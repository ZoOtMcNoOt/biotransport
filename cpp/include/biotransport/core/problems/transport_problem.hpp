#ifndef BIOTRANSPORT_CORE_PROBLEMS_TRANSPORT_PROBLEM_HPP
#define BIOTRANSPORT_CORE_PROBLEMS_TRANSPORT_PROBLEM_HPP

/**
 * @file transport_problem.hpp
 * @brief Declarative specification of a scalar transport problem.
 *
 * The canonical equation represented by this class is
 *
 * @f[
 *   \frac{\partial c}{\partial t}
 *     = \nabla \cdot (D \nabla c) - \nabla \cdot (\mathbf{v}c)
 *       + R(c,x,y,t).
 * @f]
 *
 * Diffusivity and velocity values are node centred.  Reaction terms may be
 * replaced with reaction() or composed explicitly with addReaction().  The
 * mesh is owned by the problem, so a problem remains valid when the mesh used
 * to construct it goes out of scope.
 */

#include <algorithm>
#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/physics/reactions.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace biotransport {

/** @brief Spatial discretization used for the advective flux. */
enum class AdvectionScheme {
    UPWIND,   ///< Conservative first-order upwind flux (implemented)
    CENTRAL,  ///< Reserved; the science-first solver rejects this scheme
    HYBRID,   ///< Reserved; the science-first solver rejects this scheme
    QUICK     ///< Reserved; the science-first solver rejects this scheme
};

/**
 * @brief Complete specification of a one- or two-dimensional scalar transport problem.
 *
 * Units must be mutually consistent.  For example, with metres, seconds, and
 * mol/m^3, D has units m^2/s, velocity m/s, and R mol/(m^3 s).
 *
 * Boundaries default to zero outward-normal derivative (an impermeable
 * diffusive boundary).  For Neumann conditions, the supplied value is
 * @f$\partial c/\partial n@f$, where n is the *outward* unit normal; it is not
 * a pre-multiplied diffusive flux.  Robin conditions use
 * @f$a c + b\,\partial c/\partial n = c_{rhs}@f$.
 */
class TransportProblem {
public:
    using ReactionFunc = reactions::ReactionFunc;

    /** @brief Node-centred scalar velocity component, called as v(x,y). */
    using VelocityFunc = std::function<double(double, double)>;

    explicit TransportProblem(const StructuredMesh& mesh)
        : mesh_(mesh),
          reaction_(reactions::none()),
          initial_(static_cast<std::size_t>(mesh.numNodes()), 0.0) {}

    // ---------------------------------------------------------------------
    // Diffusion
    // ---------------------------------------------------------------------

    /** @brief Use a uniform, non-negative diffusivity. */
    TransportProblem& diffusivity(double value) {
        requireFinite(value, "diffusivity");
        if (value < 0.0) {
            throw std::invalid_argument("diffusivity must be non-negative");
        }
        diffusivity_ = value;
        diffusivity_field_.clear();
        uniform_diffusivity_ = true;
        return *this;
    }

    /** @brief Use a node-centred, non-negative diffusivity field. */
    TransportProblem& diffusivityField(const std::vector<double>& values) {
        validateNodeField(values, "diffusivity field");
        for (double value : values) {
            if (value < 0.0) {
                throw std::invalid_argument("diffusivity field values must be non-negative");
            }
        }
        diffusivity_field_ = values;
        uniform_diffusivity_ = false;
        return *this;
    }

    // ---------------------------------------------------------------------
    // Reactions
    // ---------------------------------------------------------------------

    /**
     * @brief Replace the reaction term with a custom reaction.
     *
     * Because no concentration derivative bound is supplied, automatic
     * reaction-stability guarantees are unavailable.  A solve is still
     * permitted with an explicit user time step, but transport_solver.hpp
     * reports that the reaction stability bound was unknown.
     */
    TransportProblem& reaction(ReactionFunc value) {
        requireReaction(value);
        reaction_ = std::move(value);
        has_reaction_ = true;
        reaction_bound_known_ = false;
        reaction_rate_bound_ = 0.0;
        linear_reaction_rate_ = 0.0;
        return *this;
    }

    /**
     * @brief Replace the reaction term and provide max |dR/dc| [1/time].
     */
    TransportProblem& reaction(ReactionFunc value, double max_abs_dc) {
        requireReaction(value);
        validateRateBound(max_abs_dc);
        reaction_ = std::move(value);
        has_reaction_ = true;
        reaction_bound_known_ = true;
        reaction_rate_bound_ = max_abs_dc;
        linear_reaction_rate_ = 0.0;
        return *this;
    }

    /**
     * @brief Add a custom reaction to the existing reaction term.
     *
     * Composition is explicit: after addReaction(r), the evaluated source is
     * R_old + r.  This avoids the common error of silently replacing one
     * physical process while configuring another.
     */
    TransportProblem& addReaction(ReactionFunc value) {
        requireReaction(value);
        composeReaction(std::move(value));
        reaction_bound_known_ = false;
        reaction_rate_bound_ = 0.0;
        linear_reaction_rate_ = 0.0;
        return *this;
    }

    /** @brief Add a reaction with a known max |dR/dc| [1/time]. */
    TransportProblem& addReaction(ReactionFunc value, double max_abs_dc) {
        requireReaction(value);
        validateRateBound(max_abs_dc);
        composeReaction(std::move(value));
        if (reaction_bound_known_) {
            reaction_rate_bound_ += max_abs_dc;
        }
        linear_reaction_rate_ = 0.0;
        return *this;
    }

    /** @brief Remove every configured reaction term. */
    TransportProblem& clearReaction() {
        reaction_ = reactions::none();
        has_reaction_ = false;
        reaction_bound_known_ = true;
        reaction_rate_bound_ = 0.0;
        linear_reaction_rate_ = 0.0;
        return *this;
    }

    /** @brief Replace the reaction with first-order decay R=-k*c. */
    TransportProblem& linearDecay(double k) {
        validateNonNegativeFinite(k, "linear decay rate");
        reaction_ = reactions::linearDecay(k);
        has_reaction_ = (k != 0.0);
        reaction_bound_known_ = true;
        reaction_rate_bound_ = k;
        linear_reaction_rate_ = k;
        return *this;
    }

    /** @brief Add first-order decay R=-k*c. */
    TransportProblem& addLinearDecay(double k) {
        validateNonNegativeFinite(k, "linear decay rate");
        return addReaction(reactions::linearDecay(k), k);
    }

    /** @brief Replace the reaction with a constant source R=S. */
    TransportProblem& constantSource(double source) {
        requireFinite(source, "constant source");
        reaction_ = reactions::constantSource(source);
        has_reaction_ = (source != 0.0);
        reaction_bound_known_ = true;
        reaction_rate_bound_ = 0.0;
        linear_reaction_rate_ = 0.0;
        return *this;
    }

    /** @brief Add a constant source R=S. */
    TransportProblem& addConstantSource(double source) {
        requireFinite(source, "constant source");
        return addReaction(reactions::constantSource(source), 0.0);
    }

    /** @brief Replace the reaction with Michaelis-Menten consumption. */
    TransportProblem& michaelisMenten(double vmax, double km) {
        validateNonNegativeFinite(vmax, "Michaelis-Menten Vmax");
        requireFinite(km, "Michaelis-Menten Km");
        if (km <= 0.0) {
            throw std::invalid_argument("Michaelis-Menten Km must be positive");
        }
        reaction_ = reactions::michaelisMenten(vmax, km);
        has_reaction_ = (vmax != 0.0);
        reaction_bound_known_ = true;
        reaction_rate_bound_ = vmax / km;
        linear_reaction_rate_ = 0.0;
        return *this;
    }

    /** @brief Add Michaelis-Menten consumption. */
    TransportProblem& addMichaelisMenten(double vmax, double km) {
        validateNonNegativeFinite(vmax, "Michaelis-Menten Vmax");
        requireFinite(km, "Michaelis-Menten Km");
        if (km <= 0.0) {
            throw std::invalid_argument("Michaelis-Menten Km must be positive");
        }
        return addReaction(reactions::michaelisMenten(vmax, km), vmax / km);
    }

    /** @brief Replace the reaction with logistic growth R=r*c*(1-c/K). */
    TransportProblem& logisticGrowth(double r, double carrying_capacity) {
        validateLogistic(r, carrying_capacity);
        reaction_ = reactions::logistic(r, carrying_capacity);
        has_reaction_ = (r != 0.0);
        // No finite global derivative bound exists without a concentration range.
        reaction_bound_known_ = (r == 0.0);
        reaction_rate_bound_ = 0.0;
        linear_reaction_rate_ = 0.0;
        return *this;
    }

    /** @brief Add logistic growth. */
    TransportProblem& addLogisticGrowth(double r, double carrying_capacity) {
        validateLogistic(r, carrying_capacity);
        return addReaction(reactions::logistic(r, carrying_capacity));
    }

    // ---------------------------------------------------------------------
    // Advection
    // ---------------------------------------------------------------------

    /** @brief Use a uniform velocity.  vy must be zero for a 1D mesh. */
    TransportProblem& velocity(double vx, double vy = 0.0) {
        requireFinite(vx, "x velocity");
        requireFinite(vy, "y velocity");
        if (mesh_.is1D() && vy != 0.0) {
            throw std::invalid_argument("y velocity must be zero for a 1D mesh");
        }
        vx_uniform_ = vx;
        vy_uniform_ = vy;
        vx_field_.clear();
        vy_field_.clear();
        uniform_velocity_ = true;
        has_advection_ = (vx != 0.0 || vy != 0.0);
        return *this;
    }

    /** @brief Use node-centred velocity components. */
    TransportProblem& velocityField(const std::vector<double>& vx, const std::vector<double>& vy) {
        validateNodeField(vx, "x velocity field");
        std::vector<double> validated_vy;
        if (mesh_.is1D() && vy.empty()) {
            validated_vy.assign(vx.size(), 0.0);
        } else {
            validateNodeField(vy, "y velocity field");
            validated_vy = vy;
        }
        if (mesh_.is1D() && std::any_of(validated_vy.begin(), validated_vy.end(),
                                        [](double value) { return value != 0.0; })) {
            throw std::invalid_argument("y velocity field must be zero for a 1D mesh");
        }

        vx_field_ = vx;
        vy_field_ = std::move(validated_vy);
        uniform_velocity_ = false;
        has_advection_ = std::any_of(vx_field_.begin(), vx_field_.end(),
                                     [](double value) { return value != 0.0; }) ||
                         std::any_of(vy_field_.begin(), vy_field_.end(),
                                     [](double value) { return value != 0.0; });
        return *this;
    }

    /** @brief 1D convenience overload for a node-centred x velocity. */
    TransportProblem& velocityField(const std::vector<double>& vx) {
        if (!mesh_.is1D()) {
            throw std::invalid_argument("a 2D velocity field requires both vx and vy");
        }
        return velocityField(vx, {});
    }

    /** @brief Remove advection from the equation. */
    TransportProblem& clearVelocity() {
        has_advection_ = false;
        uniform_velocity_ = true;
        vx_uniform_ = 0.0;
        vy_uniform_ = 0.0;
        vx_field_.clear();
        vy_field_.clear();
        return *this;
    }

    /**
     * @brief Select the advective flux scheme.
     *
     * The value is stored for compatibility.  The science-first explicit
     * solver currently implements UPWIND only and throws for every other value.
     */
    TransportProblem& advectionScheme(AdvectionScheme scheme) noexcept {
        scheme_ = scheme;
        return *this;
    }

    // ---------------------------------------------------------------------
    // Initial and boundary conditions
    // ---------------------------------------------------------------------

    TransportProblem& initialCondition(const std::vector<double>& values) {
        validateNodeField(values, "initial condition");
        initial_ = values;
        return *this;
    }

    TransportProblem& initialCondition(double value) {
        requireFinite(value, "initial condition");
        initial_.assign(static_cast<std::size_t>(mesh_.numNodes()), value);
        return *this;
    }

    /** @brief Set c(x,0)=exp(-(x-x0)^2/(2*sigma^2)) on a 1D mesh. */
    TransportProblem& initialGaussian(double x0, double sigma) {
        if (!mesh_.is1D()) {
            throw std::invalid_argument("initialGaussian is only valid for 1D meshes");
        }
        requireFinite(x0, "Gaussian centre");
        requireFinite(sigma, "Gaussian sigma");
        if (sigma <= 0.0) {
            throw std::invalid_argument("Gaussian sigma must be positive");
        }

        initial_.resize(static_cast<std::size_t>(mesh_.numNodes()));
        for (int i = 0; i <= mesh_.nx(); ++i) {
            const double delta = mesh_.x(i) - x0;
            initial_[static_cast<std::size_t>(mesh_.index(i))] =
                std::exp(-(delta * delta) / (2.0 * sigma * sigma));
        }
        return *this;
    }

    /** @brief Set a piecewise-constant initial condition on a 1D mesh. */
    TransportProblem& initialStep(double x_step, double value_left, double value_right) {
        if (!mesh_.is1D()) {
            throw std::invalid_argument("initialStep is only valid for 1D meshes");
        }
        requireFinite(x_step, "step location");
        requireFinite(value_left, "left step value");
        requireFinite(value_right, "right step value");

        initial_.resize(static_cast<std::size_t>(mesh_.numNodes()));
        for (int i = 0; i <= mesh_.nx(); ++i) {
            initial_[static_cast<std::size_t>(mesh_.index(i))] =
                (mesh_.x(i) < x_step) ? value_left : value_right;
        }
        return *this;
    }

    /** @brief Set a circular inclusion initial condition on a 2D mesh. */
    TransportProblem& initialCircular(double x0, double y0, double radius, double value_inside,
                                      double value_outside = 0.0) {
        if (mesh_.is1D()) {
            throw std::invalid_argument("initialCircular requires a 2D mesh");
        }
        requireFinite(x0, "circle x centre");
        requireFinite(y0, "circle y centre");
        validateNonNegativeFinite(radius, "circle radius");
        requireFinite(value_inside, "inside value");
        requireFinite(value_outside, "outside value");

        initial_.resize(static_cast<std::size_t>(mesh_.numNodes()));
        for (int j = 0; j <= mesh_.ny(); ++j) {
            for (int i = 0; i <= mesh_.nx(); ++i) {
                const double delta_x = mesh_.x(i) - x0;
                const double delta_y = mesh_.y(i, j) - y0;
                const double distance = std::sqrt(delta_x * delta_x + delta_y * delta_y);
                initial_[static_cast<std::size_t>(mesh_.index(i, j))] =
                    (distance <= radius) ? value_inside : value_outside;
            }
        }
        return *this;
    }

    TransportProblem& boundary(Boundary side, const BoundaryCondition& condition) {
        const std::size_t index = boundaryIndex(side);
        validateBoundaryCondition(condition);
        if (mesh_.is1D() && (side == Boundary::Bottom || side == Boundary::Top)) {
            throw std::invalid_argument("bottom and top boundaries do not exist on a 1D mesh");
        }
        boundaries_[index] = condition;
        boundary_configured_[index] = true;
        return *this;
    }

    TransportProblem& dirichlet(Boundary side, double value) {
        requireFinite(value, "Dirichlet value");
        return boundary(side, BoundaryCondition::Dirichlet(value));
    }

    /**
     * @brief Set dc/dn on a boundary, with n pointing out of the domain.
     *
     * This parameter is a derivative, despite the historical name `flux` in
     * BoundaryCondition.  The solver multiplies it by the local diffusivity.
     */
    TransportProblem& neumann(Boundary side, double outward_normal_derivative) {
        requireFinite(outward_normal_derivative, "Neumann derivative");
        return boundary(side, BoundaryCondition::Neumann(outward_normal_derivative));
    }

    /** @brief Set a*c + b*dc/dn = c_rhs, with n the outward normal. */
    TransportProblem& robin(Boundary side, double a, double b, double c_rhs) {
        return boundary(side, BoundaryCondition::Robin(a, b, c_rhs));
    }

    // ---------------------------------------------------------------------
    // Accessors retained for the existing solver and binding surface
    // ---------------------------------------------------------------------

    const StructuredMesh& mesh() const noexcept { return mesh_; }

    bool hasUniformDiffusivity() const noexcept { return uniform_diffusivity_; }
    double diffusivity() const noexcept { return diffusivity_; }
    const std::vector<double>& diffusivityField() const noexcept { return diffusivity_field_; }

    const ReactionFunc& reaction() const noexcept { return reaction_; }
    bool hasReaction() const noexcept { return has_reaction_; }
    bool reactionStabilityBoundKnown() const noexcept { return reaction_bound_known_; }
    double reactionStabilityRateBound() const noexcept { return reaction_rate_bound_; }

    /**
     * @brief Compatibility metadata for the legacy ExplicitFD facade.
     * @return k only when the complete reaction is exactly R=-k*c; otherwise zero.
     */
    double linearReactionRate() const noexcept { return linear_reaction_rate_; }

    bool hasAdvection() const noexcept { return has_advection_; }
    bool hasUniformVelocity() const noexcept { return uniform_velocity_; }
    double vxUniform() const noexcept { return vx_uniform_; }
    double vyUniform() const noexcept { return vy_uniform_; }
    const std::vector<double>& vxField() const noexcept { return vx_field_; }
    const std::vector<double>& vyField() const noexcept { return vy_field_; }
    AdvectionScheme scheme() const noexcept { return scheme_; }

    const std::vector<double>& initial() const noexcept { return initial_; }
    const std::array<BoundaryCondition, 4>& boundaries() const noexcept { return boundaries_; }
    bool boundaryWasSet(Boundary side) const { return boundary_configured_[boundaryIndex(side)]; }

private:
    static void requireFinite(double value, const char* name) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(name) + " must be finite");
        }
    }

    static void validateNonNegativeFinite(double value, const char* name) {
        requireFinite(value, name);
        if (value < 0.0) {
            throw std::invalid_argument(std::string(name) + " must be non-negative");
        }
    }

    static void validateRateBound(double value) {
        validateNonNegativeFinite(value, "reaction derivative bound");
    }

    static void requireReaction(const ReactionFunc& value) {
        if (!value) {
            throw std::invalid_argument("reaction function must be callable");
        }
    }

    static void validateLogistic(double r, double carrying_capacity) {
        validateNonNegativeFinite(r, "logistic growth rate");
        requireFinite(carrying_capacity, "logistic carrying capacity");
        if (carrying_capacity <= 0.0) {
            throw std::invalid_argument("logistic carrying capacity must be positive");
        }
    }

    static std::size_t boundaryIndex(Boundary side) {
        switch (side) {
            case Boundary::Left:
                return 0;
            case Boundary::Right:
                return 1;
            case Boundary::Bottom:
                return 2;
            case Boundary::Top:
                return 3;
        }
        throw std::invalid_argument("invalid Boundary value");
    }

    void validateNodeField(const std::vector<double>& values, const char* name) const {
        if (values.size() != static_cast<std::size_t>(mesh_.numNodes())) {
            throw std::invalid_argument(std::string(name) + " must contain exactly " +
                                        std::to_string(mesh_.numNodes()) + " node values");
        }
        for (double value : values) {
            requireFinite(value, name);
        }
    }

    static void validateBoundaryCondition(const BoundaryCondition& condition) {
        switch (condition.type) {
            case BoundaryType::DIRICHLET:
            case BoundaryType::NEUMANN:
                requireFinite(condition.value, "boundary value");
                return;
            case BoundaryType::ROBIN:
                requireFinite(condition.a, "Robin a");
                requireFinite(condition.b, "Robin b");
                requireFinite(condition.c, "Robin c");
                if (condition.a == 0.0 && condition.b == 0.0) {
                    throw std::invalid_argument("Robin condition requires non-zero a or b");
                }
                return;
            case BoundaryType::OUTWARD_FLUX:
                throw std::invalid_argument(
                    "TransportProblem prescribes Neumann outward-normal derivatives dc/dn; "
                    "a physical OUTWARD_FLUX condition is not supported here. Use "
                    "neumann(side, -flux / D) for a uniform diffusivity, or a Robin condition.");
        }
        throw std::invalid_argument("invalid BoundaryType value");
    }

    void composeReaction(ReactionFunc value) {
        if (!has_reaction_) {
            reaction_ = std::move(value);
        } else {
            ReactionFunc previous = std::move(reaction_);
            reaction_ = [previous = std::move(previous), value = std::move(value)](
                            double concentration, double x, double y, double time) {
                return previous(concentration, x, y, time) + value(concentration, x, y, time);
            };
        }
        has_reaction_ = true;
    }

    StructuredMesh mesh_;

    bool uniform_diffusivity_ = true;
    double diffusivity_ = 0.0;
    std::vector<double> diffusivity_field_;

    ReactionFunc reaction_;
    bool has_reaction_ = false;
    bool reaction_bound_known_ = true;
    double reaction_rate_bound_ = 0.0;
    double linear_reaction_rate_ = 0.0;

    bool has_advection_ = false;
    bool uniform_velocity_ = true;
    double vx_uniform_ = 0.0;
    double vy_uniform_ = 0.0;
    std::vector<double> vx_field_;
    std::vector<double> vy_field_;
    AdvectionScheme scheme_ = AdvectionScheme::UPWIND;

    std::vector<double> initial_;
    std::array<BoundaryCondition, 4> boundaries_ = {
        BoundaryCondition::Neumann(0.0), BoundaryCondition::Neumann(0.0),
        BoundaryCondition::Neumann(0.0), BoundaryCondition::Neumann(0.0)};
    std::array<bool, 4> boundary_configured_ = {false, false, false, false};
};

/** @deprecated Use TransportProblem. */
using DiffusionProblem = TransportProblem;

}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_PROBLEMS_TRANSPORT_PROBLEM_HPP
