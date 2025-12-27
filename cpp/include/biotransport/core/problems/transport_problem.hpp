#ifndef BIOTRANSPORT_CORE_PROBLEMS_TRANSPORT_PROBLEM_HPP
#define BIOTRANSPORT_CORE_PROBLEMS_TRANSPORT_PROBLEM_HPP

/**
 * @file transport_problem.hpp
 * @brief Unified problem specification for transport equations.
 *
 * Consolidates all transport problem types into a single fluent builder:
 *   - Pure diffusion:      ∂u/∂t = D∇²u
 *   - Reaction-diffusion:  ∂u/∂t = D∇²u + R(u)
 *   - Advection-diffusion: ∂u/∂t = D∇²u - v·∇u + R(u)
 *
 * Uses composable reaction functors from reactions.hpp instead of
 * separate Problem classes for each reaction type.
 *
 * Example:
 * @code
 *   using namespace biotransport::reactions;
 *
 *   // Michaelis-Menten consumption with constant source
 *   auto problem = TransportProblem(mesh)
 *       .diffusivity(1e-9)
 *       .reaction(combine(michaelisMenten(-Vmax, Km), constantSource(S)))
 *       .dirichlet(Boundary::Left, 1.0)
 *       .neumann(Boundary::Right, 0.0)
 *       .initialCondition(initial);
 * @endcode
 */

#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/physics/reactions.hpp>
#include <functional>
#include <vector>

namespace biotransport {

/**
 * @brief Advection scheme selection.
 *
 * Defined here so TransportProblem can use it as a default member value.
 */
enum class AdvectionScheme {
    UPWIND,   ///< First-order upwind (stable, diffusive)
    CENTRAL,  ///< Central differencing (second-order, oscillatory for Pe > 2)
    HYBRID,   ///< Automatically switch based on local Peclet number
    QUICK     ///< Quadratic upstream interpolation (third-order)
};

/**
 * @brief Unified fluent builder for transport problem specification.
 *
 * Replaces separate DiffusionProblem, LinearReactionDiffusionProblem,
 * MichaelisMentenReactionDiffusionProblem, etc. with a single configurable class.
 */
class TransportProblem {
public:
    // Use the same reaction signature as reactions.hpp for compatibility
    using ReactionFunc = reactions::ReactionFunc;
    using VelocityFunc = std::function<double(double, double)>;

    explicit TransportProblem(const StructuredMesh& mesh)
        : mesh_(mesh),
          reaction_([](double, double, double, double) { return 0.0; })  // No reaction by default
    {}

    // =========================================================================
    // Diffusion configuration
    // =========================================================================

    /**
     * @brief Set uniform diffusivity.
     */
    TransportProblem& diffusivity(double D) {
        diffusivity_ = D;
        uniform_diffusivity_ = true;
        return *this;
    }

    /**
     * @brief Set spatially-varying diffusivity field.
     */
    TransportProblem& diffusivityField(const std::vector<double>& D_field) {
        diffusivity_field_ = D_field;
        uniform_diffusivity_ = false;
        return *this;
    }

    // =========================================================================
    // Reaction configuration
    // =========================================================================

    /**
     * @brief Set reaction term R(u).
     *
     * Use functors from biotransport::reactions namespace:
     * @code
     *   using namespace biotransport::reactions;
     *   problem.reaction(linearDecay(0.1));
     *   problem.reaction(michaelisMenten(Vmax, Km));
     *   problem.reaction(combine(logistic(r, K), constantSource(S)));
     * @endcode
     */
    TransportProblem& reaction(ReactionFunc R) {
        reaction_ = std::move(R);
        return *this;
    }

    // Convenience methods for common reaction types
    TransportProblem& linearDecay(double k) {
        linear_reaction_rate_ = k;  // Track for stability check
        return reaction(reactions::linearDecay(k));
    }

    TransportProblem& constantSource(double S) { return reaction(reactions::constantSource(S)); }

    TransportProblem& michaelisMenten(double Vmax, double Km) {
        return reaction(reactions::michaelisMenten(Vmax, Km));
    }

    TransportProblem& logisticGrowth(double r, double K) {
        return reaction(reactions::logistic(r, K));
    }

    // =========================================================================
    // Advection configuration
    // =========================================================================

    /**
     * @brief Set uniform velocity field.
     */
    TransportProblem& velocity(double vx, double vy = 0.0) {
        vx_uniform_ = vx;
        vy_uniform_ = vy;
        uniform_velocity_ = true;
        has_advection_ = true;
        return *this;
    }

    /**
     * @brief Set spatially-varying velocity field.
     */
    TransportProblem& velocityField(const std::vector<double>& vx, const std::vector<double>& vy) {
        vx_field_ = vx;
        vy_field_ = vy;
        uniform_velocity_ = false;
        has_advection_ = true;
        return *this;
    }

    /**
     * @brief Set advection discretization scheme.
     */
    TransportProblem& advectionScheme(AdvectionScheme scheme) {
        scheme_ = scheme;
        return *this;
    }

    // =========================================================================
    // Initial and boundary conditions
    // =========================================================================

    /**
     * @brief Set initial condition from vector.
     */
    TransportProblem& initialCondition(const std::vector<double>& values) {
        initial_ = values;  // Copy assignment
        return *this;
    }

    /**
     * @brief Set uniform initial condition.
     */
    TransportProblem& initialCondition(double value) {
        initial_.assign(mesh_.numNodes(), value);
        return *this;
    }

    /**
     * @brief Set Gaussian initial condition (1D only).
     *
     * Creates IC: u(x) = exp(-((x - x0)^2) / (2*sigma^2))
     *
     * @param x0 Center position
     * @param sigma Standard deviation (width)
     */
    TransportProblem& initialGaussian(double x0, double sigma) {
        if (!mesh_.is1D()) {
            throw std::invalid_argument("initialGaussian is only valid for 1D meshes");
        }

        initial_.resize(mesh_.numNodes());
        for (int i = 0; i <= mesh_.nx(); ++i) {
            double x = mesh_.x(i);
            double dx = x - x0;
            initial_[i] = std::exp(-(dx * dx) / (2.0 * sigma * sigma));
        }
        return *this;
    }

    /**
     * @brief Set step function initial condition (1D only).
     *
     * Creates IC: u(x) = value_left if x < x_step, else value_right
     *
     * @param x_step Step location
     * @param value_left Value for x < x_step
     * @param value_right Value for x >= x_step
     */
    TransportProblem& initialStep(double x_step, double value_left, double value_right) {
        if (!mesh_.is1D()) {
            throw std::invalid_argument("initialStep is only valid for 1D meshes");
        }

        initial_.resize(mesh_.numNodes());
        for (int i = 0; i <= mesh_.nx(); ++i) {
            double x = mesh_.x(i);
            initial_[i] = (x < x_step) ? value_left : value_right;
        }
        return *this;
    }

    /**
     * @brief Set circular initial condition (2D only).
     *
     * Creates IC: u = value_inside if distance to center <= radius, else value_outside
     *
     * @param x0 Circle center x-coordinate
     * @param y0 Circle center y-coordinate
     * @param radius Circle radius
     * @param value_inside Value inside circle
     * @param value_outside Value outside circle (default 0)
     */
    TransportProblem& initialCircular(double x0, double y0, double radius, double value_inside,
                                      double value_outside = 0.0) {
        if (mesh_.is1D()) {
            throw std::invalid_argument("initialCircular requires a 2D mesh");
        }

        initial_.resize(mesh_.numNodes());
        for (int i = 0; i <= mesh_.nx(); ++i) {
            for (int j = 0; j <= mesh_.ny(); ++j) {
                double x = mesh_.x(i);
                double y = mesh_.y(0, j);
                double dx = x - x0;
                double dy = y - y0;
                double dist = std::sqrt(dx * dx + dy * dy);

                int idx = mesh_.index(i, j);
                initial_[idx] = (dist <= radius) ? value_inside : value_outside;
            }
        }
        return *this;
    }

    /**
     * @brief Set boundary condition on a side.
     */
    TransportProblem& boundary(Boundary side, const BoundaryCondition& bc) {
        boundaries_[static_cast<int>(side)] = bc;
        return *this;
    }

    /**
     * @brief Set Dirichlet (fixed value) boundary condition.
     */
    TransportProblem& dirichlet(Boundary side, double value) {
        return boundary(side, BoundaryCondition::Dirichlet(value));
    }

    /**
     * @brief Set Neumann (fixed flux) boundary condition.
     */
    TransportProblem& neumann(Boundary side, double flux) {
        return boundary(side, BoundaryCondition::Neumann(flux));
    }

    /**
     * @brief Set Robin (mixed) boundary condition: a*u + b*du/dn = c
     */
    TransportProblem& robin(Boundary side, double a, double b, double c) {
        return boundary(side, BoundaryCondition::Robin(a, b, c));
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    const StructuredMesh& mesh() const { return mesh_; }

    bool hasUniformDiffusivity() const { return uniform_diffusivity_; }
    double diffusivity() const { return diffusivity_; }
    const std::vector<double>& diffusivityField() const { return diffusivity_field_; }

    const ReactionFunc& reaction() const { return reaction_; }

    /**
     * @brief Get the linear reaction rate for stability checking.
     *
     * Returns the coefficient k from linearDecay(k) if set, 0.0 otherwise.
     * Used by solvers to compute reaction-stable timestep: dt < 2/k.
     */
    double linearReactionRate() const { return linear_reaction_rate_; }

    bool hasAdvection() const { return has_advection_; }
    bool hasUniformVelocity() const { return uniform_velocity_; }
    double vxUniform() const { return vx_uniform_; }
    double vyUniform() const { return vy_uniform_; }
    const std::vector<double>& vxField() const { return vx_field_; }
    const std::vector<double>& vyField() const { return vy_field_; }
    AdvectionScheme scheme() const { return scheme_; }

    const std::vector<double>& initial() const { return initial_; }
    const std::array<BoundaryCondition, 4>& boundaries() const { return boundaries_; }

private:
    const StructuredMesh& mesh_;

    // Diffusion
    bool uniform_diffusivity_ = true;
    double diffusivity_ = 0.0;
    std::vector<double> diffusivity_field_;

    // Reaction
    ReactionFunc reaction_;
    double linear_reaction_rate_ = 0.0;  // For stability checking (linearDecay)

    // Advection
    bool has_advection_ = false;
    bool uniform_velocity_ = true;
    double vx_uniform_ = 0.0;
    double vy_uniform_ = 0.0;
    std::vector<double> vx_field_;
    std::vector<double> vy_field_;
    AdvectionScheme scheme_ = AdvectionScheme::HYBRID;

    // Initial and boundary conditions
    std::vector<double> initial_;
    std::array<BoundaryCondition, 4> boundaries_ = {
        BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0),
        BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)};
};

// =============================================================================
// Backward-compatible type aliases
// =============================================================================

/**
 * @brief Alias for backward compatibility.
 * @deprecated Use TransportProblem instead.
 */
using DiffusionProblem = TransportProblem;

}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_PROBLEMS_TRANSPORT_PROBLEM_HPP
