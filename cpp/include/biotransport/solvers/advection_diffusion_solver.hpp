#ifndef BIOTRANSPORT_SOLVERS_ADVECTION_DIFFUSION_SOLVER_HPP
#define BIOTRANSPORT_SOLVERS_ADVECTION_DIFFUSION_SOLVER_HPP

/**
 * @file advection_diffusion_solver.hpp
 * @brief Explicit advective-form advection-diffusion solver.
 *
 * Solves the nonconservative advective form
 *
 *   ∂C/∂t + vx ∂C/∂x + vy ∂C/∂y = D∇²C.
 *
 * For a spatially varying velocity this is not the conservative equation
 * ∂C/∂t + ∇·(vC) = D∇²C; the C∇·v term is intentionally absent. Use this
 * specialized solver only when the advective form is the intended model
 * (commonly for a discretely divergence-free prescribed velocity).
 *
 * UPWIND uses first-order upwinding, CENTRAL uses centered advection and is
 * admitted only while each directional cell Peclet number is at most two,
 * and HYBRID selects between those stencils by directional cell Peclet
 * number. QUICK is not implemented and is rejected.
 */

#include <algorithm>
#include <biotransport/core/problems/transport_problem.hpp>  // For AdvectionScheme
#include <biotransport/solvers/solver_base.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace biotransport {

// AdvectionScheme is now defined in transport_problem.hpp

/**
 * @brief Advection-diffusion solver.
 */
class AdvectionDiffusionSolver : public ExplicitSolverBase<AdvectionDiffusionSolver> {
public:
    using Base = ExplicitSolverBase<AdvectionDiffusionSolver>;
    friend Base;

    /**
     * @brief Create solver with uniform velocity.
     */
    AdvectionDiffusionSolver(const StructuredMesh& mesh, double diffusivity, double vx,
                             double vy = 0.0, AdvectionScheme scheme = AdvectionScheme::HYBRID);

    /**
     * @brief Create solver with spatially-varying velocity.
     */
    AdvectionDiffusionSolver(const StructuredMesh& mesh, double diffusivity,
                             const std::vector<double>& vx_field,
                             const std::vector<double>& vy_field,
                             AdvectionScheme scheme = AdvectionScheme::HYBRID);

    void computeNodeUpdate(int idx, int i, int j, const StencilOps& ops, double dt);

    /**
     * @brief Get the cell Peclet number.
     */
    double cellPeclet() const;

    /**
     * @brief Get maximum stable time step.
     */
    double maxTimeStep(double safety = 0.4) const;

    /**
     * @brief Check if scheme is stable for current conditions.
     */
    bool isSchemeStable() const;

    AdvectionScheme scheme() const { return scheme_; }
    void setScheme(AdvectionScheme scheme);

    double vx(int i) const;

    double vy(int i) const;

private:
    bool uniform_velocity_;
    double vx_uniform_, vy_uniform_;
    std::vector<double> vx_field_, vy_field_;
    AdvectionScheme scheme_;
    double max_vx_, max_vy_;

    void computeMaxVelocities();
    bool useUpwind(double v, double dx) const;
    double maximumDepletionRate() const;
    static void validateScheme(AdvectionScheme scheme);
    static void validateFinite(double value, const char* name);
    static void validateFiniteField(const std::vector<double>& values, const char* name);

    bool checkStabilityDerived(double dt) const;
};

// =============================================================================
// Inline implementation for simple methods
// =============================================================================

inline AdvectionDiffusionSolver::AdvectionDiffusionSolver(const StructuredMesh& mesh,
                                                          double diffusivity, double vx, double vy,
                                                          AdvectionScheme scheme)
    : Base(mesh, diffusivity),
      uniform_velocity_(true),
      vx_uniform_(vx),
      vy_uniform_(vy),
      scheme_(scheme) {
    validateFinite(vx_uniform_, "x velocity");
    validateFinite(vy_uniform_, "y velocity");
    validateScheme(scheme_);
    computeMaxVelocities();
}

inline AdvectionDiffusionSolver::AdvectionDiffusionSolver(const StructuredMesh& mesh,
                                                          double diffusivity,
                                                          const std::vector<double>& vx_field,
                                                          const std::vector<double>& vy_field,
                                                          AdvectionScheme scheme)
    : Base(mesh, diffusivity),
      uniform_velocity_(false),
      vx_uniform_(0.0),
      vy_uniform_(0.0),
      vx_field_(vx_field),
      vy_field_(vy_field),
      scheme_(scheme) {
    const auto node_count = static_cast<std::size_t>(mesh.numNodes());
    if (vx_field.size() != node_count) {
        throw std::invalid_argument("vx_field size must match mesh nodes");
    }
    if (!mesh.is1D() && vy_field.size() != node_count) {
        throw std::invalid_argument("vy_field size must match mesh nodes for 2D");
    }
    if (mesh.is1D() && !vy_field.empty() && vy_field.size() != node_count) {
        throw std::invalid_argument("vy_field must be empty or match mesh nodes for 1D");
    }
    validateFiniteField(vx_field_, "vx_field");
    validateFiniteField(vy_field_, "vy_field");
    validateScheme(scheme_);
    computeMaxVelocities();
}

inline void AdvectionDiffusionSolver::validateFinite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

inline void AdvectionDiffusionSolver::validateFiniteField(const std::vector<double>& values,
                                                          const char* name) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            throw std::invalid_argument(std::string(name) + " contains a non-finite value at " +
                                        std::to_string(index));
        }
    }
}

inline void AdvectionDiffusionSolver::validateScheme(AdvectionScheme scheme) {
    switch (scheme) {
        case AdvectionScheme::UPWIND:
        case AdvectionScheme::CENTRAL:
        case AdvectionScheme::HYBRID:
            return;
        case AdvectionScheme::QUICK:
            throw std::invalid_argument(
                "AdvectionScheme::QUICK is not implemented by AdvectionDiffusionSolver");
        default:
            throw std::invalid_argument("Unknown AdvectionScheme value");
    }
}

inline void AdvectionDiffusionSolver::setScheme(AdvectionScheme scheme) {
    validateScheme(scheme);
    scheme_ = scheme;
}

inline double AdvectionDiffusionSolver::vx(int i) const {
    if (i < 0 || i >= mesh_.numNodes()) {
        throw std::out_of_range("velocity node index is outside the mesh");
    }
    return uniform_velocity_ ? vx_uniform_ : vx_field_[static_cast<std::size_t>(i)];
}

inline double AdvectionDiffusionSolver::vy(int i) const {
    if (i < 0 || i >= mesh_.numNodes()) {
        throw std::out_of_range("velocity node index is outside the mesh");
    }
    if (mesh_.is1D()) {
        return uniform_velocity_ ? vy_uniform_ : 0.0;
    }
    return uniform_velocity_ ? vy_uniform_ : vy_field_[static_cast<std::size_t>(i)];
}

inline void AdvectionDiffusionSolver::computeMaxVelocities() {
    if (uniform_velocity_) {
        max_vx_ = std::abs(vx_uniform_);
        max_vy_ = std::abs(vy_uniform_);
    } else {
        max_vx_ = 0.0;
        max_vy_ = 0.0;
        for (size_t i = 0; i < vx_field_.size(); ++i) {
            max_vx_ = std::max(max_vx_, std::abs(vx_field_[i]));
        }
        for (size_t i = 0; i < vy_field_.size(); ++i) {
            max_vy_ = std::max(max_vy_, std::abs(vy_field_[i]));
        }
    }
}

inline double AdvectionDiffusionSolver::cellPeclet() const {
    const long double Pe_x = static_cast<long double>(max_vx_) * mesh_.dx() / diffusivity_;
    const long double Pe_y =
        mesh_.is1D() ? 0.0L : static_cast<long double>(max_vy_) * mesh_.dy() / diffusivity_;
    const long double maximum = std::max(Pe_x, Pe_y);
    if (!std::isfinite(maximum) || maximum > std::numeric_limits<double>::max()) {
        throw std::overflow_error("Cell Peclet number is not representable");
    }
    return static_cast<double>(maximum);
}

inline double AdvectionDiffusionSolver::maxTimeStep(double safety) const {
    if (!std::isfinite(safety) || safety <= 0.0 || safety > 1.0) {
        throw std::invalid_argument("safety must be finite and in (0, 1]");
    }
    if (!isSchemeStable()) {
        throw std::logic_error(
            "CENTRAL advection requires every directional cell Peclet number to be <= 2");
    }
    const long double step =
        static_cast<long double>(safety) / static_cast<long double>(maximumDepletionRate());
    if (!std::isfinite(step) || step <= 0.0L || step > std::numeric_limits<double>::max()) {
        throw std::overflow_error("Stable advection-diffusion time step is not representable");
    }
    return static_cast<double>(step);
}

inline bool AdvectionDiffusionSolver::isSchemeStable() const {
    switch (scheme_) {
        case AdvectionScheme::UPWIND:
        case AdvectionScheme::HYBRID:
            return true;
        case AdvectionScheme::CENTRAL: {
            const long double x_peclet =
                static_cast<long double>(max_vx_) * mesh_.dx() / diffusivity_;
            const long double y_peclet =
                mesh_.is1D() ? 0.0L : static_cast<long double>(max_vy_) * mesh_.dy() / diffusivity_;
            return x_peclet <= 2.0L && y_peclet <= 2.0L;
        }
        case AdvectionScheme::QUICK:
        default:
            return false;
    }
}

inline bool AdvectionDiffusionSolver::useUpwind(double v, double dx) const {
    switch (scheme_) {
        case AdvectionScheme::UPWIND:
            return true;
        case AdvectionScheme::CENTRAL:
            return false;
        case AdvectionScheme::HYBRID:
            return std::abs(v) * dx / diffusivity_ >= 2.0;
        case AdvectionScheme::QUICK:
        default:
            throw std::logic_error("Invalid advection scheme reached stencil selection");
    }
}

inline double AdvectionDiffusionSolver::maximumDepletionRate() const {
    const long double dx = mesh_.dx();
    const long double dy = mesh_.is1D() ? 1.0L : mesh_.dy();
    const long double diffusion_rate =
        mesh_.is1D() ? 2.0L * diffusivity_ / (dx * dx)
                     : 2.0L * diffusivity_ * (1.0L / (dx * dx) + 1.0L / (dy * dy));
    long double maximum_rate = diffusion_rate;
    for (int index = 0; index < mesh_.numNodes(); ++index) {
        long double rate = diffusion_rate;
        const double local_vx = vx(index);
        if (useUpwind(local_vx, static_cast<double>(dx))) {
            rate += std::abs(static_cast<long double>(local_vx)) / dx;
        }
        if (!mesh_.is1D()) {
            const double local_vy = vy(index);
            if (useUpwind(local_vy, static_cast<double>(dy))) {
                rate += std::abs(static_cast<long double>(local_vy)) / dy;
            }
        }
        maximum_rate = std::max(maximum_rate, rate);
    }
    if (!std::isfinite(maximum_rate) || maximum_rate <= 0.0L ||
        maximum_rate > std::numeric_limits<double>::max()) {
        throw std::overflow_error("Advection-diffusion depletion rate is not representable");
    }
    return static_cast<double>(maximum_rate);
}

inline bool AdvectionDiffusionSolver::checkStabilityDerived(double dt) const {
    return isSchemeStable() && dt <= maxTimeStep(1.0);
}

inline void AdvectionDiffusionSolver::computeNodeUpdate(int idx, int /*i*/, int /*j*/,
                                                        const StencilOps& ops, double dt) {
    double u = solution_[idx];

    // Diffusion term
    double diffusion = ops.diffusionTerm(solution_, idx, diffusivity_, dt);

    // Advection term: -v·∇C
    double advection = 0.0;
    double vx_local = vx(idx);
    double vy_local = vy(idx);

    const double dx = mesh_.dx();
    const double dy = mesh_.is1D() ? 1.0 : mesh_.dy();
    const int stride = iterator_.stride();

    // x-advection
    if (vx_local != 0.0) {
        if (useUpwind(vx_local, dx)) {
            if (vx_local > 0.0) {
                advection += vx_local * (solution_[idx] - solution_[idx - 1]) / dx;
            } else {
                advection += vx_local * (solution_[idx + 1] - solution_[idx]) / dx;
            }
        } else {
            advection += vx_local * (solution_[idx + 1] - solution_[idx - 1]) / (2.0 * dx);
        }
    }

    // y-advection
    if (!mesh_.is1D() && vy_local != 0.0) {
        if (useUpwind(vy_local, dy)) {
            if (vy_local > 0.0) {
                advection += vy_local * (solution_[idx] - solution_[idx - stride]) / dy;
            } else {
                advection += vy_local * (solution_[idx + stride] - solution_[idx]) / dy;
            }
        } else {
            advection +=
                vy_local * (solution_[idx + stride] - solution_[idx - stride]) / (2.0 * dy);
        }
    }

    scratch_[idx] = u + diffusion - dt * advection;
}

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_ADVECTION_DIFFUSION_SOLVER_HPP
