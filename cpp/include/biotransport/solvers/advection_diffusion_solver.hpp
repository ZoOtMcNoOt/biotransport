#ifndef BIOTRANSPORT_SOLVERS_ADVECTION_DIFFUSION_SOLVER_HPP
#define BIOTRANSPORT_SOLVERS_ADVECTION_DIFFUSION_SOLVER_HPP

/**
 * @file advection_diffusion_solver.hpp
 * @brief Advection-diffusion solver with upwind/central schemes.
 *
 * Solves: ∂C/∂t + v·∇C = D∇²C
 *
 * Uses upwind differencing for advection when Pe > 2 (convection-dominated),
 * central differencing otherwise.
 */

#include <biotransport/core/problems/transport_problem.hpp>  // For AdvectionScheme
#include <biotransport/solvers/solver_base.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>

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
    void setScheme(AdvectionScheme scheme) { scheme_ = scheme; }

    double vx(int i) const { return uniform_velocity_ ? vx_uniform_ : vx_field_[i]; }

    double vy(int i) const { return uniform_velocity_ ? vy_uniform_ : vy_field_[i]; }

private:
    bool uniform_velocity_;
    double vx_uniform_, vy_uniform_;
    std::vector<double> vx_field_, vy_field_;
    AdvectionScheme scheme_;
    double max_vx_, max_vy_;

    void computeMaxVelocities();
    bool useUpwind(double v, double dx) const;

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
    if (vx_field.size() != static_cast<size_t>(mesh.numNodes())) {
        throw std::invalid_argument("vx_field size must match mesh nodes");
    }
    if (!mesh.is1D() && vy_field.size() != static_cast<size_t>(mesh.numNodes())) {
        throw std::invalid_argument("vy_field size must match mesh nodes for 2D");
    }
    computeMaxVelocities();
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
    double Pe_x = max_vx_ * mesh_.dx() / diffusivity_;
    double Pe_y = mesh_.is1D() ? 0.0 : (max_vy_ * mesh_.dy() / diffusivity_);
    return std::max(Pe_x, Pe_y);
}

inline double AdvectionDiffusionSolver::maxTimeStep(double safety) const {
    const double dx = mesh_.dx();
    const double dy = mesh_.is1D() ? dx : mesh_.dy();

    // Diffusion limit
    double dt_diff = mesh_.is1D()
                         ? (dx * dx) / (2.0 * diffusivity_)
                         : 1.0 / (2.0 * diffusivity_ * (1.0 / (dx * dx) + 1.0 / (dy * dy)));

    // Advection limit (CFL)
    double dt_adv = std::numeric_limits<double>::max();
    if (max_vx_ > 1e-12)
        dt_adv = std::min(dt_adv, dx / max_vx_);
    if (!mesh_.is1D() && max_vy_ > 1e-12)
        dt_adv = std::min(dt_adv, dy / max_vy_);

    return safety * std::min(dt_diff, dt_adv);
}

inline bool AdvectionDiffusionSolver::isSchemeStable() const {
    if (scheme_ == AdvectionScheme::UPWIND || scheme_ == AdvectionScheme::HYBRID) {
        return true;
    }
    return cellPeclet() < 2.0;
}

inline bool AdvectionDiffusionSolver::useUpwind(double v, double dx) const {
    if (scheme_ == AdvectionScheme::UPWIND)
        return true;
    if (scheme_ == AdvectionScheme::CENTRAL)
        return false;
    return std::abs(v) * dx / diffusivity_ >= 2.0;
}

inline bool AdvectionDiffusionSolver::checkStabilityDerived(double dt) const {
    return dt <= maxTimeStep(1.0);
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
    if (std::abs(vx_local) > 1e-14) {
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
    if (!mesh_.is1D() && std::abs(vy_local) > 1e-14) {
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
