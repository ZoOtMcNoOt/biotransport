#ifndef BIOTRANSPORT_SOLVERS_DIFFUSION_SOLVERS_HPP
#define BIOTRANSPORT_SOLVERS_DIFFUSION_SOLVERS_HPP

/**
 * @file diffusion_solvers.hpp
 * @brief Unified diffusion and reaction-diffusion solvers.
 *
 * This header consolidates all diffusion-based solvers into a clean hierarchy:
 *
 * 1. DiffusionSolver - Pure diffusion (∂u/∂t = D∇²u)
 * 2. ReactionDiffusionSolver - Generic reaction-diffusion with functor
 * 3. Specialized solvers for performance-critical cases (Python bindings)
 *
 * The key insight is that most "specialized" solvers only differ in their
 * reaction term. Rather than duplicating 70+ lines of time-stepping code,
 * we use the CRTP base class and functors.
 *
 * For backward compatibility, the old class names are preserved as type aliases
 * or thin wrappers around the unified implementation.
 */

#include <biotransport/physics/reactions.hpp>
#include <biotransport/solvers/solver_base.hpp>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace biotransport {

// =============================================================================
// DiffusionSolver - Pure diffusion with no reaction term
// =============================================================================

/**
 * @brief Solver for the diffusion equation: ∂u/∂t = D∇²u
 */
class DiffusionSolver : public ExplicitSolverBase<DiffusionSolver> {
public:
    using Base = ExplicitSolverBase<DiffusionSolver>;
    friend Base;

    DiffusionSolver(const StructuredMesh& mesh, double diffusivity) : Base(mesh, diffusivity) {}

    // Required by CRTP base
    void computeNodeUpdate(int idx, int /*i*/, int /*j*/, const StencilOps& ops, double dt) {
        scratch_[idx] = solution_[idx] + ops.diffusionTerm(solution_, idx, diffusivity_, dt);
    }
};

// =============================================================================
// VariableDiffusionSolver - Spatially-varying diffusivity D(x)
// =============================================================================

/**
 * @brief Solver for diffusion with spatially-varying diffusivity: ∂u/∂t = ∇·(D(x)∇u)
 *
 * Uses flux-form discretization with face-averaged diffusivity for conservative
 * discretization. This is essential for problems like membrane diffusion where
 * D varies significantly across the domain.
 *
 * @code
 *   // Create diffusivity field
 *   std::vector<double> D_field(mesh.totalNodes());
 *   for (int i = 0; i <= mesh.nx(); ++i) {
 *       D_field[i] = (mesh.x(i) < 0.5) ? D_left : D_right;
 *   }
 *   auto solver = VariableDiffusionSolver(mesh, D_field);
 * @endcode
 */
class VariableDiffusionSolver : public ExplicitSolverBase<VariableDiffusionSolver> {
public:
    using Base = ExplicitSolverBase<VariableDiffusionSolver>;
    friend Base;

    VariableDiffusionSolver(const StructuredMesh& mesh, std::vector<double> diffusivity_field)
        : Base(mesh, computeMaxDiffusivity(diffusivity_field)),
          diffusivity_field_(std::move(diffusivity_field)) {
        if (diffusivity_field_.size() != static_cast<size_t>(mesh_.numNodes())) {
            throw std::invalid_argument("Diffusivity field size must match total nodes");
        }
        max_diffusivity_ = diffusivity_;  // Already computed by base
    }

    // Required by CRTP base
    void computeNodeUpdate(int idx, int /*i*/, int /*j*/, const StencilOps& ops, double dt) {
        scratch_[idx] =
            solution_[idx] + ops.variableDiffusionTerm(solution_, diffusivity_field_, idx, dt);
    }

    const std::vector<double>& diffusivityField() const { return diffusivity_field_; }
    double maxDiffusivity() const { return max_diffusivity_; }

private:
    std::vector<double> diffusivity_field_;
    double max_diffusivity_;

    static double computeMaxDiffusivity(const std::vector<double>& D_field) {
        if (D_field.empty()) {
            throw std::invalid_argument("Diffusivity field must not be empty");
        }
        double max_D = 0.0;
        for (double D : D_field) {
            if (D < 0.0) {
                throw std::invalid_argument("Diffusivity must be non-negative everywhere");
            }
            max_D = std::max(max_D, D);
        }
        if (max_D <= 0.0) {
            throw std::invalid_argument("At least one diffusivity value must be positive");
        }
        return max_D;
    }
};

// =============================================================================
// ReactionDiffusionSolver - Generic reaction-diffusion with callable
// =============================================================================

/**
 * @brief Solver for reaction-diffusion equations: ∂u/∂t = D∇²u + R(u, x, y, t)
 *
 * The reaction term R is provided as a callable (function, lambda, or functor).
 * This is the most flexible solver and should be preferred for new code.
 *
 * @code
 *   // Using lambda
 *   auto solver = ReactionDiffusionSolver(mesh, D,
 *       [](double u, double x, double y, double t) { return -k * u; });
 *
 *   // Using reactions library
 *   auto solver = ReactionDiffusionSolver(mesh, D, reactions::logistic(r, K));
 *   auto solver = ReactionDiffusionSolver(mesh, D, reactions::michaelisMenten(Vmax, Km));
 * @endcode
 */
class ReactionDiffusionSolver : public ExplicitSolverBase<ReactionDiffusionSolver> {
public:
    using Base = ExplicitSolverBase<ReactionDiffusionSolver>;
    using ReactionFunction = std::function<double(double u, double x, double y, double t)>;
    friend Base;

    ReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity,
                            ReactionFunction reaction)
        : Base(mesh, diffusivity), reaction_(std::move(reaction)) {
        // Pre-cache coordinates for performance
        cacheCoordinates();
    }

    void computeNodeUpdate(int idx, int i, int j, const StencilOps& ops, double dt) {
        double u = solution_[idx];
        double x = x_coords_[i];
        double y = mesh_.is1D() ? 0.0 : y_coords_[j];

        double diffusion = ops.diffusionTerm(solution_, idx, diffusivity_, dt);
        double reaction = dt * reaction_(u, x, y, time_);

        scratch_[idx] = u + diffusion + reaction;
    }

private:
    ReactionFunction reaction_;
    std::vector<double> x_coords_;
    std::vector<double> y_coords_;

    void cacheCoordinates() {
        x_coords_.resize(mesh_.nx() + 1);
        for (int i = 0; i <= mesh_.nx(); ++i) {
            x_coords_[i] = mesh_.x(i);
        }
        if (!mesh_.is1D()) {
            y_coords_.resize(mesh_.ny() + 1);
            for (int j = 0; j <= mesh_.ny(); ++j) {
                y_coords_[j] = mesh_.y(0, j);
            }
        }
    }
};

// =============================================================================
// Performance-optimized solvers for Python bindings
// =============================================================================

// These avoid std::function overhead when called from Python in tight loops.
// They're thin wrappers that could be removed if Python callback overhead
// is resolved at the binding level.

/**
 * @brief Linear reaction-diffusion: ∂u/∂t = D∇²u - k*u
 *
 * Uses IMPLICIT treatment of the decay term for unconditional stability:
 *   u_new = (u + D*dt*∇²u) / (1 + k*dt)
 *
 * This prevents oscillation when solution values become very small.
 */
class LinearReactionDiffusionSolver : public ExplicitSolverBase<LinearReactionDiffusionSolver> {
public:
    using Base = ExplicitSolverBase<LinearReactionDiffusionSolver>;
    friend Base;

    LinearReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity, double decay_rate)
        : Base(mesh, diffusivity), decay_rate_(decay_rate) {
        if (decay_rate_ < 0.0) {
            throw std::invalid_argument("Decay rate must be non-negative");
        }
        // Pre-compute denominator factor for implicit decay
        implicit_factor_ = 0.0;  // Will be set in solve() based on dt
    }

    void computeNodeUpdate(int idx, int /*i*/, int /*j*/, const StencilOps& ops, double dt) {
        double u = solution_[idx];
        double diffusion = ops.diffusionTerm(solution_, idx, diffusivity_, dt);
        // Implicit treatment: u_new * (1 + k*dt) = u + diffusion
        scratch_[idx] = (u + diffusion) / (1.0 + decay_rate_ * dt);
    }

    double decayRate() const { return decay_rate_; }

private:
    double decay_rate_;
    double implicit_factor_;
};

/**
 * @brief Logistic reaction-diffusion: ∂u/∂t = D∇²u + r*u*(1 - u/K)
 */
class LogisticReactionDiffusionSolver : public ExplicitSolverBase<LogisticReactionDiffusionSolver> {
public:
    using Base = ExplicitSolverBase<LogisticReactionDiffusionSolver>;
    friend Base;

    LogisticReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity,
                                    double growth_rate, double carrying_capacity)
        : Base(mesh, diffusivity),
          growth_rate_(growth_rate),
          carrying_capacity_(carrying_capacity) {
        if (growth_rate_ < 0.0) {
            throw std::invalid_argument("Growth rate must be non-negative");
        }
        if (carrying_capacity_ <= 0.0) {
            throw std::invalid_argument("Carrying capacity must be positive");
        }
    }

    void computeNodeUpdate(int idx, int /*i*/, int /*j*/, const StencilOps& ops, double dt) {
        double u = solution_[idx];
        double reaction = growth_rate_ * u * (1.0 - u / carrying_capacity_);
        scratch_[idx] = u + ops.diffusionTerm(solution_, idx, diffusivity_, dt) + dt * reaction;
    }

    double growthRate() const { return growth_rate_; }
    double carryingCapacity() const { return carrying_capacity_; }
    double time() const { return time_; }

private:
    double growth_rate_;
    double carrying_capacity_;
};

/**
 * @brief Michaelis-Menten reaction-diffusion: ∂u/∂t = D∇²u - Vmax*u/(Km + u)
 */
class MichaelisMentenReactionDiffusionSolver
    : public ExplicitSolverBase<MichaelisMentenReactionDiffusionSolver> {
public:
    using Base = ExplicitSolverBase<MichaelisMentenReactionDiffusionSolver>;
    friend Base;

    MichaelisMentenReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity,
                                           double vmax, double km)
        : Base(mesh, diffusivity), vmax_(vmax), km_(km) {
        if (vmax_ < 0.0) {
            throw std::invalid_argument("Vmax must be non-negative");
        }
        if (km_ <= 0.0) {
            throw std::invalid_argument("Km must be positive");
        }
    }

    void computeNodeUpdate(int idx, int /*i*/, int /*j*/, const StencilOps& ops, double dt) {
        double u = solution_[idx];
        double reaction = 0.0;
        if (u > 0.0 && vmax_ > 0.0) {
            reaction = -vmax_ * u / (km_ + u);
        }
        scratch_[idx] = u + ops.diffusionTerm(solution_, idx, diffusivity_, dt) + dt * reaction;
    }

    double vmax() const { return vmax_; }
    double km() const { return km_; }
    double time() const { return time_; }

private:
    double vmax_;
    double km_;
};

/**
 * @brief Constant source reaction-diffusion: ∂u/∂t = D∇²u + S
 */
class ConstantSourceReactionDiffusionSolver
    : public ExplicitSolverBase<ConstantSourceReactionDiffusionSolver> {
public:
    using Base = ExplicitSolverBase<ConstantSourceReactionDiffusionSolver>;
    friend Base;

    ConstantSourceReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity,
                                          double source_rate)
        : Base(mesh, diffusivity), source_rate_(source_rate) {}

    void computeNodeUpdate(int idx, int /*i*/, int /*j*/, const StencilOps& ops, double dt) {
        double u = solution_[idx];
        scratch_[idx] = u + ops.diffusionTerm(solution_, idx, diffusivity_, dt) + dt * source_rate_;
    }

    double sourceRate() const { return source_rate_; }
    double time() const { return time_; }

private:
    double source_rate_;
};

/**
 * @brief Masked Michaelis-Menten with pinned values in masked regions.
 */
class MaskedMichaelisMentenReactionDiffusionSolver
    : public ExplicitSolverBase<MaskedMichaelisMentenReactionDiffusionSolver> {
public:
    using Base = ExplicitSolverBase<MaskedMichaelisMentenReactionDiffusionSolver>;
    friend Base;

    MaskedMichaelisMentenReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity,
                                                 double vmax, double km,
                                                 std::vector<std::uint8_t> mask,
                                                 double pinned_value)
        : Base(mesh, diffusivity),
          vmax_(vmax),
          km_(km),
          mask_(std::move(mask)),
          pinned_value_(pinned_value) {
        if (vmax_ < 0.0 || !std::isfinite(vmax_)) {
            throw std::invalid_argument("Vmax must be non-negative and finite");
        }
        if (km_ <= 0.0 || !std::isfinite(km_)) {
            throw std::invalid_argument("Km must be positive and finite");
        }
        if (mask_.size() != solution_.size()) {
            throw std::invalid_argument("Mask size doesn't match mesh");
        }
    }

    void computeNodeUpdate(int idx, int /*i*/, int /*j*/, const StencilOps& ops, double dt) {
        if (mask_[idx] != 0) {
            scratch_[idx] = pinned_value_;
            return;
        }

        double u = solution_[idx];
        double reaction = -vmax_ * u / (km_ + u);
        scratch_[idx] = u + ops.diffusionTerm(solution_, idx, diffusivity_, dt) + dt * reaction;
    }

    void postStep(int step, double dt) {
        // Re-apply mask after boundary conditions
        for (std::size_t i = 0; i < mask_.size(); ++i) {
            if (mask_[i] != 0) {
                solution_[i] = pinned_value_;
            }
        }
        time_ += dt;
        (void)step;
    }

private:
    double vmax_;
    double km_;
    std::vector<std::uint8_t> mask_;
    double pinned_value_;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_DIFFUSION_SOLVERS_HPP
