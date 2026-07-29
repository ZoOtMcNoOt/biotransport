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
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace biotransport {

namespace legacy_reaction_detail {

inline void requireFinite(double value, const char* name) {
    if (!std::isfinite(value))
        throw std::invalid_argument(std::string(name) + " must be finite");
}

inline void requireNonnegativeFinite(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
    }
}

inline void requirePositiveFinite(double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
}

/**
 * @brief Safety base for legacy solvers whose state represents concentration.
 *
 * The public solve is transactional: invalid reaction updates or boundary
 * states restore both solution and time.  No negative value is clipped.
 */
template <typename Derived>
class ConcentrationExplicitSolverBase : public ExplicitSolverBase<Derived> {
public:
    using Parent = ExplicitSolverBase<Derived>;

    ConcentrationExplicitSolverBase(const StructuredMesh& mesh, double diffusivity,
                                    bool require_nonnegative = true)
        : Parent(mesh, diffusivity), require_nonnegative_(require_nonnegative) {}

    void setInitialCondition(const std::vector<double>& values) {
        if (require_nonnegative_)
            requireNonnegativeField(values, "Initial concentration");
        Parent::setInitialCondition(values);
    }

    void setDirichletBoundary(Boundary boundary, double value) {
        if (require_nonnegative_)
            requireNonnegativeFinite(value, "Dirichlet concentration");
        Parent::setDirichletBoundary(boundary, value);
    }

    void setDirichletBoundary(int boundary_id, double value) {
        if (require_nonnegative_)
            requireNonnegativeFinite(value, "Dirichlet concentration");
        Parent::setDirichletBoundary(boundary_id, value);
    }

    void setBoundaryCondition(Boundary boundary, const BoundaryCondition& condition) {
        validateConcentrationBoundary(condition);
        Parent::setBoundaryCondition(boundary, condition);
    }

    void setBoundaryCondition(int boundary_id, const BoundaryCondition& condition) {
        validateConcentrationBoundary(condition);
        Parent::setBoundaryCondition(boundary_id, condition);
    }

    void solve(double dt, int num_steps) {
        const std::vector<double> original_solution = this->solution_;
        const double original_time = this->time_;
        try {
            Parent::solve(dt, num_steps);
        } catch (...) {
            this->solution_ = original_solution;
            this->time_ = original_time;
            throw;
        }
    }

    bool requiresNonnegativeState() const { return require_nonnegative_; }

    // Called by ExplicitSolverBase after every accepted candidate swap.
    void postStep(int, double dt) {
        validateCurrentState("Updated concentration");
        this->time_ += dt;
    }

protected:
    bool require_nonnegative_;

    void setNonnegativeStatePolicy(bool required) {
        if (required)
            requireNonnegativeField(this->solution_, "Current concentration");
        require_nonnegative_ = required;
    }

    void validateCandidate(double candidate, const char* solver_name) const {
        if (!std::isfinite(candidate))
            throw std::runtime_error(std::string(solver_name) +
                                     " produced a non-finite concentration");
        if (require_nonnegative_ && candidate < 0.0) {
            throw std::runtime_error(std::string(solver_name) +
                                     " would produce a negative concentration; reduce the time "
                                     "step or revise the reaction/boundary data");
        }
    }

    void validateCurrentState(const char* name) const {
        for (std::size_t index = 0; index < this->solution_.size(); ++index) {
            const double value = this->solution_[index];
            if (!std::isfinite(value)) {
                throw std::runtime_error(std::string(name) +
                                         " contains a non-finite value at index " +
                                         std::to_string(index));
            }
            if (require_nonnegative_ && value < 0.0) {
                throw std::runtime_error(std::string(name) +
                                         " contains a negative value at index " +
                                         std::to_string(index));
            }
        }
    }

    template <typename Function>
    void forEachInteriorSerial(Function&& function) const {
        if (this->mesh_.is1D()) {
            for (int i = 1; i < this->mesh_.nx(); ++i)
                function(i, i, 0);
            return;
        }
        for (int j = 1; j < this->mesh_.ny(); ++j) {
            for (int i = 1; i < this->mesh_.nx(); ++i)
                function(this->mesh_.index(i, j), i, j);
        }
    }

private:
    static void requireNonnegativeField(const std::vector<double>& values, const char* name) {
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (!std::isfinite(values[index]) || values[index] < 0.0) {
                throw std::invalid_argument(std::string(name) +
                                            " must be finite and non-negative at index " +
                                            std::to_string(index));
            }
        }
    }

    void validateConcentrationBoundary(const BoundaryCondition& condition) const {
        if (require_nonnegative_ && condition.type == BoundaryType::DIRICHLET)
            requireNonnegativeFinite(condition.value, "Dirichlet concentration");
    }
};

}  // namespace legacy_reaction_detail

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
        scratch_[idx] = ops.diffusionStep(solution_, idx, diffusivity_, dt);
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
        : Base(mesh, validatedMaxDiffusivity(diffusivity_field,
                                             static_cast<std::size_t>(mesh.numNodes()))),
          diffusivity_field_(std::move(diffusivity_field)) {
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

    static double validatedMaxDiffusivity(const std::vector<double>& D_field,
                                          std::size_t expected_size) {
        if (D_field.empty()) {
            throw std::invalid_argument("Diffusivity field must not be empty");
        }
        if (D_field.size() != expected_size)
            throw std::invalid_argument("Diffusivity field size must match total nodes");
        double max_D = 0.0;
        for (double D : D_field) {
            if (!std::isfinite(D) || D < 0.0) {
                throw std::invalid_argument(
                    "Diffusivity must be finite and non-negative everywhere");
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
 * Callbacks are evaluated serially before each step and every returned rate
 * must be finite.  By default u is treated as a concentration: initial and
 * Dirichlet data must be non-negative and every complete Forward Euler
 * diffusion/reaction candidate is checked for positivity before mutation.
 * This state-aware check enforces a reaction time-step policy rather than
 * claiming that the diffusion CFL bound alone is sufficient.  It never clips.
 * C++ callers modeling a genuinely signed scalar may explicitly opt out with
 * setRequireNonnegativeState(false).
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
class ReactionDiffusionSolver
    : public legacy_reaction_detail::ConcentrationExplicitSolverBase<ReactionDiffusionSolver> {
public:
    using Base = legacy_reaction_detail::ConcentrationExplicitSolverBase<ReactionDiffusionSolver>;
    using EngineBase = ExplicitSolverBase<ReactionDiffusionSolver>;
    using ReactionFunction = std::function<double(double u, double x, double y, double t)>;
    friend EngineBase;

    ReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity,
                            ReactionFunction reaction, bool require_nonnegative_state = true)
        : Base(mesh, diffusivity, require_nonnegative_state),
          reaction_(std::move(reaction)),
          reaction_rates_(static_cast<std::size_t>(mesh.numNodes()), 0.0) {
        if (!reaction_)
            throw std::invalid_argument("Reaction callback must be callable");
        // Pre-cache coordinates for performance
        cacheCoordinates();
    }

    ReactionDiffusionSolver& setRequireNonnegativeState(bool required) {
        this->setNonnegativeStatePolicy(required);
        return *this;
    }

    void computeNodeUpdate(int idx, int i, int j, const StencilOps& ops, double dt) {
        (void)i;
        (void)j;
        const double candidate = this->solution_[idx] +
                                 ops.diffusionTerm(this->solution_, idx, this->diffusivity_, dt) +
                                 dt * reaction_rates_[static_cast<std::size_t>(idx)];
        // preStep validated the identical serial candidate before the possibly
        // OpenMP-parallel update loop.
        this->scratch_[idx] = candidate;
    }

private:
    ReactionFunction reaction_;
    std::vector<double> reaction_rates_;
    std::vector<double> x_coords_;
    std::vector<double> y_coords_;

    void preStep(int, double dt) {
        this->validateCurrentState("Current concentration");
        this->forEachInteriorSerial([&](int idx, int i, int j) {
            const double u = this->solution_[idx];
            const double x = x_coords_[i];
            const double y = this->mesh_.is1D() ? 0.0 : y_coords_[j];
            const double rate = reaction_(u, x, y, this->time_);
            if (!std::isfinite(rate))
                throw std::runtime_error("Reaction callback returned a non-finite rate");
            reaction_rates_[static_cast<std::size_t>(idx)] = rate;
            const double candidate =
                u + this->stencil_ops_.diffusionTerm(this->solution_, idx, this->diffusivity_, dt) +
                dt * rate;
            this->validateCandidate(candidate, "ReactionDiffusionSolver");
        });
    }

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
class LinearReactionDiffusionSolver
    : public legacy_reaction_detail::ConcentrationExplicitSolverBase<
          LinearReactionDiffusionSolver> {
public:
    using Base =
        legacy_reaction_detail::ConcentrationExplicitSolverBase<LinearReactionDiffusionSolver>;
    using EngineBase = ExplicitSolverBase<LinearReactionDiffusionSolver>;
    friend EngineBase;

    LinearReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity, double decay_rate)
        : Base(mesh, diffusivity), decay_rate_(decay_rate) {
        legacy_reaction_detail::requireNonnegativeFinite(decay_rate_, "Decay rate");
    }

    void computeNodeUpdate(int idx, int /*i*/, int /*j*/, const StencilOps& ops, double dt) {
        const double u = this->solution_[idx];
        const double diffusion = ops.diffusionTerm(this->solution_, idx, this->diffusivity_, dt);
        // Implicit treatment: u_new * (1 + k*dt) = u + diffusion
        this->scratch_[idx] = (u + diffusion) / (1.0 + decay_rate_ * dt);
    }

    double decayRate() const { return decay_rate_; }

private:
    double decay_rate_;

    void preStep(int, double dt) {
        this->validateCurrentState("Current concentration");
        this->forEachInteriorSerial([&](int idx, int, int) {
            const double candidate =
                (this->solution_[idx] +
                 this->stencil_ops_.diffusionTerm(this->solution_, idx, this->diffusivity_, dt)) /
                (1.0 + decay_rate_ * dt);
            this->validateCandidate(candidate, "LinearReactionDiffusionSolver");
        });
    }
};

/**
 * @brief Logistic reaction-diffusion: ∂u/∂t = D∇²u + r*u*(1 - u/K)
 */
class LogisticReactionDiffusionSolver
    : public legacy_reaction_detail::ConcentrationExplicitSolverBase<
          LogisticReactionDiffusionSolver> {
public:
    using Base =
        legacy_reaction_detail::ConcentrationExplicitSolverBase<LogisticReactionDiffusionSolver>;
    using EngineBase = ExplicitSolverBase<LogisticReactionDiffusionSolver>;
    friend EngineBase;

    LogisticReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity,
                                    double growth_rate, double carrying_capacity)
        : Base(mesh, diffusivity),
          growth_rate_(growth_rate),
          carrying_capacity_(carrying_capacity) {
        legacy_reaction_detail::requireNonnegativeFinite(growth_rate_, "Growth rate");
        legacy_reaction_detail::requirePositiveFinite(carrying_capacity_, "Carrying capacity");
    }

    void computeNodeUpdate(int idx, int /*i*/, int /*j*/, const StencilOps& ops, double dt) {
        const double u = this->solution_[idx];
        const double reaction = growth_rate_ * u * (1.0 - u / carrying_capacity_);
        this->scratch_[idx] =
            u + ops.diffusionTerm(this->solution_, idx, this->diffusivity_, dt) + dt * reaction;
    }

    double growthRate() const { return growth_rate_; }
    double carryingCapacity() const { return carrying_capacity_; }
    double time() const { return time_; }

private:
    double growth_rate_;
    double carrying_capacity_;

    void preStep(int, double dt) {
        this->validateCurrentState("Current concentration");
        this->forEachInteriorSerial([&](int idx, int, int) {
            const double u = this->solution_[idx];
            const double reaction = growth_rate_ * u * (1.0 - u / carrying_capacity_);
            const double candidate =
                u + this->stencil_ops_.diffusionTerm(this->solution_, idx, this->diffusivity_, dt) +
                dt * reaction;
            this->validateCandidate(candidate, "LogisticReactionDiffusionSolver");
        });
    }
};

/**
 * @brief Michaelis-Menten reaction-diffusion: ∂u/∂t = D∇²u - Vmax*u/(Km + u)
 */
class MichaelisMentenReactionDiffusionSolver
    : public legacy_reaction_detail::ConcentrationExplicitSolverBase<
          MichaelisMentenReactionDiffusionSolver> {
public:
    using Base = legacy_reaction_detail::ConcentrationExplicitSolverBase<
        MichaelisMentenReactionDiffusionSolver>;
    using EngineBase = ExplicitSolverBase<MichaelisMentenReactionDiffusionSolver>;
    friend EngineBase;

    MichaelisMentenReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity,
                                           double vmax, double km)
        : Base(mesh, diffusivity), vmax_(vmax), km_(km) {
        legacy_reaction_detail::requireNonnegativeFinite(vmax_, "Vmax");
        legacy_reaction_detail::requirePositiveFinite(km_, "Km");
    }

    void computeNodeUpdate(int idx, int /*i*/, int /*j*/, const StencilOps& ops, double dt) {
        const double u = this->solution_[idx];
        const double denominator = km_ + u;
        // u is guaranteed non-negative and Km positive, so the denominator
        // cannot be singular.  u==0 gives the exact rate zero without clipping.
        const double reaction = -vmax_ * u / denominator;
        this->scratch_[idx] =
            u + ops.diffusionTerm(this->solution_, idx, this->diffusivity_, dt) + dt * reaction;
    }

    double vmax() const { return vmax_; }
    double km() const { return km_; }
    double time() const { return time_; }

private:
    double vmax_;
    double km_;

    void preStep(int, double dt) {
        this->validateCurrentState("Current concentration");
        this->forEachInteriorSerial([&](int idx, int, int) {
            const double u = this->solution_[idx];
            const double denominator = km_ + u;
            if (!std::isfinite(denominator) || denominator <= 0.0) {
                throw std::runtime_error(
                    "Michaelis-Menten denominator Km + u must be finite and positive");
            }
            const double reaction = -vmax_ * u / denominator;
            const double candidate =
                u + this->stencil_ops_.diffusionTerm(this->solution_, idx, this->diffusivity_, dt) +
                dt * reaction;
            this->validateCandidate(candidate, "MichaelisMentenReactionDiffusionSolver");
        });
    }
};

/**
 * @brief Constant source reaction-diffusion: ∂u/∂t = D∇²u + S
 */
class ConstantSourceReactionDiffusionSolver
    : public legacy_reaction_detail::ConcentrationExplicitSolverBase<
          ConstantSourceReactionDiffusionSolver> {
public:
    using Base = legacy_reaction_detail::ConcentrationExplicitSolverBase<
        ConstantSourceReactionDiffusionSolver>;
    using EngineBase = ExplicitSolverBase<ConstantSourceReactionDiffusionSolver>;
    friend EngineBase;

    ConstantSourceReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity,
                                          double source_rate)
        : Base(mesh, diffusivity), source_rate_(source_rate) {
        legacy_reaction_detail::requireFinite(source_rate_, "Source rate");
    }

    void computeNodeUpdate(int idx, int /*i*/, int /*j*/, const StencilOps& ops, double dt) {
        const double u = this->solution_[idx];
        this->scratch_[idx] =
            u + ops.diffusionTerm(this->solution_, idx, this->diffusivity_, dt) + dt * source_rate_;
    }

    double sourceRate() const { return source_rate_; }
    double time() const { return time_; }

private:
    double source_rate_;

    void preStep(int, double dt) {
        this->validateCurrentState("Current concentration");
        this->forEachInteriorSerial([&](int idx, int, int) {
            const double candidate =
                this->solution_[idx] +
                this->stencil_ops_.diffusionTerm(this->solution_, idx, this->diffusivity_, dt) +
                dt * source_rate_;
            this->validateCandidate(candidate, "ConstantSourceReactionDiffusionSolver");
        });
    }
};

/**
 * @brief Masked Michaelis-Menten with pinned values in masked regions.
 */
class MaskedMichaelisMentenReactionDiffusionSolver
    : public legacy_reaction_detail::ConcentrationExplicitSolverBase<
          MaskedMichaelisMentenReactionDiffusionSolver> {
public:
    using Base = legacy_reaction_detail::ConcentrationExplicitSolverBase<
        MaskedMichaelisMentenReactionDiffusionSolver>;
    using EngineBase = ExplicitSolverBase<MaskedMichaelisMentenReactionDiffusionSolver>;
    friend EngineBase;

    MaskedMichaelisMentenReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity,
                                                 double vmax, double km,
                                                 std::vector<std::uint8_t> mask,
                                                 double pinned_value)
        : Base(mesh, diffusivity),
          vmax_(vmax),
          km_(km),
          mask_(std::move(mask)),
          pinned_value_(pinned_value) {
        legacy_reaction_detail::requireNonnegativeFinite(vmax_, "Vmax");
        legacy_reaction_detail::requirePositiveFinite(km_, "Km");
        legacy_reaction_detail::requireNonnegativeFinite(pinned_value_, "Pinned concentration");
        if (mask_.size() != this->solution_.size()) {
            throw std::invalid_argument("Mask size doesn't match mesh");
        }
    }

    void computeNodeUpdate(int idx, int /*i*/, int /*j*/, const StencilOps& ops, double dt) {
        if (mask_[idx] != 0) {
            this->scratch_[idx] = pinned_value_;
            return;
        }

        const double u = this->solution_[idx];
        const double reaction = -vmax_ * u / (km_ + u);
        this->scratch_[idx] =
            u + ops.diffusionTerm(this->solution_, idx, this->diffusivity_, dt) + dt * reaction;
    }

    void preStep(int, double dt) {
        this->validateCurrentState("Current concentration");
        this->forEachInteriorSerial([&](int idx, int, int) {
            if (mask_[static_cast<std::size_t>(idx)] != 0)
                return;
            const double u = this->solution_[idx];
            const double denominator = km_ + u;
            if (!std::isfinite(denominator) || denominator <= 0.0) {
                throw std::runtime_error(
                    "Masked Michaelis-Menten denominator Km + u must be finite and positive");
            }
            const double reaction = -vmax_ * u / denominator;
            const double candidate =
                u + this->stencil_ops_.diffusionTerm(this->solution_, idx, this->diffusivity_, dt) +
                dt * reaction;
            this->validateCandidate(candidate, "MaskedMichaelisMentenReactionDiffusionSolver");
        });
    }

    void postStep(int step, double dt) {
        // Re-apply mask after boundary conditions
        for (std::size_t i = 0; i < mask_.size(); ++i) {
            if (mask_[i] != 0) {
                this->solution_[i] = pinned_value_;
            }
        }
        this->validateCurrentState("Updated concentration");
        this->time_ += dt;
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
