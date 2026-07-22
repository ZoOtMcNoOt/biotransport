/**
 * @file diffusion_solver_3d.hpp
 * @brief Conservative explicit 3D diffusion and reaction--diffusion solvers.
 *
 * The discretization is vertex-centred finite volume on a uniform Cartesian
 * mesh.  Boundary nodes own half control volumes, edges own quarter control
 * volumes, and corners own eighth control volumes.  Internal diffusive fluxes
 * cancel pairwise, so homogeneous Neumann data conserve the trapezoidal-volume
 * integral up to roundoff.  Neumann values are outward derivatives du/dn.
 * Dirichlet traces meeting on an edge or corner must agree within
 * 64*epsilon*max(1, |a|, |b|); contradictory traces are rejected before any
 * public state is changed.
 */

#ifndef BIOTRANSPORT_SOLVERS_DIFFUSION_SOLVER_3D_HPP
#define BIOTRANSPORT_SOLVERS_DIFFUSION_SOLVER_3D_HPP

#include <algorithm>
#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh_3d.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace biotransport {

namespace legacy_reaction_3d_detail {

inline void requireNonnegativeField(const std::vector<double>& values, const char* name) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index]) || values[index] < 0.0) {
            throw std::invalid_argument(std::string(name) +
                                        " must be finite and non-negative at index " +
                                        std::to_string(index));
        }
    }
}

inline void requireNonnegativeFinite(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
}

inline void validateCandidate(double value, bool require_nonnegative, const char* solver_name) {
    if (!std::isfinite(value))
        throw std::runtime_error(std::string(solver_name) + " produced a non-finite concentration");
    if (require_nonnegative && value < 0.0) {
        throw std::runtime_error(std::string(solver_name) +
                                 " would produce a negative concentration; reduce the time step "
                                 "or revise the reaction/boundary data");
    }
}

inline void validateState(const std::vector<double>& values, bool require_nonnegative,
                          const char* name) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            throw std::runtime_error(std::string(name) + " contains a non-finite value at index " +
                                     std::to_string(index));
        }
        if (require_nonnegative && values[index] < 0.0) {
            throw std::runtime_error(std::string(name) + " contains a negative value at index " +
                                     std::to_string(index));
        }
    }
}

}  // namespace legacy_reaction_3d_detail

template <typename Derived>
class ExplicitSolverBase3D {
public:
    ExplicitSolverBase3D(const StructuredMesh3D& mesh, double diffusivity)
        : mesh_(mesh),
          diffusivity_(diffusivity),
          solution_(mesh.numNodes(), 0.0),
          scratch_(mesh.numNodes(), 0.0) {
        requirePositiveFinite(diffusivity, "Diffusivity");
        boundary_conditions_.fill(BoundaryCondition::Dirichlet(0.0));
    }

    void setInitialCondition(const std::vector<double>& values) {
        if (values.size() != solution_.size()) {
            throw std::invalid_argument("Initial condition size does not match mesh");
        }
        requireFiniteInput(values, "Initial condition");
        solution_ = values;
    }

    void setDirichletBoundary(Boundary3D boundary, double value) {
        requireFinite(value, "Dirichlet value");
        boundary_conditions_[checkedIndex(boundary)] = BoundaryCondition::Dirichlet(value);
    }

    void setDirichletBoundary(int boundary_id, double value) {
        setDirichletBoundary(checkedBoundary(boundary_id), value);
    }

    void setNeumannBoundary(Boundary3D boundary, double normal_derivative) {
        requireFinite(normal_derivative, "Neumann outward-normal derivative");
        boundary_conditions_[checkedIndex(boundary)] =
            BoundaryCondition::Neumann(normal_derivative);
    }

    void setNeumannBoundary(int boundary_id, double normal_derivative) {
        setNeumannBoundary(checkedBoundary(boundary_id), normal_derivative);
    }

    void setBoundaryCondition(Boundary3D boundary, const BoundaryCondition& condition) {
        (void)checkedIndex(boundary);
        if (condition.type == BoundaryType::ROBIN) {
            throw std::invalid_argument(
                "ExplicitSolverBase3D does not implement Robin boundaries; use Dirichlet or "
                "outward-derivative Neumann data");
        }
        requireFinite(condition.value, "Boundary value");
        boundary_conditions_[checkedIndex(boundary)] = condition;
    }

    void setBoundaryCondition(int boundary_id, const BoundaryCondition& condition) {
        setBoundaryCondition(checkedBoundary(boundary_id), condition);
    }

    /**
     * @brief Diffusion-only Forward Euler stability certificate.
     *
     * For a generic reaction callback this does not certify reaction stability.
     */
    bool checkStability(double dt) const {
        return std::isfinite(dt) && dt > 0.0 && dt <= maxStableTimeStep();
    }

    double maxStableTimeStep() const {
        return 1.0 / (2.0 * diffusivity_ *
                      (1.0 / (mesh_.dx() * mesh_.dx()) + 1.0 / (mesh_.dy() * mesh_.dy()) +
                       1.0 / (mesh_.dz() * mesh_.dz())));
    }

    void solve(double dt, int num_steps) {
        requirePositiveFinite(dt, "Time step");
        if (num_steps < 0) {
            throw std::invalid_argument("Number of steps must be non-negative");
        }
        if (!checkStability(dt)) {
            throw std::invalid_argument("Time step exceeds the 3D explicit diffusion limit of " +
                                        std::to_string(maxStableTimeStep()));
        }

        validateDirichletCompatibility();
        imposeDirichlet(solution_);
        for (int step_index = 0; step_index < num_steps; ++step_index) {
            derived().preStep(step_index, dt);
            for (int k = 0; k <= mesh_.nz(); ++k) {
                for (int j = 0; j <= mesh_.ny(); ++j) {
                    for (int i = 0; i <= mesh_.nx(); ++i) {
                        const std::size_t index = nodeIndex(i, j, k);
                        if (const auto fixed = dirichletValue(i, j, k)) {
                            scratch_[index] = *fixed;
                            continue;
                        }
                        derived().computeNodeUpdate(index, i, j, k, diffusionRate(i, j, k), dt);
                    }
                }
            }
            requireFiniteResult(scratch_, "3D explicit solution");
            solution_.swap(scratch_);
            derived().postStep(step_index, dt);
        }
    }

    const std::vector<double>& solution() const { return solution_; }
    const StructuredMesh3D& mesh() const { return mesh_; }
    double diffusivity() const { return diffusivity_; }
    double time() const { return time_; }

protected:
    const StructuredMesh3D& mesh_;
    double diffusivity_;
    std::vector<double> solution_;
    std::vector<double> scratch_;
    std::array<BoundaryCondition, 6> boundary_conditions_;
    double time_ = 0.0;

    void preStep(int, double) {}
    void postStep(int, double dt) { time_ += dt; }

private:
    Derived& derived() { return static_cast<Derived&>(*this); }

    static void requirePositiveFinite(double value, const char* name) {
        if (!std::isfinite(value) || value <= 0.0)
            throw std::invalid_argument(std::string(name) + " must be finite and positive");
    }

    static void requireFinite(double value, const char* name) {
        if (!std::isfinite(value))
            throw std::invalid_argument(std::string(name) + " must be finite");
    }

    static void requireFiniteInput(const std::vector<double>& values, const char* name) {
        for (double value : values) {
            requireFinite(value, name);
        }
    }

    static void requireFiniteResult(const std::vector<double>& values, const char* name) {
        for (double value : values) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(std::string(name) + " contains a non-finite value");
            }
        }
    }

    static Boundary3D checkedBoundary(int boundary_id) {
        if (boundary_id < 0 || boundary_id >= 6)
            throw std::invalid_argument("3D boundary identifier is outside [0, 5]");
        return static_cast<Boundary3D>(boundary_id);
    }

    static std::size_t checkedIndex(Boundary3D boundary) {
        const int index = to_index(boundary);
        if (index < 0 || index >= 6)
            throw std::invalid_argument("3D boundary identifier is outside [0, 5]");
        return static_cast<std::size_t>(index);
    }

    std::size_t nodeIndex(int i, int j, int k) const {
        return static_cast<std::size_t>(mesh_.index(i, j, k));
    }

    std::optional<double> dirichletValue(int i, int j, int k) const {
        double sum = 0.0;
        int count = 0;
        double reference = 0.0;
        const auto add = [&](Boundary3D face) {
            const auto& bc = boundary_conditions_[checkedIndex(face)];
            if (bc.type == BoundaryType::DIRICHLET) {
                if (count == 0) {
                    reference = bc.value;
                } else {
                    const double scale = std::max({1.0, std::abs(reference), std::abs(bc.value)});
                    if (std::abs(reference - bc.value) >
                        64.0 * std::numeric_limits<double>::epsilon() * scale) {
                        throw std::invalid_argument(
                            "Conflicting Dirichlet values meet at a three-dimensional edge or "
                            "corner");
                    }
                }
                sum += bc.value;
                ++count;
            }
        };
        if (i == 0)
            add(Boundary3D::XMin);
        if (i == mesh_.nx())
            add(Boundary3D::XMax);
        if (j == 0)
            add(Boundary3D::YMin);
        if (j == mesh_.ny())
            add(Boundary3D::YMax);
        if (k == 0)
            add(Boundary3D::ZMin);
        if (k == mesh_.nz())
            add(Boundary3D::ZMax);
        if (count == 0)
            return std::nullopt;
        return sum / static_cast<double>(count);
    }

    void validateDirichletCompatibility() const {
        for (int k = 0; k <= mesh_.nz(); ++k) {
            for (int j = 0; j <= mesh_.ny(); ++j) {
                for (int i = 0; i <= mesh_.nx(); ++i)
                    (void)dirichletValue(i, j, k);
            }
        }
    }

    void imposeDirichlet(std::vector<double>& values) const {
        for (int k = 0; k <= mesh_.nz(); ++k) {
            for (int j = 0; j <= mesh_.ny(); ++j) {
                for (int i = 0; i <= mesh_.nx(); ++i) {
                    if (const auto fixed = dirichletValue(i, j, k))
                        values[nodeIndex(i, j, k)] = *fixed;
                }
            }
        }
    }

    double xWidth(int i) const { return mesh_.dx() * ((i == 0 || i == mesh_.nx()) ? 0.5 : 1.0); }
    double yWidth(int j) const { return mesh_.dy() * ((j == 0 || j == mesh_.ny()) ? 0.5 : 1.0); }
    double zWidth(int k) const { return mesh_.dz() * ((k == 0 || k == mesh_.nz()) ? 0.5 : 1.0); }

    double diffusionRate(int i, int j, int k) const {
        const std::size_t index = nodeIndex(i, j, k);
        const double value = solution_[index];
        const double wx = xWidth(i);
        const double wy = yWidth(j);
        const double wz = zWidth(k);
        const double volume = wx * wy * wz;
        double integrated_flux = 0.0;

        const double gx = diffusivity_ * wy * wz / mesh_.dx();
        if (i > 0)
            integrated_flux += gx * (solution_[nodeIndex(i - 1, j, k)] - value);
        if (i < mesh_.nx())
            integrated_flux += gx * (solution_[nodeIndex(i + 1, j, k)] - value);

        const double gy = diffusivity_ * wx * wz / mesh_.dy();
        if (j > 0)
            integrated_flux += gy * (solution_[nodeIndex(i, j - 1, k)] - value);
        if (j < mesh_.ny())
            integrated_flux += gy * (solution_[nodeIndex(i, j + 1, k)] - value);

        const double gz = diffusivity_ * wx * wy / mesh_.dz();
        if (k > 0)
            integrated_flux += gz * (solution_[nodeIndex(i, j, k - 1)] - value);
        if (k < mesh_.nz())
            integrated_flux += gz * (solution_[nodeIndex(i, j, k + 1)] - value);

        const auto add_boundary_flux = [&](Boundary3D face, double area) {
            const auto& bc = boundary_conditions_[checkedIndex(face)];
            if (bc.type == BoundaryType::NEUMANN)
                integrated_flux += diffusivity_ * bc.value * area;
        };
        if (i == 0)
            add_boundary_flux(Boundary3D::XMin, wy * wz);
        if (i == mesh_.nx())
            add_boundary_flux(Boundary3D::XMax, wy * wz);
        if (j == 0)
            add_boundary_flux(Boundary3D::YMin, wx * wz);
        if (j == mesh_.ny())
            add_boundary_flux(Boundary3D::YMax, wx * wz);
        if (k == 0)
            add_boundary_flux(Boundary3D::ZMin, wx * wy);
        if (k == mesh_.nz())
            add_boundary_flux(Boundary3D::ZMax, wx * wy);

        return integrated_flux / volume;
    }
};

class DiffusionSolver3D : public ExplicitSolverBase3D<DiffusionSolver3D> {
public:
    using Base = ExplicitSolverBase3D<DiffusionSolver3D>;
    friend Base;

    DiffusionSolver3D(const StructuredMesh3D& mesh, double diffusivity) : Base(mesh, diffusivity) {}

private:
    void computeNodeUpdate(std::size_t index, int, int, int, double diffusion_rate, double dt) {
        scratch_[index] = solution_[index] + dt * diffusion_rate;
    }
};

/**
 * @brief IMEX solver for du/dt = D laplacian(u) - k u.
 *
 * Diffusion is Forward Euler and decay is Backward Euler.  The method is first
 * order in time; the diffusion CFL limit still applies.  This concentration
 * solver rejects negative initial/Dirichlet data and transactionally rejects
 * any negative or non-finite complete update; it never clips values.
 */
class LinearReactionDiffusionSolver3D
    : public ExplicitSolverBase3D<LinearReactionDiffusionSolver3D> {
public:
    using Base = ExplicitSolverBase3D<LinearReactionDiffusionSolver3D>;
    friend Base;

    LinearReactionDiffusionSolver3D(const StructuredMesh3D& mesh, double diffusivity,
                                    double decay_rate)
        : Base(mesh, diffusivity), decay_rate_(decay_rate) {
        if (!std::isfinite(decay_rate_) || decay_rate_ < 0.0)
            throw std::invalid_argument("Decay rate must be finite and non-negative");
    }

    void setInitialCondition(const std::vector<double>& values) {
        legacy_reaction_3d_detail::requireNonnegativeField(values, "Initial concentration");
        Base::setInitialCondition(values);
    }

    void setDirichletBoundary(Boundary3D boundary, double value) {
        legacy_reaction_3d_detail::requireNonnegativeFinite(value, "Dirichlet concentration");
        Base::setDirichletBoundary(boundary, value);
    }

    void setDirichletBoundary(int boundary_id, double value) {
        legacy_reaction_3d_detail::requireNonnegativeFinite(value, "Dirichlet concentration");
        Base::setDirichletBoundary(boundary_id, value);
    }

    void setBoundaryCondition(Boundary3D boundary, const BoundaryCondition& condition) {
        if (condition.type == BoundaryType::DIRICHLET)
            legacy_reaction_3d_detail::requireNonnegativeFinite(condition.value,
                                                                "Dirichlet concentration");
        Base::setBoundaryCondition(boundary, condition);
    }

    void setBoundaryCondition(int boundary_id, const BoundaryCondition& condition) {
        if (condition.type == BoundaryType::DIRICHLET)
            legacy_reaction_3d_detail::requireNonnegativeFinite(condition.value,
                                                                "Dirichlet concentration");
        Base::setBoundaryCondition(boundary_id, condition);
    }

    void solve(double dt, int num_steps) {
        const std::vector<double> original_solution = this->solution_;
        const double original_time = this->time_;
        try {
            Base::solve(dt, num_steps);
        } catch (...) {
            this->solution_ = original_solution;
            this->time_ = original_time;
            throw;
        }
    }

    double decayRate() const { return decay_rate_; }

private:
    double decay_rate_;

    void computeNodeUpdate(std::size_t index, int, int, int, double diffusion_rate, double dt) {
        const double candidate =
            (this->solution_[index] + dt * diffusion_rate) / (1.0 + decay_rate_ * dt);
        legacy_reaction_3d_detail::validateCandidate(candidate, true,
                                                     "LinearReactionDiffusionSolver3D");
        this->scratch_[index] = candidate;
    }

    void postStep(int, double dt) {
        legacy_reaction_3d_detail::validateState(this->solution_, true, "Updated concentration");
        this->time_ += dt;
    }
};

/**
 * @brief Explicit solver for du/dt = D laplacian(u) + R(u,x,y,z,t).
 *
 * maxStableTimeStep() certifies only the diffusion term.  Every callback rate
 * must be finite and every proposed complete update is checked before the
 * public state advances.  The default concentration policy transactionally
 * rejects negative initial, Dirichlet, boundary-result, and update values, so
 * a reaction-unsafe time step fails rather than being clipped.  C++ callers
 * modeling a signed scalar may explicitly disable that policy.
 */
class ReactionDiffusionSolver3D : public ExplicitSolverBase3D<ReactionDiffusionSolver3D> {
public:
    using Base = ExplicitSolverBase3D<ReactionDiffusionSolver3D>;
    using ReactionFunction =
        std::function<double(double u, double x, double y, double z, double t)>;
    friend Base;

    ReactionDiffusionSolver3D(const StructuredMesh3D& mesh, double diffusivity,
                              ReactionFunction reaction, bool require_nonnegative_state = true)
        : Base(mesh, diffusivity),
          reaction_(std::move(reaction)),
          require_nonnegative_state_(require_nonnegative_state) {
        if (!reaction_)
            throw std::invalid_argument("Reaction callback must be callable");
        cacheCoordinates();
    }

    void setInitialCondition(const std::vector<double>& values) {
        if (require_nonnegative_state_)
            legacy_reaction_3d_detail::requireNonnegativeField(values, "Initial concentration");
        Base::setInitialCondition(values);
    }

    void setDirichletBoundary(Boundary3D boundary, double value) {
        if (require_nonnegative_state_)
            legacy_reaction_3d_detail::requireNonnegativeFinite(value, "Dirichlet concentration");
        Base::setDirichletBoundary(boundary, value);
    }

    void setDirichletBoundary(int boundary_id, double value) {
        if (require_nonnegative_state_)
            legacy_reaction_3d_detail::requireNonnegativeFinite(value, "Dirichlet concentration");
        Base::setDirichletBoundary(boundary_id, value);
    }

    void setBoundaryCondition(Boundary3D boundary, const BoundaryCondition& condition) {
        validateBoundary(condition);
        Base::setBoundaryCondition(boundary, condition);
    }

    void setBoundaryCondition(int boundary_id, const BoundaryCondition& condition) {
        validateBoundary(condition);
        Base::setBoundaryCondition(boundary_id, condition);
    }

    ReactionDiffusionSolver3D& setRequireNonnegativeState(bool required) {
        if (required)
            legacy_reaction_3d_detail::requireNonnegativeField(this->solution_,
                                                               "Current concentration");
        require_nonnegative_state_ = required;
        return *this;
    }

    bool requiresNonnegativeState() const { return require_nonnegative_state_; }

    void solve(double dt, int num_steps) {
        const std::vector<double> original_solution = this->solution_;
        const double original_time = this->time_;
        try {
            Base::solve(dt, num_steps);
        } catch (...) {
            this->solution_ = original_solution;
            this->time_ = original_time;
            throw;
        }
    }

private:
    ReactionFunction reaction_;
    std::vector<double> x_coordinates_;
    std::vector<double> y_coordinates_;
    std::vector<double> z_coordinates_;
    bool require_nonnegative_state_;

    void computeNodeUpdate(std::size_t index, int i, int j, int k, double diffusion_rate,
                           double dt) {
        const double reaction = reaction_(solution_[index], x_coordinates_[i], y_coordinates_[j],
                                          z_coordinates_[k], time_);
        if (!std::isfinite(reaction))
            throw std::runtime_error("Reaction callback returned a non-finite rate");
        const double candidate = this->solution_[index] + dt * (diffusion_rate + reaction);
        legacy_reaction_3d_detail::validateCandidate(candidate, require_nonnegative_state_,
                                                     "ReactionDiffusionSolver3D");
        this->scratch_[index] = candidate;
    }

    void postStep(int, double dt) {
        legacy_reaction_3d_detail::validateState(this->solution_, require_nonnegative_state_,
                                                 "Updated concentration");
        this->time_ += dt;
    }

    void validateBoundary(const BoundaryCondition& condition) const {
        if (require_nonnegative_state_ && condition.type == BoundaryType::DIRICHLET) {
            legacy_reaction_3d_detail::requireNonnegativeFinite(condition.value,
                                                                "Dirichlet concentration");
        }
    }

    void cacheCoordinates() {
        x_coordinates_.resize(mesh_.nx() + 1);
        y_coordinates_.resize(mesh_.ny() + 1);
        z_coordinates_.resize(mesh_.nz() + 1);
        for (int i = 0; i <= mesh_.nx(); ++i)
            x_coordinates_[i] = mesh_.x(i);
        for (int j = 0; j <= mesh_.ny(); ++j)
            y_coordinates_[j] = mesh_.y(j);
        for (int k = 0; k <= mesh_.nz(); ++k)
            z_coordinates_[k] = mesh_.z(k);
    }
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_DIFFUSION_SOLVER_3D_HPP
