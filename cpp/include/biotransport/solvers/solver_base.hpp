#ifndef BIOTRANSPORT_SOLVERS_SOLVER_BASE_HPP
#define BIOTRANSPORT_SOLVERS_SOLVER_BASE_HPP

/**
 * @file solver_base.hpp
 * @brief CRTP base class for finite difference solvers.
 *
 * This header provides a unified template-based framework for time-stepping
 * solvers, eliminating the massive code duplication across diffusion,
 * reaction-diffusion, and advection-diffusion solvers.
 *
 * The key insight is that all these solvers share the same structure:
 *   1. Setup (check stability, resize scratch buffers)
 *   2. Time loop
 *      a. Compute spatial terms (diffusion, advection)
 *      b. Add physics-specific terms (reaction, source)
 *      c. Apply boundary conditions
 *      d. Swap buffers
 *   3. Cleanup
 *
 * Using CRTP, derived classes only need to implement their specific physics
 * via a `computeUpdate()` method, while all the boilerplate is handled here.
 */

#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/mesh_iterators.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace biotransport {

/**
 * @brief CRTP base class for explicit time-stepping solvers.
 *
 * @tparam Derived The derived solver class (CRTP pattern)
 *
 * Derived classes must implement:
 * - `void computeNodeUpdate(int idx, const StencilOps& ops, double dt)`
 *   Sets scratch_[idx] to the new value for that node.
 *
 * Optionally override:
 * - `bool checkStabilityDerived(double dt) const` for custom stability checks
 * - `void preStep(int step, double dt)` called before each time step
 * - `void postStep(int step, double dt)` called after each time step
 */
template <typename Derived>
class ExplicitSolverBase {
public:
    ExplicitSolverBase(const StructuredMesh& mesh, double diffusivity)
        : mesh_(mesh), diffusivity_(diffusivity), iterator_(mesh), stencil_ops_(mesh) {
        if (diffusivity <= 0.0) {
            throw std::invalid_argument("Diffusivity must be positive");
        }
        solution_.resize(mesh.numNodes(), 0.0);
        scratch_.resize(solution_.size(), 0.0);

        // Default boundary conditions (Dirichlet, value = 0)
        for (int i = 0; i < 4; ++i) {
            boundary_conditions_[i] = BoundaryCondition::Dirichlet(0.0);
        }
    }

    /**
     * @brief Set the initial condition.
     */
    void setInitialCondition(const std::vector<double>& values) {
        if (values.size() != solution_.size()) {
            throw std::invalid_argument("Initial condition size doesn't match mesh");
        }
        solution_ = values;
    }

    /**
     * @brief Set a Dirichlet boundary condition.
     */
    void setDirichletBoundary(Boundary boundary, double value) {
        boundary_conditions_[to_index(boundary)] = BoundaryCondition::Dirichlet(value);
    }

    void setDirichletBoundary(int boundary_id, double value) {
        setDirichletBoundary(static_cast<Boundary>(boundary_id), value);
    }

    /**
     * @brief Set a Neumann boundary condition.
     */
    void setNeumannBoundary(Boundary boundary, double flux) {
        boundary_conditions_[to_index(boundary)] = BoundaryCondition::Neumann(flux);
    }

    void setNeumannBoundary(int boundary_id, double flux) {
        setNeumannBoundary(static_cast<Boundary>(boundary_id), flux);
    }

    /**
     * @brief Set a boundary condition.
     */
    void setBoundaryCondition(Boundary boundary, const BoundaryCondition& bc) {
        boundary_conditions_[to_index(boundary)] = bc;
    }

    void setBoundaryCondition(int boundary_id, const BoundaryCondition& bc) {
        setBoundaryCondition(static_cast<Boundary>(boundary_id), bc);
    }

    /**
     * @brief Run the solver for the specified number of steps.
     */
    void solve(double dt, int num_steps) {
        if (dt <= 0.0 || num_steps <= 0) {
            throw std::invalid_argument("Time step and number of steps must be positive");
        }

        if (!checkStability(dt)) {
            throw std::runtime_error(
                "Time step may be too large for stability. "
                "Use checkStability(dt) to verify before calling solve().");
        }

        scratch_.resize(solution_.size());

        for (int step = 0; step < num_steps; ++step) {
            derived().preStep(step, dt);

            // Compute updates for all interior nodes
            iterator_.forEachInterior([this, dt](int idx, int i, int j) {
                derived().computeNodeUpdate(idx, i, j, stencil_ops_, dt);
            });

            // Apply boundary conditions
            applyBoundaryConditions(scratch_);

            // Swap buffers
            solution_.swap(scratch_);

            derived().postStep(step, dt);
        }
    }

    /**
     * @brief Get the current solution.
     */
    const std::vector<double>& solution() const { return solution_; }

    /**
     * @brief Get the mesh.
     */
    const StructuredMesh& mesh() const { return mesh_; }

    /**
     * @brief Get diffusivity.
     */
    double diffusivity() const { return diffusivity_; }

protected:
    const StructuredMesh& mesh_;
    double diffusivity_;
    std::vector<double> solution_;
    std::vector<double> scratch_;
    std::array<BoundaryCondition, 4> boundary_conditions_;

    MeshIterator iterator_;
    StencilOps stencil_ops_;

    // Time tracking for derived classes
    double time_ = 0.0;

    /**
     * @brief Compute the diffusion update term for a node.
     *
     * This is the common diffusion contribution: D * ∇²u * dt
     */
    double diffusionUpdate(int idx, double dt) const {
        return stencil_ops_.diffusionTerm(solution_, idx, diffusivity_, dt);
    }

    /**
     * @brief Check CFL stability condition.
     */
    bool checkStability(double dt) const {
        double dx = mesh_.dx();
        double max_dt = dx * dx / (2.0 * diffusivity_);

        if (!mesh_.is1D()) {
            double dy = mesh_.dy();
            max_dt = 1.0 / (2.0 * diffusivity_ * (1.0 / (dx * dx) + 1.0 / (dy * dy)));
        }

        bool stable = dt <= max_dt;

        // Let derived class add its own stability checks
        if (stable) {
            stable = derived().checkStabilityDerived(dt);
        }

        return stable;
    }

    /**
     * @brief Apply boundary conditions to a solution vector.
     */
    void applyBoundaryConditions(std::vector<double>& u) {
        if (mesh_.is1D()) {
            applyBoundaryConditions1D(u);
        } else {
            applyBoundaryConditions2D(u);
        }
    }

    // Default implementations for derived class hooks
    bool checkStabilityDerived(double /*dt*/) const { return true; }
    void preStep(int /*step*/, double /*dt*/) {}
    void postStep(int /*step*/, double dt) { time_ += dt; }

private:
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    void applyBoundaryConditions1D(std::vector<double>& u) {
        const int nx = mesh_.nx();
        const double dx = mesh_.dx();

        const auto& left_bc = boundary_conditions_[to_index(Boundary::Left)];
        const auto& right_bc = boundary_conditions_[to_index(Boundary::Right)];

        // Left boundary
        if (left_bc.type == BoundaryType::DIRICHLET) {
            u[0] = left_bc.value;
        } else {
            u[0] = u[1] - left_bc.value * dx;
        }

        // Right boundary
        if (right_bc.type == BoundaryType::DIRICHLET) {
            u[nx] = right_bc.value;
        } else {
            u[nx] = u[nx - 1] + right_bc.value * dx;
        }
    }

    void applyBoundaryConditions2D(std::vector<double>& u) {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int stride = iterator_.stride();
        const double dx = mesh_.dx();
        const double dy = mesh_.dy();

        const auto& left_bc = boundary_conditions_[to_index(Boundary::Left)];
        const auto& right_bc = boundary_conditions_[to_index(Boundary::Right)];
        const auto& bottom_bc = boundary_conditions_[to_index(Boundary::Bottom)];
        const auto& top_bc = boundary_conditions_[to_index(Boundary::Top)];

        // Left boundary (exclude corners)
        for (int j = 1; j < ny; ++j) {
            int idx = j * stride;
            if (left_bc.type == BoundaryType::DIRICHLET) {
                u[idx] = left_bc.value;
            } else {
                u[idx] = u[idx + 1] - left_bc.value * dx;
            }
        }

        // Right boundary (exclude corners)
        for (int j = 1; j < ny; ++j) {
            int idx = j * stride + nx;
            if (right_bc.type == BoundaryType::DIRICHLET) {
                u[idx] = right_bc.value;
            } else {
                u[idx] = u[idx - 1] + right_bc.value * dx;
            }
        }

        // Bottom boundary (exclude corners)
        for (int i = 1; i < nx; ++i) {
            int idx = i;
            if (bottom_bc.type == BoundaryType::DIRICHLET) {
                u[idx] = bottom_bc.value;
            } else {
                u[idx] = u[idx + stride] - bottom_bc.value * dy;
            }
        }

        // Top boundary (exclude corners)
        for (int i = 1; i < nx; ++i) {
            int idx = ny * stride + i;
            if (top_bc.type == BoundaryType::DIRICHLET) {
                u[idx] = top_bc.value;
            } else {
                u[idx] = u[idx - stride] + top_bc.value * dy;
            }
        }

        // Corners: use Dirichlet values or average neighbors
        // Bottom-left
        u[0] = (left_bc.type == BoundaryType::DIRICHLET)     ? left_bc.value
               : (bottom_bc.type == BoundaryType::DIRICHLET) ? bottom_bc.value
                                                             : 0.5 * (u[1] + u[stride]);

        // Bottom-right
        u[nx] = (right_bc.type == BoundaryType::DIRICHLET)    ? right_bc.value
                : (bottom_bc.type == BoundaryType::DIRICHLET) ? bottom_bc.value
                                                              : 0.5 * (u[nx - 1] + u[nx + stride]);

        // Top-left
        u[ny * stride] = (left_bc.type == BoundaryType::DIRICHLET) ? left_bc.value
                         : (top_bc.type == BoundaryType::DIRICHLET)
                             ? top_bc.value
                             : 0.5 * (u[ny * stride + 1] + u[(ny - 1) * stride]);

        // Top-right
        u[ny * stride + nx] = (right_bc.type == BoundaryType::DIRICHLET) ? right_bc.value
                              : (top_bc.type == BoundaryType::DIRICHLET)
                                  ? top_bc.value
                                  : 0.5 * (u[ny * stride + nx - 1] + u[(ny - 1) * stride + nx]);
    }
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_SOLVER_BASE_HPP
