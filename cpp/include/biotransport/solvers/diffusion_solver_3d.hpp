/**
 * @file diffusion_solver_3d.hpp
 * @brief 3D diffusion and reaction-diffusion solvers.
 *
 * Extends the solver framework to 3D structured meshes for:
 * - Pure diffusion (∂u/∂t = D∇²u)
 * - Reaction-diffusion (∂u/∂t = D∇²u + R(u,x,y,z,t))
 * - Linear decay (∂u/∂t = D∇²u - ku)
 *
 * Supports OpenMP parallelization for multi-core acceleration.
 */

#ifndef BIOTRANSPORT_SOLVERS_DIFFUSION_SOLVER_3D_HPP
#define BIOTRANSPORT_SOLVERS_DIFFUSION_SOLVER_3D_HPP

#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/mesh_iterators_3d.hpp>
#include <biotransport/core/mesh/structured_mesh_3d.hpp>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace biotransport {

/**
 * @brief Base class for 3D explicit time-stepping solvers using CRTP.
 */
template <typename Derived>
class ExplicitSolverBase3D {
public:
    ExplicitSolverBase3D(const StructuredMesh3D& mesh, double diffusivity)
        : mesh_(mesh), diffusivity_(diffusivity), iterator_(mesh), stencil_ops_(mesh) {
        if (diffusivity <= 0.0) {
            throw std::invalid_argument("Diffusivity must be positive");
        }
        solution_.resize(mesh.numNodes(), 0.0);
        scratch_.resize(solution_.size(), 0.0);

        // Default boundary conditions (Dirichlet, value = 0)
        for (int i = 0; i < 6; ++i) {
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
     * @brief Set a Dirichlet boundary condition on a face.
     */
    void setDirichletBoundary(Boundary3D boundary, double value) {
        boundary_conditions_[to_index(boundary)] = BoundaryCondition::Dirichlet(value);
    }

    void setDirichletBoundary(int boundary_id, double value) {
        setDirichletBoundary(static_cast<Boundary3D>(boundary_id), value);
    }

    /**
     * @brief Set a Neumann boundary condition on a face.
     */
    void setNeumannBoundary(Boundary3D boundary, double flux) {
        boundary_conditions_[to_index(boundary)] = BoundaryCondition::Neumann(flux);
    }

    void setNeumannBoundary(int boundary_id, double flux) {
        setNeumannBoundary(static_cast<Boundary3D>(boundary_id), flux);
    }

    /**
     * @brief Set a boundary condition.
     */
    void setBoundaryCondition(Boundary3D boundary, const BoundaryCondition& bc) {
        boundary_conditions_[to_index(boundary)] = bc;
    }

    void setBoundaryCondition(int boundary_id, const BoundaryCondition& bc) {
        setBoundaryCondition(static_cast<Boundary3D>(boundary_id), bc);
    }

    /**
     * @brief Check CFL stability condition for 3D explicit diffusion.
     *
     * Stability requires: dt <= 1/(2*D*(1/dx² + 1/dy² + 1/dz²))
     */
    bool checkStability(double dt) const {
        double stability_limit =
            1.0 / (2.0 * diffusivity_ *
                   (1.0 / (mesh_.dx() * mesh_.dx()) + 1.0 / (mesh_.dy() * mesh_.dy()) +
                    1.0 / (mesh_.dz() * mesh_.dz())));
        return dt <= stability_limit;
    }

    /**
     * @brief Get maximum stable time step.
     */
    double maxStableTimeStep() const {
        return 0.9 / (2.0 * diffusivity_ *
                      (1.0 / (mesh_.dx() * mesh_.dx()) + 1.0 / (mesh_.dy() * mesh_.dy()) +
                       1.0 / (mesh_.dz() * mesh_.dz())));
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
                "Use checkStability(dt) or maxStableTimeStep() to verify.");
        }

        for (int step = 0; step < num_steps; ++step) {
            derived().preStep(step, dt);

            // Compute updates for all interior nodes
            iterator_.forEachInterior([this, dt](int idx, int i, int j, int k) {
                derived().computeNodeUpdate(idx, i, j, k, stencil_ops_, dt);
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
    const StructuredMesh3D& mesh() const { return mesh_; }

    /**
     * @brief Get diffusivity.
     */
    double diffusivity() const { return diffusivity_; }

    /**
     * @brief Get current simulation time.
     */
    double time() const { return time_; }

protected:
    const StructuredMesh3D& mesh_;
    double diffusivity_;
    std::vector<double> solution_;
    std::vector<double> scratch_;
    std::array<BoundaryCondition, 6> boundary_conditions_;

    MeshIterator3D iterator_;
    StencilOps3D stencil_ops_;

    double time_ = 0.0;

    // Default implementations for hooks
    void preStep(int /*step*/, double /*dt*/) {}
    void postStep(int /*step*/, double dt) { time_ += dt; }

    /**
     * @brief Apply boundary conditions to a solution vector.
     */
    void applyBoundaryConditions(std::vector<double>& u) {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int nz = mesh_.nz();
        const int stride_j = mesh_.strideJ();
        const int stride_k = mesh_.strideK();

        // XMin face (i=0)
        applyBCFace(
            u, boundary_conditions_[to_index(Boundary3D::XMin)],
            [&](int j, int k) {
                int idx = k * stride_k + j * stride_j + 0;
                int interior = k * stride_k + j * stride_j + 1;
                return std::make_pair(idx, interior);
            },
            ny, nz, mesh_.dx());

        // XMax face (i=nx)
        applyBCFace(
            u, boundary_conditions_[to_index(Boundary3D::XMax)],
            [&](int j, int k) {
                int idx = k * stride_k + j * stride_j + nx;
                int interior = k * stride_k + j * stride_j + (nx - 1);
                return std::make_pair(idx, interior);
            },
            ny, nz, mesh_.dx());

        // YMin face (j=0)
        applyBCFace(
            u, boundary_conditions_[to_index(Boundary3D::YMin)],
            [&](int i, int k) {
                int idx = k * stride_k + 0 * stride_j + i;
                int interior = k * stride_k + 1 * stride_j + i;
                return std::make_pair(idx, interior);
            },
            nx, nz, mesh_.dy());

        // YMax face (j=ny)
        applyBCFace(
            u, boundary_conditions_[to_index(Boundary3D::YMax)],
            [&](int i, int k) {
                int idx = k * stride_k + ny * stride_j + i;
                int interior = k * stride_k + (ny - 1) * stride_j + i;
                return std::make_pair(idx, interior);
            },
            nx, nz, mesh_.dy());

        // ZMin face (k=0)
        applyBCFace(
            u, boundary_conditions_[to_index(Boundary3D::ZMin)],
            [&](int i, int j) {
                int idx = 0 * stride_k + j * stride_j + i;
                int interior = 1 * stride_k + j * stride_j + i;
                return std::make_pair(idx, interior);
            },
            nx, ny, mesh_.dz());

        // ZMax face (k=nz)
        applyBCFace(
            u, boundary_conditions_[to_index(Boundary3D::ZMax)],
            [&](int i, int j) {
                int idx = nz * stride_k + j * stride_j + i;
                int interior = (nz - 1) * stride_k + j * stride_j + i;
                return std::make_pair(idx, interior);
            },
            nx, ny, mesh_.dz());
    }

private:
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    template <typename IndexFunc>
    void applyBCFace(std::vector<double>& u, const BoundaryCondition& bc, IndexFunc&& indexFunc,
                     int n1, int n2, double h) {
        for (int idx2 = 0; idx2 <= n2; ++idx2) {
            for (int idx1 = 0; idx1 <= n1; ++idx1) {
                auto [boundary_idx, interior_idx] = indexFunc(idx1, idx2);

                if (bc.type == BoundaryType::DIRICHLET) {
                    u[boundary_idx] = bc.value;
                } else if (bc.type == BoundaryType::NEUMANN) {
                    // Ghost node approach: u_boundary = u_interior + flux * h
                    u[boundary_idx] = u[interior_idx] + bc.value * h;
                }
            }
        }
    }
};

// =============================================================================
// DiffusionSolver3D - Pure 3D diffusion
// =============================================================================

/**
 * @brief Solver for 3D diffusion equation: ∂u/∂t = D∇²u
 */
class DiffusionSolver3D : public ExplicitSolverBase3D<DiffusionSolver3D> {
public:
    using Base = ExplicitSolverBase3D<DiffusionSolver3D>;
    friend Base;

    DiffusionSolver3D(const StructuredMesh3D& mesh, double diffusivity) : Base(mesh, diffusivity) {}

    void computeNodeUpdate(int idx, int /*i*/, int /*j*/, int /*k*/, const StencilOps3D& ops,
                           double dt) {
        scratch_[idx] = solution_[idx] + ops.diffusionTerm(solution_, idx, diffusivity_, dt);
    }
};

// =============================================================================
// LinearReactionDiffusionSolver3D - With decay term
// =============================================================================

/**
 * @brief 3D solver for: ∂u/∂t = D∇²u - k*u
 *
 * Uses implicit treatment of decay for stability.
 */
class LinearReactionDiffusionSolver3D
    : public ExplicitSolverBase3D<LinearReactionDiffusionSolver3D> {
public:
    using Base = ExplicitSolverBase3D<LinearReactionDiffusionSolver3D>;
    friend Base;

    LinearReactionDiffusionSolver3D(const StructuredMesh3D& mesh, double diffusivity,
                                    double decay_rate)
        : Base(mesh, diffusivity), decay_rate_(decay_rate) {
        if (decay_rate_ < 0.0) {
            throw std::invalid_argument("Decay rate must be non-negative");
        }
    }

    void computeNodeUpdate(int idx, int /*i*/, int /*j*/, int /*k*/, const StencilOps3D& ops,
                           double dt) {
        double u = solution_[idx];
        double diffusion = ops.diffusionTerm(solution_, idx, diffusivity_, dt);
        scratch_[idx] = (u + diffusion) / (1.0 + decay_rate_ * dt);
    }

    double decayRate() const { return decay_rate_; }

private:
    double decay_rate_;
};

// =============================================================================
// ReactionDiffusionSolver3D - Generic with callable
// =============================================================================

/**
 * @brief 3D solver for: ∂u/∂t = D∇²u + R(u, x, y, z, t)
 */
class ReactionDiffusionSolver3D : public ExplicitSolverBase3D<ReactionDiffusionSolver3D> {
public:
    using Base = ExplicitSolverBase3D<ReactionDiffusionSolver3D>;
    using ReactionFunction =
        std::function<double(double u, double x, double y, double z, double t)>;
    friend Base;

    ReactionDiffusionSolver3D(const StructuredMesh3D& mesh, double diffusivity,
                              ReactionFunction reaction)
        : Base(mesh, diffusivity), reaction_(std::move(reaction)) {
        cacheCoordinates();
    }

    void computeNodeUpdate(int idx, int i, int j, int k, const StencilOps3D& ops, double dt) {
        double u = solution_[idx];
        double x = x_coords_[i];
        double y = y_coords_[j];
        double z = z_coords_[k];

        double diffusion = ops.diffusionTerm(solution_, idx, diffusivity_, dt);
        double reaction = dt * reaction_(u, x, y, z, time_);

        scratch_[idx] = u + diffusion + reaction;
    }

private:
    ReactionFunction reaction_;
    std::vector<double> x_coords_;
    std::vector<double> y_coords_;
    std::vector<double> z_coords_;

    void cacheCoordinates() {
        x_coords_.resize(mesh_.nx() + 1);
        for (int i = 0; i <= mesh_.nx(); ++i) {
            x_coords_[i] = mesh_.x(i);
        }
        y_coords_.resize(mesh_.ny() + 1);
        for (int j = 0; j <= mesh_.ny(); ++j) {
            y_coords_[j] = mesh_.y(j);
        }
        z_coords_.resize(mesh_.nz() + 1);
        for (int k = 0; k <= mesh_.nz(); ++k) {
            z_coords_[k] = mesh_.z(k);
        }
    }
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_DIFFUSION_SOLVER_3D_HPP
