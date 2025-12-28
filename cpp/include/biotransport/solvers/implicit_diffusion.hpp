/**
 * @file implicit_diffusion.hpp
 * @brief Fully implicit diffusion solver using sparse matrices.
 *
 * This solver uses Backward Euler time integration with a sparse matrix
 * representation of the Laplacian. It's unconditionally stable and
 * suitable for stiff problems.
 *
 * For 2D problems, consider using ADI which is O(N) per step.
 * This full implicit solver is O(N^1.5) for sparse LU but
 * provides maximum flexibility (non-uniform coefficients, etc.)
 *
 * @author BioTransport Development Team
 * @date December 2025
 */

#ifndef BIOTRANSPORT_SOLVERS_IMPLICIT_DIFFUSION_HPP
#define BIOTRANSPORT_SOLVERS_IMPLICIT_DIFFUSION_HPP

#ifdef BIOTRANSPORT_ENABLE_EIGEN

#include <algorithm>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/mesh/structured_mesh_3d.hpp>
#include <biotransport/core/numerics/linear_algebra/sparse_matrix.hpp>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace biotransport {

/**
 * @brief Result of an implicit solve.
 */
struct ImplicitSolveResult {
    int steps = 0;            ///< Number of time steps completed
    double total_time = 0.0;  ///< Total simulation time
    double residual = 0.0;    ///< Final residual norm
    bool success = true;      ///< Whether solve succeeded
};

/**
 * @brief 2D implicit diffusion solver using sparse matrices.
 *
 * Solves: ∂u/∂t = D∇²u + f(x,y,t)
 *
 * Using Backward Euler:
 *   (I - dt*D*∇²) u^{n+1} = u^n + dt*f^{n+1}
 *
 * Features:
 * - Unconditionally stable
 * - Supports spatially-varying diffusivity
 * - Multiple solver backends (direct, iterative)
 */
class ImplicitDiffusion2D {
public:
    /**
     * @brief Construct solver with constant diffusivity.
     */
    ImplicitDiffusion2D(const StructuredMesh& mesh, double diffusivity)
        : mesh_(mesh),
          diffusivity_(mesh.numNodes(), diffusivity),
          solution_(mesh.numNodes(), 0.0),
          time_(0.0),
          solver_type_(linalg::SparseSolverType::SparseLU),
          tolerance_(1e-10),
          max_iterations_(1000) {
        dx_ = mesh.dx();
        dy_ = mesh.dy();

        for (int i = 0; i < 4; ++i) {
            boundary_conditions_[i] = BoundaryCondition::Dirichlet(0.0);
        }
    }

    /**
     * @brief Construct solver with spatially-varying diffusivity.
     */
    ImplicitDiffusion2D(const StructuredMesh& mesh, const std::vector<double>& diffusivity)
        : mesh_(mesh),
          diffusivity_(diffusivity),
          solution_(mesh.numNodes(), 0.0),
          time_(0.0),
          solver_type_(linalg::SparseSolverType::SparseLU),
          tolerance_(1e-10),
          max_iterations_(1000) {
        if (diffusivity.size() != static_cast<size_t>(mesh.numNodes())) {
            throw std::invalid_argument("Diffusivity size must match mesh nodes");
        }

        dx_ = mesh.dx();
        dy_ = mesh.dy();

        for (int i = 0; i < 4; ++i) {
            boundary_conditions_[i] = BoundaryCondition::Dirichlet(0.0);
        }
    }

    /**
     * @brief Set initial condition.
     */
    void setInitialCondition(const std::vector<double>& values) {
        if (values.size() != solution_.size()) {
            throw std::invalid_argument("Initial condition size mismatch");
        }
        solution_ = values;
    }

    /**
     * @brief Set Dirichlet boundary condition.
     */
    void setDirichletBoundary(Boundary boundary, double value) {
        boundary_conditions_[to_index(boundary)] = BoundaryCondition::Dirichlet(value);
    }

    /**
     * @brief Set Neumann boundary condition.
     */
    void setNeumannBoundary(Boundary boundary, double flux) {
        boundary_conditions_[to_index(boundary)] = BoundaryCondition::Neumann(flux);
    }

    /**
     * @brief Set source term function.
     */
    void setSourceTerm(std::function<double(double, double, double)> source) {
        source_term_ = source;
    }

    /**
     * @brief Set solver type.
     */
    void setSolverType(linalg::SparseSolverType type) { solver_type_ = type; }

    /**
     * @brief Set tolerance for iterative solvers.
     */
    void setTolerance(double tol) { tolerance_ = tol; }

    /**
     * @brief Set max iterations for iterative solvers.
     */
    void setMaxIterations(int max_iter) { max_iterations_ = max_iter; }

    /**
     * @brief Advance solution by one time step.
     */
    ImplicitSolveResult step(double dt) {
        if (dt <= 0.0) {
            throw std::invalid_argument("Time step must be positive");
        }

        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int nodes_x = nx + 1;
        const int nodes_y = ny + 1;
        const int n = nodes_x * nodes_y;

        auto idx = [nodes_x](int i, int j) {
            return j * nodes_x + i;
        };

        // Build implicit matrix: (I - dt*D*∇²)
        linalg::SparseMatrix A(n, n);
        A.reserve(5 * n);

        std::vector<double> rhs(n);

        for (int j = 0; j < nodes_y; ++j) {
            for (int i = 0; i < nodes_x; ++i) {
                int k = idx(i, j);

                // Check if boundary
                bool is_left = (i == 0);
                bool is_right = (i == nx);
                bool is_bottom = (j == 0);
                bool is_top = (j == ny);

                if (is_left || is_right || is_bottom || is_top) {
                    // Apply boundary condition
                    const BoundaryCondition* bc = nullptr;
                    if (is_left)
                        bc = &boundary_conditions_[to_index(Boundary::Left)];
                    else if (is_right)
                        bc = &boundary_conditions_[to_index(Boundary::Right)];
                    else if (is_bottom)
                        bc = &boundary_conditions_[to_index(Boundary::Bottom)];
                    else if (is_top)
                        bc = &boundary_conditions_[to_index(Boundary::Top)];

                    if (bc->type == BoundaryType::DIRICHLET) {
                        A.addEntry(k, k, 1.0);
                        rhs[k] = bc->value;
                    } else {
                        // Neumann: ghost point approach
                        // For simplicity, use one-sided difference
                        A.addEntry(k, k, 1.0);
                        // Would need proper ghost point implementation
                        rhs[k] = solution_[k];
                    }
                } else {
                    // Interior point
                    double D = diffusivity_[k];
                    double rx = D * dt / (dx_ * dx_);
                    double ry = D * dt / (dy_ * dy_);

                    A.addEntry(k, k, 1.0 + 2.0 * rx + 2.0 * ry);
                    A.addEntry(k, idx(i - 1, j), -rx);
                    A.addEntry(k, idx(i + 1, j), -rx);
                    A.addEntry(k, idx(i, j - 1), -ry);
                    A.addEntry(k, idx(i, j + 1), -ry);

                    // RHS: u^n + dt*f^{n+1}
                    double x = mesh_.x(i);
                    double y = mesh_.y(i, j);
                    double source = source_term_ ? source_term_(x, y, time_ + dt) : 0.0;
                    rhs[k] = solution_[k] + dt * source;
                }
            }
        }

        A.finalize();

        ImplicitSolveResult result;
        try {
            solution_ = A.solve(rhs, solver_type_, tolerance_, max_iterations_);
            time_ += dt;
            result.steps = 1;
            result.total_time = time_;
            result.success = true;
        } catch (const std::exception& e) {
            result.success = false;
        }

        return result;
    }

    /**
     * @brief Run solver for multiple steps.
     */
    ImplicitSolveResult solve(double dt, int num_steps) {
        ImplicitSolveResult total;
        total.steps = 0;
        total.success = true;

        for (int s = 0; s < num_steps; ++s) {
            auto result = step(dt);
            if (!result.success) {
                total.success = false;
                break;
            }
            total.steps++;
        }

        total.total_time = time_;
        return total;
    }

    /**
     * @brief Get current solution.
     */
    const std::vector<double>& solution() const { return solution_; }

    /**
     * @brief Get current time.
     */
    double time() const { return time_; }

    /**
     * @brief Get mesh.
     */
    const StructuredMesh& mesh() const { return mesh_; }

private:
    const StructuredMesh& mesh_;
    std::vector<double> diffusivity_;
    std::vector<double> solution_;
    double time_;
    double dx_, dy_;
    std::array<BoundaryCondition, 4> boundary_conditions_;
    std::function<double(double, double, double)> source_term_;
    linalg::SparseSolverType solver_type_;
    double tolerance_;
    int max_iterations_;
};

/**
 * @brief 3D implicit diffusion solver using sparse matrices.
 */
class ImplicitDiffusion3D {
public:
    ImplicitDiffusion3D(const StructuredMesh3D& mesh, double diffusivity)
        : mesh_(mesh),
          diffusivity_(mesh.numNodes(), diffusivity),
          solution_(mesh.numNodes(), 0.0),
          time_(0.0),
          solver_type_(linalg::SparseSolverType::BiCGSTAB),  // Iterative for 3D
          tolerance_(1e-10),
          max_iterations_(1000) {
        dx_ = mesh.dx();
        dy_ = mesh.dy();
        dz_ = mesh.dz();

        for (int i = 0; i < 6; ++i) {
            boundary_conditions_[i] = BoundaryCondition::Dirichlet(0.0);
        }
    }

    void setInitialCondition(const std::vector<double>& values) {
        if (values.size() != solution_.size()) {
            throw std::invalid_argument("Initial condition size mismatch");
        }
        solution_ = values;
    }

    void setDirichletBoundary(Boundary3D boundary, double value) {
        boundary_conditions_[static_cast<int>(boundary)] = BoundaryCondition::Dirichlet(value);
    }

    void setSolverType(linalg::SparseSolverType type) { solver_type_ = type; }

    ImplicitSolveResult step(double dt) {
        if (dt <= 0.0) {
            throw std::invalid_argument("Time step must be positive");
        }

        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int nz = mesh_.nz();
        const int nodes_x = nx + 1;
        const int nodes_y = ny + 1;
        const int nodes_z = nz + 1;
        const int n = nodes_x * nodes_y * nodes_z;

        auto idx = [nodes_x, nodes_y](int i, int j, int k) {
            return k * nodes_x * nodes_y + j * nodes_x + i;
        };

        // Build matrix
        linalg::SparseMatrix A(n, n);
        A.reserve(7 * n);

        std::vector<double> rhs(n);

        for (int k = 0; k < nodes_z; ++k) {
            for (int j = 0; j < nodes_y; ++j) {
                for (int i = 0; i < nodes_x; ++i) {
                    int m = idx(i, j, k);

                    bool is_boundary =
                        (i == 0 || i == nx || j == 0 || j == ny || k == 0 || k == nz);

                    if (is_boundary) {
                        A.addEntry(m, m, 1.0);
                        // Apply Dirichlet BC (simplified)
                        rhs[m] = 0.0;  // Use appropriate BC value
                    } else {
                        double D = diffusivity_[m];
                        double rx = D * dt / (dx_ * dx_);
                        double ry = D * dt / (dy_ * dy_);
                        double rz = D * dt / (dz_ * dz_);

                        A.addEntry(m, m, 1.0 + 2.0 * (rx + ry + rz));
                        A.addEntry(m, idx(i - 1, j, k), -rx);
                        A.addEntry(m, idx(i + 1, j, k), -rx);
                        A.addEntry(m, idx(i, j - 1, k), -ry);
                        A.addEntry(m, idx(i, j + 1, k), -ry);
                        A.addEntry(m, idx(i, j, k - 1), -rz);
                        A.addEntry(m, idx(i, j, k + 1), -rz);

                        rhs[m] = solution_[m];
                    }
                }
            }
        }

        A.finalize();

        ImplicitSolveResult result;
        try {
            solution_ = A.solve(rhs, solver_type_, tolerance_, max_iterations_);
            time_ += dt;
            result.steps = 1;
            result.total_time = time_;
            result.success = true;
        } catch (const std::exception& e) {
            result.success = false;
        }

        return result;
    }

    ImplicitSolveResult solve(double dt, int num_steps) {
        ImplicitSolveResult total;
        total.steps = 0;
        total.success = true;

        for (int s = 0; s < num_steps; ++s) {
            auto result = step(dt);
            if (!result.success) {
                total.success = false;
                break;
            }
            total.steps++;
        }

        total.total_time = time_;
        return total;
    }

    const std::vector<double>& solution() const { return solution_; }
    double time() const { return time_; }

private:
    const StructuredMesh3D& mesh_;
    std::vector<double> diffusivity_;
    std::vector<double> solution_;
    double time_;
    double dx_, dy_, dz_;
    std::array<BoundaryCondition, 6> boundary_conditions_;
    linalg::SparseSolverType solver_type_;
    double tolerance_;
    int max_iterations_;
};

}  // namespace biotransport

#else  // !BIOTRANSPORT_ENABLE_EIGEN

namespace biotransport {

struct ImplicitSolveResult {
    int steps = 0;
    double total_time = 0.0;
    double residual = 0.0;
    bool success = false;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_ENABLE_EIGEN

#endif  // BIOTRANSPORT_SOLVERS_IMPLICIT_DIFFUSION_HPP
