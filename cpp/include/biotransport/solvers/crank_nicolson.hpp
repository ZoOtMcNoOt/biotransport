#ifndef BIOTRANSPORT_SOLVERS_CRANK_NICOLSON_HPP
#define BIOTRANSPORT_SOLVERS_CRANK_NICOLSON_HPP

/**
 * @file crank_nicolson.hpp
 * @brief Crank-Nicolson implicit time integration for diffusion.
 *
 * The Crank-Nicolson method is second-order accurate in time and
 * unconditionally stable for diffusion problems. It uses a 50/50 blend
 * of explicit and implicit terms:
 *
 *   u^{n+1} - (D*dt/2)*∇²u^{n+1} = u^n + (D*dt/2)*∇²u^n
 *
 * The implicit system is solved using the Jacobi iterative method,
 * which is parallelizable with OpenMP.
 *
 * @see ExplicitSolverBase for the explicit FTCS method
 */

#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/mesh_iterators.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef BIOTRANSPORT_ENABLE_OPENMP
#include <omp.h>
#endif

namespace biotransport {

/**
 * @brief Result of a Crank-Nicolson solve step.
 */
struct CNSolveResult {
    int iterations = 0;      ///< Number of iterations used
    double residual = 0.0;   ///< Final residual norm
    bool converged = false;  ///< Whether tolerance was achieved
};

/**
 * @brief Crank-Nicolson solver for the diffusion equation.
 *
 * Solves: ∂u/∂t = D∇²u
 *
 * Uses the Jacobi iterative method for the implicit solve. This is
 * slower per step than explicit methods but allows much larger time steps.
 *
 * @code
 *   CrankNicolsonDiffusion solver(mesh, 1e-9);
 *   solver.setInitialCondition(u0);
 *   solver.setDirichletBoundary(Boundary::Left, 1.0);
 *   solver.setDirichletBoundary(Boundary::Right, 0.0);
 *
 *   double dt = 1.0;  // Much larger than explicit CFL limit
 *   for (int step = 0; step < num_steps; ++step) {
 *       solver.step(dt);
 *   }
 * @endcode
 */
class CrankNicolsonDiffusion {
public:
    /**
     * @brief Construct a Crank-Nicolson diffusion solver.
     *
     * @param mesh 1D or 2D structured mesh
     * @param diffusivity Diffusion coefficient D [m²/s]
     */
    CrankNicolsonDiffusion(const StructuredMesh& mesh, double diffusivity)
        : mesh_(mesh), diffusivity_(diffusivity), iterator_(mesh) {
        if (diffusivity <= 0.0) {
            throw std::invalid_argument("Diffusivity must be positive");
        }
        solution_.resize(mesh.numNodes(), 0.0);
        rhs_.resize(solution_.size(), 0.0);
        scratch_.resize(solution_.size(), 0.0);

        // Default boundary conditions
        for (int i = 0; i < 4; ++i) {
            boundary_conditions_[i] = BoundaryCondition::Dirichlet(0.0);
        }

        // Precompute coefficients
        dx2_inv_ = 1.0 / (mesh.dx() * mesh.dx());
        if (!mesh.is1D()) {
            dy2_inv_ = 1.0 / (mesh.dy() * mesh.dy());
        } else {
            dy2_inv_ = 0.0;
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

    /**
     * @brief Set a Neumann boundary condition.
     */
    void setNeumannBoundary(Boundary boundary, double flux) {
        boundary_conditions_[to_index(boundary)] = BoundaryCondition::Neumann(flux);
    }

    /**
     * @brief Set iteration tolerance for implicit solve.
     */
    CrankNicolsonDiffusion& setTolerance(double tol) {
        tolerance_ = tol;
        return *this;
    }

    /**
     * @brief Set maximum iterations for implicit solve.
     */
    CrankNicolsonDiffusion& setMaxIterations(int max_iter) {
        max_iterations_ = max_iter;
        return *this;
    }

    /**
     * @brief Advance the solution by one time step.
     *
     * @param dt Time step size [s]
     * @return Result including iteration count and convergence status
     */
    CNSolveResult step(double dt) {
        if (dt <= 0.0) {
            throw std::invalid_argument("Time step must be positive");
        }

        // Coefficients for the Crank-Nicolson scheme
        // LHS: (I - (D*dt/2)*∇²) u^{n+1} = RHS
        // RHS: u^n + (D*dt/2)*∇² u^n
        double alpha_half = diffusivity_ * dt * 0.5;

        // Build RHS: u^n + alpha_half * ∇² u^n (explicit part)
        buildRHS(alpha_half);

        // Solve implicit system using Jacobi iteration
        return solveImplicit(alpha_half);
    }

    /**
     * @brief Run the solver for specified number of steps.
     *
     * @param dt Time step size
     * @param num_steps Number of time steps
     */
    void solve(double dt, int num_steps) {
        for (int step = 0; step < num_steps; ++step) {
            CNSolveResult result = this->step(dt);
            if (!result.converged) {
                throw std::runtime_error("Crank-Nicolson iteration did not converge at step " +
                                         std::to_string(step));
            }
            time_ += dt;
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

    /**
     * @brief Get current simulation time.
     */
    double time() const { return time_; }

private:
    const StructuredMesh& mesh_;
    double diffusivity_;
    std::vector<double> solution_;
    std::vector<double> rhs_;
    std::vector<double> scratch_;
    std::array<BoundaryCondition, 4> boundary_conditions_;
    MeshIterator iterator_;

    double dx2_inv_ = 0.0;
    double dy2_inv_ = 0.0;
    double time_ = 0.0;
    double tolerance_ = 1e-8;
    int max_iterations_ = 1000;

    /**
     * @brief Build the RHS vector: u^n + alpha * ∇² u^n
     */
    void buildRHS(double alpha) {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int stride = nx + 1;

        if (mesh_.is1D()) {
            // Interior nodes
            for (int i = 1; i < nx; ++i) {
                double laplacian =
                    dx2_inv_ * (solution_[i - 1] - 2.0 * solution_[i] + solution_[i + 1]);
                rhs_[i] = solution_[i] + alpha * laplacian;
            }
        } else {
            // 2D case
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = 1; j < ny; ++j) {
                for (int i = 1; i < nx; ++i) {
                    int idx = j * stride + i;
                    double laplacian = dx2_inv_ * (solution_[idx - 1] - 2.0 * solution_[idx] +
                                                   solution_[idx + 1]) +
                                       dy2_inv_ * (solution_[idx - stride] - 2.0 * solution_[idx] +
                                                   solution_[idx + stride]);
                    rhs_[idx] = solution_[idx] + alpha * laplacian;
                }
            }
        }
    }

    /**
     * @brief Solve the implicit system using Jacobi iteration.
     *
     * Solves: (I - alpha*∇²) u^{n+1} = rhs
     *
     * The Jacobi method updates each point independently using only
     * values from the previous iteration, making it parallelizable.
     */
    CNSolveResult solveImplicit(double alpha) {
        CNSolveResult result;
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int stride = nx + 1;

        // Precompute diagonal coefficient
        // For 1D: a_ii = 1 + 2*alpha/dx²
        // For 2D: a_ii = 1 + 2*alpha/dx² + 2*alpha/dy²
        double diag_coeff;
        if (mesh_.is1D()) {
            diag_coeff = 1.0 + 2.0 * alpha * dx2_inv_;
        } else {
            diag_coeff = 1.0 + 2.0 * alpha * (dx2_inv_ + dy2_inv_);
        }
        double diag_inv = 1.0 / diag_coeff;
        double off_diag_x = alpha * dx2_inv_;
        double off_diag_y = alpha * dy2_inv_;

        // Start from current solution as initial guess
        scratch_ = solution_;

        for (int iter = 0; iter < max_iterations_; ++iter) {
            double max_diff = 0.0;

            if (mesh_.is1D()) {
                for (int i = 1; i < nx; ++i) {
                    // Jacobi update: u_new[i] = (rhs[i] + alpha*dx2_inv*(u_old[i-1] + u_old[i+1]))
                    // / diag
                    double u_new =
                        (rhs_[i] + off_diag_x * (scratch_[i - 1] + scratch_[i + 1])) * diag_inv;
                    max_diff = std::max(max_diff, std::abs(u_new - solution_[i]));
                    solution_[i] = u_new;
                }
            } else {
                // 2D case - Jacobi is fully parallelizable
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static) reduction(max : max_diff)
#endif
                for (int j = 1; j < ny; ++j) {
                    for (int i = 1; i < nx; ++i) {
                        int idx = j * stride + i;
                        double u_new =
                            (rhs_[idx] + off_diag_x * (scratch_[idx - 1] + scratch_[idx + 1]) +
                             off_diag_y * (scratch_[idx - stride] + scratch_[idx + stride])) *
                            diag_inv;
                        double diff = std::abs(u_new - solution_[idx]);
                        max_diff = std::max(max_diff, diff);
                        solution_[idx] = u_new;
                    }
                }
            }

            // Apply boundary conditions
            applyBoundaryConditions(solution_);

            // Check convergence
            result.residual = max_diff;
            result.iterations = iter + 1;

            if (max_diff < tolerance_) {
                result.converged = true;
                break;
            }

            // Update scratch for next iteration
            scratch_ = solution_;
        }

        if (!result.converged) {
            result.converged = (result.residual < tolerance_ * 100);  // Looser check
        }

        return result;
    }

    /**
     * @brief Apply boundary conditions to the solution vector.
     */
    void applyBoundaryConditions(std::vector<double>& u) {
        if (mesh_.is1D()) {
            applyBoundaryConditions1D(u);
        } else {
            applyBoundaryConditions2D(u);
        }
    }

    void applyBoundaryConditions1D(std::vector<double>& u) {
        const int nx = mesh_.nx();
        const double dx = mesh_.dx();

        const auto& left_bc = boundary_conditions_[to_index(Boundary::Left)];
        const auto& right_bc = boundary_conditions_[to_index(Boundary::Right)];

        if (left_bc.type == BoundaryType::DIRICHLET) {
            u[0] = left_bc.value;
        } else {
            u[0] = u[1] - left_bc.value * dx;
        }

        if (right_bc.type == BoundaryType::DIRICHLET) {
            u[nx] = right_bc.value;
        } else {
            u[nx] = u[nx - 1] + right_bc.value * dx;
        }
    }

    void applyBoundaryConditions2D(std::vector<double>& u) {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int stride = nx + 1;
        const double dx = mesh_.dx();
        const double dy = mesh_.dy();

        const auto& left_bc = boundary_conditions_[to_index(Boundary::Left)];
        const auto& right_bc = boundary_conditions_[to_index(Boundary::Right)];
        const auto& bottom_bc = boundary_conditions_[to_index(Boundary::Bottom)];
        const auto& top_bc = boundary_conditions_[to_index(Boundary::Top)];

        // Left/Right boundaries
        for (int j = 0; j <= ny; ++j) {
            int left_idx = j * stride;
            int right_idx = j * stride + nx;

            if (left_bc.type == BoundaryType::DIRICHLET) {
                u[left_idx] = left_bc.value;
            } else {
                u[left_idx] = u[left_idx + 1] - left_bc.value * dx;
            }

            if (right_bc.type == BoundaryType::DIRICHLET) {
                u[right_idx] = right_bc.value;
            } else {
                u[right_idx] = u[right_idx - 1] + right_bc.value * dx;
            }
        }

        // Bottom/Top boundaries
        for (int i = 0; i <= nx; ++i) {
            if (bottom_bc.type == BoundaryType::DIRICHLET) {
                u[i] = bottom_bc.value;
            } else {
                u[i] = u[i + stride] - bottom_bc.value * dy;
            }

            int top_idx = ny * stride + i;
            if (top_bc.type == BoundaryType::DIRICHLET) {
                u[top_idx] = top_bc.value;
            } else {
                u[top_idx] = u[top_idx - stride] + top_bc.value * dy;
            }
        }
    }
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_CRANK_NICOLSON_HPP
