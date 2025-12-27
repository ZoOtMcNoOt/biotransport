/**
 * @file adi_solver.hpp
 * @brief Alternating Direction Implicit (ADI) method for 2D/3D diffusion.
 *
 * ADI splits the multidimensional implicit problem into a sequence of 1D
 * tridiagonal systems that can be solved efficiently with the Thomas algorithm.
 * This provides O(N) complexity per time step while maintaining unconditional
 * stability.
 *
 * **2D Peaceman-Rachford ADI:**
 * For solving: ∂u/∂t = D(∂²u/∂x² + ∂²u/∂y²)
 *
 * The time step is split into two half-steps:
 * - Step 1: (I - r_x*δ_x²)u* = (I + r_y*δ_y²)uⁿ     (implicit in x)
 * - Step 2: (I - r_y*δ_y²)u^{n+1} = (I + r_x*δ_x²)u* (implicit in y)
 *
 * where r_x = D*dt/(2*dx²), r_y = D*dt/(2*dy²)
 *
 * **3D Douglas-Gunn ADI:**
 * For solving: ∂u/∂t = D(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
 *
 * The time step uses three stages:
 * - Stage 1: (I - r_x*δ_x²)u* = uⁿ + dt*D*∇²uⁿ
 * - Stage 2: (I - r_y*δ_y²)u** = u* + r_y*δ_y²uⁿ
 * - Stage 3: (I - r_z*δ_z²)u^{n+1} = u** + r_z*δ_z²uⁿ
 *
 * @see CrankNicolsonDiffusion for iterative implicit solver
 * @see solve_tridiagonal for Thomas algorithm
 *
 * @author BioTransport Development Team
 * @date December 2025
 */

#ifndef BIOTRANSPORT_SOLVERS_ADI_SOLVER_HPP
#define BIOTRANSPORT_SOLVERS_ADI_SOLVER_HPP

#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/mesh/structured_mesh_3d.hpp>
#include <biotransport/core/numerics/linear_algebra/tridiagonal.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef BIOTRANSPORT_ENABLE_OPENMP
#include <omp.h>
#endif

namespace biotransport {

/**
 * @brief Result of an ADI solve step.
 */
struct ADISolveResult {
    int steps = 0;            ///< Number of time steps completed
    int substeps = 0;         ///< Number of substeps (2 for 2D, 3 for 3D)
    double time = 0.0;        ///< Current simulation time after step()
    double total_time = 0.0;  ///< Total simulation time after solve()
    bool success = true;      ///< Whether the step completed successfully
};

/**
 * @brief 2D ADI solver using Peaceman-Rachford splitting.
 *
 * Solves the 2D diffusion equation:
 *   ∂u/∂t = D(∂²u/∂x² + ∂²u/∂y²)
 *
 * The method is:
 * - Unconditionally stable (no CFL restriction)
 * - Second-order accurate in space and time
 * - O(N) per time step (N = number of grid points)
 *
 * @code
 *   StructuredMesh mesh(1.0, 1.0, 50, 50);  // 50x50 grid
 *   ADIDiffusion2D solver(mesh, 1e-5);      // D = 10⁻⁵ m²/s
 *
 *   solver.setInitialCondition(u0);
 *   solver.setDirichletBoundary(Boundary::Left, 100.0);
 *   solver.setDirichletBoundary(Boundary::Right, 0.0);
 *
 *   double dt = 0.1;  // Can be much larger than explicit CFL limit
 *   solver.solve(dt, 100);  // 100 time steps
 * @endcode
 */
class ADIDiffusion2D {
public:
    /**
     * @brief Construct a 2D ADI diffusion solver.
     *
     * @param mesh 2D structured mesh
     * @param diffusivity Diffusion coefficient D [m²/s]
     * @throws std::invalid_argument if mesh is 1D or diffusivity <= 0
     */
    ADIDiffusion2D(const StructuredMesh& mesh, double diffusivity)
        : mesh_(mesh), diffusivity_(diffusivity) {
        if (mesh.is1D()) {
            throw std::invalid_argument("ADIDiffusion2D requires a 2D mesh");
        }
        if (diffusivity <= 0.0) {
            throw std::invalid_argument("Diffusivity must be positive");
        }

        const int num_nodes = mesh.numNodes();
        solution_.resize(num_nodes, 0.0);
        intermediate_.resize(num_nodes, 0.0);

        // Pre-allocate tridiagonal system vectors for max dimension
        const int max_dim = std::max(mesh.nx() + 1, mesh.ny() + 1);
        a_.resize(max_dim);
        b_.resize(max_dim);
        c_.resize(max_dim);
        d_.resize(max_dim);

        // Default Dirichlet boundary conditions
        for (int i = 0; i < 4; ++i) {
            boundary_conditions_[i] = BoundaryCondition::Dirichlet(0.0);
        }

        // Precompute mesh spacing
        dx_ = mesh.dx();
        dy_ = mesh.dy();
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
     * @brief Advance the solution by one time step using ADI.
     *
     * @param dt Time step size [s]
     * @return ADISolveResult with status information
     */
    ADISolveResult step(double dt) {
        if (dt <= 0.0) {
            throw std::invalid_argument("Time step must be positive");
        }

        ADISolveResult result;
        result.substeps = 2;

        // Coefficients for half-step
        const double rx = diffusivity_ * dt / (2.0 * dx_ * dx_);
        const double ry = diffusivity_ * dt / (2.0 * dy_ * dy_);

        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int stride = nx + 1;  // Number of nodes per row

        // Pre-allocate max size for thread-local vectors
        const int max_dim = std::max(nx, ny);

        // ========== STEP 1: Implicit in x, explicit in y ==========
        // Solve: (I - rx*δ_x²)u* = (I + ry*δ_y²)uⁿ for each row
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel
        {
            // Thread-local tridiagonal vectors
            std::vector<double> a_local(max_dim), b_local(max_dim);
            std::vector<double> c_local(max_dim), d_local(max_dim);

#pragma omp for schedule(static)
            for (int j = 1; j < ny; ++j) {
                const int n = nx - 1;

                for (int i = 1; i < nx; ++i) {
                    const int idx = j * stride + i;
                    const int k = i - 1;

                    double u_yy =
                        solution_[idx - stride] - 2.0 * solution_[idx] + solution_[idx + stride];
                    d_local[k] = solution_[idx] + ry * u_yy;
                    a_local[k] = -rx;
                    b_local[k] = 1.0 + 2.0 * rx;
                    c_local[k] = -rx;
                }

                applyTridiagonalBCs_X_local(j, stride, rx, n, a_local, b_local, c_local, d_local);

                auto x_solution = linalg::solve_tridiagonal(
                    std::vector<double>(a_local.begin(), a_local.begin() + n),
                    std::vector<double>(b_local.begin(), b_local.begin() + n),
                    std::vector<double>(c_local.begin(), c_local.begin() + n),
                    std::vector<double>(d_local.begin(), d_local.begin() + n));

                for (int i = 1; i < nx; ++i) {
                    intermediate_[j * stride + i] = x_solution[i - 1];
                }
            }
        }
#else
        // Serial version
        for (int j = 1; j < ny; ++j) {
            const int n = nx - 1;

            for (int i = 1; i < nx; ++i) {
                const int idx = j * stride + i;
                const int k = i - 1;

                double u_yy =
                    solution_[idx - stride] - 2.0 * solution_[idx] + solution_[idx + stride];
                d_[k] = solution_[idx] + ry * u_yy;
                a_[k] = -rx;
                b_[k] = 1.0 + 2.0 * rx;
                c_[k] = -rx;
            }

            applyTridiagonalBCs_X(j, stride, rx, n);

            auto x_solution =
                linalg::solve_tridiagonal(std::vector<double>(a_.begin(), a_.begin() + n),
                                          std::vector<double>(b_.begin(), b_.begin() + n),
                                          std::vector<double>(c_.begin(), c_.begin() + n),
                                          std::vector<double>(d_.begin(), d_.begin() + n));

            for (int i = 1; i < nx; ++i) {
                intermediate_[j * stride + i] = x_solution[i - 1];
            }
        }
#endif

        // Apply boundary values to intermediate solution
        applyBoundaryConditions(intermediate_);

        // ========== STEP 2: Implicit in y, explicit in x ==========
        // Solve: (I - ry*δ_y²)u^{n+1} = (I + rx*δ_x²)u* for each column
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel
        {
            std::vector<double> a_local(max_dim), b_local(max_dim);
            std::vector<double> c_local(max_dim), d_local(max_dim);

#pragma omp for schedule(static)
            for (int i = 1; i < nx; ++i) {
                const int n = ny - 1;

                for (int j = 1; j < ny; ++j) {
                    const int idx = j * stride + i;
                    const int k = j - 1;

                    double u_xx =
                        intermediate_[idx - 1] - 2.0 * intermediate_[idx] + intermediate_[idx + 1];
                    d_local[k] = intermediate_[idx] + rx * u_xx;
                    a_local[k] = -ry;
                    b_local[k] = 1.0 + 2.0 * ry;
                    c_local[k] = -ry;
                }

                applyTridiagonalBCs_Y_local(i, stride, ry, n, a_local, b_local, c_local, d_local);

                auto y_solution = linalg::solve_tridiagonal(
                    std::vector<double>(a_local.begin(), a_local.begin() + n),
                    std::vector<double>(b_local.begin(), b_local.begin() + n),
                    std::vector<double>(c_local.begin(), c_local.begin() + n),
                    std::vector<double>(d_local.begin(), d_local.begin() + n));

                for (int j = 1; j < ny; ++j) {
                    solution_[j * stride + i] = y_solution[j - 1];
                }
            }
        }
#else
        // Serial version
        for (int i = 1; i < nx; ++i) {
            const int n = ny - 1;

            for (int j = 1; j < ny; ++j) {
                const int idx = j * stride + i;
                const int k = j - 1;

                double u_xx =
                    intermediate_[idx - 1] - 2.0 * intermediate_[idx] + intermediate_[idx + 1];
                d_[k] = intermediate_[idx] + rx * u_xx;
                a_[k] = -ry;
                b_[k] = 1.0 + 2.0 * ry;
                c_[k] = -ry;
            }

            applyTridiagonalBCs_Y(i, stride, ry, n);

            auto y_solution =
                linalg::solve_tridiagonal(std::vector<double>(a_.begin(), a_.begin() + n),
                                          std::vector<double>(b_.begin(), b_.begin() + n),
                                          std::vector<double>(c_.begin(), c_.begin() + n),
                                          std::vector<double>(d_.begin(), d_.begin() + n));

            for (int j = 1; j < ny; ++j) {
                solution_[j * stride + i] = y_solution[j - 1];
            }
        }
#endif

        // Apply boundary conditions to final solution
        applyBoundaryConditions(solution_);

        time_ += dt;
        result.time = time_;
        result.success = true;
        return result;
    }

    /**
     * @brief Run the solver for specified number of steps.
     *
     * @param dt Time step size
     * @param num_steps Number of time steps
     * @return ADISolveResult with cumulative statistics
     */
    ADISolveResult solve(double dt, int num_steps) {
        ADISolveResult total_result;
        total_result.steps = 0;
        total_result.substeps = 0;
        total_result.success = true;

        for (int step_count = 0; step_count < num_steps; ++step_count) {
            ADISolveResult result = this->step(dt);
            if (!result.success) {
                total_result.success = false;
                total_result.total_time = time_;
                return total_result;
            }
            total_result.steps++;
            total_result.substeps += result.substeps;
        }
        total_result.total_time = time_;
        return total_result;
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
    std::vector<double> intermediate_;

    // Tridiagonal system vectors (reused)
    std::vector<double> a_, b_, c_, d_;

    std::array<BoundaryCondition, 4> boundary_conditions_;

    double dx_ = 0.0;
    double dy_ = 0.0;
    double time_ = 0.0;

    /**
     * @brief Apply x-direction boundary conditions to tridiagonal system.
     */
    void applyTridiagonalBCs_X(int j, int stride, double rx, int n) {
        const auto& left_bc = boundary_conditions_[to_index(Boundary::Left)];
        const auto& right_bc = boundary_conditions_[to_index(Boundary::Right)];

        // Left boundary affects first equation
        if (left_bc.type == BoundaryType::DIRICHLET) {
            d_[0] += rx * left_bc.value;
        } else {
            // Neumann: u_0 = u_1 - flux*dx
            b_[0] -= rx;  // Adjust diagonal to account for ghost point
            d_[0] -= rx * left_bc.value * dx_;
        }

        // Right boundary affects last equation
        if (right_bc.type == BoundaryType::DIRICHLET) {
            d_[n - 1] += rx * right_bc.value;
        } else {
            // Neumann: u_{nx} = u_{nx-1} + flux*dx
            b_[n - 1] -= rx;
            d_[n - 1] += rx * right_bc.value * dx_;
        }
    }

    /**
     * @brief Apply y-direction boundary conditions to tridiagonal system.
     */
    void applyTridiagonalBCs_Y(int i, int stride, double ry, int n) {
        const auto& bottom_bc = boundary_conditions_[to_index(Boundary::Bottom)];
        const auto& top_bc = boundary_conditions_[to_index(Boundary::Top)];

        // Bottom boundary affects first equation
        if (bottom_bc.type == BoundaryType::DIRICHLET) {
            d_[0] += ry * bottom_bc.value;
        } else {
            b_[0] -= ry;
            d_[0] -= ry * bottom_bc.value * dy_;
        }

        // Top boundary affects last equation
        if (top_bc.type == BoundaryType::DIRICHLET) {
            d_[n - 1] += ry * top_bc.value;
        } else {
            b_[n - 1] -= ry;
            d_[n - 1] += ry * top_bc.value * dy_;
        }
    }

    /**
     * @brief Apply x-direction BCs to local tridiagonal vectors (for OpenMP).
     */
    void applyTridiagonalBCs_X_local(int j, int stride, double rx, int n,
                                     std::vector<double>& a_local, std::vector<double>& b_local,
                                     std::vector<double>& c_local,
                                     std::vector<double>& d_local) const {
        const auto& left_bc = boundary_conditions_[to_index(Boundary::Left)];
        const auto& right_bc = boundary_conditions_[to_index(Boundary::Right)];

        if (left_bc.type == BoundaryType::DIRICHLET) {
            d_local[0] += rx * left_bc.value;
        } else {
            b_local[0] -= rx;
            d_local[0] -= rx * left_bc.value * dx_;
        }

        if (right_bc.type == BoundaryType::DIRICHLET) {
            d_local[n - 1] += rx * right_bc.value;
        } else {
            b_local[n - 1] -= rx;
            d_local[n - 1] += rx * right_bc.value * dx_;
        }
    }

    /**
     * @brief Apply y-direction BCs to local tridiagonal vectors (for OpenMP).
     */
    void applyTridiagonalBCs_Y_local(int i, int stride, double ry, int n,
                                     std::vector<double>& a_local, std::vector<double>& b_local,
                                     std::vector<double>& c_local,
                                     std::vector<double>& d_local) const {
        const auto& bottom_bc = boundary_conditions_[to_index(Boundary::Bottom)];
        const auto& top_bc = boundary_conditions_[to_index(Boundary::Top)];

        if (bottom_bc.type == BoundaryType::DIRICHLET) {
            d_local[0] += ry * bottom_bc.value;
        } else {
            b_local[0] -= ry;
            d_local[0] -= ry * bottom_bc.value * dy_;
        }

        if (top_bc.type == BoundaryType::DIRICHLET) {
            d_local[n - 1] += ry * top_bc.value;
        } else {
            b_local[n - 1] -= ry;
            d_local[n - 1] += ry * top_bc.value * dy_;
        }
    }

    /**
     * @brief Apply boundary conditions to the solution vector.
     */
    void applyBoundaryConditions(std::vector<double>& u) {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int stride = nx + 1;

        const auto& left_bc = boundary_conditions_[to_index(Boundary::Left)];
        const auto& right_bc = boundary_conditions_[to_index(Boundary::Right)];
        const auto& bottom_bc = boundary_conditions_[to_index(Boundary::Bottom)];
        const auto& top_bc = boundary_conditions_[to_index(Boundary::Top)];

        // Left/Right boundaries
        for (int j = 0; j <= ny; ++j) {
            const int left_idx = j * stride;
            const int right_idx = j * stride + nx;

            if (left_bc.type == BoundaryType::DIRICHLET) {
                u[left_idx] = left_bc.value;
            } else {
                u[left_idx] = u[left_idx + 1] - left_bc.value * dx_;
            }

            if (right_bc.type == BoundaryType::DIRICHLET) {
                u[right_idx] = right_bc.value;
            } else {
                u[right_idx] = u[right_idx - 1] + right_bc.value * dx_;
            }
        }

        // Bottom/Top boundaries
        for (int i = 0; i <= nx; ++i) {
            if (bottom_bc.type == BoundaryType::DIRICHLET) {
                u[i] = bottom_bc.value;
            } else {
                u[i] = u[i + stride] - bottom_bc.value * dy_;
            }

            const int top_idx = ny * stride + i;
            if (top_bc.type == BoundaryType::DIRICHLET) {
                u[top_idx] = top_bc.value;
            } else {
                u[top_idx] = u[top_idx - stride] + top_bc.value * dy_;
            }
        }
    }
};

/**
 * @brief 3D ADI solver using Douglas-Gunn splitting.
 *
 * Solves the 3D diffusion equation:
 *   ∂u/∂t = D(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
 *
 * Uses a three-stage Douglas-Gunn ADI scheme that is:
 * - Unconditionally stable
 * - Second-order accurate in space and time
 * - O(N) per time step
 *
 * @code
 *   StructuredMesh3D mesh(1.0, 1.0, 1.0, 20, 20, 20);
 *   ADIDiffusion3D solver(mesh, 1e-5);
 *
 *   solver.setInitialCondition(u0);
 *   solver.setDirichletBoundary(Boundary3D::XMin, 100.0);
 *   solver.setDirichletBoundary(Boundary3D::XMax, 0.0);
 *
 *   solver.solve(0.1, 100);  // 100 steps at dt=0.1
 * @endcode
 */
class ADIDiffusion3D {
public:
    /**
     * @brief Construct a 3D ADI diffusion solver.
     *
     * @param mesh 3D structured mesh
     * @param diffusivity Diffusion coefficient D [m²/s]
     * @throws std::invalid_argument if diffusivity <= 0
     */
    ADIDiffusion3D(const StructuredMesh3D& mesh, double diffusivity)
        : mesh_(mesh), diffusivity_(diffusivity) {
        if (diffusivity <= 0.0) {
            throw std::invalid_argument("Diffusivity must be positive");
        }

        const int num_nodes = mesh.numNodes();
        solution_.resize(num_nodes, 0.0);
        stage1_.resize(num_nodes, 0.0);
        stage2_.resize(num_nodes, 0.0);

        // Pre-allocate tridiagonal vectors
        const int max_dim = std::max({mesh.nx() + 1, mesh.ny() + 1, mesh.nz() + 1});
        a_.resize(max_dim);
        b_.resize(max_dim);
        c_.resize(max_dim);
        d_.resize(max_dim);

        // Default Dirichlet boundary conditions
        for (int i = 0; i < 6; ++i) {
            boundary_conditions_[i] = BoundaryCondition::Dirichlet(0.0);
        }

        // Precompute mesh spacing
        dx_ = mesh.dx();
        dy_ = mesh.dy();
        dz_ = mesh.dz();
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
     * @brief Advance the solution by one time step using Douglas-Gunn ADI.
     *
     * @param dt Time step size [s]
     * @return ADISolveResult with status information
     */
    ADISolveResult step(double dt) {
        if (dt <= 0.0) {
            throw std::invalid_argument("Time step must be positive");
        }

        ADISolveResult result;
        result.substeps = 3;

        // Coefficients
        const double rx = diffusivity_ * dt / (dx_ * dx_);
        const double ry = diffusivity_ * dt / (dy_ * dy_);
        const double rz = diffusivity_ * dt / (dz_ * dz_);

        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int nz = mesh_.nz();
        const int stride_j = mesh_.strideJ();  // (nx+1)
        const int stride_k = mesh_.strideK();  // (nx+1)*(ny+1)

        // Pre-allocate max size for thread-local vectors
        const int max_dim = std::max({nx, ny, nz});

        // ========== STAGE 1: Implicit in x ==========
        // (I - rx*δ_x²)u* = uⁿ + dt*D*∇²uⁿ
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel
        {
            std::vector<double> a_local(max_dim), b_local(max_dim);
            std::vector<double> c_local(max_dim), d_local(max_dim);

#pragma omp for schedule(static) collapse(2)
            for (int k = 1; k < nz; ++k) {
                for (int j = 1; j < ny; ++j) {
                    const int n = nx - 1;

                    for (int i = 1; i < nx; ++i) {
                        const int idx = k * stride_k + j * stride_j + i;
                        const int m = i - 1;

                        double u_xx =
                            (solution_[idx - 1] - 2.0 * solution_[idx] + solution_[idx + 1]) /
                            (dx_ * dx_);
                        double u_yy = (solution_[idx - stride_j] - 2.0 * solution_[idx] +
                                       solution_[idx + stride_j]) /
                                      (dy_ * dy_);
                        double u_zz = (solution_[idx - stride_k] - 2.0 * solution_[idx] +
                                       solution_[idx + stride_k]) /
                                      (dz_ * dz_);

                        d_local[m] = solution_[idx] + dt * diffusivity_ * (u_xx + u_yy + u_zz);
                        a_local[m] = -rx;
                        b_local[m] = 1.0 + 2.0 * rx;
                        c_local[m] = -rx;
                    }

                    applyTridiagonalBCs_X_local(j, k, rx, n, a_local, b_local, c_local, d_local);

                    auto x_sol = linalg::solve_tridiagonal(
                        std::vector<double>(a_local.begin(), a_local.begin() + n),
                        std::vector<double>(b_local.begin(), b_local.begin() + n),
                        std::vector<double>(c_local.begin(), c_local.begin() + n),
                        std::vector<double>(d_local.begin(), d_local.begin() + n));

                    for (int i = 1; i < nx; ++i) {
                        stage1_[k * stride_k + j * stride_j + i] = x_sol[i - 1];
                    }
                }
            }
        }
#else
        // Serial version
        for (int k = 1; k < nz; ++k) {
            for (int j = 1; j < ny; ++j) {
                const int n = nx - 1;

                for (int i = 1; i < nx; ++i) {
                    const int idx = k * stride_k + j * stride_j + i;
                    const int m = i - 1;

                    double u_xx = (solution_[idx - 1] - 2.0 * solution_[idx] + solution_[idx + 1]) /
                                  (dx_ * dx_);
                    double u_yy = (solution_[idx - stride_j] - 2.0 * solution_[idx] +
                                   solution_[idx + stride_j]) /
                                  (dy_ * dy_);
                    double u_zz = (solution_[idx - stride_k] - 2.0 * solution_[idx] +
                                   solution_[idx + stride_k]) /
                                  (dz_ * dz_);

                    d_[m] = solution_[idx] + dt * diffusivity_ * (u_xx + u_yy + u_zz);
                    a_[m] = -rx;
                    b_[m] = 1.0 + 2.0 * rx;
                    c_[m] = -rx;
                }

                applyTridiagonalBCs_X(j, k, rx, n);

                auto x_sol =
                    linalg::solve_tridiagonal(std::vector<double>(a_.begin(), a_.begin() + n),
                                              std::vector<double>(b_.begin(), b_.begin() + n),
                                              std::vector<double>(c_.begin(), c_.begin() + n),
                                              std::vector<double>(d_.begin(), d_.begin() + n));

                for (int i = 1; i < nx; ++i) {
                    stage1_[k * stride_k + j * stride_j + i] = x_sol[i - 1];
                }
            }
        }
#endif
        applyBoundaryConditions(stage1_);

        // ========== STAGE 2: Implicit in y ==========
        // (I - ry*δ_y²)u** = u* + ry*δ_y²uⁿ
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel
        {
            std::vector<double> a_local(max_dim), b_local(max_dim);
            std::vector<double> c_local(max_dim), d_local(max_dim);

#pragma omp for schedule(static) collapse(2)
            for (int k = 1; k < nz; ++k) {
                for (int i = 1; i < nx; ++i) {
                    const int n = ny - 1;

                    for (int j = 1; j < ny; ++j) {
                        const int idx = k * stride_k + j * stride_j + i;
                        const int m = j - 1;

                        double u_yy_old = solution_[idx - stride_j] - 2.0 * solution_[idx] +
                                          solution_[idx + stride_j];
                        d_local[m] = stage1_[idx] + ry * u_yy_old;
                        a_local[m] = -ry;
                        b_local[m] = 1.0 + 2.0 * ry;
                        c_local[m] = -ry;
                    }

                    applyTridiagonalBCs_Y_local(i, k, ry, n, a_local, b_local, c_local, d_local);

                    auto y_sol = linalg::solve_tridiagonal(
                        std::vector<double>(a_local.begin(), a_local.begin() + n),
                        std::vector<double>(b_local.begin(), b_local.begin() + n),
                        std::vector<double>(c_local.begin(), c_local.begin() + n),
                        std::vector<double>(d_local.begin(), d_local.begin() + n));

                    for (int j = 1; j < ny; ++j) {
                        stage2_[k * stride_k + j * stride_j + i] = y_sol[j - 1];
                    }
                }
            }
        }
#else
        // Serial version
        for (int k = 1; k < nz; ++k) {
            for (int i = 1; i < nx; ++i) {
                const int n = ny - 1;

                for (int j = 1; j < ny; ++j) {
                    const int idx = k * stride_k + j * stride_j + i;
                    const int m = j - 1;

                    double u_yy_old = solution_[idx - stride_j] - 2.0 * solution_[idx] +
                                      solution_[idx + stride_j];
                    d_[m] = stage1_[idx] + ry * u_yy_old;
                    a_[m] = -ry;
                    b_[m] = 1.0 + 2.0 * ry;
                    c_[m] = -ry;
                }

                applyTridiagonalBCs_Y(i, k, ry, n);

                auto y_sol =
                    linalg::solve_tridiagonal(std::vector<double>(a_.begin(), a_.begin() + n),
                                              std::vector<double>(b_.begin(), b_.begin() + n),
                                              std::vector<double>(c_.begin(), c_.begin() + n),
                                              std::vector<double>(d_.begin(), d_.begin() + n));

                for (int j = 1; j < ny; ++j) {
                    stage2_[k * stride_k + j * stride_j + i] = y_sol[j - 1];
                }
            }
        }
#endif
        applyBoundaryConditions(stage2_);

        // ========== STAGE 3: Implicit in z ==========
        // (I - rz*δ_z²)u^{n+1} = u** + rz*δ_z²uⁿ
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel
        {
            std::vector<double> a_local(max_dim), b_local(max_dim);
            std::vector<double> c_local(max_dim), d_local(max_dim);

#pragma omp for schedule(static) collapse(2)
            for (int j = 1; j < ny; ++j) {
                for (int i = 1; i < nx; ++i) {
                    const int n = nz - 1;

                    for (int k = 1; k < nz; ++k) {
                        const int idx = k * stride_k + j * stride_j + i;
                        const int m = k - 1;

                        double u_zz_old = solution_[idx - stride_k] - 2.0 * solution_[idx] +
                                          solution_[idx + stride_k];
                        d_local[m] = stage2_[idx] + rz * u_zz_old;
                        a_local[m] = -rz;
                        b_local[m] = 1.0 + 2.0 * rz;
                        c_local[m] = -rz;
                    }

                    applyTridiagonalBCs_Z_local(i, j, rz, n, a_local, b_local, c_local, d_local);

                    auto z_sol = linalg::solve_tridiagonal(
                        std::vector<double>(a_local.begin(), a_local.begin() + n),
                        std::vector<double>(b_local.begin(), b_local.begin() + n),
                        std::vector<double>(c_local.begin(), c_local.begin() + n),
                        std::vector<double>(d_local.begin(), d_local.begin() + n));

                    for (int k = 1; k < nz; ++k) {
                        solution_[k * stride_k + j * stride_j + i] = z_sol[k - 1];
                    }
                }
            }
        }
#else
        // Serial version
        for (int j = 1; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) {
                const int n = nz - 1;

                for (int k = 1; k < nz; ++k) {
                    const int idx = k * stride_k + j * stride_j + i;
                    const int m = k - 1;

                    double u_zz_old = solution_[idx - stride_k] - 2.0 * solution_[idx] +
                                      solution_[idx + stride_k];
                    d_[m] = stage2_[idx] + rz * u_zz_old;
                    a_[m] = -rz;
                    b_[m] = 1.0 + 2.0 * rz;
                    c_[m] = -rz;
                }

                applyTridiagonalBCs_Z(i, j, rz, n);

                auto z_sol =
                    linalg::solve_tridiagonal(std::vector<double>(a_.begin(), a_.begin() + n),
                                              std::vector<double>(b_.begin(), b_.begin() + n),
                                              std::vector<double>(c_.begin(), c_.begin() + n),
                                              std::vector<double>(d_.begin(), d_.begin() + n));

                for (int k = 1; k < nz; ++k) {
                    solution_[k * stride_k + j * stride_j + i] = z_sol[k - 1];
                }
            }
        }
#endif
        applyBoundaryConditions(solution_);

        time_ += dt;
        result.time = time_;
        result.success = true;
        return result;
    }

    /**
     * @brief Run the solver for specified number of steps.
     * @return ADISolveResult with cumulative statistics
     */
    ADISolveResult solve(double dt, int num_steps) {
        ADISolveResult total_result;
        total_result.steps = 0;
        total_result.substeps = 0;
        total_result.success = true;

        for (int step_count = 0; step_count < num_steps; ++step_count) {
            ADISolveResult result = this->step(dt);
            if (!result.success) {
                total_result.success = false;
                total_result.total_time = time_;
                return total_result;
            }
            total_result.steps++;
            total_result.substeps += result.substeps;
        }
        total_result.total_time = time_;
        return total_result;
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

private:
    const StructuredMesh3D& mesh_;
    double diffusivity_;
    std::vector<double> solution_;
    std::vector<double> stage1_;
    std::vector<double> stage2_;

    std::vector<double> a_, b_, c_, d_;

    std::array<BoundaryCondition, 6> boundary_conditions_;

    double dx_ = 0.0;
    double dy_ = 0.0;
    double dz_ = 0.0;
    double time_ = 0.0;

    void applyTridiagonalBCs_X(int j, int k, double rx, int n) {
        const auto& xmin_bc = boundary_conditions_[to_index(Boundary3D::XMin)];
        const auto& xmax_bc = boundary_conditions_[to_index(Boundary3D::XMax)];

        if (xmin_bc.type == BoundaryType::DIRICHLET) {
            d_[0] += rx * xmin_bc.value;
        } else {
            b_[0] -= rx;
            d_[0] -= rx * xmin_bc.value * dx_;
        }

        if (xmax_bc.type == BoundaryType::DIRICHLET) {
            d_[n - 1] += rx * xmax_bc.value;
        } else {
            b_[n - 1] -= rx;
            d_[n - 1] += rx * xmax_bc.value * dx_;
        }
    }

    void applyTridiagonalBCs_Y(int i, int k, double ry, int n) {
        const auto& ymin_bc = boundary_conditions_[to_index(Boundary3D::YMin)];
        const auto& ymax_bc = boundary_conditions_[to_index(Boundary3D::YMax)];

        if (ymin_bc.type == BoundaryType::DIRICHLET) {
            d_[0] += ry * ymin_bc.value;
        } else {
            b_[0] -= ry;
            d_[0] -= ry * ymin_bc.value * dy_;
        }

        if (ymax_bc.type == BoundaryType::DIRICHLET) {
            d_[n - 1] += ry * ymax_bc.value;
        } else {
            b_[n - 1] -= ry;
            d_[n - 1] += ry * ymax_bc.value * dy_;
        }
    }

    void applyTridiagonalBCs_Z(int i, int j, double rz, int n) {
        const auto& zmin_bc = boundary_conditions_[to_index(Boundary3D::ZMin)];
        const auto& zmax_bc = boundary_conditions_[to_index(Boundary3D::ZMax)];

        if (zmin_bc.type == BoundaryType::DIRICHLET) {
            d_[0] += rz * zmin_bc.value;
        } else {
            b_[0] -= rz;
            d_[0] -= rz * zmin_bc.value * dz_;
        }

        if (zmax_bc.type == BoundaryType::DIRICHLET) {
            d_[n - 1] += rz * zmax_bc.value;
        } else {
            b_[n - 1] -= rz;
            d_[n - 1] += rz * zmax_bc.value * dz_;
        }
    }

    // Thread-local versions for OpenMP
    void applyTridiagonalBCs_X_local(int j, int k, double rx, int n, std::vector<double>& a_local,
                                     std::vector<double>& b_local, std::vector<double>& c_local,
                                     std::vector<double>& d_local) const {
        const auto& xmin_bc = boundary_conditions_[to_index(Boundary3D::XMin)];
        const auto& xmax_bc = boundary_conditions_[to_index(Boundary3D::XMax)];

        if (xmin_bc.type == BoundaryType::DIRICHLET) {
            d_local[0] += rx * xmin_bc.value;
        } else {
            b_local[0] -= rx;
            d_local[0] -= rx * xmin_bc.value * dx_;
        }

        if (xmax_bc.type == BoundaryType::DIRICHLET) {
            d_local[n - 1] += rx * xmax_bc.value;
        } else {
            b_local[n - 1] -= rx;
            d_local[n - 1] += rx * xmax_bc.value * dx_;
        }
    }

    void applyTridiagonalBCs_Y_local(int i, int k, double ry, int n, std::vector<double>& a_local,
                                     std::vector<double>& b_local, std::vector<double>& c_local,
                                     std::vector<double>& d_local) const {
        const auto& ymin_bc = boundary_conditions_[to_index(Boundary3D::YMin)];
        const auto& ymax_bc = boundary_conditions_[to_index(Boundary3D::YMax)];

        if (ymin_bc.type == BoundaryType::DIRICHLET) {
            d_local[0] += ry * ymin_bc.value;
        } else {
            b_local[0] -= ry;
            d_local[0] -= ry * ymin_bc.value * dy_;
        }

        if (ymax_bc.type == BoundaryType::DIRICHLET) {
            d_local[n - 1] += ry * ymax_bc.value;
        } else {
            b_local[n - 1] -= ry;
            d_local[n - 1] += ry * ymax_bc.value * dy_;
        }
    }

    void applyTridiagonalBCs_Z_local(int i, int j, double rz, int n, std::vector<double>& a_local,
                                     std::vector<double>& b_local, std::vector<double>& c_local,
                                     std::vector<double>& d_local) const {
        const auto& zmin_bc = boundary_conditions_[to_index(Boundary3D::ZMin)];
        const auto& zmax_bc = boundary_conditions_[to_index(Boundary3D::ZMax)];

        if (zmin_bc.type == BoundaryType::DIRICHLET) {
            d_local[0] += rz * zmin_bc.value;
        } else {
            b_local[0] -= rz;
            d_local[0] -= rz * zmin_bc.value * dz_;
        }

        if (zmax_bc.type == BoundaryType::DIRICHLET) {
            d_local[n - 1] += rz * zmax_bc.value;
        } else {
            b_local[n - 1] -= rz;
            d_local[n - 1] += rz * zmax_bc.value * dz_;
        }
    }

    void applyBoundaryConditions(std::vector<double>& u) {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int nz = mesh_.nz();
        const int stride_j = mesh_.strideJ();
        const int stride_k = mesh_.strideK();

        const auto& xmin_bc = boundary_conditions_[to_index(Boundary3D::XMin)];
        const auto& xmax_bc = boundary_conditions_[to_index(Boundary3D::XMax)];
        const auto& ymin_bc = boundary_conditions_[to_index(Boundary3D::YMin)];
        const auto& ymax_bc = boundary_conditions_[to_index(Boundary3D::YMax)];
        const auto& zmin_bc = boundary_conditions_[to_index(Boundary3D::ZMin)];
        const auto& zmax_bc = boundary_conditions_[to_index(Boundary3D::ZMax)];

        // X boundaries
        for (int k = 0; k <= nz; ++k) {
            for (int j = 0; j <= ny; ++j) {
                int idx_min = k * stride_k + j * stride_j + 0;
                int idx_max = k * stride_k + j * stride_j + nx;

                if (xmin_bc.type == BoundaryType::DIRICHLET) {
                    u[idx_min] = xmin_bc.value;
                } else {
                    u[idx_min] = u[idx_min + 1] - xmin_bc.value * dx_;
                }

                if (xmax_bc.type == BoundaryType::DIRICHLET) {
                    u[idx_max] = xmax_bc.value;
                } else {
                    u[idx_max] = u[idx_max - 1] + xmax_bc.value * dx_;
                }
            }
        }

        // Y boundaries
        for (int k = 0; k <= nz; ++k) {
            for (int i = 0; i <= nx; ++i) {
                int idx_min = k * stride_k + 0 * stride_j + i;
                int idx_max = k * stride_k + ny * stride_j + i;

                if (ymin_bc.type == BoundaryType::DIRICHLET) {
                    u[idx_min] = ymin_bc.value;
                } else {
                    u[idx_min] = u[idx_min + stride_j] - ymin_bc.value * dy_;
                }

                if (ymax_bc.type == BoundaryType::DIRICHLET) {
                    u[idx_max] = ymax_bc.value;
                } else {
                    u[idx_max] = u[idx_max - stride_j] + ymax_bc.value * dy_;
                }
            }
        }

        // Z boundaries
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                int idx_min = 0 * stride_k + j * stride_j + i;
                int idx_max = nz * stride_k + j * stride_j + i;

                if (zmin_bc.type == BoundaryType::DIRICHLET) {
                    u[idx_min] = zmin_bc.value;
                } else {
                    u[idx_min] = u[idx_min + stride_k] - zmin_bc.value * dz_;
                }

                if (zmax_bc.type == BoundaryType::DIRICHLET) {
                    u[idx_max] = zmax_bc.value;
                } else {
                    u[idx_max] = u[idx_max - stride_k] + zmax_bc.value * dz_;
                }
            }
        }
    }
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_ADI_SOLVER_HPP
