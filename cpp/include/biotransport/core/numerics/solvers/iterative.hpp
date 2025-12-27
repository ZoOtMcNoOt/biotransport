#ifndef BIOTRANSPORT_CORE_NUMERICS_SOLVERS_ITERATIVE_HPP
#define BIOTRANSPORT_CORE_NUMERICS_SOLVERS_ITERATIVE_HPP

/**
 * @file iterative.hpp
 * @brief Iterative solvers for linear and nonlinear systems.
 *
 * This header provides basic iterative methods commonly used in
 * finite difference/finite volume solvers:
 *   - Jacobi iteration
 *   - Gauss-Seidel iteration
 *   - Successive Over-Relaxation (SOR)
 *   - Red-Black Gauss-Seidel (for parallel efficiency)
 *
 * These are primarily used for solving the pressure Poisson equation
 * in incompressible flow solvers (Stokes, Navier-Stokes).
 */

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace biotransport {
namespace solvers {

/**
 * @brief Result of iterative solve.
 */
struct IterativeResult {
    bool converged;   ///< Whether solver converged
    int iterations;   ///< Number of iterations performed
    double residual;  ///< Final residual norm
};

/**
 * @brief Apply one Jacobi iteration for 2D Poisson equation.
 *
 * Solves: ∇²p = f on a uniform grid with Dirichlet BCs.
 *
 * @param p Current solution (modified in place)
 * @param f Right-hand side
 * @param nx Number of cells in x
 * @param ny Number of cells in y
 * @param dx Grid spacing in x
 * @param dy Grid spacing in y
 * @return Maximum change in solution
 */
inline double jacobi_step_2d(std::vector<double>& p, const std::vector<double>& f, int nx, int ny,
                             double dx, double dy) {
    const int stride = nx + 1;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double denom = 2.0 * (1.0 / dx2 + 1.0 / dy2);

    std::vector<double> p_new = p;
    double max_diff = 0.0;

    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            const int idx = j * stride + i;

            double p_old = p[idx];
            p_new[idx] = ((p[idx + 1] + p[idx - 1]) / dx2 +
                          (p[idx + stride] + p[idx - stride]) / dy2 - f[idx]) /
                         denom;

            max_diff = std::max(max_diff, std::abs(p_new[idx] - p_old));
        }
    }

    p = std::move(p_new);
    return max_diff;
}

/**
 * @brief Apply one Gauss-Seidel iteration for 2D Poisson equation.
 *
 * More efficient than Jacobi (uses updated values immediately).
 *
 * @param p Current solution (modified in place)
 * @param f Right-hand side
 * @param nx Number of cells in x
 * @param ny Number of cells in y
 * @param dx Grid spacing in x
 * @param dy Grid spacing in y
 * @return Maximum change in solution
 */
inline double gauss_seidel_step_2d(std::vector<double>& p, const std::vector<double>& f, int nx,
                                   int ny, double dx, double dy) {
    const int stride = nx + 1;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double denom = 2.0 * (1.0 / dx2 + 1.0 / dy2);

    double max_diff = 0.0;

    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            const int idx = j * stride + i;

            double p_old = p[idx];
            p[idx] = ((p[idx + 1] + p[idx - 1]) / dx2 + (p[idx + stride] + p[idx - stride]) / dy2 -
                      f[idx]) /
                     denom;

            max_diff = std::max(max_diff, std::abs(p[idx] - p_old));
        }
    }

    return max_diff;
}

/**
 * @brief Apply one SOR (Successive Over-Relaxation) iteration.
 *
 * Accelerates Gauss-Seidel with relaxation factor omega.
 * Optimal omega is typically 1.5-1.9 for Poisson problems.
 *
 * @param p Current solution (modified in place)
 * @param f Right-hand side
 * @param nx Number of cells in x
 * @param ny Number of cells in y
 * @param dx Grid spacing in x
 * @param dy Grid spacing in y
 * @param omega Relaxation factor (1 < omega < 2 for over-relaxation)
 * @return Maximum change in solution
 */
inline double sor_step_2d(std::vector<double>& p, const std::vector<double>& f, int nx, int ny,
                          double dx, double dy, double omega) {
    const int stride = nx + 1;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double denom = 2.0 * (1.0 / dx2 + 1.0 / dy2);

    double max_diff = 0.0;

    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            const int idx = j * stride + i;

            double p_old = p[idx];
            double p_gs = ((p[idx + 1] + p[idx - 1]) / dx2 +
                           (p[idx + stride] + p[idx - stride]) / dy2 - f[idx]) /
                          denom;

            p[idx] = (1.0 - omega) * p_old + omega * p_gs;

            max_diff = std::max(max_diff, std::abs(p[idx] - p_old));
        }
    }

    return max_diff;
}

/**
 * @brief Red-Black Gauss-Seidel for 2D Poisson (parallelizable).
 *
 * Updates "red" nodes (i+j even), then "black" nodes (i+j odd).
 * Each phase is embarrassingly parallel.
 *
 * @param p Current solution (modified in place)
 * @param f Right-hand side
 * @param nx Number of cells in x
 * @param ny Number of cells in y
 * @param dx Grid spacing in x
 * @param dy Grid spacing in y
 * @return Maximum change in solution
 */
inline double red_black_gs_step_2d(std::vector<double>& p, const std::vector<double>& f, int nx,
                                   int ny, double dx, double dy) {
    const int stride = nx + 1;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double denom = 2.0 * (1.0 / dx2 + 1.0 / dy2);

    double max_diff = 0.0;

    // Red nodes (i + j even)
    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            if ((i + j) % 2 != 0)
                continue;

            const int idx = j * stride + i;
            double p_old = p[idx];
            p[idx] = ((p[idx + 1] + p[idx - 1]) / dx2 + (p[idx + stride] + p[idx - stride]) / dy2 -
                      f[idx]) /
                     denom;
            max_diff = std::max(max_diff, std::abs(p[idx] - p_old));
        }
    }

    // Black nodes (i + j odd)
    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            if ((i + j) % 2 == 0)
                continue;

            const int idx = j * stride + i;
            double p_old = p[idx];
            p[idx] = ((p[idx + 1] + p[idx - 1]) / dx2 + (p[idx + stride] + p[idx - stride]) / dy2 -
                      f[idx]) /
                     denom;
            max_diff = std::max(max_diff, std::abs(p[idx] - p_old));
        }
    }

    return max_diff;
}

/**
 * @brief Compute L2 residual norm for Poisson equation.
 *
 * Residual r = f - ∇²p
 *
 * @param p Current solution
 * @param f Right-hand side
 * @param nx Number of cells in x
 * @param ny Number of cells in y
 * @param dx Grid spacing in x
 * @param dy Grid spacing in y
 * @return L2 norm of residual
 */
inline double residual_norm_2d(const std::vector<double>& p, const std::vector<double>& f, int nx,
                               int ny, double dx, double dy) {
    const int stride = nx + 1;
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);

    double sum_sq = 0.0;
    int count = 0;

    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            const int idx = j * stride + i;

            double laplacian = inv_dx2 * (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) +
                               inv_dy2 * (p[idx + stride] - 2.0 * p[idx] + p[idx - stride]);

            double residual = f[idx] - laplacian;
            sum_sq += residual * residual;
            ++count;
        }
    }

    return (count > 0) ? std::sqrt(sum_sq / count) : 0.0;
}

/**
 * @brief Solve 2D Poisson equation using Gauss-Seidel.
 *
 * @param p Initial guess / solution (modified)
 * @param f Right-hand side
 * @param nx Number of cells in x
 * @param ny Number of cells in y
 * @param dx Grid spacing in x
 * @param dy Grid spacing in y
 * @param tol Convergence tolerance
 * @param max_iter Maximum iterations
 * @return Solve result
 */
[[nodiscard]] inline IterativeResult solve_poisson_gs(std::vector<double>& p,
                                                      const std::vector<double>& f, int nx, int ny,
                                                      double dx, double dy, double tol = 1e-6,
                                                      int max_iter = 10000) {
    IterativeResult result{false, 0, 1.0};

    for (int iter = 0; iter < max_iter; ++iter) {
        double max_diff = gauss_seidel_step_2d(p, f, nx, ny, dx, dy);
        result.iterations = iter + 1;
        result.residual = max_diff;

        if (max_diff < tol) {
            result.converged = true;
            break;
        }
    }

    return result;
}

/**
 * @brief Solve 2D Poisson equation using SOR.
 *
 * @param p Initial guess / solution (modified)
 * @param f Right-hand side
 * @param nx Number of cells in x
 * @param ny Number of cells in y
 * @param dx Grid spacing in x
 * @param dy Grid spacing in y
 * @param omega Relaxation factor
 * @param tol Convergence tolerance
 * @param max_iter Maximum iterations
 * @return Solve result
 */
[[nodiscard]] inline IterativeResult solve_poisson_sor(std::vector<double>& p,
                                                       const std::vector<double>& f, int nx, int ny,
                                                       double dx, double dy, double omega = 1.5,
                                                       double tol = 1e-6, int max_iter = 10000) {
    IterativeResult result{false, 0, 1.0};

    for (int iter = 0; iter < max_iter; ++iter) {
        double max_diff = sor_step_2d(p, f, nx, ny, dx, dy, omega);
        result.iterations = iter + 1;
        result.residual = max_diff;

        if (max_diff < tol) {
            result.converged = true;
            break;
        }
    }

    return result;
}

}  // namespace solvers
}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_NUMERICS_SOLVERS_ITERATIVE_HPP
