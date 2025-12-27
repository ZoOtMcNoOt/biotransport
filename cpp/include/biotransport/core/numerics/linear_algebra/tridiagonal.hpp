#ifndef BIOTRANSPORT_CORE_NUMERICS_LINEAR_ALGEBRA_TRIDIAGONAL_HPP
#define BIOTRANSPORT_CORE_NUMERICS_LINEAR_ALGEBRA_TRIDIAGONAL_HPP

/**
 * @file tridiagonal.hpp
 * @brief Tridiagonal matrix solver using Thomas algorithm.
 *
 * The Thomas algorithm (TDMA) efficiently solves tridiagonal systems:
 *   a_i * x_{i-1} + b_i * x_i + c_i * x_{i+1} = d_i
 *
 * This is O(n) and is the foundation for implicit 1D diffusion solvers.
 *
 * Usage:
 * @code
 *   std::vector<double> a = {...};  // sub-diagonal
 *   std::vector<double> b = {...};  // diagonal
 *   std::vector<double> c = {...};  // super-diagonal
 *   std::vector<double> d = {...};  // RHS
 *   auto x = solve_tridiagonal(a, b, c, d);
 * @endcode
 */

#include <cmath>
#include <stdexcept>
#include <vector>

namespace biotransport {
namespace linalg {

/**
 * @brief Solve a tridiagonal system using the Thomas algorithm.
 *
 * Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
 *
 * @param a Sub-diagonal coefficients (size n, a[0] unused)
 * @param b Diagonal coefficients (size n)
 * @param c Super-diagonal coefficients (size n, c[n-1] unused)
 * @param d Right-hand side (size n)
 * @return Solution vector x (size n)
 *
 * @throws std::invalid_argument if sizes don't match
 * @throws std::runtime_error if matrix is singular
 */
inline std::vector<double> solve_tridiagonal(const std::vector<double>& a,
                                             const std::vector<double>& b,
                                             const std::vector<double>& c,
                                             const std::vector<double>& d) {
    const size_t n = b.size();

    if (a.size() != n || c.size() != n || d.size() != n) {
        throw std::invalid_argument("All vectors must have the same size");
    }

    if (n == 0) {
        return {};
    }

    // Working arrays for modified coefficients
    std::vector<double> c_star(n);
    std::vector<double> d_star(n);
    std::vector<double> x(n);

    // Forward sweep
    if (std::abs(b[0]) < 1e-15) {
        throw std::runtime_error("Tridiagonal matrix is singular (zero pivot at i=0)");
    }

    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for (size_t i = 1; i < n; ++i) {
        double denom = b[i] - a[i] * c_star[i - 1];
        if (std::abs(denom) < 1e-15) {
            throw std::runtime_error(
                "Tridiagonal matrix is singular (zero pivot at i=" + std::to_string(i) + ")");
        }
        c_star[i] = c[i] / denom;
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / denom;
    }

    // Back substitution
    x[n - 1] = d_star[n - 1];
    for (size_t i = n - 1; i > 0; --i) {
        x[i - 1] = d_star[i - 1] - c_star[i - 1] * x[i];
    }

    return x;
}

/**
 * @brief In-place tridiagonal solve (overwrites d with solution).
 *
 * More memory-efficient version that modifies input vectors.
 *
 * @param a Sub-diagonal coefficients (modified)
 * @param b Diagonal coefficients (modified)
 * @param c Super-diagonal coefficients (modified)
 * @param d Right-hand side, overwritten with solution
 */
inline void solve_tridiagonal_inplace(std::vector<double>& a, std::vector<double>& b,
                                      std::vector<double>& c, std::vector<double>& d) {
    const size_t n = b.size();

    if (n == 0)
        return;

    // Forward elimination
    for (size_t i = 1; i < n; ++i) {
        double m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        d[i] -= m * d[i - 1];
    }

    // Back substitution
    d[n - 1] /= b[n - 1];
    for (size_t i = n - 1; i > 0; --i) {
        d[i - 1] = (d[i - 1] - c[i - 1] * d[i]) / b[i - 1];
    }
}

/**
 * @brief Solve a cyclic (periodic) tridiagonal system.
 *
 * Uses Sherman-Morrison formula to handle the corner elements.
 * Useful for periodic boundary conditions.
 *
 * @param a Sub-diagonal coefficients
 * @param b Diagonal coefficients
 * @param c Super-diagonal coefficients
 * @param d Right-hand side
 * @param alpha Corner element (n-1, 0)
 * @param beta Corner element (0, n-1)
 * @return Solution vector
 */
inline std::vector<double> solve_cyclic_tridiagonal(std::vector<double> a, std::vector<double> b,
                                                    std::vector<double> c, std::vector<double> d,
                                                    double alpha, double beta) {
    const size_t n = b.size();

    if (n < 3) {
        throw std::invalid_argument("Cyclic tridiagonal requires n >= 3");
    }

    // Modify system using Sherman-Morrison
    double gamma = -b[0];
    b[0] -= gamma;
    b[n - 1] -= alpha * beta / gamma;

    // Create auxiliary vector u
    std::vector<double> u(n, 0.0);
    u[0] = gamma;
    u[n - 1] = alpha;

    // Solve two systems
    auto y = solve_tridiagonal(a, b, c, d);
    auto z = solve_tridiagonal(a, b, c, u);

    // Compute solution using Sherman-Morrison
    double factor = (y[0] + beta * y[n - 1] / gamma) / (1.0 + z[0] + beta * z[n - 1] / gamma);

    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = y[i] - factor * z[i];
    }

    return x;
}

}  // namespace linalg
}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_NUMERICS_LINEAR_ALGEBRA_TRIDIAGONAL_HPP
