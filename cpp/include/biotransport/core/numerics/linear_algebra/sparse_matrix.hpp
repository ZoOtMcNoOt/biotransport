/**
 * @file sparse_matrix.hpp
 * @brief Sparse matrix interface using Eigen.
 *
 * This module provides a high-level interface to Eigen's sparse matrix
 * functionality, optimized for PDE discretizations (FEM, FDM).
 *
 * Key features:
 * - SparseMatrix wrapper with convenient assembly API
 * - Multiple solver backends (LU, Cholesky, iterative)
 * - Triplet-based assembly for efficient construction
 * - Integration with biotransport mesh types
 *
 * @author BioTransport Development Team
 * @date December 2025
 */

#ifndef BIOTRANSPORT_CORE_NUMERICS_LINEAR_ALGEBRA_SPARSE_MATRIX_HPP
#define BIOTRANSPORT_CORE_NUMERICS_LINEAR_ALGEBRA_SPARSE_MATRIX_HPP

#ifdef BIOTRANSPORT_ENABLE_EIGEN

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace biotransport {
namespace linalg {

/**
 * @brief Sparse solver type enumeration.
 */
enum class SparseSolverType {
    SparseLU,           ///< Direct LU decomposition (general matrices)
    SimplicialLLT,      ///< Cholesky LLT (SPD matrices, fastest)
    SimplicialLDLT,     ///< Cholesky LDLT (symmetric matrices)
    ConjugateGradient,  ///< Iterative CG (SPD matrices, memory efficient)
    BiCGSTAB            ///< Iterative BiCGSTAB (general matrices)
};

/**
 * @brief Result of a sparse linear solve.
 */
struct SparseSolveResult {
    bool success = false;                ///< Whether solve succeeded
    int iterations = 0;                  ///< Number of iterations (for iterative solvers)
    double residual = 0.0;               ///< Final residual norm
    double factorization_time_ms = 0.0;  ///< Factorization time
    double solve_time_ms = 0.0;          ///< Solve time
    std::string error_message;           ///< Error message if failed
};

/**
 * @brief Triplet for sparse matrix assembly.
 */
struct Triplet {
    int row;
    int col;
    double value;

    Triplet(int r, int c, double v) : row(r), col(c), value(v) {}
};

/**
 * @brief Sparse matrix class wrapping Eigen::SparseMatrix.
 *
 * Provides a convenient interface for assembling and solving sparse
 * linear systems arising from PDE discretizations.
 *
 * Usage:
 * @code
 *   // Assembly from triplets
 *   SparseMatrix A(n, n);
 *   A.reserve(5 * n);  // Estimate 5 non-zeros per row
 *   A.addEntry(0, 0, 4.0);
 *   A.addEntry(0, 1, -1.0);
 *   // ... add more entries
 *   A.finalize();
 *
 *   // Solve Ax = b
 *   std::vector<double> b = {...};
 *   auto result = A.solve(b, SparseSolverType::SparseLU);
 *   std::vector<double> x = result.solution;
 * @endcode
 */
class SparseMatrix {
public:
    // Note: SparseLU requires column-major storage
    using EigenSparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;
    using EigenTriplet = Eigen::Triplet<double>;

    /**
     * @brief Construct an empty sparse matrix.
     */
    SparseMatrix() : rows_(0), cols_(0), finalized_(false) {}

    /**
     * @brief Construct a sparse matrix of given dimensions.
     */
    SparseMatrix(int rows, int cols) : rows_(rows), cols_(cols), finalized_(false) {
        matrix_.resize(rows, cols);
    }

    /**
     * @brief Reserve space for estimated number of non-zeros.
     */
    void reserve(int nnz_estimate) { triplets_.reserve(nnz_estimate); }

    /**
     * @brief Add a single entry to the matrix.
     *
     * Duplicate entries at the same position are summed.
     */
    void addEntry(int row, int col, double value) {
        if (finalized_) {
            throw std::runtime_error("Cannot add entries after finalization");
        }
        if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
            throw std::out_of_range("Matrix index out of range");
        }
        triplets_.emplace_back(row, col, value);
    }

    /**
     * @brief Add multiple entries from a vector of triplets.
     */
    void addEntries(const std::vector<Triplet>& entries) {
        for (const auto& t : entries) {
            addEntry(t.row, t.col, t.value);
        }
    }

    /**
     * @brief Add entries for a 5-point stencil at position (i, j).
     *
     * Useful for 2D Laplacian discretization.
     */
    void add5PointStencil(int idx, int idx_left, int idx_right, int idx_bottom, int idx_top,
                          double center, double off_diag) {
        addEntry(idx, idx, center);
        if (idx_left >= 0)
            addEntry(idx, idx_left, off_diag);
        if (idx_right >= 0 && idx_right < cols_)
            addEntry(idx, idx_right, off_diag);
        if (idx_bottom >= 0)
            addEntry(idx, idx_bottom, off_diag);
        if (idx_top >= 0 && idx_top < cols_)
            addEntry(idx, idx_top, off_diag);
    }

    /**
     * @brief Finalize the matrix after assembly.
     *
     * Must be called before solving. Converts triplets to compressed format.
     */
    void finalize() {
        if (finalized_)
            return;

        std::vector<EigenTriplet> eigen_triplets;
        eigen_triplets.reserve(triplets_.size());
        for (const auto& t : triplets_) {
            eigen_triplets.emplace_back(t.row, t.col, t.value);
        }

        matrix_.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
        matrix_.makeCompressed();
        finalized_ = true;

        // Clear triplets to save memory
        triplets_.clear();
        triplets_.shrink_to_fit();
    }

    /**
     * @brief Check if matrix has been finalized.
     */
    bool isFinalized() const { return finalized_; }

    /**
     * @brief Get number of rows.
     */
    int rows() const { return rows_; }

    /**
     * @brief Get number of columns.
     */
    int cols() const { return cols_; }

    /**
     * @brief Get number of non-zero entries.
     */
    int nonZeros() const { return static_cast<int>(matrix_.nonZeros()); }

    /**
     * @brief Access underlying Eigen matrix.
     */
    const EigenSparseMatrix& eigenMatrix() const { return matrix_; }

    /**
     * @brief Solve Ax = b using the specified solver.
     *
     * @param b Right-hand side vector
     * @param solver_type Type of solver to use
     * @param tolerance Convergence tolerance (for iterative solvers)
     * @param max_iterations Maximum iterations (for iterative solvers)
     * @return Solution vector
     */
    std::vector<double> solve(const std::vector<double>& b,
                              SparseSolverType solver_type = SparseSolverType::SparseLU,
                              double tolerance = 1e-10, int max_iterations = 1000) const {
        if (!finalized_) {
            throw std::runtime_error("Matrix must be finalized before solving");
        }
        if (static_cast<int>(b.size()) != rows_) {
            throw std::invalid_argument("RHS size doesn't match matrix rows");
        }

        // Convert to Eigen vector
        Eigen::VectorXd b_eigen = Eigen::Map<const Eigen::VectorXd>(b.data(), b.size());
        Eigen::VectorXd x_eigen;

        switch (solver_type) {
            case SparseSolverType::SparseLU: {
                Eigen::SparseLU<EigenSparseMatrix> solver;
                solver.compute(matrix_);
                if (solver.info() != Eigen::Success) {
                    throw std::runtime_error("SparseLU factorization failed");
                }
                x_eigen = solver.solve(b_eigen);
                break;
            }
            case SparseSolverType::SimplicialLLT: {
                Eigen::SimplicialLLT<EigenSparseMatrix> solver;
                solver.compute(matrix_);
                if (solver.info() != Eigen::Success) {
                    throw std::runtime_error("SimplicialLLT factorization failed");
                }
                x_eigen = solver.solve(b_eigen);
                break;
            }
            case SparseSolverType::SimplicialLDLT: {
                Eigen::SimplicialLDLT<EigenSparseMatrix> solver;
                solver.compute(matrix_);
                if (solver.info() != Eigen::Success) {
                    throw std::runtime_error("SimplicialLDLT factorization failed");
                }
                x_eigen = solver.solve(b_eigen);
                break;
            }
            case SparseSolverType::ConjugateGradient: {
                Eigen::ConjugateGradient<EigenSparseMatrix, Eigen::Lower | Eigen::Upper> solver;
                solver.setTolerance(tolerance);
                solver.setMaxIterations(max_iterations);
                solver.compute(matrix_);
                x_eigen = solver.solve(b_eigen);
                if (solver.info() != Eigen::Success) {
                    throw std::runtime_error("ConjugateGradient failed to converge");
                }
                break;
            }
            case SparseSolverType::BiCGSTAB: {
                Eigen::BiCGSTAB<EigenSparseMatrix> solver;
                solver.setTolerance(tolerance);
                solver.setMaxIterations(max_iterations);
                solver.compute(matrix_);
                x_eigen = solver.solve(b_eigen);
                if (solver.info() != Eigen::Success) {
                    throw std::runtime_error("BiCGSTAB failed to converge");
                }
                break;
            }
        }

        // Convert back to std::vector
        std::vector<double> x(x_eigen.data(), x_eigen.data() + x_eigen.size());
        return x;
    }

    /**
     * @brief Solve with detailed result information.
     */
    SparseSolveResult solveWithInfo(const std::vector<double>& b,
                                    SparseSolverType solver_type = SparseSolverType::SparseLU,
                                    double tolerance = 1e-10, int max_iterations = 1000) const {
        SparseSolveResult result;

        if (!finalized_) {
            result.error_message = "Matrix must be finalized before solving";
            return result;
        }

        try {
            auto x = solve(b, solver_type, tolerance, max_iterations);
            result.success = true;

            // Compute residual
            Eigen::VectorXd b_eigen = Eigen::Map<const Eigen::VectorXd>(b.data(), b.size());
            Eigen::VectorXd x_eigen = Eigen::Map<const Eigen::VectorXd>(x.data(), x.size());
            result.residual = (matrix_ * x_eigen - b_eigen).norm();

        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = e.what();
        }

        return result;
    }

    /**
     * @brief Matrix-vector product: y = A * x
     */
    std::vector<double> multiply(const std::vector<double>& x) const {
        if (!finalized_) {
            throw std::runtime_error("Matrix must be finalized before multiplication");
        }
        if (static_cast<int>(x.size()) != cols_) {
            throw std::invalid_argument("Vector size doesn't match matrix columns");
        }

        Eigen::VectorXd x_eigen = Eigen::Map<const Eigen::VectorXd>(x.data(), x.size());
        Eigen::VectorXd y_eigen = matrix_ * x_eigen;

        return std::vector<double>(y_eigen.data(), y_eigen.data() + y_eigen.size());
    }

    /**
     * @brief Clear the matrix for reuse.
     */
    void clear() {
        triplets_.clear();
        matrix_.resize(0, 0);
        finalized_ = false;
    }

    /**
     * @brief Resize the matrix (clears existing data).
     */
    void resize(int rows, int cols) {
        rows_ = rows;
        cols_ = cols;
        clear();
        matrix_.resize(rows, cols);
    }

private:
    int rows_;
    int cols_;
    bool finalized_;
    std::vector<Triplet> triplets_;
    EigenSparseMatrix matrix_;
};

/**
 * @brief Build a 2D Laplacian matrix for a structured mesh.
 *
 * Creates the discretization matrix for:
 *   -∇²u = f  on [0,Lx] × [0,Ly]
 *
 * with Dirichlet boundary conditions (rows set to identity).
 *
 * @param nx Number of cells in x
 * @param ny Number of cells in y
 * @param dx Grid spacing in x
 * @param dy Grid spacing in y
 * @return Sparse Laplacian matrix of size (nx+1)*(ny+1)
 */
inline SparseMatrix build2DLaplacian(int nx, int ny, double dx, double dy) {
    const int nodes_x = nx + 1;
    const int nodes_y = ny + 1;
    const int n = nodes_x * nodes_y;

    SparseMatrix A(n, n);
    A.reserve(5 * n);

    const double rx = 1.0 / (dx * dx);
    const double ry = 1.0 / (dy * dy);
    const double center = 2.0 * rx + 2.0 * ry;

    auto idx = [nodes_x](int i, int j) {
        return j * nodes_x + i;
    };

    for (int j = 0; j < nodes_y; ++j) {
        for (int i = 0; i < nodes_x; ++i) {
            int k = idx(i, j);

            // Boundary: identity row
            if (i == 0 || i == nx || j == 0 || j == ny) {
                A.addEntry(k, k, 1.0);
            } else {
                // Interior: 5-point stencil
                A.addEntry(k, k, center);
                A.addEntry(k, idx(i - 1, j), -rx);
                A.addEntry(k, idx(i + 1, j), -rx);
                A.addEntry(k, idx(i, j - 1), -ry);
                A.addEntry(k, idx(i, j + 1), -ry);
            }
        }
    }

    A.finalize();
    return A;
}

/**
 * @brief Build implicit diffusion matrix for Backward Euler.
 *
 * Creates: (I - α*dt*∇²) for the implicit update:
 *   (I - α*dt*∇²) u^{n+1} = u^n
 *
 * @param nx Number of cells in x
 * @param ny Number of cells in y
 * @param dx Grid spacing in x
 * @param dy Grid spacing in y
 * @param alpha Diffusion coefficient
 * @param dt Time step
 * @return Implicit diffusion matrix
 */
inline SparseMatrix buildImplicitDiffusion2D(int nx, int ny, double dx, double dy, double alpha,
                                             double dt) {
    const int nodes_x = nx + 1;
    const int nodes_y = ny + 1;
    const int n = nodes_x * nodes_y;

    SparseMatrix A(n, n);
    A.reserve(5 * n);

    const double rx = alpha * dt / (dx * dx);
    const double ry = alpha * dt / (dy * dy);

    auto idx = [nodes_x](int i, int j) {
        return j * nodes_x + i;
    };

    for (int j = 0; j < nodes_y; ++j) {
        for (int i = 0; i < nodes_x; ++i) {
            int k = idx(i, j);

            // Boundary: identity row (Dirichlet)
            if (i == 0 || i == nx || j == 0 || j == ny) {
                A.addEntry(k, k, 1.0);
            } else {
                // Interior: I - dt*α*∇²
                A.addEntry(k, k, 1.0 + 2.0 * rx + 2.0 * ry);
                A.addEntry(k, idx(i - 1, j), -rx);
                A.addEntry(k, idx(i + 1, j), -rx);
                A.addEntry(k, idx(i, j - 1), -ry);
                A.addEntry(k, idx(i, j + 1), -ry);
            }
        }
    }

    A.finalize();
    return A;
}

/**
 * @brief Build 3D implicit diffusion matrix.
 */
inline SparseMatrix buildImplicitDiffusion3D(int nx, int ny, int nz, double dx, double dy,
                                             double dz, double alpha, double dt) {
    const int nodes_x = nx + 1;
    const int nodes_y = ny + 1;
    const int nodes_z = nz + 1;
    const int n = nodes_x * nodes_y * nodes_z;

    SparseMatrix A(n, n);
    A.reserve(7 * n);

    const double rx = alpha * dt / (dx * dx);
    const double ry = alpha * dt / (dy * dy);
    const double rz = alpha * dt / (dz * dz);

    auto idx = [nodes_x, nodes_y](int i, int j, int k) {
        return k * nodes_x * nodes_y + j * nodes_x + i;
    };

    for (int k = 0; k < nodes_z; ++k) {
        for (int j = 0; j < nodes_y; ++j) {
            for (int i = 0; i < nodes_x; ++i) {
                int m = idx(i, j, k);

                // Boundary: identity row
                if (i == 0 || i == nx || j == 0 || j == ny || k == 0 || k == nz) {
                    A.addEntry(m, m, 1.0);
                } else {
                    // Interior: 7-point stencil
                    A.addEntry(m, m, 1.0 + 2.0 * rx + 2.0 * ry + 2.0 * rz);
                    A.addEntry(m, idx(i - 1, j, k), -rx);
                    A.addEntry(m, idx(i + 1, j, k), -rx);
                    A.addEntry(m, idx(i, j - 1, k), -ry);
                    A.addEntry(m, idx(i, j + 1, k), -ry);
                    A.addEntry(m, idx(i, j, k - 1), -rz);
                    A.addEntry(m, idx(i, j, k + 1), -rz);
                }
            }
        }
    }

    A.finalize();
    return A;
}

}  // namespace linalg
}  // namespace biotransport

#else  // !BIOTRANSPORT_ENABLE_EIGEN

// Stub when Eigen is not available
namespace biotransport {
namespace linalg {

enum class SparseSolverType {
    SparseLU,
    SimplicialLLT,
    SimplicialLDLT,
    ConjugateGradient,
    BiCGSTAB
};

struct SparseSolveResult {
    bool success = false;
    int iterations = 0;
    double residual = 0.0;
    double factorization_time_ms = 0.0;
    double solve_time_ms = 0.0;
    std::string error_message = "Eigen not enabled";
};

struct Triplet {
    int row, col;
    double value;
    Triplet(int r, int c, double v) : row(r), col(c), value(v) {}
};

class SparseMatrix {
public:
    SparseMatrix() = default;
    SparseMatrix(int, int) {
        throw std::runtime_error(
            "Sparse matrix requires Eigen. Rebuild with BIOTRANSPORT_EIGEN=ON");
    }
};

}  // namespace linalg
}  // namespace biotransport

#endif  // BIOTRANSPORT_ENABLE_EIGEN

#endif  // BIOTRANSPORT_CORE_NUMERICS_LINEAR_ALGEBRA_SPARSE_MATRIX_HPP
