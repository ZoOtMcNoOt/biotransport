/**
 * @file implicit_diffusion.hpp
 * @brief Conservative Backward Euler diffusion solvers using sparse matrices.
 *
 * The assembled equation is the control-volume balance
 *
 *   V_i (u_i^{n+1} - u_i^n)
 *       = dt [sum_j G_ij (u_j^{n+1} - u_i^{n+1}) + V_i f_i^{n+1}],
 *
 * with natural outward-derivative Neumann data added as boundary fluxes.
 * Harmonic face diffusivities make variable-coefficient diffusion conservative
 * and keep the free-node matrix symmetric positive definite.  Dirichlet values
 * are eliminated from neighboring rows rather than left as asymmetric columns.
 * Traces meeting at a corner or edge must agree within
 * 64*epsilon*max(1, |a|, |b|); contradictory Dirichlet data are rejected.
 */

#ifndef BIOTRANSPORT_SOLVERS_IMPLICIT_DIFFUSION_HPP
#define BIOTRANSPORT_SOLVERS_IMPLICIT_DIFFUSION_HPP

#ifdef BIOTRANSPORT_ENABLE_EIGEN

#include <algorithm>
#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/mesh/structured_mesh_3d.hpp>
#include <biotransport/core/numerics/linear_algebra/sparse_matrix.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace biotransport {

struct ImplicitSolveResult {
    int steps = 0;
    double total_time = 0.0;
    double residual = 0.0;  ///< Relative infinity norm of A u - b.
    bool success = true;
};

namespace implicit_diffusion_detail {

inline void requirePositiveFinite(double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
}

inline void requireFinite(double value, const char* name) {
    if (!std::isfinite(value))
        throw std::invalid_argument(std::string(name) + " must be finite");
}

inline void requirePositiveField(const std::vector<double>& values, const char* name) {
    for (double value : values)
        requirePositiveFinite(value, name);
}

inline void requireFiniteField(const std::vector<double>& values, const char* name) {
    for (double value : values)
        requireFinite(value, name);
}

inline void requireFiniteResult(const std::vector<double>& values, const char* name) {
    for (double value : values) {
        if (!std::isfinite(value))
            throw std::runtime_error(std::string(name) + " contains a non-finite value");
    }
}

inline double harmonic(double lhs, double rhs) {
    return 2.0 * lhs * rhs / (lhs + rhs);
}

inline double relativeInfinityResidual(const linalg::SparseMatrix& matrix,
                                       const std::vector<double>& solution,
                                       const std::vector<double>& rhs) {
    const auto product = matrix.multiply(solution);
    double numerator = 0.0;
    double denominator = 1.0;
    for (std::size_t index = 0; index < rhs.size(); ++index) {
        numerator = std::max(numerator, std::abs(product[index] - rhs[index]));
        denominator = std::max(denominator, std::abs(rhs[index]));
    }
    return numerator / denominator;
}

}  // namespace implicit_diffusion_detail

class ImplicitDiffusion2D {
public:
    ImplicitDiffusion2D(const StructuredMesh& mesh, double diffusivity)
        : mesh_(mesh), diffusivity_(mesh.numNodes(), diffusivity), solution_(mesh.numNodes(), 0.0) {
        if (mesh.is1D())
            throw std::invalid_argument("ImplicitDiffusion2D requires a 2D mesh");
        implicit_diffusion_detail::requirePositiveFinite(diffusivity, "Diffusivity");
        boundary_conditions_.fill(BoundaryCondition::Dirichlet(0.0));
    }

    ImplicitDiffusion2D(const StructuredMesh& mesh, const std::vector<double>& diffusivity)
        : mesh_(mesh), diffusivity_(diffusivity), solution_(mesh.numNodes(), 0.0) {
        if (mesh.is1D())
            throw std::invalid_argument("ImplicitDiffusion2D requires a 2D mesh");
        if (diffusivity.size() != static_cast<std::size_t>(mesh.numNodes()))
            throw std::invalid_argument("Diffusivity field size must match mesh nodes");
        implicit_diffusion_detail::requirePositiveField(diffusivity_, "Diffusivity field");
        boundary_conditions_.fill(BoundaryCondition::Dirichlet(0.0));
    }

    void setInitialCondition(const std::vector<double>& values) {
        if (values.size() != solution_.size())
            throw std::invalid_argument("Initial condition size must match mesh nodes");
        implicit_diffusion_detail::requireFiniteField(values, "Initial condition");
        solution_ = values;
    }

    void setDirichletBoundary(Boundary boundary, double value) {
        implicit_diffusion_detail::requireFinite(value, "Dirichlet value");
        boundary_conditions_[checkedIndex(boundary)] = BoundaryCondition::Dirichlet(value);
    }

    void setNeumannBoundary(Boundary boundary, double normal_derivative) {
        implicit_diffusion_detail::requireFinite(normal_derivative,
                                                 "Neumann outward-normal derivative");
        boundary_conditions_[checkedIndex(boundary)] =
            BoundaryCondition::Neumann(normal_derivative);
    }

    void setSourceTerm(std::function<double(double, double, double)> source) {
        if (!source)
            throw std::invalid_argument("Source callback must be callable");
        source_term_ = std::move(source);
    }

    void clearSourceTerm() { source_term_ = {}; }

    void setSolverType(linalg::SparseSolverType type) { solver_type_ = type; }

    void setTolerance(double tolerance) {
        implicit_diffusion_detail::requirePositiveFinite(tolerance, "Linear-solver tolerance");
        tolerance_ = tolerance;
    }

    void setMaxIterations(int max_iterations) {
        if (max_iterations <= 0)
            throw std::invalid_argument("Maximum iterations must be positive");
        max_iterations_ = max_iterations;
    }

    ImplicitSolveResult step(double dt) {
        implicit_diffusion_detail::requirePositiveFinite(dt, "Time step");
        const int total_nodes = mesh_.numNodes();
        linalg::SparseMatrix matrix(total_nodes, total_nodes);
        matrix.reserve(5 * total_nodes);
        std::vector<double> rhs(static_cast<std::size_t>(total_nodes), 0.0);

        for (int j = 0; j <= mesh_.ny(); ++j) {
            for (int i = 0; i <= mesh_.nx(); ++i) {
                const int row = mesh_.index(i, j);
                const std::size_t index = static_cast<std::size_t>(row);
                if (const auto fixed = dirichletValue(i, j)) {
                    matrix.addEntry(row, row, 1.0);
                    rhs[index] = *fixed;
                    continue;
                }

                const double volume = xWidth(i) * yWidth(j);
                double diagonal = volume;
                rhs[index] = volume * solution_[index];
                if (source_term_) {
                    const double source = source_term_(mesh_.x(i), mesh_.y(i, j), time_ + dt);
                    implicit_diffusion_detail::requireFinite(source, "Source callback result");
                    rhs[index] += dt * volume * source;
                }
                rhs[index] += dt * boundaryFlux(i, j);

                const auto add_neighbor = [&](int ni, int nj, double area, double distance) {
                    const int column = mesh_.index(ni, nj);
                    const std::size_t neighbor = static_cast<std::size_t>(column);
                    const double face_diffusivity = implicit_diffusion_detail::harmonic(
                        diffusivity_[index], diffusivity_[neighbor]);
                    const double conductance = face_diffusivity * area / distance;
                    diagonal += dt * conductance;
                    if (const auto fixed = dirichletValue(ni, nj))
                        rhs[index] += dt * conductance * *fixed;
                    else
                        matrix.addEntry(row, column, -dt * conductance);
                };

                if (i > 0)
                    add_neighbor(i - 1, j, yWidth(j), mesh_.dx());
                if (i < mesh_.nx())
                    add_neighbor(i + 1, j, yWidth(j), mesh_.dx());
                if (j > 0)
                    add_neighbor(i, j - 1, xWidth(i), mesh_.dy());
                if (j < mesh_.ny())
                    add_neighbor(i, j + 1, xWidth(i), mesh_.dy());
                matrix.addEntry(row, row, diagonal);
            }
        }

        matrix.finalize();
        std::vector<double> candidate =
            matrix.solve(rhs, solver_type_, tolerance_, max_iterations_);
        implicit_diffusion_detail::requireFiniteResult(candidate, "Implicit solution");
        const double residual =
            implicit_diffusion_detail::relativeInfinityResidual(matrix, candidate, rhs);
        if (!std::isfinite(residual))
            throw std::runtime_error("Implicit solve produced a non-finite residual");

        solution_.swap(candidate);
        time_ += dt;
        ImplicitSolveResult result;
        result.steps = 1;
        result.total_time = time_;
        result.residual = residual;
        return result;
    }

    ImplicitSolveResult solve(double dt, int num_steps) {
        implicit_diffusion_detail::requirePositiveFinite(dt, "Time step");
        if (num_steps < 0)
            throw std::invalid_argument("Number of steps must be non-negative");
        ImplicitSolveResult total;
        total.steps = 0;
        for (int count = 0; count < num_steps; ++count) {
            const auto result = step(dt);
            ++total.steps;
            total.residual = result.residual;
        }
        total.total_time = time_;
        return total;
    }

    const std::vector<double>& solution() const { return solution_; }
    const std::vector<double>& diffusivity() const { return diffusivity_; }
    double time() const { return time_; }
    const StructuredMesh& mesh() const { return mesh_; }

private:
    const StructuredMesh& mesh_;
    std::vector<double> diffusivity_;
    std::vector<double> solution_;
    double time_ = 0.0;
    std::array<BoundaryCondition, 4> boundary_conditions_;
    std::function<double(double, double, double)> source_term_;
    linalg::SparseSolverType solver_type_ = linalg::SparseSolverType::SparseLU;
    double tolerance_ = 1e-10;
    int max_iterations_ = 1000;

    static std::size_t checkedIndex(Boundary boundary) {
        const int index = to_index(boundary);
        if (index < 0 || index >= 4)
            throw std::invalid_argument("Boundary identifier is outside [0, 3]");
        return static_cast<std::size_t>(index);
    }

    std::optional<double> dirichletValue(int i, int j) const {
        double sum = 0.0;
        int count = 0;
        double reference = 0.0;
        const auto add = [&](Boundary face) {
            const auto& bc = boundary_conditions_[checkedIndex(face)];
            if (bc.type == BoundaryType::DIRICHLET) {
                if (count == 0) {
                    reference = bc.value;
                } else {
                    const double scale = std::max({1.0, std::abs(reference), std::abs(bc.value)});
                    if (std::abs(reference - bc.value) >
                        64.0 * std::numeric_limits<double>::epsilon() * scale) {
                        throw std::invalid_argument(
                            "Conflicting Dirichlet values meet at a two-dimensional corner");
                    }
                }
                sum += bc.value;
                ++count;
            }
        };
        if (i == 0)
            add(Boundary::Left);
        if (i == mesh_.nx())
            add(Boundary::Right);
        if (j == 0)
            add(Boundary::Bottom);
        if (j == mesh_.ny())
            add(Boundary::Top);
        if (count == 0)
            return std::nullopt;
        return sum / static_cast<double>(count);
    }

    double xWidth(int i) const { return mesh_.dx() * ((i == 0 || i == mesh_.nx()) ? 0.5 : 1.0); }
    double yWidth(int j) const { return mesh_.dy() * ((j == 0 || j == mesh_.ny()) ? 0.5 : 1.0); }

    double boundaryFlux(int i, int j) const {
        const std::size_t index = static_cast<std::size_t>(mesh_.index(i, j));
        double flux = 0.0;
        const auto add = [&](Boundary face, double area) {
            const auto& bc = boundary_conditions_[checkedIndex(face)];
            if (bc.type == BoundaryType::NEUMANN)
                flux += diffusivity_[index] * bc.value * area;
        };
        if (i == 0)
            add(Boundary::Left, yWidth(j));
        if (i == mesh_.nx())
            add(Boundary::Right, yWidth(j));
        if (j == 0)
            add(Boundary::Bottom, xWidth(i));
        if (j == mesh_.ny())
            add(Boundary::Top, xWidth(i));
        return flux;
    }
};

class ImplicitDiffusion3D {
public:
    ImplicitDiffusion3D(const StructuredMesh3D& mesh, double diffusivity)
        : mesh_(mesh), diffusivity_(mesh.numNodes(), diffusivity), solution_(mesh.numNodes(), 0.0) {
        implicit_diffusion_detail::requirePositiveFinite(diffusivity, "Diffusivity");
        boundary_conditions_.fill(BoundaryCondition::Dirichlet(0.0));
    }

    ImplicitDiffusion3D(const StructuredMesh3D& mesh, const std::vector<double>& diffusivity)
        : mesh_(mesh), diffusivity_(diffusivity), solution_(mesh.numNodes(), 0.0) {
        if (diffusivity.size() != static_cast<std::size_t>(mesh.numNodes()))
            throw std::invalid_argument("Diffusivity field size must match mesh nodes");
        implicit_diffusion_detail::requirePositiveField(diffusivity_, "Diffusivity field");
        boundary_conditions_.fill(BoundaryCondition::Dirichlet(0.0));
    }

    void setInitialCondition(const std::vector<double>& values) {
        if (values.size() != solution_.size())
            throw std::invalid_argument("Initial condition size must match mesh nodes");
        implicit_diffusion_detail::requireFiniteField(values, "Initial condition");
        solution_ = values;
    }

    void setDirichletBoundary(Boundary3D boundary, double value) {
        implicit_diffusion_detail::requireFinite(value, "Dirichlet value");
        boundary_conditions_[checkedIndex(boundary)] = BoundaryCondition::Dirichlet(value);
    }

    void setNeumannBoundary(Boundary3D boundary, double normal_derivative) {
        implicit_diffusion_detail::requireFinite(normal_derivative,
                                                 "Neumann outward-normal derivative");
        boundary_conditions_[checkedIndex(boundary)] =
            BoundaryCondition::Neumann(normal_derivative);
    }

    void setSourceTerm(std::function<double(double, double, double, double)> source) {
        if (!source)
            throw std::invalid_argument("Source callback must be callable");
        source_term_ = std::move(source);
    }

    void clearSourceTerm() { source_term_ = {}; }

    void setSolverType(linalg::SparseSolverType type) { solver_type_ = type; }
    void setTolerance(double tolerance) {
        implicit_diffusion_detail::requirePositiveFinite(tolerance, "Linear-solver tolerance");
        tolerance_ = tolerance;
    }
    void setMaxIterations(int max_iterations) {
        if (max_iterations <= 0)
            throw std::invalid_argument("Maximum iterations must be positive");
        max_iterations_ = max_iterations;
    }

    ImplicitSolveResult step(double dt) {
        implicit_diffusion_detail::requirePositiveFinite(dt, "Time step");
        const int total_nodes = mesh_.numNodes();
        linalg::SparseMatrix matrix(total_nodes, total_nodes);
        matrix.reserve(7 * total_nodes);
        std::vector<double> rhs(static_cast<std::size_t>(total_nodes), 0.0);

        for (int k = 0; k <= mesh_.nz(); ++k) {
            for (int j = 0; j <= mesh_.ny(); ++j) {
                for (int i = 0; i <= mesh_.nx(); ++i) {
                    const int row = mesh_.index(i, j, k);
                    const std::size_t index = static_cast<std::size_t>(row);
                    if (const auto fixed = dirichletValue(i, j, k)) {
                        matrix.addEntry(row, row, 1.0);
                        rhs[index] = *fixed;
                        continue;
                    }

                    const double volume = xWidth(i) * yWidth(j) * zWidth(k);
                    double diagonal = volume;
                    rhs[index] = volume * solution_[index];
                    if (source_term_) {
                        const double source =
                            source_term_(mesh_.x(i), mesh_.y(j), mesh_.z(k), time_ + dt);
                        implicit_diffusion_detail::requireFinite(source, "Source callback result");
                        rhs[index] += dt * volume * source;
                    }
                    rhs[index] += dt * boundaryFlux(i, j, k);

                    const auto add_neighbor = [&](int ni, int nj, int nk, double area,
                                                  double distance) {
                        const int column = mesh_.index(ni, nj, nk);
                        const std::size_t neighbor = static_cast<std::size_t>(column);
                        const double face_diffusivity = implicit_diffusion_detail::harmonic(
                            diffusivity_[index], diffusivity_[neighbor]);
                        const double conductance = face_diffusivity * area / distance;
                        diagonal += dt * conductance;
                        if (const auto fixed = dirichletValue(ni, nj, nk))
                            rhs[index] += dt * conductance * *fixed;
                        else
                            matrix.addEntry(row, column, -dt * conductance);
                    };

                    if (i > 0)
                        add_neighbor(i - 1, j, k, yWidth(j) * zWidth(k), mesh_.dx());
                    if (i < mesh_.nx())
                        add_neighbor(i + 1, j, k, yWidth(j) * zWidth(k), mesh_.dx());
                    if (j > 0)
                        add_neighbor(i, j - 1, k, xWidth(i) * zWidth(k), mesh_.dy());
                    if (j < mesh_.ny())
                        add_neighbor(i, j + 1, k, xWidth(i) * zWidth(k), mesh_.dy());
                    if (k > 0)
                        add_neighbor(i, j, k - 1, xWidth(i) * yWidth(j), mesh_.dz());
                    if (k < mesh_.nz())
                        add_neighbor(i, j, k + 1, xWidth(i) * yWidth(j), mesh_.dz());
                    matrix.addEntry(row, row, diagonal);
                }
            }
        }

        matrix.finalize();
        std::vector<double> candidate =
            matrix.solve(rhs, solver_type_, tolerance_, max_iterations_);
        implicit_diffusion_detail::requireFiniteResult(candidate, "Implicit solution");
        const double residual =
            implicit_diffusion_detail::relativeInfinityResidual(matrix, candidate, rhs);
        if (!std::isfinite(residual))
            throw std::runtime_error("Implicit solve produced a non-finite residual");

        solution_.swap(candidate);
        time_ += dt;
        ImplicitSolveResult result;
        result.steps = 1;
        result.total_time = time_;
        result.residual = residual;
        return result;
    }

    ImplicitSolveResult solve(double dt, int num_steps) {
        implicit_diffusion_detail::requirePositiveFinite(dt, "Time step");
        if (num_steps < 0)
            throw std::invalid_argument("Number of steps must be non-negative");
        ImplicitSolveResult total;
        total.steps = 0;
        for (int count = 0; count < num_steps; ++count) {
            const auto result = step(dt);
            ++total.steps;
            total.residual = result.residual;
        }
        total.total_time = time_;
        return total;
    }

    const std::vector<double>& solution() const { return solution_; }
    const std::vector<double>& diffusivity() const { return diffusivity_; }
    double time() const { return time_; }
    const StructuredMesh3D& mesh() const { return mesh_; }

private:
    const StructuredMesh3D& mesh_;
    std::vector<double> diffusivity_;
    std::vector<double> solution_;
    double time_ = 0.0;
    std::array<BoundaryCondition, 6> boundary_conditions_;
    std::function<double(double, double, double, double)> source_term_;
    linalg::SparseSolverType solver_type_ = linalg::SparseSolverType::BiCGSTAB;
    double tolerance_ = 1e-10;
    int max_iterations_ = 1000;

    static std::size_t checkedIndex(Boundary3D boundary) {
        const int index = to_index(boundary);
        if (index < 0 || index >= 6)
            throw std::invalid_argument("3D boundary identifier is outside [0, 5]");
        return static_cast<std::size_t>(index);
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

    double xWidth(int i) const { return mesh_.dx() * ((i == 0 || i == mesh_.nx()) ? 0.5 : 1.0); }
    double yWidth(int j) const { return mesh_.dy() * ((j == 0 || j == mesh_.ny()) ? 0.5 : 1.0); }
    double zWidth(int k) const { return mesh_.dz() * ((k == 0 || k == mesh_.nz()) ? 0.5 : 1.0); }

    double boundaryFlux(int i, int j, int k) const {
        const std::size_t index = static_cast<std::size_t>(mesh_.index(i, j, k));
        const double wx = xWidth(i);
        const double wy = yWidth(j);
        const double wz = zWidth(k);
        double flux = 0.0;
        const auto add = [&](Boundary3D face, double area) {
            const auto& bc = boundary_conditions_[checkedIndex(face)];
            if (bc.type == BoundaryType::NEUMANN)
                flux += diffusivity_[index] * bc.value * area;
        };
        if (i == 0)
            add(Boundary3D::XMin, wy * wz);
        if (i == mesh_.nx())
            add(Boundary3D::XMax, wy * wz);
        if (j == 0)
            add(Boundary3D::YMin, wx * wz);
        if (j == mesh_.ny())
            add(Boundary3D::YMax, wx * wz);
        if (k == 0)
            add(Boundary3D::ZMin, wx * wy);
        if (k == mesh_.nz())
            add(Boundary3D::ZMax, wx * wy);
        return flux;
    }
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
