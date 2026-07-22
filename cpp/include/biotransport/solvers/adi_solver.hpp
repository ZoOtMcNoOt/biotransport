/**
 * @file adi_solver.hpp
 * @brief Directionally split Crank--Nicolson solvers for Cartesian diffusion.
 *
 * These solvers integrate
 *
 *     du/dt = D laplacian(u)
 *
 * on uniform node-centred Cartesian meshes.  Each directional subproblem is a
 * one-dimensional Crank--Nicolson solve, so a substep is unconditionally stable
 * for the linear diffusion operator.  Symmetric (Strang) composition is used:
 * x/2-y-x/2 in two dimensions and x/2-y/2-z-y/2-x/2 in three dimensions.
 * This is second order in time and space for smooth solutions with time-
 * independent boundary data.
 *
 * Neumann data always mean the outward-normal derivative du/dn, not a Fickian
 * flux.  Boundary nodes use half control volumes, which preserves the
 * trapezoidal-volume integral exactly for homogeneous Neumann data (up to
 * roundoff).  Dirichlet traces that meet at one node must agree to within
 * 64*epsilon*max(1, |a|, |b|).  Larger discrepancies are contradictory data
 * and raise std::invalid_argument before the public solution is advanced.
 */

#ifndef BIOTRANSPORT_SOLVERS_ADI_SOLVER_HPP
#define BIOTRANSPORT_SOLVERS_ADI_SOLVER_HPP

#include <algorithm>
#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/mesh/structured_mesh_3d.hpp>
#include <biotransport/core/numerics/linear_algebra/tridiagonal.hpp>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace biotransport {

struct ADISolveResult {
    int steps = 0;            ///< Number of complete time steps.
    int substeps = 0;         ///< Directional solves (3 in 2D, 5 in 3D per step).
    double time = 0.0;        ///< Current time after step().
    double total_time = 0.0;  ///< Current time after solve().
    bool success = true;
};

/**
 * @brief Symmetric alternating-direction solver for constant-D 2D diffusion.
 */
class ADIDiffusion2D {
public:
    ADIDiffusion2D(const StructuredMesh& mesh, double diffusivity)
        : mesh_(mesh), diffusivity_(diffusivity), solution_(mesh.numNodes(), 0.0) {
        if (mesh.is1D()) {
            throw std::invalid_argument("ADIDiffusion2D requires a 2D mesh");
        }
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

    void setDirichletBoundary(Boundary boundary, double value) {
        requireFinite(value, "Dirichlet value");
        boundary_conditions_[checkedIndex(boundary)] = BoundaryCondition::Dirichlet(value);
    }

    void setNeumannBoundary(Boundary boundary, double normal_derivative) {
        requireFinite(normal_derivative, "Neumann outward-normal derivative");
        boundary_conditions_[checkedIndex(boundary)] =
            BoundaryCondition::Neumann(normal_derivative);
    }

    ADISolveResult step(double dt) {
        requirePositiveFinite(dt, "Time step");

        // Work on temporaries so a failed tridiagonal solve cannot leave a
        // partially advanced public solution.
        std::vector<double> work = solution_;
        imposeDirichlet(work);
        work = sweepX(work, 0.5 * dt);
        work = sweepY(work, dt);
        work = sweepX(work, 0.5 * dt);
        imposeDirichlet(work);
        requireFiniteResult(work, "ADI solution");

        solution_.swap(work);
        time_ += dt;

        ADISolveResult result;
        result.steps = 1;
        result.substeps = 3;
        result.time = time_;
        return result;
    }

    ADISolveResult solve(double dt, int num_steps) {
        requirePositiveFinite(dt, "Time step");
        if (num_steps < 0) {
            throw std::invalid_argument("Number of steps must be non-negative");
        }

        ADISolveResult result;
        for (int count = 0; count < num_steps; ++count) {
            const auto one_step = step(dt);
            ++result.steps;
            result.substeps += one_step.substeps;
        }
        result.time = time_;
        result.total_time = time_;
        return result;
    }

    const std::vector<double>& solution() const { return solution_; }
    const StructuredMesh& mesh() const { return mesh_; }
    double diffusivity() const { return diffusivity_; }
    double time() const { return time_; }

private:
    const StructuredMesh& mesh_;
    double diffusivity_;
    std::vector<double> solution_;
    std::array<BoundaryCondition, 4> boundary_conditions_;
    double time_ = 0.0;

    static void requirePositiveFinite(double value, const char* name) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(std::string(name) + " must be finite and positive");
        }
    }

    static void requireFinite(double value, const char* name) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(name) + " must be finite");
        }
    }

    static void requireFiniteInput(const std::vector<double>& values, const char* name) {
        for (double value : values) {
            requireFinite(value, name);
        }
    }

    static void requireFiniteResult(const std::vector<double>& values, const char* name) {
        for (double value : values) {
            if (!std::isfinite(value))
                throw std::runtime_error(std::string(name) + " contains a non-finite value");
        }
    }

    static std::size_t checkedIndex(Boundary boundary) {
        const int index = to_index(boundary);
        if (index < 0 || index >= 4) {
            throw std::invalid_argument("Boundary identifier is outside [0, 3]");
        }
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

    void imposeDirichlet(std::vector<double>& values) const {
        for (int j = 0; j <= mesh_.ny(); ++j) {
            for (int i = 0; i <= mesh_.nx(); ++i) {
                if (const auto fixed = dirichletValue(i, j)) {
                    values[static_cast<std::size_t>(mesh_.index(i, j))] = *fixed;
                }
            }
        }
    }

    std::vector<double> sweepX(const std::vector<double>& input, double duration) const {
        const int nodes = mesh_.nx() + 1;
        const double h = mesh_.dx();
        const double r = diffusivity_ * duration / (2.0 * h * h);
        std::vector<double> output(input);
        std::vector<double> a(nodes), b(nodes), c(nodes), d(nodes);

        for (int j = 0; j <= mesh_.ny(); ++j) {
            std::fill(a.begin(), a.end(), 0.0);
            std::fill(b.begin(), b.end(), 0.0);
            std::fill(c.begin(), c.end(), 0.0);
            std::fill(d.begin(), d.end(), 0.0);

            for (int i = 0; i <= mesh_.nx(); ++i) {
                const int index = mesh_.index(i, j);
                if (const auto fixed = dirichletValue(i, j)) {
                    b[i] = 1.0;
                    d[i] = *fixed;
                } else if (i == 0) {
                    const auto& bc = boundary_conditions_[checkedIndex(Boundary::Left)];
                    b[i] = 1.0 + 2.0 * r;
                    c[i] = -2.0 * r;
                    d[i] = input[index] + 2.0 * r * (input[index + 1] - input[index]) +
                           4.0 * r * bc.value * h;
                } else if (i == mesh_.nx()) {
                    const auto& bc = boundary_conditions_[checkedIndex(Boundary::Right)];
                    a[i] = -2.0 * r;
                    b[i] = 1.0 + 2.0 * r;
                    d[i] = input[index] + 2.0 * r * (input[index - 1] - input[index]) +
                           4.0 * r * bc.value * h;
                } else {
                    a[i] = -r;
                    b[i] = 1.0 + 2.0 * r;
                    c[i] = -r;
                    d[i] = input[index] +
                           r * (input[index - 1] - 2.0 * input[index] + input[index + 1]);
                }
            }

            const auto line = linalg::solve_tridiagonal(a, b, c, d);
            for (int i = 0; i <= mesh_.nx(); ++i) {
                output[static_cast<std::size_t>(mesh_.index(i, j))] = line[i];
            }
        }
        return output;
    }

    std::vector<double> sweepY(const std::vector<double>& input, double duration) const {
        const int nodes = mesh_.ny() + 1;
        const int stride = mesh_.nx() + 1;
        const double h = mesh_.dy();
        const double r = diffusivity_ * duration / (2.0 * h * h);
        std::vector<double> output(input);
        std::vector<double> a(nodes), b(nodes), c(nodes), d(nodes);

        for (int i = 0; i <= mesh_.nx(); ++i) {
            std::fill(a.begin(), a.end(), 0.0);
            std::fill(b.begin(), b.end(), 0.0);
            std::fill(c.begin(), c.end(), 0.0);
            std::fill(d.begin(), d.end(), 0.0);

            for (int j = 0; j <= mesh_.ny(); ++j) {
                const int index = mesh_.index(i, j);
                if (const auto fixed = dirichletValue(i, j)) {
                    b[j] = 1.0;
                    d[j] = *fixed;
                } else if (j == 0) {
                    const auto& bc = boundary_conditions_[checkedIndex(Boundary::Bottom)];
                    b[j] = 1.0 + 2.0 * r;
                    c[j] = -2.0 * r;
                    d[j] = input[index] + 2.0 * r * (input[index + stride] - input[index]) +
                           4.0 * r * bc.value * h;
                } else if (j == mesh_.ny()) {
                    const auto& bc = boundary_conditions_[checkedIndex(Boundary::Top)];
                    a[j] = -2.0 * r;
                    b[j] = 1.0 + 2.0 * r;
                    d[j] = input[index] + 2.0 * r * (input[index - stride] - input[index]) +
                           4.0 * r * bc.value * h;
                } else {
                    a[j] = -r;
                    b[j] = 1.0 + 2.0 * r;
                    c[j] = -r;
                    d[j] = input[index] +
                           r * (input[index - stride] - 2.0 * input[index] + input[index + stride]);
                }
            }

            const auto line = linalg::solve_tridiagonal(a, b, c, d);
            for (int j = 0; j <= mesh_.ny(); ++j) {
                output[static_cast<std::size_t>(mesh_.index(i, j))] = line[j];
            }
        }
        return output;
    }
};

/**
 * @brief Symmetric alternating-direction solver for constant-D 3D diffusion.
 */
class ADIDiffusion3D {
public:
    ADIDiffusion3D(const StructuredMesh3D& mesh, double diffusivity)
        : mesh_(mesh), diffusivity_(diffusivity), solution_(mesh.numNodes(), 0.0) {
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

    ADISolveResult step(double dt) {
        requirePositiveFinite(dt, "Time step");

        std::vector<double> work = solution_;
        imposeDirichlet(work);
        work = sweepX(work, 0.5 * dt);
        work = sweepY(work, 0.5 * dt);
        work = sweepZ(work, dt);
        work = sweepY(work, 0.5 * dt);
        work = sweepX(work, 0.5 * dt);
        imposeDirichlet(work);
        requireFiniteResult(work, "ADI solution");

        solution_.swap(work);
        time_ += dt;

        ADISolveResult result;
        result.steps = 1;
        result.substeps = 5;
        result.time = time_;
        return result;
    }

    ADISolveResult solve(double dt, int num_steps) {
        requirePositiveFinite(dt, "Time step");
        if (num_steps < 0) {
            throw std::invalid_argument("Number of steps must be non-negative");
        }

        ADISolveResult result;
        for (int count = 0; count < num_steps; ++count) {
            const auto one_step = step(dt);
            ++result.steps;
            result.substeps += one_step.substeps;
        }
        result.time = time_;
        result.total_time = time_;
        return result;
    }

    const std::vector<double>& solution() const { return solution_; }
    const StructuredMesh3D& mesh() const { return mesh_; }
    double diffusivity() const { return diffusivity_; }
    double time() const { return time_; }

private:
    const StructuredMesh3D& mesh_;
    double diffusivity_;
    std::vector<double> solution_;
    std::array<BoundaryCondition, 6> boundary_conditions_;
    double time_ = 0.0;

    static void requirePositiveFinite(double value, const char* name) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(std::string(name) + " must be finite and positive");
        }
    }

    static void requireFinite(double value, const char* name) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(name) + " must be finite");
        }
    }

    static void requireFiniteInput(const std::vector<double>& values, const char* name) {
        for (double value : values) {
            requireFinite(value, name);
        }
    }

    static void requireFiniteResult(const std::vector<double>& values, const char* name) {
        for (double value : values) {
            if (!std::isfinite(value))
                throw std::runtime_error(std::string(name) + " contains a non-finite value");
        }
    }

    static Boundary3D checkedBoundary(int boundary_id) {
        if (boundary_id < 0 || boundary_id >= 6) {
            throw std::invalid_argument("3D boundary identifier is outside [0, 5]");
        }
        return static_cast<Boundary3D>(boundary_id);
    }

    static std::size_t checkedIndex(Boundary3D boundary) {
        const int index = to_index(boundary);
        if (index < 0 || index >= 6) {
            throw std::invalid_argument("3D boundary identifier is outside [0, 5]");
        }
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

    void imposeDirichlet(std::vector<double>& values) const {
        for (int k = 0; k <= mesh_.nz(); ++k) {
            for (int j = 0; j <= mesh_.ny(); ++j) {
                for (int i = 0; i <= mesh_.nx(); ++i) {
                    if (const auto fixed = dirichletValue(i, j, k)) {
                        values[static_cast<std::size_t>(mesh_.index(i, j, k))] = *fixed;
                    }
                }
            }
        }
    }

    std::vector<double> sweepX(const std::vector<double>& input, double duration) const {
        const int nodes = mesh_.nx() + 1;
        const double h = mesh_.dx();
        const double r = diffusivity_ * duration / (2.0 * h * h);
        std::vector<double> output(input);
        std::vector<double> a(nodes), b(nodes), c(nodes), d(nodes);

        for (int k = 0; k <= mesh_.nz(); ++k) {
            for (int j = 0; j <= mesh_.ny(); ++j) {
                std::fill(a.begin(), a.end(), 0.0);
                std::fill(b.begin(), b.end(), 0.0);
                std::fill(c.begin(), c.end(), 0.0);
                std::fill(d.begin(), d.end(), 0.0);
                for (int i = 0; i <= mesh_.nx(); ++i) {
                    const int index = mesh_.index(i, j, k);
                    if (const auto fixed = dirichletValue(i, j, k)) {
                        b[i] = 1.0;
                        d[i] = *fixed;
                    } else if (i == 0) {
                        const auto& bc = boundary_conditions_[checkedIndex(Boundary3D::XMin)];
                        b[i] = 1.0 + 2.0 * r;
                        c[i] = -2.0 * r;
                        d[i] = input[index] + 2.0 * r * (input[index + 1] - input[index]) +
                               4.0 * r * bc.value * h;
                    } else if (i == mesh_.nx()) {
                        const auto& bc = boundary_conditions_[checkedIndex(Boundary3D::XMax)];
                        a[i] = -2.0 * r;
                        b[i] = 1.0 + 2.0 * r;
                        d[i] = input[index] + 2.0 * r * (input[index - 1] - input[index]) +
                               4.0 * r * bc.value * h;
                    } else {
                        a[i] = -r;
                        b[i] = 1.0 + 2.0 * r;
                        c[i] = -r;
                        d[i] = input[index] +
                               r * (input[index - 1] - 2.0 * input[index] + input[index + 1]);
                    }
                }
                const auto line = linalg::solve_tridiagonal(a, b, c, d);
                for (int i = 0; i <= mesh_.nx(); ++i)
                    output[static_cast<std::size_t>(mesh_.index(i, j, k))] = line[i];
            }
        }
        return output;
    }

    std::vector<double> sweepY(const std::vector<double>& input, double duration) const {
        const int nodes = mesh_.ny() + 1;
        const int stride = mesh_.strideJ();
        const double h = mesh_.dy();
        const double r = diffusivity_ * duration / (2.0 * h * h);
        std::vector<double> output(input);
        std::vector<double> a(nodes), b(nodes), c(nodes), d(nodes);

        for (int k = 0; k <= mesh_.nz(); ++k) {
            for (int i = 0; i <= mesh_.nx(); ++i) {
                std::fill(a.begin(), a.end(), 0.0);
                std::fill(b.begin(), b.end(), 0.0);
                std::fill(c.begin(), c.end(), 0.0);
                std::fill(d.begin(), d.end(), 0.0);
                for (int j = 0; j <= mesh_.ny(); ++j) {
                    const int index = mesh_.index(i, j, k);
                    if (const auto fixed = dirichletValue(i, j, k)) {
                        b[j] = 1.0;
                        d[j] = *fixed;
                    } else if (j == 0) {
                        const auto& bc = boundary_conditions_[checkedIndex(Boundary3D::YMin)];
                        b[j] = 1.0 + 2.0 * r;
                        c[j] = -2.0 * r;
                        d[j] = input[index] + 2.0 * r * (input[index + stride] - input[index]) +
                               4.0 * r * bc.value * h;
                    } else if (j == mesh_.ny()) {
                        const auto& bc = boundary_conditions_[checkedIndex(Boundary3D::YMax)];
                        a[j] = -2.0 * r;
                        b[j] = 1.0 + 2.0 * r;
                        d[j] = input[index] + 2.0 * r * (input[index - stride] - input[index]) +
                               4.0 * r * bc.value * h;
                    } else {
                        a[j] = -r;
                        b[j] = 1.0 + 2.0 * r;
                        c[j] = -r;
                        d[j] = input[index] + r * (input[index - stride] - 2.0 * input[index] +
                                                   input[index + stride]);
                    }
                }
                const auto line = linalg::solve_tridiagonal(a, b, c, d);
                for (int j = 0; j <= mesh_.ny(); ++j)
                    output[static_cast<std::size_t>(mesh_.index(i, j, k))] = line[j];
            }
        }
        return output;
    }

    std::vector<double> sweepZ(const std::vector<double>& input, double duration) const {
        const int nodes = mesh_.nz() + 1;
        const int stride = mesh_.strideK();
        const double h = mesh_.dz();
        const double r = diffusivity_ * duration / (2.0 * h * h);
        std::vector<double> output(input);
        std::vector<double> a(nodes), b(nodes), c(nodes), d(nodes);

        for (int j = 0; j <= mesh_.ny(); ++j) {
            for (int i = 0; i <= mesh_.nx(); ++i) {
                std::fill(a.begin(), a.end(), 0.0);
                std::fill(b.begin(), b.end(), 0.0);
                std::fill(c.begin(), c.end(), 0.0);
                std::fill(d.begin(), d.end(), 0.0);
                for (int k = 0; k <= mesh_.nz(); ++k) {
                    const int index = mesh_.index(i, j, k);
                    if (const auto fixed = dirichletValue(i, j, k)) {
                        b[k] = 1.0;
                        d[k] = *fixed;
                    } else if (k == 0) {
                        const auto& bc = boundary_conditions_[checkedIndex(Boundary3D::ZMin)];
                        b[k] = 1.0 + 2.0 * r;
                        c[k] = -2.0 * r;
                        d[k] = input[index] + 2.0 * r * (input[index + stride] - input[index]) +
                               4.0 * r * bc.value * h;
                    } else if (k == mesh_.nz()) {
                        const auto& bc = boundary_conditions_[checkedIndex(Boundary3D::ZMax)];
                        a[k] = -2.0 * r;
                        b[k] = 1.0 + 2.0 * r;
                        d[k] = input[index] + 2.0 * r * (input[index - stride] - input[index]) +
                               4.0 * r * bc.value * h;
                    } else {
                        a[k] = -r;
                        b[k] = 1.0 + 2.0 * r;
                        c[k] = -r;
                        d[k] = input[index] + r * (input[index - stride] - 2.0 * input[index] +
                                                   input[index + stride]);
                    }
                }
                const auto line = linalg::solve_tridiagonal(a, b, c, d);
                for (int k = 0; k <= mesh_.nz(); ++k)
                    output[static_cast<std::size_t>(mesh_.index(i, j, k))] = line[k];
            }
        }
        return output;
    }
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_ADI_SOLVER_HPP
