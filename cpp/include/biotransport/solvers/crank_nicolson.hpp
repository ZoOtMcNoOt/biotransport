#ifndef BIOTRANSPORT_SOLVERS_CRANK_NICOLSON_HPP
#define BIOTRANSPORT_SOLVERS_CRANK_NICOLSON_HPP

/**
 * @file crank_nicolson.hpp
 * @brief Conservative Crank--Nicolson integration for 1D/2D diffusion.
 *
 * The spatial operator is a vertex-centred finite-volume discretization of
 *
 *     du/dt = div(D grad(u)).
 *
 * Boundary nodes therefore own half control volumes.  Homogeneous Neumann
 * boundaries conserve the trapezoidal-volume integral to linear-solver
 * tolerance.  Neumann values are outward-normal derivatives du/dn; they are
 * not physical fluxes.  The symmetric positive-definite free-node system is
 * solved with diagonally preconditioned conjugate gradients.
 *
 * Crank--Nicolson is A-stable and second order in time, but it is not L-stable:
 * very large steps can produce bounded temporal oscillations for poorly
 * resolved high-frequency initial data.
 *
 * Dirichlet traces that meet at a node must agree to within
 * 64*epsilon*max(1, |a|, |b|).  Larger discrepancies are contradictory data
 * and raise std::invalid_argument before the public solution is advanced.
 */

#include <algorithm>
#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace biotransport {

struct CNSolveResult {
    int iterations = 0;      ///< PCG iterations used.
    double residual = 0.0;   ///< Relative infinity norm of the algebraic residual.
    bool converged = false;  ///< True only when residual <= configured tolerance.
};

class CrankNicolsonDiffusion {
public:
    CrankNicolsonDiffusion(const StructuredMesh& mesh, double diffusivity)
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

    void setDirichletBoundary(Boundary boundary, double value) {
        validateBoundaryForMesh(boundary);
        requireFinite(value, "Dirichlet value");
        boundary_conditions_[checkedIndex(boundary)] = BoundaryCondition::Dirichlet(value);
    }

    void setNeumannBoundary(Boundary boundary, double normal_derivative) {
        validateBoundaryForMesh(boundary);
        requireFinite(normal_derivative, "Neumann outward-normal derivative");
        boundary_conditions_[checkedIndex(boundary)] =
            BoundaryCondition::Neumann(normal_derivative);
    }

    CrankNicolsonDiffusion& setTolerance(double tolerance) {
        requirePositiveFinite(tolerance, "Linear-solver tolerance");
        tolerance_ = tolerance;
        return *this;
    }

    CrankNicolsonDiffusion& setMaxIterations(int max_iterations) {
        if (max_iterations <= 0) {
            throw std::invalid_argument("Maximum iterations must be positive");
        }
        max_iterations_ = max_iterations;
        return *this;
    }

    CNSolveResult step(double dt) {
        requirePositiveFinite(dt, "Time step");
        const double half_dt = 0.5 * dt;
        const std::size_t count = solution_.size();

        std::vector<double> old = solution_;
        imposeDirichlet(old);

        std::vector<unsigned char> free_node(count, 1);
        std::vector<double> fixed(count, 0.0);
        std::vector<double> diagonal(count, 1.0);
        std::vector<double> rhs(count, 0.0);

        forEachNode([&](int i, int j, std::size_t index) {
            if (const auto value = dirichletValue(i, j)) {
                free_node[index] = 0;
                fixed[index] = *value;
                rhs[index] = *value;
                return;
            }

            const double volume = controlVolume(i, j);
            const double sum_g = conductanceSum(i, j);
            diagonal[index] = volume + half_dt * sum_g;
            rhs[index] =
                volume * old[index] + half_dt * internalFlux(old, i, j) + dt * boundaryFlux(i, j);

            forEachNeighbor(i, j, [&](int ni, int nj, double conductance) {
                if (const auto value = dirichletValue(ni, nj)) {
                    rhs[index] += half_dt * conductance * *value;
                }
            });
        });

        auto apply_operator = [&](const std::vector<double>& input, std::vector<double>& output) {
            std::fill(output.begin(), output.end(), 0.0);
            forEachNode([&](int i, int j, std::size_t index) {
                if (!free_node[index])
                    return;
                double value = diagonal[index] * input[index];
                forEachNeighbor(i, j, [&](int ni, int nj, double conductance) {
                    const std::size_t neighbor = nodeIndex(ni, nj);
                    if (free_node[neighbor])
                        value -= half_dt * conductance * input[neighbor];
                });
                output[index] = value;
            });
        };

        std::vector<double> x(old);
        std::vector<double> residual(count, 0.0);
        std::vector<double> preconditioned(count, 0.0);
        std::vector<double> direction(count, 0.0);
        std::vector<double> product(count, 0.0);
        apply_operator(x, product);

        double rhs_scale = 1.0;
        for (std::size_t index = 0; index < count; ++index) {
            if (!free_node[index]) {
                x[index] = fixed[index];
                continue;
            }
            residual[index] = rhs[index] - product[index];
            preconditioned[index] = residual[index] / diagonal[index];
            direction[index] = preconditioned[index];
            rhs_scale = std::max(rhs_scale, std::abs(rhs[index]));
        }

        CNSolveResult result;
        result.residual = residualInfinityNorm(residual, free_node) / rhs_scale;
        if (result.residual <= tolerance_) {
            result.converged = true;
        }

        double rz = dotFree(residual, preconditioned, free_node);
        for (int iteration = 0; !result.converged && iteration < max_iterations_; ++iteration) {
            apply_operator(direction, product);
            const double denominator = dotFree(direction, product, free_node);
            if (!std::isfinite(denominator) || denominator <= 0.0 || !std::isfinite(rz)) {
                throw std::runtime_error(
                    "Crank-Nicolson PCG lost positive definiteness or produced non-finite data");
            }

            const double alpha = rz / denominator;
            for (std::size_t index = 0; index < count; ++index) {
                if (!free_node[index])
                    continue;
                x[index] += alpha * direction[index];
                residual[index] -= alpha * product[index];
            }

            result.iterations = iteration + 1;
            result.residual = residualInfinityNorm(residual, free_node) / rhs_scale;
            if (!std::isfinite(result.residual)) {
                throw std::runtime_error("Crank-Nicolson PCG produced a non-finite residual");
            }
            if (result.residual <= tolerance_) {
                result.converged = true;
                break;
            }

            for (std::size_t index = 0; index < count; ++index) {
                if (free_node[index])
                    preconditioned[index] = residual[index] / diagonal[index];
            }
            const double rz_new = dotFree(residual, preconditioned, free_node);
            const double beta = rz_new / rz;
            for (std::size_t index = 0; index < count; ++index) {
                if (free_node[index])
                    direction[index] = preconditioned[index] + beta * direction[index];
            }
            rz = rz_new;
        }

        if (result.converged) {
            for (std::size_t index = 0; index < count; ++index) {
                if (!free_node[index])
                    x[index] = fixed[index];
            }
            requireFiniteResult(x, "Crank-Nicolson solution");
            solution_.swap(x);
            time_ += dt;
        }
        return result;
    }

    void solve(double dt, int num_steps) {
        requirePositiveFinite(dt, "Time step");
        if (num_steps < 0) {
            throw std::invalid_argument("Number of steps must be non-negative");
        }
        for (int step_index = 0; step_index < num_steps; ++step_index) {
            const CNSolveResult result = step(dt);
            if (!result.converged) {
                throw std::runtime_error("Crank-Nicolson linear solve did not converge at step " +
                                         std::to_string(step_index) +
                                         "; relative residual=" + std::to_string(result.residual));
            }
        }
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
    double tolerance_ = 1e-10;
    int max_iterations_ = 10000;

    static void requirePositiveFinite(double value, const char* name) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(std::string(name) + " must be finite and positive");
        }
    }

    static void requireFinite(double value, const char* name) {
        if (!std::isfinite(value))
            throw std::invalid_argument(std::string(name) + " must be finite");
    }

    static void requireFiniteInput(const std::vector<double>& values, const char* name) {
        for (double value : values)
            requireFinite(value, name);
    }

    static void requireFiniteResult(const std::vector<double>& values, const char* name) {
        for (double value : values) {
            if (!std::isfinite(value))
                throw std::runtime_error(std::string(name) + " contains a non-finite value");
        }
    }

    static std::size_t checkedIndex(Boundary boundary) {
        const int index = to_index(boundary);
        if (index < 0 || index >= 4)
            throw std::invalid_argument("Boundary identifier is outside [0, 3]");
        return static_cast<std::size_t>(index);
    }

    void validateBoundaryForMesh(Boundary boundary) const {
        (void)checkedIndex(boundary);
        if (mesh_.is1D() && (boundary == Boundary::Bottom || boundary == Boundary::Top)) {
            throw std::invalid_argument("Bottom/Top boundaries do not exist on a 1D mesh");
        }
    }

    std::size_t nodeIndex(int i, int j) const {
        return static_cast<std::size_t>(mesh_.index(i, mesh_.is1D() ? 0 : j));
    }

    template <typename Function>
    void forEachNode(Function&& function) const {
        const int last_j = mesh_.is1D() ? 0 : mesh_.ny();
        for (int j = 0; j <= last_j; ++j) {
            for (int i = 0; i <= mesh_.nx(); ++i)
                function(i, j, nodeIndex(i, j));
        }
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
        if (!mesh_.is1D()) {
            if (j == 0)
                add(Boundary::Bottom);
            if (j == mesh_.ny())
                add(Boundary::Top);
        }
        if (count == 0)
            return std::nullopt;
        return sum / static_cast<double>(count);
    }

    void imposeDirichlet(std::vector<double>& values) const {
        forEachNode([&](int i, int j, std::size_t index) {
            if (const auto fixed = dirichletValue(i, j))
                values[index] = *fixed;
        });
    }

    double xWidth(int i) const { return mesh_.dx() * ((i == 0 || i == mesh_.nx()) ? 0.5 : 1.0); }

    double yWidth(int j) const {
        if (mesh_.is1D())
            return 1.0;
        return mesh_.dy() * ((j == 0 || j == mesh_.ny()) ? 0.5 : 1.0);
    }

    double controlVolume(int i, int j) const { return xWidth(i) * yWidth(j); }

    template <typename Function>
    void forEachNeighbor(int i, int j, Function&& function) const {
        const double x_conductance = diffusivity_ * yWidth(j) / mesh_.dx();
        if (i > 0)
            function(i - 1, j, x_conductance);
        if (i < mesh_.nx())
            function(i + 1, j, x_conductance);

        if (!mesh_.is1D()) {
            const double y_conductance = diffusivity_ * xWidth(i) / mesh_.dy();
            if (j > 0)
                function(i, j - 1, y_conductance);
            if (j < mesh_.ny())
                function(i, j + 1, y_conductance);
        }
    }

    double conductanceSum(int i, int j) const {
        double sum = 0.0;
        forEachNeighbor(i, j, [&](int, int, double conductance) { sum += conductance; });
        return sum;
    }

    double internalFlux(const std::vector<double>& values, int i, int j) const {
        const std::size_t index = nodeIndex(i, j);
        double flux = 0.0;
        forEachNeighbor(i, j, [&](int ni, int nj, double conductance) {
            flux += conductance * (values[nodeIndex(ni, nj)] - values[index]);
        });
        return flux;
    }

    double boundaryFlux(int i, int j) const {
        double flux = 0.0;
        const auto add = [&](Boundary face, double area) {
            const auto& bc = boundary_conditions_[checkedIndex(face)];
            if (bc.type == BoundaryType::NEUMANN)
                flux += diffusivity_ * bc.value * area;
        };
        if (i == 0)
            add(Boundary::Left, yWidth(j));
        if (i == mesh_.nx())
            add(Boundary::Right, yWidth(j));
        if (!mesh_.is1D()) {
            if (j == 0)
                add(Boundary::Bottom, xWidth(i));
            if (j == mesh_.ny())
                add(Boundary::Top, xWidth(i));
        }
        return flux;
    }

    static double residualInfinityNorm(const std::vector<double>& residual,
                                       const std::vector<unsigned char>& free_node) {
        double norm = 0.0;
        for (std::size_t index = 0; index < residual.size(); ++index) {
            if (free_node[index])
                norm = std::max(norm, std::abs(residual[index]));
        }
        return norm;
    }

    static double dotFree(const std::vector<double>& lhs, const std::vector<double>& rhs,
                          const std::vector<unsigned char>& free_node) {
        double value = 0.0;
        for (std::size_t index = 0; index < lhs.size(); ++index) {
            if (free_node[index])
                value += lhs[index] * rhs[index];
        }
        return value;
    }
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_CRANK_NICOLSON_HPP
