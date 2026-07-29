#ifndef BIOTRANSPORT_SOLVERS_TRANSPORT_SOLVER_HPP
#define BIOTRANSPORT_SOLVERS_TRANSPORT_SOLVER_HPP

/**
 * @file transport_solver.hpp
 * @brief Science-first conservative solver for scalar transport.
 *
 * This header implements the complete equation represented by
 * TransportProblem:
 *
 * @f[
 *   \partial_t c = \nabla\cdot(D\nabla c) - \nabla\cdot(\mathbf{v}c)
 *                  + R(c,x,y,t).
 * @f]
 *
 * The spatial operator is a nodal finite-volume balance.  Diffusive face
 * values use harmonic averaging and advective face values use conservative
 * first-order upwinding.  Boundary nodes own half control volumes, so zero
 * physical-face fluxes conserve the trapezoidal integral to roundoff.
 *
 * The implementation is intentionally strict: unsupported advection schemes,
 * malformed fields, non-finite model evaluations, ill-posed pure-advection
 * inflows, and time steps above the certified explicit limit throw exceptions.
 */

#include <algorithm>
#include <biotransport/core/problems/transport_problem.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace biotransport {

/** @brief Controls a conservative explicit transport solve. */
struct SolveOptions {
    /** Physical duration to integrate.  Zero validates and returns the initial state. */
    double final_time = 0.0;

    /**
     * Maximum step [time].  Set to zero for an automatically certified step.
     * The last step is shortened so the result lands exactly on final_time.
     */
    double time_step = 0.0;

    /** Fraction of the certified explicit stability limit used in automatic mode. */
    double safety_factor = 0.8;

    /**
     * Accuracy guard for bounded reactions: auto dt <= this/max|dR/dc|.
     * The default resolves the fastest declared reaction time scale with at
     * least ten steps.  It is an accuracy policy, separate from CFL stability.
     */
    double reaction_step_fraction = 0.1;

    /** Guard against accidentally enormous explicit runs. */
    std::size_t max_steps = 10'000'000;

    /** Throw as soon as a reaction or solution value is NaN or infinite. */
    bool check_finite = true;

    /** @brief Convenience constructor for `solve(problem, SolveOptions::until(t))`. */
    static SolveOptions until(double time) {
        SolveOptions options;
        options.final_time = time;
        return options;
    }
};

/** @brief Numerical and physical diagnostics produced by solve(). */
struct SolveDiagnostics {
    std::size_t steps = 0;
    double requested_final_time = 0.0;
    double final_time = 0.0;
    double requested_time_step = 0.0;
    double minimum_time_step = 0.0;
    double maximum_time_step = 0.0;

    /** Stability limit for diffusion plus conservative upwind advection only. */
    double transport_stable_time_step = std::numeric_limits<double>::infinity();

    /**
     * Limit including a reaction derivative bound.  NaN means the reaction
     * was custom/unbounded and no complete stability certification was possible.
     */
    double certified_stable_time_step = std::numeric_limits<double>::infinity();

    double maximum_transport_loss_rate = 0.0;
    double reaction_rate_bound = 0.0;
    bool automatic_time_step = true;
    bool reaction_stability_bound_known = true;

    double initial_mass = 0.0;
    double final_mass = 0.0;
    double mass_change = 0.0;
    double initial_minimum = 0.0;
    double initial_maximum = 0.0;
    double final_minimum = 0.0;
    double final_maximum = 0.0;
};

/** @brief Final concentration field and diagnostics from a transport solve. */
struct TransportResult {
    std::vector<double> concentration;
    double time = 0.0;
    SolveDiagnostics diagnostics;

    /** @brief Compatibility/readability alias for the final concentration field. */
    const std::vector<double>& solution() const noexcept { return concentration; }
    std::vector<double>& solution() noexcept { return concentration; }
};

namespace transport_detail {

constexpr double boundary_conflict_tolerance = 64.0 * std::numeric_limits<double>::epsilon();

inline std::size_t asSize(int value) {
    return static_cast<std::size_t>(value);
}

inline bool finite(double value) {
    return std::isfinite(value);
}

inline bool essential(const BoundaryCondition& condition) {
    return condition.type == BoundaryType::DIRICHLET ||
           (condition.type == BoundaryType::ROBIN && condition.b == 0.0);
}

inline double essentialValue(const BoundaryCondition& condition) {
    if (condition.type == BoundaryType::DIRICHLET) {
        return condition.value;
    }
    return condition.c / condition.a;
}

inline bool closeBoundaryValues(double first, double second) {
    const double scale = std::max({1.0, std::abs(first), std::abs(second)});
    return std::abs(first - second) <= boundary_conflict_tolerance * scale;
}

struct EssentialBoundaryData {
    std::vector<unsigned char> mask;
    std::vector<double> value;
};

inline void includeEssentialCandidate(const BoundaryCondition& condition, bool& found,
                                      double& selected) {
    if (!essential(condition)) {
        return;
    }
    const double candidate = essentialValue(condition);
    if (!finite(candidate)) {
        throw std::invalid_argument("essential boundary value is not finite");
    }
    if (!found) {
        found = true;
        selected = candidate;
    } else if (!closeBoundaryValues(selected, candidate)) {
        throw std::invalid_argument("conflicting essential boundary values meet at a corner");
    }
}

/**
 * Dirichlet and b=0 Robin conditions are essential.  At a corner, every
 * essential condition must agree; otherwise the model is contradictory and
 * solve() throws.  Natural conditions on both corner faces are applied
 * independently in the finite-volume balance.
 */
inline EssentialBoundaryData makeEssentialBoundaryData(const TransportProblem& problem) {
    const StructuredMesh& mesh = problem.mesh();
    EssentialBoundaryData data;
    data.mask.assign(asSize(mesh.numNodes()), 0);
    data.value.assign(asSize(mesh.numNodes()), 0.0);
    const auto& boundaries = problem.boundaries();

    const int last_y = mesh.is1D() ? 0 : mesh.ny();
    for (int j = 0; j <= last_y; ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            bool found = false;
            double selected = 0.0;
            if (i == 0) {
                includeEssentialCandidate(boundaries[0], found, selected);
            }
            if (i == mesh.nx()) {
                includeEssentialCandidate(boundaries[1], found, selected);
            }
            if (!mesh.is1D() && j == 0) {
                includeEssentialCandidate(boundaries[2], found, selected);
            }
            if (!mesh.is1D() && j == mesh.ny()) {
                includeEssentialCandidate(boundaries[3], found, selected);
            }
            if (found) {
                const std::size_t index = asSize(mesh.index(i, j));
                data.mask[index] = 1;
                data.value[index] = selected;
            }
        }
    }
    return data;
}

inline void applyEssentialBoundaries(std::vector<double>& concentration,
                                     const EssentialBoundaryData& essential_data) {
    for (std::size_t index = 0; index < concentration.size(); ++index) {
        if (essential_data.mask[index] != 0) {
            concentration[index] = essential_data.value[index];
        }
    }
}

inline double diffusivityAt(const TransportProblem& problem, std::size_t index) {
    return problem.hasUniformDiffusivity() ? problem.diffusivity()
                                           : problem.diffusivityField()[index];
}

inline double vxAt(const TransportProblem& problem, std::size_t index) {
    if (!problem.hasAdvection()) {
        return 0.0;
    }
    return problem.hasUniformVelocity() ? problem.vxUniform() : problem.vxField()[index];
}

inline double vyAt(const TransportProblem& problem, std::size_t index) {
    if (!problem.hasAdvection()) {
        return 0.0;
    }
    return problem.hasUniformVelocity() ? problem.vyUniform() : problem.vyField()[index];
}

inline double harmonicMean(double first, double second) {
    if (first == 0.0 || second == 0.0) {
        return 0.0;
    }
    const double smaller = std::min(first, second);
    const double larger = std::max(first, second);
    return smaller / (0.5 + 0.5 * smaller / larger);
}

inline double outwardDerivative(const BoundaryCondition& condition, double concentration) {
    if (condition.type == BoundaryType::NEUMANN) {
        return condition.value;
    }
    if (condition.type == BoundaryType::ROBIN && condition.b != 0.0) {
        return (condition.c - condition.a * concentration) / condition.b;
    }
    throw std::logic_error("an essential boundary has no natural derivative");
}

inline void validateBoundary(const BoundaryCondition& condition) {
    switch (condition.type) {
        case BoundaryType::DIRICHLET:
        case BoundaryType::NEUMANN:
            if (!finite(condition.value)) {
                throw std::invalid_argument("boundary value must be finite");
            }
            return;
        case BoundaryType::ROBIN:
            if (!finite(condition.a) || !finite(condition.b) || !finite(condition.c)) {
                throw std::invalid_argument("Robin coefficients must be finite");
            }
            if (condition.a == 0.0 && condition.b == 0.0) {
                throw std::invalid_argument("Robin condition requires non-zero a or b");
            }
            return;
    }
    throw std::invalid_argument("invalid BoundaryType value");
}

inline void validateProblem(const TransportProblem& problem, const SolveOptions& options) {
    const StructuredMesh& mesh = problem.mesh();
    if (mesh.nx() < 1 || !finite(mesh.dx()) || mesh.dx() <= 0.0) {
        throw std::invalid_argument("mesh must have at least one positive-width x cell");
    }
    if (!mesh.is1D() && (mesh.ny() < 1 || !finite(mesh.dy()) || mesh.dy() <= 0.0)) {
        throw std::invalid_argument("2D mesh must have at least one positive-width y cell");
    }
    if (!finite(options.final_time) || options.final_time < 0.0) {
        throw std::invalid_argument("final_time must be finite and non-negative");
    }
    if (!finite(options.time_step) || options.time_step < 0.0) {
        throw std::invalid_argument("time_step must be finite and non-negative");
    }
    if (!finite(options.safety_factor) || options.safety_factor <= 0.0 ||
        options.safety_factor > 1.0) {
        throw std::invalid_argument("safety_factor must be in (0, 1]");
    }
    if (!finite(options.reaction_step_fraction) || options.reaction_step_fraction <= 0.0 ||
        options.reaction_step_fraction > 1.0) {
        throw std::invalid_argument("reaction_step_fraction must be in (0, 1]");
    }
    if (options.max_steps == 0) {
        throw std::invalid_argument("max_steps must be positive");
    }

    const std::size_t node_count = asSize(mesh.numNodes());
    if (problem.initial().size() != node_count) {
        throw std::invalid_argument("initial condition size does not match the mesh");
    }
    for (double value : problem.initial()) {
        if (!finite(value)) {
            throw std::invalid_argument("initial condition values must be finite");
        }
    }

    if (problem.hasUniformDiffusivity()) {
        if (!finite(problem.diffusivity()) || problem.diffusivity() < 0.0) {
            throw std::invalid_argument("diffusivity must be finite and non-negative");
        }
    } else {
        if (problem.diffusivityField().size() != node_count) {
            throw std::invalid_argument("diffusivity field size does not match the mesh");
        }
        for (double value : problem.diffusivityField()) {
            if (!finite(value) || value < 0.0) {
                throw std::invalid_argument(
                    "diffusivity field values must be finite and non-negative");
            }
        }
    }

    if (problem.hasAdvection()) {
        if (problem.scheme() != AdvectionScheme::UPWIND) {
            throw std::invalid_argument(
                "transport_solver implements UPWIND advection only; CENTRAL, HYBRID, and QUICK "
                "are rejected until scientifically verified implementations are available");
        }
        if (problem.hasUniformVelocity()) {
            if (!finite(problem.vxUniform()) || !finite(problem.vyUniform())) {
                throw std::invalid_argument("velocity components must be finite");
            }
            if (mesh.is1D() && problem.vyUniform() != 0.0) {
                throw std::invalid_argument("1D problems cannot contain y velocity");
            }
        } else {
            if (problem.vxField().size() != node_count || problem.vyField().size() != node_count) {
                throw std::invalid_argument("velocity field size does not match the mesh");
            }
            for (std::size_t index = 0; index < node_count; ++index) {
                if (!finite(problem.vxField()[index]) || !finite(problem.vyField()[index])) {
                    throw std::invalid_argument("velocity field values must be finite");
                }
                if (mesh.is1D() && problem.vyField()[index] != 0.0) {
                    throw std::invalid_argument("1D problems cannot contain y velocity");
                }
            }
        }
    }

    for (const BoundaryCondition& condition : problem.boundaries()) {
        validateBoundary(condition);
    }
    if (problem.hasReaction() && !problem.reaction()) {
        throw std::invalid_argument("reaction function must be callable");
    }
}

inline bool hasEssentialAt(const EssentialBoundaryData& data, int index) {
    return data.mask[asSize(index)] != 0;
}

/**
 * A first-order equation needs concentration data at inflow.  If local
 * diffusion vanishes, a Neumann/derivative-only condition cannot provide it.
 */
inline void validateDegenerateInflows(const TransportProblem& problem,
                                      const EssentialBoundaryData& essential_data) {
    if (!problem.hasAdvection()) {
        return;
    }
    const StructuredMesh& mesh = problem.mesh();

    for (int j = 0; j <= (mesh.is1D() ? 0 : mesh.ny()); ++j) {
        const int left = mesh.index(0, j);
        const int right = mesh.index(mesh.nx(), j);
        if (diffusivityAt(problem, asSize(left)) == 0.0 && vxAt(problem, asSize(left)) > 0.0 &&
            !hasEssentialAt(essential_data, left)) {
            throw std::invalid_argument(
                "pure-advection inflow on the left requires Dirichlet (or b=0 Robin) data");
        }
        if (diffusivityAt(problem, asSize(right)) == 0.0 && vxAt(problem, asSize(right)) < 0.0 &&
            !hasEssentialAt(essential_data, right)) {
            throw std::invalid_argument(
                "pure-advection inflow on the right requires Dirichlet (or b=0 Robin) data");
        }
    }

    if (mesh.is1D()) {
        return;
    }
    for (int i = 0; i <= mesh.nx(); ++i) {
        const int bottom = mesh.index(i, 0);
        const int top = mesh.index(i, mesh.ny());
        if (diffusivityAt(problem, asSize(bottom)) == 0.0 && vyAt(problem, asSize(bottom)) > 0.0 &&
            !hasEssentialAt(essential_data, bottom)) {
            throw std::invalid_argument(
                "pure-advection inflow on the bottom requires Dirichlet (or b=0 Robin) data");
        }
        if (diffusivityAt(problem, asSize(top)) == 0.0 && vyAt(problem, asSize(top)) < 0.0 &&
            !hasEssentialAt(essential_data, top)) {
            throw std::invalid_argument(
                "pure-advection inflow on the top requires Dirichlet (or b=0 Robin) data");
        }
    }
}

inline double faceVelocityX(const TransportProblem& problem, std::size_t left, std::size_t right) {
    return 0.5 * vxAt(problem, left) + 0.5 * vxAt(problem, right);
}

inline double faceVelocityY(const TransportProblem& problem, std::size_t bottom, std::size_t top) {
    return 0.5 * vyAt(problem, bottom) + 0.5 * vyAt(problem, top);
}

inline double xFaceFlux(const TransportProblem& problem, const std::vector<double>& concentration,
                        std::size_t left, std::size_t right, double spacing) {
    const double diffusion =
        harmonicMean(diffusivityAt(problem, left), diffusivityAt(problem, right));
    const double velocity = faceVelocityX(problem, left, right);
    const double upwind = velocity >= 0.0 ? concentration[left] : concentration[right];
    // q = D grad(c) - v*c, and dc/dt = div(q) + R.
    return diffusion * (concentration[right] - concentration[left]) / spacing - velocity * upwind;
}

inline double yFaceFlux(const TransportProblem& problem, const std::vector<double>& concentration,
                        std::size_t bottom, std::size_t top, double spacing) {
    const double diffusion =
        harmonicMean(diffusivityAt(problem, bottom), diffusivityAt(problem, top));
    const double velocity = faceVelocityY(problem, bottom, top);
    const double upwind = velocity >= 0.0 ? concentration[bottom] : concentration[top];
    return diffusion * (concentration[top] - concentration[bottom]) / spacing - velocity * upwind;
}

inline double physicalXFlux(const TransportProblem& problem,
                            const std::vector<double>& concentration, std::size_t index,
                            Boundary side) {
    const BoundaryCondition& condition = problem.boundaries()[static_cast<std::size_t>(side)];
    const double derivative = outwardDerivative(condition, concentration[index]);
    const double orientation = side == Boundary::Left ? -1.0 : 1.0;
    return orientation * diffusivityAt(problem, index) * derivative -
           vxAt(problem, index) * concentration[index];
}

inline double physicalYFlux(const TransportProblem& problem,
                            const std::vector<double>& concentration, std::size_t index,
                            Boundary side) {
    const BoundaryCondition& condition = problem.boundaries()[static_cast<std::size_t>(side)];
    const double derivative = outwardDerivative(condition, concentration[index]);
    const double orientation = side == Boundary::Bottom ? -1.0 : 1.0;
    return orientation * diffusivityAt(problem, index) * derivative -
           vyAt(problem, index) * concentration[index];
}

inline double naturalRobinLoss(const BoundaryCondition& condition, double diffusivity,
                               double control_width) {
    if (condition.type != BoundaryType::ROBIN || condition.b == 0.0) {
        return 0.0;
    }
    return std::max(0.0, diffusivity * condition.a / (condition.b * control_width));
}

inline double transportLossRate(const TransportProblem& problem,
                                const EssentialBoundaryData& essential_data) {
    const StructuredMesh& mesh = problem.mesh();
    const auto& boundaries = problem.boundaries();
    double maximum_rate = 0.0;

    for (int j = 0; j <= (mesh.is1D() ? 0 : mesh.ny()); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const std::size_t centre = asSize(mesh.index(i, j));
            if (essential_data.mask[centre] != 0) {
                continue;
            }

            const double width_x = (i == 0 || i == mesh.nx()) ? 0.5 * mesh.dx() : mesh.dx();
            double loss_rate = 0.0;
            if (i > 0) {
                const std::size_t left = asSize(mesh.index(i - 1, j));
                const double diffusion =
                    harmonicMean(diffusivityAt(problem, left), diffusivityAt(problem, centre));
                const double velocity = faceVelocityX(problem, left, centre);
                loss_rate += diffusion / (mesh.dx() * width_x);
                if (velocity < 0.0) {
                    loss_rate += -velocity / width_x;
                }
            } else {
                loss_rate +=
                    naturalRobinLoss(boundaries[0], diffusivityAt(problem, centre), width_x);
                if (vxAt(problem, centre) < 0.0) {
                    loss_rate += -vxAt(problem, centre) / width_x;
                }
            }
            if (i < mesh.nx()) {
                const std::size_t right = asSize(mesh.index(i + 1, j));
                const double diffusion =
                    harmonicMean(diffusivityAt(problem, centre), diffusivityAt(problem, right));
                const double velocity = faceVelocityX(problem, centre, right);
                loss_rate += diffusion / (mesh.dx() * width_x);
                if (velocity > 0.0) {
                    loss_rate += velocity / width_x;
                }
            } else {
                loss_rate +=
                    naturalRobinLoss(boundaries[1], diffusivityAt(problem, centre), width_x);
                if (vxAt(problem, centre) > 0.0) {
                    loss_rate += vxAt(problem, centre) / width_x;
                }
            }

            if (!mesh.is1D()) {
                const double width_y = (j == 0 || j == mesh.ny()) ? 0.5 * mesh.dy() : mesh.dy();
                if (j > 0) {
                    const std::size_t bottom = asSize(mesh.index(i, j - 1));
                    const double diffusion = harmonicMean(diffusivityAt(problem, bottom),
                                                          diffusivityAt(problem, centre));
                    const double velocity = faceVelocityY(problem, bottom, centre);
                    loss_rate += diffusion / (mesh.dy() * width_y);
                    if (velocity < 0.0) {
                        loss_rate += -velocity / width_y;
                    }
                } else {
                    loss_rate +=
                        naturalRobinLoss(boundaries[2], diffusivityAt(problem, centre), width_y);
                    if (vyAt(problem, centre) < 0.0) {
                        loss_rate += -vyAt(problem, centre) / width_y;
                    }
                }
                if (j < mesh.ny()) {
                    const std::size_t top = asSize(mesh.index(i, j + 1));
                    const double diffusion =
                        harmonicMean(diffusivityAt(problem, centre), diffusivityAt(problem, top));
                    const double velocity = faceVelocityY(problem, centre, top);
                    loss_rate += diffusion / (mesh.dy() * width_y);
                    if (velocity > 0.0) {
                        loss_rate += velocity / width_y;
                    }
                } else {
                    loss_rate +=
                        naturalRobinLoss(boundaries[3], diffusivityAt(problem, centre), width_y);
                    if (vyAt(problem, centre) > 0.0) {
                        loss_rate += vyAt(problem, centre) / width_y;
                    }
                }
            }
            maximum_rate = std::max(maximum_rate, loss_rate);
        }
    }
    return maximum_rate;
}

inline double integrateMass(const StructuredMesh& mesh, const std::vector<double>& values) {
    double integral = 0.0;
    if (mesh.is1D()) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const double width = (i == 0 || i == mesh.nx()) ? 0.5 * mesh.dx() : mesh.dx();
            integral += values[asSize(mesh.index(i))] * width;
        }
        return integral;
    }

    for (int j = 0; j <= mesh.ny(); ++j) {
        const double height = (j == 0 || j == mesh.ny()) ? 0.5 * mesh.dy() : mesh.dy();
        for (int i = 0; i <= mesh.nx(); ++i) {
            const double width = (i == 0 || i == mesh.nx()) ? 0.5 * mesh.dx() : mesh.dx();
            integral += values[asSize(mesh.index(i, j))] * width * height;
        }
    }
    return integral;
}

inline std::pair<double, double> minmax(const std::vector<double>& values) {
    const auto bounds = std::minmax_element(values.begin(), values.end());
    return {*bounds.first, *bounds.second};
}

inline std::size_t plannedSteps(double final_time, double time_step) {
    if (final_time == 0.0) {
        return 0;
    }
    const long double ratio =
        static_cast<long double>(final_time) / static_cast<long double>(time_step);
    const long double approximate_count = std::ceil(ratio);
    if (!std::isfinite(approximate_count) ||
        approximate_count > static_cast<long double>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("requested solve requires too many time steps");
    }
    std::size_t count = std::max<std::size_t>(1, static_cast<std::size_t>(approximate_count));
    const long double final_time_exact = static_cast<long double>(final_time);
    const long double time_step_exact = static_cast<long double>(time_step);
    const auto final_remainder = [&]() {
        return std::fma(-static_cast<long double>(count - 1), time_step_exact, final_time_exact);
    };
    while (count > 1 && final_remainder() <= 0.0L) {
        --count;
    }
    while (final_remainder() > time_step_exact) {
        if (static_cast<double>(count) * time_step == final_time) {
            break;
        }
        if (count == std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("requested solve requires too many time steps");
        }
        ++count;
    }
    return count;
}

inline void buildInternalFluxes(const TransportProblem& problem,
                                const std::vector<double>& concentration,
                                std::vector<double>& x_flux, std::vector<double>& y_flux) {
    const StructuredMesh& mesh = problem.mesh();
    const int row_count = mesh.is1D() ? 1 : mesh.ny() + 1;
    for (int j = 0; j < row_count; ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            const std::size_t left = asSize(mesh.index(i, j));
            const std::size_t right = asSize(mesh.index(i + 1, j));
            x_flux[asSize(j * mesh.nx() + i)] =
                xFaceFlux(problem, concentration, left, right, mesh.dx());
        }
    }

    if (mesh.is1D()) {
        return;
    }
    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const std::size_t bottom = asSize(mesh.index(i, j));
            const std::size_t top = asSize(mesh.index(i, j + 1));
            y_flux[asSize(j * (mesh.nx() + 1) + i)] =
                yFaceFlux(problem, concentration, bottom, top, mesh.dy());
        }
    }
}

inline double xDivergence(const TransportProblem& problem, const std::vector<double>& concentration,
                          const std::vector<double>& x_flux, int i, int j) {
    const StructuredMesh& mesh = problem.mesh();
    const std::size_t centre = asSize(mesh.index(i, j));
    const double width = (i == 0 || i == mesh.nx()) ? 0.5 * mesh.dx() : mesh.dx();
    const double left_flux = i == 0 ? physicalXFlux(problem, concentration, centre, Boundary::Left)
                                    : x_flux[asSize(j * mesh.nx() + i - 1)];
    const double right_flux = i == mesh.nx()
                                  ? physicalXFlux(problem, concentration, centre, Boundary::Right)
                                  : x_flux[asSize(j * mesh.nx() + i)];
    return (right_flux - left_flux) / width;
}

inline double yDivergence(const TransportProblem& problem, const std::vector<double>& concentration,
                          const std::vector<double>& y_flux, int i, int j) {
    const StructuredMesh& mesh = problem.mesh();
    const std::size_t centre = asSize(mesh.index(i, j));
    const double height = (j == 0 || j == mesh.ny()) ? 0.5 * mesh.dy() : mesh.dy();
    const double bottom_flux = j == 0
                                   ? physicalYFlux(problem, concentration, centre, Boundary::Bottom)
                                   : y_flux[asSize((j - 1) * (mesh.nx() + 1) + i)];
    const double top_flux = j == mesh.ny()
                                ? physicalYFlux(problem, concentration, centre, Boundary::Top)
                                : y_flux[asSize(j * (mesh.nx() + 1) + i)];
    return (top_flux - bottom_flux) / height;
}

inline void takeStep(const TransportProblem& problem, const EssentialBoundaryData& essential_data,
                     std::vector<double>& concentration, std::vector<double>& next,
                     std::vector<double>& x_flux, std::vector<double>& y_flux, double time,
                     double dt, bool check_finite) {
    const StructuredMesh& mesh = problem.mesh();
    buildInternalFluxes(problem, concentration, x_flux, y_flux);

    for (int j = 0; j <= (mesh.is1D() ? 0 : mesh.ny()); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const std::size_t index = asSize(mesh.index(i, j));
            if (essential_data.mask[index] != 0) {
                next[index] = essential_data.value[index];
                continue;
            }

            double rate = xDivergence(problem, concentration, x_flux, i, j);
            if (!mesh.is1D()) {
                rate += yDivergence(problem, concentration, y_flux, i, j);
            }
            if (problem.hasReaction()) {
                const double reaction = problem.reaction()(concentration[index], mesh.x(i),
                                                           mesh.is1D() ? 0.0 : mesh.y(i, j), time);
                if (check_finite && !finite(reaction)) {
                    throw std::runtime_error("reaction returned a non-finite value at node " +
                                             std::to_string(index));
                }
                rate += reaction;
            }
            next[index] = concentration[index] + dt * rate;
            if (check_finite && !finite(next[index])) {
                throw std::runtime_error("transport solution became non-finite at node " +
                                         std::to_string(index));
            }
        }
    }

    applyEssentialBoundaries(next, essential_data);
    concentration.swap(next);
}

}  // namespace transport_detail

/**
 * @brief Solve every configured term in a TransportProblem.
 *
 * Automatic stepping requires a known reaction derivative bound.  Built-in
 * bounded kinetics provide one.  For a custom reaction, either use
 * TransportProblem::reaction(function, max_abs_dc) or explicitly choose
 * SolveOptions::time_step.  An explicit step with an unbounded custom reaction
 * is marked as uncertified in the diagnostics; no false stability claim is made.
 */
inline TransportResult solve(const TransportProblem& problem, const SolveOptions& options) {
    using namespace transport_detail;

    validateProblem(problem, options);
    const EssentialBoundaryData essential_data = makeEssentialBoundaryData(problem);
    validateDegenerateInflows(problem, essential_data);

    TransportResult result;
    result.concentration = problem.initial();
    // Essential values are imposed before the first flux/reaction stencil.
    applyEssentialBoundaries(result.concentration, essential_data);

    SolveDiagnostics& diagnostics = result.diagnostics;
    diagnostics.requested_final_time = options.final_time;
    diagnostics.requested_time_step = options.time_step;
    diagnostics.automatic_time_step = (options.time_step == 0.0);
    diagnostics.reaction_stability_bound_known =
        !problem.hasReaction() || problem.reactionStabilityBoundKnown();
    diagnostics.reaction_rate_bound = diagnostics.reaction_stability_bound_known
                                          ? problem.reactionStabilityRateBound()
                                          : std::numeric_limits<double>::quiet_NaN();

    const double transport_rate = transportLossRate(problem, essential_data);
    diagnostics.maximum_transport_loss_rate = transport_rate;
    diagnostics.transport_stable_time_step =
        transport_rate > 0.0 ? 1.0 / transport_rate : std::numeric_limits<double>::infinity();

    if (diagnostics.reaction_stability_bound_known) {
        const double total_rate = transport_rate + problem.reactionStabilityRateBound();
        diagnostics.certified_stable_time_step =
            total_rate > 0.0 ? 1.0 / total_rate : std::numeric_limits<double>::infinity();
    } else {
        diagnostics.certified_stable_time_step = std::numeric_limits<double>::quiet_NaN();
    }

    const auto initial_bounds = minmax(result.concentration);
    diagnostics.initial_minimum = initial_bounds.first;
    diagnostics.initial_maximum = initial_bounds.second;
    diagnostics.initial_mass = integrateMass(problem.mesh(), result.concentration);

    if (options.final_time == 0.0) {
        diagnostics.final_time = 0.0;
        diagnostics.final_minimum = diagnostics.initial_minimum;
        diagnostics.final_maximum = diagnostics.initial_maximum;
        diagnostics.final_mass = diagnostics.initial_mass;
        result.time = 0.0;
        return result;
    }

    double target_step = options.time_step;
    if (target_step == 0.0) {
        if (!diagnostics.reaction_stability_bound_known) {
            throw std::invalid_argument(
                "automatic time stepping requires a reaction derivative bound; supply "
                "reaction(function, max_abs_dc) or set SolveOptions::time_step explicitly");
        }
        if (finite(diagnostics.certified_stable_time_step)) {
            target_step = options.safety_factor * diagnostics.certified_stable_time_step;
            if (problem.reactionStabilityRateBound() > 0.0) {
                target_step = std::min(target_step, options.reaction_step_fraction /
                                                        problem.reactionStabilityRateBound());
            }
        } else {
            // A constant source/no operator has no stability ceiling; one Euler
            // step is exact for a truly constant source and transparent in diagnostics.
            target_step = options.final_time;
        }
    } else {
        const double enforced_limit = diagnostics.reaction_stability_bound_known
                                          ? diagnostics.certified_stable_time_step
                                          : diagnostics.transport_stable_time_step;
        if (finite(enforced_limit) &&
            target_step > enforced_limit * (1.0 + 16.0 * std::numeric_limits<double>::epsilon())) {
            throw std::invalid_argument(
                diagnostics.reaction_stability_bound_known
                    ? "time_step exceeds the certified explicit stability limit"
                    : "time_step exceeds the transport stability limit; custom reaction remains "
                      "uncertified");
        }
    }

    if (!finite(target_step) || target_step <= 0.0) {
        throw std::invalid_argument("the selected time step must be finite and positive");
    }

    const std::size_t step_count = plannedSteps(options.final_time, target_step);
    if (step_count > options.max_steps) {
        throw std::runtime_error(
            "solve would exceed max_steps; refine the mesh/model or raise the guard");
    }

    const StructuredMesh& mesh = problem.mesh();
    const std::size_t x_face_count = asSize(mesh.nx()) * asSize(mesh.is1D() ? 1 : mesh.ny() + 1);
    const std::size_t y_face_count = mesh.is1D() ? 0 : asSize(mesh.nx() + 1) * asSize(mesh.ny());
    std::vector<double> next(result.concentration.size(), 0.0);
    std::vector<double> x_flux(x_face_count, 0.0);
    std::vector<double> y_flux(y_face_count, 0.0);

    diagnostics.minimum_time_step = std::numeric_limits<double>::infinity();
    const long double final_time_schedule = static_cast<long double>(options.final_time);
    const long double target_step_schedule = static_cast<long double>(target_step);
    for (std::size_t step = 0; step < step_count; ++step) {
        const long double remaining_schedule =
            std::fma(-static_cast<long double>(step), target_step_schedule, final_time_schedule);
        const long double elapsed_schedule = final_time_schedule - remaining_schedule;
        double time = static_cast<double>(elapsed_schedule);
        if (step + 1 == step_count && remaining_schedule > 0.0L && time >= options.final_time) {
            time = std::nextafter(options.final_time, -std::numeric_limits<double>::infinity());
        }
        const double dt = step + 1 == step_count
                              ? std::min(target_step, static_cast<double>(remaining_schedule))
                              : target_step;
        if (!(dt > 0.0) || !finite(dt)) {
            throw std::runtime_error("floating-point time schedule produced an invalid step");
        }
        takeStep(problem, essential_data, result.concentration, next, x_flux, y_flux, time, dt,
                 options.check_finite);
        diagnostics.minimum_time_step = std::min(diagnostics.minimum_time_step, dt);
        diagnostics.maximum_time_step = std::max(diagnostics.maximum_time_step, dt);
    }

    // Assigning the requested value avoids exposing harmless summation roundoff.
    result.time = options.final_time;
    diagnostics.steps = step_count;
    diagnostics.final_time = options.final_time;
    const auto final_bounds = minmax(result.concentration);
    diagnostics.final_minimum = final_bounds.first;
    diagnostics.final_maximum = final_bounds.second;
    diagnostics.final_mass = integrateMass(mesh, result.concentration);
    diagnostics.mass_change = diagnostics.final_mass - diagnostics.initial_mass;
    return result;
}

/** @brief Convenience overload using automatic stepping to `final_time`. */
inline TransportResult solve(const TransportProblem& problem, double final_time) {
    return solve(problem, SolveOptions::until(final_time));
}

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_TRANSPORT_SOLVER_HPP
