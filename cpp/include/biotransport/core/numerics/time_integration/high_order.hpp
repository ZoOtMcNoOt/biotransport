#ifndef BIOTRANSPORT_CORE_NUMERICS_TIME_INTEGRATION_HIGH_ORDER_HPP
#define BIOTRANSPORT_CORE_NUMERICS_TIME_INTEGRATION_HIGH_ORDER_HPP

/**
 * @file high_order.hpp
 * @brief Validated explicit Runge--Kutta and high-order diffusion kernels.
 *
 * The routines in this file are deliberately small, reusable numerical
 * kernels.  They validate dimensions and finiteness at every user-callback
 * boundary, never alias stage storage with the accepted state, and shorten
 * the final step so integrations end at the requested time exactly.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace biotransport {
namespace time_integration {

using State = std::vector<double>;
using RHSFunction = std::function<State(const State&, double)>;

enum class RungeKuttaMethod { HEUN = 2, CLASSICAL_RK4 = 4 };

struct RungeKuttaResult {
    State state;
    double initial_time = 0.0;
    double time = 0.0;
    double nominal_dt = 0.0;
    double last_dt = 0.0;
    std::size_t steps = 0;
    RungeKuttaMethod method = RungeKuttaMethod::CLASSICAL_RK4;
};

namespace detail {

inline void requireFinite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

inline void validateState(const State& state, const char* name, bool allow_empty = false) {
    if (!allow_empty && state.empty()) {
        throw std::invalid_argument(std::string(name) + " must not be empty");
    }
    if (!std::all_of(state.begin(), state.end(),
                     [](double value) { return std::isfinite(value); })) {
        throw std::domain_error(std::string(name) + " must contain finite values only");
    }
}

inline State evaluateRhs(const RHSFunction& rhs, const State& state, double time,
                         const char* stage) {
    if (!rhs) {
        throw std::invalid_argument("right-hand side must be callable");
    }
    validateState(state, "Runge--Kutta stage state");
    requireFinite(time, "Runge--Kutta stage time");
    State derivative = rhs(state, time);
    if (derivative.size() != state.size()) {
        throw std::invalid_argument(std::string(stage) +
                                    " derivative dimension does not match the state");
    }
    validateState(derivative, stage);
    return derivative;
}

inline State addScaled(const State& base, const State& increment, double scale, const char* stage) {
    if (base.size() != increment.size()) {
        throw std::logic_error("internal Runge--Kutta stage dimension mismatch");
    }
    State result(base.size());
    for (std::size_t i = 0; i < base.size(); ++i) {
        result[i] = base[i] + scale * increment[i];
    }
    validateState(result, stage);
    return result;
}

inline double advancedTime(double time, double dt, double fraction = 1.0) {
    const double result = time + fraction * dt;
    if (!std::isfinite(result)) {
        throw std::overflow_error("Runge--Kutta stage time became non-finite");
    }
    if (fraction > 0.0 && result <= time) {
        throw std::runtime_error("time step is too small to advance the Runge--Kutta clock");
    }
    return result;
}

inline void validateRungeKuttaMethod(RungeKuttaMethod method) {
    switch (method) {
        case RungeKuttaMethod::HEUN:
        case RungeKuttaMethod::CLASSICAL_RK4:
            return;
    }
    throw std::invalid_argument("unsupported Runge--Kutta method");
}

inline int validateSpatialOrder(int order, bool is_2d = false) {
    if (order != 2 && order != 4 && order != 6) {
        throw std::invalid_argument("spatial order must be 2, 4, or 6");
    }
    if (is_2d && order == 6) {
        throw std::invalid_argument("sixth-order Laplacian is implemented for 1D fields only");
    }
    return order;
}

inline void validateSpacing(double spacing, const char* name) {
    requireFinite(spacing, name);
    if (spacing <= 0.0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
}

inline std::size_t minimumNodes(int order) {
    return static_cast<std::size_t>(order + 1);
}

inline double centeredLaplacianSpectralRadius(int order) {
    switch (order) {
        case 2:
            return 4.0;
        case 4:
            return 16.0 / 3.0;
        case 6:
            return 272.0 / 45.0;
        default:
            throw std::logic_error("unreachable spatial order");
    }
}

}  // namespace detail

inline State heunStep(const State& state, double time, double dt, const RHSFunction& rhs) {
    detail::validateState(state, "initial state");
    detail::requireFinite(time, "time");
    detail::requireFinite(dt, "time step");
    if (dt <= 0.0) {
        throw std::invalid_argument("time step must be positive");
    }

    const State k1 = detail::evaluateRhs(rhs, state, time, "Heun k1");
    const State predictor = detail::addScaled(state, k1, dt, "Heun predictor");
    const State k2 = detail::evaluateRhs(rhs, predictor, detail::advancedTime(time, dt), "Heun k2");

    State result(state.size());
    for (std::size_t i = 0; i < state.size(); ++i) {
        result[i] = state[i] + 0.5 * dt * (k1[i] + k2[i]);
    }
    detail::validateState(result, "Heun result");
    return result;
}

inline State classicalRk4Step(const State& state, double time, double dt, const RHSFunction& rhs) {
    detail::validateState(state, "initial state");
    detail::requireFinite(time, "time");
    detail::requireFinite(dt, "time step");
    if (dt <= 0.0) {
        throw std::invalid_argument("time step must be positive");
    }

    const State k1 = detail::evaluateRhs(rhs, state, time, "RK4 k1");
    const State stage2 = detail::addScaled(state, k1, 0.5 * dt, "RK4 stage 2");
    const State k2 =
        detail::evaluateRhs(rhs, stage2, detail::advancedTime(time, dt, 0.5), "RK4 k2");
    const State stage3 = detail::addScaled(state, k2, 0.5 * dt, "RK4 stage 3");
    const State k3 =
        detail::evaluateRhs(rhs, stage3, detail::advancedTime(time, dt, 0.5), "RK4 k3");
    const State stage4 = detail::addScaled(state, k3, dt, "RK4 stage 4");
    const State k4 = detail::evaluateRhs(rhs, stage4, detail::advancedTime(time, dt), "RK4 k4");

    State result(state.size());
    for (std::size_t i = 0; i < state.size(); ++i) {
        result[i] = state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    detail::validateState(result, "RK4 result");
    return result;
}

inline RungeKuttaResult integrateRungeKutta(
    const State& initial, double initial_time, double end_time, double dt, const RHSFunction& rhs,
    RungeKuttaMethod method = RungeKuttaMethod::CLASSICAL_RK4,
    std::size_t maximum_steps = 10000000) {
    detail::validateState(initial, "initial state");
    detail::requireFinite(initial_time, "initial time");
    detail::requireFinite(end_time, "end time");
    detail::requireFinite(dt, "time step");
    if (end_time < initial_time) {
        throw std::invalid_argument("end time must not precede initial time");
    }
    if (dt <= 0.0) {
        throw std::invalid_argument("time step must be positive");
    }
    if (maximum_steps == 0) {
        throw std::invalid_argument("maximum_steps must be positive");
    }
    if (!rhs) {
        throw std::invalid_argument("right-hand side must be callable");
    }
    detail::validateRungeKuttaMethod(method);

    const double duration = end_time - initial_time;
    if (!std::isfinite(duration)) {
        throw std::overflow_error("integration interval is not finite");
    }

    RungeKuttaResult result;
    result.state = initial;
    result.initial_time = initial_time;
    result.time = initial_time;
    result.nominal_dt = dt;
    result.method = method;

    // Accumulate elapsed duration near zero instead of repeatedly adding dt to
    // a potentially very large absolute clock.  The latter can silently turn
    // a requested step into a different floating-point increment (or no
    // increment at all) even though dt itself is well resolved.
    double elapsed = 0.0;
    const RHSFunction elapsed_rhs = [&rhs, initial_time](const State& state, double elapsed_time) {
        const double absolute_time = initial_time + elapsed_time;
        if (!std::isfinite(absolute_time)) {
            throw std::overflow_error("Runge--Kutta stage time became non-finite");
        }
        return rhs(state, absolute_time);
    };
    while (elapsed < duration) {
        if (result.steps >= maximum_steps) {
            throw std::runtime_error("Runge--Kutta integration exceeded maximum_steps");
        }
        const double remaining = duration - elapsed;
        const double step_dt = std::min(dt, remaining);
        result.state = method == RungeKuttaMethod::HEUN
                           ? heunStep(result.state, elapsed, step_dt, elapsed_rhs)
                           : classicalRk4Step(result.state, elapsed, step_dt, elapsed_rhs);

        if (step_dt == remaining) {
            elapsed = duration;
            result.time = end_time;
        } else {
            const double next_elapsed = elapsed + step_dt;
            if (!std::isfinite(next_elapsed) || next_elapsed <= elapsed) {
                throw std::runtime_error(
                    "time step is too small to advance the elapsed integration clock");
            }
            elapsed = next_elapsed;
            result.time = initial_time + elapsed;
            if (!std::isfinite(result.time)) {
                throw std::overflow_error("Runge--Kutta time became non-finite");
            }
        }
        result.last_dt = step_dt;
        ++result.steps;
    }
    return result;
}

inline State laplacian1D(const State& state, double dx, int order) {
    order = detail::validateSpatialOrder(order);
    detail::validateSpacing(dx, "dx");
    detail::validateState(state, "field");
    if (state.size() < detail::minimumNodes(order)) {
        throw std::invalid_argument("field is too short for the requested centered stencil");
    }

    const std::size_t n = state.size();
    const double inverse_dx2 = 1.0 / (dx * dx);
    State result(n, 0.0);

    // A second-order closure is used at interior points that cannot support
    // the requested centred stencil.  Boundary nodes remain zero because the
    // derivative is not evaluated at a prescribed Dirichlet boundary.
    result[1] = (state[2] - 2.0 * state[1] + state[0]) * inverse_dx2;
    result[n - 2] = (state[n - 1] - 2.0 * state[n - 2] + state[n - 3]) * inverse_dx2;
    if (order >= 4) {
        const double factor = inverse_dx2 / 12.0;
        for (std::size_t i = 2; i + 2 < n; ++i) {
            result[i] = (-state[i + 2] + 16.0 * state[i + 1] - 30.0 * state[i] +
                         16.0 * state[i - 1] - state[i - 2]) *
                        factor;
        }
    } else {
        for (std::size_t i = 2; i + 1 < n; ++i) {
            result[i] = (state[i + 1] - 2.0 * state[i] + state[i - 1]) * inverse_dx2;
        }
    }
    if (order == 6) {
        const double factor = inverse_dx2 / 180.0;
        for (std::size_t i = 3; i + 3 < n; ++i) {
            result[i] = (2.0 * state[i + 3] - 27.0 * state[i + 2] + 270.0 * state[i + 1] -
                         490.0 * state[i] + 270.0 * state[i - 1] - 27.0 * state[i - 2] +
                         2.0 * state[i - 3]) *
                        factor;
        }
    }
    detail::validateState(result, "Laplacian");
    return result;
}

inline State gradient1D(const State& state, double dx, int order) {
    if (order != 2 && order != 4) {
        throw std::invalid_argument("gradient order must be 2 or 4");
    }
    detail::validateSpacing(dx, "dx");
    detail::validateState(state, "field");
    const std::size_t minimum = order == 4 ? 5 : 3;
    if (state.size() < minimum) {
        throw std::invalid_argument("field is too short for the requested centered stencil");
    }

    const std::size_t n = state.size();
    State result(n, 0.0);
    const double centered_second = 1.0 / (2.0 * dx);
    result[1] = (state[2] - state[0]) * centered_second;
    result[n - 2] = (state[n - 1] - state[n - 3]) * centered_second;
    if (order == 2) {
        for (std::size_t i = 2; i + 1 < n; ++i) {
            result[i] = (state[i + 1] - state[i - 1]) * centered_second;
        }
    } else {
        const double factor = 1.0 / (12.0 * dx);
        for (std::size_t i = 2; i + 2 < n; ++i) {
            result[i] =
                (-state[i + 2] + 8.0 * state[i + 1] - 8.0 * state[i - 1] + state[i - 2]) * factor;
        }
    }
    detail::validateState(result, "gradient");
    return result;
}

inline State laplacian2D(const State& state, int nx, int ny, double dx, double dy, int order) {
    order = detail::validateSpatialOrder(order, true);
    detail::validateSpacing(dx, "dx");
    detail::validateSpacing(dy, "dy");
    if (nx < 2 || ny < 2) {
        throw std::invalid_argument("2D Laplacian requires at least two cells per direction");
    }
    if (order == 4 && (nx < 4 || ny < 4)) {
        throw std::invalid_argument(
            "fourth-order 2D Laplacian requires at least four cells per direction");
    }
    const std::size_t stride = static_cast<std::size_t>(nx + 1);
    const std::size_t expected = stride * static_cast<std::size_t>(ny + 1);
    if (state.size() != expected) {
        throw std::invalid_argument("2D field size does not match (nx + 1) * (ny + 1)");
    }
    detail::validateState(state, "field");

    const double inverse_dx2 = 1.0 / (dx * dx);
    const double inverse_dy2 = 1.0 / (dy * dy);
    State result(expected, 0.0);
    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            const std::size_t index =
                static_cast<std::size_t>(j) * stride + static_cast<std::size_t>(i);
            if (order == 4 && i >= 2 && i <= nx - 2 && j >= 2 && j <= ny - 2) {
                const double lap_x =
                    (-state[index + 2] + 16.0 * state[index + 1] - 30.0 * state[index] +
                     16.0 * state[index - 1] - state[index - 2]) *
                    (inverse_dx2 / 12.0);
                const double lap_y = (-state[index + 2 * stride] + 16.0 * state[index + stride] -
                                      30.0 * state[index] + 16.0 * state[index - stride] -
                                      state[index - 2 * stride]) *
                                     (inverse_dy2 / 12.0);
                result[index] = lap_x + lap_y;
            } else {
                result[index] =
                    (state[index + 1] - 2.0 * state[index] + state[index - 1]) * inverse_dx2 +
                    (state[index + stride] - 2.0 * state[index] + state[index - stride]) *
                        inverse_dy2;
            }
        }
    }
    detail::validateState(result, "Laplacian");
    return result;
}

inline double stableDiffusionTimeStep(double diffusivity, double dx, double dy, int order,
                                      double safety_factor, bool is_2d) {
    order = detail::validateSpatialOrder(order, is_2d);
    detail::requireFinite(diffusivity, "diffusivity");
    if (diffusivity <= 0.0) {
        throw std::invalid_argument("diffusivity must be positive");
    }
    detail::validateSpacing(dx, "dx");
    if (is_2d) {
        detail::validateSpacing(dy, "dy");
    }
    detail::requireFinite(safety_factor, "safety factor");
    if (safety_factor <= 0.0 || safety_factor > 1.0) {
        throw std::invalid_argument("safety factor must be in (0, 1]");
    }

    const double inverse_spacing_sum = 1.0 / (dx * dx) + (is_2d ? 1.0 / (dy * dy) : 0.0);
    const double dt =
        2.0 * safety_factor /
        (diffusivity * detail::centeredLaplacianSpectralRadius(order) * inverse_spacing_sum);
    if (!std::isfinite(dt) || dt <= 0.0) {
        throw std::overflow_error("stable diffusion time step is not finite and positive");
    }
    return dt;
}

struct HighOrderDiffusionResult {
    State solution;
    double time = 0.0;
    double nominal_dt = 0.0;
    double last_dt = 0.0;
    std::size_t steps = 0;
    int interior_order = 2;
    int boundary_closure_order = 2;
};

using DiffusionObserver = std::function<void(double, const State&)>;

inline void applyDirichletBoundaries(State& state, int nx, int ny, double left, double right,
                                     double bottom, double top, bool is_2d) {
    detail::requireFinite(left, "left Dirichlet boundary value");
    detail::requireFinite(right, "right Dirichlet boundary value");
    if (!is_2d) {
        state.front() = left;
        state.back() = right;
        return;
    }
    detail::requireFinite(bottom, "bottom Dirichlet boundary value");
    detail::requireFinite(top, "top Dirichlet boundary value");

    const std::size_t stride = static_cast<std::size_t>(nx + 1);
    for (int j = 0; j <= ny; ++j) {
        const std::size_t row = static_cast<std::size_t>(j) * stride;
        state[row] = left;
        state[row + static_cast<std::size_t>(nx)] = right;
    }
    // Horizontal values intentionally own the four corners.  This makes the
    // precedence deterministic when users prescribe discontinuous corner data.
    for (int i = 0; i <= nx; ++i) {
        state[static_cast<std::size_t>(i)] = bottom;
        state[static_cast<std::size_t>(ny) * stride + static_cast<std::size_t>(i)] = top;
    }
}

inline HighOrderDiffusionResult solveHighOrderDiffusion(
    const State& initial, int nx, int ny, double dx, double dy, double diffusivity, int order,
    double safety_factor, double end_time, double requested_dt, double left_boundary,
    double right_boundary, double bottom_boundary = 0.0, double top_boundary = 0.0,
    const DiffusionObserver& observer = {}) {
    if (ny < 0) {
        throw std::invalid_argument("ny must be nonnegative");
    }
    const bool is_2d = ny > 0;
    order = detail::validateSpatialOrder(order, is_2d);
    detail::requireFinite(end_time, "end time");
    if (end_time < 0.0) {
        throw std::invalid_argument("end time must be nonnegative");
    }
    detail::requireFinite(requested_dt, "requested time step");
    if (requested_dt < 0.0) {
        throw std::invalid_argument("requested time step must be nonnegative");
    }
    if (nx < order) {
        throw std::invalid_argument("mesh is too small for the requested spatial order");
    }
    if (is_2d && ny < order) {
        throw std::invalid_argument("mesh is too small for the requested spatial order");
    }

    const std::size_t expected =
        static_cast<std::size_t>(nx + 1) * static_cast<std::size_t>(is_2d ? ny + 1 : 1);
    if (initial.size() != expected) {
        throw std::invalid_argument("initial field size does not match the mesh");
    }
    detail::validateState(initial, "initial field");

    const double stable_dt =
        stableDiffusionTimeStep(diffusivity, dx, dy, order, safety_factor, is_2d);
    const double nominal_dt = requested_dt == 0.0 ? stable_dt : requested_dt;
    const double allowance =
        64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, stable_dt);
    if (nominal_dt > stable_dt + allowance) {
        throw std::domain_error(
            "requested time step exceeds the safety-scaled explicit "
            "diffusion stability bound");
    }

    HighOrderDiffusionResult result;
    result.solution = initial;
    result.nominal_dt = nominal_dt;
    result.interior_order = order;
    applyDirichletBoundaries(result.solution, nx, ny, left_boundary, right_boundary,
                             bottom_boundary, top_boundary, is_2d);

    constexpr std::size_t maximum_steps = 10000000;
    while (result.time < end_time) {
        if (result.steps >= maximum_steps) {
            throw std::runtime_error("diffusion solve exceeded ten million time steps");
        }
        const double remaining = end_time - result.time;
        const double step_dt = std::min(nominal_dt, remaining);
        const State laplacian = is_2d ? laplacian2D(result.solution, nx, ny, dx, dy, order)
                                      : laplacian1D(result.solution, dx, order);
        for (std::size_t i = 0; i < result.solution.size(); ++i) {
            result.solution[i] += diffusivity * step_dt * laplacian[i];
        }
        detail::validateState(result.solution, "diffusion solution");
        applyDirichletBoundaries(result.solution, nx, ny, left_boundary, right_boundary,
                                 bottom_boundary, top_boundary, is_2d);

        if (step_dt == remaining) {
            result.time = end_time;
        } else {
            const double next_time = result.time + step_dt;
            if (!std::isfinite(next_time) || next_time <= result.time) {
                throw std::runtime_error("time step is too small to advance the diffusion clock");
            }
            result.time = next_time;
        }
        result.last_dt = step_dt;
        ++result.steps;
        if (observer) {
            observer(result.time, result.solution);
        }
    }
    return result;
}

}  // namespace time_integration
}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_NUMERICS_TIME_INTEGRATION_HIGH_ORDER_HPP
