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
#include <array>
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
        result[i] = std::fma(scale, increment[i], base[i]);
    }
    validateState(result, stage);
    return result;
}

template <std::size_t N>
inline double compensatedWeightedUpdate(double base, double outer_scale,
                                        const std::array<double, N>& values,
                                        const std::array<double, N>& integer_weights,
                                        double divisor) {
    int common_exponent = std::numeric_limits<int>::min();
    if (base != 0.0) {
        (void)std::frexp(base, &common_exponent);
    }
    std::array<int, N> product_exponents{};
    std::array<double, N> product_mantissas{};
    std::array<double, N> first_product_errors{};
    std::array<double, N> second_product_errors{};
    for (std::size_t index = 0; index < N; ++index) {
        if (values[index] == 0.0 || integer_weights[index] == 0.0 || outer_scale == 0.0) {
            continue;
        }
        const double weight = integer_weights[index] / divisor;
        int value_exponent = 0;
        int weight_exponent = 0;
        int scale_exponent = 0;
        const double value_mantissa = std::frexp(values[index], &value_exponent);
        const double weight_mantissa = std::frexp(weight, &weight_exponent);
        const double scale_mantissa = std::frexp(outer_scale, &scale_exponent);
        const double first_product = scale_mantissa * weight_mantissa;
        const double first_error = std::fma(scale_mantissa, weight_mantissa, -first_product);
        const double product = first_product * value_mantissa;
        product_mantissas[index] = product;
        first_product_errors[index] = first_error * value_mantissa;
        second_product_errors[index] = std::fma(first_product, value_mantissa, -product);
        product_exponents[index] = value_exponent + weight_exponent + scale_exponent;
        common_exponent = std::max(common_exponent, product_exponents[index]);
    }
    if (common_exponent == std::numeric_limits<int>::min()) {
        return 0.0;
    }

    std::array<double, 1 + 3 * N> terms{};
    terms[0] = std::scalbn(base, -common_exponent);
    if (terms[0] == 0.0 && base != 0.0) {
        throw std::overflow_error(
            "weighted update dynamic range cannot preserve every nonzero term");
    }
    for (std::size_t index = 0; index < N; ++index) {
        if (values[index] == 0.0 || integer_weights[index] == 0.0 || outer_scale == 0.0) {
            continue;
        }
        const int exponent_delta = product_exponents[index] - common_exponent;
        terms[1 + 3 * index] = std::scalbn(product_mantissas[index], exponent_delta);
        terms[1 + 3 * index + 1] = std::scalbn(first_product_errors[index], exponent_delta);
        terms[1 + 3 * index + 2] = std::scalbn(second_product_errors[index], exponent_delta);
        if (terms[1 + 3 * index] == 0.0) {
            throw std::overflow_error(
                "weighted update dynamic range cannot preserve every nonzero term");
        }
    }
    std::sort(terms.begin(), terms.end(),
              [](double left, double right) { return std::abs(left) > std::abs(right); });
    double sum = 0.0;
    double correction = 0.0;
    for (double term : terms) {
        const double next_sum = sum + term;
        correction +=
            std::abs(sum) >= std::abs(term) ? (sum - next_sum) + term : (term - next_sum) + sum;
        sum = next_sum;
    }
    return std::scalbn(sum + correction, common_exponent);
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

template <std::size_t N>
inline double scaledStencilDerivative(const std::array<double, N>& values,
                                      const std::array<double, N>& coefficients, double spacing,
                                      double divisor, int spacing_power) {
    if (std::all_of(values.begin() + 1, values.end(),
                    [&](double value) { return value == values.front(); })) {
        return 0.0;
    }

    // Choose only as much power-of-two scaling as is needed to keep every
    // weighted term away from overflow, or to lift an entirely tiny stencil
    // out of the subnormal range.  Scaling every value by the largest sample
    // would erase a moderate signal when much larger samples cancel.
    int maximum_product_exponent = std::numeric_limits<int>::min();
    bool requires_downscaling = false;
    double positive_product_sum = 0.0;
    double negative_product_sum = 0.0;
    for (std::size_t index = 0; index < N; ++index) {
        if (values[index] == 0.0 || coefficients[index] == 0.0) {
            continue;
        }
        const double unscaled_product = values[index] * coefficients[index];
        if (!std::isfinite(unscaled_product)) {
            requires_downscaling = true;
        } else {
            double& same_sign_sum =
                unscaled_product > 0.0 ? positive_product_sum : negative_product_sum;
            const double magnitude = std::abs(unscaled_product);
            if (magnitude > std::numeric_limits<double>::max() - same_sign_sum) {
                requires_downscaling = true;
            } else {
                same_sign_sum += magnitude;
            }
        }
        int value_exponent = 0;
        int coefficient_exponent = 0;
        int product_exponent = 0;
        const double value_mantissa = std::frexp(values[index], &value_exponent);
        const double coefficient_mantissa = std::frexp(coefficients[index], &coefficient_exponent);
        (void)std::frexp(value_mantissa * coefficient_mantissa, &product_exponent);
        maximum_product_exponent = std::max(
            maximum_product_exponent, value_exponent + coefficient_exponent + product_exponent);
    }
    if (maximum_product_exponent == std::numeric_limits<int>::min()) {
        return 0.0;
    }

    constexpr int kSafeProductExponent = std::numeric_limits<double>::max_exponent - 32;
    int scale_exponent = 0;
    if (requires_downscaling) {
        scale_exponent = maximum_product_exponent - kSafeProductExponent;
    } else if (maximum_product_exponent < 0) {
        const int proposed_exponent = maximum_product_exponent;
        const bool raw_values_remain_finite = std::all_of(
            values.begin(), values.end(),
            [&](double value) { return std::isfinite(std::scalbn(value, -proposed_exponent)); });
        if (raw_values_remain_finite) {
            scale_exponent = proposed_exponent;
        }
    }

    // Sum the rounded product and its FMA residual from largest magnitude to
    // smallest with Neumaier compensation.  This preserves finite moderate
    // terms when enormous positive and negative stencil contributions cancel.
    std::array<double, 2 * N> normalized_terms{};
    for (std::size_t index = 0; index < N; ++index) {
        const double normalized_value = std::scalbn(values[index], -scale_exponent);
        if (normalized_value == 0.0 && values[index] != 0.0 && coefficients[index] != 0.0) {
            throw std::overflow_error(
                "stencil dynamic range cannot preserve every nonzero weighted term");
        }
        const double product = coefficients[index] * normalized_value;
        if (scale_exponent > 0 && product == 0.0 && normalized_value != 0.0 &&
            coefficients[index] != 0.0) {
            throw std::overflow_error(
                "stencil dynamic range cannot preserve every nonzero weighted term");
        }
        normalized_terms[2 * index] = product;
        normalized_terms[2 * index + 1] = std::fma(coefficients[index], normalized_value, -product);
    }
    std::sort(normalized_terms.begin(), normalized_terms.end(),
              [](double left, double right) { return std::abs(left) > std::abs(right); });
    double normalized_sum = 0.0;
    double correction = 0.0;
    for (double term : normalized_terms) {
        const double next_sum = normalized_sum + term;
        if (std::abs(normalized_sum) >= std::abs(term)) {
            correction += (normalized_sum - next_sum) + term;
        } else {
            correction += (term - next_sum) + normalized_sum;
        }
        normalized_sum = next_sum;
    }
    normalized_sum += correction;
    if (normalized_sum == 0.0) {
        return 0.0;
    }
    int sum_exponent = 0;
    int spacing_exponent = 0;
    int divisor_exponent = 0;
    const double sum_mantissa = std::frexp(normalized_sum, &sum_exponent);
    const double spacing_mantissa = std::frexp(spacing, &spacing_exponent);
    const double divisor_mantissa = std::frexp(divisor, &divisor_exponent);
    double denominator_mantissa = divisor_mantissa;
    for (int power = 0; power < spacing_power; ++power) {
        denominator_mantissa *= spacing_mantissa;
    }
    const int result_exponent =
        sum_exponent + scale_exponent - divisor_exponent - spacing_power * spacing_exponent;
    return std::scalbn(sum_mantissa / denominator_mantissa, result_exponent);
}

constexpr std::array<double, 3> kSecondDerivative2{1.0, -2.0, 1.0};
constexpr std::array<double, 5> kSecondDerivative4{-1.0, 16.0, -30.0, 16.0, -1.0};
constexpr std::array<double, 7> kSecondDerivative6{2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0};
constexpr std::array<double, 2> kFirstDerivative2{-1.0, 1.0};
constexpr std::array<double, 4> kFirstDerivative4{1.0, -8.0, 8.0, -1.0};

inline double scaledTripleProduct(double first, double second, double third) {
    if (first == 0.0 || second == 0.0 || third == 0.0) {
        return 0.0;
    }
    int first_exponent = 0;
    int second_exponent = 0;
    int third_exponent = 0;
    const double mantissa = std::frexp(first, &first_exponent) *
                            std::frexp(second, &second_exponent) *
                            std::frexp(third, &third_exponent);
    return std::scalbn(mantissa, first_exponent + second_exponent + third_exponent);
}

inline double scaledTripleAdd(double base, double first, double second, double third) {
    if (first == 0.0 || second == 0.0 || third == 0.0) {
        return base;
    }
    int base_exponent = 0;
    int first_exponent = 0;
    int second_exponent = 0;
    int third_exponent = 0;
    const double base_mantissa = std::frexp(base, &base_exponent);
    const double product_mantissa = std::frexp(first, &first_exponent) *
                                    std::frexp(second, &second_exponent) *
                                    std::frexp(third, &third_exponent);
    const int product_exponent = first_exponent + second_exponent + third_exponent;
    const int common_exponent = std::max(base_exponent, product_exponent);
    const double scaled_base = std::scalbn(base_mantissa, base_exponent - common_exponent);
    const double scaled_product = std::scalbn(product_mantissa, product_exponent - common_exponent);
    if ((scaled_base == 0.0 && base != 0.0) || (scaled_product == 0.0 && product_mantissa != 0.0)) {
        throw std::overflow_error(
            "diffusion update dynamic range cannot preserve every nonzero term");
    }
    return std::scalbn(scaled_base + scaled_product, common_exponent);
}

inline double scaledDiffusionTimeStep(double diffusivity, double spacing, double safety_factor,
                                      double denominator) {
    int safety_exponent = 0;
    int spacing_exponent = 0;
    int diffusivity_exponent = 0;
    int denominator_exponent = 0;
    const double safety_mantissa = std::frexp(safety_factor, &safety_exponent);
    const double spacing_mantissa = std::frexp(spacing, &spacing_exponent);
    const double diffusivity_mantissa = std::frexp(diffusivity, &diffusivity_exponent);
    const double denominator_mantissa = std::frexp(denominator, &denominator_exponent);
    const auto lower_product = [](double left, double right) {
        const double product = left * right;
        const double error = std::fma(left, right, -product);
        return error < 0.0 ? std::nextafter(product, 0.0) : product;
    };
    const auto upper_product = [](double left, double right) {
        const double product = left * right;
        const double error = std::fma(left, right, -product);
        return error > 0.0 ? std::nextafter(product, std::numeric_limits<double>::infinity())
                           : product;
    };

    // Construct a lower bound on the positive stability limit.  Returning a
    // nearest-rounded minimum subnormal when the exact limit is smaller would
    // make an explicit diffusion step unstable, so every numerator operation
    // is rounded down and every denominator operation is rounded up.
    double lower_numerator = lower_product(safety_mantissa, spacing_mantissa);
    lower_numerator = lower_product(lower_numerator, spacing_mantissa);
    const double upper_denominator = upper_product(diffusivity_mantissa, denominator_mantissa);
    double lower_mantissa = lower_numerator / upper_denominator;
    if (std::fma(-lower_mantissa, upper_denominator, lower_numerator) < 0.0) {
        lower_mantissa = std::nextafter(lower_mantissa, 0.0);
    }
    const int result_exponent =
        1 + safety_exponent + 2 * spacing_exponent - diffusivity_exponent - denominator_exponent;
    double result = std::scalbn(lower_mantissa, result_exponent);
    if (std::fpclassify(result) == FP_SUBNORMAL &&
        std::scalbn(result, -result_exponent) > lower_mantissa) {
        result = std::nextafter(result, 0.0);
    }
    return result;
}

inline std::size_t minimumNodes(int order) {
    return static_cast<std::size_t>(order + 1);
}

inline double centeredLaplacianSpectralRadius(int order) {
    switch (order) {
        case 2:
            return 4.0;
        case 4:
            return std::nextafter(16.0 / 3.0, std::numeric_limits<double>::infinity());
        case 6:
            return std::nextafter(272.0 / 45.0, std::numeric_limits<double>::infinity());
        default:
            throw std::logic_error("unreachable spatial order");
    }
}

struct ScheduledStep {
    double start = 0.0;
    double end = 0.0;
    double dt = 0.0;
};

inline double canonicalBoundary(double duration, double nominal_dt, std::size_t boundary,
                                std::size_t step_count) {
    if (boundary == 0) {
        return 0.0;
    }
    if (boundary == step_count) {
        return duration;
    }
    const long double exact_boundary =
        static_cast<long double>(boundary) * static_cast<long double>(nominal_dt);
    double rounded_boundary = static_cast<double>(exact_boundary);
    if (rounded_boundary >= duration) {
        rounded_boundary = std::nextafter(duration, 0.0);
    }
    return rounded_boundary;
}

inline ScheduledStep scheduledStep(double duration, double nominal_dt, std::size_t step,
                                   std::size_t step_count) {
    const double start = canonicalBoundary(duration, nominal_dt, step, step_count);
    const double end = canonicalBoundary(duration, nominal_dt, step + 1, step_count);
    const double step_dt = step + 1 == step_count
                               ? static_cast<double>(std::fma(-static_cast<long double>(step),
                                                              static_cast<long double>(nominal_dt),
                                                              static_cast<long double>(duration)))
                               : nominal_dt;
    return {start, end, step_dt};
}

inline std::size_t boundedStepCount(double duration, double nominal_dt, std::size_t maximum_steps,
                                    const char* limit_message) {
    if (duration == 0.0) {
        return 0;
    }
    const long double ratio =
        static_cast<long double>(duration) / static_cast<long double>(nominal_dt);
    const long double approximate_steps = std::max(1.0L, std::ceil(ratio));
    if (!std::isfinite(approximate_steps) ||
        approximate_steps > static_cast<long double>(maximum_steps)) {
        throw std::runtime_error(limit_message);
    }
    std::size_t required_steps = static_cast<std::size_t>(approximate_steps);
    const long double duration_exact = static_cast<long double>(duration);
    const long double dt_exact = static_cast<long double>(nominal_dt);
    const auto final_remainder = [&]() {
        return std::fma(-static_cast<long double>(required_steps - 1), dt_exact, duration_exact);
    };
    while (required_steps > 1 && final_remainder() <= 0.0L) {
        --required_steps;
    }
    while (final_remainder() > dt_exact) {
        if (required_steps >= maximum_steps) {
            throw std::runtime_error(limit_message);
        }
        ++required_steps;
    }

    return required_steps;
}

inline bool absoluteStageClockRepresentable(double initial_time, double start, double end,
                                            double step_dt, RungeKuttaMethod method) {
    const double absolute_start = initial_time + start;
    const double absolute_end = initial_time + end;
    if (!std::isfinite(absolute_start) || !std::isfinite(absolute_end) ||
        absolute_end <= absolute_start) {
        return false;
    }
    if (method == RungeKuttaMethod::CLASSICAL_RK4) {
        const double absolute_half = initial_time + (start + 0.5 * step_dt);
        if (!std::isfinite(absolute_half) || absolute_half <= absolute_start ||
            absolute_end <= absolute_half) {
            return false;
        }
    }
    return true;
}

inline void validateAbsoluteStageClock(double initial_time, double duration, double nominal_dt,
                                       RungeKuttaMethod method, std::size_t step_count) {
    if (step_count == 0) {
        return;
    }
    for (std::size_t step = 0; step < step_count; ++step) {
        const auto schedule = scheduledStep(duration, nominal_dt, step, step_count);
        if (!(schedule.dt > 0.0) || schedule.dt > nominal_dt) {
            throw std::invalid_argument(
                "Runge--Kutta interval cannot be partitioned into positive steps at or below dt");
        }
        if (!absoluteStageClockRepresentable(initial_time, schedule.start, schedule.end,
                                             schedule.dt, method)) {
            throw std::invalid_argument(
                "absolute Runge--Kutta stage times are not representable at this clock scale; "
                "shift the time origin or use an autonomous formulation");
        }
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
        result[i] =
            detail::compensatedWeightedUpdate(state[i], dt, std::array<double, 2>{k1[i], k2[i]},
                                              std::array<double, 2>{1.0, 1.0}, 2.0);
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
        result[i] = detail::compensatedWeightedUpdate(
            state[i], dt, std::array<double, 4>{k1[i], k2[i], k3[i], k4[i]},
            std::array<double, 4>{1.0, 2.0, 2.0, 1.0}, 6.0);
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

    // Canonical subdivision boundaries keep callback clocks contiguous while
    // every applied step remains at or below the caller's requested ceiling.
    const std::size_t planned_steps = detail::boundedStepCount(
        duration, dt, maximum_steps, "Runge--Kutta integration exceeded maximum_steps");
    detail::validateAbsoluteStageClock(initial_time, duration, dt, method, planned_steps);
    for (std::size_t planned_step = 0; planned_step < planned_steps; ++planned_step) {
        const auto schedule = detail::scheduledStep(duration, dt, planned_step, planned_steps);
        const double step_dt = schedule.dt;
        if (!(step_dt > 0.0) || !std::isfinite(step_dt)) {
            throw std::runtime_error("scheduled Runge--Kutta step is not positive and finite");
        }
        const double absolute_start = initial_time + schedule.start;
        const double absolute_end =
            planned_step + 1 == planned_steps ? end_time : initial_time + schedule.end;
        const double absolute_half = initial_time + (schedule.start + 0.5 * step_dt);
        const RHSFunction canonical_rhs = [&rhs, absolute_start, absolute_half, absolute_end,
                                           step_dt](const State& state, double local_time) {
            if (local_time <= 0.0) {
                return rhs(state, absolute_start);
            }
            if (local_time >= step_dt) {
                return rhs(state, absolute_end);
            }
            return rhs(state, absolute_half);
        };
        result.state = method == RungeKuttaMethod::HEUN
                           ? heunStep(result.state, 0.0, step_dt, canonical_rhs)
                           : classicalRk4Step(result.state, 0.0, step_dt, canonical_rhs);
        result.time = absolute_end;
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
    State result(n, 0.0);

    // A second-order closure is used at interior points that cannot support
    // the requested centred stencil.  Boundary nodes remain zero because the
    // derivative is not evaluated at a prescribed Dirichlet boundary.
    result[1] = detail::scaledStencilDerivative(std::array<double, 3>{state[0], state[1], state[2]},
                                                detail::kSecondDerivative2, dx, 1.0, 2);
    result[n - 2] = detail::scaledStencilDerivative(
        std::array<double, 3>{state[n - 3], state[n - 2], state[n - 1]}, detail::kSecondDerivative2,
        dx, 1.0, 2);
    if (order >= 4) {
        for (std::size_t i = 2; i + 2 < n; ++i) {
            result[i] = detail::scaledStencilDerivative(
                std::array<double, 5>{state[i - 2], state[i - 1], state[i], state[i + 1],
                                      state[i + 2]},
                detail::kSecondDerivative4, dx, 12.0, 2);
        }
    } else {
        for (std::size_t i = 2; i + 1 < n; ++i) {
            result[i] = detail::scaledStencilDerivative(
                std::array<double, 3>{state[i - 1], state[i], state[i + 1]},
                detail::kSecondDerivative2, dx, 1.0, 2);
        }
    }
    if (order == 6) {
        for (std::size_t i = 3; i + 3 < n; ++i) {
            result[i] = detail::scaledStencilDerivative(
                std::array<double, 7>{state[i - 3], state[i - 2], state[i - 1], state[i],
                                      state[i + 1], state[i + 2], state[i + 3]},
                detail::kSecondDerivative6, dx, 180.0, 2);
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
    result[1] = detail::scaledStencilDerivative(std::array<double, 2>{state[0], state[2]},
                                                detail::kFirstDerivative2, dx, 2.0, 1);
    result[n - 2] = detail::scaledStencilDerivative(
        std::array<double, 2>{state[n - 3], state[n - 1]}, detail::kFirstDerivative2, dx, 2.0, 1);
    if (order == 2) {
        for (std::size_t i = 2; i + 1 < n; ++i) {
            result[i] =
                detail::scaledStencilDerivative(std::array<double, 2>{state[i - 1], state[i + 1]},
                                                detail::kFirstDerivative2, dx, 2.0, 1);
        }
    } else {
        for (std::size_t i = 2; i + 2 < n; ++i) {
            result[i] = detail::scaledStencilDerivative(
                std::array<double, 4>{state[i - 2], state[i - 1], state[i + 1], state[i + 2]},
                detail::kFirstDerivative4, dx, 12.0, 1);
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

    const double reference_spacing = std::min(dx, dy);
    const double x_ratio = reference_spacing / dx;
    const double y_ratio = reference_spacing / dy;
    const double x_weight = x_ratio * x_ratio;
    const double y_weight = y_ratio * y_ratio;
    if ((x_ratio != 0.0 && x_weight == 0.0) || (y_ratio != 0.0 && y_weight == 0.0) ||
        x_ratio == 0.0 || y_ratio == 0.0) {
        throw std::overflow_error(
            "mesh anisotropy is too large to represent every directional Laplacian weight");
    }
    State result(expected, 0.0);
    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            const std::size_t index =
                static_cast<std::size_t>(j) * stride + static_cast<std::size_t>(i);
            if (order == 4 && i >= 2 && i <= nx - 2 && j >= 2 && j <= ny - 2) {
                const std::array<double, 9> values{
                    state[index - 2],      state[index - 1],      state[index],
                    state[index + 1],      state[index + 2],      state[index - 2 * stride],
                    state[index - stride], state[index + stride], state[index + 2 * stride],
                };
                const std::array<double, 9> coefficients{
                    -x_weight,       16.0 * x_weight, -30.0 * (x_weight + y_weight),
                    16.0 * x_weight, -x_weight,       -y_weight,
                    16.0 * y_weight, 16.0 * y_weight, -y_weight,
                };
                result[index] = detail::scaledStencilDerivative(values, coefficients,
                                                                reference_spacing, 12.0, 2);
            } else {
                const std::array<double, 5> values{
                    state[index - 1],      state[index + 1],      state[index],
                    state[index - stride], state[index + stride],
                };
                const std::array<double, 5> coefficients{
                    x_weight, x_weight, -2.0 * (x_weight + y_weight), y_weight, y_weight,
                };
                result[index] = detail::scaledStencilDerivative(values, coefficients,
                                                                reference_spacing, 1.0, 2);
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

    const auto upper_quotient = [](double numerator, double divisor) {
        double quotient = numerator / divisor;
        if (std::fma(-quotient, divisor, numerator) > 0.0) {
            quotient = std::nextafter(quotient, std::numeric_limits<double>::infinity());
        }
        return quotient;
    };
    const auto upper_product = [](double left, double right) {
        double product = left * right;
        if (product == 0.0 && left != 0.0 && right != 0.0) {
            return std::numeric_limits<double>::denorm_min();
        }
        if (std::fma(left, right, -product) > 0.0) {
            product = std::nextafter(product, std::numeric_limits<double>::infinity());
        }
        return product;
    };
    const auto upper_sum = [](double left, double right) {
        double sum = left + right;
        const double left_virtual = sum - right;
        const double error = (left - left_virtual) + (right - (sum - left_virtual));
        if (error > 0.0) {
            sum = std::nextafter(sum, std::numeric_limits<double>::infinity());
        }
        return sum;
    };

    const double reference_spacing = is_2d ? std::min(dx, dy) : dx;
    const double x_ratio = upper_quotient(reference_spacing, dx);
    const double y_ratio = is_2d ? upper_quotient(reference_spacing, dy) : 0.0;
    const double x_weight = upper_product(x_ratio, x_ratio);
    const double y_weight = is_2d ? upper_product(y_ratio, y_ratio) : 0.0;
    const double dimensionless_inverse_spacing = is_2d ? upper_sum(x_weight, y_weight) : x_weight;
    const double denominator = upper_product(detail::centeredLaplacianSpectralRadius(order),
                                             dimensionless_inverse_spacing);
    const double dt =
        detail::scaledDiffusionTimeStep(diffusivity, reference_spacing, safety_factor, denominator);
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
    // Both paths use the same native bound calculation, so no dimensional
    // tolerance is needed.  Even a one-ULP allowance can double a minimum
    // subnormal stability limit.
    if (nominal_dt > stable_dt) {
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
    const std::size_t planned_steps = detail::boundedStepCount(
        end_time, nominal_dt, maximum_steps, "diffusion solve exceeded ten million time steps");
    for (std::size_t planned_step = 0; planned_step < planned_steps; ++planned_step) {
        const auto schedule =
            detail::scheduledStep(end_time, nominal_dt, planned_step, planned_steps);
        const double step_dt = schedule.dt;
        if (!(step_dt > 0.0) || !std::isfinite(step_dt)) {
            throw std::runtime_error("scheduled diffusion step is not positive and finite");
        }
        const State laplacian = is_2d ? laplacian2D(result.solution, nx, ny, dx, dy, order)
                                      : laplacian1D(result.solution, dx, order);
        for (std::size_t i = 0; i < result.solution.size(); ++i) {
            result.solution[i] =
                detail::scaledTripleAdd(result.solution[i], diffusivity, step_dt, laplacian[i]);
        }
        detail::validateState(result.solution, "diffusion solution");
        applyDirichletBoundaries(result.solution, nx, ny, left_boundary, right_boundary,
                                 bottom_boundary, top_boundary, is_2d);

        result.time = planned_step + 1 == planned_steps ? end_time : schedule.end;
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
