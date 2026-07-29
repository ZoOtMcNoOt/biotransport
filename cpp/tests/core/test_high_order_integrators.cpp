/**
 * @file test_high_order_integrators.cpp
 * @brief Independent accuracy and contract tests for validated numerical kernels.
 */

#include <biotransport/core/numerics/time_integration/high_order.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ti = biotransport::time_integration;

namespace {

bool approximatelyEqual(double actual, double expected, double tolerance) {
    return std::abs(actual - expected) <= tolerance;
}

bool relativelyEqual(double actual, double expected, double tolerance = 2.0e-12) {
    return std::isfinite(actual) && std::abs(actual / expected - 1.0) <= tolerance;
}

template <typename Exception, typename Callable>
bool throws(Callable&& callable) {
    try {
        callable();
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void check(bool condition, const std::string& message, int& failures) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

double exponentialError(ti::RungeKuttaMethod method, double dt) {
    const auto result = ti::integrateRungeKutta(
        {1.0}, 0.0, 1.0, dt, [](const ti::State& state, double) { return ti::State{state[0]}; },
        method);
    return std::abs(result.state[0] - std::exp(1.0));
}

}  // namespace

int main() {
    int failures = 0;

    // Heun integrates y' = t exactly and must evaluate the nonautonomous RHS
    // at the beginning and end of each accepted step.
    {
        std::vector<double> times;
        const auto result = ti::integrateRungeKutta(
            {2.0}, 0.2, 1.0, 0.3,
            [&times](const ti::State&, double time) {
                times.push_back(time);
                return ti::State{time};
            },
            ti::RungeKuttaMethod::HEUN);
        check(approximatelyEqual(result.state[0], 2.48, 2e-15),
              "Heun failed the nonautonomous y'=t problem", failures);
        check(result.time == 1.0 && result.steps == 3,
              "Heun did not end exactly at the requested time", failures);
        check(approximatelyEqual(result.last_dt, 0.2, 2e-15),
              "Heun did not report the shortened final step", failures);
        const std::vector<double> expected_times{0.2, 0.5, 0.5, 0.8, 0.8, 1.0};
        check(times.size() == expected_times.size(),
              "Heun evaluated an unexpected number of stages", failures);
        if (times.size() == expected_times.size()) {
            for (std::size_t i = 0; i < times.size(); ++i) {
                check(approximatelyEqual(times[i], expected_times[i], 2e-15),
                      "Heun used an incorrect stage time", failures);
            }
        }
    }
    {
        constexpr double duration = 0.7000000000000001;
        std::vector<double> times;
        const auto result = ti::integrateRungeKutta(
            {0.0}, 0.0, duration, 0.1,
            [&times](const ti::State&, double time) {
                times.push_back(time);
                return ti::State{time};
            },
            ti::RungeKuttaMethod::HEUN);
        check(result.steps == 8 && result.time == duration,
              "Heun did not retain the exact-float endpoint residue", failures);
        check(relativelyEqual(result.state[0], 0.5 * duration * duration),
              "Heun clock discontinuity changed the y'=t solution", failures);
        for (std::size_t step = 0; step + 1 < result.steps; ++step) {
            check(times[2 * step + 1] == times[2 * (step + 1)],
                  "Heun endpoint and following start clocks were not contiguous", failures);
        }
    }

    // Independent global order checks for y' = y.
    {
        const double coarse = exponentialError(ti::RungeKuttaMethod::HEUN, 0.1);
        const double fine = exponentialError(ti::RungeKuttaMethod::HEUN, 0.05);
        check(coarse / fine > 3.5, "Heun did not demonstrate second-order convergence", failures);
    }
    {
        const double coarse = exponentialError(ti::RungeKuttaMethod::CLASSICAL_RK4, 0.1);
        const double fine = exponentialError(ti::RungeKuttaMethod::CLASSICAL_RK4, 0.05);
        check(coarse / fine > 12.0, "classical RK4 did not demonstrate fourth-order convergence",
              failures);
    }
    {
        const auto result = ti::integrateRungeKutta(
            {1.0}, 0.0, 1.0, 0.1, [](const ti::State&, double) { return ti::State{0.0}; },
            ti::RungeKuttaMethod::CLASSICAL_RK4, 10);
        check(result.time == 1.0 && result.steps == 10,
              "floating-point endpoint residue created an eleventh Runge--Kutta step", failures);
        check(result.last_dt > 0.0 && result.last_dt <= 0.1,
              "Runge--Kutta endpoint handling exceeded the requested step ceiling", failures);
    }
    {
        const double step = std::numeric_limits<double>::denorm_min();
        const double duration = 2.0 * step;
        const auto result = ti::integrateRungeKutta(
            {1.0}, 0.0, duration, step, [](const ti::State&, double) { return ti::State{0.0}; },
            ti::RungeKuttaMethod::HEUN, 2);
        check(result.time == duration && result.steps == 2 && result.last_dt == step,
              "Runge--Kutta endpoint handling skipped a representable subnormal step", failures);
    }
    {
        constexpr double step = 0.09375291778124752;
        constexpr double duration = 0.9375291778124752;
        check(throws<std::runtime_error>([step, duration] {
                  (void)ti::integrateRungeKutta(
                      {1.0}, 0.0, duration, step,
                      [](const ti::State&, double) { return ti::State{0.0}; },
                      ti::RungeKuttaMethod::HEUN, 10);
              }),
              "Runge--Kutta rounded an exact-float step ratio down to the step budget", failures);
    }
    {
        constexpr double step = 0.09375291778124752;
        constexpr double duration = 0.28125875334374256;
        check(throws<std::runtime_error>([step, duration] {
                  (void)ti::integrateRungeKutta(
                      {1.0}, 0.0, duration, step,
                      [](const ti::State&, double) { return ti::State{0.0}; },
                      ti::RungeKuttaMethod::HEUN, 3);
              }),
              "Runge--Kutta accepted a rounded product below the exact-float duration", failures);
        const auto result = ti::integrateRungeKutta(
            {1.0}, 0.0, duration, step, [](const ti::State&, double) { return ti::State{0.0}; },
            ti::RungeKuttaMethod::HEUN, 4);
        check(result.time == duration && result.steps == 4 && result.last_dt > 0.0 &&
                  result.last_dt <= step,
              "Runge--Kutta did not schedule the exact-float endpoint residue", failures);
    }

    // A large absolute start time must not trigger tolerance-based early
    // completion or distort the state interval by accumulating on that clock.
    {
        constexpr double initial_time = 1.0e12;
        const double end_time = initial_time + 0.01;
        const double duration = end_time - initial_time;
        int callback_calls = 0;
        check(throws<std::invalid_argument>([&] {
                  (void)ti::integrateRungeKutta(
                      {3.0}, initial_time, end_time, duration / 10.0,
                      [&](const ti::State&, double) {
                          ++callback_calls;
                          return ti::State{1.0};
                      },
                      ti::RungeKuttaMethod::HEUN);
              }),
              "large absolute clock accepted an unrepresentable exact residue", failures);
        check(callback_calls == 0, "large-clock stage representability was not preflighted",
              failures);

        const auto shifted_result = ti::integrateRungeKutta(
            {3.0}, 0.0, duration, duration / 10.0,
            [](const ti::State&, double) { return ti::State{1.0}; }, ti::RungeKuttaMethod::HEUN);
        check(shifted_result.time == duration && shifted_result.steps == 11,
              "shifted-origin integration did not retain the exact residue", failures);
        check(approximatelyEqual(shifted_result.state[0], 3.0 + duration, 5e-15),
              "shifted-origin integration distorted the duration", failures);
    }
    {
        constexpr double initial_time = 1.0e16;
        const double end_time =
            std::nextafter(initial_time, std::numeric_limits<double>::infinity());
        int callback_calls = 0;
        check(throws<std::invalid_argument>([&] {
                  (void)ti::integrateRungeKutta(
                      {0.0}, initial_time, end_time, 0.5,
                      [&](const ti::State&, double time) {
                          ++callback_calls;
                          return ti::State{time - initial_time};
                      },
                      ti::RungeKuttaMethod::HEUN);
              }),
              "nonautonomous RK accepted collapsed absolute stage times", failures);
        check(callback_calls == 0,
              "absolute-stage representability was not checked before RK callbacks", failures);
    }

    // The caller's initial storage and accepted state cannot alias stage storage.
    {
        const ti::State initial{1.0, 2.0};
        const auto result = ti::integrateRungeKutta(
            initial, 0.0, 0.5, 0.2, [](const ti::State&, double) { return ti::State{0.0, 0.0}; });
        check(initial == ti::State({1.0, 2.0}), "integration mutated the initial state", failures);
        check(result.state == initial, "zero RHS did not preserve the state", failures);
        check(result.time == 0.5 && result.steps == 3,
              "RK4 did not shorten its final step to the end time", failures);
        check(approximatelyEqual(result.last_dt, 0.1, 2e-15), "RK4 reported the wrong final step",
              failures);
    }

    // Dimension, finiteness, method, and clock failures must be explicit.
    check(throws<std::invalid_argument>([] {
              ti::integrateRungeKutta({1.0}, 0.0, 1.0, 0.1,
                                      [](const ti::State&, double) { return ti::State{1.0, 2.0}; });
          }),
          "RK integration accepted a derivative with the wrong dimension", failures);
    check(throws<std::invalid_argument>([] { ti::integrateRungeKutta({1.0}, 0.0, 1.0, 0.1, {}); }),
          "RK integration accepted an empty right-hand side", failures);
    check(throws<std::domain_error>([] {
              ti::integrateRungeKutta({1.0}, 0.0, 1.0, 0.1, [](const ti::State&, double) {
                  return ti::State{std::numeric_limits<double>::quiet_NaN()};
              });
          }),
          "RK integration accepted a non-finite derivative", failures);
    check(throws<std::domain_error>([] {
              ti::integrateRungeKutta({std::numeric_limits<double>::infinity()}, 0.0, 1.0, 0.1,
                                      [](const ti::State& state, double) { return state; });
          }),
          "RK integration accepted a non-finite initial state", failures);
    check(throws<std::invalid_argument>([] {
              ti::integrateRungeKutta(
                  {1.0}, 0.0, 1.0, 0.1, [](const ti::State& state, double) { return state; },
                  static_cast<ti::RungeKuttaMethod>(99));
          }),
          "RK integration accepted an invalid method enum", failures);
    check(throws<std::runtime_error>([] {
              ti::heunStep({1.0}, 1e20, 1.0,
                           [](const ti::State&, double) { return ti::State{1.0}; });
          }),
          "Heun accepted a step too small to advance its stage clock", failures);
    check(throws<std::overflow_error>([] {
              ti::integrateRungeKutta({1.0}, -1.0e308, 1.0e308, 1.0,
                                      [](const ti::State& state, double) { return state; });
          }),
          "RK integration accepted a non-finite interval duration", failures);

    // Known-polynomial checks isolate spatial formulas from time integration.
    {
        constexpr std::size_t cells = 40;
        constexpr double dx = 2.0 / static_cast<double>(cells);
        ti::State field(cells + 1);
        for (std::size_t i = 0; i <= cells; ++i) {
            const double x = -1.0 + static_cast<double>(i) * dx;
            field[i] = std::pow(x, 6);
        }
        const ti::State laplacian = ti::laplacian1D(field, dx, 6);
        for (std::size_t i = 3; i + 3 <= cells; ++i) {
            const double x = -1.0 + static_cast<double>(i) * dx;
            check(approximatelyEqual(laplacian[i], 30.0 * std::pow(x, 4), 2e-9),
                  "sixth-order Laplacian failed the x^6 polynomial check", failures);
        }
        check(laplacian.front() == 0.0 && laplacian.back() == 0.0,
              "Laplacian inferred values outside the prescribed boundary", failures);
    }
    {
        constexpr int nx = 12;
        constexpr int ny = 8;
        constexpr double dx = 0.1;
        constexpr double dy = 0.2;
        const std::size_t stride = static_cast<std::size_t>(nx + 1);
        ti::State field(stride * static_cast<std::size_t>(ny + 1));
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                const double x = static_cast<double>(i) * dx;
                const double y = static_cast<double>(j) * dy;
                field[static_cast<std::size_t>(j) * stride + static_cast<std::size_t>(i)] =
                    std::pow(x, 4) + std::pow(y, 4);
            }
        }
        const ti::State laplacian = ti::laplacian2D(field, nx, ny, dx, dy, 4);
        for (int j = 2; j <= ny - 2; ++j) {
            for (int i = 2; i <= nx - 2; ++i) {
                const double x = static_cast<double>(i) * dx;
                const double y = static_cast<double>(j) * dy;
                const std::size_t index =
                    static_cast<std::size_t>(j) * stride + static_cast<std::size_t>(i);
                check(approximatelyEqual(laplacian[index], 12.0 * x * x + 12.0 * y * y, 2e-10),
                      "fourth-order 2D Laplacian failed the polynomial check", failures);
            }
        }
    }
    {
        struct ScaleCase {
            double spacing;
            double amplitude;
            double expected_laplacian;
        };
        const std::vector<ScaleCase> cases{{1.0e-200, 1.0e-300, 2.0e100},
                                           {1.0e200, 1.0e300, 2.0e-100}};
        for (const auto& scale : cases) {
            ti::State quadratic(7, 0.0);
            for (std::size_t i = 0; i < quadratic.size(); ++i) {
                const double offset = static_cast<double>(i) - 3.0;
                quadratic[i] = scale.amplitude * offset * offset;
            }
            for (int order : {2, 4, 6}) {
                const auto laplacian = ti::laplacian1D(quadratic, scale.spacing, order);
                check(relativelyEqual(laplacian[3], scale.expected_laplacian),
                      "1D Laplacian lost a finite result at an extreme physical scale", failures);
            }
        }

        const ti::State constant(7, std::numeric_limits<double>::max());
        for (int order : {2, 4, 6}) {
            const auto laplacian = ti::laplacian1D(constant, 1.0, order);
            check(laplacian[3] == 0.0,
                  "Laplacian overflowed while cancelling a finite constant field", failures);
        }

        const ti::State linear{0.0, 1.0e300, 2.0e300, 3.0e300, 4.0e300};
        for (int order : {2, 4}) {
            const auto gradient = ti::gradient1D(linear, 1.0e308, order);
            check(relativelyEqual(gradient[2], 1.0e-8),
                  "1D gradient lost a finite result at a large physical scale", failures);
        }
        const double maximum = std::numeric_limits<double>::max();
        const ti::State signed_linear{-maximum, -0.5 * maximum, 0.0, 0.5 * maximum, maximum};
        for (int order : {2, 4}) {
            const auto gradient = ti::gradient1D(signed_linear, maximum, order);
            check(relativelyEqual(gradient[2], 0.5),
                  "1D gradient overflowed before a finite signed cancellation", failures);
        }

        const double adjacent_a = 1.0e308;
        const double adjacent_b =
            std::nextafter(adjacent_a, std::numeric_limits<double>::infinity());
        const double adjacent_c =
            std::nextafter(adjacent_b, std::numeric_limits<double>::infinity());
        const auto adjacent_laplacian =
            ti::laplacian1D({adjacent_a, adjacent_b, adjacent_a}, 1.0e155, 2);
        check(relativelyEqual(adjacent_laplacian[1], -3.99168061906944e-18),
              "Laplacian normalization lost an adjacent-float variation", failures);
        const auto adjacent_gradient =
            ti::gradient1D({adjacent_a, adjacent_b, adjacent_c}, 1.0e155, 2);
        check(relativelyEqual(adjacent_gradient[1], 1.99584030953472e137),
              "gradient normalization lost an adjacent-float variation", failures);

        const auto cancelling_laplacian = ti::laplacian1D({maximum, 1.0, -maximum}, 1.0, 2);
        check(cancelling_laplacian[1] == -2.0,
              "Laplacian summation erased a finite signal between cancelling extremes", failures);
        const double minimum = std::numeric_limits<double>::denorm_min();
        const auto subnormal_cancelling_laplacian =
            ti::laplacian1D({maximum, minimum, -maximum}, 1.0, 2);
        check(subnormal_cancelling_laplacian[1] == -2.0 * minimum,
              "Laplacian summation erased a subnormal signal between cancelling extremes",
              failures);
        const double amplifying_spacing = std::sqrt(2.0 * minimum);
        try {
            const auto amplified_cancelling_laplacian =
                ti::laplacian1D({maximum, minimum, -maximum}, amplifying_spacing, 2);
            check(relativelyEqual(amplified_cancelling_laplacian[1], -1.0),
                  "Laplacian scaling erased a material derivative from a subnormal signal",
                  failures);
        } catch (const std::exception& error) {
            std::cerr << "FAIL: amplified subnormal stencil threw: " << error.what() << '\n';
            ++failures;
        }
        const auto cancelling_gradient = ti::gradient1D({maximum, 0.0, 0.0, 1.0, maximum}, 1.0, 4);
        check(relativelyEqual(cancelling_gradient[2], 2.0 / 3.0),
              "gradient summation erased a finite signal between cancelling extremes", failures);
    }
    {
        const double maximum = std::numeric_limits<double>::max();
        const auto constant_rhs = [maximum](const ti::State&, double) {
            return ti::State{maximum};
        };
        check(ti::heunStep({0.0}, 0.0, 1.0, constant_rhs)[0] == maximum,
              "Heun overflowed a finite weighted stage sum", failures);
        check(ti::classicalRk4Step({0.0}, 0.0, 1.0, constant_rhs)[0] == maximum,
              "RK4 overflowed a finite weighted stage sum", failures);
        check(ti::detail::addScaled({-maximum}, {maximum}, 2.0, "test")[0] == maximum,
              "scaled stage addition overflowed before finite cancellation", failures);
        const double minimum = std::numeric_limits<double>::denorm_min();
        const auto minimum_rhs = [minimum](const ti::State&, double) {
            return ti::State{minimum};
        };
        check(ti::heunStep({0.0}, 0.0, 1.0, minimum_rhs)[0] == minimum,
              "Heun erased a minimum-subnormal weighted stage sum", failures);
        check(ti::classicalRk4Step({0.0}, 0.0, 1.0, minimum_rhs)[0] == minimum,
              "RK4 erased a minimum-subnormal weighted stage sum", failures);
        check(ti::heunStep({0.0}, 0.0, minimum,
                           [](const ti::State&, double) { return ti::State{1.0}; })[0] == minimum,
              "Heun pre-rounded minimum-subnormal stage weights to zero", failures);
    }
    {
        constexpr int nx = 4;
        constexpr int ny = 4;
        constexpr std::size_t stride = nx + 1;
        const std::vector<std::pair<double, double>> cases{{1.0e-200, 1.0e-300},
                                                           {1.0e200, 1.0e300}};
        for (const auto& scale : cases) {
            for (bool vary_x : {true, false}) {
                ti::State quadratic(stride * static_cast<std::size_t>(ny + 1), 0.0);
                for (int j = 0; j <= ny; ++j) {
                    for (int i = 0; i <= nx; ++i) {
                        const double offset = static_cast<double>(vary_x ? i - 2 : j - 2);
                        quadratic[static_cast<std::size_t>(j) * stride +
                                  static_cast<std::size_t>(i)] = scale.second * offset * offset;
                    }
                }
                const auto laplacian =
                    ti::laplacian2D(quadratic, nx, ny, scale.first, scale.first, 4);
                const double expected = 2.0 * scale.second / scale.first / scale.first;
                check(relativelyEqual(laplacian[2 * stride + 2], expected),
                      "2D Laplacian lost a finite directional result at an extreme scale",
                      failures);
            }
        }

        constexpr int cancellation_cells = 2;
        constexpr std::size_t cancellation_stride = cancellation_cells + 1;
        ti::State cancellation_field(cancellation_stride * cancellation_stride, 0.0);
        const double maximum = std::numeric_limits<double>::max();
        cancellation_field[1 * cancellation_stride + 0] = maximum;
        cancellation_field[1 * cancellation_stride + 2] = maximum;
        cancellation_field[0 * cancellation_stride + 1] = -maximum;
        cancellation_field[2 * cancellation_stride + 1] = -maximum;
        const auto cancelling_laplacian = ti::laplacian2D(cancellation_field, cancellation_cells,
                                                          cancellation_cells, 1.0, 1.0, 2);
        check(cancelling_laplacian[1 * cancellation_stride + 1] == 0.0,
              "2D directional Laplacians overflowed before a finite cancellation", failures);

        ti::State anisotropic_field(cancellation_stride * cancellation_stride, 0.0);
        anisotropic_field[1 * cancellation_stride + 0] = maximum;
        anisotropic_field[1 * cancellation_stride + 2] = maximum;
        const auto anisotropic_laplacian =
            ti::laplacian2D(anisotropic_field, 2, 2, 1.0e155, 1.0, 2);
        check(
            relativelyEqual(anisotropic_laplacian[1 * cancellation_stride + 1], 0.0359538626972462),
            "2D Laplacian overflowed while lifting finite tiny weighted products", failures);
        check(throws<std::overflow_error>(
                  [&] { (void)ti::laplacian2D(anisotropic_field, 2, 2, 1.0e163, 1.0, 2); }),
              "2D Laplacian silently discarded an underflowed directional weight", failures);
    }

    // The stability limit comes from the exact centered-stencil spectral radius.
    {
        const double dt = ti::stableDiffusionTimeStep(0.5, 0.1, 0.1, 6, 0.4, false);
        const double expected = 2.0 * 0.4 * 0.1 * 0.1 / (0.5 * (272.0 / 45.0));
        check(approximatelyEqual(dt, expected, 2e-18),
              "sixth-order diffusion stability limit is incorrect", failures);
    }
    {
        const double tiny_dt =
            ti::stableDiffusionTimeStep(1.0e-300, 1.0e-200, 1.0e-200, 2, 0.4, false);
        const double large_dt =
            ti::stableDiffusionTimeStep(1.0e300, 1.0e200, 1.0e200, 2, 0.4, false);
        check(relativelyEqual(tiny_dt, 2.0e-101),
              "stable dt underflowed through an intermediate spacing square", failures);
        check(relativelyEqual(large_dt, 2.0e99),
              "stable dt overflowed through an intermediate spacing square", failures);

        const double tiny_dt_2d =
            ti::stableDiffusionTimeStep(1.0e-300, 1.0e-200, 2.0e-200, 2, 0.4, true);
        const double large_dt_2d =
            ti::stableDiffusionTimeStep(1.0e300, 1.0e200, 2.0e200, 2, 0.4, true);
        check(relativelyEqual(tiny_dt_2d, 1.6e-101),
              "2D stable dt lost a finite microscopic result", failures);
        check(relativelyEqual(large_dt_2d, 1.6e99), "2D stable dt lost a finite macroscopic result",
              failures);

        check(throws<std::overflow_error>([] {
                  (void)ti::stableDiffusionTimeStep(1.0, std::numeric_limits<double>::denorm_min(),
                                                    1.0, 2, 0.4, false);
              }),
              "an unrepresentably small stable dt was not rejected", failures);
        check(throws<std::overflow_error>([] {
                  (void)ti::stableDiffusionTimeStep(std::numeric_limits<double>::denorm_min(),
                                                    std::numeric_limits<double>::max(), 1.0, 2, 0.4,
                                                    false);
              }),
              "an unrepresentably large stable dt was not rejected", failures);

        const double half_subnormal_spacing =
            std::nextafter(std::ldexp(1.0, -537), std::numeric_limits<double>::infinity());
        check(throws<std::overflow_error>([&] {
                  (void)ti::stableDiffusionTimeStep(1.0, half_subnormal_spacing,
                                                    half_subnormal_spacing, 2, 1.0, false);
              }),
              "an exact stability bound below the minimum subnormal was rounded up", failures);
        check(throws<std::overflow_error>([&] {
                  (void)ti::solveHighOrderDiffusion(
                      {0.0, 1.0e-308, 0.0}, 2, 0, half_subnormal_spacing, half_subnormal_spacing,
                      1.0, 2, 1.0, std::numeric_limits<double>::denorm_min(),
                      std::numeric_limits<double>::denorm_min(), 0.0, 0.0);
              }),
              "diffusion accepted a step when no positive representable stable dt exists",
              failures);

        const double conservative_2d =
            ti::stableDiffusionTimeStep(0x1.bc71d3b799602p-1, 0x1.c75d3e6a238c9p+3,
                                        0x1.e1d5f8fab3384p-1, 4, 0x1.e1fe42fa9bf6dp-3, true);
        check(conservative_2d <= 0x1.6f3665989a8d5p-4,
              "2D stability arithmetic rounded above the exact represented-input bound", failures);
        const double conservative_order4 =
            ti::stableDiffusionTimeStep(0x1.637725584671ep-1, 0x1.94c9ea85844d9p-1,
                                        0x1.94c9ea85844d9p-1, 4, 0x1.42905fe968621p-12, false);
        check(conservative_order4 <= 0x1.b39bd951c4909p-14,
              "fourth-order spectral radius rounded the stability bound upward", failures);
        check(ti::stableDiffusionTimeStep(1.0, 1.0e163, 1.0, 2, 1.0, true) < 0.5,
              "underflowed anisotropy contribution made the 2D stability bound unsafe", failures);
    }

    // Native diffusion applies boundaries at t=0, rejects unsafe dt, and
    // reports nominal versus shortened final step without mutating the input.
    {
        const ti::State initial(21, 0.0);
        const auto result = ti::solveHighOrderDiffusion(initial, 20, 0, 0.05, 0.05, 0.01, 2, 0.4,
                                                        0.1, 0.03, 1.0, 2.0);
        check(initial.front() == 0.0 && initial.back() == 0.0,
              "diffusion solver mutated its initial field", failures);
        check(result.solution.front() == 1.0 && result.solution.back() == 2.0,
              "diffusion solver did not enforce Dirichlet boundaries", failures);
        check(result.time == 0.1 && result.steps == 4,
              "diffusion solver did not end exactly at t_end", failures);
        check(result.nominal_dt == 0.03 && approximatelyEqual(result.last_dt, 0.01, 2e-15),
              "diffusion solver confused nominal and final dt", failures);
        check(result.interior_order == 2 && result.boundary_closure_order == 2,
              "diffusion result did not expose spatial accuracy scope", failures);
    }
    {
        const auto result = ti::solveHighOrderDiffusion(ti::State(3, 0.0), 2, 0, 0.5, 0.5, 0.5, 2,
                                                        0.4, 1.0, 0.1, 0.0, 0.0);
        check(result.time == 1.0 && result.steps == 10,
              "floating-point endpoint residue created an eleventh diffusion step", failures);
        check(result.last_dt > 0.0 && result.last_dt <= 0.1,
              "diffusion endpoint handling exceeded the requested step ceiling", failures);
    }
    {
        constexpr double duration = 0.7000000000000001;
        std::vector<double> observer_times;
        const auto result = ti::solveHighOrderDiffusion(
            ti::State(3, 0.0), 2, 0, 0.5, 0.5, 0.01, 2, 0.4, duration, 0.1, 0.0, 0.0, 0.0, 0.0,
            [&observer_times](double time, const ti::State&) { observer_times.push_back(time); });
        check(result.steps == 8 && observer_times.size() == 8,
              "diffusion did not retain its exact-float residue step", failures);
        check(!observer_times.empty() && observer_times.front() == 0.1 &&
                  observer_times.back() == duration,
              "diffusion observer timestamps did not match canonical step endpoints", failures);
    }
    {
        const double step = std::numeric_limits<double>::denorm_min();
        const double duration = 2.0 * step;
        const auto result = ti::solveHighOrderDiffusion(ti::State(3, 0.0), 2, 0, 0.5, 0.5, 0.5, 2,
                                                        0.4, duration, step, 0.0, 0.0);
        check(result.time == duration && result.steps == 2 && result.last_dt == step,
              "diffusion endpoint handling skipped a representable subnormal step", failures);
    }
    {
        constexpr double step = 0.09375291778124752;
        constexpr double duration = 0.9375291778124752;
        const auto result = ti::solveHighOrderDiffusion(ti::State(3, 0.0), 2, 0, 0.5, 0.5, 0.5, 2,
                                                        0.4, duration, step, 0.0, 0.0);
        check(result.time == duration && result.steps == 11,
              "diffusion rounded an exact-float step ratio down", failures);
        check(result.last_dt > 0.0 && result.last_dt <= step,
              "diffusion exact-float remainder exceeded the requested ceiling", failures);
    }
    {
        struct ScaleCase {
            double spacing;
            double diffusivity;
            double boundary;
            double expected_center;
        };
        const std::vector<ScaleCase> cases{{1.0e-200, 1.0e-300, 1.0e-300, 2.0e-301},
                                           {1.0e200, 1.0e300, 1.0e300, 2.0e299}};
        for (const auto& scale : cases) {
            const double stable_dt = ti::stableDiffusionTimeStep(scale.diffusivity, scale.spacing,
                                                                 scale.spacing, 2, 0.4, false);
            const auto result = ti::solveHighOrderDiffusion(
                ti::State(3, 0.0), 2, 0, scale.spacing, scale.spacing, scale.diffusivity, 2, 0.4,
                stable_dt, stable_dt, 0.0, scale.boundary);
            check(result.steps == 1 && relativelyEqual(result.solution[1], scale.expected_center),
                  "diffusion update lost a finite three-factor increment", failures);
        }

        const double maximum = std::numeric_limits<double>::max();
        const auto cancelling_update =
            ti::solveHighOrderDiffusion({-maximum, maximum, -maximum}, 2, 0, 2.0, 2.0, 1.0, 2, 1.0,
                                        2.0, 2.0, -maximum, -maximum);
        check(cancelling_update.solution[1] == -maximum,
              "diffusion update overflowed before finite state-increment cancellation", failures);
    }
    check(throws<std::domain_error>([] {
              ti::solveHighOrderDiffusion(ti::State(21, 0.0), 20, 0, 0.05, 0.05, 0.01, 2, 0.4, 0.1,
                                          0.051, 0.0, 0.0);
          }),
          "diffusion solver accepted a step above its safety-scaled limit", failures);
    check(throws<std::domain_error>([] {
              const double stable_dt = ti::stableDiffusionTimeStep(0.01, 0.05, 0.05, 2, 0.4, false);
              ti::solveHighOrderDiffusion(
                  ti::State(21, 0.0), 20, 0, 0.05, 0.05, 0.01, 2, 0.4, stable_dt,
                  std::nextafter(stable_dt, std::numeric_limits<double>::infinity()), 0.0, 0.0);
          }),
          "diffusion solver accepted the next representable step above its reported limit",
          failures);
    check(throws<std::domain_error>([] {
              constexpr int cells = 20;
              constexpr double domain_length = 1.0e-8;
              constexpr double dx = domain_length / static_cast<double>(cells);
              ti::solveHighOrderDiffusion(ti::State(cells + 1, 0.0), cells, 0, dx, dx, 1.0, 2, 0.4,
                                          1.0e-15, 1.0e-15, 0.0, 0.0);
          }),
          "diffusion solver accepted a grossly unstable step when the physical stability "
          "limit was far below one second",
          failures);
    check(throws<std::invalid_argument>(
              [] { ti::laplacian2D(ti::State(25, 0.0), 4, 4, 0.1, 0.1, 6); }),
          "2D Laplacian accepted the unsupported sixth-order stencil", failures);

    if (failures == 0) {
        std::cout << "All high-order integration tests passed!\n";
        return 0;
    }
    std::cerr << failures << " high-order test(s) failed.\n";
    return 1;
}
