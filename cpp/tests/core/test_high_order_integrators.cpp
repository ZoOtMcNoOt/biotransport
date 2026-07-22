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

    // A large absolute start time must not trigger tolerance-based early
    // completion or distort the state interval by accumulating on that clock.
    {
        constexpr double initial_time = 1.0e12;
        const double end_time = initial_time + 0.01;
        const double duration = end_time - initial_time;
        const auto result = ti::integrateRungeKutta(
            {3.0}, initial_time, end_time, 0.001,
            [](const ti::State&, double) { return ti::State{1.0}; }, ti::RungeKuttaMethod::HEUN);
        check(result.time == end_time && result.steps > 1,
              "large absolute time caused early integration completion", failures);
        check(approximatelyEqual(result.state[0], 3.0 + duration, 5e-15),
              "large absolute time distorted the integrated duration", failures);
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

    // The stability limit comes from the exact centered-stencil spectral radius.
    {
        const double dt = ti::stableDiffusionTimeStep(0.5, 0.1, 0.1, 6, 0.4, false);
        const double expected = 2.0 * 0.4 * 0.1 * 0.1 / (0.5 * (272.0 / 45.0));
        check(approximatelyEqual(dt, expected, 2e-18),
              "sixth-order diffusion stability limit is incorrect", failures);
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
    check(throws<std::domain_error>([] {
              ti::solveHighOrderDiffusion(ti::State(21, 0.0), 20, 0, 0.05, 0.05, 0.01, 2, 0.4, 0.1,
                                          0.051, 0.0, 0.0);
          }),
          "diffusion solver accepted a step above its safety-scaled limit", failures);
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
