/**
 * @file test_transport_solver_science.cpp
 * @brief Scientific contract tests for the conservative transport solver.
 *
 * This file intentionally does not use assert(): Release builds must execute
 * every check.  A tiny exception-based harness keeps the test standalone.
 */

#include <algorithm>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/problems/transport_problem.hpp>
#include <biotransport/physics/reactions.hpp>
#include <biotransport/solvers/transport_solver.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using biotransport::AdvectionScheme;
using biotransport::Boundary;
using biotransport::solve;
using biotransport::SolveOptions;
using biotransport::StructuredMesh;
using biotransport::TransportProblem;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(double actual, double expected, double tolerance, const std::string& message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": expected " + std::to_string(expected) + ", got " +
                                 std::to_string(actual));
    }
}

double observedOrder(double coarse_error, double fine_error) {
    if (!std::isfinite(coarse_error) || !std::isfinite(fine_error) || coarse_error <= 0.0 ||
        fine_error <= 0.0 || fine_error >= coarse_error) {
        throw std::runtime_error("observed-order inputs must be finite, positive, and decreasing");
    }
    return std::log(coarse_error / fine_error) / std::log(2.0);
}

template <typename Exception, typename Function>
void requireThrows(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void manufacturedQuadraticIsStationaryIn2D() {
    StructuredMesh mesh(12, 9, 0.0, 1.0, 0.0, 1.0);
    std::vector<double> exact(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const double x = mesh.x(i);
            const double y = mesh.y(i, j);
            exact[static_cast<std::size_t>(mesh.index(i, j))] = 3.0 + x * x + 2.0 * y * y;
        }
    }

    // laplacian(3+x^2+2y^2)=6, so R=-6 gives dc/dt=0.  The signs on
    // the right/top Neumann values exercise the outward-normal convention.
    TransportProblem problem(mesh);
    problem.diffusivity(1.0)
        .constantSource(-6.0)
        .initialCondition(exact)
        .neumann(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 2.0)
        .neumann(Boundary::Bottom, 0.0)
        .neumann(Boundary::Top, 4.0);

    SolveOptions options;
    options.final_time = 0.015;
    const auto result = solve(problem, options);

    double maximum_error = 0.0;
    for (std::size_t index = 0; index < exact.size(); ++index) {
        maximum_error =
            std::max(maximum_error, std::abs(result.concentration[index] - exact[index]));
    }
    require(maximum_error < 2e-12, "2D manufactured diffusion/source solution was not preserved");
}

void robinUsesOutwardDerivative() {
    StructuredMesh mesh(20, 0.0, 1.0);
    std::vector<double> exact(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int i = 0; i <= mesh.nx(); ++i) {
        exact[static_cast<std::size_t>(mesh.index(i))] = 1.0 + mesh.x(i);
    }

    // u=1+x has du/dn=-1 on the left and +1 on the right.
    // Left: 2*u+3*du/dn=-1. Right: 4*u+5*du/dn=13.
    TransportProblem problem(mesh);
    problem.diffusivity(0.3)
        .initialCondition(exact)
        .robin(Boundary::Left, 2.0, 3.0, -1.0)
        .robin(Boundary::Right, 4.0, 5.0, 13.0);

    SolveOptions options;
    options.final_time = 0.02;
    const auto result = solve(problem, options);
    for (std::size_t index = 0; index < exact.size(); ++index) {
        requireNear(result.concentration[index], exact[index], 2e-12,
                    "Robin manufactured solution changed");
    }
}

void conservativeFluxesPreserveMassWithVariableFields() {
    constexpr double pi = 3.141592653589793238462643383279502884;
    StructuredMesh mesh(18, 13, 0.0, 1.0, 0.0, 1.0);
    const std::size_t count = static_cast<std::size_t>(mesh.numNodes());
    std::vector<double> initial(count, 0.0);
    std::vector<double> diffusivity(count, 0.0);
    std::vector<double> vx(count, 0.0);
    std::vector<double> vy(count, 0.0);

    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const double x = mesh.x(i);
            const double y = mesh.y(i, j);
            const std::size_t index = static_cast<std::size_t>(mesh.index(i, j));
            initial[index] = 1.0 + 0.15 * std::sin(2.0 * pi * x) * std::cos(pi * y);
            diffusivity[index] = 0.01 * (1.0 + 0.6 * x);
            // Discretely conservative face fluxes telescope regardless of div(v).
            // These values also make normal advective flux zero on every wall.
            vx[index] = std::sin(pi * x) * std::cos(pi * y);
            vy[index] = -std::cos(pi * x) * std::sin(pi * y);
        }
    }

    TransportProblem problem(mesh);
    problem.diffusivityField(diffusivity)
        .velocityField(vx, vy)
        .initialCondition(initial)
        .neumann(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 0.0)
        .neumann(Boundary::Bottom, 0.0)
        .neumann(Boundary::Top, 0.0);

    SolveOptions options;
    options.final_time = 0.035;
    const auto result = solve(problem, options);
    requireNear(result.diagnostics.mass_change, 0.0, 2e-13,
                "closed variable-coefficient problem did not conserve mass");
    require(result.diagnostics.steps > 1, "conservation test did not exercise time stepping");
}

void conservativeAdvectionIncludesVelocityDivergence() {
    StructuredMesh mesh(10, 0.0, 1.0);
    std::vector<double> velocity(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int i = 0; i <= mesh.nx(); ++i) {
        velocity[static_cast<std::size_t>(mesh.index(i))] = mesh.x(i);
    }

    // With c=1 and v=x, -div(v*c)=-1.  A non-conservative v*grad(c)
    // implementation would incorrectly return zero everywhere.
    TransportProblem problem(mesh);
    problem.diffusivity(0.0)
        .velocityField(velocity)
        .initialCondition(1.0)
        .neumann(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 0.0);

    SolveOptions options;
    options.final_time = 0.01;
    options.time_step = 0.01;
    const auto result = solve(problem, options);
    for (double concentration : result.concentration) {
        requireNear(concentration, 0.99, 2e-15,
                    "solver omitted concentration times velocity divergence");
    }
}

double composedReactionError(double dt) {
    StructuredMesh mesh(4, 0.0, 1.0);
    TransportProblem problem(mesh);
    problem.diffusivity(0.0)
        .constantSource(1.0)
        .addLinearDecay(0.5)
        .initialCondition(0.0)
        .neumann(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 0.0);

    SolveOptions options;
    options.final_time = 1.0;
    options.time_step = dt;
    const auto result = solve(problem, options);
    const double exact = 2.0 * (1.0 - std::exp(-0.5));
    return std::abs(result.concentration[2] - exact);
}

void reactionsComposeAndConvergeInTime() {
    const double coarse_error = composedReactionError(0.2);
    const double fine_error = composedReactionError(0.1);
    require(coarse_error > 0.0, "reaction convergence test has zero coarse error");
    require(fine_error < 0.56 * coarse_error,
            "composed source+decay did not show first-order time convergence");
}

double canonicalDiffusionError(int cells) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr double diffusivity = 0.01;
    constexpr double final_time = 0.1;

    StructuredMesh mesh(cells, 0.0, 1.0);
    std::vector<double> initial(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int i = 0; i <= mesh.nx(); ++i) {
        initial[static_cast<std::size_t>(mesh.index(i))] = std::sin(pi * mesh.x(i));
    }

    TransportProblem problem(mesh);
    problem.diffusivity(diffusivity)
        .initialCondition(initial)
        .dirichlet(Boundary::Left, 0.0)
        .dirichlet(Boundary::Right, 0.0);

    // Explicit diffusion requires dt=O(h^2), so every stable grid study must
    // co-refine time.  Using only 2% of the CFL limit keeps the leading Euler
    // contribution at about 6% of the centered-space contribution for this mode;
    // the observed error is therefore a spatial-refinement acceptance check.
    SolveOptions options;
    options.final_time = final_time;
    options.safety_factor = 0.02;
    const auto result = solve(problem, options);

    double squared_error = 0.0;
    for (int i = 0; i <= mesh.nx(); ++i) {
        const double exact =
            std::sin(pi * mesh.x(i)) * std::exp(-diffusivity * pi * pi * final_time);
        const double error = result.concentration[static_cast<std::size_t>(mesh.index(i))] - exact;
        squared_error += error * error;
    }
    return std::sqrt(squared_error / static_cast<double>(mesh.numNodes()));
}

void canonicalSolveHasSecondOrderSpatialGridConvergence() {
    const double coarse_error = canonicalDiffusionError(20);
    const double medium_error = canonicalDiffusionError(40);
    const double fine_error = canonicalDiffusionError(80);
    const double coarse_order = observedOrder(coarse_error, medium_error);
    const double fine_order = observedOrder(medium_error, fine_error);

    require(fine_error < 1e-6, "fine-grid canonical diffusion error exceeded 1e-6");
    require(coarse_order > 1.8 && coarse_order < 2.2,
            "20-to-40 cell canonical diffusion order was outside [1.8, 2.2]: " +
                std::to_string(coarse_order));
    require(fine_order > 1.8 && fine_order < 2.2,
            "40-to-80 cell canonical diffusion order was outside [1.8, 2.2]: " +
                std::to_string(fine_order));
}

double heterogeneousAdvectionDiffusionError(int cells) {
    StructuredMesh mesh(cells, 0.0, 1.0);
    const std::size_t count = static_cast<std::size_t>(mesh.numNodes());
    std::vector<double> exact(count, 0.0);
    std::vector<double> diffusivity(count, 0.0);
    std::vector<double> velocity(count, 0.0);
    for (int i = 0; i <= mesh.nx(); ++i) {
        const double x = mesh.x(i);
        const std::size_t index = static_cast<std::size_t>(mesh.index(i));
        exact[index] = 1.0 + x;
        diffusivity[index] = 0.2 + 0.1 * x;
        velocity[index] = 0.04 + 0.02 * x;
    }

    // For c=1+x, D=0.2+0.1x, and v=0.04+0.02x,
    //
    //   d(D dc/dx)/dx - d(vc)/dx = 0.04 - 0.04x.
    //
    // The independent manufactured source R=-0.04+0.04x makes c stationary.
    // The left Dirichlet condition supplies advective inflow.  At the outflow,
    // 2c+0.3 dc/dn=4.3 exercises a genuine mixed (non-essential) Robin law.
    TransportProblem problem(mesh);
    problem.diffusivityField(diffusivity)
        .velocityField(velocity)
        .reaction([](double, double x, double, double) { return -0.04 + 0.04 * x; }, 0.0)
        .initialCondition(exact)
        .dirichlet(Boundary::Left, 1.0)
        .robin(Boundary::Right, 2.0, 0.3, 4.3);

    SolveOptions options;
    options.final_time = 0.05;
    const auto result = solve(problem, options);

    double squared_error = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const double error = result.concentration[index] - exact[index];
        squared_error += error * error;
    }
    return std::sqrt(squared_error / static_cast<double>(count));
}

void heterogeneousCoefficientsAndMixedBoundariesConverge() {
    const double coarse_error = heterogeneousAdvectionDiffusionError(20);
    const double medium_error = heterogeneousAdvectionDiffusionError(40);
    const double fine_error = heterogeneousAdvectionDiffusionError(80);
    const double coarse_order = observedOrder(coarse_error, medium_error);
    const double fine_order = observedOrder(medium_error, fine_error);

    // Conservative upwind advection is first order in space.  Requiring both
    // adjacent refinement pairs to enter its asymptotic band catches sign,
    // coefficient-placement, boundary-orientation, and source-composition
    // errors without comparing the implementation against its own stencil.
    require(coarse_order > 0.75 && coarse_order < 1.25,
            "heterogeneous mixed-BC order (20-to-40) was outside [0.75, 1.25]: " +
                std::to_string(coarse_order));
    require(fine_order > 0.75 && fine_order < 1.25,
            "heterogeneous mixed-BC order (40-to-80) was outside [0.75, 1.25]: " +
                std::to_string(fine_order));
    require(fine_error < 3e-5,
            "heterogeneous mixed-BC fine-grid error exceeded 3e-5: " + std::to_string(fine_error));
}

void diffusionUsesHarmonicFaceCoefficient() {
    StructuredMesh mesh(2, 0.0, 1.0);
    TransportProblem problem(mesh);
    problem.diffusivityField({1.0, 4.0, 4.0})
        .initialCondition(std::vector<double>{0.0, 1.0, 1.0})
        .dirichlet(Boundary::Left, 0.0)
        .dirichlet(Boundary::Right, 1.0);

    SolveOptions options;
    options.final_time = 0.01;
    options.time_step = 0.01;
    const auto result = solve(problem, options);

    // D_1/2=2*1*4/(1+4)=1.6, q_left=1.6*(1-0)/0.5=3.2.
    // dc_1/dt=(0-3.2)/0.5=-6.4, hence c_1=0.936 after one step.
    requireNear(result.concentration[1], 0.936, 2e-14,
                "variable diffusion did not use a harmonic face coefficient");
}

void finalTimeIsExactAndLastStepIsShortened() {
    StructuredMesh mesh(3, 0.0, 1.0);
    TransportProblem problem(mesh);
    problem.diffusivity(0.0).constantSource(1.0).initialCondition(0.0);

    SolveOptions options;
    options.final_time = 0.23;
    options.time_step = 0.1;
    const auto result = solve(problem, options);
    require(result.time == 0.23, "result time did not equal requested final_time exactly");
    require(result.diagnostics.final_time == 0.23,
            "diagnostic time did not equal requested final_time exactly");
    require(result.diagnostics.steps == 3, "incorrect number of shortened steps");
    requireNear(result.diagnostics.minimum_time_step, 0.03, 2e-16,
                "last step was not shortened to hit final_time");
    requireNear(result.concentration[1], 0.23, 2e-15,
                "constant source was not integrated through exact final_time");
}

void plannedScheduleNeverExceedsRequestedStep() {
    StructuredMesh mesh(3, 0.0, 1.0);
    TransportProblem problem(mesh);
    problem.diffusivity(0.0).constantSource(1.0).initialCondition(0.0);

    SolveOptions snapped_endpoint;
    snapped_endpoint.final_time = 0.1;
    snapped_endpoint.time_step = 0.01;
    snapped_endpoint.max_steps = 10;
    const auto snapped_result = solve(problem, snapped_endpoint);
    require(snapped_result.time == snapped_endpoint.final_time &&
                snapped_result.diagnostics.steps == 10,
            "a correctly rounded binary64 endpoint did not use ten nominal steps");
    require(snapped_result.diagnostics.minimum_time_step == snapped_endpoint.time_step &&
                snapped_result.diagnostics.maximum_time_step == snapped_endpoint.time_step,
            "the snapped binary64 endpoint changed a nominal step");
    requireNear(snapped_result.concentration[1], snapped_endpoint.final_time, 2.0e-16,
                "the snapped binary64 schedule did not cover the requested interval");

    SolveOptions endpoint_below = snapped_endpoint;
    endpoint_below.final_time = std::nextafter(0.1, 0.0);
    const auto below_result = solve(problem, endpoint_below);
    require(below_result.diagnostics.steps == 10,
            "the endpoint immediately below 0.1 did not use ten steps");
    require(below_result.diagnostics.maximum_time_step <= endpoint_below.time_step,
            "the schedule below the snapped endpoint exceeded the requested step");

    SolveOptions endpoint_above = snapped_endpoint;
    endpoint_above.final_time = std::nextafter(0.1, std::numeric_limits<double>::infinity());
    requireThrows<std::runtime_error>(
        [&] { (void)solve(problem, endpoint_above); },
        "the endpoint immediately above 0.1 did not honor max_steps=10");
    endpoint_above.max_steps = 11;
    const auto above_result = solve(problem, endpoint_above);
    require(above_result.diagnostics.steps == 11,
            "the endpoint immediately above 0.1 was incorrectly snapped down");
    require(above_result.diagnostics.minimum_time_step > 0.0 &&
                above_result.diagnostics.maximum_time_step <= endpoint_above.time_step,
            "the schedule above the snapped endpoint emitted an invalid step");

    SolveOptions ordinary;
    ordinary.final_time = 1.0;
    ordinary.time_step = 0.1;
    ordinary.max_steps = 10;
    const auto ordinary_result = solve(problem, ordinary);
    require(ordinary_result.time == 1.0 && ordinary_result.diagnostics.steps == 10,
            "a decimal endpoint residue consumed an extra canonical transport step");
    require(ordinary_result.diagnostics.maximum_time_step <= ordinary.time_step,
            "canonical transport applied a step above the requested ceiling");
    requireNear(ordinary_result.concentration[1], 1.0, 2.0e-15,
                "scheduled decimal steps did not cover the requested interval");

    const double tiny_step = std::numeric_limits<double>::denorm_min();
    SolveOptions tiny;
    tiny.final_time = 2.0 * tiny_step;
    tiny.time_step = tiny_step;
    tiny.max_steps = 2;
    const auto tiny_result = solve(problem, tiny);
    require(tiny_result.time == tiny.final_time && tiny_result.diagnostics.steps == 2,
            "canonical transport skipped a representable subnormal step");
    require(tiny_result.diagnostics.maximum_time_step == tiny_step,
            "canonical transport changed the requested subnormal step");
    require(tiny_result.concentration[1] == tiny.final_time,
            "canonical transport did not integrate both subnormal source steps");

    SolveOptions rounded_ratio;
    rounded_ratio.final_time = 0.9375291778124752;
    rounded_ratio.time_step = 0.09375291778124752;
    rounded_ratio.max_steps = 11;
    const auto rounded_result = solve(problem, rounded_ratio);
    require(
        rounded_result.time == rounded_ratio.final_time && rounded_result.diagnostics.steps == 11,
        "canonical transport rounded an exact-float step ratio down");
    require(rounded_result.diagnostics.maximum_time_step <= rounded_ratio.time_step,
            "canonical transport exact-float remainder exceeded the requested ceiling");
    rounded_ratio.max_steps = 10;
    requireThrows<std::runtime_error>(
        [&] { (void)solve(problem, rounded_ratio); },
        "canonical transport ignored max_steps for an exact-float endpoint residue");

    if (std::numeric_limits<std::size_t>::digits > 53) {
        SolveOptions unrepresentable_count;
        unrepresentable_count.final_time = 1.0;
        unrepresentable_count.time_step = std::ldexp(1.0, -54);
        unrepresentable_count.max_steps = std::numeric_limits<std::size_t>::max();
        requireThrows<std::overflow_error>(
            [&] { (void)solve(problem, unrepresentable_count); },
            "a step count above the exact binary64 integer range was accepted");
    }
}

void boundaryIsAppliedBeforeFirstStencilAndCornersAreDeterministic() {
    StructuredMesh mesh(4, 4, 0.0, 1.0, 0.0, 1.0);
    TransportProblem problem(mesh);
    problem.diffusivity(0.1)
        .initialCondition(0.0)
        .neumann(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 0.0)
        .dirichlet(Boundary::Bottom, 2.0)
        .dirichlet(Boundary::Top, 0.0);

    SolveOptions zero_time;
    const auto initial = solve(problem, zero_time);
    requireNear(initial.concentration[static_cast<std::size_t>(mesh.index(0, 0))], 2.0, 0.0,
                "Dirichlet corner was not imposed before any stencil");

    SolveOptions one_step;
    one_step.final_time = 0.01;
    one_step.time_step = 0.01;
    const auto evolved = solve(problem, one_step);
    require(evolved.concentration[static_cast<std::size_t>(mesh.index(2, 1))] > 0.0,
            "first stencil did not see the imposed Dirichlet boundary");

    TransportProblem contradiction(mesh);
    contradiction.diffusivity(0.1).dirichlet(Boundary::Left, 1.0).dirichlet(Boundary::Bottom, 2.0);
    requireThrows<std::invalid_argument>([&] { (void)solve(contradiction, zero_time); },
                                         "conflicting Dirichlet corner values were accepted");
}

void unsupportedAndUncertifiedModelsFailLoudly() {
    StructuredMesh mesh(10, 0.0, 1.0);
    TransportProblem central(mesh);
    central.diffusivity(0.01)
        .velocity(0.1)
        .advectionScheme(AdvectionScheme::CENTRAL)
        .initialCondition(1.0);
    SolveOptions options;
    options.final_time = 0.01;
    requireThrows<std::invalid_argument>([&] { (void)solve(central, options); },
                                         "unsupported CENTRAL scheme was silently accepted");

    TransportProblem custom(mesh);
    custom.diffusivity(0.01)
        .reaction([](double c, double, double, double) { return -c * c; })
        .initialCondition(1.0);
    requireThrows<std::invalid_argument>([&] { (void)solve(custom, options); },
                                         "automatic solve accepted an unbounded custom reaction");

    options.time_step = 0.001;
    const auto explicit_result = solve(custom, options);
    require(!explicit_result.diagnostics.reaction_stability_bound_known,
            "custom reaction incorrectly claimed a stability certificate");
    require(std::isnan(explicit_result.diagnostics.certified_stable_time_step),
            "uncertified custom reaction reported a finite stability limit");

    TransportProblem unstable(mesh);
    unstable.diffusivity(1.0).initialCondition(1.0);
    options.time_step = 0.02;
    requireThrows<std::invalid_argument>([&] { (void)solve(unstable, options); },
                                         "unstable explicit diffusion step was accepted");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"2D manufactured quadratic", manufacturedQuadraticIsStationaryIn2D},
        {"Robin outward derivative", robinUsesOutwardDerivative},
        {"variable-field conservation", conservativeFluxesPreserveMassWithVariableFields},
        {"conservative advection equation", conservativeAdvectionIncludesVelocityDivergence},
        {"reaction composition and convergence", reactionsComposeAndConvergeInTime},
        {"canonical spatial grid convergence", canonicalSolveHasSecondOrderSpatialGridConvergence},
        {"heterogeneous coefficients and mixed boundaries",
         heterogeneousCoefficientsAndMixedBoundariesConverge},
        {"harmonic diffusion", diffusionUsesHarmonicFaceCoefficient},
        {"exact final time", finalTimeIsExactAndLastStepIsShortened},
        {"bounded planned time schedule", plannedScheduleNeverExceedsRequestedStep},
        {"boundary and corner policy",
         boundaryIsAppliedBeforeFirstStencilAndCornersAreDeterministic},
        {"loud rejection", unsupportedAndUncertifiedModelsFailLoudly},
    };

    int failures = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS: " << test.first << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL: " << test.first << " -- " << error.what() << '\n';
        }
    }

    if (failures != 0) {
        std::cerr << failures << " scientific transport test(s) failed\n";
        return 1;
    }
    std::cout << "All scientific transport tests passed\n";
    return 0;
}
