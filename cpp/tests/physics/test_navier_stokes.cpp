#include "../test_support/science_test.hpp"
#include <biotransport/physics/fluid_dynamics/navier_stokes.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace biotransport;

namespace {

template <typename Exception, typename Callable>
void requireThrows(Callable&& callable, const std::string& context) {
    bool caught = false;
    try {
        callable();
    } catch (const Exception&) {
        caught = true;
    }
    SCIENCE_REQUIRE(caught, context);
}

std::vector<double> zeroField(const StructuredMesh& mesh) {
    return std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 0.0);
}

double packedDivergence(const StructuredMesh& mesh, const std::vector<double>& u,
                        const std::vector<double>& v) {
    const int nx = mesh.nx();
    const int ny = mesh.ny();
    const int stride = nx + 1;
    double maximum = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int index = j * stride + i;
            const double divergence =
                (u[index + 1] - u[index]) / mesh.dx() + (v[index + stride] - v[index]) / mesh.dy();
            maximum = std::max(maximum, std::abs(divergence));
        }
    }
    return maximum;
}

struct ManufacturedVelocityMetrics {
    double h = 0.0;
    double l2_velocity_error = 0.0;
    double maximum_velocity_error = 0.0;
    double divergence = 0.0;
    double pressure_residual = 0.0;
    bool stable = false;
};

ManufacturedVelocityMetrics runManufacturedVelocityBenchmark(int cells) {
    // A smooth, steady, divergence-free solution on [0, 1]^2:
    //
    //   u = A sin^2(pi x) sin(2 pi y)
    //   v = -A sin(2 pi x) sin^2(pi y),  p = 0.
    //
    // Both velocity components vanish on every wall.  Supplying
    // f = rho (u . grad)u - mu laplacian(u) makes this an exact solution of
    // rho (u . grad)u = -grad(p) + mu laplacian(u) + f.  The expressions below
    // are derived analytically rather than from the implementation's discrete
    // operators.
    constexpr double density = 1.3;
    constexpr double viscosity = 0.07;
    constexpr double amplitude = 0.05;
    constexpr double duration = 0.01;
    const double pi = std::acos(-1.0);
    const StructuredMesh mesh(cells, cells, 0.0, 1.0, 0.0, 1.0);
    const int stride = cells + 1;

    const auto exact_u = [=](double x, double y) {
        const double sin_x = std::sin(pi * x);
        return amplitude * sin_x * sin_x * std::sin(2.0 * pi * y);
    };
    const auto exact_v = [=](double x, double y) {
        const double sin_y = std::sin(pi * y);
        return -amplitude * std::sin(2.0 * pi * x) * sin_y * sin_y;
    };
    const auto force_x = [=](double x, double y) {
        const double sin_x = std::sin(pi * x);
        const double sin_2x = std::sin(2.0 * pi * x);
        const double cos_2x = std::cos(2.0 * pi * x);
        const double sin_y = std::sin(pi * y);
        const double sin_2y = std::sin(2.0 * pi * y);
        const double cos_2y = std::cos(2.0 * pi * y);
        const double u = amplitude * sin_x * sin_x * sin_2y;
        const double v = -amplitude * sin_2x * sin_y * sin_y;
        const double du_dx = amplitude * pi * sin_2x * sin_2y;
        const double du_dy = 2.0 * amplitude * pi * sin_x * sin_x * cos_2y;
        const double laplacian_u =
            2.0 * amplitude * pi * pi * sin_2y * (cos_2x - 2.0 * sin_x * sin_x);
        return density * (u * du_dx + v * du_dy) - viscosity * laplacian_u;
    };
    const auto force_y = [=](double x, double y) {
        const double sin_x = std::sin(pi * x);
        const double sin_2x = std::sin(2.0 * pi * x);
        const double cos_2x = std::cos(2.0 * pi * x);
        const double sin_y = std::sin(pi * y);
        const double sin_2y = std::sin(2.0 * pi * y);
        const double cos_2y = std::cos(2.0 * pi * y);
        const double u = amplitude * sin_x * sin_x * sin_2y;
        const double v = -amplitude * sin_2x * sin_y * sin_y;
        const double dv_dx = -2.0 * amplitude * pi * cos_2x * sin_y * sin_y;
        const double dv_dy = -amplitude * pi * sin_2x * sin_2y;
        const double laplacian_v =
            2.0 * amplitude * pi * pi * sin_2x * (2.0 * sin_y * sin_y - cos_2y);
        return density * (u * dv_dx + v * dv_dy) - viscosity * laplacian_v;
    };

    auto u0 = zeroField(mesh);
    auto v0 = zeroField(mesh);
    for (int j = 0; j < cells; ++j) {
        const double y = (static_cast<double>(j) + 0.5) * mesh.dy();
        for (int i = 0; i <= cells; ++i) {
            u0[j * stride + i] = exact_u(static_cast<double>(i) * mesh.dx(), y);
        }
    }
    for (int j = 0; j <= cells; ++j) {
        const double y = static_cast<double>(j) * mesh.dy();
        for (int i = 0; i < cells; ++i) {
            v0[j * stride + i] = exact_v((static_cast<double>(i) + 0.5) * mesh.dx(), y);
        }
    }
    SCIENCE_REQUIRE(packedDivergence(mesh, u0, v0) < 1e-13,
                    "manufactured initial velocity must be discretely solenoidal");

    const double kinematic_viscosity = viscosity / density;
    // The explicit viscous ceiling scales as h^2.  Multiplying by h/L makes
    // dt=O(h^3), so first-order time error is asymptotically smaller than the
    // second-order spatial error measured by this refinement study.
    const double domain_length = 1.0;
    const double dt =
        0.02 * mesh.dx() * mesh.dx() / kinematic_viscosity * (mesh.dx() / domain_length);
    NavierStokesSolver solver(mesh, density, viscosity);
    solver.setInitialVelocity(u0, v0)
        .setBodyForce(force_x, force_y)
        .setConvectionScheme(ConvectionScheme::CENTRAL)
        .setTimeStep(dt)
        .setPressureTolerance(1e-11)
        .setMaxPressureIterations(20000);
    const auto result = solver.solve(duration);

    long double squared_error = 0.0L;
    std::size_t value_count = 0;
    double maximum_error = 0.0;
    for (int j = 0; j < cells; ++j) {
        const double y = (static_cast<double>(j) + 0.5) * mesh.dy();
        for (int i = 0; i <= cells; ++i) {
            const int index = j * stride + i;
            const double error = result.u[index] - exact_u(static_cast<double>(i) * mesh.dx(), y);
            squared_error += static_cast<long double>(error) * error;
            maximum_error = std::max(maximum_error, std::abs(error));
            ++value_count;
        }
    }
    for (int j = 0; j <= cells; ++j) {
        const double y = static_cast<double>(j) * mesh.dy();
        for (int i = 0; i < cells; ++i) {
            const int index = j * stride + i;
            const double error =
                result.v[index] - exact_v((static_cast<double>(i) + 0.5) * mesh.dx(), y);
            squared_error += static_cast<long double>(error) * error;
            maximum_error = std::max(maximum_error, std::abs(error));
            ++value_count;
        }
    }

    return ManufacturedVelocityMetrics{
        mesh.dx(),
        std::sqrt(static_cast<double>(squared_error / static_cast<long double>(value_count))),
        maximum_error,
        result.divergence,
        result.pressure_residual,
        result.stable,
    };
}

void manufacturedSteadyVelocityConverges() {
    const auto coarse = runManufacturedVelocityBenchmark(8);
    const auto medium = runManufacturedVelocityBenchmark(12);
    const auto fine = runManufacturedVelocityBenchmark(16);
    const double coarse_order = std::log(coarse.l2_velocity_error / medium.l2_velocity_error) /
                                std::log(coarse.h / medium.h);
    const double fine_order =
        std::log(medium.l2_velocity_error / fine.l2_velocity_error) / std::log(medium.h / fine.h);

    science_test::report("manufactured coarse L2 velocity error", coarse.l2_velocity_error, "m/s");
    science_test::report("manufactured medium L2 velocity error", medium.l2_velocity_error, "m/s");
    science_test::report("manufactured fine L2 velocity error", fine.l2_velocity_error, "m/s");
    science_test::report("manufactured coarse-to-medium order", coarse_order);
    science_test::report("manufactured medium-to-fine order", fine_order);
    science_test::report("manufactured fine max velocity error", fine.maximum_velocity_error,
                         "m/s");
    science_test::report("manufactured fine divergence", fine.divergence, "1/s");
    science_test::report("manufactured fine pressure residual", fine.pressure_residual);

    SCIENCE_REQUIRE(coarse.stable && medium.stable && fine.stable,
                    "all manufactured-flow refinements must report projection-stable");
    SCIENCE_REQUIRE(medium.l2_velocity_error < coarse.l2_velocity_error &&
                        fine.l2_velocity_error < medium.l2_velocity_error,
                    "manufactured velocity error must decrease on every refinement");
    SCIENCE_REQUIRE(coarse_order > 1.5 && fine_order > 1.5,
                    "central manufactured velocity must demonstrate at least 1.5 order");
    SCIENCE_REQUIRE(fine.maximum_velocity_error < 2e-4,
                    "fine-grid manufactured velocity must be accurate to 0.2 mm/s");
    SCIENCE_REQUIRE(fine.divergence < 1e-9,
                    "manufactured velocity must satisfy the compatible continuity constraint");
    SCIENCE_REQUIRE(fine.pressure_residual <= 1e-11,
                    "manufactured pressure projection must meet its configured residual");
}

void constructionAndOwnedMesh() {
    NavierStokesSolver solver(StructuredMesh(8, 6, 0.0, 1.0, 0.0, 1.0), 1.0, 0.02);
    const auto result = solver.solve(0.0);
    SCIENCE_REQUIRE(result.stable, "a zero initial field must be a valid incompressible state");
    SCIENCE_REQUIRE(result.u.size() == 63, "solver must safely own a temporary 8x6 mesh");

    const StructuredMesh mesh(4, 4, 0.0, 1.0, 0.0, 1.0);
    const StructuredMesh mesh_1d(4, 0.0, 1.0);
    requireThrows<std::invalid_argument>([&] { NavierStokesSolver invalid(mesh, 0.0, 1.0); },
                                         "zero density must be rejected");
    requireThrows<std::invalid_argument>(
        [&] { NavierStokesSolver invalid(mesh, std::numeric_limits<double>::quiet_NaN(), 1.0); },
        "non-finite density must be rejected");
    requireThrows<std::invalid_argument>([&] { NavierStokesSolver invalid(mesh, 1.0, -1.0); },
                                         "negative viscosity must be rejected");
    requireThrows<std::invalid_argument>([&] { NavierStokesSolver invalid(mesh_1d, 1.0, 1.0); },
                                         "a one-dimensional mesh must be rejected");
}

void configurationContracts() {
    const StructuredMesh mesh(6, 6, 0.0, 1.0, 0.0, 1.0);
    NavierStokesSolver solver(mesh, 1.0, 0.01);

    requireThrows<std::invalid_argument>([&] { solver.setCFL(0.0); }, "CFL=0 must be rejected");
    requireThrows<std::invalid_argument>([&] { solver.setCFL(1.01); }, "CFL>1 must be rejected");
    requireThrows<std::invalid_argument>(
        [&] { solver.setCFL(std::numeric_limits<double>::infinity()); },
        "non-finite CFL must be rejected");
    requireThrows<std::invalid_argument>([&] { solver.setTimeStep(-1.0); },
                                         "negative time step must be rejected");
    requireThrows<std::invalid_argument>([&] { solver.setPressureTolerance(0.0); },
                                         "zero pressure tolerance must be rejected");
    requireThrows<std::invalid_argument>([&] { solver.setMaxPressureIterations(0); },
                                         "zero pressure iterations must be rejected");
    requireThrows<std::invalid_argument>(
        [&] { solver.setConvectionScheme(ConvectionScheme::QUICK); },
        "unimplemented QUICK convection must be rejected");
    requireThrows<std::invalid_argument>(
        [&] { solver.setConvectionScheme(ConvectionScheme::HYBRID); },
        "unimplemented HYBRID convection must be rejected");
    requireThrows<std::invalid_argument>(
        [&] { solver.setVelocityBC(Boundary::Right, VelocityBC::Outflow()); },
        "open boundaries must not silently use an incompatible pressure model");
    requireThrows<std::invalid_argument>(
        [&] { solver.setVelocityBC(Boundary::Left, VelocityBC::Inflow(1.0)); },
        "inflow shorthand must be rejected until an open-boundary model exists");
    requireThrows<std::invalid_argument>(
        [&] { solver.setInlet(Boundary::Left, [](double, double) { return 1.0; }); },
        "profile inlet must be rejected explicitly");

    solver.setConvectionScheme(ConvectionScheme::CENTRAL);
    solver.setCFL(0.5);
    solver.setTimeStep(0.0);
}

void invalidBodyForceFailsLoudly() {
    const StructuredMesh mesh(6, 6, 0.0, 1.0, 0.0, 1.0);
    NavierStokesSolver missing_callback(mesh, 1.0, 0.01);
    requireThrows<std::invalid_argument>(
        [&] {
            missing_callback.setBodyForce(std::function<double(double, double)>{},
                                          [](double, double) { return 0.0; });
        },
        "missing body-force callbacks must be rejected at configuration time");

    NavierStokesSolver nonfinite_force(mesh, 1.0, 0.01);
    nonfinite_force.setBodyForce(
        [](double, double) { return std::numeric_limits<double>::quiet_NaN(); },
        [](double, double) { return 0.0; });
    requireThrows<std::domain_error>(
        [&] { (void)nonfinite_force.solveSteps(1); },
        "a non-finite manufactured-force value must fail before state is returned");
}

void initialFieldContracts() {
    const StructuredMesh mesh(6, 5, 0.0, 1.0, 0.0, 1.0);
    NavierStokesSolver solver(mesh, 1.0, 0.01);
    auto u = zeroField(mesh);
    auto v = zeroField(mesh);

    auto short_u = u;
    short_u.pop_back();
    requireThrows<std::invalid_argument>([&] { solver.setInitialVelocity(short_u, v); },
                                         "short initial field must be rejected");
    u[3] = std::numeric_limits<double>::quiet_NaN();
    requireThrows<std::invalid_argument>([&] { solver.setInitialVelocity(u, v); },
                                         "non-finite initial field must be rejected");
}

void exactTimeAndStepSemantics() {
    const StructuredMesh mesh(8, 6, 0.0, 1.0, 0.0, 1.0);
    NavierStokesSolver solver(mesh, 1.0, 0.02);
    solver.setTimeStep(0.001);

    const auto duration_result = solver.solve(0.0035);
    SCIENCE_REQUIRE_NEAR(duration_result.time, 0.0035, 1e-15, 0.0, "exact requested final time");
    SCIENCE_REQUIRE(duration_result.time_steps == 4,
                    "final shortened step must be counted exactly once");
    SCIENCE_REQUIRE(duration_result.stable, "quiescent fixed-step solution must be stable");
    SCIENCE_REQUIRE_NEAR(duration_result.divergence, 0.0, 1e-14, 0.0, "quiescent divergence");

    const auto step_result = solver.solveSteps(7);
    SCIENCE_REQUIRE(step_result.time_steps == 7, "solveSteps(7) must take exactly seven steps");
    SCIENCE_REQUIRE_NEAR(step_result.time, 0.007, 1e-15, 0.0, "seven fixed time steps");

    requireThrows<std::invalid_argument>([&] { (void)solver.solve(-0.1); },
                                         "negative duration must be rejected");
    requireThrows<std::invalid_argument>(
        [&] { (void)solver.solve(std::numeric_limits<double>::infinity()); },
        "non-finite duration must be rejected");
    requireThrows<std::invalid_argument>([&] { (void)solver.solve(0.1, 0.01); },
                                         "unimplemented snapshot interval must be rejected");
    requireThrows<std::invalid_argument>([&] { (void)solver.solveSteps(-1); },
                                         "negative step count must be rejected");
}

void tinyPhysicalScalesPreserveStabilityAndTimeContracts() {
    const StructuredMesh microscopic_mesh(6, 6, 0.0, 1.0e-8, 0.0, 1.0e-8);
    NavierStokesSolver unstable(microscopic_mesh, 1.0, 1.0);
    unstable.setTimeStep(1.0e-15);
    requireThrows<std::domain_error>(
        [&] { (void)unstable.solveSteps(1); },
        "a unit-scale epsilon allowance must not admit a grossly unstable microscopic step");

    const StructuredMesh ordinary_mesh(6, 6, 0.0, 1.0, 0.0, 1.0);
    NavierStokesSolver exact_clock(ordinary_mesh, 1.0, 0.01);
    const double step = std::ldexp(1.0, -60);
    const double duration = 8.0 * step;
    exact_clock.setTimeStep(step);
    const auto result = exact_clock.solve(duration);
    SCIENCE_REQUIRE(result.time == duration,
                    "tiny-duration solve must report the exact requested endpoint");
    SCIENCE_REQUIRE(result.time_steps == 8,
                    "tiny-duration solve must not be snapped complete after one step");
}

void compatibleProjectionReducesDivergence() {
    const int nx = 12;
    const int ny = 10;
    const StructuredMesh mesh(nx, ny, 0.0, 1.0, 0.0, 1.0);
    auto u = zeroField(mesh);
    auto v = zeroField(mesh);
    const int stride = nx + 1;
    const double pi = std::acos(-1.0);

    for (int j = 0; j < ny; ++j) {
        const double y = (static_cast<double>(j) + 0.5) * mesh.dy();
        for (int i = 1; i < nx; ++i) {
            const double x = static_cast<double>(i) * mesh.dx();
            u[j * stride + i] = 0.2 * std::sin(pi * x) * std::sin(pi * y);
        }
    }
    const double initial_divergence = packedDivergence(mesh, u, v);
    SCIENCE_REQUIRE(initial_divergence > 0.1,
                    "test field must begin with material discrete divergence");

    NavierStokesSolver solver(mesh, 1.0, 0.01);
    solver.setInitialVelocity(u, v)
        .setTimeStep(0.0005)
        .setPressureTolerance(1e-10)
        .setMaxPressureIterations(10000);
    const auto result = solver.solveSteps(1);

    science_test::report("initial max divergence", initial_divergence, "1/s");
    science_test::report("projected max divergence", result.divergence, "1/s");
    science_test::report("relative pressure residual", result.pressure_residual);
    SCIENCE_REQUIRE(result.stable, "a converged projection must report stable");
    SCIENCE_REQUIRE(result.pressure_residual <= 1e-10,
                    "reported pressure residual must meet its contract");
    SCIENCE_REQUIRE(result.divergence < 1e-8,
                    "compatible projection must reduce max divergence quantitatively");
    SCIENCE_REQUIRE(result.divergence < initial_divergence * 1e-8,
                    "projection must reduce divergence by at least eight orders of magnitude");
}

void divergentZeroDurationIsNotStable() {
    const int nx = 8;
    const int ny = 8;
    const StructuredMesh mesh(nx, ny, 0.0, 1.0, 0.0, 1.0);
    auto u = zeroField(mesh);
    auto v = zeroField(mesh);
    const int stride = nx + 1;
    for (int j = 0; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            u[j * stride + i] = 0.1 * static_cast<double>(i);
        }
    }

    NavierStokesSolver solver(mesh, 1.0, 0.01);
    solver.setInitialVelocity(u, v);
    const auto result = solver.solve(0.0);
    SCIENCE_REQUIRE(result.divergence > 0.1, "the test state must be measurably divergent");
    SCIENCE_REQUIRE(!result.stable,
                    "a finite but materially divergent field must never be reported stable");
}

void lidDrivenCavityRemainsSolenoidal() {
    const StructuredMesh mesh(12, 12, 0.0, 1.0, 0.0, 1.0);
    NavierStokesSolver solver(mesh, 1.0, 0.1);
    solver.setVelocityBC(Boundary::Top, VelocityBC::Dirichlet(1.0, 0.0))
        .setTimeStep(0.0005)
        .setPressureTolerance(1e-9);
    const auto result = solver.solveSteps(12);

    science_test::report("cavity max speed", result.max_velocity, "m/s");
    science_test::report("cavity divergence", result.divergence, "1/s");
    SCIENCE_REQUIRE(result.stable, "bounded lid-driven cavity must remain projection-stable");
    SCIENCE_REQUIRE(result.max_velocity > 1e-4, "moving lid must transmit momentum into the fluid");
    SCIENCE_REQUIRE(result.divergence < 1e-7,
                    "lid-driven cavity velocity must remain discretely solenoidal");
}

void closedDomainForceIsBalancedByPressure() {
    const StructuredMesh mesh(10, 8, 0.0, 1.0, 0.0, 1.0);
    NavierStokesSolver solver(mesh, 1.0, 0.05);
    solver.setBodyForce(3.0, 0.0)
        .setTimeStep(0.0005)
        .setPressureTolerance(1e-10)
        .setMaxPressureIterations(10000);
    const auto result = solver.solveSteps(1);

    SCIENCE_REQUIRE(result.stable, "constant force projection must converge");
    SCIENCE_REQUIRE(result.divergence < 1e-8,
                    "pressure-balanced closed-domain force must remain divergence free");
    SCIENCE_REQUIRE(result.max_velocity < 1e-8,
                    "a conservative uniform body force in a closed domain is balanced by pressure");
}

void incompatibleFluxAndFailedPressureFailLoudly() {
    const StructuredMesh mesh(10, 10, 0.0, 1.0, 0.0, 1.0);
    NavierStokesSolver incompatible(mesh, 1.0, 0.01);
    incompatible.setVelocityBC(Boundary::Left, VelocityBC::Dirichlet(0.1, 0.0)).setTimeStep(0.001);
    requireThrows<std::domain_error>([&] { (void)incompatible.solveSteps(1); },
                                     "nonzero prescribed net flux must fail loudly");

    auto u = zeroField(mesh);
    auto v = zeroField(mesh);
    const int stride = mesh.nx() + 1;
    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 1; i < mesh.nx(); ++i) {
            u[j * stride + i] = std::sin(0.37 * static_cast<double>(i + 2 * j));
        }
    }
    NavierStokesSolver unconverged(mesh, 1.0, 0.01);
    unconverged.setInitialVelocity(u, v)
        .setTimeStep(0.0001)
        .setPressureTolerance(1e-14)
        .setMaxPressureIterations(1);
    requireThrows<std::runtime_error>([&] { (void)unconverged.solveSteps(1); },
                                      "an unconverged pressure projection must not return stable");
}

}  // namespace

int main() {
    return science_test::runSuite(
        "Navier-Stokes compatible projection",
        {{"construction owns mesh and validates parameters", constructionAndOwnedMesh},
         {"configuration rejects unsupported numerics", configurationContracts},
         {"invalid body-force callbacks fail loudly", invalidBodyForceFailsLoudly},
         {"initial fields have exact finite layout", initialFieldContracts},
         {"duration and solveSteps are exact", exactTimeAndStepSemantics},
         {"tiny physical scales preserve stability and time contracts",
          tinyPhysicalScalesPreserveStabilityAndTimeContracts},
         {"manufactured steady velocity converges", manufacturedSteadyVelocityConverges},
         {"projection quantitatively removes divergence", compatibleProjectionReducesDivergence},
         {"divergent zero-duration state is not stable", divergentZeroDurationIsNotStable},
         {"lid-driven cavity remains solenoidal", lidDrivenCavityRemainsSolenoidal},
         {"closed-domain force is pressure-balanced", closedDomainForceIsBalancedByPressure},
         {"incompatible or unconverged projection fails",
          incompatibleFluxAndFailedPressureFailLoudly}});
}
