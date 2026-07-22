/**
 * @file test_stokes.cpp
 * @brief Always-on verification tests for the steady incompressible Stokes solver.
 */

#include "../test_support/science_test.hpp"
#include <algorithm>
#include <biotransport/physics/fluid_dynamics/stokes.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace biotransport;

namespace {

double maximumMagnitude(const std::vector<double>& values) {
    double maximum = 0.0;
    for (double value : values) {
        SCIENCE_REQUIRE_FINITE(value, "field value");
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

double maximumDivergence(const StructuredMesh& mesh, const std::vector<double>& u,
                         const std::vector<double>& v) {
    const int nx = mesh.nx();
    const int ny = mesh.ny();
    const int stride = nx + 1;
    const double dx = mesh.dx();
    const double dy = mesh.dy();
    double maximum = 0.0;

    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            const int index = j * stride + i;
            const double divergence = (u[index + 1] - u[index - 1]) / (2.0 * dx) +
                                      (v[index + stride] - v[index - stride]) / (2.0 * dy);
            maximum = std::max(maximum, std::abs(divergence));
        }
    }
    return maximum;
}

double maximumMomentumResidual(const StructuredMesh& mesh, const StokesResult& result,
                               double viscosity, double force_x, double force_y) {
    const int nx = mesh.nx();
    const int ny = mesh.ny();
    const int stride = nx + 1;
    const double dx = mesh.dx();
    const double dy = mesh.dy();
    const double dx_squared = dx * dx;
    const double dy_squared = dy * dy;
    double maximum = 0.0;

    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            const int index = j * stride + i;
            const double laplacian_u =
                (result.u[index - 1] - 2.0 * result.u[index] + result.u[index + 1]) / dx_squared +
                (result.u[index - stride] - 2.0 * result.u[index] + result.u[index + stride]) /
                    dy_squared;
            const double laplacian_v =
                (result.v[index - 1] - 2.0 * result.v[index] + result.v[index + 1]) / dx_squared +
                (result.v[index - stride] - 2.0 * result.v[index] + result.v[index + stride]) /
                    dy_squared;
            const double pressure_x =
                (result.pressure[index + 1] - result.pressure[index - 1]) / (2.0 * dx);
            const double pressure_y =
                (result.pressure[index + stride] - result.pressure[index - stride]) / (2.0 * dy);

            maximum = std::max(maximum, std::abs(-pressure_x + viscosity * laplacian_u + force_x));
            maximum = std::max(maximum, std::abs(-pressure_y + viscosity * laplacian_v + force_y));
        }
    }
    return maximum;
}

void testConstructionAndPhysicalInputValidation() {
    StructuredMesh mesh_2d(8, 8, 0.0, 1.0, 0.0, 1.0);
    StructuredMesh mesh_1d(8, 0.0, 1.0);
    StokesSolver valid(mesh_2d, 1.0e-3);
    SCIENCE_REQUIRE_NEAR(valid.viscosity(), 1.0e-3, 0.0, 0.0, "stored viscosity");

    bool rejected_nonpositive_viscosity = false;
    try {
        StokesSolver invalid(mesh_2d, 0.0);
    } catch (const std::invalid_argument&) {
        rejected_nonpositive_viscosity = true;
    }
    SCIENCE_REQUIRE(rejected_nonpositive_viscosity,
                    "zero viscosity is outside the Stokes model and must be rejected");

    bool rejected_one_dimensional_mesh = false;
    try {
        StokesSolver invalid(mesh_1d, 1.0e-3);
    } catch (const std::invalid_argument&) {
        rejected_one_dimensional_mesh = true;
    }
    SCIENCE_REQUIRE(rejected_one_dimensional_mesh,
                    "an incompressible 2D solver must reject a 1D mesh");
}

void testQuiescentExactSolution() {
    StructuredMesh mesh(8, 8, 0.0, 1.0, 0.0, 1.0);
    StokesSolver solver(mesh, 1.0);
    solver.setTolerance(1.0e-12).setMaxIterations(10).setBodyForce(0.0, 0.0);

    const StokesResult result = solver.solve();
    SCIENCE_REQUIRE(result.converged, "zero forcing with no-slip walls is an exact steady state");
    SCIENCE_REQUIRE(result.iterations == 1,
                    "an exact zero initial state should converge in one outer iteration; actual=" +
                        std::to_string(result.iterations));
    SCIENCE_REQUIRE_NEAR(maximumMagnitude(result.u), 0.0, 0.0, 0.0, "quiescent x-velocity");
    SCIENCE_REQUIRE_NEAR(maximumMagnitude(result.v), 0.0, 0.0, 0.0, "quiescent y-velocity");
    SCIENCE_REQUIRE_NEAR(maximumMagnitude(result.pressure), 0.0, 0.0, 0.0,
                         "quiescent gauge pressure");
    SCIENCE_REQUIRE_NEAR(result.residual, 0.0, 0.0, 0.0, "quiescent momentum residual");
    SCIENCE_REQUIRE_NEAR(result.divergence, 0.0, 0.0, 0.0, "quiescent continuity residual");
}

void testSealedUniformForceHydrostaticEquilibrium() {
    const StructuredMesh mesh(10, 8, -0.25, 1.75, 0.5, 2.0);
    constexpr double viscosity = 0.37;
    constexpr double force_x = 7.5;
    constexpr double force_y = -2.25;

    StokesSolver solver(mesh, viscosity);
    solver.setVelocityBC(Boundary::Left, VelocityBC::NoSlip())
        .setVelocityBC(Boundary::Right, VelocityBC::NoSlip())
        .setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip())
        .setVelocityBC(Boundary::Top, VelocityBC::NoSlip())
        .setBodyForce(force_x, force_y)
        .setTolerance(1.0e-14)
        .setMaxIterations(1);

    const StokesResult result = solver.solve();
    SCIENCE_REQUIRE(result.converged,
                    "a sealed domain under a conservative uniform force is an exact equilibrium");
    SCIENCE_REQUIRE(result.iterations == 1,
                    "the analytical hydrostatic branch must not run an iterative approximation");
    SCIENCE_REQUIRE_NEAR(maximumMagnitude(result.u), 0.0, 0.0, 0.0, "hydrostatic x-velocity");
    SCIENCE_REQUIRE_NEAR(maximumMagnitude(result.v), 0.0, 0.0, 0.0, "hydrostatic y-velocity");
    SCIENCE_REQUIRE_NEAR(result.divergence, 0.0, 0.0, 0.0, "hydrostatic continuity residual");

    long double raw_pressure_sum = 0.0L;
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            raw_pressure_sum += static_cast<long double>(force_x) * mesh.x(i) +
                                static_cast<long double>(force_y) * mesh.y(i, j);
        }
    }
    const double expected_mean =
        static_cast<double>(raw_pressure_sum / static_cast<long double>(mesh.numNodes()));
    double computed_mean = 0.0;
    const int stride = mesh.nx() + 1;
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const int index = j * stride + i;
            const double expected = force_x * mesh.x(i) + force_y * mesh.y(i, j) - expected_mean;
            SCIENCE_REQUIRE_NEAR(result.pressure[index], expected, 2.0e-14, 2.0e-14,
                                 "zero-mean hydrostatic pressure");
            computed_mean += result.pressure[index];
        }
    }
    computed_mean /= static_cast<double>(mesh.numNodes());
    SCIENCE_REQUIRE_NEAR(computed_mean, 0.0, 2.0e-14, 0.0, "deterministic nodal pressure gauge");
    SCIENCE_REQUIRE(result.residual < 2.0e-13,
                    "hydrostatic momentum defect must be roundoff-sized; actual=" +
                        science_test::number(result.residual));
}

struct ChannelBenchmark {
    StructuredMesh mesh;
    StokesResult result;
    double maximum_velocity_error;
    double maximum_transverse_velocity;
    double maximum_divergence;
    double maximum_momentum_residual;
    double reference_maximum_velocity;
    double body_force;
    double height;

    ChannelBenchmark()
        : mesh(24, 12, 0.0, 2.0, 0.0, 1.0),
          result(),
          maximum_velocity_error(0.0),
          maximum_transverse_velocity(0.0),
          maximum_divergence(0.0),
          maximum_momentum_residual(0.0),
          reference_maximum_velocity(1.0),
          body_force(8.0),
          height(1.0) {
        constexpr double viscosity = 1.0;
        StokesSolver solver(mesh, viscosity);
        solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip())
            .setVelocityBC(Boundary::Top, VelocityBC::NoSlip())
            .setVelocityBC(Boundary::Left, VelocityBC::Outflow())
            .setVelocityBC(Boundary::Right, VelocityBC::Outflow())
            .setBodyForce(body_force, 0.0)
            .setTolerance(1.0e-5)
            .setMaxIterations(2000);
        result = solver.solve();

        const int stride = mesh.nx() + 1;
        for (int j = 0; j <= mesh.ny(); ++j) {
            const double y = mesh.y(0, j);
            const double exact = body_force * y * (height - y) / (2.0 * viscosity);
            for (int i = 0; i <= mesh.nx(); ++i) {
                const int index = j * stride + i;
                maximum_velocity_error =
                    std::max(maximum_velocity_error, std::abs(result.u[index] - exact));
            }
        }

        maximum_transverse_velocity = maximumMagnitude(result.v);
        maximum_divergence = maximumDivergence(mesh, result.u, result.v);
        maximum_momentum_residual =
            maximumMomentumResidual(mesh, result, viscosity, body_force, 0.0);
    }
};

const ChannelBenchmark& channelBenchmark() {
    static const ChannelBenchmark benchmark;
    return benchmark;
}

void reportChannelMetrics(const ChannelBenchmark& benchmark) {
    science_test::report("outer iterations", benchmark.result.iterations);
    science_test::report("reported converged", benchmark.result.converged ? 1.0 : 0.0);
    science_test::report("reported residual", benchmark.result.residual);
    science_test::report("reported divergence", benchmark.result.divergence, "s^-1");
    science_test::report("max velocity error / Umax",
                         benchmark.maximum_velocity_error / benchmark.reference_maximum_velocity);
    science_test::report("max |v| / Umax", benchmark.maximum_transverse_velocity /
                                               benchmark.reference_maximum_velocity);
    science_test::report("H max|div(u)| / Umax", benchmark.height * benchmark.maximum_divergence /
                                                     benchmark.reference_maximum_velocity);
    science_test::report("max momentum residual / body force",
                         benchmark.maximum_momentum_residual / benchmark.body_force);
}

void testPlanePoiseuilleAccuracyAndConservation() {
    // With f_x=8, mu=1, H=1, the exact fully developed solution is
    // u(y)=4y(1-y), v=0 and Umax=1. The centered Laplacian is exact for this quadratic.
    const ChannelBenchmark& benchmark = channelBenchmark();
    reportChannelMetrics(benchmark);

    const double relative_velocity_error =
        benchmark.maximum_velocity_error / benchmark.reference_maximum_velocity;
    const double relative_transverse_velocity =
        benchmark.maximum_transverse_velocity / benchmark.reference_maximum_velocity;
    const double dimensionless_divergence =
        benchmark.height * benchmark.maximum_divergence / benchmark.reference_maximum_velocity;
    const double normalized_momentum_residual =
        benchmark.maximum_momentum_residual / benchmark.body_force;

    SCIENCE_REQUIRE(relative_velocity_error < 5.0e-3,
                    "Poiseuille velocity must agree with the exact profile to <0.5%; actual=" +
                        science_test::number(relative_velocity_error));
    SCIENCE_REQUIRE(relative_transverse_velocity < 1.0e-5,
                    "fully developed channel flow must not create transverse velocity; actual=" +
                        science_test::number(relative_transverse_velocity));
    SCIENCE_REQUIRE(dimensionless_divergence < 1.0e-4,
                    "continuity defect H|max div(u)|/Umax must be <1e-4; actual=" +
                        science_test::number(dimensionless_divergence));
    SCIENCE_REQUIRE(normalized_momentum_residual < 2.0e-3,
                    "discrete momentum defect must be <0.2% of the driving force; actual=" +
                        science_test::number(normalized_momentum_residual));
}

void testResultDiagnosticsHaveDocumentedScientificMeaning() {
    const ChannelBenchmark& benchmark = channelBenchmark();

    SCIENCE_REQUIRE_NEAR(benchmark.result.divergence, benchmark.maximum_divergence, 1.0e-12,
                         1.0e-10, "reported maximum divergence");
    SCIENCE_REQUIRE_NEAR(
        benchmark.result.residual, benchmark.maximum_momentum_residual, 1.0e-12, 1.0e-8,
        "reported momentum residual (StokesResult::residual is documented as this quantity)");
}

void testConvergenceFlagMatchesScientificAcceptanceCriteria() {
    const ChannelBenchmark& benchmark = channelBenchmark();
    const bool meets_scientific_limits =
        benchmark.maximum_velocity_error / benchmark.reference_maximum_velocity < 5.0e-3 &&
        benchmark.height * benchmark.maximum_divergence / benchmark.reference_maximum_velocity <
            1.0e-4 &&
        benchmark.maximum_momentum_residual / benchmark.body_force < 2.0e-3;
    SCIENCE_REQUIRE(benchmark.result.converged == meets_scientific_limits,
                    "converged must reflect velocity-independent momentum and continuity "
                    "criteria; reported=" +
                        std::to_string(benchmark.result.converged) +
                        ", scientific=" + std::to_string(meets_scientific_limits));
}

}  // namespace

int main() {
    return science_test::runSuite(
        "steady Stokes flow",
        {{"construction and physical input validation", testConstructionAndPhysicalInputValidation},
         {"quiescent exact solution", testQuiescentExactSolution},
         {"sealed uniform-force hydrostatic equilibrium",
          testSealedUniformForceHydrostaticEquilibrium},
         {"plane Poiseuille accuracy and conservation", testPlanePoiseuilleAccuracyAndConservation},
         {"result diagnostics have documented scientific meaning",
          testResultDiagnosticsHaveDocumentedScientificMeaning},
         {"convergence flag matches scientific acceptance criteria",
          testConvergenceFlagMatchesScientificAcceptanceCriteria}});
}
