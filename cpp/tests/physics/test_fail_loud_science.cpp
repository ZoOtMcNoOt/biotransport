#include "../test_support/science_test.hpp"
#include <biotransport/core/numerics/linear_algebra/sparse_matrix.hpp>
#include <biotransport/core/numerics/stability.hpp>
#include <biotransport/physics/fluid_dynamics/darcy_flow.hpp>
#include <biotransport/physics/fluid_dynamics/stokes.hpp>
#include <biotransport/solvers/advection_diffusion_solver.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace biotransport;

namespace {

template <typename Exception, typename Function>
bool throws(Function&& function) {
    try {
        function();
    } catch (const Exception&) {
        return true;
    }
    return false;
}

void testAdvectionContractsAndActualEquation() {
    StructuredMesh mesh(10, 0.0, 1.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();

    SCIENCE_REQUIRE(
        throws<std::invalid_argument>([&] { (void)AdvectionDiffusionSolver(mesh, 0.1, nan); }),
        "non-finite uniform velocity must be rejected");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] {
                        (void)AdvectionDiffusionSolver(mesh, 0.1, 1.0, 0.0, AdvectionScheme::QUICK);
                    }),
                    "unimplemented QUICK must be rejected at construction");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] {
                        (void)AdvectionDiffusionSolver(mesh, 0.1, 1.0, 0.0,
                                                       static_cast<AdvectionScheme>(999));
                    }),
                    "unknown advection enums must be rejected at construction");

    std::vector<double> vx(mesh.numNodes(), 0.0);
    std::vector<double> vy;
    vx[3] = nan;
    SCIENCE_REQUIRE(
        throws<std::invalid_argument>([&] { (void)AdvectionDiffusionSolver(mesh, 0.1, vx, vy); }),
        "non-finite velocity fields must be rejected");

    AdvectionDiffusionSolver scheme_solver(mesh, 0.1, 0.2, 0.0, AdvectionScheme::UPWIND);
    SCIENCE_REQUIRE(
        throws<std::invalid_argument>([&] { scheme_solver.setScheme(AdvectionScheme::QUICK); }),
        "QUICK must also be rejected by the scheme setter");
    SCIENCE_REQUIRE(throws<std::invalid_argument>(
                        [&] { scheme_solver.setScheme(static_cast<AdvectionScheme>(-7)); }),
                    "unknown schemes must also be rejected by the scheme setter");

    AdvectionDiffusionSolver unstable_central(mesh, 0.01, 1.0, 0.0, AdvectionScheme::CENTRAL);
    SCIENCE_REQUIRE(!unstable_central.isSchemeStable(),
                    "central advection above directional Pe=2 must be inadmissible");
    SCIENCE_REQUIRE(throws<std::logic_error>([&] { (void)unstable_central.maxTimeStep(); }),
                    "no stable-step recommendation may be issued for inadmissible central flow");

    // A constant scalar in a divergent prescribed velocity is stationary for
    // v.grad(C), but not for div(v*C). This verifies the specialized solver's
    // documented nonconservative advective form.
    std::vector<double> divergent_velocity(mesh.numNodes());
    std::vector<double> constant(mesh.numNodes(), 3.0);
    for (int i = 0; i <= mesh.nx(); ++i) {
        divergent_velocity[i] = mesh.x(i);
    }
    AdvectionDiffusionSolver advective_form(mesh, 1.0, divergent_velocity, {},
                                            AdvectionScheme::CENTRAL);
    advective_form.setInitialCondition(constant);
    advective_form.setNeumannBoundary(Boundary::Left, 0.0);
    advective_form.setNeumannBoundary(Boundary::Right, 0.0);
    advective_form.solve(0.5 * advective_form.maxTimeStep(1.0), 4);
    for (double value : advective_form.solution()) {
        SCIENCE_REQUIRE_NEAR(value, 3.0, 2.0e-14, 0.0,
                             "constant solution of the advective-form equation");
    }
}

void testCombinedExplicitStabilityBounds() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    constexpr double dx = 0.1;
    constexpr double diffusivity = 0.1;
    constexpr double velocity = 1.0;
    const double expected = 1.0 / (2.0 * diffusivity / (dx * dx) + velocity / dx);

    SCIENCE_REQUIRE_NEAR(
        stability::suggest_advection_diffusion_dt_1d(dx, diffusivity, velocity, 1.0), expected,
        2.0e-17, 2.0e-15, "combined 1D upwind-diffusion bound");

    StructuredMesh mesh(10, 0.0, 1.0);
    AdvectionDiffusionSolver solver(mesh, diffusivity, velocity, 0.0, AdvectionScheme::UPWIND);
    SCIENCE_REQUIRE_NEAR(solver.maxTimeStep(1.0), expected, 2.0e-17, 2.0e-15,
                         "specialized-solver combined depletion bound");
    solver.setInitialCondition(std::vector<double>(mesh.numNodes(), 0.0));
    solver.solve(expected, 1);
    AdvectionDiffusionSolver excessive(mesh, diffusivity, velocity, 0.0, AdvectionScheme::UPWIND);
    SCIENCE_REQUIRE(throws<std::runtime_error>([&] {
                        excessive.solve(
                            std::nextafter(expected, std::numeric_limits<double>::infinity()), 1);
                    }),
                    "a step above the combined bound must be rejected");

    constexpr double dy = 0.2;
    constexpr double vy = 0.5;
    const double expected_2d =
        1.0 / (2.0 * diffusivity * (1.0 / (dx * dx) + 1.0 / (dy * dy)) + velocity / dx + vy / dy);
    SCIENCE_REQUIRE_NEAR(
        stability::suggest_advection_diffusion_dt_2d(dx, dy, diffusivity, velocity, vy, 1.0),
        expected_2d, 2.0e-17, 2.0e-15, "combined 2D upwind-diffusion bound");
    SCIENCE_REQUIRE(
        throws<std::invalid_argument>(
            [&] { (void)stability::suggest_advection_diffusion_dt_1d(dx, diffusivity, nan, 1.0); }),
        "stability helpers must reject non-finite velocities");
}

void testStokesValidationAndFailureContract() {
    StructuredMesh mesh(6, 6, 0.0, 1.0, 0.0, 1.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { (void)StokesSolver(mesh, nan); }),
                    "Stokes viscosity must be finite");

    StokesSolver solver(mesh, 1.0);
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] {
                        solver.setVelocityBC(static_cast<Boundary>(99), VelocityBC::NoSlip());
                    }),
                    "Stokes boundary IDs must be validated");
    SCIENCE_REQUIRE(throws<std::invalid_argument>(
                        [&] { solver.setVelocityBC(Boundary::Left, VelocityBC::StressFree()); }),
                    "unsupported traction boundaries must fail when configured");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] {
                        solver.setVelocityBC(Boundary::Left, VelocityBC::Dirichlet(nan, 0.0));
                    }),
                    "non-finite velocity boundary data must be rejected");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { solver.setBodyForce(nan, 0.0); }),
                    "non-finite uniform forces must be rejected");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { solver.setTolerance(0.0); }),
                    "non-positive Stokes tolerance must be rejected");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { solver.setMaxIterations(0); }),
                    "non-positive Stokes iteration limits must be rejected");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { solver.setPressureRelaxation(nan); }),
                    "non-finite pressure relaxation must be rejected");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { solver.setVelocityRelaxation(1.1); }),
                    "velocity over-relaxation outside the implemented contract must be rejected");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { (void)solver.reynolds(-1.0, 1.0, 1.0); }),
                    "invalid Reynolds-number scales must be rejected");

    std::function<double(double, double)> empty;
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { solver.setBodyForce(empty, empty); }),
                    "empty body-force callbacks must be rejected");

    StokesSolver nonfinite_force(mesh, 1.0);
    nonfinite_force.setBodyForce([&](double, double) { return nan; },
                                 [](double, double) { return 0.0; });
    SCIENCE_REQUIRE(throws<std::domain_error>([&] { (void)nonfinite_force.solve(); }),
                    "non-finite callback-defined force fields must fail before iteration");

    StokesSolver unconverged(mesh, 1.0);
    unconverged.setVelocityBC(Boundary::Top, VelocityBC::Dirichlet(1.0, 0.0))
        .setTolerance(1.0e-14)
        .setMaxIterations(1);
    SCIENCE_REQUIRE(throws<std::runtime_error>([&] { (void)unconverged.solve(); }),
                    "Stokes iteration exhaustion must throw instead of returning false");

    StokesSolver quiescent(mesh, 1.0);
    quiescent.setTolerance(1.0e-12).setMaxIterations(2);
    const StokesResult result = quiescent.solve();
    SCIENCE_REQUIRE(result.converged, "every returned Stokes result must be converged");
    SCIENCE_REQUIRE_NEAR(result.residual, 0.0, 0.0, 0.0, "quiescent Stokes momentum residual");
}

void testDarcyGradientGaugeAndFailureContracts() {
    StructuredMesh mesh(8, 6, 0.0, 1.0, 0.0, 1.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();

    DarcyFlowSolver singular(mesh, 1.0);
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { (void)singular.solve(); }),
                    "all-Neumann Darcy pressure without a gauge must be rejected");
    SCIENCE_REQUIRE(
        throws<std::invalid_argument>([&] { singular.setNeumann(Boundary::Left, nan); }),
        "Darcy pressure-gradient data must be finite");
    SCIENCE_REQUIRE(throws<std::invalid_argument>(
                        [&] { singular.setDirichlet(static_cast<Boundary>(-1), 0.0); }),
                    "Darcy boundary IDs must be validated");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { singular.setOmega(nan); }),
                    "Darcy relaxation must be finite");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { singular.setTolerance(nan); }),
                    "Darcy tolerance must be finite");

    std::vector<double> invalid_guess(mesh.numNodes(), 0.0);
    invalid_guess[2] = nan;
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { singular.setInitialGuess(invalid_guess); }),
                    "Darcy initial pressure must be finite");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] {
                        singular.setInternalPressure(std::vector<std::uint8_t>(mesh.numNodes(), 0),
                                                     1.0);
                    }),
                    "an empty internal-pressure mask cannot provide a gauge");

    // p=x exactly satisfies Laplace(p)=0, left p=0, right dp/dn=+1,
    // and zero top/bottom outward gradients. Darcy velocity is therefore -xhat.
    std::vector<double> exact(mesh.numNodes());
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            exact[mesh.index(i, j)] = mesh.x(i);
        }
    }
    DarcyFlowSolver gradient(mesh, 1.0);
    gradient.setDirichlet(Boundary::Left, 0.0)
        .setNeumann(Boundary::Right, 1.0)
        .setNeumann(Boundary::Bottom, 0.0)
        .setNeumann(Boundary::Top, 0.0)
        .setInitialGuess(exact)
        .setTolerance(1.0e-12)
        .setMaxIterations(5);
    const DarcyFlowResult result = gradient.solve();
    SCIENCE_REQUIRE(result.converged, "every returned Darcy result must be converged");
    SCIENCE_REQUIRE_NEAR(result.pressure[mesh.index(mesh.nx(), mesh.ny() / 2)], 1.0, 2.0e-13, 0.0,
                         "positive right outward pressure gradient");
    SCIENCE_REQUIRE_NEAR(result.vx[mesh.index(mesh.nx() / 2, mesh.ny() / 2)], -1.0, 2.0e-12, 0.0,
                         "Darcy velocity from outward-gradient convention");

    DarcyFlowSolver pressure_drop(mesh, 1.0);
    pressure_drop.setDirichlet(Boundary::Left, 1.0)
        .setDirichlet(Boundary::Right, 0.0)
        .setNeumann(Boundary::Bottom, 0.0)
        .setNeumann(Boundary::Top, 0.0)
        .setTolerance(1.0e-10)
        .setMaxIterations(10000);
    const DarcyFlowResult drop_result = pressure_drop.solve();
    SCIENCE_REQUIRE(drop_result.converged,
                    "a standard pressure-drop problem must meet the defect tolerance");
    SCIENCE_REQUIRE(drop_result.residual <= 1.0e-10,
                    "reported Darcy defect must control the convergence decision");

    DarcyFlowSolver unconverged(mesh, 1.0);
    unconverged.setDirichlet(Boundary::Left, 1.0)
        .setDirichlet(Boundary::Right, 0.0)
        .setTolerance(1.0e-15)
        .setMaxIterations(1);
    SCIENCE_REQUIRE(throws<std::runtime_error>([&] { (void)unconverged.solve(); }),
                    "Darcy iteration exhaustion must throw instead of returning false");
}

void testSparseSolveContracts() {
#ifdef BIOTRANSPORT_ENABLE_EIGEN
    using namespace biotransport::linalg;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    SparseMatrix identity(2, 2);
    identity.addEntry(0, 0, 1.0);
    identity.addEntry(1, 1, 1.0);
    identity.finalize();

    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] {
                        (void)identity.solve({1.0, 2.0}, static_cast<SparseSolverType>(999));
                    }),
                    "unknown sparse solver enums must be rejected");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { (void)identity.solve({nan, 2.0}); }),
                    "non-finite sparse RHS values must be rejected");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] {
                        (void)identity.solve({1.0, 2.0}, SparseSolverType::SparseLU, 0.0, 10);
                    }),
                    "non-positive sparse tolerances must be rejected");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] {
                        (void)identity.solve({1.0, 2.0}, SparseSolverType::SparseLU, 1.0e-10, 0);
                    }),
                    "non-positive sparse iteration limits must be rejected");

    const auto solution = identity.solve({1.0, 2.0});
    SCIENCE_REQUIRE_NEAR(solution[0], 1.0, 0.0, 0.0, "finite sparse solution entry zero");
    SCIENCE_REQUIRE_NEAR(solution[1], 2.0, 0.0, 0.0, "finite sparse solution entry one");

    SparseMatrix invalid_entry(1, 1);
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { invalid_entry.addEntry(0, 0, nan); }),
                    "non-finite sparse coefficients must be rejected during assembly");

    SparseMatrix singular(2, 2);
    singular.addEntry(0, 0, 1.0);
    singular.addEntry(1, 0, 1.0);
    singular.finalize();
    SCIENCE_REQUIRE(throws<std::runtime_error>([&] { (void)singular.solve({1.0, 1.0}); }),
                    "failed direct solve status must be reported after factorization/solve");
#else
    SCIENCE_REQUIRE(true, "sparse solve contracts require the optional Eigen backend");
#endif
}

}  // namespace

int main() {
    return science_test::runSuite(
        "fail-loud numerical science contracts",
        {{"advection contracts and actual equation", testAdvectionContractsAndActualEquation},
         {"combined explicit stability bounds", testCombinedExplicitStabilityBounds},
         {"Stokes validation and failure contract", testStokesValidationAndFailureContract},
         {"Darcy gradient, gauge, and failure contracts",
          testDarcyGradientGaugeAndFailureContracts},
         {"sparse solve contracts", testSparseSolveContracts}});
}
