#include "../test_support/science_test.hpp"
#include <algorithm>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/multi_species_solver.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef BIOTRANSPORT_ENABLE_OPENMP
#include <omp.h>
#endif

using namespace biotransport;

namespace {

void testHomogeneousSirInvariant() {
    StructuredMesh mesh(12, 0.0, 1.0);
    MultiSpeciesSolver solver(mesh, {0.0, 0.0, 0.0});
    solver.setReactionModel(SIRReaction(0.3, 0.1, 1000.0));
    solver.setUniformInitialCondition(0, 990.0);
    solver.setUniformInitialCondition(1, 10.0);
    solver.setUniformInitialCondition(2, 0.0);

    solver.solve(0.01, 500);

    for (int node = 0; node < mesh.numNodes(); ++node) {
        SCIENCE_REQUIRE_NEAR(solver.totalConcentration(node), 1000.0, 2.0e-12, 2.0e-14,
                             "S+I+R at each homogeneous node");
    }
    const double integrated_population =
        solver.totalMass(0) + solver.totalMass(1) + solver.totalMass(2);
    SCIENCE_REQUIRE_NEAR(integrated_population, 1000.0, 2.0e-11, 2.0e-14,
                         "integrated SIR population");
    SCIENCE_REQUIRE_NEAR(solver.time(), 5.0, 1.0e-15, 0.0, "reported final time");
}

void testClosedDiffusionConservesTrapezoidalMass() {
    StructuredMesh mesh(40, 0.0, 1.0);
    MultiSpeciesSolver solver(mesh, {0.05});
    std::vector<double> initial(static_cast<std::size_t>(mesh.numNodes()));
    for (int i = 0; i <= mesh.nx(); ++i) {
        const double x = mesh.x(i);
        initial[static_cast<std::size_t>(i)] = 0.2 + std::exp(-120.0 * (x - 0.37) * (x - 0.37));
    }
    solver.setInitialCondition(0, initial);
    const double initial_mass = solver.totalMass(0);

    solver.solve(0.8 * solver.maxStableTimeStep(), 100);

    SCIENCE_REQUIRE_NEAR(solver.totalMass(0), initial_mass, 2.0e-13, 2.0e-13,
                         "closed-domain diffusive mass");
    SCIENCE_REQUIRE(*std::min_element(solver.solution(0).begin(), solver.solution(0).end()) >= 0.0,
                    "a stable diffusion step must preserve non-negative concentrations");
}

void testDiffusionLimitAndNeumannEigenmode() {
    constexpr int cells = 20;
    constexpr double length = 2.0;
    constexpr double diffusivity = 0.25;
    StructuredMesh mesh(cells, 0.0, length);
    MultiSpeciesSolver solver(mesh, {diffusivity});
    const double dx = mesh.dx();
    const double expected_limit = dx * dx / (2.0 * diffusivity);
    SCIENCE_REQUIRE_NEAR(solver.maxStableTimeStep(), expected_limit, 0.0, 4.0e-15,
                         "reported 1D diffusion CFL limit");
    SCIENCE_REQUIRE(solver.checkStability(expected_limit),
                    "the exact diffusion CFL limit should be admissible");
    SCIENCE_REQUIRE(!solver.checkStability(
                        std::nextafter(expected_limit, std::numeric_limits<double>::infinity())),
                    "a time step above the diffusion CFL limit must be rejected");

    std::vector<double> initial(static_cast<std::size_t>(mesh.numNodes()));
    for (int i = 0; i <= cells; ++i) {
        initial[static_cast<std::size_t>(i)] =
            1.0 + 0.1 * std::cos(std::acos(-1.0) * static_cast<double>(i) / cells);
    }
    solver.setInitialCondition(0, initial);
    constexpr double dt = 0.01;
    solver.solve(dt, 1);

    const double theta = std::acos(-1.0) / cells;
    const double eigenvalue = -4.0 * std::sin(0.5 * theta) * std::sin(0.5 * theta) / (dx * dx);
    const double amplification = 1.0 + dt * diffusivity * eigenvalue;
    for (int i = 0; i <= cells; ++i) {
        const double expected =
            1.0 + 0.1 * amplification * std::cos(std::acos(-1.0) * static_cast<double>(i) / cells);
        SCIENCE_REQUIRE_NEAR(solver.concentration(0, i), expected, 3.0e-15, 3.0e-15,
                             "Neumann cosine-mode amplification");
    }
}

void testDirichletValueParticipatesInFirstStep() {
    StructuredMesh mesh(4, 0.0, 1.0);
    MultiSpeciesSolver solver(mesh, {0.1});
    solver.setUniformInitialCondition(0, 0.0);
    solver.setDirichletBoundary(0, Boundary::Left, 1.0);
    constexpr double dt = 0.01;

    solver.solve(dt, 1);

    SCIENCE_REQUIRE_NEAR(solver.concentration(0, 0), 1.0, 0.0, 0.0,
                         "accepted state satisfies the fixed boundary");
    const double expected_first_interior = dt * 0.1 / (mesh.dx() * mesh.dx());
    SCIENCE_REQUIRE_NEAR(solver.concentration(0, 1), expected_first_interior, 1.0e-15, 1.0e-14,
                         "first step uses the prescribed boundary value");
}

void testOutwardDerivativeHasCorrectMassSign() {
    StructuredMesh mesh(20, 0.0, 1.0);
    constexpr double diffusivity = 0.2;
    constexpr double outward_derivative_left = 0.5;
    constexpr double dt = 1.0e-3;
    MultiSpeciesSolver solver(mesh, {diffusivity});
    solver.setUniformInitialCondition(0, 1.0);
    solver.setNeumannBoundary(0, Boundary::Left, outward_derivative_left);
    solver.setNeumannBoundary(0, Boundary::Right, 0.0);
    const double initial_mass = solver.totalMass(0);

    solver.solve(dt, 1);

    const double expected = initial_mass + dt * diffusivity * outward_derivative_left;
    SCIENCE_REQUIRE_NEAR(solver.totalMass(0), expected, 2.0e-15, 2.0e-14,
                         "mass change from prescribed outward derivative");
}

void testReactionLimitedStepIsAtomic() {
    StructuredMesh mesh(8, 0.0, 1.0);
    MultiSpeciesSolver solver(mesh, {0.0, 0.0});
    solver.setReactionModel(LotkaVolterraReaction(0.0, 1.0, 0.0, 0.0));
    solver.setUniformInitialCondition(0, 1.0);
    solver.setUniformInitialCondition(1, 10.0);

    bool rejected = false;
    try {
        solver.solve(1.0, 1);
    } catch (const std::runtime_error&) {
        rejected = true;
    }

    SCIENCE_REQUIRE(rejected, "a step beyond the reaction positivity limit must be rejected");
    SCIENCE_REQUIRE_NEAR(solver.time(), 0.0, 0.0, 0.0,
                         "a rejected first step must not advance time");
    for (double value : solver.solution(0)) {
        SCIENCE_REQUIRE_NEAR(value, 1.0, 0.0, 0.0,
                             "a rejected step must not mutate the accepted state");
    }
}

void testSolveUntilReachesAbsoluteFinalTime() {
    StructuredMesh mesh(10, 0.0, 1.0);
    MultiSpeciesSolver solver(mesh, {0.0, 0.0, 0.0});
    solver.setReactionModel(SIRReaction(0.3, 0.1, 1000.0));
    solver.setUniformInitialCondition(0, 990.0);
    solver.setUniformInitialCondition(1, 10.0);
    solver.setUniformInitialCondition(2, 0.0);

    solver.solveUntil(1.0, 0.3);
    SCIENCE_REQUIRE_NEAR(solver.time(), 1.0, 0.0, 0.0, "first absolute final time");
    solver.solveUntil(1.125, 0.2);
    SCIENCE_REQUIRE_NEAR(solver.time(), 1.125, 0.0, 0.0, "second absolute final time");
    for (int node = 0; node < mesh.numNodes(); ++node) {
        SCIENCE_REQUIRE_NEAR(solver.totalConcentration(node), 1000.0, 3.0e-13, 3.0e-15,
                             "SIR invariant after solveUntil");
    }

    bool rejected_reverse_time = false;
    try {
        solver.solveUntil(1.0, 0.1);
    } catch (const std::invalid_argument&) {
        rejected_reverse_time = true;
    }
    SCIENCE_REQUIRE(rejected_reverse_time, "solveUntil must reject reverse integration");

    MultiSpeciesSolver diffusive(mesh, {0.1});
    std::vector<double> profile(static_cast<std::size_t>(mesh.numNodes()));
    for (int i = 0; i <= mesh.nx(); ++i) {
        profile[static_cast<std::size_t>(i)] =
            0.2 + std::exp(-40.0 * (mesh.x(i) - 0.5) * (mesh.x(i) - 0.5));
    }
    diffusive.setInitialCondition(0, profile);
    const double initial_mass = diffusive.totalMass(0);
    diffusive.solveUntil(0.1, 1.0);
    SCIENCE_REQUIRE_NEAR(diffusive.time(), 0.1, 0.0, 0.0, "automatic CFL-limited final time");
    SCIENCE_REQUIRE_NEAR(diffusive.totalMass(0), initial_mass, 2.0e-15, 2.0e-14,
                         "automatic CFL subdivision preserves closed mass");
}

void testInvalidInputsFailBeforeIntegration() {
    StructuredMesh mesh(8, 0.0, 1.0);
    bool rejected_nan_diffusivity = false;
    try {
        MultiSpeciesSolver invalid(mesh, {std::numeric_limits<double>::quiet_NaN()});
    } catch (const std::invalid_argument&) {
        rejected_nan_diffusivity = true;
    }
    SCIENCE_REQUIRE(rejected_nan_diffusivity, "NaN diffusivity must be rejected");

    MultiSpeciesSolver solver(mesh, {0.0});
    std::vector<double> invalid(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    invalid[3] = std::numeric_limits<double>::infinity();
    bool rejected_nonfinite_state = false;
    try {
        solver.setInitialCondition(0, invalid);
    } catch (const std::invalid_argument&) {
        rejected_nonfinite_state = true;
    }
    SCIENCE_REQUIRE(rejected_nonfinite_state, "non-finite initial state must be rejected");

    MultiSpeciesSolver wrong_arity(mesh, {0.0, 0.0});
    bool rejected_sir_arity = false;
    try {
        wrong_arity.setReactionModel(SIRReaction(0.3, 0.1, 1000.0));
    } catch (const std::invalid_argument&) {
        rejected_sir_arity = true;
    }
    SCIENCE_REQUIRE(rejected_sir_arity,
                    "built-in reaction arity must be rejected before integration");
}

#ifdef BIOTRANSPORT_ENABLE_OPENMP
std::vector<std::vector<double>> runThreadedBrusselator(int threads) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    StructuredMesh mesh(24, 20, 0.0, 2.4, 0.0, 2.0);
    MultiSpeciesSolver solver(mesh, {0.01, 0.02});
    solver.setReactionModel(BrusselatorReaction(1.0, 1.5));
    std::vector<double> x(static_cast<std::size_t>(mesh.numNodes()));
    std::vector<double> y(static_cast<std::size_t>(mesh.numNodes()));
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const int p = mesh.index(i, j);
            x[static_cast<std::size_t>(p)] = 1.0 + 0.01 * std::sin(0.7 * i + 0.3 * j);
            y[static_cast<std::size_t>(p)] = 1.5 + 0.01 * std::cos(0.2 * i - 0.4 * j);
        }
    }
    solver.setInitialCondition(0, x);
    solver.setInitialCondition(1, y);
    solver.solve(0.002, 40);
    return solver.allSolutions();
}

void testOpenMpThreadDeterminism() {
    const auto serial = runThreadedBrusselator(1);
    const auto parallel = runThreadedBrusselator(4);
    SCIENCE_REQUIRE(
        serial == parallel,
        "built-in multi-species reactions must be bitwise deterministic across OpenMP team sizes");
}
#endif

}  // namespace

int main() {
    return science_test::runSuite(
        "conservative multi-species reaction-diffusion",
        {
            {"homogeneous SIR conserves population", testHomogeneousSirInvariant},
            {"closed diffusion conserves trapezoidal mass",
             testClosedDiffusionConservesTrapezoidalMass},
            {"diffusion CFL and Neumann eigenmode are correct",
             testDiffusionLimitAndNeumannEigenmode},
            {"Dirichlet data participates in the first step",
             testDirichletValueParticipatesInFirstStep},
            {"outward derivative has the documented mass sign",
             testOutwardDerivativeHasCorrectMassSign},
            {"reaction-limited rejection is atomic", testReactionLimitedStepIsAtomic},
            {"solveUntil reaches exact absolute times", testSolveUntilReachesAbsoluteFinalTime},
            {"invalid numerical inputs fail before integration",
             testInvalidInputsFailBeforeIntegration},
#ifdef BIOTRANSPORT_ENABLE_OPENMP
            {"OpenMP thread count does not change results", testOpenMpThreadDeterminism},
#endif
        });
}
