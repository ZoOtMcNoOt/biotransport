/**
 * @file test_legacy_reaction_safety.cpp
 * @brief Fail-loud contracts for legacy explicit diffusion/reaction wrappers.
 */

#include "../test_support/science_test.hpp"
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/mesh/structured_mesh_3d.hpp>
#include <biotransport/solvers/diffusion_solver_3d.hpp>
#include <biotransport/solvers/diffusion_solvers.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace biotransport;

template <typename Exception, typename Function>
void requireThrows(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void requireSame(const std::vector<double>& actual, const std::vector<double>& expected,
                 const std::string& message) {
    SCIENCE_REQUIRE(actual.size() == expected.size(), message + " (size mismatch)");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        SCIENCE_REQUIRE_NEAR(actual[index], expected[index], 0.0, 0.0,
                             message + " at index " + std::to_string(index));
    }
}

template <typename Solver>
void setZeroFlux1D(Solver& solver) {
    solver.setNeumannBoundary(Boundary::Left, 0.0);
    solver.setNeumannBoundary(Boundary::Right, 0.0);
}

template <typename Solver>
void setZeroFlux3D(Solver& solver) {
    solver.setNeumannBoundary(Boundary3D::XMin, 0.0);
    solver.setNeumannBoundary(Boundary3D::XMax, 0.0);
    solver.setNeumannBoundary(Boundary3D::YMin, 0.0);
    solver.setNeumannBoundary(Boundary3D::YMax, 0.0);
    solver.setNeumannBoundary(Boundary3D::ZMin, 0.0);
    solver.setNeumannBoundary(Boundary3D::ZMax, 0.0);
}

void baseRejectsInvalidInitialAndBoundaryData() {
    StructuredMesh mesh(4, 0.0, 1.0);
    DiffusionSolver solver(mesh, 0.01);
    auto initial = std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 1.0);
    initial[2] = std::numeric_limits<double>::quiet_NaN();

    requireThrows<std::invalid_argument>([&] { solver.setInitialCondition(initial); },
                                         "Base accepted a NaN initial value");
    requireThrows<std::invalid_argument>(
        [&] {
            solver.setDirichletBoundary(Boundary::Left, std::numeric_limits<double>::quiet_NaN());
        },
        "Base accepted a NaN Dirichlet value");
    requireThrows<std::invalid_argument>(
        [&] {
            solver.setNeumannBoundary(Boundary::Right, std::numeric_limits<double>::infinity());
        },
        "Base accepted a non-finite Neumann value");
    requireThrows<std::invalid_argument>(
        [&] {
            solver.setBoundaryCondition(
                Boundary::Left,
                BoundaryCondition::Robin(1.0, std::numeric_limits<double>::quiet_NaN(), 0.0));
        },
        "Base accepted a NaN Robin coefficient");
}

void baseRejectsInvalidAndNonexistentBoundaryIdentifiers() {
    StructuredMesh mesh(4, 0.0, 1.0);
    DiffusionSolver solver(mesh, 0.01);

    requireThrows<std::invalid_argument>(
        [&] { solver.setDirichletBoundary(static_cast<Boundary>(-1), 0.0); },
        "Base accepted a negative Boundary enum");
    requireThrows<std::invalid_argument>(
        [&] { solver.setNeumannBoundary(static_cast<Boundary>(99), 0.0); },
        "Base accepted an out-of-range Boundary enum");
    requireThrows<std::invalid_argument>(
        [&] { solver.setDirichletBoundary(Boundary::Bottom, 0.0); },
        "Base accepted Bottom on a 1D mesh");
    requireThrows<std::invalid_argument>([&] { solver.setNeumannBoundary(Boundary::Top, 0.0); },
                                         "Base accepted Top on a 1D mesh");
}

void baseDiffusionRetainsResolvableUpdatesAcrossExtremeScales() {
    {
        constexpr double spacing = 1.0e155;
        StructuredMesh mesh(4, 0.0, 4.0 * spacing);
        DiffusionSolver solver(mesh, 1.0);
        std::vector<double> initial(static_cast<std::size_t>(mesh.numNodes()), 0.0);
        initial[2] = 1.0e300;
        solver.setInitialCondition(initial);
        solver.solve(1.0, 1);

        const auto& result = solver.solution();
        SCIENCE_REQUIRE(std::isfinite(result[1]) && std::isfinite(result[3]),
                        "large-spacing diffusion produced a non-finite neighbor update");
        SCIENCE_REQUIRE_NEAR(result[1], 1.0e-10, 0.0, 2.0e-14,
                             "large-spacing left-neighbor diffusion increment");
        SCIENCE_REQUIRE_NEAR(result[3], 1.0e-10, 0.0, 2.0e-14,
                             "large-spacing right-neighbor diffusion increment");
    }

    {
        constexpr double spacing = 1.0e-200;
        constexpr double dt = 1.0e-78;
        const double diffusivity = std::numeric_limits<double>::denorm_min();
        StructuredMesh mesh(4, 0.0, 4.0 * spacing);
        DiffusionSolver solver(mesh, diffusivity);
        std::vector<double> initial(static_cast<std::size_t>(mesh.numNodes()), 0.0);
        initial[2] = 1.0;
        solver.setInitialCondition(initial);
        solver.solve(dt, 1);

        constexpr double expected_lambda = 4.9406564584124654e-2;
        const auto& result = solver.solution();
        SCIENCE_REQUIRE_NEAR(result[1], expected_lambda, 0.0, 3.0e-14,
                             "subnormal-diffusivity left-neighbor update");
        SCIENCE_REQUIRE_NEAR(result[2], 1.0 - 2.0 * expected_lambda, 0.0, 3.0e-14,
                             "subnormal-diffusivity center update");
        SCIENCE_REQUIRE_NEAR(result[3], expected_lambda, 0.0, 3.0e-14,
                             "subnormal-diffusivity right-neighbor update");
    }

    {
        const double maximum = std::numeric_limits<double>::max();
        StructuredMesh mesh(2, 0.0, 2.0);
        DiffusionSolver solver(mesh, 0.5);
        solver.setInitialCondition({maximum, -maximum, maximum});
        solver.setDirichletBoundary(Boundary::Left, maximum);
        solver.setDirichletBoundary(Boundary::Right, maximum);
        solver.solve(1.0, 1);

        SCIENCE_REQUIRE_NEAR(solver.solution()[1], maximum, 0.0, 0.0,
                             "representable update after an overflowing raw increment");
    }
}

void baseUsesOutwardNeumannSignsOnBothSides() {
    StructuredMesh mesh(8, 0.0, 1.0);
    DiffusionSolver solver(mesh, 0.01);
    std::vector<double> exact(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int i = 0; i <= mesh.nx(); ++i)
        exact[static_cast<std::size_t>(i)] = mesh.x(i);
    solver.setInitialCondition(exact);
    solver.setNeumannBoundary(Boundary::Left, -1.0);
    solver.setNeumannBoundary(Boundary::Right, 1.0);
    solver.solve(0.01, 1);

    for (std::size_t index = 0; index < exact.size(); ++index) {
        SCIENCE_REQUIRE_NEAR(solver.solution()[index], exact[index], 2e-14, 0.0,
                             "outward Neumann sign at node " + std::to_string(index));
    }
}

void baseEnforcesRobinEquationAndRejectsSingularity() {
    StructuredMesh mesh(4, 0.0, 1.0);
    DiffusionSolver solver(mesh, 0.01);
    solver.setInitialCondition(std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 1.0));
    constexpr double left_a = 2.0;
    constexpr double left_b = 3.0;
    constexpr double left_c = 4.0;
    constexpr double right_a = 1.5;
    constexpr double right_b = -0.2;
    constexpr double right_c = 0.7;
    solver.setBoundaryCondition(Boundary::Left, BoundaryCondition::Robin(left_a, left_b, left_c));
    solver.setBoundaryCondition(Boundary::Right,
                                BoundaryCondition::Robin(right_a, right_b, right_c));
    solver.solve(0.01, 1);

    const auto& result = solver.solution();
    const double left_normal_derivative = (result[0] - result[1]) / mesh.dx();
    const double right_normal_derivative = (result[static_cast<std::size_t>(mesh.nx())] -
                                            result[static_cast<std::size_t>(mesh.nx() - 1)]) /
                                           mesh.dx();
    SCIENCE_REQUIRE_NEAR(left_a * result[0] + left_b * left_normal_derivative, left_c, 2e-14, 0.0,
                         "left Robin equation");
    SCIENCE_REQUIRE_NEAR(
        right_a * result[static_cast<std::size_t>(mesh.nx())] + right_b * right_normal_derivative,
        right_c, 2e-14, 0.0, "right Robin equation");

    DiffusionSolver singular(mesh, 0.01);
    const auto initial = std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 1.0);
    singular.setInitialCondition(initial);
    singular.setBoundaryCondition(Boundary::Left, BoundaryCondition::Robin(1.0, -mesh.dx(), 0.0));
    singular.setDirichletBoundary(Boundary::Right, 1.0);
    requireThrows<std::invalid_argument>([&] { singular.solve(0.01, 1); },
                                         "Base accepted a mesh-singular Robin boundary");
    requireSame(singular.solution(), initial, "Singular Robin rejection mutated public state");
}

void baseRejectsConflictingDirichletCorners() {
    StructuredMesh mesh(4, 3, 0.0, 1.0, 0.0, 1.0);
    DiffusionSolver solver(mesh, 0.01);
    const auto initial = std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 0.25);
    solver.setInitialCondition(initial);
    solver.setNeumannBoundary(Boundary::Right, 0.0);
    solver.setNeumannBoundary(Boundary::Top, 0.0);
    solver.setDirichletBoundary(Boundary::Left, 1.0);
    solver.setDirichletBoundary(Boundary::Bottom, 2.0);
    requireThrows<std::invalid_argument>([&] { solver.solve(0.01, 1); },
                                         "Base averaged conflicting corner traces");
    requireSame(solver.solution(), initial, "Corner conflict mutated public state");
}

void variableDiffusivityRejectsSizeAndNonfiniteEntriesBeforeConstruction() {
    StructuredMesh mesh(4, 0.0, 1.0);
    requireThrows<std::invalid_argument>(
        [&] { VariableDiffusionSolver solver(mesh, std::vector<double>{0.1}); },
        "Variable diffusivity accepted a wrong-sized field");

    auto field = std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 0.1);
    field[2] = std::numeric_limits<double>::quiet_NaN();
    requireThrows<std::invalid_argument>([&] { VariableDiffusionSolver solver(mesh, field); },
                                         "Variable diffusivity accepted NaN");
    field[2] = std::numeric_limits<double>::infinity();
    requireThrows<std::invalid_argument>([&] { VariableDiffusionSolver solver(mesh, field); },
                                         "Variable diffusivity accepted infinity");
    field[2] = -0.1;
    requireThrows<std::invalid_argument>([&] { VariableDiffusionSolver solver(mesh, field); },
                                         "Variable diffusivity accepted a negative entry");
    std::fill(field.begin(), field.end(), 0.0);
    requireThrows<std::invalid_argument>([&] { VariableDiffusionSolver solver(mesh, field); },
                                         "Variable diffusivity accepted an all-zero field");
}

void explicitFdRejectsUnsupportedCouplingsAndAllowsSignedPureDiffusion() {
    StructuredMesh mesh(4, 0.0, 1.0);
    const auto diffusivity_field =
        std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 0.01);
    ExplicitFD solver;

    TransportProblem advection_reaction(mesh);
    advection_reaction.diffusivity(0.01).velocity(0.1).constantSource(0.1).initialCondition(1.0);
    requireThrows<std::invalid_argument>([&] { solver.run(advection_reaction, 0.01); },
                                         "ExplicitFD silently dropped advection or reaction");

    TransportProblem advection_variable_diffusion(mesh);
    advection_variable_diffusion.diffusivityField(diffusivity_field)
        .velocity(0.1)
        .initialCondition(1.0);
    requireThrows<std::invalid_argument>(
        [&] { solver.run(advection_variable_diffusion, 0.01); },
        "ExplicitFD silently dropped advection or variable diffusivity");

    TransportProblem variable_diffusion_reaction(mesh);
    variable_diffusion_reaction.diffusivityField(diffusivity_field)
        .constantSource(0.1)
        .initialCondition(1.0);
    requireThrows<std::invalid_argument>(
        [&] { solver.run(variable_diffusion_reaction, 0.01); },
        "ExplicitFD silently dropped variable diffusivity or reaction");

    TransportProblem signed_diffusion(mesh);
    signed_diffusion.diffusivity(0.01).initialCondition(-0.25);
    const auto result = solver.run(signed_diffusion, 0.01);
    for (double value : result.solution) {
        SCIENCE_REQUIRE_NEAR(value, -0.25, 2e-15, 0.0,
                             "ExplicitFD pure diffusion preserves signed uniform state");
    }
}

void reactionConstructorsValidateCallableAndAllParameters() {
    StructuredMesh mesh(4, 0.0, 1.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    requireThrows<std::invalid_argument>(
        [&] {
            ReactionDiffusionSolver solver(mesh, 0.01, ReactionDiffusionSolver::ReactionFunction{});
        },
        "Generic reaction solver accepted an empty callback");
    requireThrows<std::invalid_argument>(
        [&] { LinearReactionDiffusionSolver solver(mesh, 0.01, nan); },
        "Linear reaction solver accepted NaN decay");
    requireThrows<std::invalid_argument>(
        [&] { LinearReactionDiffusionSolver solver(mesh, 0.01, -1.0); },
        "Linear reaction solver accepted negative decay");
    requireThrows<std::invalid_argument>(
        [&] { LogisticReactionDiffusionSolver solver(mesh, 0.01, inf, 1.0); },
        "Logistic solver accepted infinite growth");
    requireThrows<std::invalid_argument>(
        [&] { LogisticReactionDiffusionSolver solver(mesh, 0.01, 1.0, nan); },
        "Logistic solver accepted NaN capacity");
    requireThrows<std::invalid_argument>(
        [&] { LogisticReactionDiffusionSolver solver(mesh, 0.01, 1.0, 0.0); },
        "Logistic solver accepted zero capacity");
    requireThrows<std::invalid_argument>(
        [&] { MichaelisMentenReactionDiffusionSolver solver(mesh, 0.01, nan, 1.0); },
        "Michaelis-Menten solver accepted NaN Vmax");
    requireThrows<std::invalid_argument>(
        [&] { MichaelisMentenReactionDiffusionSolver solver(mesh, 0.01, 1.0, inf); },
        "Michaelis-Menten solver accepted infinite Km");
    requireThrows<std::invalid_argument>(
        [&] { ConstantSourceReactionDiffusionSolver solver(mesh, 0.01, nan); },
        "Constant-source solver accepted NaN source");

    const auto mask = std::vector<std::uint8_t>(static_cast<std::size_t>(mesh.numNodes()), 0);
    requireThrows<std::invalid_argument>(
        [&] {
            MaskedMichaelisMentenReactionDiffusionSolver solver(mesh, 0.01, 1.0, 1.0, mask, inf);
        },
        "Masked Michaelis-Menten solver accepted infinite pinned concentration");
    requireThrows<std::invalid_argument>(
        [&] {
            MaskedMichaelisMentenReactionDiffusionSolver solver(mesh, 0.01, 1.0, 1.0, mask, -0.1);
        },
        "Masked Michaelis-Menten solver accepted negative pinned concentration");
}

void concentrationModelsRejectNegativeInputsWithoutSubstitution() {
    StructuredMesh mesh(4, 0.0, 1.0);
    const auto negative = std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), -0.1);

    ReactionDiffusionSolver generic(mesh, 0.01, [](double, double, double, double) { return 0.0; });
    requireThrows<std::invalid_argument>([&] { generic.setInitialCondition(negative); },
                                         "Generic concentration mode accepted negative state");
    requireThrows<std::invalid_argument>(
        [&] { generic.setDirichletBoundary(Boundary::Left, -0.1); },
        "Generic concentration mode accepted negative Dirichlet data");

    MichaelisMentenReactionDiffusionSolver michaelis(mesh, 0.01, 1.0, 0.1);
    requireThrows<std::invalid_argument>([&] { michaelis.setInitialCondition(negative); },
                                         "Michaelis wrapper substituted for u=-Km");
    requireThrows<std::invalid_argument>(
        [&] { michaelis.setBoundaryCondition(Boundary::Left, BoundaryCondition::Dirichlet(-0.1)); },
        "Michaelis wrapper accepted negative Dirichlet data");

    ReactionDiffusionSolver signed_field(
        mesh, 0.01, [](double, double, double, double) { return 0.0; }, false);
    signed_field.setInitialCondition(negative);
    setZeroFlux1D(signed_field);
    signed_field.solve(0.01, 1);
    for (double value : signed_field.solution())
        SCIENCE_REQUIRE_NEAR(value, -0.1, 2e-15, 0.0, "explicit signed-field opt-out");
}

void genericCallbackFailuresAreFiniteAndTransactional() {
    StructuredMesh mesh(4, 0.0, 1.0);
    const auto initial = std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 1.0);

    ReactionDiffusionSolver nonfinite(mesh, 0.001, [](double, double, double, double) {
        return std::numeric_limits<double>::quiet_NaN();
    });
    nonfinite.setInitialCondition(initial);
    setZeroFlux1D(nonfinite);
    requireThrows<std::runtime_error>([&] { nonfinite.solve(0.1, 1); },
                                      "Generic solver accepted a NaN callback rate");
    requireSame(nonfinite.solution(), initial, "NaN callback rejection mutated state");

    ReactionDiffusionSolver throwing(mesh, 0.001, [](double, double, double, double) -> double {
        throw std::domain_error("callback failure");
    });
    throwing.setInitialCondition(initial);
    setZeroFlux1D(throwing);
    requireThrows<std::domain_error>([&] { throwing.solve(0.1, 1); },
                                     "Generic solver swallowed a callback exception");
    requireSame(throwing.solution(), initial, "Throwing callback mutated state");

    ReactionDiffusionSolver unsafe(mesh, 0.001,
                                   [](double u, double, double, double) { return -2.0 * u; });
    unsafe.setInitialCondition(initial);
    setZeroFlux1D(unsafe);
    requireThrows<std::runtime_error>([&] { unsafe.solve(1.0, 1); },
                                      "Generic solver accepted a reaction-negative update");
    requireSame(unsafe.solution(), initial, "Reaction positivity rejection mutated state");
}

void specializedWrappersEnforceReactionAwarePositivity() {
    StructuredMesh mesh(4, 0.0, 1.0);

    const auto high = std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 2.0);
    LogisticReactionDiffusionSolver logistic(mesh, 0.001, 10.0, 1.0);
    logistic.setInitialCondition(high);
    setZeroFlux1D(logistic);
    requireThrows<std::runtime_error>([&] { logistic.solve(0.2, 1); },
                                      "Logistic wrapper accepted a negative explicit update");
    requireSame(logistic.solution(), high, "Logistic rejection mutated state");

    const auto low = std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 0.1);
    MichaelisMentenReactionDiffusionSolver michaelis(mesh, 0.001, 10.0, 0.1);
    michaelis.setInitialCondition(low);
    setZeroFlux1D(michaelis);
    requireThrows<std::runtime_error>([&] { michaelis.solve(0.1, 1); },
                                      "Michaelis wrapper accepted a negative explicit update");
    requireSame(michaelis.solution(), low, "Michaelis rejection mutated state");

    ConstantSourceReactionDiffusionSolver sink(mesh, 0.001, -2.0);
    sink.setInitialCondition(low);
    setZeroFlux1D(sink);
    requireThrows<std::runtime_error>([&] { sink.solve(0.1, 1); },
                                      "Constant sink accepted a negative concentration update");
    requireSame(sink.solution(), low, "Constant-sink rejection mutated state");

    auto mask = std::vector<std::uint8_t>(static_cast<std::size_t>(mesh.numNodes()), 0);
    mask[0] = 1;
    MaskedMichaelisMentenReactionDiffusionSolver masked(mesh, 0.001, 10.0, 0.1, mask, 0.25);
    masked.setInitialCondition(low);
    setZeroFlux1D(masked);
    requireThrows<std::runtime_error>([&] { masked.solve(0.1, 1); },
                                      "Masked Michaelis wrapper accepted a negative update");
    requireSame(masked.solution(), low, "Masked Michaelis rejection mutated state");
}

void threeDimensionalReactionWrappersFailLoudlyAndAtomically() {
    StructuredMesh3D mesh(2, 2, 2, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    const auto initial = std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 1.0);
    const auto negative = std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), -1.0);

    requireThrows<std::invalid_argument>(
        [&] {
            ReactionDiffusionSolver3D solver(mesh, 0.001,
                                             ReactionDiffusionSolver3D::ReactionFunction{});
        },
        "3D generic solver accepted an empty callback");
    requireThrows<std::invalid_argument>(
        [&] {
            LinearReactionDiffusionSolver3D solver(mesh, 0.001,
                                                   std::numeric_limits<double>::quiet_NaN());
        },
        "3D linear solver accepted NaN decay");

    LinearReactionDiffusionSolver3D linear(mesh, 0.001, 1.0);
    requireThrows<std::invalid_argument>([&] { linear.setInitialCondition(negative); },
                                         "3D linear solver accepted negative concentration");

    ReactionDiffusionSolver3D nonfinite(mesh, 0.001, [](double, double, double, double, double) {
        return std::numeric_limits<double>::infinity();
    });
    nonfinite.setInitialCondition(initial);
    setZeroFlux3D(nonfinite);
    requireThrows<std::runtime_error>([&] { nonfinite.solve(0.1, 1); },
                                      "3D solver accepted an infinite callback rate");
    requireSame(nonfinite.solution(), initial, "3D nonfinite callback mutated state");

    ReactionDiffusionSolver3D unsafe(
        mesh, 0.001, [](double u, double, double, double, double) { return -2.0 * u; });
    unsafe.setInitialCondition(initial);
    setZeroFlux3D(unsafe);
    requireThrows<std::runtime_error>([&] { unsafe.solve(1.0, 1); },
                                      "3D solver accepted a reaction-negative update");
    requireSame(unsafe.solution(), initial, "3D positivity rejection mutated state");

    ReactionDiffusionSolver3D signed_field(
        mesh, 0.001, [](double, double, double, double, double) { return 0.0; }, false);
    signed_field.setInitialCondition(negative);
    setZeroFlux3D(signed_field);
    signed_field.solve(0.1, 1);
    for (double value : signed_field.solution())
        SCIENCE_REQUIRE_NEAR(value, -1.0, 2e-15, 0.0, "3D signed-field opt-out");
}

}  // namespace

int main() {
    return science_test::runSuite(
        "legacy reaction and boundary safety",
        {
            {"base rejects invalid initial/boundary data",
             baseRejectsInvalidInitialAndBoundaryData},
            {"base rejects invalid/nonexistent boundaries",
             baseRejectsInvalidAndNonexistentBoundaryIdentifiers},
            {"base scale-safe diffusion arithmetic",
             baseDiffusionRetainsResolvableUpdatesAcrossExtremeScales},
            {"base outward Neumann signs", baseUsesOutwardNeumannSignsOnBothSides},
            {"base Robin equation and singularity", baseEnforcesRobinEquationAndRejectsSingularity},
            {"base conflicting Dirichlet corner", baseRejectsConflictingDirichletCorners},
            {"variable diffusivity validation",
             variableDiffusivityRejectsSizeAndNonfiniteEntriesBeforeConstruction},
            {"ExplicitFD coupling rejection and signed diffusion",
             explicitFdRejectsUnsupportedCouplingsAndAllowsSignedPureDiffusion},
            {"reaction constructor validation",
             reactionConstructorsValidateCallableAndAllParameters},
            {"negative concentration input rejection",
             concentrationModelsRejectNegativeInputsWithoutSubstitution},
            {"generic callback failure atomicity",
             genericCallbackFailuresAreFiniteAndTransactional},
            {"specialized reaction positivity", specializedWrappersEnforceReactionAwarePositivity},
            {"3D reaction safety", threeDimensionalReactionWrappersFailLoudlyAndAtomically},
        });
}
