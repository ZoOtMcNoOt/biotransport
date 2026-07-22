#include "../test_support/science_test.hpp"
#include <algorithm>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/nonuniform_mesh_1d.hpp>
#include <biotransport/solvers/nonuniform_diffusion_1d.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

template <typename Exception, typename Callable>
void requireThrows(Callable&& callable, const std::string& message) {
    bool threw_expected = false;
    try {
        callable();
    } catch (const Exception&) {
        threw_expected = true;
    }
    SCIENCE_REQUIRE(threw_expected, message);
}

std::vector<double> smoothlyStretchedNodes(std::size_t cells) {
    std::vector<double> nodes(cells + 1, 0.0);
    constexpr double amplitude = 0.25;
    for (std::size_t i = 0; i <= cells; ++i) {
        const double coordinate = static_cast<double>(i) / static_cast<double>(cells);
        nodes[i] = coordinate + amplitude * std::sin(2.0 * kPi * coordinate) / (2.0 * kPi);
    }
    return nodes;
}

void meshGeometryUsesPositiveNodeCentredVolumes() {
    const biotransport::NonuniformMesh1D mesh({0.0, 0.1, 0.4, 1.0});

    SCIENCE_REQUIRE(mesh.numNodes() == 4, "mesh should retain all supplied nodes");
    SCIENCE_REQUIRE(mesh.numCells() == 3, "face count should be node count minus one");
    SCIENCE_REQUIRE_NEAR(mesh.spacing(1), 0.3, 1.0e-15, 0.0, "second face spacing");
    SCIENCE_REQUIRE_NEAR(mesh.faceCoordinate(1), 0.25, 1.0e-15, 0.0, "second face coordinate");

    const std::vector<double> expected{0.05, 0.20, 0.45, 0.30};
    double volume_sum = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        SCIENCE_REQUIRE_NEAR(mesh.controlVolume(i), expected[i], 1.0e-15, 0.0,
                             "node-centred control-volume width");
        SCIENCE_REQUIRE(mesh.controlVolume(i) > 0.0, "every control volume must be positive");
        volume_sum += mesh.controlVolume(i);
    }
    SCIENCE_REQUIRE_NEAR(volume_sum, mesh.length(), 2.0e-15, 0.0,
                         "control volumes partition the domain");
}

void uniformGridReducesToStandardFiniteVolumeStencil() {
    constexpr std::size_t cells = 10;
    constexpr double spacing = 0.1;
    constexpr double diffusivity = 0.2;
    constexpr double dt = 0.01;

    std::vector<double> nodes(cells + 1, 0.0);
    std::vector<double> initial(cells + 1, 0.0);
    for (std::size_t i = 0; i <= cells; ++i) {
        nodes[i] = spacing * static_cast<double>(i);
        initial[i] = 0.5 + 0.4 * std::sin(kPi * nodes[i]);
    }

    biotransport::NonuniformDiffusion1D solver(biotransport::NonuniformMesh1D(nodes), diffusivity);
    solver.setInitialCondition(initial);
    solver.step(dt);

    const double fourier = diffusivity * dt / (spacing * spacing);
    std::vector<double> expected(initial.size(), 0.0);
    expected.front() = initial.front() + 2.0 * fourier * (initial[1] - initial.front());
    for (std::size_t i = 1; i < cells; ++i) {
        expected[i] = initial[i] + fourier * (initial[i - 1] - 2.0 * initial[i] + initial[i + 1]);
    }
    expected.back() = initial.back() + 2.0 * fourier * (initial[cells - 1] - initial.back());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        SCIENCE_REQUIRE_NEAR(solver.solution()[i], expected[i], 3.0e-15, 2.0e-15,
                             "uniform-grid finite-volume parity");
    }
    SCIENCE_REQUIRE_NEAR(solver.maxStableTimeStep(), spacing * spacing / (2.0 * diffusivity),
                         2.0e-16, 2.0e-15, "uniform-grid Forward Euler limit");
}

double manufacturedSolutionError(std::size_t cells) {
    constexpr double diffusivity = 0.1;
    constexpr double final_time = 0.05;

    const std::vector<double> nodes = smoothlyStretchedNodes(cells);
    biotransport::NonuniformDiffusion1D solver(biotransport::NonuniformMesh1D(nodes), diffusivity);
    solver.setDirichletBoundary(biotransport::Boundary::Left, 0.0)
        .setDirichletBoundary(biotransport::Boundary::Right, 0.0);

    std::vector<double> initial(nodes.size(), 0.0);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        initial[i] = std::sin(kPi * nodes[i]);
    }
    solver.setInitialCondition(initial);
    solver.solveUntil(final_time, 0.2 * solver.maxStableTimeStep());

    const double decay = std::exp(-diffusivity * kPi * kPi * final_time);
    double squared_error = 0.0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const double exact = std::sin(kPi * nodes[i]) * decay;
        const double error = solver.solution()[i] - exact;
        squared_error += solver.mesh().controlVolume(i) * error * error;
    }
    SCIENCE_REQUIRE(solver.time() == final_time, "solveUntil must land on exact final time");
    return std::sqrt(squared_error / solver.mesh().length());
}

void manufacturedSolutionConvergesOnSmoothlyStretchedMeshes() {
    const double coarse = manufacturedSolutionError(20);
    const double medium = manufacturedSolutionError(40);
    const double fine = manufacturedSolutionError(80);
    const double first_ratio = coarse / medium;
    const double second_ratio = medium / fine;

    science_test::report("L2 error, 20 cells", coarse);
    science_test::report("L2 error, 40 cells", medium);
    science_test::report("L2 error, 80 cells", fine);
    science_test::report("20-to-40 error ratio", first_ratio);
    science_test::report("40-to-80 error ratio", second_ratio);

    SCIENCE_REQUIRE(coarse > medium && medium > fine,
                    "manufactured-solution error must decrease under refinement");
    SCIENCE_REQUIRE(first_ratio > 3.2 && second_ratio > 3.2,
                    "smooth nonuniform meshes should recover second-order spatial convergence");
}

void discontinuousDiffusivityMaintainsOneConservativeFaceFlux() {
    const std::vector<double> nodes{0.0, 0.12, 0.30, 0.55, 0.78, 1.0};
    const std::vector<double> diffusivity{1.0, 1.0, 1.0, 0.1, 0.1, 0.1};
    biotransport::NonuniformDiffusion1D solver(biotransport::NonuniformMesh1D(nodes), diffusivity);
    solver.setDirichletBoundary(biotransport::Boundary::Left, 1.0)
        .setDirichletBoundary(biotransport::Boundary::Right, 0.0);

    SCIENCE_REQUIRE_NEAR(solver.faceDiffusivities()[2], 2.0 / 11.0, 1.0e-15, 1.0e-15,
                         "harmonic diffusivity at the material interface");

    std::vector<double> resistance(solver.mesh().numCells(), 0.0);
    double total_resistance = 0.0;
    for (std::size_t face = 0; face < resistance.size(); ++face) {
        resistance[face] = solver.mesh().spacing(face) / solver.faceDiffusivities()[face];
        total_resistance += resistance[face];
    }

    std::vector<double> steady(nodes.size(), 1.0);
    double accumulated_resistance = 0.0;
    for (std::size_t node = 1; node < nodes.size(); ++node) {
        accumulated_resistance += resistance[node - 1];
        steady[node] = 1.0 - accumulated_resistance / total_resistance;
    }
    solver.setInitialCondition(steady);

    const auto initial_flux = solver.faceFluxes();
    const double expected_flux = 1.0 / total_resistance;
    for (double flux : initial_flux) {
        SCIENCE_REQUIRE_NEAR(flux, expected_flux, 2.0e-14, 2.0e-14,
                             "constant steady flux through discontinuous material");
    }

    solver.step(0.5 * solver.maxStableTimeStep());
    for (std::size_t i = 0; i < steady.size(); ++i) {
        SCIENCE_REQUIRE_NEAR(solver.solution()[i], steady[i], 2.0e-14, 2.0e-14,
                             "discrete steady state remains unchanged");
    }
}

void closedIrregularMeshConservesIntegratedMass() {
    const std::vector<double> nodes{0.0, 0.03, 0.11, 0.26, 0.50, 0.72, 1.0};
    const std::vector<double> diffusivity{0.20, 0.18, 0.16, 0.13, 0.10, 0.08, 0.07};
    const std::vector<double> initial{0.2, 1.1, 0.4, 0.9, 0.3, 0.7, 0.5};

    biotransport::NonuniformDiffusion1D solver(biotransport::NonuniformMesh1D(nodes), diffusivity);
    solver.setInitialCondition(initial);
    const double initial_mass = solver.totalMass();
    solver.solveUntil(0.1, 0.8 * solver.maxStableTimeStep());
    const auto diagnostics = solver.diagnostics();

    SCIENCE_REQUIRE_NEAR(diagnostics.total_mass, initial_mass, 5.0e-14, 5.0e-14,
                         "zero-Neumann integrated mass conservation");
    SCIENCE_REQUIRE_NEAR(diagnostics.cumulative_boundary_input, 0.0, 1.0e-16, 0.0,
                         "closed boundaries add no mass");
    SCIENCE_REQUIRE_NEAR(diagnostics.mass_balance_error, 0.0, 6.0e-14, 0.0,
                         "closed-domain mass-balance residual");
}

void outwardNormalBoundarySignsMatchFicksLaw() {
    constexpr double diffusivity = 0.2;
    constexpr double outward_derivative = 0.5;
    constexpr double dt = 1.0e-3;

    biotransport::NonuniformDiffusion1D solver(biotransport::NonuniformMesh1D({0.0, 0.2, 0.5, 1.0}),
                                               diffusivity);
    solver.setNeumannBoundary(biotransport::Boundary::Left, outward_derivative)
        .setNeumannBoundary(biotransport::Boundary::Right, 0.0)
        .setUniformInitialCondition(1.0);
    const double initial_mass = solver.totalMass();

    SCIENCE_REQUIRE_NEAR(solver.boundaryOutwardFlux(biotransport::Boundary::Left),
                         -diffusivity * outward_derivative, 1.0e-15, 0.0,
                         "left outward Fickian flux sign");
    solver.step(dt);
    const auto diagnostics = solver.diagnostics();
    const double expected_input = diffusivity * outward_derivative * dt;
    SCIENCE_REQUIRE_NEAR(diagnostics.total_mass - initial_mass, expected_input, 2.0e-15, 2.0e-14,
                         "positive outward derivative adds mass");
    SCIENCE_REQUIRE_NEAR(diagnostics.cumulative_boundary_input, expected_input, 1.0e-16, 2.0e-15,
                         "Neumann boundary input accounting");
    SCIENCE_REQUIRE_NEAR(diagnostics.mass_balance_error, 0.0, 2.0e-15, 0.0,
                         "Neumann mass-balance residual");

    biotransport::NonuniformDiffusion1D reservoir_solver(
        biotransport::NonuniformMesh1D({0.0, 0.5, 1.0}), 2.0);
    reservoir_solver.setDirichletBoundary(biotransport::Boundary::Left, 1.0)
        .setNeumannBoundary(biotransport::Boundary::Right, 0.0)
        .setInitialCondition({1.0, 0.5, 0.5});
    SCIENCE_REQUIRE_NEAR(
        reservoir_solver.boundaryOutwardFlux(biotransport::Boundary::Left), -2.0, 1.0e-15, 0.0,
        "Dirichlet reservoir flux is inward when boundary concentration is larger");
    const double reservoir_initial_mass = reservoir_solver.totalMass();
    reservoir_solver.step(0.01);
    const auto reservoir_diagnostics = reservoir_solver.diagnostics();
    SCIENCE_REQUIRE_NEAR(reservoir_diagnostics.total_mass - reservoir_initial_mass, 0.02, 2.0e-15,
                         2.0e-14, "Dirichlet reservoir supplies the conservative face input");
    SCIENCE_REQUIRE_NEAR(reservoir_diagnostics.cumulative_boundary_input, 0.02, 2.0e-15, 2.0e-14,
                         "Dirichlet reservoir input accounting");
    SCIENCE_REQUIRE_NEAR(reservoir_diagnostics.mass_balance_error, 0.0, 2.0e-15, 0.0,
                         "Dirichlet mass-balance residual");
}

void invalidInputsAndUnstableStepsFailLoudly() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    const double maximum = std::numeric_limits<double>::max();

    requireThrows<std::invalid_argument>([] { biotransport::NonuniformMesh1D mesh({0.0}); },
                                         "a one-node mesh must be rejected");
    requireThrows<std::invalid_argument>(
        [] { biotransport::NonuniformMesh1D mesh({0.0, 0.5, 0.5}); },
        "duplicate nodes must be rejected");
    requireThrows<std::invalid_argument>(
        [] { biotransport::NonuniformMesh1D mesh({0.0, 0.7, 0.6}); },
        "decreasing nodes must be rejected");
    requireThrows<std::invalid_argument>(
        [nan] { biotransport::NonuniformMesh1D mesh({0.0, nan, 1.0}); },
        "non-finite nodes must be rejected");
    requireThrows<std::invalid_argument>(
        [maximum] { biotransport::NonuniformMesh1D mesh({-maximum, maximum}); },
        "non-finite node spacing must be rejected");

    const biotransport::NonuniformMesh1D mesh({0.0, 0.2, 0.6, 1.0});
    requireThrows<std::invalid_argument>(
        [&mesh] { biotransport::NonuniformDiffusion1D solver(mesh, {0.1, 0.2}); },
        "diffusivity length mismatch must be rejected");
    requireThrows<std::invalid_argument>(
        [&mesh] { biotransport::NonuniformDiffusion1D solver(mesh, -0.1); },
        "negative diffusivity must be rejected");
    requireThrows<std::invalid_argument>(
        [&mesh, nan] { biotransport::NonuniformDiffusion1D solver(mesh, nan); },
        "non-finite diffusivity must be rejected");

    biotransport::NonuniformDiffusion1D solver(mesh, 0.1);
    requireThrows<std::invalid_argument>([&solver] { solver.setInitialCondition({1.0, 0.0}); },
                                         "initial-condition length mismatch must be rejected");
    requireThrows<std::invalid_argument>([&solver] { solver.setUniformInitialCondition(-1.0); },
                                         "negative concentration must be rejected");
    requireThrows<std::invalid_argument>(
        [&solver, infinity] { solver.setNeumannBoundary(biotransport::Boundary::Left, infinity); },
        "non-finite Neumann values must be rejected");
    requireThrows<std::invalid_argument>(
        [&solver] { solver.setDirichletBoundary(biotransport::Boundary::Bottom, 1.0); },
        "non-1D boundary identifiers must be rejected");
    requireThrows<std::invalid_argument>(
        [&solver] {
            solver.setBoundaryCondition(biotransport::Boundary::Left,
                                        biotransport::BoundaryCondition::Robin(1.0, 1.0, 1.0));
        },
        "unimplemented Robin data must be rejected rather than silently approximated");

    solver.setInitialCondition({0.1, 0.8, 0.3, 0.6});
    const std::vector<double> before = solver.solution();
    const double time_before = solver.time();
    const double unstable = std::nextafter(solver.maxStableTimeStep(), infinity);
    SCIENCE_REQUIRE(!solver.checkStability(unstable),
                    "a step above the certified CFL limit must report unstable");
    requireThrows<std::invalid_argument>([&solver, unstable] { solver.step(unstable); },
                                         "an unstable explicit step must throw");
    SCIENCE_REQUIRE(solver.solution() == before,
                    "rejected unstable step must leave concentration unchanged");
    SCIENCE_REQUIRE(solver.time() == time_before,
                    "rejected unstable step must leave time unchanged");

    solver.solveUntil(0.037, 0.9 * solver.maxStableTimeStep());
    SCIENCE_REQUIRE(solver.time() == 0.037, "absolute solve must land exactly on final_time");
    requireThrows<std::invalid_argument>([&solver] { solver.solveUntil(0.03, 0.01); },
                                         "backward solveUntil requests must be rejected");
}

}  // namespace

int main() {
    return science_test::runSuite(
        "conservative nonuniform 1D diffusion",
        {{"mesh geometry uses positive node-centred volumes",
          meshGeometryUsesPositiveNodeCentredVolumes},
         {"uniform grid reduces to the standard finite-volume stencil",
          uniformGridReducesToStandardFiniteVolumeStencil},
         {"manufactured solution converges on smoothly stretched meshes",
          manufacturedSolutionConvergesOnSmoothlyStretchedMeshes},
         {"discontinuous diffusivity maintains one conservative face flux",
          discontinuousDiffusivityMaintainsOneConservativeFaceFlux},
         {"closed irregular mesh conserves integrated mass",
          closedIrregularMeshConservesIntegratedMass},
         {"outward-normal boundary signs match Fick's law",
          outwardNormalBoundarySignsMatchFicksLaw},
         {"invalid inputs and unstable steps fail loudly",
          invalidInputsAndUnstableStepsFailLoudly}});
}
