#include "../test_support/science_test.hpp"
#include <algorithm>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/utils.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cmath>
#include <vector>

using namespace biotransport;

// Test the diffusion solver with a known analytical solution
void testDiffusion1D() {
    // Create a 1D mesh
    StructuredMesh mesh(100, 0.0, 1.0);

    // Diffusion coefficient
    double D = 0.01;

    // Set up initial condition (step function)
    std::vector<double> initial(mesh.numNodes(), 0.0);
    for (int i = 0; i <= mesh.nx(); ++i) {
        double x = mesh.x(i);
        initial[i] = (x >= 0.4 && x <= 0.6) ? 1.0 : 0.0;
    }

    TransportProblem problem(mesh);
    problem.diffusivity(D)
        .initialCondition(initial)
        .dirichlet(Boundary::Left, 0.0)
        .dirichlet(Boundary::Right, 0.0);

    // Solve for t_end = dt * num_steps = 0.0001 * 1000 = 0.1
    double t_end = 0.1;

    ExplicitFD solver;
    auto result = solver.safetyFactor(0.4).run(problem, t_end);

    // Get the solution
    const auto& solution = result.solution;

    // Check that the solution is reasonable
    // - Mass conservation (approximately)
    double initial_mass = 0.0;
    double final_mass = 0.0;
    for (int i = 0; i <= mesh.nx(); ++i) {
        initial_mass += initial[i] * mesh.dx();
        final_mass += solution[i] * mesh.dx();
    }
    // Some loss expected due to boundaries
    SCIENCE_REQUIRE_NEAR(final_mass, initial_mass, 0.1, 0.0,
                         "final mass vs initial mass (Dirichlet walls)");

    // - Symmetry (since initial condition and BCs are symmetric)
    for (int i = 0; i <= mesh.nx() / 2; ++i) {
        SCIENCE_REQUIRE_NEAR(solution[i], solution[mesh.nx() - i], 1e-6, 0.0,
                             "mirror symmetry of diffusion solution at node " + std::to_string(i));
    }

    // - Peak value decreased from initial
    double max_val = 0.0;
    for (int i = 0; i <= mesh.nx(); ++i) {
        max_val = std::max(max_val, solution[i]);
    }
    SCIENCE_REQUIRE(max_val < 1.0, "diffusion must reduce the peak below the initial plateau");
}

// Test the reaction-diffusion solver with a simple decay reaction
void testReactionDiffusion1D() {
    // Create a 1D mesh
    StructuredMesh mesh(100, 0.0, 1.0);

    // Decay rate
    double k = 0.01;

    // Diffusion coefficient
    double D = 0.01;

    // Set up initial condition (step function)
    std::vector<double> initial(mesh.numNodes(), 0.0);
    for (int i = 0; i <= mesh.nx(); ++i) {
        double x = mesh.x(i);
        initial[i] = (x >= 0.4 && x <= 0.6) ? 1.0 : 0.0;
    }

    TransportProblem problem(mesh);
    problem.diffusivity(D)
        .linearDecay(k)
        .initialCondition(initial)
        .dirichlet(Boundary::Left, 0.0)
        .dirichlet(Boundary::Right, 0.0);

    // Solve for t_end = dt * num_steps = 0.0001 * 1000 = 0.1
    double t_end = 0.1;

    ExplicitFD solver;
    auto result = solver.safetyFactor(0.4).run(problem, t_end);

    // Get the solution
    const auto& solution = result.solution;

    // Check that the solution is reasonable
    // - Total mass should be less than initial due to decay
    double initial_mass = 0.0;
    double final_mass = 0.0;
    for (int i = 0; i <= mesh.nx(); ++i) {
        initial_mass += initial[i] * mesh.dx();
        final_mass += solution[i] * mesh.dx();
    }
    SCIENCE_REQUIRE(final_mass < initial_mass, "linear decay must reduce total mass");

    // - Symmetry (since initial condition and BCs are symmetric)
    for (int i = 0; i <= mesh.nx() / 2; ++i) {
        SCIENCE_REQUIRE_NEAR(
            solution[i], solution[mesh.nx() - i], 1e-6, 0.0,
            "mirror symmetry of reaction-diffusion solution at node " + std::to_string(i));
    }

    // - Peak value decreased from initial
    double max_val = 0.0;
    for (int i = 0; i <= mesh.nx(); ++i) {
        max_val = std::max(max_val, solution[i]);
    }
    SCIENCE_REQUIRE(max_val < 1.0,
                    "reaction-diffusion must reduce the peak below the initial plateau");
}

int main() {
    return science_test::runSuite(
        "1D diffusion",
        {{"1D diffusion", testDiffusion1D}, {"1D reaction-diffusion", testReactionDiffusion1D}});
}
