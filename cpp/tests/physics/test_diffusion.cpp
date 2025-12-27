#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/utils.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace biotransport;

// Test the diffusion solver with a known analytical solution
void testDiffusion1D() {
    std::cout << "Testing 1D diffusion solver..." << std::endl;

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
    assert(std::abs(final_mass - initial_mass) < 0.1);  // Some loss expected due to boundaries

    // - Symmetry (since initial condition and BCs are symmetric)
    for (int i = 0; i <= mesh.nx() / 2; ++i) {
        assert(std::abs(solution[i] - solution[mesh.nx() - i]) < 1e-6);
    }

    // - Peak value decreased from initial
    double max_val = 0.0;
    for (int i = 0; i <= mesh.nx(); ++i) {
        max_val = std::max(max_val, solution[i]);
    }
    assert(max_val < 1.0);

    std::cout << "1D diffusion tests passed!" << std::endl;
}

// Test the reaction-diffusion solver with a simple decay reaction
void testReactionDiffusion1D() {
    std::cout << "Testing 1D reaction-diffusion solver..." << std::endl;

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
    assert(final_mass < initial_mass);  // Mass should decrease due to decay

    // - Symmetry (since initial condition and BCs are symmetric)
    for (int i = 0; i <= mesh.nx() / 2; ++i) {
        assert(std::abs(solution[i] - solution[mesh.nx() - i]) < 1e-6);
    }

    // - Peak value decreased from initial
    double max_val = 0.0;
    for (int i = 0; i <= mesh.nx(); ++i) {
        max_val = std::max(max_val, solution[i]);
    }
    assert(max_val < 1.0);

    std::cout << "1D reaction-diffusion tests passed!" << std::endl;
}

int main() {
    // Run tests
    testDiffusion1D();
    testReactionDiffusion1D();

    std::cout << "All diffusion tests passed!" << std::endl;
    return 0;
}
