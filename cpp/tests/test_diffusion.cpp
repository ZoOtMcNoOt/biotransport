#include <biotransport/solvers/diffusion.hpp>
#include <biotransport/solvers/reaction_diffusion.hpp>
#include <biotransport/core/utils.hpp>
#include <vector>
#include <cassert>
#include <iostream>
#include <cmath>

// Test the diffusion solver with a known analytical solution
void testDiffusion1D() {
    std::cout << "Testing 1D diffusion solver..." << std::endl;
    
    // Create a 1D mesh
    biotransport::StructuredMesh mesh(100, 0.0, 1.0);
    
    // Create a diffusion solver
    double D = 0.01;  // diffusion coefficient
    biotransport::DiffusionSolver solver(mesh, D);
    
    // Set up initial condition (step function)
    std::vector<double> initial(mesh.numNodes(), 0.0);
    for (int i = 0; i <= mesh.nx(); ++i) {
        double x = mesh.x(i);
        initial[i] = (x >= 0.4 && x <= 0.6) ? 1.0 : 0.0;
    }
    solver.setInitialCondition(initial);
    
    // Set boundary conditions
    solver.setDirichletBoundary(0, 0.0);
    solver.setDirichletBoundary(1, 0.0);
    
    // Solve
    double dt = 0.0001;
    int num_steps = 1000;
    solver.solve(dt, num_steps);
    
    // Get the solution
    const auto& solution = solver.solution();
    
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
    biotransport::StructuredMesh mesh(100, 0.0, 1.0);
    
    // Define reaction term (decay)
    double k = 0.01;  // decay rate
    auto reaction = [k](double u, double x, double y, double t) {
        return -k * u;
    };
    
    // Create a reaction-diffusion solver
    double D = 0.01;  // diffusion coefficient
    biotransport::ReactionDiffusionSolver solver(mesh, D, reaction);
    
    // Set up initial condition (step function)
    std::vector<double> initial(mesh.numNodes(), 0.0);
    for (int i = 0; i <= mesh.nx(); ++i) {
        double x = mesh.x(i);
        initial[i] = (x >= 0.4 && x <= 0.6) ? 1.0 : 0.0;
    }
    solver.setInitialCondition(initial);
    
    // Set boundary conditions
    solver.setDirichletBoundary(0, 0.0);
    solver.setDirichletBoundary(1, 0.0);
    
    // Solve
    double dt = 0.0001;
    int num_steps = 1000;
    solver.solve(dt, num_steps);
    
    // Get the solution
    const auto& solution = solver.solution();
    
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