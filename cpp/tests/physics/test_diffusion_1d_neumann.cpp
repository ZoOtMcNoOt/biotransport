#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace biotransport;

static double computeMass1D(const StructuredMesh& mesh, const std::vector<double>& u) {
    double mass = 0.0;
    for (int i = 0; i <= mesh.nx(); ++i) {
        mass += u[mesh.index(i)] * mesh.dx();
    }
    return mass;
}

void testDiffusion1DNeumannMassConservation() {
    std::cout << "Testing 1D diffusion (Neumann=0) mass conservation..." << std::endl;

    StructuredMesh mesh(200, 0.0, 1.0);

    const double D = 0.02;

    // Smooth deterministic initial condition
    constexpr double pi = 3.14159265358979323846;
    std::vector<double> initial(mesh.numNodes(), 0.0);
    for (int i = 0; i <= mesh.nx(); ++i) {
        const double x = mesh.x(i);
        initial[mesh.index(i)] = 1.0 + 0.5 * std::sin(2.0 * pi * x);
    }

    TransportProblem problem(mesh);
    problem.diffusivity(D)
        .initialCondition(initial)
        .neumann(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 0.0);

    const double m0 = computeMass1D(mesh, initial);

    // Run for approximately 500 steps with dt = 0.2 * dx^2 / (2*D)
    const double dt_estimate = 0.2 * (mesh.dx() * mesh.dx()) / (2.0 * D);
    const double t_end = dt_estimate * 500;

    ExplicitFD solver;
    auto result = solver.safetyFactor(0.4).run(problem, t_end);

    const double m1 = computeMass1D(mesh, result.solution);

    const double rel_err = std::abs(m1 - m0) / std::max(1e-12, std::abs(m0));
    assert(rel_err < 5e-3);

    std::cout << "1D Neumann mass conservation test passed!" << std::endl;
}

int main() {
    testDiffusion1DNeumannMassConservation();

    std::cout << "All 1D Neumann diffusion tests passed!" << std::endl;
    return 0;
}
