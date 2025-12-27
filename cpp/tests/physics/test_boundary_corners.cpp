#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace biotransport;

void testNeumannSideDoesNotOverrideDirichletCorners() {
    std::cout << "Testing boundary corner precedence (Dirichlet top/bottom)..." << std::endl;

    StructuredMesh mesh(10, 10, 0.0, 1.0, 0.0, 1.0);

    std::vector<double> initial(mesh.numNodes(), 1.0);

    // Left/right are Neumann, top/bottom are Dirichlet.
    TransportProblem problem(mesh);
    problem.diffusivity(0.01)
        .initialCondition(initial)
        .neumann(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 0.0)
        .dirichlet(Boundary::Bottom, 2.5)
        .dirichlet(Boundary::Top, -1.0);

    // Run for a very short time (1 step)
    ExplicitFD solver;
    auto result = solver.safetyFactor(0.4).run(problem, 1e-4);

    const auto& u = result.solution;

    // Bottom corners should be bottom Dirichlet.
    assert(std::abs(u[mesh.index(0, 0)] - 2.5) < 1e-12);
    assert(std::abs(u[mesh.index(mesh.nx(), 0)] - 2.5) < 1e-12);

    // Top corners should be top Dirichlet.
    assert(std::abs(u[mesh.index(0, mesh.ny())] + 1.0) < 1e-12);
    assert(std::abs(u[mesh.index(mesh.nx(), mesh.ny())] + 1.0) < 1e-12);

    std::cout << "Boundary corner precedence test passed!" << std::endl;
}

int main() {
    testNeumannSideDoesNotOverrideDirichletCorners();

    std::cout << "All boundary corner tests passed!" << std::endl;
    return 0;
}
