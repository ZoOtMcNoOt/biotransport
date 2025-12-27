#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void testExplicitFDRunUsesStableDtAndPinsDirichlet() {
    std::cout << "Testing ExplicitFD().run(DiffusionProblem, t_end)..." << std::endl;

    biotransport::StructuredMesh mesh(100, 0.0, 1.0);

    const double D = 1e-2;
    biotransport::DiffusionProblem problem(mesh);
    problem.diffusivity(D);

    std::vector<double> init(mesh.numNodes(), 1.0);
    problem.initialCondition(init);

    problem.dirichlet(biotransport::Boundary::Left, 0.0)
        .dirichlet(biotransport::Boundary::Right, 0.0);

    const double t_end = 0.1;
    biotransport::ExplicitFD runner;
    auto result = runner.run(problem, t_end);

    // dt should be positive and conservative.
    assert(result.stats.dt > 0.0);
    assert(result.stats.steps > 0);
    assert(std::abs(result.stats.t_end - t_end) < 1e-12);

    // Boundary pins.
    assert(std::abs(result.solution[mesh.index(0)] - 0.0) < 1e-12);
    assert(std::abs(result.solution[mesh.index(mesh.nx())] - 0.0) < 1e-12);

    std::cout << "ExplicitFD run test passed!" << std::endl;
}

int main() {
    testExplicitFDRunUsesStableDtAndPinsDirichlet();

    std::cout << "All ExplicitFD run tests passed!" << std::endl;
    return 0;
}
