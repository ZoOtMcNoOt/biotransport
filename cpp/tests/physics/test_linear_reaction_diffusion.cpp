#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace biotransport;

void testLinearReactionDiffusionMatchesExponentialDecayWhenNoDiffusion() {
    std::cout << "Testing linear reaction-diffusion vs exp decay (D ~ 0)..." << std::endl;

    StructuredMesh mesh(100, 0.0, 1.0);

    const double D = 1e-12;
    const double k = 0.75;

    const double u0 = 1.2;
    std::vector<double> initial(mesh.numNodes(), u0);

    TransportProblem problem(mesh);
    problem.diffusivity(D)
        .linearDecay(k)
        .initialCondition(initial)
        .neumann(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 0.0);

    const double t_end = 5.0;  // dt=1e-3 * 5000 steps

    ExplicitFD solver;
    auto result = solver.safetyFactor(0.4).run(problem, t_end);

    const double u_exact = u0 * std::exp(-k * t_end);

    const auto& u = result.solution;
    const int mid = mesh.nx() / 2;

    assert(std::abs(u[mid] - u_exact) < 2e-3);

    std::cout << "Linear decay agreement test passed!" << std::endl;
}

int main() {
    testLinearReactionDiffusionMatchesExponentialDecayWhenNoDiffusion();

    std::cout << "All linear reaction-diffusion tests passed!" << std::endl;
    return 0;
}
