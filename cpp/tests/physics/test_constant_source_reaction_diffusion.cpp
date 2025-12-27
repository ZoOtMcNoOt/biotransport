#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace biotransport;

void testConstantSource1DMatchesODEForUniformField() {
    std::cout << "Testing constant-source reaction-diffusion (1D) vs ODE..." << std::endl;

    StructuredMesh mesh(200, 0.0, 1.0);

    const double D = 1e-12;  // effectively no diffusion
    const double S = -0.3;   // negative = sink

    const double u0 = 1.25;
    std::vector<double> initial(mesh.numNodes(), u0);

    TransportProblem problem(mesh);
    problem.diffusivity(D)
        .constantSource(S)
        .initialCondition(initial)
        .neumann(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 0.0);

    const double t_end = 2.5;  // dt=1e-3 * 2500 steps

    ExplicitFD solver;
    auto result = solver.safetyFactor(0.4).run(problem, t_end);

    const double u_exact = u0 + S * t_end;

    const auto& u = result.solution;
    const int mid = mesh.nx() / 2;

    const double err = std::abs(u[mid] - u_exact);
    assert(err < 5e-3);

    std::cout << "Constant-source ODE agreement test passed!" << std::endl;
}

int main() {
    testConstantSource1DMatchesODEForUniformField();

    std::cout << "All constant-source reaction-diffusion tests passed!" << std::endl;
    return 0;
}
