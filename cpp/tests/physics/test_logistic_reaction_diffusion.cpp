#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace biotransport;

static double logisticExact(double u0, double r, double K, double t) {
    // u(t) = K / (1 + (K/u0 - 1) * exp(-r t))
    const double a = (K / u0) - 1.0;
    return K / (1.0 + a * std::exp(-r * t));
}

void testLogisticReaction1DMatchesODEForUniformField() {
    std::cout << "Testing logistic reaction-diffusion (1D) vs logistic ODE..." << std::endl;

    StructuredMesh mesh(200, 0.0, 1.0);

    const double D = 1e-12;  // effectively no diffusion
    const double r = 1.25;
    const double K = 2.0;

    const double u0 = 0.2;
    std::vector<double> initial(mesh.numNodes(), u0);

    TransportProblem problem(mesh);
    problem.diffusivity(D)
        .logisticGrowth(r, K)
        .initialCondition(initial)
        .neumann(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 0.0);

    const double t_end = 2.0;  // dt=1e-3 * 2000 steps

    ExplicitFD solver;
    auto result = solver.safetyFactor(0.4).run(problem, t_end);

    const double u_exact = logisticExact(u0, r, K, t_end);

    const auto& u = result.solution;
    const int mid = mesh.nx() / 2;

    const double err = std::abs(u[mid] - u_exact);
    assert(err < 5e-3);

    // Should remain within physical bounds for these parameters.
    assert(u[mid] > 0.0);
    assert(u[mid] < K + 1e-3);

    std::cout << "Logistic ODE agreement test passed!" << std::endl;
}

int main() {
    testLogisticReaction1DMatchesODEForUniformField();

    std::cout << "All logistic reaction-diffusion tests passed!" << std::endl;
    return 0;
}
