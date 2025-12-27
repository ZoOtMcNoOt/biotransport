#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace biotransport;

static double logisticSolution(double u0, double r, double K, double t) {
    // u(t) = K / (1 + ((K-u0)/u0) * exp(-r t))
    if (u0 <= 0.0) {
        return 0.0;
    }
    const double a = (K - u0) / u0;
    return K / (1.0 + a * std::exp(-r * t));
}

void testExplicitFDLogistic1DUniformGrowth() {
    std::cout
        << "Testing ExplicitFD.run(TransportProblem with logisticGrowth) uniform logistic growth..."
        << std::endl;

    StructuredMesh mesh(80, 0.0, 1.0);

    const double D = 0.1;
    const double r = 3.0;
    const double K = 2.5;
    const double t_end = 0.2;

    const double u0 = 0.4;
    const double expected = logisticSolution(u0, r, K, t_end);

    std::vector<double> initial(mesh.numNodes(), u0);

    TransportProblem problem(mesh);
    problem.diffusivity(D)
        .logisticGrowth(r, K)
        .initialCondition(initial)
        .neumann(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 0.0);

    ExplicitFD solver;
    const auto result = solver.safetyFactor(0.4).run(problem, t_end);

    assert(result.stats.dt > 0.0);
    assert(result.stats.steps > 0);
    assert(std::abs(result.stats.t_end - t_end) < 1e-15);

    // Field should remain (approximately) uniform and match the logistic ODE.
    const auto& u = result.solution;
    for (double v : u) {
        assert(std::abs(v - expected) < 2e-2);
        assert(v >= -1e-12);
        assert(v <= K + 1e-6);
    }

    // Average matches expected.
    const double sum = std::accumulate(u.begin(), u.end(), 0.0);
    const double avg = sum / static_cast<double>(u.size());
    assert(std::abs(avg - expected) < 1e-2);

    std::cout << "ExplicitFD logistic run test passed!" << std::endl;
}

int main() {
    testExplicitFDLogistic1DUniformGrowth();
    return 0;
}
