#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace biotransport;

void testExplicitFDConstantSource1DUniformGrowth() {
    std::cout << "Testing ExplicitFD.run(TransportProblem with constantSource) uniform growth..."
              << std::endl;

    StructuredMesh mesh(50, 0.0, 1.0);

    const double D = 0.1;
    const double S = 2.0;
    const double t_end = 0.5;
    const double expected = S * t_end;

    std::vector<double> initial(mesh.numNodes(), 0.0);

    TransportProblem problem(mesh);
    problem.diffusivity(D)
        .constantSource(S)
        .initialCondition(initial)
        .neumann(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 0.0);

    ExplicitFD solver;
    const auto result = solver.safetyFactor(0.4).run(problem, t_end);

    assert(result.stats.dt > 0.0);
    assert(result.stats.steps > 0);
    assert(std::abs(result.stats.t_end - t_end) < 1e-15);

    // Field should remain (approximately) uniform and increase by S*t.
    const auto& u = result.solution;
    for (double v : u) {
        assert(std::abs(v - expected) < 5e-2);
    }

    // Sanity: min/max metrics should match the field.
    assert(std::abs(result.stats.u_min_final - expected) < 5e-2);
    assert(std::abs(result.stats.u_max_final - expected) < 5e-2);

    // Average value should also match S*t.
    const double sum = std::accumulate(u.begin(), u.end(), 0.0);
    const double avg = sum / static_cast<double>(u.size());
    assert(std::abs(avg - expected) < 2e-2);

    std::cout << "ExplicitFD constant-source run test passed!" << std::endl;
}

int main() {
    testExplicitFDConstantSource1DUniformGrowth();
    return 0;
}
