#include "../test_support/science_test.hpp"
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cmath>
#include <numeric>
#include <vector>

using namespace biotransport;

void testExplicitFDConstantSource1DUniformGrowth() {
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

    SCIENCE_REQUIRE(result.stats.dt > 0.0, "chosen time step must be positive");
    SCIENCE_REQUIRE(result.stats.steps > 0, "solver must take at least one step");
    SCIENCE_REQUIRE_NEAR(result.stats.t_end, t_end, 1e-15, 0.0, "reported end time");

    // Field should remain (approximately) uniform and increase by S*t.
    const auto& u = result.solution;
    for (double v : u) {
        SCIENCE_REQUIRE_NEAR(v, expected, 5e-2, 0.0, "nodal concentration vs S*t");
    }

    // Sanity: min/max metrics should match the field.
    SCIENCE_REQUIRE_NEAR(result.stats.u_min_final, expected, 5e-2, 0.0, "final minimum vs S*t");
    SCIENCE_REQUIRE_NEAR(result.stats.u_max_final, expected, 5e-2, 0.0, "final maximum vs S*t");

    // Average value should also match S*t.
    const double sum = std::accumulate(u.begin(), u.end(), 0.0);
    const double avg = sum / static_cast<double>(u.size());
    SCIENCE_REQUIRE_NEAR(avg, expected, 2e-2, 0.0, "spatial average vs S*t");
}

int main() {
    return science_test::runSuite(
        "ExplicitFD constant source",
        {{"uniform growth matches S*t", testExplicitFDConstantSource1DUniformGrowth}});
}
