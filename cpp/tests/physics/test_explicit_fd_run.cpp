#include "../test_support/science_test.hpp"
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cmath>
#include <vector>

void testExplicitFDRunUsesStableDtAndPinsDirichlet() {
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
    SCIENCE_REQUIRE(result.stats.dt > 0.0, "chosen time step must be positive");
    SCIENCE_REQUIRE(result.stats.steps > 0, "solver must take at least one step");
    SCIENCE_REQUIRE_NEAR(result.stats.t_end, t_end, 1e-12, 0.0, "reported end time");

    // Boundary pins.
    SCIENCE_REQUIRE_NEAR(result.solution[mesh.index(0)], 0.0, 1e-12, 0.0,
                         "left Dirichlet boundary value");
    SCIENCE_REQUIRE_NEAR(result.solution[mesh.index(mesh.nx())], 0.0, 1e-12, 0.0,
                         "right Dirichlet boundary value");
}

int main() {
    return science_test::runSuite("ExplicitFD run",
                                  {{"stable dt and pinned Dirichlet boundaries",
                                    testExplicitFDRunUsesStableDtAndPinsDirichlet}});
}
