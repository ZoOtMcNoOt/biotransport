#include "../test_support/science_test.hpp"
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cmath>
#include <vector>

using namespace biotransport;

void testConstantSource1DMatchesODEForUniformField() {
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

    SCIENCE_REQUIRE_NEAR(u[mid], u_exact, 5e-3, 0.0,
                         "midpoint concentration vs constant-source ODE u0 + S*t");
}

int main() {
    return science_test::runSuite("constant-source reaction-diffusion",
                                  {{"uniform field matches constant-source ODE",
                                    testConstantSource1DMatchesODEForUniformField}});
}
