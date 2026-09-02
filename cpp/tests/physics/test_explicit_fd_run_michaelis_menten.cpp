#include "../test_support/science_test.hpp"
#include <algorithm>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cmath>
#include <numeric>
#include <vector>

using namespace biotransport;

static double solveMichaelisMentenODE(double u0, double vmax, double km, double t) {
    // Solve u' = -vmax * u/(km+u), u(0)=u0.
    // Implicit solution:
    //   u + km ln(u) = u0 + km ln(u0) - vmax t
    // Use Newton's method on g(u) = u + km ln(u) - rhs.

    if (u0 <= 0.0) {
        return 0.0;
    }

    const double rhs = u0 + km * std::log(u0) - vmax * t;

    double u = std::max(1e-12, u0 - vmax * t);  // reasonable initial guess
    for (int iter = 0; iter < 80; ++iter) {
        u = std::max(u, 1e-14);
        const double g = u + km * std::log(u) - rhs;
        const double gp = 1.0 + km / u;
        const double du = -g / gp;
        u += du;
        if (u < 0.0) {
            u = 0.5 * std::max(1e-14, u - du);
        }
        if (std::abs(du) < 1e-14) {
            break;
        }
    }

    return std::max(0.0, u);
}

void testExplicitFDMichaelisMenten1DUniformDecay() {
    StructuredMesh mesh(60, 0.0, 1.0);

    const double D = 0.1;
    const double vmax = 2.0;
    const double km = 0.3;
    const double t_end = 0.25;

    const double u0 = 1.2;
    const double expected = solveMichaelisMentenODE(u0, vmax, km, t_end);

    std::vector<double> initial(mesh.numNodes(), u0);

    TransportProblem problem(mesh);
    problem.diffusivity(D)
        .michaelisMenten(vmax, km)
        .initialCondition(initial)
        .neumann(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 0.0);

    ExplicitFD solver;
    const auto result = solver.safetyFactor(0.4).run(problem, t_end);

    SCIENCE_REQUIRE(result.stats.dt > 0.0, "chosen time step must be positive");
    SCIENCE_REQUIRE(result.stats.steps > 0, "solver must take at least one step");
    SCIENCE_REQUIRE_NEAR(result.stats.t_end, t_end, 1e-15, 0.0, "reported end time");

    // Field should remain (approximately) uniform and decay toward expected ODE solution.
    const auto& u = result.solution;
    for (double v : u) {
        SCIENCE_REQUIRE_NEAR(v, expected, 2e-2, 0.0, "nodal concentration vs Michaelis-Menten ODE");
    }

    // Summary stats should reflect decay.
    SCIENCE_REQUIRE(result.stats.u_max_final <= result.stats.u_max_initial + 1e-12,
                    "Michaelis-Menten decay must not raise the field maximum");
    SCIENCE_REQUIRE(result.stats.u_min_final >= -1e-12,
                    "Michaelis-Menten decay must not produce negative concentration");

    // Average matches expected.
    const double sum = std::accumulate(u.begin(), u.end(), 0.0);
    const double avg = sum / static_cast<double>(u.size());
    SCIENCE_REQUIRE_NEAR(avg, expected, 1e-2, 0.0, "spatial average vs Michaelis-Menten ODE");
}

int main() {
    return science_test::runSuite("ExplicitFD Michaelis-Menten",
                                  {{"uniform field matches Michaelis-Menten ODE",
                                    testExplicitFDMichaelisMenten1DUniformDecay}});
}
