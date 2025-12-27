#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
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
    std::cout << "Testing ExplicitFD.run(TransportProblem with michaelisMenten) uniform MM decay..."
              << std::endl;

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

    assert(result.stats.dt > 0.0);
    assert(result.stats.steps > 0);
    assert(std::abs(result.stats.t_end - t_end) < 1e-15);

    // Field should remain (approximately) uniform and decay toward expected ODE solution.
    const auto& u = result.solution;
    for (double v : u) {
        assert(std::abs(v - expected) < 2e-2);
    }

    // Summary stats should reflect decay.
    assert(result.stats.u_max_final <= result.stats.u_max_initial + 1e-12);
    assert(result.stats.u_min_final >= -1e-12);

    // Average matches expected.
    const double sum = std::accumulate(u.begin(), u.end(), 0.0);
    const double avg = sum / static_cast<double>(u.size());
    assert(std::abs(avg - expected) < 1e-2);

    std::cout << "ExplicitFD Michaelis–Menten run test passed!" << std::endl;
}

int main() {
    testExplicitFDMichaelisMenten1DUniformDecay();
    return 0;
}
