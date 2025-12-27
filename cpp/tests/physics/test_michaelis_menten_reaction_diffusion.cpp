#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace biotransport;

// Exact implicit form for Michaelis–Menten sink ODE:
//   du/dt = -Vmax * u/(Km + u)
// Integrating gives:
//   u + Km * ln(u) = u0 + Km * ln(u0) - Vmax * t
static double michaelisMentenExactU(double u0, double vmax, double km, double t) {
    const double c = u0 + km * std::log(u0) - vmax * t;

    // u(t) decreases monotonically for u0>0, vmax>0.
    double lo = 1e-15;
    double hi = u0;

    auto f = [km, c](double u) {
        return u + km * std::log(u) - c;
    };

    // Ensure bracket contains root.
    // f(lo) -> -inf, f(hi)=vmax*t>=0 so bracket should hold, but guard anyway.
    if (f(hi) < 0.0) {
        // If due to numerical issues, expand hi until f(hi) >= 0.
        hi = std::max(hi, 1.0);
        for (int k = 0; k < 100 && f(hi) < 0.0; ++k) {
            hi *= 2.0;
        }
    }

    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double fm = f(mid);
        if (fm > 0.0) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    return 0.5 * (lo + hi);
}

void testMichaelisMentenSink1DMatchesODEForUniformField() {
    std::cout << "Testing Michaelis–Menten reaction-diffusion (1D) vs ODE..." << std::endl;

    StructuredMesh mesh(200, 0.0, 1.0);

    const double D = 1e-12;  // effectively no diffusion
    const double vmax = 0.75;
    const double km = 0.4;

    const double u0 = 1.1;
    std::vector<double> initial(mesh.numNodes(), u0);

    TransportProblem problem(mesh);
    problem.diffusivity(D)
        .michaelisMenten(vmax, km)
        .initialCondition(initial)
        .neumann(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 0.0);

    const double t_end = 2.0;  // dt=5e-4 * 4000 steps

    ExplicitFD solver;
    auto result = solver.safetyFactor(0.4).run(problem, t_end);

    const double u_exact = michaelisMentenExactU(u0, vmax, km, t_end);

    const auto& u = result.solution;
    const int mid = mesh.nx() / 2;

    const double err = std::abs(u[mid] - u_exact);
    assert(err < 7e-3);

    assert(u[mid] > 0.0);
    assert(u[mid] < u0 + 1e-12);

    std::cout << "Michaelis–Menten ODE agreement test passed!" << std::endl;
}

int main() {
    testMichaelisMentenSink1DMatchesODEForUniformField();

    std::cout << "All Michaelis–Menten reaction-diffusion tests passed!" << std::endl;
    return 0;
}
