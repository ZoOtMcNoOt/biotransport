#include "../test_support/science_test.hpp"
#include <algorithm>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/physics/reactions.hpp>
#include <biotransport/solvers/diffusion_solvers.hpp>
#include <cmath>
#include <limits>
#include <vector>

using namespace biotransport;

namespace {

constexpr double kInitialConcentration = 1.1;
constexpr double kMaximumRate = 0.75;
constexpr double kMichaelisConstant = 0.4;
constexpr double kEndTime = 2.0;

// Positive root of u + Km ln(u) = u0 + Km ln(u0) - Vmax t.
double michaelisMentenExact(double initial, double maximum_rate, double michaelis_constant,
                            double time) {
    const double target = initial + michaelis_constant * std::log(initial) - maximum_rate * time;
    double lower = std::numeric_limits<double>::min();
    double upper = initial;

    for (int iteration = 0; iteration < 200; ++iteration) {
        const double midpoint = 0.5 * (lower + upper);
        const double residual = midpoint + michaelis_constant * std::log(midpoint) - target;
        if (residual > 0.0) {
            upper = midpoint;
        } else {
            lower = midpoint;
        }
    }
    return 0.5 * (lower + upper);
}

struct UniformRun {
    double concentration;
    double spatial_range;
};

UniformRun solveUniformMichaelisMenten(double dt) {
    StructuredMesh mesh(20, 0.0, 1.0);
    std::vector<double> initial(mesh.numNodes(), kInitialConcentration);

    ReactionDiffusionSolver solver(mesh, 1.0e-3,
                                   reactions::michaelisMenten(kMaximumRate, kMichaelisConstant));
    solver.setInitialCondition(initial);
    solver.setNeumannBoundary(Boundary::Left, 0.0);
    solver.setNeumannBoundary(Boundary::Right, 0.0);

    const int steps = static_cast<int>(std::llround(kEndTime / dt));
    SCIENCE_REQUIRE_NEAR(steps * dt, kEndTime, 1.0e-14, 0.0, "integrated end time");
    solver.solve(dt, steps);

    const auto& solution = solver.solution();
    const auto range = std::minmax_element(solution.begin(), solution.end());
    return {solution[mesh.nx() / 2], *range.second - *range.first};
}

void testUniformFieldMatchesMichaelisMentenOde() {
    const UniformRun coarse = solveUniformMichaelisMenten(0.02);
    const UniformRun fine = solveUniformMichaelisMenten(0.01);
    const double exact =
        michaelisMentenExact(kInitialConcentration, kMaximumRate, kMichaelisConstant, kEndTime);
    const double coarse_error = std::abs(coarse.concentration - exact);
    const double fine_error = std::abs(fine.concentration - exact);
    const double observed_order = std::log(coarse_error / fine_error) / std::log(2.0);
    const double fine_relative_error = fine_error / exact;

    science_test::report("exact concentration", exact);
    science_test::report("dt=0.02 absolute error", coarse_error);
    science_test::report("dt=0.01 absolute error", fine_error);
    science_test::report("observed temporal order", observed_order);

    SCIENCE_REQUIRE(fine.concentration > 0.0,
                    "resolved Michaelis-Menten consumption must preserve positivity");
    SCIENCE_REQUIRE(fine.concentration < kInitialConcentration,
                    "positive consumption rate must decrease concentration");
    SCIENCE_REQUIRE(fine_relative_error < 5.0e-3,
                    "dt=0.01 must resolve the analytical trajectory to <0.5% relative error; "
                    "actual=" +
                        science_test::number(fine_relative_error));
    SCIENCE_REQUIRE(observed_order > 0.9 && observed_order < 1.1,
                    "forward Euler should demonstrate first-order temporal convergence; actual=" +
                        science_test::number(observed_order));
    SCIENCE_REQUIRE(coarse.spatial_range < 1.0e-13 && fine.spatial_range < 1.0e-13,
                    "zero-flux diffusion must preserve a spatially uniform field");
}

}  // namespace

int main() {
    return science_test::runSuite("Michaelis-Menten reaction-diffusion",
                                  {{"uniform PDE reduction agrees with Michaelis-Menten ODE",
                                    testUniformFieldMatchesMichaelisMentenOde}});
}
