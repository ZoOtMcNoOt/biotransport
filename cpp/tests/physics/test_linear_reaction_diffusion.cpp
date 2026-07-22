#include "../test_support/science_test.hpp"
#include <algorithm>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/diffusion_solvers.hpp>
#include <cmath>
#include <vector>

using namespace biotransport;

namespace {

constexpr double kInitialConcentration = 1.2;
constexpr double kDecayRate = 0.75;
constexpr double kEndTime = 5.0;

struct UniformRun {
    double concentration;
    double spatial_range;
};

UniformRun solveUniformLinearDecay(double dt) {
    StructuredMesh mesh(20, 0.0, 1.0);
    std::vector<double> initial(mesh.numNodes(), kInitialConcentration);

    LinearReactionDiffusionSolver solver(mesh, 1.0e-3, kDecayRate);
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

void testUniformFieldMatchesExponentialDecay() {
    // This solver treats decay with backward Euler. It is positivity-preserving and
    // unconditionally stable, but still only first-order accurate in time.
    const UniformRun coarse = solveUniformLinearDecay(0.01);
    const UniformRun fine = solveUniformLinearDecay(0.005);
    const double exact = kInitialConcentration * std::exp(-kDecayRate * kEndTime);
    const double coarse_error = std::abs(coarse.concentration - exact);
    const double fine_error = std::abs(fine.concentration - exact);
    const double observed_order = std::log(coarse_error / fine_error) / std::log(2.0);
    const double fine_relative_error = fine_error / exact;

    science_test::report("exact concentration", exact);
    science_test::report("dt=0.01 absolute error", coarse_error);
    science_test::report("dt=0.005 absolute error", fine_error);
    science_test::report("observed temporal order", observed_order);

    SCIENCE_REQUIRE(fine.concentration > 0.0,
                    "implicit decay must preserve positivity for positive initial data");
    SCIENCE_REQUIRE(fine.concentration < kInitialConcentration,
                    "positive decay rate must decrease concentration");
    SCIENCE_REQUIRE(fine_relative_error < 8.0e-3,
                    "dt=0.005 must resolve exponential decay to <0.8% relative error; actual=" +
                        science_test::number(fine_relative_error));
    SCIENCE_REQUIRE(observed_order > 0.9 && observed_order < 1.1,
                    "backward Euler should demonstrate first-order temporal convergence; actual=" +
                        science_test::number(observed_order));
    SCIENCE_REQUIRE(coarse.spatial_range < 1.0e-13 && fine.spatial_range < 1.0e-13,
                    "zero-flux diffusion must preserve a spatially uniform field");
}

}  // namespace

int main() {
    return science_test::runSuite("linear reaction-diffusion",
                                  {{"uniform PDE reduction agrees with exponential decay",
                                    testUniformFieldMatchesExponentialDecay}});
}
