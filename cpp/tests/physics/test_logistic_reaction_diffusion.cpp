#include "../test_support/science_test.hpp"
#include <algorithm>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/physics/reactions.hpp>
#include <biotransport/solvers/diffusion_solvers.hpp>
#include <cmath>
#include <vector>

using namespace biotransport;

namespace {

constexpr double kInitialConcentration = 0.2;
constexpr double kGrowthRate = 1.25;
constexpr double kCarryingCapacity = 2.0;
constexpr double kEndTime = 2.0;

double logisticExact(double initial, double growth_rate, double capacity, double time) {
    const double ratio = capacity / initial - 1.0;
    return capacity / (1.0 + ratio * std::exp(-growth_rate * time));
}

struct UniformRun {
    double concentration;
    double spatial_range;
};

UniformRun solveUniformLogistic(double dt) {
    StructuredMesh mesh(20, 0.0, 1.0);
    std::vector<double> initial(mesh.numNodes(), kInitialConcentration);

    ReactionDiffusionSolver solver(mesh, 1.0e-3,
                                   reactions::logistic(kGrowthRate, kCarryingCapacity));
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

void testUniformFieldMatchesLogisticOde() {
    // The PDE reduces exactly to the logistic ODE for a uniform field with zero-flux walls.
    // Explicit Euler is first-order in time, so halving dt should halve the global error.
    const UniformRun coarse = solveUniformLogistic(0.02);
    const UniformRun fine = solveUniformLogistic(0.01);
    const double exact =
        logisticExact(kInitialConcentration, kGrowthRate, kCarryingCapacity, kEndTime);
    const double coarse_error = std::abs(coarse.concentration - exact);
    const double fine_error = std::abs(fine.concentration - exact);
    const double observed_order = std::log(coarse_error / fine_error) / std::log(2.0);
    const double fine_relative_error = fine_error / exact;

    science_test::report("exact concentration", exact);
    science_test::report("dt=0.02 absolute error", coarse_error);
    science_test::report("dt=0.01 absolute error", fine_error);
    science_test::report("observed temporal order", observed_order);

    SCIENCE_REQUIRE(fine.concentration > kInitialConcentration,
                    "positive growth must increase concentration");
    SCIENCE_REQUIRE(fine.concentration < kCarryingCapacity,
                    "this trajectory must approach carrying capacity from below");
    SCIENCE_REQUIRE(fine_relative_error < 3.0e-3,
                    "dt=0.01 must resolve the analytical trajectory to <0.3% relative error; "
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
    return science_test::runSuite(
        "logistic reaction-diffusion",
        {{"uniform PDE reduction agrees with logistic ODE", testUniformFieldMatchesLogisticOde}});
}
