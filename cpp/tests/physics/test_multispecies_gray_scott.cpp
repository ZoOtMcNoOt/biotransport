#include "../test_support/science_test.hpp"
#include <algorithm>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/physics/mass_transport/gray_scott.hpp>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef BIOTRANSPORT_ENABLE_OPENMP
#include <omp.h>
#endif

using namespace biotransport;

namespace {

void testHomogeneousKineticsOneStep() {
    StructuredMesh mesh(4, 3, 0.0, 4.0, 0.0, 3.0);
    GrayScottSolver solver(mesh, 0.0, 0.0, 0.04, 0.06);
    const std::size_t nodes = static_cast<std::size_t>(mesh.nx() * mesh.ny());
    std::vector<float> u(nodes, 0.8f);
    std::vector<float> v(nodes, 0.2f);
    constexpr double dt = 0.1;

    const GrayScottRunResult result = solver.simulate(u, v, 1, dt, 1, 10, 0.0, 2);

    const double uvv = 0.8 * 0.2 * 0.2;
    const double expected_u = 0.8 + dt * (-uvv + 0.04 * (1.0 - 0.8));
    const double expected_v = 0.2 + dt * (uvv - (0.04 + 0.06) * 0.2);
    const std::size_t final_offset = nodes;
    for (std::size_t p = 0; p < nodes; ++p) {
        SCIENCE_REQUIRE_NEAR(result.u_frames[final_offset + p], expected_u, 3.0e-8, 2.0e-7,
                             "homogeneous u one-step kinetics");
        SCIENCE_REQUIRE_NEAR(result.v_frames[final_offset + p], expected_v, 3.0e-8, 2.0e-7,
                             "homogeneous v one-step kinetics");
    }
    SCIENCE_REQUIRE(result.nx == mesh.nx() && result.ny == mesh.ny(),
                    "Gray-Scott output dimensions must equal periodic mesh cell counts");
    SCIENCE_REQUIRE_NEAR(result.final_time, dt, 0.0, 0.0, "Gray-Scott final time");
}

void testMeshSpacingAppearsInPeriodicLaplacian() {
    constexpr int cells = 16;
    constexpr double length = 8.0;
    constexpr double diffusivity = 0.1;
    constexpr double dt = 0.05;
    StructuredMesh mesh(cells, 4, 0.0, length, 0.0, 2.0);
    GrayScottSolver solver(mesh, diffusivity, 0.0, 0.0, 0.0);
    const std::size_t nodes = static_cast<std::size_t>(mesh.nx() * mesh.ny());
    std::vector<float> u(nodes);
    std::vector<float> v(nodes, 0.0f);
    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            u[static_cast<std::size_t>(j * mesh.nx() + i)] =
                static_cast<float>(1.0 + 0.1 * std::cos(2.0 * std::acos(-1.0) * i / cells));
        }
    }

    const auto result = solver.simulate(u, v, 1, dt, 1, 10, 0.0, 2);
    const double eigenvalue =
        -4.0 * std::pow(std::sin(std::acos(-1.0) / cells), 2) / (mesh.dx() * mesh.dx());
    const double amplification = 1.0 + dt * diffusivity * eigenvalue;
    const double expected_at_zero = 1.0 + 0.1 * amplification;
    SCIENCE_REQUIRE_NEAR(result.u_frames[nodes], expected_at_zero, 1.0e-7, 2.0e-7,
                         "periodic Fourier-mode amplification with physical dx");
}

void testPeriodicDiffusionConservesSumAndPositivity() {
    StructuredMesh mesh(12, 10, 0.0, 12.0, 0.0, 10.0);
    GrayScottSolver solver(mesh, 0.16, 0.0, 0.0, 0.0);
    const std::size_t nodes = static_cast<std::size_t>(mesh.nx() * mesh.ny());
    std::vector<float> u(nodes);
    std::vector<float> v(nodes, 0.0f);
    for (std::size_t p = 0; p < nodes; ++p) {
        u[p] = static_cast<float>(0.1 + 0.8 * static_cast<double>((37 * p + 11) % 101) / 100.0);
    }
    const double initial_sum = std::accumulate(u.begin(), u.end(), 0.0);

    const auto result = solver.simulate(u, v, 50, 0.5, 50, 100, 0.0, 2);
    const auto final_begin = result.u_frames.end() - static_cast<std::ptrdiff_t>(nodes);
    const double final_sum = std::accumulate(final_begin, result.u_frames.end(), 0.0);

    SCIENCE_REQUIRE_NEAR(final_sum, initial_sum, 2.0e-5, 5.0e-7,
                         "periodic diffusive concentration sum");
    SCIENCE_REQUIRE(*std::min_element(final_begin, result.u_frames.end()) >= 0.0f,
                    "admissible Gray-Scott diffusion must preserve positivity");
}

void testUnstableAndNonfiniteInputsFailLoudly() {
    StructuredMesh mesh(8, 8, 0.0, 0.8, 0.0, 0.8);
    GrayScottSolver solver(mesh, 0.16, 0.08, 0.04, 0.06);
    const std::size_t nodes = static_cast<std::size_t>(mesh.nx() * mesh.ny());
    std::vector<float> u(nodes, 1.0f);
    std::vector<float> v(nodes, 0.0f);

    bool rejected_dt = false;
    try {
        static_cast<void>(solver.simulate(u, v, 1, 1.0, 1, 1, 0.0, 1));
    } catch (const std::runtime_error&) {
        rejected_dt = true;
    }
    SCIENCE_REQUIRE(rejected_dt, "unstable Gray-Scott dt must be rejected instead of clipped");

    u[0] = std::numeric_limits<float>::quiet_NaN();
    bool rejected_state = false;
    try {
        static_cast<void>(solver.simulate(u, v, 1, 1.0e-3, 1, 1, 0.0, 1));
    } catch (const std::invalid_argument&) {
        rejected_state = true;
    }
    SCIENCE_REQUIRE(rejected_state, "non-finite Gray-Scott initial state must be rejected");
}

void testEarlyStopTracksBothSpecies() {
    StructuredMesh mesh(6, 5, 0.0, 6.0, 0.0, 5.0);
    GrayScottSolver solver(mesh, 0.16, 0.08, 0.04, 0.06);
    const std::size_t nodes = static_cast<std::size_t>(mesh.nx() * mesh.ny());
    std::vector<float> u(nodes, 1.0f);
    std::vector<float> v(nodes, 0.0f);

    const auto result = solver.simulate(u, v, 20, 0.5, 10, 1, 1.0e-8, 1);

    SCIENCE_REQUIRE(result.steps_run == 1,
                    "the exact homogeneous steady state should stop at the first check");
    SCIENCE_REQUIRE(result.frame_steps.back() == 1,
                    "an early-stop result must include the accepted final state");
    SCIENCE_REQUIRE_NEAR(result.final_time, 0.5, 0.0, 0.0, "early-stop final time");
}

void testZeroToleranceDisablesEarlyStop() {
    StructuredMesh mesh(6, 5, 0.0, 6.0, 0.0, 5.0);
    GrayScottSolver solver(mesh, 0.16, 0.08, 0.04, 0.06);
    const std::size_t nodes = static_cast<std::size_t>(mesh.nx() * mesh.ny());
    const std::vector<float> u(nodes, 1.0f);
    const std::vector<float> v(nodes, 0.0f);

    const auto result = solver.simulate(u, v, 4, 0.5, 3, 1, 0.0, 1);

    SCIENCE_REQUIRE(result.steps_run == 4, "stable_tol=0 must disable early termination");
    SCIENCE_REQUIRE(result.frame_steps.back() == 4,
                    "a complete run must save its final accepted state");
    SCIENCE_REQUIRE_NEAR(result.final_time, 2.0, 0.0, 0.0, "full-run final model time");
}

void testEarlyStopRequiresCurrentCheck() {
    StructuredMesh mesh(6, 5, 0.0, 6.0, 0.0, 5.0);
    GrayScottSolver solver(mesh, 0.16, 0.08, 0.04, 0.06);
    const std::size_t nodes = static_cast<std::size_t>(mesh.nx() * mesh.ny());
    std::vector<float> u(nodes, 1.0f);
    std::vector<float> v(nodes, 0.0f);
    u[0] = 0.5f;
    v[0] = 0.25f;

    const auto result = solver.simulate(u, v, 10, 0.5, 3, 2, 10.0, 2);

    SCIENCE_REQUIRE(result.steps_run == 4,
                    "early termination must occur on a current check, not a latched result");
    SCIENCE_REQUIRE(result.frame_steps.back() == 4,
                    "early termination must save the checked state");
}

void testTimeStepMustAdvanceFloatKernel() {
    StructuredMesh mesh(4, 4, 0.0, 4.0, 0.0, 4.0);
    GrayScottSolver solver(mesh, 0.16, 0.08, 0.04, 0.06);
    const std::size_t nodes = static_cast<std::size_t>(mesh.nx() * mesh.ny());
    const std::vector<float> u(nodes, 1.0f);
    const std::vector<float> v(nodes, 0.0f);
    const double underflowing_dt =
        0.25 * static_cast<double>(std::numeric_limits<float>::denorm_min());

    bool rejected = false;
    try {
        static_cast<void>(solver.simulate(u, v, 1, underflowing_dt, 1, 1, 0.0, 1));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    SCIENCE_REQUIRE(rejected,
                    "a positive dt that rounds to zero in the float kernel must be rejected");

    const double underflowing_parameter =
        0.25 * static_cast<double>(std::numeric_limits<float>::denorm_min());
    bool rejected_parameter = false;
    try {
        static_cast<void>(GrayScottSolver(mesh, underflowing_parameter, 0.08, 0.04, 0.06));
    } catch (const std::invalid_argument&) {
        rejected_parameter = true;
    }
    SCIENCE_REQUIRE(rejected_parameter,
                    "a positive coefficient that rounds to zero must be rejected");
}

#ifdef BIOTRANSPORT_ENABLE_OPENMP
GrayScottRunResult runGrayScottWithThreads(int threads) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    StructuredMesh mesh(24, 20, 0.0, 24.0, 0.0, 20.0);
    GrayScottSolver solver(mesh, 0.16, 0.08, 0.035, 0.06);
    const std::size_t nodes = static_cast<std::size_t>(mesh.nx() * mesh.ny());
    std::vector<float> u(nodes, 1.0f);
    std::vector<float> v(nodes, 0.0f);
    for (std::size_t p = 0; p < nodes; ++p) {
        if ((p % 29) < 3) {
            u[p] = 0.5f;
            v[p] = 0.25f;
        }
    }
    return solver.simulate(u, v, 80, 0.5, 40, 200, 0.0, 2);
}

void testOpenMpThreadDeterminism() {
    const auto serial = runGrayScottWithThreads(1);
    const auto parallel = runGrayScottWithThreads(4);
    SCIENCE_REQUIRE(serial.u_frames == parallel.u_frames && serial.v_frames == parallel.v_frames &&
                        serial.frame_steps == parallel.frame_steps,
                    "Gray-Scott output must be bitwise deterministic across OpenMP team sizes");
}
#endif

}  // namespace

int main() {
    return science_test::runSuite(
        "Gray-Scott reaction-diffusion",
        {
            {"homogeneous one-step kinetics are exact", testHomogeneousKineticsOneStep},
            {"mesh spacing scales the periodic Laplacian",
             testMeshSpacingAppearsInPeriodicLaplacian},
            {"periodic diffusion conserves concentration",
             testPeriodicDiffusionConservesSumAndPositivity},
            {"unstable and non-finite inputs fail loudly",
             testUnstableAndNonfiniteInputsFailLoudly},
            {"early-stop detection tracks both species", testEarlyStopTracksBothSpecies},
            {"zero steady tolerance disables early stop", testZeroToleranceDisablesEarlyStop},
            {"early stop requires a current stability check", testEarlyStopRequiresCurrentCheck},
            {"time step must advance the float kernel", testTimeStepMustAdvanceFloatKernel},
#ifdef BIOTRANSPORT_ENABLE_OPENMP
            {"OpenMP thread count does not change Gray-Scott output", testOpenMpThreadDeterminism},
#endif
        });
}
