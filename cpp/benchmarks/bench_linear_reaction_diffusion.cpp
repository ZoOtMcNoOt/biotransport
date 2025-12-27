/**
 * @file bench_linear_reaction_diffusion.cpp
 * @brief Benchmark for linear reaction-diffusion solver performance
 *
 * Tests LinearReactionDiffusionSolver (diffusion with first-order decay)
 * on grids of increasing size.
 */

#include "bench_utils.hpp"
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/physics/mass_transport/linear_reaction_diffusion.hpp>
#include <cmath>
#include <iostream>
#include <vector>

using namespace biotransport;
using namespace biotransport::bench;

// Generate initial condition: uniform concentration
std::vector<double> uniformIC(const StructuredMesh& mesh, double value) {
    return std::vector<double>(mesh.numNodes(), value);
}

BenchmarkResult benchmarkSize(int n, int num_warmup, int num_runs) {
    // Square grid n x n
    const double L = 1.0;
    const double D = 0.01;
    const double k = 0.1;  // Decay rate

    StructuredMesh mesh(n, n, 0.0, L, 0.0, L);

    // Stability condition: dt <= dx^2 / (4 * D)
    const double dx = mesh.dx();
    const double dt = 0.2 * dx * dx / (4.0 * D);  // 20% safety factor

    // Fixed number of steps for consistent comparison
    const int num_steps = 100;

    auto ic = uniformIC(mesh, 1.0);

    auto benchFunc = [&]() {
        LinearReactionDiffusionSolver solver(mesh, D, k);
        solver.setInitialCondition(ic);
        solver.setDirichletBoundary(Boundary::Left, 0.0);
        solver.setDirichletBoundary(Boundary::Right, 0.0);
        solver.setDirichletBoundary(Boundary::Bottom, 0.0);
        solver.setDirichletBoundary(Boundary::Top, 0.0);
        solver.solve(dt, num_steps);
    };

    std::string name = "linear_rxn_diff_" + std::to_string(n) + "x" + std::to_string(n);
    std::string desc = "Linear reaction-diffusion on " + std::to_string(n) + "x" +
                       std::to_string(n) + " grid, D=" + std::to_string(D) +
                       ", k=" + std::to_string(k);

    return runBenchmark(name, desc, n, n, num_steps, dt, num_warmup, num_runs, benchFunc);
}

int main(int argc, char* argv[]) {
    std::cout << "=================================================\n";
    std::cout << " BioTransport Linear Reaction-Diffusion Benchmark\n";
    std::cout << "=================================================\n";

#ifdef BIOTRANSPORT_HAS_OPENMP
    std::cout << "OpenMP: ENABLED\n";
#else
    std::cout << "OpenMP: DISABLED\n";
#endif

    const int num_warmup = 2;
    const int num_runs = 5;

    // Test various grid sizes
    std::vector<int> grid_sizes = {32, 64, 128, 256, 512};

    std::vector<BenchmarkResult> results;
    results.reserve(grid_sizes.size());

    for (int n : grid_sizes) {
        std::cout << "\nBenchmarking " << n << "x" << n << " grid...\n";
        auto result = benchmarkSize(n, num_warmup, num_runs);
        printResult(result);
        results.push_back(result);
    }

    // Write results to JSON
    writeJSON(results, "bench_linear_reaction_diffusion_results.json");

    // Summary table
    std::cout << "\n=== Summary (median times) ===\n";
    std::cout << std::setw(12) << "Grid" << std::setw(15) << "Time (ms)" << std::setw(18)
              << "Cells/s" << "\n";
    std::cout << std::string(45, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::setw(6) << r.nx << "x" << std::setw(4) << r.ny << std::fixed
                  << std::setprecision(2) << std::setw(15) << r.stats.median_ms << std::scientific
                  << std::setprecision(2) << std::setw(18) << r.cells_per_second << "\n";
    }

    return 0;
}
