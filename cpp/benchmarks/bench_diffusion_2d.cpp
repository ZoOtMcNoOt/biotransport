/**
 * @file bench_diffusion_2d.cpp
 * @brief Benchmark for 2D diffusion solver performance
 *
 * Tests DiffusionSolver on grids of increasing size to measure scaling behavior.
 */

#include "bench_utils.hpp"
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/physics/mass_transport/diffusion.hpp>
#include <cmath>
#include <iostream>
#include <vector>

using namespace biotransport;
using namespace biotransport::bench;

// Generate initial condition: Gaussian pulse centered in domain
std::vector<double> gaussianIC(const StructuredMesh& mesh, double L) {
    std::vector<double> ic(mesh.numNodes(), 0.0);

    const double cx = L / 2.0;
    const double cy = L / 2.0;
    const double sigma = L / 8.0;
    const double sigma2 = sigma * sigma;

    const int stride = mesh.nx() + 1;
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const double x = i * mesh.dx();
            const double y = j * mesh.dy();
            const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            ic[j * stride + i] = std::exp(-r2 / (2.0 * sigma2));
        }
    }

    return ic;
}

BenchmarkResult benchmarkSize(int n, int num_warmup, int num_runs) {
    // Square grid n x n
    const double L = 1.0;
    const double D = 0.01;

    StructuredMesh mesh(n, n, 0.0, L, 0.0, L);

    // Stability condition: dt <= dx^2 / (4 * D)
    const double dx = mesh.dx();
    const double dt = 0.2 * dx * dx / (4.0 * D);  // 20% safety factor

    // Fixed number of steps for consistent comparison
    const int num_steps = 100;

    auto ic = gaussianIC(mesh, L);

    auto benchFunc = [&]() {
        DiffusionSolver solver(mesh, D);
        solver.setInitialCondition(ic);
        solver.setDirichletBoundary(Boundary::Left, 0.0);
        solver.setDirichletBoundary(Boundary::Right, 0.0);
        solver.setDirichletBoundary(Boundary::Bottom, 0.0);
        solver.setDirichletBoundary(Boundary::Top, 0.0);
        solver.solve(dt, num_steps);
    };

    std::string name = "diffusion_2d_" + std::to_string(n) + "x" + std::to_string(n);
    std::string desc = "2D diffusion on " + std::to_string(n) + "x" + std::to_string(n) +
                       " grid, D=" + std::to_string(D);

    return runBenchmark(name, desc, n, n, num_steps, dt, num_warmup, num_runs, benchFunc);
}

int main(int argc, char* argv[]) {
    std::cout << "======================================\n";
    std::cout << " BioTransport 2D Diffusion Benchmark\n";
    std::cout << "======================================\n";

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
    writeJSON(results, "bench_diffusion_2d_results.json");

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
