/**
 * @file bench_variable_diffusion.cpp
 * @brief Benchmark for variable diffusion solver performance
 *
 * Tests VariableDiffusionSolver (spatially varying D) on grids of increasing size.
 */

#include "bench_utils.hpp"
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/physics/mass_transport/variable_diffusion.hpp>
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

// Generate diffusivity field: varies smoothly across domain
std::vector<double> variableD(const StructuredMesh& mesh, double D_min, double D_max) {
    std::vector<double> D(mesh.numNodes());

    const int stride = mesh.nx() + 1;
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            // Linear variation in x direction
            const double t = static_cast<double>(i) / mesh.nx();
            D[j * stride + i] = D_min + t * (D_max - D_min);
        }
    }

    return D;
}

BenchmarkResult benchmarkSize(int n, int num_warmup, int num_runs) {
    // Square grid n x n
    const double L = 1.0;
    const double D_min = 0.005;
    const double D_max = 0.02;

    StructuredMesh mesh(n, n, 0.0, L, 0.0, L);

    // Stability condition based on max D: dt <= dx^2 / (4 * D_max)
    const double dx = mesh.dx();
    const double dt = 0.2 * dx * dx / (4.0 * D_max);  // 20% safety factor

    // Fixed number of steps for consistent comparison
    const int num_steps = 100;

    auto ic = gaussianIC(mesh, L);
    auto D = variableD(mesh, D_min, D_max);

    auto benchFunc = [&]() {
        VariableDiffusionSolver solver(mesh, D);
        solver.setInitialCondition(ic);
        solver.setDirichletBoundary(Boundary::Left, 0.0);
        solver.setDirichletBoundary(Boundary::Right, 0.0);
        solver.setDirichletBoundary(Boundary::Bottom, 0.0);
        solver.setDirichletBoundary(Boundary::Top, 0.0);
        solver.solve(dt, num_steps);
    };

    std::string name = "variable_diff_" + std::to_string(n) + "x" + std::to_string(n);
    std::string desc = "Variable diffusion on " + std::to_string(n) + "x" + std::to_string(n) +
                       " grid, D=[" + std::to_string(D_min) + ", " + std::to_string(D_max) + "]";

    return runBenchmark(name, desc, n, n, num_steps, dt, num_warmup, num_runs, benchFunc);
}

int main(int argc, char* argv[]) {
    std::cout << "==========================================\n";
    std::cout << " BioTransport Variable Diffusion Benchmark\n";
    std::cout << "==========================================\n";

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
    writeJSON(results, "bench_variable_diffusion_results.json");

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
