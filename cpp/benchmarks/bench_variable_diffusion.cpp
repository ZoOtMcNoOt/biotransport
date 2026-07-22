/** Benchmark canonical conservative variable-coefficient diffusion. */

#include "bench_utils.hpp"
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/problems/transport_problem.hpp>
#include <biotransport/solvers/transport_solver.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using biotransport::Boundary;
using biotransport::SolveOptions;
using biotransport::StructuredMesh;
using biotransport::TransportProblem;
using biotransport::TransportResult;
using biotransport::bench::BenchmarkCase;
using biotransport::bench::BenchmarkOptions;
using biotransport::bench::BenchmarkResult;
using biotransport::bench::RunOutcome;

std::vector<double> gaussianInitialCondition(const StructuredMesh& mesh) {
    constexpr double centre = 0.5;
    constexpr double sigma = 0.125;
    std::vector<double> values(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const double delta_x = mesh.x(i) - centre;
            const double delta_y = mesh.y(i, j) - centre;
            values[static_cast<std::size_t>(mesh.index(i, j))] =
                std::exp(-(delta_x * delta_x + delta_y * delta_y) / (2.0 * sigma * sigma));
        }
    }
    return values;
}

std::vector<double> diffusivityField(const StructuredMesh& mesh, double minimum, double maximum) {
    std::vector<double> values(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const double fraction = static_cast<double>(i) / mesh.nx();
            values[static_cast<std::size_t>(mesh.index(i, j))] =
                minimum + fraction * (maximum - minimum);
        }
    }
    return values;
}

BenchmarkResult benchmarkSize(int cells_per_axis, const BenchmarkOptions& options) {
    constexpr double minimum_diffusivity = 0.005;
    constexpr double maximum_diffusivity = 0.020;
    constexpr double cfl_fraction = 0.2;
    constexpr double final_step_fraction = 0.5;
    constexpr double mass_tolerance = 5.0e-12;

    const StructuredMesh mesh(cells_per_axis, cells_per_axis, 0.0, 1.0, 0.0, 1.0);
    const std::vector<double> initial = gaussianInitialCondition(mesh);
    const std::vector<double> diffusivity =
        diffusivityField(mesh, minimum_diffusivity, maximum_diffusivity);
    const double time_step = cfl_fraction * mesh.dx() * mesh.dx() / (4.0 * maximum_diffusivity);
    const double final_time =
        time_step * (static_cast<double>(options.steps - 1) + final_step_fraction);

    BenchmarkCase workload;
    workload.name = "variable_diffusion_" + std::to_string(cells_per_axis) + "x" +
                    std::to_string(cells_per_axis);
    workload.description =
        "Closed 2D Gaussian diffusion with linearly varying D; mass conservation is checked";
    workload.implementation = "canonical TransportProblem conservative harmonic-face diffusion";
    workload.parallel_kernel = "none (canonical scalar transport is serial)";
    workload.openmp_effective_for_workload = false;
    workload.cells_x = cells_per_axis;
    workload.cells_y = cells_per_axis;
    workload.cell_count =
        static_cast<std::uint64_t>(cells_per_axis) * static_cast<std::uint64_t>(cells_per_axis);
    workload.node_count = static_cast<std::uint64_t>(mesh.numNodes());
    workload.species_count = 1;
    workload.step_count = options.steps;
    workload.maximum_time_step = time_step;
    workload.final_time = final_time;
    workload.parameters = {{"minimum_diffusivity", minimum_diffusivity},
                           {"maximum_diffusivity", maximum_diffusivity},
                           {"cfl_fraction", cfl_fraction},
                           {"final_step_fraction", final_step_fraction}};

    auto run = [&]() -> RunOutcome {
        TransportProblem problem(mesh);
        problem.diffusivityField(diffusivity)
            .initialCondition(initial)
            .neumann(Boundary::Left, 0.0)
            .neumann(Boundary::Right, 0.0)
            .neumann(Boundary::Bottom, 0.0)
            .neumann(Boundary::Top, 0.0);
        SolveOptions solve_options;
        solve_options.final_time = final_time;
        solve_options.time_step = time_step;
        solve_options.max_steps = static_cast<std::size_t>(options.steps);
        const TransportResult result = biotransport::solve(problem, solve_options);
        if (result.diagnostics.steps != static_cast<std::size_t>(options.steps)) {
            throw std::runtime_error("canonical solver did not execute the declared step count");
        }
        const double initial_mass = result.diagnostics.initial_mass;
        const double final_mass = result.diagnostics.final_mass;
        const double absolute_error = std::abs(final_mass - initial_mass);
        return {biotransport::bench::makeCorrectnessEvidence(
            "closed_boundary_trapezoidal_mass_conservation", initial_mass, final_mass,
            absolute_error, biotransport::bench::relativeError(initial_mass, final_mass),
            mass_tolerance, biotransport::bench::solutionChecksum(result.concentration))};
    };

    return biotransport::bench::runBenchmark(workload, options.warmup_runs, options.timed_runs,
                                             run);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const BenchmarkOptions options =
            biotransport::bench::parseOptions(argc, argv, "bench_variable_diffusion_results.json");
        if (options.show_help) {
            biotransport::bench::printUsage(
                argv[0], "Canonical variable-coefficient diffusion performance evidence");
            return 0;
        }

        std::vector<BenchmarkResult> results;
        results.reserve(options.sizes.size());
        for (int size : options.sizes) {
            results.push_back(benchmarkSize(size, options));
            biotransport::bench::printResult(results.back());
        }
        biotransport::bench::writeJson(results, options, "bench_variable_diffusion");
        std::cout << "JSON evidence written to " << options.output_path << '\n';
        return biotransport::bench::allCorrect(results) ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << '\n';
        return 1;
    }
}
