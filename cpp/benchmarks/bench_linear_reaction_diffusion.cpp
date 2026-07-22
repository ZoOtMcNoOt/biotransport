/** Benchmark canonical scalar linear reaction-diffusion. */

#include "bench_utils.hpp"
#include <algorithm>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/problems/transport_problem.hpp>
#include <biotransport/solvers/transport_solver.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
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

BenchmarkResult benchmarkSize(int cells_per_axis, const BenchmarkOptions& options) {
    constexpr double diffusivity = 0.01;
    constexpr double decay_rate = 0.1;
    constexpr double initial_concentration = 1.0;
    constexpr double cfl_fraction = 0.2;
    constexpr double final_step_fraction = 0.5;
    constexpr double solution_tolerance = 5.0e-12;

    const StructuredMesh mesh(cells_per_axis, cells_per_axis, 0.0, 1.0, 0.0, 1.0);
    const double time_step = cfl_fraction * mesh.dx() * mesh.dx() / (4.0 * diffusivity);
    const double final_time =
        time_step * (static_cast<double>(options.steps - 1) + final_step_fraction);
    const double expected_concentration =
        initial_concentration * std::pow(1.0 - decay_rate * time_step, options.steps - 1) *
        (1.0 - decay_rate * final_step_fraction * time_step);

    BenchmarkCase workload;
    workload.name = "linear_reaction_diffusion_" + std::to_string(cells_per_axis) + "x" +
                    std::to_string(cells_per_axis);
    workload.description =
        "Uniform closed-domain diffusion with first-order decay; discrete Euler decay is checked";
    workload.implementation = "canonical TransportProblem plus conservative solve";
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
    workload.parameters = {{"diffusivity", diffusivity},
                           {"decay_rate", decay_rate},
                           {"initial_concentration", initial_concentration},
                           {"cfl_fraction", cfl_fraction},
                           {"final_step_fraction", final_step_fraction}};

    auto run = [&]() -> RunOutcome {
        TransportProblem problem(mesh);
        problem.diffusivity(diffusivity)
            .linearDecay(decay_rate)
            .initialCondition(initial_concentration)
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

        const double mean =
            std::accumulate(result.concentration.begin(), result.concentration.end(), 0.0) /
            static_cast<double>(result.concentration.size());
        double maximum_error = 0.0;
        for (double value : result.concentration) {
            maximum_error = std::max(maximum_error, std::abs(value - expected_concentration));
        }
        const double relative_error =
            maximum_error / std::max(std::abs(expected_concentration), 1.0e-300);
        return {biotransport::bench::makeCorrectnessEvidence(
            "uniform_field_matches_discrete_forward_euler_linear_decay", expected_concentration,
            mean, maximum_error, relative_error, solution_tolerance,
            biotransport::bench::solutionChecksum(result.concentration))};
    };

    return biotransport::bench::runBenchmark(workload, options.warmup_runs, options.timed_runs,
                                             run);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const BenchmarkOptions options = biotransport::bench::parseOptions(
            argc, argv, "bench_linear_reaction_diffusion_results.json");
        if (options.show_help) {
            biotransport::bench::printUsage(
                argv[0], "Canonical scalar linear reaction-diffusion performance evidence");
            return 0;
        }

        std::vector<BenchmarkResult> results;
        results.reserve(options.sizes.size());
        for (int size : options.sizes) {
            results.push_back(benchmarkSize(size, options));
            biotransport::bench::printResult(results.back());
        }
        biotransport::bench::writeJson(results, options, "bench_linear_reaction_diffusion");
        std::cout << "JSON evidence written to " << options.output_path << '\n';
        return biotransport::bench::allCorrect(results) ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << '\n';
        return 1;
    }
}
