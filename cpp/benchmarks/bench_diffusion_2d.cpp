/** Benchmark conservative 2D diffusion through the native multispecies API. */

#include "bench_utils.hpp"
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/multi_species_solver.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using biotransport::Boundary;
using biotransport::MultiSpeciesSolver;
using biotransport::StructuredMesh;
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

void setClosedBoundaries(MultiSpeciesSolver& solver) {
    solver.setAllSpeciesNeumann(Boundary::Left, 0.0);
    solver.setAllSpeciesNeumann(Boundary::Right, 0.0);
    solver.setAllSpeciesNeumann(Boundary::Bottom, 0.0);
    solver.setAllSpeciesNeumann(Boundary::Top, 0.0);
}

BenchmarkResult benchmarkSize(int cells_per_axis, const BenchmarkOptions& options) {
    constexpr double diffusivity = 0.01;
    constexpr double cfl_fraction = 0.5;
    constexpr double mass_tolerance = 5.0e-12;
    const StructuredMesh mesh(cells_per_axis, cells_per_axis, 0.0, 1.0, 0.0, 1.0);
    const std::vector<double> initial = gaussianInitialCondition(mesh);

    MultiSpeciesSolver reference_solver(mesh, {diffusivity});
    setClosedBoundaries(reference_solver);
    reference_solver.setInitialCondition(0, initial);
    const double initial_mass = reference_solver.totalMass(0);
    const double time_step = cfl_fraction * reference_solver.maxStableTimeStep();

    BenchmarkCase workload;
    workload.name = "conservative_diffusion_2d_" + std::to_string(cells_per_axis) + "x" +
                    std::to_string(cells_per_axis);
    workload.description =
        "Closed 2D Gaussian diffusion; mass conservation is checked on every run";
    workload.implementation = "MultiSpeciesSolver conservative nodal finite volume";
    workload.parallel_kernel = "MultiSpeciesSolver::computeCandidateStep node loop";
    workload.openmp_effective_for_workload = biotransport::build::nativeBuildInfo().openmp_enabled;
    workload.cells_x = cells_per_axis;
    workload.cells_y = cells_per_axis;
    workload.cell_count =
        static_cast<std::uint64_t>(cells_per_axis) * static_cast<std::uint64_t>(cells_per_axis);
    workload.node_count = static_cast<std::uint64_t>(mesh.numNodes());
    workload.species_count = 1;
    workload.step_count = options.steps;
    workload.maximum_time_step = time_step;
    workload.final_time = time_step * static_cast<double>(options.steps);
    workload.parameters = {{"diffusivity", diffusivity}, {"cfl_fraction", cfl_fraction}};

    auto run = [&]() -> RunOutcome {
        MultiSpeciesSolver solver(mesh, {diffusivity});
        setClosedBoundaries(solver);
        solver.setInitialCondition(0, initial);
        solver.solve(time_step, options.steps);
        const double final_mass = solver.totalMass(0);
        const double absolute_error = std::abs(final_mass - initial_mass);
        return {biotransport::bench::makeCorrectnessEvidence(
            "closed_boundary_trapezoidal_mass_conservation", initial_mass, final_mass,
            absolute_error, biotransport::bench::relativeError(initial_mass, final_mass),
            mass_tolerance, biotransport::bench::solutionChecksum(solver.solution(0)))};
    };

    return biotransport::bench::runBenchmark(workload, options.warmup_runs, options.timed_runs,
                                             run);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const BenchmarkOptions options =
            biotransport::bench::parseOptions(argc, argv, "bench_diffusion_2d_results.json");
        if (options.show_help) {
            biotransport::bench::printUsage(argv[0],
                                            "Conservative 2D diffusion performance evidence");
            return 0;
        }

        std::vector<BenchmarkResult> results;
        results.reserve(options.sizes.size());
        for (int size : options.sizes) {
            results.push_back(benchmarkSize(size, options));
            biotransport::bench::printResult(results.back());
        }
        biotransport::bench::writeJson(results, options, "bench_diffusion_2d");
        std::cout << "JSON evidence written to " << options.output_path << '\n';
        return biotransport::bench::allCorrect(results) ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << '\n';
        return 1;
    }
}
