/**
 * @file bench_utils.hpp
 * @brief Bounded benchmark runner and machine-readable evidence writer.
 */

#ifndef BIOTRANSPORT_BENCH_UTILS_HPP
#define BIOTRANSPORT_BENCH_UTILS_HPP

#include "bench_build_config.hpp"
#include <algorithm>
#include <biotransport/core/build_info.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(BIOTRANSPORT_ENABLE_OPENMP) && defined(_OPENMP)
#include <omp.h>
#endif

#ifndef BIOTRANSPORT_BENCH_BUILD_TYPE
#define BIOTRANSPORT_BENCH_BUILD_TYPE "unknown"
#endif

namespace biotransport {
namespace bench {

constexpr const char* schema_version = "biotransport.performance.v1";

struct BenchmarkOptions {
    std::vector<int> sizes{64, 128, 256};
    int steps = 50;
    int warmup_runs = 1;
    int timed_runs = 5;
    std::string output_path;
    bool show_help = false;
};

struct BenchmarkStats {
    std::vector<double> samples_ms;
    double mean_ms = 0.0;
    double population_stddev_ms = 0.0;
    double minimum_ms = 0.0;
    double maximum_ms = 0.0;
    double median_ms = 0.0;

    void compute(const std::vector<double>& samples) {
        if (samples.empty()) {
            throw std::invalid_argument("timing samples must not be empty");
        }
        for (double sample : samples) {
            if (!std::isfinite(sample) || sample <= 0.0) {
                throw std::runtime_error("every elapsed time must be finite and positive");
            }
        }

        samples_ms = samples;
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        minimum_ms = sorted.front();
        maximum_ms = sorted.back();
        const std::size_t count = sorted.size();
        median_ms =
            count % 2 == 0 ? 0.5 * (sorted[count / 2 - 1] + sorted[count / 2]) : sorted[count / 2];
        mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(count);
        double squared_deviation = 0.0;
        for (double sample : samples) {
            squared_deviation += (sample - mean_ms) * (sample - mean_ms);
        }
        population_stddev_ms = std::sqrt(squared_deviation / static_cast<double>(count));
    }
};

struct CorrectnessEvidence {
    std::string invariant;
    std::string tolerance_basis = "relative_error";
    double reference_value = 0.0;
    double observed_value = 0.0;
    double absolute_error = 0.0;
    double relative_error = 0.0;
    double tolerance = 0.0;
    double solution_checksum = 0.0;
    bool passed = false;
    bool repeatable_across_runs = true;
    double maximum_checksum_difference = 0.0;
};

struct BenchmarkCase {
    std::string name;
    std::string description;
    std::string implementation;
    std::string timed_scope = "construct_configure_solve_and_compute_invariant";
    std::string parallel_kernel;
    bool openmp_effective_for_workload = false;
    int cells_x = 0;
    int cells_y = 0;
    std::uint64_t cell_count = 0;
    std::uint64_t node_count = 0;
    std::size_t species_count = 1;
    int step_count = 0;
    double maximum_time_step = 0.0;
    double final_time = 0.0;
    std::vector<std::pair<std::string, double>> parameters;
};

struct RunOutcome {
    CorrectnessEvidence correctness;
};

struct BenchmarkResult {
    BenchmarkCase workload;
    BenchmarkStats timing;
    CorrectnessEvidence correctness;
    double node_species_steps_per_second = 0.0;
};

inline double relativeError(double reference, double observed) {
    const double denominator = std::max(std::abs(reference), 1.0e-300);
    return std::abs(observed - reference) / denominator;
}

inline CorrectnessEvidence makeCorrectnessEvidence(std::string invariant, double reference,
                                                   double observed, double absolute_error,
                                                   double relative_error, double tolerance,
                                                   double checksum) {
    CorrectnessEvidence evidence;
    evidence.invariant = std::move(invariant);
    evidence.reference_value = reference;
    evidence.observed_value = observed;
    evidence.absolute_error = absolute_error;
    evidence.relative_error = relative_error;
    evidence.tolerance = tolerance;
    evidence.solution_checksum = checksum;
    evidence.passed = relative_error <= tolerance;
    return evidence;
}

inline void validateCorrectness(const CorrectnessEvidence& evidence) {
    const double values[] = {evidence.reference_value,
                             evidence.observed_value,
                             evidence.absolute_error,
                             evidence.relative_error,
                             evidence.tolerance,
                             evidence.solution_checksum,
                             evidence.maximum_checksum_difference};
    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("correctness evidence contains a non-finite value");
        }
    }
    if (evidence.invariant.empty()) {
        throw std::runtime_error("correctness evidence must name its invariant");
    }
    if (evidence.absolute_error < 0.0 || evidence.relative_error < 0.0 ||
        evidence.tolerance < 0.0 || evidence.maximum_checksum_difference < 0.0) {
        throw std::runtime_error("correctness errors and tolerance must be non-negative");
    }
}

inline double solutionChecksum(const std::vector<double>& values) {
    long double checksum = 0.0L;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            throw std::runtime_error("cannot checksum a non-finite solution");
        }
        const long double weight = static_cast<long double>(index % 251U + 1U);
        checksum += weight * static_cast<long double>(values[index]);
    }
    const double result = static_cast<double>(checksum);
    if (!std::isfinite(result)) {
        throw std::runtime_error("solution checksum overflowed");
    }
    return result;
}

template <typename Function>
BenchmarkResult runBenchmark(const BenchmarkCase& workload, int warmup_runs, int timed_runs,
                             Function&& function) {
    if (warmup_runs < 0 || timed_runs < 2) {
        throw std::invalid_argument(
            "warmup_runs must be non-negative and timed_runs must be at least two");
    }

    for (int run = 0; run < warmup_runs; ++run) {
        const RunOutcome outcome = function();
        validateCorrectness(outcome.correctness);
        if (!outcome.correctness.passed) {
            throw std::runtime_error("correctness invariant failed during benchmark warmup");
        }
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(timed_runs));
    CorrectnessEvidence aggregate;
    bool have_evidence = false;
    for (int run = 0; run < timed_runs; ++run) {
        const auto start = std::chrono::steady_clock::now();
        const RunOutcome outcome = function();
        const auto end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        validateCorrectness(outcome.correctness);
        samples.push_back(elapsed);

        if (!have_evidence) {
            aggregate = outcome.correctness;
            have_evidence = true;
        } else {
            const double checksum_difference =
                std::abs(outcome.correctness.solution_checksum - aggregate.solution_checksum);
            aggregate.maximum_checksum_difference =
                std::max(aggregate.maximum_checksum_difference, checksum_difference);
            if (checksum_difference != 0.0) {
                aggregate.repeatable_across_runs = false;
            }
            aggregate.absolute_error =
                std::max(aggregate.absolute_error, outcome.correctness.absolute_error);
            aggregate.relative_error =
                std::max(aggregate.relative_error, outcome.correctness.relative_error);
            aggregate.passed = aggregate.passed && outcome.correctness.passed;
        }
    }
    aggregate.passed = aggregate.passed && aggregate.repeatable_across_runs;

    BenchmarkResult result;
    result.workload = workload;
    result.correctness = aggregate;
    result.timing.compute(samples);
    const long double work = static_cast<long double>(workload.node_count) *
                             static_cast<long double>(workload.species_count) *
                             static_cast<long double>(workload.step_count);
    result.node_species_steps_per_second =
        static_cast<double>(work / (result.timing.median_ms / 1000.0L));
    if (!std::isfinite(result.node_species_steps_per_second) ||
        result.node_species_steps_per_second <= 0.0) {
        throw std::runtime_error("computed throughput is not finite and positive");
    }
    return result;
}

inline int parseInteger(const std::string& text, const std::string& option) {
    std::size_t parsed = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument(option + " requires an integer");
    }
    if (parsed != text.size() || value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(option + " requires an integer");
    }
    return static_cast<int>(value);
}

inline std::vector<int> parseSizes(const std::string& text) {
    std::vector<int> values;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string token = text.substr(start, comma - start);
        if (token.empty()) {
            throw std::invalid_argument("--sizes requires comma-separated integers");
        }
        const int value = parseInteger(token, "--sizes");
        if (value < 8 || value > 2048) {
            throw std::invalid_argument("each --sizes value must be in [8, 2048]");
        }
        values.push_back(value);
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (values.empty()) {
        throw std::invalid_argument("--sizes must not be empty");
    }
    return values;
}

inline BenchmarkOptions parseOptions(int argc, char* argv[], std::string default_output) {
    BenchmarkOptions options;
    options.output_path = std::move(default_output);
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto requireValue = [&](const std::string& option) -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument(option + " requires a value");
            }
            return argv[++index];
        };

        if (argument == "--sizes") {
            options.sizes = parseSizes(requireValue(argument));
        } else if (argument == "--steps") {
            options.steps = parseInteger(requireValue(argument), argument);
        } else if (argument == "--warmup") {
            options.warmup_runs = parseInteger(requireValue(argument), argument);
        } else if (argument == "--runs") {
            options.timed_runs = parseInteger(requireValue(argument), argument);
        } else if (argument == "--output") {
            options.output_path = requireValue(argument);
            if (options.output_path.empty()) {
                throw std::invalid_argument("--output must not be empty");
            }
        } else if (argument == "--quick") {
            options.sizes = {64};
            options.steps = 10;
            options.warmup_runs = 0;
            options.timed_runs = 3;
        } else if (argument == "--help" || argument == "-h") {
            options.show_help = true;
        } else {
            throw std::invalid_argument("unknown benchmark option: " + argument);
        }
    }

    if (options.steps < 1 || options.steps > 10'000) {
        throw std::invalid_argument("--steps must be in [1, 10000]");
    }
    if (options.warmup_runs < 0 || options.warmup_runs > 20) {
        throw std::invalid_argument("--warmup must be in [0, 20]");
    }
    if (options.timed_runs < 2 || options.timed_runs > 100) {
        throw std::invalid_argument("--runs must be in [2, 100]");
    }
    return options;
}

inline void printUsage(const std::string& executable, const std::string& workload) {
    std::cout << workload << "\n\n"
              << "Usage: " << executable
              << " [--sizes N,N] [--steps N] [--warmup N] [--runs N]"
                 " [--output FILE] [--quick]\n"
              << "Defaults are bounded: sizes=64,128,256; steps=50; warmup=1; runs=5.\n"
              << "Use OMP_NUM_THREADS to control an OpenMP-enabled build.\n";
}

inline std::string jsonEscape(const std::string& value) {
    std::ostringstream output;
    for (unsigned char character : value) {
        switch (character) {
            case '\"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(character) << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

inline void writeString(std::ostream& output, const std::string& value) {
    output << '\"' << jsonEscape(value) << '\"';
}

inline void writeNumber(std::ostream& output, double value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("refusing to emit non-finite JSON number");
    }
    output << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
}

inline std::string utcTimestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

inline std::string environmentValue(const char* name) {
#ifdef _WIN32
    char* buffer = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&buffer, &size, name) != 0 || buffer == nullptr) {
        return "";
    }
    const std::string result(buffer);
    std::free(buffer);
    return result;
#else
    const char* value = std::getenv(name);
    return value == nullptr ? "" : std::string(value);
#endif
}

inline std::string cpuModel() {
    const std::string windows_model = environmentValue("PROCESSOR_IDENTIFIER");
    if (!windows_model.empty()) {
        return windows_model;
    }
#ifdef __linux__
    std::ifstream cpu_info("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpu_info, line)) {
        const std::string key = "model name";
        if (line.compare(0, key.size(), key) == 0) {
            const std::size_t colon = line.find(':');
            if (colon != std::string::npos) {
                const std::size_t first = line.find_first_not_of(" \t", colon + 1);
                return first == std::string::npos ? "unknown" : line.substr(first);
            }
        }
    }
#endif
    return "unknown";
}

inline std::string configuredFlags(const std::string& build_type) {
    std::string flags = BIOTRANSPORT_BENCH_CXX_FLAGS;
    const char* configuration_flags = "";
    if (build_type == "Debug") {
        configuration_flags = BIOTRANSPORT_BENCH_CXX_FLAGS_DEBUG;
    } else if (build_type == "Release") {
        configuration_flags = BIOTRANSPORT_BENCH_CXX_FLAGS_RELEASE;
    } else if (build_type == "RelWithDebInfo") {
        configuration_flags = BIOTRANSPORT_BENCH_CXX_FLAGS_RELWITHDEBINFO;
    } else if (build_type == "MinSizeRel") {
        configuration_flags = BIOTRANSPORT_BENCH_CXX_FLAGS_MINSIZEREL;
    }
    if (*configuration_flags != '\0') {
        if (!flags.empty()) {
            flags += ' ';
        }
        flags += configuration_flags;
    }
    return flags;
}

struct ParallelMetadata {
    int maximum_threads = 1;
    int observed_threads = 1;
    int thread_limit = 1;
    bool dynamic_teams = false;
};

inline ParallelMetadata parallelMetadata() {
    ParallelMetadata metadata;
#if defined(BIOTRANSPORT_ENABLE_OPENMP) && defined(_OPENMP)
    metadata.maximum_threads = omp_get_max_threads();
#if _OPENMP >= 200805
    metadata.thread_limit = omp_get_thread_limit();
#else
    metadata.thread_limit = metadata.maximum_threads;
#endif
    metadata.dynamic_teams = omp_get_dynamic() != 0;
#pragma omp parallel
    {
#pragma omp single
        metadata.observed_threads = omp_get_num_threads();
    }
#endif
    return metadata;
}

inline void writeJson(const std::vector<BenchmarkResult>& results, const BenchmarkOptions& options,
                      const std::string& program_name) {
    if (results.empty()) {
        throw std::invalid_argument("cannot write an empty benchmark result set");
    }
    std::ofstream output(options.output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("could not open benchmark output: " + options.output_path);
    }

    const build::NativeBuildInfo native = build::nativeBuildInfo();
    const ParallelMetadata parallel = parallelMetadata();
    const std::string build_type = BIOTRANSPORT_BENCH_BUILD_TYPE;
    const unsigned int logical_threads = std::thread::hardware_concurrency();

    output << "{\n  \"schema_version\": ";
    writeString(output, schema_version);
    output << ",\n  \"generated_utc\": ";
    writeString(output, utcTimestamp());
    output << ",\n  \"program\": ";
    writeString(output, program_name);
    output << ",\n  \"provenance\": {\n"
           << "    \"project_version\": ";
    writeString(output, BIOTRANSPORT_BENCH_PROJECT_VERSION);
    output << ",\n    \"revision\": ";
    writeString(output, BIOTRANSPORT_BENCH_GIT_REVISION);
    output << ",\n    \"revision_discoverable\": "
           << (std::string(BIOTRANSPORT_BENCH_GIT_REVISION) != "unknown" ? "true" : "false")
           << ",\n    \"revision_dirty\": "
           << (BIOTRANSPORT_BENCH_GIT_DIRTY != 0 ? "true" : "false") << ",\n"
           << "    \"compiler\": {\"id\": ";
    writeString(output, native.compiler_id);
    output << ", \"version\": ";
    writeString(output, native.compiler_version);
    output << ", \"cpp_standard\": " << native.cpp_standard << ", \"cpp_standard_name\": ";
    writeString(output, native.cpp_standard_name);
    output << "},\n    \"build\": {\"type\": ";
    writeString(output, build_type);
    output << ", \"generator\": ";
    writeString(output, BIOTRANSPORT_BENCH_GENERATOR);
    output << ", \"configured_cxx_flags\": ";
    writeString(output, configuredFlags(build_type));
    output << ", \"openmp_cxx_flags\": ";
    writeString(output, BIOTRANSPORT_BENCH_OPENMP_FLAGS);
    output << ", \"benchmark_target_flags\": ";
    writeString(output, BIOTRANSPORT_BENCH_TARGET_FLAGS);
    output << ", \"flags_scope\": "
              "\"CMake cache flags; implicit compiler and toolchain flags may be absent\", "
              "\"assertions_enabled\": "
           << (native.assertions_enabled ? "true" : "false")
           << ", \"eigen_enabled\": " << (native.eigen_enabled ? "true" : "false")
           << "},\n    \"platform\": {\"operating_system\": ";
    writeString(output, BIOTRANSPORT_BENCH_SYSTEM_NAME);
    output << ", \"architecture\": ";
    writeString(output, BIOTRANSPORT_BENCH_SYSTEM_PROCESSOR);
    output << ", \"cpu_model\": ";
    writeString(output, cpuModel());
    output << ", \"logical_threads\": " << logical_threads << "},\n"
           << "    \"openmp\": {\"compile_definition\": "
           << (native.openmp_compile_definition ? "true" : "false")
           << ", \"enabled\": " << (native.openmp_enabled ? "true" : "false")
           << ", \"specification_date\": " << native.openmp_specification_date
           << ", \"maximum_threads\": " << parallel.maximum_threads
           << ", \"observed_threads\": " << parallel.observed_threads
           << ", \"thread_limit\": " << parallel.thread_limit
           << ", \"dynamic_teams\": " << (parallel.dynamic_teams ? "true" : "false")
           << ", \"omp_num_threads_environment\": ";
    writeString(output, environmentValue("OMP_NUM_THREADS"));
    output << ", \"omp_proc_bind_environment\": ";
    writeString(output, environmentValue("OMP_PROC_BIND"));
    output << ", \"omp_places_environment\": ";
    writeString(output, environmentValue("OMP_PLACES"));
    output << "}\n  },\n  \"configuration\": {\n"
           << "    \"warmup_runs\": " << options.warmup_runs << ",\n"
           << "    \"timed_runs\": " << options.timed_runs << ",\n"
           << "    \"requested_steps\": " << options.steps << ",\n"
           << "    \"requested_sizes_cells_per_axis\": [";
    for (std::size_t index = 0; index < options.sizes.size(); ++index) {
        output << (index == 0 ? "" : ", ") << options.sizes[index];
    }
    output << "]\n  },\n  \"results\": [\n";

    for (std::size_t index = 0; index < results.size(); ++index) {
        const BenchmarkResult& result = results[index];
        validateCorrectness(result.correctness);
        output << "    {\n      \"name\": ";
        writeString(output, result.workload.name);
        output << ",\n      \"description\": ";
        writeString(output, result.workload.description);
        output << ",\n      \"implementation\": ";
        writeString(output, result.workload.implementation);
        output << ",\n      \"timed_scope\": ";
        writeString(output, result.workload.timed_scope);
        output << ",\n      \"parallel\": {\"kernel\": ";
        writeString(output, result.workload.parallel_kernel);
        output << ", \"openmp_effective_for_workload\": "
               << (result.workload.openmp_effective_for_workload ? "true" : "false")
               << ", \"observed_threads\": " << parallel.observed_threads << "},\n"
               << "      \"workload\": {\n"
               << "        \"cells_x\": " << result.workload.cells_x << ",\n"
               << "        \"cells_y\": " << result.workload.cells_y << ",\n"
               << "        \"cell_count\": " << result.workload.cell_count << ",\n"
               << "        \"node_count\": " << result.workload.node_count << ",\n"
               << "        \"species_count\": " << result.workload.species_count << ",\n"
               << "        \"step_count\": " << result.workload.step_count << ",\n"
               << "        \"maximum_time_step\": ";
        writeNumber(output, result.workload.maximum_time_step);
        output << ",\n        \"final_time\": ";
        writeNumber(output, result.workload.final_time);
        output << ",\n        \"parameters\": {";
        for (std::size_t parameter_index = 0; parameter_index < result.workload.parameters.size();
             ++parameter_index) {
            const auto& parameter = result.workload.parameters[parameter_index];
            output << (parameter_index == 0 ? "" : ", ");
            writeString(output, parameter.first);
            output << ": ";
            writeNumber(output, parameter.second);
        }
        output << "}\n      },\n      \"correctness\": {\n"
               << "        \"status\": " << (result.correctness.passed ? "\"pass\"" : "\"fail\"")
               << ",\n"
               << "        \"invariant\": ";
        writeString(output, result.correctness.invariant);
        output << ",\n        \"tolerance_basis\": ";
        writeString(output, result.correctness.tolerance_basis);
        output << ",\n        \"reference_value\": ";
        writeNumber(output, result.correctness.reference_value);
        output << ",\n        \"observed_value\": ";
        writeNumber(output, result.correctness.observed_value);
        output << ",\n        \"absolute_error\": ";
        writeNumber(output, result.correctness.absolute_error);
        output << ",\n        \"relative_error\": ";
        writeNumber(output, result.correctness.relative_error);
        output << ",\n        \"tolerance\": ";
        writeNumber(output, result.correctness.tolerance);
        output << ",\n        \"solution_checksum\": ";
        writeNumber(output, result.correctness.solution_checksum);
        output << ",\n        \"checksum_definition\": "
                  "\"sum(((node_index mod 251) + 1) * value[node_index]) in long double\",\n"
               << "        \"repeatable_across_runs\": "
               << (result.correctness.repeatable_across_runs ? "true" : "false")
               << ",\n        \"maximum_checksum_difference\": ";
        writeNumber(output, result.correctness.maximum_checksum_difference);
        output << "\n      },\n      \"timing_ms\": {\n"
               << "        \"clock\": \"std::chrono::steady_clock\",\n"
               << "        \"samples\": [";
        for (std::size_t sample_index = 0; sample_index < result.timing.samples_ms.size();
             ++sample_index) {
            output << (sample_index == 0 ? "" : ", ");
            writeNumber(output, result.timing.samples_ms[sample_index]);
        }
        output << "],\n        \"mean\": ";
        writeNumber(output, result.timing.mean_ms);
        output << ",\n        \"population_stddev\": ";
        writeNumber(output, result.timing.population_stddev_ms);
        output << ",\n        \"minimum\": ";
        writeNumber(output, result.timing.minimum_ms);
        output << ",\n        \"maximum\": ";
        writeNumber(output, result.timing.maximum_ms);
        output << ",\n        \"median\": ";
        writeNumber(output, result.timing.median_ms);
        output << "\n      },\n      \"throughput\": {\n"
               << "        \"basis\": \"median elapsed time\",\n"
               << "        \"node_species_steps_per_second\": ";
        writeNumber(output, result.node_species_steps_per_second);
        output << "\n      }\n    }" << (index + 1 == results.size() ? "" : ",") << "\n";
    }
    output << "  ]\n}\n";
    if (!output) {
        throw std::runtime_error("failed while writing benchmark JSON");
    }
}

inline void printResult(const BenchmarkResult& result) {
    std::cout << result.workload.name << ": median=" << std::fixed << std::setprecision(3)
              << result.timing.median_ms
              << " ms, invariant=" << (result.correctness.passed ? "pass" : "FAIL")
              << ", checksum=" << std::scientific << std::setprecision(8)
              << result.correctness.solution_checksum << '\n';
}

inline bool allCorrect(const std::vector<BenchmarkResult>& results) {
    return std::all_of(results.begin(), results.end(),
                       [](const BenchmarkResult& result) { return result.correctness.passed; });
}

}  // namespace bench
}  // namespace biotransport

#endif
