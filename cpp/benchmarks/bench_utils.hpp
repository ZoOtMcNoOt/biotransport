/**
 * @file bench_utils.hpp
 * @brief Lightweight benchmarking utilities for biotransport
 *
 * Provides timing, statistics, and JSON output for performance tracking.
 */

#ifndef BIOTRANSPORT_BENCH_UTILS_HPP
#define BIOTRANSPORT_BENCH_UTILS_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace biotransport {
namespace bench {

/**
 * High-resolution timer for benchmarking
 */
class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }

    void stop() { end_ = std::chrono::high_resolution_clock::now(); }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(end_ - start_).count();
    }

    double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(end_ - start_).count();
    }

    double elapsed_s() const { return std::chrono::duration<double>(end_ - start_).count(); }

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_;
};

/**
 * Statistics for a set of timing measurements
 */
struct BenchmarkStats {
    double mean_ms = 0.0;
    double std_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double median_ms = 0.0;
    int num_runs = 0;

    void compute(std::vector<double>& times_ms) {
        if (times_ms.empty())
            return;

        num_runs = static_cast<int>(times_ms.size());

        // Sort for median
        std::sort(times_ms.begin(), times_ms.end());
        min_ms = times_ms.front();
        max_ms = times_ms.back();

        if (num_runs % 2 == 0) {
            median_ms = (times_ms[num_runs / 2 - 1] + times_ms[num_runs / 2]) / 2.0;
        } else {
            median_ms = times_ms[num_runs / 2];
        }

        mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / num_runs;

        double variance = 0.0;
        for (double t : times_ms) {
            variance += (t - mean_ms) * (t - mean_ms);
        }
        std_ms = std::sqrt(variance / num_runs);
    }
};

/**
 * Result of a single benchmark
 */
struct BenchmarkResult {
    std::string name;
    std::string description;
    BenchmarkStats stats;

    // Problem parameters
    int nx = 0;
    int ny = 0;
    int num_steps = 0;
    double dt = 0.0;

    // Throughput metrics
    double cells_per_second = 0.0;
    double steps_per_second = 0.0;

    void computeThroughput() {
        if (stats.mean_ms > 0) {
            double total_cells = static_cast<double>(nx) * ny * num_steps;
            cells_per_second = total_cells / (stats.mean_ms / 1000.0);
            steps_per_second = num_steps / (stats.mean_ms / 1000.0);
        }
    }
};

/**
 * Print benchmark result to console
 */
inline void printResult(const BenchmarkResult& result) {
    std::cout << "\n=== " << result.name << " ===\n";
    std::cout << result.description << "\n";
    std::cout << "Grid: " << result.nx << " x " << result.ny << " (" << (result.nx * result.ny)
              << " cells)\n";
    std::cout << "Steps: " << result.num_steps << ", dt: " << result.dt << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Time: " << result.stats.mean_ms << " ± " << result.stats.std_ms << " ms "
              << "(min: " << result.stats.min_ms << ", max: " << result.stats.max_ms
              << ", median: " << result.stats.median_ms << ")\n";
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "Throughput: " << result.cells_per_second << " cell-steps/s, "
              << result.steps_per_second << " steps/s\n";
}

/**
 * Write benchmark results to JSON file
 */
inline void writeJSON(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << filename << " for writing\n";
        return;
    }

    file << "{\n";
    file << "  \"benchmark_version\": \"1.0\",\n";

    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    file << "  \"timestamp\": \"" << std::ctime(&time_t);
    // Remove newline from ctime
    file.seekp(-1, std::ios_base::cur);
    file << "\",\n";

    file << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        file << "    {\n";
        file << "      \"name\": \"" << r.name << "\",\n";
        file << "      \"description\": \"" << r.description << "\",\n";
        file << "      \"grid\": { \"nx\": " << r.nx << ", \"ny\": " << r.ny << " },\n";
        file << "      \"num_steps\": " << r.num_steps << ",\n";
        file << "      \"dt\": " << std::scientific << r.dt << ",\n";
        file << std::fixed << std::setprecision(6);
        file << "      \"timing_ms\": {\n";
        file << "        \"mean\": " << r.stats.mean_ms << ",\n";
        file << "        \"std\": " << r.stats.std_ms << ",\n";
        file << "        \"min\": " << r.stats.min_ms << ",\n";
        file << "        \"max\": " << r.stats.max_ms << ",\n";
        file << "        \"median\": " << r.stats.median_ms << ",\n";
        file << "        \"num_runs\": " << r.stats.num_runs << "\n";
        file << "      },\n";
        file << std::scientific << std::setprecision(6);
        file << "      \"throughput\": {\n";
        file << "        \"cells_per_second\": " << r.cells_per_second << ",\n";
        file << "        \"steps_per_second\": " << r.steps_per_second << "\n";
        file << "      }\n";
        file << "    }" << (i < results.size() - 1 ? "," : "") << "\n";
    }

    file << "  ]\n";
    file << "}\n";

    file.close();
    std::cout << "\nResults written to: " << filename << "\n";
}

/**
 * Run a benchmark function multiple times and collect statistics
 */
template <typename Func>
BenchmarkResult runBenchmark(const std::string& name, const std::string& description, int nx,
                             int ny, int num_steps, double dt, int num_warmup, int num_runs,
                             Func&& func) {
    BenchmarkResult result;
    result.name = name;
    result.description = description;
    result.nx = nx;
    result.ny = ny;
    result.num_steps = num_steps;
    result.dt = dt;

    // Warmup runs (not timed)
    for (int i = 0; i < num_warmup; ++i) {
        func();
    }

    // Timed runs
    Timer timer;
    std::vector<double> times;
    times.reserve(num_runs);

    for (int i = 0; i < num_runs; ++i) {
        timer.start();
        func();
        timer.stop();
        times.push_back(timer.elapsed_ms());
    }

    result.stats.compute(times);
    result.computeThroughput();

    return result;
}

}  // namespace bench
}  // namespace biotransport

#endif  // BIOTRANSPORT_BENCH_UTILS_HPP
