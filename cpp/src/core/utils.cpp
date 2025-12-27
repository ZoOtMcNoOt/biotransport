#include <algorithm>
#include <biotransport/core/utils.hpp>
#include <cmath>
#include <iostream>

namespace biotransport {
namespace utils {

bool writeCsv1D(const std::string& filename, const std::vector<double>& x,
                const std::vector<double>& solution) {
    if (x.size() != solution.size()) {
        std::cerr << "Error: x and solution vectors must have the same size" << std::endl;
        return false;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    // Write header
    file << "x,solution\n";

    // Write data
    for (size_t i = 0; i < x.size(); ++i) {
        file << x[i] << "," << solution[i] << "\n";
    }

    return true;
}

bool writeCsv2D(const std::string& filename, const std::vector<double>& x,
                const std::vector<double>& y, const std::vector<double>& solution, int nx, int ny) {
    if (x.size() != nx || y.size() != ny || solution.size() != nx * ny) {
        std::cerr << "Error: Inconsistent dimensions in writeCsv2D" << std::endl;
        return false;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    // Write header
    file << "x,y,solution\n";

    // Write data
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            file << x[i] << "," << y[j] << "," << solution[j * nx + i] << "\n";
        }
    }

    return true;
}

double l2Norm(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size for L2 norm calculation");
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }

    return std::sqrt(sum / a.size());
}

double maxDifference(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vectors must have the same size for max difference calculation");
    }

    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = std::abs(a[i] - b[i]);
        max_diff = std::max(max_diff, diff);
    }

    return max_diff;
}

}  // namespace utils
}  // namespace biotransport
