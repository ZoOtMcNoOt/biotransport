/**
 * @file utils.hpp
 * @brief General utility functions for the biotransport library.
 *
 * Provides common utilities for:
 *   - CSV file I/O for solution data
 *   - Vector norm calculations for error analysis
 */

#ifndef BIOTRANSPORT_CORE_UTILS_HPP
#define BIOTRANSPORT_CORE_UTILS_HPP

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace biotransport {

/**
 * @brief Utility functions for file I/O and error analysis.
 */
namespace utils {

/**
 * @brief Write 1D solution data to a CSV file.
 *
 * @param filename Output filename
 * @param x X coordinates
 * @param solution Solution values
 * @return true if successful, false otherwise
 */
bool writeCsv1D(const std::string& filename, const std::vector<double>& x,
                const std::vector<double>& solution);

/**
 * @brief Write 2D solution data to a CSV file.
 *
 * @param filename Output filename
 * @param x X coordinates
 * @param y Y coordinates
 * @param solution Solution values (flattened row-major)
 * @param nx Number of points in x direction
 * @param ny Number of points in y direction
 * @return true if successful, false otherwise
 */
bool writeCsv2D(const std::string& filename, const std::vector<double>& x,
                const std::vector<double>& y, const std::vector<double>& solution, int nx, int ny);

/**
 * @brief Calculate the L2 norm of the difference between two vectors.
 *
 * @param a First vector
 * @param b Second vector
 * @return L2 norm of (a-b)
 */
double l2Norm(const std::vector<double>& a, const std::vector<double>& b);

/**
 * @brief Calculate the maximum absolute difference between two vectors.
 *
 * @param a First vector
 * @param b Second vector
 * @return Maximum absolute value of (a[i] - b[i])
 */
double maxDifference(const std::vector<double>& a, const std::vector<double>& b);

}  // namespace utils

}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_UTILS_HPP
