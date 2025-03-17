#ifndef BIOTRANSPORT_UTILS_HPP
#define BIOTRANSPORT_UTILS_HPP

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

namespace biotransport {

/**
 * Utility functions for the biotransport library.
 */
namespace utils {

/**
 * Write solution data to a CSV file.
 * 
 * @param filename Output filename
 * @param x X coordinates
 * @param solution Solution values
 * @return true if successful
 */
bool writeCsv1D(const std::string& filename, 
                const std::vector<double>& x,
                const std::vector<double>& solution);

/**
 * Write 2D solution data to a CSV file.
 * 
 * @param filename Output filename
 * @param x X coordinates
 * @param y Y coordinates
 * @param solution Solution values (flattened)
 * @param nx Number of points in x direction
 * @param ny Number of points in y direction
 * @return true if successful
 */
bool writeCsv2D(const std::string& filename,
                const std::vector<double>& x,
                const std::vector<double>& y,
                const std::vector<double>& solution,
                int nx, int ny);

/**
 * Calculate the L2 norm of the difference between two vectors.
 * 
 * @param a First vector
 * @param b Second vector
 * @return L2 norm of (a-b)
 */
double l2Norm(const std::vector<double>& a, const std::vector<double>& b);

/**
 * Calculate the maximum absolute difference between two vectors.
 * 
 * @param a First vector
 * @param b Second vector
 * @return Maximum absolute difference
 */
double maxDifference(const std::vector<double>& a, const std::vector<double>& b);

} // namespace utils

} // namespace biotransport

#endif // BIOTRANSPORT_UTILS_HPP