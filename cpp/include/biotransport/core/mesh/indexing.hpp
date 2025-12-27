/**
 * @file indexing.hpp
 * @brief Common 2D grid indexing utilities.
 *
 * Provides consistent indexing functions for 2D structured grids
 * to avoid code duplication across solvers.
 */

#ifndef BIOTRANSPORT_CORE_MESH_INDEXING_HPP
#define BIOTRANSPORT_CORE_MESH_INDEXING_HPP

#include <cstddef>

namespace biotransport {

/**
 * @brief Convert 2D grid indices to flat array index.
 *
 * Uses row-major ordering: index = j * stride + i
 *
 * @param i Column index (x-direction)
 * @param j Row index (y-direction)
 * @param stride Number of elements per row (typically nx + 1)
 * @return Flat array index
 */
[[nodiscard]] constexpr std::size_t grid_index(int i, int j, int stride) noexcept {
    return static_cast<std::size_t>(j) * static_cast<std::size_t>(stride) +
           static_cast<std::size_t>(i);
}

/**
 * @brief Shorthand alias for grid_index.
 *
 * For code that prefers terse naming.
 */
[[nodiscard]] constexpr std::size_t idx(int i, int j, int stride) noexcept {
    return grid_index(i, j, stride);
}

/**
 * @brief Wrap index for periodic boundary conditions.
 *
 * Maps -1 to n-1 and n to 0.
 *
 * @param a Index that may be out of bounds
 * @param n Domain size
 * @return Wrapped index in [0, n)
 */
[[nodiscard]] constexpr int wrap_index(int a, int n) noexcept {
    if (a < 0)
        return n - 1;
    if (a >= n)
        return 0;
    return a;
}

}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_MESH_INDEXING_HPP
