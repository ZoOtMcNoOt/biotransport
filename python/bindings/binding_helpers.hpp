/**
 * @file binding_helpers.hpp
 * @brief Common helper utilities for Python bindings.
 *
 * Eliminates code duplication across binding files by providing
 * reusable templates for common patterns like vector-to-numpy conversion.
 */

#ifndef BIOTRANSPORT_BINDINGS_HELPERS_HPP
#define BIOTRANSPORT_BINDINGS_HELPERS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace biotransport {
namespace bindings {

/**
 * @brief Copy a std::vector<double> into an owning NumPy array.
 *
 * Getters use owned arrays by default so later C++ vector swaps, reallocations,
 * and object destruction cannot invalidate Python data.
 */
inline py::array_t<double> to_numpy(const std::vector<double>& vec) {
    py::array_t<double> result(static_cast<py::ssize_t>(vec.size()));
    std::copy(vec.begin(), vec.end(), result.mutable_data());
    return result;
}

/**
 * @brief Wrap a mutable std::vector<double> reference as a writeable NumPy array.
 *
 * Creates a view into the existing data without copying.
 * Modifications to the array will affect the original vector.
 *
 * @param vec Reference to vector (must outlive the returned array)
 * @return Writeable NumPy array view of the vector data
 */
inline py::array_t<double> to_numpy_mutable(std::vector<double>& vec) {
    return py::array_t<double>({static_cast<py::ssize_t>(vec.size())},
                               {static_cast<py::ssize_t>(sizeof(double))}, vec.data());
}

/**
 * @brief Copy a std::vector<float> into an owning NumPy array.
 */
inline py::array_t<float> to_numpy(const std::vector<float>& vec) {
    py::array_t<float> result(static_cast<py::ssize_t>(vec.size()));
    std::copy(vec.begin(), vec.end(), result.mutable_data());
    return result;
}

/**
 * @brief Copy a flat vector into an owning 2D NumPy array.
 *
 * Interprets the vector as row-major 2D data with shape (ny+1, nx+1).
 *
 * @param vec Flat vector data
 * @param nx Number of cells in x (array width will be nx+1)
 * @param ny Number of cells in y (array height will be ny+1)
 * @return 2D NumPy array view
 */
inline py::array_t<double> to_numpy_2d(const std::vector<double>& vec, int nx, int ny) {
    const auto width = static_cast<py::ssize_t>(nx + 1);
    const auto height = static_cast<py::ssize_t>(ny + 1);
    if (nx < 0 || ny < 0 || static_cast<std::size_t>(width * height) != vec.size()) {
        throw std::invalid_argument("Vector size does not match requested 2D NumPy shape");
    }
    py::array_t<double> result({height, width});
    std::copy(vec.begin(), vec.end(), result.mutable_data());
    return result;
}

/**
 * @brief Copy a std::vector<double> to a new NumPy array.
 *
 * Compatibility spelling for the owning-copy behavior of to_numpy().
 *
 * @param vec Vector to copy
 * @return NumPy array owning a copy of the data
 */
inline py::array_t<double> copy_to_numpy(const std::vector<double>& vec) {
    return to_numpy(vec);
}

/**
 * @brief Compatibility helper returning an owned NumPy copy.
 *
 * The historical implementation returned a view tied to ``base``. That was
 * unsafe for solver-owned vectors that are swapped during stepping. The name
 * remains temporarily to avoid a broad mechanical binding rewrite.
 */
template <typename T>
inline py::array_t<T> to_numpy_with_base(const std::vector<T>& vec, py::object base) {
    (void)base;
    py::array_t<T> result(static_cast<py::ssize_t>(vec.size()));
    std::copy(vec.begin(), vec.end(), result.mutable_data());
    return result;
}

/**
 * @brief Copy a flat vector into an owning 3D array (frames × ny × nx).
 *
 * For time-series data where each frame is a 2D grid.
 * The `base` object keeps the underlying data alive.
 *
 * @tparam T Element type (double, float, etc.)
 * @param vec Flat vector containing all frame data
 * @param frames Number of time frames
 * @param ny Grid height
 * @param nx Grid width
 * @param base Python object that owns the vector
 * @return 3D NumPy array view with shape (frames, ny, nx)
 */
template <typename T>
inline py::array_t<T> to_numpy_3d(const std::vector<T>& vec, py::ssize_t frames, py::ssize_t ny,
                                  py::ssize_t nx, py::object base) {
    (void)base;
    if (frames < 0 || ny < 0 || nx < 0 ||
        static_cast<std::size_t>(frames * ny * nx) != vec.size()) {
        throw std::invalid_argument("Vector size does not match requested 3D NumPy shape");
    }
    py::array_t<T> result({frames, ny, nx});
    std::copy(vec.begin(), vec.end(), result.mutable_data());
    return result;
}

}  // namespace bindings
}  // namespace biotransport

#endif  // BIOTRANSPORT_BINDINGS_HELPERS_HPP
