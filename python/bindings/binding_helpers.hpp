/**
 * @file binding_helpers.hpp
 * @brief Common helper utilities for Python bindings.
 *
 * Eliminates code duplication across binding files by providing
 * reusable templates for common patterns like vector-to-numpy conversion.
 */

#ifndef BIOTRANSPORT_BINDINGS_HELPERS_HPP
#define BIOTRANSPORT_BINDINGS_HELPERS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>

namespace py = pybind11;

namespace biotransport {
namespace bindings {

/**
 * @brief Wrap a const std::vector<double> reference as a read-only NumPy array.
 *
 * Creates a view into the existing data without copying.
 * The resulting array is not writeable.
 *
 * @param vec Reference to vector (must outlive the returned array)
 * @return NumPy array view of the vector data
 */
inline py::array_t<double> to_numpy(const std::vector<double>& vec) {
    return py::array_t<double>(
        {static_cast<py::ssize_t>(vec.size())},
        {static_cast<py::ssize_t>(sizeof(double))},
        vec.data()
    );
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
    return py::array_t<double>(
        {static_cast<py::ssize_t>(vec.size())},
        {static_cast<py::ssize_t>(sizeof(double))},
        vec.data()
    );
}

/**
 * @brief Wrap a const std::vector<float> reference as a read-only NumPy array.
 */
inline py::array_t<float> to_numpy(const std::vector<float>& vec) {
    return py::array_t<float>(
        {static_cast<py::ssize_t>(vec.size())},
        {static_cast<py::ssize_t>(sizeof(float))},
        vec.data()
    );
}

/**
 * @brief Create a 2D NumPy array view of a flat vector.
 *
 * Interprets the vector as row-major 2D data with shape (ny+1, nx+1).
 *
 * @param vec Flat vector data
 * @param nx Number of cells in x (array width will be nx+1)
 * @param ny Number of cells in y (array height will be ny+1)
 * @return 2D NumPy array view
 */
inline py::array_t<double> to_numpy_2d(const std::vector<double>& vec, int nx, int ny) {
    return py::array_t<double>(
        {static_cast<py::ssize_t>(ny + 1), static_cast<py::ssize_t>(nx + 1)},
        {static_cast<py::ssize_t>((nx + 1) * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))},
        vec.data()
    );
}

/**
 * @brief Copy a std::vector<double> to a new NumPy array.
 *
 * Unlike to_numpy(), this creates an owning copy of the data.
 * Use when the vector may go out of scope.
 *
 * @param vec Vector to copy
 * @return NumPy array owning a copy of the data
 */
inline py::array_t<double> copy_to_numpy(const std::vector<double>& vec) {
    py::array_t<double> result(static_cast<py::ssize_t>(vec.size()));
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    std::copy(vec.begin(), vec.end(), ptr);
    return result;
}

/**
 * @brief Wrap a vector as a NumPy array view with owner reference.
 *
 * The `base` object keeps the underlying data alive as long as the
 * NumPy array exists. Use this when exposing member data from a class.
 *
 * @tparam T Element type (double, float, etc.)
 * @param vec Vector to wrap
 * @param base Python object that owns the vector (keeps it alive)
 * @return NumPy array view with lifetime tied to base
 */
template <typename T>
inline py::array_t<T> to_numpy_with_base(const std::vector<T>& vec, py::object base) {
    return py::array_t<T>(
        {static_cast<py::ssize_t>(vec.size())},
        {static_cast<py::ssize_t>(sizeof(T))},
        vec.data(),
        std::move(base)
    );
}

/**
 * @brief Wrap a flat vector as a 3D NumPy array view (frames × ny × nx).
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
inline py::array_t<T> to_numpy_3d(const std::vector<T>& vec,
                                   py::ssize_t frames,
                                   py::ssize_t ny,
                                   py::ssize_t nx,
                                   py::object base) {
    return py::array_t<T>(
        {frames, ny, nx},
        {static_cast<py::ssize_t>(ny * nx * sizeof(T)),
         static_cast<py::ssize_t>(nx * sizeof(T)),
         static_cast<py::ssize_t>(sizeof(T))},
        vec.data(),
        std::move(base)
    );
}

} // namespace bindings
} // namespace biotransport

#endif // BIOTRANSPORT_BINDINGS_HELPERS_HPP
