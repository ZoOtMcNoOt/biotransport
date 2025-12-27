/**
 * @file utilities_bindings.hpp
 * @brief Python bindings for utility modules
 * 
 * This module provides bindings for:
 * - dimensionless submodule (Re, Sc, Pe, Bi, Fo, Sh)
 * - analytical submodule (canonical solutions for verification)
 */

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace biotransport {
namespace bindings {

/**
 * @brief Register all utility bindings (dimensionless, analytical)
 * @param m The pybind11 module to register with
 */
void register_utilities_bindings(py::module_& m);

} // namespace bindings
} // namespace biotransport
