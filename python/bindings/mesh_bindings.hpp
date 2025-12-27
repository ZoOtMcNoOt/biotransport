/**
 * @file mesh_bindings.hpp
 * @brief Python bindings for mesh-related classes
 * 
 * This module provides bindings for:
 * - StructuredMesh (1D/2D Cartesian grids)
 * - CylindricalMesh (radial/axisymmetric/3D cylindrical)
 * - Boundary enums and BoundaryCondition types
 */

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace biotransport {
namespace bindings {

/**
 * @brief Register all mesh-related bindings
 * @param m The pybind11 module to register with
 */
void register_mesh_bindings(py::module_& m);

} // namespace bindings
} // namespace biotransport
