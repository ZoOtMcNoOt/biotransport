/**
 * @file fluid_bindings.hpp
 * @brief Python bindings for fluid dynamics solvers
 * 
 * This module provides bindings for:
 * - DarcyFlowSolver (porous media)
 * - StokesSolver (viscous flow)
 * - NavierStokesSolver (inertial flow)
 * - Non-Newtonian models (Power-law, Carreau, Casson, etc.)
 */

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace biotransport {
namespace bindings {

/**
 * @brief Register all fluid dynamics bindings
 * @param m The pybind11 module to register with
 */
void register_fluid_bindings(py::module_& m);

} // namespace bindings
} // namespace biotransport
