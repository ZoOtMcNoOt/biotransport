/**
 * @file nonuniform_bindings.hpp
 * @brief Python registration for conservative nonuniform 1D transport.
 */

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace biotransport {
namespace bindings {

void register_nonuniform_bindings(py::module_& module);

}  // namespace bindings
}  // namespace biotransport
