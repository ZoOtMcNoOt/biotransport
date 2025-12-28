/**
 * @file sparse_bindings.hpp
 * @brief Python bindings for sparse matrix and implicit solvers.
 */

#pragma once

#include <pybind11/pybind11.h>

namespace biotransport {
namespace bindings {

/**
 * @brief Register sparse matrix bindings.
 * @param m Module to register bindings in.
 */
void register_sparse_bindings(pybind11::module_& m);

}  // namespace bindings
}  // namespace biotransport
