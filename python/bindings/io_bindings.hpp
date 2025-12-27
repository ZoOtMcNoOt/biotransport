/**
 * @file io_bindings.hpp
 * @brief Python bindings for I/O functions
 */

#ifndef IO_BINDINGS_HPP
#define IO_BINDINGS_HPP

#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_io_bindings(py::module& m);

#endif  // IO_BINDINGS_HPP
