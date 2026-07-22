#pragma once

#include <pybind11/pybind11.h>

namespace biotransport::bindings {

void register_balance_bindings(pybind11::module_& module);

}  // namespace biotransport::bindings
