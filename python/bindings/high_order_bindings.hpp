#ifndef BIOTRANSPORT_PYTHON_HIGH_ORDER_BINDINGS_HPP
#define BIOTRANSPORT_PYTHON_HIGH_ORDER_BINDINGS_HPP

#include <pybind11/pybind11.h>

namespace biotransport {
namespace bindings {

void register_high_order_bindings(pybind11::module_& module);

}  // namespace bindings
}  // namespace biotransport

#endif  // BIOTRANSPORT_PYTHON_HIGH_ORDER_BINDINGS_HPP
