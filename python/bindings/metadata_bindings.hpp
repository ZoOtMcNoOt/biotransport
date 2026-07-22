#ifndef BIOTRANSPORT_PYTHON_METADATA_BINDINGS_HPP
#define BIOTRANSPORT_PYTHON_METADATA_BINDINGS_HPP

#include <pybind11/pybind11.h>

namespace biotransport {
namespace bindings {

void register_metadata_bindings(pybind11::module_& module);

}  // namespace bindings
}  // namespace biotransport

#endif  // BIOTRANSPORT_PYTHON_METADATA_BINDINGS_HPP
