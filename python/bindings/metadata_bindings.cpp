/**
 * @file metadata_bindings.cpp
 * @brief Python access to path-free native build provenance.
 */

#include "metadata_bindings.hpp"

#include <biotransport/core/build_info.hpp>

namespace py = pybind11;

namespace biotransport {
namespace bindings {

void register_metadata_bindings(py::module_& module) {
    module.def(
        "native_build_info",
        []() {
            const build::NativeBuildInfo info = build::nativeBuildInfo();

            py::dict compiler;
            compiler["id"] = info.compiler_id;
            compiler["version"] = info.compiler_version;

            py::dict cxx;
            cxx["standard"] = info.cpp_standard;
            cxx["standard_name"] = info.cpp_standard_name;
            cxx["assertions_enabled"] = info.assertions_enabled;

            py::dict eigen;
            eigen["compile_definition"] = info.eigen_enabled;
            eigen["enabled"] = info.eigen_enabled;
            if (info.eigen_enabled)
                eigen["version"] = info.eigen_version;
            else
                eigen["version"] = py::none();

            py::dict openmp;
            openmp["compile_definition"] = info.openmp_compile_definition;
            openmp["enabled"] = info.openmp_enabled;
            if (info.openmp_specification_date > 0)
                openmp["specification_date"] = info.openmp_specification_date;
            else
                openmp["specification_date"] = py::none();

            py::dict features;
            features["eigen"] = eigen;
            features["openmp"] = openmp;

            py::dict result;
            result["compiler"] = compiler;
            result["cxx"] = cxx;
            result["features"] = features;
            return result;
        },
        R"doc(Return path-free metadata for the loaded native extension build.

The result reports compiler identity/version, the effective C++ language
standard, assertion mode, and the actual Eigen/OpenMP compile-time feature
definitions. It deliberately omits source paths, compiler executable paths,
hostnames, usernames, command lines, and environment variables.
)doc");
}

}  // namespace bindings
}  // namespace biotransport
