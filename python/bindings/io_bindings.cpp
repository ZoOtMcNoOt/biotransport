/**
 * @file io_bindings.cpp
 * @brief Python bindings for I/O and visualization functions
 */

#include "io_bindings.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/io/vtk_export.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

void register_io_bindings(py::module& m) {
    using namespace biotransport;
    using namespace biotransport::io;

    // write_vtk function: single snapshot
    m.def(
        "write_vtk",
        [](const StructuredMesh& mesh, py::array_t<double> solution, const std::string& filename,
           const std::string& field_name) {
            // Convert numpy array to vector
            py::buffer_info info = solution.request();
            if (info.ndim != 1) {
                throw std::runtime_error("Solution must be a 1D array");
            }

            std::vector<double> sol_vec(static_cast<double*>(info.ptr),
                                        static_cast<double*>(info.ptr) + info.size);

            // Call C++ function
            write_vtk(mesh, sol_vec, filename, field_name);
        },
        py::arg("mesh"), py::arg("solution"), py::arg("filename"), py::arg("field_name") = "scalar",
        R"(
        Write solution to VTK file for ParaView visualization.

        Parameters:
            mesh: StructuredMesh object
            solution: numpy array of solution values
            filename: output filename (e.g., "output.vtk")
            field_name: name of the scalar field (default: "scalar")

        Example:
            >>> mesh = bt.StructuredMesh1D(0.0, 1.0, 100)
            >>> solution = np.sin(np.linspace(0, np.pi, 101))
            >>> bt.write_vtk(mesh, solution, "result.vtk", "temperature")
        )");

    // write_vtk_series function: multiple snapshots
    m.def(
        "write_vtk_series",
        [](const StructuredMesh& mesh, const std::vector<py::array_t<double>>& solutions,
           const std::vector<double>& times, const std::string& prefix,
           const std::string& field_name) {
            // Convert list of numpy arrays to vector of vectors
            std::vector<std::vector<double>> sol_vecs;
            sol_vecs.reserve(solutions.size());

            for (const auto& sol : solutions) {
                py::buffer_info info = sol.request();
                if (info.ndim != 1) {
                    throw std::runtime_error("Each solution must be a 1D array");
                }
                sol_vecs.emplace_back(static_cast<double*>(info.ptr),
                                      static_cast<double*>(info.ptr) + info.size);
            }

            // Call C++ function
            write_vtk_series(mesh, sol_vecs, times, prefix, field_name);
        },
        py::arg("mesh"), py::arg("solutions"), py::arg("times"), py::arg("prefix"),
        py::arg("field_name") = "scalar",
        R"(
        Write time series to multiple VTK files for animation.

        Creates files named prefix_0000.vtk, prefix_0001.vtk, etc.

        Parameters:
            mesh: StructuredMesh object
            solutions: list of numpy arrays (one per time step)
            times: list of time values (seconds)
            prefix: filename prefix (e.g., "simulation")
            field_name: name of the scalar field (default: "scalar")

        Example:
            >>> mesh = bt.StructuredMesh1D(0.0, 1.0, 100)
            >>> solutions = [sol_t0, sol_t1, sol_t2]  # List of numpy arrays
            >>> times = [0.0, 0.5, 1.0]
            >>> bt.write_vtk_series(mesh, solutions, times, "sim", "concentration")
            # Creates: sim_0000.vtk, sim_0001.vtk, sim_0002.vtk
        )");

    // write_vtk_series_with_metadata: includes .pvd file
    m.def(
        "write_vtk_series_with_metadata",
        [](const StructuredMesh& mesh, const std::vector<py::array_t<double>>& solutions,
           const std::vector<double>& times, const std::string& prefix,
           const std::string& field_name) {
            // Convert list of numpy arrays to vector of vectors
            std::vector<std::vector<double>> sol_vecs;
            sol_vecs.reserve(solutions.size());

            for (const auto& sol : solutions) {
                py::buffer_info info = sol.request();
                if (info.ndim != 1) {
                    throw std::runtime_error("Each solution must be a 1D array");
                }
                sol_vecs.emplace_back(static_cast<double*>(info.ptr),
                                      static_cast<double*>(info.ptr) + info.size);
            }

            // Call C++ function
            write_vtk_series_with_metadata(mesh, sol_vecs, times, prefix, field_name);
        },
        py::arg("mesh"), py::arg("solutions"), py::arg("times"), py::arg("prefix"),
        py::arg("field_name") = "scalar",
        R"(
        Write time series with ParaView metadata file (.pvd).

        Creates individual VTK files plus a .pvd file that ParaView can open
        to load the entire series with proper time information.

        Parameters:
            mesh: StructuredMesh object
            solutions: list of numpy arrays (one per time step)
            times: list of time values (seconds)
            prefix: filename prefix (e.g., "simulation")
            field_name: name of the scalar field (default: "scalar")

        Example:
            >>> bt.write_vtk_series_with_metadata(mesh, solutions, times, "sim")
            # Creates: sim.pvd + sim_0000.vtk, sim_0001.vtk, ...
            # Open sim.pvd in ParaView to load entire series
        )");
}
