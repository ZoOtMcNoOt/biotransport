/**
 * @file vtk_export.hpp
 * @brief VTK export utilities for ParaView visualization
 */

#ifndef BIOTRANSPORT_IO_VTK_EXPORT_HPP
#define BIOTRANSPORT_IO_VTK_EXPORT_HPP

#include <biotransport/core/mesh/structured_mesh.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace biotransport {
namespace io {

/**
 * @brief Write solution to VTK Legacy format file for ParaView.
 *
 * Creates a structured points dataset that can be opened in ParaView.
 * The VTK Legacy format is simple and widely supported.
 *
 * @param mesh StructuredMesh containing the domain
 * @param solution Solution field values at mesh nodes
 * @param filename Output filename (e.g., "output.vtk")
 * @param field_name Name of the scalar field (default: "scalar")
 *
 * @throws std::runtime_error if file cannot be opened or sizes don't match
 */
inline void write_vtk(const StructuredMesh& mesh, const std::vector<double>& solution,
                      const std::string& filename, const std::string& field_name = "scalar") {
    if (solution.size() != static_cast<std::size_t>(mesh.numNodes())) {
        throw std::runtime_error("Solution size doesn't match mesh nodes");
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // VTK Legacy format header
    file << "# vtk DataFile Version 3.0\n";
    file << "BioTransport simulation output\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";

    if (mesh.is1D()) {
        // 1D mesh: extend to 2D with ny=nz=0 for ParaView compatibility
        file << "DIMENSIONS " << (mesh.nx() + 1) << " 1 1\n";
        file << "ORIGIN " << mesh.x(0) << " 0.0 0.0\n";
        file << "SPACING " << mesh.dx() << " 1.0 1.0\n";
    } else {
        // 2D mesh: extend to 3D with nz=0
        file << "DIMENSIONS " << (mesh.nx() + 1) << " " << (mesh.ny() + 1) << " 1\n";
        file << "ORIGIN " << mesh.x(0) << " " << mesh.y(0, 0) << " 0.0\n";
        file << "SPACING " << mesh.dx() << " " << mesh.dy() << " 1.0\n";
    }

    // Point data section
    file << "POINT_DATA " << mesh.numNodes() << "\n";
    file << "SCALARS " << field_name << " double 1\n";
    file << "LOOKUP_TABLE default\n";

    // Write solution values
    file << std::scientific << std::setprecision(10);
    for (std::size_t i = 0; i < solution.size(); ++i) {
        file << solution[i] << "\n";
    }

    file.close();
}

/**
 * @brief Write time series of solutions to VTK files for animation.
 *
 * Creates multiple VTK files with sequential naming for time series animation
 * in ParaView. Files are named as: prefix_XXXX.vtk where XXXX is the time index.
 *
 * @param mesh StructuredMesh containing the domain
 * @param solutions Vector of solution snapshots (each matching mesh size)
 * @param times Vector of time values for each snapshot (seconds)
 * @param prefix Output filename prefix (e.g., "simulation")
 * @param field_name Name of the scalar field (default: "scalar")
 *
 * @throws std::runtime_error if sizes don't match or files cannot be written
 *
 * Example:
 *   write_vtk_series(mesh, {sol0, sol1, sol2}, {0.0, 1.0, 2.0}, "sim");
 *   // Creates: sim_0000.vtk, sim_0001.vtk, sim_0002.vtk
 */
inline void write_vtk_series(const StructuredMesh& mesh,
                             const std::vector<std::vector<double>>& solutions,
                             const std::vector<double>& times, const std::string& prefix,
                             const std::string& field_name = "scalar") {
    if (solutions.size() != times.size()) {
        throw std::runtime_error("Number of solutions must match number of times");
    }

    for (std::size_t i = 0; i < solutions.size(); ++i) {
        // Generate filename with zero-padded index
        std::ostringstream filename;
        filename << prefix << "_" << std::setfill('0') << std::setw(4) << i << ".vtk";

        // Write this snapshot
        write_vtk(mesh, solutions[i], filename.str(), field_name);
    }
}

/**
 * @brief Write time series with additional metadata file for ParaView.
 *
 * In addition to individual VTK files, this creates a .pvd (ParaView Data) file
 * that ParaView can use to load the entire time series at once with proper
 * time values.
 *
 * @param mesh StructuredMesh containing the domain
 * @param solutions Vector of solution snapshots
 * @param times Vector of time values (seconds)
 * @param prefix Output filename prefix
 * @param field_name Name of the scalar field (default: "scalar")
 *
 * Example:
 *   write_vtk_series_with_metadata(mesh, solutions, times, "sim");
 *   // Creates: sim.pvd (metadata) + sim_0000.vtk, sim_0001.vtk, ...
 *   // Open sim.pvd in ParaView to load entire series with time info
 */
inline void write_vtk_series_with_metadata(const StructuredMesh& mesh,
                                           const std::vector<std::vector<double>>& solutions,
                                           const std::vector<double>& times,
                                           const std::string& prefix,
                                           const std::string& field_name = "scalar") {
    // Write individual VTK files
    write_vtk_series(mesh, solutions, times, prefix, field_name);

    // Create PVD metadata file
    std::string pvd_filename = prefix + ".pvd";
    std::ofstream pvd(pvd_filename);
    if (!pvd.is_open()) {
        throw std::runtime_error("Cannot create PVD file: " + pvd_filename);
    }

    pvd << "<?xml version=\"1.0\"?>\n";
    pvd << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    pvd << "  <Collection>\n";

    for (std::size_t i = 0; i < solutions.size(); ++i) {
        std::ostringstream filename;
        filename << prefix << "_" << std::setfill('0') << std::setw(4) << i << ".vtk";

        pvd << "    <DataSet timestep=\"" << times[i] << "\" file=\"" << filename.str() << "\"/>\n";
    }

    pvd << "  </Collection>\n";
    pvd << "</VTKFile>\n";
    pvd.close();
}

}  // namespace io
}  // namespace biotransport

#endif  // BIOTRANSPORT_IO_VTK_EXPORT_HPP
