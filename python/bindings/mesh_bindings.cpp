/**
 * @file mesh_bindings.cpp
 * @brief Python bindings for mesh-related classes
 */

// Ensure M_PI is defined on MSVC
#define _USE_MATH_DEFINES
#include <cmath>

#include "mesh_bindings.hpp"
#include "binding_helpers.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/mesh/cylindrical_mesh.hpp>
#include <biotransport/core/boundary.hpp>

namespace biotransport {
namespace bindings {

void register_mesh_bindings(py::module_& m) {
    // =========================================================================
    // StructuredMesh
    // =========================================================================
    py::class_<StructuredMesh>(m, "StructuredMesh")
        .def(py::init<int, double, double>(), 
             py::arg("nx"), py::arg("xmin"), py::arg("xmax"))
        .def(py::init<int, int, double, double, double, double>(),
             py::arg("nx"), py::arg("ny"), 
             py::arg("xmin"), py::arg("xmax"),
             py::arg("ymin"), py::arg("ymax"))
        .def("num_nodes", &StructuredMesh::numNodes)
        .def("num_cells", &StructuredMesh::numCells)
        .def("dx", &StructuredMesh::dx)
        .def("dy", &StructuredMesh::dy)
        .def("is_1d", &StructuredMesh::is1D)
        .def("nx", &StructuredMesh::nx)
        .def("ny", &StructuredMesh::ny)
        .def("x", &StructuredMesh::x, py::arg("i"))
        .def("y", &StructuredMesh::y, py::arg("i"), py::arg("j") = 0)
        .def("index", &StructuredMesh::index, py::arg("i"), py::arg("j") = 0);

    // =========================================================================
    // Boundary Enums and Types
    // =========================================================================
    
    // Boundary type enum
    py::enum_<BoundaryType>(m, "BoundaryType")
        .value("DIRICHLET", BoundaryType::DIRICHLET)
        .value("NEUMANN", BoundaryType::NEUMANN)
        .export_values();

    // Boundary side enum
    py::enum_<Boundary>(m, "Boundary")
        .value("Left", Boundary::Left)
        .value("Right", Boundary::Right)
        .value("Bottom", Boundary::Bottom)
        .value("Top", Boundary::Top)
        .export_values();

    // Boundary condition (type + value)
    py::class_<BoundaryCondition>(m, "BoundaryCondition")
        .def(py::init<BoundaryType, double>(), py::arg("type"), py::arg("value"))
        .def_readwrite("type", &BoundaryCondition::type)
        .def_readwrite("value", &BoundaryCondition::value)
        .def_static("dirichlet", &BoundaryCondition::Dirichlet, py::arg("value"))
        .def_static("neumann", &BoundaryCondition::Neumann, py::arg("flux"));

    // =========================================================================
    // CylindricalMesh
    // =========================================================================
    py::enum_<CylindricalMeshType>(m, "CylindricalMeshType")
        .value("AXISYMMETRIC_RZ", CylindricalMeshType::AXISYMMETRIC_RZ)
        .value("RADIAL_R", CylindricalMeshType::RADIAL_R)
        .value("FULL_3D", CylindricalMeshType::FULL_3D)
        .export_values();

    py::class_<CylindricalMesh>(m, "CylindricalMesh")
        // 1D radial constructor
        .def(py::init<int, double, double>(),
             py::arg("nr"), py::arg("rmin"), py::arg("rmax"),
             "Create a 1D radial mesh.")
        // 2D axisymmetric constructor
        .def(py::init<int, int, double, double, double, double>(),
             py::arg("nr"), py::arg("nz"), py::arg("rmin"), py::arg("rmax"),
             py::arg("zmin"), py::arg("zmax"),
             "Create a 2D axisymmetric (r, z) mesh.")
        // 3D cylindrical constructor
        .def(py::init<int, int, int, double, double, double, double, double, double>(),
             py::arg("nr"), py::arg("ntheta"), py::arg("nz"),
             py::arg("rmin"), py::arg("rmax"),
             py::arg("thetamin"), py::arg("thetamax"),
             py::arg("zmin"), py::arg("zmax"),
             "Create a full 3D cylindrical mesh.")
        .def("num_nodes", &CylindricalMesh::numNodes)
        .def("num_cells", &CylindricalMesh::numCells)
        .def("type", &CylindricalMesh::type)
        .def("is_radial", &CylindricalMesh::isRadial)
        .def("is_axisymmetric", &CylindricalMesh::isAxisymmetric)
        .def("is_3d", &CylindricalMesh::is3D)
        .def("nr", &CylindricalMesh::nr)
        .def("ntheta", &CylindricalMesh::ntheta)
        .def("nz", &CylindricalMesh::nz)
        .def("dr", &CylindricalMesh::dr)
        .def("dtheta", &CylindricalMesh::dtheta)
        .def("dz", &CylindricalMesh::dz)
        .def("rmin", &CylindricalMesh::rmin)
        .def("rmax", &CylindricalMesh::rmax)
        .def("zmin", &CylindricalMesh::zmin)
        .def("zmax", &CylindricalMesh::zmax)
        .def("r", &CylindricalMesh::r, py::arg("i"))
        .def("theta", &CylindricalMesh::theta, py::arg("j"))
        .def("z", &CylindricalMesh::z, py::arg("k"))
        .def("index", &CylindricalMesh::index,
             py::arg("i"), py::arg("j") = 0, py::arg("k") = 0)
        .def("has_axis_singularity", &CylindricalMesh::hasAxisSingularity)
        .def("x", &CylindricalMesh::x, py::arg("i"), py::arg("j") = 0)
        .def("y", &CylindricalMesh::y, py::arg("i"), py::arg("j") = 0)
        .def("cell_volume", &CylindricalMesh::cellVolume,
             py::arg("i"), py::arg("j") = 0, py::arg("k") = 0)
        .def("cell_area", &CylindricalMesh::cellArea, py::arg("i"))
        .def("cross_section_area", &CylindricalMesh::crossSectionArea)
        .def("gradient_r", [](const CylindricalMesh& mesh, const std::vector<double>& phi) {
             return copy_to_numpy(mesh.gradientR(phi));
        }, py::arg("phi"))
        .def("gradient_z", [](const CylindricalMesh& mesh, const std::vector<double>& phi) {
             return copy_to_numpy(mesh.gradientZ(phi));
        }, py::arg("phi"))
        .def("laplacian", [](const CylindricalMesh& mesh, const std::vector<double>& phi) {
             return copy_to_numpy(mesh.laplacian(phi));
        }, py::arg("phi"))
        .def("divergence", [](const CylindricalMesh& mesh,
                              const std::vector<double>& vr, const std::vector<double>& vz) {
             return copy_to_numpy(mesh.divergence(vr, vz));
        }, py::arg("vr"), py::arg("vz"));
}

} // namespace bindings
} // namespace biotransport
