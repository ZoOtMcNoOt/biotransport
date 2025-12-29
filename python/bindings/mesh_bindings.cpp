/**
 * @file mesh_bindings.cpp
 * @brief Python bindings for mesh-related classes
 */

// Ensure M_PI is defined on MSVC
#define _USE_MATH_DEFINES
#include "mesh_bindings.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "binding_helpers.hpp"
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/cylindrical_mesh.hpp>
#include <biotransport/core/mesh/mesh_iterators.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/mesh/structured_mesh_3d.hpp>
#include <cmath>

namespace biotransport {
namespace bindings {

void register_mesh_bindings(py::module_& m) {
    // =========================================================================
    // StructuredMesh
    // =========================================================================
    py::class_<StructuredMesh>(m, "StructuredMesh")
        .def(py::init<int, double, double>(), py::arg("nx"), py::arg("xmin"), py::arg("xmax"))
        .def(py::init<int, int, double, double, double, double>(), py::arg("nx"), py::arg("ny"),
             py::arg("xmin"), py::arg("xmax"), py::arg("ymin"), py::arg("ymax"))
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
    // StructuredMesh3D
    // =========================================================================
    py::class_<StructuredMesh3D>(m, "StructuredMesh3D")
        .def(py::init<int, int, int, double, double, double, double, double, double>(),
             py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("xmin"), py::arg("xmax"),
             py::arg("ymin"), py::arg("ymax"), py::arg("zmin"), py::arg("zmax"),
             "Create a 3D structured mesh with specified dimensions and bounds.")
        .def(py::init<int, double>(), py::arg("n"), py::arg("length"),
             "Create a cubic mesh with n cells and given side length.")
        .def("num_nodes", &StructuredMesh3D::numNodes)
        .def("num_cells", &StructuredMesh3D::numCells)
        .def("dx", &StructuredMesh3D::dx)
        .def("dy", &StructuredMesh3D::dy)
        .def("dz", &StructuredMesh3D::dz)
        .def("nx", &StructuredMesh3D::nx)
        .def("ny", &StructuredMesh3D::ny)
        .def("nz", &StructuredMesh3D::nz)
        .def("xmin", &StructuredMesh3D::xmin)
        .def("xmax", &StructuredMesh3D::xmax)
        .def("ymin", &StructuredMesh3D::ymin)
        .def("ymax", &StructuredMesh3D::ymax)
        .def("zmin", &StructuredMesh3D::zmin)
        .def("zmax", &StructuredMesh3D::zmax)
        .def("x", &StructuredMesh3D::x, py::arg("i"))
        .def("y", &StructuredMesh3D::y, py::arg("j"))
        .def("z", &StructuredMesh3D::z, py::arg("k"))
        .def("index", &StructuredMesh3D::index, py::arg("i"), py::arg("j"), py::arg("k"))
        .def("ijk", &StructuredMesh3D::ijk, py::arg("idx"),
             "Convert linear index to (i, j, k) tuple.");

    // Boundary3D enum
    py::enum_<Boundary3D>(m, "Boundary3D")
        .value("XMin", Boundary3D::XMin)
        .value("XMax", Boundary3D::XMax)
        .value("YMin", Boundary3D::YMin)
        .value("YMax", Boundary3D::YMax)
        .value("ZMin", Boundary3D::ZMin)
        .value("ZMax", Boundary3D::ZMax)
        .export_values();

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
        .def(py::init<int, double, double>(), py::arg("nr"), py::arg("rmin"), py::arg("rmax"),
             "Create a 1D radial mesh.")
        // 2D axisymmetric constructor
        .def(py::init<int, int, double, double, double, double>(), py::arg("nr"), py::arg("nz"),
             py::arg("rmin"), py::arg("rmax"), py::arg("zmin"), py::arg("zmax"),
             "Create a 2D axisymmetric (r, z) mesh.")
        // 3D cylindrical constructor
        .def(py::init<int, int, int, double, double, double, double, double, double>(),
             py::arg("nr"), py::arg("ntheta"), py::arg("nz"), py::arg("rmin"), py::arg("rmax"),
             py::arg("thetamin"), py::arg("thetamax"), py::arg("zmin"), py::arg("zmax"),
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
        .def("index", &CylindricalMesh::index, py::arg("i"), py::arg("j") = 0, py::arg("k") = 0)
        .def("has_axis_singularity", &CylindricalMesh::hasAxisSingularity)
        .def("x", &CylindricalMesh::x, py::arg("i"), py::arg("j") = 0)
        .def("y", &CylindricalMesh::y, py::arg("i"), py::arg("j") = 0)
        .def("cell_volume", &CylindricalMesh::cellVolume, py::arg("i"), py::arg("j") = 0,
             py::arg("k") = 0)
        .def("cell_area", &CylindricalMesh::cellArea, py::arg("i"))
        .def("cross_section_area", &CylindricalMesh::crossSectionArea)
        .def(
            "gradient_r",
            [](const CylindricalMesh& mesh, const std::vector<double>& phi) {
                return copy_to_numpy(mesh.gradientR(phi));
            },
            py::arg("phi"))
        .def(
            "gradient_z",
            [](const CylindricalMesh& mesh, const std::vector<double>& phi) {
                return copy_to_numpy(mesh.gradientZ(phi));
            },
            py::arg("phi"))
        .def(
            "laplacian",
            [](const CylindricalMesh& mesh, const std::vector<double>& phi) {
                return copy_to_numpy(mesh.laplacian(phi));
            },
            py::arg("phi"))
        .def(
            "divergence",
            [](const CylindricalMesh& mesh, const std::vector<double>& vr,
               const std::vector<double>& vz) { return copy_to_numpy(mesh.divergence(vr, vz)); },
            py::arg("vr"), py::arg("vz"));

    // =========================================================================
    // StencilOps - Higher-order finite difference stencils
    // =========================================================================
    py::class_<StencilOps>(m, "StencilOps",
                           "Finite difference stencil operations on structured meshes.")
        .def(py::init<const StructuredMesh&>(), py::arg("mesh"),
             "Create stencil operator for the given mesh.")
        // 2nd-order (existing)
        .def("laplacian", &StencilOps::laplacian, py::arg("u"), py::arg("idx"),
             "Compute 2nd-order Laplacian at node idx.")
        // 4th-order
        .def("laplacian_4th_order", &StencilOps::laplacian4thOrder, py::arg("u"), py::arg("idx"),
             "Compute 4th-order Laplacian at node idx. Requires 2-node padding.")
        // 6th-order (1D only)
        .def("laplacian_6th_order", &StencilOps::laplacian6thOrder, py::arg("u"), py::arg("idx"),
             "Compute 6th-order Laplacian at node idx (1D only). Requires 3-node padding.")
        // Gradients
        .def("grad_x", &StencilOps::gradX, py::arg("u"), py::arg("idx"),
             "Compute 2nd-order x-gradient at node idx.")
        .def("grad_y", &StencilOps::gradY, py::arg("u"), py::arg("idx"),
             "Compute 2nd-order y-gradient at node idx.")
        .def("grad_x_4th_order", &StencilOps::gradX4thOrder, py::arg("u"), py::arg("idx"),
             "Compute 4th-order x-gradient at node idx.")
        .def("grad_y_4th_order", &StencilOps::gradY4thOrder, py::arg("u"), py::arg("idx"),
             "Compute 4th-order y-gradient at node idx.")
        // Bulk operations (1D)
        .def(
            "laplacian_4th_order_bulk_1d",
            [](const StencilOps& ops, const std::vector<double>& u) {
                return copy_to_numpy(ops.laplacian4thOrderBulk1D(u));
            },
            py::arg("u"), "Apply 4th-order Laplacian to entire 1D array. Returns numpy array.")
        .def(
            "laplacian_6th_order_bulk_1d",
            [](const StencilOps& ops, const std::vector<double>& u) {
                return copy_to_numpy(ops.laplacian6thOrderBulk1D(u));
            },
            py::arg("u"), "Apply 6th-order Laplacian to entire 1D array. Returns numpy array.")
        // Bulk operations (2D)
        .def(
            "laplacian_4th_order_bulk_2d",
            [](const StencilOps& ops, const std::vector<double>& u) {
                return copy_to_numpy(ops.laplacian4thOrderBulk2D(u));
            },
            py::arg("u"),
            "Apply 4th-order Laplacian to entire 2D array (row-major). Returns numpy array.")
        // Bulk gradient (1D)
        .def(
            "gradient_4th_order_bulk_1d",
            [](const StencilOps& ops, const std::vector<double>& u) {
                return copy_to_numpy(ops.gradient4thOrderBulk1D(u));
            },
            py::arg("u"), "Apply 4th-order gradient to entire 1D array. Returns numpy array.")
        // Accessors
        .def("inv_dx2", &StencilOps::invDx2, "Get precomputed 1/dx².")
        .def("inv_dy2", &StencilOps::invDy2, "Get precomputed 1/dy².")
        .def("inv_12_dx2", &StencilOps::inv12Dx2, "Get precomputed 1/(12*dx²).")
        .def("inv_12_dy2", &StencilOps::inv12Dy2, "Get precomputed 1/(12*dy²).")
        .def("stride", &StencilOps::stride, "Get row stride for 2D indexing.");
}

}  // namespace bindings
}  // namespace biotransport
