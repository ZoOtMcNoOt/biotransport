#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <biotransport/core/mesh.hpp>
#include <biotransport/solvers/diffusion.hpp>
#include <biotransport/solvers/reaction_diffusion.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "BioTransport library Python bindings";
    
    // Mesh class
    py::class_<biotransport::StructuredMesh>(m, "StructuredMesh")
        .def(py::init<int, double, double>(), 
             py::arg("nx"), py::arg("xmin"), py::arg("xmax"))
        .def(py::init<int, int, double, double, double, double>(),
             py::arg("nx"), py::arg("ny"), 
             py::arg("xmin"), py::arg("xmax"),
             py::arg("ymin"), py::arg("ymax"))
        .def("num_nodes", &biotransport::StructuredMesh::numNodes)
        .def("num_cells", &biotransport::StructuredMesh::numCells)
        .def("dx", &biotransport::StructuredMesh::dx)
        .def("dy", &biotransport::StructuredMesh::dy)
        .def("is_1d", &biotransport::StructuredMesh::is1D)
        .def("nx", &biotransport::StructuredMesh::nx)
        .def("ny", &biotransport::StructuredMesh::ny)
        .def("x", &biotransport::StructuredMesh::x, py::arg("i"))
        .def("y", &biotransport::StructuredMesh::y, py::arg("i"), py::arg("j") = 0)
        .def("index", &biotransport::StructuredMesh::index, py::arg("i"), py::arg("j") = 0);
    
    // Boundary type enum
    py::enum_<biotransport::BoundaryType>(m, "BoundaryType")
        .value("DIRICHLET", biotransport::BoundaryType::DIRICHLET)
        .value("NEUMANN", biotransport::BoundaryType::NEUMANN)
        .export_values();
    
    // Diffusion solver
    py::class_<biotransport::DiffusionSolver>(m, "DiffusionSolver")
        .def(py::init<const biotransport::StructuredMesh&, double>(),
             py::arg("mesh"), py::arg("diffusivity"))
        .def("set_initial_condition", &biotransport::DiffusionSolver::setInitialCondition, 
             py::arg("values"))
        .def("set_dirichlet_boundary", &biotransport::DiffusionSolver::setDirichletBoundary,
             py::arg("boundary_id"), py::arg("value"))
        .def("set_neumann_boundary", &biotransport::DiffusionSolver::setNeumannBoundary,
             py::arg("boundary_id"), py::arg("flux"))
        .def("solve", &biotransport::DiffusionSolver::solve,
             py::arg("dt"), py::arg("num_steps"))
        .def("solution", [](const biotransport::DiffusionSolver& solver) {
            const auto& sol = solver.solution();
            return py::array_t<double>(
                {sol.size()},                  // Shape
                {sizeof(double)},              // Strides
                sol.data(),                    // Data pointer
                py::capsule([]() {})           // Dummy capsule
            );
        });
        
    // Reaction-diffusion solver
    py::class_<biotransport::ReactionDiffusionSolver, biotransport::DiffusionSolver>
        (m, "ReactionDiffusionSolver")
        .def(py::init<const biotransport::StructuredMesh&, double, 
                     biotransport::ReactionDiffusionSolver::ReactionFunction>(),
             py::arg("mesh"), py::arg("diffusivity"), py::arg("reaction"))
        .def("solve", &biotransport::ReactionDiffusionSolver::solve,
             py::arg("dt"), py::arg("num_steps"));
}