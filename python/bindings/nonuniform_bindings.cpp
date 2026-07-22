/**
 * @file nonuniform_bindings.cpp
 * @brief Python bindings for conservative diffusion on fitted 1D meshes.
 */

#include "nonuniform_bindings.hpp"

#include <pybind11/stl.h>

#include "binding_helpers.hpp"
#include <biotransport/core/mesh/nonuniform_mesh_1d.hpp>
#include <biotransport/solvers/nonuniform_diffusion_1d.hpp>
#include <utility>
#include <vector>

namespace biotransport {
namespace bindings {

void register_nonuniform_bindings(py::module_& module) {
    py::class_<NonuniformMesh1D>(
        module, "NonuniformMesh1D",
        "Validated fitted 1D node mesh with positive node-centred control volumes.")
        .def(py::init<std::vector<double>>(), py::arg("nodes"),
             "Create a mesh from finite, strictly increasing node coordinates.")
        .def("num_nodes", &NonuniformMesh1D::numNodes)
        .def("num_cells", &NonuniformMesh1D::numCells)
        .def("x", &NonuniformMesh1D::x, py::arg("node"))
        .def(
            "nodes", [](const NonuniformMesh1D& mesh) { return copy_to_numpy(mesh.nodes()); },
            "Return an owned copy of node coordinates.")
        .def("spacing", &NonuniformMesh1D::spacing, py::arg("face"))
        .def("face_coordinate", &NonuniformMesh1D::faceCoordinate, py::arg("face"))
        .def("control_volume", &NonuniformMesh1D::controlVolume, py::arg("node"))
        .def(
            "control_volumes",
            [](const NonuniformMesh1D& mesh) { return copy_to_numpy(mesh.controlVolumes()); },
            "Return an owned copy of node-centred control-volume widths.")
        .def("xmin", &NonuniformMesh1D::xmin)
        .def("xmax", &NonuniformMesh1D::xmax)
        .def("length", &NonuniformMesh1D::length)
        .def("minimum_spacing", &NonuniformMesh1D::minimumSpacing);

    py::class_<NonuniformDiffusionDiagnostics>(
        module, "NonuniformDiffusionDiagnostics",
        "Mass-balance, flux, range, time, and stability diagnostics.")
        .def_readonly("steps", &NonuniformDiffusionDiagnostics::steps)
        .def_readonly("reference_time", &NonuniformDiffusionDiagnostics::reference_time)
        .def_readonly("time", &NonuniformDiffusionDiagnostics::time)
        .def_readonly("stability_limit", &NonuniformDiffusionDiagnostics::stability_limit)
        .def_readonly("reference_mass", &NonuniformDiffusionDiagnostics::reference_mass)
        .def_readonly("total_mass", &NonuniformDiffusionDiagnostics::total_mass)
        .def_readonly("cumulative_boundary_input",
                      &NonuniformDiffusionDiagnostics::cumulative_boundary_input)
        .def_readonly("mass_balance_error", &NonuniformDiffusionDiagnostics::mass_balance_error)
        .def_readonly("minimum_concentration",
                      &NonuniformDiffusionDiagnostics::minimum_concentration)
        .def_readonly("maximum_concentration",
                      &NonuniformDiffusionDiagnostics::maximum_concentration)
        .def_readonly("left_outward_flux", &NonuniformDiffusionDiagnostics::left_outward_flux)
        .def_readonly("right_outward_flux", &NonuniformDiffusionDiagnostics::right_outward_flux);

    py::class_<NonuniformDiffusion1D>(
        module, "NonuniformDiffusion1D",
        "Conservative Forward Euler diffusion on a fitted nonuniform 1D mesh.")
        .def(py::init<NonuniformMesh1D, double>(), py::arg("mesh"), py::arg("diffusivity"),
             "Create a solver with one finite non-negative diffusivity value.")
        .def(py::init<NonuniformMesh1D, std::vector<double>>(), py::arg("mesh"),
             py::arg("nodal_diffusivity"),
             "Create a solver with finite non-negative nodal diffusivity values. Face values "
             "use the harmonic mean.")
        .def(
            "set_initial_condition",
            [](NonuniformDiffusion1D& solver,
               std::vector<double> concentration) -> NonuniformDiffusion1D& {
                return solver.setInitialCondition(std::move(concentration));
            },
            py::arg("concentration"), py::return_value_policy::reference_internal)
        .def("set_uniform_initial_condition", &NonuniformDiffusion1D::setUniformInitialCondition,
             py::arg("concentration"), py::return_value_policy::reference_internal)
        .def("set_boundary_condition", &NonuniformDiffusion1D::setBoundaryCondition,
             py::arg("boundary"), py::arg("condition"), py::return_value_policy::reference_internal,
             "Set a Left or Right Dirichlet/Neumann condition. Robin is rejected.")
        .def("set_dirichlet_boundary", &NonuniformDiffusion1D::setDirichletBoundary,
             py::arg("boundary"), py::arg("concentration"),
             py::return_value_policy::reference_internal)
        .def("set_neumann_boundary", &NonuniformDiffusion1D::setNeumannBoundary,
             py::arg("boundary"), py::arg("outward_normal_derivative"),
             py::return_value_policy::reference_internal,
             "Set dc/dn, where n points outward. The outward Fickian flux is -D dc/dn.")
        .def("boundary_condition", &NonuniformDiffusion1D::boundaryCondition, py::arg("boundary"),
             py::return_value_policy::copy)
        .def("check_stability", &NonuniformDiffusion1D::checkStability, py::arg("dt"))
        .def("max_stable_time_step", &NonuniformDiffusion1D::maxStableTimeStep,
             "Return the exact local conductance/control-volume Forward Euler limit.")
        .def("step", &NonuniformDiffusion1D::step, py::arg("dt"))
        .def("solve", &NonuniformDiffusion1D::solve, py::arg("dt"), py::arg("num_steps"))
        .def("solve_until", &NonuniformDiffusion1D::solveUntil, py::arg("final_time"),
             py::arg("maximum_dt"),
             "Advance to an exact absolute final time using stable equal substeps.")
        .def(
            "solution",
            [](const NonuniformDiffusion1D& solver) { return copy_to_numpy(solver.solution()); },
            "Return an owned copy of nodal concentrations.")
        .def(
            "diffusivity",
            [](const NonuniformDiffusion1D& solver) { return copy_to_numpy(solver.diffusivity()); },
            "Return an owned copy of nodal diffusivity values.")
        .def(
            "face_diffusivities",
            [](const NonuniformDiffusion1D& solver) {
                return copy_to_numpy(solver.faceDiffusivities());
            },
            "Return harmonic diffusivity at each face.")
        .def(
            "face_fluxes",
            [](const NonuniformDiffusion1D& solver) { return copy_to_numpy(solver.faceFluxes()); },
            "Return Fickian face fluxes, positive toward increasing x.")
        .def("mesh", &NonuniformDiffusion1D::mesh, py::return_value_policy::reference_internal)
        .def("time", &NonuniformDiffusion1D::time)
        .def("steps", &NonuniformDiffusion1D::steps)
        .def("total_mass", &NonuniformDiffusion1D::totalMass)
        .def("boundary_outward_flux", &NonuniformDiffusion1D::boundaryOutwardFlux,
             py::arg("boundary"), "Return physical Fickian flux leaving the boundary.")
        .def("reset_balance_reference", &NonuniformDiffusion1D::resetBalanceReference)
        .def("diagnostics", &NonuniformDiffusion1D::diagnostics);
}

}  // namespace bindings
}  // namespace biotransport
