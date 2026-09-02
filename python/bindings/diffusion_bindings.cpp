/**
 * @file diffusion_bindings.cpp
 * @brief Python bindings for diffusion and reaction-diffusion solvers
 */

#include "diffusion_bindings.hpp"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "binding_helpers.hpp"
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/mesh/structured_mesh_3d.hpp>
#include <biotransport/core/problems/transport_problem.hpp>
#include <biotransport/physics/heat_transfer/bioheat_cryotherapy.hpp>
#include <biotransport/physics/mass_transport/gray_scott.hpp>
#include <biotransport/physics/mass_transport/membrane_diffusion.hpp>
#include <biotransport/physics/mass_transport/tumor_drug_delivery.hpp>
#include <biotransport/physics/reactions.hpp>
#include <biotransport/solvers/adi_solver.hpp>
#include <biotransport/solvers/advection_diffusion_solver.hpp>
#include <biotransport/solvers/crank_nicolson.hpp>
#include <biotransport/solvers/diffusion_solver_3d.hpp>
#include <biotransport/solvers/diffusion_solvers.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <biotransport/solvers/multi_species_solver.hpp>
#include <biotransport/solvers/nernst_planck_solver.hpp>
#include <cmath>
#include <string>

namespace biotransport {
namespace bindings {

namespace {

Boundary checkedBoundary(int boundary_id) {
    if (boundary_id < to_index(Boundary::Left) || boundary_id > to_index(Boundary::Top)) {
        throw py::value_error("boundary_id must be between 0 (Left) and 3 (Top)");
    }
    return static_cast<Boundary>(boundary_id);
}

Boundary3D checkedBoundary3D(int boundary_id) {
    constexpr int first_boundary = static_cast<int>(Boundary3D::XMin);
    constexpr int last_boundary = static_cast<int>(Boundary3D::ZMax);
    if (boundary_id < first_boundary || boundary_id > last_boundary) {
        throw py::value_error("boundary_id must be between 0 (XMin) and 5 (ZMax)");
    }
    return static_cast<Boundary3D>(boundary_id);
}

void copyPythonReactionRates(py::handle values, std::vector<double>& rates,
                             const char* value_source) {
    if (py::isinstance<py::array>(values)) {
        const auto array = py::reinterpret_borrow<py::array>(values);
        if (array.ndim() != 1) {
            throw py::value_error(std::string("reaction callback ") + value_source +
                                  " must be one-dimensional; got " + std::to_string(array.ndim()) +
                                  " dimensions");
        }
    }

    if (PyUnicode_Check(values.ptr()) || PyBytes_Check(values.ptr()) ||
        !PySequence_Check(values.ptr())) {
        throw py::type_error(
            "reaction callback must return None after mutating rates, or return a "
            "one-dimensional numeric sequence");
    }

    const auto sequence = py::reinterpret_borrow<py::sequence>(values);
    const auto actual_size = static_cast<std::size_t>(py::len(sequence));
    for (std::size_t i = 0; i < actual_size; ++i) {
        const py::object item = sequence[static_cast<py::ssize_t>(i)];
        if (!PyUnicode_Check(item.ptr()) && !PyBytes_Check(item.ptr()) &&
            PySequence_Check(item.ptr())) {
            throw py::value_error(std::string("reaction callback ") + value_source +
                                  " must be one-dimensional");
        }
    }
    if (actual_size != rates.size()) {
        throw py::value_error(std::string("reaction callback ") + value_source + " contains " +
                              std::to_string(actual_size) + " rates; expected exactly " +
                              std::to_string(rates.size()));
    }

    std::vector<double> validated_rates(rates.size());
    for (std::size_t i = 0; i < rates.size(); ++i) {
        try {
            validated_rates[i] = py::cast<double>(sequence[static_cast<py::ssize_t>(i)]);
        } catch (const py::cast_error&) {
            throw py::type_error("reaction callback rate at index " + std::to_string(i) +
                                 " must be a real number");
        }
        if (!std::isfinite(validated_rates[i])) {
            throw py::value_error("reaction callback rate at index " + std::to_string(i) +
                                  " must be finite");
        }
    }
    rates = std::move(validated_rates);
}

}  // namespace

void register_diffusion_bindings(py::module_& m) {
    // =========================================================================
    // DiffusionSolver (base class)
    // =========================================================================
    py::class_<DiffusionSolver>(m, "DiffusionSolver")
        .def(py::init<const StructuredMesh&, double>(), py::arg("mesh"), py::arg("diffusivity"),
             py::keep_alive<1, 2>())
        .def("set_initial_condition", &DiffusionSolver::setInitialCondition, py::arg("values"))
        .def(
            "set_dirichlet_boundary",
            [](DiffusionSolver& solver, int boundary_id, double value) {
                solver.setDirichletBoundary(checkedBoundary(boundary_id), value);
            },
            py::arg("boundary_id"), py::arg("value"))
        .def("set_dirichlet_boundary",
             py::overload_cast<Boundary, double>(&DiffusionSolver::setDirichletBoundary),
             py::arg("boundary"), py::arg("value"))
        .def(
            "set_neumann_boundary",
            [](DiffusionSolver& solver, int boundary_id, double normal_derivative) {
                solver.setNeumannBoundary(checkedBoundary(boundary_id), normal_derivative);
            },
            py::arg("boundary_id"), py::arg("normal_derivative"))
        .def("set_neumann_boundary",
             py::overload_cast<Boundary, double>(&DiffusionSolver::setNeumannBoundary),
             py::arg("boundary"), py::arg("normal_derivative"))
        .def(
            "set_boundary_condition",
            [](DiffusionSolver& solver, int boundary_id, const BoundaryCondition& bc) {
                solver.setBoundaryCondition(checkedBoundary(boundary_id), bc);
            },
            py::arg("boundary_id"), py::arg("bc"))
        .def("set_boundary_condition",
             py::overload_cast<Boundary, const BoundaryCondition&>(
                 &DiffusionSolver::setBoundaryCondition),
             py::arg("boundary"), py::arg("bc"))
        .def(
            "time", [](const DiffusionSolver& s) { return s.time(); },
            "Current simulation time advanced by solve()")
        .def(
            "check_stability",
            [](const DiffusionSolver& s, double dt) { return s.checkStability(dt); }, py::arg("dt"),
            "Return whether dt satisfies the explicit stability condition")
        .def("max_stable_time_step", &DiffusionSolver::maxStableTimeStep,
             "Largest explicit step accepted by check_stability() for pure diffusion; "
             "infinity when the diffusivity is zero")
        .def("mesh", &DiffusionSolver::mesh, py::return_value_policy::reference_internal,
             "The mesh this solver was built on")
        .def("solve", &DiffusionSolver::solve, py::arg("dt"), py::arg("num_steps"))
        .def("solution", [](const DiffusionSolver& solver) {
            return to_numpy_with_base(solver.solution(), py::cast(&solver));
        });

    // =========================================================================
    // CrankNicolsonDiffusion (implicit solver)
    // =========================================================================
    py::class_<CNSolveResult>(m, "CNSolveResult", "Result of a Crank-Nicolson solve step")
        .def(py::init<>())
        .def_readonly("iterations", &CNSolveResult::iterations, "Number of iterations used")
        .def_readonly("residual", &CNSolveResult::residual, "Final residual norm")
        .def_readonly("converged", &CNSolveResult::converged, "Whether tolerance was achieved");

    py::class_<CrankNicolsonDiffusion>(m, "CrankNicolsonDiffusion",
                                       R"(Crank-Nicolson implicit solver for the diffusion equation.

        The linear diffusion update is A-stable and second-order in time, but
        it is not L-stable: very large steps can produce bounded oscillations
        and poor temporal accuracy. Algebraic convergence is checked explicitly.

        Example:
            >>> mesh = bt.StructuredMesh(100, 0.0, 1.0)
            >>> solver = bt.CrankNicolsonDiffusion(mesh, 1e-5)
            >>> solver.set_initial_condition(u0)
            >>> solver.set_dirichlet_boundary(bt.Boundary.Left, 1.0)
            >>> solver.solve(dt=0.1, num_steps=100)  # dt >> explicit CFL limit
        )")
        .def(py::init<const StructuredMesh&, double>(), py::arg("mesh"), py::arg("diffusivity"),
             py::keep_alive<1, 2>(), "Create a Crank-Nicolson diffusion solver")
        .def("set_initial_condition", &CrankNicolsonDiffusion::setInitialCondition,
             py::arg("values"), "Set the initial condition")
        .def("set_dirichlet_boundary", &CrankNicolsonDiffusion::setDirichletBoundary,
             py::arg("boundary"), py::arg("value"), "Set a Dirichlet boundary condition")
        .def("set_neumann_boundary", &CrankNicolsonDiffusion::setNeumannBoundary,
             py::arg("boundary"), py::arg("normal_derivative"),
             "Set the outward-normal derivative du/dn (not physical flux)")
        .def("set_tolerance", &CrankNicolsonDiffusion::setTolerance, py::arg("tol"),
             "Set convergence tolerance for implicit solve")
        .def("set_max_iterations", &CrankNicolsonDiffusion::setMaxIterations, py::arg("max_iter"),
             "Set maximum iterations for implicit solve")
        .def("step", &CrankNicolsonDiffusion::step, py::arg("dt"),
             "Advance solution by one time step, returns CNSolveResult")
        .def("mesh", &CrankNicolsonDiffusion::mesh, py::return_value_policy::reference_internal,
             "The mesh this solver was built on")
        .def("solve", &CrankNicolsonDiffusion::solve, py::arg("dt"), py::arg("num_steps"),
             "Run solver for specified number of steps")
        .def(
            "solution",
            [](const CrankNicolsonDiffusion& solver) {
                return to_numpy_with_base(solver.solution(), py::cast(&solver));
            },
            "Return an owned copy of the current solution")
        .def("time", &CrankNicolsonDiffusion::time, "Get current simulation time")
        .def_property_readonly("diffusivity", &CrankNicolsonDiffusion::diffusivity,
                               "Diffusion coefficient");

    // =========================================================================
    // ADI Solvers (Alternating Direction Implicit)
    // =========================================================================
    py::class_<ADISolveResult>(m, "ADISolveResult", "Result of an ADI solve step")
        .def(py::init<>())
        .def_readonly("steps", &ADISolveResult::steps, "Number of time steps completed")
        .def_readonly("substeps", &ADISolveResult::substeps,
                      "Directional solves (3 per 2D step, 5 per 3D step)")
        .def_readonly("time", &ADISolveResult::time, "Current simulation time after step()")
        .def_readonly("total_time", &ADISolveResult::total_time,
                      "Total simulation time after solve()")
        .def_readonly("success", &ADISolveResult::success,
                      "Whether the step completed successfully");

    py::class_<ADIDiffusion2D>(m, "ADIDiffusion2D",
                               R"(2D symmetric directionally split Crank-Nicolson solver.

        Uses x/2-y-x/2 symmetric composition. Each directional linear-diffusion
        subproblem is unconditionally stable; time-independent boundary data and
        smooth solutions are required for the stated second-order convergence.

        Example:
            >>> mesh = bt.StructuredMesh(50, 50, 0.0, 1.0, 0.0, 1.0)
            >>> solver = bt.ADIDiffusion2D(mesh, 1e-5)
            >>> solver.set_initial_condition(u0)
            >>> solver.set_dirichlet_boundary(bt.Boundary.Left, 100.0)
            >>> solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)
            >>> solver.solve(dt=0.1, num_steps=100)  # No CFL restriction
        )")
        .def(py::init<const StructuredMesh&, double>(), py::arg("mesh"), py::arg("diffusivity"),
             py::keep_alive<1, 2>(), "Create a 2D ADI diffusion solver")
        .def("set_initial_condition", &ADIDiffusion2D::setInitialCondition, py::arg("values"),
             "Set the initial condition")
        .def("set_dirichlet_boundary", &ADIDiffusion2D::setDirichletBoundary, py::arg("boundary"),
             py::arg("value"), "Set a Dirichlet boundary condition")
        .def("set_neumann_boundary", &ADIDiffusion2D::setNeumannBoundary, py::arg("boundary"),
             py::arg("normal_derivative"),
             "Set the outward-normal derivative du/dn (not physical flux)")
        .def("step", &ADIDiffusion2D::step, py::arg("dt"),
             "Advance solution by one time step, returns ADISolveResult")
        .def("mesh", &ADIDiffusion2D::mesh, py::return_value_policy::reference_internal,
             "The mesh this solver was built on")
        .def("solve", &ADIDiffusion2D::solve, py::arg("dt"), py::arg("num_steps"),
             "Run solver for specified number of steps")
        .def(
            "solution",
            [](const ADIDiffusion2D& solver) {
                return to_numpy_with_base(solver.solution(), py::cast(&solver));
            },
            "Return an owned copy of the current solution")
        .def("time", &ADIDiffusion2D::time, "Get current simulation time")
        .def_property_readonly("diffusivity", &ADIDiffusion2D::diffusivity,
                               "Diffusion coefficient");

    py::class_<ADIDiffusion3D>(m, "ADIDiffusion3D",
                               R"(3D symmetric directionally split Crank-Nicolson solver.

        Uses x/2-y/2-z-y/2-x/2 symmetric composition. Each directional
        linear-diffusion subproblem is unconditionally stable; time-independent
        boundary data and smooth solutions are required for second-order convergence.

        Example:
            >>> mesh = bt.StructuredMesh3D(20, 20, 20, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
            >>> solver = bt.ADIDiffusion3D(mesh, 1e-5)
            >>> solver.set_initial_condition(u0)
            >>> solver.set_dirichlet_boundary(bt.Boundary3D.XMin, 100.0)
            >>> solver.set_dirichlet_boundary(bt.Boundary3D.XMax, 0.0)
            >>> solver.solve(dt=0.1, num_steps=100)
        )")
        .def(py::init<const StructuredMesh3D&, double>(), py::arg("mesh"), py::arg("diffusivity"),
             py::keep_alive<1, 2>(), "Create a 3D ADI diffusion solver")
        .def("set_initial_condition", &ADIDiffusion3D::setInitialCondition, py::arg("values"),
             "Set the initial condition")
        .def("set_dirichlet_boundary",
             py::overload_cast<Boundary3D, double>(&ADIDiffusion3D::setDirichletBoundary),
             py::arg("boundary"), py::arg("value"), "Set a Dirichlet boundary condition")
        .def(
            "set_dirichlet_boundary",
            [](ADIDiffusion3D& solver, int boundary_id, double value) {
                solver.setDirichletBoundary(checkedBoundary3D(boundary_id), value);
            },
            py::arg("boundary_id"), py::arg("value"), "Set a Dirichlet BC by integer ID")
        .def("set_neumann_boundary",
             py::overload_cast<Boundary3D, double>(&ADIDiffusion3D::setNeumannBoundary),
             py::arg("boundary"), py::arg("normal_derivative"),
             "Set the outward-normal derivative du/dn (not physical flux)")
        .def(
            "set_neumann_boundary",
            [](ADIDiffusion3D& solver, int boundary_id, double normal_derivative) {
                solver.setNeumannBoundary(checkedBoundary3D(boundary_id), normal_derivative);
            },
            py::arg("boundary_id"), py::arg("normal_derivative"),
            "Set outward-normal derivative by integer boundary ID")
        .def("step", &ADIDiffusion3D::step, py::arg("dt"),
             "Advance solution by one time step, returns ADISolveResult")
        .def("mesh", &ADIDiffusion3D::mesh, py::return_value_policy::reference_internal,
             "The mesh this solver was built on")
        .def("solve", &ADIDiffusion3D::solve, py::arg("dt"), py::arg("num_steps"),
             "Run solver for specified number of steps")
        .def(
            "solution",
            [](const ADIDiffusion3D& solver) {
                return to_numpy_with_base(solver.solution(), py::cast(&solver));
            },
            "Return an owned copy of the current solution")
        .def("time", &ADIDiffusion3D::time, "Get current simulation time")
        .def_property_readonly("diffusivity", &ADIDiffusion3D::diffusivity,
                               "Diffusion coefficient");

    // =========================================================================
    // Derived Diffusion Solvers
    // =========================================================================

    // ReactionDiffusionSolver (custom reaction function)
    py::class_<ReactionDiffusionSolver>(m, "ReactionDiffusionSolver")
        .def(py::init<const StructuredMesh&, double, ReactionDiffusionSolver::ReactionFunction>(),
             py::arg("mesh"), py::arg("diffusivity"), py::arg("reaction"), py::keep_alive<1, 2>())
        .def(
            "time", [](const ReactionDiffusionSolver& s) { return s.time(); },
            "Current simulation time advanced by solve()")
        .def(
            "check_stability",
            [](const ReactionDiffusionSolver& s, double dt) { return s.checkStability(dt); },
            py::arg("dt"), "Return whether dt satisfies the explicit stability condition")
        .def("mesh", &ReactionDiffusionSolver::mesh, py::return_value_policy::reference_internal,
             "The mesh this solver was built on")
        .def("solve", &ReactionDiffusionSolver::solve, py::arg("dt"), py::arg("num_steps"))
        .def("solution",
             [](const ReactionDiffusionSolver& solver) {
                 return to_numpy_with_base(solver.solution(), py::cast(&solver));
             })
        .def("set_initial_condition", &ReactionDiffusionSolver::setInitialCondition,
             py::arg("values"))
        .def(
            "set_dirichlet_boundary",
            [](ReactionDiffusionSolver& solver, int boundary_id, double value) {
                solver.setDirichletBoundary(checkedBoundary(boundary_id), value);
            },
            py::arg("boundary_id"), py::arg("value"))
        .def("set_dirichlet_boundary",
             py::overload_cast<Boundary, double>(&ReactionDiffusionSolver::setDirichletBoundary),
             py::arg("boundary"), py::arg("value"))
        .def(
            "set_neumann_boundary",
            [](ReactionDiffusionSolver& solver, int boundary_id, double normal_derivative) {
                solver.setNeumannBoundary(checkedBoundary(boundary_id), normal_derivative);
            },
            py::arg("boundary_id"), py::arg("normal_derivative"))
        .def("set_neumann_boundary",
             py::overload_cast<Boundary, double>(&ReactionDiffusionSolver::setNeumannBoundary),
             py::arg("boundary"), py::arg("normal_derivative"))
        .def(
            "set_boundary",
            [](ReactionDiffusionSolver& solver, int boundary_id, const BoundaryCondition& bc) {
                solver.setBoundaryCondition(checkedBoundary(boundary_id), bc);
            },
            py::arg("boundary_id"), py::arg("bc"))
        .def(
            "set_boundary",
            [](ReactionDiffusionSolver& solver, Boundary boundary, const BoundaryCondition& bc) {
                solver.setBoundaryCondition(boundary, bc);
            },
            py::arg("boundary"), py::arg("bc"));

    // Linear reaction-diffusion (first-order decay)
    py::class_<LinearReactionDiffusionSolver>(m, "LinearReactionDiffusionSolver")
        .def(py::init<const StructuredMesh&, double, double>(), py::arg("mesh"),
             py::arg("diffusivity"), py::arg("decay_rate"), py::keep_alive<1, 2>())
        .def(
            "time", [](const LinearReactionDiffusionSolver& s) { return s.time(); },
            "Current simulation time advanced by solve()")
        .def(
            "check_stability",
            [](const LinearReactionDiffusionSolver& s, double dt) { return s.checkStability(dt); },
            py::arg("dt"), "Return whether dt satisfies the explicit stability condition")
        .def("mesh", &LinearReactionDiffusionSolver::mesh,
             py::return_value_policy::reference_internal, "The mesh this solver was built on")
        .def("solve", &LinearReactionDiffusionSolver::solve, py::arg("dt"), py::arg("num_steps"))
        .def("solution",
             [](const LinearReactionDiffusionSolver& solver) {
                 return to_numpy_with_base(solver.solution(), py::cast(&solver));
             })
        .def("set_initial_condition", &LinearReactionDiffusionSolver::setInitialCondition,
             py::arg("values"))
        .def(
            "set_boundary",
            [](LinearReactionDiffusionSolver& solver, int boundary_id,
               const BoundaryCondition& bc) {
                solver.setBoundaryCondition(checkedBoundary(boundary_id), bc);
            },
            py::arg("boundary_id"), py::arg("bc"))
        .def(
            "set_boundary",
            [](LinearReactionDiffusionSolver& solver, Boundary boundary,
               const BoundaryCondition& bc) { solver.setBoundaryCondition(boundary, bc); },
            py::arg("boundary"), py::arg("bc"));

    // Logistic reaction-diffusion
    py::class_<LogisticReactionDiffusionSolver>(m, "LogisticReactionDiffusionSolver")
        .def(py::init<const StructuredMesh&, double, double, double>(), py::arg("mesh"),
             py::arg("diffusivity"), py::arg("growth_rate"), py::arg("carrying_capacity"),
             py::keep_alive<1, 2>())
        .def(
            "time", [](const LogisticReactionDiffusionSolver& s) { return s.time(); },
            "Current simulation time advanced by solve()")
        .def(
            "check_stability",
            [](const LogisticReactionDiffusionSolver& s, double dt) {
                return s.checkStability(dt);
            },
            py::arg("dt"), "Return whether dt satisfies the explicit stability condition")
        .def("mesh", &LogisticReactionDiffusionSolver::mesh,
             py::return_value_policy::reference_internal, "The mesh this solver was built on")
        .def("solve", &LogisticReactionDiffusionSolver::solve, py::arg("dt"), py::arg("num_steps"))
        .def("solution",
             [](const LogisticReactionDiffusionSolver& solver) {
                 return to_numpy_with_base(solver.solution(), py::cast(&solver));
             })
        .def("set_initial_condition", &LogisticReactionDiffusionSolver::setInitialCondition,
             py::arg("values"))
        .def(
            "set_boundary",
            [](LogisticReactionDiffusionSolver& solver, int boundary_id,
               const BoundaryCondition& bc) {
                solver.setBoundaryCondition(checkedBoundary(boundary_id), bc);
            },
            py::arg("boundary_id"), py::arg("bc"))
        .def(
            "set_boundary",
            [](LogisticReactionDiffusionSolver& solver, Boundary boundary,
               const BoundaryCondition& bc) { solver.setBoundaryCondition(boundary, bc); },
            py::arg("boundary"), py::arg("bc"));

    // Michaelis-Menten reaction-diffusion
    py::class_<MichaelisMentenReactionDiffusionSolver>(m, "MichaelisMentenReactionDiffusionSolver")
        .def(py::init<const StructuredMesh&, double, double, double>(), py::arg("mesh"),
             py::arg("diffusivity"), py::arg("vmax"), py::arg("km"), py::keep_alive<1, 2>())
        .def(
            "time", [](const MichaelisMentenReactionDiffusionSolver& s) { return s.time(); },
            "Current simulation time advanced by solve()")
        .def(
            "check_stability",
            [](const MichaelisMentenReactionDiffusionSolver& s, double dt) {
                return s.checkStability(dt);
            },
            py::arg("dt"), "Return whether dt satisfies the explicit stability condition")
        .def("mesh", &MichaelisMentenReactionDiffusionSolver::mesh,
             py::return_value_policy::reference_internal, "The mesh this solver was built on")
        .def("solve", &MichaelisMentenReactionDiffusionSolver::solve, py::arg("dt"),
             py::arg("num_steps"))
        .def("solution",
             [](const MichaelisMentenReactionDiffusionSolver& solver) {
                 return to_numpy_with_base(solver.solution(), py::cast(&solver));
             })
        .def("set_initial_condition", &MichaelisMentenReactionDiffusionSolver::setInitialCondition,
             py::arg("values"))
        .def(
            "set_boundary",
            [](MichaelisMentenReactionDiffusionSolver& solver, int boundary_id,
               const BoundaryCondition& bc) {
                solver.setBoundaryCondition(checkedBoundary(boundary_id), bc);
            },
            py::arg("boundary_id"), py::arg("bc"))
        .def(
            "set_boundary",
            [](MichaelisMentenReactionDiffusionSolver& solver, Boundary boundary,
               const BoundaryCondition& bc) { solver.setBoundaryCondition(boundary, bc); },
            py::arg("boundary"), py::arg("bc"));

    // Masked Michaelis-Menten
    py::class_<MaskedMichaelisMentenReactionDiffusionSolver>(
        m, "MaskedMichaelisMentenReactionDiffusionSolver")
        .def(py::init<const StructuredMesh&, double, double, double, std::vector<std::uint8_t>,
                      double>(),
             py::arg("mesh"), py::arg("diffusivity"), py::arg("vmax"), py::arg("km"),
             py::arg("mask"), py::arg("pinned_value"), py::keep_alive<1, 2>())
        .def(
            "time", [](const MaskedMichaelisMentenReactionDiffusionSolver& s) { return s.time(); },
            "Current simulation time advanced by solve()")
        .def(
            "check_stability",
            [](const MaskedMichaelisMentenReactionDiffusionSolver& s, double dt) {
                return s.checkStability(dt);
            },
            py::arg("dt"), "Return whether dt satisfies the explicit stability condition")
        .def("mesh", &MaskedMichaelisMentenReactionDiffusionSolver::mesh,
             py::return_value_policy::reference_internal, "The mesh this solver was built on")
        .def("solve", &MaskedMichaelisMentenReactionDiffusionSolver::solve, py::arg("dt"),
             py::arg("num_steps"))
        .def("solution",
             [](const MaskedMichaelisMentenReactionDiffusionSolver& solver) {
                 return to_numpy_with_base(solver.solution(), py::cast(&solver));
             })
        .def("set_initial_condition",
             &MaskedMichaelisMentenReactionDiffusionSolver::setInitialCondition, py::arg("values"))
        .def(
            "set_boundary",
            [](MaskedMichaelisMentenReactionDiffusionSolver& solver, int boundary_id,
               const BoundaryCondition& bc) {
                solver.setBoundaryCondition(checkedBoundary(boundary_id), bc);
            },
            py::arg("boundary_id"), py::arg("bc"))
        .def(
            "set_boundary",
            [](MaskedMichaelisMentenReactionDiffusionSolver& solver, Boundary boundary,
               const BoundaryCondition& bc) { solver.setBoundaryCondition(boundary, bc); },
            py::arg("boundary"), py::arg("bc"));

    // Constant source reaction-diffusion
    py::class_<ConstantSourceReactionDiffusionSolver>(m, "ConstantSourceReactionDiffusionSolver")
        .def(py::init<const StructuredMesh&, double, double>(), py::arg("mesh"),
             py::arg("diffusivity"), py::arg("source_rate"), py::keep_alive<1, 2>())
        .def(
            "time", [](const ConstantSourceReactionDiffusionSolver& s) { return s.time(); },
            "Current simulation time advanced by solve()")
        .def(
            "check_stability",
            [](const ConstantSourceReactionDiffusionSolver& s, double dt) {
                return s.checkStability(dt);
            },
            py::arg("dt"), "Return whether dt satisfies the explicit stability condition")
        .def("mesh", &ConstantSourceReactionDiffusionSolver::mesh,
             py::return_value_policy::reference_internal, "The mesh this solver was built on")
        .def("solve", &ConstantSourceReactionDiffusionSolver::solve, py::arg("dt"),
             py::arg("num_steps"))
        .def("solution",
             [](const ConstantSourceReactionDiffusionSolver& solver) {
                 return to_numpy_with_base(solver.solution(), py::cast(&solver));
             })
        .def("set_initial_condition", &ConstantSourceReactionDiffusionSolver::setInitialCondition,
             py::arg("values"))
        .def(
            "set_boundary",
            [](ConstantSourceReactionDiffusionSolver& solver, int boundary_id,
               const BoundaryCondition& bc) {
                solver.setBoundaryCondition(checkedBoundary(boundary_id), bc);
            },
            py::arg("boundary_id"), py::arg("bc"))
        .def(
            "set_boundary",
            [](ConstantSourceReactionDiffusionSolver& solver, Boundary boundary,
               const BoundaryCondition& bc) { solver.setBoundaryCondition(boundary, bc); },
            py::arg("boundary"), py::arg("bc"));

    // =========================================================================
    // Gray-Scott (two-species pattern formation)
    // =========================================================================
    py::class_<GrayScottRunResult>(m, "GrayScottRunResult")
        .def(py::init<>())
        .def_readonly("nx", &GrayScottRunResult::nx)
        .def_readonly("ny", &GrayScottRunResult::ny)
        .def_readonly("frames", &GrayScottRunResult::frames)
        .def_readonly("steps_run", &GrayScottRunResult::steps_run)
        .def_readonly("final_time", &GrayScottRunResult::final_time)
        .def_readonly("frame_steps", &GrayScottRunResult::frame_steps)
        .def("u_frames",
             [](const GrayScottRunResult& r) {
                 return to_numpy_3d(r.u_frames, static_cast<py::ssize_t>(r.frames),
                                    static_cast<py::ssize_t>(r.ny), static_cast<py::ssize_t>(r.nx),
                                    py::cast(&r));
             })
        .def("v_frames", [](const GrayScottRunResult& r) {
            return to_numpy_3d(r.v_frames, static_cast<py::ssize_t>(r.frames),
                               static_cast<py::ssize_t>(r.ny), static_cast<py::ssize_t>(r.nx),
                               py::cast(&r));
        });

    py::class_<GrayScottSolver>(m, "GrayScottSolver")
        .def(py::init<const StructuredMesh&, double, double, double, double>(), py::arg("mesh"),
             py::arg("Du"), py::arg("Dv"), py::arg("f"), py::arg("k"), py::keep_alive<1, 2>())
        .def("simulate", &GrayScottSolver::simulate, py::arg("u0"), py::arg("v0"),
             py::arg("total_steps"), py::arg("dt"), py::arg("steps_between_frames") = 1000,
             py::arg("check_interval") = 1000, py::arg("stable_tol") = 1e-4,
             py::arg("min_frames_before_early_stop") = 6);

    // =========================================================================
    // Tumor drug delivery (pressure + transport)
    // =========================================================================
    py::class_<TumorDrugDeliverySaved>(m, "TumorDrugDeliverySaved")
        .def(py::init<>())
        .def_readonly("nx", &TumorDrugDeliverySaved::nx)
        .def_readonly("ny", &TumorDrugDeliverySaved::ny)
        .def_readonly("frames", &TumorDrugDeliverySaved::frames)
        .def_readonly("times_s", &TumorDrugDeliverySaved::times_s)
        .def_readonly("final_time_s", &TumorDrugDeliverySaved::final_time_s)
        .def_readonly("stability_limit_s", &TumorDrugDeliverySaved::stability_limit_s)
        .def_readonly("free_amount_per_depth", &TumorDrugDeliverySaved::free_amount_per_depth)
        .def_readonly("bound_amount_per_depth", &TumorDrugDeliverySaved::bound_amount_per_depth)
        .def_readonly("cellular_amount_per_depth",
                      &TumorDrugDeliverySaved::cellular_amount_per_depth)
        .def_readonly("total_amount_per_depth", &TumorDrugDeliverySaved::total_amount_per_depth)
        .def_readonly("cumulative_net_vascular_exchange_per_depth",
                      &TumorDrugDeliverySaved::cumulative_net_vascular_exchange_per_depth)
        .def_readonly("cumulative_boundary_outflow_per_depth",
                      &TumorDrugDeliverySaved::cumulative_boundary_outflow_per_depth)
        .def_readonly("mass_balance_error_per_depth",
                      &TumorDrugDeliverySaved::mass_balance_error_per_depth)
        .def("free",
             [](const TumorDrugDeliverySaved& r) {
                 return to_numpy_3d(r.free, static_cast<py::ssize_t>(r.frames),
                                    static_cast<py::ssize_t>(r.ny), static_cast<py::ssize_t>(r.nx),
                                    py::cast(&r));
             })
        .def("bound",
             [](const TumorDrugDeliverySaved& r) {
                 return to_numpy_3d(r.bound, static_cast<py::ssize_t>(r.frames),
                                    static_cast<py::ssize_t>(r.ny), static_cast<py::ssize_t>(r.nx),
                                    py::cast(&r));
             })
        .def("cellular",
             [](const TumorDrugDeliverySaved& r) {
                 return to_numpy_3d(r.cellular, static_cast<py::ssize_t>(r.frames),
                                    static_cast<py::ssize_t>(r.ny), static_cast<py::ssize_t>(r.nx),
                                    py::cast(&r));
             })
        .def("total", [](const TumorDrugDeliverySaved& r) {
            return to_numpy_3d(r.total, static_cast<py::ssize_t>(r.frames),
                               static_cast<py::ssize_t>(r.ny), static_cast<py::ssize_t>(r.nx),
                               py::cast(&r));
        });

    py::class_<TumorDrugDeliverySolver>(m, "TumorDrugDeliverySolver")
        .def(py::init<const StructuredMesh&, std::vector<std::uint8_t>, std::vector<double>, double,
                      double>(),
             py::arg("mesh"), py::arg("tumor_mask"), py::arg("hydraulic_conductivity"),
             py::arg("p_boundary"), py::arg("p_tumor"), py::keep_alive<1, 2>())
        .def("solve_pressure_sor", &TumorDrugDeliverySolver::solvePressureSOR,
             py::arg("max_iter") = 20000, py::arg("tol") = 1e-10, py::arg("omega") = 1.8)
        .def("simulate", &TumorDrugDeliverySolver::simulate, py::arg("pressure"),
             py::arg("diffusivity"), py::arg("vessel_wall_solute_permeability"),
             py::arg("vascular_surface_area_density"), py::arg("k_binding"), py::arg("k_uptake"),
             py::arg("c_plasma"), py::arg("dt"), py::arg("num_steps"), py::arg("times_to_save_s"));

    // =========================================================================
    // Bioheat cryotherapy (temperature + damage)
    // =========================================================================
    py::class_<BioheatSaved>(m, "BioheatSaved")
        .def(py::init<>())
        .def_readonly("nx", &BioheatSaved::nx)
        .def_readonly("ny", &BioheatSaved::ny)
        .def_readonly("frames", &BioheatSaved::frames)
        .def_readonly("times_s", &BioheatSaved::times_s)
        .def_readonly("minimum_temperature_K", &BioheatSaved::minimum_temperature_K)
        .def_readonly("maximum_temperature_K", &BioheatSaved::maximum_temperature_K)
        .def_readonly("maximum_stable_dt_s", &BioheatSaved::maximum_stable_dt_s)
        .def("temperature_K",
             [](const BioheatSaved& r) {
                 return to_numpy_3d(r.temperature_K, static_cast<py::ssize_t>(r.frames),
                                    static_cast<py::ssize_t>(r.ny), static_cast<py::ssize_t>(r.nx),
                                    py::cast(&r));
             })
        .def("damage",
             [](const BioheatSaved& r) {
                 return to_numpy_3d(r.damage, static_cast<py::ssize_t>(r.frames),
                                    static_cast<py::ssize_t>(r.ny), static_cast<py::ssize_t>(r.nx),
                                    py::cast(&r));
             })
        .def("frozen_fraction", [](const BioheatSaved& r) {
            return to_numpy_3d(r.frozen_fraction, static_cast<py::ssize_t>(r.frames),
                               static_cast<py::ssize_t>(r.ny), static_cast<py::ssize_t>(r.nx),
                               py::cast(&r));
        });

    py::class_<BioheatCryotherapySolver>(m, "BioheatCryotherapySolver")
        .def(py::init<const StructuredMesh&, std::vector<std::uint8_t>, std::vector<double>,
                      std::vector<double>, double, double, double, double, double, double, double,
                      double, double, double, double, double, double, double, double>(),
             py::arg("mesh"), py::arg("probe_mask"), py::arg("perfusion_map"), py::arg("q_met_map"),
             py::arg("rho_tissue"), py::arg("rho_blood"), py::arg("c_blood"), py::arg("k_unfrozen"),
             py::arg("k_frozen"), py::arg("c_unfrozen"), py::arg("c_frozen"), py::arg("T_body_K"),
             py::arg("T_probe_K"), py::arg("T_freeze_K"), py::arg("T_freeze_range_K"),
             py::arg("L_fusion"), py::arg("A"), py::arg("E_a"), py::arg("R_gas"))
        .def("set_initial_temperature_K", &BioheatCryotherapySolver::setInitialTemperatureK,
             py::arg("temperature_K"), py::return_value_policy::reference_internal)
        .def("set_initial_temperature_field_K",
             &BioheatCryotherapySolver::setInitialTemperatureFieldK, py::arg("temperature_K"),
             py::return_value_policy::reference_internal)
        .def("set_arterial_temperature_K", &BioheatCryotherapySolver::setArterialTemperatureK,
             py::arg("temperature_K"), py::return_value_policy::reference_internal)
        .def("set_boundary_temperature_K", &BioheatCryotherapySolver::setBoundaryTemperatureK,
             py::arg("temperature_K"), py::return_value_policy::reference_internal)
        .def("frozen_fraction", &BioheatCryotherapySolver::frozenFraction, py::arg("temperature_K"))
        .def("thermal_conductivity", &BioheatCryotherapySolver::thermalConductivity,
             py::arg("temperature_K"))
        .def("effective_specific_heat", &BioheatCryotherapySolver::effectiveSpecificHeat,
             py::arg("temperature_K"))
        .def("arrhenius_heat_injury_rate", &BioheatCryotherapySolver::arrheniusHeatInjuryRate,
             py::arg("temperature_K"))
        .def("maximum_stable_time_step_s", &BioheatCryotherapySolver::maximumStableTimeStep)
        .def("simulate", &BioheatCryotherapySolver::simulate, py::arg("dt"), py::arg("num_steps"),
             py::arg("times_to_save_s"));

    // =========================================================================
    // Membrane Diffusion
    // =========================================================================
    py::class_<MembraneDiffusionResult>(
        m, "MembraneDiffusionResult",
        "Steady 1D membrane result. Concentration and flux use the caller's consistent "
        "amount unit; this API does not assume that amount is molar.")
        .def(py::init<>())
        .def_readonly("flux", &MembraneDiffusionResult::flux, "Steady flux [amount/(m² s)]")
        .def_readonly("permeability", &MembraneDiffusionResult::permeability,
                      "External-concentration permeability P [m/s]")
        .def_readonly("effective_diffusivity", &MembraneDiffusionResult::effective_diffusivity,
                      "Equivalent external-gradient coefficient P*L [m²/s], including "
                      "partition and any enabled hindrance")
        .def(
            "x", [](const MembraneDiffusionResult& r) { return to_numpy(r.x); },
            "Return an owned coordinate array [m]")
        .def(
            "concentration",
            [](const MembraneDiffusionResult& r) { return to_numpy(r.concentration); },
            "Return an owned intramembrane concentration profile [amount/m³]");

    py::class_<MembraneDiffusion1DSolver>(m, "MembraneDiffusion1DSolver")
        .def(py::init<>())
        .def("set_membrane_thickness", &MembraneDiffusion1DSolver::setMembraneThickness,
             py::arg("L"), py::return_value_policy::reference_internal)
        .def("set_diffusivity", &MembraneDiffusion1DSolver::setDiffusivity, py::arg("D"),
             py::return_value_policy::reference_internal)
        .def("set_partition_coefficient", &MembraneDiffusion1DSolver::setPartitionCoefficient,
             py::arg("Phi"), py::return_value_policy::reference_internal)
        .def("set_left_concentration", &MembraneDiffusion1DSolver::setLeftConcentration,
             py::arg("C"), py::return_value_policy::reference_internal)
        .def("set_right_concentration", &MembraneDiffusion1DSolver::setRightConcentration,
             py::arg("C"), py::return_value_policy::reference_internal)
        .def("set_hindered_diffusion", &MembraneDiffusion1DSolver::setHinderedDiffusion,
             py::arg("solute_radius"), py::arg("pore_radius"),
             py::return_value_policy::reference_internal)
        .def("disable_hindered_diffusion", &MembraneDiffusion1DSolver::disableHinderedDiffusion,
             py::return_value_policy::reference_internal)
        .def("set_num_nodes", &MembraneDiffusion1DSolver::setNumNodes, py::arg("n"),
             py::return_value_policy::reference_internal)
        .def("solve", &MembraneDiffusion1DSolver::solve)
        .def("compute_flux", &MembraneDiffusion1DSolver::computeFlux)
        .def("compute_permeability", &MembraneDiffusion1DSolver::computePermeability)
        .def("membrane_thickness", &MembraneDiffusion1DSolver::membraneThickness)
        .def("diffusivity", &MembraneDiffusion1DSolver::diffusivity)
        .def("partition_coefficient", &MembraneDiffusion1DSolver::partitionCoefficient)
        .def("left_concentration", &MembraneDiffusion1DSolver::leftConcentration)
        .def("right_concentration", &MembraneDiffusion1DSolver::rightConcentration)
        .def("is_hindered_diffusion", &MembraneDiffusion1DSolver::isHinderedDiffusion)
        .def("lambda_ratio", &MembraneDiffusion1DSolver::lambda);

    py::class_<MultiLayerMembraneSolver>(m, "MultiLayerMembraneSolver")
        .def(py::init<>())
        .def("add_layer", &MultiLayerMembraneSolver::addLayer, py::arg("thickness"),
             py::arg("diffusivity"), py::arg("partition_coefficient") = 1.0,
             py::return_value_policy::reference_internal)
        .def("set_left_concentration", &MultiLayerMembraneSolver::setLeftConcentration,
             py::arg("C"), py::return_value_policy::reference_internal)
        .def("set_right_concentration", &MultiLayerMembraneSolver::setRightConcentration,
             py::arg("C"), py::return_value_policy::reference_internal)
        .def("clear_layers", &MultiLayerMembraneSolver::clearLayers,
             py::return_value_policy::reference_internal)
        .def("solve", &MultiLayerMembraneSolver::solve)
        .def("total_thickness", &MultiLayerMembraneSolver::totalThickness)
        .def("num_layers", &MultiLayerMembraneSolver::numLayers);

    // Renkin hindrance function
    m.def("renkin_hindrance", &renkin_hindrance, py::arg("lambda_ratio"),
          "Compute Renkin hindrance factor H for spherical solute in cylindrical pore.\n"
          "H = (1-λ)² × (1 - 2.104λ + 2.09λ³ - 0.95λ⁵)\n"
          "where λ = solute_radius / pore_radius.");

    // =========================================================================
    // Advection-Diffusion
    // =========================================================================
    py::enum_<AdvectionScheme>(m, "AdvectionScheme")
        .value("UPWIND", AdvectionScheme::UPWIND)
        .value("CENTRAL", AdvectionScheme::CENTRAL)
        .value("HYBRID", AdvectionScheme::HYBRID)
        .value("QUICK", AdvectionScheme::QUICK)
        .export_values();

    py::class_<AdvectionDiffusionSolver>(m, "AdvectionDiffusionSolver")
        .def(py::init<const StructuredMesh&, double, double, double, AdvectionScheme>(),
             py::arg("mesh"), py::arg("diffusivity"), py::arg("vx"), py::arg("vy") = 0.0,
             py::arg("scheme") = AdvectionScheme::HYBRID, py::keep_alive<1, 2>())
        .def(py::init<const StructuredMesh&, double, const std::vector<double>&,
                      const std::vector<double>&, AdvectionScheme>(),
             py::arg("mesh"), py::arg("diffusivity"), py::arg("vx_field"), py::arg("vy_field"),
             py::arg("scheme") = AdvectionScheme::HYBRID, py::keep_alive<1, 2>())
        .def(
            "time", [](const AdvectionDiffusionSolver& s) { return s.time(); },
            "Current simulation time advanced by solve()")
        .def(
            "check_stability",
            [](const AdvectionDiffusionSolver& s, double dt) { return s.checkStability(dt); },
            py::arg("dt"), "Return whether dt satisfies the explicit stability condition")
        .def("mesh", &AdvectionDiffusionSolver::mesh, py::return_value_policy::reference_internal,
             "The mesh this solver was built on")
        .def("solve", &AdvectionDiffusionSolver::solve, py::arg("dt"), py::arg("num_steps"))
        .def("cell_peclet", &AdvectionDiffusionSolver::cellPeclet)
        .def("max_time_step", &AdvectionDiffusionSolver::maxTimeStep, py::arg("safety") = 0.4)
        .def("is_scheme_stable", &AdvectionDiffusionSolver::isSchemeStable)
        .def("scheme", &AdvectionDiffusionSolver::scheme)
        .def("set_scheme", &AdvectionDiffusionSolver::setScheme, py::arg("scheme"))
        .def("solution",
             [](const AdvectionDiffusionSolver& solver) {
                 return to_numpy_with_base(solver.solution(), py::cast(&solver));
             })
        .def("set_initial_condition", &AdvectionDiffusionSolver::setInitialCondition,
             py::arg("values"))
        .def(
            "set_boundary",
            [](AdvectionDiffusionSolver& solver, int boundary_id, const BoundaryCondition& bc) {
                solver.setBoundaryCondition(checkedBoundary(boundary_id), bc);
            },
            py::arg("boundary_id"), py::arg("bc"))
        .def(
            "set_boundary",
            [](AdvectionDiffusionSolver& solver, Boundary boundary, const BoundaryCondition& bc) {
                solver.setBoundaryCondition(boundary, bc);
            },
            py::arg("boundary"), py::arg("bc"));

    // =========================================================================
    // ExplicitFD Facade (Problem + run)
    // =========================================================================
    py::class_<SolverStats>(m, "SolverStats")
        .def(py::init<>())
        .def_readonly("dt", &SolverStats::dt)
        .def_readonly("steps", &SolverStats::steps)
        .def_readonly("t_end", &SolverStats::t_end)
        .def_readonly("u_min_initial", &SolverStats::u_min_initial)
        .def_readonly("u_max_initial", &SolverStats::u_max_initial)
        .def_readonly("u_min_final", &SolverStats::u_min_final)
        .def_readonly("u_max_final", &SolverStats::u_max_final)
        .def_readonly("mass_initial", &SolverStats::mass_initial)
        .def_readonly("mass_final", &SolverStats::mass_final)
        .def_readonly("mass_abs_drift", &SolverStats::mass_abs_drift)
        .def_readonly("mass_rel_drift", &SolverStats::mass_rel_drift)
        .def_readonly("wall_time_s", &SolverStats::wall_time_s);

    py::class_<RunResult>(m, "RunResult")
        .def(py::init<>())
        .def_readonly("stats", &RunResult::stats)
        .def("solution", [](const RunResult& result) {
            return to_numpy_with_base(result.solution, py::cast(&result));
        });

    // =========================================================================
    // TransportProblem - Unified problem specification
    // =========================================================================
    py::class_<TransportProblem>(m, "TransportProblem")
        .def(py::init<const StructuredMesh&>(), py::arg("mesh"), py::keep_alive<1, 2>())
        .def("diffusivity",
             static_cast<TransportProblem& (TransportProblem::*)(double)>(
                 &TransportProblem::diffusivity),
             py::arg("diffusivity"), py::return_value_policy::reference_internal,
             "Set uniform diffusivity")
        .def("diffusivity",
             static_cast<double (TransportProblem::*)() const>(&TransportProblem::diffusivity),
             "Get diffusivity value")
        .def("diffusivity_field",
             static_cast<TransportProblem& (TransportProblem::*)(const std::vector<double>&)>(
                 &TransportProblem::diffusivityField),
             py::arg("D_field"), py::return_value_policy::reference_internal)
        .def("reaction",
             py::overload_cast<TransportProblem::ReactionFunc>(&TransportProblem::reaction),
             py::arg("function"), py::return_value_policy::reference_internal,
             "Replace the configured reaction with R(c, x, y, t). The explicit stability bound "
             "is then unknown.")
        .def("reaction",
             py::overload_cast<TransportProblem::ReactionFunc, double>(&TransportProblem::reaction),
             py::arg("function"), py::arg("max_abs_dc"),
             py::return_value_policy::reference_internal,
             "Replace the reaction and provide an upper bound for |dR/dc| in 1/time.")
        .def("add_reaction",
             py::overload_cast<TransportProblem::ReactionFunc>(&TransportProblem::addReaction),
             py::arg("function"), py::return_value_policy::reference_internal,
             "Add R(c, x, y, t) to the existing reaction. The explicit stability bound is then "
             "unknown.")
        .def("add_reaction",
             py::overload_cast<TransportProblem::ReactionFunc, double>(
                 &TransportProblem::addReaction),
             py::arg("function"), py::arg("max_abs_dc"),
             py::return_value_policy::reference_internal,
             "Add a reaction and its upper bound for |dR/dc| in 1/time.")
        .def("linear_decay", &TransportProblem::linearDecay, py::arg("k"),
             py::return_value_policy::reference_internal,
             "Replace the reaction with first-order decay R=-k*c")
        .def("add_linear_decay", &TransportProblem::addLinearDecay, py::arg("k"),
             py::return_value_policy::reference_internal,
             "Add first-order decay R=-k*c to the existing reaction")
        .def("constant_source", &TransportProblem::constantSource, py::arg("S"),
             py::return_value_policy::reference_internal,
             "Replace the reaction with a constant source R=S")
        .def("add_constant_source", &TransportProblem::addConstantSource, py::arg("S"),
             py::return_value_policy::reference_internal,
             "Add a constant source R=S to the existing reaction")
        .def("michaelis_menten", &TransportProblem::michaelisMenten, py::arg("Vmax"), py::arg("Km"),
             py::return_value_policy::reference_internal,
             "Replace the reaction with Michaelis-Menten consumption")
        .def("add_michaelis_menten", &TransportProblem::addMichaelisMenten, py::arg("Vmax"),
             py::arg("Km"), py::return_value_policy::reference_internal,
             "Add Michaelis-Menten consumption to the existing reaction")
        .def("logistic_growth", &TransportProblem::logisticGrowth, py::arg("r"), py::arg("K"),
             py::return_value_policy::reference_internal,
             "Replace the reaction with logistic growth")
        .def("add_logistic_growth", &TransportProblem::addLogisticGrowth, py::arg("r"),
             py::arg("K"), py::return_value_policy::reference_internal,
             "Add logistic growth to the existing reaction")
        .def("clear_reaction", &TransportProblem::clearReaction,
             py::return_value_policy::reference_internal, "Remove every configured reaction term")
        .def("velocity", &TransportProblem::velocity, py::arg("vx"), py::arg("vy") = 0.0,
             py::return_value_policy::reference_internal)
        .def("velocity_field",
             static_cast<TransportProblem& (TransportProblem::*)(const std::vector<double>&)>(
                 &TransportProblem::velocityField),
             py::arg("vx"), py::return_value_policy::reference_internal,
             "Set a node-centred x velocity field for a 1D mesh")
        .def("velocity_field",
             static_cast<TransportProblem& (TransportProblem::*)(const std::vector<double>&,
                                                                 const std::vector<double>&)>(
                 &TransportProblem::velocityField),
             py::arg("vx"), py::arg("vy"), py::return_value_policy::reference_internal,
             "Set node-centred x and y velocity fields")
        .def("advection_scheme",
             static_cast<TransportProblem& (TransportProblem::*)(AdvectionScheme)>(
                 &TransportProblem::advectionScheme),
             py::arg("scheme"), py::return_value_policy::reference_internal)
        .def(
            "initial_condition",
            [](TransportProblem& self, const std::vector<double>& values) -> TransportProblem& {
                // Explicit copy to avoid any dangling reference issues
                std::vector<double> values_copy(values.begin(), values.end());
                return self.initialCondition(values_copy);
            },
            py::arg("values"), py::return_value_policy::reference_internal)
        .def("initial_condition",
             static_cast<TransportProblem& (TransportProblem::*)(double)>(
                 &TransportProblem::initialCondition),
             py::arg("value"), py::return_value_policy::reference_internal)
        .def("boundary", &TransportProblem::boundary, py::arg("side"), py::arg("bc"),
             py::return_value_policy::reference_internal)
        .def("dirichlet", &TransportProblem::dirichlet, py::arg("side"), py::arg("value"),
             py::return_value_policy::reference_internal)
        .def("neumann", &TransportProblem::neumann, py::arg("side"), py::arg("normal_derivative"),
             py::return_value_policy::reference_internal)
        .def("robin", &TransportProblem::robin, py::arg("side"), py::arg("a"), py::arg("b"),
             py::arg("c"), py::return_value_policy::reference_internal)
        // Accessors
        .def("mesh", &TransportProblem::mesh, py::return_value_policy::reference_internal)
        .def("has_uniform_diffusivity", &TransportProblem::hasUniformDiffusivity,
             "Whether the diffusivity is represented by one uniform value")
        .def("has_advection", &TransportProblem::hasAdvection,
             "Whether a nonzero velocity field is configured")
        .def("has_reaction", &TransportProblem::hasReaction,
             "Whether any reaction term is configured")
        .def("reaction_stability_bound_known", &TransportProblem::reactionStabilityBoundKnown,
             "Whether an explicit |dR/dc| stability bound is available")
        .def("reaction_stability_rate_bound", &TransportProblem::reactionStabilityRateBound,
             "Return the configured |dR/dc| bound in 1/time")
        .def("initial",
             [](const TransportProblem& prob) -> py::array_t<double> {
                 const std::vector<double>& vec = prob.initial();
                 // Create a copy to avoid any reference issues
                 std::vector<double> vec_copy(vec.begin(), vec.end());
                 py::array_t<double> result(vec_copy.size());
                 auto r = result.mutable_unchecked<1>();
                 for (size_t i = 0; i < vec_copy.size(); ++i) {
                     r(i) = vec_copy[i];
                 }
                 return result;
             })
        .def("boundaries", &TransportProblem::boundaries, py::return_value_policy::copy);

    // ExplicitFD facade - now uses unified TransportProblem
    py::class_<ExplicitFD>(m, "ExplicitFD")
        .def(py::init<>())
        .def("safety_factor", &ExplicitFD::safetyFactor, py::arg("factor"),
             py::return_value_policy::reference_internal)
        .def("run", &ExplicitFD::run, py::arg("problem"), py::arg("t_end"));

    // =========================================================================
    // 3D Diffusion Solvers
    // =========================================================================

    py::class_<DiffusionSolver3D>(m, "DiffusionSolver3D",
                                  R"(Conservative explicit 3D solver for ∂u/∂t = D∇²u.

        The enforced Forward Euler diffusion CFL bound is available through
        max_stable_time_step().

        Example:
            >>> mesh = bt.StructuredMesh3D(20, 20, 20, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
            >>> solver = bt.DiffusionSolver3D(mesh, 1e-5)
            >>> solver.set_initial_condition(u0)
            >>> solver.solve(dt, num_steps)
        )")
        .def(py::init<const StructuredMesh3D&, double>(), py::arg("mesh"), py::arg("diffusivity"),
             py::keep_alive<1, 2>())
        .def("set_initial_condition", &DiffusionSolver3D::setInitialCondition, py::arg("values"))
        .def(
            "set_dirichlet_boundary",
            [](DiffusionSolver3D& solver, int boundary_id, double value) {
                solver.setDirichletBoundary(checkedBoundary3D(boundary_id), value);
            },
            py::arg("boundary_id"), py::arg("value"))
        .def("set_dirichlet_boundary",
             py::overload_cast<Boundary3D, double>(&DiffusionSolver3D::setDirichletBoundary),
             py::arg("boundary"), py::arg("value"))
        .def(
            "set_neumann_boundary",
            [](DiffusionSolver3D& solver, int boundary_id, double normal_derivative) {
                solver.setNeumannBoundary(checkedBoundary3D(boundary_id), normal_derivative);
            },
            py::arg("boundary_id"), py::arg("normal_derivative"))
        .def("set_neumann_boundary",
             py::overload_cast<Boundary3D, double>(&DiffusionSolver3D::setNeumannBoundary),
             py::arg("boundary"), py::arg("normal_derivative"))
        .def("solve", &DiffusionSolver3D::solve, py::arg("dt"), py::arg("num_steps"))
        .def("check_stability", &DiffusionSolver3D::checkStability, py::arg("dt"),
             "Check if the given time step satisfies the CFL stability condition.")
        .def("max_stable_time_step", &DiffusionSolver3D::maxStableTimeStep,
             "Get the maximum stable time step for explicit integration.")
        .def("time", &DiffusionSolver3D::time, "Get current simulation time.")
        .def("solution",
             [](const DiffusionSolver3D& solver) {
                 return to_numpy_with_base(solver.solution(), py::cast(&solver));
             })
        .def("mesh", &DiffusionSolver3D::mesh, py::return_value_policy::reference_internal);

    py::class_<LinearReactionDiffusionSolver3D>(
        m, "LinearReactionDiffusionSolver3D",
        R"(3D IMEX reaction-diffusion solver: ∂u/∂t = D∇²u - k*u

        Decay uses Backward Euler, but diffusion uses Forward Euler; the
        explicit diffusion CFL limit still applies. The method is first order in time.

        Example:
            >>> mesh = bt.StructuredMesh3D(20, 1.0)  # 20x20x20 unit cube
            >>> solver = bt.LinearReactionDiffusionSolver3D(mesh, D=1e-5, decay_rate=0.01)
        )")
        .def(py::init<const StructuredMesh3D&, double, double>(), py::arg("mesh"),
             py::arg("diffusivity"), py::arg("decay_rate"), py::keep_alive<1, 2>())
        .def("set_initial_condition", &LinearReactionDiffusionSolver3D::setInitialCondition,
             py::arg("values"))
        .def(
            "set_dirichlet_boundary",
            [](LinearReactionDiffusionSolver3D& solver, int boundary_id, double value) {
                solver.setDirichletBoundary(checkedBoundary3D(boundary_id), value);
            },
            py::arg("boundary_id"), py::arg("value"))
        .def("set_dirichlet_boundary",
             py::overload_cast<Boundary3D, double>(
                 &LinearReactionDiffusionSolver3D::setDirichletBoundary),
             py::arg("boundary"), py::arg("value"))
        .def(
            "set_neumann_boundary",
            [](LinearReactionDiffusionSolver3D& solver, int boundary_id, double normal_derivative) {
                solver.setNeumannBoundary(checkedBoundary3D(boundary_id), normal_derivative);
            },
            py::arg("boundary_id"), py::arg("normal_derivative"))
        .def("set_neumann_boundary",
             py::overload_cast<Boundary3D, double>(
                 &LinearReactionDiffusionSolver3D::setNeumannBoundary),
             py::arg("boundary"), py::arg("normal_derivative"))
        .def("solve", &LinearReactionDiffusionSolver3D::solve, py::arg("dt"), py::arg("num_steps"))
        .def("check_stability", &LinearReactionDiffusionSolver3D::checkStability, py::arg("dt"))
        .def("max_stable_time_step", &LinearReactionDiffusionSolver3D::maxStableTimeStep)
        .def("decay_rate", &LinearReactionDiffusionSolver3D::decayRate)
        .def("time", &LinearReactionDiffusionSolver3D::time)
        .def("solution",
             [](const LinearReactionDiffusionSolver3D& solver) {
                 return to_numpy_with_base(solver.solution(), py::cast(&solver));
             })
        .def("mesh", &LinearReactionDiffusionSolver3D::mesh,
             py::return_value_policy::reference_internal);

    // =========================================================================
    // Multi-Species Reaction-Diffusion Solver
    // =========================================================================
    py::class_<MultiSpeciesSolver>(m, "MultiSpeciesSolver",
                                   R"(Generic N-species reaction-diffusion solver.

        Solves the coupled system:
            ∂u_i/∂t = D_i ∇²u_i + R_i(u_1, ..., u_N, x, y, t)

        for i = 1, ..., N species with individual diffusion coefficients and
        user-defined reaction kinetics.

        Example (Lotka-Volterra predator-prey):
            >>> mesh = bt.StructuredMesh(50, 50, 0.0, 1.0, 0.0, 1.0)
            >>> solver = bt.MultiSpeciesSolver(mesh, [D_prey, D_pred])
            >>> solver.set_reaction_model(bt.LotkaVolterraReaction(alpha, beta, gamma, delta))
            >>> solver.set_initial_condition(0, prey_ic)   # Species 0: prey
            >>> solver.set_initial_condition(1, pred_ic)   # Species 1: predator
            >>> solver.solve(dt, num_steps)
            >>> prey = solver.solution(0)
            >>> pred = solver.solution(1)

        Example (SIR epidemiological model):
            >>> solver = bt.MultiSpeciesSolver(mesh, [D_S, D_I, D_R])
            >>> solver.set_reaction_model(bt.SIRReaction(beta=0.3, gamma=0.1, N=1000))
            >>> solver.set_initial_condition(0, S0)  # Susceptible
            >>> solver.set_initial_condition(1, I0)  # Infected
            >>> solver.set_initial_condition(2, R0)  # Recovered
        )")
        .def(py::init<const StructuredMesh&, const std::vector<double>&, size_t>(), py::arg("mesh"),
             py::arg("diffusivities"), py::arg("num_species") = 0, py::keep_alive<1, 2>(),
             "Create a multi-species solver with specified diffusivities")
        .def(
            "set_reaction_function",
            [](MultiSpeciesSolver& solver, py::function reaction) {
                solver.setReactionFunction([reaction = std::move(reaction)](
                                               std::vector<double>& rates,
                                               const std::vector<double>& concentrations, double x,
                                               double y, double t) {
                    py::gil_scoped_acquire acquire;

                    py::list python_rates(rates.size());
                    py::list python_concentrations(concentrations.size());
                    for (std::size_t i = 0; i < rates.size(); ++i) {
                        python_rates[static_cast<py::ssize_t>(i)] = rates[i];
                    }
                    for (std::size_t i = 0; i < concentrations.size(); ++i) {
                        python_concentrations[static_cast<py::ssize_t>(i)] = concentrations[i];
                    }

                    py::object returned_rates =
                        reaction(python_rates, python_concentrations, x, y, t);
                    if (returned_rates.is_none()) {
                        copyPythonReactionRates(python_rates, rates, "mutated rates list");
                    } else {
                        // A returned sequence is authoritative. This makes callbacks that
                        // return rates unambiguous even if they also happen to mutate the list.
                        copyPythonReactionRates(returned_rates, rates, "returned rate sequence");
                    }
                });
            },
            py::arg("reaction"),
            R"doc(Set a custom reaction function ``f(rates, concentrations, x, y, t)``.

``rates`` is a zero-initialized mutable list with one entry per species;
``concentrations`` is an input list in the same species order. The callback may
either mutate ``rates`` and return ``None``, or return a one-dimensional rate
sequence. A returned sequence is authoritative if both forms are used. Exactly
one finite real rate per species is required. Rates have units of concentration
per unit time.)doc")
        .def(
            "set_reaction_model",
            [](MultiSpeciesSolver& solver, const LotkaVolterraReaction& model) {
                solver.setReactionModel(model);
            },
            py::arg("model"), "Set a Lotka-Volterra reaction model")
        .def(
            "set_reaction_model",
            [](MultiSpeciesSolver& solver, const SIRReaction& model) {
                solver.setReactionModel(model);
            },
            py::arg("model"), "Set an SIR epidemiological model")
        .def(
            "set_reaction_model",
            [](MultiSpeciesSolver& solver, const SEIRReaction& model) {
                solver.setReactionModel(model);
            },
            py::arg("model"), "Set an SEIR epidemiological model")
        .def(
            "set_reaction_model",
            [](MultiSpeciesSolver& solver, const BrusselatorReaction& model) {
                solver.setReactionModel(model);
            },
            py::arg("model"), "Set a Brusselator oscillator model")
        .def(
            "set_reaction_model",
            [](MultiSpeciesSolver& solver, const CompetitiveInhibitionReaction& model) {
                solver.setReactionModel(model);
            },
            py::arg("model"), "Set a competitive inhibition model")
        .def(
            "set_reaction_model",
            [](MultiSpeciesSolver& solver, const EnzymeCascadeReaction& model) {
                solver.setReactionModel(model);
            },
            py::arg("model"), "Set an enzyme cascade model")
        .def("set_initial_condition", &MultiSpeciesSolver::setInitialCondition,
             py::arg("species_idx"), py::arg("values"),
             "Set initial condition for a specific species")
        .def("set_uniform_initial_condition", &MultiSpeciesSolver::setUniformInitialCondition,
             py::arg("species_idx"), py::arg("value"),
             "Set uniform initial condition for a species")
        .def("set_dirichlet_boundary",
             py::overload_cast<size_t, Boundary, double>(&MultiSpeciesSolver::setDirichletBoundary),
             py::arg("species_idx"), py::arg("boundary"), py::arg("value"),
             "Set Dirichlet boundary for a specific species")
        .def(
            "set_dirichlet_boundary",
            [](MultiSpeciesSolver& solver, size_t species_idx, int boundary_id, double value) {
                solver.setDirichletBoundary(species_idx, checkedBoundary(boundary_id), value);
            },
            py::arg("species_idx"), py::arg("boundary_id"), py::arg("value"),
            "Set Dirichlet boundary for a specific species (by ID)")
        .def("set_neumann_boundary",
             py::overload_cast<size_t, Boundary, double>(&MultiSpeciesSolver::setNeumannBoundary),
             py::arg("species_idx"), py::arg("boundary"), py::arg("normal_derivative"),
             "Set the outward-normal concentration derivative for a specific species")
        .def(
            "set_neumann_boundary",
            [](MultiSpeciesSolver& solver, size_t species_idx, int boundary_id,
               double normal_derivative) {
                solver.setNeumannBoundary(species_idx, checkedBoundary(boundary_id),
                                          normal_derivative);
            },
            py::arg("species_idx"), py::arg("boundary_id"), py::arg("normal_derivative"),
            "Set the outward-normal concentration derivative for a species (by ID)")
        .def("set_all_species_dirichlet", &MultiSpeciesSolver::setAllSpeciesDirichlet,
             py::arg("boundary"), py::arg("value"), "Set same Dirichlet boundary for all species")
        .def("set_all_species_neumann", &MultiSpeciesSolver::setAllSpeciesNeumann,
             py::arg("boundary"), py::arg("normal_derivative"),
             "Set the same outward-normal concentration derivative for all species")
        .def("check_stability", &MultiSpeciesSolver::checkStability, py::arg("dt"),
             "Check the exact Forward Euler diffusion CFL bound; reaction stability is separate")
        .def("max_stable_time_step", &MultiSpeciesSolver::maxStableTimeStep,
             "Exact forward-Euler diffusion CFL ceiling; reactions may impose a smaller step")
        .def("solve", &MultiSpeciesSolver::solve, py::arg("dt"), py::arg("num_steps"),
             "Run solver for specified number of time steps")
        .def("solve_until", &MultiSpeciesSolver::solveUntil, py::arg("final_time"),
             py::arg("maximum_dt"),
             "Advance to an exact absolute final time using equal stable substeps no larger "
             "than maximum_dt")
        .def(
            "solution",
            [](const MultiSpeciesSolver& solver, size_t species_idx) {
                return to_numpy_with_base(solver.solution(species_idx), py::cast(&solver));
            },
            py::arg("species_idx"), "Return an owned solution copy for a specific species")
        .def(
            "all_solutions",
            [](const MultiSpeciesSolver& solver) {
                py::list result;
                for (size_t s = 0; s < solver.numSpecies(); ++s) {
                    result.append(to_numpy_with_base(solver.solution(s), py::cast(&solver)));
                }
                return result;
            },
            "Return owned solution copies for all species")
        .def("mesh", &MultiSpeciesSolver::mesh, py::return_value_policy::reference_internal)
        .def("num_species", &MultiSpeciesSolver::numSpecies, "Get number of species")
        .def("diffusivity", &MultiSpeciesSolver::diffusivity, py::arg("species_idx"),
             "Get diffusivity for a species")
        .def("time", &MultiSpeciesSolver::time, "Get current simulation time")
        .def("reset_time", &MultiSpeciesSolver::resetTime, "Reset time to zero")
        .def("total_concentration", &MultiSpeciesSolver::totalConcentration, py::arg("node_idx"),
             "Get total concentration across all species at a node")
        .def("concentration", &MultiSpeciesSolver::concentration, py::arg("species_idx"),
             py::arg("node_idx"), "Get concentration of a species at a node")
        .def("solution_norm", &MultiSpeciesSolver::solutionNorm, py::arg("species_idx"),
             "Compute L2 norm of a species solution")
        .def("total_mass", &MultiSpeciesSolver::totalMass, py::arg("species_idx"),
             "Compute total mass (integral) of a species");

    // =========================================================================
    // Reaction Models (for use with MultiSpeciesSolver)
    // =========================================================================
    py::class_<LotkaVolterraReaction>(m, "LotkaVolterraReaction",
                                      R"(Lotka-Volterra predator-prey with carrying capacity.

        For 2 species (prey u, predator v):
            du/dt = α·u·(1 - u/K) - β·u·v   (logistic prey growth)
            dv/dt = δ·u·v - γ·v             (predator dynamics)

        The carrying capacity K prevents unbounded prey growth.

        Parameters:
            alpha: prey growth rate
            beta: predation rate
            gamma: predator death rate
            delta: predator reproduction rate from prey
            carrying_capacity: maximum prey population (default=100)

        Example:
            >>> model = bt.LotkaVolterraReaction(alpha=1.0, beta=0.1, gamma=1.0, delta=0.1, carrying_capacity=50)
            >>> solver.set_reaction_model(model)
        )")
        .def(py::init<double, double, double, double, double>(), py::arg("alpha"), py::arg("beta"),
             py::arg("gamma"), py::arg("delta"), py::arg("carrying_capacity") = 100.0)
        .def_property_readonly("alpha", &LotkaVolterraReaction::alpha)
        .def_property_readonly("beta", &LotkaVolterraReaction::beta)
        .def_property_readonly("gamma", &LotkaVolterraReaction::gamma)
        .def_property_readonly("delta", &LotkaVolterraReaction::delta)
        .def_property_readonly("carrying_capacity", &LotkaVolterraReaction::carrying_capacity);

    py::class_<SIRReaction>(m, "SIRReaction",
                            R"(SIR epidemiological model.

        For 3 species (S, I, R):
            dS/dt = -β·S·I / N         (susceptible become infected)
            dI/dt = β·S·I / N - γ·I    (infected from S, recover)
            dR/dt = γ·I                (recovered from infected)

        Parameters:
            beta: transmission rate
            gamma: recovery rate
            total_population: N, for normalization

        N is a reference population in the same units as local S, I, and R.
        For spatial density fields it is a local reference density, not the
        domain-integrated population. The approximation R₀ = β/γ assumes S≈N.

        Example:
            >>> model = bt.SIRReaction(beta=0.3, gamma=0.1, total_population=1000)
            >>> print(f"R0 = {model.R0}")  # 3.0
        )")
        .def(py::init<double, double, double>(), py::arg("beta"), py::arg("gamma"),
             py::arg("total_population"))
        .def_property_readonly("beta", &SIRReaction::beta)
        .def_property_readonly("gamma", &SIRReaction::gamma)
        .def_property_readonly("N", &SIRReaction::N)
        .def_property_readonly("R0", &SIRReaction::R0, "Basic reproduction number");

    py::class_<SEIRReaction>(m, "SEIRReaction",
                             R"(SEIR epidemiological model with exposed (latent) period.

        For 4 species (S, E, I, R):
            dS/dt = -β·S·I / N
            dE/dt = β·S·I / N - σ·E    (exposed become infectious after incubation)
            dI/dt = σ·E - γ·I
            dR/dt = γ·I

        Parameters:
            beta: transmission rate
            sigma: rate of becoming infectious (1/incubation period)
            gamma: recovery rate
            total_population: reference N in the same local units as S/E/I/R;
                              for density fields this is not a domain integral

        Example:
            >>> model = bt.SEIRReaction(beta=0.5, sigma=0.2, gamma=0.1, total_population=1e6)
        )")
        .def(py::init<double, double, double, double>(), py::arg("beta"), py::arg("sigma"),
             py::arg("gamma"), py::arg("total_population"))
        .def_property_readonly("beta", &SEIRReaction::beta)
        .def_property_readonly("sigma", &SEIRReaction::sigma)
        .def_property_readonly("gamma", &SEIRReaction::gamma)
        .def_property_readonly("N", &SEIRReaction::N);

    py::class_<BrusselatorReaction>(m, "BrusselatorReaction",
                                    R"(Conventional nondimensional Brusselator model.

        Classic 2-species autocatalytic system exhibiting limit cycle oscillations:
            dX/dt = A - (B+1)·X + X²·Y
            dY/dt = B·X - X²·Y

        For B > 1 + A², the system exhibits sustained oscillations.

        Example:
            >>> model = bt.BrusselatorReaction(A=1.0, B=3.0)
            >>> print(f"Oscillatory: {model.is_oscillatory}")  # True
        )")
        .def(py::init<double, double>(), py::arg("A"), py::arg("B"))
        .def_property_readonly("A", &BrusselatorReaction::A)
        .def_property_readonly("B", &BrusselatorReaction::B)
        .def_property_readonly("is_oscillatory", &BrusselatorReaction::isOscillatory,
                               "Check if parameters lead to oscillatory behavior");

    py::class_<CompetitiveInhibitionReaction>(m, "CompetitiveInhibitionReaction",
                                              R"(Competitive enzyme inhibition model.

        Models substrate (S) competing with inhibitor (I) for enzyme:
            dS/dt = -Vmax · S / (Km · (1 + I/Ki) + S)
            dI/dt = -k_decay · I  (optional inhibitor decay)
            dP/dt = Vmax · S / (Km · (1 + I/Ki) + S)

        For 3 species: [Substrate, Inhibitor, Product]

        Example:
            >>> model = bt.CompetitiveInhibitionReaction(vmax=10.0, km=5.0, ki=2.0)
        )")
        .def(py::init<double, double, double, double>(), py::arg("vmax"), py::arg("km"),
             py::arg("ki"), py::arg("inhibitor_decay") = 0.0)
        .def_property_readonly("vmax", &CompetitiveInhibitionReaction::vmax)
        .def_property_readonly("km", &CompetitiveInhibitionReaction::km)
        .def_property_readonly("ki", &CompetitiveInhibitionReaction::ki);

    py::class_<EnzymeCascadeReaction>(m, "EnzymeCascadeReaction",
                                      R"(Linear enzyme cascade reaction kinetics.

        Models a cascade of enzyme activations with Michaelis-Menten kinetics:
            E₀ → E₁ → E₂ → ... → Eₙ

        Each enzyme is activated by the previous one:
            dE_i/dt = (Vmax,i · E_{i-1}) / (Km,i + E_{i-1}) - k_deg,i · E_i

        Parameters:
            vmax_values: maximum reaction rates (N-1 values)
            km_values: Michaelis constants (N-1 values)
            kdeg_values: degradation rates (N values)

        Example (3-enzyme cascade):
            >>> model = bt.EnzymeCascadeReaction(
            ...     vmax_values=[10.0, 20.0],
            ...     km_values=[1.0, 2.0],
            ...     kdeg_values=[0.1, 0.05, 0.02]
            ... )
        )")
        .def(py::init<const std::vector<double>&, const std::vector<double>&,
                      const std::vector<double>&>(),
             py::arg("vmax_values"), py::arg("km_values"), py::arg("kdeg_values"))
        .def_property_readonly("num_enzymes", &EnzymeCascadeReaction::numEnzymes);

    // =========================================================================
    // Nernst-Planck Electrochemical Transport
    // =========================================================================

    // Physical constants submodule
    auto constants_mod = m.def_submodule("constants", "Physical constants for electrochemistry");
    constants_mod.attr("FARADAY") = constants::FARADAY;
    constants_mod.attr("GAS_CONSTANT") = constants::GAS_CONSTANT;
    constants_mod.attr("BOLTZMANN") = constants::BOLTZMANN;
    constants_mod.attr("ELEMENTARY_CHARGE") = constants::ELEMENTARY_CHARGE;
    constants_mod.attr("VACUUM_PERMITTIVITY") = constants::VACUUM_PERMITTIVITY;

    // Ion species
    py::class_<IonSpecies>(m, "IonSpecies",
                           R"(Represents an ion species with transport properties.

        Automatically computes electrical mobility from diffusion coefficient
        using the Einstein relation: μ = |z|FD/(RT)

        Example:
            >>> Na = bt.IonSpecies("Na+", valence=1, diffusivity=1.33e-9)
            >>> K = bt.IonSpecies("K+", 1, 1.96e-9)
            >>> Cl = bt.IonSpecies("Cl-", -1, 2.03e-9)
        )")
        .def(py::init<const std::string&, int, double, double>(), py::arg("name"),
             py::arg("valence"), py::arg("diffusivity"), py::arg("temperature") = 310.0)
        .def_readonly("name", &IonSpecies::name, "Species name (e.g., 'Na+', 'K+', 'Cl-')")
        .def_readonly("valence", &IonSpecies::valence, "Ion charge number (z)")
        .def_readonly("diffusivity", &IonSpecies::diffusivity, "Diffusion coefficient [m²/s]")
        .def_readonly("mobility", &IonSpecies::mobility,
                      "Electrical mobility magnitude at mobility_temperature [m²/(V·s)]")
        .def_readonly("mobility_temperature", &IonSpecies::mobility_temperature,
                      "Absolute temperature used to compute mobility [K]")
        .def("mobility_at", &IonSpecies::mobilityAt, py::arg("temperature"),
             "Evaluate electrical mobility magnitude at an absolute temperature [K]")
        .def_static("thermal_voltage", &IonSpecies::thermalVoltage, py::arg("temperature") = 310.0,
                    "Get thermal voltage V_T = RT/F at given temperature");

    // Common ions submodule
    auto ions_mod = m.def_submodule(
        "ions",
        "Representative aqueous infinite-dilution ion species. Quantitative studies "
        "should supply coefficients measured or corrected at the model temperature.");
    ions_mod.def("sodium", &ions::sodium, "Representative Na+ parameters.");
    ions_mod.def("potassium", &ions::potassium, "Representative K+ parameters.");
    ions_mod.def("chloride", &ions::chloride, "Representative Cl- parameters.");
    ions_mod.def("calcium", &ions::calcium, "Representative Ca2+ parameters.");
    ions_mod.def("magnesium", &ions::magnesium, "Representative Mg2+ parameters.");
    ions_mod.def("hydrogen", &ions::hydrogen, "Representative H+ parameters.");
    ions_mod.def("hydroxide", &ions::hydroxide, "Representative OH- parameters.");
    ions_mod.def("bicarbonate", &ions::bicarbonate, "Representative HCO3- parameters.");

    // Single-ion Nernst-Planck solver
    py::class_<NernstPlanckSolver>(m, "NernstPlanckSolver",
                                   R"(Solver for single-ion Nernst-Planck transport.

        Solves the equation:
            ∂c/∂t = D∇²c + (zFD/RT) ∇·(c ∇φ)

        where:
            c = ion concentration [mol/m³]
            D = diffusion coefficient [m²/s]
            z = ion valence
            F = Faraday constant (96485 C/mol)
            R = gas constant (8.314 J/(mol·K))
            T = temperature [K]
            φ = electric potential [V]

        The electric potential is prescribed. This class does not solve
        Poisson's equation, membrane gating, or neural action-potential models.

        Example (ion transport in uniform electric field):
            >>> mesh = bt.StructuredMesh(100, 0.0, 1e-3)  # 1mm domain
            >>> Na = bt.ions.sodium()
            >>> solver = bt.NernstPlanckSolver(mesh, Na, temperature=310.0)
            >>> solver.set_initial_condition(c0)
            >>> solver.set_uniform_field(Ex=1000.0)  # 1 kV/m field
            >>> solver.set_dirichlet_boundary(bt.Boundary.Left, 100.0)  # 100 mM
            >>> solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)
            >>> solver.solve(dt, num_steps)
        )")
        .def(py::init<const StructuredMesh&, const IonSpecies&, double>(), py::arg("mesh"),
             py::arg("ion"), py::arg("temperature") = 310.0, py::keep_alive<1, 2>(),
             "Create solver for single ion species")
        .def("set_initial_condition", &NernstPlanckSolver::setInitialCondition, py::arg("values"),
             "Set initial concentration field")
        .def("set_potential_field",
             py::overload_cast<const std::vector<double>&>(&NernstPlanckSolver::setPotentialField),
             py::arg("phi"), "Set electric potential field (static array)")
        .def("set_uniform_field", &NernstPlanckSolver::setUniformField, py::arg("Ex"),
             py::arg("Ey") = 0.0, "Set uniform electric field [V/m]")
        .def("set_dirichlet_boundary",
             py::overload_cast<Boundary, double>(&NernstPlanckSolver::setDirichletBoundary),
             py::arg("boundary"), py::arg("value"), "Set fixed concentration boundary")
        .def(
            "set_dirichlet_boundary",
            [](NernstPlanckSolver& solver, int boundary_id, double value) {
                solver.setDirichletBoundary(checkedBoundary(boundary_id), value);
            },
            py::arg("boundary_id"), py::arg("value"))
        .def("set_outward_flux_boundary", &NernstPlanckSolver::setOutwardFluxBoundary,
             py::arg("boundary"), py::arg("outward_molar_flux"),
             "Prescribe the outward total molar flux N.n [mol/(m^2 s)], positive when ions "
             "leave the domain. This is a physical flux, not a concentration derivative.")
        .def(
            "set_outward_flux_boundary",
            [](NernstPlanckSolver& solver, int boundary_id, double outward_molar_flux) {
                solver.setOutwardFluxBoundary(checkedBoundary(boundary_id), outward_molar_flux);
            },
            py::arg("boundary_id"), py::arg("outward_molar_flux"),
            "Prescribe the outward total molar flux on a boundary given by integer id")
        .def(
            "set_neumann_boundary",
            [](NernstPlanckSolver& solver, Boundary boundary, double flux) {
                warn_deprecated("NernstPlanckSolver.set_neumann_boundary",
                                "set_outward_flux_boundary(boundary, outward_molar_flux)",
                                "the value is a physical molar flux, positive leaving the domain, "
                                "not the outward-normal derivative that set_neumann_boundary "
                                "means on every scalar diffusion solver");
                solver.setOutwardFluxBoundary(boundary, flux);
            },
            py::arg("boundary"), py::arg("flux"),
            "Deprecated spelling of set_outward_flux_boundary(); installs the same condition")
        .def(
            "set_neumann_boundary",
            [](NernstPlanckSolver& solver, int boundary_id, double flux) {
                warn_deprecated("NernstPlanckSolver.set_neumann_boundary",
                                "set_outward_flux_boundary(boundary, outward_molar_flux)",
                                "the value is a physical molar flux, positive leaving the domain, "
                                "not the outward-normal derivative that set_neumann_boundary "
                                "means on every scalar diffusion solver");
                solver.setOutwardFluxBoundary(checkedBoundary(boundary_id), flux);
            },
            py::arg("boundary_id"), py::arg("flux"),
            "Deprecated spelling of set_outward_flux_boundary(); installs the same condition")
        .def("check_stability", &NernstPlanckSolver::checkStability, py::arg("dt"),
             "Check the positivity bound of the fitted diffusion-drift operator")
        .def("maximum_stable_time_step", &NernstPlanckSolver::maximumStableTimeStep,
             "Largest explicit step allowed by the fitted homogeneous operator")
        .def("recommended_time_step", &NernstPlanckSolver::recommendedTimeStep,
             py::arg("safety") = 0.9, "Return safety times the fitted-operator stability bound")
        .def("solve", &NernstPlanckSolver::solve, py::arg("dt"), py::arg("num_steps"),
             "Run simulation for specified time steps")
        .def(
            "solution",
            [](const NernstPlanckSolver& solver) { return to_numpy(solver.solution()); },
            "Return an owned copy of the current concentration field")
        .def(
            "potential",
            [](const NernstPlanckSolver& solver) { return to_numpy(solver.potential()); },
            "Return an owned copy of the current electric potential field")
        .def(
            "compute_current_density",
            [](const NernstPlanckSolver& solver) {
                return to_numpy(solver.computeCurrentDensity());
            },
            "Return interleaved Cartesian ionic current-density components [A/m²]")
        .def("time", &NernstPlanckSolver::time, "Get current simulation time")
        .def("ion", &NernstPlanckSolver::ion, py::return_value_policy::reference_internal,
             "Get ion species parameters")
        .def("thermal_voltage", &NernstPlanckSolver::thermalVoltage,
             "Get thermal voltage V_T = RT/F")
        .def("electrical_mobility", &NernstPlanckSolver::electricalMobility,
             "Mobility magnitude evaluated at this solver's temperature [m²/(V·s)]")
        .def("mesh", &NernstPlanckSolver::mesh, py::return_value_policy::reference_internal);

    // Multi-ion solver
    py::class_<MultiIonSolver>(m, "MultiIonSolver",
                               R"(Solver for multiple ion species in one prescribed potential.

        Each species advances independently against the same prescribed
        potential. This class does not solve Poisson's equation and does not
        enforce electroneutrality.

        Example (Na+, K+, Cl- transport):
            >>> mesh = bt.StructuredMesh(100, 0.0, 1e-3)
            >>> ions = [bt.ions.sodium(), bt.ions.potassium(), bt.ions.chloride()]
            >>> solver = bt.MultiIonSolver(mesh, ions)
            >>> solver.set_initial_condition(0, Na_ic)  # Na+
            >>> solver.set_initial_condition(1, K_ic)   # K+
            >>> solver.set_initial_condition(2, Cl_ic)  # Cl-
            >>> solver.set_uniform_field(Ex=500.0)
            >>> solver.solve(dt, num_steps)
        )")
        .def(py::init<const StructuredMesh&, std::vector<IonSpecies>, double>(), py::arg("mesh"),
             py::arg("ions"), py::arg("temperature") = 310.0, py::keep_alive<1, 2>(),
             "Create multi-ion solver")
        .def("set_initial_condition", &MultiIonSolver::setInitialCondition, py::arg("species"),
             py::arg("values"), "Set initial concentration for a species")
        .def("set_dirichlet_boundary",
             py::overload_cast<size_t, Boundary, double>(&MultiIonSolver::setDirichletBoundary),
             py::arg("species"), py::arg("boundary"), py::arg("value"),
             "Set Dirichlet boundary for a species")
        .def(
            "set_dirichlet_boundary",
            [](MultiIonSolver& solver, size_t species, int boundary_id, double value) {
                solver.setDirichletBoundary(species, checkedBoundary(boundary_id), value);
            },
            py::arg("species"), py::arg("boundary_id"), py::arg("value"))
        .def("set_outward_flux_boundary", &MultiIonSolver::setOutwardFluxBoundary,
             py::arg("species"), py::arg("boundary"), py::arg("outward_molar_flux"),
             "Prescribe the outward total molar flux of one species [mol/(m^2 s)], positive "
             "when ions leave the domain. This is a physical flux, not a derivative.")
        .def(
            "set_neumann_boundary",
            [](MultiIonSolver& solver, size_t species, Boundary boundary, double flux) {
                warn_deprecated("MultiIonSolver.set_neumann_boundary",
                                "set_outward_flux_boundary(species, boundary, outward_molar_flux)",
                                "the value is a physical molar flux, positive leaving the domain, "
                                "not the outward-normal derivative that set_neumann_boundary "
                                "means on every scalar diffusion solver");
                solver.setOutwardFluxBoundary(species, boundary, flux);
            },
            py::arg("species"), py::arg("boundary"), py::arg("flux"),
            "Deprecated spelling of set_outward_flux_boundary(); installs the same condition")
        .def("set_potential_field", &MultiIonSolver::setPotentialField, py::arg("phi"),
             "Set electric potential field")
        .def("set_uniform_field", &MultiIonSolver::setUniformField, py::arg("Ex"),
             py::arg("Ey") = 0.0, "Set uniform electric field")
        .def("set_electroneutrality_mode", &MultiIonSolver::setElectroneutralityMode,
             py::arg("enable"), py::arg("background_charge") = 0.0,
             "Compatibility method: enable=True is rejected because electroneutral coupling "
             "is not implemented")
        .def("check_stability", &MultiIonSolver::checkStability, py::arg("dt"),
             "Check the positivity bound for every species")
        .def("maximum_stable_time_step", &MultiIonSolver::maximumStableTimeStep,
             "Largest fitted-operator explicit step over all species")
        .def("recommended_time_step", &MultiIonSolver::recommendedTimeStep, py::arg("safety") = 0.9,
             "Return safety times the multi-species stability bound")
        .def("solve", &MultiIonSolver::solve, py::arg("dt"), py::arg("num_steps"), "Run simulation")
        .def(
            "concentration",
            [](const MultiIonSolver& solver, size_t species) {
                return to_numpy(solver.concentration(species));
            },
            py::arg("species"), "Return an owned concentration copy for a species")
        .def(
            "potential", [](const MultiIonSolver& solver) { return to_numpy(solver.potential()); },
            "Return an owned copy of the electric potential field")
        .def(
            "charge_density",
            [](const MultiIonSolver& solver) { return to_numpy(solver.chargeDensity()); },
            "Compute total charge density [C/m³]")
        .def("time", &MultiIonSolver::time, "Get current simulation time")
        .def("num_species", &MultiIonSolver::numSpecies, "Get number of ion species")
        .def("ion", &MultiIonSolver::ion, py::arg("index"),
             py::return_value_policy::reference_internal, "Get ion species by index")
        .def("electrical_mobility", &MultiIonSolver::electricalMobility, py::arg("species"),
             "Mobility magnitude for a species at this solver's temperature [m²/(V·s)]")
        .def("mesh", &MultiIonSolver::mesh, py::return_value_policy::reference_internal);

    // GHK utilities submodule
    auto ghk_mod = m.def_submodule("ghk", "Goldman-Hodgkin-Katz utilities");
    ghk_mod.def("nernst_potential", &ghk::nernstPotential, py::arg("z"), py::arg("c_in"),
                py::arg("c_out"), py::arg("temperature") = 310.0,
                R"(Compute Nernst equilibrium potential for an ion.

        E = (RT/zF) * ln(c_out / c_in)

        Args:
            z: Ion valence
            c_in: Intracellular concentration [mol/m³]
            c_out: Extracellular concentration [mol/m³]
            temperature: Temperature [K] (default 310K = body temp)

        Returns:
            Equilibrium potential [V]

        Example:
            >>> E_K = bt.ghk.nernst_potential(z=1, c_in=140e-3, c_out=5e-3)
            >>> print(f"E_K = {E_K*1000:.1f} mV")  # ~-90 mV
        )");

    ghk_mod.def("ghk_voltage", &ghk::ghkVoltage, py::arg("P_K"), py::arg("K_in"), py::arg("K_out"),
                py::arg("P_Na"), py::arg("Na_in"), py::arg("Na_out"), py::arg("P_Cl"),
                py::arg("Cl_in"), py::arg("Cl_out"), py::arg("temperature") = 310.0,
                R"(Goldman-Hodgkin-Katz voltage equation for membrane potential.

        V_m = (RT/F) * ln((P_K[K]_o + P_Na[Na]_o + P_Cl[Cl]_i) /
                          (P_K[K]_i + P_Na[Na]_i + P_Cl[Cl]_o))

        Args:
            P_K, P_Na, P_Cl: Relative permeabilities
            K_in, K_out: Potassium concentrations [mol/m³]
            Na_in, Na_out: Sodium concentrations [mol/m³]
            Cl_in, Cl_out: Chloride concentrations [mol/m³]
            temperature: Temperature [K]

        Returns:
            Membrane potential [V]

        Example (resting potential):
            >>> V_m = bt.ghk.ghk_voltage(
            ...     P_K=1.0, K_in=140e-3, K_out=5e-3,
            ...     P_Na=0.04, Na_in=14e-3, Na_out=140e-3,
            ...     P_Cl=0.45, Cl_in=4e-3, Cl_out=120e-3
            ... )
            >>> print(f"Resting potential: {V_m*1000:.1f} mV")  # ~-70 mV
        )");
}

}  // namespace bindings
}  // namespace biotransport
