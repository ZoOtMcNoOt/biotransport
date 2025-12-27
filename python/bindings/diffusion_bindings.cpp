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
#include <biotransport/solvers/advection_diffusion_solver.hpp>
#include <biotransport/solvers/crank_nicolson.hpp>
#include <biotransport/solvers/diffusion_solver_3d.hpp>
#include <biotransport/solvers/diffusion_solvers.hpp>
#include <biotransport/solvers/explicit_fd.hpp>

namespace biotransport {
namespace bindings {

void register_diffusion_bindings(py::module_& m) {
    // =========================================================================
    // DiffusionSolver (base class)
    // =========================================================================
    py::class_<DiffusionSolver>(m, "DiffusionSolver")
        .def(py::init<const StructuredMesh&, double>(), py::arg("mesh"), py::arg("diffusivity"))
        .def("set_initial_condition", &DiffusionSolver::setInitialCondition, py::arg("values"))
        .def("set_dirichlet_boundary",
             py::overload_cast<int, double>(&DiffusionSolver::setDirichletBoundary),
             py::arg("boundary_id"), py::arg("value"))
        .def("set_dirichlet_boundary",
             py::overload_cast<Boundary, double>(&DiffusionSolver::setDirichletBoundary),
             py::arg("boundary"), py::arg("value"))
        .def("set_neumann_boundary",
             py::overload_cast<int, double>(&DiffusionSolver::setNeumannBoundary),
             py::arg("boundary_id"), py::arg("flux"))
        .def("set_neumann_boundary",
             py::overload_cast<Boundary, double>(&DiffusionSolver::setNeumannBoundary),
             py::arg("boundary"), py::arg("flux"))
        .def("set_boundary_condition",
             py::overload_cast<int, const BoundaryCondition&>(
                 &DiffusionSolver::setBoundaryCondition),
             py::arg("boundary_id"), py::arg("bc"))
        .def("set_boundary_condition",
             py::overload_cast<Boundary, const BoundaryCondition&>(
                 &DiffusionSolver::setBoundaryCondition),
             py::arg("boundary"), py::arg("bc"))
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

        Second-order accurate in time, unconditionally stable.
        Allows much larger time steps than explicit methods.

        Example:
            >>> mesh = bt.StructuredMesh(100, 0.0, 1.0)
            >>> solver = bt.CrankNicolsonDiffusion(mesh, 1e-5)
            >>> solver.set_initial_condition(u0)
            >>> solver.set_dirichlet_boundary(bt.Boundary.Left, 1.0)
            >>> solver.solve(dt=0.1, num_steps=100)  # dt >> explicit CFL limit
        )")
        .def(py::init<const StructuredMesh&, double>(), py::arg("mesh"), py::arg("diffusivity"),
             "Create a Crank-Nicolson diffusion solver")
        .def("set_initial_condition", &CrankNicolsonDiffusion::setInitialCondition,
             py::arg("values"), "Set the initial condition")
        .def("set_dirichlet_boundary", &CrankNicolsonDiffusion::setDirichletBoundary,
             py::arg("boundary"), py::arg("value"), "Set a Dirichlet boundary condition")
        .def("set_neumann_boundary", &CrankNicolsonDiffusion::setNeumannBoundary,
             py::arg("boundary"), py::arg("flux"), "Set a Neumann boundary condition")
        .def("set_tolerance", &CrankNicolsonDiffusion::setTolerance, py::arg("tol"),
             "Set convergence tolerance for implicit solve")
        .def("set_max_iterations", &CrankNicolsonDiffusion::setMaxIterations, py::arg("max_iter"),
             "Set maximum iterations for implicit solve")
        .def("step", &CrankNicolsonDiffusion::step, py::arg("dt"),
             "Advance solution by one time step, returns CNSolveResult")
        .def("solve", &CrankNicolsonDiffusion::solve, py::arg("dt"), py::arg("num_steps"),
             "Run solver for specified number of steps")
        .def(
            "solution",
            [](const CrankNicolsonDiffusion& solver) {
                return to_numpy_with_base(solver.solution(), py::cast(&solver));
            },
            "Get the current solution as numpy array")
        .def("time", &CrankNicolsonDiffusion::time, "Get current simulation time")
        .def_property_readonly("diffusivity", &CrankNicolsonDiffusion::diffusivity,
                               "Diffusion coefficient");

    // =========================================================================
    // Derived Diffusion Solvers
    // =========================================================================

    // ReactionDiffusionSolver (custom reaction function)
    py::class_<ReactionDiffusionSolver>(m, "ReactionDiffusionSolver")
        .def(py::init<const StructuredMesh&, double, ReactionDiffusionSolver::ReactionFunction>(),
             py::arg("mesh"), py::arg("diffusivity"), py::arg("reaction"))
        .def("solve", &ReactionDiffusionSolver::solve, py::arg("dt"), py::arg("num_steps"))
        .def("solution",
             [](const ReactionDiffusionSolver& solver) {
                 return to_numpy_with_base(solver.solution(), py::cast(&solver));
             })
        .def("set_initial_condition", &ReactionDiffusionSolver::setInitialCondition,
             py::arg("values"))
        .def("set_boundary",
             py::overload_cast<int, const BoundaryCondition&>(
                 &ReactionDiffusionSolver::setBoundaryCondition),
             py::arg("side"), py::arg("bc"));

    // Linear reaction-diffusion (first-order decay)
    py::class_<LinearReactionDiffusionSolver>(m, "LinearReactionDiffusionSolver")
        .def(py::init<const StructuredMesh&, double, double>(), py::arg("mesh"),
             py::arg("diffusivity"), py::arg("decay_rate"))
        .def("solve", &LinearReactionDiffusionSolver::solve, py::arg("dt"), py::arg("num_steps"))
        .def("solution",
             [](const LinearReactionDiffusionSolver& solver) {
                 return to_numpy_with_base(solver.solution(), py::cast(&solver));
             })
        .def("set_initial_condition", &LinearReactionDiffusionSolver::setInitialCondition,
             py::arg("values"))
        .def("set_boundary",
             py::overload_cast<int, const BoundaryCondition&>(
                 &LinearReactionDiffusionSolver::setBoundaryCondition),
             py::arg("side"), py::arg("bc"));

    // Logistic reaction-diffusion
    py::class_<LogisticReactionDiffusionSolver>(m, "LogisticReactionDiffusionSolver")
        .def(py::init<const StructuredMesh&, double, double, double>(), py::arg("mesh"),
             py::arg("diffusivity"), py::arg("growth_rate"), py::arg("carrying_capacity"))
        .def("solve", &LogisticReactionDiffusionSolver::solve, py::arg("dt"), py::arg("num_steps"))
        .def("solution",
             [](const LogisticReactionDiffusionSolver& solver) {
                 return to_numpy_with_base(solver.solution(), py::cast(&solver));
             })
        .def("set_initial_condition", &LogisticReactionDiffusionSolver::setInitialCondition,
             py::arg("values"))
        .def("set_boundary",
             py::overload_cast<int, const BoundaryCondition&>(
                 &LogisticReactionDiffusionSolver::setBoundaryCondition),
             py::arg("side"), py::arg("bc"));

    // Michaelis-Menten reaction-diffusion
    py::class_<MichaelisMentenReactionDiffusionSolver>(m, "MichaelisMentenReactionDiffusionSolver")
        .def(py::init<const StructuredMesh&, double, double, double>(), py::arg("mesh"),
             py::arg("diffusivity"), py::arg("vmax"), py::arg("km"))
        .def("solve", &MichaelisMentenReactionDiffusionSolver::solve, py::arg("dt"),
             py::arg("num_steps"))
        .def("solution",
             [](const MichaelisMentenReactionDiffusionSolver& solver) {
                 return to_numpy_with_base(solver.solution(), py::cast(&solver));
             })
        .def("set_initial_condition", &MichaelisMentenReactionDiffusionSolver::setInitialCondition,
             py::arg("values"))
        .def("set_boundary",
             py::overload_cast<int, const BoundaryCondition&>(
                 &MichaelisMentenReactionDiffusionSolver::setBoundaryCondition),
             py::arg("side"), py::arg("bc"));

    // Masked Michaelis-Menten
    py::class_<MaskedMichaelisMentenReactionDiffusionSolver>(
        m, "MaskedMichaelisMentenReactionDiffusionSolver")
        .def(py::init<const StructuredMesh&, double, double, double, std::vector<std::uint8_t>,
                      double>(),
             py::arg("mesh"), py::arg("diffusivity"), py::arg("vmax"), py::arg("km"),
             py::arg("mask"), py::arg("pinned_value"))
        .def("solve", &MaskedMichaelisMentenReactionDiffusionSolver::solve, py::arg("dt"),
             py::arg("num_steps"))
        .def("solution",
             [](const MaskedMichaelisMentenReactionDiffusionSolver& solver) {
                 return to_numpy_with_base(solver.solution(), py::cast(&solver));
             })
        .def("set_initial_condition",
             &MaskedMichaelisMentenReactionDiffusionSolver::setInitialCondition, py::arg("values"))
        .def("set_boundary",
             py::overload_cast<int, const BoundaryCondition&>(
                 &MaskedMichaelisMentenReactionDiffusionSolver::setBoundaryCondition),
             py::arg("side"), py::arg("bc"));

    // Constant source reaction-diffusion
    py::class_<ConstantSourceReactionDiffusionSolver>(m, "ConstantSourceReactionDiffusionSolver")
        .def(py::init<const StructuredMesh&, double, double>(), py::arg("mesh"),
             py::arg("diffusivity"), py::arg("source_rate"))
        .def("solve", &ConstantSourceReactionDiffusionSolver::solve, py::arg("dt"),
             py::arg("num_steps"))
        .def("solution",
             [](const ConstantSourceReactionDiffusionSolver& solver) {
                 return to_numpy_with_base(solver.solution(), py::cast(&solver));
             })
        .def("set_initial_condition", &ConstantSourceReactionDiffusionSolver::setInitialCondition,
             py::arg("values"))
        .def("set_boundary",
             py::overload_cast<int, const BoundaryCondition&>(
                 &ConstantSourceReactionDiffusionSolver::setBoundaryCondition),
             py::arg("side"), py::arg("bc"));

    // =========================================================================
    // Gray-Scott (two-species pattern formation)
    // =========================================================================
    py::class_<GrayScottRunResult>(m, "GrayScottRunResult")
        .def(py::init<>())
        .def_readonly("nx", &GrayScottRunResult::nx)
        .def_readonly("ny", &GrayScottRunResult::ny)
        .def_readonly("frames", &GrayScottRunResult::frames)
        .def_readonly("steps_run", &GrayScottRunResult::steps_run)
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
             py::arg("Du"), py::arg("Dv"), py::arg("f"), py::arg("k"))
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
             py::arg("p_boundary"), py::arg("p_tumor"))
        .def("solve_pressure_sor", &TumorDrugDeliverySolver::solvePressureSOR,
             py::arg("max_iter") = 20000, py::arg("tol") = 1e-10, py::arg("omega") = 1.8)
        .def("simulate", &TumorDrugDeliverySolver::simulate, py::arg("pressure"),
             py::arg("diffusivity"), py::arg("permeability"), py::arg("vessel_density"),
             py::arg("k_binding"), py::arg("k_uptake"), py::arg("c_plasma"), py::arg("dt"),
             py::arg("num_steps"), py::arg("times_to_save_s"));

    // =========================================================================
    // Bioheat cryotherapy (temperature + damage)
    // =========================================================================
    py::class_<BioheatSaved>(m, "BioheatSaved")
        .def(py::init<>())
        .def_readonly("nx", &BioheatSaved::nx)
        .def_readonly("ny", &BioheatSaved::ny)
        .def_readonly("frames", &BioheatSaved::frames)
        .def_readonly("times_s", &BioheatSaved::times_s)
        .def("temperature_K",
             [](const BioheatSaved& r) {
                 return to_numpy_3d(r.temperature_K, static_cast<py::ssize_t>(r.frames),
                                    static_cast<py::ssize_t>(r.ny), static_cast<py::ssize_t>(r.nx),
                                    py::cast(&r));
             })
        .def("damage", [](const BioheatSaved& r) {
            return to_numpy_3d(r.damage, static_cast<py::ssize_t>(r.frames),
                               static_cast<py::ssize_t>(r.ny), static_cast<py::ssize_t>(r.nx),
                               py::cast(&r));
        });

    py::class_<BioheatCryotherapySolver>(m, "BioheatCryotherapySolver")
        .def(py::init<const StructuredMesh&, std::vector<std::uint8_t>, std::vector<double>,
                      std::vector<double>, double, double, double, double, double, double, double,
                      double, double, double, double, double, double, double, double>(),
             py::arg("mesh"), py::arg("probe_mask"), py::arg("perfusion_map"), py::arg("q_met_map"),
             py::arg("rho_tissue"), py::arg("rho_blood"), py::arg("c_blood"), py::arg("k_unfrozen"),
             py::arg("k_frozen"), py::arg("c_unfrozen"), py::arg("c_frozen"), py::arg("T_body"),
             py::arg("T_probe"), py::arg("T_freeze"), py::arg("T_freeze_range"),
             py::arg("L_fusion"), py::arg("A"), py::arg("E_a"), py::arg("R_gas"))
        .def("simulate", &BioheatCryotherapySolver::simulate, py::arg("dt"), py::arg("num_steps"),
             py::arg("times_to_save_s"));

    // =========================================================================
    // Membrane Diffusion
    // =========================================================================
    py::class_<MembraneDiffusionResult>(m, "MembraneDiffusionResult")
        .def(py::init<>())
        .def_readonly("flux", &MembraneDiffusionResult::flux)
        .def_readonly("permeability", &MembraneDiffusionResult::permeability)
        .def_readonly("effective_diffusivity", &MembraneDiffusionResult::effective_diffusivity)
        .def("x", [](const MembraneDiffusionResult& r) { return to_numpy(r.x); })
        .def("concentration",
             [](const MembraneDiffusionResult& r) { return to_numpy(r.concentration); });

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
             py::arg("scheme") = AdvectionScheme::HYBRID)
        .def(py::init<const StructuredMesh&, double, const std::vector<double>&,
                      const std::vector<double>&, AdvectionScheme>(),
             py::arg("mesh"), py::arg("diffusivity"), py::arg("vx_field"), py::arg("vy_field"),
             py::arg("scheme") = AdvectionScheme::HYBRID)
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
        .def("set_boundary",
             py::overload_cast<int, const BoundaryCondition&>(
                 &AdvectionDiffusionSolver::setBoundaryCondition),
             py::arg("side"), py::arg("bc"));

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
        .def("linear_decay", &TransportProblem::linearDecay, py::arg("k"),
             py::return_value_policy::reference_internal)
        .def("constant_source", &TransportProblem::constantSource, py::arg("S"),
             py::return_value_policy::reference_internal)
        .def("michaelis_menten", &TransportProblem::michaelisMenten, py::arg("Vmax"), py::arg("Km"),
             py::return_value_policy::reference_internal)
        .def("logistic_growth", &TransportProblem::logisticGrowth, py::arg("r"), py::arg("K"),
             py::return_value_policy::reference_internal)
        .def("velocity", &TransportProblem::velocity, py::arg("vx"), py::arg("vy") = 0.0,
             py::return_value_policy::reference_internal)
        .def("velocity_field",
             static_cast<TransportProblem& (TransportProblem::*)(const std::vector<double>&,
                                                                 const std::vector<double>&)>(
                 &TransportProblem::velocityField),
             py::arg("vx"), py::arg("vy"), py::return_value_policy::reference_internal)
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
        .def("neumann", &TransportProblem::neumann, py::arg("side"), py::arg("flux"),
             py::return_value_policy::reference_internal)
        .def("robin", &TransportProblem::robin, py::arg("side"), py::arg("a"), py::arg("b"),
             py::arg("c"), py::return_value_policy::reference_internal)
        // Accessors
        .def("mesh", &TransportProblem::mesh, py::return_value_policy::reference_internal)
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
                                  R"(3D diffusion solver for the equation: ∂u/∂t = D∇²u

        Supports OpenMP parallelization for multi-core acceleration.

        Example:
            >>> mesh = bt.StructuredMesh3D(20, 20, 20, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
            >>> solver = bt.DiffusionSolver3D(mesh, 1e-5)
            >>> solver.set_initial_condition(u0)
            >>> solver.solve(dt, num_steps)
        )")
        .def(py::init<const StructuredMesh3D&, double>(), py::arg("mesh"), py::arg("diffusivity"))
        .def("set_initial_condition", &DiffusionSolver3D::setInitialCondition, py::arg("values"))
        .def("set_dirichlet_boundary",
             py::overload_cast<int, double>(&DiffusionSolver3D::setDirichletBoundary),
             py::arg("boundary_id"), py::arg("value"))
        .def("set_dirichlet_boundary",
             py::overload_cast<Boundary3D, double>(&DiffusionSolver3D::setDirichletBoundary),
             py::arg("boundary"), py::arg("value"))
        .def("set_neumann_boundary",
             py::overload_cast<int, double>(&DiffusionSolver3D::setNeumannBoundary),
             py::arg("boundary_id"), py::arg("flux"))
        .def("set_neumann_boundary",
             py::overload_cast<Boundary3D, double>(&DiffusionSolver3D::setNeumannBoundary),
             py::arg("boundary"), py::arg("flux"))
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

    py::class_<LinearReactionDiffusionSolver3D>(m, "LinearReactionDiffusionSolver3D",
                                                R"(3D reaction-diffusion solver: ∂u/∂t = D∇²u - k*u

        Uses implicit treatment of decay term for unconditional stability.

        Example:
            >>> mesh = bt.StructuredMesh3D(20, 1.0)  # 20x20x20 unit cube
            >>> solver = bt.LinearReactionDiffusionSolver3D(mesh, D=1e-5, decay_rate=0.01)
        )")
        .def(py::init<const StructuredMesh3D&, double, double>(), py::arg("mesh"),
             py::arg("diffusivity"), py::arg("decay_rate"))
        .def("set_initial_condition", &LinearReactionDiffusionSolver3D::setInitialCondition,
             py::arg("values"))
        .def("set_dirichlet_boundary",
             py::overload_cast<int, double>(&LinearReactionDiffusionSolver3D::setDirichletBoundary),
             py::arg("boundary_id"), py::arg("value"))
        .def("set_dirichlet_boundary",
             py::overload_cast<Boundary3D, double>(
                 &LinearReactionDiffusionSolver3D::setDirichletBoundary),
             py::arg("boundary"), py::arg("value"))
        .def("set_neumann_boundary",
             py::overload_cast<int, double>(&LinearReactionDiffusionSolver3D::setNeumannBoundary),
             py::arg("boundary_id"), py::arg("flux"))
        .def("set_neumann_boundary",
             py::overload_cast<Boundary3D, double>(
                 &LinearReactionDiffusionSolver3D::setNeumannBoundary),
             py::arg("boundary"), py::arg("flux"))
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
}

}  // namespace bindings
}  // namespace biotransport
