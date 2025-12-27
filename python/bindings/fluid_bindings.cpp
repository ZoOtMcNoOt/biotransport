/**
 * @file fluid_bindings.cpp
 * @brief Python bindings for fluid dynamics solvers
 */

// Ensure M_PI is defined on MSVC
#define _USE_MATH_DEFINES
#include <cmath>

#include "fluid_bindings.hpp"
#include "binding_helpers.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/boundary.hpp>
#include <biotransport/physics/fluid_dynamics/darcy_flow.hpp>
#include <biotransport/physics/fluid_dynamics/stokes.hpp>
#include <biotransport/physics/fluid_dynamics/navier_stokes.hpp>
#include <biotransport/physics/fluid_dynamics/non_newtonian.hpp>

namespace biotransport {
namespace bindings {

void register_fluid_bindings(py::module_& m) {
    // =========================================================================
    // Darcy Flow Solver (porous media)
    // =========================================================================
    py::class_<DarcyFlowResult>(m, "DarcyFlowResult")
        .def(py::init<>())
        .def_readonly("iterations", &DarcyFlowResult::iterations)
        .def_readonly("residual", &DarcyFlowResult::residual)
        .def_readonly("converged", &DarcyFlowResult::converged)
        .def("pressure", [](const DarcyFlowResult& r) { return to_numpy(r.pressure); })
        .def("vx", [](const DarcyFlowResult& r) { return to_numpy(r.vx); })
        .def("vy", [](const DarcyFlowResult& r) { return to_numpy(r.vy); });

    py::class_<DarcyFlowSolver>(m, "DarcyFlowSolver")
        .def(py::init<const StructuredMesh&, double>(),
             py::arg("mesh"), py::arg("kappa"),
             py::keep_alive<1, 2>())
        .def(py::init<const StructuredMesh&, const std::vector<double>&>(),
             py::arg("mesh"), py::arg("kappa"),
             py::keep_alive<1, 2>())
        .def("set_dirichlet", &DarcyFlowSolver::setDirichlet,
             py::arg("side"), py::arg("pressure"),
             py::return_value_policy::reference_internal)
        .def("set_neumann", &DarcyFlowSolver::setNeumann,
             py::arg("side"), py::arg("flux"),
             py::return_value_policy::reference_internal)
        .def("set_internal_pressure", &DarcyFlowSolver::setInternalPressure,
             py::arg("mask"), py::arg("pressure"),
             py::return_value_policy::reference_internal)
        .def("set_omega", &DarcyFlowSolver::setOmega,
             py::arg("omega"),
             py::return_value_policy::reference_internal)
        .def("set_tolerance", &DarcyFlowSolver::setTolerance,
             py::arg("tol"),
             py::return_value_policy::reference_internal)
        .def("set_max_iterations", &DarcyFlowSolver::setMaxIterations,
             py::arg("max_iter"),
             py::return_value_policy::reference_internal)
        .def("set_initial_guess", &DarcyFlowSolver::setInitialGuess,
             py::arg("pressure"),
             py::return_value_policy::reference_internal)
        .def("solve", &DarcyFlowSolver::solve)
        .def("kappa", [](const DarcyFlowSolver& s) { return to_numpy(s.kappa()); });

    // =========================================================================
    // Stokes Flow Solver
    // =========================================================================
    py::enum_<VelocityBCType>(m, "VelocityBCType")
        .value("DIRICHLET", VelocityBCType::DIRICHLET)
        .value("NEUMANN", VelocityBCType::NEUMANN)
        .value("NOSLIP", VelocityBCType::NOSLIP)
        .value("INFLOW", VelocityBCType::INFLOW)
        .value("OUTFLOW", VelocityBCType::OUTFLOW)
        .export_values();

    py::class_<VelocityBC>(m, "VelocityBC")
        .def_readwrite("type", &VelocityBC::type)
        .def_readwrite("u_value", &VelocityBC::u_value)
        .def_readwrite("v_value", &VelocityBC::v_value)
        .def_static("no_slip", &VelocityBC::NoSlip)
        .def_static("inflow", &VelocityBC::Inflow,
             py::arg("u"), py::arg("v") = 0.0)
        .def_static("outflow", &VelocityBC::Outflow)
        .def_static("dirichlet", &VelocityBC::Dirichlet,
             py::arg("u"), py::arg("v"))
        .def_static("stress_free", &VelocityBC::StressFree);

    py::class_<StokesResult>(m, "StokesResult")
        .def(py::init<>())
        .def_readonly("iterations", &StokesResult::iterations)
        .def_readonly("residual", &StokesResult::residual)
        .def_readonly("divergence", &StokesResult::divergence)
        .def_readonly("converged", &StokesResult::converged)
        .def("u", [](const StokesResult& r) { return to_numpy(r.u); })
        .def("v", [](const StokesResult& r) { return to_numpy(r.v); })
        .def("pressure", [](const StokesResult& r) { return to_numpy(r.pressure); });

    py::class_<StokesSolver>(m, "StokesSolver")
        .def(py::init<const StructuredMesh&, double>(),
             py::arg("mesh"), py::arg("viscosity"),
             py::keep_alive<1, 2>())
        .def("set_velocity_bc", &StokesSolver::setVelocityBC,
             py::arg("side"), py::arg("bc"),
             py::return_value_policy::reference_internal)
        .def("set_body_force", py::overload_cast<double, double>(&StokesSolver::setBodyForce),
             py::arg("fx"), py::arg("fy"),
             py::return_value_policy::reference_internal)
        .def("set_tolerance", &StokesSolver::setTolerance,
             py::arg("tol"),
             py::return_value_policy::reference_internal)
        .def("set_max_iterations", &StokesSolver::setMaxIterations,
             py::arg("max_iter"),
             py::return_value_policy::reference_internal)
        .def("set_pressure_relaxation", &StokesSolver::setPressureRelaxation,
             py::arg("omega_p"),
             py::return_value_policy::reference_internal)
        .def("set_velocity_relaxation", &StokesSolver::setVelocityRelaxation,
             py::arg("omega_v"),
             py::return_value_policy::reference_internal)
        .def("solve", &StokesSolver::solve)
        .def("viscosity", &StokesSolver::viscosity)
        .def("reynolds", &StokesSolver::reynolds,
             py::arg("L"), py::arg("U"), py::arg("rho"));

    // =========================================================================
    // Navier-Stokes Solver
    // =========================================================================
    py::enum_<ConvectionScheme>(m, "ConvectionScheme")
        .value("UPWIND", ConvectionScheme::UPWIND)
        .value("CENTRAL", ConvectionScheme::CENTRAL)
        .value("QUICK", ConvectionScheme::QUICK)
        .value("HYBRID", ConvectionScheme::HYBRID)
        .export_values();

    py::class_<NavierStokesResult>(m, "NavierStokesResult")
        .def(py::init<>())
        .def_readonly("time", &NavierStokesResult::time)
        .def_readonly("time_steps", &NavierStokesResult::time_steps)
        .def_readonly("max_velocity", &NavierStokesResult::max_velocity)
        .def_readonly("reynolds", &NavierStokesResult::reynolds)
        .def_readonly("stable", &NavierStokesResult::stable)
        .def("u", [](const NavierStokesResult& r) { return to_numpy(r.u); })
        .def("v", [](const NavierStokesResult& r) { return to_numpy(r.v); })
        .def("pressure", [](const NavierStokesResult& r) { return to_numpy(r.pressure); });

    py::class_<NavierStokesSolver>(m, "NavierStokesSolver")
        .def(py::init<const StructuredMesh&, double, double>(),
             py::arg("mesh"), py::arg("density"), py::arg("viscosity"),
             py::keep_alive<1, 2>())
        .def("set_velocity_bc", &NavierStokesSolver::setVelocityBC,
             py::arg("side"), py::arg("bc"),
             py::return_value_policy::reference_internal)
        .def("set_body_force", py::overload_cast<double, double>(&NavierStokesSolver::setBodyForce),
             py::arg("fx"), py::arg("fy"),
             py::return_value_policy::reference_internal)
        .def("set_initial_velocity", &NavierStokesSolver::setInitialVelocity,
             py::arg("u0"), py::arg("v0"),
             py::return_value_policy::reference_internal)
        .def("set_convection_scheme", &NavierStokesSolver::setConvectionScheme,
             py::arg("scheme"),
             py::return_value_policy::reference_internal)
        .def("set_cfl", &NavierStokesSolver::setCFL,
             py::arg("cfl"),
             py::return_value_policy::reference_internal)
        .def("set_time_step", &NavierStokesSolver::setTimeStep,
             py::arg("dt"),
             py::return_value_policy::reference_internal)
        .def("set_pressure_tolerance", &NavierStokesSolver::setPressureTolerance,
             py::arg("tol"),
             py::return_value_policy::reference_internal)
        .def("set_max_pressure_iterations", &NavierStokesSolver::setMaxPressureIterations,
             py::arg("max_iter"),
             py::return_value_policy::reference_internal)
        .def("solve", &NavierStokesSolver::solve,
             py::arg("duration"), py::arg("output_interval") = 0.0)
        .def("solve_steps", &NavierStokesSolver::solveSteps,
             py::arg("num_steps"))
        .def("density", &NavierStokesSolver::density)
        .def("viscosity", &NavierStokesSolver::viscosity)
        .def("kinematic_viscosity", &NavierStokesSolver::kinematicViscosity)
        .def("reynolds", &NavierStokesSolver::reynolds,
             py::arg("L"), py::arg("U"));

    // =========================================================================
    // Non-Newtonian Fluid Models
    // =========================================================================
    py::enum_<FluidModel>(m, "FluidModel")
        .value("NEWTONIAN", FluidModel::NEWTONIAN)
        .value("POWER_LAW", FluidModel::POWER_LAW)
        .value("CARREAU", FluidModel::CARREAU)
        .value("CARREAU_YASUDA", FluidModel::CARREAU_YASUDA)
        .value("CROSS", FluidModel::CROSS)
        .value("BINGHAM", FluidModel::BINGHAM)
        .value("HERSCHEL_BULKLEY", FluidModel::HERSCHEL_BULKLEY)
        .value("CASSON", FluidModel::CASSON)
        .export_values();

    // Base ViscosityModel class
    py::class_<ViscosityModel>(m, "ViscosityModel")
        .def("viscosity", &ViscosityModel::viscosity, py::arg("gamma_dot"))
        .def("shear_stress", &ViscosityModel::shearStress, py::arg("gamma_dot"))
        .def("name", &ViscosityModel::name)
        .def("type", &ViscosityModel::type);

    // Newtonian model
    py::class_<NewtonianModel, ViscosityModel>(m, "NewtonianModel")
        .def(py::init<double>(), py::arg("mu0"))
        .def("mu0", &NewtonianModel::mu0);

    // Power-law model
    py::class_<PowerLawModel, ViscosityModel>(m, "PowerLawModel")
        .def(py::init<double, double, double>(),
             py::arg("K"), py::arg("n"), py::arg("gamma_min") = 1e-10)
        .def("K", &PowerLawModel::K)
        .def("n", &PowerLawModel::n)
        .def("is_shear_thinning", &PowerLawModel::isShearThinning)
        .def("is_shear_thickening", &PowerLawModel::isShearThickening);

    // Carreau model
    py::class_<CarreauModel, ViscosityModel>(m, "CarreauModel")
        .def(py::init<double, double, double, double>(),
             py::arg("mu0"), py::arg("mu_inf"), py::arg("lambda_"), py::arg("n"))
        .def("mu0", &CarreauModel::mu0)
        .def("mu_inf", &CarreauModel::muInf)
        .def("lambda_", &CarreauModel::lambda)
        .def("n", &CarreauModel::n);

    // Carreau-Yasuda model
    py::class_<CarreauYasudaModel, ViscosityModel>(m, "CarreauYasudaModel")
        .def(py::init<double, double, double, double, double>(),
             py::arg("mu0"), py::arg("mu_inf"), py::arg("lambda_"), py::arg("a"), py::arg("n"))
        .def("mu0", &CarreauYasudaModel::mu0)
        .def("mu_inf", &CarreauYasudaModel::muInf)
        .def("lambda_", &CarreauYasudaModel::lambda)
        .def("a", &CarreauYasudaModel::a)
        .def("n", &CarreauYasudaModel::n);

    // Cross model
    py::class_<CrossModel, ViscosityModel>(m, "CrossModel")
        .def(py::init<double, double, double, double>(),
             py::arg("mu0"), py::arg("mu_inf"), py::arg("K"), py::arg("m"))
        .def("mu0", &CrossModel::mu0)
        .def("mu_inf", &CrossModel::muInf)
        .def("K", &CrossModel::K)
        .def("m", &CrossModel::m);

    // Bingham model
    py::class_<BinghamModel, ViscosityModel>(m, "BinghamModel")
        .def(py::init<double, double, double>(),
             py::arg("tau_y"), py::arg("mu_p"), py::arg("epsilon") = 1e-6)
        .def("yield_stress", &BinghamModel::yieldStress)
        .def("plastic_viscosity", &BinghamModel::plasticViscosity)
        .def("bingham_number", &BinghamModel::binghamNumber,
             py::arg("L"), py::arg("U"));

    // Herschel-Bulkley model
    py::class_<HerschelBulkleyModel, ViscosityModel>(m, "HerschelBulkleyModel")
        .def(py::init<double, double, double, double>(),
             py::arg("tau_y"), py::arg("K"), py::arg("n"), py::arg("epsilon") = 1e-6)
        .def("yield_stress", &HerschelBulkleyModel::yieldStress)
        .def("K", &HerschelBulkleyModel::K)
        .def("n", &HerschelBulkleyModel::n);

    // Casson model (blood rheology)
    py::class_<CassonModel, ViscosityModel>(m, "CassonModel")
        .def(py::init<double, double, double>(),
             py::arg("tau_y"), py::arg("mu_p"), py::arg("epsilon") = 1e-6)
        .def("yield_stress", &CassonModel::yieldStress)
        .def("plastic_viscosity", &CassonModel::plasticViscosity);

    // Blood rheology utility functions
    m.def("blood_casson_model", &bloodCassonModel,
          py::arg("hematocrit"),
          "Create Casson model for blood at given hematocrit (0-0.7).");

    m.def("blood_carreau_model", &bloodCarreauModel,
          py::arg("hematocrit"),
          "Create Carreau model for blood at given hematocrit (0-0.7).");

    m.def("pipe_wall_shear_rate", &pipeWallShearRate,
          py::arg("Q"), py::arg("R"),
          "Wall shear rate in pipe flow: gamma_w = 4Q/(pi*R^3).");
}

} // namespace bindings
} // namespace biotransport
