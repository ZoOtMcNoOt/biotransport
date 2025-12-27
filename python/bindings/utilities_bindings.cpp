/**
 * @file utilities_bindings.cpp
 * @brief Python bindings for utility modules (dimensionless, analytical)
 */

#include "utilities_bindings.hpp"
#include <pybind11/stl.h>

#include <biotransport/core/dimensionless.hpp>
#include <biotransport/core/analytical.hpp>

namespace biotransport {
namespace bindings {

void register_utilities_bindings(py::module_& m) {
    // =========================================================================
    // Dimensionless number utilities (biotransport::dimensionless)
    // =========================================================================
    auto dimensionless_mod = m.def_submodule("dimensionless",
        "Dimensionless number utilities for biotransport analysis (Re, Sc, Pe, Bi, Fo, Sh).");

    dimensionless_mod.def("reynolds", &dimensionless::reynolds,
        py::arg("density"), py::arg("velocity"), py::arg("length"), py::arg("viscosity"),
        "Reynolds number: Re = ρvL/μ (inertial/viscous forces).");

    dimensionless_mod.def("reynolds_kinematic", &dimensionless::reynolds_kinematic,
        py::arg("velocity"), py::arg("length"), py::arg("kinematic_viscosity"),
        "Reynolds number from kinematic viscosity: Re = vL/ν.");

    dimensionless_mod.def("schmidt", &dimensionless::schmidt,
        py::arg("viscosity"), py::arg("density"), py::arg("diffusivity"),
        "Schmidt number: Sc = μ/(ρD) (momentum/mass diffusivity).");

    dimensionless_mod.def("schmidt_kinematic", &dimensionless::schmidt_kinematic,
        py::arg("kinematic_viscosity"), py::arg("diffusivity"),
        "Schmidt number from kinematic viscosity: Sc = ν/D.");

    dimensionless_mod.def("peclet", &dimensionless::peclet,
        py::arg("velocity"), py::arg("length"), py::arg("diffusivity"),
        "Peclet number: Pe = vL/D (convective/diffusive transport).");

    dimensionless_mod.def("biot", &dimensionless::biot,
        py::arg("h_m"), py::arg("length"), py::arg("diffusivity"),
        "Biot number (mass): Bi = h_m·L/D (external/internal resistance).");

    dimensionless_mod.def("fourier", &dimensionless::fourier,
        py::arg("diffusivity"), py::arg("time"), py::arg("length"),
        "Fourier number: Fo = Dt/L² (dimensionless time for diffusion).");

    dimensionless_mod.def("sherwood", &dimensionless::sherwood,
        py::arg("h_m"), py::arg("length"), py::arg("diffusivity"),
        "Sherwood number: Sh = h_m·L/D (convective/diffusive mass transfer).");

    dimensionless_mod.def("is_convection_dominated", &dimensionless::is_convection_dominated,
        py::arg("pe"), py::arg("threshold") = 1.0,
        "Check if Pe > threshold (convection-dominated regime).");

    dimensionless_mod.def("is_lumped_valid", &dimensionless::is_lumped_valid,
        py::arg("bi"), py::arg("threshold") = 0.1,
        "Check if Bi < threshold (lumped parameter assumption valid).");

    // =========================================================================
    // Analytical solutions (biotransport::analytical)
    // =========================================================================
    auto analytical_mod = m.def_submodule("analytical",
        "Analytical solutions for canonical transport problems (verification/teaching).");

    // Diffusion
    analytical_mod.def("diffusion_1d_semi_infinite", &analytical::diffusion_1d_semi_infinite,
        py::arg("x"), py::arg("t"), py::arg("diffusivity"), py::arg("C_surface"), py::arg("C_initial"),
        "1D semi-infinite diffusion: C(x,t) using erfc solution.");

    analytical_mod.def("diffusion_penetration_depth", &analytical::diffusion_penetration_depth,
        py::arg("diffusivity"), py::arg("t"),
        "Penetration depth δ ≈ √(Dt) for diffusion.");

    analytical_mod.def("lumped_exponential", &analytical::lumped_exponential,
        py::arg("C_0"), py::arg("C_inf"), py::arg("t"), py::arg("tau"),
        "Lumped parameter exponential: C(t) = C_∞ + (C_0 - C_∞)exp(-t/τ).");

    // Pipe flow
    analytical_mod.def("poiseuille_velocity", &analytical::poiseuille_velocity,
        py::arg("r"), py::arg("radius"), py::arg("dp_dz"), py::arg("viscosity"),
        "Poiseuille velocity profile in circular pipe: vz(r).");

    analytical_mod.def("poiseuille_max_velocity", &analytical::poiseuille_max_velocity,
        py::arg("radius"), py::arg("dp_dz"), py::arg("viscosity"),
        "Maximum velocity in Poiseuille flow (centerline).");

    analytical_mod.def("poiseuille_flow_rate", &analytical::poiseuille_flow_rate,
        py::arg("radius"), py::arg("dp_dz"), py::arg("viscosity"),
        "Volumetric flow rate Q for Poiseuille flow (Hagen-Poiseuille).");

    analytical_mod.def("poiseuille_wall_shear", &analytical::poiseuille_wall_shear,
        py::arg("radius"), py::arg("dp_dz"),
        "Wall shear stress τ_w for Poiseuille flow.");

    analytical_mod.def("couette_velocity", &analytical::couette_velocity,
        py::arg("y"), py::arg("half_height"), py::arg("dp_dx"), py::arg("viscosity"),
        "Pressure-driven Couette velocity profile: vx(y).");

    analytical_mod.def("couette_max_velocity", &analytical::couette_max_velocity,
        py::arg("half_height"), py::arg("dp_dx"), py::arg("viscosity"),
        "Maximum velocity in pressure-driven Couette flow.");

    // Bernoulli
    analytical_mod.def("bernoulli_velocity", &analytical::bernoulli_velocity,
        py::arg("v1"), py::arg("p1"), py::arg("z1"), py::arg("p2"), py::arg("z2"),
        py::arg("density"), py::arg("g") = 9.81,
        "Solve Bernoulli equation for v2 given conditions at points 1 and 2.");

    // Kinetics
    analytical_mod.def("first_order_decay", &analytical::first_order_decay,
        py::arg("C_0"), py::arg("k"), py::arg("t"),
        "First-order decay: C(t) = C_0·exp(-kt).");

    analytical_mod.def("logistic_growth", &analytical::logistic_growth,
        py::arg("C_0"), py::arg("carrying_capacity"), py::arg("growth_rate"), py::arg("t"),
        "Logistic growth: C(t) = K / (1 + ((K-C_0)/C_0)·exp(-rt)).");

    // Taylor-Couette flow (rotating cylinders) — NASA Bioreactor
    analytical_mod.def("taylor_couette_velocity", &analytical::taylor_couette_velocity,
        py::arg("r"), py::arg("a"), py::arg("b"), py::arg("omega_a"), py::arg("omega_b"),
        "Taylor-Couette velocity profile vθ(r) between concentric rotating cylinders.\n"
        "a = inner radius, b = outer radius, omega_a/b = angular velocities.");

    analytical_mod.def("taylor_couette_torque", &analytical::taylor_couette_torque,
        py::arg("a"), py::arg("b"), py::arg("omega_a"), py::arg("omega_b"), py::arg("viscosity"),
        "Torque per unit length on inner cylinder in Taylor-Couette flow.");

    // Viscoelastic models — HW6 Problem 5
    analytical_mod.def("maxwell_relaxation", &analytical::maxwell_relaxation,
        py::arg("E"), py::arg("eta"), py::arg("epsilon_0"), py::arg("t"),
        "Maxwell model stress relaxation: σ(t) = E·ε₀·exp(-t/τ) where τ = η/E.");

    analytical_mod.def("maxwell_relaxation_time", &analytical::maxwell_relaxation_time,
        py::arg("E"), py::arg("eta"),
        "Maxwell relaxation time τ = η/E.");

    analytical_mod.def("kelvin_voigt_creep", &analytical::kelvin_voigt_creep,
        py::arg("E"), py::arg("eta"), py::arg("sigma_0"), py::arg("t"),
        "Kelvin-Voigt creep: ε(t) = (σ₀/E)·(1 - exp(-t/τ)) where τ = η/E.");

    analytical_mod.def("sls_relaxation", &analytical::sls_relaxation,
        py::arg("E1"), py::arg("E2"), py::arg("eta"), py::arg("epsilon_0"), py::arg("t"),
        "Standard Linear Solid stress relaxation: σ(t) = ε₀·(E₁ + E₂·exp(-t/τ)).");

    analytical_mod.def("sls_creep", &analytical::sls_creep,
        py::arg("E1"), py::arg("E2"), py::arg("eta"), py::arg("sigma_0"), py::arg("t"),
        "Standard Linear Solid creep response.");

    analytical_mod.def("burgers_creep", &analytical::burgers_creep,
        py::arg("E1"), py::arg("mu1"), py::arg("E2"), py::arg("mu2"), py::arg("sigma_0"), py::arg("t"),
        "Burgers 4-parameter model creep: ε(t) = σ₀·J(t).");

    analytical_mod.def("burgers_compliance", &analytical::burgers_compliance,
        py::arg("E1"), py::arg("mu1"), py::arg("E2"), py::arg("mu2"), py::arg("t"),
        "Burgers creep compliance: J(t) = 1/E₁ + t/μ₁ + (1/E₂)·(1 - exp(-t/τ₂)).");

    // Complex modulus utilities — dynamic viscoelasticity
    analytical_mod.def("complex_modulus_magnitude", &analytical::complex_modulus_magnitude,
        py::arg("G1"), py::arg("G2"),
        "Complex modulus magnitude |G*| = √(G₁² + G₂²).");

    analytical_mod.def("loss_tangent", &analytical::loss_tangent,
        py::arg("G1"), py::arg("G2"),
        "Loss tangent tan(δ) = G₂/G₁. Ratio of energy dissipated to stored per cycle.");

    analytical_mod.def("phase_angle", &analytical::phase_angle,
        py::arg("G1"), py::arg("G2"),
        "Phase angle δ = atan2(G₂, G₁) in radians.");
}

} // namespace bindings
} // namespace biotransport
