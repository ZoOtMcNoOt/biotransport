#ifndef BIOTRANSPORT_CORE_ANALYTICAL_HPP
#define BIOTRANSPORT_CORE_ANALYTICAL_HPP

/**
 * @file analytical.hpp
 * @brief Analytical solutions for canonical transport problems.
 *
 * These closed-form solutions are used for verification, teaching, and
 * building intuition. They correspond to standard BMEN 341 problems.
 * All functions are header-only.
 */

#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace biotransport {
namespace analytical {

// ============================================================================
// Diffusion solutions
// ============================================================================

/**
 * @brief 1D semi-infinite diffusion with constant surface concentration.
 *
 * Solves: ∂C/∂t = D ∂²C/∂x²
 * BC: C(0,t) = C_surface, C(∞,t) = C_initial
 * IC: C(x,0) = C_initial
 *
 * Solution: C(x,t) = C_initial + (C_surface - C_initial) * erfc(x / (2√(Dt)))
 *
 * @param x             Position [m]
 * @param t             Time [s]
 * @param diffusivity   Diffusivity D [m²/s]
 * @param C_surface     Surface concentration
 * @param C_initial     Initial (bulk) concentration
 * @return Concentration at (x, t)
 */
inline double diffusion_1d_semi_infinite(double x, double t, double diffusivity, double C_surface,
                                         double C_initial) {
    if (t <= 0.0) {
        return (x == 0.0) ? C_surface : C_initial;
    }
    if (diffusivity <= 0.0) {
        throw std::invalid_argument("diffusivity must be positive");
    }
    double eta = x / (2.0 * std::sqrt(diffusivity * t));
    return C_initial + (C_surface - C_initial) * std::erfc(eta);
}

/**
 * @brief Penetration depth for diffusion (approximate distance of influence).
 *
 * δ ≈ √(D t)  or more precisely 4√(Dt) for 99% penetration
 *
 * @param diffusivity   Diffusivity D [m²/s]
 * @param t             Time [s]
 * @return Penetration depth [m]
 */
inline double diffusion_penetration_depth(double diffusivity, double t) {
    if (diffusivity < 0.0 || t < 0.0) {
        throw std::invalid_argument("diffusivity and time must be non-negative");
    }
    return std::sqrt(diffusivity * t);
}

/**
 * @brief Lumped parameter exponential decay/approach.
 *
 * Used when Bi << 1. The solution is:
 * (C - C_∞) / (C_0 - C_∞) = exp(-Bi × Fo) = exp(-h_m A t / (V ρ))
 *
 * Alternatively expressed with time constant τ:
 * C(t) = C_∞ + (C_0 - C_∞) exp(-t/τ)
 *
 * @param C_0       Initial concentration
 * @param C_inf     Final (equilibrium) concentration
 * @param t         Time [s]
 * @param tau       Time constant τ [s]
 * @return Concentration at time t
 */
inline double lumped_exponential(double C_0, double C_inf, double t, double tau) {
    if (tau <= 0.0) {
        throw std::invalid_argument("time constant tau must be positive");
    }
    return C_inf + (C_0 - C_inf) * std::exp(-t / tau);
}

// ============================================================================
// Pipe flow solutions (Poiseuille / Couette)
// ============================================================================

/**
 * @brief Poiseuille flow velocity profile in a circular pipe.
 *
 * vz(r) = (a² / 4μ) × (-dp/dz) × (1 - r²/a²)
 *
 * @param r             Radial position [m]
 * @param radius        Pipe radius a [m]
 * @param dp_dz         Pressure gradient dp/dz [Pa/m] (negative for flow in +z)
 * @param viscosity     Dynamic viscosity μ [Pa·s]
 * @return Axial velocity vz [m/s]
 */
inline double poiseuille_velocity(double r, double radius, double dp_dz, double viscosity) {
    if (viscosity <= 0.0) {
        throw std::invalid_argument("viscosity must be positive");
    }
    if (radius <= 0.0) {
        throw std::invalid_argument("radius must be positive");
    }
    if (std::abs(r) > radius) {
        return 0.0;  // Outside pipe wall
    }
    double r_ratio = r / radius;
    return (radius * radius / (4.0 * viscosity)) * (-dp_dz) * (1.0 - r_ratio * r_ratio);
}

/**
 * @brief Maximum velocity in Poiseuille flow (at centerline r=0).
 *
 * v_max = (a² / 4μ) × (-dp/dz)
 *
 * @param radius        Pipe radius a [m]
 * @param dp_dz         Pressure gradient dp/dz [Pa/m]
 * @param viscosity     Dynamic viscosity μ [Pa·s]
 * @return Maximum velocity [m/s]
 */
inline double poiseuille_max_velocity(double radius, double dp_dz, double viscosity) {
    return poiseuille_velocity(0.0, radius, dp_dz, viscosity);
}

/**
 * @brief Volumetric flow rate for Poiseuille flow (Hagen-Poiseuille equation).
 *
 * Q = (π a⁴ / 8μ) × (-dp/dz)
 *
 * @param radius        Pipe radius a [m]
 * @param dp_dz         Pressure gradient dp/dz [Pa/m]
 * @param viscosity     Dynamic viscosity μ [Pa·s]
 * @return Volumetric flow rate Q [m³/s]
 */
inline double poiseuille_flow_rate(double radius, double dp_dz, double viscosity) {
    if (viscosity <= 0.0) {
        throw std::invalid_argument("viscosity must be positive");
    }
    if (radius <= 0.0) {
        throw std::invalid_argument("radius must be positive");
    }
    double a4 = radius * radius * radius * radius;
    return (M_PI * a4 / (8.0 * viscosity)) * (-dp_dz);
}

/**
 * @brief Wall shear stress for Poiseuille flow.
 *
 * τ_w = (a / 2) × |dp/dz|
 *
 * @param radius    Pipe radius a [m]
 * @param dp_dz     Pressure gradient dp/dz [Pa/m]
 * @return Wall shear stress [Pa]
 */
inline double poiseuille_wall_shear(double radius, double dp_dz) {
    return (radius / 2.0) * std::abs(dp_dz);
}

/**
 * @brief Couette flow velocity profile between parallel plates (pressure-driven).
 *
 * For plates at y = ±h/2 with origin at centerline:
 * vx(y) = (1 / 2μ) × (dp/dx) × (h²/4 - y²)
 *
 * @param y             Position from centerline [m]
 * @param half_height   Half channel height h/2 [m]
 * @param dp_dx         Pressure gradient dp/dx [Pa/m] (negative for flow in +x)
 * @param viscosity     Dynamic viscosity μ [Pa·s]
 * @return Velocity vx [m/s]
 */
inline double couette_velocity(double y, double half_height, double dp_dx, double viscosity) {
    if (viscosity <= 0.0) {
        throw std::invalid_argument("viscosity must be positive");
    }
    if (half_height <= 0.0) {
        throw std::invalid_argument("half_height must be positive");
    }
    if (std::abs(y) > half_height) {
        return 0.0;  // Outside channel
    }
    return (1.0 / (2.0 * viscosity)) * (-dp_dx) * (half_height * half_height - y * y);
}

/**
 * @brief Maximum velocity in pressure-driven Couette flow (at centerline).
 *
 * v_max = (h² / 8μ) × (-dp/dx)
 *
 * @param half_height   Half channel height h/2 [m]
 * @param dp_dx         Pressure gradient dp/dx [Pa/m]
 * @param viscosity     Dynamic viscosity μ [Pa·s]
 * @return Maximum velocity [m/s]
 */
inline double couette_max_velocity(double half_height, double dp_dx, double viscosity) {
    return couette_velocity(0.0, half_height, dp_dx, viscosity);
}

// ============================================================================
// Bernoulli / inviscid flow
// ============================================================================

/**
 * @brief Bernoulli equation: solve for velocity at point 2 given conditions.
 *
 * p₁ + ρgz₁ + ½ρv₁² = p₂ + ρgz₂ + ½ρv₂²
 *
 * Solves for v₂:
 * v₂ = √(v₁² + 2(p₁ - p₂)/ρ + 2g(z₁ - z₂))
 *
 * @param v1        Velocity at point 1 [m/s]
 * @param p1        Pressure at point 1 [Pa]
 * @param z1        Elevation at point 1 [m]
 * @param p2        Pressure at point 2 [Pa]
 * @param z2        Elevation at point 2 [m]
 * @param density   Fluid density ρ [kg/m³]
 * @param g         Gravitational acceleration [m/s²] (default 9.81)
 * @return Velocity at point 2 [m/s]
 */
inline double bernoulli_velocity(double v1, double p1, double z1, double p2, double z2,
                                 double density, double g = 9.81) {
    if (density <= 0.0) {
        throw std::invalid_argument("density must be positive");
    }
    double v2_squared = v1 * v1 + 2.0 * (p1 - p2) / density + 2.0 * g * (z1 - z2);
    if (v2_squared < 0.0) {
        throw std::invalid_argument("Bernoulli equation yields negative v²; check inputs");
    }
    return std::sqrt(v2_squared);
}

// ============================================================================
// Exponential decay / reaction kinetics
// ============================================================================

/**
 * @brief First-order decay (linear kinetics).
 *
 * dC/dt = -k C  →  C(t) = C_0 exp(-kt)
 *
 * @param C_0   Initial concentration
 * @param k     Decay rate constant [1/s]
 * @param t     Time [s]
 * @return Concentration at time t
 */
inline double first_order_decay(double C_0, double k, double t) {
    return C_0 * std::exp(-k * t);
}

/**
 * @brief Logistic growth solution.
 *
 * dC/dt = r C (1 - C/K)
 *
 * C(t) = K / (1 + ((K - C_0)/C_0) exp(-r t))
 *
 * @param C_0               Initial concentration
 * @param carrying_capacity Carrying capacity K
 * @param growth_rate       Growth rate r [1/s]
 * @param t                 Time [s]
 * @return Concentration at time t
 */
inline double logistic_growth(double C_0, double carrying_capacity, double growth_rate, double t) {
    if (C_0 <= 0.0) {
        throw std::invalid_argument("C_0 must be positive");
    }
    double ratio = (carrying_capacity - C_0) / C_0;
    return carrying_capacity / (1.0 + ratio * std::exp(-growth_rate * t));
}

// ============================================================================
// Taylor-Couette flow (rotating cylinders) — NASA Bioreactor problem
// ============================================================================

/**
 * @brief Taylor-Couette flow velocity profile between concentric rotating cylinders.
 *
 * For fluid between inner cylinder (r = a, ω = ωa) and outer cylinder (r = b, ω = ωb):
 *
 * vθ(r) = A·r + B/r
 *
 * where:
 *   A = (b²ωb - a²ωa) / (b² - a²)
 *   B = a²b²(ωa - ωb) / (b² - a²)
 *
 * From HW: NASA Bioreactor practice problem (BMEN 341).
 *
 * @param r         Radial position [m] (must be in [a, b])
 * @param a         Inner cylinder radius [m]
 * @param b         Outer cylinder radius [m]
 * @param omega_a   Angular velocity of inner cylinder [rad/s]
 * @param omega_b   Angular velocity of outer cylinder [rad/s]
 * @return Tangential velocity vθ [m/s]
 */
inline double taylor_couette_velocity(double r, double a, double b, double omega_a,
                                      double omega_b) {
    if (a <= 0.0 || b <= 0.0) {
        throw std::invalid_argument("cylinder radii must be positive");
    }
    if (a >= b) {
        throw std::invalid_argument("inner radius a must be less than outer radius b");
    }
    if (r < a || r > b) {
        throw std::invalid_argument("r must be in range [a, b]");
    }

    double a2 = a * a;
    double b2 = b * b;
    double denom = b2 - a2;

    double A = (b2 * omega_b - a2 * omega_a) / denom;
    double B = a2 * b2 * (omega_a - omega_b) / denom;

    return A * r + B / r;
}

/**
 * @brief Torque per unit length on inner cylinder in Taylor-Couette flow.
 *
 * τ = 4πμ (ωa - ωb) a²b² / (b² - a²)
 *
 * @param a         Inner cylinder radius [m]
 * @param b         Outer cylinder radius [m]
 * @param omega_a   Angular velocity of inner cylinder [rad/s]
 * @param omega_b   Angular velocity of outer cylinder [rad/s]
 * @param viscosity Dynamic viscosity μ [Pa·s]
 * @return Torque per unit length [N·m/m = N]
 */
inline double taylor_couette_torque(double a, double b, double omega_a, double omega_b,
                                    double viscosity) {
    if (a <= 0.0 || b <= 0.0) {
        throw std::invalid_argument("cylinder radii must be positive");
    }
    if (a >= b) {
        throw std::invalid_argument("inner radius a must be less than outer radius b");
    }
    if (viscosity <= 0.0) {
        throw std::invalid_argument("viscosity must be positive");
    }

    double a2 = a * a;
    double b2 = b * b;
    return 4.0 * M_PI * viscosity * (omega_a - omega_b) * a2 * b2 / (b2 - a2);
}

// ============================================================================
// Viscoelastic material models — HW6 Problem 5
// ============================================================================

/**
 * @brief Maxwell model stress relaxation.
 *
 * Spring (E) in series with dashpot (η).
 * For constant strain ε₀ applied at t=0:
 *
 * σ(t) = E·ε₀·exp(-E·t/η) = E·ε₀·exp(-t/τ)
 *
 * where τ = η/E is the relaxation time.
 *
 * @param E         Elastic modulus [Pa]
 * @param eta       Viscosity [Pa·s]
 * @param epsilon_0 Applied constant strain (dimensionless)
 * @param t         Time [s]
 * @return Stress σ(t) [Pa]
 */
inline double maxwell_relaxation(double E, double eta, double epsilon_0, double t) {
    if (E <= 0.0 || eta <= 0.0) {
        throw std::invalid_argument("E and eta must be positive");
    }
    double tau = eta / E;
    return E * epsilon_0 * std::exp(-t / tau);
}

/**
 * @brief Maxwell model relaxation time.
 *
 * τ = η / E
 *
 * @param E     Elastic modulus [Pa]
 * @param eta   Viscosity [Pa·s]
 * @return Relaxation time τ [s]
 */
inline double maxwell_relaxation_time(double E, double eta) {
    if (E <= 0.0) {
        throw std::invalid_argument("E must be positive");
    }
    return eta / E;
}

/**
 * @brief Kelvin-Voigt model creep response.
 *
 * Spring (E) in parallel with dashpot (η).
 * For constant stress σ₀ applied at t=0:
 *
 * ε(t) = (σ₀/E)·(1 - exp(-E·t/η)) = (σ₀/E)·(1 - exp(-t/τ))
 *
 * where τ = η/E is the retardation time.
 *
 * @param E       Elastic modulus [Pa]
 * @param eta     Viscosity [Pa·s]
 * @param sigma_0 Applied constant stress [Pa]
 * @param t       Time [s]
 * @return Strain ε(t) (dimensionless)
 */
inline double kelvin_voigt_creep(double E, double eta, double sigma_0, double t) {
    if (E <= 0.0 || eta <= 0.0) {
        throw std::invalid_argument("E and eta must be positive");
    }
    double tau = eta / E;
    return (sigma_0 / E) * (1.0 - std::exp(-t / tau));
}

/**
 * @brief Standard Linear Solid (SLS) stress relaxation.
 *
 * Also known as Zener model. Spring E₁ in series with (E₂ parallel to η).
 * For constant strain ε₀ applied at t=0:
 *
 * σ(t) = ε₀·(E₁ + E₂·exp(-E₂·t/η))
 *
 * At t=0: σ = ε₀(E₁ + E₂) (both springs respond)
 * At t→∞: σ = ε₀·E₁ (dashpot fully relaxed)
 *
 * From HW6 Problem 5a (BMEN 341).
 *
 * @param E1        Equilibrium spring modulus [Pa]
 * @param E2        Relaxing spring modulus [Pa]
 * @param eta       Dashpot viscosity [Pa·s]
 * @param epsilon_0 Applied constant strain (dimensionless)
 * @param t         Time [s]
 * @return Stress σ(t) [Pa]
 */
inline double sls_relaxation(double E1, double E2, double eta, double epsilon_0, double t) {
    if (E1 <= 0.0 || E2 <= 0.0 || eta <= 0.0) {
        throw std::invalid_argument("E1, E2, and eta must be positive");
    }
    double tau = eta / E2;
    return epsilon_0 * (E1 + E2 * std::exp(-t / tau));
}

/**
 * @brief Standard Linear Solid (SLS) creep response.
 *
 * For constant stress σ₀ applied at t=0:
 *
 * ε(t) = σ₀·[1/E₁ + (1/E₂)·(1 - exp(-t/τ))]
 *
 * where τ = η(E₁ + E₂)/(E₁·E₂) is the retardation time.
 *
 * @param E1      Equilibrium spring modulus [Pa]
 * @param E2      Relaxing spring modulus [Pa]
 * @param eta     Dashpot viscosity [Pa·s]
 * @param sigma_0 Applied constant stress [Pa]
 * @param t       Time [s]
 * @return Strain ε(t) (dimensionless)
 */
inline double sls_creep(double E1, double E2, double eta, double sigma_0, double t) {
    if (E1 <= 0.0 || E2 <= 0.0 || eta <= 0.0) {
        throw std::invalid_argument("E1, E2, and eta must be positive");
    }
    double tau = eta * (E1 + E2) / (E1 * E2);
    return sigma_0 * (1.0 / E1 + (1.0 / E2) * (1.0 - std::exp(-t / tau)));
}

/**
 * @brief Burgers model creep response (4-parameter model).
 *
 * Maxwell element (E₁, μ₁) in series with Kelvin-Voigt element (E₂, μ₂).
 * Creep compliance J(t) = ε(t)/σ₀:
 *
 * J(t) = 1/E₁ + t/μ₁ + (1/E₂)·(1 - exp(-E₂·t/μ₂))
 *
 * Components:
 * - 1/E₁: instantaneous elastic strain
 * - t/μ₁: viscous flow (permanent deformation)
 * - (1/E₂)(1 - exp(-E₂t/μ₂)): delayed elastic response
 *
 * From HW6 Problem 5b (BMEN 341).
 *
 * @param E1      Maxwell spring modulus [Pa]
 * @param mu1     Maxwell dashpot viscosity [Pa·s]
 * @param E2      Kelvin-Voigt spring modulus [Pa]
 * @param mu2     Kelvin-Voigt dashpot viscosity [Pa·s]
 * @param sigma_0 Applied constant stress [Pa]
 * @param t       Time [s]
 * @return Strain ε(t) (dimensionless)
 */
inline double burgers_creep(double E1, double mu1, double E2, double mu2, double sigma_0,
                            double t) {
    if (E1 <= 0.0 || mu1 <= 0.0 || E2 <= 0.0 || mu2 <= 0.0) {
        throw std::invalid_argument("all parameters must be positive");
    }
    double tau2 = mu2 / E2;
    double J = (1.0 / E1) + (t / mu1) + (1.0 / E2) * (1.0 - std::exp(-t / tau2));
    return sigma_0 * J;
}

/**
 * @brief Burgers model creep compliance.
 *
 * J(t) = 1/E₁ + t/μ₁ + (1/E₂)·(1 - exp(-E₂·t/μ₂))
 *
 * @param E1    Maxwell spring modulus [Pa]
 * @param mu1   Maxwell dashpot viscosity [Pa·s]
 * @param E2    Kelvin-Voigt spring modulus [Pa]
 * @param mu2   Kelvin-Voigt dashpot viscosity [Pa·s]
 * @param t     Time [s]
 * @return Creep compliance J(t) [1/Pa]
 */
inline double burgers_compliance(double E1, double mu1, double E2, double mu2, double t) {
    if (E1 <= 0.0 || mu1 <= 0.0 || E2 <= 0.0 || mu2 <= 0.0) {
        throw std::invalid_argument("all parameters must be positive");
    }
    double tau2 = mu2 / E2;
    return (1.0 / E1) + (t / mu1) + (1.0 / E2) * (1.0 - std::exp(-t / tau2));
}

// ============================================================================
// Complex modulus utilities (dynamic viscoelasticity) — HW6 Problem 5c
// ============================================================================

/**
 * @brief Complex modulus magnitude |G*| from storage and loss moduli.
 *
 * |G*| = √(G₁² + G₂²)
 *
 * @param G1    Storage modulus (elastic component) [Pa]
 * @param G2    Loss modulus (viscous component) [Pa]
 * @return Complex modulus magnitude [Pa]
 */
inline double complex_modulus_magnitude(double G1, double G2) {
    return std::sqrt(G1 * G1 + G2 * G2);
}

/**
 * @brief Loss tangent tan(δ) from storage and loss moduli.
 *
 * tan(δ) = G₂/G₁
 *
 * Physical meaning: ratio of energy dissipated to energy stored per cycle.
 * - tan(δ) → 0: elastic solid
 * - tan(δ) → ∞: viscous fluid
 *
 * @param G1    Storage modulus [Pa]
 * @param G2    Loss modulus [Pa]
 * @return Loss tangent (dimensionless)
 */
inline double loss_tangent(double G1, double G2) {
    if (G1 <= 0.0) {
        throw std::invalid_argument("G1 must be positive");
    }
    return G2 / G1;
}

/**
 * @brief Phase angle δ from storage and loss moduli.
 *
 * δ = atan(G₂/G₁)
 *
 * @param G1    Storage modulus [Pa]
 * @param G2    Loss modulus [Pa]
 * @return Phase angle δ [radians]
 */
inline double phase_angle(double G1, double G2) {
    return std::atan2(G2, G1);
}

}  // namespace analytical
}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_ANALYTICAL_HPP
