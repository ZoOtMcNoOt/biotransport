/**
 * @file test_analytical.cpp
 * @brief Tests for analytical solution utilities.
 */

#include <biotransport/core/analytical.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace ana = biotransport::analytical;

constexpr double EPS = 1e-6;

bool approx_eq(double a, double b, double tol = EPS) {
    return std::abs(a - b) < tol;
}

bool rel_eq(double a, double b, double rel_tol = 1e-4) {
    if (std::abs(b) < 1e-12)
        return std::abs(a) < 1e-12;
    return std::abs(a - b) / std::abs(b) < rel_tol;
}

int main() {
    int failures = 0;

    // =========================================================================
    // Semi-infinite diffusion tests
    // =========================================================================
    {
        // At x=0, C should equal C_surface
        double C = ana::diffusion_1d_semi_infinite(0.0, 1.0, 1e-9, 1.0, 0.0);
        if (!approx_eq(C, 1.0)) {
            std::cerr << "FAIL: diffusion_1d_semi_infinite at x=0 expected 1.0, got " << C << "\n";
            ++failures;
        }
    }

    {
        // At t=0, C(x>0) should equal C_initial
        double C = ana::diffusion_1d_semi_infinite(0.001, 0.0, 1e-9, 1.0, 0.0);
        if (!approx_eq(C, 0.0)) {
            std::cerr << "FAIL: diffusion_1d_semi_infinite at t=0 expected 0.0, got " << C << "\n";
            ++failures;
        }
    }

    {
        // At very large x, C should approach C_initial
        double C = ana::diffusion_1d_semi_infinite(1.0, 1.0, 1e-9, 1.0, 0.0);
        if (!approx_eq(C, 0.0, 0.01)) {
            std::cerr << "FAIL: diffusion_1d_semi_infinite at large x expected ~0, got " << C
                      << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Penetration depth tests
    // =========================================================================
    {
        // δ = sqrt(D*t) = sqrt(1e-9 * 100) = 1e-3.5 ≈ 3.16e-4
        double delta = ana::diffusion_penetration_depth(1e-9, 100.0);
        double expected = std::sqrt(1e-9 * 100.0);
        if (!rel_eq(delta, expected)) {
            std::cerr << "FAIL: diffusion_penetration_depth expected " << expected << ", got "
                      << delta << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Lumped exponential tests
    // =========================================================================
    {
        // C(0) = C_0
        double C = ana::lumped_exponential(10.0, 0.0, 0.0, 1.0);
        if (!approx_eq(C, 10.0)) {
            std::cerr << "FAIL: lumped_exponential at t=0 expected 10.0, got " << C << "\n";
            ++failures;
        }
    }

    {
        // C(∞) = C_inf
        double C = ana::lumped_exponential(10.0, 2.0, 1000.0, 1.0);
        if (!approx_eq(C, 2.0, 0.01)) {
            std::cerr << "FAIL: lumped_exponential at t→∞ expected 2.0, got " << C << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Poiseuille flow tests
    // =========================================================================
    {
        // At r=0 (centerline), v = v_max
        double radius = 0.001;   // 1 mm
        double dp_dz = -1000.0;  // Pa/m (negative = flow in +z)
        double mu = 0.001;       // Pa·s

        double v_max = ana::poiseuille_max_velocity(radius, dp_dz, mu);
        double v_center = ana::poiseuille_velocity(0.0, radius, dp_dz, mu);

        if (!approx_eq(v_max, v_center)) {
            std::cerr << "FAIL: poiseuille v_max != v(r=0)\n";
            ++failures;
        }

        // v_max = a² / (4μ) × (-dp/dz) = (1e-6) / (4 * 0.001) * 1000 = 0.25 m/s
        if (!rel_eq(v_max, 0.25)) {
            std::cerr << "FAIL: poiseuille_max_velocity expected 0.25, got " << v_max << "\n";
            ++failures;
        }
    }

    {
        // At r=a (wall), v = 0
        double v_wall = ana::poiseuille_velocity(0.001, 0.001, -1000.0, 0.001);
        if (!approx_eq(v_wall, 0.0)) {
            std::cerr << "FAIL: poiseuille_velocity at wall expected 0.0, got " << v_wall << "\n";
            ++failures;
        }
    }

    {
        // Hagen-Poiseuille: Q = π a⁴ / (8μ) × (-dp/dz)
        double radius = 0.001;
        double dp_dz = -1000.0;
        double mu = 0.001;
        double Q = ana::poiseuille_flow_rate(radius, dp_dz, mu);
        double expected = M_PI * std::pow(radius, 4) / (8.0 * mu) * 1000.0;
        if (!rel_eq(Q, expected)) {
            std::cerr << "FAIL: poiseuille_flow_rate expected " << expected << ", got " << Q
                      << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Couette flow tests
    // =========================================================================
    {
        // At y=0 (centerline), v = v_max
        double h2 = 0.001;  // half-height
        double dp_dx = -100.0;
        double mu = 0.001;

        double v_center = ana::couette_velocity(0.0, h2, dp_dx, mu);
        double v_max = ana::couette_max_velocity(h2, dp_dx, mu);

        if (!approx_eq(v_center, v_max)) {
            std::cerr << "FAIL: couette v(y=0) != v_max\n";
            ++failures;
        }
    }

    {
        // At y = ±h/2 (walls), v = 0
        double v_wall = ana::couette_velocity(0.001, 0.001, -100.0, 0.001);
        if (!approx_eq(v_wall, 0.0)) {
            std::cerr << "FAIL: couette_velocity at wall expected 0.0, got " << v_wall << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Bernoulli tests
    // =========================================================================
    {
        // Simple case: same elevation, no initial velocity, pressure drop
        // v2 = sqrt(2 * (p1 - p2) / rho)
        double v2 = ana::bernoulli_velocity(0.0, 101325.0, 0.0, 100000.0, 0.0, 1000.0);
        double expected = std::sqrt(2.0 * (101325.0 - 100000.0) / 1000.0);
        if (!rel_eq(v2, expected)) {
            std::cerr << "FAIL: bernoulli_velocity expected " << expected << ", got " << v2 << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // First-order decay tests
    // =========================================================================
    {
        // C(0) = C_0
        double C = ana::first_order_decay(100.0, 0.1, 0.0);
        if (!approx_eq(C, 100.0)) {
            std::cerr << "FAIL: first_order_decay at t=0 expected 100.0, got " << C << "\n";
            ++failures;
        }
    }

    {
        // Half-life: C(t_half) = C_0 / 2 where t_half = ln(2) / k
        double k = 0.1;
        double t_half = std::log(2.0) / k;
        double C = ana::first_order_decay(100.0, k, t_half);
        if (!rel_eq(C, 50.0)) {
            std::cerr << "FAIL: first_order_decay at half-life expected 50.0, got " << C << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Logistic growth tests
    // =========================================================================
    {
        // C(0) = C_0
        double C = ana::logistic_growth(10.0, 100.0, 0.5, 0.0);
        if (!approx_eq(C, 10.0)) {
            std::cerr << "FAIL: logistic_growth at t=0 expected 10.0, got " << C << "\n";
            ++failures;
        }
    }

    {
        // C(∞) → K
        double C = ana::logistic_growth(10.0, 100.0, 0.5, 100.0);
        if (!rel_eq(C, 100.0)) {
            std::cerr << "FAIL: logistic_growth at t→∞ expected 100.0, got " << C << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Taylor-Couette flow tests (NASA bioreactor)
    // =========================================================================
    {
        // Outer cylinder stationary (omega_b=0), inner rotating
        double a = 0.01;        // 1 cm inner radius
        double b = 0.02;        // 2 cm outer radius
        double omega_a = 10.0;  // rad/s
        double omega_b = 0.0;

        // At inner wall (r=a), velocity = a * omega_a
        double v_a = ana::taylor_couette_velocity(a, a, b, omega_a, omega_b);
        double expected_a = a * omega_a;  // 0.1 m/s
        if (!rel_eq(v_a, expected_a)) {
            std::cerr << "FAIL: taylor_couette_velocity at r=a expected " << expected_a << ", got "
                      << v_a << "\n";
            ++failures;
        }

        // At outer wall (r=b), velocity = 0 (stationary)
        double v_b = ana::taylor_couette_velocity(b, a, b, omega_a, omega_b);
        if (!approx_eq(v_b, 0.0, 1e-10)) {
            std::cerr << "FAIL: taylor_couette_velocity at r=b expected 0, got " << v_b << "\n";
            ++failures;
        }
    }

    {
        // Torque should be proportional to viscosity and angular velocity difference
        double a = 0.01, b = 0.02;
        double omega_a = 10.0, omega_b = 0.0;
        double mu = 0.001;
        double T = ana::taylor_couette_torque(a, b, omega_a, omega_b, mu);
        // Torque should be positive when inner rotates faster
        if (T <= 0.0) {
            std::cerr << "FAIL: taylor_couette_torque expected positive, got " << T << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Maxwell model tests
    // =========================================================================
    {
        double E = 1000.0;   // Pa
        double eta = 100.0;  // Pa·s
        double eps0 = 0.01;  // strain

        // At t=0, stress = E * epsilon_0
        double sigma0 = ana::maxwell_relaxation(E, eta, eps0, 0.0);
        if (!rel_eq(sigma0, E * eps0)) {
            std::cerr << "FAIL: maxwell_relaxation at t=0 expected " << E * eps0 << ", got "
                      << sigma0 << "\n";
            ++failures;
        }

        // At t >> tau, stress → 0
        double tau = ana::maxwell_relaxation_time(E, eta);
        double sigma_late = ana::maxwell_relaxation(E, eta, eps0, 10.0 * tau);
        if (!approx_eq(sigma_late, 0.0, 0.001)) {
            std::cerr << "FAIL: maxwell_relaxation at t>>tau expected ~0, got " << sigma_late
                      << "\n";
            ++failures;
        }

        // Check relaxation time
        double expected_tau = eta / E;  // 0.1 s
        if (!rel_eq(tau, expected_tau)) {
            std::cerr << "FAIL: maxwell_relaxation_time expected " << expected_tau << ", got "
                      << tau << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Kelvin-Voigt model tests
    // =========================================================================
    {
        double E = 1000.0;
        double eta = 100.0;
        double sigma0 = 10.0;

        // At t=0, strain = 0
        double eps0 = ana::kelvin_voigt_creep(E, eta, sigma0, 0.0);
        if (!approx_eq(eps0, 0.0, 1e-10)) {
            std::cerr << "FAIL: kelvin_voigt_creep at t=0 expected 0, got " << eps0 << "\n";
            ++failures;
        }

        // At t→∞, strain = sigma_0 / E
        double tau = eta / E;
        double eps_inf = ana::kelvin_voigt_creep(E, eta, sigma0, 100.0 * tau);
        double expected_inf = sigma0 / E;
        if (!rel_eq(eps_inf, expected_inf)) {
            std::cerr << "FAIL: kelvin_voigt_creep at t→∞ expected " << expected_inf << ", got "
                      << eps_inf << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Standard Linear Solid tests
    // =========================================================================
    {
        double E1 = 500.0, E2 = 500.0, eta = 100.0;
        double eps0 = 0.01;

        // At t=0, stress = epsilon_0 * (E1 + E2)
        double sigma0 = ana::sls_relaxation(E1, E2, eta, eps0, 0.0);
        double expected0 = eps0 * (E1 + E2);
        if (!rel_eq(sigma0, expected0)) {
            std::cerr << "FAIL: sls_relaxation at t=0 expected " << expected0 << ", got " << sigma0
                      << "\n";
            ++failures;
        }

        // At t→∞, stress = epsilon_0 * E1
        double tau = eta / E2;
        double sigma_inf = ana::sls_relaxation(E1, E2, eta, eps0, 100.0 * tau);
        double expected_inf = eps0 * E1;
        if (!rel_eq(sigma_inf, expected_inf)) {
            std::cerr << "FAIL: sls_relaxation at t→∞ expected " << expected_inf << ", got "
                      << sigma_inf << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Burgers model tests
    // =========================================================================
    {
        double E1 = 1000.0, mu1 = 1000.0;  // Maxwell arm
        double E2 = 500.0, mu2 = 50.0;     // Kelvin-Voigt arm
        double sigma0 = 10.0;

        // At t=0, strain = sigma_0 / E1 (instantaneous elastic)
        double eps0 = ana::burgers_creep(E1, mu1, E2, mu2, sigma0, 0.0);
        double expected0 = sigma0 / E1;
        if (!rel_eq(eps0, expected0)) {
            std::cerr << "FAIL: burgers_creep at t=0 expected " << expected0 << ", got " << eps0
                      << "\n";
            ++failures;
        }

        // Compliance at t=0 should be 1/E1
        double J0 = ana::burgers_compliance(E1, mu1, E2, mu2, 0.0);
        if (!rel_eq(J0, 1.0 / E1)) {
            std::cerr << "FAIL: burgers_compliance at t=0 expected " << 1.0 / E1 << ", got " << J0
                      << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Complex modulus utilities tests
    // =========================================================================
    {
        double G1 = 3.0, G2 = 4.0;  // 3-4-5 triangle

        double Gstar = ana::complex_modulus_magnitude(G1, G2);
        if (!rel_eq(Gstar, 5.0)) {
            std::cerr << "FAIL: complex_modulus_magnitude expected 5.0, got " << Gstar << "\n";
            ++failures;
        }

        double tan_delta = ana::loss_tangent(G1, G2);
        if (!rel_eq(tan_delta, 4.0 / 3.0)) {
            std::cerr << "FAIL: loss_tangent expected " << 4.0 / 3.0 << ", got " << tan_delta
                      << "\n";
            ++failures;
        }

        double delta = ana::phase_angle(G1, G2);
        double expected_delta = std::atan2(4.0, 3.0);
        if (!rel_eq(delta, expected_delta)) {
            std::cerr << "FAIL: phase_angle expected " << expected_delta << ", got " << delta
                      << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Error handling
    // =========================================================================
    {
        bool caught = false;
        try {
            ana::poiseuille_velocity(0.0, 0.001, -1000.0, 0.0);  // zero viscosity
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        if (!caught) {
            std::cerr << "FAIL: poiseuille_velocity should throw on zero viscosity\n";
            ++failures;
        }
    }

    if (failures == 0) {
        std::cout << "All analytical tests passed!\n";
        return 0;
    } else {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }
}
