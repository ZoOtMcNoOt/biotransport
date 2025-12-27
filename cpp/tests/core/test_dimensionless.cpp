/**
 * @file test_dimensionless.cpp
 * @brief Tests for dimensionless number utilities.
 */

#include <biotransport/core/dimensionless.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace dim = biotransport::dimensionless;

constexpr double EPS = 1e-10;

bool approx_eq(double a, double b, double tol = EPS) {
    return std::abs(a - b) < tol;
}

int main() {
    std::cout << "Starting dimensionless tests..." << std::endl;
    std::cout.flush();

    int failures = 0;

    // =========================================================================
    // Reynolds number tests
    // =========================================================================
    {
        // Water at 20°C: ρ ≈ 998 kg/m³, μ ≈ 0.001 Pa·s
        // v = 1 m/s, L = 0.01 m (1 cm pipe)
        // Re = 998 * 1 * 0.01 / 0.001 = 9980
        double Re = dim::reynolds(998.0, 1.0, 0.01, 0.001);
        if (!approx_eq(Re, 9980.0, 1.0)) {
            std::cerr << "FAIL: reynolds() expected ~9980, got " << Re << "\n";
            ++failures;
        }
    }

    {
        // Kinematic version: ν = μ/ρ = 0.001/998 ≈ 1.002e-6 m²/s
        double Re = dim::reynolds_kinematic(1.0, 0.01, 1.002e-6);
        if (!approx_eq(Re, 9980.04, 1.0)) {
            std::cerr << "FAIL: reynolds_kinematic() expected ~9980, got " << Re << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Schmidt number tests
    // =========================================================================
    {
        // Sc = μ / (ρ D) with D = 1e-9 m²/s (typical small molecule in water)
        // Sc = 0.001 / (998 * 1e-9) ≈ 1002
        double Sc = dim::schmidt(0.001, 998.0, 1e-9);
        if (!approx_eq(Sc, 1002.004, 1.0)) {
            std::cerr << "FAIL: schmidt() expected ~1002, got " << Sc << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Peclet number tests
    // =========================================================================
    {
        // Pe = vL/D = 1.0 * 0.01 / 1e-9 = 1e7
        double Pe = dim::peclet(1.0, 0.01, 1e-9);
        if (!approx_eq(Pe, 1e7, 1e3)) {
            std::cerr << "FAIL: peclet() expected 1e7, got " << Pe << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Biot number tests
    // =========================================================================
    {
        // Bi = h_m * L / D
        // h_m = 1e-5 m/s, L = 0.001 m, D = 1e-9 m²/s
        // Bi = 1e-5 * 0.001 / 1e-9 = 10
        double Bi = dim::biot(1e-5, 0.001, 1e-9);
        if (!approx_eq(Bi, 10.0)) {
            std::cerr << "FAIL: biot() expected 10, got " << Bi << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Fourier number tests
    // =========================================================================
    {
        // Fo = D * t / L^2
        // D = 1e-9, t = 1000 s, L = 0.001 m
        // Fo = 1e-9 * 1000 / 1e-6 = 1.0
        double Fo = dim::fourier(1e-9, 1000.0, 0.001);
        if (!approx_eq(Fo, 1.0)) {
            std::cerr << "FAIL: fourier() expected 1.0, got " << Fo << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Sherwood number tests
    // =========================================================================
    {
        // Sh = h_m * L / D (same formula as Bi, different interpretation)
        double Sh = dim::sherwood(1e-4, 0.01, 1e-9);
        // Sh = 1e-4 * 0.01 / 1e-9 = 1e-6 / 1e-9 = 1000
        if (!approx_eq(Sh, 1000.0, 1.0)) {
            std::cerr << "FAIL: sherwood() expected 1000, got " << Sh << "\n";
            ++failures;
        }
    }

    // =========================================================================
    // Boolean checks
    // =========================================================================
    {
        if (!dim::is_convection_dominated(10.0)) {
            std::cerr << "FAIL: is_convection_dominated(10) should be true\n";
            ++failures;
        }
        if (dim::is_convection_dominated(0.5)) {
            std::cerr << "FAIL: is_convection_dominated(0.5) should be false\n";
            ++failures;
        }
        if (!dim::is_lumped_valid(0.05)) {
            std::cerr << "FAIL: is_lumped_valid(0.05) should be true\n";
            ++failures;
        }
        if (dim::is_lumped_valid(0.5)) {
            std::cerr << "FAIL: is_lumped_valid(0.5) should be false\n";
            ++failures;
        }
    }

    // =========================================================================
    // Error handling
    // =========================================================================
    {
        bool caught = false;
        try {
            dim::reynolds(1.0, 1.0, 1.0, 0.0);  // zero viscosity
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        if (!caught) {
            std::cerr << "FAIL: reynolds() should throw on zero viscosity\n";
            ++failures;
        }
    }

    if (failures == 0) {
        std::cout << "All dimensionless tests passed!\n";
        return 0;
    } else {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }
}
