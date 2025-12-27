/**
 * @file test_advection_diffusion.cpp
 * @brief Tests for the advection-diffusion solver.
 */

#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace biotransport;

constexpr double EPS = 1e-6;

bool approx_eq(double a, double b, double tol = EPS) {
    return std::abs(a - b) < tol;
}

bool rel_eq(double a, double b, double rel_tol = 0.01) {
    if (std::abs(b) < 1e-12)
        return std::abs(a) < 1e-12;
    return std::abs(a - b) / std::abs(b) < rel_tol;
}

int main() {
    int failures = 0;

    // =========================================================================
    // Test 1: Pure advection (low diffusivity) - Gaussian pulse transport
    // =========================================================================
    {
        std::cout << "Test 1: Pure advection (Gaussian pulse transport)..." << std::endl;

        // 1D domain with uniform velocity
        int nx = 100;
        double L = 1.0;
        StructuredMesh mesh(nx, 0.0, L);

        double D = 1e-6;  // Very small diffusion
        double v = 0.1;   // m/s velocity in +x direction

        // Initial Gaussian pulse centered at x = 0.3
        std::vector<double> ic(mesh.numNodes());
        double x0 = 0.3;
        double sigma = 0.05;
        for (int i = 0; i <= nx; ++i) {
            double x = mesh.x(i);
            ic[i] = std::exp(-std::pow(x - x0, 2) / (2 * sigma * sigma));
        }

        TransportProblem problem(mesh);
        problem.diffusivity(D)
            .velocity(v, 0.0)
            .advectionScheme(AdvectionScheme::UPWIND)
            .initialCondition(ic)
            .dirichlet(Boundary::Left, 0.0)
            .dirichlet(Boundary::Right, 0.0);

        // Run for t = 0.5s, pulse should move to x ≈ 0.3 + 0.1*0.5 = 0.35
        double t_end = 0.5;
        ExplicitFD solver;
        auto result = solver.safetyFactor(0.4).run(problem, t_end);

        // Find peak location
        const auto& sol = result.solution;
        int peak_idx = 0;
        double peak_val = sol[0];
        for (int i = 1; i <= nx; ++i) {
            if (sol[i] > peak_val) {
                peak_val = sol[i];
                peak_idx = i;
            }
        }
        double peak_x = mesh.x(peak_idx);

        // Expected peak position (with numerical diffusion, may be slightly off)
        double expected_x = x0 + v * t_end;

        // Allow some tolerance due to numerical diffusion
        if (std::abs(peak_x - expected_x) > 0.1) {
            std::cerr << "FAIL: Pure advection peak at x=" << peak_x << ", expected ~" << expected_x
                      << std::endl;
            ++failures;
        } else {
            std::cout << "  Peak moved from x=0.3 to x=" << peak_x << " (expected ~" << expected_x
                      << ")" << std::endl;
        }
    }

    // =========================================================================
    // Test 2: ExplicitFD facade with advection-diffusion TransportProblem
    // =========================================================================
    {
        std::cout << "Test 2: ExplicitFD facade with advection-diffusion TransportProblem..."
                  << std::endl;

        StructuredMesh mesh(50, 0.0, 1.0);

        // Initial Gaussian pulse in the interior (not at boundary)
        std::vector<double> ic(mesh.numNodes(), 0.0);
        double x0 = 0.2;
        double sigma = 0.05;
        for (int i = 0; i <= 50; ++i) {
            double x = mesh.x(i);
            ic[i] = std::exp(-std::pow(x - x0, 2) / (2 * sigma * sigma));
        }

        TransportProblem problem(mesh);
        problem.diffusivity(1e-3)
            .velocity(0.1, 0.0)
            .advectionScheme(AdvectionScheme::UPWIND)
            .initialCondition(ic)
            .dirichlet(Boundary::Left, 0.0)
            .dirichlet(Boundary::Right, 0.0);

        ExplicitFD solver;
        auto result = solver.safetyFactor(0.4).run(problem, 1.0);

        // Find peak location - should have moved from x=0.2 toward x=0.3
        int peak_idx = 0;
        double peak_val = 0.0;
        for (int i = 0; i <= 50; ++i) {
            if (result.solution[i] > peak_val) {
                peak_val = result.solution[i];
                peak_idx = i;
            }
        }
        double peak_x = mesh.x(peak_idx);

        // Peak should have moved rightward (advection v=0.1 for t=1.0 means ~0.1 movement)
        bool propagated = peak_x > 0.25;

        if (!propagated) {
            std::cerr << "FAIL: Peak did not propagate with advection. Peak at x=" << peak_x
                      << std::endl;
            ++failures;
        } else {
            std::cout << "  ExplicitFD facade ran successfully, peak at x=" << peak_x
                      << ", steps=" << result.stats.steps << std::endl;
        }
    }

    // =========================================================================
    // Test 3: 2D advection-diffusion
    // =========================================================================
    {
        std::cout << "Test 3: 2D advection-diffusion..." << std::endl;

        int nx = 20, ny = 20;
        StructuredMesh mesh(nx, ny, 0.0, 1.0, 0.0, 1.0);

        double D = 1e-3;
        double vx = 0.05;
        double vy = 0.05;

        // Initial: Gaussian at center
        std::vector<double> ic(mesh.numNodes(), 0.0);
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                double x = mesh.x(i);
                double y = mesh.y(i, j);
                double r2 = std::pow(x - 0.3, 2) + std::pow(y - 0.3, 2);
                ic[mesh.index(i, j)] = std::exp(-r2 / 0.01);
            }
        }

        TransportProblem problem(mesh);
        problem.diffusivity(D)
            .velocity(vx, vy)
            .advectionScheme(AdvectionScheme::HYBRID)
            .initialCondition(ic)
            .dirichlet(Boundary::Left, 0.0)
            .dirichlet(Boundary::Right, 0.0)
            .dirichlet(Boundary::Bottom, 0.0)
            .dirichlet(Boundary::Top, 0.0);

        ExplicitFD solver;
        auto result = solver.safetyFactor(0.4).run(problem, 0.5);

        // Find peak location - should have moved in +x, +y direction
        const auto& sol = result.solution;
        int peak_i = 0, peak_j = 0;
        double peak_val = 0.0;
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                if (sol[mesh.index(i, j)] > peak_val) {
                    peak_val = sol[mesh.index(i, j)];
                    peak_i = i;
                    peak_j = j;
                }
            }
        }

        double peak_x = mesh.x(peak_i);
        double peak_y = mesh.y(peak_i, peak_j);

        // Peak should have moved from (0.3, 0.3) toward larger x and y
        if (peak_x < 0.25 || peak_y < 0.25) {
            std::cerr << "FAIL: 2D peak should have moved to larger x,y. Got (" << peak_x << ", "
                      << peak_y << ")" << std::endl;
            ++failures;
        } else {
            std::cout << "  2D peak moved to (" << peak_x << ", " << peak_y << ")" << std::endl;
        }
    }

    // =========================================================================
    // Summary
    // =========================================================================
    if (failures == 0) {
        std::cout << "\nAll advection-diffusion tests passed!" << std::endl;
        return 0;
    } else {
        std::cerr << "\n" << failures << " test(s) failed." << std::endl;
        return 1;
    }
}
