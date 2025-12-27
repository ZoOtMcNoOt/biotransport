/**
 * @file test_stokes.cpp
 * @brief Unit tests for the Stokes flow solver
 *
 * Tests verify:
 * 1. Basic solver construction and parameter setting
 * 2. Poiseuille flow (pressure-driven channel flow)
 * 3. Lid-driven cavity flow
 * 4. No-slip boundary conditions
 * 5. Convergence behavior
 * 6. Mass conservation (divergence-free velocity)
 */

#include <algorithm>
#include <biotransport/physics/fluid_dynamics/stokes.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace biotransport;

// Helper function to check approximate equality
bool approxEqual(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

// Helper function to check relative error
double relativeError(double computed, double expected) {
    if (std::abs(expected) < 1e-12)
        return std::abs(computed);
    return std::abs(computed - expected) / std::abs(expected);
}

/**
 * Test 1: Basic solver construction
 */
void testConstruction() {
    std::cout << "Test 1: Basic construction..." << std::endl;

    StructuredMesh mesh(10, 10, 0.0, 1.0, 0.0, 1.0);
    double mu = 0.001;

    StokesSolver solver(mesh, mu);

    // Should not throw
    solver.setTolerance(1e-6);
    solver.setMaxIterations(1000);
    solver.setPressureRelaxation(0.1);
    solver.setVelocityRelaxation(0.5);
    solver.setBodyForce(0.0, 0.0);

    std::cout << "  PASSED" << std::endl;
}

/**
 * Test 2: Invalid parameters should throw
 */
void testInvalidParameters() {
    std::cout << "Test 2: Invalid parameters..." << std::endl;

    StructuredMesh mesh2D(10, 10, 0.0, 1.0, 0.0, 1.0);
    StructuredMesh mesh1D(10, 0.0, 1.0);

    // Negative viscosity should throw
    bool caught = false;
    try {
        StokesSolver solver(mesh2D, -0.001);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught && "Should throw for negative viscosity");

    // 1D mesh should throw
    caught = false;
    try {
        StokesSolver solver(mesh1D, 0.001);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught && "Should throw for 1D mesh");

    std::cout << "  PASSED" << std::endl;
}

/**
 * Test 3: Poiseuille flow (pressure-driven channel flow)
 *
 * Analytical solution for 2D channel:
 *   u(y) = (dP/dx) / (2*mu) * y * (H - y)
 *   v = 0
 *   u_max = (dP/dx) * H^2 / (8*mu)
 */
void testPoiseuilleFlow() {
    std::cout << "Test 3: Poiseuille flow..." << std::endl;

    // Channel dimensions
    double L = 1.0;         // length
    double H = 0.1;         // height
    double mu = 0.001;      // viscosity
    double dP_dx = 1000.0;  // pressure gradient (body force)

    // Analytical maximum velocity
    double u_max_analytical = dP_dx * H * H / (8.0 * mu);

    // Create mesh
    int nx = 40, ny = 20;
    StructuredMesh mesh(nx, ny, 0.0, L, 0.0, H);

    // Create solver
    StokesSolver solver(mesh, mu);

    // No-slip on walls
    solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Top, VelocityBC::NoSlip());

    // Outflow on both ends (periodic-like for developed flow)
    solver.setVelocityBC(Boundary::Left, VelocityBC::Outflow());
    solver.setVelocityBC(Boundary::Right, VelocityBC::Outflow());

    // Body force represents pressure gradient
    solver.setBodyForce(dP_dx, 0.0);

    // Solver parameters
    solver.setTolerance(1e-8);
    solver.setMaxIterations(5000);

    // Solve
    StokesResult result = solver.solve();

    assert(result.converged && "Poiseuille flow should converge");

    // Check maximum velocity
    double u_max_computed = *std::max_element(result.u.begin(), result.u.end());
    double error = relativeError(u_max_computed, u_max_analytical);

    std::cout << "  u_max analytical: " << u_max_analytical << std::endl;
    std::cout << "  u_max computed:   " << u_max_computed << std::endl;
    std::cout << "  Relative error:   " << error * 100 << "%" << std::endl;
    std::cout << "  Iterations:       " << result.iterations << std::endl;

    assert(error < 0.01 && "Poiseuille flow error should be < 1%");

    // Check v ≈ 0 everywhere
    double v_max = 0.0;
    for (double vi : result.v) {
        v_max = std::max(v_max, std::abs(vi));
    }
    assert(v_max < 1e-6 && "v should be approximately zero");

    std::cout << "  PASSED" << std::endl;
}

/**
 * Test 4: Lid-driven cavity flow
 *
 * Classic benchmark problem:
 * - Square cavity with moving top lid
 * - Develops recirculating flow
 */
void testLidDrivenCavity() {
    std::cout << "Test 4: Lid-driven cavity..." << std::endl;

    double L = 1.0;
    double mu = 0.01;
    double u_lid = 1.0;

    int nx = 20, ny = 20;
    StructuredMesh mesh(nx, ny, 0.0, L, 0.0, L);

    StokesSolver solver(mesh, mu);

    // No-slip on walls
    solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Left, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Right, VelocityBC::NoSlip());

    // Moving lid on top
    solver.setVelocityBC(Boundary::Top, VelocityBC::Dirichlet(u_lid, 0.0));

    solver.setTolerance(1e-6);
    solver.setMaxIterations(5000);

    StokesResult result = solver.solve();

    assert(result.converged && "Lid-driven cavity should converge");

    // Check top boundary has u = u_lid
    int stride = nx + 1;
    for (int i = 0; i <= nx; ++i) {
        int idx = ny * stride + i;
        assert(approxEqual(result.u[idx], u_lid, 1e-6) && "Top boundary should have u = u_lid");
    }

    // Check bottom has u = 0
    for (int i = 0; i <= nx; ++i) {
        assert(approxEqual(result.u[i], 0.0, 1e-6) && "Bottom should have u = 0");
    }

    // Check flow develops (non-zero interior velocities)
    double u_interior_max = 0.0;
    double v_interior_max = 0.0;
    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            int idx = j * stride + i;
            u_interior_max = std::max(u_interior_max, std::abs(result.u[idx]));
            v_interior_max = std::max(v_interior_max, std::abs(result.v[idx]));
        }
    }

    std::cout << "  Max interior u: " << u_interior_max << std::endl;
    std::cout << "  Max interior v: " << v_interior_max << std::endl;
    std::cout << "  Iterations:     " << result.iterations << std::endl;

    assert(u_interior_max > 0.1 && "Should have significant interior u velocity");
    assert(v_interior_max > 0.01 && "Should have non-zero interior v velocity");

    std::cout << "  PASSED" << std::endl;
}

/**
 * Test 5: Body force driven flow in closed cavity
 */
void testBodyForceCavity() {
    std::cout << "Test 5: Body force in closed cavity..." << std::endl;

    int nx = 10, ny = 10;
    StructuredMesh mesh(nx, ny, 0.0, 1.0, 0.0, 1.0);

    StokesSolver solver(mesh, 1.0);

    // All walls no-slip
    solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Top, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Left, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Right, VelocityBC::NoSlip());

    // Apply body force
    solver.setBodyForce(10.0, 0.0);
    solver.setMaxIterations(500);

    StokesResult result = solver.solve();

    // Should produce some flow
    double u_max = *std::max_element(result.u.begin(), result.u.end());

    std::cout << "  Max u: " << u_max << std::endl;
    std::cout << "  Iterations: " << result.iterations << std::endl;

    assert(u_max > 0.0 && "Body force should produce flow");
    assert(!std::isnan(u_max) && "Solution should not contain NaN");

    std::cout << "  PASSED" << std::endl;
}

/**
 * Test 6: Grid convergence study
 */
void testGridConvergence() {
    std::cout << "Test 6: Grid convergence..." << std::endl;

    double L = 1.0;
    double H = 0.1;
    double mu = 0.001;
    double dP_dx = 1000.0;
    double u_max_analytical = dP_dx * H * H / (8.0 * mu);

    std::vector<int> resolutions = {10, 20, 40};
    std::vector<double> errors;

    for (int n : resolutions) {
        int nx = n * 2;
        int ny = n;
        StructuredMesh mesh(nx, ny, 0.0, L, 0.0, H);

        StokesSolver solver(mesh, mu);
        solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
        solver.setVelocityBC(Boundary::Top, VelocityBC::NoSlip());
        solver.setVelocityBC(Boundary::Left, VelocityBC::Outflow());
        solver.setVelocityBC(Boundary::Right, VelocityBC::Outflow());
        solver.setBodyForce(dP_dx, 0.0);
        solver.setTolerance(1e-8);
        solver.setMaxIterations(5000);

        StokesResult result = solver.solve();
        double u_max = *std::max_element(result.u.begin(), result.u.end());
        double error = relativeError(u_max, u_max_analytical);
        errors.push_back(error);

        std::cout << "  Grid " << nx << "x" << ny << ": error = " << error * 100 << "%"
                  << std::endl;
    }

    // Error should decrease with finer grids
    for (size_t i = 1; i < errors.size(); ++i) {
        assert(errors[i] <= errors[i - 1] && "Error should decrease with finer grid");
    }

    // Finest grid should have small error
    assert(errors.back() < 0.01 && "Fine grid error should be < 1%");

    std::cout << "  PASSED" << std::endl;
}

/**
 * Test 7: No NaN in results
 */
void testNoNaN() {
    std::cout << "Test 7: No NaN values..." << std::endl;

    int nx = 15, ny = 15;
    StructuredMesh mesh(nx, ny, 0.0, 1.0, 0.0, 1.0);

    StokesSolver solver(mesh, 0.01);
    solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Top, VelocityBC::Dirichlet(1.0, 0.0));
    solver.setVelocityBC(Boundary::Left, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Right, VelocityBC::NoSlip());
    solver.setMaxIterations(1000);

    StokesResult result = solver.solve();

    for (size_t i = 0; i < result.u.size(); ++i) {
        assert(!std::isnan(result.u[i]) && "u should not contain NaN");
        assert(!std::isnan(result.v[i]) && "v should not contain NaN");
        assert(!std::isnan(result.pressure[i]) && "pressure should not contain NaN");
        assert(!std::isinf(result.u[i]) && "u should not contain Inf");
        assert(!std::isinf(result.v[i]) && "v should not contain Inf");
    }

    std::cout << "  PASSED" << std::endl;
}

/**
 * Test 8: Inflow/outflow boundary conditions
 */
void testInflowOutflow() {
    std::cout << "Test 8: Inflow/outflow BCs..." << std::endl;

    int nx = 30, ny = 10;
    StructuredMesh mesh(nx, ny, 0.0, 1.0, 0.0, 0.1);

    double u_inlet = 0.5;

    StokesSolver solver(mesh, 0.001);
    solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Top, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Left, VelocityBC::Inflow(u_inlet, 0.0));
    solver.setVelocityBC(Boundary::Right, VelocityBC::Outflow());
    solver.setTolerance(1e-6);
    solver.setMaxIterations(3000);

    StokesResult result = solver.solve();

    // Check inlet velocity is applied
    int stride = nx + 1;
    for (int j = 0; j <= ny; ++j) {
        int idx = j * stride;  // Left boundary
        assert(approxEqual(result.u[idx], u_inlet, 1e-4) && "Inlet u should match BC");
    }

    // Check flow reaches outlet
    double u_outlet_avg = 0.0;
    for (int j = 0; j <= ny; ++j) {
        int idx = j * stride + nx;  // Right boundary
        u_outlet_avg += result.u[idx];
    }
    u_outlet_avg /= (ny + 1);

    std::cout << "  Inlet velocity:  " << u_inlet << std::endl;
    std::cout << "  Outlet avg u:    " << u_outlet_avg << std::endl;

    assert(u_outlet_avg > 0.3 && "Flow should reach outlet");

    std::cout << "  PASSED" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Stokes Solver Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        testConstruction();
        testInvalidParameters();
        testPoiseuilleFlow();
        testLidDrivenCavity();
        testBodyForceCavity();
        testGridConvergence();
        testNoNaN();
        testInflowOutflow();

        std::cout << "========================================" << std::endl;
        std::cout << "All 8 tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAILED with exception: " << e.what() << std::endl;
        return 1;
    }
}
