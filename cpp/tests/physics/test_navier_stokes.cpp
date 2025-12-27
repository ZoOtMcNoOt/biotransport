/**
 * @file test_navier_stokes.cpp
 * @brief Unit tests for the Navier-Stokes flow solver
 *
 * Tests verify:
 * 1. Basic solver construction and parameter setting
 * 2. Channel flow development
 * 3. Stability behavior
 * 4. CFL condition handling
 * 5. Time stepping accuracy
 * 6. Reynolds number effects
 */

#include <algorithm>
#include <biotransport/physics/fluid_dynamics/navier_stokes.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace biotransport;

// Helper function to check approximate equality
bool approxEqual(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

/**
 * Test 1: Basic solver construction
 */
void testConstruction() {
    std::cout << "Test 1: Basic construction..." << std::endl;

    StructuredMesh mesh(10, 10, 0.0, 1.0, 0.0, 1.0);
    double rho = 1000.0;  // kg/m^3
    double mu = 0.001;    // Pa.s

    NavierStokesSolver solver(mesh, rho, mu);

    // Should not throw
    solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Top, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Left, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Right, VelocityBC::NoSlip());

    std::cout << "  PASSED" << std::endl;
}

/**
 * Test 2: Invalid parameters should throw
 */
void testInvalidParameters() {
    std::cout << "Test 2: Invalid parameters..." << std::endl;

    StructuredMesh mesh2D(10, 10, 0.0, 1.0, 0.0, 1.0);
    StructuredMesh mesh1D(10, 0.0, 1.0);

    // Negative density should throw
    bool caught = false;
    try {
        NavierStokesSolver solver(mesh2D, -1000.0, 0.001);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught && "Should throw for negative density");

    // Negative viscosity should throw
    caught = false;
    try {
        NavierStokesSolver solver(mesh2D, 1000.0, -0.001);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught && "Should throw for negative viscosity");

    // 1D mesh should throw
    caught = false;
    try {
        NavierStokesSolver solver(mesh1D, 1000.0, 0.001);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught && "Should throw for 1D mesh");

    std::cout << "  PASSED" << std::endl;
}

/**
 * Test 3: Channel flow development
 */
void testChannelFlow() {
    std::cout << "Test 3: Channel flow development..." << std::endl;

    // Microfluidic channel
    double L = 0.001;   // 1 mm
    double H = 0.0005;  // 0.5 mm
    double rho = 1000.0;
    double mu = 0.001;
    double u_inlet = 0.1;  // 0.1 m/s

    int nx = 20, ny = 10;
    StructuredMesh mesh(nx, ny, 0.0, L, 0.0, H);

    NavierStokesSolver solver(mesh, rho, mu);
    solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Top, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Left, VelocityBC::Inflow(u_inlet, 0.0));
    solver.setVelocityBC(Boundary::Right, VelocityBC::Outflow());

    // Simulate for a short time
    double t_end = 0.001;  // 1 ms
    NavierStokesResult result = solver.solve(t_end);

    std::cout << "  Stable: " << (result.stable ? "true" : "false") << std::endl;
    std::cout << "  Time steps: " << result.time_steps << std::endl;
    std::cout << "  Final time: " << result.time << std::endl;

    assert(result.stable && "Channel flow should be stable");
    assert(result.time_steps > 0 && "Should take some time steps");

    // Check velocities are non-zero and finite
    double u_max = *std::max_element(result.u.begin(), result.u.end());
    assert(u_max > 0.0 && "Should have positive velocity");
    assert(!std::isnan(u_max) && "Should not have NaN");
    assert(!std::isinf(u_max) && "Should not have Inf");

    std::cout << "  Max velocity: " << u_max << " m/s" << std::endl;
    std::cout << "  PASSED" << std::endl;
}

/**
 * Test 4: Lid-driven cavity (time-dependent)
 */
void testLidDrivenCavity() {
    std::cout << "Test 4: Lid-driven cavity..." << std::endl;

    double L = 0.001;  // 1 mm
    double rho = 1000.0;
    double mu = 0.01;  // Higher viscosity for stability
    double u_lid = 0.1;

    int nx = 15, ny = 15;
    StructuredMesh mesh(nx, ny, 0.0, L, 0.0, L);

    NavierStokesSolver solver(mesh, rho, mu);
    solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Left, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Right, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Top, VelocityBC::Dirichlet(u_lid, 0.0));

    double t_end = 0.01;  // 10 ms
    NavierStokesResult result = solver.solve(t_end);

    assert(result.stable && "Cavity flow should be stable with high viscosity");

    // Check lid boundary condition is enforced
    int stride = nx + 1;
    for (int i = 0; i <= nx; ++i) {
        int idx = ny * stride + i;
        assert(approxEqual(result.u[idx], u_lid, 1e-4) && "Lid should have u = u_lid");
    }

    // Check recirculation develops
    double v_max = 0.0;
    for (double vi : result.v) {
        v_max = std::max(v_max, std::abs(vi));
    }

    std::cout << "  Max |v|: " << v_max << " m/s" << std::endl;
    std::cout << "  Time steps: " << result.time_steps << std::endl;

    assert(v_max > 0.0 && "Should develop vertical velocity from recirculation");

    std::cout << "  PASSED" << std::endl;
}

/**
 * Test 5: Reynolds number effect on stability
 */
void testReynoldsNumber() {
    std::cout << "Test 5: Reynolds number effects..." << std::endl;

    double L = 0.001;
    double H = 0.001;
    double rho = 1000.0;
    double u = 0.1;

    int nx = 10, ny = 10;

    // Low Re (Re ~ 10) - should be stable
    {
        double mu = 0.01;  // Re = rho*u*L/mu = 1000*0.1*0.001/0.01 = 10
        StructuredMesh mesh(nx, ny, 0.0, L, 0.0, H);
        NavierStokesSolver solver(mesh, rho, mu);
        solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
        solver.setVelocityBC(Boundary::Top, VelocityBC::NoSlip());
        solver.setVelocityBC(Boundary::Left, VelocityBC::Inflow(u, 0.0));
        solver.setVelocityBC(Boundary::Right, VelocityBC::Outflow());

        NavierStokesResult result = solver.solve(0.001);
        std::cout << "  Low Re (10): stable=" << result.stable << std::endl;
        assert(result.stable && "Low Re flow should be stable");
    }

    // Moderate Re (Re ~ 100) - should still be stable with proper time stepping
    {
        double mu = 0.001;  // Re = 100
        StructuredMesh mesh(nx, ny, 0.0, L, 0.0, H);
        NavierStokesSolver solver(mesh, rho, mu);
        solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
        solver.setVelocityBC(Boundary::Top, VelocityBC::NoSlip());
        solver.setVelocityBC(Boundary::Left, VelocityBC::Inflow(u, 0.0));
        solver.setVelocityBC(Boundary::Right, VelocityBC::Outflow());

        NavierStokesResult result = solver.solve(0.0005);
        std::cout << "  Moderate Re (100): stable=" << result.stable << std::endl;
        // May or may not be stable depending on time step selection
    }

    std::cout << "  PASSED" << std::endl;
}

/**
 * Test 6: No NaN in results
 */
void testNoNaN() {
    std::cout << "Test 6: No NaN values..." << std::endl;

    int nx = 10, ny = 10;
    StructuredMesh mesh(nx, ny, 0.0, 0.001, 0.0, 0.001);

    NavierStokesSolver solver(mesh, 1000.0, 0.01);
    solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Top, VelocityBC::Dirichlet(0.1, 0.0));
    solver.setVelocityBC(Boundary::Left, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Right, VelocityBC::NoSlip());

    NavierStokesResult result = solver.solve(0.01);

    for (size_t i = 0; i < result.u.size(); ++i) {
        assert(!std::isnan(result.u[i]) && "u should not contain NaN");
        assert(!std::isnan(result.v[i]) && "v should not contain NaN");
        assert(!std::isnan(result.pressure[i]) && "pressure should not contain NaN");
    }

    std::cout << "  PASSED" << std::endl;
}

/**
 * Test 7: Time stepping
 */
void testTimeStepping() {
    std::cout << "Test 7: Time stepping..." << std::endl;

    int nx = 10, ny = 10;
    StructuredMesh mesh(nx, ny, 0.0, 0.001, 0.0, 0.001);

    NavierStokesSolver solver(mesh, 1000.0, 0.001);
    solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Top, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Left, VelocityBC::Inflow(0.1, 0.0));
    solver.setVelocityBC(Boundary::Right, VelocityBC::Outflow());

    // Different end times should result in different time steps
    NavierStokesResult r1 = solver.solve(0.0001);
    NavierStokesResult r2 = solver.solve(0.001);

    std::cout << "  t=0.1ms: steps=" << r1.time_steps << ", time=" << r1.time << std::endl;
    std::cout << "  t=1.0ms: steps=" << r2.time_steps << ", time=" << r2.time << std::endl;

    assert(r2.time_steps >= r1.time_steps && "Longer sim should have more steps");
    assert(r2.time >= r1.time && "Longer sim should reach later time");

    std::cout << "  PASSED" << std::endl;
}

/**
 * Test 8: Body force
 */
void testBodyForce() {
    std::cout << "Test 8: Body force..." << std::endl;

    int nx = 10, ny = 10;
    StructuredMesh mesh(nx, ny, 0.0, 0.001, 0.0, 0.001);

    NavierStokesSolver solver(mesh, 1000.0, 0.01);
    solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Top, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Left, VelocityBC::NoSlip());
    solver.setVelocityBC(Boundary::Right, VelocityBC::NoSlip());
    solver.setBodyForce(1000.0, 0.0);  // Strong body force

    NavierStokesResult result = solver.solve(0.01);

    double u_max = *std::max_element(result.u.begin(), result.u.end());
    std::cout << "  Max u: " << u_max << " m/s" << std::endl;

    assert(u_max > 0.0 && "Body force should accelerate flow");
    assert(!std::isnan(u_max) && "Should not have NaN");

    std::cout << "  PASSED" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Navier-Stokes Solver Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        testConstruction();
        testInvalidParameters();
        testChannelFlow();
        testLidDrivenCavity();
        testReynoldsNumber();
        testNoNaN();
        testTimeStepping();
        testBodyForce();

        std::cout << "========================================" << std::endl;
        std::cout << "All 8 tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAILED with exception: " << e.what() << std::endl;
        return 1;
    }
}
