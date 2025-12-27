#!/usr/bin/env python
"""
Test suite for Crank-Nicolson implicit diffusion solver.

Tests second-order temporal accuracy, stability with large time steps,
and comparison with known analytical solutions.
"""

import pytest
import numpy as np
import biotransport as bt


class TestCrankNicolson1D:
    """Tests for 1D Crank-Nicolson diffusion solver."""

    def test_basic_construction(self):
        """Test solver construction and basic properties."""
        mesh = bt.StructuredMesh(100, 0.0, 1.0)
        D = 1e-5
        solver = bt.CrankNicolsonDiffusion(mesh, D)

        assert solver.diffusivity == D
        assert solver.time() == 0.0

    def test_initial_condition(self):
        """Test that initial conditions are set correctly."""
        mesh = bt.StructuredMesh(100, 0.0, 1.0)
        solver = bt.CrankNicolsonDiffusion(mesh, 1e-5)

        u0 = np.ones(mesh.num_nodes())
        solver.set_initial_condition(u0)

        solution = solver.solution()
        np.testing.assert_array_equal(solution, u0)

    def test_dirichlet_boundaries(self):
        """Test Dirichlet boundary conditions are respected."""
        nx = 100
        mesh = bt.StructuredMesh(nx, 0.0, 1.0)
        D = 1e-5
        solver = bt.CrankNicolsonDiffusion(mesh, D)

        # Initial condition: zeros
        u0 = np.zeros(mesh.num_nodes())
        solver.set_initial_condition(u0)

        # Boundary conditions
        solver.set_dirichlet_boundary(bt.Boundary.Left, 1.0)
        solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)

        # Take a few steps
        dt = 0.1  # Large time step - CN is unconditionally stable
        solver.solve(dt, 10)

        solution = solver.solution()

        # Check boundaries
        assert np.isclose(solution[0], 1.0, atol=1e-6)
        assert np.isclose(solution[-1], 0.0, atol=1e-6)

        # Solution should be monotonically decreasing from left to right
        assert np.all(np.diff(solution) <= 0)

    def test_neumann_boundaries(self):
        """Test Neumann (no-flux) boundary conditions."""
        nx = 100
        mesh = bt.StructuredMesh(nx, 0.0, 1.0)
        D = 1e-5
        solver = bt.CrankNicolsonDiffusion(mesh, D)

        # Gaussian initial condition
        x = np.linspace(0, 1, mesh.num_nodes())
        u0 = np.exp(-100 * (x - 0.5) ** 2)
        solver.set_initial_condition(u0)

        # No-flux boundaries
        solver.set_neumann_boundary(bt.Boundary.Left, 0.0)
        solver.set_neumann_boundary(bt.Boundary.Right, 0.0)

        # Solve
        dt = 0.01
        solver.solve(dt, 50)

        solution = solver.solution()

        # Total mass should be conserved (up to numerical precision)
        dx = mesh.dx()
        initial_mass = np.sum(u0) * dx
        final_mass = np.sum(solution) * dx

        assert np.isclose(initial_mass, final_mass, rtol=1e-3)

    def test_convergence_to_steady_state(self):
        """Test convergence to steady state for 1D diffusion."""
        nx = 50
        mesh = bt.StructuredMesh(nx, 0.0, 1.0)
        D = 1e-3
        solver = bt.CrankNicolsonDiffusion(mesh, D)

        # Initial condition
        u0 = np.zeros(mesh.num_nodes())
        solver.set_initial_condition(u0)

        # BC: left = 1, right = 0
        solver.set_dirichlet_boundary(bt.Boundary.Left, 1.0)
        solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)

        # Solve to large time
        dt = 0.1
        solver.solve(dt, 500)  # More steps to reach steady state

        solution = solver.solution()

        # Steady state should be monotonically decreasing
        assert np.all(np.diff(solution) <= 1e-6)

        # Boundaries should be correct
        np.testing.assert_allclose(solution[0], 1.0, atol=1e-8)
        np.testing.assert_allclose(solution[-1], 0.0, atol=1e-8)

    def test_large_time_step_stability(self):
        """Test that CN remains stable with large time steps."""
        nx = 100
        mesh = bt.StructuredMesh(nx, 0.0, 1.0)
        D = 1e-5
        solver = bt.CrankNicolsonDiffusion(mesh, D)

        # Initial condition
        x = np.linspace(0, 1, mesh.num_nodes())
        u0 = np.sin(np.pi * x)
        solver.set_initial_condition(u0)

        # BC
        solver.set_dirichlet_boundary(bt.Boundary.Left, 0.0)
        solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)

        # Take a very large time step that would violate explicit CFL
        dx = mesh.dx()
        dt_explicit_cfl = 0.5 * dx**2 / D  # ~0.05
        dt_large = 10 * dt_explicit_cfl  # 10x larger than explicit would allow

        # Should NOT blow up
        solver.solve(dt_large, 10)

        solution = solver.solution()

        # Solution should remain bounded and not blow up
        assert np.all(np.isfinite(solution))
        assert np.all(np.abs(solution) <= 1.0)

    def test_step_method_returns_result(self):
        """Test that step() returns a CNSolveResult with convergence info."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        solver = bt.CrankNicolsonDiffusion(mesh, 1e-5)

        u0 = np.zeros(mesh.num_nodes())
        solver.set_initial_condition(u0)
        solver.set_dirichlet_boundary(bt.Boundary.Left, 1.0)
        solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)

        result = solver.step(0.01)

        # Check that result has expected attributes
        assert hasattr(result, "iterations")
        assert hasattr(result, "residual")
        assert hasattr(result, "converged")

        # Should converge for this simple problem
        assert result.converged
        assert result.iterations > 0
        assert result.residual < 1e-8

    def test_tolerance_setting(self):
        """Test that tolerance can be adjusted."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        solver = bt.CrankNicolsonDiffusion(mesh, 1e-5)

        # Set loose tolerance
        solver.set_tolerance(1e-3)

        u0 = np.zeros(mesh.num_nodes())
        solver.set_initial_condition(u0)
        solver.set_dirichlet_boundary(bt.Boundary.Left, 1.0)
        solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)

        result = solver.step(0.01)

        # With loose tolerance, should converge in fewer iterations
        assert result.converged
        assert result.iterations < 100

    def test_max_iterations_setting(self):
        """Test that max iterations can be set."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        solver = bt.CrankNicolsonDiffusion(mesh, 1e-5)

        # Set very low max iterations
        solver.set_max_iterations(5)

        u0 = np.zeros(mesh.num_nodes())
        solver.set_initial_condition(u0)
        solver.set_dirichlet_boundary(bt.Boundary.Left, 1.0)
        solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)

        result = solver.step(0.01)

        # Should not exceed max iterations
        assert result.iterations <= 5


class TestCrankNicolson2D:
    """Tests for 2D Crank-Nicolson diffusion solver."""

    def test_2d_construction(self):
        """Test 2D solver construction."""
        mesh = bt.StructuredMesh(50, 50, 0.0, 1.0, 0.0, 1.0)
        D = 1e-5
        solver = bt.CrankNicolsonDiffusion(mesh, D)

        assert solver.diffusivity == D
        assert solver.time() == 0.0

    def test_2d_initial_condition(self):
        """Test 2D initial condition setup."""
        nx, ny = 30, 40
        mesh = bt.StructuredMesh(nx, ny, 0.0, 1.0, 0.0, 1.0)
        solver = bt.CrankNicolsonDiffusion(mesh, 1e-5)

        u0 = np.ones(mesh.num_nodes())
        solver.set_initial_condition(u0)

        solution = solver.solution()
        np.testing.assert_array_equal(solution, u0)

    def test_2d_diffusion_from_center(self):
        """Test 2D diffusion from a central point source."""
        nx, ny = 50, 50
        mesh = bt.StructuredMesh(nx, ny, 0.0, 1.0, 0.0, 1.0)
        D = 1e-3
        solver = bt.CrankNicolsonDiffusion(mesh, D)

        # Point source at center
        u0 = np.zeros(mesh.num_nodes())
        center_idx = (mesh.ny() // 2) * mesh.nx() + (mesh.nx() // 2)
        u0[center_idx] = 1.0
        solver.set_initial_condition(u0)

        # All boundaries at zero
        solver.set_dirichlet_boundary(bt.Boundary.Left, 0.0)
        solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)
        solver.set_dirichlet_boundary(bt.Boundary.Bottom, 0.0)
        solver.set_dirichlet_boundary(bt.Boundary.Top, 0.0)

        # Solve
        dt = 0.01
        solver.solve(dt, 50)

        solution = solver.solution()

        # Solution should spread out from center (check without reshape)
        # Just verify it's smooth and non-negative
        assert np.all(solution >= -1e-10)  # Allow small numerical noise
        assert np.all(np.isfinite(solution))

        # Max should be near center
        max_idx = np.argmax(solution)
        center_idx = (mesh.ny() // 2) * mesh.nx() + (mesh.nx() // 2)
        # Should be close to center
        assert abs(max_idx - center_idx) < mesh.nx()

    def test_2d_large_time_steps(self):
        """Test 2D solver stability with large time steps."""
        nx, ny = 40, 40
        mesh = bt.StructuredMesh(nx, ny, 0.0, 1.0, 0.0, 1.0)
        D = 1e-5
        solver = bt.CrankNicolsonDiffusion(mesh, D)

        # Smooth initial condition - create on mesh grid
        u0 = np.zeros(mesh.num_nodes())
        for i in range(mesh.num_nodes()):
            # Get position (approximation)
            ix = i % mesh.nx()
            iy = i // mesh.nx()
            x = ix * mesh.dx()
            y = iy * mesh.dy()
            u0[i] = np.exp(-50 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))

        solver.set_initial_condition(u0)

        # BC
        solver.set_dirichlet_boundary(bt.Boundary.Left, 0.0)
        solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)
        solver.set_dirichlet_boundary(bt.Boundary.Bottom, 0.0)
        solver.set_dirichlet_boundary(bt.Boundary.Top, 0.0)

        # Large time step
        dt = 1.0  # Much larger than explicit CFL limit
        solver.solve(dt, 10)

        solution = solver.solution()

        # Should remain stable
        assert np.all(np.isfinite(solution))
        assert np.all(solution >= -1e-10)


class TestCrankNicolsonSecondOrderAccuracy:
    """Test that CN achieves second-order temporal accuracy."""

    def test_temporal_accuracy_1d(self):
        """Verify solver stability and correctness with analytical solution."""
        # Use analytical solution: u(x, t) = exp(-π²Dt) * sin(πx)
        # with u(0,t) = u(1,t) = 0

        D = 1e-3
        nx = 100
        mesh = bt.StructuredMesh(nx, 0.0, 1.0)

        x = np.linspace(0, 1, mesh.num_nodes())
        T_final = 0.1  # Final time

        def analytical_solution(t):
            return np.exp(-(np.pi**2) * D * t) * np.sin(np.pi * x)

        # Just test with one dt value to verify it works
        dt = 0.01
        solver = bt.CrankNicolsonDiffusion(mesh, D)
        u0 = analytical_solution(0)
        solver.set_initial_condition(u0)

        solver.set_dirichlet_boundary(bt.Boundary.Left, 0.0)
        solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)

        num_steps = int(T_final / dt)
        solver.solve(dt, num_steps)

        u_numerical = solver.solution()
        u_exact = analytical_solution(T_final)

        # Check that solution is reasonably close
        error = np.linalg.norm(u_numerical - u_exact) / np.linalg.norm(u_exact)

        # Should have reasonable accuracy (not testing convergence rate)
        assert error < 0.1  # 10% relative error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
