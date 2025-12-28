"""Tests for ADI (Alternating Direction Implicit) solvers."""

import unittest

import numpy as np

import biotransport as bt


class TestADIDiffusion2D(unittest.TestCase):
    """Tests for 2D ADI diffusion solver."""

    def test_construction(self):
        """Test ADI solver construction."""
        mesh = bt.StructuredMesh(20, 20, 0.0, 1.0, 0.0, 1.0)
        D = 0.01
        solver = bt.ADIDiffusion2D(mesh, D)
        self.assertIsNotNone(solver)

    def test_initial_condition(self):
        """Test setting initial condition."""
        mesh = bt.StructuredMesh(10, 10, 0.0, 1.0, 0.0, 1.0)
        solver = bt.ADIDiffusion2D(mesh, 0.01)

        # Gaussian IC
        x, y = bt.xy_grid(mesh)
        ic = np.exp(-50 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
        solver.set_initial_condition(ic.flatten().tolist())

        solution = np.array(solver.solution())
        self.assertEqual(len(solution), mesh.num_nodes())
        np.testing.assert_array_almost_equal(solution, ic.flatten())

    def test_dirichlet_boundaries(self):
        """Test Dirichlet boundary conditions."""
        mesh = bt.StructuredMesh(10, 10, 0.0, 1.0, 0.0, 1.0)
        solver = bt.ADIDiffusion2D(mesh, 0.01)

        # Set all boundaries to zero
        solver.set_dirichlet_boundary(bt.Boundary.Left, 0.0)
        solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)
        solver.set_dirichlet_boundary(bt.Boundary.Bottom, 0.0)
        solver.set_dirichlet_boundary(bt.Boundary.Top, 0.0)

        # Initial hot spot in center
        x, y = bt.xy_grid(mesh)
        ic = np.exp(-50 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
        solver.set_initial_condition(ic.flatten().tolist())

        # Solve
        result = solver.solve(0.01, 10)
        self.assertTrue(result.success)
        self.assertEqual(result.steps, 10)

    def test_large_timestep_stability(self):
        """ADI should be stable for large time steps."""
        mesh = bt.StructuredMesh(20, 20, 0.0, 1.0, 0.0, 1.0)
        D = 0.01
        dx = 1.0 / 20

        # Explicit stability limit
        dt_explicit_max = dx**2 / (4 * D)

        # ADI can use much larger timesteps
        dt_adi = 10 * dt_explicit_max  # 10x larger than explicit limit

        solver = bt.ADIDiffusion2D(mesh, D)
        x, y = bt.xy_grid(mesh)
        ic = np.exp(-50 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
        solver.set_initial_condition(ic.flatten().tolist())

        # Should not explode
        result = solver.solve(dt_adi, 5)
        solution = np.array(solver.solution())

        self.assertTrue(result.success)
        self.assertFalse(np.any(np.isnan(solution)))
        self.assertFalse(np.any(np.isinf(solution)))
        # Solution should stay bounded
        self.assertLess(np.max(np.abs(solution)), 10.0)

    def test_convergence_to_steady_state(self):
        """Test ADI converges to steady state."""
        mesh = bt.StructuredMesh(20, 20, 0.0, 1.0, 0.0, 1.0)
        solver = bt.ADIDiffusion2D(mesh, 0.1)

        # Dirichlet: left=1, right=0
        solver.set_dirichlet_boundary(bt.Boundary.Left, 1.0)
        solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)
        solver.set_dirichlet_boundary(bt.Boundary.Bottom, 0.5)
        solver.set_dirichlet_boundary(bt.Boundary.Top, 0.5)

        # Start from uniform
        ic = 0.5 * np.ones(mesh.num_nodes())
        solver.set_initial_condition(ic.tolist())

        # Run to steady state
        solver.solve(0.01, 500)
        solution = np.array(solver.solution()).reshape(21, 21)

        # Check left boundary ≈ 1, right boundary ≈ 0
        self.assertAlmostEqual(solution[10, 0], 1.0, places=1)
        self.assertAlmostEqual(solution[10, -1], 0.0, places=1)


class TestADIDiffusion3D(unittest.TestCase):
    """Tests for 3D ADI diffusion solver."""

    def test_construction(self):
        """Test 3D ADI solver construction."""
        mesh = bt.StructuredMesh3D(10, 10, 10, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        solver = bt.ADIDiffusion3D(mesh, 0.01)
        self.assertIsNotNone(solver)

    def test_solve_3d(self):
        """Test 3D ADI solver runs without error."""
        mesh = bt.StructuredMesh3D(8, 8, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        solver = bt.ADIDiffusion3D(mesh, 0.01)

        # Simple IC
        ic = np.zeros(mesh.num_nodes())
        center_idx = mesh.num_nodes() // 2
        ic[center_idx] = 1.0
        solver.set_initial_condition(ic.tolist())

        # Solve
        result = solver.solve(0.001, 5)
        solution = np.array(solver.solution())

        self.assertTrue(result.success)
        self.assertFalse(np.any(np.isnan(solution)))


class TestADISolveResult(unittest.TestCase):
    """Tests for ADISolveResult struct."""

    def test_result_attributes(self):
        """Test result has expected attributes."""
        mesh = bt.StructuredMesh(10, 10, 0.0, 1.0, 0.0, 1.0)
        solver = bt.ADIDiffusion2D(mesh, 0.01)
        ic = np.ones(mesh.num_nodes())
        solver.set_initial_condition(ic.tolist())

        result = solver.solve(0.01, 5)

        self.assertTrue(hasattr(result, "success"))
        self.assertTrue(hasattr(result, "steps"))
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.steps, int)


if __name__ == "__main__":
    unittest.main()
