"""Tests for ADI (Alternating Direction Implicit) solvers."""

import unittest

import numpy as np

import biotransport as bt


def trapezoidal_mass_2d(mesh, values):
    field = np.asarray(values).reshape(mesh.ny() + 1, mesh.nx() + 1)
    weights = np.ones_like(field)
    weights[:, (0, -1)] *= 0.5
    weights[(0, -1), :] *= 0.5
    return np.sum(weights * field) * mesh.dx() * mesh.dy()


def trapezoidal_mass_3d(mesh, values):
    field = np.asarray(values).reshape(mesh.nz() + 1, mesh.ny() + 1, mesh.nx() + 1)
    weights = np.ones_like(field)
    weights[:, :, (0, -1)] *= 0.5
    weights[:, (0, -1), :] *= 0.5
    weights[(0, -1), :, :] *= 0.5
    return np.sum(weights * field) * mesh.dx() * mesh.dy() * mesh.dz()


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
        self.assertEqual(result.substeps, 30)
        self.assertAlmostEqual(result.total_time, 0.1)

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
        # A-stability bounds the linear solve, but CN-type sweeps are not
        # monotonicity preserving for arbitrarily large time steps.
        self.assertLess(np.max(np.abs(solution)), 10.0)

    def test_homogeneous_neumann_conserves_control_volume_mass(self):
        mesh = bt.StructuredMesh(15, 11, -0.1, 1.2, 0.0, 0.9)
        solver = bt.ADIDiffusion2D(mesh, 0.15)
        x, y = bt.xy_grid(mesh)
        initial = 1 + 0.2 * np.sin(2.1 * x) + 0.1 * np.cos(1.3 * y)
        solver.set_initial_condition(initial.ravel())
        for face in (
            bt.Boundary.Left,
            bt.Boundary.Right,
            bt.Boundary.Bottom,
            bt.Boundary.Top,
        ):
            solver.set_neumann_boundary(face, 0.0)
        initial_mass = trapezoidal_mass_2d(mesh, initial)
        solver.solve(0.11, 9)
        self.assertAlmostEqual(
            trapezoidal_mass_2d(mesh, solver.solution()), initial_mass, places=12
        )

    def test_outward_neumann_derivative_preserves_linear_field(self):
        mesh = bt.StructuredMesh(15, 11, -0.1, 1.2, 0.0, 0.9)
        solver = bt.ADIDiffusion2D(mesh, 0.15)
        x, y = bt.xy_grid(mesh)
        exact = 2 + 0.4 * x - 0.7 * y
        solver.set_initial_condition(exact.ravel())
        solver.set_neumann_boundary(bt.Boundary.Left, -0.4)
        solver.set_neumann_boundary(bt.Boundary.Right, 0.4)
        solver.set_neumann_boundary(bt.Boundary.Bottom, 0.7)
        solver.set_neumann_boundary(bt.Boundary.Top, -0.7)
        solver.solve(0.17, 6)
        np.testing.assert_allclose(
            solver.solution(), exact.ravel(), rtol=0.0, atol=3e-13
        )

    def test_second_order_temporal_convergence(self):
        mesh = bt.StructuredMesh(32, 28, 0.0, 1.0, 0.0, 1.0)
        x, y = bt.xy_grid(mesh)
        initial = np.sin(np.pi * x) * np.sin(np.pi * y)

        def run(steps):
            solver = bt.ADIDiffusion2D(mesh, 0.2)
            solver.set_initial_condition(initial.ravel())
            solver.solve(0.08 / steps, steps)
            return np.asarray(solver.solution())

        reference = run(256)
        coarse_error = np.max(np.abs(run(4) - reference))
        medium_error = np.max(np.abs(run(8) - reference))
        self.assertGreater(coarse_error / medium_error, 3.8)

    def test_convergence_to_steady_state(self):
        """Test ADI converges to steady state."""
        mesh = bt.StructuredMesh(20, 20, 0.0, 1.0, 0.0, 1.0)
        solver = bt.ADIDiffusion2D(mesh, 0.1)

        # Dirichlet pressure-like trace in x and insulated boundaries in y.
        # Four different Dirichlet face constants would prescribe two values
        # at each corner and are therefore intentionally rejected.
        solver.set_dirichlet_boundary(bt.Boundary.Left, 1.0)
        solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)
        solver.set_neumann_boundary(bt.Boundary.Bottom, 0.0)
        solver.set_neumann_boundary(bt.Boundary.Top, 0.0)

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
        self.assertEqual(result.substeps, 25)
        self.assertFalse(np.any(np.isnan(solution)))

    def test_homogeneous_neumann_conserves_control_volume_mass(self):
        mesh = bt.StructuredMesh3D(7, 5, 4, 0.0, 1.0, -0.2, 0.7, 0.0, 1.3)
        solver = bt.ADIDiffusion3D(mesh, 0.12)
        initial = np.empty(mesh.num_nodes())
        for k in range(mesh.nz() + 1):
            for j in range(mesh.ny() + 1):
                for i in range(mesh.nx() + 1):
                    initial[mesh.index(i, j, k)] = (
                        1 + 0.2 * np.sin(mesh.x(i)) + 0.1 * np.cos(mesh.y(j))
                    )
        solver.set_initial_condition(initial)
        for face in (
            bt.Boundary3D.XMin,
            bt.Boundary3D.XMax,
            bt.Boundary3D.YMin,
            bt.Boundary3D.YMax,
            bt.Boundary3D.ZMin,
            bt.Boundary3D.ZMax,
        ):
            solver.set_neumann_boundary(face, 0.0)
        initial_mass = trapezoidal_mass_3d(mesh, initial)
        solver.solve(0.09, 7)
        self.assertAlmostEqual(
            trapezoidal_mass_3d(mesh, solver.solution()), initial_mass, places=12
        )


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

    def test_invalid_inputs_fail_before_mutating_state(self):
        mesh = bt.StructuredMesh(6, 5, 0.0, 1.0, 0.0, 1.0)
        solver = bt.ADIDiffusion2D(mesh, 0.1)
        initial = np.ones(mesh.num_nodes())
        solver.set_initial_condition(initial)
        with self.assertRaises(ValueError):
            solver.step(np.inf)
        with self.assertRaises(ValueError):
            solver.solve(0.1, -1)
        np.testing.assert_array_equal(solver.solution(), initial)
        self.assertEqual(solver.time(), 0.0)


if __name__ == "__main__":
    unittest.main()
