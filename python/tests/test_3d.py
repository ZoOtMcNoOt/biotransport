"""Tests for 3D mesh and solver functionality."""

import unittest

import numpy as np

import biotransport as bt


class TestStructuredMesh3D(unittest.TestCase):
    """Tests for 3D structured mesh."""

    def test_construction(self):
        """Test 3D mesh construction."""
        mesh = bt.StructuredMesh3D(10, 10, 10, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        self.assertIsNotNone(mesh)

    def test_dimensions(self):
        """Test mesh dimension accessors."""
        nx, ny, nz = 10, 15, 20
        mesh = bt.StructuredMesh3D(nx, ny, nz, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0)

        self.assertEqual(mesh.nx(), nx)
        self.assertEqual(mesh.ny(), ny)
        self.assertEqual(mesh.nz(), nz)
        self.assertEqual(mesh.num_nodes(), (nx + 1) * (ny + 1) * (nz + 1))

    def test_spacing(self):
        """Test grid spacing calculations."""
        mesh = bt.StructuredMesh3D(10, 20, 30, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0)

        self.assertAlmostEqual(mesh.dx(), 0.1, places=10)
        self.assertAlmostEqual(mesh.dy(), 0.1, places=10)
        self.assertAlmostEqual(mesh.dz(), 0.1, places=10)

    def test_coordinate_access(self):
        """Test coordinate accessor methods."""
        mesh = bt.StructuredMesh3D(5, 5, 5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

        # Check corners
        self.assertAlmostEqual(mesh.x(0), 0.0)
        self.assertAlmostEqual(mesh.x(5), 1.0)
        self.assertAlmostEqual(mesh.y(0), 0.0)
        self.assertAlmostEqual(mesh.y(5), 1.0)
        self.assertAlmostEqual(mesh.z(0), 0.0)
        self.assertAlmostEqual(mesh.z(5), 1.0)


class TestDiffusionSolver3D(unittest.TestCase):
    """Tests for 3D diffusion solver."""

    def test_construction(self):
        """Test 3D solver construction."""
        mesh = bt.StructuredMesh3D(5, 5, 5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        solver = bt.DiffusionSolver3D(mesh, 0.01)
        self.assertIsNotNone(solver)

    def test_initial_condition(self):
        """Test setting initial condition."""
        mesh = bt.StructuredMesh3D(5, 5, 5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        solver = bt.DiffusionSolver3D(mesh, 0.01)

        ic = np.ones(mesh.num_nodes())
        solver.set_initial_condition(ic.tolist())

        solution = np.array(solver.solution())
        np.testing.assert_array_almost_equal(solution, ic)

    def test_boundary_conditions(self):
        """Test setting 3D boundary conditions."""
        mesh = bt.StructuredMesh3D(5, 5, 5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        solver = bt.DiffusionSolver3D(mesh, 0.01)

        # Set all 6 faces
        solver.set_dirichlet_boundary(bt.Boundary3D.XMin, 1.0)
        solver.set_dirichlet_boundary(bt.Boundary3D.XMax, 0.0)
        solver.set_dirichlet_boundary(bt.Boundary3D.YMin, 0.5)
        solver.set_dirichlet_boundary(bt.Boundary3D.YMax, 0.5)
        solver.set_dirichlet_boundary(bt.Boundary3D.ZMin, 0.5)
        solver.set_dirichlet_boundary(bt.Boundary3D.ZMax, 0.5)

        ic = 0.5 * np.ones(mesh.num_nodes())
        solver.set_initial_condition(ic.tolist())

        # Should run without error
        solver.solve(0.0001, 10)
        solution = np.array(solver.solution())
        self.assertFalse(np.any(np.isnan(solution)))

    def test_diffusion_from_hot_center(self):
        """Test heat diffuses from hot center."""
        mesh = bt.StructuredMesh3D(10, 10, 10, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        solver = bt.DiffusionSolver3D(mesh, 0.01)

        # Hot spot at center
        ic = np.zeros(mesh.num_nodes())
        # Find center index (approximately)
        nx, ny, nz = 11, 11, 11  # num_nodes per dimension
        center_i, center_j, center_k = nx // 2, ny // 2, nz // 2
        center_idx = center_i + center_j * nx + center_k * nx * ny
        ic[center_idx] = 100.0

        solver.set_initial_condition(ic.tolist())

        # Solve
        solver.solve(0.0001, 50)
        solution = np.array(solver.solution())

        # Peak should decrease (heat spreads)
        self.assertLess(np.max(solution), 100.0)
        # Should still be bounded
        self.assertFalse(np.any(np.isnan(solution)))

    def test_stability(self):
        """Test explicit solver respects CFL condition."""
        mesh = bt.StructuredMesh3D(10, 10, 10, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        D = 0.01
        dx = 0.1

        # CFL for 3D: dt < dx^2 / (6*D)
        dt_stable = dx**2 / (6 * D) * 0.5  # 50% of limit

        solver = bt.DiffusionSolver3D(mesh, D)
        ic = np.random.rand(mesh.num_nodes())
        solver.set_initial_condition(ic.tolist())

        solver.solve(dt_stable, 10)
        solution = np.array(solver.solution())

        # Should remain bounded
        self.assertTrue(np.all(np.isfinite(solution)))


class TestBoundary3D(unittest.TestCase):
    """Tests for 3D boundary enum."""

    def test_boundary_enum_values(self):
        """Test Boundary3D enum has expected values."""
        self.assertIsNotNone(bt.Boundary3D.XMin)
        self.assertIsNotNone(bt.Boundary3D.XMax)
        self.assertIsNotNone(bt.Boundary3D.YMin)
        self.assertIsNotNone(bt.Boundary3D.YMax)
        self.assertIsNotNone(bt.Boundary3D.ZMin)
        self.assertIsNotNone(bt.Boundary3D.ZMax)


if __name__ == "__main__":
    unittest.main()
