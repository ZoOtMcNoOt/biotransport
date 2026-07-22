"""Tests for 3D mesh and solver functionality."""

import unittest

import numpy as np

import biotransport as bt


def trapezoidal_mass(mesh, values):
    """Integral for the vertex-centred control volumes used by the solver."""
    field = np.asarray(values).reshape(mesh.nz() + 1, mesh.ny() + 1, mesh.nx() + 1)
    weights = np.ones_like(field)
    weights[:, :, (0, -1)] *= 0.5
    weights[:, (0, -1), :] *= 0.5
    weights[(0, -1), :, :] *= 0.5
    return np.sum(weights * field) * mesh.dx() * mesh.dy() * mesh.dz()


def set_homogeneous_neumann(solver):
    for face in (
        bt.Boundary3D.XMin,
        bt.Boundary3D.XMax,
        bt.Boundary3D.YMin,
        bt.Boundary3D.YMax,
        bt.Boundary3D.ZMin,
        bt.Boundary3D.ZMax,
    ):
        solver.set_neumann_boundary(face, 0.0)


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

        # A constant Dirichlet value on every face is compatible at all edges
        # and corners. Conflicting face traces are rejected instead of being
        # silently resolved by face-application order.
        solver.set_dirichlet_boundary(bt.Boundary3D.XMin, 0.5)
        solver.set_dirichlet_boundary(bt.Boundary3D.XMax, 0.5)
        solver.set_dirichlet_boundary(bt.Boundary3D.YMin, 0.5)
        solver.set_dirichlet_boundary(bt.Boundary3D.YMax, 0.5)
        solver.set_dirichlet_boundary(bt.Boundary3D.ZMin, 0.5)
        solver.set_dirichlet_boundary(bt.Boundary3D.ZMax, 0.5)

        ic = 0.5 * np.ones(mesh.num_nodes())
        solver.set_initial_condition(ic.tolist())

        # Should run without error
        solver.solve(0.0001, 10)

    def test_conflicting_dirichlet_edges_fail_loudly(self):
        mesh = bt.StructuredMesh3D(5, 5, 5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        solver = bt.DiffusionSolver3D(mesh, 0.01)
        solver.set_dirichlet_boundary(bt.Boundary3D.XMin, 1.0)
        solver.set_dirichlet_boundary(bt.Boundary3D.YMin, 0.0)
        solver.set_initial_condition(np.full(mesh.num_nodes(), 0.5).tolist())

        with self.assertRaisesRegex(ValueError, "Conflicting Dirichlet"):
            solver.solve(0.0001, 1)
        solution = np.array(solver.solution())
        self.assertFalse(np.any(np.isnan(solution)))

    def test_diffusion_from_hot_center(self):
        """Test heat diffuses from hot center."""
        mesh = bt.StructuredMesh3D(10, 10, 10, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        solver = bt.DiffusionSolver3D(mesh, 0.01)
        set_homogeneous_neumann(solver)

        # Hot spot at center
        ic = np.zeros(mesh.num_nodes())
        center_idx = mesh.index(mesh.nx() // 2, mesh.ny() // 2, mesh.nz() // 2)
        ic[center_idx] = 100.0
        solver.set_initial_condition(ic.tolist())
        initial_mass = trapezoidal_mass(mesh, ic)

        solver.solve(0.5 * solver.max_stable_time_step(), 50)
        solution = np.array(solver.solution())

        self.assertLess(np.max(solution), 100.0)
        self.assertTrue(np.all(np.isfinite(solution)))
        self.assertAlmostEqual(
            trapezoidal_mass(mesh, solution), initial_mass, places=12
        )

    def test_outward_neumann_derivative_preserves_linear_field(self):
        """A harmonic linear field is stationary with its exact outward derivatives."""
        mesh = bt.StructuredMesh3D(7, 5, 4, 0.0, 1.0, -0.5, 0.5, 0.0, 2.0)
        solver = bt.DiffusionSolver3D(mesh, 0.3)
        exact = np.empty(mesh.num_nodes())
        for k in range(mesh.nz() + 1):
            for j in range(mesh.ny() + 1):
                for i in range(mesh.nx() + 1):
                    exact[mesh.index(i, j, k)] = (
                        4 + mesh.x(i) + 2 * mesh.y(j) + 3 * mesh.z(k)
                    )
        solver.set_initial_condition(exact)
        solver.set_neumann_boundary(bt.Boundary3D.XMin, -1.0)
        solver.set_neumann_boundary(bt.Boundary3D.XMax, 1.0)
        solver.set_neumann_boundary(bt.Boundary3D.YMin, -2.0)
        solver.set_neumann_boundary(bt.Boundary3D.YMax, 2.0)
        solver.set_neumann_boundary(bt.Boundary3D.ZMin, -3.0)
        solver.set_neumann_boundary(bt.Boundary3D.ZMax, 3.0)
        solver.solve(0.8 * solver.max_stable_time_step(), 12)
        np.testing.assert_allclose(solver.solution(), exact, rtol=0.0, atol=3e-13)

    def test_stability(self):
        """Test explicit solver respects CFL condition."""
        mesh = bt.StructuredMesh3D(10, 10, 10, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        D = 0.01
        dx = 0.1

        # CFL for 3D: dt < dx^2 / (6*D)
        dt_stable = dx**2 / (6 * D) * 0.5  # 50% of limit

        solver = bt.DiffusionSolver3D(mesh, D)
        ic = np.random.default_rng(17).random(mesh.num_nodes())
        solver.set_initial_condition(ic.tolist())

        solver.solve(dt_stable, 10)
        solution = np.array(solver.solution())

        self.assertTrue(np.all(np.isfinite(solution)))
        with self.assertRaises(ValueError):
            solver.solve(np.nextafter(solver.max_stable_time_step(), np.inf), 1)

    def test_invalid_physical_inputs_are_rejected(self):
        mesh = bt.StructuredMesh3D(3, 3, 3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            bt.DiffusionSolver3D(mesh, 0.0)
        solver = bt.DiffusionSolver3D(mesh, 0.1)
        with self.assertRaises(ValueError):
            solver.set_initial_condition([0.0] * (mesh.num_nodes() - 1))
        with self.assertRaises(ValueError):
            solver.solve(np.inf, 1)


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
