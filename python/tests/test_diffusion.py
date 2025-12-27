import unittest
import numpy as np
from biotransport import StructuredMesh, DiffusionSolver, ReactionDiffusionSolver


class TestDiffusion(unittest.TestCase):
    def test_1d_mesh(self):
        """Test 1D mesh creation and properties."""
        mesh = StructuredMesh(10, 0.0, 1.0)

        self.assertEqual(mesh.nx(), 10)
        self.assertTrue(mesh.is_1d())
        self.assertEqual(mesh.num_nodes(), 11)
        self.assertEqual(mesh.num_cells(), 10)
        self.assertAlmostEqual(mesh.dx(), 0.1)

        # Test node coordinates
        self.assertAlmostEqual(mesh.x(0), 0.0)
        self.assertAlmostEqual(mesh.x(5), 0.5)
        self.assertAlmostEqual(mesh.x(10), 1.0)

    def test_2d_mesh(self):
        """Test 2D mesh creation and properties."""
        mesh = StructuredMesh(5, 5, 0.0, 1.0, 0.0, 1.0)

        self.assertEqual(mesh.nx(), 5)
        self.assertEqual(mesh.ny(), 5)
        self.assertFalse(mesh.is_1d())
        self.assertEqual(mesh.num_nodes(), 36)
        self.assertEqual(mesh.num_cells(), 25)
        self.assertAlmostEqual(mesh.dx(), 0.2)
        self.assertAlmostEqual(mesh.dy(), 0.2)

        # Test node coordinates
        self.assertAlmostEqual(mesh.x(0), 0.0)
        self.assertAlmostEqual(mesh.x(5), 1.0)
        self.assertAlmostEqual(mesh.y(0, 0), 0.0)
        self.assertAlmostEqual(mesh.y(0, 5), 1.0)

    def test_diffusion_solver(self):
        """Test basic diffusion solver functionality."""
        mesh = StructuredMesh(50, 0.0, 1.0)
        D = 0.01
        solver = DiffusionSolver(mesh, D)

        # Create initial condition (Gaussian)
        x = np.array([mesh.x(i) for i in range(mesh.nx() + 1)])
        initial = np.exp(-100 * (x - 0.5) ** 2)
        solver.set_initial_condition(initial)

        # Set boundary conditions
        solver.set_dirichlet_boundary(0, 0.0)
        solver.set_dirichlet_boundary(1, 0.0)

        # Solve
        dt = 0.0001
        num_steps = 100
        solver.solve(dt, num_steps)

        # Get solution
        solution = solver.solution()

        # Basic checks
        self.assertEqual(len(solution), mesh.num_nodes())
        self.assertLess(max(solution), max(initial))  # Peak should decrease
        self.assertGreater(max(solution), 0.0)  # Solution shouldn't be all zero

    def test_reaction_diffusion_solver(self):
        """Test reaction-diffusion solver with a decay term."""
        mesh = StructuredMesh(50, 0.0, 1.0)
        D = 0.01

        # Define decay reaction
        k = 0.1  # decay rate

        def reaction(u, x, y, t):
            return -k * u

        solver = ReactionDiffusionSolver(mesh, D, reaction)

        # Create initial condition (Gaussian)
        x = np.array([mesh.x(i) for i in range(mesh.nx() + 1)])
        initial = np.exp(-100 * (x - 0.5) ** 2)
        solver.set_initial_condition(initial)

        # Set boundary conditions
        solver.set_dirichlet_boundary(0, 0.0)
        solver.set_dirichlet_boundary(1, 0.0)

        # Solve
        dt = 0.0001
        num_steps = 100
        solver.solve(dt, num_steps)

        # Get solution
        solution = solver.solution()

        # Basic checks
        self.assertEqual(len(solution), mesh.num_nodes())
        self.assertLess(max(solution), max(initial))  # Peak should decrease

        # Compare with diffusion-only solution
        diffusion_solver = DiffusionSolver(mesh, D)
        diffusion_solver.set_initial_condition(initial)
        diffusion_solver.set_dirichlet_boundary(0, 0.0)
        diffusion_solver.set_dirichlet_boundary(1, 0.0)
        diffusion_solver.solve(dt, num_steps)
        diffusion_solution = diffusion_solver.solution()

        # Reaction-diffusion solution should have lower values due to decay
        self.assertLess(sum(solution), sum(diffusion_solution))


if __name__ == "__main__":
    unittest.main()
