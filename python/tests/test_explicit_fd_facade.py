import unittest

import numpy as np

from biotransport import Boundary, ExplicitFD, DiffusionProblem, StructuredMesh


class TestExplicitFDFacade(unittest.TestCase):
    def test_diffusion_problem_run_1d_dirichlet(self):
        mesh = StructuredMesh(50, 0.0, 1.0)

        # Initial condition: Gaussian bump
        x = np.array([mesh.x(i) for i in range(mesh.nx() + 1)])
        initial = np.exp(-100.0 * (x - 0.5) ** 2)

        problem = (
            DiffusionProblem(mesh)
            .diffusivity(0.01)
            .initial_condition(initial)
            .dirichlet(Boundary.Left, 0.0)
            .dirichlet(Boundary.Right, 0.0)
        )

        solver = ExplicitFD().safety_factor(0.9)
        result = solver.run(problem, 0.01)

        sol = result.solution()
        self.assertEqual(len(sol), mesh.num_nodes())

        # Facade should produce a stable run with positive dt/steps
        self.assertGreater(result.stats.dt, 0.0)
        self.assertGreater(result.stats.steps, 0)
        self.assertAlmostEqual(result.stats.t_end, 0.01)

        # Dirichlet boundaries should be enforced (within numerical tolerance)
        self.assertAlmostEqual(float(sol[0]), 0.0, places=12)
        self.assertAlmostEqual(float(sol[-1]), 0.0, places=12)

    def test_diffusion_problem_run_2d_neumann_preserves_constant(self):
        mesh = StructuredMesh(20, 20, 0.0, 1.0, 0.0, 1.0)
        initial = np.ones(mesh.num_nodes(), dtype=np.float64)

        problem = (
            DiffusionProblem(mesh)
            .diffusivity(0.1)
            .initial_condition(initial)
            .neumann(Boundary.Left, 0.0)
            .neumann(Boundary.Right, 0.0)
            .neumann(Boundary.Bottom, 0.0)
            .neumann(Boundary.Top, 0.0)
        )

        result = ExplicitFD().run(problem, 0.1)
        sol = result.solution()

        self.assertEqual(len(sol), mesh.num_nodes())
        self.assertGreater(result.stats.dt, 0.0)
        self.assertGreater(result.stats.steps, 0)

        # Constant field is an exact steady-state of diffusion with Neumann zero flux.
        self.assertAlmostEqual(float(np.min(sol)), 1.0, places=12)
        self.assertAlmostEqual(float(np.max(sol)), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
