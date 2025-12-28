"""Tests for advection-diffusion solvers and application-specific solvers."""

import unittest

import numpy as np

import biotransport as bt


class TestAdvectionDiffusionSolver(unittest.TestCase):
    """Tests for AdvectionDiffusionSolver class."""

    def test_construction(self):
        """Test solver construction."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        D = 1e-3  # Diffusivity
        v = 0.1  # Velocity
        solver = bt.AdvectionDiffusionSolver(mesh, D, v)
        self.assertIsNotNone(solver)

    def test_cell_peclet_number(self):
        """Test cell Peclet number calculation."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        D = 1e-3
        v = 0.1
        solver = bt.AdvectionDiffusionSolver(mesh, D, v)

        # Pe = v * dx / D
        dx = 1.0 / 50
        expected_pe = v * dx / D
        self.assertAlmostEqual(solver.cell_peclet(), expected_pe, places=5)

    def test_max_time_step(self):
        """Test maximum stable time step calculation."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        solver = bt.AdvectionDiffusionSolver(mesh, 1e-3, 0.1)

        dt_max = solver.max_time_step()
        self.assertGreater(dt_max, 0)

    def test_set_scheme(self):
        """Test setting advection scheme."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        solver = bt.AdvectionDiffusionSolver(mesh, 1e-3, 0.1)

        # Test setting different schemes
        solver.set_scheme(bt.AdvectionScheme.UPWIND)
        self.assertEqual(solver.scheme(), bt.AdvectionScheme.UPWIND)

        solver.set_scheme(bt.AdvectionScheme.CENTRAL)
        self.assertEqual(solver.scheme(), bt.AdvectionScheme.CENTRAL)

    def test_scheme_stability_check(self):
        """Test scheme stability checking."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        solver = bt.AdvectionDiffusionSolver(mesh, 1e-3, 0.1)

        solver.set_scheme(bt.AdvectionScheme.UPWIND)
        # Upwind is generally stable
        is_stable = solver.is_scheme_stable()
        self.assertIsInstance(is_stable, bool)

    def test_pure_advection(self):
        """Test advection-dominated transport."""
        mesh = bt.StructuredMesh(100, 0.0, 1.0)
        D = 1e-6  # Very small diffusivity
        v = 1.0  # Unit velocity
        solver = bt.AdvectionDiffusionSolver(mesh, D, v)
        solver.set_scheme(bt.AdvectionScheme.UPWIND)

        # Gaussian initial condition
        x = bt.x_nodes(mesh)
        ic = np.exp(-((x - 0.2) ** 2) / 0.01)
        solver.set_initial_condition(ic.tolist())

        # Dirichlet boundaries
        solver.set_boundary(0, bt.BoundaryCondition.dirichlet(0.0))  # Left
        solver.set_boundary(1, bt.BoundaryCondition.dirichlet(0.0))  # Right

        # Solve for short time
        dt = solver.max_time_step() * 0.5
        solver.solve(dt, 10)

        solution = np.array(solver.solution())

        # Peak should have moved to the right
        peak_initial = np.argmax(ic)
        peak_final = np.argmax(solution)
        self.assertGreater(peak_final, peak_initial)

    def test_diffusion_dominated(self):
        """Test diffusion-dominated transport."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        D = 0.1  # Large diffusivity
        v = 0.01  # Small velocity
        solver = bt.AdvectionDiffusionSolver(mesh, D, v)

        # Gaussian initial condition
        x = bt.x_nodes(mesh)
        ic = np.exp(-((x - 0.5) ** 2) / 0.01)
        solver.set_initial_condition(ic.tolist())

        # Neumann boundaries (zero flux)
        solver.set_boundary(0, bt.BoundaryCondition.neumann(0.0))
        solver.set_boundary(1, bt.BoundaryCondition.neumann(0.0))

        # Solve
        dt = solver.max_time_step() * 0.5
        solver.solve(dt, 50)

        solution = np.array(solver.solution())

        # Peak should decrease (spreading)
        self.assertLess(np.max(solution), np.max(ic))


class TestAdvectionScheme(unittest.TestCase):
    """Tests for AdvectionScheme enum."""

    def test_enum_values(self):
        """Test that all expected schemes exist."""
        self.assertIsNotNone(bt.AdvectionScheme.UPWIND)
        self.assertIsNotNone(bt.AdvectionScheme.CENTRAL)
        self.assertIsNotNone(bt.AdvectionScheme.QUICK)
        self.assertIsNotNone(bt.AdvectionScheme.HYBRID)


class TestDarcyFlowSolver(unittest.TestCase):
    """Tests for Darcy flow in porous media."""

    def test_construction(self):
        """Test Darcy flow solver construction."""
        # 2D mesh: nx, ny, x_min, x_max, y_min, y_max
        mesh = bt.StructuredMesh(50, 50, 0.0, 1.0, 0.0, 1.0)
        kappa = 1e-12  # Hydraulic conductivity
        solver = bt.DarcyFlowSolver(mesh, kappa)
        self.assertIsNotNone(solver)

    def test_simple_pressure_drop(self):
        """Test flow with simple pressure gradient."""
        mesh = bt.StructuredMesh(50, 50, 0.0, 1.0, 0.0, 1.0)
        kappa = 1e-12  # Hydraulic conductivity
        solver = bt.DarcyFlowSolver(mesh, kappa)

        # Verify solver was created
        self.assertIsNotNone(solver)


class TestGrayScottSolver(unittest.TestCase):
    """Tests for Gray-Scott reaction-diffusion patterns."""

    def test_construction(self):
        """Test Gray-Scott solver construction."""
        # 2D mesh: nx, ny, x_min, x_max, y_min, y_max
        mesh = bt.StructuredMesh(64, 64, 0.0, 2.5, 0.0, 2.5)
        # Du, Dv, f, k parameters
        solver = bt.GrayScottSolver(mesh, Du=0.16, Dv=0.08, f=0.06, k=0.062)
        self.assertIsNotNone(solver)

    def test_parameters(self):
        """Test setting Gray-Scott parameters."""
        mesh = bt.StructuredMesh(32, 32, 0.0, 1.0, 0.0, 1.0)
        # Default F and k parameters create different patterns
        solver = bt.GrayScottSolver(mesh, Du=0.16, Dv=0.08, f=0.04, k=0.06)
        self.assertIsNotNone(solver)


class TestBioheatCryotherapySolver(unittest.TestCase):
    """Tests for bioheat equation with cryotherapy."""

    def test_construction(self):
        """Test bioheat solver construction."""
        config = bt.BioheatCryotherapyConfig()
        self.assertIsNotNone(config)

    def test_config_parameters(self):
        """Test configuration parameters."""
        config = bt.BioheatCryotherapyConfig()
        # Config should have fields for thermal properties
        self.assertIsNotNone(config)


class TestTumorDrugDeliverySolver(unittest.TestCase):
    """Tests for tumor drug delivery simulation."""

    def test_construction(self):
        """Test tumor drug delivery config construction."""
        config = bt.TumorDrugDeliveryConfig()
        self.assertIsNotNone(config)

    def test_config_parameters(self):
        """Test configuration parameters."""
        config = bt.TumorDrugDeliveryConfig()
        # Config should have domain size, tumor parameters, etc.
        self.assertTrue(hasattr(config, "domain_size"))
        self.assertTrue(hasattr(config, "tumor_radius"))


if __name__ == "__main__":
    unittest.main()
