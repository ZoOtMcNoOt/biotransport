"""Tests for multi-species reaction-diffusion systems."""

import unittest

import numpy as np

import biotransport as bt


class TestMultiSpeciesSolver(unittest.TestCase):
    """Tests for MultiSpeciesSolver class."""

    def test_construction(self):
        """Test solver construction with diffusivities."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        diffusivities = [1e-9, 2e-9]
        solver = bt.MultiSpeciesSolver(mesh, diffusivities)
        self.assertIsNotNone(solver)

    def test_num_species(self):
        """Test species count accessor."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        diffusivities = [1e-9, 2e-9, 3e-9]
        solver = bt.MultiSpeciesSolver(mesh, diffusivities)
        self.assertEqual(solver.num_species(), 3)

    def test_diffusivity_accessor(self):
        """Test getting diffusivity for each species."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        diffusivities = [1e-9, 2e-9]
        solver = bt.MultiSpeciesSolver(mesh, diffusivities)
        self.assertAlmostEqual(solver.diffusivity(0), 1e-9)
        self.assertAlmostEqual(solver.diffusivity(1), 2e-9)

    def test_set_initial_condition(self):
        """Test setting initial conditions for species."""
        mesh = bt.StructuredMesh(30, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [1e-9, 1e-9])

        n = mesh.num_nodes()
        u0 = np.ones(n) * 100.0
        v0 = np.ones(n) * 50.0

        solver.set_initial_condition(0, u0.tolist())
        solver.set_initial_condition(1, v0.tolist())

        u = np.array(solver.solution(0))
        v = np.array(solver.solution(1))

        np.testing.assert_array_almost_equal(u, u0)
        np.testing.assert_array_almost_equal(v, v0)

    def test_uniform_initial_condition(self):
        """Test setting uniform initial condition."""
        mesh = bt.StructuredMesh(30, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [1e-9, 1e-9])

        solver.set_uniform_initial_condition(0, 100.0)
        solver.set_uniform_initial_condition(1, 50.0)

        u = np.array(solver.solution(0))
        v = np.array(solver.solution(1))

        self.assertTrue(np.all(u == 100.0))
        self.assertTrue(np.all(v == 50.0))

    def test_pure_diffusion(self):
        """Test two-species pure diffusion (no reactions)."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [1e-3, 2e-3])  # Different diffusivities

        # Gaussian initial conditions
        x = bt.x_nodes(mesh)
        u0 = 100.0 * np.exp(-((x - 0.5) ** 2) / 0.01)
        v0 = 50.0 * np.exp(-((x - 0.5) ** 2) / 0.01)

        solver.set_initial_condition(0, u0.tolist())
        solver.set_initial_condition(1, v0.tolist())

        # Solve
        solver.solve(1e-4, 100)

        u = np.array(solver.solution(0))
        v = np.array(solver.solution(1))

        # Both should diffuse (peaks decrease)
        self.assertLess(np.max(u), np.max(u0))
        self.assertLess(np.max(v), np.max(v0))

        # Species 1 (higher D) should spread more
        self.assertLess(np.max(v), np.max(u))

    def test_stability_check(self):
        """Test stability criterion."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [1e-3, 2e-3])

        is_stable = solver.check_stability(1e-4)
        self.assertIsInstance(is_stable, bool)

    def test_max_stable_time_step(self):
        """Test maximum stable time step calculation."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [1e-3, 2e-3])

        dt_max = solver.max_stable_time_step()
        self.assertGreater(dt_max, 0)

    def test_all_solutions(self):
        """Test getting all solutions at once."""
        mesh = bt.StructuredMesh(20, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [1e-9, 1e-9])

        solver.set_uniform_initial_condition(0, 100.0)
        solver.set_uniform_initial_condition(1, 50.0)

        all_sols = solver.all_solutions()
        self.assertEqual(len(all_sols), 2)
        self.assertEqual(len(all_sols[0]), mesh.num_nodes())

    def test_total_mass(self):
        """Test total mass calculation."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [1e-9, 1e-9])

        solver.set_uniform_initial_condition(0, 100.0)

        mass = solver.total_mass(0)
        # Mass should be approximately concentration * length
        expected_mass = 100.0 * 1.0
        self.assertAlmostEqual(mass, expected_mass, delta=5.0)


class TestLotkaVolterraReaction(unittest.TestCase):
    """Tests for Lotka-Volterra predator-prey model."""

    def test_construction(self):
        """Test Lotka-Volterra construction."""
        # alpha, beta, gamma, delta
        lv = bt.LotkaVolterraReaction(1.0, 0.1, 0.1, 0.02)
        self.assertIsNotNone(lv)

    def test_parameter_accessors(self):
        """Test accessing parameters."""
        lv = bt.LotkaVolterraReaction(alpha=1.0, beta=0.1, gamma=0.1, delta=0.02)
        self.assertAlmostEqual(lv.alpha, 1.0)
        self.assertAlmostEqual(lv.beta, 0.1)
        self.assertAlmostEqual(lv.gamma, 0.1)
        self.assertAlmostEqual(lv.delta, 0.02)

    def test_integration_with_solver(self):
        """Test using Lotka-Volterra with MultiSpeciesSolver."""
        mesh = bt.StructuredMesh(50, 0.0, 10.0)
        solver = bt.MultiSpeciesSolver(mesh, [0.01, 0.01])

        # Create and set reaction model
        lv = bt.LotkaVolterraReaction(1.0, 0.1, 0.1, 0.02)
        solver.set_reaction_model(lv)

        # Initial populations: prey=40, predator=9
        solver.set_uniform_initial_condition(0, 40.0)  # Prey
        solver.set_uniform_initial_condition(1, 9.0)  # Predator

        # Neumann boundaries (no flux)
        solver.set_all_species_neumann(bt.Boundary.Left, 0.0)
        solver.set_all_species_neumann(bt.Boundary.Right, 0.0)

        # Solve for short time
        solver.solve(0.01, 100)

        prey = np.array(solver.solution(0))
        predator = np.array(solver.solution(1))

        # Both species should remain positive
        self.assertTrue(np.all(prey > 0))
        self.assertTrue(np.all(predator > 0))


class TestSIRReaction(unittest.TestCase):
    """Tests for SIR epidemic model."""

    def test_construction(self):
        """Test SIR construction."""
        sir = bt.SIRReaction(beta=0.3, gamma=0.1, total_population=1000.0)
        self.assertIsNotNone(sir)

    def test_parameter_accessors(self):
        """Test accessing SIR parameters."""
        sir = bt.SIRReaction(beta=0.3, gamma=0.1, total_population=1000.0)
        self.assertAlmostEqual(sir.beta, 0.3)
        self.assertAlmostEqual(sir.gamma, 0.1)
        self.assertAlmostEqual(sir.N, 1000.0)

    def test_basic_reproduction_number(self):
        """Test R0 calculation."""
        # R0 = beta / gamma
        sir = bt.SIRReaction(beta=0.3, gamma=0.1, total_population=1000.0)
        self.assertAlmostEqual(sir.R0, 3.0)

    def test_integration_with_solver(self):
        """Test SIR with MultiSpeciesSolver (3 species: S, I, R)."""
        mesh = bt.StructuredMesh(50, 0.0, 100.0)
        # Diffusion coefficients for each compartment
        solver = bt.MultiSpeciesSolver(mesh, [0.1, 0.05, 0.05])

        sir = bt.SIRReaction(beta=0.3, gamma=0.1, total_population=1000.0)
        solver.set_reaction_model(sir)

        # Initial: S=990, I=10, R=0 (uniform)
        solver.set_uniform_initial_condition(0, 990.0)  # Susceptible
        solver.set_uniform_initial_condition(1, 10.0)  # Infected
        solver.set_uniform_initial_condition(2, 0.0)  # Recovered

        # Neumann boundaries
        solver.set_all_species_neumann(bt.Boundary.Left, 0.0)
        solver.set_all_species_neumann(bt.Boundary.Right, 0.0)

        # Solve
        solver.solve(0.1, 50)

        S = np.array(solver.solution(0))
        infected = np.array(solver.solution(1))
        R = np.array(solver.solution(2))

        # All compartments should be non-negative
        self.assertTrue(np.all(S >= 0))
        self.assertTrue(np.all(infected >= 0))
        self.assertTrue(np.all(R >= 0))


class TestBrusselatorReaction(unittest.TestCase):
    """Tests for Brusselator oscillatory reaction model."""

    def test_construction(self):
        """Test Brusselator construction."""
        br = bt.BrusselatorReaction(A=1.0, B=3.0)
        self.assertIsNotNone(br)

    def test_parameter_accessors(self):
        """Test accessing Brusselator parameters."""
        br = bt.BrusselatorReaction(A=1.0, B=3.0)
        self.assertAlmostEqual(br.A, 1.0)
        self.assertAlmostEqual(br.B, 3.0)

    def test_oscillatory_condition(self):
        """Test oscillatory condition check."""
        # Oscillatory when B > 1 + A^2
        br_osc = bt.BrusselatorReaction(A=1.0, B=3.0)  # B=3 > 1+1=2
        self.assertTrue(br_osc.is_oscillatory)

        br_stable = bt.BrusselatorReaction(A=1.0, B=1.5)  # B=1.5 < 2
        self.assertFalse(br_stable.is_oscillatory)

    def test_integration_with_solver(self):
        """Test Brusselator with MultiSpeciesSolver."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [0.01, 0.01])

        br = bt.BrusselatorReaction(A=1.0, B=3.0)
        solver.set_reaction_model(br)

        # Steady state: u=A, v=B/A
        # Add small perturbation
        solver.set_uniform_initial_condition(0, 1.0 + 0.1)  # u = A + perturbation
        solver.set_uniform_initial_condition(1, 3.0 + 0.1)  # v = B/A + perturbation

        # Neumann boundaries
        solver.set_all_species_neumann(bt.Boundary.Left, 0.0)
        solver.set_all_species_neumann(bt.Boundary.Right, 0.0)

        # Solve
        solver.solve(1e-4, 100)

        u = np.array(solver.solution(0))
        v = np.array(solver.solution(1))

        # Solutions should remain positive and bounded
        self.assertTrue(np.all(u > 0))
        self.assertTrue(np.all(v > 0))
        self.assertTrue(np.all(u < 100))
        self.assertTrue(np.all(v < 100))


class TestSEIRReaction(unittest.TestCase):
    """Tests for SEIR epidemic model (with exposed class)."""

    def test_construction(self):
        """Test SEIR construction."""
        seir = bt.SEIRReaction(beta=0.3, sigma=0.2, gamma=0.1, total_population=1000.0)
        self.assertIsNotNone(seir)


class TestCompetitiveInhibitionReaction(unittest.TestCase):
    """Tests for enzyme competitive inhibition model."""

    def test_construction(self):
        """Test competitive inhibition construction."""
        # Vmax, Km, Ki
        ci = bt.CompetitiveInhibitionReaction(100.0, 10.0, 5.0)
        self.assertIsNotNone(ci)


class TestEnzymeCascadeReaction(unittest.TestCase):
    """Tests for enzyme cascade reaction model."""

    def test_construction(self):
        """Test enzyme cascade construction."""
        # 2-step cascade with vmax, km for each enzyme
        # kdeg has one more element (degradation for each intermediate + product)
        cascade = bt.EnzymeCascadeReaction(
            vmax_values=[100.0, 80.0],
            km_values=[10.0, 8.0],
            kdeg_values=[0.1, 0.1, 0.1],  # 3 elements: S1, S2, P
        )
        self.assertIsNotNone(cascade)


if __name__ == "__main__":
    unittest.main()
