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
        """The reported value is the exact diffusion-only Euler CFL limit."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [1e-3, 2e-3])

        dt_max = solver.max_stable_time_step()
        expected = mesh.dx() ** 2 / (2.0 * 2e-3)
        self.assertAlmostEqual(dt_max, expected, places=15)
        self.assertTrue(solver.check_stability(dt_max))
        self.assertFalse(solver.check_stability(np.nextafter(dt_max, np.inf)))

    def test_all_solutions(self):
        """Test getting all solutions at once."""
        mesh = bt.StructuredMesh(20, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [1e-9, 1e-9])

        solver.set_uniform_initial_condition(0, 100.0)
        solver.set_uniform_initial_condition(1, 50.0)

        all_sols = solver.all_solutions()
        self.assertEqual(len(all_sols), 2)
        self.assertEqual(len(all_sols[0]), mesh.num_nodes())

    def test_solution_arrays_are_owned_snapshots(self):
        """Python arrays do not alias the solver's rotating C++ work buffers."""
        mesh = bt.StructuredMesh(8, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [0.0])
        solver.set_uniform_initial_condition(0, 2.0)

        snapshot = solver.solution(0)
        snapshot[0] = 99.0
        self.assertEqual(solver.concentration(0, 0), 2.0)

        solver.solve(0.1, 1)
        self.assertEqual(snapshot[1], 2.0)
        all_snapshots = solver.all_solutions()
        all_snapshots[0][0] = 88.0
        self.assertEqual(solver.concentration(0, 0), 2.0)

    def test_total_mass(self):
        """Test total mass calculation."""
        mesh = bt.StructuredMesh(50, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [1e-9, 1e-9])

        solver.set_uniform_initial_condition(0, 100.0)

        mass = solver.total_mass(0)
        # Nodal fields use trapezoidal quadrature, including half-weight end nodes.
        expected_mass = 100.0 * 1.0
        self.assertAlmostEqual(mass, expected_mass, places=12)

    def test_default_closed_boundaries_conserve_diffusive_mass_1d(self):
        """The default zero-normal-derivative boundary is conservative."""
        mesh = bt.StructuredMesh(40, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [0.05])
        x = bt.x_nodes(mesh)
        initial = 0.2 + np.exp(-120.0 * (x - 0.37) ** 2)
        solver.set_initial_condition(0, initial)
        initial_mass = solver.total_mass(0)

        solver.solve(0.8 * solver.max_stable_time_step(), 100)

        self.assertAlmostEqual(solver.total_mass(0), initial_mass, places=12)
        self.assertTrue(np.all(np.asarray(solver.solution(0)) >= 0.0))

    def test_default_closed_boundaries_conserve_diffusive_mass_2d(self):
        """Half/quarter boundary control volumes conserve the 2-D integral."""
        mesh = bt.StructuredMesh(16, 12, 0.0, 2.0, 0.0, 1.5)
        solver = bt.MultiSpeciesSolver(mesh, [0.03])
        rng = np.random.default_rng(1234)
        initial = 0.1 + rng.random(mesh.num_nodes())
        solver.set_initial_condition(0, initial)
        initial_mass = solver.total_mass(0)

        solver.solve(0.75 * solver.max_stable_time_step(), 80)

        self.assertAlmostEqual(solver.total_mass(0), initial_mass, places=11)

    def test_reaction_positivity_violation_fails_loudly(self):
        """A reaction-limited step is rejected rather than clipped."""
        mesh = bt.StructuredMesh(8, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [0.0, 0.0])
        solver.set_reaction_model(
            bt.LotkaVolterraReaction(alpha=0.0, beta=1.0, gamma=0.0, delta=0.0)
        )
        solver.set_uniform_initial_condition(0, 1.0)
        solver.set_uniform_initial_condition(1, 10.0)

        with self.assertRaisesRegex(RuntimeError, "positivity limit"):
            solver.solve(1.0, 1)

        np.testing.assert_array_equal(solver.solution(0), np.ones(mesh.num_nodes()))
        self.assertEqual(solver.time(), 0.0)

    def test_python_reaction_callback_copies_mutated_rates(self):
        """Mutations to the Python rates list drive the native update."""
        mesh = bt.StructuredMesh(4, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [0.0, 0.0])
        solver.set_uniform_initial_condition(0, 2.0)
        solver.set_uniform_initial_condition(1, 1.0)

        def reaction(rates, concentrations, x, y, time):
            self.assertEqual(len(rates), 2)
            self.assertEqual(rates, [0.0, 0.0])
            self.assertEqual(concentrations, [2.0, 1.0])
            self.assertGreaterEqual(x, 0.0)
            self.assertEqual(y, 0.0)
            self.assertEqual(time, 0.0)
            rates[:] = [-concentrations[0], concentrations[0]]

        solver.set_reaction_function(reaction)
        solver.solve(0.1, 1)

        np.testing.assert_allclose(solver.solution(0), 1.8, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(solver.solution(1), 1.2, rtol=0.0, atol=1e-15)

    def test_python_reaction_callback_accepts_returned_sequence(self):
        """Returning a 1-D sequence is the concise custom-kinetics form."""
        mesh = bt.StructuredMesh(3, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [0.0, 0.0])
        solver.set_uniform_initial_condition(0, 1.0)
        solver.set_uniform_initial_condition(1, 3.0)

        def reaction(rates, concentrations, _x, _y, _time):
            rates[:] = [100.0, 100.0]  # Ignored when a sequence is returned.
            return np.array([concentrations[1], -concentrations[0]])

        solver.set_reaction_function(reaction)
        solver.solve(0.25, 1)

        np.testing.assert_allclose(solver.solution(0), 1.75, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(solver.solution(1), 2.75, rtol=0.0, atol=1e-15)

    def test_python_reaction_callback_validates_rates(self):
        """Bad callback output fails at its scientific boundary with clear errors."""
        mesh = bt.StructuredMesh(2, 0.0, 1.0)

        cases = [
            (lambda *_args: [0.0], ValueError, "expected exactly 2"),
            (lambda *_args: [[0.0, 0.0]], ValueError, "one-dimensional"),
            (lambda *_args: [0.0, np.nan], ValueError, "index 1 must be finite"),
            (lambda *_args: 1.0, TypeError, "must return None"),
        ]
        for reaction, error_type, message in cases:
            with self.subTest(message=message):
                solver = bt.MultiSpeciesSolver(mesh, [0.0, 0.0])
                solver.set_uniform_initial_condition(0, 1.0)
                solver.set_uniform_initial_condition(1, 1.0)
                solver.set_reaction_function(reaction)
                with self.assertRaisesRegex(error_type, message):
                    solver.solve(0.1, 1)
                self.assertEqual(solver.time(), 0.0)

        def changes_rate_count(rates, *_args):
            rates.append(0.0)

        solver = bt.MultiSpeciesSolver(mesh, [0.0, 0.0])
        solver.set_reaction_function(changes_rate_count)
        with self.assertRaisesRegex(ValueError, "expected exactly 2"):
            solver.solve(0.1, 1)

    def test_dirichlet_value_participates_in_first_step(self):
        """Fixed boundary data is visible to its neighbor immediately."""
        mesh = bt.StructuredMesh(4, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [0.1])
        solver.set_uniform_initial_condition(0, 0.0)
        solver.set_dirichlet_boundary(0, bt.Boundary.Left, 1.0)

        solver.solve(0.01, 1)

        self.assertEqual(solver.concentration(0, 0), 1.0)
        self.assertAlmostEqual(
            solver.concentration(0, 1), 0.01 * 0.1 / mesh.dx() ** 2, places=15
        )

    def test_builtin_reaction_arity_fails_before_solve(self):
        """Invalid built-in model/solver combinations never enter OpenMP."""
        mesh = bt.StructuredMesh(8, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "at least 3 species"):
            solver.set_reaction_model(bt.SIRReaction(0.3, 0.1, 1000.0))

    def test_sir_homogeneous_kinetics_conserve_population(self):
        """S+I+R is an invariant of the homogeneous SIR reaction equations."""
        mesh = bt.StructuredMesh(10, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [0.0, 0.0, 0.0])
        solver.set_reaction_model(bt.SIRReaction(0.3, 0.1, 1000.0))
        solver.set_uniform_initial_condition(0, 990.0)
        solver.set_uniform_initial_condition(1, 10.0)
        solver.set_uniform_initial_condition(2, 0.0)

        solver.solve(0.01, 500)

        total = sum(solver.total_mass(species) for species in range(3))
        self.assertAlmostEqual(total, 1000.0, places=9)
        self.assertAlmostEqual(solver.time(), 5.0, places=14)

    def test_solve_until_reaches_absolute_final_time(self):
        mesh = bt.StructuredMesh(10, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [0.0, 0.0, 0.0])
        solver.set_reaction_model(bt.SIRReaction(0.3, 0.1, 1000.0))
        solver.set_uniform_initial_condition(0, 990.0)
        solver.set_uniform_initial_condition(1, 10.0)
        solver.set_uniform_initial_condition(2, 0.0)

        solver.solve_until(1.0, maximum_dt=0.3)
        self.assertEqual(solver.time(), 1.0)
        solver.solve_until(1.125, maximum_dt=0.2)
        self.assertEqual(solver.time(), 1.125)
        with self.assertRaisesRegex(ValueError, "must not precede"):
            solver.solve_until(1.0, maximum_dt=0.1)

        diffusive = bt.MultiSpeciesSolver(mesh, [0.1])
        x = bt.x_nodes(mesh)
        profile = 0.2 + np.exp(-40.0 * (x - 0.5) ** 2)
        diffusive.set_initial_condition(0, profile)
        initial_mass = diffusive.total_mass(0)
        diffusive.solve_until(0.1, maximum_dt=1.0)
        self.assertEqual(diffusive.time(), 0.1)
        self.assertAlmostEqual(diffusive.total_mass(0), initial_mass, places=14)

    def test_invalid_numeric_inputs_are_rejected(self):
        mesh = bt.StructuredMesh(8, 0.0, 1.0)
        with self.assertRaises(ValueError):
            bt.MultiSpeciesSolver(mesh, [np.nan])
        solver = bt.MultiSpeciesSolver(mesh, [0.0])
        invalid = np.zeros(mesh.num_nodes())
        invalid[3] = np.inf
        with self.assertRaises(ValueError):
            solver.set_initial_condition(0, invalid)
        with self.assertRaises(IndexError):
            solver.concentration(0, mesh.num_nodes())


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
        no_recovery = bt.SIRReaction(beta=0.3, gamma=0.0, total_population=1000.0)
        self.assertTrue(np.isinf(no_recovery.R0))

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

    def test_negative_inhibitor_decay_is_rejected(self):
        with self.assertRaises(ValueError):
            bt.CompetitiveInhibitionReaction(100.0, 10.0, 5.0, -0.1)


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

    def test_solver_species_count_must_match_cascade(self):
        mesh = bt.StructuredMesh(8, 0.0, 1.0)
        solver = bt.MultiSpeciesSolver(mesh, [0.0, 0.0])
        cascade = bt.EnzymeCascadeReaction([1.0, 1.0], [1.0, 1.0], [0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "must match"):
            solver.set_reaction_model(cascade)


class TestGrayScottScienceContract(unittest.TestCase):
    """Independent checks for the cell-centred periodic Gray-Scott kernel."""

    def test_homogeneous_one_step_matches_kinetics(self):
        mesh = bt.StructuredMesh(4, 3, 0.0, 4.0, 0.0, 3.0)
        solver = bt.GrayScottSolver(mesh, Du=0.0, Dv=0.0, f=0.04, k=0.06)
        u0 = np.full(mesh.nx() * mesh.ny(), 0.8, dtype=np.float32)
        v0 = np.full(mesh.nx() * mesh.ny(), 0.2, dtype=np.float32)
        dt = 0.1

        result = solver.simulate(u0, v0, total_steps=1, dt=dt, steps_between_frames=1)

        uvv = 0.8 * 0.2**2
        expected_u = 0.8 + dt * (-uvv + 0.04 * (1.0 - 0.8))
        expected_v = 0.2 + dt * (uvv - (0.04 + 0.06) * 0.2)
        np.testing.assert_allclose(result.u_frames()[-1], expected_u, rtol=2e-7)
        np.testing.assert_allclose(result.v_frames()[-1], expected_v, rtol=2e-7)
        self.assertEqual(result.u_frames().shape, (2, mesh.ny(), mesh.nx()))
        self.assertAlmostEqual(result.final_time, dt, places=15)

    def test_periodic_diffusion_conserves_sum(self):
        mesh = bt.StructuredMesh(12, 10, 0.0, 12.0, 0.0, 10.0)
        solver = bt.GrayScottSolver(mesh, Du=0.16, Dv=0.0, f=0.0, k=0.0)
        rng = np.random.default_rng(9)
        u0 = rng.random(mesh.nx() * mesh.ny(), dtype=np.float32)
        v0 = np.zeros_like(u0)

        result = solver.simulate(
            u0, v0, total_steps=50, dt=0.5, steps_between_frames=50
        )

        self.assertAlmostEqual(
            float(np.sum(result.u_frames()[-1], dtype=np.float64)),
            float(np.sum(u0, dtype=np.float64)),
            places=5,
        )
        self.assertTrue(np.all(result.u_frames()[-1] >= 0.0))

    def test_unstable_step_and_nonfinite_state_are_rejected(self):
        mesh = bt.StructuredMesh(8, 8, 0.0, 0.8, 0.0, 0.8)
        solver = bt.GrayScottSolver(mesh, Du=0.16, Dv=0.08, f=0.04, k=0.06)
        u0 = np.ones(mesh.nx() * mesh.ny(), dtype=np.float32)
        v0 = np.zeros_like(u0)
        with self.assertRaisesRegex(RuntimeError, "positivity limit"):
            solver.simulate(u0, v0, total_steps=1, dt=1.0)

        u0[0] = np.nan
        with self.assertRaises(ValueError):
            solver.simulate(u0, v0, total_steps=1, dt=1e-3)

    def test_zero_stability_tolerance_runs_to_final_time(self):
        mesh = bt.StructuredMesh(6, 5, 0.0, 6.0, 0.0, 5.0)
        solver = bt.GrayScottSolver(mesh, Du=0.16, Dv=0.08, f=0.04, k=0.06)
        u0 = np.ones(mesh.nx() * mesh.ny(), dtype=np.float32)
        v0 = np.zeros_like(u0)

        result = solver.simulate(
            u0,
            v0,
            total_steps=4,
            dt=0.5,
            steps_between_frames=3,
            check_interval=1,
            stable_tol=0.0,
            min_frames_before_early_stop=1,
        )

        self.assertEqual(result.steps_run, 4)
        self.assertEqual(result.frame_steps[-1], 4)
        self.assertEqual(result.final_time, 2.0)

    def test_early_stop_occurs_only_on_a_current_check(self):
        mesh = bt.StructuredMesh(6, 5, 0.0, 6.0, 0.0, 5.0)
        solver = bt.GrayScottSolver(mesh, Du=0.16, Dv=0.08, f=0.04, k=0.06)
        u0 = np.ones(mesh.nx() * mesh.ny(), dtype=np.float32)
        v0 = np.zeros_like(u0)
        u0[0] = 0.5
        v0[0] = 0.25

        result = solver.simulate(
            u0,
            v0,
            total_steps=10,
            dt=0.5,
            steps_between_frames=3,
            check_interval=2,
            stable_tol=10.0,
            min_frames_before_early_stop=2,
        )

        self.assertEqual(result.steps_run, 4)
        self.assertEqual(result.frame_steps[-1], 4)


if __name__ == "__main__":
    unittest.main()
