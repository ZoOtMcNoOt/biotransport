"""Tests for Nernst-Planck electrochemical ion transport."""

import unittest

import numpy as np

import biotransport as bt


class TestIonSpecies(unittest.TestCase):
    """Tests for IonSpecies class."""

    def test_construction(self):
        """Test IonSpecies construction."""
        # Sodium ion
        na = bt.IonSpecies("Na+", 1, 1.33e-9)
        self.assertIsNotNone(na)

    def test_properties(self):
        """Test ion property accessors."""
        # Chloride ion
        cl = bt.IonSpecies("Cl-", -1, 2.03e-9)
        self.assertEqual(cl.name, "Cl-")
        self.assertEqual(cl.valence, -1)
        self.assertAlmostEqual(cl.diffusivity, 2.03e-9)

    def test_mobility_calculation(self):
        """Test Einstein relation for mobility."""
        # At body temp (310K), mobility = D * z * F / (R * T)
        D = 1.33e-9  # Na+ diffusivity
        z = 1
        T = 310.0  # K

        na = bt.IonSpecies("Na+", z, D, T)

        # Mobility should be positive for positive ions
        self.assertGreater(na.mobility, 0)
        self.assertEqual(na.mobility_temperature, T)
        self.assertAlmostEqual(na.mobility_at(T), na.mobility, places=18)
        self.assertGreater(na.mobility_at(298.0), na.mobility)

    def test_thermal_voltage(self):
        """Test thermal voltage calculation."""
        # Vt = RT/F ≈ 26.7 mV at 310K
        Vt = bt.IonSpecies.thermal_voltage(310.0)
        self.assertAlmostEqual(Vt * 1000, 26.7, places=0)  # ~27 mV


class TestNernstPlanckSolver(unittest.TestCase):
    """Tests for single-ion Nernst-Planck solver."""

    def test_construction(self):
        """Test solver construction."""
        mesh = bt.StructuredMesh(50, 0.0, 1e-3)  # 1 mm domain
        ion = bt.IonSpecies("Na+", 1, 1.33e-9)
        solver = bt.NernstPlanckSolver(mesh, ion)
        self.assertIsNotNone(solver)

    def test_thermal_voltage_method(self):
        """Test thermal voltage accessor on solver."""
        mesh = bt.StructuredMesh(20, 0.0, 1e-3)
        ion = bt.IonSpecies("K+", 1, 1.96e-9)
        solver = bt.NernstPlanckSolver(mesh, ion)
        # Thermal voltage at room temp ~25mV
        Vt = solver.thermal_voltage()
        self.assertGreater(Vt, 0.02)  # > 20 mV
        self.assertLess(Vt, 0.03)  # < 30 mV
        self.assertAlmostEqual(
            solver.electrical_mobility(), ion.mobility_at(310.0), places=18
        )

    def test_pure_diffusion(self):
        """Test with zero electric field (pure diffusion)."""
        mesh = bt.StructuredMesh(50, 0.0, 1e-3)
        ion = bt.IonSpecies("K+", 1, 1.96e-9)
        solver = bt.NernstPlanckSolver(mesh, ion)

        # Gaussian initial concentration
        x = bt.x_nodes(mesh)
        ic = 100.0 * np.exp(-((x - 0.5e-3) ** 2) / (0.1e-3) ** 2)
        solver.set_initial_condition(ic.tolist())

        # Zero electric field (uniform potential = zero field)
        solver.set_uniform_field(0.0)

        # Solve
        solver.solve(1e-6, 100)
        solution = np.array(solver.solution())

        # Should diffuse (peak decreases)
        self.assertLess(np.max(solution), np.max(ic))
        # Mass should be conserved (with the default zero-total-flux walls)
        weights = np.ones_like(solution)
        weights[[0, -1]] = 0.5
        initial_amount = mesh.dx() * np.sum(weights * ic)
        final_amount = mesh.dx() * np.sum(weights * solution)
        self.assertAlmostEqual(final_amount, initial_amount, delta=2e-13)
        self.assertFalse(np.any(np.isnan(solution)))

    def test_boltzmann_equilibrium_is_stationary(self):
        """The fitted flux must exactly preserve c proportional to exp(-z phi/Vt)."""
        mesh = bt.StructuredMesh(80, 0.0, 1e-3)
        ion = bt.IonSpecies("Na+", 1, 1.33e-9)
        solver = bt.NernstPlanckSolver(mesh, ion, 310.0)
        x = bt.x_nodes(mesh)
        potential = 0.04 * x / 1e-3
        concentration = 100.0 * np.exp(-potential / solver.thermal_voltage())
        solver.set_potential_field(potential.tolist())
        solver.set_initial_condition(concentration.tolist())

        current = np.asarray(solver.compute_current_density()).reshape(-1, 2)
        current_scale = bt.constants.FARADAY * ion.diffusivity * 100.0 / mesh.dx()
        self.assertLess(np.max(np.abs(current[:, 0])) / current_scale, 3e-13)
        solver.solve(1e-3, 20)
        np.testing.assert_allclose(
            solver.solution(), concentration, rtol=3e-13, atol=2e-14
        )

    def test_outward_flux_has_declared_mass_balance(self):
        """A positive right-wall flux removes exactly flux times elapsed time."""
        mesh = bt.StructuredMesh(100, 0.0, 1e-3)
        solver = bt.NernstPlanckSolver(mesh, bt.IonSpecies("X+", 1, 1e-9))
        initial = np.full(mesh.num_nodes(), 10.0)
        solver.set_initial_condition(initial.tolist())
        outward_flux = 1e-4
        solver.set_outward_flux_boundary(bt.Boundary.Right, outward_flux)
        dt = 1e-4
        steps = 100
        solver.solve(dt, steps)

        weights = np.ones_like(initial)
        weights[[0, -1]] = 0.5
        initial_amount = mesh.dx() * np.sum(weights * initial)
        final_amount = mesh.dx() * np.sum(weights * np.asarray(solver.solution()))
        self.assertAlmostEqual(
            final_amount,
            initial_amount - outward_flux * dt * steps,
            delta=2e-14,
        )

    def test_electromigration(self):
        """Test ion migration in electric field."""
        mesh = bt.StructuredMesh(50, 0.0, 1e-3)
        ion = bt.IonSpecies("Na+", 1, 1.33e-9)  # Positive ion
        solver = bt.NernstPlanckSolver(mesh, ion)

        # Uniform initial concentration
        ic = 100.0 * np.ones(mesh.num_nodes())
        solver.set_initial_condition(ic.tolist())

        # Constant electric field pointing right (positive x)
        solver.set_uniform_field(1000.0)  # 1000 V/m

        # Dirichlet boundaries
        solver.set_dirichlet_boundary(0, 100.0)  # Left
        solver.set_dirichlet_boundary(1, 100.0)  # Right

        solver.solve(1e-7, 50)
        solution = np.array(solver.solution())

        # Should remain bounded and physical
        self.assertTrue(np.all(solution >= 0))
        self.assertFalse(np.any(np.isnan(solution)))

    def test_stability_check(self):
        """Test stability criterion for electrodiffusion."""
        mesh = bt.StructuredMesh(50, 0.0, 1e-3)
        ion = bt.IonSpecies("Na+", 1, 1.33e-9)
        solver = bt.NernstPlanckSolver(mesh, ion)

        # Check stability for a given dt
        dt = 1e-6
        is_stable = solver.check_stability(dt)
        self.assertIsInstance(is_stable, bool)
        maximum = solver.maximum_stable_time_step()
        self.assertTrue(solver.check_stability(maximum))
        self.assertFalse(solver.check_stability(1.01 * maximum))
        self.assertAlmostEqual(solver.recommended_time_step(0.5), 0.5 * maximum)

    def test_invalid_physical_inputs_are_rejected(self):
        """Temperatures, concentrations, fields, and boundary values need a real domain."""
        with self.assertRaises(ValueError):
            bt.IonSpecies("", 1, 1e-9)
        with self.assertRaises(ValueError):
            bt.IonSpecies("Na+", 1, 1e-9, 0.0)

        mesh = bt.StructuredMesh(8, 0.0, 1.0)
        solver = bt.NernstPlanckSolver(mesh, bt.IonSpecies("Na+", 1, 1e-9))
        with self.assertRaises(ValueError):
            solver.set_initial_condition([-1.0] * mesh.num_nodes())
        with self.assertRaises(ValueError):
            solver.set_potential_field([float("nan")] * mesh.num_nodes())
        with self.assertRaises(ValueError):
            solver.set_dirichlet_boundary(bt.Boundary.Left, -1.0)


class TestMultiIonSolver(unittest.TestCase):
    """Tests for multi-ion Nernst-Planck solver."""

    def test_construction(self):
        """Test multi-ion solver construction."""
        mesh = bt.StructuredMesh(50, 0.0, 1e-3)
        ions = [
            bt.IonSpecies("Na+", 1, 1.33e-9),
            bt.IonSpecies("K+", 1, 1.96e-9),
            bt.IonSpecies("Cl-", -1, 2.03e-9),
        ]
        solver = bt.MultiIonSolver(mesh, ions)
        self.assertIsNotNone(solver)

    def test_num_species(self):
        """Test species count accessor."""
        mesh = bt.StructuredMesh(20, 0.0, 1e-3)
        ions = [
            bt.IonSpecies("Na+", 1, 1.33e-9),
            bt.IonSpecies("Cl-", -1, 2.03e-9),
        ]
        solver = bt.MultiIonSolver(mesh, ions)
        self.assertEqual(solver.num_species(), 2)
        maximum = solver.maximum_stable_time_step()
        self.assertTrue(solver.check_stability(maximum))
        self.assertAlmostEqual(solver.recommended_time_step(0.5), 0.5 * maximum)
        self.assertAlmostEqual(
            solver.electrical_mobility(0), ions[0].mobility_at(310.0), places=18
        )

    def test_ion_accessor(self):
        """Test accessing ion by index."""
        mesh = bt.StructuredMesh(20, 0.0, 1e-3)
        ions = [
            bt.IonSpecies("Na+", 1, 1.33e-9),
            bt.IonSpecies("Cl-", -1, 2.03e-9),
        ]
        solver = bt.MultiIonSolver(mesh, ions)
        # Access ion properties
        self.assertEqual(solver.ion(0).valence, 1)
        self.assertEqual(solver.ion(1).valence, -1)

    def test_set_concentrations(self):
        """Test setting initial concentrations for multiple ions."""
        mesh = bt.StructuredMesh(20, 0.0, 1e-3)
        ions = [
            bt.IonSpecies("Na+", 1, 1.33e-9),
            bt.IonSpecies("Cl-", -1, 2.03e-9),
        ]
        solver = bt.MultiIonSolver(mesh, ions)

        n = mesh.num_nodes()
        na_conc = 140.0 * np.ones(n)  # 140 mM
        cl_conc = 140.0 * np.ones(n)  # 140 mM (electroneutral)

        solver.set_initial_condition(0, na_conc.tolist())
        solver.set_initial_condition(1, cl_conc.tolist())

        # Retrieve and verify
        na_retrieved = np.array(solver.concentration(0))
        cl_retrieved = np.array(solver.concentration(1))

        np.testing.assert_array_almost_equal(na_retrieved, na_conc)
        np.testing.assert_array_almost_equal(cl_retrieved, cl_conc)

    def test_charge_density(self):
        """Test charge density calculation."""
        mesh = bt.StructuredMesh(20, 0.0, 1e-3)
        ions = [
            bt.IonSpecies("Na+", 1, 1.33e-9),
            bt.IonSpecies("Cl-", -1, 2.03e-9),
        ]
        solver = bt.MultiIonSolver(mesh, ions)

        n = mesh.num_nodes()
        # Set electroneutral initial conditions
        solver.set_initial_condition(0, (140.0 * np.ones(n)).tolist())
        solver.set_initial_condition(1, (140.0 * np.ones(n)).tolist())

        # Charge density should be near zero
        rho = np.array(solver.charge_density())
        self.assertLess(np.max(np.abs(rho)), 1e-10)

    def test_species_conserve_independently_in_prescribed_potential(self):
        """The multi-ion class is conservative but intentionally not self-coupled."""
        mesh = bt.StructuredMesh(48, 0.0, 8e-4)
        ions = [
            bt.IonSpecies("X+", 1, 1.2e-9),
            bt.IonSpecies("Y-", -1, 1.8e-9),
        ]
        solver = bt.MultiIonSolver(mesh, ions, 305.0)
        x = bt.x_nodes(mesh)
        xi = x / 8e-4
        fields = [
            4.0 + 3.0 * np.sin(np.pi * xi) ** 2,
            7.0 + 2.0 * np.exp(-(((xi - 0.65) / 0.15) ** 2)),
        ]
        solver.set_potential_field((0.02 * (xi - 0.3 * xi**2)).tolist())
        for species, field in enumerate(fields):
            solver.set_initial_condition(species, field.tolist())

        weights = np.ones(mesh.num_nodes())
        weights[[0, -1]] = 0.5
        initial_amounts = [mesh.dx() * np.sum(weights * field) for field in fields]
        solver.solve(1e-4, 80)
        for species, initial_amount in enumerate(initial_amounts):
            concentration = np.asarray(solver.concentration(species))
            final_amount = mesh.dx() * np.sum(weights * concentration)
            self.assertAlmostEqual(final_amount, initial_amount, delta=2e-13)
            self.assertGreaterEqual(np.min(concentration), 0.0)

    def test_unimplemented_electroneutrality_fails_loudly(self):
        """The class must not imply a self-consistent field it does not solve."""
        mesh = bt.StructuredMesh(8, 0.0, 1.0)
        solver = bt.MultiIonSolver(
            mesh,
            [bt.IonSpecies("Na+", 1, 1e-9), bt.IonSpecies("Cl-", -1, 1e-9)],
        )
        with self.assertRaises(RuntimeError):
            solver.set_electroneutrality_mode(True)
        with self.assertRaises(RuntimeError):
            solver.set_electroneutrality_mode(False, background_charge=1.0)


class TestGHKEquation(unittest.TestCase):
    """Tests for Goldman-Hodgkin-Katz utilities."""

    def test_ghk_voltage(self):
        """Test GHK voltage calculation."""
        # Typical neuron values
        # Inside: [K+]=140, [Na+]=10, [Cl-]=10
        # Outside: [K+]=5, [Na+]=145, [Cl-]=110

        P_K = 1.0
        P_Na = 0.04
        P_Cl = 0.45

        # Should give resting potential around -70 mV
        V = bt.ghk.ghk_voltage(
            P_K=P_K,
            K_in=140,
            K_out=5,
            P_Na=P_Na,
            Na_in=10,
            Na_out=145,
            P_Cl=P_Cl,
            Cl_in=10,
            Cl_out=110,
        )

        # Resting potential should be negative
        self.assertLess(V, 0)
        # Should be around -60 to -80 mV
        self.assertLess(V * 1000, -50)  # < -50 mV
        self.assertGreater(V * 1000, -90)  # > -90 mV

    def test_nernst_potential(self):
        """Test Nernst potential calculation for single ion."""
        # E = (RT/zF) * ln(c_out/c_in)
        # For K+ at 310K with [K+]out=5mM, [K+]in=140mM
        # E_K ≈ -90 mV
        E_K = bt.ghk.nernst_potential(z=1, c_in=140.0, c_out=5.0, temperature=310.0)
        self.assertLess(E_K * 1000, -80)  # < -80 mV
        self.assertGreater(E_K * 1000, -100)  # > -100 mV

    def test_single_permeant_ion_reduces_to_nernst(self):
        """GHK must recover Nernst in its one-permeant-ion limits."""
        nernst_k = bt.ghk.nernst_potential(1, 140.0, 5.0, 310.0)
        ghk_k = bt.ghk.ghk_voltage(1.0, 140.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 310.0)
        self.assertAlmostEqual(ghk_k, nernst_k, places=15)

        nernst_cl = bt.ghk.nernst_potential(-1, 10.0, 110.0, 310.0)
        ghk_cl = bt.ghk.ghk_voltage(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 10.0, 110.0, 310.0
        )
        self.assertAlmostEqual(ghk_cl, nernst_cl, places=15)

    def test_invalid_ghk_domain_is_rejected(self):
        with self.assertRaises(ValueError):
            bt.ghk.ghk_voltage(0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0)


class TestPhysicalConstants(unittest.TestCase):
    """Tests for physical constants module."""

    def test_faraday_constant(self):
        """Test Faraday constant value."""
        self.assertAlmostEqual(bt.constants.FARADAY, 96485.0, places=0)

    def test_gas_constant(self):
        """Test gas constant value."""
        self.assertAlmostEqual(bt.constants.GAS_CONSTANT, 8.314, places=2)

    def test_boltzmann_constant(self):
        """Test Boltzmann constant value."""
        self.assertAlmostEqual(bt.constants.BOLTZMANN, 1.38e-23, delta=1e-25)


if __name__ == "__main__":
    unittest.main()
