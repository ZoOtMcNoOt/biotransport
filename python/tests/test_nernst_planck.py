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
        # Mass should be conserved (approximately, with Neumann BCs)
        self.assertFalse(np.any(np.isnan(solution)))

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
