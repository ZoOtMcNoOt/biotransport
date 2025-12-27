"""
Tests for membrane diffusion solver.

Verifies analytical solutions for steady-state membrane transport:
  j = D * Phi * (C_left - C_right) / L
  P = D * Phi / L
"""

import unittest
from biotransport import (
    MembraneDiffusion1DSolver,
    MultiLayerMembraneSolver,
    renkin_hindrance,
)


class TestMembraneDiffusion(unittest.TestCase):
    """Tests for steady-state membrane diffusion."""

    def test_simple_membrane_flux(self):
        """Test flux calculation against analytical solution."""
        L = 100e-6  # 100 um
        D = 1e-10  # m^2/s
        Phi = 0.5
        C_left = 10.0
        C_right = 2.0

        solver = (
            MembraneDiffusion1DSolver()
            .set_membrane_thickness(L)
            .set_diffusivity(D)
            .set_partition_coefficient(Phi)
            .set_left_concentration(C_left)
            .set_right_concentration(C_right)
        )

        result = solver.solve()

        # Analytical: j = D * Phi * (C_left - C_right) / L
        j_analytical = D * Phi * (C_left - C_right) / L
        P_analytical = D * Phi / L

        self.assertAlmostEqual(result.flux, j_analytical, places=15)
        self.assertAlmostEqual(result.permeability, P_analytical, places=15)
        self.assertAlmostEqual(result.effective_diffusivity, D, places=15)

    def test_permeability_calculation(self):
        """Test permeability P = D * Phi / L."""
        L = 50e-6
        D = 5e-11
        Phi = 0.8

        solver = (
            MembraneDiffusion1DSolver()
            .set_membrane_thickness(L)
            .set_diffusivity(D)
            .set_partition_coefficient(Phi)
        )

        P = solver.compute_permeability()
        P_expected = D * Phi / L

        self.assertAlmostEqual(P, P_expected, places=15)

    def test_concentration_profile_linear(self):
        """Verify concentration profile is linear at steady state."""
        solver = (
            MembraneDiffusion1DSolver()
            .set_membrane_thickness(1e-4)
            .set_diffusivity(1e-9)
            .set_partition_coefficient(0.5)
            .set_left_concentration(100.0)
            .set_right_concentration(20.0)
            .set_num_nodes(101)
        )

        result = solver.solve()
        x = result.x()
        c = result.concentration()

        # Check linearity: C = C_left*Phi + (C_right*Phi - C_left*Phi) * x/L
        C_left_mem = 0.5 * 100.0
        C_right_mem = 0.5 * 20.0
        L = 1e-4

        for i in range(len(x)):
            c_expected = C_left_mem + (C_right_mem - C_left_mem) * (x[i] / L)
            self.assertAlmostEqual(c[i], c_expected, places=10)

    def test_zero_flux_equal_concentrations(self):
        """Flux should be zero when concentrations are equal."""
        solver = (
            MembraneDiffusion1DSolver()
            .set_membrane_thickness(1e-4)
            .set_diffusivity(1e-10)
            .set_partition_coefficient(0.3)
            .set_left_concentration(5.0)
            .set_right_concentration(5.0)
        )

        result = solver.solve()
        self.assertAlmostEqual(result.flux, 0.0, places=15)

    def test_renkin_hindrance_limits(self):
        """Test Renkin hindrance at limiting values."""
        # No hindrance when lambda = 0
        self.assertAlmostEqual(renkin_hindrance(0.0), 1.0, places=10)

        # Complete hindrance when lambda >= 1
        self.assertAlmostEqual(renkin_hindrance(1.0), 0.0, places=10)

        # Monotonically decreasing
        H1 = renkin_hindrance(0.2)
        H2 = renkin_hindrance(0.5)
        H3 = renkin_hindrance(0.8)
        self.assertGreater(H1, H2)
        self.assertGreater(H2, H3)

        # Known value check: H(0.5) from Renkin equation
        # H = (1-λ)² × (1 - 2.104λ + 2.09λ³ - 0.95λ⁵)
        self.assertAlmostEqual(renkin_hindrance(0.5), 0.04489, places=4)

    def test_hindered_diffusion_reduces_flux(self):
        """Hindered diffusion should reduce effective diffusivity and flux."""
        L = 100e-6
        D = 1e-10
        Phi = 1.0
        solute_r = 3e-9
        pore_r = 10e-9

        solver_bulk = (
            MembraneDiffusion1DSolver()
            .set_membrane_thickness(L)
            .set_diffusivity(D)
            .set_partition_coefficient(Phi)
            .set_left_concentration(10.0)
            .set_right_concentration(0.0)
        )

        solver_hindered = (
            MembraneDiffusion1DSolver()
            .set_membrane_thickness(L)
            .set_diffusivity(D)
            .set_partition_coefficient(Phi)
            .set_left_concentration(10.0)
            .set_right_concentration(0.0)
            .set_hindered_diffusion(solute_r, pore_r)
        )

        result_bulk = solver_bulk.solve()
        result_hindered = solver_hindered.solve()

        # Hindered flux should be less
        self.assertLess(result_hindered.flux, result_bulk.flux)

        # Check effective diffusivity is reduced by H factor
        H = renkin_hindrance(solute_r / pore_r)
        self.assertAlmostEqual(result_hindered.effective_diffusivity, D * H, places=15)

    def test_multilayer_resistance_series(self):
        """Test multi-layer membrane with resistances in series."""
        # Two identical layers should have double resistance
        L = 50e-6
        D = 1e-10
        Phi = 1.0

        # Single layer
        single = (
            MembraneDiffusion1DSolver()
            .set_membrane_thickness(L)
            .set_diffusivity(D)
            .set_partition_coefficient(Phi)
            .set_left_concentration(10.0)
            .set_right_concentration(0.0)
        )
        result_single = single.solve()

        # Double thickness single layer
        double_thick = (
            MembraneDiffusion1DSolver()
            .set_membrane_thickness(2 * L)
            .set_diffusivity(D)
            .set_partition_coefficient(Phi)
            .set_left_concentration(10.0)
            .set_right_concentration(0.0)
        )
        result_double = double_thick.solve()

        # Two identical layers
        multi = (
            MultiLayerMembraneSolver()
            .add_layer(L, D, Phi)
            .add_layer(L, D, Phi)
            .set_left_concentration(10.0)
            .set_right_concentration(0.0)
        )
        result_multi = multi.solve()

        # Flux should be half (double resistance)
        self.assertAlmostEqual(result_multi.flux, result_single.flux / 2, places=10)
        self.assertAlmostEqual(result_multi.flux, result_double.flux, places=10)

    def test_multilayer_heterogeneous(self):
        """Test multi-layer with different properties."""
        # Two layers with different D and Phi
        L1, D1, Phi1 = 100e-6, 1e-10, 0.5
        L2, D2, Phi2 = 200e-6, 2e-10, 0.8
        C_left, C_right = 10.0, 0.0

        solver = (
            MultiLayerMembraneSolver()
            .add_layer(L1, D1, Phi1)
            .add_layer(L2, D2, Phi2)
            .set_left_concentration(C_left)
            .set_right_concentration(C_right)
        )

        result = solver.solve()

        # Analytical: R_total = L1/(D1*Phi1) + L2/(D2*Phi2)
        R1 = L1 / (D1 * Phi1)
        R2 = L2 / (D2 * Phi2)
        R_total = R1 + R2
        j_expected = (C_left - C_right) / R_total
        P_expected = 1.0 / R_total

        self.assertAlmostEqual(result.flux, j_expected, places=12)
        self.assertAlmostEqual(result.permeability, P_expected, places=12)

    def test_fluent_api_chaining(self):
        """Test that fluent API returns self for chaining."""
        solver = MembraneDiffusion1DSolver()
        result = (
            solver.set_membrane_thickness(100e-6)
            .set_diffusivity(1e-10)
            .set_partition_coefficient(0.5)
            .set_left_concentration(10.0)
            .set_right_concentration(0.0)
            .set_num_nodes(51)
        )
        # Chained result should be the same solver
        self.assertIsInstance(result, MembraneDiffusion1DSolver)


if __name__ == "__main__":
    unittest.main()
