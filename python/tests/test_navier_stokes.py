"""
Unit tests for the Navier-Stokes flow solver.

Tests verify:
1. Basic solver construction and parameter setting
2. Channel flow development
3. Stability behavior
4. Time stepping
5. Result data validity (no NaN/Inf)
"""

import unittest
import numpy as np
import biotransport as bt


class TestNavierStokesConstruction(unittest.TestCase):
    """Test Navier-Stokes solver construction and configuration."""

    def test_basic_construction(self):
        """Test basic solver creation."""
        mesh = bt.StructuredMesh(10, 10, 0.0, 1.0, 0.0, 1.0)
        rho = 1000.0  # kg/m^3
        mu = 0.001  # Pa.s
        solver = bt.NavierStokesSolver(mesh, rho, mu)
        self.assertIsNotNone(solver)

    def test_set_boundary_conditions(self):
        """Test setting boundary conditions."""
        mesh = bt.StructuredMesh(10, 10, 0.0, 1.0, 0.0, 1.0)
        solver = bt.NavierStokesSolver(mesh, 1000.0, 0.001)

        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(0.1))
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())


class TestChannelFlow(unittest.TestCase):
    """Test channel flow development."""

    def test_channel_flow_stable(self):
        """Test that channel flow is stable."""
        # Microfluidic channel
        L = 0.001  # 1 mm
        H = 0.0005  # 0.5 mm
        rho = 1000.0
        mu = 0.001
        u_inlet = 0.1  # 0.1 m/s

        mesh = bt.StructuredMesh(20, 10, 0.0, L, 0.0, H)
        solver = bt.NavierStokesSolver(mesh, rho, mu)

        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(u_inlet, 0.0))
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())

        t_end = 0.001  # 1 ms
        result = solver.solve(t_end)

        self.assertTrue(result.stable, "Channel flow should be stable")
        self.assertGreater(result.time_steps, 0, "Should take some time steps")

    def test_channel_flow_produces_velocity(self):
        """Test that channel flow produces non-zero velocity."""
        mesh = bt.StructuredMesh(20, 10, 0.0, 0.001, 0.0, 0.0005)
        solver = bt.NavierStokesSolver(mesh, 1000.0, 0.001)

        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(0.1))
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())

        result = solver.solve(0.001)
        u = result.u()

        self.assertGreater(np.max(u), 0.0, "Should have positive velocity")
        self.assertFalse(np.any(np.isnan(u)), "Should not have NaN")


class TestLidDrivenCavity(unittest.TestCase):
    """Test lid-driven cavity (time-dependent)."""

    def test_lid_driven_cavity_stable(self):
        """Test that lid-driven cavity is stable with high viscosity."""
        L = 0.001  # 1 mm
        rho = 1000.0
        mu = 0.01  # Higher viscosity for stability
        u_lid = 0.1

        mesh = bt.StructuredMesh(15, 15, 0.0, L, 0.0, L)
        solver = bt.NavierStokesSolver(mesh, rho, mu)

        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.dirichlet(u_lid, 0.0))

        result = solver.solve(0.01)  # 10 ms

        self.assertTrue(result.stable, "Cavity should be stable with high viscosity")

    def test_lid_bc_applied(self):
        """Test that lid BC is correctly applied."""
        nx, ny = 15, 15
        mesh = bt.StructuredMesh(nx, ny, 0.0, 0.001, 0.0, 0.001)
        u_lid = 0.1

        solver = bt.NavierStokesSolver(mesh, 1000.0, 0.01)
        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.dirichlet(u_lid, 0.0))

        result = solver.solve(0.01)
        u = result.u().reshape((ny + 1, nx + 1))

        # Check lid boundary has u = u_lid
        np.testing.assert_allclose(
            u[-1, :], u_lid, atol=1e-4, err_msg="Lid should have u = u_lid"
        )


class TestReynoldsNumber(unittest.TestCase):
    """Test Reynolds number effects on stability."""

    def test_low_reynolds_stable(self):
        """Test that low Reynolds number flow is stable."""
        L = 0.001
        rho = 1000.0
        mu = 0.01  # Re ~ 10
        u = 0.1

        mesh = bt.StructuredMesh(10, 10, 0.0, L, 0.0, L)
        solver = bt.NavierStokesSolver(mesh, rho, mu)

        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(u))
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())

        result = solver.solve(0.001)

        self.assertTrue(result.stable, "Low Re flow should be stable")


class TestResultValidity(unittest.TestCase):
    """Test that results are valid."""

    def test_no_nan_in_results(self):
        """Test that solution contains no NaN values."""
        mesh = bt.StructuredMesh(10, 10, 0.0, 0.001, 0.0, 0.001)
        solver = bt.NavierStokesSolver(mesh, 1000.0, 0.01)

        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.dirichlet(0.1, 0.0))
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.no_slip())

        result = solver.solve(0.01)

        self.assertFalse(np.any(np.isnan(result.u())), "u should not contain NaN")
        self.assertFalse(np.any(np.isnan(result.v())), "v should not contain NaN")
        self.assertFalse(
            np.any(np.isnan(result.pressure())), "pressure should not contain NaN"
        )

    def test_result_shapes(self):
        """Test that result arrays have correct shapes."""
        nx, ny = 10, 8
        mesh = bt.StructuredMesh(nx, ny, 0.0, 0.001, 0.0, 0.001)
        solver = bt.NavierStokesSolver(mesh, 1000.0, 0.01)

        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(0.1))
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())

        result = solver.solve(0.001)

        expected_size = (nx + 1) * (ny + 1)
        self.assertEqual(len(result.u()), expected_size)
        self.assertEqual(len(result.v()), expected_size)
        self.assertEqual(len(result.pressure()), expected_size)

    def test_result_attributes(self):
        """Test that result has expected attributes."""
        mesh = bt.StructuredMesh(10, 10, 0.0, 0.001, 0.0, 0.001)
        solver = bt.NavierStokesSolver(mesh, 1000.0, 0.01)

        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(0.1))
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())

        result = solver.solve(0.001)

        # Check attributes exist
        self.assertTrue(hasattr(result, "stable"))
        self.assertTrue(hasattr(result, "time"))
        self.assertTrue(hasattr(result, "time_steps"))


class TestTimeStepping(unittest.TestCase):
    """Test time stepping behavior."""

    def test_longer_sim_more_steps(self):
        """Test that longer simulation has more time steps."""
        mesh = bt.StructuredMesh(10, 10, 0.0, 0.001, 0.0, 0.001)
        solver = bt.NavierStokesSolver(mesh, 1000.0, 0.001)

        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(0.1))
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())

        r1 = solver.solve(0.0001)
        r2 = solver.solve(0.001)

        self.assertGreaterEqual(
            r2.time_steps, r1.time_steps, "Longer sim should have >= time steps"
        )


class TestBodyForce(unittest.TestCase):
    """Test body force functionality."""

    def test_body_force_accelerates_flow(self):
        """Test that body force accelerates flow."""
        mesh = bt.StructuredMesh(10, 10, 0.0, 0.001, 0.0, 0.001)
        solver = bt.NavierStokesSolver(mesh, 1000.0, 0.01)

        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.no_slip())
        solver.set_body_force(1000.0, 0.0)

        result = solver.solve(0.01)
        u = result.u()

        self.assertGreater(np.max(u), 0.0, "Body force should accelerate flow")
        self.assertFalse(np.any(np.isnan(u)), "Should not have NaN")


if __name__ == "__main__":
    unittest.main()
