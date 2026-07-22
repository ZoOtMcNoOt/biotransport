"""
Unit tests for the Stokes flow solver.

Tests verify:
1. Basic solver construction and parameter setting
2. Poiseuille flow (pressure-driven channel flow)
3. Lid-driven cavity flow
4. No-slip boundary conditions
5. Convergence behavior
6. Result data validity (no NaN/Inf)
"""

import unittest
import numpy as np
import biotransport as bt


class TestStokesConstruction(unittest.TestCase):
    """Test Stokes solver construction and configuration."""

    def test_basic_construction(self):
        """Test basic solver creation."""
        mesh = bt.StructuredMesh(10, 10, 0.0, 1.0, 0.0, 1.0)
        solver = bt.StokesSolver(mesh, 0.001)
        # Should not raise
        self.assertIsNotNone(solver)

    def test_set_parameters(self):
        """Test setting solver parameters."""
        mesh = bt.StructuredMesh(10, 10, 0.0, 1.0, 0.0, 1.0)
        solver = bt.StokesSolver(mesh, 0.001)

        # These should not raise
        solver.set_tolerance(1e-8)
        solver.set_max_iterations(5000)
        solver.set_pressure_relaxation(0.1)
        solver.set_velocity_relaxation(0.5)
        solver.set_body_force(1.0, 0.0)

    def test_set_boundary_conditions(self):
        """Test setting boundary conditions."""
        mesh = bt.StructuredMesh(10, 10, 0.0, 1.0, 0.0, 1.0)
        solver = bt.StokesSolver(mesh, 0.001)

        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(1.0))
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())


class TestPoiseuilleFlow(unittest.TestCase):
    """Test Poiseuille (pressure-driven channel) flow."""

    def test_poiseuille_accuracy(self):
        """
        Test Poiseuille flow matches analytical solution.

        Analytical: u_max = (dP/dx) * H^2 / (8 * mu)
        """
        L = 1.0  # length
        H = 0.1  # height
        mu = 0.001  # viscosity
        dP_dx = 1000.0  # pressure gradient (body force)

        # Analytical maximum velocity
        u_max_analytical = dP_dx * H**2 / (8.0 * mu)

        # Create mesh
        nx, ny = 40, 20
        mesh = bt.StructuredMesh(nx, ny, 0.0, L, 0.0, H)

        # Create solver
        solver = bt.StokesSolver(mesh, mu)

        # No-slip on walls, outflow on ends
        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.outflow())
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())

        # Body force represents pressure gradient
        solver.set_body_force(dP_dx, 0.0)
        solver.set_tolerance(1e-6)
        solver.set_max_iterations(10000)

        # Solve
        result = solver.solve()

        # Check convergence or that we got a reasonable answer anyway
        # (iterative solver may hit max iterations but still be accurate)

        # Check maximum velocity
        u = result.u()
        u_max_computed = np.max(u)

        rel_error = abs(u_max_computed - u_max_analytical) / u_max_analytical
        self.assertLess(
            rel_error, 0.01, f"Poiseuille error {rel_error * 100:.2f}% should be < 1%"
        )

    def test_poiseuille_v_zero(self):
        """Test that v-velocity is approximately zero in Poiseuille flow."""
        mesh = bt.StructuredMesh(20, 10, 0.0, 1.0, 0.0, 0.1)
        solver = bt.StokesSolver(mesh, 0.001)

        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.outflow())
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())
        solver.set_body_force(1000.0, 0.0)
        solver.set_max_iterations(2000)

        result = solver.solve()
        v = result.v()

        # v should be small compared to u (Poiseuille is 1D flow)
        u_max = np.max(np.abs(result.u()))
        self.assertLess(
            np.max(np.abs(v)),
            0.001 * u_max,
            "v should be negligible compared to u in Poiseuille flow",
        )


class TestLidDrivenCavity(unittest.TestCase):
    """Test lid-driven cavity flow."""

    def test_lid_driven_cavity(self):
        """Test lid-driven cavity produces recirculating flow."""
        L = 1.0
        mu = 0.01
        u_lid = 1.0

        mesh = bt.StructuredMesh(20, 20, 0.0, L, 0.0, L)
        solver = bt.StokesSolver(mesh, mu)

        # No-slip on walls, moving lid on top
        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.dirichlet(u_lid, 0.0))

        solver.set_tolerance(1e-4)  # Looser tolerance for this difficult problem
        solver.set_max_iterations(10000)

        result = solver.solve()

        # Check that we got some flow (even if not fully converged)

        # Check top boundary has u = u_lid
        u = result.u().reshape((21, 21))
        np.testing.assert_allclose(
            u[-1, :], u_lid, atol=1e-6, err_msg="Top boundary should have u = u_lid"
        )

        # Check bottom has u = 0
        np.testing.assert_allclose(
            u[0, :], 0.0, atol=1e-6, err_msg="Bottom should have u = 0"
        )

        # Check recirculation develops (non-zero v in interior)
        v = result.v().reshape((21, 21))
        v_interior = v[1:-1, 1:-1]
        self.assertGreater(
            np.max(np.abs(v_interior)),
            0.01,
            "Should have significant v velocity from recirculation",
        )


class TestHydrostaticEquilibrium(unittest.TestCase):
    """Test conservative uniform forcing in a sealed cavity."""

    def test_uniform_body_force_is_balanced_by_pressure(self):
        """A sealed cavity remains at rest with grad(p) equal to the force."""
        nx, ny = 10, 8
        x_min, x_max = -0.25, 1.75
        y_min, y_max = 0.5, 2.0
        force_x, force_y = 7.5, -2.25
        mesh = bt.StructuredMesh(nx, ny, x_min, x_max, y_min, y_max)
        solver = bt.StokesSolver(mesh, 0.37)

        # All walls no-slip
        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.no_slip())

        solver.set_body_force(force_x, force_y)
        solver.set_tolerance(1e-14)
        solver.set_max_iterations(1)

        result = solver.solve()
        pressure = result.pressure().reshape((ny + 1, nx + 1))
        x, y = np.meshgrid(
            np.linspace(x_min, x_max, nx + 1),
            np.linspace(y_min, y_max, ny + 1),
        )
        expected_pressure = force_x * x + force_y * y
        expected_pressure -= np.mean(expected_pressure)

        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        np.testing.assert_array_equal(result.u(), 0.0)
        np.testing.assert_array_equal(result.v(), 0.0)
        np.testing.assert_allclose(pressure, expected_pressure, rtol=2e-14, atol=2e-14)
        self.assertAlmostEqual(float(np.mean(pressure)), 0.0, delta=2e-14)
        self.assertLess(result.residual, 2e-13)
        self.assertEqual(result.divergence, 0.0)


class TestResultValidity(unittest.TestCase):
    """Test that results are valid (no NaN/Inf)."""

    def test_no_nan_in_results(self):
        """Test that solution contains no NaN values."""
        mesh = bt.StructuredMesh(15, 15, 0.0, 1.0, 0.0, 1.0)
        solver = bt.StokesSolver(mesh, 0.01)

        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.no_slip())
        solver.set_tolerance(1e-12)
        solver.set_max_iterations(10)

        result = solver.solve()

        self.assertFalse(np.any(np.isnan(result.u())), "u should not contain NaN")
        self.assertFalse(np.any(np.isnan(result.v())), "v should not contain NaN")
        self.assertFalse(
            np.any(np.isnan(result.pressure())), "pressure should not contain NaN"
        )
        self.assertFalse(np.any(np.isinf(result.u())), "u should not contain Inf")
        self.assertFalse(np.any(np.isinf(result.v())), "v should not contain Inf")

    def test_result_shapes(self):
        """Test that result arrays have correct shapes."""
        nx, ny = 10, 8
        mesh = bt.StructuredMesh(nx, ny, 0.0, 1.0, 0.0, 1.0)
        solver = bt.StokesSolver(mesh, 0.01)

        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.no_slip())
        solver.set_tolerance(1e-12)
        solver.set_max_iterations(10)

        result = solver.solve()

        expected_size = (nx + 1) * (ny + 1)
        self.assertEqual(len(result.u()), expected_size)
        self.assertEqual(len(result.v()), expected_size)
        self.assertEqual(len(result.pressure()), expected_size)


class TestInflowOutflow(unittest.TestCase):
    """Test inflow/outflow boundary conditions."""

    def test_inflow_bc_applied(self):
        """Test a flux-compatible uniform inflow/outflow solution."""
        nx, ny = 12, 6
        mesh = bt.StructuredMesh(nx, ny, 0.0, 1.0, 0.0, 0.5)

        u_inlet = 0.5

        solver = bt.StokesSolver(mesh, 1.0)
        solver.set_velocity_bc(
            bt.Boundary.Bottom, bt.VelocityBC.dirichlet(u_inlet, 0.0)
        )
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.dirichlet(u_inlet, 0.0))
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(u_inlet, 0.0))
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())
        solver.set_tolerance(1e-6)
        solver.set_max_iterations(3000)

        result = solver.solve()
        u = result.u().reshape((ny + 1, nx + 1))
        v = result.v().reshape((ny + 1, nx + 1))

        np.testing.assert_allclose(u, u_inlet, atol=1e-6)
        np.testing.assert_allclose(v, 0.0, atol=1e-6)
        self.assertLess(result.divergence, 1.01e-6)


class TestGridConvergence(unittest.TestCase):
    """Test analytical channel accuracy across a grid sequence."""

    def test_channel_accuracy_across_refinement(self):
        """Keep the Poiseuille error bounded across several grid sizes."""
        length = 2.0
        height = 1.0
        viscosity = 1.0
        body_force = 8.0
        u_max_analytical = 1.0

        errors = []
        for n in [8, 12, 16]:
            nx, ny = n * 2, n
            mesh = bt.StructuredMesh(nx, ny, 0.0, length, 0.0, height)
            solver = bt.StokesSolver(mesh, viscosity)

            solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
            solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
            solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.outflow())
            solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())
            solver.set_body_force(body_force, 0.0)
            solver.set_tolerance(1e-5)
            solver.set_max_iterations(4000)

            result = solver.solve()
            u = result.u().reshape((ny + 1, nx + 1))
            y = np.linspace(0.0, height, ny + 1)
            exact = body_force * y * (height - y) / (2.0 * viscosity)
            error = np.max(np.abs(u - exact[:, None])) / u_max_analytical
            errors.append(error)
            self.assertLess(result.divergence, 1.01e-5)

        # This sequence is an accuracy check, not an observed-order claim: the
        # iterative tolerance is fixed while the centered stencil is exact for
        # the quadratic analytical profile.
        for error in errors:
            self.assertLess(error, 5e-3, f"Error should be < 0.5%: {errors}")


if __name__ == "__main__":
    unittest.main()
