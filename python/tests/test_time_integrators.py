"""Tests for higher-order time integration methods (RK4, Heun).

These tests verify:
1. Basic functionality of integrators
2. Correct convergence order (4th for RK4, 2nd for Heun)
3. Accuracy comparison with analytical solutions
"""

import numpy as np
import pytest

import biotransport as bt


# ============================================================================
# Test RK4 step function
# ============================================================================


class TestRK4Step:
    """Tests for the standalone rk4_step function."""

    def test_simple_ode(self):
        """Test RK4 on dy/dt = -y with y(0) = 1 (solution: e^-t)."""
        u = np.array([1.0])

        def rhs(u_state, t):
            return -u_state

        # One step
        dt = 0.1
        u_new = bt.rk4_step(u, rhs, 0.0, dt)

        # Analytical: e^{-0.1} ≈ 0.9048
        expected = np.exp(-dt)
        assert abs(u_new[0] - expected) < 1e-6, f"Expected {expected}, got {u_new[0]}"

    def test_oscillator(self):
        """Test RK4 on simple harmonic oscillator."""
        # dy/dt = v, dv/dt = -y  =>  y'' = -y
        # Solution: y = cos(t), v = -sin(t) for y(0)=1, v(0)=0
        u = np.array([1.0, 0.0])  # [y, v]

        def rhs(u_state, t):
            return np.array([u_state[1], -u_state[0]])

        dt = 0.1
        for _ in range(10):  # 10 steps to t=1.0
            u = bt.rk4_step(u, rhs, 0.0, dt)

        t_final = 1.0
        y_expected = np.cos(t_final)
        v_expected = -np.sin(t_final)

        # RK4 should be very accurate
        assert abs(u[0] - y_expected) < 1e-6, f"y error: {abs(u[0] - y_expected)}"
        assert abs(u[1] - v_expected) < 1e-6, f"v error: {abs(u[1] - v_expected)}"

    def test_fourth_order_convergence(self):
        """Verify RK4 achieves 4th-order convergence."""
        # Solve dy/dt = y from t=0 to t=1 (solution: e^t)
        t_end = 1.0

        def rhs(u_state, t):
            return u_state

        errors = []
        dts = [0.1, 0.05, 0.025]

        for dt in dts:
            u = np.array([1.0])
            t = 0.0
            steps = int(t_end / dt)

            for _ in range(steps):
                u = bt.rk4_step(u, rhs, t, dt)
                t += dt

            error = abs(u[0] - np.exp(t_end))
            errors.append(error)

        # Compute convergence order
        order_1 = np.log(errors[0] / errors[1]) / np.log(dts[0] / dts[1])
        order_2 = np.log(errors[1] / errors[2]) / np.log(dts[1] / dts[2])

        # Should be close to 4
        assert order_1 > 3.8, f"Order 1 = {order_1}, expected ~4"
        assert order_2 > 3.8, f"Order 2 = {order_2}, expected ~4"


# ============================================================================
# Test Heun step function
# ============================================================================


class TestHeunStep:
    """Tests for the standalone heun_step function."""

    def test_simple_ode(self):
        """Test Heun on dy/dt = -y."""
        u = np.array([1.0])

        def rhs(u_state, t):
            return -u_state

        dt = 0.1
        u_new = bt.heun_step(u, rhs, 0.0, dt)

        # Heun should be accurate (2nd order)
        expected = np.exp(-dt)
        assert abs(u_new[0] - expected) < 1e-3, f"Expected {expected}, got {u_new[0]}"

    def test_second_order_convergence(self):
        """Verify Heun achieves 2nd-order convergence."""
        t_end = 1.0

        def rhs(u_state, t):
            return u_state

        errors = []
        dts = [0.1, 0.05, 0.025]

        for dt in dts:
            u = np.array([1.0])
            t = 0.0
            steps = int(t_end / dt)

            for _ in range(steps):
                u = bt.heun_step(u, rhs, t, dt)
                t += dt

            error = abs(u[0] - np.exp(t_end))
            errors.append(error)

        # Compute convergence order
        order_1 = np.log(errors[0] / errors[1]) / np.log(dts[0] / dts[1])
        order_2 = np.log(errors[1] / errors[2]) / np.log(dts[1] / dts[2])

        # Should be close to 2
        assert order_1 > 1.8, f"Order 1 = {order_1}, expected ~2"
        assert order_2 > 1.8, f"Order 2 = {order_2}, expected ~2"


# ============================================================================
# Test Euler step function
# ============================================================================


class TestEulerStep:
    """Tests for the standalone euler_step function."""

    def test_simple_ode(self):
        """Test Euler on dy/dt = -y."""
        u = np.array([1.0])

        def rhs(u_state, t):
            return -u_state

        dt = 0.1
        u_new = bt.euler_step(u, rhs, 0.0, dt)

        # Euler: u_new = u + dt * (-u) = u * (1 - dt) = 0.9
        expected = 1.0 * (1 - dt)
        assert abs(u_new[0] - expected) < 1e-10

    def test_first_order_convergence(self):
        """Verify Euler achieves 1st-order convergence."""
        t_end = 1.0

        def rhs(u_state, t):
            return u_state

        errors = []
        dts = [0.1, 0.05, 0.025]

        for dt in dts:
            u = np.array([1.0])
            t = 0.0
            steps = int(t_end / dt)

            for _ in range(steps):
                u = bt.euler_step(u, rhs, t, dt)
                t += dt

            error = abs(u[0] - np.exp(t_end))
            errors.append(error)

        # Compute convergence order
        order_1 = np.log(errors[0] / errors[1]) / np.log(dts[0] / dts[1])
        order_2 = np.log(errors[1] / errors[2]) / np.log(dts[1] / dts[2])

        # Should be close to 1
        assert 0.8 < order_1 < 1.2, f"Order 1 = {order_1}, expected ~1"
        assert 0.8 < order_2 < 1.2, f"Order 2 = {order_2}, expected ~1"


# ============================================================================
# Test RK4Integrator class
# ============================================================================


class TestRK4Integrator:
    """Tests for the RK4Integrator class with transport problems."""

    def test_initialization(self):
        """Test RK4Integrator can be constructed."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)
        ic = bt.gaussian(mesh, center=0.5, width=0.1)
        problem = bt.Problem(mesh).diffusivity(0.01).initial_condition(ic)

        integrator = bt.RK4Integrator(problem)
        assert integrator is not None
        assert integrator.D == 0.01

    def test_max_stable_dt(self):
        """Test that max_stable_dt returns a positive value."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)
        ic = bt.gaussian(mesh, center=0.5, width=0.1)
        problem = bt.Problem(mesh).diffusivity(0.01).initial_condition(ic)

        integrator = bt.RK4Integrator(problem)
        dt = integrator.max_stable_dt()

        assert dt > 0
        # RK4 should allow larger dt than Euler CFL
        dx = 1.0 / 50
        dt_euler = dx * dx / (2 * 0.01)  # Euler CFL limit
        assert dt >= dt_euler  # RK4 should be at least as permissive

    def test_solve_basic(self):
        """Test that solve runs and returns a result."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)
        ic = bt.gaussian(mesh, center=0.5, width=0.1)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(ic)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        integrator = bt.RK4Integrator(problem)
        result = integrator.solve(t_end=0.1)

        assert result is not None
        assert isinstance(result, bt.IntegrationResult)
        assert len(result.solution) == len(ic)
        assert result.time > 0
        assert result.stats["method"] == "rk4"
        assert result.stats["steps"] > 0

    def test_solve_diffusion_accuracy(self):
        """Test RK4 accuracy on a diffusion problem."""
        # Use a simple exponential decay test case
        n = 51
        mesh = bt.mesh_1d(n, 0.0, 1.0)
        D = 0.1

        # Initial condition: sin(pi*x) - eigenfunction of Laplacian
        x = list(bt.x_nodes(mesh))
        ic = [np.sin(np.pi * xi) for xi in x]

        problem = (
            bt.Problem(mesh)
            .diffusivity(D)
            .initial_condition(ic)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        # Analytical solution: u(x,t) = sin(pi*x) * exp(-pi^2 * D * t)
        t_end = 0.1
        decay_factor = np.exp(-(np.pi**2) * D * t_end)
        expected = [np.sin(np.pi * xi) * decay_factor for xi in x]

        integrator = bt.RK4Integrator(problem)
        result = integrator.solve(t_end=t_end)

        # Compare with analytical solution
        error = np.sqrt(np.mean((np.array(result.solution) - np.array(expected)) ** 2))

        # RK4 should achieve good accuracy
        assert error < 0.01, f"RMSE = {error}, expected < 0.01"

    def test_store_history(self):
        """Test that history storage works."""
        mesh = bt.mesh_1d(20, 0.0, 1.0)
        ic = bt.gaussian(mesh, center=0.5, width=0.1)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(ic)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        integrator = bt.RK4Integrator(problem)
        result = integrator.solve(t_end=0.05, store_history=True)

        assert "history" in result.stats
        history = result.stats["history"]
        assert len(history) > 1  # Should have multiple snapshots
        assert len(history[0]) == len(ic)


# ============================================================================
# Test HeunIntegrator class
# ============================================================================


class TestHeunIntegrator:
    """Tests for the HeunIntegrator class."""

    def test_initialization(self):
        """Test HeunIntegrator can be constructed."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)
        ic = bt.gaussian(mesh, center=0.5, width=0.1)
        problem = bt.Problem(mesh).diffusivity(0.01).initial_condition(ic)

        integrator = bt.HeunIntegrator(problem)
        assert integrator is not None

    def test_solve_basic(self):
        """Test that solve runs and returns a result."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)
        ic = bt.gaussian(mesh, center=0.5, width=0.1)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(ic)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        integrator = bt.HeunIntegrator(problem)
        result = integrator.solve(t_end=0.1)

        assert result is not None
        assert result.stats["method"] == "heun"


# ============================================================================
# Test integrate() convenience function
# ============================================================================


class TestIntegrateFunction:
    """Tests for the integrate() convenience function."""

    def test_integrate_euler(self):
        """Test integrate with euler method."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)
        ic = bt.gaussian(mesh, center=0.5, width=0.1)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(ic)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        result = bt.integrate(problem, t_end=0.1, method="euler")
        assert result.stats["method"] == "euler"

    def test_integrate_heun(self):
        """Test integrate with heun method."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)
        ic = bt.gaussian(mesh, center=0.5, width=0.1)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(ic)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        result = bt.integrate(problem, t_end=0.1, method="heun")
        assert result.stats["method"] == "heun"

    def test_integrate_rk4(self):
        """Test integrate with rk4 method."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)
        ic = bt.gaussian(mesh, center=0.5, width=0.1)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(ic)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        result = bt.integrate(problem, t_end=0.1, method="rk4")
        assert result.stats["method"] == "rk4"

    def test_integrate_invalid_method(self):
        """Test that invalid method raises error."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)
        ic = bt.gaussian(mesh, center=0.5, width=0.1)
        problem = bt.Problem(mesh).diffusivity(0.01).initial_condition(ic)

        with pytest.raises(ValueError, match="Unknown integration method"):
            bt.integrate(problem, t_end=0.1, method="invalid")


# ============================================================================
# Test accuracy comparison between methods
# ============================================================================


class TestMethodAccuracyComparison:
    """Compare accuracy of different time integration methods."""

    def test_rk4_more_accurate_than_euler(self):
        """Verify RK4 gives better accuracy than Euler for same dt."""
        n = 51
        mesh = bt.mesh_1d(n, 0.0, 1.0)
        D = 0.1

        x = list(bt.x_nodes(mesh))
        ic = [np.sin(np.pi * xi) for xi in x]

        problem = (
            bt.Problem(mesh)
            .diffusivity(D)
            .initial_condition(ic)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        t_end = 0.1
        decay_factor = np.exp(-(np.pi**2) * D * t_end)
        expected = np.array([np.sin(np.pi * xi) * decay_factor for xi in x])

        # Use same timestep for both
        dt = 0.001

        result_euler = bt.integrate(problem, t_end=t_end, method="euler", dt=dt)
        result_rk4 = bt.integrate(problem, t_end=t_end, method="rk4", dt=dt)

        error_euler = np.sqrt(np.mean((result_euler.solution - expected) ** 2))
        error_rk4 = np.sqrt(np.mean((result_rk4.solution - expected) ** 2))

        # RK4 should be more accurate
        assert (
            error_rk4 < error_euler
        ), f"RK4 error ({error_rk4}) should be less than Euler error ({error_euler})"

    def test_heun_more_accurate_than_euler(self):
        """Verify Heun gives better accuracy than Euler for same dt."""
        n = 51
        mesh = bt.mesh_1d(n, 0.0, 1.0)
        D = 0.1

        x = list(bt.x_nodes(mesh))
        ic = [np.sin(np.pi * xi) for xi in x]

        problem = (
            bt.Problem(mesh)
            .diffusivity(D)
            .initial_condition(ic)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        t_end = 0.1
        decay_factor = np.exp(-(np.pi**2) * D * t_end)
        expected = np.array([np.sin(np.pi * xi) * decay_factor for xi in x])

        dt = 0.001

        result_euler = bt.integrate(problem, t_end=t_end, method="euler", dt=dt)
        result_heun = bt.integrate(problem, t_end=t_end, method="heun", dt=dt)

        error_euler = np.sqrt(np.mean((result_euler.solution - expected) ** 2))
        error_heun = np.sqrt(np.mean((result_heun.solution - expected) ** 2))

        # Heun should be more accurate
        assert (
            error_heun < error_euler
        ), f"Heun error ({error_heun}) should be less than Euler error ({error_euler})"
