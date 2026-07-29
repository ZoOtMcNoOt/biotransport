"""Tests for higher-order time integration methods (RK4, Heun).

These tests verify:
1. Basic functionality of integrators
2. Correct convergence order (4th for RK4, 2nd for Heun)
3. Accuracy comparison with analytical solutions
"""

import warnings
from fractions import Fraction

import numpy as np
import pytest

import biotransport as bt
from biotransport.time_integrators import (
    _RK4_NEGATIVE_REAL_AXIS_RADIUS,
    _resolve_legacy_step,
    _validated_ode_array,
)


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

    def test_rejects_unrepresentable_large_clock_midpoint(self):
        with pytest.raises(ValueError, match="midpoint time is not representable"):
            bt.rk4_step(np.array([1.0]), lambda u, t: u, 1.0e16, 2.0)

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


@pytest.mark.parametrize("stepper", [bt.euler_step, bt.heun_step, bt.rk4_step])
class TestStandaloneStepContracts:
    """Generic ODE steps must reject corruption instead of broadcasting it."""

    def test_rhs_cannot_mutate_the_callers_state(self, stepper):
        state = np.array([1.0, 2.0])

        def mutating_rhs(stage_state, _time):
            stage_state[:] = 100.0
            return np.zeros_like(stage_state)

        result = stepper(state, mutating_rhs, 0.0, 0.1)

        np.testing.assert_array_equal(state, [1.0, 2.0])
        np.testing.assert_array_equal(result, [1.0, 2.0])

    @pytest.mark.parametrize(
        "rhs",
        [
            lambda _state, _time: 1.0,
            lambda _state, _time: np.array([1.0, 2.0, 3.0]),
        ],
    )
    def test_rejects_rhs_shapes_that_numpy_would_broadcast(self, stepper, rhs):
        with pytest.raises(ValueError, match="shape"):
            stepper(np.ones(2), rhs, 0.0, 0.1)

    @pytest.mark.parametrize(
        "rhs",
        [
            lambda state, _time: np.full_like(state, np.nan),
            lambda state, _time: np.full_like(state, np.inf),
        ],
    )
    def test_rejects_nonfinite_rhs_values(self, stepper, rhs):
        with pytest.raises(FloatingPointError, match="finite"):
            stepper(np.ones(2), rhs, 0.0, 0.1)

    def test_rejects_finite_inputs_whose_update_overflows(self, stepper):
        with pytest.raises(FloatingPointError, match="non-finite state"):
            stepper(
                np.array([1.0e308]),
                lambda _state, _time: np.array([1.0e308]),
                0.0,
                1.0e308,
            )

    @pytest.mark.parametrize(
        ("state", "time", "dt"),
        [
            (np.array([np.nan]), 0.0, 0.1),
            (np.ones(1), np.inf, 0.1),
            (np.ones(1), 0.0, np.nan),
        ],
    )
    def test_rejects_nonfinite_step_inputs(self, stepper, state, time, dt):
        with pytest.raises(ValueError, match="finite"):
            stepper(state, lambda value, _time: value, time, dt)

    @pytest.mark.parametrize("dt", [0.0, 0.5])
    def test_rejects_zero_or_unrepresentable_large_clock_step(self, stepper, dt):
        message = "nonzero" if dt == 0.0 else "distinct finite"
        with pytest.raises(ValueError, match=message):
            stepper(
                np.ones(1),
                lambda value, _time: value,
                1.0e16,
                dt,
            )

    def test_rejects_boolean_state_as_non_numerical(self, stepper):
        with pytest.raises(TypeError, match="numeric"):
            stepper(
                np.array([True]),
                lambda state, _time: np.zeros_like(state),
                0.0,
                0.1,
            )

    def test_rejects_actively_masked_state(self, stepper):
        state = np.ma.array([1.0, 999.0], mask=[False, True])

        with pytest.raises(ValueError, match="masked"):
            stepper(state, lambda value, _time: value, 0.0, 0.1)

    def test_rejects_actively_masked_rhs_stage(self, stepper):
        def masked_rhs(_state, _time):
            return np.ma.array([1.0, 999.0], mask=[False, True])

        with pytest.raises(ValueError, match="masked"):
            stepper(np.ones(2), masked_rhs, 0.0, 0.1)


@pytest.mark.parametrize("stepper", [bt.heun_step, bt.rk4_step])
def test_stage_average_preserves_a_representable_minimum_subnormal(stepper):
    minimum_subnormal = np.nextafter(0.0, 1.0)

    result = stepper(
        np.array([0.0]),
        lambda _state, _time: np.array([minimum_subnormal]),
        0.0,
        1.0,
    )

    assert result[0] == minimum_subnormal


def test_rk4_stage_average_preserves_moderate_terms_across_huge_cancellation():
    stage_values = iter((np.finfo(float).max, 1.0, 1.0, -np.finfo(float).max))

    result = bt.rk4_step(
        np.array([0.0]),
        lambda _state, _time: np.array([next(stage_values)]),
        0.0,
        1.0,
    )

    assert result[0] == pytest.approx(2.0 / 3.0)


def test_heun_stage_average_preserves_adjacent_huge_difference():
    largest = np.finfo(float).max
    adjacent = np.nextafter(largest, 0.0)
    stage_values = iter((largest, -adjacent))

    result = bt.heun_step(
        np.array([0.0]),
        lambda _state, _time: np.array([next(stage_values)]),
        0.0,
        1.0,
    )

    assert result[0] == (largest - adjacent) / 2.0


def test_ode_array_contract_rejects_hidden_mask_and_complex_diffusion_data():
    with pytest.raises(ValueError, match="masked"):
        _validated_ode_array(
            np.ma.array([1.0, 2.0], mask=[False, True]),
            "field",
        )
    with pytest.raises(TypeError, match="real"):
        _validated_ode_array(
            np.array([1.0 + 2.0j]),
            "diffusion field",
            allow_complex=False,
        )


# ============================================================================
# Test RK4Integrator class
# ============================================================================


class TestRK4Integrator:
    """Tests for the RK4Integrator class with transport problems."""

    def test_negative_real_axis_ceiling_is_conservatively_rounded(self):
        radius = Fraction.from_float(_RK4_NEGATIVE_REAL_AXIS_RADIUS)
        z = -radius
        stability_polynomial = 1 + z + z * z / 2 + z * z * z / 6 + z * z * z * z / 24

        assert abs(stability_polynomial) <= 1

    def test_initialization(self):
        """Test RK4Integrator can be constructed."""
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
        assert integrator is not None
        assert integrator.D == 0.01

    def test_max_stable_dt(self):
        """Test that max_stable_dt returns a positive value."""
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
        dt = integrator.max_stable_dt()

        assert dt > 0
        dx = 1.0 / 50
        expected = 0.5 * 2.785293563405282 * dx * dx / (4 * 0.01)
        assert dt == pytest.approx(expected)

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
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(ic)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        integrator = bt.HeunIntegrator(problem)
        assert integrator is not None

    def test_max_stable_dt_matches_euler_limit(self):
        """Heun/RK2 has the Euler negative-real-axis stability interval."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        integrator = bt.HeunIntegrator(problem)
        expected = 0.8 * mesh.dx() ** 2 / (2 * 0.01)
        assert integrator.max_stable_dt() == pytest.approx(expected)

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

    def test_default_step_damps_highest_frequency_mode(self):
        """The default Heun step must not amplify the diffusion grid mode."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)
        initial = np.array(
            [0.0 if index in (0, 50) else (-1.0) ** index for index in range(51)]
        )
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(initial)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )
        integrator = bt.HeunIntegrator(problem)

        result = integrator.solve(
            t_end=integrator.max_stable_dt(),
            dt=integrator.max_stable_dt(),
        )

        assert np.max(np.abs(result.solution[1:-1])) <= np.max(np.abs(initial[1:-1]))


class TestExtremeScaleLegacyDiffusion:
    """Finite diffusion must survive extreme but representable input scales."""

    @staticmethod
    def _impulse_problem(dx, diffusivity, amplitude):
        mesh = bt.mesh_1d(4, 0.0, 4.0 * dx)
        return (
            bt.Problem(mesh)
            .diffusivity(diffusivity)
            .initial_condition([0.0, 0.0, amplitude, 0.0, 0.0])
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    def test_large_spacing_and_field_preserve_neighbor_diffusion(self, integrator):
        problem = self._impulse_problem(1.0e155, 1.0, 1.0e300)
        solver = integrator(problem)

        result = solver.solve(t_end=1.0, dt=1.0)

        assert np.isinf(solver.max_stable_dt())
        assert np.all(np.isfinite(result.solution))
        assert result.solution[1] == pytest.approx(1.0e-10)
        assert result.solution[3] == pytest.approx(1.0e-10)

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    def test_subnormal_diffusivity_and_tiny_spacing_preserve_lambda(self, integrator):
        minimum_subnormal = np.nextafter(0.0, 1.0)
        dx = 1.0e-200
        dt = 1.0e-78
        diffusion_number = minimum_subnormal / dx / dx * dt
        problem = self._impulse_problem(dx, minimum_subnormal, 1.0)
        solver = integrator(problem)

        result = solver.solve(t_end=dt, dt=dt)

        assert diffusion_number == pytest.approx(0.04940656458412465)
        assert solver.max_stable_dt() > dt
        assert np.all(np.isfinite(result.solution))
        assert 0.04 < result.solution[1] < diffusion_number
        assert result.solution[1] == pytest.approx(result.solution[3])

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    def test_subnormal_final_remainder_is_not_rounded_up(self, integrator):
        minimum_subnormal = np.nextafter(0.0, 1.0)
        mesh = bt.mesh_1d(2, 0.0, 2.0e-154)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0)
            .initial_condition([0.0, 0.1, 0.0])
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        exact = integrator(problem).solve(
            t_end=3.0 * minimum_subnormal,
            dt=2.0 * minimum_subnormal,
        )
        overadvanced = integrator(problem).solve(
            t_end=4.0 * minimum_subnormal,
            dt=2.0 * minimum_subnormal,
        )

        assert exact.time == 3.0 * minimum_subnormal
        assert exact.stats["steps"] == 2
        assert exact.stats["dt"] == 2.0 * minimum_subnormal
        assert exact.stats["final_dt"] == minimum_subnormal
        assert exact.solution[1] != overadvanced.solution[1]

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    def test_stability_ceiling_never_rounds_up_to_minimum_subnormal(self, integrator):
        dx = np.nextafter(np.ldexp(1.0, -537), np.inf)
        mesh = bt.mesh_1d(1, 0.0, dx)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        with pytest.raises(FloatingPointError, match="below binary64 range"):
            integrator(problem, safety_factor=1.0).max_stable_dt()


class TestLegacyProblemValidation:
    """Legacy Python wrappers must reject physics they cannot preserve."""

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    def test_rejects_multidimensional_mesh(self, integrator):
        mesh = bt.mesh_2d(4, 4)
        problem = bt.Problem(mesh).diffusivity(0.01).initial_condition(0.0)

        with pytest.raises(ValueError, match="only 1D diffusion"):
            integrator(problem)

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    @pytest.mark.parametrize(
        "physics", ["variable diffusivity", "reaction", "advection"]
    )
    def test_rejects_unrepresented_physics(self, integrator, physics):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )
        if physics == "variable diffusivity":
            problem.diffusivity_field(np.full(mesh.num_nodes(), 0.01))
        elif physics == "reaction":
            problem.constant_source(1.0)
        else:
            problem.velocity(0.1)

        with pytest.raises(ValueError, match=physics):
            integrator(problem)

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    def test_rejects_non_dirichlet_boundaries(self, integrator):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .neumann(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        with pytest.raises(ValueError, match="Dirichlet left/right"):
            integrator(problem)

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    @pytest.mark.parametrize("safety_factor", [0.0, -0.1, 1.1, np.inf, np.nan])
    def test_rejects_invalid_safety_factor(self, integrator, safety_factor):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        with pytest.raises(ValueError, match="safety_factor"):
            integrator(problem, safety_factor=safety_factor)

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    def test_rejects_requested_step_above_stability_limit(self, integrator):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.1)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )
        solver = integrator(problem)

        with pytest.raises(ValueError, match="stability limit"):
            solver.solve(t_end=1.0, dt=1.01 * solver.max_stable_dt())

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    def test_subnormal_stability_limit_has_no_normal_scale_allowance(self, integrator):
        mesh = bt.mesh_1d(10, 0.0, 1.0e-159)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )
        solver = integrator(problem)
        stable_dt = solver.max_stable_dt()
        minimum_subnormal = np.nextafter(0.0, 1.0)
        requested_dt = stable_dt + 16.0 * minimum_subnormal

        assert 0.0 < stable_dt < np.finfo(float).tiny
        with pytest.raises(ValueError, match="stability limit"):
            solver.solve(t_end=requested_dt, dt=requested_dt)

    def test_minimum_subnormal_stability_limit_is_strict(self):
        minimum_subnormal = np.nextafter(0.0, 1.0)

        with pytest.raises(ValueError, match="stability limit"):
            _resolve_legacy_step(
                2.0 * minimum_subnormal,
                2.0 * minimum_subnormal,
                minimum_subnormal,
                "legacy regression",
            )

    def test_planned_step_count_uses_exact_float_ratio(self):
        stable_dt = 0.09375291778124752
        final_time = 0.9375291778124752

        num_steps, resolved_dt = _resolve_legacy_step(
            final_time,
            stable_dt,
            stable_dt,
            "legacy regression",
        )

        assert num_steps == 11
        assert resolved_dt <= stable_dt

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    def test_rejects_impractical_python_step_count_before_loop(self, integrator):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.0)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )
        solver = integrator(problem)

        with pytest.raises(RuntimeError, match="step limit"):
            solver.solve(t_end=1.0, dt=5.0e-8)

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    def test_store_history_requires_a_boolean(self, integrator):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        with pytest.raises(TypeError, match="store_history"):
            integrator(problem).solve(t_end=0.1, store_history="yes")

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            (
                lambda problem, mesh: problem.diffusivity_field(
                    np.full(mesh.num_nodes(), 0.01)
                ),
                "variable diffusivity",
            ),
            (
                lambda problem, _mesh: problem.constant_source(1.0),
                "reactions or sources",
            ),
            (
                lambda problem, _mesh: problem.reaction(
                    lambda concentration, _x, _y, _time: -concentration
                ),
                "reactions or sources",
            ),
            (
                lambda problem, _mesh: problem.velocity(0.1),
                "advection",
            ),
        ],
    )
    def test_solve_revalidates_post_construction_physics(
        self, integrator, mutation, message
    ):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )
        solver = integrator(problem)
        mutation(problem, mesh)

        with pytest.raises(ValueError, match=message):
            solver.solve(t_end=0.01)

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    def test_solve_revalidates_post_construction_boundaries(self, integrator):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )
        solver = integrator(problem)
        problem.neumann(bt.Boundary.Left, 0.0)

        with pytest.raises(ValueError, match="Dirichlet left/right"):
            solver.solve(t_end=0.01)

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    def test_supported_live_problem_updates_are_refreshed(self, integrator):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )
        solver = integrator(problem)
        updated_initial = np.linspace(1.0, 2.0, mesh.num_nodes())
        (
            problem.diffusivity(0.02)
            .initial_condition(updated_initial)
            .dirichlet(bt.Boundary.Left, 1.0)
            .dirichlet(bt.Boundary.Right, 2.0)
        )

        result = solver.solve(t_end=0.001)

        np.testing.assert_allclose(result.solution, updated_initial, atol=2e-15)
        assert solver.D == pytest.approx(0.02)
        np.testing.assert_array_equal(solver.u0, updated_initial)
        assert solver.left_bc.value == 1.0
        assert solver.right_bc.value == 2.0

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    def test_captured_public_state_is_read_only_or_copied(self, integrator):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.0)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )
        solver = integrator(problem)

        exposed_initial = solver.u0
        exposed_boundary = solver.left_bc
        with pytest.raises(ValueError, match="read-only"):
            exposed_initial[:] = 99.0
        with pytest.raises(ValueError):
            exposed_initial.setflags(write=True)
        with pytest.raises(AttributeError):
            exposed_boundary.type = bt.BoundaryType.NEUMANN
        with pytest.raises(AttributeError):
            exposed_boundary.value = 99.0
        with pytest.raises(AttributeError):
            solver.u0 = np.zeros(2)
        with pytest.raises(AttributeError):
            solver.D = 0.1
        with pytest.raises(AttributeError):
            solver.safety = 0.1

        result = solver.solve(t_end=0.01)
        np.testing.assert_array_equal(result.solution, np.zeros(mesh.num_nodes()))

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    def test_python_contract_overrides_cannot_spoof_native_problem_state(
        self, integrator
    ):
        class LyingProblem(bt.TransportProblem):
            def has_reaction(self):
                return False

        mesh = bt.mesh_1d(10)
        problem = (
            LyingProblem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .constant_source(1.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        with pytest.raises(TypeError, match="overrides.*has_reaction"):
            integrator(problem)

    @pytest.mark.parametrize("integrator", [bt.RK4Integrator, bt.HeunIntegrator])
    def test_python_complex_initial_override_is_rejected(self, integrator):
        class ComplexInitialProblem(bt.TransportProblem):
            def initial(self):
                return np.full(11, 1.0 + 2.0j)

        mesh = bt.mesh_1d(10)
        problem = (
            ComplexInitialProblem(mesh)
            .diffusivity(0.01)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        with pytest.raises(TypeError, match="overrides.*initial"):
            integrator(problem)


# ============================================================================
# Test integrate() convenience function
# ============================================================================


class TestIntegrateFunction:
    """Tests for the integrate() convenience function."""

    def test_omitted_method_preserves_legacy_rk4_and_warns(self):
        """Omission remains compatible while announcing the future native default."""
        mesh = bt.mesh_1d(20, 0.0, 1.0)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        with pytest.warns(FutureWarning, match="method='rk4'.*biotransport.solve"):
            result = bt.integrate(problem, t_end=0.01, dt=0.005)

        assert result.stats["method"] == "rk4"
        assert "diagnostics" not in result.stats

    def test_integrate_euler(self):
        """Explicit Euler selects the canonical native solver."""
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
        assert "diagnostics" in result.stats

    def test_integrate_euler_forwards_dt_and_lands_exactly(self):
        """Euler delegates its requested ceiling and shortens the final step."""
        mesh = bt.mesh_1d(10, 0.0, 1.0)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        result = bt.integrate(
            problem,
            t_end=0.1,
            method="euler",
            dt=0.03,
        )

        diagnostics = result.stats["diagnostics"]
        assert result.time == pytest.approx(0.1)
        assert result.stats["t_end"] == pytest.approx(0.1)
        assert diagnostics.requested_time_step == pytest.approx(0.03)
        assert diagnostics.steps == 4
        assert diagnostics.maximum_time_step == pytest.approx(0.03)
        assert diagnostics.minimum_time_step == pytest.approx(0.01)

    def test_integrate_euler_preserves_configured_source(self):
        """The canonical Euler path must not inherit legacy simplifications."""
        mesh = bt.mesh_1d(10, 0.0, 1.0)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .constant_source(1.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        result = bt.integrate(problem, t_end=0.01, method="euler", dt=0.005)

        assert np.max(result.solution[1:-1]) == pytest.approx(0.01)

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
        """Explicit RK4 remains silent and selects the legacy wrapper."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)
        ic = bt.gaussian(mesh, center=0.5, width=0.1)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(ic)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = bt.integrate(problem, t_end=0.1, method="rk4")
        assert result.stats["method"] == "rk4"

    def test_integrate_invalid_method(self):
        """Test that invalid method raises error."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)
        ic = bt.gaussian(mesh, center=0.5, width=0.1)
        problem = bt.Problem(mesh).diffusivity(0.01).initial_condition(ic)

        with pytest.raises(ValueError, match="Unknown integration method"):
            bt.integrate(problem, t_end=0.1, method="invalid")

        with pytest.raises(TypeError, match="string"):
            bt.integrate(problem, t_end=0.1, method=None)


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
        # Compare temporal error against the exact solution of the same
        # semi-discrete Laplacian. Comparing with the continuum eigenvalue can
        # let Euler's time error accidentally cancel the spatial error.
        discrete_eigenvalue = (
            -4.0 * D * np.sin(0.5 * np.pi * mesh.dx()) ** 2 / mesh.dx() ** 2
        )
        expected = np.asarray(ic) * np.exp(discrete_eigenvalue * t_end)

        # Use same timestep for both
        dt = 0.001

        result_euler = bt.integrate(problem, t_end=t_end, method="euler", dt=dt)
        result_rk4 = bt.integrate(problem, t_end=t_end, method="rk4", dt=dt)

        error_euler = np.sqrt(np.mean((result_euler.solution - expected) ** 2))
        error_rk4 = np.sqrt(np.mean((result_rk4.solution - expected) ** 2))

        # RK4 should be more accurate
        assert error_rk4 < error_euler, (
            f"RK4 error ({error_rk4}) should be less than Euler error ({error_euler})"
        )

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
        discrete_eigenvalue = (
            -4.0 * D * np.sin(0.5 * np.pi * mesh.dx()) ** 2 / mesh.dx() ** 2
        )
        expected = np.asarray(ic) * np.exp(discrete_eigenvalue * t_end)

        dt = 0.001

        result_euler = bt.integrate(problem, t_end=t_end, method="euler", dt=dt)
        result_heun = bt.integrate(problem, t_end=t_end, method="heun", dt=dt)

        error_euler = np.sqrt(np.mean((result_euler.solution - expected) ** 2))
        error_heun = np.sqrt(np.mean((result_heun.solution - expected) ** 2))

        # Heun should be more accurate
        assert error_heun < error_euler, (
            f"Heun error ({error_heun}) should be less than Euler error ({error_euler})"
        )
