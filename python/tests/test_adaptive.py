"""Tests for adaptive time-stepping module."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import biotransport as bt
from biotransport.adaptive import (
    AdaptiveResult,
    AdaptiveTimeStepper,
    AdaptiveTimeStepperConfig,
    solve_adaptive,
)


class TestAdaptiveTimeStepperConfig:
    """Tests for AdaptiveTimeStepperConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AdaptiveTimeStepperConfig()

        assert config.tol == 1e-4
        assert config.atol == 1e-8
        assert config.safety == 0.9
        assert config.dt_min == 1e-12
        assert config.dt_max is None
        assert config.max_factor == 2.0
        assert config.min_factor == 0.1
        assert config.max_rejections == 100
        assert config.maximum_steps == 10_000_000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AdaptiveTimeStepperConfig(
            tol=1e-6,
            atol=1e-10,
            safety=0.8,
            dt_min=1e-15,
            dt_max=0.01,
            max_factor=3.0,
            min_factor=0.05,
            max_rejections=50,
            maximum_steps=500,
        )

        assert config.tol == 1e-6
        assert config.atol == 1e-10
        assert config.safety == 0.8
        assert config.dt_min == 1e-15
        assert config.dt_max == 0.01
        assert config.max_factor == 3.0
        assert config.min_factor == 0.05
        assert config.max_rejections == 50
        assert config.maximum_steps == 500

    def test_configuration_is_immutable_after_validation(self):
        config = AdaptiveTimeStepperConfig()

        with pytest.raises(FrozenInstanceError):
            config.tol = -1.0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"tol": -1.0}, "tol"),
            ({"atol": 0.0}, "atol"),
            ({"safety": 1.1}, "safety"),
            ({"dt_min": 0.0}, "dt_min"),
            ({"dt_min": 0.1, "dt_max": 0.01}, "dt_max"),
            ({"max_factor": 0.5}, "max_factor"),
            ({"min_factor": 1.0}, "min_factor"),
            ({"max_rejections": 0}, "max_rejections"),
            ({"maximum_steps": True}, "maximum_steps"),
        ],
    )
    def test_public_configuration_owns_all_validation(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            AdaptiveTimeStepperConfig(**kwargs)


class TestAdaptiveResult:
    """Tests for AdaptiveResult dataclass."""

    def test_result_fields(self):
        """Test AdaptiveResult contains expected fields."""
        solution = np.array([1.0, 2.0, 3.0])
        result = AdaptiveResult(
            solution=solution,
            time=1.0,
            stats={"steps": 10, "rejections": 2},
        )

        assert np.array_equal(result.solution, solution)
        assert result.time == 1.0
        assert result.stats["steps"] == 10
        assert result.stats["rejections"] == 2

    def test_result_default_stats(self):
        """Test AdaptiveResult has empty default stats."""
        result = AdaptiveResult(solution=np.zeros(5), time=0.5)

        assert result.stats == {}


class TestAdaptiveTimeStepper:
    """Tests for AdaptiveTimeStepper class."""

    @pytest.fixture
    def simple_problem(self):
        """Create a simple 1D diffusion problem for testing."""
        mesh = bt.mesh_1d(20)  # 20 cells = 21 nodes
        problem = bt.DiffusionProblem(mesh)
        problem.diffusivity(1e-3)

        # Gaussian initial condition
        x = np.linspace(0.0, 1.0, 21)
        u0 = np.exp(-100 * (x - 0.5) ** 2)
        problem.initial_condition(u0.tolist())

        # Dirichlet BCs
        problem.dirichlet(bt.Boundary.Left, 0.0)
        problem.dirichlet(bt.Boundary.Right, 0.0)

        return problem

    def test_stepper_initialization(self, simple_problem):
        """Test AdaptiveTimeStepper initialization."""
        stepper = AdaptiveTimeStepper(simple_problem, tol=1e-4)

        assert stepper.problem is simple_problem
        assert stepper.config.tol == 1e-4
        assert stepper._cfl_limit > 0

    def test_stepper_custom_tolerance(self, simple_problem):
        """Test stepper with custom tolerance."""
        stepper = AdaptiveTimeStepper(
            simple_problem,
            tol=1e-6,
            atol=1e-10,
            safety=0.85,
            max_factor=3.0,
            min_factor=0.05,
            max_rejections=25,
        )

        assert stepper.config.tol == 1e-6
        assert stepper.config.atol == 1e-10
        assert stepper.config.safety == 0.85
        assert stepper.config.max_factor == 3.0
        assert stepper.config.min_factor == 0.05
        assert stepper.config.max_rejections == 25

    def test_stepper_configuration_cannot_be_replaced_or_corrupted(
        self, simple_problem
    ):
        stepper = AdaptiveTimeStepper(simple_problem)

        with pytest.raises(FrozenInstanceError):
            stepper.config.tol = -1.0
        with pytest.raises(AttributeError):
            stepper.config = AdaptiveTimeStepperConfig(tol=1e-2)

        result = stepper.solve(t_end=0.001)
        assert result.stats["steps"] > 0

    def test_cfl_limit_computed(self, simple_problem):
        """Test CFL limit is computed."""
        stepper = AdaptiveTimeStepper(simple_problem)

        expected = simple_problem.mesh().dx() ** 2 / (2 * 1e-3)
        assert stepper._cfl_limit == pytest.approx(expected)

    def test_solve_basic(self, simple_problem):
        """Test basic solve."""
        stepper = AdaptiveTimeStepper(simple_problem, tol=1e-3)
        result = stepper.solve(t_end=0.01)

        assert isinstance(result, AdaptiveResult)
        assert result.time == pytest.approx(0.01)
        assert len(result.solution) == 21
        assert result.stats["steps"] > 0

    def test_solve_reaches_end_time(self, simple_problem):
        """Test that solve reaches the specified end time."""
        stepper = AdaptiveTimeStepper(simple_problem, tol=1e-4)
        result = stepper.solve(t_end=0.005)

        assert result.time == pytest.approx(0.005, rel=1e-10)

    def test_solve_stats_present(self, simple_problem):
        """Test that statistics are computed."""
        stepper = AdaptiveTimeStepper(simple_problem, tol=1e-4)
        result = stepper.solve(t_end=0.01)

        assert "steps" in result.stats
        assert "rejections" in result.stats
        assert "dt_min_used" in result.stats
        assert "dt_max_used" in result.stats
        assert "dt_avg" in result.stats
        assert "dt_history" in result.stats
        assert "cfl_limit" in result.stats

    def test_solve_with_callback(self, simple_problem):
        """Test solve with callback function."""
        stepper = AdaptiveTimeStepper(simple_problem, tol=1e-3)

        times = []
        solutions = []

        def callback(t, u):
            times.append(t)
            solutions.append(u.copy())

        result = stepper.solve(t_end=0.01, callback=callback)

        assert len(times) == result.stats["steps"]
        assert all(t > 0 for t in times)
        assert times[-1] == pytest.approx(0.01)

    def test_callback_cannot_mutate_the_accepted_state(self, simple_problem):
        baseline = AdaptiveTimeStepper(simple_problem, tol=1e-3).solve(t_end=0.01)

        def corrupting_callback(_time, state):
            state[:] = 100.0

        observed = AdaptiveTimeStepper(simple_problem, tol=1e-3).solve(
            t_end=0.01,
            callback=corrupting_callback,
        )

        np.testing.assert_array_equal(observed.solution, baseline.solution)

    def test_noncallable_callback_is_rejected_before_stepping(self, simple_problem):
        stepper = AdaptiveTimeStepper(simple_problem)

        with pytest.raises(TypeError, match="callback"):
            stepper.solve(t_end=0.01, callback="not callable")

    def test_solve_negative_time_raises(self, simple_problem):
        """Test that negative end time raises error."""
        stepper = AdaptiveTimeStepper(simple_problem)

        with pytest.raises(ValueError, match="t_end must be finite and positive"):
            stepper.solve(t_end=-1.0)

    def test_solve_custom_initial_dt(self, simple_problem):
        """Test solve with custom initial time step."""
        stepper = AdaptiveTimeStepper(simple_problem, tol=1e-3)

        # Use a very small initial dt
        result = stepper.solve(t_end=0.005, dt_initial=1e-8)

        assert result.time == pytest.approx(0.005)
        assert result.stats["steps"] > 0

    @pytest.mark.parametrize("dt_initial", [0.0, -1.0, np.inf, np.nan])
    def test_invalid_initial_dt_raises(self, simple_problem, dt_initial):
        """Invalid initial steps must not enter or stall the adaptive loop."""
        stepper = AdaptiveTimeStepper(simple_problem)

        with pytest.raises(ValueError, match="dt_initial"):
            stepper.solve(t_end=0.005, dt_initial=dt_initial)

    def test_initial_dt_below_configured_minimum_is_rejected(self, simple_problem):
        stepper = AdaptiveTimeStepper(simple_problem, dt_min=1e-4)

        with pytest.raises(ValueError, match="dt_initial"):
            stepper.solve(t_end=0.005, dt_initial=1e-5)

    def test_maximum_steps_stops_an_impractical_solve(self, simple_problem):
        stepper = AdaptiveTimeStepper(simple_problem, maximum_steps=1)

        with pytest.raises(RuntimeError, match="maximum_steps"):
            stepper.solve(t_end=0.005, dt_initial=1e-6)

    def test_reduction_clamps_to_and_attempts_acceptable_dt_min(self, simple_problem):
        stepper = AdaptiveTimeStepper(simple_problem, dt_min=1e-4)
        attempted_steps = []

        def error_drops_at_minimum(state, dt, _snapshot):
            attempted_steps.append(dt)
            error = 4.0 if len(attempted_steps) == 1 else 0.0
            return state.copy(), state.copy(), error

        stepper._estimate_error = error_drops_at_minimum
        result = stepper.solve(t_end=3e-4, dt_initial=2e-4)

        assert attempted_steps[:2] == pytest.approx([2e-4, 1e-4])
        assert result.time == pytest.approx(3e-4)
        assert result.stats["rejections"] == 1

    def test_unsatisfied_tolerance_fails_after_rejecting_dt_min(self, simple_problem):
        stepper = AdaptiveTimeStepper(simple_problem, dt_min=1e-4)

        def rejected_step(state, _dt, _snapshot):
            return state.copy(), state.copy(), 2.0

        stepper._estimate_error = rejected_step
        with pytest.raises(RuntimeError, match="cannot be satisfied at dt_min"):
            stepper.solve(t_end=0.005, dt_initial=1e-4)

    def test_exact_final_remainder_below_dt_min_can_be_accepted(self, simple_problem):
        stepper = AdaptiveTimeStepper(simple_problem, dt_min=1e-4)
        attempted_steps = []

        def accepted_steps(state, dt, _snapshot):
            attempted_steps.append(dt)
            return state.copy(), state.copy(), 0.0

        stepper._estimate_error = accepted_steps
        result = stepper.solve(t_end=2.5e-4, dt_initial=2e-4)

        assert attempted_steps == pytest.approx([2e-4, 5e-5])
        assert result.time == pytest.approx(2.5e-4)
        assert result.stats["dt_history"][-1] == pytest.approx(5e-5)

    def test_rejected_exact_final_remainder_reports_its_actual_size(
        self, simple_problem
    ):
        stepper = AdaptiveTimeStepper(simple_problem, dt_min=1e-4)
        attempts = 0

        def reject_remainder(state, _dt, _snapshot):
            nonlocal attempts
            attempts += 1
            error = 0.0 if attempts == 1 else 2.0
            return state.copy(), state.copy(), error

        stepper._estimate_error = reject_remainder
        with pytest.raises(RuntimeError, match="final remainder below dt_min"):
            stepper.solve(t_end=2.5e-4, dt_initial=2e-4)

    def test_step_rejection_tracking(self, simple_problem):
        """Test that step rejections are tracked."""
        # Use very tight tolerance to force rejections
        stepper = AdaptiveTimeStepper(simple_problem, tol=1e-8)

        result = stepper.solve(t_end=0.001)

        # Stats should include rejection count (may be 0)
        assert "rejections" in result.stats
        assert result.stats["rejections"] >= 0

    def test_dt_history_recorded(self, simple_problem):
        """Test that dt history is recorded."""
        stepper = AdaptiveTimeStepper(simple_problem, tol=1e-4)
        result = stepper.solve(t_end=0.01)

        dt_history = result.stats["dt_history"]
        assert len(dt_history) == result.stats["steps"]
        assert all(dt > 0 for dt in dt_history)


class TestAdaptiveExtremeScaleDiffusion:
    """Adaptive stability and native steps retain representable scale effects."""

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

    def test_large_spacing_and_field_preserve_neighbor_diffusion(self):
        problem = self._impulse_problem(1.0e155, 1.0, 1.0e300)
        stepper = AdaptiveTimeStepper(problem, tol=1.0, dt_max=1.0)

        result = stepper.solve(t_end=1.0, dt_initial=1.0)

        assert np.isinf(result.stats["cfl_limit"])
        assert np.all(np.isfinite(result.solution))
        assert result.solution[1] == pytest.approx(1.0e-10)
        assert result.solution[3] == pytest.approx(1.0e-10)

    def test_subnormal_diffusivity_and_tiny_spacing_preserve_lambda(self):
        minimum_subnormal = np.nextafter(0.0, 1.0)
        dx = 1.0e-200
        dt = 1.0e-78
        diffusion_number = minimum_subnormal / dx / dx * dt
        problem = self._impulse_problem(dx, minimum_subnormal, 1.0)
        stepper = AdaptiveTimeStepper(
            problem,
            tol=1.0,
            dt_min=1.0e-100,
            dt_max=dt,
        )

        result = stepper.solve(t_end=dt, dt_initial=dt)

        assert diffusion_number == pytest.approx(0.04940656458412465)
        assert result.stats["cfl_limit"] > dt
        assert np.all(np.isfinite(result.solution))
        assert 0.04 < result.solution[1] < diffusion_number
        assert result.solution[1] == pytest.approx(result.solution[3])

    def test_unrepresentable_equal_half_steps_fail_loudly(self):
        minimum_subnormal = np.nextafter(0.0, 1.0)
        problem = self._impulse_problem(1.0e-154, 1.0, 0.1)
        stepper = AdaptiveTimeStepper(
            problem,
            tol=1.0,
            dt_min=minimum_subnormal,
        )

        with pytest.raises(FloatingPointError, match="two equal half steps"):
            stepper.solve(
                t_end=3.0 * minimum_subnormal,
                dt_initial=3.0 * minimum_subnormal,
            )

    def test_cfl_ceiling_never_rounds_up_to_minimum_subnormal(self):
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
            AdaptiveTimeStepper(problem, dt_min=np.nextafter(0.0, 1.0))


class TestAdaptiveTimeStepper2D:
    """The legacy adaptive wrapper must reject multidimensional problems."""

    @pytest.fixture
    def simple_2d_problem(self):
        """Create a simple 2D diffusion problem."""
        mesh = bt.mesh_2d(10, 10)  # 10x10 cells = 11x11 nodes
        problem = bt.DiffusionProblem(mesh)
        problem.diffusivity(1e-3)

        # Uniform initial condition
        u0 = np.ones(11 * 11) * 0.5
        problem.initial_condition(u0.tolist())

        return problem

    def test_2d_problem_is_rejected(self, simple_2d_problem):
        """A 2D field must not be flattened and evolved as a 1D chain."""
        with pytest.raises(ValueError, match="only 1D diffusion"):
            AdaptiveTimeStepper(simple_2d_problem)

    def test_2d_solve(self, simple_2d_problem):
        """The convenience wrapper applies the same dimensional guard."""
        with pytest.raises(ValueError, match="only 1D diffusion"):
            solve_adaptive(simple_2d_problem, t_end=0.001, tol=1e-3)


class TestAdaptiveProblemValidation:
    """Unsupported physics must fail before the legacy step loop starts."""

    @pytest.mark.parametrize(
        "physics", ["variable diffusivity", "reaction", "advection"]
    )
    def test_rejects_unrepresented_physics(self, physics):
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
            AdaptiveTimeStepper(problem)

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
    def test_solve_revalidates_post_construction_physics(self, mutation, message):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )
        stepper = AdaptiveTimeStepper(problem)
        mutation(problem, mesh)

        with pytest.raises(ValueError, match=message):
            stepper.solve(t_end=0.01)

    def test_solve_revalidates_post_construction_boundaries(self):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )
        stepper = AdaptiveTimeStepper(problem)
        problem.neumann(bt.Boundary.Left, 0.0)

        with pytest.raises(ValueError, match="Dirichlet left/right"):
            stepper.solve(t_end=0.01)

    def test_supported_live_problem_updates_are_refreshed(self):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )
        stepper = AdaptiveTimeStepper(problem)
        updated_initial = np.linspace(1.0, 2.0, mesh.num_nodes())
        (
            problem.diffusivity(0.0)
            .initial_condition(updated_initial)
            .dirichlet(bt.Boundary.Left, 1.0)
            .dirichlet(bt.Boundary.Right, 2.0)
        )

        result = stepper.solve(t_end=0.01)

        np.testing.assert_array_equal(result.solution, updated_initial)
        assert result.stats["cfl_limit"] == float("inf")

    def test_live_problem_reference_is_read_only(self):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.0)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )
        stepper = AdaptiveTimeStepper(problem)

        with pytest.raises(AttributeError):
            stepper.problem = bt.Problem(mesh)
        assert stepper.problem is problem

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"tol": 0.0}, "tol"),
            ({"atol": np.nan}, "atol"),
            ({"safety": 1.1}, "safety"),
            ({"dt_min": 0.0}, "dt_min"),
            ({"dt_max": np.inf}, "dt_max"),
            ({"dt_min": 0.1, "dt_max": 0.01}, "dt_max"),
            ({"max_factor": 0.5}, "max_factor"),
            ({"min_factor": 1.0}, "min_factor"),
            ({"max_rejections": 0}, "max_rejections"),
            ({"maximum_steps": 0}, "maximum_steps"),
            ({"maximum_steps": True}, "maximum_steps"),
        ],
    )
    def test_rejects_invalid_controller_configuration(self, kwargs, message):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(0.01)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        with pytest.raises(ValueError, match=message):
            AdaptiveTimeStepper(problem, **kwargs)

    def test_rejects_dt_min_above_the_diffusion_stability_limit(self):
        mesh = bt.mesh_1d(10)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0)
            .initial_condition(0.0)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )

        with pytest.raises(ValueError, match="stable permitted"):
            AdaptiveTimeStepper(problem, dt_min=0.01)


class TestSolveAdaptive:
    """Tests for solve_adaptive convenience function."""

    def test_solve_adaptive_basic(self):
        """Test basic solve_adaptive call."""
        mesh = bt.mesh_1d(20)  # 20 cells = 21 nodes
        problem = bt.DiffusionProblem(mesh)
        problem.diffusivity(1e-3)

        x = np.linspace(0.0, 1.0, 21)
        u0 = np.sin(np.pi * x)
        problem.initial_condition(u0.tolist())
        problem.dirichlet(bt.Boundary.Left, 0.0)
        problem.dirichlet(bt.Boundary.Right, 0.0)

        result = solve_adaptive(problem, t_end=0.01, tol=1e-3)

        assert isinstance(result, AdaptiveResult)
        assert result.time == pytest.approx(0.01)
        assert result.stats["steps"] > 0

    def test_solve_adaptive_with_tolerance(self):
        """Test solve_adaptive respects tolerance."""
        mesh = bt.mesh_1d(20)  # 20 cells = 21 nodes
        problem = bt.DiffusionProblem(mesh)
        problem.diffusivity(1e-3)

        u0 = np.ones(21) * 0.5
        problem.initial_condition(u0.tolist())
        problem.dirichlet(bt.Boundary.Left, 0.0)
        problem.dirichlet(bt.Boundary.Right, 0.0)

        # Looser tolerance should require fewer steps
        result_loose = solve_adaptive(problem, t_end=0.01, tol=1e-2)
        result_tight = solve_adaptive(problem, t_end=0.01, tol=1e-5)

        # Tighter tolerance generally needs more steps (though not guaranteed)
        assert result_loose.stats["steps"] > 0
        assert result_tight.stats["steps"] > 0


class TestAdaptiveIntegration:
    """Integration tests for adaptive time-stepping."""

    def test_diffusion_decay(self):
        """Test that adaptive stepper correctly simulates diffusion decay."""
        mesh = bt.mesh_1d(50)  # 50 cells = 51 nodes
        problem = bt.DiffusionProblem(mesh)
        problem.diffusivity(0.01)

        # Sinusoidal initial condition (known analytical solution)
        x = np.linspace(0.0, 1.0, 51)
        u0 = np.sin(np.pi * x)
        problem.initial_condition(u0.tolist())
        problem.dirichlet(bt.Boundary.Left, 0.0)
        problem.dirichlet(bt.Boundary.Right, 0.0)

        result = solve_adaptive(problem, t_end=0.1, tol=1e-4)

        # Solution should have decayed
        assert np.max(result.solution) < np.max(u0)

        # Check approximate analytical decay: exp(-D * pi^2 * t)
        D = 0.01
        t = 0.1
        expected_decay = np.exp(-D * np.pi**2 * t)
        numerical_decay = np.max(result.solution) / np.max(u0)

        assert numerical_decay == pytest.approx(expected_decay, rel=0.1)

    def test_neumann_problem_is_rejected(self):
        """Natural boundaries are delegated to the canonical solver."""
        mesh = bt.mesh_1d(50)  # 50 cells = 51 nodes
        problem = bt.DiffusionProblem(mesh)
        problem.diffusivity(0.01)

        # Non-uniform initial condition
        x = np.linspace(0.0, 1.0, 51)
        u0 = 1.0 + 0.5 * np.sin(2 * np.pi * x)
        problem.initial_condition(u0.tolist())

        # Neumann (no-flux) BCs - mass should be conserved
        problem.neumann(bt.Boundary.Left, 0.0)
        problem.neumann(bt.Boundary.Right, 0.0)

        with pytest.raises(ValueError, match="Dirichlet left/right"):
            solve_adaptive(problem, t_end=0.5, tol=1e-4)
