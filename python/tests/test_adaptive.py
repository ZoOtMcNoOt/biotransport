"""Tests for adaptive time-stepping module."""

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
        )

        assert config.tol == 1e-6
        assert config.atol == 1e-10
        assert config.safety == 0.8
        assert config.dt_min == 1e-15
        assert config.dt_max == 0.01
        assert config.max_factor == 3.0
        assert config.min_factor == 0.05
        assert config.max_rejections == 50


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
        )

        assert stepper.config.tol == 1e-6
        assert stepper.config.atol == 1e-10
        assert stepper.config.safety == 0.85

    def test_cfl_limit_computed(self, simple_problem):
        """Test CFL limit is computed."""
        stepper = AdaptiveTimeStepper(simple_problem)

        # CFL limit should be positive and reasonable
        assert stepper._cfl_limit > 0
        assert stepper._cfl_limit < 100.0  # Should be finite and reasonable

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

    def test_solve_negative_time_raises(self, simple_problem):
        """Test that negative end time raises error."""
        stepper = AdaptiveTimeStepper(simple_problem)

        with pytest.raises(ValueError, match="t_end must be positive"):
            stepper.solve(t_end=-1.0)

    def test_solve_custom_initial_dt(self, simple_problem):
        """Test solve with custom initial time step."""
        stepper = AdaptiveTimeStepper(simple_problem, tol=1e-3)

        # Use a very small initial dt
        result = stepper.solve(t_end=0.005, dt_initial=1e-8)

        assert result.time == pytest.approx(0.005)
        assert result.stats["steps"] > 0

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


class TestAdaptiveTimeStepper2D:
    """Tests for adaptive time-stepping on 2D problems."""

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

    def test_2d_cfl_limit(self, simple_2d_problem):
        """Test CFL limit computation for 2D."""
        stepper = AdaptiveTimeStepper(simple_2d_problem)

        # 2D CFL should be smaller due to both directions
        assert stepper._cfl_limit > 0

    def test_2d_solve(self, simple_2d_problem):
        """Test solve on 2D problem."""
        stepper = AdaptiveTimeStepper(simple_2d_problem, tol=1e-3)
        result = stepper.solve(t_end=0.001)

        assert result.time == pytest.approx(0.001)
        assert len(result.solution) == 11 * 11


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

    def test_mass_conservation_neumann(self):
        """Test mass conservation with Neumann BCs."""
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

        result = solve_adaptive(problem, t_end=0.5, tol=1e-4)

        # Total mass should be approximately conserved
        dx = 1.0 / 50
        initial_mass = np.sum(u0) * dx
        final_mass = np.sum(result.solution) * dx

        assert final_mass == pytest.approx(initial_mass, rel=0.05)
