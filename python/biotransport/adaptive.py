"""
Adaptive Time-Stepping for Transport Simulations

Provides error-controlled time integration with automatic step size adjustment.
This is essential for robust simulations where the optimal time step varies
during the computation (e.g., reaction fronts, transient heat sources).

Key features:
- Error estimation via Richardson extrapolation (comparing dt and dt/2)
- Automatic step size increase/decrease based on local error
- Step rejection when error exceeds tolerance
- Comprehensive statistics tracking

Example:
    >>> mesh = bt.mesh_1d(100, 0.0, 1.0)
    >>> problem = bt.Problem(mesh).diffusivity(1e-5).initial_condition(u0)
    >>> stepper = bt.AdaptiveTimeStepper(problem, tol=1e-4)
    >>> result = stepper.solve(t_end=10.0)
    >>> print(f"Steps: {result.stats['steps']}, Rejections: {result.stats['rejections']}")
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ._core import (
    DiffusionSolver,
    TransportProblem,
)


@dataclass
class AdaptiveResult:
    """Result of an adaptive time-stepping simulation."""

    solution: np.ndarray
    """Final solution field."""

    time: float
    """Final simulation time reached."""

    stats: dict = field(default_factory=dict)
    """Statistics including steps, rejections, dt history."""


@dataclass
class AdaptiveTimeStepperConfig:
    """Configuration for adaptive time-stepping."""

    tol: float = 1e-4
    """Relative error tolerance for step acceptance."""

    atol: float = 1e-8
    """Absolute error tolerance (for near-zero solutions)."""

    safety: float = 0.9
    """Safety factor for step size adjustment."""

    dt_min: float = 1e-12
    """Minimum allowed time step."""

    dt_max: Optional[float] = None
    """Maximum allowed time step (None = CFL limit)."""

    max_factor: float = 2.0
    """Maximum factor for step size increase."""

    min_factor: float = 0.1
    """Minimum factor for step size decrease."""

    max_rejections: int = 100
    """Maximum consecutive rejections before error."""


class AdaptiveTimeStepper:
    """
    Adaptive time-stepping controller for transport simulations.

    Uses local error estimation via step-doubling (Richardson extrapolation)
    to automatically adjust the time step for efficiency and accuracy.

    The error is estimated by comparing:
    - One step of size dt
    - Two steps of size dt/2

    The difference gives an O(dt²) error estimate for explicit methods.

    Example:
        >>> stepper = AdaptiveTimeStepper(problem, tol=1e-4)
        >>> result = stepper.solve(t_end=10.0)
    """

    def __init__(
        self,
        problem: TransportProblem,
        *,
        tol: float = 1e-4,
        atol: float = 1e-8,
        safety: float = 0.9,
        dt_min: float = 1e-12,
        dt_max: Optional[float] = None,
        verbose: bool = False,
    ):
        """
        Create an adaptive time-stepper.

        Parameters
        ----------
        problem : TransportProblem
            The transport problem to solve.
        tol : float
            Relative error tolerance for step acceptance.
        atol : float
            Absolute error tolerance (for near-zero values).
        safety : float
            Safety factor for step size adjustment (< 1).
        dt_min : float
            Minimum allowed time step.
        dt_max : float, optional
            Maximum allowed time step. Defaults to CFL limit.
        verbose : bool
            Print step information during solve.
        """
        self.problem = problem
        self.config = AdaptiveTimeStepperConfig(
            tol=tol,
            atol=atol,
            safety=safety,
            dt_min=dt_min,
            dt_max=dt_max,
        )
        self.verbose = verbose

        # Create solver instance for stepping
        self._mesh = problem.mesh()
        self._D = problem.diffusivity()
        self._initial = np.array(problem.initial())

        # Compute CFL limit
        self._cfl_limit = self._compute_cfl_limit()

        if dt_max is None:
            self.config.dt_max = self._cfl_limit

    def _compute_cfl_limit(self) -> float:
        """Compute the maximum stable time step based on CFL condition."""
        mesh = self._mesh
        D = self._D
        dx2 = mesh.dx() ** 2

        if mesh.is_1d():
            return 0.9 * dx2 / (2.0 * D)
        else:
            dy2 = mesh.dy() ** 2
            return 0.9 / (2.0 * D * (1.0 / dx2 + 1.0 / dy2))

    def _create_solver(self, initial: np.ndarray):
        """Create a fresh solver with the given initial condition."""
        # Get the linear reaction rate if present
        # Note: We access the problem's internal state
        problem = self.problem

        # Create appropriate solver type
        solver = DiffusionSolver(self._mesh, self._D)
        solver.set_initial_condition(initial.tolist())

        # Set boundary conditions
        boundaries = problem.boundaries()
        for i in range(4):
            solver.set_boundary_condition(i, boundaries[i])

        return solver

    def _step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Take a single time step from state u with step size dt."""
        solver = self._create_solver(u)
        solver.solve(dt, 1)
        return np.array(solver.solution())

    def _estimate_error(
        self, u: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Estimate local error using step-doubling.

        Returns:
            (u_full, u_half, error) where:
            - u_full: Solution after one step of dt
            - u_half: Solution after two steps of dt/2 (more accurate)
            - error: Maximum relative error estimate
        """
        # One step of size dt
        u_full = self._step(u, dt)

        # Two steps of size dt/2
        u_mid = self._step(u, dt / 2)
        u_half = self._step(u_mid, dt / 2)

        # Error estimate: |u_half - u_full| / (atol + rtol * |u_half|)
        diff = np.abs(u_half - u_full)
        scale = self.config.atol + self.config.tol * np.abs(u_half)
        error = np.max(diff / scale)

        return u_full, u_half, error

    def solve(
        self,
        t_end: float,
        dt_initial: Optional[float] = None,
        callback: Optional[Callable[[float, np.ndarray], None]] = None,
    ) -> AdaptiveResult:
        """
        Solve the problem to time t_end with adaptive time-stepping.

        Parameters
        ----------
        t_end : float
            Final simulation time.
        dt_initial : float, optional
            Initial time step guess. Defaults to CFL limit.
        callback : callable, optional
            Function called after each accepted step: callback(t, u)

        Returns
        -------
        AdaptiveResult
            Solution and statistics.
        """
        if t_end <= 0:
            raise ValueError("t_end must be positive")

        # Initialize
        u = self._initial.copy()
        t = 0.0
        dt = dt_initial if dt_initial else self.config.safety * self._cfl_limit

        # Ensure dt doesn't exceed CFL
        dt = min(dt, self._cfl_limit)

        # Statistics tracking
        steps = 0
        rejections = 0
        consecutive_rejections = 0
        dt_history = []

        while t < t_end:
            # Don't overshoot
            if t + dt > t_end:
                dt = t_end - t

            # Estimate error
            u_full, u_half, error = self._estimate_error(u, dt)

            if error <= 1.0:
                # Accept step (use the more accurate u_half)
                u = u_half
                t += dt
                steps += 1
                consecutive_rejections = 0
                dt_history.append(dt)

                if callback:
                    callback(t, u)

                if self.verbose:
                    print(f"  t={t:.6e}, dt={dt:.6e}, error={error:.2e} (accepted)")

                # Increase step size for next step
                if error > 0:
                    factor = self.config.safety * (1.0 / error) ** 0.5
                    factor = min(factor, self.config.max_factor)
                else:
                    factor = self.config.max_factor

                dt = min(dt * factor, self.config.dt_max, self._cfl_limit)

            else:
                # Reject step
                rejections += 1
                consecutive_rejections += 1

                if consecutive_rejections > self.config.max_rejections:
                    raise RuntimeError(
                        f"Too many consecutive step rejections ({consecutive_rejections}). "
                        f"Consider increasing tolerance or decreasing dt_min."
                    )

                if self.verbose:
                    print(f"  t={t:.6e}, dt={dt:.6e}, error={error:.2e} (REJECTED)")

                # Decrease step size
                factor = max(
                    self.config.safety * (1.0 / error) ** 0.5, self.config.min_factor
                )
                dt = max(dt * factor, self.config.dt_min)

        # Build statistics
        stats = {
            "steps": steps,
            "rejections": rejections,
            "dt_min_used": min(dt_history) if dt_history else 0,
            "dt_max_used": max(dt_history) if dt_history else 0,
            "dt_avg": np.mean(dt_history) if dt_history else 0,
            "dt_history": dt_history,
            "cfl_limit": self._cfl_limit,
            "final_error": error,
        }

        return AdaptiveResult(solution=u, time=t, stats=stats)


def solve_adaptive(
    problem: TransportProblem,
    t_end: float,
    *,
    tol: float = 1e-4,
    verbose: bool = False,
) -> AdaptiveResult:
    """
    Convenience function for adaptive time-stepping.

    Parameters
    ----------
    problem : TransportProblem
        The transport problem to solve.
    t_end : float
        Final simulation time.
    tol : float
        Relative error tolerance.
    verbose : bool
        Print step information.

    Returns
    -------
    AdaptiveResult
        Solution and statistics.

    Example
    -------
    >>> result = bt.solve_adaptive(problem, t_end=10.0, tol=1e-4)
    >>> print(f"Finished in {result.stats['steps']} steps")
    """
    stepper = AdaptiveTimeStepper(problem, tol=tol, verbose=verbose)
    return stepper.solve(t_end)
