"""Legacy adaptive stepping for a narrow diffusion problem.

This Python orchestrator uses step doubling around the original native
``DiffusionSolver``.  It supports only one-dimensional, uniform-diffusivity
diffusion with fixed Dirichlet values at both ends.  It rejects reactions,
sources, advection, variable diffusivity, multidimensional meshes, and natural
boundary conditions rather than silently dropping configured physics.

Use :func:`biotransport.solve` for general transport.  This module remains for
legacy pure-diffusion workflows that specifically need adaptive step doubling.

Key features:
- Error estimation via step doubling (comparing dt and dt/2)
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
from .time_integrators import _validate_legacy_diffusion_problem


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
    Legacy adaptive controller for supported 1D uniform diffusion.

    Uses local error estimation via step doubling to automatically adjust the
    time step.  It does not compose the complete :class:`TransportProblem`
    operator; unsupported configurations raise during construction.

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
        self._validate_config(tol, atol, safety, dt_min, dt_max)
        self.config = AdaptiveTimeStepperConfig(
            tol=tol,
            atol=atol,
            safety=safety,
            dt_min=dt_min,
            dt_max=dt_max,
        )
        self.verbose = verbose

        # Validate before extracting the intentionally narrow solver state.
        self._mesh, self._D, self._left_bc, self._right_bc = (
            _validate_legacy_diffusion_problem(problem, "AdaptiveTimeStepper")
        )
        self._initial = np.asarray(problem.initial(), dtype=float).copy()
        self._initial[0] = self._left_bc.value
        self._initial[-1] = self._right_bc.value

        # Compute CFL limit
        self._cfl_limit = self._compute_cfl_limit()

        if dt_max is None:
            self.config.dt_max = self._cfl_limit

    @staticmethod
    def _validate_config(
        tol: float,
        atol: float,
        safety: float,
        dt_min: float,
        dt_max: Optional[float],
    ) -> None:
        """Reject invalid controller parameters before entering the solve loop."""
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be finite and positive")
        if not np.isfinite(atol) or atol <= 0.0:
            raise ValueError("atol must be finite and positive")
        if not np.isfinite(safety) or not 0.0 < safety <= 1.0:
            raise ValueError("safety must be in (0, 1]")
        if not np.isfinite(dt_min) or dt_min <= 0.0:
            raise ValueError("dt_min must be finite and positive")
        if dt_max is not None:
            if not np.isfinite(dt_max) or dt_max <= 0.0:
                raise ValueError("dt_max must be finite and positive when provided")
            if dt_max < dt_min:
                raise ValueError("dt_max must be greater than or equal to dt_min")

    def _compute_cfl_limit(self) -> float:
        """Compute the maximum stable time step based on CFL condition."""
        mesh = self._mesh
        D = self._D
        if D == 0.0:
            return float("inf")
        dx2 = mesh.dx() ** 2
        return dx2 / (2.0 * D)

    def _create_solver(self, initial: np.ndarray):
        """Create a fresh solver with the given initial condition."""
        # Configuration was restricted to pure diffusion during construction.
        solver = DiffusionSolver(self._mesh, self._D)
        solver.set_initial_condition(initial.tolist())

        solver.set_boundary_condition(0, self._left_bc)
        solver.set_boundary_condition(1, self._right_bc)

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
        t_end = float(t_end)
        if not np.isfinite(t_end) or t_end <= 0.0:
            raise ValueError("t_end must be finite and positive")
        if dt_initial is not None:
            dt_initial = float(dt_initial)
            if not np.isfinite(dt_initial) or dt_initial <= 0.0:
                raise ValueError("dt_initial must be finite and positive")

        # Initialize
        u = self._initial.copy()
        t = 0.0
        if dt_initial is not None:
            dt = dt_initial
        elif np.isfinite(self._cfl_limit):
            dt = self.config.safety * self._cfl_limit
        else:
            dt = t_end

        # Respect both the user ceiling and the exact diffusion stability limit.
        dt = min(dt, self.config.dt_max, self._cfl_limit, t_end)

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
            if not np.isfinite(error):
                raise FloatingPointError(
                    "adaptive error estimate became non-finite; reduce the time step "
                    "or inspect the initial field"
                )

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
    Convenience wrapper for the limited legacy adaptive stepper.

    Parameters
    ----------
    problem : TransportProblem
        A 1D uniform-diffusion problem with Dirichlet endpoint conditions.
        General transport configurations raise instead of being simplified.
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
