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
from numbers import Integral
from typing import Callable, Optional

import numpy as np

from ._core import (
    BoundaryCondition,
    DiffusionSolver,
    TransportProblem,
)
from .time_integrators import (
    _LegacyDiffusionSnapshot,
    _capture_legacy_diffusion_problem,
    _finite_real_scalar,
    _scaled_positive_ratio,
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


@dataclass(frozen=True)
class AdaptiveTimeStepperConfig:
    """Immutable configuration that owns every adaptive-controller contract.

    Every field is normalized and validated during construction.  A solver
    keeps a private instance and exposes it read-only, so controller behavior
    cannot be changed silently between construction and :meth:`solve`.
    """

    tol: float = 1e-4
    """Relative error tolerance for step acceptance."""

    atol: float = 1e-8
    """Absolute error tolerance (for near-zero solutions)."""

    safety: float = 0.9
    """Safety factor for step size adjustment."""

    dt_min: float = 1e-12
    """Controller floor; an exact final remainder may be smaller."""

    dt_max: Optional[float] = None
    """Maximum allowed time step (None = CFL limit)."""

    max_factor: float = 2.0
    """Maximum factor for step size increase."""

    min_factor: float = 0.1
    """Minimum factor for step size decrease."""

    max_rejections: int = 100
    """Maximum consecutive rejections before error."""

    maximum_steps: int = 10_000_000
    """Maximum accepted steps before the Python controller stops."""

    def __post_init__(self) -> None:
        """Normalize and validate every value owned by the controller."""
        tol = _finite_real_scalar(self.tol, "tol")
        if tol <= 0.0:
            raise ValueError("tol must be finite and positive")
        atol = _finite_real_scalar(self.atol, "atol")
        if atol <= 0.0:
            raise ValueError("atol must be finite and positive")
        safety = _finite_real_scalar(self.safety, "safety")
        if not 0.0 < safety <= 1.0:
            raise ValueError("safety must be in (0, 1]")
        dt_min = _finite_real_scalar(self.dt_min, "dt_min")
        if dt_min <= 0.0:
            raise ValueError("dt_min must be finite and positive")
        dt_max = (
            None if self.dt_max is None else _finite_real_scalar(self.dt_max, "dt_max")
        )
        if dt_max is not None:
            if dt_max <= 0.0:
                raise ValueError("dt_max must be finite and positive when provided")
            if dt_max < dt_min:
                raise ValueError("dt_max must be greater than or equal to dt_min")

        max_factor = _finite_real_scalar(self.max_factor, "max_factor")
        if max_factor < 1.0:
            raise ValueError("max_factor must be greater than or equal to 1")
        min_factor = _finite_real_scalar(self.min_factor, "min_factor")
        if not 0.0 < min_factor < 1.0:
            raise ValueError("min_factor must be in (0, 1)")

        if (
            isinstance(self.max_rejections, bool)
            or not isinstance(self.max_rejections, Integral)
            or int(self.max_rejections) <= 0
        ):
            raise ValueError("max_rejections must be a positive integer")
        if (
            isinstance(self.maximum_steps, bool)
            or not isinstance(self.maximum_steps, Integral)
            or int(self.maximum_steps) <= 0
        ):
            raise ValueError("maximum_steps must be a positive integer")

        object.__setattr__(self, "tol", tol)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "safety", safety)
        object.__setattr__(self, "dt_min", dt_min)
        object.__setattr__(self, "dt_max", dt_max)
        object.__setattr__(self, "max_factor", max_factor)
        object.__setattr__(self, "min_factor", min_factor)
        object.__setattr__(self, "max_rejections", int(self.max_rejections))
        object.__setattr__(self, "maximum_steps", int(self.maximum_steps))


class AdaptiveTimeStepper:
    """
    Legacy adaptive controller for supported 1D uniform diffusion.

    .. warning::
       This is a legacy compatibility wrapper that orchestrates native
       diffusion steps from Python.  Prefer :func:`biotransport.solve` for the
       complete transport operator and a fully native solve loop.

    Uses local error estimation via step doubling to automatically adjust the
    time step.  It does not compose the complete :class:`TransportProblem`
    operator; unsupported configurations raise during construction and are
    rechecked at every solve.

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
        max_factor: float = 2.0,
        min_factor: float = 0.1,
        max_rejections: int = 100,
        maximum_steps: int = 10_000_000,
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
            Controller floor for ordinary proposals.  A shorter exact-final
            remainder is permitted so the result can land exactly on t_end.
        dt_max : float, optional
            Maximum allowed time step. Defaults to CFL limit.
        max_factor : float
            Largest factor by which an accepted step may grow (at least 1).
        min_factor : float
            Smallest rejection reduction factor, strictly between 0 and 1.
        max_rejections : int
            Maximum consecutive rejected attempts before raising.
        maximum_steps : int
            Maximum accepted steps before raising instead of running
            indefinitely.
        verbose : bool
            Print step information during solve.
        """
        self._problem = problem
        self._config = AdaptiveTimeStepperConfig(
            tol=tol,
            atol=atol,
            safety=safety,
            dt_min=dt_min,
            dt_max=dt_max,
            max_factor=max_factor,
            min_factor=min_factor,
            max_rejections=max_rejections,
            maximum_steps=maximum_steps,
        )
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be bool")
        self.verbose = verbose

        # Validate and own every value used by the initial solve.
        self._last_snapshot = _capture_legacy_diffusion_problem(
            problem,
            "AdaptiveTimeStepper",
        )

        # Compute CFL limit
        self._cfl_limit = self._compute_cfl_limit(self._last_snapshot)

        effective_dt_max = (
            self._cfl_limit
            if self.config.dt_max is None
            else min(self.config.dt_max, self._cfl_limit)
        )
        if effective_dt_max < self.config.dt_min:
            raise ValueError(
                "dt_min exceeds every stable permitted time step; decrease dt_min"
            )

    @property
    def config(self) -> AdaptiveTimeStepperConfig:
        """Return the immutable, validated controller configuration."""
        return self._config

    @property
    def problem(self) -> TransportProblem:
        """The live problem revalidated at the beginning of every solve."""
        return self._problem

    @staticmethod
    def _compute_cfl_limit(snapshot: _LegacyDiffusionSnapshot) -> float:
        """Compute the maximum stable time step based on CFL condition."""
        mesh = snapshot.mesh
        D = snapshot.diffusivity
        if D == 0.0:
            return float("inf")
        dx = mesh.dx()
        cfl_limit = _scaled_positive_ratio(
            (dx, dx),
            (2.0, D),
            conservative=True,
        )
        if cfl_limit == 0.0:
            raise FloatingPointError(
                "adaptive diffusion stability limit is below binary64 range"
            )
        return cfl_limit

    @staticmethod
    def _create_solver(
        initial: np.ndarray,
        snapshot: _LegacyDiffusionSnapshot,
    ) -> DiffusionSolver:
        """Create a fresh solver with the given initial condition."""
        # Configuration was restricted to pure diffusion during construction.
        solver = DiffusionSolver(snapshot.mesh, snapshot.diffusivity)
        solver.set_initial_condition(initial.tolist())

        solver.set_boundary_condition(
            0,
            BoundaryCondition.dirichlet(snapshot.left_value),
        )
        solver.set_boundary_condition(
            1,
            BoundaryCondition.dirichlet(snapshot.right_value),
        )

        return solver

    def _step(
        self,
        u: np.ndarray,
        dt: float,
        snapshot: _LegacyDiffusionSnapshot,
    ) -> np.ndarray:
        """Take a single time step from state u with step size dt."""
        if snapshot.diffusivity == 0.0:
            unchanged = u.copy()
            unchanged[0] = snapshot.left_value
            unchanged[-1] = snapshot.right_value
            return unchanged
        solver = self._create_solver(u, snapshot)
        solver.solve(dt, 1)
        return np.array(solver.solution())

    def _estimate_error(
        self,
        u: np.ndarray,
        dt: float,
        snapshot: _LegacyDiffusionSnapshot,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Estimate local error using step-doubling.

        Returns:
            (u_full, u_half, error) where:
            - u_full: Solution after one step of dt
            - u_half: Solution after two steps of dt/2 (more accurate)
            - error: Maximum relative error estimate
        """
        half_dt = dt / 2.0
        if half_dt <= 0.0 or half_dt + half_dt != dt:
            raise FloatingPointError(
                "adaptive step doubling cannot represent two equal half steps "
                "for the proposed dt"
            )

        # One step of size dt
        u_full = self._step(u, dt, snapshot)

        # Two steps of size dt/2
        u_mid = self._step(u, half_dt, snapshot)
        u_half = self._step(u_mid, half_dt, snapshot)

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
            Function called after each accepted step as ``callback(t, u)``.
            The array is an isolated copy and cannot alter the accepted state.

        Returns
        -------
        AdaptiveResult
            Solution and statistics.
        """
        t_end = _finite_real_scalar(t_end, "t_end")
        if t_end <= 0.0:
            raise ValueError("t_end must be finite and positive")
        if dt_initial is not None:
            dt_initial = _finite_real_scalar(dt_initial, "dt_initial")
            if dt_initial <= 0.0:
                raise ValueError("dt_initial must be finite and positive")
            if dt_initial < self.config.dt_min:
                raise ValueError("dt_initial must be greater than or equal to dt_min")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")

        # The problem is mutable.  Revalidate and own a complete supported
        # snapshot at every solve so no post-construction physics is omitted.
        snapshot = _capture_legacy_diffusion_problem(
            self._problem,
            "AdaptiveTimeStepper",
        )
        cfl_limit = self._compute_cfl_limit(snapshot)
        configured_dt_max = (
            cfl_limit if self.config.dt_max is None else self.config.dt_max
        )
        if min(configured_dt_max, cfl_limit) < self.config.dt_min:
            raise ValueError(
                "dt_min exceeds every stable permitted time step; decrease dt_min"
            )
        self._last_snapshot = snapshot
        self._cfl_limit = cfl_limit

        # Initialize from the fresh owned snapshot.
        u = snapshot.initial.copy()
        u[0] = snapshot.left_value
        u[-1] = snapshot.right_value
        t = 0.0
        if dt_initial is not None:
            dt = dt_initial
        elif np.isfinite(cfl_limit):
            dt = self.config.safety * cfl_limit
        else:
            dt = t_end

        # Respect both the user ceiling and the exact diffusion stability limit.
        dt = min(dt, configured_dt_max, cfl_limit, t_end)

        # Statistics tracking
        steps = 0
        rejections = 0
        consecutive_rejections = 0
        dt_history = []

        while t < t_end:
            if steps >= self.config.maximum_steps:
                raise RuntimeError(
                    "adaptive solve exceeded maximum_steps before reaching t_end"
                )

            # Don't overshoot, and retain the exact remaining interval so the
            # accepted final step can land on t_end without roundoff drift.
            remaining = t_end - t
            dt = min(dt, remaining)
            exact_final_below_minimum = dt == remaining and dt < self.config.dt_min
            if not np.isfinite(dt) or dt <= 0.0:
                raise FloatingPointError(
                    "adaptive time step became non-finite or non-positive"
                )

            # Estimate error
            u_full, u_half, error = self._estimate_error(u, dt, snapshot)
            if not np.isfinite(error):
                raise FloatingPointError(
                    "adaptive error estimate became non-finite; reduce the time step "
                    "or inspect the initial field"
                )

            if error <= 1.0:
                # Accept step (use the more accurate u_half)
                u = u_half
                if dt == remaining:
                    t = t_end
                else:
                    next_time = t + dt
                    if not np.isfinite(next_time) or next_time <= t:
                        raise FloatingPointError(
                            "adaptive time step is too small to advance time"
                        )
                    t = next_time
                steps += 1
                consecutive_rejections = 0
                dt_history.append(dt)

                if callback is not None:
                    callback(t, u.copy())

                if self.verbose:
                    print(f"  t={t:.6e}, dt={dt:.6e}, error={error:.2e} (accepted)")

                # Increase step size for next step
                if error > 0:
                    factor = self.config.safety * (1.0 / error) ** 0.5
                    factor = min(factor, self.config.max_factor)
                else:
                    factor = self.config.max_factor

                dt = min(dt * factor, configured_dt_max, cfl_limit)

            else:
                # Reject step
                rejections += 1
                consecutive_rejections += 1

                if exact_final_below_minimum:
                    raise RuntimeError(
                        "adaptive tolerance cannot be satisfied by the exact "
                        "final remainder below dt_min"
                    )
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
                proposed_dt = dt * factor
                if dt <= self.config.dt_min:
                    raise RuntimeError(
                        "adaptive tolerance cannot be satisfied at dt_min"
                    )
                # The minimum is a permitted step, not merely a lower bound
                # for the proposal.  Try it once before declaring failure.
                dt = max(proposed_dt, self.config.dt_min)

        # Build statistics
        stats = {
            "steps": steps,
            "rejections": rejections,
            "dt_min_used": min(dt_history) if dt_history else 0,
            "dt_max_used": max(dt_history) if dt_history else 0,
            "dt_avg": np.mean(dt_history) if dt_history else 0,
            "dt_history": dt_history,
            "cfl_limit": cfl_limit,
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
