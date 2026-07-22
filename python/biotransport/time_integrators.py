"""Legacy Python time-integration orchestration.

The standalone :func:`euler_step`, :func:`heun_step`, and :func:`rk4_step`
functions are generic ODE building blocks.  The :class:`HeunIntegrator` and
:class:`RK4Integrator` problem wrappers are deliberately narrower: they only
implement one-dimensional, uniform-diffusivity diffusion with fixed
Dirichlet values at the left and right boundaries.  They reject other
transport physics instead of silently dropping it.

For general transport problems, including reactions, sources, advection,
variable diffusivity, multidimensional meshes, or natural boundary
conditions, use :func:`biotransport.solve`.  The ``euler`` branch of
:func:`integrate` delegates to that canonical C++ solver.

Example:
    >>> mesh = bt.mesh_1d(100, 0.0, 1.0)
    >>> problem = bt.Problem(mesh).diffusivity(1e-5).initial_condition(u0)
    >>>
    >>> # RK4 integration
    >>> integrator = bt.RK4Integrator(problem)
    >>> result = integrator.solve(t_end=1.0, dt=0.01)
    >>> print(result.solution)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ._core import (
    BoundaryCondition,
    BoundaryType,
    StructuredMesh,
    TransportProblem,
)


_RK4_NEGATIVE_REAL_AXIS_RADIUS = 2.785293563405282


def _problem_flag(problem: TransportProblem, name: str, component: str) -> bool:
    """Read a native problem capability required for honest legacy dispatch."""
    accessor = getattr(problem, name, None)
    if accessor is None:
        raise RuntimeError(
            f"{component} requires a current biotransport native extension with "
            f"TransportProblem.{name}(). Rebuild the extension before using this "
            "legacy Python integrator."
        )
    return bool(accessor())


def _validate_legacy_diffusion_problem(
    problem: TransportProblem,
    component: str,
) -> tuple[StructuredMesh, float, BoundaryCondition, BoundaryCondition]:
    """Validate the intentionally narrow problem contract of Python steppers."""
    mesh = problem.mesh()
    if not mesh.is_1d():
        raise ValueError(
            f"{component} supports only 1D diffusion; use biotransport.solve "
            "for 2D or 3D transport."
        )
    if not _problem_flag(problem, "has_uniform_diffusivity", component):
        raise ValueError(
            f"{component} does not support variable diffusivity; "
            "use biotransport.solve."
        )
    if _problem_flag(problem, "has_reaction", component):
        raise ValueError(
            f"{component} does not support reactions or sources; "
            "use biotransport.solve."
        )
    if _problem_flag(problem, "has_advection", component):
        raise ValueError(
            f"{component} does not support advection; use biotransport.solve."
        )

    diffusivity = float(problem.diffusivity())
    if not np.isfinite(diffusivity) or diffusivity < 0.0:
        raise ValueError(f"{component} requires a finite, non-negative diffusivity")

    boundaries = problem.boundaries()
    left_bc = boundaries[0]
    right_bc = boundaries[1]
    if (
        left_bc.type != BoundaryType.DIRICHLET
        or right_bc.type != BoundaryType.DIRICHLET
    ):
        raise ValueError(
            f"{component} supports Dirichlet left/right boundaries only; "
            "use biotransport.solve for Neumann or Robin conditions."
        )

    return mesh, diffusivity, left_bc, right_bc


def _validate_safety_factor(value: float, component: str) -> float:
    safety = float(value)
    if not np.isfinite(safety) or not 0.0 < safety <= 1.0:
        raise ValueError(f"{component} safety_factor must be in (0, 1]")
    return safety


def _validate_solve_times(
    t_end: float, dt: Optional[float]
) -> tuple[float, Optional[float]]:
    final_time = float(t_end)
    if not np.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("t_end must be finite and positive")
    if dt is None:
        return final_time, None

    requested_dt = float(dt)
    if not np.isfinite(requested_dt) or requested_dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    return final_time, requested_dt


@dataclass
class IntegrationResult:
    """Result of a time integration."""

    solution: np.ndarray
    """Final solution field."""

    time: float
    """Final simulation time reached."""

    stats: dict = field(default_factory=dict)
    """Statistics including steps, wall time, etc."""


def _compute_diffusion_rhs(
    mesh: StructuredMesh,
    u: np.ndarray,
    D: float,
) -> np.ndarray:
    """Compute the 1D uniform-diffusion right-hand side.

    The caller owns enforcement of the validated Dirichlet boundary values.

    Args:
        mesh: The computational mesh
        u: Current solution array
        D: Diffusion coefficient
    Returns:
        Array of du/dt values
    """
    n = len(u)
    dx = mesh.dx()
    dx2 = dx * dx

    dudt = np.zeros(n)

    # Interior nodes: central difference
    for i in range(1, n - 1):
        dudt[i] = D * (u[i - 1] - 2 * u[i] + u[i + 1]) / dx2

    # Fixed Dirichlet nodes do not evolve.
    dudt[0] = 0.0
    dudt[-1] = 0.0

    return dudt


def euler_step(u: np.ndarray, rhs: Callable, t: float, dt: float) -> np.ndarray:
    """Forward Euler step: u^{n+1} = u^n + dt * f(u^n, t^n).

    First-order accurate: O(dt).

    Args:
        u: Current state
        rhs: Function that computes du/dt given (u, t)
        t: Current time
        dt: Time step

    Returns:
        New state at t + dt
    """
    return u + dt * rhs(u, t)


def heun_step(u: np.ndarray, rhs: Callable, t: float, dt: float) -> np.ndarray:
    """Heun's method (improved Euler / RK2).

    Second-order accurate: O(dt²).

    k1 = f(u^n, t^n)
    k2 = f(u^n + dt*k1, t^n + dt)
    u^{n+1} = u^n + dt/2 * (k1 + k2)

    Args:
        u: Current state
        rhs: Function that computes du/dt given (u, t)
        t: Current time
        dt: Time step

    Returns:
        New state at t + dt
    """
    k1 = rhs(u, t)
    k2 = rhs(u + dt * k1, t + dt)
    return u + dt / 2 * (k1 + k2)


def rk4_step(u: np.ndarray, rhs: Callable, t: float, dt: float) -> np.ndarray:
    """Classic 4th-order Runge-Kutta step.

    Fourth-order accurate: O(dt⁴).

    k1 = f(u^n, t^n)
    k2 = f(u^n + dt/2*k1, t^n + dt/2)
    k3 = f(u^n + dt/2*k2, t^n + dt/2)
    k4 = f(u^n + dt*k3, t^n + dt)
    u^{n+1} = u^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Args:
        u: Current state
        rhs: Function that computes du/dt given (u, t)
        t: Current time
        dt: Time step

    Returns:
        New state at t + dt
    """
    k1 = rhs(u, t)
    k2 = rhs(u + dt / 2 * k1, t + dt / 2)
    k3 = rhs(u + dt / 2 * k2, t + dt / 2)
    k4 = rhs(u + dt * k3, t + dt)
    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class RK4Integrator:
    """Legacy RK4 wrapper for a narrow 1D diffusion problem.

    This integrator uses method of lines: discretize in space first,
    then integrate the resulting ODE system in time using RK4.

    Only a uniform, non-negative diffusivity and Dirichlet conditions at both
    ends are supported.  General transport physics belongs in
    :func:`biotransport.solve`.

    Compared to Forward Euler (1st order), RK4 provides:
    - 4th-order time accuracy (error ~ O(dt⁴))
    - Better accuracy for the same time step
    - Allows larger stable time steps for some problems

    Note: RK4 requires 4 function evaluations per step vs 1 for Euler,
    but the improved accuracy often allows much larger time steps.

    Example:
        >>> mesh = bt.mesh_1d(50, 0.0, 1.0)
        >>> problem = (bt.Problem(mesh).diffusivity(0.01).initial_condition(u0)
        ...            .dirichlet(bt.Boundary.Left, 0.0)
        ...            .dirichlet(bt.Boundary.Right, 0.0))
        >>> integrator = bt.RK4Integrator(problem)
        >>> result = integrator.solve(t_end=1.0, dt=0.01)
    """

    def __init__(
        self,
        problem: TransportProblem,
        *,
        safety_factor: float = 0.5,
    ):
        """Initialize the RK4 integrator.

        Args:
            problem: The transport problem to solve
            safety_factor: Factor applied to CFL-based dt (default 0.5 for RK4)
        """
        self.problem = problem
        self.mesh, self.D, self.left_bc, self.right_bc = (
            _validate_legacy_diffusion_problem(problem, "RK4Integrator")
        )
        self.safety = _validate_safety_factor(safety_factor, "RK4Integrator")

        # Get initial condition
        self.u0 = np.asarray(problem.initial(), dtype=float).copy()

    def max_stable_dt(self) -> float:
        """Compute the maximum stable time step for RK4.

        For centered 1D diffusion, the most negative semi-discrete eigenvalue
        approaches ``-4D/dx²``.  Classical RK4 is stable on the negative real
        axis through approximately ``-2.7852935634``, so
        ``dt <= 2.7852935634 * dx² / (4D)``.
        """
        if self.D == 0.0:
            return float("inf")
        dx = self.mesh.dx()
        stability_limit = _RK4_NEGATIVE_REAL_AXIS_RADIUS * dx * dx / (4.0 * self.D)
        return self.safety * stability_limit

    def solve(
        self,
        t_end: float,
        *,
        dt: Optional[float] = None,
        store_history: bool = False,
    ) -> IntegrationResult:
        """Solve the problem to t_end using RK4.

        Args:
            t_end: End time
            dt: Time step (uses stable dt if not provided)
            store_history: If True, store solution at each step

        Returns:
            IntegrationResult with final solution and statistics
        """
        import time as time_module

        t_end, dt = _validate_solve_times(t_end, dt)

        # Determine time step
        dt_max = self.max_stable_dt()
        if dt is None:
            dt = t_end if not np.isfinite(dt_max) else dt_max
        else:
            # Enforce stability - use at most the stable dt
            dt = min(dt, dt_max)

        # Ensure we don't exceed t_end
        num_steps = int(np.ceil(t_end / dt))
        dt = t_end / num_steps

        # Initialize
        u = self.u0.copy()
        u[0] = self.left_bc.value
        u[-1] = self.right_bc.value
        t = 0.0
        history = [u.copy()] if store_history else None

        # RHS function for diffusion
        def rhs(u_state: np.ndarray, _t_val: float) -> np.ndarray:
            return _compute_diffusion_rhs(self.mesh, u_state, self.D)

        # Time integration loop
        start = time_module.perf_counter()

        for _ in range(num_steps):
            u = rk4_step(u, rhs, t, dt)

            # Apply boundary conditions
            u[0] = self.left_bc.value
            u[-1] = self.right_bc.value

            t += dt

            if store_history:
                history.append(u.copy())

        elapsed = time_module.perf_counter() - start

        # Build result
        stats = {
            "steps": num_steps,
            "dt": dt,
            "t_end": t,
            "wall_time_s": elapsed,
            "method": "rk4",
        }

        if store_history:
            stats["history"] = history

        return IntegrationResult(
            solution=u,
            time=t,
            stats=stats,
        )


class HeunIntegrator:
    """Legacy Heun wrapper for a narrow 1D diffusion problem.

    Within its supported uniform-diffusion contract, this offers:
    - 2nd-order time accuracy (error ~ O(dt²))
    - Only 2 function evaluations per step (vs 4 for RK4)

    Reactions, sources, advection, variable diffusivity, multidimensional
    meshes, and non-Dirichlet conditions are rejected.  Use
    :func:`biotransport.solve` for those problems.

    Example:
        >>> integrator = bt.HeunIntegrator(problem)
        >>> result = integrator.solve(t_end=1.0, dt=0.01)
    """

    def __init__(
        self,
        problem: TransportProblem,
        *,
        safety_factor: float = 0.8,
    ):
        """Initialize the Heun integrator.

        Args:
            problem: The transport problem to solve
            safety_factor: Factor applied to CFL-based dt
        """
        self.problem = problem
        self.mesh, self.D, self.left_bc, self.right_bc = (
            _validate_legacy_diffusion_problem(problem, "HeunIntegrator")
        )
        self.safety = _validate_safety_factor(safety_factor, "HeunIntegrator")

        self.u0 = np.asarray(problem.initial(), dtype=float).copy()

    def max_stable_dt(self) -> float:
        """Return the safety-scaled centered-diffusion stability limit.

        Explicit Heun/RK2 reaches ``-2`` on the negative real axis, exactly
        the same limit as Forward Euler for this semi-discrete operator.
        """
        if self.D == 0.0:
            return float("inf")
        dx = self.mesh.dx()
        dt_cfl = dx * dx / (2 * self.D)
        return self.safety * dt_cfl

    def solve(
        self,
        t_end: float,
        *,
        dt: Optional[float] = None,
        store_history: bool = False,
    ) -> IntegrationResult:
        """Solve the problem to t_end using Heun's method."""
        import time as time_module

        t_end, dt = _validate_solve_times(t_end, dt)
        dt_max = self.max_stable_dt()
        if dt is None:
            dt = t_end if not np.isfinite(dt_max) else dt_max
        else:
            # Enforce stability
            dt = min(dt, dt_max)

        num_steps = int(np.ceil(t_end / dt))
        dt = t_end / num_steps

        u = self.u0.copy()
        u[0] = self.left_bc.value
        u[-1] = self.right_bc.value
        t = 0.0
        history = [u.copy()] if store_history else None

        def rhs(u_state: np.ndarray, _t_val: float) -> np.ndarray:
            return _compute_diffusion_rhs(self.mesh, u_state, self.D)

        start = time_module.perf_counter()

        for _ in range(num_steps):
            u = heun_step(u, rhs, t, dt)

            u[0] = self.left_bc.value
            u[-1] = self.right_bc.value

            t += dt

            if store_history:
                history.append(u.copy())

        elapsed = time_module.perf_counter() - start

        stats = {
            "steps": num_steps,
            "dt": dt,
            "t_end": t,
            "wall_time_s": elapsed,
            "method": "heun",
        }

        if store_history:
            stats["history"] = history

        return IntegrationResult(
            solution=u,
            time=t,
            stats=stats,
        )


def integrate(
    problem: TransportProblem,
    t_end: float,
    *,
    method: str = "rk4",
    dt: Optional[float] = None,
) -> IntegrationResult:
    """Dispatch to canonical Euler or limited legacy Python integrators.

    ``method="euler"`` uses the canonical C++ transport solver and preserves
    every configured problem term.  ``"heun"`` and ``"rk4"`` retain the
    legacy Python orchestration and therefore accept only 1D uniform diffusion
    with Dirichlet conditions at both ends.

    Args:
        problem: The transport problem to solve
        t_end: End time
        method: Integration method - "euler", "heun", or "rk4"
        dt: Time step (uses method-specific stable dt if not provided)

    Returns:
        IntegrationResult with solution and statistics

    Example:
        >>> result = bt.integrate(problem, t_end=1.0, method="rk4")
        >>> print(f"Final solution: {result.solution}")
    """
    normalized_method = method.lower()
    if normalized_method not in {"euler", "heun", "rk4"}:
        raise ValueError(
            f"Unknown integration method: {method}. Choose from: 'euler', 'heun', 'rk4'"
        )

    if normalized_method == "euler":
        import time as time_module

        from .run import solve

        t_end, dt = _validate_solve_times(t_end, dt)
        start = time_module.perf_counter()
        run_result = solve(
            problem,
            end_time=t_end,
            time_step=dt,
            method="explicit",
        )
        elapsed = time_module.perf_counter() - start
        diagnostics = run_result.diagnostics
        return IntegrationResult(
            solution=np.asarray(run_result.concentration, dtype=float).copy(),
            time=float(run_result.time),
            stats={
                "steps": diagnostics.steps,
                "dt": diagnostics.maximum_time_step,
                "dt_min": diagnostics.minimum_time_step,
                "t_end": run_result.time,
                "wall_time_s": elapsed,
                "method": "euler",
                "diagnostics": diagnostics,
            },
        )

    if normalized_method == "heun":
        integrator = HeunIntegrator(problem)
        return integrator.solve(t_end, dt=dt)

    integrator = RK4Integrator(problem)
    return integrator.solve(t_end, dt=dt)
