"""
Time Integration Methods for Transport Simulations.

Provides higher-order time integration methods beyond Forward Euler:
- Heun's method (RK2) - 2nd order accurate
- Classic RK4 - 4th order accurate

These methods offer better time accuracy for problems with:
- Smooth solutions
- Stiff reactions
- Long-time integration

Note: Higher-order methods require more function evaluations per step
but can use larger time steps while maintaining accuracy.

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
    TransportProblem,
    StructuredMesh,
)


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
    boundary_values: tuple,
) -> np.ndarray:
    """Compute the right-hand side for the diffusion equation: du/dt = D * d²u/dx².

    Uses second-order central differences for the Laplacian.

    Args:
        mesh: The computational mesh
        u: Current solution array
        D: Diffusion coefficient
        boundary_values: Tuple of (left_value, right_value) for Dirichlet BCs

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

    # Boundary nodes: Dirichlet (du/dt = 0 for fixed values)
    # Note: BCs are applied after the update, so we set dudt to maintain them
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
    """4th-order Runge-Kutta integrator for diffusion problems.

    This integrator uses method of lines: discretize in space first,
    then integrate the resulting ODE system in time using RK4.

    Compared to Forward Euler (1st order), RK4 provides:
    - 4th-order time accuracy (error ~ O(dt⁴))
    - Better accuracy for the same time step
    - Allows larger stable time steps for some problems

    Note: RK4 requires 4 function evaluations per step vs 1 for Euler,
    but the improved accuracy often allows much larger time steps.

    Example:
        >>> mesh = bt.mesh_1d(50, 0.0, 1.0)
        >>> problem = bt.Problem(mesh).diffusivity(0.01).initial_condition(u0)
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
        self.mesh = problem.mesh()
        self.D = problem.diffusivity()
        self.safety = safety_factor

        # Get initial condition
        self.u0 = np.array(problem.initial())

        # Get boundary conditions
        boundaries = problem.boundaries()
        self.left_bc = boundaries[0]
        self.right_bc = boundaries[1]

    def max_stable_dt(self) -> float:
        """Compute the maximum stable time step for RK4.

        For RK4 with the diffusion equation, stability requires:
        dt ≤ 2.78 * dx² / (4D)  (approximately)

        This is more permissive than Forward Euler's dt ≤ dx²/(2D).
        """
        dx = self.mesh.dx()
        # RK4 stability region is larger than Euler
        # For 1D diffusion: dt_max ≈ 2.78 * dx²/(4D) ≈ 0.7 * dx²/D
        # Use conservative factor
        dt_cfl = dx * dx / (2 * self.D)
        # RK4 can use about 2x larger dt than Euler
        return self.safety * 2.0 * dt_cfl

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

        # Determine time step
        dt_max = self.max_stable_dt()
        if dt is None:
            dt = dt_max
        else:
            # Enforce stability - use at most the stable dt
            dt = min(dt, dt_max)

        # Ensure we don't exceed t_end
        num_steps = int(np.ceil(t_end / dt))
        dt = t_end / num_steps

        # Initialize
        u = self.u0.copy()
        t = 0.0
        history = [u.copy()] if store_history else None

        # RHS function for diffusion
        def rhs(u_state: np.ndarray, t_val: float) -> np.ndarray:
            return _compute_diffusion_rhs(
                self.mesh,
                u_state,
                self.D,
                (self.left_bc.value, self.right_bc.value),
            )

        # Time integration loop
        start = time_module.perf_counter()

        for step in range(num_steps):
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
    """2nd-order Heun (improved Euler) integrator for diffusion problems.

    This is a good compromise between Forward Euler and RK4:
    - 2nd-order time accuracy (error ~ O(dt²))
    - Only 2 function evaluations per step (vs 4 for RK4)

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
        self.mesh = problem.mesh()
        self.D = problem.diffusivity()
        self.safety = safety_factor

        self.u0 = np.array(problem.initial())

        boundaries = problem.boundaries()
        self.left_bc = boundaries[0]
        self.right_bc = boundaries[1]

    def max_stable_dt(self) -> float:
        """Compute the maximum stable time step for Heun's method."""
        dx = self.mesh.dx()
        dt_cfl = dx * dx / (2 * self.D)
        # Heun has slightly better stability than Euler
        return self.safety * 1.5 * dt_cfl

    def solve(
        self,
        t_end: float,
        *,
        dt: Optional[float] = None,
        store_history: bool = False,
    ) -> IntegrationResult:
        """Solve the problem to t_end using Heun's method."""
        import time as time_module

        dt_max = self.max_stable_dt()
        if dt is None:
            dt = dt_max
        else:
            # Enforce stability
            dt = min(dt, dt_max)

        num_steps = int(np.ceil(t_end / dt))
        dt = t_end / num_steps

        u = self.u0.copy()
        t = 0.0
        history = [u.copy()] if store_history else None

        def rhs(u_state: np.ndarray, t_val: float) -> np.ndarray:
            return _compute_diffusion_rhs(
                self.mesh,
                u_state,
                self.D,
                (self.left_bc.value, self.right_bc.value),
            )

        start = time_module.perf_counter()

        for step in range(num_steps):
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
    """Solve a transport problem using the specified time integration method.

    This is a convenience function that selects the appropriate integrator.

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
    if method.lower() == "euler":
        # Use the standard ExplicitFD for Euler
        from .run import solve

        run_result = solve(problem, t_end, dt=dt)
        return IntegrationResult(
            solution=np.array(run_result.solution()),
            time=t_end,
            stats={
                "steps": run_result.stats.steps,
                "dt": run_result.stats.dt,
                "t_end": run_result.stats.t_end,
                "wall_time_s": run_result.stats.wall_time_s,
                "method": "euler",
            },
        )

    elif method.lower() == "heun":
        integrator = HeunIntegrator(problem)
        return integrator.solve(t_end, dt=dt)

    elif method.lower() == "rk4":
        integrator = RK4Integrator(problem)
        return integrator.solve(t_end, dt=dt)

    else:
        raise ValueError(
            f"Unknown integration method: {method}. "
            f"Choose from: 'euler', 'heun', 'rk4'"
        )
