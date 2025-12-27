"""Beginner-friendly run helpers.

The goal is a single, obvious entry point for the common case:
- configure a *Problem*
- run to an end time with conservative, stable defaults

Advanced users can still call `ExplicitFD().run(...)` directly.
"""

from __future__ import annotations

from typing import Sequence

from ._core import ExplicitFD, TransportProblem


def run(problem, t_end: float, *, solver: ExplicitFD | None = None):
    """Run a configured problem to `t_end` using the ExplicitFD façade.

    Args:
        problem: A TransportProblem instance
        t_end: End time in seconds
        solver: Optional ExplicitFD instance (creates one if not provided)

    Returns:
        RunResult: Result containing `.solution()` and `.stats`
    """
    if solver is None:
        solver = ExplicitFD()
    return solver.run(problem, float(t_end))


def solve(problem, t: float, *, safety_factor: float = 0.9):
    """Solve a transport problem to time t.

    This is the simplest way to run a simulation. Just configure your
    problem and call solve().

    Args:
        problem: A TransportProblem (or Problem) instance
        t: End time in seconds
        safety_factor: CFL safety factor (default 0.9)

    Returns:
        RunResult: Result with .solution() and .stats

    Example:
        >>> import biotransport as bt
        >>> mesh = bt.mesh_1d(100)
        >>> problem = bt.Problem(mesh).diffusivity(0.01).initial(0.0)
        >>> result = bt.solve(problem, t=0.1)
        >>> print(result.solution())
    """
    solver = ExplicitFD().safety_factor(safety_factor)
    return solver.run(problem, float(t))


def run_checkpoints(
    mesh,
    checkpoints: Sequence[float],
    diffusivity: float,
    initial_condition: list | None = None,
    boundaries: dict | None = None,
    *,
    solver: ExplicitFD | None = None,
    on_checkpoint=None,
) -> dict:
    """Run simulation saving results at multiple checkpoint times.

    This is a convenience function for multi-time simulations. It handles
    the boilerplate of chaining runs and updating initial conditions.

    Args:
        mesh: StructuredMesh instance
        checkpoints: List of times to save solutions (must be sorted, > 0)
        diffusivity: Diffusion coefficient (uniform)
        initial_condition: Initial field values (defaults to zeros)
        boundaries: Dict mapping Boundary to BoundaryCondition (optional)
        solver: Optional ExplicitFD instance
        on_checkpoint: Optional callback(t, solution) called at each checkpoint

    Returns:
        dict: Mapping from checkpoint time to solution array

    Example:
        >>> mesh = StructuredMesh(0, 0.01, 100)
        >>> ic = [1.0 if x < 0.002 else 0.0 for x in mesh.x_coords()]
        >>> results = run_checkpoints(
        ...     mesh,
        ...     checkpoints=[0.1, 0.5, 1.0, 5.0],
        ...     diffusivity=1e-9,
        ...     initial_condition=ic,
        ...     boundaries={bt.Boundary.LEFT: bt.BoundaryCondition.dirichlet(1.0)},
        ... )
        >>> # results[1.0] is the solution at t=1.0s
    """

    if solver is None:
        solver = ExplicitFD()

    # Validate checkpoints
    if not checkpoints:
        raise ValueError("checkpoints must not be empty")

    sorted_times = sorted(checkpoints)
    if sorted_times[0] <= 0:
        raise ValueError("All checkpoint times must be > 0")

    # Initialize
    if initial_condition is None:
        n_nodes = (mesh.nx() + 1) * (mesh.ny() + 1) if mesh.ny() > 0 else mesh.nx() + 1
        current_ic = [0.0] * n_nodes
    else:
        current_ic = list(initial_condition)

    results = {}
    current_time = 0.0

    for target_time in sorted_times:
        duration = target_time - current_time
        if duration <= 0:
            continue

        # Build problem for this segment
        problem = TransportProblem(mesh)
        problem.diffusivity(diffusivity)
        problem.initialCondition(current_ic)

        # Apply boundaries if provided
        if boundaries:
            for boundary, bc in boundaries.items():
                problem.boundary(boundary, bc)

        # Run segment
        result = solver.run(problem, duration)
        solution = list(result.solution())

        # Store result
        results[target_time] = solution

        # Callback
        if on_checkpoint:
            on_checkpoint(target_time, solution)

        # Update for next segment
        current_ic = solution
        current_time = target_time

    return results
