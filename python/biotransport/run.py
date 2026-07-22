"""Thin, user-friendly access to the canonical C++ transport solver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real

from ._core import (
    BoundaryCondition,
    SolveOptions,
    TransportProblem,
    TransportResult,
    solve_transport,
)


def solve(
    problem: TransportProblem,
    end_time: float | None = None,
    *,
    t: float | None = None,
    time_step: float | None = None,
    dt: float | None = None,
    safety_factor: float = 0.8,
    reaction_step_fraction: float = 0.1,
    max_steps: int = 10_000_000,
    check_finite: bool = True,
    method: str = "conservative",
) -> TransportResult:
    """Solve a scalar transport problem entirely in the C++ core.

    Parameters
    ----------
    problem:
        Physics configured with :class:`biotransport.Problem`.
    end_time:
        Requested physical end time. The result lands on this time exactly.
    time_step:
        Maximum explicit step. When omitted, the C++ solver selects a certified
        transport-stable step. Custom reactions require either this argument or
        a declared derivative bound.
    safety_factor:
        Fraction of the certified explicit stability limit used automatically.
    reaction_step_fraction:
        Accuracy guard relative to a known reaction timescale.

    Notes
    -----
    ``t`` and ``dt`` remain as compatibility aliases. ``method`` accepts
    ``"conservative"`` or ``"explicit"``; other algorithms are exposed through
    their specialized APIs until they share this scientific contract.
    """
    if end_time is not None and t is not None:
        raise TypeError("Pass either end_time or t, not both")
    if end_time is None:
        end_time = t
    if end_time is None:
        raise TypeError("end_time is required")

    if time_step is not None and dt is not None:
        raise TypeError("Pass either time_step or dt, not both")
    if time_step is None:
        time_step = dt

    normalized_method = method.lower().replace("-", "_")
    if normalized_method not in {"conservative", "explicit", "explicit_euler"}:
        raise ValueError(
            "The intuitive solve() API currently supports the verified conservative "
            "explicit solver only. Use a specialized solver explicitly for other methods."
        )

    options = SolveOptions()
    options.final_time = float(end_time)
    options.time_step = 0.0 if time_step is None else float(time_step)
    options.safety_factor = float(safety_factor)
    options.reaction_step_fraction = float(reaction_step_fraction)
    options.max_steps = int(max_steps)
    options.check_finite = bool(check_finite)
    return solve_transport(problem, options)


def run(problem: TransportProblem, t_end: float, **kwargs) -> TransportResult:
    """Compatibility alias for :func:`solve`; computation remains in C++."""
    return solve(problem, end_time=t_end, **kwargs)


def run_checkpoints(
    mesh,
    checkpoints: Sequence[float],
    diffusivity: float,
    initial_condition=None,
    boundaries: Mapping[object, BoundaryCondition] | None = None,
    **solve_kwargs,
) -> dict[float, object]:
    """Solve pure diffusion in C++ and return fields at requested times.

    This helper is deliberately scoped to uniform diffusion. For reactions or
    advection, construct a :class:`Problem` and call :func:`solve` so configured
    terms cannot be lost while rebuilding checkpoint segments.
    """
    if not checkpoints:
        raise ValueError("checkpoints must not be empty")
    times = sorted(float(value) for value in checkpoints)
    if times[0] <= 0.0 or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError(
            "checkpoints must be unique, strictly increasing, and positive"
        )

    node_count = mesh.num_nodes()
    if initial_condition is None:
        current = [0.0] * node_count
    elif isinstance(initial_condition, Real):
        current = [float(initial_condition)] * node_count
    else:
        current = list(initial_condition)
    if len(current) != node_count:
        raise ValueError(
            f"initial_condition has {len(current)} values; the mesh requires {node_count}"
        )
    results: dict[float, object] = {}
    current_time = 0.0

    for target_time in times:
        problem = (
            TransportProblem(mesh)
            .diffusivity(float(diffusivity))
            .initial_condition(current)
        )
        if boundaries:
            for side, condition in boundaries.items():
                problem.boundary(side, condition)

        result = solve(problem, end_time=target_time - current_time, **solve_kwargs)
        current = result.concentration.tolist()
        results[target_time] = result.concentration
        current_time = target_time

    return results
