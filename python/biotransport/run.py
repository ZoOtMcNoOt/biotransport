"""Thin, user-friendly access to the canonical C++ transport solver."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from ._core import (
    Boundary,
    BoundaryCondition,
    SolveDiagnostics,
    SolveOptions,
    TransportProblem,
    solve_transport,
)
from .solution import Solution


@dataclass(frozen=True)
class CheckpointResult(Mapping[float, np.ndarray]):
    """Owned checkpoint fields plus per-segment native diagnostics.

    The object behaves as a read-only mapping from absolute checkpoint time to
    a read-only NumPy field, so existing ``result[time]`` and ``dict(result)``
    usage remains natural. Diagnostics are keyed by the same absolute times;
    each diagnostic's requested/final time is the duration of that segment.
    """

    fields: Mapping[float, np.ndarray]
    diagnostics: Mapping[float, SolveDiagnostics]
    total_steps: int

    def __post_init__(self) -> None:
        field_items = list(self.fields.items())
        diagnostic_items = list(self.diagnostics.items())
        field_times = tuple(
            _finite_real(time, f"fields key {index}")
            for index, (time, _field) in enumerate(field_items)
        )
        diagnostic_times = tuple(
            _finite_real(time, f"diagnostics key {index}")
            for index, (time, _diagnostic) in enumerate(diagnostic_items)
        )
        if (
            field_times != tuple(sorted(field_times))
            or any(time <= 0.0 for time in field_times)
            or any(right <= left for left, right in zip(field_times, field_times[1:]))
        ):
            raise ValueError(
                "checkpoint field times must be strictly increasing and positive"
            )
        if field_times != diagnostic_times:
            raise ValueError(
                "checkpoint fields and diagnostics must share ordered keys"
            )
        if not field_times:
            raise ValueError("at least one checkpoint field is required")

        owned_fields: dict[float, np.ndarray] = {}
        field_size: int | None = None
        for time, (_original_time, field) in zip(field_times, field_items):
            snapshot = _owned_finite_real_array(field, "checkpoint field")
            if snapshot.size == 0:
                raise ValueError("checkpoint fields must not be empty")
            if field_size is None:
                field_size = int(snapshot.size)
            elif snapshot.size != field_size:
                raise ValueError(
                    "all checkpoint fields must have the same number of values"
                )
            snapshot.setflags(write=False)
            owned_fields[time] = snapshot

        owned_diagnostics: dict[float, SolveDiagnostics] = {}
        previous_time = 0.0
        for time, (_original_time, diagnostic) in zip(
            diagnostic_times, diagnostic_items
        ):
            if not isinstance(diagnostic, SolveDiagnostics):
                raise TypeError(
                    "checkpoint diagnostics must be SolveDiagnostics objects"
                )
            segment_duration = time - previous_time
            requested_time = _finite_real(
                diagnostic.requested_final_time,
                f"diagnostics[{time}].requested_final_time",
            )
            final_time = _finite_real(
                diagnostic.final_time, f"diagnostics[{time}].final_time"
            )
            if requested_time != segment_duration or final_time != segment_duration:
                raise ValueError(
                    "each checkpoint diagnostic time must equal its segment duration"
                )
            if int(diagnostic.steps) <= 0:
                raise ValueError(
                    "each positive checkpoint segment must report at least one step"
                )
            owned_diagnostics[time] = diagnostic
            previous_time = time

        if isinstance(self.total_steps, bool) or not isinstance(
            self.total_steps, Integral
        ):
            raise TypeError("total_steps must be a non-negative integer")
        if self.total_steps < 0:
            raise ValueError("total_steps must be non-negative")
        diagnostic_steps = sum(
            int(diagnostic.steps) for diagnostic in owned_diagnostics.values()
        )
        if int(self.total_steps) != diagnostic_steps:
            raise ValueError(
                "total_steps must equal the sum of segment diagnostic steps"
            )
        object.__setattr__(self, "fields", MappingProxyType(owned_fields))
        object.__setattr__(self, "diagnostics", MappingProxyType(owned_diagnostics))
        object.__setattr__(self, "total_steps", int(self.total_steps))

    @property
    def times(self) -> tuple[float, ...]:
        """Return sorted absolute checkpoint times."""

        return tuple(self.fields)

    def __getitem__(self, time: float) -> np.ndarray:
        return self.fields[time]

    def __iter__(self) -> Iterator[float]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)


def _finite_real(value: object, name: str) -> float:
    """Return one finite real value without accepting booleans or text."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _owned_finite_real_array(value: object, name: str) -> np.ndarray:
    """Copy one flat, finite, genuinely real numeric array."""

    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ValueError(f"{name} must not contain masked values")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a flat array")
    result = raw.astype(np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _saved_times(
    final_time: float,
    save_every: float | None,
    save_at: Sequence[float] | None,
    frames: int | None,
) -> tuple[float, ...]:
    """Work out which times to stop and record a field at.

    Always ends exactly on ``final_time``. Returns an empty tuple when nothing
    needs to be saved along the way.
    """

    chosen = [
        name
        for name, value in (
            ("save_every", save_every),
            ("save_at", save_at),
            ("frames", frames),
        )
        if value is not None
    ]
    if len(chosen) > 1:
        raise TypeError(
            f"pass only one of save_every, save_at or frames (got {', '.join(chosen)})"
        )
    if not chosen or final_time == 0.0:
        return ()

    if save_every is not None:
        interval = _finite_real(save_every, "save_every")
        if interval <= 0.0:
            raise ValueError("save_every must be positive")
        if interval > final_time:
            raise ValueError(
                f"save_every={interval:g} is longer than end_time={final_time:g}, "
                f"so nothing between the start and the end would be saved"
            )
        count = int(math.floor(final_time / interval + 1.0e-9))
        times = [interval * (index + 1) for index in range(count)]
    elif frames is not None:
        if isinstance(frames, bool) or not isinstance(frames, Integral):
            raise TypeError("frames must be a positive integer")
        count = int(frames)
        if count < 1:
            raise ValueError("frames must be at least 1")
        times = [final_time * (index + 1) / count for index in range(count)]
    else:
        assert save_at is not None
        if isinstance(save_at, (str, bytes)):
            raise TypeError("save_at must be a sequence of times")
        times = sorted(
            _finite_real(value, f"save_at[{index}]")
            for index, value in enumerate(save_at)
        )
        if any(value <= 0.0 for value in times):
            raise ValueError("save_at times must be positive")
        if any(value > final_time for value in times):
            raise ValueError(
                f"save_at contains times beyond end_time={final_time:g}; "
                f"raise end_time or drop those entries"
            )

    # Land exactly on the end, and drop anything that duplicates it.
    times = [value for value in times if value < final_time * (1.0 - 1.0e-12)]
    times.append(final_time)
    return tuple(times)


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
    save_every: float | None = None,
    save_at: Sequence[float] | None = None,
    frames: int | None = None,
    steady: bool = False,
) -> Solution:
    """Run a transport problem forward in time.

    All the arithmetic happens in the C++ core. This function checks your
    arguments, hands them over, and wraps what comes back in a
    :class:`~biotransport.Solution` that knows its own mesh.

    Args:
        problem: Physics built with :class:`biotransport.Problem`.
        end_time: How long to run for. The solver lands on this time exactly,
            shortening its last step if it has to.
        time_step: Largest step to take. Leave it out and the core picks a step
            that is provably stable for your grid and coefficients. A custom
            reaction without a declared derivative bound needs this set.
        safety_factor: What fraction of the certified stability limit to use
            when choosing a step automatically. Default 0.8.
        reaction_step_fraction: Accuracy guard against a known reaction
            timescale. Default 0.1.
        max_steps: Refuse to run longer than this many steps. Counted across
            every segment when you are saving frames.
        check_finite: Reject a run that produces a NaN or infinity.
        save_every: Also record the field at this time interval, so you can
            plot or animate the evolution.
        save_at: Record the field at these specific times.
        frames: Record this many equally spaced fields, ending at ``end_time``.
        steady: Skip the transient entirely and solve for the steady state.
            Equivalent to :func:`biotransport.solve_steady`; leave ``end_time``
            out when you use it.

    Returns:
        A :class:`~biotransport.Solution`. It behaves like the native result
        (``concentration``, ``time``, ``diagnostics``) and adds geometry, saved
        frames, plotting and comparison.

    Raises:
        ValueError: If the configuration is outside what this solver certifies
            -- an unstable step, an unverified method, an uncertified reaction
            with automatic stepping. The library refuses rather than quietly
            solving a different problem.

    Example:
        >>> sol = bt.solve(problem, end_time=0.1)
        >>> sol.plot()

        Recording the evolution so you can watch it:

        >>> sol = bt.solve(problem, end_time=0.1, save_every=0.01)
        >>> sol.plot(times=[0.0, 0.02, 0.05, 0.1])

    Note:
        ``t`` and ``dt`` still work as aliases for ``end_time`` and
        ``time_step``. ``method`` accepts ``"conservative"`` or ``"explicit"``,
        which name the same verified algorithm; other schemes live behind their
        own APIs until they carry the same evidence.

        Saving frames splits the run into segments that each start their clock
        at zero, so a reaction written as ``R(c, x, y, t)`` gets its clock
        shifted back to absolute time. That happens automatically for problems
        built with :class:`biotransport.Problem`. Segment boundaries also pin
        step boundaries, so a saved run can follow a very slightly different
        (equally valid) discrete path than the same run done in one shot.
    """
    if not isinstance(problem, TransportProblem):
        raise TypeError("problem must be a TransportProblem")
    if steady:
        from .steady import solve_steady

        if end_time is not None or t is not None:
            raise TypeError(
                "a steady solve has no end time -- drop end_time, or leave "
                "steady out to march through the transient"
            )
        if save_every is not None or save_at is not None or frames is not None:
            raise TypeError(
                "a steady solve produces a single field, so there are no frames to save"
            )
        return solve_steady(problem)
    if end_time is not None and t is not None:
        raise TypeError("Pass either end_time or t, not both")
    if end_time is None:
        end_time = t
    if end_time is None:
        raise TypeError("end_time is required")
    final_time = _finite_real(end_time, "end_time")
    if final_time < 0.0:
        raise ValueError("end_time must be non-negative")

    if time_step is not None and dt is not None:
        raise TypeError("Pass either time_step or dt, not both")
    if time_step is None:
        time_step = dt
    requested_step = 0.0
    if time_step is not None:
        requested_step = _finite_real(time_step, "time_step")
        if requested_step <= 0.0:
            raise ValueError("time_step must be positive when provided")

    if not isinstance(method, str):
        raise TypeError("method must be a string")
    normalized_method = method.lower().replace("-", "_")
    if normalized_method not in {"conservative", "explicit", "explicit_euler"}:
        raise ValueError(
            f"solve() runs the verified conservative explicit scheme; {method!r} is "
            f"not one of its names ('conservative', 'explicit', 'explicit_euler'). "
            f"Other algorithms have their own classes -- CrankNicolsonDiffusion, "
            f"ADIDiffusion2D, ImplicitDiffusion2D -- because they carry different "
            f"stability and accuracy evidence."
        )

    safety = _finite_real(safety_factor, "safety_factor")
    if not 0.0 < safety <= 1.0:
        raise ValueError("safety_factor must be in (0, 1]")
    reaction_fraction = _finite_real(reaction_step_fraction, "reaction_step_fraction")
    if reaction_fraction <= 0.0:
        raise ValueError("reaction_step_fraction must be positive")
    if isinstance(max_steps, bool) or not isinstance(max_steps, Integral):
        raise TypeError("max_steps must be a positive integer")
    step_limit = int(max_steps)
    if step_limit <= 0:
        raise ValueError("max_steps must be a positive integer")
    if not isinstance(check_finite, bool):
        raise TypeError("check_finite must be a boolean")

    def _options(duration: float, remaining: int) -> SolveOptions:
        options = SolveOptions()
        options.final_time = duration
        options.time_step = requested_step
        options.safety_factor = safety
        options.reaction_step_fraction = reaction_fraction
        options.max_steps = remaining
        options.check_finite = check_finite
        return options

    mesh = problem.mesh()
    starting_field = _starting_field(problem, mesh)
    checkpoints = _saved_times(final_time, save_every, save_at, frames)

    # The common case: one native solve, no problem mutation at all.
    if len(checkpoints) <= 1:
        try:
            native = solve_transport(problem, _options(final_time, step_limit))
        except Exception as error:
            raise _explain_step_error(problem, mesh, requested_step, error) from None
        times: list[float] = []
        fields: list[np.ndarray] = []
        if starting_field is not None and native.time > 0.0:
            times.append(0.0)
            fields.append(starting_field)
        times.append(float(native.time))
        fields.append(np.asarray(native.concentration, dtype=np.float64))
        return Solution(
            mesh=mesh,
            times=times,
            fields=fields,
            diagnostics=[native.diagnostics],
            problem=problem,
            total_steps=int(native.diagnostics.steps),
        )

    return _solve_in_segments(
        problem,
        mesh=mesh,
        starting_field=starting_field,
        checkpoints=checkpoints,
        options_for=_options,
        step_limit=step_limit,
        requested_step=requested_step,
    )


def _step_limits(problem: TransportProblem, mesh) -> list[tuple[str, str, float]]:
    """Per-process explicit step limits, as ``(process, formula, limit)``.

    Computed from the configured coefficients so an error can say *which* term
    is throttling the run, not just that something is.
    """

    recipe = getattr(problem, "_recipe", None)
    if recipe is None:
        return []

    limits: list[tuple[str, str, float]] = []
    dx = float(mesh.x(1) - mesh.x(0))
    is_1d = bool(mesh.is_1d())

    diffusivity = recipe._diffusivity_scale()
    if diffusivity and diffusivity > 0.0:
        if is_1d:
            limits.append(("diffusion", "dx^2 / (2 D)", dx * dx / (2.0 * diffusivity)))
        else:
            dy = float(mesh.y(0, 1) - mesh.y(0, 0))
            limit = 1.0 / (2.0 * diffusivity * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
            limits.append(("diffusion", "1 / (2 D (1/dx^2 + 1/dy^2))", limit))

    speed = recipe._speed()
    if speed and speed > 0.0:
        limits.append(("advection", "dx / |v|", dx / speed))

    if problem.reaction_stability_bound_known():
        bound = float(problem.reaction_stability_rate_bound())
        if bound > 0.0:
            limits.append(("reaction", "1 / max|dR/dc|", 1.0 / bound))

    return limits


def _explain_step_error(
    problem: TransportProblem,
    mesh,
    requested_step: float,
    error: Exception,
) -> Exception:
    """Rebuild a bare stability complaint into something a student can act on."""

    text = str(error).lower()
    # Two different failures land here and they need opposite advice: a step that
    # is too large, and automatic stepping with nothing to size itself from.
    unbounded_reaction = "derivative bound" in text or "max_abs_dc" in text
    too_large = "stability" in text or "time_step" in text or "time step" in text
    if not (unbounded_reaction or too_large):
        return error

    try:
        if unbounded_reaction:
            lines = [str(error).rstrip("."), ""]
            lines.append(
                "  Automatic stepping sizes the step from the fastest process in the "
                "problem, and it"
            )
            lines.append(
                "  cannot differentiate your reaction to find out how fast that is. So:"
            )
            lines.append("")
            lines.append(
                "    declare the bound         problem.reaction(f, max_abs_dc=B)"
            )
            lines.append(
                "                              where B >= max |dR/dc| over the "
                "concentrations you expect"
            )
            lines.append(
                "    or pick the step yourself bt.solve(problem, end_time=..., "
                "time_step=...)"
            )
            transport = _step_limits(problem, mesh)
            if transport:
                smallest = min(limit for _name, _formula, limit in transport)
                lines.append("")
                lines.append(
                    f"  transport alone would allow about dt = {smallest:.4g}, so start "
                    f"below that."
                )
            return type(error)("\n".join(lines))

        limits = _step_limits(problem, mesh)
        certified = None
        stable_probe = getattr(problem, "stable_time_step", None)
        if callable(stable_probe):
            certified = float(stable_probe())

        lines = [str(error).rstrip(".")]
        lines.append("")
        if requested_step > 0.0:
            lines.append(f"  you asked for dt = {requested_step:.4g}")
        if certified and certified > 0.0:
            lines.append(f"  the certified stable limit here is dt = {certified:.4g}")
        if limits:
            lines.append("")
            lines.append(
                "  which process is squeezing you (indicative scalings; the certified "
                "limit above"
            )
            lines.append("  combines them):")
            for index, (name, formula, limit) in enumerate(
                sorted(limits, key=lambda item: item[2])
            ):
                marker = "  <-- smallest" if index == 0 else ""
                lines.append(f"    {name:<10} {formula:<28} = {limit:.4g}{marker}")
        lines.append("")
        lines.append("  your options:")
        lines.append(
            "    - leave time_step out entirely and let the solver choose a stable step"
        )
        if limits and min(limits, key=lambda item: item[2])[0] == "diffusion":
            lines.append(
                "    - use a coarser mesh: the diffusion limit scales as dx^2, so "
                "halving the cell count buys you 4x the step"
            )
        lines.append(
            "    - use an implicit solver, which has no step limit: "
            "CrankNicolsonDiffusion (1D), ADIDiffusion2D or ImplicitDiffusion2D (2D)"
        )
        lines.append(
            "    - if you only want the final steady answer, skip the transient "
            "with bt.solve_steady(problem)"
        )
        return type(error)("\n".join(lines))
    except Exception:  # noqa: BLE001 - diagnosis must never mask the real error
        return error


def _starting_field(problem: TransportProblem, mesh) -> np.ndarray | None:
    """Read the configured initial condition back, when the core exposes it."""

    getter = getattr(problem, "initial", None)
    if not callable(getter):
        return None
    try:
        raw = getter()
    except TypeError:
        return None
    field = np.asarray(raw, dtype=np.float64).reshape(-1)
    if field.size != int(mesh.num_nodes()):
        return None
    return field


def _solve_in_segments(
    problem: TransportProblem,
    *,
    mesh,
    starting_field: np.ndarray | None,
    checkpoints: tuple[float, ...],
    options_for,
    step_limit: int,
    requested_step: float = 0.0,
) -> Solution:
    """Advance through the checkpoints, recording a field at each one.

    Each segment reuses the same problem object, so every configured term
    survives; only the initial condition is replaced between segments. The
    problem is put back the way it was found before returning.
    """

    if starting_field is None:
        raise ValueError(
            "saving frames needs to read the problem's initial condition, which "
            "this problem does not expose. Build it with bt.Problem(mesh) and set "
            "an initial condition."
        )

    recipe = getattr(problem, "_recipe", None)
    needs_clock_shift = recipe is not None and recipe.has_custom_reaction

    times: list[float] = [0.0]
    fields: list[np.ndarray] = [starting_field.copy()]
    diagnostics: list[SolveDiagnostics] = []
    current = starting_field.copy()
    elapsed = 0.0
    total_steps = 0

    try:
        for target in checkpoints:
            remaining = step_limit - total_steps
            if remaining <= 0:
                raise RuntimeError(
                    f"max_steps={step_limit} ran out at t={elapsed:g} before "
                    f"reaching t={checkpoints[-1]:g}. Raise max_steps, or shorten "
                    f"end_time."
                )
            if needs_clock_shift:
                recipe.replay_reactions(problem, elapsed)
            # Native call: the field is already validated, and going through a
            # recording override would be redundant work inside the loop.
            TransportProblem.initial_condition(problem, current.tolist())

            duration = target - elapsed
            try:
                native = solve_transport(problem, options_for(duration, remaining))
            except Exception as error:
                raise _explain_step_error(
                    problem, mesh, requested_step, error
                ) from None
            if native.time != duration:
                raise RuntimeError(
                    "the native solver did not land exactly on the requested "
                    f"segment: asked for {duration}, reached {native.time}"
                )
            current = np.asarray(native.concentration, dtype=np.float64)
            times.append(target)
            fields.append(current.copy())
            diagnostics.append(native.diagnostics)
            total_steps += int(native.diagnostics.steps)
            elapsed = target
    finally:
        # Leave the problem exactly as the caller configured it.
        if needs_clock_shift:
            recipe.replay_reactions(problem, 0.0)
        TransportProblem.initial_condition(problem, starting_field.tolist())

    return Solution(
        mesh=mesh,
        times=times,
        fields=fields,
        diagnostics=diagnostics,
        problem=problem,
        total_steps=total_steps,
    )


def run(problem: TransportProblem, t_end: float, **kwargs) -> Solution:
    """Older name for :func:`solve`, kept so existing scripts keep working.

    New code should call ``bt.solve(problem, end_time=...)``.
    """
    return solve(problem, end_time=t_end, **kwargs)


def run_checkpoints(
    mesh,
    checkpoints: Sequence[float],
    diffusivity: float,
    initial_condition=None,
    boundaries: Mapping[Boundary, BoundaryCondition] | None = None,
    **solve_kwargs,
) -> CheckpointResult:
    """Solve pure diffusion in C++ and return fields at requested times.

    Superseded by ``bt.solve(problem, end_time=..., save_at=[...])``, which does
    the same thing for *any* configured problem and hands back a
    :class:`~biotransport.Solution` you can plot directly. This function stays
    for older scripts.

    It is deliberately scoped to uniform diffusion, because it rebuilds the
    problem from scratch for each segment and so cannot carry a reaction or a
    velocity across. Checkpoints may
    be supplied in any order; returned keys are sorted physical times. Each
    checkpoint segment lands exactly on its requested duration and keeps its
    native diagnostics.

    Segment boundaries partition time stepping. A run is bitwise comparable
    with a one-shot solve only when the same requested step divides every
    segment; automatic or shortened steps may produce a slightly different
    valid discrete trajectory. ``max_steps`` is enforced cumulatively across
    all segments.
    """
    if isinstance(checkpoints, (str, bytes)):
        raise TypeError("checkpoints must be a sequence of real times")
    try:
        raw_checkpoints = list(checkpoints)
    except TypeError as error:
        raise TypeError("checkpoints must be a sequence of real times") from error
    if not raw_checkpoints:
        raise ValueError("checkpoints must not be empty")
    times = sorted(
        _finite_real(value, f"checkpoints[{index}]")
        for index, value in enumerate(raw_checkpoints)
    )
    if times[0] <= 0.0 or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError(
            "checkpoints must be unique, strictly increasing, and positive"
        )

    if not hasattr(mesh, "num_nodes") or not callable(mesh.num_nodes):
        raise TypeError("mesh must provide num_nodes()")
    node_count = int(mesh.num_nodes())
    if node_count <= 0:
        raise ValueError("mesh must contain at least one node")
    diffusion = _finite_real(diffusivity, "diffusivity")
    if diffusion < 0.0:
        raise ValueError("diffusivity must be non-negative")

    if initial_condition is None:
        current = [0.0] * node_count
    elif isinstance(initial_condition, Real) and not isinstance(
        initial_condition, bool
    ):
        initial_value = _finite_real(initial_condition, "initial_condition")
        current = [initial_value] * node_count
    else:
        if isinstance(initial_condition, (str, bytes)):
            raise TypeError("initial_condition must be a real scalar or field")
        try:
            raw_initial = list(initial_condition)
        except TypeError as error:
            raise TypeError(
                "initial_condition must be a real scalar or field"
            ) from error
        current = [
            _finite_real(value, f"initial_condition[{index}]")
            for index, value in enumerate(raw_initial)
        ]
    if len(current) != node_count:
        raise ValueError(
            f"initial_condition has {len(current)} values; the mesh requires {node_count}"
        )
    if boundaries is not None and not isinstance(boundaries, Mapping):
        raise TypeError("boundaries must be a mapping or None")
    if boundaries is not None:
        for side, condition in boundaries.items():
            if not isinstance(side, Boundary):
                raise TypeError("boundary mapping keys must be Boundary values")
            if not isinstance(condition, BoundaryCondition):
                raise TypeError(
                    "boundary mapping values must be BoundaryCondition objects"
                )
    if "end_time" in solve_kwargs or "t" in solve_kwargs:
        raise TypeError(
            "run_checkpoints controls each segment end time; do not pass end_time or t"
        )

    total_step_limit = solve_kwargs.pop("max_steps", 10_000_000)
    if isinstance(total_step_limit, bool) or not isinstance(total_step_limit, Integral):
        raise TypeError("max_steps must be a positive integer")
    total_step_limit = int(total_step_limit)
    if total_step_limit <= 0:
        raise ValueError("max_steps must be a positive integer")

    fields: dict[float, np.ndarray] = {}
    diagnostics: dict[float, SolveDiagnostics] = {}
    current_time = 0.0
    total_steps = 0

    for target_time in times:
        remaining_steps = total_step_limit - total_steps
        if remaining_steps <= 0:
            raise RuntimeError(
                "cumulative max_steps was exhausted before the final checkpoint"
            )
        problem = (
            TransportProblem(mesh)
            .diffusivity(diffusion)
            .initial_condition(np.asarray(current, dtype=np.float64))
        )
        if boundaries:
            for side, condition in boundaries.items():
                problem.boundary(side, condition)

        segment_duration = target_time - current_time
        result = solve(
            problem,
            end_time=segment_duration,
            max_steps=remaining_steps,
            **solve_kwargs,
        )
        if result.time != segment_duration:
            raise RuntimeError(
                "the native solver did not land exactly on the requested checkpoint "
                f"segment: requested {segment_duration}, reached {result.time}"
            )
        concentration = result.concentration
        current = concentration.tolist()
        fields[target_time] = concentration
        diagnostics[target_time] = result.diagnostics
        total_steps += int(result.diagnostics.steps)
        current_time = target_time

    return CheckpointResult(
        fields=fields,
        diagnostics=diagnostics,
        total_steps=total_steps,
    )
