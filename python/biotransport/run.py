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
    TransportResult,
    solve_transport,
)
from ._deprecation import deprecated_callable, warn_deprecated
from .results import Result, Snapshots

#: Identifier of the scientific contract behind :func:`solve`.
CANONICAL_CONTRACT_ID = "transport.canonical_explicit"


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


def _wrap_transport_result(native: TransportResult) -> Result:
    """Present a native ``TransportResult`` through the shared :class:`Result`."""
    diagnostics = native.diagnostics
    return Result(
        fields={"concentration": native.concentration},
        time=native.time,
        steps=int(diagnostics.steps),
        diagnostics=diagnostics,
        mesh=native.mesh,
        contract=CANONICAL_CONTRACT_ID,
        snapshots=Snapshots(
            tuple(native.snapshot_times), tuple(native.snapshot_fields)
        ),
        native=native,
    )


def _validated_save_times(save_times: object, final_time: float) -> list[float]:
    if save_times is None:
        return []
    if isinstance(save_times, (str, bytes)):
        raise TypeError("save_times must be a sequence of real times")
    try:
        raw = list(save_times)  # type: ignore[call-overload]
    except TypeError as error:
        raise TypeError("save_times must be a sequence of real times") from error
    values = [
        _finite_real(value, f"save_times[{index}]") for index, value in enumerate(raw)
    ]
    if any(value < 0.0 or value > final_time for value in values):
        raise ValueError("save_times must lie within [0, end_time]")
    if any(later <= earlier for earlier, later in zip(values, values[1:])):
        raise ValueError("save_times must be strictly increasing")
    return values


def solve(
    problem: TransportProblem,
    end_time: float | None = None,
    *,
    time_step: float | None = None,
    save_times: Sequence[float] | None = None,
    safety_factor: float = 0.8,
    reaction_step_fraction: float = 0.1,
    max_steps: int = 10_000_000,
    check_finite: bool = True,
    method: str = "conservative",
    t: float | None = None,
    dt: float | None = None,
) -> Result:
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
    save_times:
        Absolute times in ``[0, end_time]``, strictly increasing, at which the
        field is recorded. The C++ solver partitions its step schedule so each
        snapshot is captured exactly at that clock, and every configured term
        (including a time-dependent reaction) is preserved. The snapshots are
        returned as ``result.snapshots``.
    safety_factor:
        Fraction of the certified explicit stability limit used automatically.
    reaction_step_fraction:
        Accuracy guard relative to a known reaction timescale.

    Returns
    -------
    Result
        ``result.concentration`` (the final field), ``result.time`` (exactly
        ``end_time``), ``result.steps``, ``result.diagnostics``, ``result.mesh``,
        ``result.snapshots`` and ``result.plot()``.

    Notes
    -----
    ``t`` and ``dt`` are deprecated spellings of ``end_time`` and ``time_step``
    and emit :class:`~biotransport.BioTransportDeprecationWarning`. ``method``
    accepts ``"conservative"`` or ``"explicit"``; other algorithms are exposed
    through their specialized APIs until they share this scientific contract.
    """
    if not isinstance(problem, TransportProblem):
        raise TypeError("problem must be a TransportProblem")
    if end_time is not None and t is not None:
        raise TypeError("Pass either end_time or t, not both")
    if end_time is None and t is not None:
        warn_deprecated(
            "solve(t=...)",
            "solve(end_time=...)",
            reason="end_time is the one spelling for the requested end time",
        )
        end_time = t
    if end_time is None:
        raise TypeError("end_time is required")
    final_time = _finite_real(end_time, "end_time")
    if final_time < 0.0:
        raise ValueError("end_time must be non-negative")

    if time_step is not None and dt is not None:
        raise TypeError("Pass either time_step or dt, not both")
    if time_step is None and dt is not None:
        warn_deprecated(
            "solve(dt=...)",
            "solve(time_step=...)",
            reason="time_step is the one spelling for the maximum explicit step",
        )
        time_step = dt
    requested_step = 0.0
    if time_step is not None:
        requested_step = _finite_real(time_step, "time_step")
        if requested_step <= 0.0:
            raise ValueError("time_step must be positive when provided")
    save_time_values = _validated_save_times(save_times, final_time)

    if not isinstance(method, str):
        raise TypeError("method must be a string")
    normalized_method = method.lower().replace("-", "_")
    if normalized_method not in {"conservative", "explicit", "explicit_euler"}:
        raise ValueError(
            "The intuitive solve() API currently supports the verified conservative "
            "explicit solver only. Use a specialized solver explicitly for other methods."
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

    options = SolveOptions()
    options.final_time = final_time
    options.time_step = requested_step
    options.safety_factor = safety
    options.reaction_step_fraction = reaction_fraction
    options.max_steps = step_limit
    options.check_finite = check_finite
    options.save_times = save_time_values
    return _wrap_transport_result(solve_transport(problem, options))


def run(problem: TransportProblem, t_end: float, **kwargs) -> Result:
    """Deprecated alias for :func:`solve`; computation remains in C++."""
    return solve(problem, end_time=t_end, **kwargs)


@deprecated_callable(
    "bt.solve(problem, end_time=..., save_times=[...]).snapshots",
    reason=(
        "save_times keeps every configured term (reaction, advection, variable "
        "diffusivity), passes the absolute time to the reaction, and lands on each "
        "snapshot inside the C++ solver; run_checkpoints only supported uniform "
        "diffusion"
    ),
    name="biotransport.run_checkpoints",
)
def run_checkpoints(
    mesh,
    checkpoints: Sequence[float],
    diffusivity: float,
    initial_condition=None,
    boundaries: Mapping[Boundary, BoundaryCondition] | None = None,
    **solve_kwargs,
) -> CheckpointResult:
    """Solve pure diffusion in C++ and return fields at requested times.

    This helper is deliberately scoped to uniform diffusion. For reactions or
    advection, construct a :class:`Problem` and call :func:`solve` so configured
    terms cannot be lost while rebuilding checkpoint segments. Checkpoints may
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
