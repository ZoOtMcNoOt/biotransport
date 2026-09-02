"""One lifecycle for every native stepping solver.

The compiled specialized solvers (Crank--Nicolson, ADI, implicit, the explicit
reaction-diffusion family, 3D diffusion, nonuniform 1D, multi-species,
Nernst--Planck) each expose their own low-level ``solve(dt, num_steps)`` or
``solve_until`` entry point and their own boundary-setter spellings.  This
module installs, on every registered class, the vocabulary the canonical
:func:`biotransport.solve` already uses:

* ``solver.solve_until(end_time, time_step=None, *, save_times=None)`` returns
  a :class:`~biotransport.Result`;
* ``solver.dirichlet(side, value)``, ``solver.neumann(side, normal_derivative)``,
  ``solver.robin(side, a, b, rhs)``, ``solver.boundary(side, condition)`` and,
  for the Nernst--Planck solvers only, ``solver.outward_flux(side, molar_flux)``
  return the solver so calls chain.

The module performs no numerics.  It partitions time into segments at the
requested save times, asks the native solver to advance each segment with
equal substeps no larger than ``time_step``, verifies that the native clock
landed on the requested time, and packages the native fields.  An automatic
``time_step`` is offered only when the native class certifies its own explicit
stability limit; every other class requires an explicit ``time_step`` so no
uncertified claim is made on its behalf.  Unsupported boundary kinds raise
instead of being reinterpreted.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from . import _core
from ._deprecation import warn_deprecated
from .results import Result, Snapshots

__all__ = ["StepDiagnostics", "registered_stepping_classes", "solve_until"]

_EPS = float(np.finfo(np.float64).eps)


def _clock_tolerance(steps: int, target: float) -> float:
    """Largest acceptable gap between the native clock and its target.

    Native solvers accumulate ``time += dt`` once per step, so the clock can
    drift by up to one rounding error per step relative to the target. The
    bound is 64 ulp plus four ulp per step taken during this call, which still
    rejects any solver that takes the wrong number of steps or the wrong step.
    """
    return (64.0 + 4.0 * steps) * _EPS * max(1.0, abs(target))


@dataclass(frozen=True)
class StepDiagnostics:
    """Bookkeeping for one :func:`solve_until` call on a native stepping solver.

    ``stability_limit`` is the native solver's own explicit certificate when it
    exposes one and ``None`` otherwise; ``automatic_time_step`` records whether
    the step was derived from that certificate.  Time-step fields are the
    substeps actually used across all segments.
    """

    solver: str
    contract: str
    steps: int
    segments: int
    start_time: float
    requested_final_time: float
    final_time: float
    requested_time_step: float | None
    minimum_time_step: float
    maximum_time_step: float
    stability_limit: float | None
    automatic_time_step: bool


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _single_field(solver: Any) -> dict[str, np.ndarray]:
    return {"concentration": np.asarray(solver.solution(), dtype=np.float64)}


def _species_fields(solver: Any) -> dict[str, np.ndarray]:
    return {
        f"species_{index}": np.asarray(solver.solution(index), dtype=np.float64)
        for index in range(int(solver.num_species()))
    }


def _nernst_planck_fields(solver: Any) -> dict[str, np.ndarray]:
    return {
        "concentration": np.asarray(solver.solution(), dtype=np.float64),
        "potential": np.asarray(solver.potential(), dtype=np.float64),
    }


def _multi_ion_fields(solver: Any) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    for index in range(int(solver.num_species())):
        name = str(solver.ion(index).name) or f"ion_{index}"
        if name in fields:
            name = f"{name}_{index}"
        fields[name] = np.asarray(solver.concentration(index), dtype=np.float64)
    fields["potential"] = np.asarray(solver.potential(), dtype=np.float64)
    return fields


@dataclass(frozen=True)
class _Protocol:
    contract: str
    fields: Callable[[Any], dict[str, np.ndarray]]
    primary: str
    stability: str | None  # name of the native certificate accessor, if any
    boundary_style: str  # condition | condition_legacy | dirichlet_neumann | species | nernst_planck | multi_ion
    native_solve_until: bool = False  # class already lands exactly on absolute times
    multi_field: bool = False


_PROTOCOLS: dict[type, _Protocol] = {
    _core.DiffusionSolver: _Protocol(
        "diffusion.forward_euler_1d_2d",
        _single_field,
        "concentration",
        "max_stable_time_step",
        "condition",
    ),
    _core.ReactionDiffusionSolver: _Protocol(
        "reaction.generic_explicit",
        _single_field,
        "concentration",
        None,
        "condition_legacy",
    ),
    _core.LinearReactionDiffusionSolver: _Protocol(
        "reaction.linear_imex_1d_2d",
        _single_field,
        "concentration",
        None,
        "condition_legacy",
    ),
    _core.LogisticReactionDiffusionSolver: _Protocol(
        "reaction.logistic_specialized",
        _single_field,
        "concentration",
        None,
        "condition_legacy",
    ),
    _core.MichaelisMentenReactionDiffusionSolver: _Protocol(
        "reaction.michaelis_menten_specialized",
        _single_field,
        "concentration",
        None,
        "condition_legacy",
    ),
    _core.MaskedMichaelisMentenReactionDiffusionSolver: _Protocol(
        "reaction.masked_michaelis_menten",
        _single_field,
        "concentration",
        None,
        "condition_legacy",
    ),
    _core.ConstantSourceReactionDiffusionSolver: _Protocol(
        "reaction.constant_source_specialized",
        _single_field,
        "concentration",
        None,
        "condition_legacy",
    ),
    _core.AdvectionDiffusionSolver: _Protocol(
        "transport.legacy_advection_diffusion",
        _single_field,
        "concentration",
        None,
        "condition_legacy",
    ),
    _core.DiffusionSolver3D: _Protocol(
        "diffusion.forward_euler_3d",
        _single_field,
        "concentration",
        "max_stable_time_step",
        "dirichlet_neumann",
    ),
    _core.LinearReactionDiffusionSolver3D: _Protocol(
        "reaction.linear_imex_3d",
        _single_field,
        "concentration",
        "max_stable_time_step",
        "dirichlet_neumann",
    ),
    _core.CrankNicolsonDiffusion: _Protocol(
        "diffusion.crank_nicolson",
        _single_field,
        "concentration",
        None,
        "dirichlet_neumann",
    ),
    _core.ADIDiffusion2D: _Protocol(
        "diffusion.adi_2d",
        _single_field,
        "concentration",
        None,
        "dirichlet_neumann",
    ),
    _core.ADIDiffusion3D: _Protocol(
        "diffusion.adi_3d",
        _single_field,
        "concentration",
        None,
        "dirichlet_neumann",
    ),
    _core.ImplicitDiffusion2D: _Protocol(
        "diffusion.backward_euler_2d",
        _single_field,
        "concentration",
        None,
        "dirichlet_neumann",
    ),
    _core.ImplicitDiffusion3D: _Protocol(
        "diffusion.backward_euler_3d",
        _single_field,
        "concentration",
        None,
        "dirichlet_neumann",
    ),
    _core.NonuniformDiffusion1D: _Protocol(
        "diffusion.nonuniform_forward_euler_1d",
        _single_field,
        "concentration",
        "max_stable_time_step",
        "condition",
        native_solve_until=True,
    ),
    _core.MultiSpeciesSolver: _Protocol(
        "reaction.multispecies",
        _species_fields,
        "species_0",
        "max_stable_time_step",
        "species",
        native_solve_until=True,
        multi_field=True,
    ),
    _core.NernstPlanckSolver: _Protocol(
        "electrochem.nernst_planck",
        _nernst_planck_fields,
        "concentration",
        "maximum_stable_time_step",
        "nernst_planck",
        multi_field=True,
    ),
    _core.MultiIonSolver: _Protocol(
        "electrochem.multi_ion",
        _multi_ion_fields,
        "potential",
        "maximum_stable_time_step",
        "multi_ion",
        multi_field=True,
    ),
}


def registered_stepping_classes() -> tuple[type, ...]:
    """Return the native classes that carry the shared stepping vocabulary."""
    return tuple(_PROTOCOLS)


def _protocol_for(solver: Any) -> _Protocol:
    protocol = next(
        (_PROTOCOLS[base] for base in type(solver).__mro__ if base in _PROTOCOLS),
        None,
    )
    if protocol is None:
        names = ", ".join(sorted(cls.__name__ for cls in _PROTOCOLS))
        raise TypeError(
            f"{type(solver).__name__} is not a registered stepping solver; "
            f"solve_until supports: {names}"
        )
    return protocol


def _validated_save_times(save_times: object, start: float, end: float) -> list[float]:
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
    if any(value < start or value > end for value in values):
        raise ValueError("save_times must lie within [solver.time(), end_time]")
    if any(later <= earlier for earlier, later in zip(values, values[1:])):
        raise ValueError("save_times must be strictly increasing")
    return values


def _substeps(duration: float, time_step: float) -> tuple[int, float]:
    """Equal substeps covering ``duration`` with none larger than ``time_step``."""
    count = max(1, int(math.ceil(duration / time_step - 1e-12)))
    while count * time_step < duration * (1.0 - 1e-15):
        count += 1
    return count, duration / count


def solve_until(
    solver: Any,
    end_time: float,
    time_step: float | None = None,
    *,
    save_times: Sequence[float] | None = None,
    safety_factor: float = 0.8,
    **deprecated: Any,
) -> Result:
    """Advance a native stepping solver to ``end_time`` and return a :class:`Result`.

    Parameters
    ----------
    solver:
        Any registered native stepping solver (see :func:`registered_stepping_classes`).
    end_time:
        Absolute time to reach; must not precede ``solver.time()``.
    time_step:
        Upper bound on each substep. When omitted, it is
        ``safety_factor`` times the solver's own explicit stability certificate;
        classes without a certificate (Crank--Nicolson, ADI, implicit, and the
        legacy reaction/advection family) require an explicit value.
    save_times:
        Absolute times in ``[solver.time(), end_time]`` at which the primary field
        is recorded into ``result.snapshots``.
    safety_factor:
        Fraction of the certificate used for an automatic step.

    Notes
    -----
    Each segment between consecutive save times (and ``end_time``) is advanced
    with equal substeps of at most ``time_step``. After every segment the native
    clock must agree with the target to within the rounding error a native
    ``time += dt`` accumulation can incur (64 ulp plus 4 ulp per step); otherwise a
    ``RuntimeError`` is raised rather than silently accepting a drifted clock.
    """
    if "maximum_dt" in deprecated:
        if time_step is not None:
            raise TypeError("Pass either time_step or maximum_dt, not both")
        warn_deprecated(
            "solve_until(maximum_dt=...)",
            "solve_until(time_step=...)",
            reason="time_step is the one spelling for the maximum substep",
        )
        time_step = deprecated.pop("maximum_dt")
    if deprecated:
        raise TypeError(f"unexpected keyword arguments: {sorted(deprecated)}")

    protocol = _protocol_for(solver)
    start = float(solver.time())
    target = _finite_real(end_time, "end_time")
    if target < start:
        raise ValueError(
            f"end_time={target} must not precede the solver's current time {start}"
        )
    safety = _finite_real(safety_factor, "safety_factor")
    if not 0.0 < safety <= 1.0:
        raise ValueError("safety_factor must be in (0, 1]")

    limit: float | None = None
    if protocol.stability is not None:
        limit = float(getattr(solver, protocol.stability)())
    requested_step: float | None = None
    if time_step is None:
        if limit is None or not math.isfinite(limit) or limit <= 0.0:
            raise TypeError(
                f"{type(solver).__name__} does not certify an explicit stability limit; "
                "choose time_step explicitly (an implicit scheme's step is an accuracy "
                "decision, not a stability one)"
            )
        step = safety * limit
        automatic = True
    else:
        requested_step = _finite_real(time_step, "time_step")
        if requested_step <= 0.0:
            raise ValueError("time_step must be positive")
        step = requested_step
        automatic = False

    saves = _validated_save_times(save_times, start, target)
    segment_ends = saves + [target]

    total_steps = 0
    segments = 0
    minimum_step = math.inf
    maximum_step = 0.0
    snapshot_times: list[float] = []
    snapshot_fields: list[np.ndarray] = []
    previous = start
    for index, segment_end in enumerate(segment_ends):
        duration = segment_end - previous
        if duration > 0.0:
            if protocol.native_solve_until:
                solver._native_solve_until(segment_end, step)
                count = int(math.ceil(duration / step - 1e-12)) or 1
                used = duration / count
            else:
                count, used = _substeps(duration, step)
                solver.solve(used, count)
            total_steps += count
            segments += 1
            minimum_step = min(minimum_step, used)
            maximum_step = max(maximum_step, used)
            clock = float(solver.time())
            if abs(clock - segment_end) > _clock_tolerance(total_steps, segment_end):
                raise RuntimeError(
                    f"{type(solver).__name__} clock {clock!r} did not land on the "
                    f"requested time {segment_end!r}"
                )
        if index < len(saves):
            snapshot_times.append(segment_end)
            snapshot_fields.append(protocol.fields(solver)[protocol.primary])
        previous = segment_end

    if total_steps == 0:
        minimum_step = 0.0

    fields = protocol.fields(solver)
    diagnostics = StepDiagnostics(
        solver=type(solver).__name__,
        contract=protocol.contract,
        steps=total_steps,
        segments=segments,
        start_time=start,
        requested_final_time=target,
        final_time=float(solver.time()),
        requested_time_step=requested_step,
        minimum_time_step=minimum_step,
        maximum_time_step=maximum_step,
        stability_limit=limit,
        automatic_time_step=automatic,
    )
    return Result(
        fields=fields,
        time=target,
        steps=total_steps,
        diagnostics=diagnostics,
        mesh=solver.mesh(),
        contract=protocol.contract,
        snapshots=Snapshots(snapshot_times, snapshot_fields),
        native=solver,
        primary=protocol.primary,
    )


# ---------------------------------------------------------------------------
# Fluent boundary verbs
# ---------------------------------------------------------------------------


def _reject(solver: Any, kind: str, hint: str) -> None:
    raise TypeError(
        f"{type(solver).__name__} does not support {kind} conditions; {hint}"
    )


def _species_indices(solver: Any, species: int | None) -> range | tuple[int]:
    if species is None:
        return range(int(solver.num_species()))
    if isinstance(species, bool) or int(species) != species:
        raise TypeError("species must be an integer index")
    return (int(species),)


def _apply_condition(
    solver: Any, side: Any, condition: Any, species: int | None
) -> None:
    protocol = _protocol_for(solver)
    style = protocol.boundary_style
    kind = condition.type
    if species is not None and style not in {"species", "multi_ion"}:
        raise TypeError(
            f"{type(solver).__name__} solves one field; species= is not accepted"
        )

    if style == "condition":
        if kind == _core.BoundaryType.OUTWARD_FLUX:
            _reject(solver, "physical-flux", "it prescribes derivatives; use neumann()")
        solver.set_boundary_condition(side, condition)
    elif style == "condition_legacy":
        if kind == _core.BoundaryType.OUTWARD_FLUX:
            _reject(solver, "physical-flux", "it prescribes derivatives; use neumann()")
        solver.set_boundary(side, condition)
    elif style == "dirichlet_neumann":
        if kind == _core.BoundaryType.DIRICHLET:
            solver.set_dirichlet_boundary(side, condition.value)
        elif kind == _core.BoundaryType.NEUMANN:
            solver.set_neumann_boundary(side, condition.value)
        elif kind == _core.BoundaryType.ROBIN:
            _reject(solver, "Robin", "it accepts dirichlet() and neumann() only")
        else:
            _reject(solver, "physical-flux", "it prescribes derivatives; use neumann()")
    elif style == "species":
        for index in _species_indices(solver, species):
            if kind == _core.BoundaryType.DIRICHLET:
                solver.set_dirichlet_boundary(index, side, condition.value)
            elif kind == _core.BoundaryType.NEUMANN:
                solver.set_neumann_boundary(index, side, condition.value)
            elif kind == _core.BoundaryType.ROBIN:
                _reject(solver, "Robin", "it accepts dirichlet() and neumann() only")
            else:
                _reject(
                    solver, "physical-flux", "it prescribes derivatives; use neumann()"
                )
    elif style == "nernst_planck":
        if kind == _core.BoundaryType.DIRICHLET:
            solver.set_dirichlet_boundary(side, condition.value)
        elif kind == _core.BoundaryType.OUTWARD_FLUX:
            solver.set_outward_flux_boundary(side, condition.value)
        elif kind == _core.BoundaryType.NEUMANN:
            _reject(
                solver,
                "derivative (Neumann)",
                "it prescribes an outward molar flux; use outward_flux(side, molar_flux)",
            )
        else:
            _reject(solver, "Robin", "it accepts dirichlet() and outward_flux() only")
    elif style == "multi_ion":
        for index in _species_indices(solver, species):
            if kind == _core.BoundaryType.DIRICHLET:
                solver.set_dirichlet_boundary(index, side, condition.value)
            elif kind == _core.BoundaryType.OUTWARD_FLUX:
                solver.set_outward_flux_boundary(index, side, condition.value)
            elif kind == _core.BoundaryType.NEUMANN:
                _reject(
                    solver,
                    "derivative (Neumann)",
                    "it prescribes an outward molar flux; use outward_flux(side, molar_flux)",
                )
            else:
                _reject(
                    solver, "Robin", "it accepts dirichlet() and outward_flux() only"
                )
    else:  # pragma: no cover - registry invariant
        raise AssertionError(f"unknown boundary style {style!r}")


def _dirichlet(
    self: Any, side: Any, value: float, *, species: int | None = None
) -> Any:
    """Fix the field value on ``side`` and return the solver."""
    _apply_condition(
        self,
        side,
        _core.BoundaryCondition.dirichlet(_finite_real(value, "value")),
        species,
    )
    return self


def _neumann(
    self: Any, side: Any, normal_derivative: float, *, species: int | None = None
) -> Any:
    """Prescribe the outward-normal derivative on ``side`` and return the solver."""
    _apply_condition(
        self,
        side,
        _core.BoundaryCondition.neumann(
            _finite_real(normal_derivative, "normal_derivative")
        ),
        species,
    )
    return self


def _robin(
    self: Any, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
) -> Any:
    """Prescribe ``a*u + b*du/dn = rhs`` on ``side`` and return the solver."""
    _apply_condition(
        self,
        side,
        _core.BoundaryCondition.robin(
            _finite_real(a, "a"), _finite_real(b, "b"), _finite_real(rhs, "rhs")
        ),
        species,
    )
    return self


def _boundary(
    self: Any, side: Any, condition: Any, *, species: int | None = None
) -> Any:
    """Install a :class:`~biotransport.BoundaryCondition` and return the solver."""
    if not isinstance(condition, _core.BoundaryCondition):
        raise TypeError("condition must be a BoundaryCondition")
    _apply_condition(self, side, condition, species)
    return self


def _outward_flux(
    self: Any, side: Any, molar_flux: float, *, species: int | None = None
) -> Any:
    """Prescribe a physical outward flux (Nernst--Planck only) and return the solver."""
    _apply_condition(
        self,
        side,
        _core.BoundaryCondition.outward_flux(_finite_real(molar_flux, "molar_flux")),
        species,
    )
    return self


def _method_solve_until(
    self: Any, end_time: float, time_step: float | None = None, **kw: Any
):
    return solve_until(self, end_time, time_step, **kw)


_method_solve_until.__doc__ = solve_until.__doc__
_method_solve_until.__name__ = "solve_until"
_method_solve_until.__qualname__ = "solve_until"


def _install() -> None:
    """Attach the shared vocabulary to every registered native class (idempotent)."""
    for cls, protocol in _PROTOCOLS.items():
        if protocol.native_solve_until and not hasattr(cls, "_native_solve_until"):
            cls._native_solve_until = cls.solve_until
        if getattr(cls, "solve_until", None) is not _method_solve_until:
            cls.solve_until = _method_solve_until
        for name, function in (
            ("dirichlet", _dirichlet),
            ("neumann", _neumann),
            ("robin", _robin),
            ("boundary", _boundary),
            ("outward_flux", _outward_flux),
        ):
            setattr(cls, name, function)


_install()
