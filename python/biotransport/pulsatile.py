"""Validated time-dependent scalar boundary-value generators.

The classes in this module generate scalar values as functions of time.  A
name such as :class:`ArterialPressureBC` describes the units and shape of the
generated signal; it does *not* turn a scalar diffusion problem into a fluid,
pressure-propagation, or compliant-vessel model.

``solve_pulsatile`` is retained as a compatibility reference implementation
for exactly one equation on a uniform one-dimensional mesh::

    du/dt = D d2u/dx2

It supports uniform ``D``, time-varying Dirichlet data on the left and/or
right boundary, and static Dirichlet or outward-derivative Neumann data on
the remaining boundary.  It rejects advection, reactions, variable
diffusivity, Robin data, and multidimensional meshes instead of silently
dropping those terms.  The loop is Python/NumPy, not the native C++ transport
solver, and emits a runtime warning when used.

Times are seconds for the waveform classes whose rates are expressed in Hz,
beats/minute, or breaths/minute.  For the reference diffusion solve, spatial
units and field units may be chosen freely but must be mutually consistent:
``D`` must then have units ``length**2 / second``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
import math
from numbers import Integral
from typing import Any, Optional, cast
import warnings

import numpy as np

from ._core import (
    Boundary,
    BoundaryType,
    TransportProblem,
)


_REFERENCE_SOLVER_WARNING = (
    "solve_pulsatile is a legacy Python/NumPy reference solver for uniform 1D "
    "scalar diffusion only. Time-varying values are strong Dirichlet scalar "
    "data; waveform names do not add pressure, flow, or vascular physics."
)


def _finite(value: float, name: str) -> float:
    """Return ``value`` as a finite float or raise a focused error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _fraction(value: float, name: str, *, closed: bool = False) -> float:
    result = _finite(value, name)
    valid = 0.0 <= result <= 1.0 if closed else 0.0 < result < 1.0
    if not valid:
        interval = "[0, 1]" if closed else "(0, 1)"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _time(value: float) -> float:
    return _finite(value, "time")


def _waveform_value(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must return a real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} returned a non-finite value")
    return result


def _smoothstep(value: float) -> float:
    """Cubic interpolation from zero to one with zero endpoint slopes."""
    clipped = min(1.0, max(0.0, value))
    return clipped * clipped * (3.0 - 2.0 * clipped)


# =============================================================================
# Abstract Base Class
# =============================================================================


class PulsatileBC(ABC):
    """Abstract base class for time-varying boundary conditions.

    Subclasses must implement __call__(t) to return the BC value at time t.
    """

    @abstractmethod
    def __call__(self, t: float) -> float:
        """Evaluate the boundary condition value at time t.

        Args:
            t: Current simulation time (seconds)

        Returns:
            Boundary condition value at time t
        """
        pass

    @abstractmethod
    def period(self) -> float:
        """Return the period of the pulsatile waveform.

        Returns:
            Period in seconds (0 for non-periodic BCs)
        """
        pass


# =============================================================================
# Basic Waveform Types
# =============================================================================


@dataclass
class ConstantBC(PulsatileBC):
    """Constant (time-invariant) boundary condition.

    Useful as a baseline or for combining with other waveforms.

    Attributes:
        value: The constant boundary value
    """

    value: float = 0.0

    def __post_init__(self) -> None:
        self.value = _finite(self.value, "value")

    def __call__(self, t: float) -> float:
        _time(t)
        self.value = _finite(self.value, "value")
        return self.value

    def period(self) -> float:
        self.value = _finite(self.value, "value")
        return 0.0  # Non-periodic


@dataclass
class SinusoidalBC(PulsatileBC):
    """Sinusoidal time-varying boundary condition.

    value(t) = mean + amplitude * sin(2 * pi * frequency * t + phase)

    Attributes:
        mean: Mean (DC) value
        amplitude: Oscillation amplitude (peak deviation from mean)
        frequency: Oscillation frequency in Hz
        phase: Phase offset in radians (default 0)
    """

    mean: float = 0.0
    amplitude: float = 1.0
    frequency: float = 1.0  # Hz
    phase: float = 0.0  # radians

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.mean = _finite(self.mean, "mean")
        self.amplitude = _finite(self.amplitude, "amplitude")
        self.frequency = _positive(self.frequency, "frequency")
        self.phase = _finite(self.phase, "phase")

    def __call__(self, t: float) -> float:
        self._validate()
        time = _time(t)
        return float(
            self.mean
            + self.amplitude * np.sin(2.0 * np.pi * self.frequency * time + self.phase)
        )

    def period(self) -> float:
        self._validate()
        return 1.0 / self.frequency


@dataclass
class RampBC(PulsatileBC):
    """Linear ramp boundary condition.

    value(t) = start_value + (end_value - start_value) * (t - t_start) / duration

    Clamps to start_value before t_start and end_value after t_start + duration.

    Attributes:
        start_value: Initial value
        end_value: Final value
        t_start: Ramp start time (default 0)
        duration: Ramp duration (default 1.0)
    """

    start_value: float = 0.0
    end_value: float = 1.0
    t_start: float = 0.0
    duration: float = 1.0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.start_value = _finite(self.start_value, "start_value")
        self.end_value = _finite(self.end_value, "end_value")
        self.t_start = _finite(self.t_start, "t_start")
        self.duration = _positive(self.duration, "duration")

    def __call__(self, t: float) -> float:
        self._validate()
        time = _time(t)
        if time <= self.t_start:
            return self.start_value
        if time >= self.t_start + self.duration:
            return self.end_value
        fraction = (time - self.t_start) / self.duration
        return self.start_value + fraction * (self.end_value - self.start_value)

    def period(self) -> float:
        self._validate()
        return 0.0  # Non-periodic


@dataclass
class StepBC(PulsatileBC):
    """Step function boundary condition.

    value(t) = value_before if t < t_step else value_after

    Attributes:
        value_before: Value before step time
        value_after: Value after step time
        t_step: Time of step change
    """

    value_before: float = 0.0
    value_after: float = 1.0
    t_step: float = 0.0

    def __post_init__(self) -> None:
        self.value_before = _finite(self.value_before, "value_before")
        self.value_after = _finite(self.value_after, "value_after")
        self.t_step = _finite(self.t_step, "t_step")

    def __call__(self, t: float) -> float:
        self.__post_init__()
        return self.value_before if _time(t) < self.t_step else self.value_after

    def period(self) -> float:
        self.__post_init__()
        return 0.0  # Non-periodic


@dataclass
class SquareWaveBC(PulsatileBC):
    """Square wave boundary condition.

    Alternates between high_value and low_value with given frequency.

    Attributes:
        high_value: Value during "on" phase
        low_value: Value during "off" phase
        frequency: Oscillation frequency in Hz
        duty_cycle: Fraction of period at high_value (0 to 1, default 0.5)
        phase: Phase offset as fraction of period (0 to 1, default 0)
    """

    high_value: float = 1.0
    low_value: float = 0.0
    frequency: float = 1.0
    duty_cycle: float = 0.5
    phase: float = 0.0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.high_value = _finite(self.high_value, "high_value")
        self.low_value = _finite(self.low_value, "low_value")
        self.frequency = _positive(self.frequency, "frequency")
        self.duty_cycle = _fraction(self.duty_cycle, "duty_cycle", closed=True)
        self.phase = _finite(self.phase, "phase")

    def __call__(self, t: float) -> float:
        self._validate()
        period = 1.0 / self.frequency
        shifted_time = _time(t) + self.phase * period
        time_in_cycle = shifted_time % period
        if time_in_cycle < self.duty_cycle * period:
            return self.high_value
        return self.low_value

    def period(self) -> float:
        self._validate()
        return 1.0 / self.frequency


@dataclass
class CustomBC(PulsatileBC):
    """Custom time-varying boundary condition from user-provided function.

    Attributes:
        func: Callable that takes time t and returns BC value
        T: Period of the waveform (0 for non-periodic)
    """

    func: Callable[[float], float] = field(default=lambda t: 0.0)
    T: float = 0.0  # Period (0 = non-periodic)

    def __post_init__(self) -> None:
        if not callable(self.func):
            raise TypeError("func must be callable")
        self.T = _nonnegative(self.T, "T")

    def __call__(self, t: float) -> float:
        self.__post_init__()
        return _waveform_value(self.func(_time(t)), "CustomBC.func")

    def period(self) -> float:
        self.__post_init__()
        return self.T


# =============================================================================
# Physiological Cardiac Waveforms
# =============================================================================


@dataclass
class ArterialPressureBC(PulsatileBC):
    """Synthetic periodic arterial-pressure protocol in mmHg.

    This is an uncalibrated, smooth template with one systolic peak, a
    late-systolic shoulder, and a diastolic return.  ``systolic`` and
    ``diastolic`` are the exact maximum and minimum of the template.  It is
    useful for deterministic examples, not as a patient-specific waveform or
    a validated hemodynamics model.

    Attributes:
        systolic: Systolic (peak) pressure in mmHg (default 120)
        diastolic: Diastolic (minimum) pressure in mmHg (default 80)
        heart_rate: Heart rate in beats per minute (default 72)
        systolic_fraction: Fraction of cycle in systole (default 0.35)
    """

    systolic: float = 120.0  # mmHg
    diastolic: float = 80.0  # mmHg
    heart_rate: float = 72.0  # bpm
    systolic_fraction: float = 0.35

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.systolic = _positive(self.systolic, "systolic")
        self.diastolic = _nonnegative(self.diastolic, "diastolic")
        if self.systolic <= self.diastolic:
            raise ValueError("systolic must be greater than diastolic")
        self.heart_rate = _positive(self.heart_rate, "heart_rate")
        self.systolic_fraction = _fraction(self.systolic_fraction, "systolic_fraction")

    def __call__(self, t: float) -> float:
        self._validate()
        period = 60.0 / self.heart_rate
        phase = (_time(t) % period) / period
        peak_phase = 0.35 * self.systolic_fraction
        shoulder = 0.12

        if phase <= peak_phase:
            normalized = _smoothstep(phase / peak_phase)
        elif phase <= self.systolic_fraction:
            descent = _smoothstep(
                (phase - peak_phase) / (self.systolic_fraction - peak_phase)
            )
            normalized = 1.0 - (1.0 - shoulder) * descent
        else:
            diastolic_phase = (phase - self.systolic_fraction) / (
                1.0 - self.systolic_fraction
            )
            normalized = shoulder * (1.0 - _smoothstep(diastolic_phase))

        return self.diastolic + (self.systolic - self.diastolic) * normalized

    def period(self) -> float:
        self._validate()
        return 60.0 / self.heart_rate


@dataclass
class VenousPressureBC(PulsatileBC):
    """Synthetic central-venous-pressure protocol in mmHg.

    Three Gaussian components label the conventional A, C, and V features.
    Their timing and widths are fixed illustrative values; this is not a
    calibrated venous-return or right-heart model.

    Attributes:
        mean_pressure: Mean venous pressure in mmHg (default 8)
        amplitude: Pressure variation amplitude in mmHg (default 4)
        heart_rate: Heart rate in beats per minute (default 72)
    """

    mean_pressure: float = 8.0  # mmHg (central venous pressure)
    amplitude: float = 4.0  # mmHg
    heart_rate: float = 72.0  # bpm

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.mean_pressure = _finite(self.mean_pressure, "mean_pressure")
        self.amplitude = _nonnegative(self.amplitude, "amplitude")
        self.heart_rate = _positive(self.heart_rate, "heart_rate")

    def __call__(self, t: float) -> float:
        self._validate()
        period = 60.0 / self.heart_rate
        phase = (_time(t) % period) / period

        # Venous waveform: A wave (atrial contraction), C wave (AV valve bulging),
        # V wave (atrial filling)
        p = self.mean_pressure

        # Simplified 3-wave pattern
        # A wave at phase ~0.1, C wave at ~0.15, V wave at ~0.5
        a_wave = 0.4 * self.amplitude * np.exp(-((phase - 0.1) ** 2) / 0.005)
        c_wave = 0.2 * self.amplitude * np.exp(-((phase - 0.15) ** 2) / 0.002)
        v_wave = 0.4 * self.amplitude * np.exp(-((phase - 0.5) ** 2) / 0.02)

        return float(p + a_wave + c_wave + v_wave)

    def period(self) -> float:
        self._validate()
        return 60.0 / self.heart_rate


@dataclass
class CardiacOutputBC(PulsatileBC):
    """Synthetic periodic volumetric-flow protocol in L/min.

    The ejection segment is a squared-sine lobe with the specified
    ``peak_flow``.  The non-ejection segment is a smaller squared-sine return
    lobe whose amplitude is selected so the cycle average is exactly
    ``mean_flow``.  The value and its first derivative are continuous between
    cycles and segments.
    Parameter combinations that cannot satisfy both that average and the peak
    bound are rejected.  The generated scalar is neither a velocity profile
    nor a Navier--Stokes inlet condition until a caller performs the required
    geometry- and model-specific conversion.

    Attributes:
        mean_flow: Cycle-mean volumetric flow rate in L/min
        peak_flow: Peak ejection flow rate in L/min
        heart_rate: Heart rate in beats per minute (default 72)
        ejection_fraction: Fraction of cycle during ejection (default 0.3)
    """

    mean_flow: float = 5.0  # L/min for cardiac output
    peak_flow: float = 25.0  # L/min peak
    heart_rate: float = 72.0  # bpm
    ejection_fraction: float = 0.3

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.mean_flow = _nonnegative(self.mean_flow, "mean_flow")
        self.peak_flow = _positive(self.peak_flow, "peak_flow")
        self.heart_rate = _positive(self.heart_rate, "heart_rate")
        self.ejection_fraction = _fraction(self.ejection_fraction, "ejection_fraction")
        tail_amplitude = self._tail_amplitude()
        tolerance = 64.0 * np.finfo(float).eps * self.peak_flow
        if tail_amplitude < -tolerance or tail_amplitude > self.peak_flow + tolerance:
            lower, upper = self._admissible_mean_range()
            raise ValueError(
                "mean_flow is incompatible with peak_flow and ejection_fraction "
                "for this non-negative peak-bounded template; expected "
                f"{lower:.12g} <= mean_flow <= {upper:.12g}"
            )

    def _tail_integral(self) -> float:
        return 0.5 * (1.0 - self.ejection_fraction)

    def _ejection_mean(self) -> float:
        return 0.5 * self.peak_flow * self.ejection_fraction

    def _tail_amplitude(self) -> float:
        return (self.mean_flow - self._ejection_mean()) / self._tail_integral()

    def _admissible_mean_range(self) -> tuple[float, float]:
        ejection_mean = self._ejection_mean()
        return ejection_mean, ejection_mean + self.peak_flow * self._tail_integral()

    def __call__(self, t: float) -> float:
        self._validate()
        period = 60.0 / self.heart_rate
        phase = (_time(t) % period) / period

        # Squared-sine lobes are C1 at the segment and cycle boundaries.
        if phase < self.ejection_fraction:
            ejection_phase = phase / self.ejection_fraction
            return float(self.peak_flow * np.sin(np.pi * ejection_phase) ** 2)

        diastole_phase = (phase - self.ejection_fraction) / (
            1.0 - self.ejection_fraction
        )
        return float(
            max(0.0, self._tail_amplitude()) * np.sin(np.pi * diastole_phase) ** 2
        )

    def period(self) -> float:
        self._validate()
        return 60.0 / self.heart_rate


@dataclass
class RespiratoryBC(PulsatileBC):
    """Synthetic respiratory-cycle scalar protocol.

    ``mean`` is retained for compatibility but is the baseline/minimum, not
    the mathematical cycle mean.  ``amplitude`` is the excursion above that
    baseline.  The scalar has no pressure, volume, or concentration meaning
    until the caller assigns units and couples it to an appropriate model.

    Attributes:
        mean: Baseline/minimum value (legacy parameter name)
        amplitude: Breath amplitude
        respiratory_rate: Breaths per minute (default 12)
        inspiration_fraction: Fraction of cycle during inspiration (default 0.4)
    """

    mean: float = 0.0
    amplitude: float = 1.0
    respiratory_rate: float = 12.0  # breaths per minute
    inspiration_fraction: float = 0.4

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.mean = _finite(self.mean, "mean")
        self.amplitude = _nonnegative(self.amplitude, "amplitude")
        self.respiratory_rate = _positive(self.respiratory_rate, "respiratory_rate")
        self.inspiration_fraction = _fraction(
            self.inspiration_fraction, "inspiration_fraction"
        )

    def __call__(self, t: float) -> float:
        self._validate()
        period = 60.0 / self.respiratory_rate
        phase = (_time(t) % period) / period

        if phase < self.inspiration_fraction:
            # Inspiration: rise
            insp_phase = phase / self.inspiration_fraction
            return float(
                self.mean + self.amplitude * (0.5 - 0.5 * np.cos(np.pi * insp_phase))
            )

        # Expiration: fall
        exp_phase = (phase - self.inspiration_fraction) / (
            1.0 - self.inspiration_fraction
        )
        return float(
            self.mean + self.amplitude * (0.5 + 0.5 * np.cos(np.pi * exp_phase))
        )

    def period(self) -> float:
        self._validate()
        return 60.0 / self.respiratory_rate


@dataclass
class DrugInfusionBC(PulsatileBC):
    """Prescribed concentration protocol with bolus and maintenance phases.

    This returns a boundary concentration, not a dose or infusion rate.  It
    does not include a compartmental pharmacokinetic model, clearance, or
    vascular/tissue exchange.

    Attributes:
        bolus_concentration: Concentration during bolus phase
        maintenance_concentration: Concentration during maintenance
        bolus_duration: Duration of bolus in seconds
        infusion_start: Time infusion begins (default 0)
    """

    bolus_concentration: float = 1.0
    maintenance_concentration: float = 0.1
    bolus_duration: float = 60.0  # seconds
    infusion_start: float = 0.0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.bolus_concentration = _nonnegative(
            self.bolus_concentration, "bolus_concentration"
        )
        self.maintenance_concentration = _nonnegative(
            self.maintenance_concentration, "maintenance_concentration"
        )
        self.bolus_duration = _positive(self.bolus_duration, "bolus_duration")
        self.infusion_start = _finite(self.infusion_start, "infusion_start")

    def __call__(self, t: float) -> float:
        self._validate()
        time = _time(t)
        if time < self.infusion_start:
            return 0.0
        if time < self.infusion_start + self.bolus_duration:
            # Bolus phase with slight exponential decay
            phase = (time - self.infusion_start) / self.bolus_duration
            return float(self.bolus_concentration * np.exp(-0.5 * phase))
        return self.maintenance_concentration

    def period(self) -> float:
        self._validate()
        return 0.0  # Non-periodic


# =============================================================================
# Composite Waveforms
# =============================================================================


@dataclass
class CompositeBC(PulsatileBC):
    """Composite boundary condition from multiple waveforms.

    Combines multiple :class:`PulsatileBC` objects using addition or
    multiplication.  ``period()`` returns a common declared period when one
    can be established from the components.  It returns zero for an empty or
    genuinely non-periodic composition.

    Attributes:
        components: List of PulsatileBC objects to combine
        operation: 'add' or 'multiply' (default 'add')
    """

    components: list[PulsatileBC] = field(default_factory=list)
    operation: str = "add"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.operation not in {"add", "multiply"}:
            raise ValueError("operation must be 'add' or 'multiply'")
        if not isinstance(self.components, list):
            self.components = list(self.components)
        if any(not isinstance(component, PulsatileBC) for component in self.components):
            raise TypeError("every component must be a PulsatileBC")

    def __call__(self, t: float) -> float:
        self._validate()
        time = _time(t)
        if not self.components:
            return 0.0

        if self.operation == "multiply":
            result = 1.0
            for bc in self.components:
                result *= bc(time)
        else:
            result = 0.0
            for bc in self.components:
                result += bc(time)

        return _waveform_value(result, "CompositeBC")

    def period(self) -> float:
        self._validate()
        if not self.components:
            return 0.0

        periods: list[float] = []
        for component in self.components:
            period = _nonnegative(component.period(), "component period")
            if period == 0.0:
                # A constant does not destroy another component's periodicity.
                if isinstance(component, ConstantBC):
                    continue
                return 0.0
            periods.append(period)

        if not periods:
            return 0.0

        rational_periods = [
            Fraction(period).limit_denominator(10_000) for period in periods
        ]
        numerator = 1
        denominator = 0
        for rational_period in rational_periods:
            numerator = math.lcm(numerator, rational_period.numerator)
            denominator = (
                rational_period.denominator
                if denominator == 0
                else math.gcd(denominator, rational_period.denominator)
            )
        common_period = float(Fraction(numerator, denominator))

        # Treat an enormous inferred repeat interval as no useful established period.
        if not math.isfinite(common_period) or common_period > 1_000_000.0 * max(
            periods
        ):
            return 0.0
        for period in periods:
            cycles = common_period / period
            if not math.isclose(cycles, round(cycles), rel_tol=1e-10, abs_tol=1e-10):
                return 0.0
        return common_period


# =============================================================================
# Solver with Pulsatile BCs
# =============================================================================


@dataclass
class PulsatileResult:
    """Result from the legacy one-dimensional diffusion reference solve.

    Attributes:
        solution: Final solution field
        time: Final simulation time
        time_history: Initial, requested, and final snapshot times
        solution_history: Solution copies aligned with ``time_history``
        bc_history: Dynamic Dirichlet values aligned with ``time_history``
        stats: Numerical metadata, including the precise equation and CFL limit
    """

    solution: np.ndarray
    time: float
    time_history: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    solution_history: list[np.ndarray] = field(default_factory=list)
    bc_history: dict[Boundary, list[float]] = field(default_factory=dict)
    stats: dict[str, object] = field(default_factory=dict)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validated_dynamic_boundaries(
    pulsatile_bcs: Mapping[Boundary, PulsatileBC],
) -> dict[Boundary, PulsatileBC]:
    if not isinstance(pulsatile_bcs, Mapping):
        raise TypeError("pulsatile_bcs must be a mapping")

    supported_sides = {Boundary.Left, Boundary.Right}
    result: dict[Boundary, PulsatileBC] = {}
    for side, waveform in pulsatile_bcs.items():
        if side not in supported_sides:
            raise ValueError(
                "the 1D reference solver accepts only Boundary.Left and Boundary.Right"
            )
        if not isinstance(waveform, PulsatileBC):
            raise TypeError(
                "each time-varying boundary must be a PulsatileBC; wrap arbitrary "
                "callables with CustomBC"
            )
        result[side] = waveform
    return result


def _validate_reference_problem(
    problem: TransportProblem,
    dynamic_boundaries: Mapping[Boundary, PulsatileBC],
) -> tuple[Any, float, np.ndarray, tuple[Any, Any]]:
    if not isinstance(problem, TransportProblem):
        raise TypeError("problem must be a TransportProblem")

    # The runtime bindings expose these accessors; their shared stub is updated
    # by the integration owner, so keep this compatibility module independent
    # of stub rollout order.
    bound_problem = cast(Any, problem)
    mesh = bound_problem.mesh()
    if not mesh.is_1d():
        raise ValueError("solve_pulsatile supports only a one-dimensional mesh")
    if not bound_problem.has_uniform_diffusivity():
        raise NotImplementedError(
            "solve_pulsatile does not implement variable diffusivity; use the native "
            "transport solver for static boundaries"
        )
    if bound_problem.has_advection():
        raise NotImplementedError(
            "solve_pulsatile solves diffusion only and will not discard configured advection"
        )
    if bound_problem.has_reaction():
        raise NotImplementedError(
            "solve_pulsatile solves diffusion only and will not discard configured reactions"
        )

    diffusivity = _nonnegative(bound_problem.diffusivity(), "problem diffusivity")
    spacing = _positive(mesh.dx(), "mesh spacing")
    del spacing

    initial = np.asarray(bound_problem.initial(), dtype=np.float64)
    if initial.ndim != 1 or initial.size != mesh.num_nodes():
        raise ValueError(
            "problem initial condition must contain one value per mesh node"
        )
    if not np.all(np.isfinite(initial)):
        raise ValueError("problem initial condition must contain only finite values")
    initial = np.array(initial, dtype=np.float64, copy=True)

    boundaries = bound_problem.boundaries()
    static_boundaries = (boundaries[0], boundaries[1])
    for side, boundary in zip((Boundary.Left, Boundary.Right), static_boundaries):
        if side in dynamic_boundaries:
            continue
        if boundary.type == BoundaryType.ROBIN:
            raise NotImplementedError(
                "solve_pulsatile does not implement Robin boundaries because their "
                "explicit stability limit depends on the Robin coefficients"
            )
        if boundary.type not in {BoundaryType.DIRICHLET, BoundaryType.NEUMANN}:
            raise ValueError("unsupported static boundary type")
        _finite(boundary.value, f"static {side.name} boundary value")

    return mesh, diffusivity, initial, static_boundaries


def _evaluate_dynamic_boundaries(
    dynamic_boundaries: Mapping[Boundary, PulsatileBC], time: float
) -> dict[Boundary, float]:
    return {
        side: _waveform_value(waveform(time), f"{side.name} boundary waveform")
        for side, waveform in dynamic_boundaries.items()
    }


def _impose_dirichlet_boundaries(
    solution: np.ndarray,
    static_boundaries: tuple[Any, Any],
    dynamic_values: Mapping[Boundary, float],
) -> None:
    for index, side, boundary in (
        (0, Boundary.Left, static_boundaries[0]),
        (-1, Boundary.Right, static_boundaries[1]),
    ):
        if side in dynamic_values:
            solution[index] = dynamic_values[side]
        elif boundary.type == BoundaryType.DIRICHLET:
            solution[index] = boundary.value


def solve_pulsatile(
    problem: TransportProblem,
    t_end: float,
    pulsatile_bcs: Mapping[Boundary, PulsatileBC],
    dt: Optional[float] = None,
    save_every: Optional[int] = None,
    callback: Optional[Callable[[float, np.ndarray], None]] = None,
    *,
    max_steps: int = 1_000_000,
) -> PulsatileResult:
    """Run the legacy reference solver for uniform 1D scalar diffusion.

    The discretization is second-order centered in space and forward Euler in
    time.  Values in ``pulsatile_bcs`` are imposed as strong Dirichlet data at
    the new time level.  A static Neumann value is interpreted as the outward
    derivative ``du/dn`` and applied with a centered ghost point.  The
    explicit diffusion number must satisfy ``D*dt/dx**2 <= 1/2``.

    This function does not call the C++ transport solver.  It is retained for
    compatibility and transparent reference calculations while a native
    dynamic-boundary interface is absent.  Every call emits ``RuntimeWarning``.

    Args:
        problem: A 1D, uniform-diffusivity, diffusion-only problem.
        t_end: Non-negative final time in seconds.
        pulsatile_bcs: Left/right scalar waveforms, imposed as Dirichlet data.
        dt: Positive time step in seconds.  If omitted, 90% of the exact
            forward-Euler diffusion stability limit is used.  When ``D=0``
            there is no diffusion-derived time scale, so ``dt`` is required.
        save_every: Save every positive integer number of steps.  The initial
            and final states are always returned.
        callback: Called after each accepted step with ``(time, read-only copy)``.
        max_steps: Guard against accidentally running a long Python loop.

    Returns:
        Final field, aligned histories, and numerical metadata.

    Raises:
        NotImplementedError: If the problem contains physics this reference
            scheme does not implement.
        ValueError: If a value is outside its domain or the requested time
            step violates the explicit diffusion stability bound.
    """
    final_time = _nonnegative(t_end, "t_end")
    step_limit = _positive_integer(max_steps, "max_steps")
    if save_every is not None:
        save_every = _positive_integer(save_every, "save_every")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable or None")

    dynamic_boundaries = _validated_dynamic_boundaries(pulsatile_bcs)
    mesh, diffusivity, solution, static_boundaries = _validate_reference_problem(
        problem, dynamic_boundaries
    )
    spacing = _positive(mesh.dx(), "mesh spacing")
    spacing_squared = spacing * spacing
    if spacing_squared == 0.0:
        raise ValueError("mesh spacing is too small to square in float64 arithmetic")
    max_stable_dt = (
        spacing_squared / (2.0 * diffusivity) if diffusivity > 0.0 else math.inf
    )

    if dt is None:
        if final_time > 0.0 and diffusivity == 0.0:
            raise ValueError(
                "dt is required when diffusivity is zero because no diffusion "
                "stability scale exists to choose it"
            )
        nominal_dt = 0.9 * max_stable_dt if final_time > 0.0 else 0.0
        if not math.isfinite(nominal_dt):
            nominal_dt = final_time
    else:
        nominal_dt = _positive(dt, "dt")

    planned_steps = 0
    if final_time > 0.0:
        first_step = min(nominal_dt, final_time)
        diffusion_number = diffusivity * first_step / spacing_squared
        stability_tolerance = 64.0 * np.finfo(float).eps
        if diffusion_number > 0.5 + stability_tolerance:
            raise ValueError(
                "dt violates the forward-Euler 1D diffusion stability limit: "
                f"D*dt/dx^2={diffusion_number:.12g} > 0.5; use dt <= "
                f"{max_stable_dt:.12g}"
            )
        step_ratio = final_time / nominal_dt
        nearest_integer = round(step_ratio)
        if nearest_integer >= 1 and math.isclose(
            step_ratio, nearest_integer, rel_tol=1e-13, abs_tol=1e-13
        ):
            planned_steps = int(nearest_integer)
        else:
            planned_steps = math.ceil(step_ratio)
        if planned_steps > step_limit:
            raise ValueError(
                f"the requested solve needs more than max_steps={step_limit}; "
                "use the native C++ solver for long runs or raise max_steps explicitly"
            )

    warnings.warn(_REFERENCE_SOLVER_WARNING, RuntimeWarning, stacklevel=2)

    time = 0.0
    step = 0
    last_dt = 0.0
    max_diffusion_number = 0.0
    dynamic_values = _evaluate_dynamic_boundaries(dynamic_boundaries, time)
    _impose_dirichlet_boundaries(solution, static_boundaries, dynamic_values)

    time_history = [time]
    solution_history = [solution.copy()]
    bc_history = {side: [dynamic_values[side]] for side in dynamic_boundaries}

    for step_index in range(planned_steps):
        next_time = (
            final_time
            if step_index + 1 == planned_steps
            else min((step_index + 1) * nominal_dt, final_time)
        )
        step_dt = next_time - time
        if step_dt <= 0.0:
            raise RuntimeError("time integration stopped making forward progress")

        diffusion_number = diffusivity * step_dt / spacing_squared
        next_solution = solution.copy()
        if solution.size > 2:
            next_solution[1:-1] = solution[1:-1] + diffusion_number * (
                solution[:-2] - 2.0 * solution[1:-1] + solution[2:]
            )

        for index, neighbor, side, boundary in (
            (0, 1, Boundary.Left, static_boundaries[0]),
            (-1, -2, Boundary.Right, static_boundaries[1]),
        ):
            if side in dynamic_boundaries or boundary.type == BoundaryType.DIRICHLET:
                continue
            outward_derivative = _finite(
                boundary.value, f"static {side.name} outward derivative"
            )
            next_solution[index] = solution[index] + 2.0 * diffusion_number * (
                solution[neighbor] - solution[index] + outward_derivative * spacing
            )

        next_dynamic_values = _evaluate_dynamic_boundaries(
            dynamic_boundaries, next_time
        )
        _impose_dirichlet_boundaries(
            next_solution, static_boundaries, next_dynamic_values
        )
        if not np.all(np.isfinite(next_solution)):
            raise FloatingPointError("the reference solve produced a non-finite value")

        solution = next_solution
        dynamic_values = next_dynamic_values
        time = next_time
        step += 1
        last_dt = step_dt
        max_diffusion_number = max(max_diffusion_number, diffusion_number)

        if save_every is not None and step % save_every == 0:
            time_history.append(time)
            solution_history.append(solution.copy())
            for side in dynamic_boundaries:
                bc_history[side].append(dynamic_values[side])

        if callback is not None:
            callback_solution = solution.copy()
            callback_solution.setflags(write=False)
            callback(time, callback_solution)

    if time_history[-1] != time:
        time_history.append(time)
        solution_history.append(solution.copy())
        for side in dynamic_boundaries:
            bc_history[side].append(dynamic_values[side])

    positive_periods = []
    for waveform in dynamic_boundaries.values():
        waveform_period = waveform.period()
        if waveform_period > 0.0:
            positive_periods.append(waveform_period)
    minimum_period = min(positive_periods) if positive_periods else None
    steps_per_minimum_period = (
        minimum_period / nominal_dt
        if minimum_period is not None and nominal_dt > 0.0
        else None
    )

    return PulsatileResult(
        solution=solution,
        time=time,
        time_history=np.asarray(time_history, dtype=np.float64),
        solution_history=solution_history,
        bc_history=bc_history,
        stats={
            "implementation": "legacy-python-numpy-reference",
            "equation": "du/dt = D*d2u/dx2",
            "dynamic_boundary_semantics": "strong Dirichlet scalar value",
            "static_neumann_semantics": "outward-normal derivative du/dn",
            "steps": step,
            "dt": nominal_dt,
            "last_dt": last_dt,
            "t_end": time,
            "max_stable_dt": max_stable_dt,
            "max_diffusion_number": max_diffusion_number,
            "minimum_waveform_period": minimum_period,
            "steps_per_minimum_period": steps_per_minimum_period,
        },
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def heart_rate_to_period(bpm: float) -> float:
    """Convert heart rate in BPM to period in seconds.

    Args:
        bpm: Heart rate in beats per minute

    Returns:
        Period in seconds
    """
    return 60.0 / _positive(bpm, "bpm")


def period_to_heart_rate(T: float) -> float:
    """Convert period in seconds to heart rate in BPM.

    Args:
        T: Period in seconds

    Returns:
        Heart rate in beats per minute
    """
    return 60.0 / _positive(T, "T")


def sample_waveform(
    bc: PulsatileBC, t_start: float = 0.0, t_end: float = 1.0, num_points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a pulsatile BC waveform over a time range.

    Useful for visualization and verification.

    Args:
        bc: PulsatileBC object to sample
        t_start: Start time
        t_end: End time
        num_points: Number of sample points

    Returns:
        Tuple of (times, values) arrays
    """
    if not isinstance(bc, PulsatileBC):
        raise TypeError("bc must be a PulsatileBC")
    start = _finite(t_start, "t_start")
    end = _finite(t_end, "t_end")
    if end < start:
        raise ValueError("t_end must be greater than or equal to t_start")
    count = _positive_integer(num_points, "num_points")

    times = np.linspace(start, end, count, dtype=np.float64)
    values = np.asarray([bc(float(time)) for time in times], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("waveform sampling produced a non-finite value")
    return times, values
