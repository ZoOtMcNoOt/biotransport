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
from numbers import Integral, Real
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
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
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
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must return a real scalar")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} returned a non-finite value") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} returned a non-finite value")
    return result


def _smoothstep(value: float) -> float:
    """Cubic interpolation from zero to one with zero endpoint slopes."""
    clipped = min(1.0, max(0.0, value))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _period_from_rate(numerator: float, rate: float, name: str) -> float:
    """Return a representable positive period for a validated rate."""
    period = numerator / rate
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError(f"{name} produces a period outside the finite float64 range")
    return period


def _cycle_phase(time: float, period: float) -> float:
    """Reduce a finite time to a unit-cycle phase without forming rate*time."""
    phase = (_time(time) % period) / period
    if not math.isfinite(phase):
        raise ValueError("waveform phase could not be represented as a finite value")
    return phase


def _exact_step_count(final_time: float, nominal_dt: float) -> int:
    """Return ceil(final_time / nominal_dt) using the exact float values."""
    final_numerator, final_denominator = final_time.as_integer_ratio()
    step_numerator, step_denominator = nominal_dt.as_integer_ratio()
    numerator = final_numerator * step_denominator
    denominator = final_denominator * step_numerator
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(remainder != 0)


def _violates_diffusion_stability(
    diffusivity: float, step_dt: float, spacing: float
) -> bool:
    """Compare 2*D*dt <= dx**2 using the exact values of the input floats."""
    return _exact_diffusion_number(diffusivity, step_dt, spacing) > Fraction(1, 2)


def _exact_diffusion_number(
    diffusivity: float, step_dt: float, spacing: float
) -> Fraction:
    """Return the exact D*dt/dx**2 represented by three finite floats."""
    if diffusivity == 0.0:
        return Fraction(0)
    return (
        Fraction.from_float(diffusivity)
        * Fraction.from_float(step_dt)
        / Fraction.from_float(spacing) ** 2
    )


def _diffusion_number(diffusivity: float, step_dt: float, spacing: float) -> float:
    """Evaluate a validated diffusion number without overflow or underflow."""
    result = float(_exact_diffusion_number(diffusivity, step_dt, spacing))
    if not math.isfinite(result):
        raise ValueError("the diffusion number cannot be represented as a finite float")
    return result


def _maximum_stable_time_step(diffusivity: float, spacing: float) -> float:
    """Return a conservative float for dx**2/(2*D) without intermediates."""
    if diffusivity == 0.0:
        return math.inf
    exact_bound = Fraction.from_float(spacing) ** 2 / (
        2 * Fraction.from_float(diffusivity)
    )
    try:
        result = float(exact_bound)
    except OverflowError:
        # Every finite float time step is then below the mathematical bound.
        return math.inf
    if result > 0.0 and Fraction.from_float(result) > exact_bound:
        result = math.nextafter(result, 0.0)
    return result


def _finite_fraction_to_float(value: Fraction, name: str) -> float:
    try:
        result = float(value)
    except OverflowError as exc:
        raise FloatingPointError(f"{name} exceeds the finite float64 range") from exc
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} exceeds the finite float64 range")
    return result


def _advance_diffusion_interior(
    solution: np.ndarray,
    next_solution: np.ndarray,
    diffusion_number: float,
    exact_diffusion_number: Fraction,
) -> None:
    if solution.size <= 2 or exact_diffusion_number == 0:
        return

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        stencil = solution[:-2] - 2.0 * solution[1:-1] + solution[2:]
        updated = solution[1:-1] + diffusion_number * stencil

    needs_exact_path = (
        0.0 < diffusion_number < np.finfo(float).tiny
        or (diffusion_number == 0.0 and exact_diffusion_number > 0)
        or not np.all(np.isfinite(stencil))
        or not np.all(np.isfinite(updated))
    )
    if not needs_exact_path:
        next_solution[1:-1] = updated
        return

    for index in range(1, solution.size - 1):
        center = Fraction.from_float(float(solution[index]))
        exact_stencil = (
            Fraction.from_float(float(solution[index - 1]))
            - 2 * center
            + Fraction.from_float(float(solution[index + 1]))
        )
        exact_updated = center + exact_diffusion_number * exact_stencil
        next_solution[index] = _finite_fraction_to_float(
            exact_updated, "diffusion update"
        )


def _advance_neumann_boundary(
    *,
    center: float,
    neighbor: float,
    outward_derivative: float,
    spacing: float,
    diffusion_number: float,
    exact_diffusion_number: Fraction,
) -> float:
    if exact_diffusion_number == 0:
        return center

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        boundary_stencil = neighbor - center + outward_derivative * spacing
        updated = center + 2.0 * diffusion_number * boundary_stencil

    needs_exact_path = (
        0.0 < diffusion_number < np.finfo(float).tiny
        or (diffusion_number == 0.0 and exact_diffusion_number > 0)
        or not math.isfinite(boundary_stencil)
        or not math.isfinite(updated)
    )
    if not needs_exact_path:
        return updated

    exact_center = Fraction.from_float(center)
    exact_stencil = (
        Fraction.from_float(neighbor)
        - exact_center
        + Fraction.from_float(outward_derivative) * Fraction.from_float(spacing)
    )
    return _finite_fraction_to_float(
        exact_center + 2 * exact_diffusion_number * exact_stencil,
        "Neumann boundary update",
    )


def _diagnostic_diffusion_number(exact_value: Fraction) -> float:
    """Return a nonzero conservative float diagnostic for a positive exact CFL."""
    rounded = float(exact_value)
    if exact_value > 0 and rounded == 0.0:
        return math.nextafter(0.0, math.inf)
    return rounded


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
        period = _period_from_rate(1.0, self.frequency, "frequency")
        angle = math.tau * _cycle_phase(time, period) + math.remainder(
            self.phase, math.tau
        )
        return _waveform_value(
            self.mean + self.amplitude * math.sin(angle), "SinusoidalBC"
        )

    def period(self) -> float:
        self._validate()
        return _period_from_rate(1.0, self.frequency, "frequency")


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
        return _waveform_value(
            (1.0 - fraction) * self.start_value + fraction * self.end_value,
            "RampBC",
        )

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
        period = _period_from_rate(1.0, self.frequency, "frequency")
        phase = (_cycle_phase(t, period) + self.phase % 1.0) % 1.0
        if phase < self.duty_cycle:
            return self.high_value
        return self.low_value

    def period(self) -> float:
        self._validate()
        return _period_from_rate(1.0, self.frequency, "frequency")


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
        period = _period_from_rate(60.0, self.heart_rate, "heart_rate")
        phase = _cycle_phase(t, period)
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

        return _waveform_value(
            self.diastolic + (self.systolic - self.diastolic) * normalized,
            "ArterialPressureBC",
        )

    def period(self) -> float:
        self._validate()
        return _period_from_rate(60.0, self.heart_rate, "heart_rate")


@dataclass
class VenousPressureBC(PulsatileBC):
    """Synthetic central-venous-pressure protocol in mmHg.

    Three Gaussian components label the conventional A, C, and V features.
    Their timing and widths are fixed illustrative values; this is not a
    calibrated venous-return or right-heart model. ``mean_pressure`` is the
    exact cycle mean; the internal baseline is shifted to compensate for the
    positive mean of the Gaussian components.

    Attributes:
        mean_pressure: Cycle-mean venous pressure in mmHg (default 8)
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

    @staticmethod
    def _gaussian_cycle_mean(center: float, denominator: float) -> float:
        scale = math.sqrt(denominator)
        return (
            0.5
            * math.sqrt(math.pi * denominator)
            * (math.erf((1.0 - center) / scale) + math.erf(center / scale))
        )

    def _component_cycle_mean(self) -> float:
        return self.amplitude * (
            0.4 * self._gaussian_cycle_mean(0.1, 0.005)
            + 0.2 * self._gaussian_cycle_mean(0.15, 0.002)
            + 0.4 * self._gaussian_cycle_mean(0.5, 0.02)
        )

    def __call__(self, t: float) -> float:
        self._validate()
        period = _period_from_rate(60.0, self.heart_rate, "heart_rate")
        phase = _cycle_phase(t, period)

        # Venous waveform: A wave (atrial contraction), C wave (AV valve bulging),
        # V wave (atrial filling)
        baseline = self.mean_pressure - self._component_cycle_mean()

        # Simplified 3-wave pattern
        # A wave at phase ~0.1, C wave at ~0.15, V wave at ~0.5
        a_wave = 0.4 * self.amplitude * math.exp(-((phase - 0.1) ** 2) / 0.005)
        c_wave = 0.2 * self.amplitude * math.exp(-((phase - 0.15) ** 2) / 0.002)
        v_wave = 0.4 * self.amplitude * math.exp(-((phase - 0.5) ** 2) / 0.02)

        return _waveform_value(baseline + a_wave + c_wave + v_wave, "VenousPressureBC")

    def period(self) -> float:
        self._validate()
        return _period_from_rate(60.0, self.heart_rate, "heart_rate")


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
        self._bounded_tail_amplitude()

    def _bounded_tail_amplitude(self) -> float:
        tail_amplitude = self._tail_amplitude()
        tolerance = (64.0 * np.finfo(float).eps) * self.peak_flow
        invalid = (
            not math.isfinite(tail_amplitude)
            or (tail_amplitude < 0.0 and -tail_amplitude > tolerance)
            or (
                tail_amplitude > self.peak_flow
                and tail_amplitude - self.peak_flow > tolerance
            )
        )
        if invalid:
            lower, upper = self._admissible_mean_range()
            raise ValueError(
                "mean_flow is incompatible with peak_flow and ejection_fraction "
                "for this non-negative peak-bounded template; expected "
                f"{lower:.12g} <= mean_flow <= {upper:.12g}"
            )
        return min(self.peak_flow, max(0.0, tail_amplitude))

    def _tail_integral(self) -> float:
        return 0.5 * (1.0 - self.ejection_fraction)

    def _ejection_mean(self) -> float:
        return 0.5 * self.peak_flow * self.ejection_fraction

    def _tail_amplitude(self) -> float:
        return (self.mean_flow - self._ejection_mean()) / self._tail_integral()

    def _admissible_mean_range(self) -> tuple[float, float]:
        return self._ejection_mean(), 0.5 * self.peak_flow

    def __call__(self, t: float) -> float:
        self._validate()
        period = _period_from_rate(60.0, self.heart_rate, "heart_rate")
        phase = _cycle_phase(t, period)

        # Squared-sine lobes are C1 at the segment and cycle boundaries.
        if phase < self.ejection_fraction:
            ejection_phase = phase / self.ejection_fraction
            return _waveform_value(
                self.peak_flow * math.sin(math.pi * ejection_phase) ** 2,
                "CardiacOutputBC",
            )

        diastole_phase = (phase - self.ejection_fraction) / (
            1.0 - self.ejection_fraction
        )
        return _waveform_value(
            self._bounded_tail_amplitude() * math.sin(math.pi * diastole_phase) ** 2,
            "CardiacOutputBC",
        )

    def period(self) -> float:
        self._validate()
        return _period_from_rate(60.0, self.heart_rate, "heart_rate")


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
        period = _period_from_rate(60.0, self.respiratory_rate, "respiratory_rate")
        phase = _cycle_phase(t, period)

        if phase < self.inspiration_fraction:
            # Inspiration: rise
            insp_phase = phase / self.inspiration_fraction
            return _waveform_value(
                self.mean
                + self.amplitude * (0.5 - 0.5 * math.cos(math.pi * insp_phase)),
                "RespiratoryBC",
            )

        # Expiration: fall
        exp_phase = (phase - self.inspiration_fraction) / (
            1.0 - self.inspiration_fraction
        )
        return _waveform_value(
            self.mean + self.amplitude * (0.5 + 0.5 * math.cos(math.pi * exp_phase)),
            "RespiratoryBC",
        )

    def period(self) -> float:
        self._validate()
        return _period_from_rate(60.0, self.respiratory_rate, "respiratory_rate")


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


@dataclass(frozen=True)
class _BoundarySnapshot:
    type: BoundaryType
    value: float
    a: float
    b: float
    c: float


@dataclass(frozen=True)
class _ReferenceProblemSnapshot:
    mesh_signature: tuple[int, int, int, float, float, float]
    diffusivity: float
    initial: np.ndarray
    boundaries: tuple[_BoundarySnapshot, ...]


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


def _snapshot_reference_problem(
    problem: TransportProblem,
) -> _ReferenceProblemSnapshot:
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

    initial = np.asarray(bound_problem.initial(), dtype=np.float64)
    if initial.ndim != 1 or initial.size != mesh.num_nodes():
        raise ValueError(
            "problem initial condition must contain one value per mesh node"
        )
    if not np.all(np.isfinite(initial)):
        raise ValueError("problem initial condition must contain only finite values")
    initial = np.array(initial, dtype=np.float64, copy=True)
    initial.setflags(write=False)

    boundaries = bound_problem.boundaries()
    boundary_snapshots: list[_BoundarySnapshot] = []
    for side, boundary in zip(
        (Boundary.Left, Boundary.Right, Boundary.Bottom, Boundary.Top), boundaries
    ):
        if boundary.type == BoundaryType.ROBIN:
            raise NotImplementedError(
                "solve_pulsatile does not implement Robin boundaries because their "
                "explicit stability limit depends on the Robin coefficients; remove "
                "the Robin condition before applying a dynamic Dirichlet override"
            )
        if boundary.type not in {BoundaryType.DIRICHLET, BoundaryType.NEUMANN}:
            raise ValueError("unsupported static boundary type")
        value = _finite(boundary.value, f"{side.name} boundary value")
        boundary_snapshots.append(
            _BoundarySnapshot(
                type=boundary.type,
                value=value,
                a=_finite(boundary.a, f"{side.name} boundary a"),
                b=_finite(boundary.b, f"{side.name} boundary b"),
                c=_finite(boundary.c, f"{side.name} boundary c"),
            )
        )

    return _ReferenceProblemSnapshot(
        mesh_signature=(
            int(mesh.nx()),
            int(mesh.ny()),
            int(mesh.num_nodes()),
            spacing,
            _finite(mesh.x(0), "mesh left coordinate"),
            _finite(mesh.x(mesh.nx()), "mesh right coordinate"),
        ),
        diffusivity=diffusivity,
        initial=initial,
        boundaries=tuple(boundary_snapshots),
    )


def _assert_reference_problem_unchanged(
    problem: TransportProblem,
    expected: _ReferenceProblemSnapshot,
) -> None:
    try:
        current = _snapshot_reference_problem(problem)
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise RuntimeError(
            "the TransportProblem was mutated during solve_pulsatile; waveform "
            "and solve callbacks must not change the problem"
        ) from exc

    unchanged = (
        current.mesh_signature == expected.mesh_signature
        and current.diffusivity == expected.diffusivity
        and current.boundaries == expected.boundaries
        and np.array_equal(current.initial, expected.initial)
    )
    if not unchanged:
        raise RuntimeError(
            "the TransportProblem was mutated during solve_pulsatile; waveform "
            "and solve callbacks must not change the problem"
        )


def _waveform_may_call_user_code(waveform: PulsatileBC) -> bool:
    if type(waveform) is CompositeBC:
        return any(
            _waveform_may_call_user_code(component) for component in waveform.components
        )
    return type(waveform) not in {
        ConstantBC,
        SinusoidalBC,
        RampBC,
        StepBC,
        SquareWaveBC,
        ArterialPressureBC,
        VenousPressureBC,
        CardiacOutputBC,
        RespiratoryBC,
        DrugInfusionBC,
    }


def _evaluate_dynamic_boundaries(
    dynamic_boundaries: Mapping[Boundary, PulsatileBC],
    time: float,
    *,
    problem: TransportProblem,
    expected_problem: _ReferenceProblemSnapshot,
) -> dict[Boundary, float]:
    values: dict[Boundary, float] = {}
    for side, waveform in dynamic_boundaries.items():
        values[side] = _waveform_value(waveform(time), f"{side.name} boundary waveform")
        if _waveform_may_call_user_code(waveform):
            _assert_reference_problem_unchanged(problem, expected_problem)
    return values


def _impose_dirichlet_boundaries(
    solution: np.ndarray,
    static_boundaries: tuple[_BoundarySnapshot, _BoundarySnapshot],
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
    The validated problem is snapshotted before integration; waveform and solve
    callbacks must not mutate it.

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
        RuntimeError: If user callback code mutates ``problem`` during the solve.
    """
    final_time = _nonnegative(t_end, "t_end")
    step_limit = _positive_integer(max_steps, "max_steps")
    if save_every is not None:
        save_every = _positive_integer(save_every, "save_every")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable or None")

    dynamic_boundaries = _validated_dynamic_boundaries(pulsatile_bcs)
    problem_snapshot = _snapshot_reference_problem(problem)
    diffusivity = problem_snapshot.diffusivity
    solution = np.array(problem_snapshot.initial, dtype=np.float64, copy=True)
    static_boundaries = (
        problem_snapshot.boundaries[0],
        problem_snapshot.boundaries[1],
    )
    spacing = problem_snapshot.mesh_signature[3]
    max_stable_dt = _maximum_stable_time_step(diffusivity, spacing)
    if final_time > 0.0 and diffusivity > 0.0 and max_stable_dt == 0.0:
        raise ValueError(
            "no positive float64 time step can satisfy the diffusion stability "
            "limit for this diffusivity and mesh spacing"
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
    final_step_dt = 0.0
    if final_time > 0.0:
        planned_steps = _exact_step_count(final_time, nominal_dt)
        if planned_steps > step_limit:
            raise ValueError(
                f"the requested solve needs more than max_steps={step_limit}; "
                "use the native C++ solver for long runs or raise max_steps explicitly"
            )

        first_step = min(nominal_dt, final_time)
        if _violates_diffusion_stability(diffusivity, first_step, spacing):
            diffusion_number = _diffusion_number(diffusivity, first_step, spacing)
            raise ValueError(
                "dt violates the forward-Euler 1D diffusion stability limit: "
                f"D*dt/dx^2={diffusion_number:.17g} > 0.5; use dt <= "
                f"{max_stable_dt:.17g}"
            )

        exact_remainder = Fraction.from_float(final_time) - (
            planned_steps - 1
        ) * Fraction.from_float(nominal_dt)
        final_step_dt = float(exact_remainder)
        if final_step_dt <= 0.0 or final_step_dt > nominal_dt:
            raise ValueError(
                "the requested time grid cannot be represented safely in float64"
            )

    nominal_exact_diffusion_number = (
        _exact_diffusion_number(diffusivity, nominal_dt, spacing)
        if planned_steps > 1
        else Fraction(0)
    )
    final_exact_diffusion_number = (
        _exact_diffusion_number(diffusivity, final_step_dt, spacing)
        if planned_steps > 0
        else Fraction(0)
    )
    nominal_diffusion_number = float(nominal_exact_diffusion_number)
    final_diffusion_number = float(final_exact_diffusion_number)

    warnings.warn(_REFERENCE_SOLVER_WARNING, RuntimeWarning, stacklevel=2)

    time = 0.0
    step = 0
    last_dt = 0.0
    max_exact_diffusion_number = Fraction(0)
    dynamic_values = _evaluate_dynamic_boundaries(
        dynamic_boundaries,
        time,
        problem=problem,
        expected_problem=problem_snapshot,
    )
    _impose_dirichlet_boundaries(solution, static_boundaries, dynamic_values)

    time_history = [time]
    solution_history = [solution.copy()]
    bc_history = {side: [dynamic_values[side]] for side in dynamic_boundaries}

    for step_index in range(planned_steps):
        is_final_step = step_index + 1 == planned_steps
        next_time = final_time if is_final_step else (step_index + 1) * nominal_dt
        if not is_final_step and next_time >= final_time:
            # Exact float arithmetic says this nominal endpoint is still below
            # t_end, but nearest-float rounding can map both to the same value.
            # Preserve a strictly increasing observable clock without dropping
            # the real final remainder step.
            next_time = math.nextafter(final_time, -math.inf)
        step_dt = final_step_dt if is_final_step else nominal_dt
        if step_dt <= 0.0:
            raise RuntimeError("time integration stopped making forward progress")
        if next_time <= time:
            raise ValueError(
                "the requested time grid cannot make forward progress in float64; "
                "increase dt or reduce t_end"
            )

        diffusion_number = (
            final_diffusion_number if is_final_step else nominal_diffusion_number
        )
        exact_diffusion_number = (
            final_exact_diffusion_number
            if is_final_step
            else nominal_exact_diffusion_number
        )
        next_solution = solution.copy()
        _advance_diffusion_interior(
            solution,
            next_solution,
            diffusion_number,
            exact_diffusion_number,
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
            next_solution[index] = _advance_neumann_boundary(
                center=float(solution[index]),
                neighbor=float(solution[neighbor]),
                outward_derivative=outward_derivative,
                spacing=spacing,
                diffusion_number=diffusion_number,
                exact_diffusion_number=exact_diffusion_number,
            )

        next_dynamic_values = _evaluate_dynamic_boundaries(
            dynamic_boundaries,
            next_time,
            problem=problem,
            expected_problem=problem_snapshot,
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
        max_exact_diffusion_number = max(
            max_exact_diffusion_number, exact_diffusion_number
        )

        if save_every is not None and step % save_every == 0:
            time_history.append(time)
            solution_history.append(solution.copy())
            for side in dynamic_boundaries:
                bc_history[side].append(dynamic_values[side])

        if callback is not None:
            callback_solution = solution.copy()
            callback_solution.setflags(write=False)
            callback(time, callback_solution)
            _assert_reference_problem_unchanged(problem, problem_snapshot)

    if time_history[-1] != time:
        time_history.append(time)
        solution_history.append(solution.copy())
        for side in dynamic_boundaries:
            bc_history[side].append(dynamic_values[side])

    positive_periods = []
    for waveform in dynamic_boundaries.values():
        waveform_period = _nonnegative(waveform.period(), "boundary waveform period")
        if _waveform_may_call_user_code(waveform):
            _assert_reference_problem_unchanged(problem, problem_snapshot)
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
            "max_diffusion_number": _diagnostic_diffusion_number(
                max_exact_diffusion_number
            ),
            "max_diffusion_number_exact": (
                f"{max_exact_diffusion_number.numerator}/"
                f"{max_exact_diffusion_number.denominator}"
            ),
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
    rate = _positive(bpm, "bpm")
    return _period_from_rate(60.0, rate, "bpm")


def period_to_heart_rate(T: float) -> float:
    """Convert period in seconds to heart rate in BPM.

    Args:
        T: Period in seconds

    Returns:
        Heart rate in beats per minute
    """
    period = _positive(T, "T")
    heart_rate = 60.0 / period
    if not math.isfinite(heart_rate) or heart_rate <= 0.0:
        raise ValueError("T produces a heart rate outside the finite float64 range")
    return heart_rate


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
