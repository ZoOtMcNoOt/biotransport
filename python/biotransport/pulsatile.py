"""
Pulsatile Boundary Conditions for Time-Varying Transport Simulations.

Provides time-dependent boundary condition functions for simulating
physiological processes with cardiac cycle variations:

- Arterial pressure waveforms
- Venous flow patterns
- Drug infusion profiles
- Periodic heating/cooling

Example:
    >>> import biotransport as bt
    >>>
    >>> # Create a sinusoidal BC
    >>> bc = bt.SinusoidalBC(mean=100, amplitude=20, frequency=1.0)  # 1 Hz
    >>> print(bc(t=0.5))  # Value at t=0.5s
    >>>
    >>> # Use arterial pressure waveform
    >>> arterial = bt.ArterialPressureBC(systolic=120, diastolic=80, heart_rate=72)
    >>> print(arterial(t=0.25))
    >>>
    >>> # Solve with pulsatile BC
    >>> result = bt.solve_pulsatile(
    ...     problem,
    ...     t_end=5.0,
    ...     pulsatile_bcs={bt.Boundary.Left: arterial}
    ... )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import numpy as np

from ._core import (
    Boundary,
    TransportProblem,
)


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

    def __call__(self, t: float) -> float:
        return self.value

    def period(self) -> float:
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

    def __call__(self, t: float) -> float:
        return self.mean + self.amplitude * np.sin(
            2.0 * np.pi * self.frequency * t + self.phase
        )

    def period(self) -> float:
        return 1.0 / self.frequency if self.frequency > 0 else 0.0


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

    def __call__(self, t: float) -> float:
        if t <= self.t_start:
            return self.start_value
        elif t >= self.t_start + self.duration:
            return self.end_value
        else:
            frac = (t - self.t_start) / self.duration
            return self.start_value + frac * (self.end_value - self.start_value)

    def period(self) -> float:
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

    def __call__(self, t: float) -> float:
        return self.value_before if t < self.t_step else self.value_after

    def period(self) -> float:
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

    def __call__(self, t: float) -> float:
        T = 1.0 / self.frequency if self.frequency > 0 else 1.0
        t_shifted = t + self.phase * T
        t_in_cycle = t_shifted % T
        return self.high_value if t_in_cycle < self.duty_cycle * T else self.low_value

    def period(self) -> float:
        return 1.0 / self.frequency if self.frequency > 0 else 0.0


@dataclass
class CustomBC(PulsatileBC):
    """Custom time-varying boundary condition from user-provided function.

    Attributes:
        func: Callable that takes time t and returns BC value
        T: Period of the waveform (0 for non-periodic)
    """

    func: Callable[[float], float] = field(default=lambda t: 0.0)
    T: float = 0.0  # Period (0 = non-periodic)

    def __call__(self, t: float) -> float:
        return self.func(t)

    def period(self) -> float:
        return self.T


# =============================================================================
# Physiological Cardiac Waveforms
# =============================================================================


@dataclass
class ArterialPressureBC(PulsatileBC):
    """Arterial pressure waveform boundary condition.

    Generates a realistic arterial pressure waveform with systolic peak,
    dicrotic notch, and diastolic decay. Based on a sum of harmonics
    approximation to the cardiac pressure waveform.

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

    def __call__(self, t: float) -> float:
        # Period of cardiac cycle
        T = 60.0 / self.heart_rate
        omega = 2.0 * np.pi / T

        # Pressure range
        pp = self.systolic - self.diastolic  # Pulse pressure
        mean = (self.systolic + 2.0 * self.diastolic) / 3.0  # MAP approximation

        # Fourier-based arterial waveform (4 harmonics)
        # Coefficients approximate a typical arterial pressure shape
        p = mean
        p += 0.50 * pp * np.sin(omega * t)  # Fundamental
        p += 0.25 * pp * np.sin(2.0 * omega * t - 0.2)  # 2nd harmonic
        p += 0.10 * pp * np.sin(3.0 * omega * t - 0.4)  # 3rd harmonic
        p += 0.05 * pp * np.sin(4.0 * omega * t - 0.6)  # 4th harmonic

        return p

    def period(self) -> float:
        return 60.0 / self.heart_rate


@dataclass
class VenousPressureBC(PulsatileBC):
    """Venous pressure waveform boundary condition.

    Generates a venous pressure waveform with characteristic A, C, and V waves.
    Lower amplitude and slower variations than arterial pressure.

    Attributes:
        mean_pressure: Mean venous pressure in mmHg (default 8)
        amplitude: Pressure variation amplitude in mmHg (default 4)
        heart_rate: Heart rate in beats per minute (default 72)
    """

    mean_pressure: float = 8.0  # mmHg (central venous pressure)
    amplitude: float = 4.0  # mmHg
    heart_rate: float = 72.0  # bpm

    def __call__(self, t: float) -> float:
        T = 60.0 / self.heart_rate
        phase = (t % T) / T

        # Venous waveform: A wave (atrial contraction), C wave (AV valve bulging),
        # V wave (atrial filling)
        p = self.mean_pressure

        # Simplified 3-wave pattern
        # A wave at phase ~0.1, C wave at ~0.15, V wave at ~0.5
        a_wave = 0.4 * self.amplitude * np.exp(-((phase - 0.1) ** 2) / 0.005)
        c_wave = 0.2 * self.amplitude * np.exp(-((phase - 0.15) ** 2) / 0.002)
        v_wave = 0.4 * self.amplitude * np.exp(-((phase - 0.5) ** 2) / 0.02)

        return p + a_wave + c_wave + v_wave

    def period(self) -> float:
        return 60.0 / self.heart_rate


@dataclass
class CardiacOutputBC(PulsatileBC):
    """Pulsatile flow/velocity boundary condition for cardiac output.

    Generates a flow waveform representing blood flow rate through
    a vessel during the cardiac cycle.

    Attributes:
        mean_flow: Mean flow rate (units depend on application)
        peak_flow: Peak systolic flow rate
        heart_rate: Heart rate in beats per minute (default 72)
        ejection_fraction: Fraction of cycle during ejection (default 0.3)
    """

    mean_flow: float = 5.0  # L/min for cardiac output
    peak_flow: float = 25.0  # L/min peak
    heart_rate: float = 72.0  # bpm
    ejection_fraction: float = 0.3

    def __call__(self, t: float) -> float:
        T = 60.0 / self.heart_rate
        phase = (t % T) / T

        # Ejection phase: half-sine wave
        if phase < self.ejection_fraction:
            # Normalize phase within ejection period
            ejection_phase = phase / self.ejection_fraction
            return self.peak_flow * np.sin(np.pi * ejection_phase)
        else:
            # Diastole: low/zero flow (or slight retrograde)
            diastole_phase = (phase - self.ejection_fraction) / (
                1.0 - self.ejection_fraction
            )
            # Exponential decay to near-zero
            return self.mean_flow * 0.1 * np.exp(-3.0 * diastole_phase)

    def period(self) -> float:
        return 60.0 / self.heart_rate


@dataclass
class RespiratoryBC(PulsatileBC):
    """Respiratory modulation boundary condition.

    Generates a respiratory waveform for modeling ventilation effects
    on pressure or concentration.

    Attributes:
        mean: Mean value
        amplitude: Breath amplitude
        respiratory_rate: Breaths per minute (default 12)
        inspiration_fraction: Fraction of cycle during inspiration (default 0.4)
    """

    mean: float = 0.0
    amplitude: float = 1.0
    respiratory_rate: float = 12.0  # breaths per minute
    inspiration_fraction: float = 0.4

    def __call__(self, t: float) -> float:
        T = 60.0 / self.respiratory_rate
        phase = (t % T) / T

        if phase < self.inspiration_fraction:
            # Inspiration: rise
            insp_phase = phase / self.inspiration_fraction
            return self.mean + self.amplitude * (0.5 - 0.5 * np.cos(np.pi * insp_phase))
        else:
            # Expiration: fall
            exp_phase = (phase - self.inspiration_fraction) / (
                1.0 - self.inspiration_fraction
            )
            return self.mean + self.amplitude * (0.5 + 0.5 * np.cos(np.pi * exp_phase))

    def period(self) -> float:
        return 60.0 / self.respiratory_rate


@dataclass
class DrugInfusionBC(PulsatileBC):
    """Drug infusion boundary condition with bolus and maintenance phases.

    Models IV drug administration with optional loading dose followed
    by continuous infusion.

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

    def __call__(self, t: float) -> float:
        if t < self.infusion_start:
            return 0.0
        elif t < self.infusion_start + self.bolus_duration:
            # Bolus phase with slight exponential decay
            phase = (t - self.infusion_start) / self.bolus_duration
            return self.bolus_concentration * np.exp(-0.5 * phase)
        else:
            return self.maintenance_concentration

    def period(self) -> float:
        return 0.0  # Non-periodic


# =============================================================================
# Composite Waveforms
# =============================================================================


@dataclass
class CompositeBC(PulsatileBC):
    """Composite boundary condition from multiple waveforms.

    Combines multiple PulsatileBC objects using addition or multiplication.

    Attributes:
        components: List of PulsatileBC objects to combine
        operation: 'add' or 'multiply' (default 'add')
    """

    components: list = field(default_factory=list)
    operation: str = "add"

    def __call__(self, t: float) -> float:
        if not self.components:
            return 0.0

        if self.operation == "multiply":
            result = 1.0
            for bc in self.components:
                result *= bc(t)
        else:  # add
            result = 0.0
            for bc in self.components:
                result += bc(t)

        return result

    def period(self) -> float:
        # Return LCM of periods or max period as approximation
        periods = [bc.period() for bc in self.components if bc.period() > 0]
        return max(periods) if periods else 0.0


# =============================================================================
# Solver with Pulsatile BCs
# =============================================================================


@dataclass
class PulsatileResult:
    """Result from a pulsatile simulation.

    Attributes:
        solution: Final solution field
        time: Final simulation time
        time_history: Array of time points if snapshots were saved
        solution_history: List of solution snapshots
        bc_history: Dict mapping boundary to list of BC values over time
        stats: Solver statistics
    """

    solution: np.ndarray
    time: float
    time_history: np.ndarray = field(default_factory=lambda: np.array([]))
    solution_history: list = field(default_factory=list)
    bc_history: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def solve_pulsatile(
    problem: TransportProblem,
    t_end: float,
    pulsatile_bcs: Dict[Boundary, PulsatileBC],
    dt: Optional[float] = None,
    save_every: Optional[int] = None,
    callback: Optional[Callable[[float, np.ndarray], None]] = None,
) -> PulsatileResult:
    """Solve a transport problem with time-varying boundary conditions.

    This function performs explicit time integration while updating
    boundary conditions at each timestep according to the provided
    pulsatile BC functions.

    Args:
        problem: TransportProblem with initial condition and static BCs
        t_end: End time for simulation
        pulsatile_bcs: Dict mapping Boundary enum to PulsatileBC objects
        dt: Time step (if None, uses stability-limited dt)
        save_every: Save solution every N steps (None = only final)
        callback: Optional function called each step with (t, solution)

    Returns:
        PulsatileResult with final and optionally saved solutions

    Example:
        >>> bc_left = bt.ArterialPressureBC(systolic=120, diastolic=80)
        >>> result = bt.solve_pulsatile(
        ...     problem,
        ...     t_end=5.0,
        ...     pulsatile_bcs={bt.Boundary.Left: bc_left}
        ... )
    """
    mesh = problem.mesh()
    D = problem.diffusivity()

    # Compute stable timestep if not provided
    if dt is None:
        dx = mesh.dx()
        dt = 0.4 * dx * dx / D  # CFL-like stability condition

    # Initialize solution
    u = np.array(problem.initial(), dtype=np.float64)
    dx = mesh.dx()
    dx2 = dx * dx

    # Tracking
    t = 0.0
    step = 0
    time_history = [0.0]
    solution_history = [u.copy()]
    bc_history = {side: [bc(0.0)] for side, bc in pulsatile_bcs.items()}

    # Get static boundaries for sides without pulsatile BCs
    static_bcs = problem.boundaries()

    while t < t_end:
        # Adjust final step
        if t + dt > t_end:
            dt = t_end - t

        # Update pulsatile BCs for current time
        left_val = (
            pulsatile_bcs[Boundary.Left](t)
            if Boundary.Left in pulsatile_bcs
            else static_bcs[0].value
        )
        right_val = (
            pulsatile_bcs[Boundary.Right](t)
            if Boundary.Right in pulsatile_bcs
            else static_bcs[1].value
        )

        # Compute RHS (diffusion only for now, 1D)
        dudt = np.zeros_like(u)
        for i in range(1, len(u) - 1):
            dudt[i] = D * (u[i - 1] - 2 * u[i] + u[i + 1]) / dx2

        # Forward Euler step
        u = u + dt * dudt

        # Apply boundary conditions
        u[0] = left_val
        u[-1] = right_val

        t += dt
        step += 1

        # Save history
        if save_every is not None and step % save_every == 0:
            time_history.append(t)
            solution_history.append(u.copy())
            for side, bc in pulsatile_bcs.items():
                bc_history[side].append(bc(t))

        # Callback
        if callback is not None:
            callback(t, u)

    # Always save final state
    if save_every is None or step % save_every != 0:
        time_history.append(t)
        solution_history.append(u.copy())
        for side, bc in pulsatile_bcs.items():
            bc_history[side].append(bc(t))

    return PulsatileResult(
        solution=u,
        time=t,
        time_history=np.array(time_history),
        solution_history=solution_history,
        bc_history=bc_history,
        stats={"steps": step, "dt": dt, "t_end": t},
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
    return 60.0 / bpm


def period_to_heart_rate(T: float) -> float:
    """Convert period in seconds to heart rate in BPM.

    Args:
        T: Period in seconds

    Returns:
        Heart rate in beats per minute
    """
    return 60.0 / T


def sample_waveform(
    bc: PulsatileBC, t_start: float = 0.0, t_end: float = 1.0, num_points: int = 100
) -> tuple:
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
    times = np.linspace(t_start, t_end, num_points)
    values = np.array([bc(t) for t in times])
    return times, values
