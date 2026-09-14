"""Owned, explicitly timed concentration protocols for transport experiments."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Sequence

from .mesh_utils import _finite_float
from .units import Dimension, Quantity


def _nonnegative(value: float | Quantity, name: str, dimension: Dimension) -> float:
    if isinstance(value, Quantity):
        value = value.require(dimension)
    result = _finite_float(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True, init=False)
class ConcentrationSchedule:
    """A prescribed concentration with explicit change times.

    Args:
        times: Strictly increasing times in seconds, starting at zero.
        values: Nonnegative concentrations in mol/m^3, one per time. Quantity
            values are converted with dimension checks, as are quantity times.
        interpolation: ``'step'`` holds each value until the next time and is
            right-continuous at jumps. ``'linear'`` joins neighboring values.
            Both hold the last value indefinitely.

    The schedule owns immutable tuples. For a pulse from 10 through 20 seconds,
    use ``ConcentrationSchedule([0, 10, 20], [0, 1, 0])``. The coupled solver
    splits integration at connected schedule knots, even between output frames.
    """

    times: tuple[float, ...]
    values: tuple[float, ...]
    interpolation: str

    def __init__(
        self,
        times: Sequence[float | Quantity],
        values: Sequence[float | Quantity],
        *,
        interpolation: str = "step",
    ):
        if isinstance(times, (str, bytes)) or isinstance(values, (str, bytes)):
            raise TypeError(
                "times and values must be sequences of real numbers or quantities"
            )
        time_values = tuple(
            _nonnegative(t, "schedule time", Dimension.TIME) for t in times
        )
        concentrations = tuple(
            _nonnegative(c, "schedule concentration", Dimension.MOLAR_CONCENTRATION)
            for c in values
        )
        if not time_values or len(time_values) != len(concentrations):
            raise ValueError("a schedule needs matching nonempty times and values")
        if time_values[0] != 0.0 or any(
            b <= a for a, b in zip(time_values, time_values[1:])
        ):
            raise ValueError(
                "schedule times must start at zero and be strictly increasing"
            )
        if interpolation not in ("step", "linear"):
            raise ValueError("interpolation must be 'step' or 'linear'")
        object.__setattr__(self, "times", time_values)
        object.__setattr__(self, "values", concentrations)
        object.__setattr__(self, "interpolation", interpolation)

    def _value(self, time: float, *, left: bool = False) -> float:
        search = bisect_left if left else bisect_right
        index = max(0, search(self.times, time) - 1)
        if self.interpolation == "step" or index == len(self.times) - 1:
            return self.values[index]
        fraction = (time - self.times[index]) / (
            self.times[index + 1] - self.times[index]
        )
        # Convex interpolation preserves nonnegativity and avoids cancellation
        # when a decreasing ramp reaches zero exactly at its final knot.
        return (1.0 - fraction) * self.values[index] + fraction * self.values[index + 1]

    def at(self, time: float | Quantity, *, side: str = "right") -> float:
        """Concentration at a nonnegative time with an explicit limit at jumps.

        ``side='right'`` returns the new value; ``side='left'`` returns the value
        immediately before a jump. At time zero both return the initial value.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        return self._value(
            _nonnegative(time, "time", Dimension.TIME), left=side == "left"
        )
