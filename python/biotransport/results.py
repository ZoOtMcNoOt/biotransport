"""Result containers shared by every BioTransport solve entry point.

A :class:`Result` bundles the solved field(s), the exact physical time, the
step count, the solver's own diagnostics object, a copy of the mesh, and any
snapshots requested through ``save_times``.  It carries the identifier of the
scientific contract that produced it so evidence and exclusions can be looked
up through :mod:`biotransport.contracts`.  It performs no numerics.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np

__all__ = ["Result", "Snapshots"]


def _owned_read_only(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    array = array.astype(np.float64, copy=True).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _finite_time(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class Snapshots(Mapping[float, np.ndarray]):
    """Fields recorded at requested absolute times, as an immutable mapping.

    Keys are the exact ``save_times`` values in increasing order; each value is
    an owned, read-only float64 array.  ``snapshots[0.5]`` looks a time up
    exactly; use :meth:`at` for a tolerance-based lookup.
    """

    __slots__ = ("_times", "_fields")

    def __init__(self, times: Sequence[float], fields: Sequence[Any]) -> None:
        time_values = tuple(
            _finite_time(t, f"snapshot time {i}") for i, t in enumerate(times)
        )
        field_values = tuple(
            _owned_read_only(values, f"snapshot field {i}")
            for i, values in enumerate(fields)
        )
        if len(time_values) != len(field_values):
            raise ValueError("snapshot times and fields must have the same length")
        if any(t < 0.0 for t in time_values):
            raise ValueError("snapshot times must be non-negative")
        if any(b <= a for a, b in zip(time_values, time_values[1:])):
            raise ValueError("snapshot times must be strictly increasing")
        sizes = {values.size for values in field_values}
        if len(sizes) > 1:
            raise ValueError("all snapshot fields must have the same number of values")
        self._times = time_values
        self._fields = field_values

    @property
    def times(self) -> tuple[float, ...]:
        """Snapshot times in increasing order."""
        return self._times

    @property
    def fields(self) -> tuple[np.ndarray, ...]:
        """Read-only fields in the same order as :attr:`times`."""
        return self._fields

    def __getitem__(self, time: float) -> np.ndarray:
        key = _finite_time(time, "time")
        for recorded, values in zip(self._times, self._fields):
            if recorded == key:
                return values
        raise KeyError(time)

    def __iter__(self) -> Iterator[float]:
        return iter(self._times)

    def __len__(self) -> int:
        return len(self._times)

    def at(self, time: float, *, abs_tol: float = 0.0) -> np.ndarray:
        """Return the field whose time is within ``abs_tol`` of ``time``."""
        key = _finite_time(time, "time")
        tolerance = _finite_time(abs_tol, "abs_tol")
        if tolerance < 0.0:
            raise ValueError("abs_tol must be non-negative")
        for recorded, values in zip(self._times, self._fields):
            if abs(recorded - key) <= tolerance:
                return values
        raise KeyError(f"no snapshot within {tolerance} of t={key}")

    def stacked(self) -> np.ndarray:
        """Return all fields as one ``(len(times), n_nodes)`` array (a copy)."""
        if not self._fields:
            return np.empty((0, 0), dtype=np.float64)
        return np.stack(self._fields, axis=0)

    def __repr__(self) -> str:
        return f"Snapshots(times={self._times!r})"


@dataclass(frozen=True)
class Result:
    """The outcome of one solve: field(s), time, steps, diagnostics, mesh, snapshots.

    ``fields`` maps a field name to an owned, read-only float64 array;
    ``primary`` names the field returned by :attr:`field`.  ``diagnostics`` is
    the solver's own diagnostics object (for the canonical path a
    :class:`~biotransport.SolveDiagnostics`).  ``contract`` is the identifier
    of the scientific contract that produced the result, resolvable through
    :func:`biotransport.contracts.get_contract` or
    :func:`biotransport.contracts.get_python_numerical_contract`.
    """

    fields: Mapping[str, np.ndarray]
    time: float
    steps: int
    diagnostics: Any
    mesh: Any
    contract: str
    snapshots: Snapshots = field(default_factory=lambda: Snapshots((), ()))
    native: Any = None
    primary: str = "concentration"

    def __post_init__(self) -> None:
        if not isinstance(self.fields, Mapping) or not self.fields:
            raise TypeError("fields must be a non-empty mapping of name -> values")
        owned = {}
        for name, values in self.fields.items():
            if not isinstance(name, str) or not name:
                raise TypeError("field names must be non-empty strings")
            owned[name] = _owned_read_only(values, f"field {name!r}")
        if self.primary not in owned:
            raise ValueError(
                f"primary field {self.primary!r} is not among {sorted(owned)}"
            )
        object.__setattr__(self, "fields", MappingProxyType(owned))
        object.__setattr__(self, "time", _finite_time(self.time, "time"))
        if (
            isinstance(self.steps, bool)
            or int(self.steps) != self.steps
            or self.steps < 0
        ):
            raise ValueError("steps must be a non-negative integer")
        object.__setattr__(self, "steps", int(self.steps))
        if not isinstance(self.contract, str) or not self.contract:
            raise TypeError("contract must be a non-empty string")
        if not isinstance(self.snapshots, Snapshots):
            raise TypeError("snapshots must be a Snapshots mapping")

    @property
    def field(self) -> np.ndarray:
        """The primary field (read-only)."""
        return self.fields[self.primary]

    @property
    def concentration(self) -> np.ndarray:
        """The ``"concentration"`` field, when this result has one."""
        try:
            return self.fields["concentration"]
        except KeyError:
            raise AttributeError(
                f"this result has no 'concentration' field; available fields: "
                f"{sorted(self.fields)}"
            ) from None

    def as_grid(self, name: str | None = None) -> np.ndarray:
        """Reshape a field to the mesh's node grid.

        1D meshes return the flat vector; 2D meshes return ``(ny + 1, nx + 1)``;
        3D meshes return ``(nz + 1, ny + 1, nx + 1)``.  The array is a read-only
        view of the stored field.
        """
        values = self.fields[self.primary if name is None else name]
        mesh = self.mesh
        if hasattr(mesh, "nz"):
            shape = (int(mesh.nz()) + 1, int(mesh.ny()) + 1, int(mesh.nx()) + 1)
        elif hasattr(mesh, "is_1d") and not mesh.is_1d():
            shape = (int(mesh.ny()) + 1, int(mesh.nx()) + 1)
        else:
            return values
        grid = values.reshape(shape, order="C")
        grid.setflags(write=False)
        return grid

    def plot(self, **kwargs: Any):
        """Plot the primary field with :func:`biotransport.plot`."""
        from .visualization import plot

        return plot(self, **kwargs)

    def __repr__(self) -> str:
        return (
            f"Result(contract={self.contract!r}, time={self.time!r}, steps={self.steps}, "
            f"fields={sorted(self.fields)}, snapshots={len(self.snapshots)})"
        )
