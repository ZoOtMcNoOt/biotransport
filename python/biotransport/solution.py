"""The object you get back from a solve.

A :class:`Solution` knows three things the raw C++ result does not: the mesh it
was computed on, the times that were saved along the way, and the problem that
produced it. That is enough for it to plot itself, compare itself against an
exact answer, and explain its own numerics -- without you threading a mesh
through every call.

It is also a drop-in replacement for the native ``TransportResult``:
``concentration``, ``time``, ``solution`` and ``diagnostics`` mean exactly what
they always meant, and refer to the final state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .mesh_utils import as_2d, x_nodes, y_nodes, xy_grid

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.animation import FuncAnimation

    from ._core import SolveDiagnostics


__all__ = ["Solution", "ErrorReport"]


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _read_only(array: np.ndarray) -> np.ndarray:
    """Return a float64 copy that cannot be written through.

    Used for the frames a Solution stores, so nothing downstream can corrupt
    them. Public accessors hand out writable copies instead -- see :func:`_fresh`.
    """

    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _fresh(array: np.ndarray) -> np.ndarray:
    """Return a writable float64 copy, matching native result semantics."""

    return np.array(array, dtype=np.float64, copy=True)


def _trapezoid_weights(nodes: np.ndarray) -> np.ndarray:
    """Control-volume widths for node-centred data, summing to the domain length.

    Interior nodes own half of each neighbouring cell; boundary nodes own a half
    cell. These are the same weights the C++ core uses for its mass diagnostic,
    so error norms computed here are consistent with the reported mass change.
    """

    if nodes.size < 2:
        raise ValueError("at least two nodes are required")
    widths = np.diff(nodes)
    weights = np.empty(nodes.size, dtype=np.float64)
    weights[0] = 0.5 * widths[0]
    weights[-1] = 0.5 * widths[-1]
    if nodes.size > 2:
        weights[1:-1] = 0.5 * (widths[:-1] + widths[1:])
    return weights


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------


class ErrorReport:
    """How far a numerical field is from a reference field.

    Produced by :meth:`Solution.compare`. Printing it gives you a short,
    readable verdict; the individual numbers are attributes if you want to
    assert on them in a test.

    Attributes:
        max_abs: Largest absolute difference anywhere (the L-infinity norm).
        rms: Root-mean-square difference over the nodes.
        l2: Domain-averaged L2 norm, ``sqrt(int e^2 dV / V)``, using the same
            control-volume weights as the solver's mass accounting. This is the
            norm to quote in a convergence study, because it does not change
            meaning when you refine the grid.
        rel_max_abs: ``max_abs`` divided by the peak-to-peak range of the
            reference field. ``None`` when the reference field is constant.
        rel_l2: ``l2`` divided by the peak-to-peak range of the reference field.
            ``None`` when the reference field is constant.
        n: Number of nodes compared.
    """

    __slots__ = ("max_abs", "rms", "l2", "rel_max_abs", "rel_l2", "n", "_at_node")

    max_abs: float
    rms: float
    l2: float
    rel_max_abs: float | None
    rel_l2: float | None
    n: int
    _at_node: int

    def __init__(
        self,
        *,
        max_abs: float,
        rms: float,
        l2: float,
        rel_max_abs: float | None,
        rel_l2: float | None,
        n: int,
        at_node: int,
    ) -> None:
        self.max_abs = max_abs
        self.rms = rms
        self.l2 = l2
        self.rel_max_abs = rel_max_abs
        self.rel_l2 = rel_l2
        self.n = n
        self._at_node = at_node

    @property
    def worst_node(self) -> int:
        """Flat index of the node with the largest absolute error."""

        return self._at_node

    def __repr__(self) -> str:
        parts = [
            f"max|error| = {self.max_abs:.4g}",
            f"L2 = {self.l2:.4g}",
        ]
        if self.rel_l2 is not None:
            parts.append(f"relative L2 = {100.0 * self.rel_l2:.3g}% of range")
        return f"ErrorReport({', '.join(parts)}, over {self.n} nodes)"

    def __str__(self) -> str:
        lines = [
            f"Compared {self.n} nodes against the reference field.",
            f"  largest absolute error  {self.max_abs:.6g}  (at node {self._at_node})",
            f"  RMS error               {self.rms:.6g}",
            f"  L2 error                {self.l2:.6g}",
        ]
        if self.rel_l2 is not None and self.rel_max_abs is not None:
            lines.append(
                f"  as a fraction of the reference range: "
                f"{100.0 * self.rel_max_abs:.3g}% peak, "
                f"{100.0 * self.rel_l2:.3g}% L2"
            )
        else:
            lines.append(
                "  the reference field is constant, so relative errors are undefined"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Solution
# ---------------------------------------------------------------------------


class Solution:
    """A solved transport problem, with everything needed to interpret it.

    You do not construct this yourself -- :func:`biotransport.solve` returns it.

    The final state is available under the same names the native result uses::

        sol.concentration   # flat node values at the final time
        sol.time            # the final time actually reached
        sol.diagnostics     # native numerics report for the last segment

    On top of that, a ``Solution`` remembers its geometry and its saved frames::

        sol.x               # node coordinates
        sol.c               # final field, shaped for the mesh (2D stays 2D)
        sol.times           # every time that was saved
        sol.at(0.05)        # the field at (the saved time nearest) t = 0.05
        sol.plot()          # the right plot for this geometry
        sol.summary()       # a readable account of what the solver did
    """

    __slots__ = (
        "_mesh",
        "_times",
        "_fields",
        "_diagnostics",
        "_problem",
        "_total_steps",
        "_steady",
        "_newton",
    )

    # Bare annotations, so type checkers know the slot types. They create no
    # class attributes, which is what __slots__ requires.
    _mesh: Any
    _times: tuple[float, ...]
    _fields: tuple[np.ndarray, ...]
    _diagnostics: tuple[SolveDiagnostics, ...]
    _problem: Any
    _total_steps: int
    _steady: bool
    _newton: Any

    def __init__(
        self,
        *,
        mesh: Any,
        times: Sequence[float],
        fields: Sequence[np.ndarray],
        diagnostics: Sequence[SolveDiagnostics] = (),
        problem: Any = None,
        total_steps: int = 0,
        steady: bool = False,
        newton: Any = None,
    ) -> None:
        if len(times) != len(fields):
            raise ValueError("times and fields must have the same length")
        if not times:
            raise ValueError("a Solution needs at least one saved frame")

        checked_times = tuple(
            _finite_float(value, f"times[{index}]") for index, value in enumerate(times)
        )
        if any(
            later <= earlier for earlier, later in zip(checked_times, checked_times[1:])
        ):
            raise ValueError("saved times must be strictly increasing")

        node_count = int(mesh.num_nodes())
        checked_fields = []
        for index, field in enumerate(fields):
            flat = np.asarray(field, dtype=np.float64).reshape(-1)
            if flat.size != node_count:
                raise ValueError(
                    f"fields[{index}] has {flat.size} values but the mesh has "
                    f"{node_count} nodes"
                )
            checked_fields.append(_read_only(flat))

        object.__setattr__(self, "_mesh", mesh)
        object.__setattr__(self, "_times", checked_times)
        object.__setattr__(self, "_fields", tuple(checked_fields))
        object.__setattr__(self, "_diagnostics", tuple(diagnostics))
        object.__setattr__(self, "_problem", problem)
        object.__setattr__(self, "_total_steps", int(total_steps))
        object.__setattr__(self, "_steady", bool(steady))
        object.__setattr__(self, "_newton", newton)

    def __setattr__(self, name: str, value: object) -> None:  # pragma: no cover
        raise AttributeError("Solution is read-only")

    # -- geometry -----------------------------------------------------------

    @property
    def mesh(self) -> Any:
        """The mesh this solution lives on."""

        return self._mesh

    @property
    def problem(self) -> Any:
        """The problem that produced this solution, when it is known."""

        return self._problem

    @property
    def is_1d(self) -> bool:
        """True for a 1D mesh."""

        is_1d = getattr(self._mesh, "is_1d", None)
        return bool(is_1d()) if callable(is_1d) else False

    @property
    def x(self) -> np.ndarray:
        """Node coordinates along x."""

        return x_nodes(self._mesh)

    @property
    def y(self) -> np.ndarray:
        """Node coordinates along y. Only meaningful for a 2D mesh."""

        return y_nodes(self._mesh)

    @property
    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """``(X, Y)`` meshgrid arrays for a 2D mesh."""

        return xy_grid(self._mesh)

    # -- fields -------------------------------------------------------------

    def _shape(self, flat: np.ndarray) -> np.ndarray:
        """Give a flat node vector the natural shape for this mesh."""

        if self.is_1d:
            return _fresh(flat)
        return _fresh(as_2d(self._mesh, flat))

    @property
    def c(self) -> np.ndarray:
        """The final field, shaped for the mesh.

        1D solutions come back as a flat array; 2D solutions come back with
        shape ``(ny + 1, nx + 1)`` so they line up with :attr:`grid` and can be
        handed straight to ``contourf`` or ``imshow``.
        """

        return self._shape(self._fields[-1])

    @property
    def c0(self) -> np.ndarray:
        """The first saved field, shaped for the mesh.

        This is the initial condition when the problem recorded one, which is
        the case for any problem built with :class:`biotransport.Problem`.
        """

        return self._shape(self._fields[0])

    @property
    def concentration(self) -> np.ndarray:
        """Flat node values at the final time (native result compatibility)."""

        return _fresh(self._fields[-1])

    @property
    def solution(self) -> np.ndarray:
        """Alias for :attr:`concentration` (native result compatibility)."""

        return _fresh(self._fields[-1])

    @property
    def time(self) -> float:
        """The final time reached."""

        return self._times[-1]

    @property
    def t(self) -> float:
        """Alias for :attr:`time`."""

        return self._times[-1]

    @property
    def times(self) -> tuple[float, ...]:
        """Every time that was saved, in increasing order."""

        return self._times

    @property
    def fields(self) -> tuple[np.ndarray, ...]:
        """Every saved field as a flat array, aligned with :attr:`times`."""

        return tuple(_fresh(field) for field in self._fields)

    @property
    def history(self) -> np.ndarray:
        """Saved frames stacked into one array of shape ``(len(times), nodes)``."""

        return np.vstack(self._fields)

    @property
    def steady(self) -> bool:
        """True when this came from a steady solve, where :attr:`time` has no meaning."""

        return self._steady

    @property
    def newton(self) -> Any:
        """Newton convergence report from a steady solve, otherwise ``None``."""

        return self._newton

    @property
    def diagnostics(self) -> SolveDiagnostics | None:
        """Native numerics report for the final segment.

        ``None`` for a steady solve, which reports convergence through
        :attr:`newton` instead.
        """

        return self._diagnostics[-1] if self._diagnostics else None

    @property
    def all_diagnostics(self) -> tuple[SolveDiagnostics, ...]:
        """Native numerics reports, one per solved segment."""

        return self._diagnostics

    @property
    def steps(self) -> int:
        """Total time steps taken across every segment."""

        return self._total_steps

    def at(self, time: float) -> np.ndarray:
        """The field at the saved time nearest ``time``, shaped for the mesh.

        Frames only exist at the times in :attr:`times`, so this snaps to the
        closest one rather than interpolating. Pass ``save_every`` or ``save_at``
        to :func:`biotransport.solve` if you need finer resolution in time.

        Args:
            time: The time you are interested in.

        Returns:
            The saved field closest to ``time``.
        """

        target = _finite_float(time, "time")
        index = int(np.argmin([abs(saved - target) for saved in self._times]))
        return self._shape(self._fields[index])

    def frame(self, index: int) -> np.ndarray:
        """The ``index``-th saved field, shaped for the mesh."""

        return self._shape(self._fields[index])

    def nearest_time(self, time: float) -> float:
        """The saved time that :meth:`at` would give you for ``time``."""

        target = _finite_float(time, "time")
        index = int(np.argmin([abs(saved - target) for saved in self._times]))
        return self._times[index]

    def __getitem__(self, time: float) -> np.ndarray:
        return self.at(time)

    def __len__(self) -> int:
        return len(self._times)

    # -- scalar reductions --------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        """Control-volume weights per node, summing to the domain size.

        These are flat, matching :attr:`concentration`, so integrate with
        ``float(sol.weights @ sol.concentration)``. Pairing them with :attr:`c`
        will not work on a 2D mesh, because ``c`` is shaped -- flatten it first,
        or just call :meth:`total`, which does this for you.

        On a cylindrical or spherical mesh these are shell measures rather than
        widths, so an integral counts the outside of the domain more heavily
        than the centre, as it should.
        """

        mesh = self._mesh
        control_volume = getattr(mesh, "control_volume", None)
        if not callable(control_volume):  # pragma: no cover - older extensions
            if self.is_1d:
                return _trapezoid_weights(x_nodes(mesh))
            return np.outer(
                _trapezoid_weights(y_nodes(mesh)), _trapezoid_weights(x_nodes(mesh))
            ).reshape(-1)

        # control_volume carries the radial measure and reduces to dx (dx/2 at
        # the ends) on a Cartesian mesh, so one expression covers both.
        radial = np.array(
            [control_volume(i) for i in range(mesh.nx() + 1)], dtype=np.float64
        )
        if self.is_1d:
            return radial
        axial = np.array(
            [mesh.axial_height(j) for j in range(mesh.ny() + 1)], dtype=np.float64
        )
        return np.outer(axial, radial).reshape(-1)

    def total(self, time: float | None = None) -> float:
        """Integrate the field over the domain (its total mass or heat content).

        Args:
            time: Which saved frame to integrate. Defaults to the final one.
        """

        field = self._fields[-1] if time is None else self.at(time).reshape(-1)
        return float(np.asarray(self.weights) @ field)

    def mean(self, time: float | None = None) -> float:
        """Volume-averaged value of the field."""

        weights = np.asarray(self.weights)
        field = self._fields[-1] if time is None else self.at(time).reshape(-1)
        return float((weights @ field) / weights.sum())

    def peak(self, time: float | None = None) -> float:
        """Largest value in the field."""

        field = self._fields[-1] if time is None else self.at(time).reshape(-1)
        return float(np.max(field))

    def trace(
        self,
        index: int | None = None,
        *,
        at: float | tuple[float, float] | Sequence[float] | None = None,
    ) -> np.ndarray:
        """How the value at one point evolved across the saved frames.

        The numerical equivalent of leaving a probe in the domain.

        Args:
            index: Flat node index to follow.
            at: Position to follow instead of an index -- an ``x`` coordinate on a
                1D mesh, or an ``(x, y)`` pair on a 2D one. The nearest node is
                used.

        Returns:
            One value per entry in :attr:`times`.

        Example:
            >>> sol.trace(at=0.003)          # 1D
            >>> sol.trace(at=(0.25, 0.5))    # 2D
        """

        if (index is None) == (at is None):
            raise TypeError("pass exactly one of index or at")

        if at is not None:
            coords_x = x_nodes(self._mesh)
            if self.is_1d:
                if np.ndim(at) != 0:
                    raise TypeError(
                        "a 1D mesh takes a single position, for example at=0.5"
                    )
                index = int(np.argmin(np.abs(coords_x - _finite_float(at, "at"))))
            else:
                position = np.asarray(at, dtype=np.float64).reshape(-1)
                if position.size != 2:
                    raise TypeError(
                        "a 2D mesh takes an (x, y) pair, for example at=(0.25, 0.5)"
                    )
                coords_y = y_nodes(self._mesh)
                column = int(np.argmin(np.abs(coords_x - position[0])))
                row = int(np.argmin(np.abs(coords_y - position[1])))
                # Fields are stored flat in C order over (ny + 1, nx + 1).
                index = row * coords_x.size + column

        return np.array([field[index] for field in self._fields], dtype=np.float64)

    # -- fluxes and rates ---------------------------------------------------

    def flux(self, time: float | None = None) -> Any:
        """The transport flux ``J = -D grad(c) + v c`` on the interior faces.

        Rebuilt from the saved field using the same face formulas the solver uses
        -- harmonic mean diffusivity, first-order upwinding -- so this is the flux
        the solver saw, not a fresh finite difference.

        Args:
            time: Which saved frame to use. Defaults to the final one.

        Returns:
            On a 1D mesh, one value per interior face (``nx`` of them, sitting
            between consecutive nodes). On a 2D mesh, a ``(Jx, Jy)`` pair.

        Example:
            >>> J = sol.flux()
            >>> print(f"peak flux {abs(J).max():.3g}")
        """

        from .fluxes import interior_flux

        field = self._fields[-1] if time is None else self.at(time).reshape(-1)
        return interior_flux(self._require_problem("flux"), self._mesh, field)

    def flux_at(self, side: Any, time: float | None = None) -> Any:
        """Outward flux through one boundary, per unit area.

        Positive means material is leaving through that side. This is the number
        most transport questions are really asking for: the oxygen flux into the
        tissue, the drug permeation rate through the membrane.

        Args:
            side: ``"left"``, ``"right"``, ``"bottom"`` or ``"top"``.
            time: Which saved frame to use. Defaults to the final one.

        Returns:
            A scalar on a 1D mesh; one value per boundary node on a 2D mesh.
        """

        from .fluxes import outward_flux

        field = self._fields[-1] if time is None else self.at(time).reshape(-1)
        return outward_flux(
            self._require_problem("flux_at"), self._mesh, field, side, self.time
        )

    def rate(self, side: Any, time: float | None = None) -> float:
        """Total outward transfer through a side, integrated over its area.

        In 1D this equals :meth:`flux_at`, because the boundary is a point. In 2D
        it is the flux integrated along that edge -- the quantity you would report
        as a total delivery or clearance rate.
        """

        from .fluxes import transfer_rate

        field = self._fields[-1] if time is None else self.at(time).reshape(-1)
        return transfer_rate(
            self._require_problem("rate"), self._mesh, field, side, self.time
        )

    def uptake(self, time: float | None = None) -> float:
        """Volume-integrated reaction rate.

        Negative for a consuming reaction, positive for a source. Compare against
        the boundary rates to see what fraction of what entered is being consumed.
        """

        from .fluxes import total_reaction

        field = self._fields[-1] if time is None else self.at(time).reshape(-1)
        return total_reaction(
            self._require_problem("uptake"),
            self._mesh,
            field,
            np.asarray(self.weights),
            self.time,
        )

    def balance(self, time: float | None = None) -> Any:
        """Account for where everything went, and check the books close.

        Reports what is stored, what is crossing each boundary, what the reaction
        is producing or consuming, and the residual in

            d(stored)/dt = (net inward transfer) + (reaction)

        On a steady solution the left side is zero, so the residual is a genuine
        check on the whole discretization -- it is the same statement a transport
        course spends weeks on, and it catches a boundary sign error immediately.

        Returns:
            A :class:`~biotransport.FluxReport`. Print it.

        Example:
            >>> print(bt.solve_steady(problem).balance())
        """

        from .fluxes import FluxReport, transfer_rate, total_reaction

        problem = self._require_problem("balance")
        mesh = self._mesh
        field = self._fields[-1] if time is None else self.at(time).reshape(-1)
        weights = np.asarray(self.weights)

        sides = ("left", "right") if self.is_1d else ("left", "right", "bottom", "top")

        def supply_rate(state: np.ndarray, when: float) -> float:
            """Net inward transfer plus reaction, at one instant."""

            leaving = sum(
                transfer_rate(problem, mesh, state, name, when) for name in sides
            )
            return -float(leaving) + total_reaction(problem, mesh, state, weights, when)

        when = self.time if time is None else self.nearest_time(time)
        by_side = {
            name: transfer_rate(problem, mesh, field, name, when) for name in sides
        }
        entered = -float(sum(by_side.values()))
        produced = total_reaction(problem, mesh, field, weights, when)

        stored = float(weights @ field)
        stored_initially = float(weights @ self._fields[0])

        residual: float | None = None
        supplied: float | None = None
        closure_error: float | None = None

        if self._steady:
            # No accumulation, so the instantaneous statement must close. This is
            # a real test of the discretization.
            residual = entered + produced
        elif len(self._times) >= 3:
            # A single frame cannot give d(stored)/dt, so check the integrated
            # form instead: the time integral of (in + reaction) should equal the
            # change in what is stored. The trapezoid rule over saved frames is
            # what limits the agreement, so it tightens as frames are added.
            rates = [
                supply_rate(state, moment)
                for state, moment in zip(self._fields, self._times)
            ]
            supplied = float(np.trapezoid(rates, self._times))
            closure_error = supplied - (stored - stored_initially)

        return FluxReport(
            stored=stored,
            stored_initially=stored_initially,
            entered=entered,
            produced=produced,
            by_side=by_side,
            residual=residual,
            supplied=supplied,
            closure_error=closure_error,
            steady=self._steady,
            frames=len(self._times),
        )

    def _require_problem(self, what: str) -> Any:
        if self._problem is None:
            raise ValueError(
                f"{what}() needs the problem that produced this solution, which "
                f"this Solution does not carry."
            )
        return self._problem

    # -- comparison ---------------------------------------------------------

    def compare(
        self,
        reference: Callable[..., Any] | Sequence[float] | np.ndarray,
        *,
        time: float | None = None,
        plot: bool = False,
        ax: Axes | None = None,
    ) -> ErrorReport:
        """Measure this solution against an exact or reference answer.

        This is the "did I get it right?" call. ``reference`` may be

        * a function of position -- ``f(x)`` on a 1D mesh, ``f(X, Y)`` on a 2D
          mesh, evaluated on the node coordinates;
        * a function of position and time -- ``f(x, t)`` or ``f(X, Y, t)``; or
        * an array of node values you computed some other way.

        Args:
            reference: The field to compare against, as described above.
            time: Which saved frame to compare. Defaults to the final one.
            plot: Draw the two fields and their difference.
            ax: Axes to draw into when ``plot`` is true.

        Returns:
            An :class:`ErrorReport`. Print it for a readable verdict.

        Example:
            >>> exact = lambda x: np.exp(-x)
            >>> print(sol.compare(exact))
        """

        when = self.time if time is None else self.nearest_time(time)
        numeric = self._fields[-1] if time is None else self.at(time).reshape(-1)
        expected = self._evaluate_reference(reference, when)

        error = numeric - expected
        weights = np.asarray(self.weights)
        domain = float(weights.sum())
        worst = int(np.argmax(np.abs(error)))
        max_abs = float(np.abs(error[worst]))
        rms = float(np.sqrt(np.mean(error**2)))
        l2 = float(np.sqrt(float(weights @ (error**2)) / domain))

        spread = float(np.max(expected) - np.min(expected))
        rel_max: float | None = None
        rel_l2: float | None = None
        if spread > 0.0:
            rel_max = max_abs / spread
            rel_l2 = l2 / spread

        if plot:
            self._plot_comparison(expected, numeric, when, ax=ax)

        return ErrorReport(
            max_abs=max_abs,
            rms=rms,
            l2=l2,
            rel_max_abs=rel_max,
            rel_l2=rel_l2,
            n=int(numeric.size),
            at_node=worst,
        )

    def _evaluate_reference(
        self,
        reference: Callable[..., Any] | Sequence[float] | np.ndarray,
        when: float,
    ) -> np.ndarray:
        """Turn a callable or array reference into a flat node vector."""

        if callable(reference):
            if self.is_1d:
                coords: tuple[np.ndarray, ...] = (x_nodes(self._mesh),)
            else:
                coords = xy_grid(self._mesh)
            try:
                raw = reference(*coords, when)
            except TypeError:
                raw = reference(*coords)
            values = np.asarray(raw, dtype=np.float64)
        else:
            values = np.asarray(reference, dtype=np.float64)

        flat = values.reshape(-1)
        expected_size = int(self._mesh.num_nodes())
        if flat.size == 1:
            flat = np.full(expected_size, float(flat[0]))
        if flat.size != expected_size:
            raise ValueError(
                f"the reference field has {flat.size} values but the mesh has "
                f"{expected_size} nodes"
            )
        if not np.all(np.isfinite(flat)):
            raise ValueError("the reference field must be finite everywhere")
        return flat

    def _plot_comparison(
        self,
        expected: np.ndarray,
        numeric: np.ndarray,
        when: float,
        *,
        ax: Axes | None = None,
    ) -> Axes:
        import matplotlib.pyplot as plt

        if self.is_1d:
            if ax is None:
                _fig, ax = plt.subplots(figsize=(7.5, 4.5))
            coords = x_nodes(self._mesh)
            ax.plot(coords, expected, "-", lw=2.5, alpha=0.45, label="reference")
            ax.plot(coords, numeric, "--", lw=1.8, label=f"biotransport (t = {when:g})")
            ax.set_xlabel("position")
            ax.set_ylabel("value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            return ax

        if ax is None:
            _fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        else:
            axes = np.atleast_1d(ax)
        X, Y = xy_grid(self._mesh)
        panels = (
            ("reference", as_2d(self._mesh, expected)),
            (f"biotransport (t = {when:g})", as_2d(self._mesh, numeric)),
            ("difference", as_2d(self._mesh, numeric - expected)),
        )
        for panel_ax, (label, field) in zip(np.atleast_1d(axes), panels):
            mesh_plot = panel_ax.contourf(X, Y, field, 40, cmap="viridis")
            panel_ax.set_title(label)
            panel_ax.set_xlabel("x")
            panel_ax.set_ylabel("y")
            panel_ax.figure.colorbar(mesh_plot, ax=panel_ax)
        return np.atleast_1d(axes)[0]

    # -- plotting -----------------------------------------------------------

    def plot(
        self,
        times: float | Sequence[float] | None = None,
        *,
        kind: Literal["auto", "line", "contour", "heatmap", "surface"] = "auto",
        ax: Axes | None = None,
        label: str | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        clabel: str | None = None,
        levels: int = 40,
        cmap: str = "viridis",
        colorbar: bool = True,
        legend: bool | None = None,
        save: str | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Draw this solution, picking a sensible plot for the geometry.

        A 1D solution becomes a line; a 2D solution becomes a filled contour.
        Extra keyword arguments go through to Matplotlib, so ``lw``, ``color``,
        ``alpha`` and friends work as you would expect.

        Args:
            times: Which saved frames to draw. A single time draws one curve;
                a list of times overlays them with a legend; ``"all"`` is
                spelled ``sol.times``. Defaults to the final frame.
            kind: ``"line"``, ``"contour"``, ``"heatmap"`` or ``"surface"``.
                ``"auto"`` picks line for 1D and contour for 2D.
            ax: Draw into these axes instead of making a new figure.
            label: Legend label for a single 1D curve.
            title: Axes title.
            save: Path to save the figure to.

        Returns:
            The Matplotlib ``Axes`` that was drawn into. Get the figure with
            ``ax.figure`` if you want to save or further compose it.

        Example:
            >>> sol.plot()                              # final state
            >>> sol.plot(times=[0.0, 0.01, 0.05])       # overlay snapshots
            >>> sol.plot(kind="surface")                # 2D as a surface
        """

        if kind not in {"auto", "line", "contour", "heatmap", "surface"}:
            raise ValueError(
                "kind must be 'auto', 'line', 'contour', 'heatmap' or 'surface'"
            )

        if self.is_1d:
            if kind not in {"auto", "line"}:
                raise ValueError(f"a 1D solution cannot be drawn as {kind!r}")
            ax = self._plot_1d(
                times,
                ax=ax,
                label=label,
                xlabel=xlabel,
                ylabel=ylabel,
                legend=legend,
                **kwargs,
            )
        else:
            if kind == "line":
                raise ValueError(
                    "a 2D solution cannot be drawn as a line; try kind='contour', "
                    "'heatmap' or 'surface'"
                )
            requested = np.asarray(self.time if times is None else times)
            if requested.ndim > 1 or requested.size != 1:
                raise ValueError(
                    "2D plots draw one frame at a time; pass a single time, or "
                    "call sol.animate() to see the evolution"
                )
            when = requested.reshape(-1)[0]
            ax = self._plot_2d(
                float(when),
                kind="contour" if kind == "auto" else kind,
                ax=ax,
                xlabel=xlabel,
                ylabel=ylabel,
                clabel=clabel,
                levels=levels,
                cmap=cmap,
                colorbar=colorbar,
                **kwargs,
            )

        if title:
            ax.set_title(title)
        if save:
            ax.figure.savefig(save, dpi=150, bbox_inches="tight")
        return ax

    def _plot_1d(
        self,
        times: float | Sequence[float] | None,
        *,
        ax: Axes | None,
        label: str | None,
        xlabel: str | None,
        ylabel: str | None,
        legend: bool | None,
        **kwargs: Any,
    ) -> Axes:
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, ax = plt.subplots(figsize=(7.5, 4.5))

        coords = x_nodes(self._mesh)
        if times is None:
            requested: list[float] = [self.time]
        elif isinstance(times, (list, tuple, np.ndarray)):
            requested = [_finite_float(value, "times") for value in times]
        else:
            requested = [_finite_float(times, "times")]

        multiple = len(requested) > 1
        for when in requested:
            snapped = self.nearest_time(when)
            curve_label = label if not multiple else f"t = {snapped:g}"
            if multiple and label:
                curve_label = f"{label} (t = {snapped:g})"
            ax.plot(
                coords,
                self.at(when).reshape(-1),
                label=curve_label,
                **kwargs,
            )

        ax.set_xlabel(xlabel or "position")
        ax.set_ylabel(ylabel or "value")
        ax.grid(True, alpha=0.3)
        show_legend = legend if legend is not None else (multiple or bool(label))
        if show_legend:
            ax.legend()
        return ax

    def _plot_2d(
        self,
        when: float,
        *,
        kind: str,
        ax: Axes | None,
        xlabel: str | None,
        ylabel: str | None,
        clabel: str | None,
        levels: int,
        cmap: str,
        colorbar: bool,
        **kwargs: Any,
    ) -> Axes:
        import matplotlib.pyplot as plt

        field = self.at(when)
        X, Y = xy_grid(self._mesh)

        if kind == "surface":
            if ax is None:
                fig = plt.figure(figsize=(8.5, 6.5))
                ax = fig.add_subplot(111, projection="3d")
            drawn = ax.plot_surface(X, Y, field, cmap=cmap, edgecolor="none", **kwargs)
            ax.set_zlabel(clabel or "value")
        else:
            if ax is None:
                _fig, ax = plt.subplots(figsize=(7.0, 5.5))
            if kind == "heatmap":
                coords_x = x_nodes(self._mesh)
                coords_y = y_nodes(self._mesh)
                drawn = ax.imshow(
                    field,
                    origin="lower",
                    aspect="auto",
                    extent=(
                        float(coords_x[0]),
                        float(coords_x[-1]),
                        float(coords_y[0]),
                        float(coords_y[-1]),
                    ),
                    cmap=cmap,
                    **kwargs,
                )
            else:
                drawn = ax.contourf(X, Y, field, levels, cmap=cmap, **kwargs)

        if colorbar:
            ax.figure.colorbar(drawn, ax=ax, label=clabel or "value")
        ax.set_xlabel(xlabel or "x")
        ax.set_ylabel(ylabel or "y")
        return ax

    def animate(
        self,
        *,
        interval: int = 80,
        cmap: str = "viridis",
        title: str | None = None,
        save: str | None = None,
        **kwargs: Any,
    ) -> FuncAnimation:
        """Animate the saved frames.

        Needs more than one saved frame, so solve with ``save_every`` or
        ``save_at`` first.

        Args:
            interval: Milliseconds between frames.
            save: Path to write the animation to. ``.gif`` uses Pillow and
                ``.mp4`` needs ffmpeg on your PATH.

        Returns:
            A Matplotlib ``FuncAnimation``. In a notebook, keep a reference to
            it or it will be garbage collected before it renders.

        Example:
            >>> sol = bt.solve(problem, end_time=0.2, save_every=0.005)
            >>> anim = sol.animate(save="diffusion.gif")
        """

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        if len(self._times) < 2:
            raise ValueError(
                "animate() needs more than one saved frame. Solve with "
                "save_every=... or save_at=[...] to record the evolution."
            )

        stacked = np.vstack(self._fields)
        low = float(np.min(stacked))
        high = float(np.max(stacked))
        if high == low:
            high = low + 1.0
        pad = 0.05 * (high - low)

        if self.is_1d:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            coords = x_nodes(self._mesh)
            (line,) = ax.plot(coords, self._fields[0], **kwargs)
            ax.set_xlim(float(coords[0]), float(coords[-1]))
            ax.set_ylim(low - pad, high + pad)
            ax.set_xlabel("position")
            ax.set_ylabel("value")
            ax.grid(True, alpha=0.3)
            heading = ax.set_title(title or f"t = {self._times[0]:g}")

            def update(index: int) -> Any:
                line.set_ydata(self._fields[index])
                heading.set_text(
                    f"{title + '  ' if title else ''}t = {self._times[index]:g}"
                )
                return line, heading

        else:
            fig, ax = plt.subplots(figsize=(7.0, 5.5))
            coords_x = x_nodes(self._mesh)
            coords_y = y_nodes(self._mesh)
            image = ax.imshow(
                as_2d(self._mesh, self._fields[0]),
                origin="lower",
                aspect="auto",
                extent=(
                    float(coords_x[0]),
                    float(coords_x[-1]),
                    float(coords_y[0]),
                    float(coords_y[-1]),
                ),
                cmap=cmap,
                vmin=low,
                vmax=high,
                **kwargs,
            )
            fig.colorbar(image, ax=ax, label="value")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            heading = ax.set_title(title or f"t = {self._times[0]:g}")

            def update(index: int) -> Any:
                image.set_data(as_2d(self._mesh, self._fields[index]))
                heading.set_text(
                    f"{title + '  ' if title else ''}t = {self._times[index]:g}"
                )
                return image, heading

        animation = FuncAnimation(
            fig,
            update,
            frames=len(self._times),
            interval=int(interval),
            blit=False,
        )
        if save:
            animation.save(save, dpi=120)
        return animation

    # -- explaining itself --------------------------------------------------

    def summary(self) -> str:
        """A readable account of what the solver did and what it certifies.

        Prints the discretization, the step the solver chose and why, the
        stability margin, conservation, and the dimensionless numbers that
        govern the problem. This is the fastest way to find out whether a
        result is trustworthy or whether the grid is too coarse.
        """

        diagnostics = self.diagnostics
        mesh = self._mesh
        lines: list[str] = []

        lines.append("Steady solution summary" if self._steady else "Solution summary")
        lines.append("=" * 60)

        if self.is_1d:
            coords = x_nodes(mesh)
            lines.append(
                f"  grid          {mesh.nx()} cells on "
                f"[{coords[0]:g}, {coords[-1]:g}], dx = {coords[1] - coords[0]:.4g}"
            )
        else:
            coords_x = x_nodes(mesh)
            coords_y = y_nodes(mesh)
            lines.append(
                f"  grid          {mesh.nx()} x {mesh.ny()} cells on "
                f"[{coords_x[0]:g}, {coords_x[-1]:g}] x "
                f"[{coords_y[0]:g}, {coords_y[-1]:g}]"
            )
            lines.append(
                f"                dx = {coords_x[1] - coords_x[0]:.4g}, "
                f"dy = {coords_y[1] - coords_y[0]:.4g}"
            )

        if self._steady:
            newton = self._newton
            lines.append(
                f"  method        Newton on the steady operator, converged in "
                f"{self.steps} iterations"
            )
            if newton is not None:
                lines.append(
                    f"  residual      {float(newton.residual_norm):.3g} (dimensionless) "
                    f"(solved by {newton.linear_solver})"
                )
            lines.append(
                f"  range         {float(np.min(self._fields[-1])):.6g} to "
                f"{float(np.max(self._fields[-1])):.6g}"
            )
            numbers = self.dimensionless()
            if numbers:
                lines.append("")
                lines.append("Dimensionless numbers for this problem")
                lines.append("-" * 60)
                for name, (value, note) in numbers.items():
                    if name.startswith(("Fourier", "sqrt(D t)")):
                        continue  # meaningless once time has been removed
                    lines.append(f"  {name:<14}{value:<12.4g}{note}")
            return "\n".join(lines)

        if diagnostics is None:  # pragma: no cover - only steady solves lack these
            raise RuntimeError("a transient solution must carry native diagnostics")

        lines.append(f"  time          reached t = {self.time:g} in {self.steps} steps")
        used = float(diagnostics.maximum_time_step)
        stable = float(diagnostics.certified_stable_time_step)
        lines.append(
            f"  step size     used dt = {used:.4g}"
            + ("  (chosen automatically)" if diagnostics.automatic_time_step else "")
        )
        if stable > 0.0:
            margin = used / stable
            lines.append(
                f"  stability     certified limit dt = {stable:.4g}; "
                f"this run used {100.0 * margin:.0f}% of it"
            )
        if not diagnostics.reaction_stability_bound_known:
            lines.append(
                "  warning       the reaction had no declared derivative bound, so "
                "its stability is uncertified"
            )

        initial_mass = float(diagnostics.initial_mass)
        change = float(diagnostics.mass_change)
        final_mass = float(diagnostics.final_mass)
        lines.append(
            f"  conservation  total went from {initial_mass:.6g} to {final_mass:.6g}"
        )
        # A percentage is only informative when the starting total is a
        # meaningful share of the final one. Fixed-value boundaries on an empty
        # domain start it near zero, where a relative change is a huge
        # meaningless number rather than a useful check.
        if initial_mass != 0.0 and abs(initial_mass) > 0.01 * abs(final_mass):
            lines.append(
                f"                net change {change:+.4g} "
                f"({100.0 * change / abs(initial_mass):+.3g}%)"
            )
        else:
            lines.append(
                f"                net change {change:+.4g} "
                f"(it started essentially empty, so material entered "
                f"through the boundaries)"
            )
        lines.append(
            f"  range         {float(diagnostics.final_minimum):.6g} to "
            f"{float(diagnostics.final_maximum):.6g} "
            f"(started {float(diagnostics.initial_minimum):.6g} to "
            f"{float(diagnostics.initial_maximum):.6g})"
        )
        if len(self._times) > 1:
            lines.append(
                f"  saved frames  {len(self._times)} between "
                f"t = {self._times[0]:g} and t = {self._times[-1]:g}"
            )

        numbers = self.dimensionless()
        if numbers:
            lines.append("")
            lines.append("Dimensionless numbers for this problem")
            lines.append("-" * 60)
            for name, (value, note) in numbers.items():
                lines.append(f"  {name:<14}{value:<12.4g}{note}")

        return "\n".join(lines)

    def dimensionless(self) -> dict[str, tuple[float, str]]:
        """Dimensionless groups implied by this problem, with a plain reading.

        Returns a mapping from name to ``(value, interpretation)``. Empty when
        the problem is not available -- build problems with
        :class:`biotransport.Problem` to get this.
        """

        problem = self._problem
        recipe = getattr(problem, "_recipe", None)
        if recipe is None:
            return {}
        return recipe.dimensionless_numbers(self._mesh, self.time)

    def __repr__(self) -> str:
        shape = (
            f"{self._mesh.nx()} cells"
            if self.is_1d
            else f"{self._mesh.nx()}x{self._mesh.ny()} cells"
        )
        frames = "" if len(self._times) < 2 else f", {len(self._times)} frames"
        when = "steady state" if self._steady else f"t={self.time:g}"
        return (
            f"<Solution {when} on {shape}{frames}, "
            f"range [{float(np.min(self._fields[-1])):.4g}, "
            f"{float(np.max(self._fields[-1])):.4g}]>"
        )

    def _repr_html_(self) -> str:
        """Render a compact table in Jupyter."""

        diagnostics = self.diagnostics
        geometry = (
            f"{self._mesh.nx()} cells"
            if self.is_1d
            else f"{self._mesh.nx()} &times; {self._mesh.ny()} cells"
        )
        if self._steady:
            rows = [
                ("state", "steady"),
                ("Newton iterations", f"{self.steps}"),
                ("grid", geometry),
            ]
        else:
            rows = [
                ("final time", f"{self.time:g}"),
                ("steps", f"{self.steps:,}"),
                ("grid", geometry),
                ("saved frames", str(len(self._times))),
            ]
        rows.append(
            (
                "value range",
                f"{float(np.min(self._fields[-1])):.4g} &hellip; "
                f"{float(np.max(self._fields[-1])):.4g}",
            )
        )
        if diagnostics is not None:
            rows.append(
                ("net change in total", f"{float(diagnostics.mass_change):+.4g}")
            )
        for name, (value, note) in self.dimensionless().items():
            rows.append((name, f"{value:.4g} <em>&mdash; {note}</em>"))

        body = "".join(
            f"<tr><th style='text-align:left;padding:2px 12px 2px 0;"
            f"font-weight:600'>{name}</th>"
            f"<td style='padding:2px 0'>{value}</td></tr>"
            for name, value in rows
        )
        return (
            "<div style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
            "font-size:0.85em'>"
            "<strong>biotransport.Solution</strong>"
            f"<table style='border-collapse:collapse;margin-top:4px'>{body}</table>"
            "<div style='opacity:0.6;margin-top:4px'>"
            "sol.plot() &middot; sol.summary() &middot; sol.compare(exact)"
            "</div></div>"
        )
