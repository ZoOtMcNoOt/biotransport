"""Building a transport problem.

:class:`Problem` is the front door of the library. You hand it a mesh, describe
the physics with a chain of small calls, and pass it to
:func:`biotransport.solve`.

It is a thin Python layer over the C++ ``TransportProblem`` -- every number you
set lands in the same native object, and no arithmetic happens here. What the
layer adds is memory: it remembers *how* you described the problem, which is
what lets the solver save frames with a correct clock, report the dimensionless
groups that govern your setup, and print a description you can check by eye.
Recorded settings change only after native validation succeeds, so a rejected
edit leaves the previous executable model and its description intact.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from numbers import Real
from typing import Any

import numpy as np

from ._core import Boundary, BoundaryCondition, TransportProblem

__all__ = ["Problem"]


_SIDE_NAMES = {
    "left": Boundary.Left,
    "right": Boundary.Right,
    "bottom": Boundary.Bottom,
    "top": Boundary.Top,
    # Words people actually reach for.
    "west": Boundary.Left,
    "east": Boundary.Right,
    "south": Boundary.Bottom,
    "north": Boundary.Top,
    "inlet": Boundary.Left,
    "outlet": Boundary.Right,
}


def _as_side(side: Boundary | str) -> Boundary:
    """Accept ``Boundary.Left`` or the string ``"left"``."""

    if isinstance(side, Boundary):
        return side
    if isinstance(side, str):
        resolved = _SIDE_NAMES.get(side.strip().casefold())
        if resolved is not None:
            return resolved
        raise ValueError(
            f"{side!r} is not a boundary. Use 'left', 'right', 'bottom' or 'top' "
            f"(or bt.Boundary.Left and friends)."
        )
    raise TypeError(
        "a boundary must be a string like 'left' or a bt.Boundary value, "
        f"not {type(side).__name__}"
    )


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class _Reaction:
    """One recorded reaction term, so the stack can be replayed."""

    __slots__ = ("mode", "kind", "args", "function", "bound")

    def __init__(
        self,
        mode: str,
        kind: str,
        *,
        args: tuple[float, ...] = (),
        function: Callable[..., float] | None = None,
        bound: float | None = None,
    ) -> None:
        self.mode = mode  # "replace" or "add"
        self.kind = kind
        self.args = args
        self.function = function
        self.bound = bound

    @property
    def is_custom(self) -> bool:
        return self.function is not None

    def describe(self) -> str:
        if self.kind == "linear_decay":
            return f"first-order decay, k = {self.args[0]:g}"
        if self.kind == "constant_source":
            return f"constant source, S = {self.args[0]:g}"
        if self.kind == "michaelis_menten":
            return (
                f"Michaelis-Menten uptake, Vmax = {self.args[0]:g}, "
                f"Km = {self.args[1]:g}"
            )
        if self.kind == "logistic_growth":
            return f"logistic growth, r = {self.args[0]:g}, K = {self.args[1]:g}"
        name = getattr(self.function, "__name__", "custom")
        if name == "<lambda>":
            name = "custom"
        bound = (
            f", |dR/dc| <= {self.bound:g}"
            if self.bound is not None
            else ", no declared derivative bound"
        )
        return f"{name} reaction{bound}"


class _Recipe:
    """Everything the user said, kept so it can be replayed or explained."""

    def __init__(self) -> None:
        self.diffusivity: float | None = None
        self.diffusivity_field: np.ndarray | None = None
        self.velocity: tuple[float, float] | None = None
        self.velocity_field: tuple[np.ndarray, np.ndarray | None] | None = None
        self.reactions: list[_Reaction] = []
        self.boundaries: dict[Boundary, str] = {}
        self.advection_scheme: Any = None

    # -- replay -------------------------------------------------------------

    @property
    def has_custom_reaction(self) -> bool:
        return any(term.is_custom for term in self.reactions)

    def replay_reactions(self, target: TransportProblem, time_offset: float) -> None:
        """Re-apply the recorded reaction stack, shifting custom clocks.

        A saved-frame run solves in segments, and each native segment starts its
        clock at zero. A reaction written as ``R(c, x, y, t)`` would therefore
        see the wrong ``t`` from the second segment onward, so custom terms are
        re-registered with the elapsed time added back.

        Every call here goes straight to the native methods rather than through
        :class:`Problem`'s recording overrides. Going through them would rewrite
        this recipe while we are reading it -- ``clear_reaction`` in particular
        would empty the list we are about to iterate.
        """

        TransportProblem.clear_reaction(target)
        for term in self.reactions:
            if term.function is not None:
                function = _shift_clock(term.function, time_offset)
                native = (
                    TransportProblem.reaction
                    if term.mode == "replace"
                    else TransportProblem.add_reaction
                )
                if term.bound is None:
                    native(target, function)
                else:
                    native(target, function, term.bound)
                continue

            method_name = term.kind if term.mode == "replace" else f"add_{term.kind}"
            getattr(TransportProblem, method_name)(target, *term.args)

    def reaction_rate(
        self,
        values: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        time: float,
    ) -> np.ndarray:
        """Evaluate the configured reaction ``R(c, x, y, t)`` at every node.

        Replays the recorded stack in order, so a ``replace`` call resets the sum
        exactly as it does inside the solver.
        """

        field = np.asarray(values, dtype=np.float64)
        total = np.zeros_like(field)
        for term in self.reactions:
            if term.mode == "replace":
                total = np.zeros_like(field)
            if term.kind == "linear_decay":
                total = total - term.args[0] * field
            elif term.kind == "constant_source":
                total = total + term.args[0]
            elif term.kind == "michaelis_menten":
                vmax, km = term.args
                # Match the native nonnegative uptake law, including c <= 0.
                # Scale the ratio to avoid a pole at -Km or overflow in Km+c.
                positive = np.maximum(field, 0.0)
                scale = np.maximum(positive, km)
                relative = positive / scale
                total = total - vmax * (relative / (km / scale + relative))
            elif term.kind == "logistic_growth":
                rate, capacity = term.args
                total = total + rate * field * (1.0 - field / capacity)
            elif term.function is not None:
                total = total + np.array(
                    [
                        float(term.function(value, xi, yi, time))
                        for value, xi, yi in zip(field, x, y)
                    ],
                    dtype=np.float64,
                )
        return total

    # -- explanation --------------------------------------------------------

    def characteristic_length(self, mesh: Any) -> float:
        """Domain extent used as the length scale for dimensionless groups."""

        span_x = float(mesh.x(mesh.nx()) - mesh.x(0))
        if bool(mesh.is_1d()):
            return span_x
        span_y = float(mesh.y(0, mesh.ny()) - mesh.y(0, 0))
        return max(span_x, span_y)

    def _speed(self) -> float | None:
        if self.velocity is not None:
            vx, vy = self.velocity
            return float(math.hypot(vx, vy))
        if self.velocity_field is not None:
            field_x, field_y = self.velocity_field
            magnitude = np.abs(np.asarray(field_x, dtype=np.float64))
            if field_y is not None:
                magnitude = np.hypot(magnitude, np.asarray(field_y, dtype=np.float64))
            return float(np.max(magnitude))
        return None

    def _diffusivity_scale(self) -> float | None:
        if self.diffusivity is not None:
            return float(self.diffusivity)
        if self.diffusivity_field is not None:
            values = np.asarray(self.diffusivity_field, dtype=np.float64)
            positive = values[values > 0.0]
            return float(np.mean(positive)) if positive.size else None
        return None

    def _decay_rate(self) -> float | None:
        total = 0.0
        found = False
        for term in self.reactions:
            if term.kind == "linear_decay":
                if term.mode == "replace":
                    total = float(term.args[0])
                else:
                    total += float(term.args[0])
                found = True
            elif term.kind == "michaelis_menten":
                rate = float(term.args[0]) / float(term.args[1])
                total = rate if term.mode == "replace" else total + rate
                found = True
        return total if found else None

    def dimensionless_numbers(
        self, mesh: Any, time: float
    ) -> dict[str, tuple[float, str]]:
        """Dimensionless groups for this problem, each with a plain reading."""

        numbers: dict[str, tuple[float, str]] = {}
        length = self.characteristic_length(mesh)
        diffusivity = self._diffusivity_scale()
        speed = self._speed()
        decay = self._decay_rate()
        dx = float(mesh.x(1) - mesh.x(0))

        if diffusivity and diffusivity > 0.0 and length > 0.0:
            fourier = diffusivity * time / (length * length)
            numbers["Fourier"] = (
                fourier,
                _read_fourier(fourier),
            )
            depth = math.sqrt(diffusivity * time) if time > 0.0 else 0.0
            numbers["sqrt(D t) / L"] = (
                depth / length,
                f"diffusion has spread about {depth:.3g} into a domain of {length:.3g}",
            )

        if speed and speed > 0.0 and diffusivity and diffusivity > 0.0:
            peclet = speed * length / diffusivity
            numbers["Peclet"] = (peclet, _read_peclet(peclet))
            grid_peclet = speed * dx / diffusivity
            numbers["grid Peclet"] = (grid_peclet, _read_grid_peclet(grid_peclet))
        elif speed and speed > 0.0:
            numbers["Peclet"] = (
                float("inf"),
                "pure advection: there is no diffusion to compete with",
            )

        if decay and decay > 0.0 and diffusivity and diffusivity > 0.0 and length > 0.0:
            damkohler = decay * length * length / diffusivity
            numbers["Damkohler"] = (damkohler, _read_damkohler(damkohler))
            penetration = math.sqrt(diffusivity / decay)
            numbers["sqrt(D/k) / L"] = (
                penetration / length,
                f"reaction consumes the solute within about {penetration:.3g} "
                f"of the source",
            )

        return numbers

    def describe(self, mesh: Any) -> list[str]:
        """Human-readable lines describing the configured physics."""

        lines: list[str] = []
        if self.diffusivity is not None:
            lines.append(f"  diffusion     D = {self.diffusivity:g} (uniform)")
        elif self.diffusivity_field is not None:
            values = np.asarray(self.diffusivity_field)
            lines.append(
                f"  diffusion     D varies from {float(values.min()):g} "
                f"to {float(values.max()):g}"
            )
        else:
            lines.append("  diffusion     not set (D = 0)")

        if self.velocity is not None:
            vx, vy = self.velocity
            if bool(mesh.is_1d()):
                lines.append(f"  advection     v = {vx:g}")
            else:
                lines.append(f"  advection     v = ({vx:g}, {vy:g})")
        elif self.velocity_field is not None:
            field_x, _field_y = self.velocity_field
            values = np.asarray(field_x)
            lines.append(
                f"  advection     v varies from {float(values.min()):g} "
                f"to {float(values.max()):g}"
            )

        if self.reactions:
            for index, term in enumerate(self.reactions):
                prefix = "  reaction    " if index == 0 else "              "
                joiner = "" if index == 0 else "+ "
                lines.append(f"{prefix}  {joiner}{term.describe()}")

        if self.boundaries:
            sides = ", ".join(
                f"{side.name.lower()}: {text}"
                for side, text in sorted(
                    self.boundaries.items(), key=lambda item: item[0].name
                )
            )
            lines.append(f"  boundaries    {sides}")
        unset = [
            side.name.lower()
            for side in _relevant_sides(mesh)
            if side not in self.boundaries
        ]
        if unset:
            lines.append(
                f"                {', '.join(unset)}: default zero-gradient "
                f"(no diffusive flux)"
            )
        return lines


def _relevant_sides(mesh: Any) -> tuple[Boundary, ...]:
    if bool(mesh.is_1d()):
        return (Boundary.Left, Boundary.Right)
    return (Boundary.Left, Boundary.Right, Boundary.Bottom, Boundary.Top)


def _shift_clock(
    function: Callable[..., float], offset: float
) -> Callable[[float, float, float, float], float]:
    """Wrap ``R(c, x, y, t)`` so it sees absolute rather than segment-local time."""

    if offset == 0.0:
        return function

    def shifted(c: float, x: float, y: float, t: float) -> float:
        return function(c, x, y, t + offset)

    return shifted


def _read_fourier(value: float) -> str:
    # Deliberately a statement about diffusion only. A large Fourier number means
    # diffusion has had time to cross the domain -- it does NOT mean the problem
    # has reached steady state, because a reaction or a moving boundary can keep
    # it evolving long after that. Saying "equilibrated" here would be wrong for
    # every reaction-limited case.
    if value < 0.01:
        return "diffusion has barely started; the profile is still close to t = 0"
    if value < 0.1:
        return "early: diffusion has only reached a fraction of the domain"
    if value < 1.0:
        return "diffusion is working its way across the domain"
    return "diffusion has had time to cross the domain (check reaction terms too)"


def _read_peclet(value: float) -> str:
    if value < 0.1:
        return "diffusion dominates; advection is nearly irrelevant"
    if value < 10.0:
        return "advection and diffusion are comparable"
    if value < 1000.0:
        return "advection dominates; expect a sharp travelling front"
    return "advection overwhelms diffusion; the profile is nearly transported intact"


def _read_grid_peclet(value: float) -> str:
    if value <= 2.0:
        return "the grid resolves the front (<= 2 is comfortable)"
    return (
        f"above 2, so upwinding adds noticeable numerical diffusion; "
        f"refine the grid by about {value / 2.0:.0f}x to sharpen the front"
    )


def _read_damkohler(value: float) -> str:
    if value < 0.1:
        return "diffusion is much faster than reaction; the field is nearly uniform"
    if value < 10.0:
        return "reaction and diffusion are balanced; expect a curved profile"
    reach = 100.0 / math.sqrt(value)
    if value < 1000.0:
        return (
            f"reaction outpaces diffusion; solute reaches only about "
            f"{reach:.0f}% of the way across"
        )
    return (
        f"reaction dominates; solute is confined to a thin layer about "
        f"{reach:.2g}% of the domain"
    )


class Problem(TransportProblem):
    """A scalar transport problem: diffusion, advection and reaction on a mesh.

    The equation being solved is

    .. math::

        \\frac{\\partial c}{\\partial t}
        = \\nabla\\cdot(D\\nabla c) - \\nabla\\cdot(\\mathbf{v}c) + R(c, x, t)

    Every setter returns the problem, so you can chain them::

        problem = (
            bt.Problem(mesh)
            .diffusivity(1e-9)
            .velocity(1e-4)
            .linear_decay(0.5)
            .initial(bt.gaussian(mesh, center=0.3, width=0.05))
            .dirichlet("left", 1.0)
            .neumann("right", 0.0)
        )

    Units are yours to keep consistent -- the solver works in whatever system
    you feed it. If you would rather have that checked, build your numbers with
    :mod:`biotransport.units`.

    Args:
        mesh: A mesh from :func:`biotransport.mesh_1d` or
            :func:`biotransport.mesh_2d`.
    """

    # Some setters below are also getters, which means this class shadows a few
    # of the native read accessors. Every one of them delegates to the native
    # method, so the C++ object remains the single source of truth for solver
    # state -- this class only adds a record of how that state was described.
    # Components that verify they are reading native values look for this flag.
    _NATIVE_STATE_IS_AUTHORITATIVE = True

    _recipe: _Recipe

    def __init__(self, mesh: Any) -> None:
        super().__init__(mesh)
        object.__setattr__(self, "_recipe", _Recipe())

    # -- transport terms ----------------------------------------------------

    def diffusivity(self, value: Any = None) -> Any:
        """Set a uniform diffusivity, or read the current one back.

        Called with a number this sets ``D``; called with nothing it returns the
        uniform value the problem is currently using.
        """

        if value is None:
            return super().diffusivity()
        if np.ndim(value) != 0:
            return self.diffusivity_field(value)
        amount = _finite(value, "diffusivity")
        if amount < 0.0:
            raise ValueError(
                f"diffusivity must be non-negative, got {amount:g}. A negative D "
                f"would make the problem ill-posed (heat flowing up its own gradient)."
            )
        super().diffusivity(amount)
        self._recipe.diffusivity = amount
        self._recipe.diffusivity_field = None
        return self

    def diffusivity_field(self, values: Any) -> Problem:
        """Set a node-by-node diffusivity for a heterogeneous medium."""

        # Native setters copy their fields. Retain an independent Python copy as
        # well, so later edits to caller-owned arrays cannot rewrite the recipe.
        array = np.array(values, dtype=np.float64, copy=True).reshape(-1)
        super().diffusivity_field(array)
        self._recipe.diffusivity_field = array
        self._recipe.diffusivity = None
        return self

    def velocity(self, vx: float, vy: float = 0.0) -> Problem:
        """Set a uniform velocity. ``vy`` must be zero on a 1D mesh."""

        speed_x = _finite(vx, "vx")
        speed_y = _finite(vy, "vy")
        super().velocity(speed_x, speed_y)
        self._recipe.velocity = (speed_x, speed_y)
        self._recipe.velocity_field = None
        return self

    def velocity_field(self, vx: Any, vy: Any = None) -> Problem:
        """Set a node-by-node velocity field.

        Pass ``vy`` as well on a 2D mesh; omit it on a 1D one.
        """

        array_x = np.array(vx, dtype=np.float64, copy=True).reshape(-1)
        array_y = (
            None
            if vy is None
            else np.array(vy, dtype=np.float64, copy=True).reshape(-1)
        )
        if array_y is None:
            super().velocity_field(array_x)
        else:
            super().velocity_field(array_x, array_y)
        self._recipe.velocity_field = (array_x, array_y)
        self._recipe.velocity = None
        return self

    def advection_scheme(self, scheme: Any) -> Problem:
        """Choose the advection discretization. Only upwinding is verified."""

        super().advection_scheme(scheme)
        self._recipe.advection_scheme = scheme
        return self

    # -- reactions ----------------------------------------------------------

    def linear_decay(self, k: float) -> Problem:
        """Replace the reaction with first-order decay, ``R = -k c``."""

        rate = _finite(k, "k")
        super().linear_decay(rate)
        self._recipe.reactions = [_Reaction("replace", "linear_decay", args=(rate,))]
        return self

    def add_linear_decay(self, k: float) -> Problem:
        """Add first-order decay to whatever reaction is already configured."""

        rate = _finite(k, "k")
        super().add_linear_decay(rate)
        self._recipe.reactions.append(_Reaction("add", "linear_decay", args=(rate,)))
        return self

    def constant_source(self, S: float) -> Problem:
        """Replace the reaction with a constant source, ``R = S``."""

        rate = _finite(S, "S")
        super().constant_source(rate)
        self._recipe.reactions = [_Reaction("replace", "constant_source", args=(rate,))]
        return self

    def add_constant_source(self, S: float) -> Problem:
        """Add a constant source to the existing reaction."""

        rate = _finite(S, "S")
        super().add_constant_source(rate)
        self._recipe.reactions.append(_Reaction("add", "constant_source", args=(rate,)))
        return self

    def michaelis_menten(self, Vmax: float, Km: float) -> Problem:
        """Replace the reaction with saturable uptake, ``R = -Vmax c / (Km + c)``.

        This is the workhorse for oxygen or nutrient consumption by tissue.
        """

        vmax = _finite(Vmax, "Vmax")
        km = _finite(Km, "Km")
        if km <= 0.0:
            raise ValueError("Km must be positive")
        super().michaelis_menten(vmax, km)
        self._recipe.reactions = [
            _Reaction("replace", "michaelis_menten", args=(vmax, km))
        ]
        return self

    def add_michaelis_menten(self, Vmax: float, Km: float) -> Problem:
        """Add saturable uptake to the existing reaction."""

        vmax = _finite(Vmax, "Vmax")
        km = _finite(Km, "Km")
        if km <= 0.0:
            raise ValueError("Km must be positive")
        super().add_michaelis_menten(vmax, km)
        self._recipe.reactions.append(
            _Reaction("add", "michaelis_menten", args=(vmax, km))
        )
        return self

    def logistic_growth(self, r: float, K: float) -> Problem:
        """Replace the reaction with logistic growth, ``R = r c (1 - c/K)``."""

        rate = _finite(r, "r")
        capacity = _finite(K, "K")
        super().logistic_growth(rate, capacity)
        self._recipe.reactions = [
            _Reaction("replace", "logistic_growth", args=(rate, capacity))
        ]
        return self

    def add_logistic_growth(self, r: float, K: float) -> Problem:
        """Add logistic growth to the existing reaction."""

        rate = _finite(r, "r")
        capacity = _finite(K, "K")
        super().add_logistic_growth(rate, capacity)
        self._recipe.reactions.append(
            _Reaction("add", "logistic_growth", args=(rate, capacity))
        )
        return self

    def reaction(
        self,
        function: Callable[..., float],
        max_abs_dc: float | None = None,
    ) -> Problem:
        """Replace the reaction with your own ``R(c, x, y, t)``.

        The solver picks its own time step from the fastest process in the
        problem, and it cannot differentiate your function to find out how fast
        the reaction is. Declare ``max_abs_dc`` -- an upper bound on
        ``|dR/dc|`` over the concentrations you expect -- and automatic stepping
        keeps working. Without it you must choose ``time_step`` yourself.

        Args:
            function: Called as ``function(c, x, y, t)``. On a 1D mesh ``y`` is
                zero.
            max_abs_dc: Upper bound on ``|dR/dc|``, in units of 1/time.
        """

        if not callable(function):
            raise TypeError("a reaction must be callable as function(c, x, y, t)")
        bound = None if max_abs_dc is None else _finite(max_abs_dc, "max_abs_dc")
        if bound is not None and bound < 0.0:
            raise ValueError("max_abs_dc must be non-negative")
        if bound is None:
            super().reaction(function)
        else:
            super().reaction(function, bound)
        self._recipe.reactions = [
            _Reaction("replace", "custom", function=function, bound=bound)
        ]
        return self

    def add_reaction(
        self,
        function: Callable[..., float],
        max_abs_dc: float | None = None,
    ) -> Problem:
        """Add your own ``R(c, x, y, t)`` to the existing reaction."""

        if not callable(function):
            raise TypeError("a reaction must be callable as function(c, x, y, t)")
        bound = None if max_abs_dc is None else _finite(max_abs_dc, "max_abs_dc")
        if bound is not None and bound < 0.0:
            raise ValueError("max_abs_dc must be non-negative")
        if bound is None:
            super().add_reaction(function)
        else:
            super().add_reaction(function, bound)
        self._recipe.reactions.append(
            _Reaction("add", "custom", function=function, bound=bound)
        )
        return self

    def clear_reaction(self) -> Problem:
        """Remove every reaction term."""

        super().clear_reaction()
        self._recipe.reactions = []
        return self

    # -- initial condition --------------------------------------------------

    def initial(self, values: Any = None) -> Any:
        """Set the starting field, or read the current one back.

        Accepts a single number for a uniform start, or one value per mesh node.
        Called with no argument it returns the current initial condition, which
        is what the native ``initial()`` getter does.
        """

        if values is None:
            return super().initial()
        return self.initial_condition(values)

    def initial_condition(self, values: Any) -> Problem:
        """Set the starting field from a number or one value per node."""

        if np.ndim(values) == 0:
            super().initial_condition(_finite(values, "initial_condition"))
            return self

        array = np.asarray(values, dtype=np.float64).reshape(-1)
        expected = int(self.mesh().num_nodes())
        if array.size != expected:
            mesh = self.mesh()
            if bool(mesh.is_1d()):
                hint = f"a 1D mesh with {mesh.nx()} cells has {expected} nodes"
            else:
                hint = (
                    f"a {mesh.nx()}x{mesh.ny()} mesh has "
                    f"({mesh.ny()} + 1) x ({mesh.nx()} + 1) = {expected} nodes"
                )
            raise ValueError(
                f"the initial condition has {array.size} values but the mesh needs "
                f"{expected}: {hint}. Remember a mesh of n cells has n + 1 nodes."
            )
        if not np.all(np.isfinite(array)):
            bad = int(np.argmax(~np.isfinite(array)))
            raise ValueError(
                f"the initial condition must be finite everywhere, but node {bad} "
                f"is {array[bad]}"
            )
        super().initial_condition(array)
        return self

    # -- boundaries ---------------------------------------------------------

    def dirichlet(self, side: Boundary | str, value: float) -> Problem:
        """Hold a side at a fixed value, ``c = value``."""

        resolved = _as_side(side)
        amount = _finite(value, "value")
        super().dirichlet(resolved, amount)
        self._recipe.boundaries[resolved] = f"held at {amount:g}"
        return self

    def neumann(self, side: Boundary | str, gradient: float) -> Problem:
        """Fix the outward normal derivative on a side, ``dc/dn = gradient``.

        This is a *derivative*, not a flux. The outward diffusive flux that
        results is ``-D * gradient``, so ``gradient = 0`` means no diffusive
        flux -- a sealed wall as far as diffusion is concerned. If the velocity
        points through that side, advection still carries material across it.
        """

        resolved = _as_side(side)
        slope = _finite(gradient, "gradient")
        super().neumann(resolved, slope)
        self._recipe.boundaries[resolved] = (
            "sealed (no diffusive flux)" if slope == 0.0 else f"dc/dn = {slope:g}"
        )
        return self

    def sealed(self, side: Boundary | str) -> Problem:
        """Shorthand for a zero-gradient wall, ``dc/dn = 0``."""

        return self.neumann(side, 0.0)

    # The native signature names the right-hand side `c`, which collides with the
    # concentration in the very equation it appears in. Renaming it to `rhs` is a
    # deliberate, documented divergence; `c` still works as a keyword, so every
    # existing call keeps running.
    def robin(  # type: ignore[override]
        self,
        side: Boundary | str,
        a: float,
        b: float,
        rhs: float | None = None,
        *,
        c: float | None = None,
    ) -> Problem:
        """Impose ``a*c + b*dc/dn = rhs`` on a side.

        For convective exchange with a bath held at ``c_inf`` through a film
        coefficient ``h``, that is ``a = h``, ``b = D``, ``rhs = h * c_inf``.
        Setting ``b = 0`` reduces this to a fixed value.

        Args:
            side: ``"left"``, ``"right"``, ``"bottom"``, ``"top"``, or a
                :class:`Boundary` value.
            a: Coefficient on the concentration.
            b: Coefficient on the outward normal derivative.
            rhs: The right-hand side of the relation.
            c: Deprecated spelling of ``rhs``. The native binding names this
                argument ``c``, which collides with the concentration the
                equation is about, so ``rhs`` is preferred.
        """

        if rhs is None and c is None:
            raise TypeError("robin() needs a right-hand side: pass rhs=")
        if rhs is not None and c is not None:
            raise TypeError("pass either rhs or c, not both")
        resolved = _as_side(side)
        coefficient_a = _finite(a, "a")
        coefficient_b = _finite(b, "b")
        right_hand_side = _finite(rhs if rhs is not None else c, "rhs")
        super().robin(resolved, coefficient_a, coefficient_b, right_hand_side)
        self._recipe.boundaries[resolved] = (
            f"{coefficient_a:g}*c + {coefficient_b:g}*dc/dn = {right_hand_side:g}"
        )
        return self

    def boundary(self, side: Boundary | str, bc: BoundaryCondition) -> Problem:
        """Apply a prebuilt :class:`BoundaryCondition` to a side."""

        resolved = _as_side(side)
        super().boundary(resolved, bc)
        self._recipe.boundaries[resolved] = str(bc.type)
        return self

    # -- explaining itself --------------------------------------------------

    def describe(self) -> str:
        """A readable account of the problem you have built.

        Worth printing before a long run -- it is the quickest way to catch a
        boundary you forgot or a term that got overwritten.
        """

        mesh = self.mesh()
        lines = ["Transport problem", "=" * 60]
        if bool(mesh.is_1d()):
            shape = ""
            is_radial = getattr(mesh, "is_radial", None)
            if callable(is_radial) and is_radial():
                name = str(mesh.geometry()).rsplit(".", 1)[-1].lower()
                shape = f", {name} (radius r)"
            lines.append(
                f"  mesh          {mesh.nx()} cells on "
                f"[{mesh.x(0):g}, {mesh.x(mesh.nx()):g}] "
                f"({mesh.num_nodes()} nodes){shape}"
            )
        else:
            is_radial = getattr(mesh, "is_radial", None)
            axes = "r x z, axisymmetric" if callable(is_radial) and is_radial() else ""
            lines.append(
                f"  mesh          {mesh.nx()} x {mesh.ny()} cells on "
                f"[{mesh.x(0):g}, {mesh.x(mesh.nx()):g}] x "
                f"[{mesh.y(0, 0):g}, {mesh.y(0, mesh.ny()):g}] "
                f"({mesh.num_nodes()} nodes)" + (f", {axes}" if axes else "")
            )
        lines.extend(self._recipe.describe(mesh))
        return "\n".join(lines)

    def stable_time_step(self, safety_factor: float = 1.0) -> float:
        """Return the certified explicit stability ceiling, scaled by a fraction.

        This is a stability bound, not the selected integration step. Reaction
        accuracy guards and saved intervals can require shorter steps. Use
        :func:`biotransport.plan_transport` for the actual native schedule, or
        :meth:`biotransport.Experiment.plan` for a run with saved frames.

        Args:
            safety_factor: Fraction of the certified limit to report. The default
                of 1.0 gives the certified limit itself.

        Returns:
            The step size, in your time units.
        """

        from ._core import SolveOptions, solve_transport

        fraction = _finite(safety_factor, "safety_factor")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("safety_factor must be in (0, 1]")

        # A zero-length solve costs nothing and still populates the diagnostics.
        # The certified limit it reports is the raw stability ceiling, so the
        # safety factor is applied here rather than passed in -- passing it would
        # be silently ignored.
        probe = SolveOptions()
        probe.final_time = 0.0
        certified = float(
            solve_transport(self, probe).diagnostics.certified_stable_time_step
        )
        return certified * fraction

    def __repr__(self) -> str:
        mesh = self.mesh()
        geometry = (
            f"{mesh.nx()} cells"
            if bool(mesh.is_1d())
            else f"{mesh.nx()}x{mesh.ny()} cells"
        )
        terms = []
        if self._recipe.diffusivity or self._recipe.diffusivity_field is not None:
            terms.append("diffusion")
        if self.has_advection():
            terms.append("advection")
        if self.has_reaction():
            terms.append("reaction")
        return f"<Problem on {geometry}: {' + '.join(terms) or 'nothing configured'}>"

    def _repr_html_(self) -> str:
        return (
            "<pre style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
            f"font-size:0.85em;line-height:1.45'>{self.describe()}</pre>"
        )
