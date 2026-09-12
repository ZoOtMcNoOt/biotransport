"""Fluxes, transfer rates, and the conservation statement behind them.

Transport homework mostly asks for a *rate*, not a field: what is the oxygen flux
into the tissue, how fast does the drug cross the membrane, what fraction of what
was delivered got consumed. The solver computes those fluxes on every face of
every step and then throws them away, so this module reconstructs them from the
returned field.

Reconstructed, not recorded -- but not invented either. Every formula here is the
one the C++ core uses: harmonic mean diffusivity on a face, conservative
first-order upwinding for advection, and half control volumes at the boundaries.

How exact that makes it is worth knowing. Interior face fluxes are exactly the
ones the solver computed. Boundary fluxes are exact wherever the wall states its
own gradient, and exact on a fixed-value wall in 1D, where the boundary cell's
balance has a single unknown. In 2D the only soft spot is the corners: a corner
node owns two physical walls but only one balance, so the split between them is a
convention. The consequence is that a 2D balance closes at second order in the
mesh spacing rather than to roundoff -- :meth:`biotransport.Solution.balance`
reports the residual so you can see which regime you are in instead of trusting
either claim.

Sign convention throughout: the physical transport flux is

    J = -D grad(c) + v c

so a positive ``J`` points along +x (or +y). Boundary quantities are reported as
**outward** -- positive means leaving the domain through that face.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._core import Boundary, BoundaryType

__all__ = ["FluxReport"]


_SIDE_NAMES = {
    "left": Boundary.Left,
    "right": Boundary.Right,
    "bottom": Boundary.Bottom,
    "top": Boundary.Top,
}


def _as_side(side: Boundary | str) -> Boundary:
    if isinstance(side, Boundary):
        return side
    if isinstance(side, str):
        resolved = _SIDE_NAMES.get(side.strip().casefold())
        if resolved is not None:
            return resolved
    raise ValueError(
        f"{side!r} is not a boundary. Use 'left', 'right', 'bottom' or 'top'."
    )


def _harmonic(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Face average matching the core's ``harmonicMean``, zeros included.

    A zero on either side gives zero, which is what makes an impermeable layer
    actually impermeable rather than merely very slow.
    """

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    smaller = np.minimum(first, second)
    larger = np.maximum(first, second)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(larger > 0.0, smaller / larger, 0.0)
        result = smaller / (0.5 + 0.5 * ratio)
    return np.where((first == 0.0) | (second == 0.0), 0.0, result)


def _diffusivity_nodes(problem: Any, mesh: Any) -> np.ndarray:
    """Nodal diffusivity, uniform or per-node."""

    count = int(mesh.num_nodes())
    recipe = getattr(problem, "_recipe", None)
    if recipe is not None and recipe.diffusivity_field is not None:
        return np.asarray(recipe.diffusivity_field, dtype=np.float64).reshape(-1)
    return np.full(count, float(problem.diffusivity()), dtype=np.float64)


def _velocity_nodes(problem: Any, mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    """Nodal ``(vx, vy)``. Zero when no advection was configured."""

    count = int(mesh.num_nodes())
    zero = np.zeros(count, dtype=np.float64)
    recipe = getattr(problem, "_recipe", None)
    if recipe is None:
        return zero, zero.copy()
    if recipe.velocity is not None:
        vx, vy = recipe.velocity
        return np.full(count, vx), np.full(count, vy)
    if recipe.velocity_field is not None:
        field_x, field_y = recipe.velocity_field
        vx = np.asarray(field_x, dtype=np.float64).reshape(-1)
        vy = (
            zero.copy()
            if field_y is None
            else np.asarray(field_y, dtype=np.float64).reshape(-1)
        )
        return vx, vy
    return zero, zero.copy()


def _face_flux(
    left_values: np.ndarray,
    right_values: np.ndarray,
    left_diffusivity: np.ndarray,
    right_diffusivity: np.ndarray,
    left_velocity: np.ndarray,
    right_velocity: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """Physical flux ``J = -D grad(c) + v c`` on a set of faces.

    Mirrors ``xFaceFlux``/``yFaceFlux`` in the core, negated: the core carries the
    quantity ``q = D grad(c) - v c`` internally, and ``J = -q``.
    """

    diffusivity = _harmonic(left_diffusivity, right_diffusivity)
    velocity = 0.5 * (left_velocity + right_velocity)
    upwind = np.where(velocity >= 0.0, left_values, right_values)
    return -diffusivity * (right_values - left_values) / spacing + velocity * upwind


class FluxReport:
    """Where material is going, and whether the books balance.

    Returned by :meth:`biotransport.Solution.balance`. Print it for the readable
    version; the numbers are attributes when you want to assert on them.

    The balance being checked is the integral statement every transport course
    spends weeks on::

        d(stored)/dt = (net inward transfer) + (reaction)

    How it is checked depends on what you have. A steady solution has no
    accumulation, so the instantaneous statement must close exactly, and
    :attr:`residual` is a genuine test of the whole discretization. A transient
    solution is checked in integrated form over the saved frames instead, since
    a single frame cannot tell you ``d(stored)/dt``.

    Attributes:
        stored: Total amount currently in the domain.
        stored_initially: Total at the first saved frame.
        accumulated: ``stored - stored_initially``.
        entered: Net inward transfer rate across all boundaries, at this instant.
        produced: Volume-integrated reaction rate, at this instant.
        by_side: Outward transfer rate through each side, keyed by name. Positive
            means leaving.
        residual: For a steady solution, ``entered + produced`` -- which should be
            zero. ``None`` for a transient one, where :attr:`closure_error` is the
            meaningful check.
        supplied: Time-integrated ``entered + produced`` across the saved frames,
            by the trapezoid rule. ``None`` when fewer than three frames were
            saved.
        closure_error: ``supplied - accumulated``. Two things limit it: the
            trapezoid rule over the saved frames, and the solver's first-order
            time stepping. Refining either shrinks it -- with a small ``time_step``
            and a few dozen frames it reaches roughly 1e-5 relative. ``None`` when
            it could not be computed.
        steady: Whether this came from a steady solve.
        frames: How many saved frames the integrated check used.
    """

    __slots__ = (
        "stored",
        "stored_initially",
        "accumulated",
        "entered",
        "produced",
        "by_side",
        "residual",
        "supplied",
        "closure_error",
        "steady",
        "frames",
    )

    stored: float
    stored_initially: float
    accumulated: float
    entered: float
    produced: float
    by_side: dict[str, float]
    residual: float | None
    supplied: float | None
    closure_error: float | None
    steady: bool
    frames: int

    def __init__(
        self,
        *,
        stored: float,
        stored_initially: float,
        entered: float,
        produced: float,
        by_side: dict[str, float],
        residual: float | None,
        supplied: float | None,
        closure_error: float | None,
        steady: bool,
        frames: int,
    ) -> None:
        self.stored = stored
        self.stored_initially = stored_initially
        self.accumulated = stored - stored_initially
        self.entered = entered
        self.produced = produced
        self.by_side = by_side
        self.residual = residual
        self.supplied = supplied
        self.closure_error = closure_error
        self.steady = steady
        self.frames = frames

    def __repr__(self) -> str:
        check = (
            f"residual={self.residual:.3g}"
            if self.residual is not None
            else f"closure={self.closure_error:.3g}"
            if self.closure_error is not None
            else "no closure check"
        )
        return (
            f"FluxReport(stored={self.stored:.6g}, in={self.entered:.4g}/time, "
            f"produced={self.produced:.4g}/time, {check})"
        )

    def __str__(self) -> str:
        lines = ["Transport balance", "=" * 62]
        lines.append(f"  stored now          {self.stored:.8g}")
        if not self.steady:
            lines.append(f"  stored at the start {self.stored_initially:.8g}")
            lines.append(f"  accumulated         {self.accumulated:+.6g}")
        lines.append("")
        when = "at steady state" if self.steady else "at this instant"
        lines.append(f"  rates {when} (amount per unit time)")
        for name, rate in self.by_side.items():
            if rate == 0.0:
                direction = "sealed"
            else:
                direction = "leaving" if rate > 0 else "entering"
            lines.append(f"    {name:<9} {rate:+.6g}   ({direction})")
        lines.append(f"    {'net in':<9} {self.entered:+.6g}")
        lines.append(f"    {'reaction':<9} {self.produced:+.6g}")
        lines.append("")

        if self.steady:
            total = self.entered + self.produced
            scale = max(abs(self.entered), abs(self.produced), 1.0e-300)
            lines.append("  nothing accumulates at steady state, so these must cancel:")
            lines.append(f"    in + reaction     {total:+.4g}")
            lines.append(
                f"    relative to the flow through it: {abs(total) / scale:.3g}"
            )
        elif self.closure_error is not None and self.supplied is not None:
            scale = max(abs(self.accumulated), abs(self.supplied), 1.0e-300)
            lines.append(f"  integrated over the {self.frames} saved frames:")
            lines.append(f"    supplied          {self.supplied:+.6g}")
            lines.append(f"    accumulated       {self.accumulated:+.6g}")
            lines.append(f"    difference        {self.closure_error:+.4g}")
            lines.append(f"    relative          {abs(self.closure_error) / scale:.3g}")
            lines.append(
                "  two things set that difference: the trapezoid rule over the"
            )
            lines.append("  saved frames, and the solver's first-order time stepping.")
            lines.append("  Refining either shrinks it -- more frames, or a smaller")
            lines.append("  time_step than the stability limit allows.")
        else:
            lines.append(
                "  save at least three frames (save_every=... or frames=...) to"
            )
            lines.append("  check the integrated balance over time.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The computations, driven by Solution
# ---------------------------------------------------------------------------


def interior_flux(problem: Any, mesh: Any, field: np.ndarray):
    """Physical flux on interior faces.

    Returns one array on a 1D mesh (length ``nx``), or ``(Jx, Jy)`` on a 2D mesh
    with shapes ``(ny + 1, nx)`` and ``(ny, nx + 1)``.
    """

    diffusivity = _diffusivity_nodes(problem, mesh)
    vx, vy = _velocity_nodes(problem, mesh)
    values = np.asarray(field, dtype=np.float64).reshape(-1)

    if bool(mesh.is_1d()):
        return _face_flux(
            values[:-1],
            values[1:],
            diffusivity[:-1],
            diffusivity[1:],
            vx[:-1],
            vx[1:],
            float(mesh.dx()),
        )

    shape = (mesh.ny() + 1, mesh.nx() + 1)
    grid = values.reshape(shape)
    d_grid = diffusivity.reshape(shape)
    vx_grid = vx.reshape(shape)
    vy_grid = vy.reshape(shape)

    flux_x = _face_flux(
        grid[:, :-1],
        grid[:, 1:],
        d_grid[:, :-1],
        d_grid[:, 1:],
        vx_grid[:, :-1],
        vx_grid[:, 1:],
        float(mesh.dx()),
    )
    flux_y = _face_flux(
        grid[:-1, :],
        grid[1:, :],
        d_grid[:-1, :],
        d_grid[1:, :],
        vy_grid[:-1, :],
        vy_grid[1:, :],
        float(mesh.dy()),
    )
    return flux_x, flux_y


def outward_flux(
    problem: Any,
    mesh: Any,
    field: np.ndarray,
    side: Boundary | str,
    time: float,
):
    """Outward flux through one physical boundary, per unit area.

    Positive means material is leaving through that side.

    A natural boundary (Neumann or Robin) states its own derivative, so the flux
    there is exact. A fixed-value boundary does not, so its flux comes from the
    balance on the boundary half control volume -- which is also exact, because a
    node held at a fixed value has no accumulation term.
    """

    resolved = _as_side(side)
    is_1d = bool(mesh.is_1d())
    if is_1d and resolved in (Boundary.Bottom, Boundary.Top):
        raise ValueError("a 1D mesh has only 'left' and 'right' boundaries")

    values = np.asarray(field, dtype=np.float64).reshape(-1)
    diffusivity = _diffusivity_nodes(problem, mesh)
    vx, vy = _velocity_nodes(problem, mesh)
    condition = problem.boundaries()[int(resolved.value)]

    shape = (mesh.ny() + 1, mesh.nx() + 1) if not is_1d else (values.size,)
    if is_1d:
        selector: Any = 0 if resolved is Boundary.Left else -1
        edge_values = np.atleast_1d(values[selector])
        edge_diffusivity = np.atleast_1d(diffusivity[selector])
        edge_velocity = np.atleast_1d(vx[selector])
    else:
        grid = values.reshape(shape)
        d_grid = diffusivity.reshape(shape)
        v_grid = (vx if resolved in (Boundary.Left, Boundary.Right) else vy).reshape(
            shape
        )
        if resolved is Boundary.Left:
            edge_values, edge_diffusivity, edge_velocity = (
                grid[:, 0],
                d_grid[:, 0],
                v_grid[:, 0],
            )
        elif resolved is Boundary.Right:
            edge_values, edge_diffusivity, edge_velocity = (
                grid[:, -1],
                d_grid[:, -1],
                v_grid[:, -1],
            )
        elif resolved is Boundary.Bottom:
            edge_values, edge_diffusivity, edge_velocity = (
                grid[0, :],
                d_grid[0, :],
                v_grid[0, :],
            )
        else:
            edge_values, edge_diffusivity, edge_velocity = (
                grid[-1, :],
                d_grid[-1, :],
                v_grid[-1, :],
            )

    # Outward normal points along -x / -y on the left and bottom sides.
    normal_sign = -1.0 if resolved in (Boundary.Left, Boundary.Bottom) else 1.0

    # Compare pybind11 enums with ==, never `is`: each attribute access hands back
    # a fresh Python object, so identity comparison is always False and a Neumann
    # wall would silently take the Robin path and divide by b = 0.
    kind = condition.type
    is_neumann = kind == BoundaryType.NEUMANN
    is_natural_robin = kind == BoundaryType.ROBIN and condition.b != 0.0

    if is_neumann or is_natural_robin:
        if is_neumann:
            derivative = np.full(edge_values.shape, float(condition.value))
        else:
            derivative = (
                float(condition.c) - float(condition.a) * edge_values
            ) / float(condition.b)
        result = -edge_diffusivity * derivative + normal_sign * edge_velocity * (
            edge_values
        )
    else:
        result = _dirichlet_wall_flux(
            problem, mesh, values, resolved, normal_sign, time
        )

    return float(result[0]) if is_1d else np.asarray(result, dtype=np.float64)


def _dirichlet_wall_flux(
    problem: Any,
    mesh: Any,
    values: np.ndarray,
    side: Boundary,
    normal_sign: float,
    time: float,
) -> np.ndarray:
    """Wall flux on a fixed-value boundary, from the boundary cell's own balance.

    A fixed-value node cannot accumulate, so everything else in its control-volume
    balance is known and the wall flux is what closes it::

        0 = (net flux in through the faces we know) + R * volume - outward * area

    In 1D that leaves one unknown and the answer is exact. In 2D an edge node also
    exchanges with its neighbours *along* the boundary, and this accounts for that
    -- omitting it is what makes a naive one-sided estimate wrong wherever the
    field varies along the wall.

    A corner node is the one place this cannot be exact: it owns two physical
    walls and one balance, so the two wall fluxes are not separately determined.
    Its residual is split between them in proportion to their face areas, which
    is consistent and converges under refinement, but it is a convention rather
    than a derivation.
    """

    is_1d = bool(mesh.is_1d())
    recipe = getattr(problem, "_recipe", None)
    coords_x, coords_y = _node_coordinates(mesh)
    if recipe is None:
        reaction = np.zeros(values.size, dtype=np.float64)
    else:
        reaction = recipe.reaction_rate(values, coords_x, coords_y, time)

    if is_1d:
        flux = np.asarray(interior_flux(problem, mesh, values))
        node = 0 if side == Boundary.Left else mesh.nx()
        # Curved geometry weights each face by its area and divides by the shell
        # measure. These reduce to 1, 1 and dx/2 on a Cartesian mesh.
        control_volume = getattr(mesh, "control_volume", None)
        if callable(control_volume):
            volume = float(control_volume(node))
            wall_area = float(
                mesh.lower_face_area(node)
                if side == Boundary.Left
                else mesh.upper_face_area(node)
            )
            interior_area = float(
                mesh.upper_face_area(node)
                if side == Boundary.Left
                else mesh.lower_face_area(node)
            )
        else:  # pragma: no cover - meshes without the geometry accessors
            volume, wall_area, interior_area = 0.5 * float(mesh.dx()), 1.0, 1.0

        if side == Boundary.Left:
            inward = -interior_area * flux[0]
            edge_reaction = reaction[0]
        else:
            inward = interior_area * flux[-1]
            edge_reaction = reaction[-1]

        if wall_area == 0.0:
            # The centre of a cylinder or sphere has no area, so no material can
            # cross it. Symmetry, stated as a number.
            return np.atleast_1d(0.0)
        # 0 = inward + R*volume - outward*wall_area
        return np.atleast_1d((inward + volume * edge_reaction) / wall_area)

    rows, columns = mesh.ny() + 1, mesh.nx() + 1
    flux_x, flux_y = interior_flux(problem, mesh, values)
    reaction_grid = reaction.reshape((rows, columns))

    # Face areas and control volumes, written out rather than assumed. On a
    # Cartesian mesh these reduce to the plain widths and heights; on an
    # axisymmetric one they carry the radial metric. Keeping them explicit is
    # what lets one balance serve both.
    radial = np.array(
        [mesh.control_volume(i) for i in range(columns)], dtype=np.float64
    )
    axial = np.array([mesh.axial_height(j) for j in range(rows)], dtype=np.float64)
    lower_area = np.array(
        [mesh.lower_face_area(i) for i in range(columns)], dtype=np.float64
    )
    upper_area = np.array(
        [mesh.upper_face_area(i) for i in range(columns)], dtype=np.float64
    )

    volume = np.outer(axial, radial)  # (rows, columns)

    if side in (Boundary.Left, Boundary.Right):
        column = 0 if side == Boundary.Left else columns - 1
        # Wall is a radial face: area = r_wall * dz.
        wall_area = (
            lower_area[column] if side == Boundary.Left else upper_area[column]
        ) * axial
        # The opposite radial face is the one we know a flux for.
        interior_area = (
            upper_area[column] if side == Boundary.Left else lower_area[column]
        ) * axial
        interior_flux_values = flux_x[:, 0] if side == Boundary.Left else flux_x[:, -1]
        inward = normal_sign * interior_area * interior_flux_values
        # Axial faces of this column carry area equal to the radial measure.
        below = np.concatenate([[0.0], flux_y[:, column]])
        above = np.concatenate([flux_y[:, column], [0.0]])
        inward = inward + (below - above) * radial[column]
        cell_volume = volume[:, column]
        edge_reaction = reaction_grid[:, column]
    else:
        row = 0 if side == Boundary.Bottom else rows - 1
        # Wall is an axial face: area = the radial measure of that column.
        wall_area = radial.copy()
        interior_flux_values = (
            flux_y[0, :] if side == Boundary.Bottom else flux_y[-1, :]
        )
        inward = normal_sign * radial * interior_flux_values
        # Radial faces of this row, each with its own area.
        inner = np.concatenate([[0.0], flux_x[row, :]]) * lower_area * axial[row]
        outer = np.concatenate([flux_x[row, :], [0.0]]) * upper_area * axial[row]
        inward = inward + (inner - outer)
        cell_volume = volume[row, :]
        edge_reaction = reaction_grid[row, :]

    # 0 = inward + R * V - outward * wall_area
    with np.errstate(divide="ignore", invalid="ignore"):
        outward = np.where(
            wall_area > 0.0,
            (inward + cell_volume * edge_reaction)
            / np.where(wall_area > 0.0, wall_area, 1.0),
            0.0,  # a wall of zero area is the axis, which nothing can cross
        )
    return outward


def _select_edge(values: np.ndarray, mesh: Any, side: Boundary) -> np.ndarray:
    if bool(mesh.is_1d()):
        return np.atleast_1d(values[0 if side is Boundary.Left else -1])
    grid = values.reshape((mesh.ny() + 1, mesh.nx() + 1))
    if side is Boundary.Left:
        return grid[:, 0]
    if side is Boundary.Right:
        return grid[:, -1]
    if side is Boundary.Bottom:
        return grid[0, :]
    return grid[-1, :]


def _node_coordinates(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    from .mesh_utils import x_nodes, xy_grid

    if bool(mesh.is_1d()):
        coords = x_nodes(mesh)
        return coords, np.zeros_like(coords)
    grid_x, grid_y = xy_grid(mesh)
    return grid_x.reshape(-1), grid_y.reshape(-1)


def _neighbour_face_flux(
    problem: Any,
    mesh: Any,
    values: np.ndarray,
    diffusivity: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    side: Boundary,
) -> np.ndarray:
    """The interior face flux adjacent to a boundary, along that side."""

    result = interior_flux(problem, mesh, values)
    if bool(mesh.is_1d()):
        flux = np.asarray(result)
        return np.atleast_1d(flux[0] if side is Boundary.Left else flux[-1])

    flux_x, flux_y = result
    if side is Boundary.Left:
        return flux_x[:, 0]
    if side is Boundary.Right:
        return flux_x[:, -1]
    if side is Boundary.Bottom:
        return flux_y[0, :]
    return flux_y[-1, :]


def _edge_weights(mesh: Any, side: Boundary) -> np.ndarray:
    """Control-volume measure along a boundary, summing to that side's area."""

    from .solution import _trapezoid_weights
    from .mesh_utils import x_nodes, y_nodes

    if bool(mesh.is_1d()):
        # A 1D boundary is a point, whose "area" is the geometric face factor:
        # 1 for a slab, r for a cylinder, r^2 for a sphere.
        face_area = getattr(mesh, "lower_face_area", None)
        if callable(face_area):
            node = 0 if side == Boundary.Left else mesh.nx()
            area = (
                mesh.lower_face_area(node)
                if side == Boundary.Left
                else mesh.upper_face_area(node)
            )
            return np.array([float(area)])
        return np.ones(1)
    control_volume = getattr(mesh, "control_volume", None)
    if not callable(control_volume):  # pragma: no cover - older extensions
        if side in (Boundary.Left, Boundary.Right):
            return _trapezoid_weights(y_nodes(mesh))
        return _trapezoid_weights(x_nodes(mesh))

    if side in (Boundary.Left, Boundary.Right):
        # A radial wall: each node owns (wall area factor) x (its axial height).
        # The factor is 1 on a Cartesian mesh and r_wall on an axisymmetric one.
        wall = (
            mesh.lower_face_area(0)
            if side == Boundary.Left
            else mesh.upper_face_area(mesh.nx())
        )
        return float(wall) * np.array(
            [mesh.axial_height(j) for j in range(mesh.ny() + 1)], dtype=np.float64
        )
    # An axial wall: each node owns the radial measure of its column, which is
    # the true annulus area rather than a width.
    return np.array([control_volume(i) for i in range(mesh.nx() + 1)], dtype=np.float64)


def transfer_rate(
    problem: Any, mesh: Any, field: np.ndarray, side: Boundary | str, time: float
) -> float:
    """Total outward transfer through a side, integrated over its area."""

    resolved = _as_side(side)
    flux = outward_flux(problem, mesh, field, resolved, time)
    weights = _edge_weights(mesh, resolved)
    return float(np.sum(np.atleast_1d(flux) * weights))


def total_reaction(
    problem: Any, mesh: Any, field: np.ndarray, weights: np.ndarray, time: float
) -> float:
    """Volume integral of the reaction rate."""

    recipe = getattr(problem, "_recipe", None)
    if recipe is None or not recipe.reactions:
        return 0.0
    values = np.asarray(field, dtype=np.float64).reshape(-1)
    x_nodes, y_nodes = _node_coordinates(mesh)
    rate = recipe.reaction_rate(values, x_nodes, y_nodes, time)
    return float(np.asarray(weights) @ rate)
