"""Visualization tools for biotransport simulations.

One function, :func:`plot`, renders any field the library produces:

- pass a :class:`~biotransport.Result` (or any result that carries its mesh)
  on its own, or ``(mesh, values)`` for raw arrays and native results;
- 1D meshes become a line plot, 2D meshes a filled contour (``kind="contour"``)
  or a 3D surface (``kind="surface"``);
- the Matplotlib figure is returned so callers can add to it, and ``save_to=``
  writes it to disk in the same call.

Matplotlib is imported lazily, so importing :mod:`biotransport` never pays
for a plotting backend unless a plot is actually drawn.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import ArrayLike

from ._deprecation import deprecated_callable, warn_deprecated
from .mesh_utils import as_1d, as_2d, x_nodes, xy_grid
from .utils import get_result_path

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D

    from ._core import StructuredMesh


def _pyplot():
    """Import ``matplotlib.pyplot`` on first use."""
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------------------
# Private renderers
# ---------------------------------------------------------------------------


def _line(
    mesh: StructuredMesh,
    values: ArrayLike,
    *,
    title: str | None = None,
    xlabel: str = "Position",
    ylabel: str = "Value",
    ax: Axes | None = None,
) -> Figure:
    if not mesh.is_1d():
        raise ValueError("Mesh must be 1D for 1D plotting")

    x = x_nodes(mesh)
    y = as_1d(mesh, values)

    if ax is None:
        fig, ax = _pyplot().subplots(figsize=(10, 6))
    else:
        fig = cast("Figure", ax.figure)

    ax.plot(x, y, "b-")
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    return fig


def _contour(
    mesh: StructuredMesh,
    values: ArrayLike,
    *,
    title: str | None = None,
    colorbar_label: str = "Value",
    xlabel: str = "X",
    ylabel: str = "Y",
    ax: Axes | None = None,
) -> Figure:
    if mesh.is_1d():
        raise ValueError("Mesh must be 2D for 2D plotting")

    X, Y = xy_grid(mesh)
    Z = as_2d(mesh, values)

    if ax is None:
        fig, ax = _pyplot().subplots(figsize=(10, 8))
    else:
        fig = cast("Figure", ax.figure)

    contour = ax.contourf(X, Y, Z, 50, cmap="viridis")
    fig.colorbar(contour, ax=ax, label=colorbar_label)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def _surface(
    mesh: StructuredMesh,
    values: ArrayLike,
    *,
    title: str | None = None,
    zlabel: str = "Value",
    xlabel: str = "X",
    ylabel: str = "Y",
    ax: Axes3D | None = None,
) -> Figure:
    if mesh.is_1d():
        raise ValueError("Mesh must be 2D for 3D surface plotting")

    X, Y = xy_grid(mesh)
    Z = as_2d(mesh, values)

    if ax is None:
        fig = _pyplot().figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = cast("Figure", ax.figure)

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    return fig


def _field_values(source):
    """Extract plottable values from a result-like object or return it unchanged."""
    if hasattr(source, "field") and not isinstance(source, np.ndarray):
        return source.field
    if hasattr(source, "concentration"):
        candidate = source.concentration
        return candidate() if callable(candidate) else candidate
    if hasattr(source, "solution"):
        candidate = source.solution
        return candidate() if callable(candidate) else candidate
    return source


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot(
    mesh_or_result,
    values=None,
    *,
    title: str | None = None,
    kind: str = "auto",
    show: bool = False,
    save_to: str | os.PathLike[str] | None = None,
    **kwargs,
):
    """Plot a field on its mesh and return the Matplotlib figure.

    Detects 1D versus 2D from the mesh and chooses the plot type. Accepts a
    :class:`~biotransport.Result` (or any result that carries its mesh) on its
    own, or ``(mesh, values)`` where ``values`` is an array or an object
    exposing ``field``, ``concentration`` or ``solution`` data.

    Args:
        mesh_or_result: A result carrying its mesh, or a structured mesh
        values: Field values or a result object (when a mesh was given)
        title: Plot title (optional)
        kind: ``'auto'`` (default), ``'line'`` (1D), ``'contour'`` or
            ``'surface'`` (2D)
        show: Whether to call ``plt.show()`` (default False; the figure is
            returned so callers can add to it first)
        save_to: Optional path; when given the figure is saved there at
            150 dpi with tight bounding box before returning
        **kwargs: Forwarded to the renderer: ``ax``, ``xlabel``, ``ylabel``
            for every kind; ``colorbar_label`` for contours; ``zlabel`` for
            surfaces

    Returns:
        Matplotlib figure

    Examples:
        >>> result = bt.solve(problem, end_time=0.1)
        >>> bt.plot(result)                       # or result.plot()
        >>> bt.plot(mesh, values, kind="surface", save_to="field.png")
    """
    if "solution" in kwargs:
        warn_deprecated(
            "plot(solution=...)",
            "plot(mesh, values)",
            reason="the second argument is named values",
        )
        if values is not None:
            raise TypeError("pass the field once, as values")
        values = kwargs.pop("solution")
    if not isinstance(kind, str):
        raise TypeError("kind must be a string")
    normalized_kind = kind.casefold()
    if not isinstance(show, bool):
        raise TypeError("show must be a boolean")
    if values is None:
        mesh = getattr(mesh_or_result, "mesh", None)
        if mesh is not None and (
            hasattr(mesh_or_result, "field") or hasattr(mesh_or_result, "concentration")
        ):
            source = mesh_or_result
        elif hasattr(mesh_or_result, "concentration") or hasattr(
            mesh_or_result, "solution"
        ):
            raise ValueError(
                "This result does not carry its mesh. "
                "Pass both objects: bt.plot(mesh, result)."
            )
        else:
            raise TypeError("field values or a solver result are required")
    else:
        mesh = mesh_or_result
        source = values
    if not hasattr(mesh, "is_1d") or not callable(mesh.is_1d):
        raise TypeError(
            "mesh_or_result must be a structured mesh or a result that carries one"
        )

    field = _field_values(source)
    if bool(mesh.is_1d()):
        if normalized_kind not in {"auto", "line"}:
            raise ValueError("a 1D mesh supports kind='auto' or kind='line'")
        fig = _line(mesh, field, title=title, **kwargs)
    else:
        if normalized_kind not in {"auto", "contour", "surface"}:
            raise ValueError(
                "a 2D mesh supports kind='auto', kind='contour', or kind='surface'"
            )
        if normalized_kind == "surface":
            fig = _surface(mesh, field, title=title, **kwargs)
        else:
            fig = _contour(mesh, field, title=title, **kwargs)

    if save_to is not None:
        fig.savefig(os.fspath(save_to), dpi=150, bbox_inches="tight")
    if show:
        _pyplot().show()
    return fig


# ---------------------------------------------------------------------------
# Deprecated spellings (removed in 0.4.0)
# ---------------------------------------------------------------------------

_ONE_PLOT = "one plot() function detects the mesh dimension and plot kind"


@deprecated_callable(
    "bt.plot(mesh, values)", reason=_ONE_PLOT, name="biotransport.plot_1d_solution"
)
def plot_1d_solution(
    mesh: StructuredMesh,
    solution: ArrayLike,
    title: str | None = None,
    xlabel: str = "Position",
    ylabel: str = "Value",
    ax: Axes | None = None,
) -> Figure:
    """Deprecated: use :func:`plot`."""
    return _line(mesh, solution, title=title, xlabel=xlabel, ylabel=ylabel, ax=ax)


@deprecated_callable(
    "bt.plot(mesh, values)", reason=_ONE_PLOT, name="biotransport.plot_2d_solution"
)
def plot_2d_solution(
    mesh: StructuredMesh,
    solution: ArrayLike,
    title: str | None = None,
    colorbar_label: str = "Value",
    ax: Axes | None = None,
) -> Figure:
    """Deprecated: use :func:`plot`."""
    return _contour(mesh, solution, title=title, colorbar_label=colorbar_label, ax=ax)


@deprecated_callable(
    "bt.plot(mesh, values, kind='surface')",
    reason=_ONE_PLOT,
    name="biotransport.plot_2d_surface",
)
def plot_2d_surface(
    mesh: StructuredMesh,
    solution: ArrayLike,
    title: str | None = None,
    zlabel: str = "Value",
    ax: Axes3D | None = None,
) -> Figure:
    """Deprecated: use :func:`plot` with ``kind='surface'``."""
    return _surface(mesh, solution, title=title, zlabel=zlabel, ax=ax)


@deprecated_callable(
    "bt.plot(mesh, values, kind=...)", reason=_ONE_PLOT, name="biotransport.plot_field"
)
def plot_field(
    mesh: StructuredMesh,
    values: ArrayLike,
    *,
    title: str | None = None,
    ax: Axes | Axes3D | None = None,
    kind: Literal["contour", "surface"] = "contour",
    xlabel: str | None = None,
    ylabel: str | None = None,
    colorbar_label: str = "Value",
    zlabel: str = "Value",
) -> Figure:
    """Deprecated: use :func:`plot`."""
    if kind not in {"contour", "surface"}:
        raise ValueError("kind must be 'contour' or 'surface'")
    labels = {}
    if xlabel:
        labels["xlabel"] = xlabel
    if ylabel:
        labels["ylabel"] = ylabel
    if mesh.is_1d():
        return _line(mesh, values, title=title, ax=ax, **labels)
    if kind == "surface":
        return _surface(mesh, values, title=title, zlabel=zlabel, ax=ax, **labels)
    return _contour(
        mesh, values, title=title, colorbar_label=colorbar_label, ax=ax, **labels
    )


@deprecated_callable(
    "bt.plot(mesh, values, save_to=...)", reason=_ONE_PLOT, name="biotransport.plot_1d"
)
def plot_1d(
    mesh: StructuredMesh,
    solution: ArrayLike,
    title: str | None = None,
    xlabel: str = "Position",
    ylabel: str = "Value",
    *,
    save_as: tuple[str, str] | None = None,
    show_grid: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """Deprecated: use :func:`plot` with ``save_to=``."""
    fig = _line(mesh, solution, title=title, xlabel=xlabel, ylabel=ylabel, ax=ax)
    axes = fig.axes[0] if ax is None else ax
    if show_grid:
        axes.grid(True, alpha=0.3)
    if save_as is not None:
        filename, example_name = save_as
        fig.savefig(
            get_result_path(filename, example_name), dpi=150, bbox_inches="tight"
        )
    return fig


@deprecated_callable(
    "bt.plot(mesh, values, save_to=...)", reason=_ONE_PLOT, name="biotransport.plot_2d"
)
def plot_2d(
    mesh: StructuredMesh,
    solution: ArrayLike,
    title: str | None = None,
    xlabel: str = "X",
    ylabel: str = "Y",
    colorbar_label: str = "Value",
    *,
    save_as: tuple[str, str] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Deprecated: use :func:`plot` with ``save_to=``."""
    fig = _contour(
        mesh,
        solution,
        title=title,
        colorbar_label=colorbar_label,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
    )
    if save_as is not None:
        filename, example_name = save_as
        fig.savefig(
            get_result_path(filename, example_name), dpi=150, bbox_inches="tight"
        )
    return fig
