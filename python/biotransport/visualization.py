"""Visualization tools for biotransport simulations.

These functions are intentionally beginner-friendly:
- accept either flat solver outputs or already-shaped NumPy arrays
- avoid slow Python loops over mesh indexing
- return a Matplotlib figure for easy saving/customization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from numpy.typing import ArrayLike

from .mesh_utils import as_1d, as_2d, x_nodes, xy_grid
from .utils import get_result_path


def _pyplot():
    """Import pyplot on first use.

    Importing Matplotlib costs about a third of a second, which is most of the
    cost of ``import biotransport``. Deferring it keeps the import fast for the
    many scripts and test runs that never draw anything.
    """

    import matplotlib.pyplot as plt

    return plt


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D

    from ._core import StructuredMesh


def plot_1d_solution(
    mesh: StructuredMesh,
    solution: ArrayLike,
    title: str | None = None,
    xlabel: str = "Position",
    ylabel: str = "Value",
    ax: Axes | None = None,
) -> Figure:
    """Plot a 1D solution on a mesh.

    Args:
        mesh: The 1D mesh
        solution: The solution values (array-like)
        title: Plot title
        xlabel: x-axis label
        ylabel: y-axis label
        ax: Optional Matplotlib axes to plot into
    """

    if not mesh.is_1d():
        raise ValueError("Mesh must be 1D for 1D plotting")

    x = x_nodes(mesh)
    y = as_1d(mesh, solution)

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


def plot_2d_solution(
    mesh: StructuredMesh,
    solution: ArrayLike,
    title: str | None = None,
    colorbar_label: str = "Value",
    ax: Axes | None = None,
    *,
    colorbar: bool = True,
) -> Figure:
    """Plot a 2D solution on a mesh as a contour plot.

    Args:
        mesh: The 2D mesh
        solution: The solution values (flat or shaped)
        title: Plot title
        colorbar_label: Label for the colorbar
        ax: Optional Matplotlib axes to plot into
        colorbar: Whether to attach a colorbar. Pass ``False`` when drawing
            repeatedly into the same axes, because each colorbar steals width
            from the axes it is attached to.
    """

    if mesh.is_1d():
        raise ValueError("Mesh must be 2D for 2D plotting")

    X, Y = xy_grid(mesh)
    Z = as_2d(mesh, solution)

    if ax is None:
        fig, ax = _pyplot().subplots(figsize=(10, 8))
    else:
        fig = cast("Figure", ax.figure)

    contour = ax.contourf(X, Y, Z, 50, cmap="viridis")
    if colorbar:
        fig.colorbar(contour, ax=ax, label=colorbar_label)

    if title:
        ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return fig


def plot_2d_surface(
    mesh: StructuredMesh,
    solution: ArrayLike,
    title: str | None = None,
    zlabel: str = "Value",
    ax: Axes3D | None = None,
) -> Figure:
    """Plot a 2D solution as a 3D surface.

    Args:
        mesh: The 2D mesh
        solution: The solution values (flat or shaped)
        title: Plot title
        zlabel: z-axis label
        ax: Optional Matplotlib 3D axes to plot into
    """

    if mesh.is_1d():
        raise ValueError("Mesh must be 2D for 3D surface plotting")

    X, Y = xy_grid(mesh)
    Z = as_2d(mesh, solution)

    if ax is None:
        fig = _pyplot().figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = cast("Figure", ax.figure)

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

    if title:
        ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel(zlabel)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    return fig


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
    """Plot a field with one obvious function call.

    For 1D meshes this calls :func:`plot_1d_solution`.
    For 2D meshes this calls :func:`plot_2d_solution` (default) or
    :func:`plot_2d_surface` when ``kind='surface'``.

    Args:
        mesh: The mesh
        values: Field values (flat or shaped)
        title: Optional plot title
        ax: Optional Matplotlib axes to plot into
        kind: For 2D meshes, "contour" (default) or "surface"
        xlabel: Optional x-axis label override
        ylabel: Optional y-axis label override
        colorbar_label: For 2D contour plots, label for the colorbar
        zlabel: For 2D surface plots, z-axis label

    Returns:
        Matplotlib figure.
    """

    if mesh.is_1d():
        fig = plot_1d_solution(
            mesh,
            values,
            title=title,
            xlabel=xlabel or "Position",
            ylabel=ylabel or "Value",
            ax=ax,
        )
        return fig

    if kind not in {"surface", "contour"}:
        raise ValueError("kind must be 'contour' or 'surface'")

    if kind == "surface":
        fig = plot_2d_surface(mesh, values, title=title, zlabel=zlabel, ax=ax)
    else:
        fig = plot_2d_solution(
            mesh,
            values,
            title=title,
            colorbar_label=colorbar_label,
            ax=ax,
        )

    # Label the axes we actually drew into. Reaching for fig.axes[0] would
    # relabel the figure's first subplot, which is the wrong one whenever the
    # caller passed an ax belonging to a multi-panel figure.
    target = ax if ax is not None else fig.axes[0]
    if xlabel:
        target.set_xlabel(xlabel)
    if ylabel:
        target.set_ylabel(ylabel)
    return fig


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
    """Enhanced 1D plotting with automatic saving.

    Convenience function that wraps plot_1d_solution with additional features:
    - Automatic file saving using get_result_path
    - Grid display control

    Args:
        mesh: The 1D mesh
        solution: The solution values
        title: Plot title
        xlabel: x-axis label
        ylabel: y-axis label
        save_as: Optional (filename, example_name) tuple for automatic saving
        show_grid: Whether to display grid lines (default True)
        ax: Optional Matplotlib axes to plot into

    Returns:
        Matplotlib figure

    Example:
        >>> bt.plot_1d(mesh, solution,
        ...            title='Concentration',
        ...            xlabel='Position (mm)',
        ...            ylabel='Concentration (mM)',
        ...            save_as=('result.png', 'diffusion_1d'))
    """
    fig = plot_1d_solution(
        mesh, solution, title=title, xlabel=xlabel, ylabel=ylabel, ax=ax
    )

    if ax is None:
        ax = fig.axes[0]

    if show_grid:
        ax.grid(True, alpha=0.3)

    if save_as is not None:
        filename, example_name = save_as
        filepath = get_result_path(filename, example_name)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")

    return fig


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
    """Enhanced 2D plotting with automatic saving.

    Convenience function that wraps plot_2d_solution with additional features:
    - Automatic file saving using get_result_path

    Args:
        mesh: The 2D mesh
        solution: The solution values (flat or shaped)
        title: Plot title
        xlabel: x-axis label
        ylabel: y-axis label
        colorbar_label: Label for the colorbar
        save_as: Optional (filename, example_name) tuple for automatic saving
        ax: Optional Matplotlib axes to plot into

    Returns:
        Matplotlib figure

    Example:
        >>> bt.plot_2d(mesh, solution,
        ...            title='Concentration Field',
        ...            colorbar_label='Concentration (mM)',
        ...            save_as=('result.png', 'diffusion_2d'))
    """
    fig = plot_2d_solution(
        mesh, solution, title=title, colorbar_label=colorbar_label, ax=ax
    )

    if ax is None:
        ax = fig.axes[0]

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save_as is not None:
        filename, example_name = save_as
        filepath = get_result_path(filename, example_name)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")

    return fig


def plot(
    mesh_or_result,
    solution=None,
    *,
    title: str | None = None,
    kind: str = "auto",
    show: bool = True,
    **kwargs,
):
    """Plot a field on a mesh.

    Detects 1D vs 2D and picks the plot type. Accepts a
    :class:`~biotransport.Solution` on its own, since it knows its own mesh, or a
    ``(mesh, values)`` pair when you have a bare array.

    For anything that came out of :func:`biotransport.solve`, prefer
    ``sol.plot()`` -- it does not call ``plt.show()``, it takes ``times=`` to
    overlay snapshots, and it returns the Axes so you can compose further.

    Args:
        mesh_or_result: A :class:`~biotransport.Solution`, or a structured mesh.
        solution: Field values, or an object exposing ``concentration`` or
            ``solution``. Omit when the first argument is a ``Solution``.
        title: Plot title.
        kind: ``'auto'`` (default), ``'contour'``, ``'surface'`` or ``'line'``.
        show: Whether to call ``plt.show()`` (default True).
        **kwargs: Passed through to the underlying plot function.

    Returns:
        Matplotlib figure.

    Examples:
        >>> sol = bt.solve(problem, end_time=0.1)
        >>> bt.plot(sol, show=False)              # a Solution knows its mesh
        >>> bt.plot(mesh, values, kind='surface')  # a bare array needs one
    """
    if not isinstance(kind, str):
        raise TypeError("kind must be a string")
    normalized_kind = kind.casefold()
    if not isinstance(show, bool):
        raise TypeError("show must be a boolean")
    if solution is None:
        # A Solution carries its own mesh, so it can be plotted on its own.
        own_mesh = getattr(mesh_or_result, "mesh", None)
        if own_mesh is not None and not callable(own_mesh):
            solution = mesh_or_result
            mesh_or_result = own_mesh
        elif hasattr(mesh_or_result, "concentration") or hasattr(
            mesh_or_result, "solution"
        ):
            raise ValueError(
                "this result does not carry a mesh, so pass both: "
                "bt.plot(mesh, result)."
            )
        else:
            raise TypeError("field values or a solver result are required")
    if not hasattr(mesh_or_result, "is_1d") or not callable(mesh_or_result.is_1d):
        raise TypeError("mesh_or_result must be a structured mesh")

    values = solution
    if hasattr(solution, "concentration"):
        candidate = solution.concentration
        values = candidate() if callable(candidate) else candidate
    elif hasattr(solution, "solution"):
        candidate = solution.solution
        values = candidate() if callable(candidate) else candidate

    mesh = mesh_or_result
    is_1d = bool(mesh.is_1d())
    if is_1d:
        if normalized_kind not in {"auto", "line"}:
            raise ValueError("a 1D mesh supports kind='auto' or kind='line'")
        fig = plot_1d_solution(mesh, values, title=title, **kwargs)
    else:
        if normalized_kind not in {"auto", "contour", "surface"}:
            raise ValueError(
                "a 2D mesh supports kind='auto', kind='contour', or kind='surface'"
            )
        if normalized_kind == "surface":
            fig = plot_2d_surface(mesh, values, title=title, **kwargs)
        else:
            fig = plot_2d_solution(mesh, values, title=title, **kwargs)

    if show:
        _pyplot().show()

    return fig
