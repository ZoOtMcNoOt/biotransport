"""Visualization tools for biotransport simulations.

These functions are intentionally beginner-friendly:
- accept either flat solver outputs or already-shaped NumPy arrays
- avoid slow Python loops over mesh indexing
- return a Matplotlib figure for easy saving/customization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

from .mesh_utils import as_1d, as_2d, x_nodes, xy_grid
from .utils import get_result_path

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    from .._core import StructuredMesh


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
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

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
) -> Figure:
    """Plot a 2D solution on a mesh as a contour plot.

    Args:
        mesh: The 2D mesh
        solution: The solution values (flat or shaped)
        title: Plot title
        colorbar_label: Label for the colorbar
        ax: Optional Matplotlib axes to plot into
    """

    if mesh.is_1d():
        raise ValueError("Mesh must be 2D for 2D plotting")

    X, Y = xy_grid(mesh)
    Z = as_2d(mesh, solution)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    contour = ax.contourf(X, Y, Z, 50, cmap="viridis")
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
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

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

    if kind == "surface":
        fig = plot_2d_surface(mesh, values, title=title, zlabel=zlabel, ax=ax)
        if xlabel:
            ax = fig.axes[0]
            ax.set_xlabel(xlabel)
        if ylabel:
            ax = fig.axes[0]
            ax.set_ylabel(ylabel)
        return fig
    if kind == "contour":
        fig = plot_2d_solution(
            mesh,
            values,
            title=title,
            colorbar_label=colorbar_label,
            ax=ax,
        )
        if xlabel:
            ax = fig.axes[0]
            ax.set_xlabel(xlabel)
        if ylabel:
            ax = fig.axes[0]
            ax.set_ylabel(ylabel)
        return fig

    raise ValueError("kind must be 'contour' or 'surface'")


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
    """Universal plotting function - the simplest way to visualize results.

    Automatically detects 1D vs 2D and chooses the right plot type.
    Can accept either (mesh, solution) or just a RunResult.

    Args:
        mesh_or_result: Either a mesh or a RunResult from bt.solve()
        solution: Solution values (optional if mesh_or_result is a RunResult)
        title: Plot title (optional)
        kind: Plot type - 'auto' (default), 'contour', 'surface', or 'line'
        show: Whether to call plt.show() (default True)
        **kwargs: Additional arguments passed to underlying plot functions

    Returns:
        Matplotlib figure

    Examples:
        >>> # From a result
        >>> result = bt.solve(problem, t=0.1)
        >>> bt.plot(mesh, result.solution())

        >>> # 3D surface plot
        >>> bt.plot(mesh, solution, kind='surface')
    """
    # Handle RunResult input
    if hasattr(mesh_or_result, "solution") and solution is None:
        raise ValueError(
            "When passing a RunResult, you still need the mesh. "
            "Use: bt.plot(mesh, result.solution())"
        )

    mesh = mesh_or_result

    if mesh.is_1d():
        fig = plot_1d_solution(mesh, solution, title=title)
    else:
        if kind == "surface":
            fig = plot_2d_surface(mesh, solution, title=title, **kwargs)
        else:
            fig = plot_2d_solution(mesh, solution, title=title, **kwargs)

    if show:
        plt.show()

    return fig
