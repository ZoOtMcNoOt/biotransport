"""Initial condition helper functions.

These functions create common initial conditions for transport problems,
reducing boilerplate and making code more readable.

Example:
    >>> import biotransport as bt
    >>> mesh = bt.mesh_1d(100)
    >>> problem = (
    ...     bt.Problem(mesh)
    ...     .diffusivity(0.01)
    ...     .initial(bt.gaussian(mesh, center=0.5, width=0.1))
    ... )
"""

from __future__ import annotations

import numpy as np

from .mesh_utils import x_nodes, xy_grid


def gaussian(
    mesh,
    center: float | None = None,
    width: float = 0.1,
    amplitude: float = 1.0,
    *,
    center_x: float | None = None,
    center_y: float | None = None,
):
    """Create a Gaussian (bell curve) initial condition.

    For 1D: exp(-((x - center)^2) / (2 * width^2))
    For 2D: Centered at (center_x, center_y) with same width in both directions

    Args:
        mesh: The mesh to create the IC for
        center: Center position for 1D, or both x and y for 2D (default 0.5)
        width: Standard deviation / width parameter (default 0.1)
        amplitude: Peak amplitude (default 1.0)
        center_x: X center for 2D (overrides center)
        center_y: Y center for 2D (overrides center)

    Returns:
        list: Initial condition values for all mesh nodes

    Example:
        >>> ic = bt.gaussian(mesh, center=0.5, width=0.1)  # 1D
        >>> ic = bt.gaussian(mesh, center=0.0, width=0.1)  # 2D centered at origin
        >>> ic = bt.gaussian(mesh, center_x=0.2, center_y=0.3, width=0.1)  # 2D
    """
    if mesh.is_1d():
        x = x_nodes(mesh)
        c = center if center is not None else 0.5
        values = amplitude * np.exp(-((x - c) ** 2) / (2 * width**2))
    else:
        X, Y = xy_grid(mesh)
        # Resolve center coordinates
        cx = (
            center_x
            if center_x is not None
            else (center if center is not None else 0.5)
        )
        cy = (
            center_y
            if center_y is not None
            else (center if center is not None else 0.5)
        )
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2
        values = amplitude * np.exp(-dist2 / (2 * width**2))
        values = values.ravel(order="C")  # Row-major to match solver expectations

    return values.tolist()


def step(mesh, position: float = 0.5, left: float = 1.0, right: float = 0.0):
    """Create a step function initial condition (1D only).

    Value is `left` for x < position, `right` for x >= position.

    Args:
        mesh: The 1D mesh
        position: Step location (default 0.5)
        left: Value for x < position (default 1.0)
        right: Value for x >= position (default 0.0)

    Returns:
        list: Initial condition values

    Example:
        >>> ic = bt.step(mesh, position=0.3, left=1.0, right=0.0)
    """
    if not mesh.is_1d():
        raise ValueError("step() is only valid for 1D meshes")

    x = x_nodes(mesh)
    values = np.where(x < position, left, right)
    return values.tolist()


def uniform(mesh, value: float = 0.0):
    """Create a uniform (constant) initial condition.

    Args:
        mesh: The mesh
        value: Constant value everywhere (default 0.0)

    Returns:
        list: Initial condition values

    Example:
        >>> ic = bt.uniform(mesh, 1.0)
    """
    return [value] * mesh.num_nodes()


def circle(
    mesh,
    center_x: float = 0.5,
    center_y: float = 0.5,
    radius: float = 0.2,
    inside: float = 1.0,
    outside: float = 0.0,
):
    """Create a circular initial condition (2D only).

    Value is `inside` within the circle, `outside` elsewhere.

    Args:
        mesh: The 2D mesh
        center_x: Circle center x-coordinate (default 0.5)
        center_y: Circle center y-coordinate (default 0.5)
        radius: Circle radius (default 0.2)
        inside: Value inside circle (default 1.0)
        outside: Value outside circle (default 0.0)

    Returns:
        list: Initial condition values

    Example:
        >>> ic = bt.circle(mesh, center_x=0.5, center_y=0.5, radius=0.1)
    """
    if mesh.is_1d():
        raise ValueError("circle() is only valid for 2D meshes")

    X, Y = xy_grid(mesh)
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    values = np.where(dist <= radius, inside, outside)
    return values.ravel(order="C").tolist()


def sinusoidal(mesh, periods: float = 1.0, amplitude: float = 1.0, offset: float = 0.0):
    """Create a sinusoidal initial condition (1D only).

    Creates sin(2π * periods * x / L) where L is the domain length.

    Args:
        mesh: The 1D mesh
        periods: Number of complete periods across domain (default 1.0)
        amplitude: Wave amplitude (default 1.0)
        offset: Vertical offset (default 0.0)

    Returns:
        list: Initial condition values

    Example:
        >>> ic = bt.sinusoidal(mesh, periods=2, amplitude=0.5)
    """
    if not mesh.is_1d():
        raise ValueError("sinusoidal() is only valid for 1D meshes")

    x = x_nodes(mesh)
    L = x[-1] - x[0]
    values = offset + amplitude * np.sin(2 * np.pi * periods * (x - x[0]) / L)
    return values.tolist()
