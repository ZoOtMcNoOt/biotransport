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

from .mesh_utils import _finite_float, x_nodes, xy_grid


def _validated_node_count(mesh) -> int:
    """Return the node count for every mesh type supported by the library."""
    from ._core import (
        CylindricalMesh,
        NonuniformMesh1D,
        StructuredMesh,
        StructuredMesh3D,
    )

    supported_types = (
        StructuredMesh,
        StructuredMesh3D,
        CylindricalMesh,
        NonuniformMesh1D,
    )
    if not isinstance(mesh, supported_types):
        raise TypeError(
            "mesh must be a StructuredMesh, StructuredMesh3D, "
            "CylindricalMesh, or NonuniformMesh1D"
        )
    node_count = int(mesh.num_nodes())
    if node_count <= 0:
        raise ValueError("mesh must contain at least one node")
    return node_count


def _require_structured_mesh(mesh) -> bool:
    """Return Cartesian dimensionality or reject unsupported mesh semantics."""
    from ._core import StructuredMesh

    if not isinstance(mesh, StructuredMesh):
        raise TypeError("initial-condition helpers require a StructuredMesh")
    is_1d = getattr(mesh, "is_1d", None)
    num_nodes = getattr(mesh, "num_nodes", None)
    if not callable(is_1d) or not callable(num_nodes):
        raise TypeError("initial-condition helpers require a StructuredMesh")
    _validated_node_count(mesh)
    return bool(is_1d())


def _finite_values(values: np.ndarray, name: str) -> list[float]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1, order="C")
    if not np.all(np.isfinite(flattened)):
        raise ValueError(f"{name} parameters produced non-finite values")
    return flattened.tolist()


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
    is_1d = _require_structured_mesh(mesh)
    width = _finite_float(width, "width")
    amplitude = _finite_float(amplitude, "amplitude")
    if width <= 0.0:
        raise ValueError("width must be greater than zero")
    if center is not None:
        center = _finite_float(center, "center")
    if center_x is not None:
        center_x = _finite_float(center_x, "center_x")
    if center_y is not None:
        center_y = _finite_float(center_y, "center_y")

    if is_1d:
        if center_x is not None or center_y is not None:
            raise ValueError("center_x and center_y are only valid for 2D meshes")
        x = x_nodes(mesh)
        c = center if center is not None else 0.5
        with np.errstate(over="ignore"):
            values = amplitude * np.exp(-0.5 * ((x - c) / width) ** 2)
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
        with np.errstate(over="ignore"):
            scaled_radius = np.hypot((X - cx) / width, (Y - cy) / width)
            values = amplitude * np.exp(-0.5 * scaled_radius**2)

    return _finite_values(values, "gaussian")


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
    if not _require_structured_mesh(mesh):
        raise ValueError("step() is only valid for 1D meshes")

    position = _finite_float(position, "position")
    left = _finite_float(left, "left")
    right = _finite_float(right, "right")
    x = x_nodes(mesh)
    values = np.where(x < position, left, right)
    return _finite_values(values, "step")


def uniform(mesh, value: float = 0.0):
    """Create a uniform (constant) initial condition.

    Args:
        mesh: Any supported Cartesian, cylindrical, or fitted 1D mesh. Unlike
            coordinate-dependent helpers, ``uniform`` also supports 3D meshes.
        value: Constant value everywhere (default 0.0)

    Returns:
        list: Initial condition values

    Example:
        >>> ic = bt.uniform(mesh, 1.0)
    """
    node_count = _validated_node_count(mesh)
    value = _finite_float(value, "value")
    return [value] * node_count


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
    if _require_structured_mesh(mesh):
        raise ValueError("circle() is only valid for 2D meshes")

    center_x = _finite_float(center_x, "center_x")
    center_y = _finite_float(center_y, "center_y")
    radius = _finite_float(radius, "radius")
    inside = _finite_float(inside, "inside")
    outside = _finite_float(outside, "outside")
    if radius < 0.0:
        raise ValueError("radius must be non-negative")

    X, Y = xy_grid(mesh)
    dist = np.hypot(X - center_x, Y - center_y)
    values = np.where(dist <= radius, inside, outside)
    return _finite_values(values, "circle")


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
    if not _require_structured_mesh(mesh):
        raise ValueError("sinusoidal() is only valid for 1D meshes")

    periods = _finite_float(periods, "periods")
    amplitude = _finite_float(amplitude, "amplitude")
    offset = _finite_float(offset, "offset")
    x = x_nodes(mesh)
    L = x[-1] - x[0]
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("mesh domain length must be finite and positive")
    with np.errstate(invalid="ignore", over="ignore"):
        values = offset + amplitude * np.sin(2 * np.pi * periods * (x - x[0]) / L)
    return _finite_values(values, "sinusoidal")
