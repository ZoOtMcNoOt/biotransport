"""Mesh + field convenience helpers (Python-level).

These helpers are intentionally small and dependency-free beyond NumPy.
They provide a stable, beginner-friendly way to:
- get coordinate arrays from a :class:`biotransport.StructuredMesh`
- reshape flat solver outputs into 2D arrays

Why these exist:
- Undergrad users should not need to write repeated loops like
  ``np.array([mesh.x(i) for i in range(mesh.nx()+1)])``.
- Plotting should not require slow Python loops over ``mesh.index(i, j)``.

All functions accept the bound C++ :class:`biotransport.StructuredMesh`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def x_nodes(mesh) -> np.ndarray:
    """Return x-coordinates of mesh nodes as a 1D NumPy array."""

    n = int(mesh.nx()) + 1
    return np.fromiter((mesh.x(i) for i in range(n)), dtype=np.float64, count=n)


def y_nodes(mesh) -> np.ndarray:
    """Return y-coordinates of mesh nodes as a 1D NumPy array (2D meshes only)."""

    if mesh.is_1d():
        raise ValueError("y_nodes is only valid for 2D meshes")

    n = int(mesh.ny()) + 1
    return np.fromiter((mesh.y(0, j) for j in range(n)), dtype=np.float64, count=n)


def xy_grid(mesh) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) meshgrid arrays for a 2D mesh."""

    if mesh.is_1d():
        raise ValueError("xy_grid is only valid for 2D meshes")

    x = x_nodes(mesh)
    y = y_nodes(mesh)
    return np.meshgrid(x, y)


def as_1d(mesh, values) -> np.ndarray:
    """Coerce values into a float64 1D array and validate length."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    expected = int(mesh.num_nodes())
    if arr.size != expected:
        raise ValueError(f"Expected {expected} values, got {arr.size}")

    return arr


def as_2d(mesh, values) -> np.ndarray:
    """Coerce values into a float64 (ny+1, nx+1) array.

    Accepts either:
    - a flat vector length = mesh.num_nodes() (row-major / C order), or
    - an already-shaped 2D array.
    """

    if mesh.is_1d():
        raise ValueError("as_2d is only valid for 2D meshes")

    nx = int(mesh.nx()) + 1
    ny = int(mesh.ny()) + 1

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape != (ny, nx):
            raise ValueError(f"Expected shape {(ny, nx)}, got {arr.shape}")
        return arr

    flat = arr.reshape(-1)
    expected = int(mesh.num_nodes())
    if flat.size != expected:
        raise ValueError(f"Expected {expected} values, got {flat.size}")

    return flat.reshape((ny, nx), order="C")


# ===========================================================================
# Mesh creation convenience functions
# ===========================================================================


def mesh_1d(n: int, x_min: float = 0.0, x_max: float = 1.0):
    """Create a 1D mesh with n cells from x_min to x_max.

    This is a convenience wrapper for StructuredMesh with a more intuitive API.

    Args:
        n: Number of cells (results in n+1 nodes)
        x_min: Left boundary coordinate (default 0.0)
        x_max: Right boundary coordinate (default 1.0)

    Returns:
        StructuredMesh: A 1D mesh ready for use with solvers

    Example:
        >>> mesh = mesh_1d(100)  # 100 cells, domain [0, 1]
        >>> mesh = mesh_1d(50, 0.0, 0.01)  # 50 cells, domain [0, 0.01]
    """
    from ._core import StructuredMesh

    return StructuredMesh(n, x_min, x_max)


def mesh_2d(
    nx: int,
    ny: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
):
    """Create a 2D mesh with nx × ny cells.

    This is a convenience wrapper for StructuredMesh with a more intuitive API.

    Args:
        nx: Number of cells in x direction
        ny: Number of cells in y direction
        x_min: Left boundary x coordinate (default 0.0)
        x_max: Right boundary x coordinate (default 1.0)
        y_min: Bottom boundary y coordinate (default 0.0)
        y_max: Top boundary y coordinate (default 1.0)

    Returns:
        StructuredMesh: A 2D mesh ready for use with solvers

    Example:
        >>> mesh = mesh_2d(50, 50)  # 50×50 cells, unit square
        >>> mesh = mesh_2d(100, 50, x_max=0.01, y_max=0.005)  # 100×50 cells, 10mm × 5mm
        >>> mesh = mesh_2d(50, 50, -1.0, 1.0, -1.0, 1.0)  # centered at origin
    """
    from ._core import StructuredMesh

    return StructuredMesh(nx, ny, x_min, x_max, y_min, y_max)


# ===========================================================================
# Cylindrical mesh helpers
# ===========================================================================


def r_nodes(mesh) -> np.ndarray:
    """Return r-coordinates of cylindrical mesh nodes as a 1D NumPy array.

    Works with CylindricalMesh objects (1D radial, 2D axisymmetric, or 3D).
    """
    if not hasattr(mesh, "nr"):
        raise ValueError("r_nodes is only valid for CylindricalMesh")

    n = int(mesh.nr()) + 1
    return np.fromiter((mesh.r(i) for i in range(n)), dtype=np.float64, count=n)


def z_nodes(mesh) -> np.ndarray:
    """Return z-coordinates of cylindrical mesh nodes as a 1D NumPy array.

    Only valid for 2D axisymmetric or 3D cylindrical meshes.
    """
    if not hasattr(mesh, "nz"):
        raise ValueError("z_nodes is only valid for 2D or 3D CylindricalMesh")

    n = int(mesh.nz()) + 1
    return np.fromiter((mesh.z(k) for k in range(n)), dtype=np.float64, count=n)


def rz_grid(mesh) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R, Z) meshgrid arrays for a 2D axisymmetric cylindrical mesh.

    Only valid for 2D axisymmetric meshes.
    """
    if not hasattr(mesh, "nz"):
        raise ValueError("rz_grid is only valid for 2D CylindricalMesh")
    if hasattr(mesh, "ntheta") and mesh.ntheta() > 0:
        raise ValueError("rz_grid is not valid for 3D cylindrical meshes")

    r = r_nodes(mesh)
    z = z_nodes(mesh)
    return np.meshgrid(r, z)
