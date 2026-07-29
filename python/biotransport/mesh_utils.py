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

import operator
from numbers import Real
from typing import Tuple

import numpy as np


def _require_callable(mesh, name: str):
    method = getattr(mesh, name, None)
    if not callable(method):
        raise TypeError(f"mesh must provide {name}()")
    return method


def _validated_coordinates(values: np.ndarray, name: str) -> np.ndarray:
    if values.size == 0:
        raise ValueError(f"mesh has no {name} coordinates")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"mesh {name} coordinates must be finite")
    if values.size > 1 and not np.all(np.diff(values) > 0.0):
        raise ValueError(f"mesh {name} coordinates must be strictly increasing")
    return values


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_cell_count(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _numeric_array(values, name: str = "values") -> np.ndarray:
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{name} must not contain masked values")
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc

    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError(
            f"{name} must contain real values; complex values are not supported"
        )
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(
            f"{name} must contain real numeric values; boolean values are not supported"
        )
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError(
            f"{name} must contain real numeric values; text and object values "
            "are not supported"
        )

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            arr = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def x_nodes(mesh) -> np.ndarray:
    """Return x-coordinates of mesh nodes as a 1D NumPy array."""

    nx = _require_callable(mesh, "nx")
    x = _require_callable(mesh, "x")
    n = int(nx()) + 1
    if n < 2:
        raise ValueError("mesh must contain at least one x cell")
    values = np.fromiter((x(i) for i in range(n)), dtype=np.float64, count=n)
    return _validated_coordinates(values, "x")


def y_nodes(mesh) -> np.ndarray:
    """Return y-coordinates of mesh nodes as a 1D NumPy array (2D meshes only)."""

    is_1d = _require_callable(mesh, "is_1d")
    if is_1d():
        raise ValueError("y_nodes is only valid for 2D meshes")

    ny = _require_callable(mesh, "ny")
    y = _require_callable(mesh, "y")
    n = int(ny()) + 1
    if n < 2:
        raise ValueError("mesh must contain at least one y cell")
    values = np.fromiter((y(0, j) for j in range(n)), dtype=np.float64, count=n)
    return _validated_coordinates(values, "y")


def xy_grid(mesh) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) meshgrid arrays for a 2D mesh."""

    is_1d = _require_callable(mesh, "is_1d")
    if is_1d():
        raise ValueError("xy_grid is only valid for 2D meshes")

    x = x_nodes(mesh)
    y = y_nodes(mesh)
    return np.meshgrid(x, y)


def as_1d(mesh, values) -> np.ndarray:
    """Coerce finite values into a float64 vector and validate native shape.

    A 2D field is flattened only when its shape is exactly
    ``(mesh.ny() + 1, mesh.nx() + 1)``.
    """

    arr = _numeric_array(values)
    if arr.ndim not in (1, 2):
        raise ValueError("values must be a 1D vector or a correctly shaped 2D field")

    num_nodes = _require_callable(mesh, "num_nodes")
    expected = int(num_nodes())
    if arr.ndim == 2:
        is_1d = _require_callable(mesh, "is_1d")
        if is_1d():
            raise ValueError("2D values cannot be flattened for a 1D mesh")
        nx = int(_require_callable(mesh, "nx")()) + 1
        ny = int(_require_callable(mesh, "ny")()) + 1
        if arr.shape != (ny, nx):
            raise ValueError(f"Expected shape {(ny, nx)}, got {arr.shape}")
        arr = arr.reshape(-1, order="C")

    if arr.size != expected:
        raise ValueError(f"Expected {expected} values, got {arr.size}")

    return arr


def as_2d(mesh, values) -> np.ndarray:
    """Coerce finite values into a float64 (ny+1, nx+1) array.

    Accepts either:
    - a flat vector length = mesh.num_nodes() (row-major / C order), or
    - an already-shaped 2D array with the exact native shape.
    """

    is_1d = _require_callable(mesh, "is_1d")
    if is_1d():
        raise ValueError("as_2d is only valid for 2D meshes")

    nx = int(_require_callable(mesh, "nx")()) + 1
    ny = int(_require_callable(mesh, "ny")()) + 1

    arr = _numeric_array(values)
    if arr.ndim == 2:
        if arr.shape != (ny, nx):
            raise ValueError(f"Expected shape {(ny, nx)}, got {arr.shape}")
        return arr
    if arr.ndim != 1:
        raise ValueError("values must be a flat vector or a correctly shaped 2D field")

    expected = int(_require_callable(mesh, "num_nodes")())
    if arr.size != expected:
        raise ValueError(f"Expected {expected} values, got {arr.size}")

    return arr.reshape((ny, nx), order="C")


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

    n = _positive_cell_count(n, "n")
    x_min = _finite_float(x_min, "x_min")
    x_max = _finite_float(x_max, "x_max")
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")

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

    nx = _positive_cell_count(nx, "nx")
    ny = _positive_cell_count(ny, "ny")
    x_min = _finite_float(x_min, "x_min")
    x_max = _finite_float(x_max, "x_max")
    y_min = _finite_float(y_min, "y_min")
    y_max = _finite_float(y_max, "y_max")
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")
    if y_max <= y_min:
        raise ValueError("y_max must be greater than y_min")

    return StructuredMesh(nx, ny, x_min, x_max, y_min, y_max)


# ===========================================================================
# Cylindrical mesh helpers
# ===========================================================================


def r_nodes(mesh) -> np.ndarray:
    """Return r-coordinates of cylindrical mesh nodes as a 1D NumPy array.

    Works with CylindricalMesh objects (1D radial, 2D axisymmetric, or 3D).
    """
    is_radial = getattr(mesh, "is_radial", None)
    is_axisymmetric = getattr(mesh, "is_axisymmetric", None)
    is_3d = getattr(mesh, "is_3d", None)
    if not all(callable(method) for method in (is_radial, is_axisymmetric, is_3d)):
        raise ValueError("r_nodes is only valid for CylindricalMesh")

    n = int(_require_callable(mesh, "nr")()) + 1
    if n < 2:
        raise ValueError("mesh must contain at least one radial cell")
    values = np.fromiter(
        (_require_callable(mesh, "r")(i) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    return _validated_coordinates(values, "r")


def z_nodes(mesh) -> np.ndarray:
    """Return z-coordinates of cylindrical mesh nodes as a 1D NumPy array.

    Only valid for 2D axisymmetric or 3D cylindrical meshes.
    """
    is_axisymmetric = getattr(mesh, "is_axisymmetric", None)
    is_3d = getattr(mesh, "is_3d", None)
    if not callable(is_axisymmetric) or not callable(is_3d):
        raise ValueError("z_nodes is only valid for 2D or 3D CylindricalMesh")
    if not is_axisymmetric() and not is_3d():
        raise ValueError("z_nodes is only valid for 2D or 3D CylindricalMesh")

    n = int(_require_callable(mesh, "nz")()) + 1
    if n < 2:
        raise ValueError("mesh must contain at least one axial cell")
    z = _require_callable(mesh, "z")
    values = np.fromiter((z(k) for k in range(n)), dtype=np.float64, count=n)
    return _validated_coordinates(values, "z")


def rz_grid(mesh) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R, Z) meshgrid arrays for a 2D axisymmetric cylindrical mesh.

    Only valid for 2D axisymmetric meshes.
    """
    is_axisymmetric = getattr(mesh, "is_axisymmetric", None)
    if not callable(is_axisymmetric):
        raise ValueError("rz_grid is only valid for 2D CylindricalMesh")
    if not is_axisymmetric():
        is_3d = getattr(mesh, "is_3d", None)
        if callable(is_3d) and is_3d():
            raise ValueError("rz_grid is not valid for 3D cylindrical meshes")
        raise ValueError("rz_grid is only valid for 2D axisymmetric CylindricalMesh")

    r = r_nodes(mesh)
    z = z_nodes(mesh)
    return np.meshgrid(r, z)
