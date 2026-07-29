"""VTK file export utilities for ParaView visualization.

This module provides functions to export simulation results to VTK Legacy format,
which can be opened in ParaView, VisIt, or other scientific visualization tools.

VTK Legacy format is simple ASCII text that's easy to inspect and widely supported.
For time-series data, use write_vtk_series() to create a PVD collection file.

Example:
    >>> import biotransport as bt
    >>> import numpy as np
    >>>
    >>> # Create mesh and solution
    >>> mesh = bt.StructuredMesh(50, 50, 0.0, 1.0, 0.0, 1.0)
    >>> x, y = bt.xy_grid(mesh)
    >>> concentration = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.01)
    >>>
    >>> # Export to VTK
    >>> bt.write_vtk(mesh, {"concentration": concentration}, "result.vtk")

References:
    - VTK File Formats: https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
    - ParaView: https://www.paraview.org/
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING
from xml.sax.saxutils import quoteattr

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ._core import CylindricalMesh, StructuredMesh


def _validate_title(title: str) -> str:
    if not isinstance(title, str):
        raise TypeError("title must be a string")
    if not title:
        raise ValueError("title must not be empty")
    if "\n" in title or "\r" in title:
        raise ValueError("title must not contain newlines")
    try:
        encoded = title.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("title must contain only ASCII characters") from exc
    if len(encoded) > 255:
        raise ValueError("title must be at most 255 ASCII bytes")
    return title


def _mesh_layout(mesh) -> tuple[str, tuple[int, ...], int]:
    """Return supported geometry kind, native field shape, and node count."""
    from ._core import CylindricalMesh, StructuredMesh
    from .mesh_utils import r_nodes, x_nodes, y_nodes, z_nodes

    shape: tuple[int, ...]
    if isinstance(mesh, StructuredMesh):
        num_nodes = int(mesh.num_nodes())
        x_nodes(mesh)
        if mesh.is_1d():
            shape = (int(mesh.nx()) + 1,)
            kind = "cartesian_1d"
        else:
            y_nodes(mesh)
            shape = (int(mesh.ny()) + 1, int(mesh.nx()) + 1)
            kind = "cartesian_2d"
    elif isinstance(mesh, CylindricalMesh):
        if mesh.is_3d():
            raise ValueError(
                "VTK export does not support full 3D cylindrical meshes yet"
            )
        num_nodes = int(mesh.num_nodes())
        r_nodes(mesh)
        if mesh.is_radial():
            shape = (int(mesh.nr()) + 1,)
            kind = "radial_1d"
        elif mesh.is_axisymmetric():
            z_nodes(mesh)
            shape = (int(mesh.nz()) + 1, int(mesh.nr()) + 1)
            kind = "axisymmetric_2d"
        else:
            raise ValueError("unsupported cylindrical mesh geometry")
    else:
        raise TypeError("mesh must be a StructuredMesh or CylindricalMesh")

    if num_nodes <= 0 or int(np.prod(shape)) != num_nodes:
        raise ValueError("mesh reports an inconsistent node count")
    return kind, shape, num_nodes


def _safe_field_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("field names must be strings")
    if not name:
        raise ValueError("field names must not be empty")
    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not any(character.isalnum() for character in safe_name):
        raise ValueError(f"Field name {name!r} has no usable letters or digits")
    return safe_name


def _normalize_fields(
    fields: Mapping[str, ArrayLike],
    expected_shape: tuple[int, ...],
    num_nodes: int,
) -> dict[str, np.ndarray]:
    """Validate fields fully before a file or directory is created."""
    from .mesh_utils import _numeric_array

    if not isinstance(fields, Mapping):
        raise TypeError("fields must be a mapping of names to scalar arrays")

    normalized: dict[str, np.ndarray] = {}
    source_names: dict[str, str] = {}
    for name, data in fields.items():
        safe_name = _safe_field_name(name)
        if safe_name in normalized:
            other_name = source_names[safe_name]
            raise ValueError(
                f"Field names {other_name!r} and {name!r} both sanitize to "
                f"{safe_name!r}"
            )

        arr = _numeric_array(data, f"Field {name!r}")
        if arr.ndim == 0 or arr.ndim > len(expected_shape):
            raise ValueError(
                f"Field {name!r} must be flat or have shape {expected_shape}; "
                f"got shape {arr.shape}"
            )
        if arr.ndim == 1 and arr.size != num_nodes:
            raise ValueError(
                f"Field {name!r} has {arr.size} values, but mesh has {num_nodes} nodes"
            )
        if arr.ndim > 1 and arr.shape != expected_shape:
            raise ValueError(
                f"Field {name!r} must have shape {expected_shape}; got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Field {name!r} must contain only finite values")

        normalized[safe_name] = arr.reshape(-1, order="C")
        source_names[safe_name] = name

    return normalized


def write_vtk(
    mesh: StructuredMesh | CylindricalMesh,
    fields: Mapping[str, ArrayLike],
    filename: str | os.PathLike[str],
    *,
    title: str = "BioTransport Export",
) -> Path:
    """Write mesh and scalar fields to VTK Legacy ASCII format.

    Creates a .vtk file that can be opened in ParaView or VisIt for
    publication-quality visualization.

    Args:
        mesh: StructuredMesh (1D or 2D) or CylindricalMesh.
        fields: Dictionary mapping field names to numpy arrays.
            Each array must be finite and either flat with
            ``mesh.num_nodes()`` values or have the mesh's exact native shape.
        filename: Output file path. Extension .vtk will be added if missing.
        title: Title string embedded in VTK file header.

    Returns:
        Path to the written file.

    Raises:
        ValueError: If geometry, field shape/data, title, or field names are
            invalid. Full 3D cylindrical meshes are not yet supported.
        TypeError: If mesh type is not supported.

    Example:
        >>> mesh = bt.StructuredMesh(100, 0.0, 1.0)  # 1D
        >>> temperature = np.linspace(300, 400, mesh.num_nodes())
        >>> bt.write_vtk(mesh, {"temperature": temperature}, "heat.vtk")
    """
    title = _validate_title(title)
    mesh_kind, expected_shape, num_nodes = _mesh_layout(mesh)
    normalized_fields = _normalize_fields(fields, expected_shape, num_nodes)

    try:
        filepath = Path(filename)
    except TypeError as exc:
        raise TypeError("filename must be a path-like value") from exc
    if filepath.suffix.lower() != ".vtk":
        filepath = filepath.with_suffix(".vtk")

    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Write VTK file
    with open(filepath, "w", encoding="ascii") as f:
        _write_vtk_header(f, title)

        if mesh_kind == "cartesian_1d":
            _write_vtk_1d_geometry(f, mesh)
        elif mesh_kind == "radial_1d":
            _write_vtk_radial_geometry(f, mesh)
        elif mesh_kind == "axisymmetric_2d":
            _write_vtk_cylindrical_geometry(f, mesh)
        else:
            _write_vtk_cartesian_2d_geometry(f, mesh)

        _write_vtk_point_data(f, normalized_fields, num_nodes)

    return filepath


def write_vtk_series(
    mesh: StructuredMesh | CylindricalMesh,
    time_fields: Sequence[tuple[float, Mapping[str, ArrayLike]]],
    base_filename: str | os.PathLike[str],
    *,
    title: str = "BioTransport Time Series",
) -> Path:
    """Write a time series of fields to VTK files with a PVD collection.

    Creates multiple .vtk files (one per timestep) and a .pvd file that
    ParaView can open to animate the time series.

    Args:
        mesh: StructuredMesh (1D or 2D) or CylindricalMesh.
        time_fields: Sequence of (time, fields_dict) tuples.
            Times must be finite and strictly increasing. Each fields_dict maps
            field names to arrays accepted by :func:`write_vtk`.
        base_filename: Base path for output files. Will create:
            - base_filename.pvd (collection file)
            - base_filename_0000.vtk, base_filename_0001.vtk, ...
        title: Title string for individual VTK file headers.

    Returns:
        Path to the PVD collection file.

    Example:
        >>> mesh = bt.StructuredMesh(50, 50, 0.0, 1.0, 0.0, 1.0)
        >>> snapshots = []
        >>> for t in [0.0, 0.1, 0.2, 0.3]:
        ...     c = np.exp(-t) * np.ones(mesh.num_nodes())
        ...     snapshots.append((t, {"concentration": c}))
        >>> bt.write_vtk_series(mesh, snapshots, "results/diffusion")
    """
    title = _validate_title(title)
    mesh_kind, expected_shape, num_nodes = _mesh_layout(mesh)
    del mesh_kind

    try:
        base_path = Path(base_filename)
    except TypeError as exc:
        raise TypeError("base_filename must be a path-like value") from exc

    snapshots: list[tuple[float, dict[str, np.ndarray], str]] = []
    previous_time: float | None = None
    for index, snapshot in enumerate(time_fields):
        try:
            time_val, fields = snapshot
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"time_fields[{index}] must be a (time, fields) pair"
            ) from exc
        if isinstance(time_val, (bool, np.bool_)) or not isinstance(time_val, Real):
            raise TypeError(f"time_fields[{index}] time must be a real number")
        try:
            time_float = float(time_val)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"time_fields[{index}] time must be a real number") from exc
        if not np.isfinite(time_float):
            raise ValueError(f"time_fields[{index}] time must be finite")
        if previous_time is not None and time_float <= previous_time:
            raise ValueError("time values must be strictly increasing")

        normalized = _normalize_fields(fields, expected_shape, num_nodes)
        snapshot_title = _validate_title(f"{title} t={time_float:.6g}")
        snapshots.append((time_float, normalized, snapshot_title))
        previous_time = time_float

    base_path.parent.mkdir(parents=True, exist_ok=True)
    vtk_files: list[tuple[float, Path]] = []

    # Write individual VTK files
    for idx, (time_val, fields, snapshot_title) in enumerate(snapshots):
        vtk_name = f"{base_path.stem}_{idx:04d}.vtk"
        vtk_path = base_path.parent / vtk_name
        write_vtk(mesh, fields, vtk_path, title=snapshot_title)
        vtk_files.append((time_val, vtk_path))

    # Write PVD collection file
    pvd_path = base_path.with_suffix(".pvd")
    _write_pvd_file(pvd_path, vtk_files)

    return pvd_path


def _write_vtk_header(f, title: str) -> None:
    """Write VTK Legacy file header."""
    f.write("# vtk DataFile Version 3.0\n")
    f.write(f"{title}\n")
    f.write("ASCII\n")


def _write_vtk_1d_geometry(f, mesh) -> None:
    """Write 1D mesh geometry as VTK STRUCTURED_POINTS (line)."""
    # Get mesh properties
    nx = mesh.nx()
    num_nodes = nx + 1
    dx = mesh.dx()
    xmin = mesh.x(0) if hasattr(mesh, "x") else 0.0

    # For 1D, we write as a 2D slice (ny=1, nz=1)
    f.write("DATASET STRUCTURED_POINTS\n")
    f.write(f"DIMENSIONS {num_nodes} 1 1\n")
    f.write(f"ORIGIN {xmin} 0.0 0.0\n")
    f.write(f"SPACING {dx} 1.0 1.0\n")


def _write_vtk_cartesian_2d_geometry(f, mesh) -> None:
    """Write 2D Cartesian mesh as STRUCTURED_POINTS."""
    nx = mesh.nx()
    ny = mesh.ny()
    num_x = nx + 1
    num_y = ny + 1

    dx = mesh.dx()
    dy = mesh.dy()

    # Get origin
    xmin = mesh.x(0) if hasattr(mesh, "x") else 0.0
    ymin = mesh.y(0, 0) if hasattr(mesh, "y") else 0.0

    f.write("DATASET STRUCTURED_POINTS\n")
    f.write(f"DIMENSIONS {num_x} {num_y} 1\n")
    f.write(f"ORIGIN {xmin} {ymin} 0.0\n")
    f.write(f"SPACING {dx} {dy} 1.0\n")


def _write_vtk_radial_geometry(f, mesh) -> None:
    """Write a radial mesh as an explicit line in the positive x direction."""
    num_r = mesh.nr() + 1
    f.write("DATASET STRUCTURED_GRID\n")
    f.write(f"DIMENSIONS {num_r} 1 1\n")
    f.write(f"POINTS {num_r} double\n")
    for i in range(num_r):
        f.write(f"{mesh.r(i)} 0.0 0.0\n")


def _write_vtk_cylindrical_geometry(f, mesh) -> None:
    """Write cylindrical mesh as STRUCTURED_GRID with explicit coordinates."""
    nr = mesh.nr()
    nz = mesh.nz()
    num_r = nr + 1
    num_z = nz + 1
    num_nodes = num_r * num_z

    # Write as structured grid with explicit points
    f.write("DATASET STRUCTURED_GRID\n")
    f.write(f"DIMENSIONS {num_r} {num_z} 1\n")
    f.write(f"POINTS {num_nodes} double\n")

    # Get r and z coordinates
    for j in range(num_z):
        for i in range(num_r):
            r = mesh.r(i)
            z = mesh.z(j) if hasattr(mesh, "z") else j * mesh.dz()
            # Map (r, z) to (x, y, 0) for 2D visualization
            f.write(f"{r} {z} 0.0\n")


def _write_vtk_point_data(f, fields: Mapping[str, ArrayLike], num_nodes: int) -> None:
    """Write scalar point data to VTK file."""
    if not fields:
        return

    f.write(f"\nPOINT_DATA {num_nodes}\n")

    for name, data in fields.items():
        arr = np.asarray(data, dtype=np.float64).ravel()
        f.write(f"SCALARS {name} double 1\n")
        f.write("LOOKUP_TABLE default\n")

        # Write data values
        for val in arr:
            f.write(f"{val}\n")


def _write_pvd_file(pvd_path: Path, vtk_files: list[tuple[float, Path]]) -> None:
    """Write ParaView Data (PVD) collection file for time series."""
    with open(pvd_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")

        for time_val, vtk_path in vtk_files:
            # Use relative path from PVD file location
            rel_path = quoteattr(vtk_path.name)
            f.write(f'    <DataSet timestep="{time_val}" file={rel_path}/>\n')

        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")
