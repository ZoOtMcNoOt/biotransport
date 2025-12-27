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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike

    from ._core import CylindricalMesh, StructuredMesh


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
            Each array must have length equal to mesh.num_nodes().
        filename: Output file path. Extension .vtk will be added if missing.
        title: Title string embedded in VTK file header.

    Returns:
        Path to the written file.

    Raises:
        ValueError: If field array length doesn't match mesh node count.
        TypeError: If mesh type is not supported.

    Example:
        >>> mesh = bt.StructuredMesh(100, 0.0, 1.0)  # 1D
        >>> temperature = np.linspace(300, 400, mesh.num_nodes())
        >>> bt.write_vtk(mesh, {"temperature": temperature}, "heat.vtk")
    """
    filepath = Path(filename)
    if filepath.suffix.lower() != ".vtk":
        filepath = filepath.with_suffix(".vtk")

    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Determine mesh type and dimensions
    # CylindricalMesh has 'nr' method, StructuredMesh has 'nx'
    is_cylindrical = hasattr(mesh, "nr") and not hasattr(mesh, "nx")
    is_1d = mesh.is_1d() if hasattr(mesh, "is_1d") else False

    num_nodes = mesh.num_nodes()

    # Validate field sizes
    for name, data in fields.items():
        arr = np.asarray(data).ravel()
        if len(arr) != num_nodes:
            raise ValueError(
                f"Field '{name}' has {len(arr)} values, but mesh has {num_nodes} nodes"
            )

    # Write VTK file
    with open(filepath, "w", encoding="ascii") as f:
        _write_vtk_header(f, title)

        if is_1d:
            _write_vtk_1d_geometry(f, mesh)
        else:
            _write_vtk_2d_geometry(f, mesh, is_cylindrical)

        _write_vtk_point_data(f, fields, num_nodes)

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
            Each fields_dict maps field names to numpy arrays.
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
    base_path = Path(base_filename)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    vtk_files: list[tuple[float, Path]] = []

    # Write individual VTK files
    for idx, (time_val, fields) in enumerate(time_fields):
        vtk_name = f"{base_path.stem}_{idx:04d}.vtk"
        vtk_path = base_path.parent / vtk_name
        write_vtk(mesh, fields, vtk_path, title=f"{title} t={time_val:.6g}")
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


def _write_vtk_2d_geometry(f, mesh, is_cylindrical: bool) -> None:
    """Write 2D mesh geometry as VTK STRUCTURED_GRID or RECTILINEAR_GRID."""
    if is_cylindrical:
        _write_vtk_cylindrical_geometry(f, mesh)
    else:
        _write_vtk_cartesian_2d_geometry(f, mesh)


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

        # Sanitize field name (VTK doesn't like spaces/special chars)
        safe_name = name.replace(" ", "_").replace("-", "_")

        f.write(f"SCALARS {safe_name} double 1\n")
        f.write("LOOKUP_TABLE default\n")

        # Write data values
        for val in arr:
            f.write(f"{val}\n")


def _write_pvd_file(pvd_path: Path, vtk_files: list[tuple[float, Path]]) -> None:
    """Write ParaView Data (PVD) collection file for time series."""
    with open(pvd_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(
            '<VTKFile type="Collection" version="0.1" ' 'byte_order="LittleEndian">\n'
        )
        f.write("  <Collection>\n")

        for time_val, vtk_path in vtk_files:
            # Use relative path from PVD file location
            rel_path = vtk_path.name
            f.write(f'    <DataSet timestep="{time_val}" file="{rel_path}"/>\n')

        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")
