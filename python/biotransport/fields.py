"""Spatial field builders for defining variable properties.

This module provides a declarative API for building spatially-varying fields
such as diffusivity, reaction rates, or source terms.

Example usage:
    >>> from biotransport import SpatialField
    >>> D_field = (
    ...     SpatialField(mesh)
    ...     .default(D_medium)
    ...     .region_box(x_min, x_max, D_membrane)
    ...     .build()
    ... )
    >>> problem.diffusivity_field(D_field)
"""

from __future__ import annotations

from numbers import Real
from typing import Optional, Tuple

import numpy as np

from ._deprecation import deprecated_callable

from .mesh_utils import r_nodes, rz_grid, x_nodes, xy_grid, y_nodes, z_nodes


def _finite_scalar(value: float, name: str) -> float:
    """Return ``value`` as a finite float or raise a user-facing error."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mesh_kind(mesh) -> str:
    """Identify supported structured mesh geometry without guessing."""
    is_1d = getattr(mesh, "is_1d", None)
    if callable(is_1d):
        return "cartesian_1d" if is_1d() else "cartesian_2d"

    is_radial = getattr(mesh, "is_radial", None)
    is_axisymmetric = getattr(mesh, "is_axisymmetric", None)
    is_3d = getattr(mesh, "is_3d", None)
    if callable(is_radial) and callable(is_axisymmetric) and callable(is_3d):
        if is_3d():
            raise ValueError(
                "SpatialField does not support full 3D cylindrical meshes; "
                "provide a flat field explicitly"
            )
        if is_radial():
            return "radial_1d"
        if is_axisymmetric():
            return "axisymmetric_2d"

    raise TypeError(
        "mesh must be a StructuredMesh or a radial/axisymmetric CylindricalMesh"
    )


class SpatialField:
    """Declarative builder for spatial fields on meshes.

    Supports 1D/2D Cartesian meshes plus radial and axisymmetric cylindrical
    meshes.  For an axisymmetric mesh, ``x`` and ``y`` region coordinates mean
    radial and axial coordinates, respectively. Fields are built by setting a
    default value and then defining regions with different values.
    """

    def __init__(self, mesh):
        """Initialize a spatial field builder.

        Args:
            mesh: A StructuredMesh or CylindricalMesh object
        """
        self._mesh_kind = _mesh_kind(mesh)
        num_nodes = getattr(mesh, "num_nodes", None)
        if not callable(num_nodes):
            raise TypeError("mesh must provide num_nodes()")
        node_count = int(num_nodes())
        if node_count <= 0:
            raise ValueError("mesh must contain at least one node")

        if self._mesh_kind == "cartesian_1d":
            coordinate_count = x_nodes(mesh).size
        elif self._mesh_kind == "cartesian_2d":
            coordinate_count = x_nodes(mesh).size * y_nodes(mesh).size
        elif self._mesh_kind == "radial_1d":
            coordinate_count = r_nodes(mesh).size
        else:
            coordinate_count = r_nodes(mesh).size * z_nodes(mesh).size
        if coordinate_count != node_count:
            raise ValueError("mesh reports an inconsistent node count")

        self.mesh = mesh
        self._field = np.zeros(node_count, dtype=np.float64)
        self._default_value = 0.0

    def _is_1d(self) -> bool:
        return self._mesh_kind in {"cartesian_1d", "radial_1d"}

    def _x_coordinates(self) -> np.ndarray:
        if self._mesh_kind == "cartesian_1d":
            return x_nodes(self.mesh)
        return r_nodes(self.mesh)

    def _coordinate_grid(self) -> tuple[np.ndarray, np.ndarray]:
        if self._mesh_kind == "cartesian_2d":
            return xy_grid(self.mesh)
        return rz_grid(self.mesh)

    def default(self, value: float) -> SpatialField:
        """Set the default value for the entire field.

        Args:
            value: Default field value

        Returns:
            self for method chaining

        Raises:
            ValueError: If ``value`` is not finite.
        """
        validated_value = _finite_scalar(value, "value")
        self._default_value = validated_value
        self._field[:] = validated_value
        return self

    def region_box(
        self,
        x_min: float,
        x_max: float,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        *,
        value: float,
    ) -> SpatialField:
        """Set field value in a rectangular/interval region.

        For 1D meshes: Sets value in interval [x_min, x_max]
        For 2D meshes: Sets value in rectangle [x_min, x_max] × [y_min, y_max]

        Args:
            x_min: Minimum x coordinate
            x_max: Maximum x coordinate
            y_min: Minimum y coordinate (2D only)
            y_max: Maximum y coordinate (2D only)
            value: Field value in this region

        Returns:
            self for method chaining

        Raises:
            ValueError: If bounds are reversed or non-finite, ``value`` is
                non-finite, required 2D bounds are missing, or the region
                contains no mesh nodes.
        """
        x_min = _finite_scalar(x_min, "x_min")
        x_max = _finite_scalar(x_max, "x_max")
        value = _finite_scalar(value, "value")
        if x_min > x_max:
            raise ValueError("x_min must be less than or equal to x_max")

        if self._is_1d():
            # 1D case: interval
            if y_min is not None or y_max is not None:
                raise ValueError("y_min and y_max should not be provided for 1D meshes")

            x = self._x_coordinates()
            mask = (x >= x_min) & (x <= x_max)

        else:
            # 2D case: rectangle
            if y_min is None or y_max is None:
                raise ValueError("y_min and y_max required for 2D meshes")
            y_min = _finite_scalar(y_min, "y_min")
            y_max = _finite_scalar(y_max, "y_max")
            if y_min > y_max:
                raise ValueError("y_min must be less than or equal to y_max")

            X, Y = self._coordinate_grid()
            mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)

            # Flatten mask and apply
            mask_flat = mask.ravel()
            mask = mask_flat

        if not np.any(mask):
            raise ValueError("region_box does not select any mesh nodes")
        self._field[mask] = value

        return self

    def region_circle(
        self, x0: float, y0: float, radius: float, *, value: float
    ) -> SpatialField:
        """Set field value in a circular region (2D only).

        Sets value where sqrt((x-x0)^2 + (y-y0)^2) <= radius

        Args:
            x0: Circle center x coordinate
            y0: Circle center y coordinate
            radius: Circle radius
            value: Field value inside circle

        Returns:
            self for method chaining

        Raises:
            ValueError: If the mesh is 1D, inputs are invalid, or the circle
                contains no mesh nodes.
        """
        if self._is_1d():
            raise ValueError("region_circle is only valid for 2D meshes")

        x0 = _finite_scalar(x0, "x0")
        y0 = _finite_scalar(y0, "y0")
        radius = _finite_scalar(radius, "radius")
        value = _finite_scalar(value, "value")
        if radius < 0.0:
            raise ValueError("radius must be non-negative")

        X, Y = self._coordinate_grid()
        dist = np.hypot(X - x0, Y - y0)
        mask = dist <= radius

        # Flatten mask and apply
        mask_flat = mask.ravel()
        if not np.any(mask_flat):
            raise ValueError("region_circle does not select any mesh nodes")
        self._field[mask_flat] = value

        return self

    def region_annulus(
        self, x0: float, y0: float, r_inner: float, r_outer: float, *, value: float
    ) -> SpatialField:
        """Set field value in an annular region (2D only).

        Sets value where r_inner <= sqrt((x-x0)^2 + (y-y0)^2) <= r_outer

        Args:
            x0: Annulus center x coordinate
            y0: Annulus center y coordinate
            r_inner: Inner radius
            r_outer: Outer radius
            value: Field value in annulus

        Returns:
            self for method chaining

        Raises:
            ValueError: If the mesh is 1D, radii are invalid, inputs are
                non-finite, or the annulus contains no mesh nodes.
        """
        if self._is_1d():
            raise ValueError("region_annulus is only valid for 2D meshes")

        x0 = _finite_scalar(x0, "x0")
        y0 = _finite_scalar(y0, "y0")
        r_inner = _finite_scalar(r_inner, "r_inner")
        r_outer = _finite_scalar(r_outer, "r_outer")
        value = _finite_scalar(value, "value")
        if r_inner < 0.0:
            raise ValueError("r_inner must be non-negative")
        if r_outer < r_inner:
            raise ValueError("r_outer must be greater than or equal to r_inner")

        X, Y = self._coordinate_grid()
        dist = np.hypot(X - x0, Y - y0)
        mask = (dist >= r_inner) & (dist <= r_outer)

        # Flatten mask and apply
        mask_flat = mask.ravel()
        if not np.any(mask_flat):
            raise ValueError("region_annulus does not select any mesh nodes")
        self._field[mask_flat] = value

        return self

    def build(self) -> np.ndarray:
        """Build and return the field as a flat ``float64`` NumPy array.

        The array is an independent copy, so later builder edits do not change
        it. It can be passed directly to ``Problem`` field methods and to the
        native solvers.
        """
        return self._field.copy()

    @deprecated_callable(
        "SpatialField.build()",
        reason="build() now returns the same NumPy array",
        name="SpatialField.build_array",
    )
    def build_array(self) -> np.ndarray:
        """Deprecated alias of :meth:`build`."""
        return self.build()


def layered_1d(
    mesh, layers: list[Tuple[float, float, float]], default: float = 0.0
) -> np.ndarray:
    """Create a 1D layered field with different values in different regions.

    Convenience function for creating piecewise constant fields in 1D.

    Args:
        mesh: 1D StructuredMesh or radial CylindricalMesh
        layers: List of (x_min, x_max, value) tuples defining each layer
        default: Default value outside all layers

    Returns:
        Field as a flat NumPy array suitable for TransportProblem methods

    Example:
        >>> D_field = layered_1d(mesh, [
        ...     (0.0, 2.0, 1e-9),      # Layer 1
        ...     (2.0, 3.0, 1e-10),     # Layer 2 (membrane)
        ...     (3.0, 5.0, 1e-9),      # Layer 3
        ... ])
        >>> problem.diffusivity_field(D_field)
    """
    builder = SpatialField(mesh)
    if not builder._is_1d():
        raise ValueError("layered_1d only works with 1D meshes")
    builder.default(default)

    for x_min, x_max, value in layers:
        builder.region_box(x_min, x_max, value=value)

    return builder.build()
