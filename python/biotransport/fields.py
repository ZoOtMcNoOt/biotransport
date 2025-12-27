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

from typing import Optional, Tuple

import numpy as np

from .mesh_utils import x_nodes, xy_grid


class SpatialField:
    """Declarative builder for spatial fields on meshes.

    Supports 1D and 2D structured meshes. Fields are built by setting a default
    value and then defining regions with different values.
    """

    def __init__(self, mesh):
        """Initialize a spatial field builder.

        Args:
            mesh: A StructuredMesh or CylindricalMesh object
        """
        self.mesh = mesh
        self._field = np.zeros(mesh.num_nodes(), dtype=np.float64)
        self._default_value = 0.0

    def default(self, value: float) -> SpatialField:
        """Set the default value for the entire field.

        Args:
            value: Default field value

        Returns:
            self for method chaining
        """
        self._default_value = value
        self._field[:] = value
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
        """
        if self.mesh.is_1d():
            # 1D case: interval
            if y_min is not None or y_max is not None:
                raise ValueError("y_min and y_max should not be provided for 1D meshes")

            x = x_nodes(self.mesh)
            mask = (x >= x_min) & (x <= x_max)
            self._field[mask] = value

        else:
            # 2D case: rectangle
            if y_min is None or y_max is None:
                raise ValueError("y_min and y_max required for 2D meshes")

            X, Y = xy_grid(self.mesh)
            mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)

            # Flatten mask and apply
            mask_flat = mask.ravel()
            self._field[mask_flat] = value

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
        """
        if self.mesh.is_1d():
            raise ValueError("region_circle is only valid for 2D meshes")

        X, Y = xy_grid(self.mesh)
        dist = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
        mask = dist <= radius

        # Flatten mask and apply
        mask_flat = mask.ravel()
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
        """
        if self.mesh.is_1d():
            raise ValueError("region_annulus is only valid for 2D meshes")

        X, Y = xy_grid(self.mesh)
        dist = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
        mask = (dist >= r_inner) & (dist <= r_outer)

        # Flatten mask and apply
        mask_flat = mask.ravel()
        self._field[mask_flat] = value

        return self

    def build(self) -> list:
        """Build and return the field as a Python list.

        Returns:
            Field values as a list (suitable for TransportProblem methods)
        """
        return self._field.tolist()

    def build_array(self) -> np.ndarray:
        """Build and return the field as a NumPy array.

        Returns:
            Field values as a 1D numpy array
        """
        return self._field.copy()


def layered_1d(
    mesh, layers: list[Tuple[float, float, float]], default: float = 0.0
) -> list:
    """Create a 1D layered field with different values in different regions.

    Convenience function for creating piecewise constant fields in 1D.

    Args:
        mesh: 1D StructuredMesh
        layers: List of (x_min, x_max, value) tuples defining each layer
        default: Default value outside all layers

    Returns:
        Field as a list suitable for TransportProblem methods

    Example:
        >>> D_field = layered_1d(mesh, [
        ...     (0.0, 2.0, 1e-9),      # Layer 1
        ...     (2.0, 3.0, 1e-10),     # Layer 2 (membrane)
        ...     (3.0, 5.0, 1e-9),      # Layer 3
        ... ])
        >>> problem.diffusivity_field(D_field)
    """
    if not mesh.is_1d():
        raise ValueError("layered_1d only works with 1D meshes")

    builder = SpatialField(mesh).default(default)

    for x_min, x_max, value in layers:
        builder.region_box(x_min, x_max, value=value)

    return builder.build()
