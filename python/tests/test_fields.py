"""Tests for fields module (SpatialField builder)."""

import numpy as np
import pytest

import biotransport as bt
from biotransport.fields import SpatialField


class TestSpatialFieldInit:
    """Tests for SpatialField initialization."""

    def test_init_1d_mesh(self):
        """Test initialization with 1D mesh."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)  # 50 cells = 51 nodes
        field = SpatialField(mesh)

        assert field.mesh is mesh
        assert field._field.shape == (51,)
        assert np.all(field._field == 0.0)

    def test_init_2d_mesh(self):
        """Test initialization with 2D mesh."""
        mesh = bt.mesh_2d(10, 10, 0.0, 1.0, 0.0, 1.0)  # 10x10 cells = 11x11 nodes
        field = SpatialField(mesh)

        assert field.mesh is mesh
        assert field._field.shape == (11 * 11,)


class TestSpatialFieldDefault:
    """Tests for SpatialField.default() method."""

    def test_default_sets_all_values(self):
        """Test that default sets all field values."""
        mesh = bt.mesh_1d(100, 0.0, 1.0)  # 100 cells = 101 nodes
        field = SpatialField(mesh).default(5.0)

        assert np.all(field._field == 5.0)
        assert field._default_value == 5.0

    def test_default_returns_self(self):
        """Test that default returns self for chaining."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)  # 50 cells = 51 nodes
        field = SpatialField(mesh)
        result = field.default(1.0)

        assert result is field


class TestSpatialFieldRegionBox1D:
    """Tests for region_box on 1D meshes."""

    def test_region_box_1d_basic(self):
        """Test basic 1D region box."""
        mesh = bt.mesh_1d(100, 0.0, 1.0)  # 100 cells = 101 nodes
        field = SpatialField(mesh).default(1.0).region_box(0.4, 0.6, value=2.0).build()

        # Check values in different regions
        x = np.linspace(0.0, 1.0, 101)
        for i, xi in enumerate(x):
            if 0.4 <= xi <= 0.6:
                assert field[i] == pytest.approx(2.0)
            else:
                assert field[i] == pytest.approx(1.0)

    def test_region_box_1d_at_boundary(self):
        """Test region box at domain boundary."""
        mesh = bt.mesh_1d(10, 0.0, 1.0)  # 10 cells = 11 nodes
        field = SpatialField(mesh).default(0.0).region_box(0.0, 0.2, value=1.0).build()

        # First few nodes should be 1.0
        assert field[0] == pytest.approx(1.0)
        assert field[2] == pytest.approx(1.0)
        # Nodes past 0.2 should be 0.0
        assert field[5] == pytest.approx(0.0)

    def test_region_box_1d_multiple_regions(self):
        """Test multiple region boxes."""
        mesh = bt.mesh_1d(100, 0.0, 1.0)  # 100 cells = 101 nodes
        field = (
            SpatialField(mesh)
            .default(1.0)
            .region_box(0.1, 0.2, value=2.0)
            .region_box(0.8, 0.9, value=3.0)
            .build()
        )

        # Check middle region is default
        assert field[50] == pytest.approx(1.0)
        # Check first region
        assert field[15] == pytest.approx(2.0)
        # Check second region
        assert field[85] == pytest.approx(3.0)

    def test_region_box_1d_overlapping_last_wins(self):
        """Test that overlapping regions use last value."""
        mesh = bt.mesh_1d(100, 0.0, 1.0)  # 100 cells = 101 nodes
        field = (
            SpatialField(mesh)
            .default(0.0)
            .region_box(0.3, 0.7, value=1.0)
            .region_box(0.4, 0.6, value=2.0)
            .build()
        )

        # Overlap region should have second value
        assert field[50] == pytest.approx(2.0)
        # Non-overlap but in first region
        assert field[35] == pytest.approx(1.0)

    def test_region_box_1d_rejects_y_params(self):
        """Test that y params raise error for 1D mesh."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)  # 50 cells = 51 nodes
        field = SpatialField(mesh).default(1.0)

        with pytest.raises(ValueError, match="y_min and y_max should not be provided"):
            field.region_box(0.2, 0.4, 0.0, 0.5, value=2.0)


class TestSpatialFieldRegionBox2D:
    """Tests for region_box on 2D meshes."""

    def test_region_box_2d_basic(self):
        """Test basic 2D region box."""
        mesh = bt.mesh_2d(10, 10, 0.0, 1.0, 0.0, 1.0)  # 10x10 cells = 11x11 nodes
        field = (
            SpatialField(mesh)
            .default(0.0)
            .region_box(0.3, 0.7, 0.3, 0.7, value=1.0)
            .build()
        )

        # Center should be 1.0
        center_idx = 5 * 11 + 5  # Row 5, col 5
        assert field[center_idx] == pytest.approx(1.0)

        # Corner should be 0.0
        assert field[0] == pytest.approx(0.0)

    def test_region_box_2d_requires_y_params(self):
        """Test that 2D mesh requires y params."""
        mesh = bt.mesh_2d(10, 10, 0.0, 1.0, 0.0, 1.0)  # 10x10 cells = 11x11 nodes
        field = SpatialField(mesh).default(1.0)

        with pytest.raises(ValueError, match="y_min and y_max required"):
            field.region_box(0.2, 0.4, value=2.0)

    def test_region_box_2d_multiple_regions(self):
        """Test multiple 2D region boxes."""
        mesh = bt.mesh_2d(10, 10, 0.0, 1.0, 0.0, 1.0)  # 10x10 cells = 11x11 nodes
        field = (
            SpatialField(mesh)
            .default(0.0)
            .region_box(0.0, 0.3, 0.0, 0.3, value=1.0)  # Bottom-left
            .region_box(0.7, 1.0, 0.7, 1.0, value=2.0)  # Top-right
            .build()
        )

        # Bottom-left corner
        assert field[0] == pytest.approx(1.0)
        # Top-right corner
        assert field[-1] == pytest.approx(2.0)
        # Center should be default
        center_idx = 5 * 11 + 5
        assert field[center_idx] == pytest.approx(0.0)


class TestSpatialFieldBuild:
    """Tests for SpatialField.build() method."""

    def test_build_returns_list(self):
        """Test that build returns a list (for C++ binding compatibility)."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)  # 50 cells = 51 nodes
        field = SpatialField(mesh).default(1.0).build()

        # SpatialField.build() returns a list for compatibility with C++ bindings
        assert isinstance(field, list)
        assert len(field) == 51
        assert all(v == 1.0 for v in field)

    def test_build_returns_copy(self):
        """Test that build returns a copy, not reference."""
        mesh = bt.mesh_1d(10, 0.0, 1.0)  # 10 cells = 11 nodes
        builder = SpatialField(mesh).default(1.0)
        field1 = builder.build()
        field2 = builder.build()

        # Modify one, should not affect other
        field1[0] = 999.0
        assert field2[0] != 999.0


class TestSpatialFieldChaining:
    """Tests for method chaining patterns."""

    def test_full_chain(self):
        """Test complete method chain."""
        mesh = bt.mesh_1d(100, 0.0, 10.0)  # 100 cells = 101 nodes
        field = (
            SpatialField(mesh)
            .default(1e-9)  # Default diffusivity
            .region_box(3.0, 4.0, value=1e-12)  # Membrane region
            .region_box(6.0, 7.0, value=1e-12)  # Another membrane
            .build()
        )

        # Check diffusivity profile
        assert field[0] == pytest.approx(1e-9)  # Before first membrane
        assert field[35] == pytest.approx(1e-12)  # In first membrane
        assert field[50] == pytest.approx(1e-9)  # Between membranes
        assert field[65] == pytest.approx(1e-12)  # In second membrane
        assert field[-1] == pytest.approx(1e-9)  # After second membrane


class TestSpatialFieldIntegration:
    """Integration tests with solvers."""

    def test_field_with_diffusion_problem(self):
        """Test using SpatialField with a diffusion problem."""
        mesh = bt.mesh_1d(50, 0.0, 1.0)  # 50 cells = 51 nodes

        # Create spatially varying diffusivity
        D = (
            SpatialField(mesh)
            .default(1e-5)
            .region_box(0.4, 0.6, value=1e-7)  # Low diffusivity region
            .build()
        )

        # This should work with the solver
        problem = bt.DiffusionProblem(mesh)
        problem.diffusivity_field(D)

        # Check field was set (build returns list, not numpy array)
        assert len(D) == 51
