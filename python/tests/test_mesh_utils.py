"""Tests for mesh_utils module."""

import numpy as np
import pytest

import biotransport as bt
from biotransport.mesh_utils import (
    x_nodes,
    y_nodes,
    xy_grid,
    as_1d,
    as_2d,
    mesh_1d,
    mesh_2d,
    r_nodes,
    z_nodes,
    rz_grid,
)


class TestXNodes:
    """Tests for x_nodes function."""

    def test_x_nodes_1d(self):
        """Test x_nodes on 1D mesh."""
        mesh = bt.mesh_1d(10, 0.0, 1.0)  # 10 cells = 11 nodes
        x = x_nodes(mesh)

        assert x.shape == (11,)
        assert x[0] == pytest.approx(0.0)
        assert x[-1] == pytest.approx(1.0)
        # Check uniform spacing
        assert np.allclose(np.diff(x), 0.1)

    def test_x_nodes_2d(self):
        """Test x_nodes on 2D mesh."""
        mesh = bt.mesh_2d(4, 2, 0.0, 2.0, 0.0, 1.0)  # 4x2 cells = 5x3 nodes
        x = x_nodes(mesh)

        # Should return x coordinates of first row (nx+1 values)
        assert x.shape == (5,)
        assert x[0] == pytest.approx(0.0)
        assert x[-1] == pytest.approx(2.0)


class TestYNodes:
    """Tests for y_nodes function."""

    def test_y_nodes_2d(self):
        """Test y_nodes on 2D mesh."""
        mesh = bt.mesh_2d(4, 6, 0.0, 1.0, 0.0, 3.0)  # 4x6 cells = 5x7 nodes
        y = y_nodes(mesh)

        assert y.shape == (7,)
        assert y[0] == pytest.approx(0.0)
        assert y[-1] == pytest.approx(3.0)

    def test_y_nodes_1d_raises(self):
        """Test y_nodes raises on 1D mesh."""
        mesh = bt.mesh_1d(10)

        with pytest.raises(ValueError, match="only valid for 2D"):
            y_nodes(mesh)


class TestXYGrid:
    """Tests for xy_grid function."""

    def test_xy_grid_basic(self):
        """Test xy_grid returns proper meshgrid."""
        mesh = bt.mesh_2d(4, 3, 0.0, 1.0, 0.0, 2.0)  # 4x3 cells = 5x4 nodes
        X, Y = xy_grid(mesh)

        assert X.shape == (4, 5)  # (ny+1, nx+1)
        assert Y.shape == (4, 5)

        # Check X varies along columns, Y along rows
        assert np.all(X[0, :] == X[1, :])  # Same x for different y
        assert np.all(Y[:, 0] == Y[:, 1])  # Same y for different x

    def test_xy_grid_values(self):
        """Test xy_grid coordinate values."""
        mesh = bt.mesh_2d(2, 4, -1.0, 1.0, -2.0, 2.0)  # 2x4 cells = 3x5 nodes
        X, Y = xy_grid(mesh)

        # Check corners
        assert X[0, 0] == pytest.approx(-1.0)  # Bottom-left x
        assert X[0, -1] == pytest.approx(1.0)  # Bottom-right x
        assert Y[0, 0] == pytest.approx(-2.0)  # Bottom y
        assert Y[-1, 0] == pytest.approx(2.0)  # Top y

    def test_xy_grid_1d_raises(self):
        """Test xy_grid raises on 1D mesh."""
        mesh = bt.mesh_1d(10)

        with pytest.raises(ValueError, match="only valid for 2D"):
            xy_grid(mesh)


class TestAs1d:
    """Tests for as_1d function."""

    def test_as_1d_from_list(self):
        """Test as_1d converts list to array."""
        mesh = bt.mesh_1d(5)  # 6 nodes
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = as_1d(mesh, data)

        assert isinstance(result, np.ndarray)
        assert result.shape == (6,)
        assert result.dtype == np.float64

    def test_as_1d_from_2d_array(self):
        """Test as_1d flattens 2D array."""
        mesh = bt.mesh_2d(2, 2)  # 3x3 = 9 nodes
        data_2d = np.ones((3, 3))
        result = as_1d(mesh, data_2d)

        assert result.shape == (9,)

    def test_as_1d_wrong_length_raises(self):
        """Test as_1d raises on wrong length."""
        mesh = bt.mesh_1d(10)  # 11 nodes
        data = [1.0, 2.0, 3.0]  # Wrong length

        with pytest.raises(ValueError, match="Expected 11 values"):
            as_1d(mesh, data)


class TestAs2d:
    """Tests for as_2d function."""

    def test_as_2d_from_flat(self):
        """Test as_2d reshapes flat array."""
        mesh = bt.mesh_2d(3, 2)  # 4x3 = 12 nodes
        data_1d = np.arange(12, dtype=np.float64)
        result = as_2d(mesh, data_1d)

        assert result.shape == (3, 4)  # (ny+1, nx+1)

    def test_as_2d_from_2d(self):
        """Test as_2d validates already 2D array."""
        mesh = bt.mesh_2d(3, 2)  # 4x3 = 12 nodes
        data_2d = np.arange(12).reshape((3, 4))
        result = as_2d(mesh, data_2d)

        assert result.shape == (3, 4)
        assert np.array_equal(result, data_2d)

    def test_as_2d_round_trip(self):
        """Test round-trip conversion."""
        mesh = bt.mesh_2d(4, 3)  # 5x4 = 20 nodes
        original = np.arange(20).reshape(4, 5)
        flat = as_1d(mesh, original)
        restored = as_2d(mesh, flat)

        assert np.array_equal(original, restored)

    def test_as_2d_1d_mesh_raises(self):
        """Test as_2d raises on 1D mesh."""
        mesh = bt.mesh_1d(10)
        data = np.ones(11)

        with pytest.raises(ValueError, match="only valid for 2D"):
            as_2d(mesh, data)


class TestMesh1d:
    """Tests for mesh_1d convenience function."""

    def test_mesh_1d_basic(self):
        """Test basic 1D mesh creation."""
        mesh = mesh_1d(100, 0.0, 10.0)  # 100 cells = 101 nodes

        assert mesh.is_1d()
        assert mesh.num_nodes() == 101

    def test_mesh_1d_bounds(self):
        """Test mesh bounds are correct."""
        mesh = mesh_1d(50, -5.0, 5.0)  # 50 cells = 51 nodes
        x = x_nodes(mesh)

        assert x[0] == pytest.approx(-5.0)
        assert x[-1] == pytest.approx(5.0)

    def test_mesh_1d_negative_domain(self):
        """Test mesh with negative coordinates."""
        mesh = mesh_1d(20, -10.0, -5.0)  # 20 cells = 21 nodes
        x = x_nodes(mesh)

        assert x[0] == pytest.approx(-10.0)
        assert x[-1] == pytest.approx(-5.0)

    def test_mesh_1d_default_bounds(self):
        """Test mesh_1d with default bounds."""
        mesh = mesh_1d(10)  # Default 0.0 to 1.0
        x = x_nodes(mesh)

        assert x[0] == pytest.approx(0.0)
        assert x[-1] == pytest.approx(1.0)


class TestMesh2d:
    """Tests for mesh_2d convenience function."""

    def test_mesh_2d_basic(self):
        """Test basic 2D mesh creation."""
        mesh = mesh_2d(20, 10, 0.0, 2.0, 0.0, 1.0)

        assert not mesh.is_1d()
        assert mesh.num_nodes() == 21 * 11

    def test_mesh_2d_bounds(self):
        """Test mesh bounds are correct."""
        mesh = mesh_2d(5, 5, -1.0, 1.0, -1.0, 1.0)
        x = x_nodes(mesh)
        y = y_nodes(mesh)

        assert x[0] == pytest.approx(-1.0)
        assert x[-1] == pytest.approx(1.0)
        assert y[0] == pytest.approx(-1.0)
        assert y[-1] == pytest.approx(1.0)

    def test_mesh_2d_default_bounds(self):
        """Test mesh_2d with default bounds."""
        mesh = mesh_2d(10, 10)  # Default 0.0 to 1.0
        x = x_nodes(mesh)
        y = y_nodes(mesh)

        assert x[0] == pytest.approx(0.0)
        assert x[-1] == pytest.approx(1.0)
        assert y[0] == pytest.approx(0.0)
        assert y[-1] == pytest.approx(1.0)


class TestCylindricalMesh:
    """Tests for cylindrical mesh utilities."""

    def test_r_nodes(self):
        """Test r_nodes on cylindrical mesh."""
        mesh = bt.CylindricalMesh(10, 20, 0.0, 1.0, 0.0, 2.0)
        r = r_nodes(mesh)

        assert r.shape == (11,)  # 10 cells = 11 nodes
        assert r[0] == pytest.approx(0.0)
        assert r[-1] == pytest.approx(1.0)

    def test_z_nodes(self):
        """Test z_nodes on cylindrical mesh."""
        mesh = bt.CylindricalMesh(10, 20, 0.0, 1.0, 0.0, 2.0)
        z = z_nodes(mesh)

        assert z.shape == (21,)  # 20 cells = 21 nodes
        assert z[0] == pytest.approx(0.0)
        assert z[-1] == pytest.approx(2.0)

    def test_rz_grid(self):
        """Test rz_grid returns proper meshgrid."""
        mesh = bt.CylindricalMesh(5, 10, 0.0, 0.5, 0.0, 1.0)
        R, Z = rz_grid(mesh)

        assert R.shape == (11, 6)  # (nz+1, nr+1)
        assert Z.shape == (11, 6)
        assert R[0, 0] == pytest.approx(0.0)
        assert R[0, -1] == pytest.approx(0.5)
        assert Z[0, 0] == pytest.approx(0.0)
        assert Z[-1, 0] == pytest.approx(1.0)
