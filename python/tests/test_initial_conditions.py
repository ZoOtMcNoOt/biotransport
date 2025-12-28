"""Tests for initial_conditions module."""

import numpy as np
import pytest

import biotransport as bt
from biotransport.initial_conditions import (
    gaussian,
    step,
    uniform,
    circle,
    sinusoidal,
)


class TestGaussian:
    """Tests for gaussian initial condition."""

    def test_gaussian_1d_returns_list(self):
        """Test 1D Gaussian returns a list."""
        mesh = bt.mesh_1d(50)  # 50 cells = 51 nodes
        ic = gaussian(mesh, amplitude=1.0, width=0.1)

        assert isinstance(ic, list)
        assert len(ic) == 51

    def test_gaussian_1d_default_center(self):
        """Test 1D Gaussian with default center at domain midpoint."""
        mesh = bt.mesh_1d(50)
        ic = gaussian(mesh, amplitude=1.0, width=0.1)
        ic_arr = np.array(ic)

        # Maximum should be near center (around index 25)
        max_idx = np.argmax(ic_arr)
        assert 20 <= max_idx <= 30

    def test_gaussian_1d_custom_center(self):
        """Test 1D Gaussian with custom center."""
        mesh = bt.mesh_1d(100)
        ic = gaussian(mesh, amplitude=2.0, width=0.1, center=0.25)
        ic_arr = np.array(ic)

        # Maximum should be near x=0.25 (around index 25)
        max_idx = np.argmax(ic_arr)
        assert 20 <= max_idx <= 30

    def test_gaussian_1d_amplitude(self):
        """Test Gaussian amplitude is correct."""
        mesh = bt.mesh_1d(100, 0.0, 10.0)
        ic = gaussian(mesh, amplitude=5.0, width=1.0, center=5.0)
        ic_arr = np.array(ic)

        # Peak should be close to amplitude
        assert np.max(ic_arr) == pytest.approx(5.0, rel=0.01)

    def test_gaussian_2d_returns_flat_list(self):
        """Test 2D Gaussian returns flattened list."""
        mesh = bt.mesh_2d(20, 20)  # 20x20 cells = 21x21 nodes
        ic = gaussian(mesh, amplitude=1.0, width=0.1)

        assert isinstance(ic, list)
        assert len(ic) == 21 * 21

    def test_gaussian_2d_custom_center(self):
        """Test 2D Gaussian with custom center."""
        mesh = bt.mesh_2d(20, 20, 0.0, 2.0, 0.0, 2.0)
        ic = gaussian(mesh, amplitude=3.0, width=0.2, center_x=1.0, center_y=1.0)
        ic_arr = np.array(ic)

        assert np.max(ic_arr) == pytest.approx(3.0, rel=0.1)

    def test_gaussian_decays_with_distance(self):
        """Test that Gaussian decays away from center."""
        mesh = bt.mesh_1d(100, 0.0, 10.0)
        ic = gaussian(mesh, amplitude=1.0, width=1.0, center=5.0)
        ic_arr = np.array(ic)

        # Value at center should be larger than at edges
        center_idx = 50
        edge_idx = 0
        assert ic_arr[center_idx] > ic_arr[edge_idx]


class TestStep:
    """Tests for step initial condition."""

    def test_step_1d_returns_list(self):
        """Test step returns a list."""
        mesh = bt.mesh_1d(100)
        ic = step(mesh)

        assert isinstance(ic, list)
        assert len(ic) == 101

    def test_step_1d_basic(self):
        """Test basic 1D step function."""
        mesh = bt.mesh_1d(100)
        ic = step(mesh, position=0.5, left=1.0, right=0.0)
        ic_arr = np.array(ic)

        # Left half should be 1.0, right half 0.0
        assert np.all(ic_arr[:40] == pytest.approx(1.0))
        assert np.all(ic_arr[60:] == pytest.approx(0.0))

    def test_step_1d_custom_position(self):
        """Test step function with custom interface position."""
        mesh = bt.mesh_1d(100)
        ic = step(mesh, position=0.25, left=2.0, right=1.0)
        ic_arr = np.array(ic)

        # Check values on each side
        assert ic_arr[10] == pytest.approx(2.0)  # x = 0.1
        assert ic_arr[50] == pytest.approx(1.0)  # x = 0.5

    def test_step_1d_reversed_values(self):
        """Test step where right > left."""
        mesh = bt.mesh_1d(100)
        ic = step(mesh, position=0.5, left=0.0, right=5.0)
        ic_arr = np.array(ic)

        assert ic_arr[0] == pytest.approx(0.0)
        assert ic_arr[100] == pytest.approx(5.0)


class TestUniform:
    """Tests for uniform initial condition."""

    def test_uniform_1d_returns_list(self):
        """Test uniform returns a list."""
        mesh = bt.mesh_1d(50)
        ic = uniform(mesh, value=3.14)

        assert isinstance(ic, list)
        assert len(ic) == 51

    def test_uniform_1d_values(self):
        """Test uniform IC values are correct."""
        mesh = bt.mesh_1d(50)
        ic = uniform(mesh, value=3.14)
        ic_arr = np.array(ic)

        assert np.all(ic_arr == pytest.approx(3.14))

    def test_uniform_2d(self):
        """Test uniform IC on 2D mesh."""
        mesh = bt.mesh_2d(10, 10)
        ic = uniform(mesh, value=2.718)
        ic_arr = np.array(ic)

        assert len(ic) == 11 * 11
        assert np.all(ic_arr == pytest.approx(2.718))

    def test_uniform_zero(self):
        """Test uniform zero IC."""
        mesh = bt.mesh_1d(100, -1.0, 1.0)
        ic = uniform(mesh, value=0.0)
        ic_arr = np.array(ic)

        assert np.all(ic_arr == 0.0)

    def test_uniform_negative(self):
        """Test uniform negative value."""
        mesh = bt.mesh_1d(20, 0.0, 5.0)
        ic = uniform(mesh, value=-1.5)
        ic_arr = np.array(ic)

        assert np.all(ic_arr == pytest.approx(-1.5))


class TestCircle:
    """Tests for circle initial condition (2D only)."""

    def test_circle_returns_list(self):
        """Test circle returns a list."""
        mesh = bt.mesh_2d(20, 20)
        ic = circle(
            mesh, center_x=0.5, center_y=0.5, radius=0.2, inside=1.0, outside=0.0
        )

        assert isinstance(ic, list)
        assert len(ic) == 21 * 21

    def test_circle_has_both_values(self):
        """Test circular IC has both inside and outside values."""
        mesh = bt.mesh_2d(20, 20)
        ic = circle(
            mesh, center_x=0.5, center_y=0.5, radius=0.2, inside=1.0, outside=0.0
        )
        ic_arr = np.array(ic)

        # Should have both values present - check for any 1.0s and 0.0s
        assert 1.0 in ic_arr  # Should have some inside values
        assert 0.0 in ic_arr  # Should have some outside values

    def test_circle_at_corner(self):
        """Test circle centered at corner."""
        mesh = bt.mesh_2d(20, 20)
        ic = circle(
            mesh, center_x=0.0, center_y=0.0, radius=0.3, inside=5.0, outside=1.0
        )
        ic_arr = np.array(ic)

        # Corner node should be inside
        assert ic_arr[0] == pytest.approx(5.0)

    def test_circle_large_radius(self):
        """Test circle that covers entire domain."""
        mesh = bt.mesh_2d(10, 10)
        ic = circle(
            mesh, center_x=0.5, center_y=0.5, radius=2.0, inside=10.0, outside=0.0
        )
        ic_arr = np.array(ic)

        # All nodes should be inside
        assert np.all(ic_arr == pytest.approx(10.0))


class TestSinusoidal:
    """Tests for sinusoidal initial condition (1D only)."""

    def test_sinusoidal_returns_list(self):
        """Test sinusoidal returns a list."""
        mesh = bt.mesh_1d(100)
        ic = sinusoidal(mesh, periods=1, amplitude=1.0)

        assert isinstance(ic, list)
        assert len(ic) == 101

    def test_sinusoidal_amplitude(self):
        """Test sinusoidal with different amplitude."""
        mesh = bt.mesh_1d(100)
        ic = sinusoidal(mesh, periods=1, amplitude=3.0)
        ic_arr = np.array(ic)

        assert np.max(ic_arr) == pytest.approx(3.0, rel=0.05)
        assert np.min(ic_arr) == pytest.approx(-3.0, rel=0.05)

    def test_sinusoidal_higher_periods(self):
        """Test sinusoidal with higher periods."""
        mesh = bt.mesh_1d(200)
        ic1 = sinusoidal(mesh, periods=1, amplitude=1.0)
        ic2 = sinusoidal(mesh, periods=3, amplitude=1.0)
        ic1_arr = np.array(ic1)
        ic2_arr = np.array(ic2)

        # Higher periods should have more zero crossings
        crossings1 = np.sum(np.diff(np.sign(ic1_arr)) != 0)
        crossings2 = np.sum(np.diff(np.sign(ic2_arr)) != 0)
        assert crossings2 > crossings1

    def test_sinusoidal_with_offset(self):
        """Test sinusoidal with offset."""
        mesh = bt.mesh_1d(100)
        ic = sinusoidal(mesh, periods=1, amplitude=1.0, offset=2.0)
        ic_arr = np.array(ic)

        # Values should oscillate around 2.0
        assert np.mean(ic_arr) == pytest.approx(2.0, rel=0.1)
