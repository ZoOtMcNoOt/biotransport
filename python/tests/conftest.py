"""
Pytest fixtures for biotransport tests.

Provides reusable test fixtures for common mesh and solver configurations.
"""

import pytest
import numpy as np
import biotransport as bt


# =============================================================================
# Mesh Fixtures
# =============================================================================


@pytest.fixture
def mesh_1d_small():
    """Small 1D mesh for quick tests."""
    return bt.StructuredMesh(10, 0.0, 1.0)


@pytest.fixture
def mesh_1d_medium():
    """Medium 1D mesh for accuracy tests."""
    return bt.StructuredMesh(50, 0.0, 1.0)


@pytest.fixture
def mesh_2d_small():
    """Small 2D mesh (10x10) for quick tests."""
    return bt.StructuredMesh(10, 10, 0.0, 1.0, 0.0, 1.0)


@pytest.fixture
def mesh_2d_medium():
    """Medium 2D mesh (20x20) for accuracy tests."""
    return bt.StructuredMesh(20, 20, 0.0, 1.0, 0.0, 1.0)


@pytest.fixture
def mesh_2d_channel():
    """2D mesh for channel flow (aspect ratio 4:1)."""
    return bt.StructuredMesh(40, 10, 0.0, 4.0, 0.0, 1.0)


@pytest.fixture
def mesh_2d_microfluidic():
    """Microfluidic channel mesh (1mm x 0.5mm)."""
    return bt.StructuredMesh(20, 10, 0.0, 0.001, 0.0, 0.0005)


# =============================================================================
# Stokes Solver Fixtures
# =============================================================================


@pytest.fixture
def stokes_solver_lid_driven(mesh_2d_small):
    """Stokes solver configured for lid-driven cavity."""
    solver = bt.StokesSolver(mesh_2d_small, 0.001)
    solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.dirichlet(1.0, 0.0))
    solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.no_slip())
    return solver


@pytest.fixture
def stokes_solver_channel(mesh_2d_channel):
    """Stokes solver configured for channel flow."""
    solver = bt.StokesSolver(mesh_2d_channel, 0.001)
    solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(1.0))
    solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())
    return solver


# =============================================================================
# Navier-Stokes Solver Fixtures
# =============================================================================


@pytest.fixture
def navier_stokes_water():
    """Navier-Stokes solver with water properties."""
    mesh = bt.StructuredMesh(20, 10, 0.0, 0.001, 0.0, 0.0005)
    rho = 1000.0  # kg/m^3
    mu = 0.001  # Pa.s
    return bt.NavierStokesSolver(mesh, rho, mu)


@pytest.fixture
def navier_stokes_channel(mesh_2d_microfluidic):
    """Navier-Stokes solver for microfluidic channel."""
    rho = 1000.0
    mu = 0.001
    solver = bt.NavierStokesSolver(mesh_2d_microfluidic, rho, mu)
    solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(0.1))
    solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())
    return solver


# =============================================================================
# Diffusion Solver Fixtures
# =============================================================================


@pytest.fixture
def diffusion_solver_1d(mesh_1d_medium):
    """1D diffusion solver with default settings."""
    D = 1e-9  # m^2/s
    return bt.DiffusionSolver(mesh_1d_medium, D)


@pytest.fixture
def diffusion_solver_2d(mesh_2d_small):
    """2D diffusion solver with default settings."""
    D = 1e-9  # m^2/s
    return bt.DiffusionSolver(mesh_2d_small, D)


# =============================================================================
# Initial Condition Helpers
# =============================================================================


@pytest.fixture
def gaussian_ic_1d():
    """Factory for 1D Gaussian initial condition."""

    def _make(mesh, amplitude=1.0, center=0.5, width=0.1):
        x = np.array([mesh.x(i) for i in range(mesh.num_nodes())])
        return amplitude * np.exp(-(((x - center) / width) ** 2))

    return _make


@pytest.fixture
def gaussian_ic_2d():
    """Factory for 2D Gaussian initial condition."""

    def _make(mesh, amplitude=1.0, cx=0.5, cy=0.5, width=0.1):
        ic = np.zeros(mesh.num_nodes())
        for j in range(mesh.ny() + 1):
            for i in range(mesh.nx() + 1):
                x = mesh.x(i)
                y = mesh.y(i, j)
                r2 = (x - cx) ** 2 + (y - cy) ** 2
                ic[mesh.index(i, j)] = amplitude * np.exp(-r2 / width**2)
        return ic

    return _make


# =============================================================================
# Validation Helpers
# =============================================================================


@pytest.fixture
def assert_no_nan_inf():
    """Fixture that returns a validation function for NaN/Inf checking."""

    def _check(array, name="array"):
        arr = np.asarray(array)
        assert not np.any(np.isnan(arr)), f"{name} contains NaN values"
        assert not np.any(np.isinf(arr)), f"{name} contains Inf values"

    return _check


@pytest.fixture
def assert_bounded():
    """Fixture that returns a validation function for bounds checking."""

    def _check(array, min_val, max_val, name="array"):
        arr = np.asarray(array)
        assert np.all(arr >= min_val), f"{name} has values below {min_val}"
        assert np.all(arr <= max_val), f"{name} has values above {max_val}"

    return _check
