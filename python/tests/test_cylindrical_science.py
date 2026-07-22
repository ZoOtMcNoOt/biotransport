"""Independent science contracts for cylindrical coordinates."""

from __future__ import annotations

import math

import numpy as np
import pytest

import biotransport as bt


def test_periodic_full_mesh_has_no_duplicate_theta_endpoint() -> None:
    mesh = bt.CylindricalMesh(3, 12, 4, 0.2, 1.0, -math.pi, math.pi, -1.0, 2.0)

    assert mesh.num_nodes() == 4 * 12 * 5
    assert mesh.num_cells() == 3 * 12 * 4
    assert mesh.theta(0) == pytest.approx(-math.pi)
    assert mesh.theta(11) == pytest.approx(-math.pi + 11 * 2 * math.pi / 12)
    with pytest.raises(IndexError, match="periodic endpoint|ntheta"):
        mesh.theta(12)


def test_full_mesh_rejects_partial_wedges_and_undersampled_periodicity() -> None:
    with pytest.raises(ValueError, match="complete turn"):
        bt.CylindricalMesh(3, 12, 4, 0.2, 1.0, 0.0, math.pi, 0.0, 1.0)
    with pytest.raises(ValueError, match="at least 3"):
        bt.CylindricalMesh(3, 2, 4, 0.2, 1.0, 0.0, 2 * math.pi, 0.0, 1.0)


@pytest.mark.parametrize("rmin", [-1.0, -np.inf, np.nan])
def test_radius_domain_validation(rmin: float) -> None:
    with pytest.raises(ValueError):
        bt.CylindricalMesh(8, rmin, 1.0)


def test_nodal_measures_integrate_exact_domain_volume() -> None:
    axisymmetric = bt.CylindricalMesh(9, 11, 0.4, 1.7, -0.3, 2.2)
    volume = sum(
        axisymmetric.cell_volume(i, 0, k)
        for k in range(axisymmetric.nz() + 1)
        for i in range(axisymmetric.nr() + 1)
    )
    exact = math.pi * (1.7**2 - 0.4**2) * 2.5
    assert volume == pytest.approx(exact, rel=2e-14)

    full = bt.CylindricalMesh(7, 16, 8, 0.2, 1.3, 0.4, 0.4 + 2 * math.pi, -0.5, 1.5)
    volume = sum(
        full.cell_volume(i, j, k)
        for k in range(full.nz() + 1)
        for j in range(full.ntheta())
        for i in range(full.nr() + 1)
    )
    exact = math.pi * (1.3**2 - 0.2**2) * 2.0
    assert volume == pytest.approx(exact, rel=2e-13)


def test_axis_limit_and_metric_operators_are_polynomial_exact() -> None:
    mesh = bt.CylindricalMesh(10, 9, 0.0, 2.0, -1.0, 2.0)
    r = np.array([mesh.r(i) for i in range(mesh.nr() + 1)])
    z = np.array([mesh.z(k) for k in range(mesh.nz() + 1)])
    radius, axial = np.meshgrid(r, z)

    # C-order flattening matches the library's radial-fastest storage contract.
    phi = radius**2 + 3.0 * axial**2
    grad_r = np.asarray(mesh.gradient_r(phi.ravel())).reshape(phi.shape)
    grad_z = np.asarray(mesh.gradient_z(phi.ravel())).reshape(phi.shape)
    laplacian = np.asarray(mesh.laplacian(phi.ravel())).reshape(phi.shape)

    np.testing.assert_allclose(grad_r, 2.0 * radius, atol=2e-13)
    np.testing.assert_allclose(grad_z, 6.0 * axial, atol=2e-13)
    np.testing.assert_allclose(laplacian, 10.0, atol=3e-12)


def test_axisymmetric_divergence_uses_regular_axis_limit() -> None:
    mesh = bt.CylindricalMesh(10, 9, 0.0, 2.0, -1.0, 2.0)
    r = np.array([mesh.r(i) for i in range(mesh.nr() + 1)])
    z = np.array([mesh.z(k) for k in range(mesh.nz() + 1)])
    radius, axial = np.meshgrid(r, z)

    divergence = np.asarray(
        mesh.divergence((2 * radius).ravel(), (-0.5 * axial).ravel())
    )
    np.testing.assert_allclose(divergence, 3.5, atol=2e-13)

    invalid_vr = 2 * radius
    invalid_vr[:, 0] = 1.0
    with pytest.raises(ValueError, match="vr=0 at r=0"):
        mesh.divergence(invalid_vr.ravel(), (-0.5 * axial).ravel())


def test_full_periodic_metric_operators() -> None:
    mesh = bt.CylindricalMesh(4, 128, 4, 0.75, 1.5, -math.pi, math.pi, -0.4, 0.8)
    r = np.array([mesh.r(i) for i in range(mesh.nr() + 1)])
    theta = np.array([mesh.theta(j) for j in range(mesh.ntheta())])
    z = np.array([mesh.z(k) for k in range(mesh.nz() + 1)])
    axial, angle, radius = np.meshgrid(z, theta, r, indexing="ij")

    phi = radius**2 + axial**2 + np.cos(angle)
    gradient_theta = np.asarray(mesh.gradient_theta(phi.ravel())).reshape(phi.shape)
    laplacian = np.asarray(mesh.laplacian(phi.ravel())).reshape(phi.shape)
    divergence = np.asarray(
        mesh.divergence(radius.ravel(), np.sin(angle).ravel(), axial.ravel())
    ).reshape(phi.shape)

    np.testing.assert_allclose(gradient_theta, -np.sin(angle) / radius, atol=8e-4)
    np.testing.assert_allclose(laplacian, 6.0 - np.cos(angle) / radius**2, atol=1.5e-3)
    np.testing.assert_allclose(divergence, 3.0 + np.cos(angle) / radius, atol=8e-4)


def test_operator_inputs_fail_loudly() -> None:
    mesh = bt.CylindricalMesh(8, 0.0, 1.0)
    with pytest.raises(ValueError, match="exactly"):
        mesh.laplacian(np.ones(3))
    bad = np.ones(mesh.num_nodes())
    bad[2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        mesh.gradient_r(bad)

    full_with_axis = bt.CylindricalMesh(4, 16, 4, 0.0, 1.0, 0.0, 2 * math.pi, 0.0, 1.0)
    with pytest.raises(ValueError, match="rmin > 0|annulus"):
        full_with_axis.laplacian(np.ones(full_with_axis.num_nodes()))
