"""Independent checks for public closed-form reference solutions."""

import math

import pytest

import biotransport as bt


def test_diffusion_length_has_one_explicit_convention() -> None:
    diffusivity = 1.7e-9
    time = 23.0
    expected = math.sqrt(diffusivity * time)

    assert bt.analytical.diffusion_length(diffusivity, time) == pytest.approx(expected)
    # Historical alias is retained, but it no longer claims a threshold-defined depth.
    assert bt.analytical.diffusion_penetration_depth(
        diffusivity, time
    ) == pytest.approx(expected)


def test_plane_poiseuille_uses_pressure_gradient_sign_and_wall_values() -> None:
    half_height = 2.0e-3
    viscosity = 3.5e-3
    pressure_gradient = -120.0
    expected_center = -pressure_gradient * half_height**2 / (2.0 * viscosity)

    assert bt.analytical.plane_poiseuille_velocity(
        0.0, half_height, pressure_gradient, viscosity
    ) == pytest.approx(expected_center)
    assert bt.analytical.plane_poiseuille_max_velocity(
        half_height, pressure_gradient, viscosity
    ) == pytest.approx(expected_center)
    assert bt.analytical.plane_poiseuille_velocity(
        half_height, half_height, pressure_gradient, viscosity
    ) == pytest.approx(0.0)


def test_couette_is_linear_between_fixed_and_translating_walls() -> None:
    gap = 4.0e-3
    wall_speed = 0.24

    assert bt.analytical.couette_velocity(0.0, gap, wall_speed) == 0.0
    assert bt.analytical.couette_velocity(gap / 4.0, gap, wall_speed) == pytest.approx(
        wall_speed / 4.0
    )
    assert bt.analytical.couette_velocity(gap, gap, wall_speed) == pytest.approx(
        wall_speed
    )
    assert bt.analytical.couette_max_velocity(wall_speed) == wall_speed


@pytest.mark.parametrize("y", [-1e-6, 4.001e-3])
def test_couette_rejects_points_outside_the_gap(y: float) -> None:
    with pytest.raises(ValueError, match="between 0 and gap_height"):
        bt.analytical.couette_velocity(y, 4.0e-3, 0.24)
