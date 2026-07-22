"""Science contracts for the public rheology and fluid-boundary API."""

from __future__ import annotations

import math

import pytest

import biotransport as bt


@pytest.mark.parametrize(
    ("factory", "arguments"),
    [
        (bt.NewtonianModel, (math.nan,)),
        (bt.PowerLawModel, (1.0, 0.5, 0.0)),
        (bt.CarreauModel, (1.0, 0.1, 1.0, 1.1)),
        (bt.CarreauYasudaModel, (1.0, 2.0, 1.0, 2.0, 0.5)),
        (bt.CrossModel, (1.0, 0.001, 1.0, 2.0)),
        (bt.BinghamModel, (0.1, math.inf)),
        (bt.HerschelBulkleyModel, (0.1, 0.5, 0.8, math.nan)),
        (bt.CassonModel, (-0.1, 0.003)),
    ],
)
def test_nonphysical_constitutive_parameters_fail_loudly(
    factory: object, arguments: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError):
        factory(*arguments)  # type: ignore[operator]


def test_casson_stress_is_signed_and_regularized_at_zero() -> None:
    model = bt.CassonModel(0.005, 0.003)

    assert model.shear_stress(0.0) == 0.0
    assert model.shear_stress(-10.0) == pytest.approx(-model.shear_stress(10.0))
    assert math.isfinite(model.viscosity(0.0))
    with pytest.raises(ValueError, match="finite"):
        model.viscosity(math.nan)


def test_cross_model_accepts_monotone_fits_with_exponent_above_one() -> None:
    model = bt.CrossModel(0.056, 0.00345, 1.007, 1.028)

    assert math.isfinite(model.viscosity(100.0))


def test_blood_casson_matches_documented_merrill_parameterization() -> None:
    model = bt.blood_casson_model(0.45)

    assert model.yield_stress() == pytest.approx(0.9e-7 * 39.0**3)
    assert model.plastic_viscosity() == pytest.approx(
        0.0012 * (1.0 + 0.025 * 45.0 + 7.35e-4 * 45.0**2)
    )


@pytest.mark.parametrize("hematocrit", [-0.01, 0.600001, 0.67, math.nan, math.inf])
def test_blood_helpers_enforce_validated_hematocrit_domain(
    hematocrit: float,
) -> None:
    with pytest.raises(ValueError):
        bt.blood_casson_model(hematocrit)
    with pytest.raises(ValueError):
        bt.blood_carreau_model(hematocrit)


def test_reference_blood_carreau_fit_is_recovered_at_45_percent() -> None:
    model = bt.blood_carreau_model(0.45)

    assert model.mu0() == pytest.approx(0.056)
    assert model.mu_inf() == pytest.approx(0.00345)
    assert model.lambda_() == pytest.approx(3.313)
    assert model.n() == pytest.approx(0.3568)


def test_pipe_wall_shear_rate_is_a_validated_magnitude() -> None:
    positive = bt.pipe_wall_shear_rate(2e-9, 1e-3)
    negative = bt.pipe_wall_shear_rate(-2e-9, 1e-3)

    assert positive > 0.0
    assert negative == pytest.approx(positive)
    assert bt.pipe_wall_shear_rate(0.0, 1e-3) == 0.0
    with pytest.raises(ValueError, match="positive"):
        bt.pipe_wall_shear_rate(1e-9, 0.0)
    with pytest.raises(ValueError, match="finite"):
        bt.pipe_wall_shear_rate(math.nan, 1e-3)


def test_stokes_stress_free_condition_does_not_masquerade_as_zero_gradient() -> None:
    mesh = bt.StructuredMesh(4, 4, 0.0, 1.0, 0.0, 1.0)
    solver = bt.StokesSolver(mesh, 0.01)

    with pytest.raises(ValueError, match="traction is not implemented"):
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.stress_free())
