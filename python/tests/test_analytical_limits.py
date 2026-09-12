"""Independent high-precision and asymptotic checks for exact steady references."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

import biotransport as bt


def _decimal_slab(x, diffusivity, rate, length, surface):
    with localcontext() as context:
        context.prec = 80
        d, k, span, value = map(
            Decimal.from_float, (diffusivity, rate, length, surface)
        )
        modulus = (k / d).sqrt()

        def cosh(argument):
            return (argument.exp() + (-argument).exp()) / 2

        denominator = cosh(modulus * span)
        return np.array(
            [
                float(
                    value
                    * cosh(modulus * (span - Decimal.from_float(float(position))))
                    / denominator
                )
                for position in x
            ]
        )


@pytest.mark.parametrize("surface", [2.5, 1e250, -2.5])
def test_high_thiele_profile_matches_decimal_hyperbolic_solution(surface):
    x = np.array([0.0, 0.001, 0.01, 0.5, 0.9, 1.0])
    expected = _decimal_slab(x, 1e-6, 1.0, 1.0, surface)
    result = bt.analytical.steady_slab_first_order(
        x, D=1e-6, k=1.0, L=1.0, c_surface=surface
    )
    np.testing.assert_allclose(result, expected, rtol=2e-13, atol=0)
    assert result[0] == surface


def test_reference_avoids_overflow_in_the_diffusivity_ratio():
    length = 1e-155
    x = np.array([0.0, length / 2, length])
    expected = _decimal_slab(x, 1e-310, 1.0, length, 1.0)
    result = bt.analytical.steady_slab_first_order(
        x, D=1e-310, k=1.0, L=length, c_surface=1.0
    )
    np.testing.assert_allclose(result, expected, rtol=2e-14, atol=0)
    assert bt.analytical.thiele_modulus(
        D=1e-310, k=1.0, length=length
    ) == pytest.approx(1.0)


def test_thiele_modulus_can_remain_finite_when_inverse_length_overflows():
    diffusivity, rate, length = 5e-324, 1e308, 1e-310
    with localcontext() as context:
        context.prec = 80
        expected = float(
            (Decimal.from_float(rate) / Decimal.from_float(diffusivity)).sqrt()
            * Decimal.from_float(length)
        )
    actual = bt.analytical.thiele_modulus(D=diffusivity, k=rate, length=length)
    assert actual == pytest.approx(expected, rel=2e-14)


@pytest.mark.parametrize("phi", [1e-8, 1e-4, 0.05, 0.0999, 0.1, 1.0, 20.0])
def test_sphere_effectiveness_matches_decimal_closed_form(phi):
    with localcontext() as context:
        context.prec = 80
        p = Decimal.from_float(phi)
        exponential = (2 * p).exp()
        coth = (exponential + 1) / (exponential - 1)
        expected = float(3 * (p * coth - 1) / (p * p))
    assert bt.analytical.effectiveness_factor(phi, "sphere") == pytest.approx(
        expected, rel=3e-14
    )


def _decimal_cylinder_effectiveness(phi):
    # From the positive defining power series for I0 and I1. Their common
    # phi/2 prefactor cancels in 2*I1/(phi*I0). No SciPy Bessel evaluation is used.
    with localcontext() as context:
        context.prec = 80
        p = Decimal.from_float(phi)
        square = p * p / 4
        term_zero = term_one = total_zero = total_one = Decimal(1)
        for order in range(1, 3000):
            term_zero *= square / (order * order)
            term_one *= square / (order * (order + 1))
            total_zero += term_zero
            total_one += term_one
            if order > phi and max(
                term_zero / total_zero, term_one / total_one
            ) < Decimal("1e-65"):
                return float(total_one / total_zero)
    raise AssertionError("high-precision Bessel reference did not converge")


@pytest.mark.parametrize("phi", [1.0, 25.0, 700.0, 1000.0])
def test_cylinder_effectiveness_matches_decimal_bessel_series(phi):
    expected = _decimal_cylinder_effectiveness(phi)
    actual = bt.analytical.effectiveness_factor(phi, "cylinder")
    assert actual == pytest.approx(expected, rel=2e-14)
    assert np.isfinite(actual) and 0 < actual <= 1


@pytest.mark.parametrize(
    "geometry, coefficient", [("slab", 1.0), ("cylinder", 2.0), ("sphere", 3.0)]
)
def test_effectiveness_keeps_large_and_small_modulus_limits(geometry, coefficient):
    assert bt.analytical.effectiveness_factor(5e-324, geometry) == 1.0
    value = bt.analytical.effectiveness_factor(1e300, geometry)
    assert value > 0 and np.isfinite(value)
    assert value * 1e300 == pytest.approx(coefficient, rel=2e-14)


@pytest.mark.parametrize("phi", [np.nan, np.inf, -np.inf, -1e-10])
def test_effectiveness_rejects_invalid_modulus(phi):
    with pytest.raises(ValueError, match="finite and non-negative"):
        bt.analytical.effectiveness_factor(phi)


def test_zero_reaction_limit_still_validates_geometry():
    with pytest.raises(ValueError, match="geometry"):
        bt.analytical.effectiveness_factor(0.0, "cube")


@pytest.mark.parametrize("x", [-0.001, 1.001])
def test_slab_reference_rejects_depths_outside_its_domain(x):
    with pytest.raises(ValueError, match=r"\[0, L\]"):
        bt.analytical.steady_slab_first_order(x, D=1.0, k=1.0, L=1.0, c_surface=1.0)


@pytest.mark.parametrize("cells, length", [(11, 0.1), (37, 1e-4), (149, 0.01)])
def test_slab_reference_accepts_roundoff_at_native_mesh_endpoint(cells, length):
    mesh = bt.mesh_1d(cells, 0.0, length)
    coordinates = bt.x_nodes(mesh)
    profile = bt.analytical.steady_slab_first_order(
        coordinates, D=1e-9, k=1e-4, L=length, c_surface=1.0
    )
    endpoint = bt.analytical.steady_slab_first_order(
        length, D=1e-9, k=1e-4, L=length, c_surface=1.0
    )
    assert np.all(np.isfinite(profile))
    assert profile[-1] == pytest.approx(endpoint, rel=1e-14, abs=0)


def _decimal_radial(positions, diffusivity, rate, radius, surface, geometry):
    with localcontext() as context:
        context.prec = 80
        d, k, span, value = map(
            Decimal.from_float, (diffusivity, rate, radius, surface)
        )
        modulus = (k / d).sqrt()

        def radial_basis(z):
            if geometry == "sphere":
                return (z.exp() - (-z).exp()) / (2 * z) if z else Decimal(1)
            # Positive defining power series, independently of SciPy's i0e.
            square = z * z / 4
            term = total = Decimal(1)
            for order in range(1, 3000):
                term *= square / (order * order)
                total += term
                if order > z and term / total < Decimal("1e-65"):
                    return total
            raise AssertionError("high-precision Bessel reference did not converge")

        denominator = radial_basis(modulus * span)
        return np.array(
            [
                float(
                    value
                    * radial_basis(modulus * Decimal.from_float(float(r)))
                    / denominator
                )
                for r in positions
            ]
        )


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
@pytest.mark.parametrize(
    "phi, surface", [(1e-8, 2.5), (0.0999, -2.5), (1.0, 2.5), (1000.0, 1e250)]
)
def test_radial_reference_matches_independent_high_precision_basis(
    geometry, phi, surface
):
    radii = np.array([0.0, 1e-12, 0.1, 0.5, 0.9, 1.0])
    expected = _decimal_radial(radii, 1.0, phi**2, 1.0, surface, geometry)
    actual = bt.analytical.steady_radial_first_order(
        radii, D=1.0, k=phi**2, R=1.0, c_surface=surface, geometry=geometry
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=0)
    assert actual[-1] == surface


@pytest.mark.parametrize("geometry", ["sphere", "cylinder", "spherical", "cylindrical"])
def test_radial_reference_zero_reaction_and_scalar_api(geometry):
    value = bt.analytical.steady_radial_first_order(
        0.0, D=1.0, k=0.0, R=1.0, c_surface=2.0, geometry=geometry
    )
    assert type(value) is float and value == 2.0


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
def test_radial_reference_keeps_origin_and_surface_limits_at_extreme_modulus(geometry):
    result = bt.analytical.steady_radial_first_order(
        [0.0, 0.5, 1.0],
        D=1e-310,
        k=1.0,
        R=1.0,
        c_surface=np.finfo(float).max,
        geometry=geometry,
    )
    np.testing.assert_array_equal(result, [0.0, 0.0, np.finfo(float).max])
    tiny_radius = 1e-155
    r = np.array([0.0, tiny_radius / 2, tiny_radius])
    expected = _decimal_radial(r, 1e-310, 1.0, tiny_radius, 1.0, geometry)
    result = bt.analytical.steady_radial_first_order(
        r, D=1e-310, k=1.0, R=tiny_radius, c_surface=1.0, geometry=geometry
    )
    np.testing.assert_allclose(result, expected, rtol=3e-14, atol=0)


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
def test_radial_reference_preserves_boundary_layer_distance(geometry):
    r = np.array([np.nextafter(1.0, 0.0), 1.0])
    # At phi=1e16, the next representable inner point lies 1.11 reaction
    # lengths below the surface. Large-argument asymptotes differ by <1 ulp.
    expected = np.exp(-1e16 * (1.0 - r))
    result = bt.analytical.steady_radial_first_order(
        r, D=1.0, k=1e32, R=1.0, c_surface=1.0, geometry=geometry
    )
    np.testing.assert_allclose(result, expected, rtol=2e-14, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"r": -0.001},
        {"r": 1.001},
        {"r": np.nan},
        {"k": -1.0},
        {"k": np.inf},
        {"geometry": "annulus"},
        {"c_surface": np.nan},
    ],
)
def test_radial_reference_rejects_invalid_data_even_at_zero_reaction(kwargs):
    parameters = dict(r=0.0, D=1.0, k=0.0, R=1.0, c_surface=1.0)
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        bt.analytical.steady_radial_first_order(**parameters)


@pytest.mark.parametrize("geometry", ["cylindrical", "spherical"])
@pytest.mark.parametrize("cells, radius", [(11, 0.1), (37, 1e-4), (149, 0.01)])
def test_radial_reference_accepts_native_endpoint_roundoff(geometry, cells, radius):
    mesh = bt.mesh_1d(cells, 0.0, radius, geometry)
    profile = bt.analytical.steady_radial_first_order(
        bt.x_nodes(mesh), D=1e-9, k=1e-4, R=radius, c_surface=1.0, geometry=geometry
    )
    assert np.all(np.isfinite(profile))
    assert profile[-1] == 1.0
