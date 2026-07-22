"""Scientific contract tests for the explicit runtime-units boundary."""

import math
from dataclasses import FrozenInstanceError

import pytest

from biotransport import units


def test_temperature_round_trip_and_si_storage() -> None:
    body = units.temperature(37.0, "degC")

    assert body.dimension is units.Dimension.ABSOLUTE_TEMPERATURE
    assert body.si_unit == "K"
    assert body.si_value == pytest.approx(310.15)
    assert body.to("degC") == pytest.approx(37.0)
    assert body.to("degF") == pytest.approx(98.6)
    assert units.convert(98.6, "degF", "degC") == pytest.approx(37.0)


def test_absolute_temperature_and_temperature_difference_are_distinct() -> None:
    arterial = units.temperature(37.0, "degC")
    cooling = units.temperature_difference(-5.0, "delta_degC")
    cooled = arterial + cooling

    assert cooled.to("degC") == pytest.approx(32.0)
    assert (arterial - cooled).to("delta_K") == pytest.approx(5.0)
    assert units.temperature_difference(9.0, "delta_degF").to(
        "delta_degC"
    ) == pytest.approx(5.0)

    with pytest.raises(units.DimensionError, match="adding two absolute"):
        _ = arterial + cooled
    with pytest.raises(units.DimensionError, match="cannot convert"):
        arterial.to("delta_degC")


@pytest.mark.parametrize(
    ("value", "unit"),
    [
        (-0.01, "K"),
        (-273.16, "degC"),
        (-459.68, "degF"),
    ],
)
def test_absolute_zero_is_enforced(value: float, unit: str) -> None:
    with pytest.raises(units.QuantityDomainError, match="absolute zero"):
        units.temperature(value, unit)


def test_pressure_conversions_use_explicit_reference_units() -> None:
    pressure = units.pressure(20.0, "mmHg")

    assert pressure.si_value == pytest.approx(2666.4477483)
    assert pressure.to("kPa") == pytest.approx(2.6664477483)
    assert units.pressure(1.0, "atm").to("torr") == pytest.approx(760.0)


def test_molar_concentration_scales_are_not_confused() -> None:
    physiological = units.concentration(140.0, "mM")

    # 1 mM is exactly 1 mol/m^3; 140 mM is not 0.140 mol/m^3.
    assert physiological.si_value == pytest.approx(140.0)
    assert physiological.to("mol/L") == pytest.approx(0.140)
    assert units.concentration(250.0, "uM").to("mM") == pytest.approx(0.250)


def test_transport_coefficients_keep_semantic_dimensions_separate() -> None:
    diffusion = units.diffusivity(1e-5, "cm^2/s")
    membrane = units.solute_permeability(2.0, "um/s")
    porous = units.intrinsic_permeability(1.0, "darcy")
    mobility = units.darcy_mobility(5e-12)

    assert diffusion.si_value == pytest.approx(1e-9)
    assert membrane.si_value == pytest.approx(2e-6)
    assert porous.si_value == pytest.approx(9.869233e-13)
    assert mobility.si_value == pytest.approx(5e-12)
    assert units.permeability(1e-7, "m/s").dimension is (
        units.Dimension.SOLUTE_PERMEABILITY
    )

    for other_unit in ("m/s", "m^2", "m^2/(Pa*s)"):
        if units.get_unit(other_unit).dimension is not diffusion.dimension:
            with pytest.raises(units.DimensionError, match="cannot convert"):
                diffusion.to(other_unit)


def test_mass_specific_perfusion_requires_density_context() -> None:
    clinical = units.mass_specific_perfusion(60.0, "mL/(min*100g)")

    with pytest.raises(units.ConversionContextError, match="tissue_density_kg_m3"):
        clinical.to("1/s")
    with pytest.raises(units.ConversionContextError, match="tissue_density_kg_m3"):
        units.perfusion_rate(60.0, "mL/(min*100g)")

    volumetric = units.perfusion_rate(
        60.0,
        "mL/(min*100g)",
        tissue_density_kg_m3=1000.0,
    )
    assert volumetric.dimension is units.Dimension.VOLUMETRIC_PERFUSION
    assert volumetric.si_value == pytest.approx(0.01)
    assert clinical.to("1/min", tissue_density_kg_m3=1000.0) == pytest.approx(0.6)


def test_perfusion_context_conversion_round_trip() -> None:
    perfusion = units.perfusion_rate(0.01, "1/s")

    mass_specific = perfusion.to("mL/(min*100g)", tissue_density_kg_m3=1000.0)
    assert mass_specific == pytest.approx(60.0)
    assert units.convert(
        mass_specific,
        "mL/(min*100g)",
        "1/s",
        tissue_density_kg_m3=1000.0,
    ) == pytest.approx(0.01)

    with pytest.raises(units.QuantityDomainError, match="must be positive"):
        perfusion.to("mL/(kg*min)", tissue_density_kg_m3=0.0)


def test_length_time_amount_and_energy_round_trip() -> None:
    assert units.length(100.0, "um").to("mm") == pytest.approx(0.1)
    assert units.time(2.0, "h").to("min") == pytest.approx(120.0)
    assert units.amount(2.5, "umol").to("nmol") == pytest.approx(2500.0)
    assert units.energy(1.0, "kcal").to("kJ") == pytest.approx(4.184)


@pytest.mark.parametrize(
    ("factory", "unit"),
    [
        (units.concentration, "mM"),
        (units.diffusivity, "m^2/s"),
        (units.solute_permeability, "m/s"),
        (units.intrinsic_permeability, "m^2"),
        (units.darcy_mobility, "m^2/(Pa*s)"),
        (units.perfusion_rate, "1/s"),
        (units.mass_specific_perfusion, "m^3/(kg*s)"),
        (units.length, "m"),
        (units.time, "s"),
        (units.amount, "mol"),
    ],
)
def test_nonnegative_quantity_domains(factory, unit: str) -> None:
    with pytest.raises(units.QuantityDomainError, match="non-negative"):
        factory(-1.0, unit)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_input_is_rejected_everywhere(value: float) -> None:
    with pytest.raises(units.QuantityDomainError, match="finite"):
        units.quantity(value, "Pa")


def test_bool_and_non_numeric_values_are_not_silently_coerced() -> None:
    with pytest.raises(TypeError, match="real number"):
        units.pressure(True)
    with pytest.raises(TypeError, match="real number"):
        units.pressure("20")  # type: ignore[arg-type]


def test_dimension_named_factories_reject_a_valid_but_wrong_unit() -> None:
    with pytest.raises(units.DimensionError, match="diffusivity requires"):
        units.diffusivity(1.0, "m/s")
    with pytest.raises(units.DimensionError, match="pressure requires"):
        units.pressure(1.0, "J")


def test_cross_dimension_arithmetic_comparison_and_ratio_are_rejected() -> None:
    distance = units.length(1.0, "m")
    duration = units.time(1.0, "s")

    with pytest.raises(units.DimensionError, match="cannot combine"):
        _ = distance + duration
    with pytest.raises(units.DimensionError, match="cannot combine"):
        _ = distance / duration
    with pytest.raises(units.DimensionError, match="cannot combine"):
        _ = distance < duration
    with pytest.raises(units.DimensionError, match="multiplication"):
        _ = distance * duration
    with pytest.raises(TypeError, match="another Quantity"):
        _ = distance < 1.0


def test_same_dimension_arithmetic_and_solver_handoff() -> None:
    one_mm = units.length(1.0, "mm")
    two_mm = one_mm * 2.0

    assert (one_mm + two_mm).to("mm") == pytest.approx(3.0)
    assert two_mm / one_mm == pytest.approx(2.0)
    assert two_mm.require(units.Dimension.LENGTH) == pytest.approx(0.002)
    with pytest.raises(units.DimensionError, match="expected time"):
        two_mm.require(units.Dimension.TIME)


def test_quantity_and_unit_registry_are_immutable() -> None:
    distance = units.length(1.0)

    with pytest.raises(FrozenInstanceError):
        distance.si_value = 2.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        units.get_unit("m").scale = 2.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        units.UNITS["furlong"] = units.get_unit("m")  # type: ignore[index]


def test_conversion_overflow_is_rejected_as_nonfinite() -> None:
    with pytest.raises(units.QuantityDomainError, match="non-finite"):
        units.pressure(1e308, "MPa")


def test_unknown_units_fail_loudly_and_registry_is_discoverable() -> None:
    with pytest.raises(units.UnknownUnitError, match="available_units"):
        units.quantity(1.0, "banana")

    concentration_units = units.available_units(units.Dimension.MOLAR_CONCENTRATION)
    assert concentration_units == ("mol/m^3", "mol/L", "mM", "uM", "nM", "pM")
    assert len(units.available_units()) > len(concentration_units)


def test_unicode_aliases_and_human_whitespace_are_supported() -> None:
    assert units.length(2.0, " µm ").si_value == pytest.approx(2e-6)
    assert units.diffusivity(3.0, "µm² / s").si_value == pytest.approx(3e-12)
    assert units.concentration(4.0, "mol · m⁻³").si_value == pytest.approx(4.0)


def test_format_always_names_the_requested_unit() -> None:
    assert units.pressure(20.0, "mmHg").format("kPa", precision=5) == "2.6664 kPa"


def test_documented_end_to_end_api_example() -> None:
    # Inputs can be written in familiar units, then handed to native solvers as
    # explicit SI scalars only after a dimension check.
    temperature_k = units.temperature(37.0, "degC").require(
        units.Dimension.ABSOLUTE_TEMPERATURE
    )
    concentration_mol_m3 = units.concentration(140.0, "mM").require(
        units.Dimension.MOLAR_CONCENTRATION
    )
    diffusivity_m2_s = units.diffusivity(1.33e-5, "cm^2/s").require(
        units.Dimension.DIFFUSIVITY
    )

    assert temperature_k == pytest.approx(310.15)
    assert concentration_mol_m3 == pytest.approx(140.0)
    assert diffusivity_m2_s == pytest.approx(1.33e-9)
