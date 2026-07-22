"""Independent unit and limiting-case checks for the bioheat cryotherapy model."""

from dataclasses import replace
import math

import numpy as np
import pytest

import biotransport as bt
from biotransport.provenance import (
    EvidenceLevel,
    ParameterStatus,
    TemperatureContext,
    Uncertainty,
    UncertaintyKind,
    ValidityRange,
)


def _solver(
    *,
    perfusion: float = 0.0,
    metabolic_source: float = 0.0,
    latent_heat: float = 0.0,
    arrhenius_A: float = 0.0,
    arrhenius_E: float = 0.0,
):
    mesh = bt.StructuredMesh(4, 4, 0.0, 0.04, 0.0, 0.04)
    nodes = (mesh.nx() + 1) * (mesh.ny() + 1)
    return bt.BioheatCryotherapySolver(
        mesh,
        [0] * nodes,
        [perfusion] * nodes,
        [metabolic_source] * nodes,
        1000.0,  # rho_tissue [kg/m^3]
        1000.0,  # rho_blood [kg/m^3]
        4000.0,  # c_blood [J/(kg K)]
        1.0,  # k_unfrozen [W/(m K)]
        1.0,  # k_frozen [W/(m K)]
        1000.0,  # c_unfrozen [J/(kg K)]
        1000.0,  # c_frozen [J/(kg K)]
        300.0,  # initial/arterial/boundary default [K]
        200.0,  # probe [K]
        250.0,  # freezing center [K]
        2.0,  # two-sigma mushy-zone width [K]
        latent_heat,
        arrhenius_A,
        arrhenius_E,
        8.31446261815324,
    )


def test_config_uses_explicit_temperature_units_and_celsius_constructor():
    config = bt.BioheatCryotherapyConfig.from_celsius(
        probe_C=-150.0,
        freeze_C=-1.0,
        initial_C=37.0,
        arterial_C=37.0,
        boundary_C=37.0,
    )

    assert config.T_probe_K == pytest.approx(123.15)
    assert config.T_freeze_K == pytest.approx(272.15)
    assert config.T_initial_K == pytest.approx(310.15)
    assert config.T_probe_C == pytest.approx(-150.0)
    assert config.dt <= config.maximum_stable_dt_s

    with pytest.raises(TypeError):
        bt.BioheatCryotherapyConfig(T_probe=-150.0)
    with pytest.raises(ValueError, match="absolute zero"):
        bt.BioheatCryotherapyConfig.kelvin_from_celsius(-273.15)


def test_config_rejects_nonphysical_and_unstable_inputs():
    with pytest.raises(ValueError, match="w_b_normal"):
        bt.BioheatCryotherapyConfig(w_b_normal=-1.0)
    with pytest.raises(ValueError, match="T_probe_K"):
        bt.BioheatCryotherapyConfig(T_probe_K=280.0)
    with pytest.raises(ValueError, match="finite"):
        bt.BioheatCryotherapyConfig(q_met_normal=math.nan)
    with pytest.raises(ValueError, match="stability"):
        bt.BioheatCryotherapyConfig(nx=1000, ny=1000, dt=0.05)
    with pytest.raises(ValueError, match="probe_position"):
        bt.BioheatCryotherapyConfig(probe_position=0.01)


def test_config_provenance_is_traceable_and_never_silently_stale():
    config = bt.BioheatCryotherapyConfig()
    generated = config.provenance

    assert generated.model_identifier == "bioheat_cryotherapy"
    assert generated.record("rho_tissue").value == config.rho_tissue
    assert generated.record("rho_tissue").status is ParameterStatus.ILLUSTRATIVE
    assert "patient-specific" in generated.notes

    source_record = replace(
        generated.record("rho_tissue"),
        source_identifier="synthetic-bioheat-test-fixture",
        citation="Synthetic test fixture; not a tissue-property source.",
        url="https://example.org/synthetic-bioheat-fixture",
        population_or_material="synthetic homogeneous thermal phantom",
        temperature=TemperatureContext(
            value=293.15,
            unit="K",
            description="Synthetic fixture reference temperature.",
        ),
        measurement_method="deterministic synthetic fixture construction",
        validity_range=ValidityRange(
            lower=1000.0,
            upper=1100.0,
            unit="kg/m^3",
            description="Synthetic fixture test interval.",
        ),
        uncertainty=Uncertainty(
            kind=UncertaintyKind.EXACT,
            unit="kg/m^3",
            description="Exactly specified synthetic value.",
        ),
        evidence_level=EvidenceLevel.VALIDATED,
        status=ParameterStatus.RECOMMENDED,
    )
    attached = generated.with_record(source_record)
    config.attach_parameter_provenance(attached)
    assert config.provenance.record("rho_tissue").status is ParameterStatus.RECOMMENDED

    config.rho_tissue = 1040.0
    with pytest.raises(ValueError, match="stale"):
        _ = config.provenance

    config.reset_parameter_provenance_as_illustrative()
    config.validate()
    assert config.provenance.record("rho_tissue").value == 1040.0
    assert config.provenance.record("rho_tissue").status is ParameterStatus.ILLUSTRATIVE


def test_apparent_heat_capacity_has_mass_specific_units():
    config = bt.BioheatCryotherapyConfig()
    sigma_K = config.T_freeze_range_K / 2.0
    expected_peak = 0.5 * (
        config.c_tissue_unfrozen + config.c_tissue_frozen
    ) + config.L_fusion / (math.sqrt(2.0 * math.pi) * sigma_K)

    assert config.apparent_specific_heat_at_freezing == pytest.approx(expected_peak)
    # The old implementation erroneously introduced another factor of rho_tissue.
    assert config.apparent_specific_heat_at_freezing < 1.0e6


def test_frozen_fraction_is_bounded_and_monotone():
    config = bt.BioheatCryotherapyConfig()
    warm = config.frozen_fraction(config.T_freeze_K + 10.0)
    center = config.frozen_fraction(config.T_freeze_K)
    cold = config.frozen_fraction(config.T_freeze_K - 10.0)

    assert 0.0 <= warm < center < cold <= 1.0
    assert center == pytest.approx(0.5)


def test_cpp_property_queries_match_the_documented_phase_model():
    solver = _solver(latent_heat=333000.0, arrhenius_A=2.0, arrhenius_E=0.0)

    assert solver.frozen_fraction(250.0) == pytest.approx(0.5)
    assert solver.thermal_conductivity(250.0) == pytest.approx(1.0)
    expected_capacity = 1000.0 + 333000.0 / math.sqrt(2.0 * math.pi)
    assert solver.effective_specific_heat(250.0) == pytest.approx(expected_capacity)
    assert solver.arrhenius_heat_injury_rate(250.0) == pytest.approx(2.0)
    assert solver.maximum_stable_time_step_s() > 0.1


def test_config_creates_a_fully_configured_cpp_solver():
    config = bt.BioheatCryotherapyConfig(
        domain_size_x=0.04,
        domain_size_y=0.04,
        nx=4,
        ny=4,
        T_initial_K=290.0,
        T_arterial_K=300.0,
        T_boundary_K=290.0,
    )
    mesh = bt.StructuredMesh(4, 4, 0.0, 0.04, 0.0, 0.04)
    nodes = mesh.num_nodes()
    solver = config.create_solver(
        mesh,
        probe_mask=[0] * nodes,
        perfusion_map=[0.01] * nodes,
        q_met_map=[0.0] * nodes,
    )

    saved = solver.simulate(0.1, 1, [0.1])
    assert saved.temperature_K()[0, 2, 2] > config.T_initial_K

    mismatched_mesh = bt.StructuredMesh(5, 4, 0.0, 0.04, 0.0, 0.04)
    with pytest.raises(ValueError, match="configured nx and ny"):
        config.create_solver(
            mismatched_mesh,
            probe_mask=[0] * mismatched_mesh.num_nodes(),
            perfusion_map=[0.0] * mismatched_mesh.num_nodes(),
            q_met_map=[0.0] * mismatched_mesh.num_nodes(),
        )

    shifted_mesh = bt.StructuredMesh(4, 4, -0.02, 0.02, -0.02, 0.02)
    with pytest.raises(ValueError, match="domain bounds"):
        config.create_solver(
            shifted_mesh,
            probe_mask=[0] * shifted_mesh.num_nodes(),
            perfusion_map=[0.0] * shifted_mesh.num_nodes(),
            q_met_map=[0.0] * shifted_mesh.num_nodes(),
        )


def test_uniform_pennes_equilibrium_remains_uniform():
    solver = _solver(perfusion=0.01)
    saved = solver.simulate(0.1, 10, [0.0, 1.0])

    np.testing.assert_allclose(saved.temperature_K(), 300.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(saved.damage(), 0.0, atol=0.0, rtol=0.0)


def test_metabolic_source_has_watts_per_cubic_metre_scaling():
    solver = _solver(metabolic_source=1000.0)
    saved = solver.simulate(0.1, 1, [0.1])
    temperature = saved.temperature_K()[0]

    # q/(rho*c) = 1000/(1000*1000) = 1e-3 K/s at the center.
    assert temperature[2, 2] == pytest.approx(300.0001, abs=1e-12)
    assert temperature[0, 0] == pytest.approx(300.0)


def test_arrhenius_integral_is_heat_injury_without_artificial_freeze_factor():
    solver = _solver(arrhenius_A=2.0, arrhenius_E=0.0)
    saved = solver.simulate(0.1, 1, [0.1])

    np.testing.assert_allclose(saved.damage(), 0.2, atol=1e-15, rtol=0.0)


def test_off_grid_save_times_are_reached_exactly_and_duplicates_coalesce():
    solver = _solver()
    saved = solver.simulate(0.1, 1, [0.1, 0.035, 0.0, 0.035])

    assert saved.times_s == pytest.approx([0.0, 0.035, 0.1], abs=0.0)
    assert saved.frames == 3


def test_result_exposes_phase_and_stability_diagnostics():
    solver = _solver()
    saved = solver.simulate(0.1, 1, [0.0, 0.1])

    assert saved.maximum_stable_dt_s > 0.1
    assert saved.frozen_fraction().shape == saved.temperature_K().shape
    assert saved.minimum_temperature_K == pytest.approx([300.0, 300.0])
    assert saved.maximum_temperature_K == pytest.approx([300.0, 300.0])


def test_solver_rejects_nonfinite_maps_bad_masks_and_unstable_steps():
    mesh = bt.StructuredMesh(4, 4, 0.0, 0.04, 0.0, 0.04)
    nodes = (mesh.nx() + 1) * (mesh.ny() + 1)
    base_arguments = [
        mesh,
        [0] * nodes,
        [0.0] * nodes,
        [0.0] * nodes,
        1000.0,
        1000.0,
        4000.0,
        1.0,
        1.0,
        1000.0,
        1000.0,
        300.0,
        200.0,
        250.0,
        2.0,
        0.0,
        0.0,
        0.0,
        8.314,
    ]

    nonfinite = list(base_arguments)
    nonfinite[2] = [0.0] * nodes
    nonfinite[2][4] = math.nan
    with pytest.raises(ValueError, match="perfusion"):
        bt.BioheatCryotherapySolver(*nonfinite)

    bad_mask = list(base_arguments)
    bad_mask[1] = [0] * nodes
    bad_mask[1][4] = 2
    with pytest.raises(ValueError, match="probe_mask"):
        bt.BioheatCryotherapySolver(*bad_mask)

    solver = _solver()
    with pytest.raises(ValueError, match="stability"):
        solver.simulate(100.0, 1, [])
    with pytest.raises(ValueError, match="save times"):
        solver.simulate(0.1, 1, [0.2])
