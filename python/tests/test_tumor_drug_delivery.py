"""Science-facing tests for tumor drug delivery configuration and bindings."""

from dataclasses import replace
import math

import biotransport as bt
from biotransport.provenance import ParameterSetProvenance, ParameterStatus
import numpy as np
import pytest


def _constant(mesh: bt.StructuredMesh, value: float) -> list[float]:
    return [value] * mesh.num_nodes()


def _uniform_solver() -> tuple[
    bt.StructuredMesh, bt.TumorDrugDeliverySolver, list[float]
]:
    mesh = bt.StructuredMesh(2, 2, 0.0, 1.0, 0.0, 1.0)
    solver = bt.TumorDrugDeliverySolver(
        mesh,
        [0] * mesh.num_nodes(),
        _constant(mesh, 1.0),
        0.0,
        0.0,
    )
    return mesh, solver, solver.solve_pressure_sor(max_iter=10, tol=1e-14, omega=1.0)


def test_config_converts_dimensional_vascular_surface_area() -> None:
    config = bt.TumorDrugDeliveryConfig(MVD_normal=100.0, vessel_radius=5e-6)

    expected = 2.0 * math.pi * 5e-6 * 100.0 * 1e6
    assert config.vascular_surface_area_normal == pytest.approx(expected)
    assert config.vascular_exchange_rate_normal == pytest.approx(
        config.P_vessel_normal * expected
    )
    assert config.IFP_tumor_Pa == pytest.approx(config.IFP_tumor * 133.322387415)
    assert "no Starling/lymphatic pressure solve" in config.describe()
    assert "1/m" in config.describe()
    assert "P*S_v" in config.describe()


def test_config_factory_accepts_round_tripped_provenance() -> None:
    config = bt.TumorDrugDeliveryConfig()
    manifest = config.provenance
    restored = ParameterSetProvenance.from_json(manifest.to_json())
    rebuilt = bt.TumorDrugDeliveryConfig(parameter_provenance=restored)

    assert rebuilt.provenance.fingerprint() == manifest.fingerprint()
    assert all(
        record.status is ParameterStatus.ILLUSTRATIVE
        for record in rebuilt.provenance.records
    )

    stale = restored.with_record(replace(restored.record("IFP_tumor"), value=19.0))
    with pytest.raises(ValueError, match="stale"):
        bt.TumorDrugDeliveryConfig(parameter_provenance=stale)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("domain_size", 0.0),
        ("tumor_radius", -1.0),
        ("rim_thickness", 3e-3),
        ("D_drug_tumor", -1e-12),
        ("k_binding", -0.1),
        ("MVD_normal", -1.0),
        ("vessel_radius", 0.0),
        ("P_vessel_tumor", -1e-7),
        ("C_plasma", -1.0),
        ("K_hydraulic_normal", 0.0),
        ("IFP_normal", 30.0),
        ("nx", 1),
        ("ny", True),
    ],
)
def test_config_rejects_nonphysical_domains(keyword: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        bt.TumorDrugDeliveryConfig(**{keyword: value})


def test_config_requires_tumor_strictly_inside_domain() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        bt.TumorDrugDeliveryConfig(tumor_center=(1e-3, 2.5e-3), tumor_radius=1e-3)


def test_config_rejects_nonrepresentable_unit_conversions() -> None:
    with pytest.raises(ValueError, match="pascals"):
        bt.TumorDrugDeliveryConfig(IFP_tumor=1e308)
    with pytest.raises(ValueError, match="vascular surface area density"):
        bt.TumorDrugDeliveryConfig(vessel_radius=1e308, MVD_normal=1e308)


def test_pressure_field_is_bounded_symmetric_and_converged() -> None:
    mesh = bt.StructuredMesh(10, 10, 0.0, 1.0, 0.0, 1.0)
    mask = np.zeros(mesh.num_nodes(), dtype=np.uint8)
    mask[mesh.index(5, 5)] = 1
    solver = bt.TumorDrugDeliverySolver(
        mesh,
        mask.tolist(),
        _constant(mesh, 1.0),
        0.0,
        10.0,
    )

    pressure = np.asarray(
        solver.solve_pressure_sor(max_iter=20_000, tol=1e-11, omega=1.6)
    )
    assert np.all(np.isfinite(pressure))
    assert pressure.min() >= -1e-10
    assert pressure.max() <= 10.0 + 1e-10
    assert pressure[mesh.index(5, 5)] == 10.0
    assert pressure[mesh.index(3, 5)] == pytest.approx(
        pressure[mesh.index(7, 5)], abs=2e-9
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        solver.solve_pressure_sor(max_iter=1, tol=1e-30, omega=1.0)
    with pytest.raises(RuntimeError, match="did not converge"):
        solver.solve_pressure_sor(max_iter=1, tol=1e-10, omega=1e-14)


def test_vascular_exchange_is_not_normalized_and_saves_exact_times() -> None:
    mesh, solver, pressure = _uniform_solver()

    def run(surface_area: float):
        return solver.simulate(
            pressure,
            _constant(mesh, 0.0),
            _constant(mesh, 0.1),
            _constant(mesh, surface_area),
            0.0,
            0.0,
            1.0,
            0.01,
            20,
            [0.2, 0.0, 0.01, 0.155, 0.155],
        )

    one = run(1.0)
    two = run(2.0)
    assert one.times_s == [0.0, 0.01, 0.155, 0.2]
    assert one.frames == 4
    free_one = one.free()
    free_two = two.free()
    assert free_one.shape == (4, 3, 3)
    assert np.all(np.isfinite(free_one))
    assert np.all(free_one >= 0.0)
    assert free_two[1, 1, 1] / free_one[1, 1, 1] == pytest.approx(2.0, abs=5e-3)


def test_reaction_compartments_are_nonnegative_and_sum_to_total() -> None:
    mesh, solver, pressure = _uniform_solver()
    result = solver.simulate(
        pressure,
        _constant(mesh, 0.0),
        _constant(mesh, 0.1),
        _constant(mesh, 1.0),
        0.2,
        0.3,
        1.0,
        0.01,
        20,
        [0.2],
    )

    free = result.free()
    bound = result.bound()
    cellular = result.cellular()
    total = result.total()
    assert np.all(free >= 0.0)
    assert np.all(bound >= 0.0)
    assert np.all(cellular >= 0.0)
    np.testing.assert_allclose(total, free + bound + cellular, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(bound / cellular, 2.0 / 3.0, rtol=1e-14)
    exact_free = 0.1 / 0.6 * (1.0 - math.exp(-0.6 * 0.2))
    assert free[-1, 0, 0] == pytest.approx(exact_free, abs=7e-5)
    assert result.final_time_s == pytest.approx(0.2)
    assert result.stability_limit_s > 0.0
    assert result.total_amount_per_depth[-1] == pytest.approx(
        result.cumulative_net_vascular_exchange_per_depth[-1], abs=5e-15
    )
    assert result.cumulative_boundary_outflow_per_depth[-1] == 0.0
    assert result.mass_balance_error_per_depth[-1] == pytest.approx(0.0, abs=5e-15)


def test_transport_rejects_unstable_or_ambiguous_inputs() -> None:
    mesh, solver, pressure = _uniform_solver()
    tiny = solver.simulate(
        pressure,
        _constant(mesh, 0.0),
        _constant(mesh, 0.0),
        _constant(mesh, 0.0),
        0.0,
        0.0,
        0.0,
        1e-16,
        1,
        [1e-16],
    )
    assert tiny.times_s == [1e-16]
    assert tiny.final_time_s == 1e-16

    with pytest.raises(ValueError, match="stability limit"):
        solver.simulate(
            pressure,
            _constant(mesh, 1.0),
            _constant(mesh, 0.0),
            _constant(mesh, 0.0),
            0.0,
            0.0,
            1.0,
            1.0,
            1,
            [1.0],
        )
    with pytest.raises(ValueError, match="save times"):
        solver.simulate(
            pressure,
            _constant(mesh, 0.0),
            _constant(mesh, 0.0),
            _constant(mesh, 0.0),
            0.0,
            0.0,
            1.0,
            0.1,
            1,
            [0.2],
        )
    with pytest.raises(ValueError, match="surface area density"):
        solver.simulate(
            pressure,
            _constant(mesh, 0.0),
            _constant(mesh, 0.1),
            [-1.0] + _constant(mesh, 1.0)[1:],
            0.0,
            0.0,
            1.0,
            0.1,
            1,
            [0.1],
        )
