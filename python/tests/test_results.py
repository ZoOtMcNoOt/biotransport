"""The shared Result/Snapshots containers and bt.solve(save_times=...)."""

from __future__ import annotations

import numpy as np
import pytest

import biotransport as bt
from biotransport import BioTransportDeprecationWarning, Result, Snapshots
from biotransport.contracts import get_contract


def _problem(mesh=None):
    mesh = bt.mesh_1d(20) if mesh is None else mesh
    return (
        bt.Problem(mesh)
        .diffusivity(0.01)
        .linear_decay(0.3)
        .initial_condition(bt.gaussian(mesh, center=0.5, width=0.1))
        .dirichlet(bt.Boundary.Left, 0.0)
        .neumann(bt.Boundary.Right, 0.0)
    )


def test_solve_returns_a_result_that_carries_everything() -> None:
    result = bt.solve(_problem(), end_time=0.05, time_step=0.01)

    assert isinstance(result, Result)
    assert result.time == 0.05
    assert result.steps == result.diagnostics.steps == 5
    assert result.contract == "transport.canonical_explicit"
    assert get_contract(result.contract).native_symbols == ("solve_transport",)
    assert result.mesh.num_nodes() == 21
    assert result.field is result.concentration
    assert result.fields.keys() == {"concentration"}
    assert not result.concentration.flags.writeable
    assert len(result.snapshots) == 0
    np.testing.assert_array_equal(result.concentration, result.native.concentration)
    with pytest.raises(TypeError):
        result.fields["other"] = np.zeros(3)  # type: ignore[index]


def test_save_times_snapshots_match_one_shot_solves_bitwise() -> None:
    problem = _problem()
    result = bt.solve(
        problem, end_time=0.1, time_step=0.01, save_times=[0.0, 0.05, 0.1]
    )

    assert result.snapshots.times == (0.0, 0.05, 0.1)
    assert list(result.snapshots) == [0.0, 0.05, 0.1]
    assert result.diagnostics.steps == 10
    np.testing.assert_array_equal(result.snapshots[0.1], result.concentration)
    np.testing.assert_array_equal(
        result.snapshots[0.05],
        bt.solve(problem, end_time=0.05, time_step=0.01).concentration,
    )
    np.testing.assert_array_equal(
        result.concentration,
        bt.solve(problem, end_time=0.1, time_step=0.01).concentration,
    )
    stacked = result.snapshots.stacked()
    assert stacked.shape == (3, 21)
    np.testing.assert_array_equal(
        stacked[1], result.snapshots.at(0.05000001, abs_tol=1e-6)
    )
    with pytest.raises(KeyError):
        result.snapshots[0.07]
    assert not result.snapshots[0.05].flags.writeable


def test_save_times_preserve_every_configured_term_and_absolute_time() -> None:
    mesh = bt.mesh_1d(8)
    problem = (
        bt.Problem(mesh)
        .diffusivity(0.0)
        .initial_condition(0.0)
        .reaction(lambda c, x, y, t: 1.0 if t >= 0.05 else 0.0, max_abs_dc=0.0)
    )
    direct = bt.solve(problem, end_time=0.1, time_step=0.01)
    partitioned = bt.solve(problem, end_time=0.1, time_step=0.01, save_times=[0.05])

    assert direct.concentration[3] == pytest.approx(0.05)
    np.testing.assert_array_equal(direct.concentration, partitioned.concentration)
    np.testing.assert_array_equal(
        partitioned.snapshots[0.05], np.zeros(mesh.num_nodes())
    )


@pytest.mark.parametrize(
    ("save_times", "error", "match"),
    [
        ([0.05, 0.05], ValueError, "strictly increasing"),
        ([0.2], ValueError, r"within \[0, end_time\]"),
        ([-0.01], ValueError, r"within \[0, end_time\]"),
        ([0.06, 0.05], ValueError, "strictly increasing"),
        ("0.05", TypeError, "sequence"),
        ([float("nan")], ValueError, "finite"),
        ([True], TypeError, "real number"),
    ],
)
def test_save_times_are_validated_in_python(save_times, error, match) -> None:
    with pytest.raises(error, match=match):
        bt.solve(_problem(), end_time=0.1, time_step=0.01, save_times=save_times)


def test_save_time_at_zero_records_the_initial_state_with_essential_values() -> None:
    result = bt.solve(_problem(), end_time=0.0, save_times=[0.0])
    assert result.steps == 0
    np.testing.assert_array_equal(result.snapshots[0.0], result.concentration)
    assert result.concentration[0] == 0.0


def test_as_grid_reshapes_2d_fields_and_plot_works_on_a_result_alone() -> None:
    mesh = bt.mesh_2d(6, 4)
    problem = bt.Problem(mesh).diffusivity(0.01).initial_condition(1.0)
    result = bt.solve(problem, end_time=0.01)

    grid = result.as_grid()
    assert grid.shape == (5, 7)
    assert not grid.flags.writeable
    np.testing.assert_array_equal(grid.ravel(), result.concentration)
    figure = result.plot(show=False)
    assert figure is not None
    figure_via_module = bt.plot(result, show=False)
    assert figure_via_module is not None


def test_deprecated_t_and_dt_keywords_warn_and_match_the_new_spellings() -> None:
    problem = _problem()
    expected = bt.solve(problem, end_time=0.05, time_step=0.01)
    with pytest.warns(
        BioTransportDeprecationWarning, match=r"solve\(end_time=\.\.\.\)"
    ):
        legacy_time = bt.solve(problem, t=0.05, time_step=0.01)
    with pytest.warns(
        BioTransportDeprecationWarning, match=r"solve\(time_step=\.\.\.\)"
    ):
        legacy_step = bt.solve(problem, end_time=0.05, dt=0.01)
    np.testing.assert_array_equal(legacy_time.concentration, expected.concentration)
    np.testing.assert_array_equal(legacy_step.concentration, expected.concentration)
    with pytest.raises(TypeError, match="either end_time or t"):
        bt.solve(problem, end_time=0.05, t=0.05)
    with pytest.raises(TypeError, match="either time_step or dt"):
        bt.solve(problem, end_time=0.05, time_step=0.01, dt=0.01)


def test_run_checkpoints_is_deprecated_in_favour_of_save_times() -> None:
    mesh = bt.mesh_1d(10)
    with pytest.warns(BioTransportDeprecationWarning, match="save_times"):
        legacy = bt.run_checkpoints(mesh, [0.05, 0.1], 0.01, time_step=0.01)
    problem = bt.Problem(mesh).diffusivity(0.01).initial_condition(0.0)
    modern = bt.solve(problem, end_time=0.1, time_step=0.01, save_times=[0.05, 0.1])
    np.testing.assert_array_equal(legacy[0.1], modern.snapshots[0.1])


def test_snapshots_validate_their_inputs() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        Snapshots((0.2, 0.1), (np.zeros(2), np.zeros(2)))
    with pytest.raises(ValueError, match="same length"):
        Snapshots((0.1,), ())
    with pytest.raises(ValueError, match="same number of values"):
        Snapshots((0.1, 0.2), (np.zeros(2), np.zeros(3)))
    with pytest.raises(ValueError, match="non-negative"):
        Snapshots((-0.1,), (np.zeros(2),))
    assert Snapshots((), ()).stacked().shape == (0, 0)


def test_result_validates_its_inputs() -> None:
    mesh = bt.mesh_1d(2)
    diagnostics = bt.solve(
        bt.Problem(mesh).initial_condition(0.0), end_time=0.0
    ).diagnostics
    with pytest.raises(ValueError, match="primary field"):
        Result({"a": np.zeros(3)}, 0.0, 0, diagnostics, mesh, "x", primary="b")
    with pytest.raises(TypeError, match="non-empty mapping"):
        Result({}, 0.0, 0, diagnostics, mesh, "x")
    with pytest.raises(TypeError, match="contract"):
        Result({"concentration": np.zeros(3)}, 0.0, 0, diagnostics, mesh, "")
    with pytest.raises(ValueError, match="steps"):
        Result({"concentration": np.zeros(3)}, 0.0, -1, diagnostics, mesh, "x")
    result = Result({"u": np.zeros(3)}, 0.0, 0, diagnostics, mesh, "x", primary="u")
    with pytest.raises(AttributeError, match="available fields"):
        result.concentration
