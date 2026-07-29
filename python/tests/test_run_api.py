"""Fail-loud and checkpoint semantics for the friendly canonical adapters."""

from __future__ import annotations

import numpy as np
import pytest

import biotransport as bt


def _constant_problem() -> bt.Problem:
    mesh = bt.mesh_1d(4, 0.0, 1.0)
    return bt.Problem(mesh).diffusivity(0.0).initial_condition(2.0)


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("end_time", True, TypeError),
        ("end_time", np.nan, ValueError),
        ("end_time", -1.0, ValueError),
        ("time_step", False, TypeError),
        ("time_step", np.inf, ValueError),
        ("time_step", 0.0, ValueError),
        ("safety_factor", 0.0, ValueError),
        ("safety_factor", 1.1, ValueError),
        ("reaction_step_fraction", np.nan, ValueError),
        ("max_steps", True, TypeError),
        ("max_steps", 2.5, TypeError),
        ("max_steps", 0, ValueError),
        ("check_finite", "yes", TypeError),
        ("method", None, TypeError),
    ],
)
def test_solve_rejects_ambiguous_or_nonfinite_options(keyword, value, error) -> None:
    arguments = {"end_time": 0.1, keyword: value}

    with pytest.raises(error):
        bt.solve(_constant_problem(), **arguments)


def test_solve_zero_time_is_an_exact_no_op() -> None:
    result = bt.solve(_constant_problem(), end_time=0.0)

    assert result.time == 0.0
    assert result.diagnostics.steps == 0
    np.testing.assert_array_equal(result.concentration, 2.0)


@pytest.mark.parametrize(
    ("checkpoints", "error"),
    [
        ([True], TypeError),
        ([np.nan], ValueError),
        ([np.inf], ValueError),
        ([0.0], ValueError),
        ([-0.1], ValueError),
        ([0.1, 0.1], ValueError),
        ("0.1", TypeError),
    ],
)
def test_run_checkpoints_rejects_invalid_times(checkpoints, error) -> None:
    mesh = bt.mesh_1d(4, 0.0, 1.0)

    with pytest.raises(error):
        bt.run_checkpoints(mesh, checkpoints, diffusivity=0.0)


def test_run_checkpoints_returns_owned_fields_and_segment_diagnostics() -> None:
    mesh = bt.mesh_1d(4, 0.0, 1.0)
    result = bt.run_checkpoints(
        mesh,
        np.array([0.2, 0.1]),
        diffusivity=0.0,
        initial_condition=2.0,
        time_step=0.05,
    )

    assert isinstance(result, bt.CheckpointResult)
    assert result.times == (0.1, 0.2)
    assert tuple(result) == result.times
    assert result.total_steps == 4
    assert result.diagnostics[0.1].steps == 2
    assert result.diagnostics[0.2].steps == 2
    assert result.diagnostics[0.1].final_time == 0.1
    assert result.diagnostics[0.2].final_time == 0.1
    np.testing.assert_array_equal(result[0.1], 2.0)
    np.testing.assert_array_equal(result[0.2], 2.0)
    assert not result[0.1].flags.writeable
    assert not np.shares_memory(result[0.1], result[0.2])


def test_checkpoint_result_constructor_copies_and_checks_metadata() -> None:
    native = bt.solve(_constant_problem(), end_time=0.1, time_step=0.1)
    source = np.full(5, 2.0)
    result = bt.CheckpointResult(
        fields={0.1: source},
        diagnostics={0.1: native.diagnostics},
        total_steps=1,
    )
    source[0] = -1.0

    np.testing.assert_array_equal(result[0.1], 2.0)
    assert not result[0.1].flags.writeable
    with pytest.raises(ValueError, match="sum of segment"):
        bt.CheckpointResult(
            fields={0.1: source},
            diagnostics={0.1: native.diagnostics},
            total_steps=2,
        )


def test_checkpoint_result_rejects_empty_or_inconsistent_fields() -> None:
    first = bt.solve(_constant_problem(), end_time=0.1, time_step=0.1).diagnostics
    second = bt.solve(_constant_problem(), end_time=0.1, time_step=0.1).diagnostics

    with pytest.raises(ValueError, match="at least one"):
        bt.CheckpointResult(fields={}, diagnostics={}, total_steps=0)
    with pytest.raises(ValueError, match="must not be empty"):
        bt.CheckpointResult(
            fields={0.1: np.array([])},
            diagnostics={0.1: first},
            total_steps=1,
        )
    with pytest.raises(ValueError, match="same number"):
        bt.CheckpointResult(
            fields={0.1: np.ones(2), 0.2: np.ones(3)},
            diagnostics={0.1: first, 0.2: second},
            total_steps=2,
        )


def test_checkpoint_result_rejects_diagnostic_time_mismatch() -> None:
    diagnostic = bt.solve(_constant_problem(), end_time=0.1, time_step=0.1).diagnostics

    with pytest.raises(ValueError, match="segment duration"):
        bt.CheckpointResult(
            fields={0.2: np.ones(5)},
            diagnostics={0.2: diagnostic},
            total_steps=1,
        )


@pytest.mark.parametrize(
    ("checkpoint_time", "diagnostic_time"),
    [
        (1.0e-300, 2.0e-300),
        (1.0, 1.0 + 1.0e-13),
    ],
)
def test_checkpoint_result_requires_exact_diagnostic_times(
    checkpoint_time: float,
    diagnostic_time: float,
) -> None:
    diagnostic = bt.solve(
        _constant_problem(),
        end_time=diagnostic_time,
        time_step=diagnostic_time,
    ).diagnostics

    with pytest.raises(ValueError, match="segment duration"):
        bt.CheckpointResult(
            fields={checkpoint_time: np.ones(5)},
            diagnostics={checkpoint_time: diagnostic},
            total_steps=1,
        )


def test_checkpoint_result_rejects_keys_that_collide_after_float_normalization() -> (
    None
):
    diagnostic = bt.solve(_constant_problem(), end_time=0.1, time_step=0.1).diagnostics
    first = 2**53
    second = first + 1

    with pytest.raises(ValueError, match="strictly increasing"):
        bt.CheckpointResult(
            fields={first: np.ones(5), second: np.ones(5)},
            diagnostics={first: diagnostic, second: diagnostic},
            total_steps=2,
        )


@pytest.mark.parametrize(
    "field",
    [
        np.array([1.0 + 2.0j]),
        np.array([True]),
        np.array(["1.0"]),
    ],
)
def test_checkpoint_result_rejects_non_real_numeric_fields(field) -> None:
    native = bt.solve(_constant_problem(), end_time=0.1, time_step=0.1)

    with pytest.raises(TypeError, match="real numeric"):
        bt.CheckpointResult(
            fields={0.1: field},
            diagnostics={0.1: native.diagnostics},
            total_steps=1,
        )


def test_checkpoint_result_rejects_masked_field_values() -> None:
    native = bt.solve(_constant_problem(), end_time=0.1, time_step=0.1)
    field = np.ma.array(np.ones(5), mask=[False, False, True, False, False])

    with pytest.raises(ValueError, match="masked"):
        bt.CheckpointResult(
            fields={0.1: field},
            diagnostics={0.1: native.diagnostics},
            total_steps=1,
        )


def test_run_checkpoints_matches_one_shot_when_dt_divides_every_segment() -> None:
    mesh = bt.mesh_1d(10, 0.0, 1.0)
    initial = np.sin(np.pi * bt.x_nodes(mesh))
    boundaries = {
        bt.Boundary.Left: bt.BoundaryCondition.dirichlet(0.0),
        bt.Boundary.Right: bt.BoundaryCondition.dirichlet(0.0),
    }

    checkpoints = bt.run_checkpoints(
        mesh,
        [0.001, 0.002],
        diffusivity=0.1,
        initial_condition=initial,
        boundaries=boundaries,
        time_step=0.0005,
    )
    one_shot_problem = (
        bt.Problem(mesh)
        .diffusivity(0.1)
        .initial_condition(initial)
        .dirichlet(bt.Boundary.Left, 0.0)
        .dirichlet(bt.Boundary.Right, 0.0)
    )
    one_shot = bt.solve(one_shot_problem, end_time=0.002, time_step=0.0005)

    np.testing.assert_array_equal(checkpoints[0.002], one_shot.concentration)
    assert checkpoints.total_steps == one_shot.diagnostics.steps


def test_run_checkpoints_enforces_max_steps_cumulatively() -> None:
    mesh = bt.mesh_1d(4, 0.0, 1.0)

    with pytest.raises(RuntimeError, match="cumulative max_steps"):
        bt.run_checkpoints(
            mesh,
            [0.1, 0.2],
            diffusivity=0.0,
            time_step=0.1,
            max_steps=1,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("diffusivity", np.nan, ValueError),
        ("diffusivity", -1.0, ValueError),
        ("initial_condition", True, TypeError),
        ("initial_condition", [0.0, 0.0, np.inf, 0.0, 0.0], ValueError),
        ("boundaries", [], TypeError),
        (
            "boundaries",
            {"left": bt.BoundaryCondition.dirichlet(0.0)},
            TypeError,
        ),
        ("boundaries", {bt.Boundary.Left: 0.0}, TypeError),
    ],
)
def test_run_checkpoints_rejects_invalid_problem_data(keyword, value, error) -> None:
    mesh = bt.mesh_1d(4, 0.0, 1.0)
    arguments = {
        "mesh": mesh,
        "checkpoints": [0.1],
        "diffusivity": 0.0,
        keyword: value,
    }

    with pytest.raises(error):
        bt.run_checkpoints(**arguments)


def test_run_checkpoints_owns_segment_time_keywords() -> None:
    mesh = bt.mesh_1d(4, 0.0, 1.0)

    with pytest.raises(TypeError, match="controls each segment end time"):
        bt.run_checkpoints(mesh, [0.1], 0.0, end_time=0.1)
