"""Scientific and planning contracts for reusable teaching experiments."""

import copy
import math

import numpy as np
import pytest

import biotransport as bt
from biotransport.studio.examples import examples


def closed_uptake(geometry="cartesian", mode="transient"):
    document = copy.deepcopy(examples()[0])
    document["domain"].update(geometry=geometry, length=1.0, cells=24)
    document["run"].update(mode=mode, duration=0.1, frames=4)
    document["components"][0]["parameters"]["coefficient"] = 0.2
    document["components"][1]["parameters"]["value"] = 1.0
    for index, side in ((2, "left"), (3, "right")):
        document["components"][index] = {
            "id": side,
            "type": "boundary.sealed",
            "parameters": {"side": side},
        }
    document["components"].append(
        {
            "id": "uptake",
            "type": "reaction.uptake",
            "parameters": {"Vmax": 0.3, "Km": 0.25},
        }
    )
    return document


@pytest.mark.parametrize("geometry", ["cartesian", "cylindrical", "spherical"])
def test_saturable_uptake_matches_native_and_integrated_ode(geometry):
    document = closed_uptake(geometry)
    experiment = bt.Experiment.from_dict(document)
    direct = (
        bt.Problem(bt.mesh_1d(24, 0, 1, geometry))
        .diffusivity(0.2)
        .initial(1)
        .sealed("left")
        .sealed("right")
        .add_michaelis_menten(0.3, 0.25)
    )
    solution = experiment.run()
    np.testing.assert_array_equal(
        solution.history, bt.solve(direct, end_time=0.1, frames=4).history
    )
    # Separate variables: c + Km*log(c) = c0 + Km*log(c0) - Vmax*t.
    c = solution.c
    # Taylor's remainder bounds Euler's error in this integrated identity:
    # |F''| <= Km/(c0 - Vmax*t)^2 and |dc/dt| <= Vmax.
    largest_step = max(d.maximum_time_step for d in solution.all_diagnostics)
    error_bound = 0.25 * 0.3**2 * largest_step * 0.1 / (2 * (1 - 0.3 * 0.1) ** 2)
    assert np.max(np.abs(c + 0.25 * np.log(c) - 0.97)) <= error_bound + 1e-12


@pytest.mark.parametrize("geometry", ["cartesian", "cylindrical", "spherical"])
def test_closed_source_and_saturable_uptake_have_correct_equilibrium(geometry):
    document = closed_uptake(geometry, "steady")
    document["components"].append(
        {"id": "source", "type": "reaction.source", "parameters": {"rate": 0.1}}
    )
    solution = bt.Experiment.from_dict(document).run()
    np.testing.assert_allclose(solution.c, 0.25 * 0.1 / (0.3 - 0.1), atol=1e-9)


@pytest.mark.parametrize("parameter,value", [("Vmax", -1), ("Km", 0), ("Km", math.inf)])
def test_uptake_parameter_errors_locate_the_field(parameter, value):
    document = closed_uptake()
    document["components"][-1]["parameters"][parameter] = value
    with pytest.raises(bt.ExperimentValidationError) as captured:
        bt.Experiment.from_dict(document)
    assert any(issue["path"].endswith(parameter) for issue in captured.value.issues)


@pytest.mark.parametrize("production", [0.3, 0.4])
def test_closed_production_cannot_exceed_saturable_capacity_at_steady_state(production):
    document = closed_uptake(mode="steady")
    document["components"].append(
        {"id": "source", "type": "reaction.source", "parameters": {"rate": production}}
    )
    with pytest.raises(bt.ExperimentValidationError, match="uptake capacity"):
        bt.Experiment.from_dict(document)


@pytest.mark.parametrize("geometry", ["cartesian", "cylindrical", "spherical"])
def test_experiment_plan_matches_saved_schedule_and_storage(geometry):
    experiment = bt.Experiment.from_dict(closed_uptake(geometry))
    plan = experiment.plan()
    solution = experiment.run()
    assert isinstance(plan, bt.ExperimentPlan)
    assert plan.planned_steps == solution.steps
    assert plan.saved_states == len(solution.times)
    assert plan.field_storage_bytes == solution.history.nbytes
    assert plan.selected_time_step == max(
        d.maximum_time_step for d in solution.all_diagnostics
    )
    data = plan.to_dict()
    data["planned_steps"] = -1
    assert plan.planned_steps == solution.steps


def test_experiment_step_budget_rejects_before_stepping(monkeypatch):
    import biotransport.experiment as implementation

    experiment = bt.Experiment.from_dict(closed_uptake())
    monkeypatch.setattr(
        implementation, "solve", lambda *a, **k: pytest.fail("must not solve")
    )
    plan = experiment.plan(max_steps=1)
    assert not plan.within_step_budget
    with pytest.raises(RuntimeError, match="max_steps=1"):
        experiment.run(max_steps=1)


def test_steady_plan_does_not_invent_time_steps_or_predict_convergence(monkeypatch):
    import biotransport.experiment as implementation

    monkeypatch.setattr(
        implementation, "solve_steady", lambda *a, **k: pytest.fail("must not solve")
    )
    plan = bt.Experiment.from_dict(examples()[3]).plan(max_steps=1)
    assert plan.planned_steps is plan.selected_time_step is None
    assert plan.saved_states == 1 and plan.within_step_budget
    assert plan.field_storage_bytes == plan.nodes * 8


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_plan_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="positive integer"):
        bt.Experiment.from_dict(closed_uptake()).plan(max_steps=budget)


def test_uptake_accounting_matches_native_clipping_at_negative_concentrations():
    mesh = bt.mesh_1d(4)
    field = np.array([-1.0, -0.25, -0.1, 0.0, 0.5])
    problem = bt.Problem(mesh).diffusivity(0).initial(field).michaelis_menten(0.3, 0.25)
    rates = problem._recipe.reaction_rate(field, np.linspace(0, 1, 5), np.zeros(5), 0)
    np.testing.assert_allclose(rates, [0, 0, 0, 0, -0.2])
    result = bt.solve(problem, end_time=0.001)
    np.testing.assert_allclose((result.c - field) / 0.001, rates, atol=1e-12)
