"""Native preflight follows the exact schedule and never executes the model."""

import math

import numpy as np
import pytest

import biotransport as bt
from biotransport._core import TransportPlan, plan_transport


def _options(end_time, **settings):
    options = bt.SolveOptions.until(end_time)
    for key, value in settings.items():
        setattr(options, key, value)
    return options


@pytest.mark.parametrize("geometry", ["cartesian", "cylindrical", "spherical"])
def test_plan_matches_solve_geometry_and_reaction_schedule(geometry):
    problem = (
        bt.Problem(bt.mesh_1d(8, 0, 1, geometry))
        .diffusivity(0.25)
        .linear_decay(0.9)
        .add_constant_source(0.3)
        .initial(0.2)
        .sealed("left")
        .dirichlet("right", 1)
    )
    original = np.asarray(problem.initial()).copy()
    options = _options(0.25)
    plan = plan_transport(problem, options)
    assert isinstance(plan, TransportPlan)
    np.testing.assert_array_equal(problem.initial(), original)
    assert plan.diagnostics.steps == 0
    assert plan.diagnostics.final_time == 0
    assert plan.diagnostics.requested_final_time == 0.25
    result = bt.solve_transport(problem, options)
    assert plan.planned_steps == result.diagnostics.steps
    assert plan.selected_time_step == result.diagnostics.maximum_time_step
    assert plan.within_step_budget
    assert plan.diagnostics.initial_mass == result.diagnostics.initial_mass
    assert plan.diagnostics.initial_maximum == 1
    assert (
        plan.diagnostics.certified_stable_time_step
        == result.diagnostics.certified_stable_time_step
    )


def test_plan_includes_reaction_accuracy_limit():
    problem = bt.Problem(bt.mesh_1d(2)).diffusivity(0).linear_decay(1).initial(1)
    plan = plan_transport(problem, _options(1))
    assert plan.diagnostics.certified_stable_time_step == 1
    assert plan.selected_time_step == 0.1
    assert plan.planned_steps == 10
    stricter = plan_transport(problem, _options(1, reaction_step_fraction=0.025))
    assert stricter.selected_time_step == 0.025
    assert stricter.planned_steps == 40


def test_plan_reports_budget_excess_without_evolving():
    problem = bt.Problem(bt.mesh_1d(2)).diffusivity(0).linear_decay(1).initial(1)
    options = _options(1, max_steps=3)
    plan = plan_transport(problem, options)
    assert plan.planned_steps == 10
    assert not plan.within_step_budget
    with pytest.raises(RuntimeError, match="max_steps"):
        bt.solve_transport(problem, options)


@pytest.mark.parametrize(
    "duration, expected",
    [(math.nextafter(0.1, 0), 10), (0.1, 10), (math.nextafter(0.1, 1), 11)],
)
def test_plan_uses_native_binary64_endpoint_rules(duration, expected):
    problem = bt.Problem(bt.mesh_1d(2)).diffusivity(0).constant_source(1).initial(0)
    options = _options(duration, time_step=0.01, max_steps=11)
    plan = plan_transport(problem, options)
    assert plan.planned_steps == expected
    assert bt.solve_transport(problem, options).diagnostics.steps == expected


def test_segment_plans_match_saved_run_exactly():
    problem = bt.Problem(bt.mesh_1d(2)).diffusivity(0).linear_decay(1).initial(1)
    duration, frames = 0.7, 6
    # This is the public equally spaced frame contract; only native code counts steps.
    times = [duration * (index + 1) / frames for index in range(frames)]
    total_steps = 0
    previous = 0
    for time in times:
        total_steps += plan_transport(problem, _options(time - previous)).planned_steps
        previous = time
    result = bt.solve(problem, end_time=duration, frames=frames)
    assert result.steps == total_steps
    assert result.steps > plan_transport(problem, _options(duration)).planned_steps


def test_plan_never_calls_reactions_or_changes_initial_fields():
    calls = []

    def reaction(value, x, y, time):
        calls.append(time)
        return -value

    problem = (
        bt.Problem(bt.mesh_1d(2))
        .diffusivity(0)
        .reaction(reaction, max_abs_dc=1)
        .initial(1)
    )
    options = _options(0.2)
    plan = plan_transport(problem, options)
    assert not calls
    np.testing.assert_array_equal(problem.initial(), np.ones(3))
    bt.solve_transport(problem, options)
    assert calls
    assert plan.diagnostics.steps == 0


def test_plan_is_readonly_and_diagnostics_are_independent():
    problem = bt.Problem(bt.mesh_1d(2)).diffusivity(0).linear_decay(1).initial(1)
    options = _options(1)
    plan = plan_transport(problem, options)
    for field, value in [
        ("selected_time_step", 0.3),
        ("planned_steps", 4),
        ("within_step_budget", False),
    ]:
        with pytest.raises(AttributeError):
            setattr(plan, field, value)
    assert plan.diagnostics is not plan.diagnostics
    with pytest.raises(AttributeError):
        plan.diagnostics.steps = 3
    problem.linear_decay(2)
    options.final_time = 2
    assert plan.planned_steps == 10
    assert plan.diagnostics.reaction_rate_bound == 1


def test_zero_duration_and_constant_source_plans():
    problem = bt.Problem(bt.mesh_1d(2)).diffusivity(0).constant_source(1).initial(3)
    zero = plan_transport(problem, _options(0))
    assert zero.planned_steps == 0
    assert zero.selected_time_step == 0
    assert zero.within_step_budget
    assert zero.diagnostics.initial_mass == zero.diagnostics.final_mass == 3
    source = plan_transport(problem, _options(2))
    assert source.planned_steps == 1
    assert source.selected_time_step == 2


def test_scientifically_invalid_plans_fail_before_running():
    problem = bt.Problem(bt.mesh_1d(10)).diffusivity(1).initial(1)
    with pytest.raises(ValueError, match="stability"):
        plan_transport(problem, _options(1, time_step=1))
    problem.reaction(lambda value, x, y, t: -value * value)
    with pytest.raises(ValueError, match="derivative bound"):
        plan_transport(problem, _options(1))
    explicit = plan_transport(problem, _options(1, time_step=0.001))
    assert not explicit.diagnostics.reaction_stability_bound_known
    assert math.isnan(explicit.diagnostics.certified_stable_time_step)
    with pytest.raises(ValueError, match="max_steps"):
        plan_transport(problem, _options(1, max_steps=0))
