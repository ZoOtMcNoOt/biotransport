"""Scientific-contract tests for time-dependent scalar boundary protocols."""

from __future__ import annotations

import numpy as np
import pytest

import biotransport as bt


def _problem_1d(*, cells: int = 20, diffusivity: float = 0.1, initial: float = 0.0):
    mesh = bt.mesh_1d(cells, 0.0, 1.0)
    problem = bt.Problem(mesh).diffusivity(diffusivity).initial_condition(initial)
    return mesh, problem


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: bt.SinusoidalBC(frequency=0.0), "frequency must be positive"),
        (lambda: bt.RampBC(duration=0.0), "duration must be positive"),
        (lambda: bt.SquareWaveBC(duty_cycle=1.1), "duty_cycle"),
        (
            lambda: bt.ArterialPressureBC(systolic=80.0, diastolic=120.0),
            "systolic must be greater",
        ),
        (lambda: bt.RespiratoryBC(inspiration_fraction=1.0), "inspiration_fraction"),
        (lambda: bt.DrugInfusionBC(bolus_duration=-1.0), "bolus_duration"),
        (lambda: bt.CompositeBC(operation="subtract"), "operation"),
    ],
)
def test_waveform_parameters_fail_loudly(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_arterial_template_has_declared_extrema_and_period():
    waveform = bt.ArterialPressureBC(
        systolic=120.0,
        diastolic=80.0,
        heart_rate=60.0,
        systolic_fraction=0.4,
    )

    peak_time = 0.35 * 0.4 * waveform.period()
    assert waveform(0.0) == pytest.approx(80.0)
    assert waveform(peak_time) == pytest.approx(120.0)
    assert waveform(0.37) == pytest.approx(waveform(1.37))


def test_cardiac_output_mean_is_the_declared_cycle_mean():
    waveform = bt.CardiacOutputBC(
        mean_flow=5.0,
        peak_flow=25.0,
        heart_rate=72.0,
        ejection_fraction=0.3,
    )
    count = 20_000
    times = (np.arange(count, dtype=np.float64) + 0.5) * waveform.period() / count
    sampled_mean = np.mean([waveform(float(time)) for time in times])

    assert sampled_mean == pytest.approx(5.0, rel=2e-5)
    assert max(waveform(float(time)) for time in times) <= 25.0


def test_composite_reports_only_an_established_common_period():
    periodic = bt.CompositeBC(
        [bt.SinusoidalBC(frequency=1.0), bt.SinusoidalBC(frequency=2.0 / 3.0)]
    )
    nonperiodic = bt.CompositeBC([bt.ConstantBC(2.0), bt.RampBC()])

    assert periodic.period() == pytest.approx(3.0)
    assert nonperiodic.period() == 0.0


def test_sampling_and_rate_conversions_validate_their_domains():
    with pytest.raises(ValueError, match="bpm must be positive"):
        bt.heart_rate_to_period(0.0)
    with pytest.raises(ValueError, match="T must be positive"):
        bt.period_to_heart_rate(-1.0)
    with pytest.raises(ValueError, match="t_end"):
        bt.sample_waveform(bt.ConstantBC(1.0), 1.0, 0.0)
    with pytest.raises(ValueError, match="num_points"):
        bt.sample_waveform(bt.ConstantBC(1.0), num_points=0)


def test_reference_solver_preserves_mass_with_zero_neumann_boundaries():
    mesh = bt.mesh_1d(40, 0.0, 1.0)
    x = np.array([mesh.x(i) for i in range(mesh.num_nodes())])
    initial = np.exp(-(((x - 0.37) / 0.11) ** 2))
    problem = bt.Problem(mesh).diffusivity(0.05).initial_condition(initial)

    with pytest.warns(RuntimeWarning, match="legacy Python/NumPy reference"):
        result = bt.solve_pulsatile(problem, 0.05, {}, dt=0.001)

    initial_mass = mesh.dx() * (
        0.5 * initial[0] + np.sum(initial[1:-1]) + 0.5 * initial[-1]
    )
    final_mass = mesh.dx() * (
        0.5 * result.solution[0]
        + np.sum(result.solution[1:-1])
        + 0.5 * result.solution[-1]
    )
    assert final_mass == pytest.approx(initial_mass, rel=0.0, abs=2e-14)
    assert result.solution[-1] != pytest.approx(0.0, abs=1e-15)


def test_static_neumann_values_are_outward_derivatives_with_correct_signs():
    mesh = bt.mesh_1d(20, 0.0, 1.0)
    linear = np.array([mesh.x(i) for i in range(mesh.num_nodes())])
    problem = (
        bt.Problem(mesh)
        .diffusivity(0.1)
        .initial_condition(linear)
        .neumann(bt.Boundary.Left, -1.0)
        .neumann(bt.Boundary.Right, 1.0)
    )

    with pytest.warns(RuntimeWarning):
        result = bt.solve_pulsatile(problem, 0.05, {}, dt=0.001)

    np.testing.assert_allclose(result.solution, linear, rtol=0.0, atol=2e-14)
    assert result.stats["static_neumann_semantics"] == "outward-normal derivative du/dn"


def test_reference_solver_applies_dynamic_dirichlet_at_the_new_time_level():
    _, problem = _problem_1d(diffusivity=0.01)
    waveform = bt.CustomBC(lambda time: 2.0 * time, T=0.0)

    with pytest.warns(RuntimeWarning):
        result = bt.solve_pulsatile(
            problem,
            0.03,
            {bt.Boundary.Left: waveform},
            dt=0.01,
            save_every=2,
        )

    assert result.solution[0] == pytest.approx(0.06)
    assert result.bc_history[bt.Boundary.Left][-1] == pytest.approx(0.06)
    assert len(result.time_history) == len(result.solution_history)
    assert len(result.time_history) == len(result.bc_history[bt.Boundary.Left])
    assert result.stats["dynamic_boundary_semantics"] == "strong Dirichlet scalar value"


def test_reference_solver_rejects_an_unstable_step():
    _, problem = _problem_1d(cells=10, diffusivity=1.0)

    with pytest.raises(ValueError, match="stability limit"):
        bt.solve_pulsatile(problem, 0.1, {}, dt=0.01)


def test_reference_solver_rejects_physics_it_does_not_implement():
    _, reaction_problem = _problem_1d()
    reaction_problem.linear_decay(1.0)
    with pytest.raises(NotImplementedError, match="reactions"):
        bt.solve_pulsatile(reaction_problem, 0.1, {}, dt=0.001)

    _, advection_problem = _problem_1d()
    advection_problem.velocity(0.1)
    with pytest.raises(NotImplementedError, match="advection"):
        bt.solve_pulsatile(advection_problem, 0.1, {}, dt=0.001)

    mesh = bt.mesh_1d(10, 0.0, 1.0)
    variable_problem = (
        bt.Problem(mesh)
        .diffusivity_field(np.ones(mesh.num_nodes()))
        .initial_condition(0.0)
    )
    with pytest.raises(NotImplementedError, match="variable diffusivity"):
        bt.solve_pulsatile(variable_problem, 0.1, {}, dt=0.001)


def test_reference_solver_rejects_unsupported_boundaries_and_dimensions():
    _, robin_problem = _problem_1d()
    robin_problem.robin(bt.Boundary.Right, 1.0, 1.0, 0.0)
    with pytest.raises(NotImplementedError, match="Robin"):
        bt.solve_pulsatile(robin_problem, 0.1, {}, dt=0.001)

    mesh_2d = bt.mesh_2d(5, 4, 0.0, 1.0, 0.0, 1.0)
    problem_2d = bt.Problem(mesh_2d).diffusivity(0.1).initial_condition(0.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        bt.solve_pulsatile(problem_2d, 0.1, {}, dt=0.001)

    _, problem = _problem_1d()
    with pytest.raises(ValueError, match="only Boundary.Left"):
        bt.solve_pulsatile(
            problem,
            0.1,
            {bt.Boundary.Top: bt.ConstantBC(1.0)},
            dt=0.001,
        )


def test_zero_diffusivity_requires_an_explicit_temporal_resolution():
    _, problem = _problem_1d(diffusivity=0.0)
    with pytest.raises(ValueError, match="dt is required"):
        bt.solve_pulsatile(problem, 1.0, {bt.Boundary.Left: bt.ConstantBC(1.0)})

    with pytest.warns(RuntimeWarning):
        result = bt.solve_pulsatile(
            problem,
            1.0,
            {bt.Boundary.Left: bt.ConstantBC(1.0)},
            dt=0.1,
        )
    assert result.solution[0] == 1.0
    assert np.all(result.solution[1:] == 0.0)


def test_callback_receives_a_read_only_copy_and_step_guard_is_enforced():
    _, problem = _problem_1d(diffusivity=0.01)
    callback_fields = []

    def callback(_time, field):
        callback_fields.append(field)
        assert not field.flags.writeable

    with pytest.warns(RuntimeWarning):
        result = bt.solve_pulsatile(problem, 0.02, {}, dt=0.01, callback=callback)

    assert len(callback_fields) == result.stats["steps"]
    with pytest.raises(ValueError, match="more than max_steps"):
        bt.solve_pulsatile(problem, 0.1, {}, dt=0.001, max_steps=5)
