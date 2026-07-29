"""Scientific-contract tests for time-dependent scalar boundary protocols."""

from __future__ import annotations

from fractions import Fraction
import math

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


def test_periodic_waveforms_reduce_extreme_phases_without_nan():
    sinusoid = bt.SinusoidalBC(frequency=1e308, phase=1e308)
    square = bt.SquareWaveBC(frequency=1e308, phase=1e308)

    assert math.isfinite(sinusoid(1e308))
    assert math.isfinite(square(1e308))
    with pytest.raises(ValueError, match="period outside the finite float64 range"):
        bt.SinusoidalBC(frequency=np.nextafter(0.0, 1.0)).period()


def test_waveform_derived_arithmetic_is_finite_or_fails_loudly():
    maximum = np.finfo(float).max
    ramp = bt.RampBC(start_value=-maximum, end_value=maximum)
    assert ramp(0.5) == pytest.approx(0.0)

    respiratory = bt.RespiratoryBC(
        mean=maximum,
        amplitude=maximum,
        respiratory_rate=60.0,
        inspiration_fraction=0.5,
    )
    with pytest.raises(ValueError, match="non-finite"):
        respiratory(0.5)

    with pytest.raises(ValueError, match="mean_flow is incompatible"):
        bt.CardiacOutputBC(
            mean_flow=maximum,
            peak_flow=maximum,
            heart_rate=60.0,
            ejection_fraction=np.nextafter(1.0, 0.0),
        )


def test_venous_mean_pressure_is_the_declared_cycle_mean():
    waveform = bt.VenousPressureBC(mean_pressure=8.0, amplitude=4.0, heart_rate=72.0)
    count = 20_000
    times = (np.arange(count, dtype=np.float64) + 0.5) * waveform.period() / count
    sampled_mean = np.mean([waveform(float(time)) for time in times])

    assert sampled_mean == pytest.approx(8.0, rel=0.0, abs=2e-9)


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
    subnormal = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="outside the finite float64 range"):
        bt.heart_rate_to_period(subnormal)
    with pytest.raises(ValueError, match="outside the finite float64 range"):
        bt.period_to_heart_rate(subnormal)


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

    with pytest.raises(NotImplementedError, match="Robin"):
        bt.solve_pulsatile(
            robin_problem,
            0.1,
            {bt.Boundary.Right: bt.ConstantBC(1.0)},
            dt=0.001,
        )

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


def test_step_planning_never_rounds_down_a_tiny_final_remainder():
    mesh = bt.mesh_1d(10, 0.0, 1.0)
    problem = bt.Problem(mesh).diffusivity(1.0).initial_condition(0.0)
    stable_dt = math.nextafter(0.5 * mesh.dx() ** 2, 0.0)
    final_time = stable_dt * (1.0 + 3e-14)

    with pytest.raises(ValueError, match="more than max_steps=1"):
        bt.solve_pulsatile(problem, final_time, {}, dt=stable_dt, max_steps=1)

    with pytest.warns(RuntimeWarning):
        result = bt.solve_pulsatile(problem, final_time, {}, dt=stable_dt, max_steps=2)
    assert result.stats["steps"] == 2
    assert result.stats["max_diffusion_number"] <= 0.5


def test_step_planning_uses_the_exact_float_ratio_not_rounded_products():
    _, problem = _problem_1d(diffusivity=0.0)
    step_dt = 0.09375291778124752
    final_time = 0.28125875334374256

    with pytest.raises(ValueError, match="more than max_steps=3"):
        bt.solve_pulsatile(problem, final_time, {}, dt=step_dt, max_steps=3)

    callback_times = []
    with pytest.warns(RuntimeWarning):
        result = bt.solve_pulsatile(
            problem,
            final_time,
            {},
            dt=step_dt,
            max_steps=4,
            callback=lambda time, _field: callback_times.append(time),
        )
    assert result.stats["steps"] == 4
    assert result.time == final_time
    assert result.stats["last_dt"] > 0.0
    assert np.all(np.diff(callback_times) > 0.0)


def test_decimal_endpoint_keeps_its_exact_remainder_step():
    _, problem = _problem_1d(diffusivity=0.0)

    with pytest.warns(RuntimeWarning):
        result = bt.solve_pulsatile(problem, 0.1, {}, dt=0.0001, max_steps=1001)

    assert result.stats["steps"] == 1001
    assert result.time == 0.1
    assert 0.0 < result.stats["last_dt"] <= 0.0001


def test_subnormal_dt_fails_with_the_step_guard_instead_of_overflowing():
    _, problem = _problem_1d(diffusivity=0.0)
    subnormal_dt = np.nextafter(0.0, 1.0)

    with pytest.raises(ValueError, match="more than max_steps=10"):
        bt.solve_pulsatile(problem, 1.0, {}, dt=subnormal_dt, max_steps=10)


def test_tiny_diffusivity_is_not_lost_when_the_stable_dt_overflows():
    mesh = bt.mesh_1d(10, 0.0, 1.0)
    diffusivity = np.nextafter(0.0, 1.0)
    initial = np.zeros(mesh.num_nodes())
    initial[5] = 1.0
    problem = bt.Problem(mesh).diffusivity(diffusivity).initial_condition(initial)
    step_dt = 1e308
    expected_diffusion_number = diffusivity * step_dt / mesh.dx() ** 2

    with pytest.warns(RuntimeWarning):
        result = bt.solve_pulsatile(problem, step_dt, {}, dt=step_dt, max_steps=1)

    expected = initial.copy()
    expected[4] = expected_diffusion_number
    expected[5] = 1.0 - 2.0 * expected_diffusion_number
    expected[6] = expected_diffusion_number
    np.testing.assert_allclose(result.solution, expected, rtol=2e-16, atol=0.0)
    assert math.isinf(result.stats["max_stable_dt"])
    assert result.stats["max_diffusion_number"] == pytest.approx(
        expected_diffusion_number, rel=1e-15
    )


@pytest.mark.parametrize("peak", [1e300, np.finfo(float).max])
def test_subnormal_cfl_preserves_representable_large_field_updates(peak):
    mesh = bt.mesh_1d(4, 0.0, 4.0)
    diffusivity = np.nextafter(0.0, 1.0)
    step_dt = 0.25
    initial = np.zeros(mesh.num_nodes())
    initial[2] = peak
    problem = bt.Problem(mesh).diffusivity(diffusivity).initial_condition(initial)
    exact_lambda = (
        Fraction.from_float(diffusivity)
        * Fraction.from_float(step_dt)
        / Fraction.from_float(mesh.dx()) ** 2
    )
    expected_increment = float(exact_lambda * Fraction.from_float(peak))

    with pytest.warns(RuntimeWarning):
        result = bt.solve_pulsatile(problem, step_dt, {}, dt=step_dt, max_steps=1)

    assert np.all(np.isfinite(result.solution))
    assert result.solution[1] == expected_increment
    assert result.solution[3] == expected_increment
    assert result.stats["max_diffusion_number"] > 0.0
    assert result.stats["max_diffusion_number_exact"] == (
        f"{exact_lambda.numerator}/{exact_lambda.denominator}"
    )


def test_subnormal_cfl_preserves_representable_neumann_update():
    mesh = bt.mesh_1d(1, 0.0, 1.0)
    diffusivity = np.nextafter(0.0, 1.0)
    step_dt = 0.25
    derivative = 1e300
    problem = (
        bt.Problem(mesh)
        .diffusivity(diffusivity)
        .initial_condition(0.0)
        .neumann(bt.Boundary.Left, derivative)
        .dirichlet(bt.Boundary.Right, 0.0)
    )
    exact_lambda = (
        Fraction.from_float(diffusivity)
        * Fraction.from_float(step_dt)
        / Fraction.from_float(mesh.dx()) ** 2
    )
    expected = float(2 * exact_lambda * Fraction.from_float(derivative))

    with pytest.warns(RuntimeWarning):
        result = bt.solve_pulsatile(problem, step_dt, {}, dt=step_dt, max_steps=1)

    assert result.solution[0] == expected
    assert result.solution[1] == 0.0


def test_stable_limit_avoids_large_spacing_intermediate_overflow():
    mesh = bt.mesh_1d(10, 0.0, 1e156)
    problem = bt.Problem(mesh).diffusivity(1e308).initial_condition(0.0)
    expected_limit = 0.5 * (mesh.dx() / 1e308) * mesh.dx()

    with pytest.warns(RuntimeWarning):
        automatic = bt.solve_pulsatile(problem, 100.0, {})
    with pytest.warns(RuntimeWarning):
        explicit = bt.solve_pulsatile(problem, 100.0, {}, dt=40.0)

    assert automatic.stats["max_stable_dt"] == pytest.approx(expected_limit, rel=2e-15)
    assert automatic.stats["dt"] <= automatic.stats["max_stable_dt"]
    assert automatic.stats["max_diffusion_number"] <= 0.5
    assert explicit.stats["max_stable_dt"] == automatic.stats["max_stable_dt"]
    assert explicit.stats["max_diffusion_number"] == pytest.approx(0.4)


def test_stable_limit_avoids_tiny_spacing_intermediate_underflow():
    mesh = bt.mesh_1d(1, 0.0, 1e-200)
    problem = bt.Problem(mesh).diffusivity(1e-100).initial_condition(0.0)
    expected_limit = 0.5 * (mesh.dx() / 1e-100) * mesh.dx()

    with pytest.warns(RuntimeWarning):
        result = bt.solve_pulsatile(problem, 1e-300, {})

    assert result.stats["max_stable_dt"] == pytest.approx(expected_limit, rel=2e-15)
    assert result.stats["dt"] <= result.stats["max_stable_dt"]
    assert result.stats["max_diffusion_number"] <= 0.5


def test_custom_waveform_cannot_mutate_the_transport_problem():
    _, problem = _problem_1d(diffusivity=0.01)

    def mutate_problem(_time):
        problem.diffusivity(0.02)
        return 1.0

    waveform = bt.CustomBC(mutate_problem)
    with pytest.warns(RuntimeWarning):
        with pytest.raises(RuntimeError, match="TransportProblem was mutated"):
            bt.solve_pulsatile(
                problem,
                0.01,
                {bt.Boundary.Left: waveform},
                dt=0.01,
            )


def test_solve_callback_cannot_mutate_the_transport_problem():
    _, problem = _problem_1d(diffusivity=0.01)

    def mutate_problem(_time, _field):
        problem.initial_condition(3.0)

    with pytest.warns(RuntimeWarning):
        with pytest.raises(RuntimeError, match="TransportProblem was mutated"):
            bt.solve_pulsatile(problem, 0.02, {}, dt=0.01, callback=mutate_problem)
