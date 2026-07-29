import numpy as np
import pytest

import biotransport as bt


def test_constant_source_runs_entirely_through_canonical_solve() -> None:
    mesh = bt.mesh_1d(20)
    problem = (
        bt.Problem(mesh).diffusivity(0.0).constant_source(2.0).initial_condition(0.0)
    )

    result = bt.solve(problem, end_time=0.25)

    np.testing.assert_allclose(result.concentration, 0.5)
    assert result.time == 0.25
    assert result.diagnostics.steps == 1
    assert result.diagnostics.reaction_stability_bound_known


def test_zero_flux_diffusion_conserves_trapezoidal_mass() -> None:
    mesh = bt.mesh_1d(80)
    x = bt.x_nodes(mesh)
    initial = np.exp(-(((x - 0.5) / 0.08) ** 2))
    problem = bt.Problem(mesh).diffusivity(0.01).initial_condition(initial)

    result = bt.solve(problem, end_time=0.01)

    scale = max(1.0, abs(result.diagnostics.initial_mass))
    assert abs(result.diagnostics.mass_change) <= 1e-12 * scale
    assert result.diagnostics.final_time == 0.01


def test_requested_step_is_honored_and_last_step_lands_exactly() -> None:
    mesh = bt.mesh_1d(10)
    problem = bt.Problem(mesh).diffusivity(0.01).initial_condition(1.0)

    result = bt.solve(problem, end_time=0.02, time_step=0.007)

    assert result.time == 0.02
    assert result.diagnostics.steps == 3
    assert result.diagnostics.maximum_time_step == pytest.approx(0.007)
    assert result.diagnostics.minimum_time_step == pytest.approx(0.006)


def test_custom_reaction_requires_step_or_derivative_bound() -> None:
    mesh = bt.mesh_1d(10)
    problem = (
        bt.Problem(mesh)
        .diffusivity(0.0)
        .initial_condition(1.0)
        .reaction(lambda concentration, x, y, time: -concentration)
    )

    with pytest.raises(ValueError, match="derivative bound"):
        bt.solve(problem, end_time=0.1)

    result = bt.solve(problem, end_time=0.1, time_step=0.01)
    assert result.diagnostics.steps == 10
    assert not result.diagnostics.reaction_stability_bound_known


def test_requested_step_planning_uses_portable_binary64_endpoints() -> None:
    problem = (
        bt.Problem(bt.mesh_1d(3))
        .diffusivity(0.0)
        .constant_source(1.0)
        .initial_condition(0.0)
    )

    snapped = bt.solve(problem, end_time=0.1, time_step=0.01, max_steps=10)
    assert snapped.diagnostics.steps == 10
    assert snapped.diagnostics.minimum_time_step == 0.01
    assert snapped.diagnostics.maximum_time_step == 0.01

    below = bt.solve(
        problem,
        end_time=np.nextafter(0.1, 0.0),
        time_step=0.01,
        max_steps=10,
    )
    assert below.diagnostics.steps == 10
    assert below.diagnostics.maximum_time_step <= 0.01

    endpoint_above = np.nextafter(0.1, np.inf)
    with pytest.raises(RuntimeError, match="max_steps"):
        bt.solve(
            problem,
            end_time=endpoint_above,
            time_step=0.01,
            max_steps=10,
        )
    above = bt.solve(
        problem,
        end_time=endpoint_above,
        time_step=0.01,
        max_steps=11,
    )
    assert above.diagnostics.steps == 11
    assert 0.0 < above.diagnostics.minimum_time_step <= 0.01
    assert above.diagnostics.maximum_time_step <= 0.01

    adversarial_end = 0.9375291778124752
    adversarial_step = 0.09375291778124752
    with pytest.raises(RuntimeError, match="max_steps"):
        bt.solve(
            problem,
            end_time=adversarial_end,
            time_step=adversarial_step,
            max_steps=10,
        )
    adversarial = bt.solve(
        problem,
        end_time=adversarial_end,
        time_step=adversarial_step,
        max_steps=11,
    )
    assert adversarial.diagnostics.steps == 11
    assert 0.0 < adversarial.diagnostics.minimum_time_step <= adversarial_step
    assert adversarial.diagnostics.maximum_time_step <= adversarial_step


def test_unverified_method_is_rejected_instead_of_substituted() -> None:
    mesh = bt.mesh_1d(10)
    problem = bt.Problem(mesh).diffusivity(0.01)

    with pytest.raises(ValueError, match="verified conservative"):
        bt.solve(problem, end_time=0.1, method="crank_nicolson")


def test_one_component_velocity_field_is_intuitive_for_1d() -> None:
    mesh = bt.mesh_1d(10)
    velocity = np.linspace(0.1, 0.2, 11)
    problem = bt.Problem(mesh)

    returned = problem.velocity_field(velocity)

    assert returned is problem
    assert problem.has_advection()


def test_two_component_velocity_field_remains_available_for_2d() -> None:
    mesh = bt.mesh_2d(4, 3)
    node_count = (mesh.nx() + 1) * (mesh.ny() + 1)
    vx = np.full(node_count, 0.1)
    vy = np.full(node_count, -0.2)
    problem = bt.Problem(mesh)

    returned = problem.velocity_field(vx, vy)

    assert returned is problem
    assert problem.has_advection()


def test_one_component_velocity_field_fails_clearly_for_2d() -> None:
    mesh = bt.mesh_2d(4, 3)
    vx = np.full((mesh.nx() + 1) * (mesh.ny() + 1), 0.1)
    problem = bt.Problem(mesh)

    with pytest.raises(ValueError, match="2D velocity field requires both vx and vy"):
        problem.velocity_field(vx)


def test_rejected_1d_y_velocity_field_does_not_mutate_problem() -> None:
    mesh = bt.mesh_1d(10)
    node_count = mesh.nx() + 1
    problem = (
        bt.Problem(mesh)
        .diffusivity(0.0)
        .initial_condition(2.0)
        .velocity_field(np.full(node_count, 0.1))
        .dirichlet(bt.Boundary.Left, 2.0)
    )

    with pytest.raises(ValueError, match="y velocity field must be zero"):
        problem.velocity_field(np.full(node_count, 0.5), np.full(node_count, 0.25))

    result = bt.solve(problem, end_time=0.0)
    np.testing.assert_array_equal(result.concentration, 2.0)
    assert problem.has_advection()
