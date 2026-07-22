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


def test_unverified_method_is_rejected_instead_of_substituted() -> None:
    mesh = bt.mesh_1d(10)
    problem = bt.Problem(mesh).diffusivity(0.01)

    with pytest.raises(ValueError, match="verified conservative"):
        bt.solve(problem, end_time=0.1, method="crank_nicolson")
