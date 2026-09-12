"""Rejected edits must leave the executable model and its recorded physics intact."""

from __future__ import annotations

import numpy as np
import pytest

import biotransport as bt


def _custom_source(c, x, y, t):
    # Time dependence makes a lost callback recipe observable across saved frames.
    return 0.1 + 2 * t - 0.05 * c


def _problem(dimensions=1):
    mesh = bt.mesh_1d(10) if dimensions == 1 else bt.mesh_2d(4, 3)
    problem = (
        bt.Problem(mesh)
        .diffusivity(0.02)
        .velocity(0.01)
        .initial(np.linspace(0.4, 0.6, mesh.num_nodes()))
        .advection_scheme(bt.AdvectionScheme.UPWIND)
        .dirichlet("left", 0.4)
        .dirichlet("right", 0.6)
        .reaction(_custom_source, max_abs_dc=0.05)
        .add_constant_source(0.01)
    )
    return problem


def _field_state(array):
    return None if array is None else (array.shape, array.dtype.str, array.tobytes())


def _recorded_state(problem):
    recipe = problem._recipe
    return (
        recipe.diffusivity,
        _field_state(recipe.diffusivity_field),
        recipe.velocity,
        None
        if recipe.velocity_field is None
        else tuple(_field_state(field) for field in recipe.velocity_field),
        recipe.advection_scheme,
        tuple(recipe.boundaries.items()),
        tuple(
            (term.mode, term.kind, term.args, term.function, term.bound)
            for term in recipe.reactions
        ),
    )


def _native_state(problem):
    return (
        problem.diffusivity(),
        problem.has_uniform_diffusivity(),
        problem.has_advection(),
        problem.has_reaction(),
        problem.reaction_stability_bound_known(),
        problem.reaction_stability_rate_bound(),
        _field_state(problem.initial()),
        tuple((bc.type, bc.value, bc.a, bc.b, bc.c) for bc in problem.boundaries()),
    )


def _assert_rejected_edit_is_transactional(problem, edit):
    recorded = _recorded_state(problem)
    native = _native_state(problem)
    description = problem.describe()
    one_shot = bt.solve(problem, end_time=0.05, time_step=0.005)
    frames = bt.solve(problem, end_time=0.05, time_step=0.005, frames=5)

    with pytest.raises((TypeError, ValueError, OverflowError)):
        edit(problem)

    assert _native_state(problem) == native
    np.testing.assert_array_equal(
        bt.solve(problem, end_time=0.05, time_step=0.005).c, one_shot.c
    )
    # This also verifies replay retains the prior custom function and its clock.
    np.testing.assert_array_equal(
        bt.solve(problem, end_time=0.05, time_step=0.005, frames=5).history,
        frames.history,
    )
    assert _recorded_state(problem) == recorded
    assert problem.describe() == description


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda p: p.diffusivity(-1), id="negative-uniform-diffusion"),
        pytest.param(lambda p: p.diffusivity_field([1, 2]), id="diffusion-size"),
        pytest.param(lambda p: p.diffusivity([-1] * 11), id="negative-diffusion-field"),
        pytest.param(
            lambda p: p.diffusivity_field([np.nan] * 11), id="nan-diffusion-field"
        ),
        pytest.param(lambda p: p.velocity(0.2, 1), id="transverse-1d-velocity"),
        pytest.param(lambda p: p.velocity(np.inf), id="infinite-velocity"),
        pytest.param(lambda p: p.velocity_field([1, 2]), id="velocity-size"),
        pytest.param(
            lambda p: p.velocity_field([np.inf] * 11), id="infinite-velocity-field"
        ),
        pytest.param(
            lambda p: p.velocity_field([0.2] * 11, [1] * 11), id="transverse-1d-field"
        ),
        pytest.param(lambda p: p.advection_scheme("UPWIND"), id="invalid-scheme-type"),
        pytest.param(lambda p: p.linear_decay(-1), id="negative-decay"),
        pytest.param(lambda p: p.add_linear_decay(-1), id="negative-additive-decay"),
        pytest.param(lambda p: p.constant_source(np.nan), id="nan-source"),
        pytest.param(
            lambda p: p.add_constant_source(np.inf), id="infinite-additive-source"
        ),
        pytest.param(lambda p: p.michaelis_menten(-1, 0.1), id="negative-vmax"),
        pytest.param(
            lambda p: p.add_michaelis_menten(-1, 0.1), id="negative-additive-vmax"
        ),
        pytest.param(lambda p: p.michaelis_menten(1, 0), id="zero-km"),
        pytest.param(lambda p: p.add_michaelis_menten(1, 0), id="zero-additive-km"),
        pytest.param(lambda p: p.logistic_growth(-1, 1), id="negative-growth"),
        pytest.param(
            lambda p: p.add_logistic_growth(-1, 1), id="negative-additive-growth"
        ),
        pytest.param(lambda p: p.logistic_growth(1, 0), id="zero-capacity"),
        pytest.param(
            lambda p: p.add_logistic_growth(1, 0), id="zero-additive-capacity"
        ),
        pytest.param(lambda p: p.reaction(None), id="noncallable-reaction"),
        pytest.param(
            lambda p: p.add_reaction(None), id="noncallable-additive-reaction"
        ),
        pytest.param(
            lambda p: p.reaction(_custom_source, -1), id="negative-reaction-bound"
        ),
        pytest.param(
            lambda p: p.add_reaction(_custom_source, -1), id="negative-additive-bound"
        ),
        pytest.param(lambda p: p.initial([1, 2]), id="initial-size"),
        pytest.param(
            lambda p: p.initial_condition([np.nan] * 11), id="nan-initial-field"
        ),
        pytest.param(lambda p: p.initial(np.inf), id="infinite-initial-value"),
        pytest.param(
            lambda p: p.dirichlet("bottom", 1), id="nonexistent-dirichlet-side"
        ),
        pytest.param(lambda p: p.neumann("top", 1), id="nonexistent-neumann-side"),
        pytest.param(lambda p: p.sealed("bottom"), id="nonexistent-sealed-side"),
        pytest.param(lambda p: p.robin("top", 1, 1, 0), id="nonexistent-robin-side"),
        pytest.param(lambda p: p.robin("left", 0, 0, 1), id="degenerate-robin"),
        pytest.param(
            lambda p: p.boundary("left", object()), id="invalid-boundary-type"
        ),
        pytest.param(
            lambda p: p.boundary("left", bt.BoundaryCondition.dirichlet(np.nan)),
            id="nonfinite-boundary-object",
        ),
    ],
)
def test_invalid_edits_preserve_native_physics_and_custom_recipe(edit):
    _assert_rejected_edit_is_transactional(_problem(), edit)


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda p: p.velocity_field([0.2] * 20), id="missing-y-component"),
        pytest.param(lambda p: p.velocity_field([0.2] * 20, [0.1]), id="wrong-y-size"),
        pytest.param(
            lambda p: p.velocity_field([0.2] * 20, [np.nan] * 20), id="nonfinite-y"
        ),
    ],
)
def test_invalid_two_dimensional_fields_are_transactional(edit):
    _assert_rejected_edit_is_transactional(_problem(dimensions=2), edit)


@pytest.mark.parametrize("dimensions", [1, 2])
def test_recorded_coefficient_fields_own_their_storage(dimensions):
    problem = _problem(dimensions)
    count = problem.mesh().num_nodes()
    diffusion = np.linspace(0.01, 0.02, count)
    vx = np.linspace(0.01, 0.02, count)
    vy = np.linspace(0.02, 0.03, count) if dimensions == 2 else None
    problem.diffusivity_field(diffusion).velocity_field(vx, vy)
    recorded = _recorded_state(problem)
    description = problem.describe()
    groups = problem._recipe.dimensionless_numbers(problem.mesh(), 0.05)
    solution = bt.solve(problem, end_time=0.05, time_step=0.005, frames=5)

    diffusion[:] = 9
    vx[:] = 8
    if vy is not None:
        vy[:] = 7

    # C++ already owns coefficient copies; Python diagnostics must describe them.
    np.testing.assert_array_equal(
        bt.solve(problem, end_time=0.05, time_step=0.005, frames=5).history,
        solution.history,
    )
    assert _recorded_state(problem) == recorded
    assert problem.describe() == description
    assert problem._recipe.dimensionless_numbers(problem.mesh(), 0.05) == groups
