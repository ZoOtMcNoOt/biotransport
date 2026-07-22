"""Scientific and contract tests for the high-order native kernels."""

import math

import numpy as np
import pytest

import biotransport as bt
from biotransport.high_order import (
    HighOrderDiffusionSolver,
    d2dx2,
    ddx,
    gradient_4th_order,
    integrate_explicit_runge_kutta,
    laplacian_2nd_order,
    laplacian_4th_order,
    laplacian_6th_order,
    verify_order_of_accuracy,
)


@pytest.mark.parametrize(
    ("order", "operator", "polynomial", "exact", "margin"),
    [
        (2, laplacian_2nd_order, lambda x: x**2, lambda x: 2.0 + 0.0 * x, 1),
        (4, laplacian_4th_order, lambda x: x**4, lambda x: 12.0 * x**2, 2),
        (6, laplacian_6th_order, lambda x: x**6, lambda x: 30.0 * x**4, 3),
    ],
)
def test_laplacian_is_exact_for_resolved_polynomial(
    order, operator, polynomial, exact, margin
):
    x = np.linspace(-1.0, 1.0, 41)
    numerical = operator(polynomial(x), x[1] - x[0])
    np.testing.assert_allclose(
        numerical[margin:-margin], exact(x)[margin:-margin], rtol=2e-9, atol=2e-9
    )
    assert numerical[0] == 0.0
    assert numerical[-1] == 0.0


@pytest.mark.parametrize(
    ("operator", "margin", "minimum_order"),
    [
        (laplacian_2nd_order, 1, 1.9),
        (laplacian_4th_order, 2, 3.8),
        (laplacian_6th_order, 3, 5.5),
    ],
)
def test_laplacian_observed_order_against_known_derivative(
    operator, margin, minimum_order
):
    k = 2.0 * np.pi
    result = verify_order_of_accuracy(
        lambda cells: lambda values: operator(values, 1.0 / cells),
        lambda x: np.sin(k * x),
        lambda x: -(k**2) * np.sin(k * x),
        grid_sizes=(20, 40, 80),
        interior_margin=margin,
    )
    assert result["observed_orders"][-1] > minimum_order
    assert result["norm"] == "L_inf"


def test_gradient_known_polynomial_and_fourth_order_convergence():
    x = np.linspace(-1.0, 1.0, 41)
    dx = x[1] - x[0]
    numerical = gradient_4th_order(x**4, dx)
    np.testing.assert_allclose(numerical[2:-2], 4.0 * x[2:-2] ** 3, atol=2e-12)

    errors = []
    for cells in (20, 40, 80):
        grid = np.linspace(0.0, 1.0, cells + 1)
        result = ddx(np.sin(2.0 * np.pi * grid), 1.0 / cells, order=4)
        exact = 2.0 * np.pi * np.cos(2.0 * np.pi * grid)
        errors.append(np.max(np.abs(result[2:-2] - exact[2:-2])))
    observed = math.log(errors[-2] / errors[-1], 2.0)
    assert observed > 3.8


def test_preferred_and_legacy_mesh_forms_agree_and_preserve_shape():
    mesh = bt.StructuredMesh(20, 0.0, 1.0)
    x = np.linspace(0.0, 1.0, 21)
    preferred = laplacian_4th_order(x**4, mesh.dx(), mesh=mesh)
    legacy = laplacian_4th_order(mesh, x**4)
    np.testing.assert_array_equal(preferred, legacy)
    assert preferred.shape == x.shape


def test_two_dimensional_laplacian_preserves_shape_and_is_exact_deep_inside():
    nx, ny = 12, 8
    dx, dy = 0.1, 0.2
    x = np.arange(nx + 1) * dx
    y = np.arange(ny + 1) * dy
    xx, yy = np.meshgrid(x, y)
    field = xx**4 + yy**4
    exact = 12.0 * xx**2 + 12.0 * yy**2

    result = laplacian_4th_order(field, dx, dy)
    assert result.shape == field.shape
    np.testing.assert_allclose(result[2:-2, 2:-2], exact[2:-2, 2:-2], atol=2e-11)
    assert np.all(result[0, :] == 0.0)
    assert np.all(result[:, 0] == 0.0)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: laplacian_2nd_order(np.ones(5), 0.0), ValueError),
        (lambda: laplacian_2nd_order(np.ones(5), np.nan), ValueError),
        (lambda: laplacian_2nd_order(np.ones((2, 2, 2)), 1.0), ValueError),
        (lambda: laplacian_2nd_order(np.array([1.0, np.nan, 2.0]), 1.0), ValueError),
        (lambda: laplacian_4th_order(np.ones(4), 1.0), ValueError),
        (lambda: laplacian_6th_order(np.ones((8, 8)), 1.0), ValueError),
        (lambda: gradient_4th_order(np.ones(4), 1.0), ValueError),
        (lambda: d2dx2(np.ones(8), 1.0, order=3), ValueError),
        (lambda: ddx(np.ones(8), 1.0, order=6), ValueError),
        (lambda: laplacian_2nd_order(["1", "2", "3"], 1.0), TypeError),
    ],
)
def test_spatial_operator_validation(call, error):
    with pytest.raises(error):
        call()


def test_mesh_shape_and_spacing_are_not_silently_ignored():
    mesh = bt.StructuredMesh(10, 0.0, 1.0)
    with pytest.raises(ValueError, match="shape"):
        laplacian_4th_order(np.ones(10), mesh.dx(), mesh=mesh)
    with pytest.raises(ValueError, match="does not match"):
        laplacian_4th_order(np.ones(11), 0.2, mesh=mesh)


@pytest.mark.parametrize(
    ("order", "spectral_radius"),
    [(2, 4.0), (4, 16.0 / 3.0), (6, 272.0 / 45.0)],
)
def test_stable_dt_uses_exact_centered_stencil_spectral_radius(order, spectral_radius):
    mesh = bt.StructuredMesh(10, 0.0, 1.0)
    solver = HighOrderDiffusionSolver(mesh, D=0.5, order=order, safety_factor=0.4)
    expected = 2.0 * 0.4 * mesh.dx() ** 2 / (0.5 * spectral_radius)
    assert solver.compute_stable_dt() == pytest.approx(expected, rel=2e-15)


def test_two_dimensional_stable_dt_accounts_for_both_grid_directions():
    mesh = bt.StructuredMesh(10, 8, 0.0, 1.0, 0.0, 2.0)
    solver = HighOrderDiffusionSolver(mesh, D=0.25, order=4, safety_factor=0.7)
    inverse_spacing_sum = 1.0 / mesh.dx() ** 2 + 1.0 / mesh.dy() ** 2
    expected = 2.0 * 0.7 / (0.25 * (16.0 / 3.0) * inverse_spacing_sum)
    assert solver.compute_stable_dt() == pytest.approx(expected, rel=2e-15)


def test_native_diffusion_ends_exactly_and_reports_nominal_and_last_step():
    mesh = bt.StructuredMesh(20, 0.0, 1.0)
    solver = HighOrderDiffusionSolver(mesh, D=0.01, order=2)
    initial = np.zeros(21)
    result = solver.solve(initial, t_end=0.1, dt=0.03)

    assert result.time == 0.1
    assert result.steps == 4
    assert result.dt == 0.03
    assert result.last_dt == pytest.approx(0.01)
    assert result.order == result.interior_order == 2
    assert result.boundary_order == 2
    assert result.temporal_order == 1


def test_native_diffusion_matches_sine_mode_and_preserves_input():
    mesh = bt.StructuredMesh(40, 0.0, 1.0)
    x = np.linspace(0.0, 1.0, 41)
    initial = np.sin(np.pi * x)
    original = initial.copy()
    diffusivity = 0.1
    end_time = 0.01
    solver = HighOrderDiffusionSolver(mesh, D=diffusivity, order=4)

    result = solver.solve(initial, end_time)
    exact = np.exp(-diffusivity * np.pi**2 * end_time) * np.sin(np.pi * x)
    np.testing.assert_array_equal(initial, original)
    np.testing.assert_allclose(result.solution[1:-1], exact[1:-1], atol=2e-4)


def test_diffusion_boundaries_are_applied_at_zero_time_and_corner_rule_is_explicit():
    mesh = bt.StructuredMesh(4, 4, 0.0, 1.0, 0.0, 1.0)
    solver = HighOrderDiffusionSolver(mesh, D=0.1, order=4)
    solver.set_boundary(bt.Boundary.Left, 1.0)
    solver.set_boundary(bt.Boundary.Right, 2.0)
    solver.set_boundary(bt.Boundary.Bottom, 3.0)
    solver.set_boundary(bt.Boundary.Top, 4.0)

    result = solver.solve(np.zeros(25), t_end=0.0)
    assert result.solution.shape == (25,)
    field = result.solution.reshape(5, 5)
    np.testing.assert_array_equal(field[0, :], np.full(5, 3.0))
    np.testing.assert_array_equal(field[-1, :], np.full(5, 4.0))
    np.testing.assert_array_equal(field[1:-1, 0], np.full(3, 1.0))
    np.testing.assert_array_equal(field[1:-1, -1], np.full(3, 2.0))


def test_diffusion_callback_receives_isolated_shaped_copy():
    mesh = bt.StructuredMesh(8, 8, 0.0, 1.0, 0.0, 1.0)
    solver = HighOrderDiffusionSolver(mesh, D=0.01, order=2)
    seen_shapes = []

    def callback(_time, state):
        seen_shapes.append(state.shape)
        state[:] = np.nan

    result = solver.solve(np.ones((9, 9)), t_end=0.01, callback=callback)
    assert seen_shapes
    assert set(seen_shapes) == {(9, 9)}
    assert np.all(np.isfinite(result.solution))


def test_diffusion_rejects_unsafe_or_ill_defined_inputs():
    mesh = bt.StructuredMesh(20, 0.0, 1.0)
    solver = HighOrderDiffusionSolver(mesh, D=0.01, order=2)
    with pytest.raises(ValueError, match="stability"):
        solver.solve(np.zeros(21), t_end=0.1, dt=1.01 * solver.compute_stable_dt())
    with pytest.raises(ValueError, match="shape"):
        solver.solve(np.zeros(20), t_end=0.1)
    with pytest.raises(ValueError, match="finite"):
        solver.solve(np.full(21, np.nan), t_end=0.1)
    with pytest.raises(ValueError, match="nonnegative"):
        solver.solve(np.zeros(21), t_end=-1.0)
    with pytest.raises(ValueError, match="not a boundary"):
        solver.set_boundary(bt.Boundary.Bottom, 0.0)

    with pytest.raises(ValueError, match="positive"):
        HighOrderDiffusionSolver(mesh, D=0.0)
    with pytest.raises(TypeError, match="real number"):
        HighOrderDiffusionSolver(mesh, D=np.bool_(True))
    with pytest.raises(ValueError, match="safety_factor"):
        HighOrderDiffusionSolver(mesh, D=0.1, safety_factor=1.1)
    with pytest.raises(ValueError, match="1D"):
        HighOrderDiffusionSolver(
            bt.StructuredMesh(8, 8, 0.0, 1.0, 0.0, 1.0), D=0.1, order=6
        )


def test_heun_uses_correct_nonautonomous_stage_times():
    times = []

    def rhs(_state, time):
        times.append(time)
        return np.array([time])

    result = integrate_explicit_runge_kutta(
        np.array([2.0]), rhs, t_end=1.0, dt=0.3, t_start=0.2, method="heun"
    )
    expected = 2.0 + 0.5 * (1.0**2 - 0.2**2)
    assert result.solution[0] == pytest.approx(expected, abs=2e-15)
    assert result.time == 1.0
    assert result.steps == 3
    assert result.last_dt == pytest.approx(0.2)
    assert result.order == 2
    assert times == pytest.approx([0.2, 0.5, 0.5, 0.8, 0.8, 1.0])


@pytest.mark.parametrize(("method", "minimum_order"), [("heun", 1.8), ("rk4", 3.8)])
def test_runge_kutta_observed_order_for_exponential_growth(method, minimum_order):
    errors = []
    for step in (0.2, 0.1, 0.05):
        result = integrate_explicit_runge_kutta(
            [1.0], lambda state: state, 1.0, step, method=method, autonomous=True
        )
        errors.append(abs(result.solution[0] - math.e))
    observed = math.log(errors[-2] / errors[-1], 2.0)
    assert observed > minimum_order


def test_runge_kutta_preserves_multidimensional_shape_and_initial_state():
    initial = np.arange(6.0).reshape(2, 3)
    original = initial.copy()
    shapes = []

    def rhs(state):
        shapes.append(state.shape)
        return -state

    result = integrate_explicit_runge_kutta(
        initial, rhs, 0.3, 0.1, method="classical rk4", autonomous=True
    )
    np.testing.assert_array_equal(initial, original)
    assert result.solution.shape == initial.shape
    assert set(shapes) == {(2, 3)}
    np.testing.assert_allclose(result.solution, initial * np.exp(-0.3), rtol=2e-6)


def test_runge_kutta_large_absolute_clock_does_not_skip_the_interval():
    initial_time = 1.0e12
    end_time = initial_time + 0.01
    duration = end_time - initial_time

    result = integrate_explicit_runge_kutta(
        [3.0],
        lambda _state, _time: np.ones(1),
        end_time,
        0.001,
        t_start=initial_time,
        method="heun",
    )

    assert result.time == end_time
    assert result.steps > 1
    assert result.solution[0] == pytest.approx(3.0 + duration, abs=5e-15)


def test_runge_kutta_callback_cannot_alias_accepted_state():
    initial = np.array([1.0])

    def mutating_rhs(state):
        state[:] = 100.0
        return np.zeros_like(state)

    result = integrate_explicit_runge_kutta(
        initial, mutating_rhs, 1.0, 0.25, autonomous=True
    )
    assert initial[0] == 1.0
    assert result.solution[0] == 1.0


@pytest.mark.parametrize(
    "rhs",
    [
        lambda _state: np.array([1.0, 2.0]),
        lambda _state: np.array([np.nan]),
        lambda _state: np.array([np.inf]),
    ],
)
def test_runge_kutta_rejects_wrong_dimension_and_nonfinite_derivatives(rhs):
    with pytest.raises(ValueError):
        integrate_explicit_runge_kutta([1.0], rhs, 1.0, 0.1, autonomous=True)


def test_runge_kutta_signature_and_integration_validation_are_explicit():
    with pytest.raises(TypeError):
        integrate_explicit_runge_kutta(
            [1.0], lambda state: state, 1.0, 0.1, autonomous=False
        )
    with pytest.raises(ValueError, match="precede"):
        integrate_explicit_runge_kutta(
            [1.0], lambda state: state, -1.0, 0.1, autonomous=True
        )
    with pytest.raises(ValueError, match="positive"):
        integrate_explicit_runge_kutta(
            [1.0], lambda state: state, 1.0, 0.0, autonomous=True
        )
    with pytest.raises(ValueError, match="method"):
        integrate_explicit_runge_kutta(
            [1.0], lambda state: state, 1.0, 0.1, method="bogus", autonomous=True
        )
    with pytest.raises(ValueError, match="maximum_steps"):
        integrate_explicit_runge_kutta(
            [1.0], lambda state: state, 1.0, 0.1, autonomous=True, maximum_steps=0
        )
    with pytest.raises(RuntimeError, match="maximum_steps"):
        integrate_explicit_runge_kutta(
            [1.0], lambda state: state, 1.0, 0.1, autonomous=True, maximum_steps=2
        )
    with pytest.raises(ValueError, match="empty"):
        integrate_explicit_runge_kutta(
            [], lambda state: state, 1.0, 0.1, autonomous=True
        )
    with pytest.raises(OverflowError, match="interval"):
        integrate_explicit_runge_kutta(
            [1.0],
            lambda state: state,
            1.0e308,
            1.0,
            t_start=-1.0e308,
            autonomous=True,
        )


def test_verify_order_requires_independent_exact_derivative_and_valid_shapes():
    with pytest.raises(TypeError):
        verify_order_of_accuracy(lambda _n: lambda u: u, lambda x: x)  # type: ignore

    with pytest.raises(ValueError, match="preserve"):
        verify_order_of_accuracy(
            lambda _n: lambda u: u,
            lambda x: x[:-1],
            lambda x: x,
            grid_sizes=(20, 40),
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        verify_order_of_accuracy(
            lambda _n: lambda u: u,
            lambda x: x,
            lambda x: x,
            grid_sizes=(40, 20),
        )
