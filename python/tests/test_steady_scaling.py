"""Scale invariance and sparse steady solves against independent exact fields."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import biotransport as bt


@pytest.mark.parametrize(
    "diffusivity, concentration, length",
    [
        (1.0, 1.0, 1.0),
        (1.0e-20, 1.0, 1.0),
        (1.0, 1.0e-20, 1.0),
        (1.0e-20, 1.0e-20, 1.0),
        (1.0e20, 1.0e20, 1.0),
        (2.0e-9, 0.05, 1.0e-4),
        (1.0e-4, 1.0e-12, 1.0e3),
    ],
)
def test_equivalent_first_order_slabs_have_the_same_profile(
    diffusivity, concentration, length
):
    mesh = bt.mesh_1d(100, 0.0, length)
    problem = (
        bt.Problem(mesh)
        .diffusivity(diffusivity)
        .linear_decay(diffusivity / length**2)
        .initial(concentration)
        .dirichlet("left", concentration)
        .sealed("right")
    )
    result = bt.solve_steady(problem)
    exact = np.cosh(1.0 - result.x / length) / np.cosh(1.0)
    # The conservative half-cell boundary has max error 0.020565/N^2 for
    # phi=1. Its second-order refinement and balance are checked separately.
    np.testing.assert_allclose(result.c / concentration, exact, atol=2.1e-6, rtol=0)
    assert result.newton.converged
    assert result.newton.iterations > 0
    assert result.newton.linear_solver == "sparse_direct"
    np.testing.assert_array_equal(result.newton.solution, result.c)


@pytest.mark.parametrize("diffusivity", [1.0e-20, 1.0, 1.0e20])
def test_boundary_gradient_is_enforced_when_the_interior_residual_is_zero(diffusivity):
    mesh = bt.mesh_1d(40, 0.0, 2.0)
    x = np.linspace(0.0, 2.0, 41)
    problem = (
        bt.Problem(mesh)
        .diffusivity(diffusivity)
        .initial(1.0 + 0.25 * x)
        .dirichlet("left", 1.0)
        .sealed("right")
    )
    result = bt.solve_steady(problem)
    np.testing.assert_allclose(result.c, 1.0, atol=1e-10, rtol=0)


@pytest.mark.parametrize("concentration", [1.0e-20, 1.0, 1.0e20])
def test_source_and_nonzero_outward_gradient_preserve_a_quadratic_solution(
    concentration,
):
    diffusivity, length = 2e-9, 1e-4
    mesh = bt.mesh_1d(60, 0.0, length)
    # c/C = 2 + x/L - (x/L)^2; c'(0) = C/L, so dc/dn = -C/L on the left.
    problem = (
        bt.Problem(mesh)
        .diffusivity(diffusivity)
        .constant_source(2 * diffusivity * concentration / length**2)
        .initial(0.0)
        .neumann("left", -concentration / length)
        .dirichlet("right", 2 * concentration)
    )
    result = bt.solve_steady(problem)
    position = result.x / length
    exact = 2 + position - position**2
    np.testing.assert_allclose(result.c / concentration, exact, atol=1e-10, rtol=0)


def test_sparse_jacobian_matches_directional_derivative_with_variable_diffusivity():
    mesh = bt.mesh_1d(20)
    x = np.linspace(0, 1, 21)
    solver = bt.NonlinearDiffusionSolver(mesh, D=np.where(x < 0.5, 0.2, 1.0))
    solver.set_reaction(lambda u: u**3, lambda u: 3 * u**2)
    solver.set_boundary(bt.Boundary.Left, -0.3, "neumann")
    solver.set_boundary(bt.Boundary.Right, 0.7)
    state = 0.5 + 0.2 * np.sin(x)
    direction = np.cos(2.3 * x)
    matrix = solver._jacobian_1d(state)
    assert sparse.issparse(matrix)
    assert matrix.nnz <= 3 * state.size
    delta = 1e-5
    difference = (
        solver._residual_1d(state + delta * direction)
        - solver._residual_1d(state - delta * direction)
    ) / (2 * delta)
    np.testing.assert_allclose(matrix @ direction, difference, rtol=1e-7, atol=1e-8)


def test_two_node_dirichlet_problem_uses_a_valid_sparse_matrix():
    problem = (
        bt.Problem(bt.mesh_1d(1))
        .diffusivity(1.0)
        .initial(0.0)
        .dirichlet("left", 1.0)
        .dirichlet("right", 2.0)
    )
    np.testing.assert_allclose(bt.solve_steady(problem).c, [1.0, 2.0], atol=1e-12)


@pytest.mark.parametrize("cells", [17, 99, 100])
def test_sparse_pure_neumann_operator_refuses_an_undetermined_constant(cells):
    solver = bt.NonlinearDiffusionSolver(bt.mesh_1d(cells), D=1.0)
    solver.set_boundary(bt.Boundary.Left, 0.0, "neumann")
    solver.set_boundary(bt.Boundary.Right, 0.0, "neumann")
    with pytest.raises(bt.NewtonLinearSolveError, match="singular"):
        solver.solve(np.linspace(0.0, 1.0, cells + 1))


def test_constant_nullspace_uses_least_squares_only_with_explicit_opt_in():
    matrix = sparse.csr_matrix([[2.0, -2.0], [-3.0, 3.0]])
    solver = bt.NewtonRaphsonSolver(
        lambda u: matrix @ u - [4.0, -6.0],
        lambda u: matrix,
        n=2,
        allow_least_squares=True,
    )
    result = solver.solve([0.0, 0.0])
    assert result.converged
    assert result.used_least_squares
    assert result.linear_solver == "sparse_least_squares"
    np.testing.assert_allclose(result.solution, [1.0, -1.0], atol=1e-10)


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_sparse_reaction_resonance_refuses_nonconstant_nullspaces(mode):
    cells = 4
    rate = 4 * cells**2 * np.sin(mode * np.pi / (2 * cells)) ** 2
    solver = bt.NonlinearDiffusionSolver(bt.mesh_1d(cells), D=1.0)
    solver.set_reaction(lambda u: -rate * u, lambda u: np.full_like(u, -rate))
    solver.set_boundary(bt.Boundary.Left, 0.0)
    solver.set_boundary(bt.Boundary.Right, 0.0)
    with pytest.raises(bt.NewtonLinearSolveError, match="singular"):
        solver.solve(np.linspace(0.0, 1.0, cells + 1))


def test_nonresonant_negative_reaction_still_solves_without_changing_random_state():
    solver = bt.NonlinearDiffusionSolver(bt.mesh_1d(40), D=1.0)
    solver.set_reaction(lambda u: -2.0 * u, lambda u: np.full_like(u, -2.0))
    solver.set_boundary(bt.Boundary.Left, 1.0)
    solver.set_boundary(bt.Boundary.Right, 0.0)
    before = np.random.get_state()
    result = solver.solve(np.linspace(1.0, 0.0, 41))
    after = np.random.get_state()
    x = np.linspace(0.0, 1.0, 41)
    exact = np.sin(np.sqrt(2.0) * (1.0 - x)) / np.sin(np.sqrt(2.0))
    assert result.converged
    np.testing.assert_allclose(result.solution, exact, atol=3e-5, rtol=0)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_sparse_condition_estimation_preserves_subnormal_equation_scaling():
    scale = 1e-310
    matrix = sparse.eye(2, format="csr") * scale
    solver = bt.NewtonRaphsonSolver(lambda u: scale * (u - 1.0), lambda u: matrix, n=2)
    result = solver.solve([0.0, 0.0])
    assert result.converged
    assert result.linear_solver == "sparse_direct"
    np.testing.assert_allclose(result.solution, 1.0, atol=1e-14, rtol=0)


@pytest.mark.parametrize("concentration", [1e-20, 1.0, 1e20])
def test_scaled_nonlinear_reaction_and_added_source_preserve_known_equilibrium(
    concentration,
):
    problem = (
        bt.Problem(bt.mesh_1d(30))
        .diffusivity(1e-9)
        .michaelis_menten(Vmax=1e-9 * concentration, Km=0.5 * concentration)
        .add_constant_source((2.0 / 3.0) * 1e-9 * concentration)
        .initial(0.1 * concentration)
        .dirichlet("left", concentration)
        .dirichlet("right", concentration)
    )
    result = bt.solve_steady(problem)
    np.testing.assert_allclose(result.c / concentration, 1.0, atol=1e-9, rtol=0)


@pytest.mark.parametrize(
    "concentration, length", [(1e-20, 1e-4), (1.0, 1.0), (1e20, 1e3)]
)
def test_rectangle_poisson_source_matches_series_at_the_centre(concentration, length):
    diffusivity = 2e-9
    mesh = bt.mesh_2d(40, 20, 0.0, 2 * length, 0.0, length)
    problem = (
        bt.Problem(mesh)
        .diffusivity(diffusivity)
        .constant_source(diffusivity * concentration / length**2)
        .initial(0.0)
    )
    for side in ("left", "right", "bottom", "top"):
        problem.dirichlet(side, 0.0)
    result = bt.solve_steady(problem)
    # Separation of variables on [0,2] x [0,1], -laplacian(u)=1, u=0 on all sides.
    odd = np.arange(1, 200, 2)
    sign = (-1.0) ** np.arange(odd.size)
    exact_centre = 4 / np.pi**3 * np.sum(sign / odd**3 * (1 - 1 / np.cosh(odd * np.pi)))
    centre = result.c.reshape(21, 41)[10, 20] / concentration
    assert centre == pytest.approx(exact_centre, abs=2e-4)


def test_large_sparse_slab_retains_second_order_accuracy():
    problem = (
        bt.Problem(bt.mesh_1d(10_000))
        .diffusivity(1.0)
        .linear_decay(1.0)
        .initial(1.0)
        .dirichlet("left", 1.0)
        .sealed("right")
    )
    result = bt.solve_steady(problem)
    exact = np.cosh(1 - result.x) / np.cosh(1.0)
    np.testing.assert_allclose(result.c, exact, atol=3e-10, rtol=0)
    assert result.newton.linear_solver == "sparse_direct"


@pytest.mark.parametrize(
    "guess",
    [
        np.ones(11, dtype=bool),
        np.ones(11, dtype=complex),
        np.full(11, np.nan),
        np.ma.array(np.ones(11), mask=[True] + [False] * 10),
    ],
)
def test_normalization_rejects_unsafe_guess_values_without_coercion(guess):
    problem = (
        bt.Problem(bt.mesh_1d(10))
        .diffusivity(1.0)
        .initial(0.0)
        .dirichlet("left", 1.0)
        .dirichlet("right", 0.0)
    )
    with pytest.raises(ValueError, match="Initial guess"):
        bt.solve_steady(problem, guess=guess)


@pytest.mark.parametrize(
    "settings", [{"tol": True}, {"max_iterations": 1.5}, {"verbose": "yes"}]
)
def test_steady_settings_keep_the_lower_solver_validation(settings):
    problem = (
        bt.Problem(bt.mesh_1d(10))
        .diffusivity(1.0)
        .initial(0.0)
        .dirichlet("left", 1.0)
        .dirichlet("right", 0.0)
    )
    with pytest.raises(ValueError):
        bt.solve_steady(problem, **settings)
