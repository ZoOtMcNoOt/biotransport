"""Strict numerical-contract tests for the public Python Newton solvers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import biotransport as bt
from biotransport.newton_raphson import (
    ConvergenceCriterion,
    NewtonEvaluationError,
    NewtonLinearSolveError,
    NewtonLineSearchError,
    NewtonRaphsonSolver,
    NewtonSolverError,
    NonlinearDiffusionSolver,
    bistable,
    exponential_decay,
    hill_kinetics,
    michaelis_menten,
)


class TestNewtonRaphsonSolver:
    def test_typed_failures_are_top_level_public_api(self) -> None:
        assert bt.NewtonSolverError is NewtonSolverError
        assert bt.NewtonEvaluationError is NewtonEvaluationError
        assert bt.NewtonLinearSolveError is NewtonLinearSolveError
        assert bt.NewtonLineSearchError is NewtonLineSearchError

    def test_dense_direct_solve_converges_and_reports_method(self) -> None:
        def residual(u: np.ndarray) -> np.ndarray:
            return u**3 - u - 1.0

        def jacobian(u: np.ndarray) -> np.ndarray:
            return np.array([[3.0 * u[0] ** 2 - 1.0]])

        result = NewtonRaphsonSolver(residual, jacobian, n=1).solve([1.5])

        assert result.converged
        assert result.solution[0] == pytest.approx(1.3247179572, abs=1e-9)
        assert result.linear_solver == "dense_direct"
        assert not result.used_least_squares
        assert result.least_squares_rank is None
        assert all(np.isfinite(result.residual_history))

    def test_scaled_finite_difference_jacobian_converges(self) -> None:
        solver = NewtonRaphsonSolver(lambda u: u**2 - 4.0, n=1)
        positive = solver.solve([3.0])
        negative = solver.solve([-3.0])

        assert positive.converged and negative.converged
        assert positive.solution[0] == pytest.approx(2.0, abs=1e-8)
        assert negative.solution[0] == pytest.approx(-2.0, abs=1e-8)

    def test_sparse_jacobian_uses_sparse_direct_path(self) -> None:
        def residual(u: np.ndarray) -> np.ndarray:
            return np.array([u[0] ** 2 - 2.0])

        def jacobian(u: np.ndarray) -> sparse.csr_matrix:
            return sparse.csr_matrix([[2.0 * u[0]]])

        result = NewtonRaphsonSolver(residual, jacobian, n=1).solve([1.0])
        assert result.converged
        assert result.solution[0] == pytest.approx(np.sqrt(2.0), abs=1e-9)
        assert result.linear_solver == "sparse_direct"

    def test_singular_jacobian_fails_by_default(self) -> None:
        def residual(u: np.ndarray) -> np.ndarray:
            value = u[0] + u[1] - 1.0
            return np.array([value, value])

        singular = np.array([[1.0, 1.0], [1.0, 1.0]])
        solver = NewtonRaphsonSolver(residual, lambda _u: singular, n=2)

        with pytest.raises(
            NewtonLinearSolveError, match="least-squares fallback is disabled"
        ):
            solver.solve([0.0, 0.0])

    def test_least_squares_requires_opt_in_and_is_diagnostic(self) -> None:
        def residual(u: np.ndarray) -> np.ndarray:
            value = u[0] + u[1] - 1.0
            return np.array([value, value])

        singular = np.array([[1.0, 1.0], [1.0, 1.0]])
        solver = NewtonRaphsonSolver(
            residual,
            lambda _u: singular,
            n=2,
            allow_least_squares=True,
        )
        result = solver.solve([0.0, 0.0])

        assert result.converged
        np.testing.assert_allclose(result.solution, [0.5, 0.5], atol=1e-10)
        assert result.used_least_squares
        assert result.linear_solver == "dense_least_squares"
        assert result.least_squares_rank == 1

    def test_sparse_least_squares_rejects_condition_limit_exit(self) -> None:
        jacobian = sparse.diags([1.0, 1.0e-10, 0.0], format="csr")
        solver = NewtonRaphsonSolver(
            lambda _u: np.array([-1.0, -1.0, 0.0]),
            lambda _u: jacobian,
            n=3,
            allow_least_squares=True,
        )

        with pytest.raises(NewtonLinearSolveError, match="accuracy contract"):
            solver.solve(np.zeros(3))

    def test_failed_armijo_search_raises_instead_of_accepting_last_trial(self) -> None:
        solver = NewtonRaphsonSolver(
            lambda _u: np.array([1.0]),
            lambda _u: np.array([[1.0]]),
            n=1,
        ).set_parameters(line_search_max_iter=4)

        with pytest.raises(NewtonLineSearchError, match="failed after 4 trials"):
            solver.solve([0.0])

    def test_nonfinite_line_search_result_raises(self) -> None:
        def residual(u: np.ndarray) -> np.ndarray:
            return np.array([1.0 if u[0] == 0.0 else np.nan])

        solver = NewtonRaphsonSolver(residual, lambda _u: np.array([[1.0]]), n=1)
        with pytest.raises(NewtonEvaluationError, match="non-finite"):
            solver.solve([0.0])

    def test_finite_difference_never_calls_residual_with_nonfinite_trial(self) -> None:
        finite_calls = []

        def residual(u: np.ndarray) -> np.ndarray:
            finite_calls.append(bool(np.all(np.isfinite(u))))
            return np.ones(1)

        solver = NewtonRaphsonSolver(residual, n=1)
        with pytest.raises(NewtonEvaluationError, match="trial state is non-finite"):
            solver.solve([np.finfo(np.float64).max])

        assert finite_calls == [True]

    @pytest.mark.parametrize(
        "initial",
        [np.array([[1.0]]), np.array([np.nan]), np.array([np.inf]), np.array([])],
    )
    def test_invalid_initial_shapes_and_values_raise(self, initial: np.ndarray) -> None:
        solver = NewtonRaphsonSolver(lambda u: u, n=1)
        with pytest.raises(ValueError, match="Initial guess"):
            solver.solve(initial)

    @pytest.mark.parametrize(
        ("callback", "message"),
        [
            (lambda _u: 1.0, "must return shape"),
            (lambda _u: np.array([1.0, 2.0]), "must return shape"),
            (lambda _u: np.array([np.inf]), "non-finite"),
        ],
    )
    def test_invalid_residual_outputs_raise(self, callback, message: str) -> None:
        with pytest.raises(NewtonEvaluationError, match=message):
            NewtonRaphsonSolver(callback, n=1).solve([0.0])

    @pytest.mark.parametrize(
        "jacobian",
        [
            lambda _u: np.array([1.0]),
            lambda _u: np.array([[np.nan]]),
            lambda _u: sparse.csr_matrix([[np.inf]]),
        ],
    )
    def test_invalid_jacobian_outputs_raise(self, jacobian) -> None:
        solver = NewtonRaphsonSolver(lambda u: u + 1.0, jacobian, n=1)
        with pytest.raises(NewtonEvaluationError):
            solver.solve([0.0])

    @pytest.mark.parametrize(
        ("parameter", "value"),
        [
            ("max_iterations", 0),
            ("tol_residual", 0.0),
            ("tol_update", np.nan),
            ("damping", 1.1),
            ("fd_epsilon", -1.0),
            ("line_search_alpha", 1.0),
            ("line_search_max_iter", 0),
        ],
    )
    def test_invalid_settings_raise(self, parameter: str, value) -> None:
        solver = NewtonRaphsonSolver(lambda u: u, n=1)
        with pytest.raises(ValueError):
            solver.set_parameters(**{parameter: value})

    def test_max_iterations_returns_a_nonconverged_result(self) -> None:
        solver = NewtonRaphsonSolver(lambda u: u**2 - 2.0, n=1)
        solver.set_parameters(max_iterations=1, tol_residual=1e-14)
        result = solver.solve([10.0])

        assert not result.converged
        assert result.iterations == 1
        assert result.residual_history[-1] < result.residual_history[0]

    def test_update_criterion_and_disabled_line_search_remain_supported(self) -> None:
        solver = NewtonRaphsonSolver(
            lambda u: u**2 - 4.0,
            lambda u: np.array([[2.0 * u[0]]]),
            n=1,
        )
        solver.set_parameters(
            criterion=ConvergenceCriterion.UPDATE,
            use_line_search=False,
            damping=0.5,
        )
        result = solver.solve([3.0])
        assert result.converged
        assert result.solution[0] == pytest.approx(2.0, abs=1e-8)

    def test_tiny_damping_cannot_manufacture_update_convergence(self) -> None:
        solver = NewtonRaphsonSolver(
            lambda u: u - 100.0,
            lambda _u: np.ones((1, 1)),
            n=1,
        ).set_parameters(
            criterion=ConvergenceCriterion.UPDATE,
            use_line_search=False,
            damping=1.0e-13,
            tol_update=1.0e-10,
            max_iterations=3,
        )

        result = solver.solve([0.0])

        assert not result.converged
        assert result.iterations == 3
        assert result.residual_norm == pytest.approx(100.0, rel=1e-12)

    def test_both_criterion_cannot_converge_on_residual_scaling_alone(self) -> None:
        scale = 1.0e-21
        target = 1.0e12
        solver = NewtonRaphsonSolver(
            lambda u: scale * (u - target),
            lambda _u: np.array([[scale]]),
            n=1,
        )

        result = solver.solve([0.0])

        assert result.converged
        assert result.iterations == 1
        assert result.solution[0] == pytest.approx(target)
        assert result.update_norm == pytest.approx(0.0)

    def test_exact_root_does_not_require_a_nonsingular_jacobian(self) -> None:
        solver = NewtonRaphsonSolver(
            lambda u: u**2,
            lambda u: np.array([[2.0 * u[0]]]),
            n=1,
        ).set_parameters(criterion=ConvergenceCriterion.UPDATE)

        result = solver.solve([0.0])

        assert result.converged
        assert result.iterations == 0


class TestNonlinearDiffusionSolver:
    @staticmethod
    def _zero_dirichlet(solver: NonlinearDiffusionSolver) -> NonlinearDiffusionSolver:
        return solver.set_boundary(bt.Boundary.Left, 0.0).set_boundary(
            bt.Boundary.Right, 0.0
        )

    def test_missing_boundary_conditions_raise_before_newton(self) -> None:
        mesh = bt.StructuredMesh(10, 0.0, 1.0)
        solver = NonlinearDiffusionSolver(mesh, D=1.0)
        solver.set_boundary(bt.Boundary.Left, 0.0)

        with pytest.raises(ValueError, match="missing: Right"):
            solver.solve()

    def test_unknown_boundaries_and_boundary_types_raise(self) -> None:
        mesh = bt.StructuredMesh(10, 0.0, 1.0)
        solver = NonlinearDiffusionSolver(mesh, D=1.0)

        with pytest.raises(ValueError, match="do not exist"):
            solver.set_boundary(bt.Boundary.Bottom, 0.0)
        with pytest.raises(ValueError, match="dirichlet.*neumann"):
            solver.set_boundary(bt.Boundary.Left, 0.0, bc_type="periodic")
        with pytest.raises(ValueError, match="finite"):
            solver.set_boundary(bt.Boundary.Left, np.nan)

    def test_neumann_boundary_requires_three_nodes(self) -> None:
        solver = NonlinearDiffusionSolver(bt.StructuredMesh(1, 0.0, 1.0), D=1.0)
        with pytest.raises(ValueError, match="at least three nodes"):
            solver.set_boundary(bt.Boundary.Left, 0.0, bc_type="neumann")

    def test_2d_neumann_and_variable_diffusivity_are_explicitly_unsupported(
        self,
    ) -> None:
        mesh = bt.StructuredMesh(4, 3, 0.0, 1.0, 0.0, 1.0)
        with pytest.raises(NotImplementedError, match="Variable diffusivity"):
            NonlinearDiffusionSolver(mesh, D=np.ones((4, 5)))

        solver = NonlinearDiffusionSolver(mesh, D=1.0)
        with pytest.raises(NotImplementedError, match="Neumann"):
            solver.set_boundary(bt.Boundary.Left, 0.0, bc_type="neumann")

    @pytest.mark.parametrize("diffusivity", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_scalar_diffusivity_raises(self, diffusivity: float) -> None:
        mesh = bt.StructuredMesh(5, 0.0, 1.0)
        with pytest.raises(ValueError, match="diffusivity"):
            NonlinearDiffusionSolver(mesh, D=diffusivity)

    def test_invalid_fields_and_reaction_outputs_raise(self) -> None:
        mesh = bt.StructuredMesh(5, 0.0, 1.0)
        solver = self._zero_dirichlet(NonlinearDiffusionSolver(mesh, D=1.0))

        with pytest.raises(ValueError, match="Source"):
            solver.set_source(np.ones(5))
        with pytest.raises(ValueError, match="finite"):
            solver.set_source([0.0, 0.0, np.nan, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Initial guess"):
            solver.solve(np.zeros((6, 1)))

        solver.set_reaction(lambda _u: 1.0)
        with pytest.raises(
            NewtonEvaluationError, match="Reaction callback must return shape"
        ):
            solver.solve(np.zeros(6))

        solver.set_reaction(lambda u: np.ones_like(u), lambda _u: np.array([np.nan]))
        with pytest.raises(NewtonEvaluationError, match="Reaction derivative"):
            solver.solve(np.zeros(6))

    def test_poisson_solution_is_second_order(self) -> None:
        errors = []
        for cells in (10, 20, 40):
            mesh = bt.StructuredMesh(cells, 0.0, 1.0)
            x = np.linspace(0.0, 1.0, cells + 1)
            source = np.sin(np.pi * x)
            exact = source / np.pi**2
            solver = self._zero_dirichlet(NonlinearDiffusionSolver(mesh, D=1.0))
            result = solver.set_source(source).solve(np.zeros(cells + 1))

            assert result.converged
            errors.append(float(np.max(np.abs(result.solution - exact))))

        assert errors[0] / errors[1] > 3.9
        assert errors[1] / errors[2] > 3.9

    def test_variable_diffusivity_has_one_harmonic_interface_flux(self) -> None:
        cells = 6
        mesh = bt.StructuredMesh(cells, 0.0, 1.0)
        nodal_diffusivity = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1])
        solver = NonlinearDiffusionSolver(mesh, D=nodal_diffusivity)
        solver.set_boundary(bt.Boundary.Left, 1.0)
        solver.set_boundary(bt.Boundary.Right, 0.0)

        assert solver.face_diffusivity[2] == pytest.approx(2.0 / 11.0)
        resistance = mesh.dx() / solver.face_diffusivity
        exact = 1.0 - np.concatenate(([0.0], np.cumsum(resistance))) / np.sum(
            resistance
        )
        result = solver.solve(np.linspace(1.0, 0.0, cells + 1))

        assert result.converged
        np.testing.assert_allclose(result.solution, exact, rtol=2e-10, atol=2e-10)
        flux = (
            solver.face_diffusivity
            * (result.solution[:-1] - result.solution[1:])
            / mesh.dx()
        )
        np.testing.assert_allclose(
            flux, np.full(cells, flux[0]), rtol=2e-10, atol=2e-10
        )

    def test_diffusivity_assignment_rebuilds_faces_and_inplace_mutation_fails(
        self,
    ) -> None:
        mesh = bt.StructuredMesh(3, 0.0, 1.0)
        solver = NonlinearDiffusionSolver(mesh, D=np.ones(4))

        exposed = solver.D
        assert isinstance(exposed, np.ndarray)
        with pytest.raises(ValueError, match="read-only"):
            exposed[0] = 2.0

        solver.D = np.array([1.0, 2.0, 2.0, 2.0])
        assert solver.face_diffusivity[0] == pytest.approx(4.0 / 3.0)

    def test_1d_neumann_values_are_outward_normal_derivatives(self) -> None:
        cells = 20
        mesh = bt.StructuredMesh(cells, 0.0, 1.0)
        x = np.linspace(0.0, 1.0, cells + 1)
        solver = NonlinearDiffusionSolver(mesh, D=1.0)
        solver.set_boundary(bt.Boundary.Left, 1.0, bc_type="neumann")
        solver.set_boundary(bt.Boundary.Right, 0.0, bc_type="dirichlet")
        result = solver.solve(np.zeros(cells + 1))

        assert result.converged
        np.testing.assert_allclose(result.solution, 1.0 - x, atol=2e-10)
        outward_derivative = (
            3.0 * result.solution[0] - 4.0 * result.solution[1] + result.solution[2]
        ) / (2.0 * mesh.dx())
        assert outward_derivative == pytest.approx(1.0, abs=1e-10)

    def test_neumann_boundary_is_second_order_for_a_manufactured_cubic(self) -> None:
        errors = []
        for cells in (10, 20, 40):
            mesh = bt.StructuredMesh(cells, 0.0, 1.0)
            x = np.linspace(0.0, 1.0, cells + 1)
            exact = x**3
            source = -6.0 * x
            solver = NonlinearDiffusionSolver(mesh, D=1.0).set_source(source)
            solver.set_boundary(bt.Boundary.Left, 0.0, bc_type="neumann")
            solver.set_boundary(bt.Boundary.Right, 1.0, bc_type="dirichlet")
            result = solver.solve(exact * 0.5)

            assert result.converged
            errors.append(float(np.max(np.abs(result.solution - exact))))

        assert errors[0] / errors[1] > 3.9
        assert errors[1] / errors[2] > 3.9

    def test_unanchored_pure_neumann_problem_fails_as_singular(self) -> None:
        mesh = bt.StructuredMesh(10, 0.0, 1.0)
        solver = NonlinearDiffusionSolver(mesh, D=1.0)
        solver.set_boundary(bt.Boundary.Left, 0.0, bc_type="neumann")
        solver.set_boundary(bt.Boundary.Right, 0.0, bc_type="neumann")

        with pytest.raises(NewtonLinearSolveError):
            solver.solve(np.linspace(0.0, 1.0, 11))

    def test_reaction_diffusion_with_analytic_derivative(self) -> None:
        cells = 40
        mesh = bt.StructuredMesh(cells, 0.0, 1.0)
        reaction, derivative = exponential_decay(10.0)
        solver = NonlinearDiffusionSolver(mesh, D=1.0)
        solver.set_reaction(reaction, derivative)
        solver.set_boundary(bt.Boundary.Left, 1.0)
        solver.set_boundary(bt.Boundary.Right, 0.0)
        result = solver.solve(np.linspace(1.0, 0.0, cells + 1))

        assert result.converged
        assert result.residual_norm < 1e-9
        assert np.all(result.solution >= -1e-12)

    def test_2d_poisson_with_scalar_diffusivity(self) -> None:
        cells = 8
        mesh = bt.StructuredMesh(cells, cells, 0.0, 1.0, 0.0, 1.0)
        coordinate = np.linspace(0.0, 1.0, cells + 1)
        x, y = np.meshgrid(coordinate, coordinate)
        exact = np.sin(np.pi * x) * np.sin(np.pi * y)
        source = 2.0 * np.pi**2 * exact

        solver = NonlinearDiffusionSolver(mesh, D=1.0).set_source(source)
        for boundary in (
            bt.Boundary.Left,
            bt.Boundary.Right,
            bt.Boundary.Bottom,
            bt.Boundary.Top,
        ):
            solver.set_boundary(boundary, 0.0)
        result = solver.solve()

        assert result.converged
        assert result.solution.shape == exact.shape
        assert np.max(np.abs(result.solution - exact)) < 0.02

    def test_conflicting_2d_dirichlet_corner_traces_raise(self) -> None:
        mesh = bt.StructuredMesh(2, 2, 0.0, 1.0, 0.0, 1.0)
        solver = NonlinearDiffusionSolver(mesh, D=1.0)
        solver.set_boundary(bt.Boundary.Left, 0.0)
        solver.set_boundary(bt.Boundary.Right, 0.0)
        solver.set_boundary(bt.Boundary.Bottom, 1.0)
        solver.set_boundary(bt.Boundary.Top, 0.0)

        with pytest.raises(ValueError, match="Inconsistent 2D Dirichlet traces"):
            solver.solve()

    def test_roundoff_equivalent_2d_corner_traces_are_accepted(self) -> None:
        mesh = bt.StructuredMesh(2, 2, 0.0, 1.0, 0.0, 1.0)
        solver = NonlinearDiffusionSolver(mesh, D=1.0)
        solver.set_boundary(bt.Boundary.Left, 0.1 + 0.2)
        solver.set_boundary(bt.Boundary.Right, 0.3)
        solver.set_boundary(bt.Boundary.Bottom, 0.3)
        solver.set_boundary(bt.Boundary.Top, 0.3)

        result = solver.solve()

        assert result.converged
        np.testing.assert_allclose(result.solution, 0.3, rtol=0.0, atol=1.0e-14)


class TestReactionHelpers:
    def test_helpers_return_expected_values_and_derivatives(self) -> None:
        michaelis, michaelis_derivative = michaelis_menten(2.0, 0.5)
        hill, hill_derivative = hill_kinetics(1.0, 1.0, 2.0)
        bistable_reaction, bistable_derivative = bistable(0.3)
        decay, decay_derivative = exponential_decay(0.5)
        values = np.array([0.0, 0.5, 1.0])

        assert michaelis(values)[1] == pytest.approx(1.0)
        assert np.all(michaelis_derivative(values) > 0.0)
        assert hill(values)[2] == pytest.approx(0.5)
        assert np.all(np.isfinite(hill_derivative(values)))
        np.testing.assert_allclose(bistable_reaction([0.0, 0.3, 1.0]), 0.0, atol=1e-15)
        assert np.all(np.isfinite(bistable_derivative(values)))
        np.testing.assert_allclose(decay(values), 0.5 * values)
        np.testing.assert_allclose(decay_derivative(values), 0.5)

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: michaelis_menten(-1.0, 0.5),
            lambda: michaelis_menten(1.0, 0.0),
            lambda: hill_kinetics(1.0, 1.0, 0.5),
            lambda: hill_kinetics(np.nan, 1.0, 2.0),
            lambda: exponential_decay(-1.0),
            lambda: bistable(np.inf),
        ],
    )
    def test_invalid_helper_parameters_raise(self, factory) -> None:
        with pytest.raises(ValueError):
            factory()
