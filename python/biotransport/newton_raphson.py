"""Validated Newton iteration and steady nonlinear diffusion.

The general solver treats user callbacks as an untrusted numerical boundary:
shapes and finite values are checked at every evaluation. Singular Jacobians
and failed line searches raise by default. A least-squares linear step is
available only through the explicit ``allow_least_squares`` opt-in and is
reported in :class:`NewtonResult`.

``NonlinearDiffusionSolver`` solves ``-div(D grad(u)) + R(u) = S`` on uniform
``StructuredMesh`` grids. Nodal variable diffusivity is supported in 1D with
harmonic face values. Variable diffusivity and Neumann boundaries are currently
rejected in 2D rather than being approximated by a different equation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, lsmr, spsolve

from ._core import Boundary, StructuredMesh


class ConvergenceCriterion(Enum):
    """Criterion used to accept a Newton solution."""

    RESIDUAL = "residual"
    UPDATE = "update"
    BOTH = "both"


class NewtonSolverError(RuntimeError):
    """Base class for numerical failures during Newton iteration."""


class NewtonEvaluationError(NewtonSolverError):
    """A residual or Jacobian callback returned invalid numerical data."""


class NewtonLinearSolveError(NewtonSolverError):
    """The Newton linear system could not be solved as configured."""


class NewtonLineSearchError(NewtonSolverError):
    """Armijo backtracking failed to find an acceptable finite step."""


@dataclass
class NewtonResult:
    """Solution and auditable convergence diagnostics."""

    solution: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float
    update_norm: float
    residual_history: list[float]
    linear_solver: str = "none"
    used_least_squares: bool = False
    least_squares_rank: Optional[int] = None
    line_search_backtracks: int = 0
    applied_update_norm: float = 0.0


class NewtonRaphsonSolver:
    """Solve a finite nonlinear system ``F(u) = 0`` with Newton iteration.

    Parameters
    ----------
    residual_func:
        Callback returning one finite vector with shape ``(n,)``.
    jacobian_func:
        Optional callback returning a finite dense or sparse ``(n, n)``
        matrix. A scaled forward-difference Jacobian is used when omitted.
    n:
        Number of unknowns.
    allow_least_squares:
        Explicitly permit a least-squares Newton step if the direct Jacobian
        solve is singular. It is disabled by default and surfaced in the
        returned diagnostics.
    """

    def __init__(
        self,
        residual_func: Callable[[np.ndarray], Any],
        jacobian_func: Optional[Callable[[np.ndarray], Any]] = None,
        n: int = 1,
        *,
        allow_least_squares: bool = False,
    ):
        if not callable(residual_func):
            raise TypeError("residual_func must be callable")
        if jacobian_func is not None and not callable(jacobian_func):
            raise TypeError("jacobian_func must be callable or None")
        self.n = self._positive_integer(n, "n")
        self.residual_func = residual_func
        self.jacobian_func = jacobian_func

        self.max_iterations = 50
        self.tol_residual = 1.0e-10
        self.tol_update = 1.0e-10
        self.criterion = ConvergenceCriterion.BOTH
        self.fd_epsilon = 1.0e-8
        self.use_line_search = True
        self.line_search_alpha = 1.0e-4
        self.line_search_max_iter = 10
        self.damping = 1.0
        self.verbose = False
        self.allow_least_squares = self._boolean(
            allow_least_squares, "allow_least_squares"
        )
        self.least_squares_rcond: Optional[float] = None

    @staticmethod
    def _positive_integer(value: Any, name: str) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @staticmethod
    def _positive_finite(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite positive number")
        converted = float(value)
        if not np.isfinite(converted) or converted <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
        return converted

    @staticmethod
    def _boolean(value: Any, name: str) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
        return bool(value)

    def set_parameters(
        self,
        max_iterations: Optional[int] = None,
        tol_residual: Optional[float] = None,
        tol_update: Optional[float] = None,
        criterion: Optional[ConvergenceCriterion] = None,
        use_line_search: Optional[bool] = None,
        damping: Optional[float] = None,
        verbose: Optional[bool] = None,
        allow_least_squares: Optional[bool] = None,
        fd_epsilon: Optional[float] = None,
        line_search_alpha: Optional[float] = None,
        line_search_max_iter: Optional[int] = None,
    ) -> "NewtonRaphsonSolver":
        """Validate and update solver settings; return ``self`` for chaining."""

        if max_iterations is not None:
            self.max_iterations = self._positive_integer(
                max_iterations, "max_iterations"
            )
        if tol_residual is not None:
            self.tol_residual = self._positive_finite(tol_residual, "tol_residual")
        if tol_update is not None:
            self.tol_update = self._positive_finite(tol_update, "tol_update")
        if criterion is not None:
            if not isinstance(criterion, ConvergenceCriterion):
                raise ValueError("criterion must be a ConvergenceCriterion")
            self.criterion = criterion
        if use_line_search is not None:
            self.use_line_search = self._boolean(use_line_search, "use_line_search")
        if damping is not None:
            converted = self._positive_finite(damping, "damping")
            if converted > 1.0:
                raise ValueError("damping must be in (0, 1]")
            self.damping = converted
        if verbose is not None:
            self.verbose = self._boolean(verbose, "verbose")
        if allow_least_squares is not None:
            self.allow_least_squares = self._boolean(
                allow_least_squares, "allow_least_squares"
            )
        if fd_epsilon is not None:
            self.fd_epsilon = self._positive_finite(fd_epsilon, "fd_epsilon")
        if line_search_alpha is not None:
            converted = self._positive_finite(line_search_alpha, "line_search_alpha")
            if converted >= 1.0:
                raise ValueError("line_search_alpha must be in (0, 1)")
            self.line_search_alpha = converted
        if line_search_max_iter is not None:
            self.line_search_max_iter = self._positive_integer(
                line_search_max_iter, "line_search_max_iter"
            )
        return self

    def _validate_settings(self) -> None:
        """Catch invalid direct attribute mutation before numerical work begins."""

        self.n = self._positive_integer(self.n, "n")
        self.max_iterations = self._positive_integer(
            self.max_iterations, "max_iterations"
        )
        self.tol_residual = self._positive_finite(self.tol_residual, "tol_residual")
        self.tol_update = self._positive_finite(self.tol_update, "tol_update")
        self.fd_epsilon = self._positive_finite(self.fd_epsilon, "fd_epsilon")
        self.line_search_max_iter = self._positive_integer(
            self.line_search_max_iter, "line_search_max_iter"
        )
        if not isinstance(self.criterion, ConvergenceCriterion):
            raise ValueError("criterion must be a ConvergenceCriterion")
        self.use_line_search = self._boolean(self.use_line_search, "use_line_search")
        self.verbose = self._boolean(self.verbose, "verbose")
        self.allow_least_squares = self._boolean(
            self.allow_least_squares, "allow_least_squares"
        )
        self.damping = self._positive_finite(self.damping, "damping")
        if self.damping > 1.0:
            raise ValueError("damping must be in (0, 1]")
        self.line_search_alpha = self._positive_finite(
            self.line_search_alpha, "line_search_alpha"
        )
        if self.line_search_alpha >= 1.0:
            raise ValueError("line_search_alpha must be in (0, 1)")
        if self.least_squares_rcond is not None:
            if (
                isinstance(self.least_squares_rcond, bool)
                or not isinstance(self.least_squares_rcond, Real)
                or not np.isfinite(float(self.least_squares_rcond))
                or float(self.least_squares_rcond) < 0.0
            ):
                raise ValueError(
                    "least_squares_rcond must be None or finite and non-negative"
                )

    def _initial_vector(self, value: Any) -> np.ndarray:
        try:
            if np.iscomplexobj(value):
                raise ValueError("Initial guess must be real-valued")
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Initial guess must be a numeric vector") from error
        if vector.shape != (self.n,):
            raise ValueError(
                f"Initial guess must have shape ({self.n},), got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("Initial guess must contain only finite values")
        return vector.copy()

    def _evaluate_residual(self, u: np.ndarray) -> np.ndarray:
        try:
            raw = self.residual_func(u.copy())
            if np.iscomplexobj(raw):
                raise NewtonEvaluationError("Residual callback must return real values")
            residual = np.asarray(raw, dtype=np.float64)
        except NewtonSolverError:
            raise
        except Exception as error:
            raise NewtonEvaluationError(
                "Residual callback raised an exception"
            ) from error
        if residual.shape != (self.n,):
            raise NewtonEvaluationError(
                f"Residual callback must return shape ({self.n},), got {residual.shape}"
            )
        if not np.all(np.isfinite(residual)):
            raise NewtonEvaluationError("Residual callback returned non-finite values")
        return residual.copy()

    def _evaluate_jacobian(self, u: np.ndarray) -> Any:
        if self.jacobian_func is None:
            raise AssertionError("Internal error: no Jacobian callback is configured")
        try:
            raw = self.jacobian_func(u.copy())
        except NewtonSolverError:
            raise
        except Exception as error:
            raise NewtonEvaluationError(
                "Jacobian callback raised an exception"
            ) from error

        if sparse.issparse(raw):
            try:
                if np.iscomplexobj(raw.data):
                    raise NewtonEvaluationError(
                        "Jacobian callback must return real values"
                    )
                matrix = raw.astype(np.float64).tocsr(copy=True)
            except (TypeError, ValueError) as error:
                raise NewtonEvaluationError(
                    "Sparse Jacobian must be numeric"
                ) from error
            if matrix.shape != (self.n, self.n):
                raise NewtonEvaluationError(
                    f"Jacobian callback must return shape ({self.n}, {self.n}), "
                    f"got {matrix.shape}"
                )
            if not np.all(np.isfinite(matrix.data)):
                raise NewtonEvaluationError(
                    "Jacobian callback returned non-finite values"
                )
            return matrix

        try:
            if np.iscomplexobj(raw):
                raise NewtonEvaluationError("Jacobian callback must return real values")
            matrix = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise NewtonEvaluationError("Dense Jacobian must be numeric") from error
        if matrix.shape != (self.n, self.n):
            raise NewtonEvaluationError(
                f"Jacobian callback must return shape ({self.n}, {self.n}), got {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise NewtonEvaluationError("Jacobian callback returned non-finite values")
        return matrix.copy()

    def _compute_jacobian_fd(self, u: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        matrix: np.ndarray = np.empty((self.n, self.n), dtype=np.float64)
        for column in range(self.n):
            perturbation = self.fd_epsilon * max(1.0, abs(float(u[column])))
            trial = u.copy()
            with np.errstate(over="ignore", invalid="ignore"):
                trial[column] += perturbation
            if not np.all(np.isfinite(trial)):
                raise NewtonEvaluationError(
                    "Finite-difference trial state is non-finite for component "
                    f"{column}"
                )
            if trial[column] == u[column]:
                trial[column] = np.nextafter(u[column], np.inf)
                perturbation = float(trial[column] - u[column])
            if (
                not np.all(np.isfinite(trial))
                or not np.isfinite(perturbation)
                or perturbation <= 0.0
            ):
                raise NewtonEvaluationError(
                    f"Could not construct a finite-difference perturbation for component {column}"
                )
            matrix[:, column] = (
                self._evaluate_residual(trial) - baseline
            ) / perturbation
        if not np.all(np.isfinite(matrix)):
            raise NewtonEvaluationError(
                "Finite-difference Jacobian contains non-finite values"
            )
        return matrix

    def _compute_jacobian(self, u: np.ndarray, residual: np.ndarray) -> Any:
        if self.jacobian_func is not None:
            return self._evaluate_jacobian(u)
        return self._compute_jacobian_fd(u, residual)

    @staticmethod
    def _finite_norm(values: np.ndarray, name: str) -> float:
        scale = float(np.max(np.abs(values))) if values.size else 0.0
        if scale == 0.0:
            return 0.0
        norm = scale * float(np.linalg.norm(values / scale))
        if not np.isfinite(norm):
            raise NewtonEvaluationError(f"{name} norm is not finite")
        return norm

    def _least_squares_step(
        self, matrix: Any, rhs: np.ndarray
    ) -> Tuple[np.ndarray, str, Optional[int]]:
        if sparse.issparse(matrix):
            try:
                output = lsmr(
                    matrix,
                    rhs,
                    atol=min(self.tol_residual, 1.0e-12),
                    btol=min(self.tol_residual, 1.0e-12),
                    maxiter=max(100, 10 * self.n),
                )
            except Exception as error:
                raise NewtonLinearSolveError(
                    "Sparse least-squares solve failed"
                ) from error
            step = np.asarray(output[0], dtype=np.float64)
            termination_code = int(output[1])
            if termination_code in (3, 6, 7):
                reasons = {
                    3: "the estimated condition number exceeded conlim",
                    6: "the estimated condition number exceeded the machine limit",
                    7: "the iteration limit was reached",
                }
                raise NewtonLinearSolveError(
                    "Sparse least-squares solve did not meet the configured "
                    f"accuracy contract: {reasons[termination_code]} "
                    f"(LSMR istop={termination_code})"
                )
            method = "sparse_least_squares"
            rank = None
        else:
            try:
                step, _, rank_value, _ = np.linalg.lstsq(
                    matrix, rhs, rcond=self.least_squares_rcond
                )
            except np.linalg.LinAlgError as error:
                raise NewtonLinearSolveError(
                    "Dense least-squares solve failed"
                ) from error
            method = "dense_least_squares"
            rank = int(rank_value)

        step = np.asarray(step, dtype=np.float64).reshape(-1)
        if step.shape != (self.n,) or not np.all(np.isfinite(step)):
            raise NewtonLinearSolveError(
                "Least-squares solve returned an invalid Newton step"
            )
        return step, method, rank

    def _linear_step(
        self, matrix: Any, residual: np.ndarray
    ) -> Tuple[np.ndarray, str, bool, Optional[int]]:
        rhs = -residual
        direct_error: Optional[BaseException] = None

        if sparse.issparse(matrix):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    step = np.asarray(spsolve(matrix, rhs), dtype=np.float64).reshape(
                        -1
                    )
                if step.shape != (self.n,) or not np.all(np.isfinite(step)):
                    raise np.linalg.LinAlgError("sparse solve returned non-finite data")
                return step, "sparse_direct", False, None
            except (
                MatrixRankWarning,
                np.linalg.LinAlgError,
                RuntimeError,
                ValueError,
            ) as error:
                direct_error = error
        else:
            try:
                numerical_rank = int(np.linalg.matrix_rank(matrix))
                if numerical_rank < self.n:
                    raise np.linalg.LinAlgError(
                        f"dense Jacobian has numerical rank {numerical_rank} < {self.n}"
                    )
                step = np.asarray(
                    np.linalg.solve(matrix, rhs), dtype=np.float64
                ).reshape(-1)
                if step.shape != (self.n,) or not np.all(np.isfinite(step)):
                    raise np.linalg.LinAlgError("dense solve returned non-finite data")
                return step, "dense_direct", False, None
            except np.linalg.LinAlgError as error:
                direct_error = error

        if not self.allow_least_squares:
            raise NewtonLinearSolveError(
                "Jacobian solve failed or was singular; least-squares fallback is disabled. "
                "Set allow_least_squares=True only when a minimum-norm step is scientifically "
                "appropriate."
            ) from direct_error

        step, method, rank = self._least_squares_step(matrix, rhs)
        return step, method, True, rank

    def _line_search(
        self, u: np.ndarray, direction: np.ndarray, residual: np.ndarray
    ) -> Tuple[float, np.ndarray, int]:
        baseline_norm = self._finite_norm(residual, "Residual")
        for attempt in range(self.line_search_max_iter):
            step_length = self.damping * (0.5**attempt)
            candidate = u + step_length * direction
            if not np.all(np.isfinite(candidate)):
                raise NewtonEvaluationError(
                    "Newton line-search candidate is non-finite"
                )
            candidate_residual = self._evaluate_residual(candidate)
            candidate_norm = self._finite_norm(candidate_residual, "Trial residual")
            required_norm = (1.0 - self.line_search_alpha * step_length) * baseline_norm
            if candidate_norm <= required_norm:
                return step_length, candidate_residual, attempt
        raise NewtonLineSearchError(
            f"Armijo line search failed after {self.line_search_max_iter} trials; "
            f"initial residual norm was {baseline_norm:.6e}"
        )

    def _make_result(
        self,
        solution: np.ndarray,
        converged: bool,
        iterations: int,
        residual_norm: float,
        update_norm: float,
        residual_history: list[float],
        linear_solver: str,
        used_least_squares: bool,
        least_squares_rank: Optional[int],
        line_search_backtracks: int,
        applied_update_norm: float = 0.0,
    ) -> NewtonResult:
        return NewtonResult(
            solution=solution.copy(),
            converged=converged,
            iterations=iterations,
            residual_norm=residual_norm,
            update_norm=update_norm,
            residual_history=list(residual_history),
            linear_solver=linear_solver,
            used_least_squares=used_least_squares,
            least_squares_rank=least_squares_rank,
            line_search_backtracks=line_search_backtracks,
            applied_update_norm=applied_update_norm,
        )

    def solve(self, u0: Any) -> NewtonResult:
        """Solve from ``u0`` without mutating the caller's array."""

        self._validate_settings()
        u = self._initial_vector(u0)
        residual = self._evaluate_residual(u)
        residual_norm = self._finite_norm(residual, "Residual")
        residual_history = [residual_norm]
        update_norm = 0.0
        applied_update_norm = 0.0
        linear_solver = "none"
        used_least_squares = False
        least_squares_rank: Optional[int] = None
        line_search_backtracks = 0

        if self.verbose:
            print(f"Newton iteration 0: ||F|| = {residual_norm:.3e}")

        # An exact root needs no Jacobian (which may legitimately be singular
        # there). Residual-only mode may also accept its configured tolerance.
        # BOTH mode must still inspect the Newton correction; otherwise simple
        # residual scaling could manufacture convergence with a huge update.
        if residual_norm == 0.0 or (
            self.criterion == ConvergenceCriterion.RESIDUAL
            and residual_norm <= self.tol_residual
        ):
            return self._make_result(
                u,
                True,
                0,
                residual_norm,
                0.0,
                residual_history,
                linear_solver,
                used_least_squares,
                least_squares_rank,
                line_search_backtracks,
            )

        for iteration in range(1, self.max_iterations + 1):
            matrix = self._compute_jacobian(u, residual)
            direction, method, used_fallback, rank = self._linear_step(matrix, residual)
            if used_fallback or linear_solver == "none":
                linear_solver = method
            used_least_squares = used_least_squares or used_fallback
            if rank is not None:
                least_squares_rank = rank

            try:
                direction_norm = self._finite_norm(direction, "Newton step")
            except NewtonEvaluationError as error:
                raise NewtonLinearSolveError(
                    "Newton step norm is non-finite"
                ) from error
            # Convergence must use the undamped Newton correction. Otherwise a
            # tiny user damping factor or a heavily backtracked line search can
            # manufacture an arbitrarily small applied displacement far from a
            # root. ``update_norm`` reports this correction, while
            # ``applied_update_norm`` separately reports the damped displacement.
            correction_converged = direction_norm <= self.tol_update

            if self.criterion == ConvergenceCriterion.UPDATE and correction_converged:
                return self._make_result(
                    u,
                    True,
                    iteration - 1,
                    residual_norm,
                    direction_norm,
                    residual_history,
                    linear_solver,
                    used_least_squares,
                    least_squares_rank,
                    line_search_backtracks,
                    applied_update_norm,
                )
            if (
                self.criterion == ConvergenceCriterion.BOTH
                and residual_norm <= self.tol_residual
                and correction_converged
            ):
                return self._make_result(
                    u,
                    True,
                    iteration - 1,
                    residual_norm,
                    direction_norm,
                    residual_history,
                    linear_solver,
                    used_least_squares,
                    least_squares_rank,
                    line_search_backtracks,
                    applied_update_norm,
                )

            if self.use_line_search:
                step_length, next_residual, backtracks = self._line_search(
                    u, direction, residual
                )
                line_search_backtracks += backtracks
            else:
                step_length = self.damping
                candidate = u + step_length * direction
                if not np.all(np.isfinite(candidate)):
                    raise NewtonEvaluationError(
                        "Newton update produced a non-finite candidate"
                    )
                next_residual = self._evaluate_residual(candidate)

            u = u + step_length * direction
            residual = next_residual
            update_norm = direction_norm
            applied_update_norm = step_length * direction_norm
            residual_norm = self._finite_norm(residual, "Residual")
            residual_history.append(residual_norm)

            if self.verbose:
                print(
                    f"Newton iteration {iteration}: ||F|| = {residual_norm:.3e}, "
                    f"||Newton correction|| = {update_norm:.3e}, "
                    f"||applied update|| = {applied_update_norm:.3e}"
                )

            residual_converged = residual_norm <= self.tol_residual
            converged = (
                (self.criterion == ConvergenceCriterion.RESIDUAL and residual_converged)
                or (
                    self.criterion == ConvergenceCriterion.UPDATE
                    and correction_converged
                )
                or (
                    self.criterion == ConvergenceCriterion.BOTH
                    and residual_converged
                    and correction_converged
                )
            )
            if converged:
                return self._make_result(
                    u,
                    True,
                    iteration,
                    residual_norm,
                    update_norm,
                    residual_history,
                    linear_solver,
                    used_least_squares,
                    least_squares_rank,
                    line_search_backtracks,
                    applied_update_norm,
                )

        return self._make_result(
            u,
            False,
            self.max_iterations,
            residual_norm,
            update_norm,
            residual_history,
            linear_solver,
            used_least_squares,
            least_squares_rank,
            line_search_backtracks,
            applied_update_norm,
        )


class NonlinearDiffusionSolver:
    """Steady solver for ``-div(D grad(u)) + R(u) = S``.

    Scalar positive diffusivity is supported in 1D and 2D. A positive nodal
    diffusivity vector is supported in 1D and converted to harmonic face
    values, giving one conservative interface flux. Variable diffusivity in 2D
    and 2D Neumann conditions are explicitly unsupported.

    One boundary condition is required for every domain side before ``solve``.
    One-dimensional Neumann values are outward-normal derivatives ``du/dn``.
    """

    def __init__(
        self, mesh: StructuredMesh, D: Union[float, np.ndarray, list[float]] = 1.0
    ):
        required_methods = ("is_1d", "nx", "dx", "num_nodes")
        if any(not hasattr(mesh, name) for name in required_methods):
            raise TypeError("mesh must be a StructuredMesh-compatible object")
        self.mesh = mesh
        self.is_1d = bool(mesh.is_1d())
        self.nx = int(mesh.nx())
        self.dx = float(mesh.dx())
        self._grid_shape: Tuple[int, ...]
        if not np.isfinite(self.dx) or self.dx <= 0.0:
            raise ValueError("Mesh spacing dx must be finite and positive")

        if self.is_1d:
            self.n = self.nx + 1
            self.ny: Optional[int] = None
            self.dy: Optional[float] = None
            self._grid_shape = (self.n,)
        else:
            if not hasattr(mesh, "ny") or not hasattr(mesh, "dy"):
                raise TypeError("A 2D mesh must provide ny() and dy()")
            self.ny = int(mesh.ny())
            self.dy = float(mesh.dy())
            if not np.isfinite(self.dy) or self.dy <= 0.0:
                raise ValueError("Mesh spacing dy must be finite and positive")
            self.n = (self.nx + 1) * (self.ny + 1)
            self._grid_shape = (self.ny + 1, self.nx + 1)

        self._scalar_diffusivity: Optional[float]
        self._nodal_diffusivity: Optional[np.ndarray]
        self._face_diffusivity: Optional[np.ndarray]
        self._configure_diffusivity(D)

        self.reaction_func: Optional[Callable[[np.ndarray], Any]] = None
        self.reaction_deriv: Optional[Callable[[np.ndarray], Any]] = None
        self.source: Optional[np.ndarray] = None
        self._bcs: dict[Boundary, Tuple[str, float]] = {}

        self.max_iterations = 50
        self.tol = 1.0e-10
        self.verbose = False
        self.use_line_search = True
        self.damping = 1.0
        self.allow_least_squares = False

    def _configure_diffusivity(self, diffusivity: Any) -> None:
        if isinstance(diffusivity, (str, bytes, bool, np.bool_)):
            raise ValueError("Diffusivity must be numeric")
        try:
            raw_array = np.asarray(diffusivity)
            if np.issubdtype(raw_array.dtype, np.bool_):
                raise ValueError("Diffusivity must not be boolean")
            if np.iscomplexobj(raw_array):
                raise ValueError("Diffusivity must be real-valued")
            array = np.asarray(diffusivity, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Diffusivity must be numeric") from error

        if array.ndim == 0:
            value = float(array)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("Scalar diffusivity must be finite and positive")
            self._scalar_diffusivity = value
            self._nodal_diffusivity = None
            if self.is_1d:
                self._face_diffusivity = np.full(self.n - 1, value, dtype=np.float64)
            else:
                self._face_diffusivity = None
            return

        if not self.is_1d:
            raise NotImplementedError(
                "Variable diffusivity is not implemented for 2D NonlinearDiffusionSolver; "
                "only scalar D is supported"
            )
        if array.shape != (self.n,):
            raise ValueError(
                f"One-dimensional nodal diffusivity must have shape ({self.n},), "
                f"got {array.shape}"
            )
        if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
            raise ValueError("Nodal diffusivity must contain finite positive values")

        nodal = array.copy()
        low = np.minimum(nodal[:-1], nodal[1:])
        high = np.maximum(nodal[:-1], nodal[1:])
        face = low * (2.0 / (1.0 + low / high))
        if not np.all(np.isfinite(face)) or np.any(face <= 0.0):
            raise ValueError("Harmonic face diffusivity is not finite and positive")
        self._scalar_diffusivity = None
        self._nodal_diffusivity = nodal
        self._face_diffusivity = face

    @property
    def D(self) -> Union[float, np.ndarray]:
        """Return scalar diffusivity or a read-only copy of nodal values.

        Assign ``solver.D = value`` to reconfigure diffusivity. In-place array
        mutation is rejected because it would otherwise desynchronize the
        cached harmonic face coefficients.
        """

        if self._scalar_diffusivity is not None:
            return self._scalar_diffusivity
        if self._nodal_diffusivity is None:
            raise AssertionError("Internal error: diffusivity is not configured")
        values = self._nodal_diffusivity.copy()
        values.setflags(write=False)
        return values

    @D.setter
    def D(self, diffusivity: Any) -> None:
        """Reconfigure diffusivity and rebuild conservative face values."""

        self._configure_diffusivity(diffusivity)

    @property
    def face_diffusivity(self) -> np.ndarray:
        """Return an owned 1D array of conservative harmonic face values."""

        if not self.is_1d or self._face_diffusivity is None:
            raise NotImplementedError(
                "Face diffusivity is exposed only for 1D problems"
            )
        return self._face_diffusivity.copy()

    def set_reaction(
        self,
        func: Callable[[np.ndarray], Any],
        derivative: Optional[Callable[[np.ndarray], Any]] = None,
    ) -> "NonlinearDiffusionSolver":
        if not callable(func):
            raise TypeError("Reaction function must be callable")
        if derivative is not None and not callable(derivative):
            raise TypeError("Reaction derivative must be callable or None")
        self.reaction_func = func
        self.reaction_deriv = derivative
        return self

    def _coerce_field(self, value: Any, name: str) -> np.ndarray:
        try:
            if np.iscomplexobj(value):
                raise ValueError(f"{name} must be real-valued")
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be numeric") from error
        accepted_shapes = {(self.n,)} if self.is_1d else {(self.n,), self._grid_shape}
        if array.shape not in accepted_shapes:
            expected = " or ".join(str(shape) for shape in sorted(accepted_shapes))
            raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array.reshape(-1).copy()

    def set_source(self, source: Any) -> "NonlinearDiffusionSolver":
        """Set a finite nodal source field with exactly the mesh shape."""

        self.source = self._coerce_field(source, "Source")
        return self

    def set_boundary(
        self, boundary: Boundary, value: float, bc_type: str = "dirichlet"
    ) -> "NonlinearDiffusionSolver":
        valid_boundaries = (
            Boundary.Left,
            Boundary.Right,
            Boundary.Bottom,
            Boundary.Top,
        )
        if boundary not in valid_boundaries:
            raise ValueError("Unknown boundary identifier")
        if self.is_1d and boundary in (Boundary.Bottom, Boundary.Top):
            raise ValueError("Bottom and Top boundaries do not exist on a 1D mesh")
        if not isinstance(bc_type, str):
            raise ValueError("bc_type must be 'dirichlet' or 'neumann'")
        normalized_type = bc_type.strip().lower()
        if normalized_type not in ("dirichlet", "neumann"):
            raise ValueError("bc_type must be 'dirichlet' or 'neumann'")
        if self.is_1d and normalized_type == "neumann" and self.n < 3:
            raise ValueError(
                "A 1D Neumann boundary requires at least three nodes for the "
                "second-order outward derivative"
            )
        if not self.is_1d and normalized_type == "neumann":
            raise NotImplementedError(
                "Two-dimensional Neumann boundaries are not implemented by "
                "NonlinearDiffusionSolver"
            )
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Boundary value must be finite")
        converted_value = float(value)
        if not np.isfinite(converted_value):
            raise ValueError("Boundary value must be finite")
        self._bcs[boundary] = (normalized_type, converted_value)
        return self

    def set_parameters(
        self,
        max_iterations: Optional[int] = None,
        tol: Optional[float] = None,
        verbose: Optional[bool] = None,
        use_line_search: Optional[bool] = None,
        damping: Optional[float] = None,
        allow_least_squares: Optional[bool] = None,
    ) -> "NonlinearDiffusionSolver":
        """Validate and update Newton settings for this PDE solve."""

        validator = NewtonRaphsonSolver(lambda u: u, n=1)
        if max_iterations is not None:
            self.max_iterations = validator._positive_integer(
                max_iterations, "max_iterations"
            )
        if tol is not None:
            self.tol = validator._positive_finite(tol, "tol")
        if verbose is not None:
            self.verbose = validator._boolean(verbose, "verbose")
        if use_line_search is not None:
            self.use_line_search = validator._boolean(
                use_line_search, "use_line_search"
            )
        if damping is not None:
            converted = validator._positive_finite(damping, "damping")
            if converted > 1.0:
                raise ValueError("damping must be in (0, 1]")
            self.damping = converted
        if allow_least_squares is not None:
            self.allow_least_squares = validator._boolean(
                allow_least_squares, "allow_least_squares"
            )
        return self

    def _validate_problem(self) -> None:
        required = (
            (Boundary.Left, Boundary.Right)
            if self.is_1d
            else (Boundary.Left, Boundary.Right, Boundary.Bottom, Boundary.Top)
        )
        unknown = [boundary for boundary in self._bcs if boundary not in required]
        if unknown:
            raise ValueError(
                f"Boundary set contains unsupported identifiers: {unknown}"
            )
        missing = [boundary.name for boundary in required if boundary not in self._bcs]
        if missing:
            raise ValueError(
                "A boundary condition is required on every side; missing: "
                + ", ".join(missing)
            )
        for boundary in required:
            bc_type, value = self._bcs[boundary]
            if bc_type not in ("dirichlet", "neumann") or not np.isfinite(value):
                raise ValueError(f"Boundary data for {boundary.name} are invalid")
            if not self.is_1d and bc_type == "neumann":
                raise NotImplementedError(
                    "Two-dimensional Neumann boundaries are unsupported"
                )

        if not self.is_1d:
            corner_pairs = (
                (Boundary.Left, Boundary.Bottom),
                (Boundary.Right, Boundary.Bottom),
                (Boundary.Left, Boundary.Top),
                (Boundary.Right, Boundary.Top),
            )
            for first, second in corner_pairs:
                first_value = self._bcs[first][1]
                second_value = self._bcs[second][1]
                scale = max(1.0, abs(first_value), abs(second_value))
                corner_tolerance = 64.0 * np.finfo(np.float64).eps * scale
                if abs(first_value - second_value) > corner_tolerance:
                    raise ValueError(
                        "Inconsistent 2D Dirichlet traces at the "
                        f"{first.name}-{second.name} corner: {first_value} != "
                        f"{second_value}. Set matching values at each corner."
                    )

    def _evaluate_reaction(self, u: np.ndarray) -> np.ndarray:
        if self.reaction_func is None:
            return np.zeros_like(u)
        try:
            raw = self.reaction_func(u.copy())
            if np.iscomplexobj(raw):
                raise NewtonEvaluationError("Reaction callback must return real values")
            result = np.asarray(raw, dtype=np.float64)
        except NewtonSolverError:
            raise
        except Exception as error:
            raise NewtonEvaluationError(
                "Reaction callback raised an exception"
            ) from error
        if result.shape != u.shape:
            raise NewtonEvaluationError(
                f"Reaction callback must return shape {u.shape}, got {result.shape}"
            )
        if not np.all(np.isfinite(result)):
            raise NewtonEvaluationError("Reaction callback returned non-finite values")
        return result

    def _evaluate_reaction_derivative(self, u: np.ndarray) -> np.ndarray:
        if self.reaction_deriv is None:
            raise AssertionError(
                "Internal error: reaction derivative is not configured"
            )
        try:
            raw = self.reaction_deriv(u.copy())
            if np.iscomplexobj(raw):
                raise NewtonEvaluationError(
                    "Reaction derivative must return real values"
                )
            result = np.asarray(raw, dtype=np.float64)
        except NewtonSolverError:
            raise
        except Exception as error:
            raise NewtonEvaluationError(
                "Reaction derivative callback raised an exception"
            ) from error
        if result.shape != u.shape:
            raise NewtonEvaluationError(
                f"Reaction derivative must return shape {u.shape}, got {result.shape}"
            )
        if not np.all(np.isfinite(result)):
            raise NewtonEvaluationError(
                "Reaction derivative returned non-finite values"
            )
        return result

    def _apply_bcs_1d(self, u: np.ndarray, residual: np.ndarray) -> np.ndarray:
        left_type, left_value = self._bcs[Boundary.Left]
        if left_type == "dirichlet":
            residual[0] = u[0] - left_value
        else:
            residual[0] = (3.0 * u[0] - 4.0 * u[1] + u[2]) / (
                2.0 * self.dx
            ) - left_value

        right_type, right_value = self._bcs[Boundary.Right]
        if right_type == "dirichlet":
            residual[-1] = u[-1] - right_value
        else:
            residual[-1] = (3.0 * u[-1] - 4.0 * u[-2] + u[-3]) / (
                2.0 * self.dx
            ) - right_value
        return residual

    def _apply_bcs_2d(self, u: np.ndarray, residual: np.ndarray) -> np.ndarray:
        values = {boundary: self._bcs[boundary][1] for boundary in self._bcs}
        residual[:, 0] = u[:, 0] - values[Boundary.Left]
        residual[:, -1] = u[:, -1] - values[Boundary.Right]
        residual[0, :] = u[0, :] - values[Boundary.Bottom]
        residual[-1, :] = u[-1, :] - values[Boundary.Top]
        return residual

    def _residual_1d(self, u: np.ndarray) -> np.ndarray:
        if u.shape != (self.n,) or not np.all(np.isfinite(u)):
            raise NewtonEvaluationError(
                "1D nonlinear diffusion state has invalid shape or values"
            )
        if self._face_diffusivity is None:
            raise AssertionError("Internal error: 1D face diffusivity is missing")
        residual = np.zeros_like(u)
        right_flux = self._face_diffusivity[1:] * (u[2:] - u[1:-1]) / self.dx
        left_flux = self._face_diffusivity[:-1] * (u[1:-1] - u[:-2]) / self.dx
        residual[1:-1] = -(right_flux - left_flux) / self.dx
        residual += self._evaluate_reaction(u)
        if self.source is not None:
            residual -= self.source
        return self._apply_bcs_1d(u, residual)

    def _residual_2d(self, u_flat: np.ndarray) -> np.ndarray:
        if self.ny is None or self.dy is None or self._scalar_diffusivity is None:
            raise AssertionError("Internal error: 2D solver geometry is incomplete")
        if u_flat.shape != (self.n,) or not np.all(np.isfinite(u_flat)):
            raise NewtonEvaluationError(
                "2D nonlinear diffusion state has invalid shape or values"
            )
        u = u_flat.reshape(self._grid_shape)
        residual = np.zeros_like(u)
        laplacian = (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / self.dx**2 + (
            u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]
        ) / self.dy**2
        residual[1:-1, 1:-1] = -self._scalar_diffusivity * laplacian
        residual += self._evaluate_reaction(u)
        if self.source is not None:
            residual -= self.source.reshape(self._grid_shape)
        return self._apply_bcs_2d(u, residual).reshape(-1)

    def _jacobian_1d(self, u: np.ndarray) -> np.ndarray:
        if self._face_diffusivity is None:
            raise AssertionError("Internal error: 1D face diffusivity is missing")
        matrix: np.ndarray = np.zeros((self.n, self.n), dtype=np.float64)
        inverse_dx2 = 1.0 / self.dx**2
        for index in range(1, self.n - 1):
            left = self._face_diffusivity[index - 1] * inverse_dx2
            right = self._face_diffusivity[index] * inverse_dx2
            matrix[index, index - 1] = -left
            matrix[index, index] = left + right
            matrix[index, index + 1] = -right

        if self.reaction_deriv is not None:
            derivative = self._evaluate_reaction_derivative(u)
            diagonal = np.arange(1, self.n - 1)
            matrix[diagonal, diagonal] += derivative[1:-1]

        left_type, _ = self._bcs[Boundary.Left]
        if left_type == "dirichlet":
            matrix[0, 0] = 1.0
        else:
            matrix[0, 0] = 3.0 / (2.0 * self.dx)
            matrix[0, 1] = -4.0 / (2.0 * self.dx)
            matrix[0, 2] = 1.0 / (2.0 * self.dx)

        right_type, _ = self._bcs[Boundary.Right]
        if right_type == "dirichlet":
            matrix[-1, -1] = 1.0
        else:
            matrix[-1, -1] = 3.0 / (2.0 * self.dx)
            matrix[-1, -2] = -4.0 / (2.0 * self.dx)
            matrix[-1, -3] = 1.0 / (2.0 * self.dx)
        return matrix

    def _default_initial_guess(self) -> np.ndarray:
        guess: np.ndarray = np.zeros(self.n, dtype=np.float64)
        if self.is_1d:
            if self._bcs[Boundary.Left][0] == "dirichlet":
                guess[0] = self._bcs[Boundary.Left][1]
            if self._bcs[Boundary.Right][0] == "dirichlet":
                guess[-1] = self._bcs[Boundary.Right][1]
            return guess

        grid = guess.reshape(self._grid_shape)
        dummy = np.zeros_like(grid)
        self._apply_bcs_2d(grid, dummy)
        # Boundary residual is grid - target, so subtract it to impose target.
        grid -= dummy
        return grid.reshape(-1)

    def solve(self, initial_guess: Optional[Any] = None) -> NewtonResult:
        """Validate the PDE and solve it from a flat or mesh-shaped guess."""

        self._validate_problem()
        if initial_guess is None:
            initial = self._default_initial_guess()
        else:
            initial = self._coerce_field(initial_guess, "Initial guess")

        residual_func: Callable[[np.ndarray], np.ndarray]
        jacobian_func: Optional[Callable[[np.ndarray], np.ndarray]]
        if self.is_1d:
            residual_func = self._residual_1d
            jacobian_func = (
                self._jacobian_1d
                if self.reaction_func is None or self.reaction_deriv is not None
                else None
            )
        else:
            residual_func = self._residual_2d
            jacobian_func = None

        solver = NewtonRaphsonSolver(
            residual_func=residual_func,
            jacobian_func=jacobian_func,
            n=self.n,
            allow_least_squares=self.allow_least_squares,
        )
        solver.set_parameters(
            max_iterations=self.max_iterations,
            tol_residual=self.tol,
            tol_update=self.tol,
            criterion=ConvergenceCriterion.BOTH,
            use_line_search=self.use_line_search,
            damping=self.damping,
            verbose=self.verbose,
        )
        result = solver.solve(initial)
        if not self.is_1d:
            result.solution = result.solution.reshape(self._grid_shape)
        return result


def _finite_parameter(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not np.isfinite(converted) or (positive and converted <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return converted


def michaelis_menten(
    vmax: float, km: float
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return ``R(u)=Vmax*u/(Km+u)`` and its derivative."""

    vmax_value = _finite_parameter(vmax, "vmax")
    if vmax_value < 0.0:
        raise ValueError("vmax must be non-negative")
    km_value = _finite_parameter(km, "km", positive=True)

    def reaction(u: np.ndarray) -> np.ndarray:
        values = np.asarray(u, dtype=np.float64)
        return vmax_value * values / (km_value + values)

    def derivative(u: np.ndarray) -> np.ndarray:
        values = np.asarray(u, dtype=np.float64)
        return vmax_value * km_value / (km_value + values) ** 2

    return reaction, derivative


def hill_kinetics(
    vmax: float, km: float, n: float
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return a non-negative-concentration Hill law and its derivative."""

    vmax_value = _finite_parameter(vmax, "vmax")
    if vmax_value < 0.0:
        raise ValueError("vmax must be non-negative")
    km_value = _finite_parameter(km, "km", positive=True)
    exponent = _finite_parameter(n, "n", positive=True)
    if exponent < 1.0:
        raise ValueError("n must be at least 1 so the derivative is finite at u=0")

    def reaction(u: np.ndarray) -> np.ndarray:
        values = np.asarray(u, dtype=np.float64)
        if np.any(values < 0.0):
            raise ValueError("Hill kinetics requires non-negative concentrations")
        powered = values**exponent
        return vmax_value * powered / (km_value**exponent + powered)

    def derivative(u: np.ndarray) -> np.ndarray:
        values = np.asarray(u, dtype=np.float64)
        if np.any(values < 0.0):
            raise ValueError("Hill kinetics requires non-negative concentrations")
        powered = values**exponent
        numerator = (
            vmax_value * exponent * values ** (exponent - 1.0) * km_value**exponent
        )
        return numerator / (km_value**exponent + powered) ** 2

    return reaction, derivative


def bistable(
    a: float = 0.0,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return ``R(u)=u(1-u)(u-a)`` and its derivative."""

    threshold = _finite_parameter(a, "a")

    def reaction(u: np.ndarray) -> np.ndarray:
        values = np.asarray(u, dtype=np.float64)
        return values * (1.0 - values) * (values - threshold)

    def derivative(u: np.ndarray) -> np.ndarray:
        values = np.asarray(u, dtype=np.float64)
        return (
            (1.0 - values) * (values - threshold)
            - values * (values - threshold)
            + values * (1.0 - values)
        )

    return reaction, derivative


def exponential_decay(
    k: float,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return the non-negative first-order term ``R(u)=k*u`` and derivative."""

    rate = _finite_parameter(k, "k")
    if rate < 0.0:
        raise ValueError("k must be non-negative")

    def reaction(u: np.ndarray) -> np.ndarray:
        return rate * np.asarray(u, dtype=np.float64)

    def derivative(u: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(u, dtype=np.float64), rate)

    return reaction, derivative
