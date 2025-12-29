"""
Newton-Raphson Iteration for Nonlinear Steady-State Problems
=============================================================

Solves nonlinear systems F(u) = 0 using Newton-Raphson iteration:

    u^{n+1} = u^n - J^{-1} F(u^n)

where J is the Jacobian matrix ∂F/∂u.

Common biotransport applications:
- Nonlinear reaction-diffusion: -D∇²u + R(u) = 0
- Michaelis-Menten kinetics: R(u) = Vmax*u/(Km + u)
- Nonlinear boundary conditions
- Coupled nonlinear systems

Features:
- Automatic Jacobian via finite differences (or user-provided)
- Line search for global convergence
- Multiple convergence criteria (residual, update, both)
- Damping for improved stability

Example usage:
    >>> solver = NonlinearDiffusionSolver(mesh, D=1.0)
    >>> solver.set_reaction(lambda u: u**2 - 1)  # Bistable
    >>> solver.set_boundary(Boundary.Left, 1.0)
    >>> solver.set_boundary(Boundary.Right, -1.0)
    >>> result = solver.solve(initial_guess)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ._core import Boundary, StructuredMesh


class ConvergenceCriterion(Enum):
    """Convergence criterion for Newton iteration."""

    RESIDUAL = "residual"  # ||F(u)|| < tol
    UPDATE = "update"  # ||du|| < tol
    BOTH = "both"  # Both residual and update


@dataclass
class NewtonResult:
    """Result from Newton-Raphson solver."""

    solution: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float
    update_norm: float
    residual_history: list[float]


class NewtonRaphsonSolver:
    """
    General Newton-Raphson solver for nonlinear systems F(u) = 0.

    Parameters
    ----------
    residual_func : callable
        Function F(u) returning the residual vector
    jacobian_func : callable, optional
        Function J(u) returning the Jacobian matrix (dense or sparse)
        If None, Jacobian is computed via finite differences
    n : int
        Problem size (length of u)

    Example
    -------
    >>> def residual(u):
    ...     return u**3 - u - 1  # Find root of x³ - x - 1 = 0
    >>> solver = NewtonRaphsonSolver(residual, n=1)
    >>> result = solver.solve(np.array([1.5]))
    """

    def __init__(
        self,
        residual_func: Callable[[np.ndarray], np.ndarray],
        jacobian_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        n: int = 1,
    ):
        self.residual_func = residual_func
        self.jacobian_func = jacobian_func
        self.n = n

        # Default solver parameters
        self.max_iterations = 50
        self.tol_residual = 1e-10
        self.tol_update = 1e-10
        self.criterion = ConvergenceCriterion.BOTH
        self.fd_epsilon = 1e-8  # Finite difference step
        self.use_line_search = True
        self.line_search_alpha = 1e-4  # Armijo parameter
        self.line_search_max_iter = 10
        self.damping = 1.0  # Initial damping factor
        self.verbose = False

    def set_parameters(
        self,
        max_iterations: Optional[int] = None,
        tol_residual: Optional[float] = None,
        tol_update: Optional[float] = None,
        criterion: Optional[ConvergenceCriterion] = None,
        use_line_search: Optional[bool] = None,
        damping: Optional[float] = None,
        verbose: Optional[bool] = None,
    ) -> "NewtonRaphsonSolver":
        """Set solver parameters. Returns self for chaining."""
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if tol_residual is not None:
            self.tol_residual = tol_residual
        if tol_update is not None:
            self.tol_update = tol_update
        if criterion is not None:
            self.criterion = criterion
        if use_line_search is not None:
            self.use_line_search = use_line_search
        if damping is not None:
            self.damping = damping
        if verbose is not None:
            self.verbose = verbose
        return self

    def _compute_jacobian_fd(self, u: np.ndarray) -> np.ndarray:
        """Compute Jacobian via forward finite differences."""
        F0 = self.residual_func(u)
        n = len(u)
        J = np.zeros((n, n))

        for j in range(n):
            u_pert = u.copy()
            u_pert[j] += self.fd_epsilon
            F_pert = self.residual_func(u_pert)
            J[:, j] = (F_pert - F0) / self.fd_epsilon

        return J

    def _compute_jacobian(self, u: np.ndarray) -> np.ndarray:
        """Compute Jacobian (user-provided or finite difference)."""
        if self.jacobian_func is not None:
            return self.jacobian_func(u)
        return self._compute_jacobian_fd(u)

    def _line_search(
        self, u: np.ndarray, du: np.ndarray, F0: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Armijo line search to find step length.

        Returns (step_length, F_new)
        """
        norm_F0 = np.linalg.norm(F0)
        step = self.damping

        for _ in range(self.line_search_max_iter):
            u_new = u + step * du
            F_new = self.residual_func(u_new)
            norm_F_new = np.linalg.norm(F_new)

            # Armijo condition: sufficient decrease
            if norm_F_new <= (1 - self.line_search_alpha * step) * norm_F0:
                return step, F_new

            step *= 0.5

        # Failed to find good step, use current best
        return step, F_new

    def solve(self, u0: np.ndarray) -> NewtonResult:
        """
        Solve F(u) = 0 using Newton-Raphson iteration.

        Parameters
        ----------
        u0 : np.ndarray
            Initial guess

        Returns
        -------
        NewtonResult
            Solution and convergence information
        """
        u = np.array(u0, dtype=np.float64).copy()
        residual_history = []

        F = self.residual_func(u)
        residual_norm = np.linalg.norm(F)
        residual_history.append(residual_norm)

        if self.verbose:
            print(f"Newton iteration 0: ||F|| = {residual_norm:.3e}")

        for iteration in range(1, self.max_iterations + 1):
            # Check residual convergence
            if residual_norm < self.tol_residual:
                if self.criterion in (
                    ConvergenceCriterion.RESIDUAL,
                    ConvergenceCriterion.BOTH,
                ):
                    return NewtonResult(
                        solution=u,
                        converged=True,
                        iterations=iteration - 1,
                        residual_norm=residual_norm,
                        update_norm=0.0,
                        residual_history=residual_history,
                    )

            # Compute Jacobian and solve J*du = -F
            J = self._compute_jacobian(u)

            # Solve linear system
            if sparse.issparse(J):
                du = spsolve(J, -F)
            else:
                try:
                    du = np.linalg.solve(J, -F)
                except np.linalg.LinAlgError:
                    # Singular Jacobian - try pseudoinverse
                    du = np.linalg.lstsq(J, -F, rcond=None)[0]

            # Line search or direct update
            if self.use_line_search:
                step, F = self._line_search(u, du, F)
                u = u + step * du
            else:
                u = u + self.damping * du
                F = self.residual_func(u)

            update_norm = np.linalg.norm(du)
            residual_norm = np.linalg.norm(F)
            residual_history.append(residual_norm)

            if self.verbose:
                print(
                    f"Newton iteration {iteration}: ||F|| = {residual_norm:.3e}, ||du|| = {update_norm:.3e}"
                )

            # Check convergence
            converged_residual = residual_norm < self.tol_residual
            converged_update = update_norm < self.tol_update

            if self.criterion == ConvergenceCriterion.RESIDUAL and converged_residual:
                return NewtonResult(
                    solution=u,
                    converged=True,
                    iterations=iteration,
                    residual_norm=residual_norm,
                    update_norm=update_norm,
                    residual_history=residual_history,
                )
            elif self.criterion == ConvergenceCriterion.UPDATE and converged_update:
                return NewtonResult(
                    solution=u,
                    converged=True,
                    iterations=iteration,
                    residual_norm=residual_norm,
                    update_norm=update_norm,
                    residual_history=residual_history,
                )
            elif (
                self.criterion == ConvergenceCriterion.BOTH
                and converged_residual
                and converged_update
            ):
                return NewtonResult(
                    solution=u,
                    converged=True,
                    iterations=iteration,
                    residual_norm=residual_norm,
                    update_norm=update_norm,
                    residual_history=residual_history,
                )

        # Did not converge
        return NewtonResult(
            solution=u,
            converged=False,
            iterations=self.max_iterations,
            residual_norm=residual_norm,
            update_norm=update_norm,
            residual_history=residual_history,
        )


class NonlinearDiffusionSolver:
    """
    Newton-Raphson solver for nonlinear reaction-diffusion:

        -D ∇²u + R(u) = S

    where R(u) is a nonlinear reaction term.

    Parameters
    ----------
    mesh : StructuredMesh
        Computational mesh
    D : float or np.ndarray
        Diffusion coefficient (scalar or field)

    Example
    -------
    >>> mesh = bt.mesh_1d(50, 0, 1)
    >>> solver = NonlinearDiffusionSolver(mesh, D=1.0)
    >>> solver.set_reaction(lambda u: u**2)  # R(u) = u²
    >>> solver.set_boundary(Boundary.Left, 0.0)
    >>> solver.set_boundary(Boundary.Right, 1.0)
    >>> result = solver.solve(initial_guess)
    """

    def __init__(self, mesh: StructuredMesh, D: Union[float, np.ndarray] = 1.0):
        self.mesh = mesh
        self.D = D
        self.is_1d = mesh.is_1d()

        if self.is_1d:
            self.nx = mesh.nx()
            self.n = self.nx + 1
            self.dx = mesh.dx()
        else:
            self.nx = mesh.nx()
            self.ny = mesh.ny()
            self.n = (self.nx + 1) * (self.ny + 1)
            self.dx = mesh.dx()
            self.dy = mesh.dy()

        # Default: no reaction, no source
        self.reaction_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.reaction_deriv: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.source: Optional[np.ndarray] = None

        # Boundary conditions: dict of {Boundary: (type, value)}
        self._bcs: dict[Boundary, tuple[str, float]] = {}

        # Newton parameters
        self.max_iterations = 50
        self.tol = 1e-10
        self.verbose = False

    def set_reaction(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        derivative: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> "NonlinearDiffusionSolver":
        """
        Set the nonlinear reaction term R(u).

        Parameters
        ----------
        func : callable
            Reaction function R(u), applied pointwise
        derivative : callable, optional
            Derivative R'(u) for analytical Jacobian
            If None, finite differences are used

        Returns
        -------
        self for chaining
        """
        self.reaction_func = func
        self.reaction_deriv = derivative
        return self

    def set_source(self, source: np.ndarray) -> "NonlinearDiffusionSolver":
        """Set source term S."""
        self.source = np.array(source, dtype=np.float64)
        return self

    def set_boundary(
        self, boundary: Boundary, value: float, bc_type: str = "dirichlet"
    ) -> "NonlinearDiffusionSolver":
        """
        Set boundary condition.

        Parameters
        ----------
        boundary : Boundary
            Which boundary (Left, Right, Bottom, Top)
        value : float
            Boundary value
        bc_type : str
            'dirichlet' (fixed value) or 'neumann' (fixed flux)
        """
        self._bcs[boundary] = (bc_type.lower(), value)
        return self

    def set_parameters(
        self,
        max_iterations: Optional[int] = None,
        tol: Optional[float] = None,
        verbose: Optional[bool] = None,
    ) -> "NonlinearDiffusionSolver":
        """Set Newton iteration parameters."""
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if tol is not None:
            self.tol = tol
        if verbose is not None:
            self.verbose = verbose
        return self

    def _apply_bcs_1d(self, u: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """Apply boundary conditions to residual (1D)."""
        # Left boundary
        if Boundary.Left in self._bcs:
            bc_type, value = self._bcs[Boundary.Left]
            if bc_type == "dirichlet":
                residual[0] = u[0] - value
            else:  # Neumann
                residual[0] = (u[1] - u[0]) / self.dx - value

        # Right boundary
        if Boundary.Right in self._bcs:
            bc_type, value = self._bcs[Boundary.Right]
            if bc_type == "dirichlet":
                residual[-1] = u[-1] - value
            else:  # Neumann
                residual[-1] = (u[-1] - u[-2]) / self.dx - value

        return residual

    def _apply_bcs_2d(self, u: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """Apply boundary conditions to residual (2D)."""
        # Left boundary (x=0)
        if Boundary.Left in self._bcs:
            bc_type, value = self._bcs[Boundary.Left]
            if bc_type == "dirichlet":
                residual[:, 0] = u[:, 0] - value

        # Right boundary (x=L)
        if Boundary.Right in self._bcs:
            bc_type, value = self._bcs[Boundary.Right]
            if bc_type == "dirichlet":
                residual[:, -1] = u[:, -1] - value

        # Bottom boundary (y=0)
        if Boundary.Bottom in self._bcs:
            bc_type, value = self._bcs[Boundary.Bottom]
            if bc_type == "dirichlet":
                residual[0, :] = u[0, :] - value

        # Top boundary (y=H)
        if Boundary.Top in self._bcs:
            bc_type, value = self._bcs[Boundary.Top]
            if bc_type == "dirichlet":
                residual[-1, :] = u[-1, :] - value

        return residual

    def _residual_1d(self, u: np.ndarray) -> np.ndarray:
        """Compute residual for 1D problem: -D*d²u/dx² + R(u) - S = 0."""
        residual = np.zeros_like(u)
        D = self.D if np.isscalar(self.D) else self.D
        inv_dx2 = 1.0 / (self.dx * self.dx)

        # Interior: -D * (u[i+1] - 2*u[i] + u[i-1])/dx² + R(u) - S = 0
        for i in range(1, len(u) - 1):
            Di = D if np.isscalar(D) else D[i]
            laplacian = (u[i + 1] - 2 * u[i] + u[i - 1]) * inv_dx2
            residual[i] = -Di * laplacian

        # Add reaction term
        if self.reaction_func is not None:
            residual += self.reaction_func(u)

        # Subtract source
        if self.source is not None:
            residual -= self.source

        # Apply boundary conditions
        residual = self._apply_bcs_1d(u, residual)

        return residual

    def _residual_2d(self, u_flat: np.ndarray) -> np.ndarray:
        """Compute residual for 2D problem."""
        ny, nx = self.ny + 1, self.nx + 1
        u = u_flat.reshape(ny, nx)
        residual = np.zeros_like(u)

        D = self.D
        inv_dx2 = 1.0 / (self.dx * self.dx)
        inv_dy2 = 1.0 / (self.dy * self.dy)

        # Interior points
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                Di = D if np.isscalar(D) else D[j, i]
                laplacian = (u[j, i + 1] - 2 * u[j, i] + u[j, i - 1]) * inv_dx2 + (
                    u[j + 1, i] - 2 * u[j, i] + u[j - 1, i]
                ) * inv_dy2
                residual[j, i] = -Di * laplacian

        # Add reaction term
        if self.reaction_func is not None:
            residual += self.reaction_func(u)

        # Subtract source
        if self.source is not None:
            source = self.source.reshape(ny, nx)
            residual -= source

        # Apply boundary conditions
        residual = self._apply_bcs_2d(u, residual)

        return residual.flatten()

    def _jacobian_1d(self, u: np.ndarray) -> np.ndarray:
        """Compute analytical Jacobian for 1D problem."""
        n = len(u)
        J = np.zeros((n, n))
        D = self.D
        inv_dx2 = 1.0 / (self.dx * self.dx)

        # Interior points: d/du_i of (-D * laplacian + R)
        for i in range(1, n - 1):
            Di = D if np.isscalar(D) else D[i]
            # d/du_{i-1}: -D * (1/dx²)
            J[i, i - 1] = -Di * inv_dx2
            # d/du_i: -D * (-2/dx²) + R'(u_i)
            J[i, i] = 2 * Di * inv_dx2
            # d/du_{i+1}: -D * (1/dx²)
            J[i, i + 1] = -Di * inv_dx2

        # Add reaction derivative
        if self.reaction_deriv is not None:
            dR = self.reaction_deriv(u)
            for i in range(1, n - 1):
                J[i, i] += dR[i]

        # Boundary conditions
        if Boundary.Left in self._bcs:
            bc_type, _ = self._bcs[Boundary.Left]
            if bc_type == "dirichlet":
                J[0, :] = 0
                J[0, 0] = 1
            else:  # Neumann
                J[0, :] = 0
                J[0, 0] = -1 / self.dx
                J[0, 1] = 1 / self.dx

        if Boundary.Right in self._bcs:
            bc_type, _ = self._bcs[Boundary.Right]
            if bc_type == "dirichlet":
                J[-1, :] = 0
                J[-1, -1] = 1
            else:  # Neumann
                J[-1, :] = 0
                J[-1, -1] = 1 / self.dx
                J[-1, -2] = -1 / self.dx

        return J

    def solve(self, initial_guess: Optional[np.ndarray] = None) -> NewtonResult:
        """
        Solve the nonlinear problem using Newton-Raphson iteration.

        Parameters
        ----------
        initial_guess : np.ndarray, optional
            Initial guess for the solution. If None, uses zeros.

        Returns
        -------
        NewtonResult
            Solution and convergence information
        """
        if initial_guess is None:
            initial_guess = np.zeros(self.n)

        # Set up residual and Jacobian functions
        if self.is_1d:
            residual_func = self._residual_1d
            jacobian_func = self._jacobian_1d if self.reaction_deriv else None
        else:
            residual_func = self._residual_2d
            jacobian_func = None  # Use FD for 2D

        # Create Newton solver
        solver = NewtonRaphsonSolver(
            residual_func=residual_func,
            jacobian_func=jacobian_func,
            n=self.n,
        )
        solver.set_parameters(
            max_iterations=self.max_iterations,
            tol_residual=self.tol,
            tol_update=self.tol,
            verbose=self.verbose,
        )

        # Solve
        result = solver.solve(initial_guess.flatten())

        # Reshape for 2D
        if not self.is_1d:
            result.solution = result.solution.reshape(self.ny + 1, self.nx + 1)

        return result


# Common reaction terms for convenience
def michaelis_menten(vmax: float, km: float) -> tuple[Callable, Callable]:
    """
    Michaelis-Menten kinetics: R(u) = Vmax * u / (Km + u)

    Returns (reaction_func, derivative_func)
    """

    def reaction(u):
        return vmax * u / (km + u)

    def derivative(u):
        return vmax * km / (km + u) ** 2

    return reaction, derivative


def hill_kinetics(vmax: float, km: float, n: float) -> tuple[Callable, Callable]:
    """
    Hill kinetics: R(u) = Vmax * u^n / (Km^n + u^n)

    Returns (reaction_func, derivative_func)
    """

    def reaction(u):
        u_n = np.abs(u) ** n
        return vmax * u_n / (km**n + u_n)

    def derivative(u):
        u_n = np.abs(u) ** n
        return vmax * n * (np.abs(u) ** (n - 1)) * km**n / (km**n + u_n) ** 2

    return reaction, derivative


def bistable(a: float = 0.0) -> tuple[Callable, Callable]:
    """
    Bistable reaction: R(u) = u * (1 - u) * (u - a)

    Has stable fixed points at u=0 and u=1, unstable at u=a.

    Returns (reaction_func, derivative_func)
    """

    def reaction(u):
        return u * (1 - u) * (u - a)

    def derivative(u):
        return (1 - u) * (u - a) + u * (-(u - a)) + u * (1 - u)

    return reaction, derivative


def exponential_decay(k: float) -> tuple[Callable, Callable]:
    """
    First-order decay: R(u) = k * u

    Returns (reaction_func, derivative_func)
    """

    def reaction(u):
        return k * u

    def derivative(u):
        return np.full_like(u, k)

    return reaction, derivative
