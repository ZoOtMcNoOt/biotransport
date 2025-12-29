"""
Higher-Order Finite Difference Schemes for Improved Spatial Accuracy.

This module provides 4th-order and 6th-order accurate finite difference
operators for spatial derivatives, significantly improving accuracy compared
to standard 2nd-order schemes.

Standard 2nd-order central difference:
    d²u/dx² ≈ (u[i+1] - 2*u[i] + u[i-1]) / dx²
    Truncation error: O(dx²)

4th-order central difference:
    d²u/dx² ≈ (-u[i+2] + 16*u[i+1] - 30*u[i] + 16*u[i-1] - u[i-2]) / (12*dx²)
    Truncation error: O(dx⁴)

6th-order central difference:
    d²u/dx² ≈ (2*u[i+3] - 27*u[i+2] + 270*u[i+1] - 490*u[i]
              + 270*u[i-1] - 27*u[i-2] + 2*u[i-3]) / (180*dx²)
    Truncation error: O(dx⁶)

Example:
    >>> import biotransport as bt
    >>> from biotransport.high_order import HighOrderDiffusionSolver
    >>>
    >>> mesh = bt.mesh_1d(50, 0, 1)
    >>> solver = HighOrderDiffusionSolver(mesh, D=0.01, order=4)
    >>> result = solver.solve(initial, t_end=1.0)
"""

from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import numpy as np

from ._core import StructuredMesh, Boundary


# =============================================================================
# Finite Difference Stencils
# =============================================================================


def laplacian_2nd_order(
    u: np.ndarray, dx: float, dy: Optional[float] = None
) -> np.ndarray:
    """
    Compute Laplacian using 2nd-order central differences.

    ∇²u ≈ (u[i+1] - 2*u[i] + u[i-1]) / dx²

    Args:
        u: Solution field (1D or 2D array)
        dx: Grid spacing in x
        dy: Grid spacing in y (for 2D, defaults to dx)

    Returns:
        Laplacian field (interior values only)
    """
    if u.ndim == 1:
        lap = np.zeros_like(u)
        inv_dx2 = 1.0 / (dx * dx)
        lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) * inv_dx2
        return lap
    else:
        dy = dy or dx
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) * inv_dx2 + (
            u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]
        ) * inv_dy2
        return lap


def laplacian_4th_order(
    u: np.ndarray, dx: float, dy: Optional[float] = None
) -> np.ndarray:
    """
    Compute Laplacian using 4th-order central differences.

    d²u/dx² ≈ (-u[i+2] + 16*u[i+1] - 30*u[i] + 16*u[i-1] - u[i-2]) / (12*dx²)

    Truncation error: O(dx⁴)

    Args:
        u: Solution field (1D or 2D array)
        dx: Grid spacing in x
        dy: Grid spacing in y (for 2D, defaults to dx)

    Returns:
        Laplacian field (requires 2 ghost cells on each side)
    """
    if u.ndim == 1:
        n = len(u)
        lap = np.zeros_like(u)
        inv_12dx2 = 1.0 / (12.0 * dx * dx)

        # 4th-order stencil for interior (needs i-2 to i+2)
        for i in range(2, n - 2):
            lap[i] = (
                -u[i + 2] + 16 * u[i + 1] - 30 * u[i] + 16 * u[i - 1] - u[i - 2]
            ) * inv_12dx2

        # Use 2nd-order at nodes adjacent to boundaries
        inv_dx2 = 1.0 / (dx * dx)
        if n > 2:
            lap[1] = (u[2] - 2 * u[1] + u[0]) * inv_dx2
            lap[n - 2] = (u[n - 1] - 2 * u[n - 2] + u[n - 3]) * inv_dx2

        return lap
    else:
        # 2D case
        dy = dy or dx
        ny, nx = u.shape
        lap = np.zeros_like(u)
        inv_12dx2 = 1.0 / (12.0 * dx * dx)
        inv_12dy2 = 1.0 / (12.0 * dy * dy)
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)

        # 4th-order interior (i,j in [2, n-2])
        for j in range(2, ny - 2):
            for i in range(2, nx - 2):
                lap_x = (
                    -u[j, i + 2]
                    + 16 * u[j, i + 1]
                    - 30 * u[j, i]
                    + 16 * u[j, i - 1]
                    - u[j, i - 2]
                ) * inv_12dx2
                lap_y = (
                    -u[j + 2, i]
                    + 16 * u[j + 1, i]
                    - 30 * u[j, i]
                    + 16 * u[j - 1, i]
                    - u[j - 2, i]
                ) * inv_12dy2
                lap[j, i] = lap_x + lap_y

        # Use 2nd-order in transition zone (1 cell from boundary)
        # Top/bottom rows at j=1 and j=ny-2
        for j in [1, ny - 2]:
            for i in range(1, nx - 1):
                lap[j, i] = (u[j, i + 1] - 2 * u[j, i] + u[j, i - 1]) * inv_dx2 + (
                    u[j + 1, i] - 2 * u[j, i] + u[j - 1, i]
                ) * inv_dy2
        # Left/right columns at i=1 and i=nx-2
        for i in [1, nx - 2]:
            for j in range(2, ny - 2):
                lap[j, i] = (u[j, i + 1] - 2 * u[j, i] + u[j, i - 1]) * inv_dx2 + (
                    u[j + 1, i] - 2 * u[j, i] + u[j - 1, i]
                ) * inv_dy2

        return lap


def laplacian_6th_order(
    u: np.ndarray, dx: float, dy: Optional[float] = None
) -> np.ndarray:
    """
    Compute Laplacian using 6th-order central differences.

    d²u/dx² ≈ (2*u[i+3] - 27*u[i+2] + 270*u[i+1] - 490*u[i]
              + 270*u[i-1] - 27*u[i-2] + 2*u[i-3]) / (180*dx²)

    Truncation error: O(dx⁶)

    Args:
        u: Solution field (1D array)
        dx: Grid spacing

    Returns:
        Laplacian field (requires 3 ghost cells on each side)
    """
    if u.ndim != 1:
        raise NotImplementedError("6th-order 2D Laplacian not yet implemented")

    n = len(u)
    lap = np.zeros_like(u)
    inv_180dx2 = 1.0 / (180.0 * dx * dx)

    # 6th-order stencil for deep interior (needs i-3 to i+3)
    for i in range(3, n - 3):
        lap[i] = (
            2 * u[i + 3]
            - 27 * u[i + 2]
            + 270 * u[i + 1]
            - 490 * u[i]
            + 270 * u[i - 1]
            - 27 * u[i - 2]
            + 2 * u[i - 3]
        ) * inv_180dx2

    # Use 4th-order at i=2 and i=n-3
    inv_12dx2 = 1.0 / (12.0 * dx * dx)
    if n > 4:
        for i in [2, n - 3]:
            if 2 <= i <= n - 3:
                lap[i] = (
                    -u[i + 2] + 16 * u[i + 1] - 30 * u[i] + 16 * u[i - 1] - u[i - 2]
                ) * inv_12dx2

    # Use 2nd-order at i=1 and i=n-2
    inv_dx2 = 1.0 / (dx * dx)
    if n > 2:
        lap[1] = (u[2] - 2 * u[1] + u[0]) * inv_dx2
        lap[n - 2] = (u[n - 1] - 2 * u[n - 2] + u[n - 3]) * inv_dx2

    return lap


def gradient_4th_order(u: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute first derivative using 4th-order central differences.

    du/dx ≈ (-u[i+2] + 8*u[i+1] - 8*u[i-1] + u[i-2]) / (12*dx)

    Truncation error: O(dx⁴)

    Args:
        u: Solution field (1D array)
        dx: Grid spacing

    Returns:
        Gradient field
    """
    n = len(u)
    grad = np.zeros_like(u)
    inv_12dx = 1.0 / (12.0 * dx)

    # 4th-order interior
    for i in range(2, n - 2):
        grad[i] = (-u[i + 2] + 8 * u[i + 1] - 8 * u[i - 1] + u[i - 2]) * inv_12dx

    # 2nd-order at boundaries
    inv_2dx = 1.0 / (2.0 * dx)
    if n > 2:
        grad[1] = (u[2] - u[0]) * inv_2dx
        grad[n - 2] = (u[n - 1] - u[n - 3]) * inv_2dx

    return grad


# =============================================================================
# High-Order Diffusion Solver
# =============================================================================


@dataclass
class HighOrderResult:
    """Result from high-order solver.

    Attributes:
        solution: Final solution field
        time: Final simulation time
        steps: Number of time steps taken
        dt: Time step used
        order: Spatial accuracy order used
    """

    solution: np.ndarray
    time: float
    steps: int
    dt: float
    order: int


class HighOrderDiffusionSolver:
    """
    Diffusion solver with selectable spatial accuracy order.

    Supports 2nd, 4th, and 6th-order accurate spatial discretization
    for improved accuracy on smooth solutions.

    The CFL stability condition is adjusted for higher-order schemes:
    - 2nd-order: dt <= safety * dx² / (2*D)
    - 4th-order: dt <= safety * dx² / (2.5*D)
    - 6th-order: dt <= safety * dx² / (3*D)

    Example:
        >>> solver = HighOrderDiffusionSolver(mesh, D=0.01, order=4)
        >>> result = solver.solve(initial, t_end=1.0)
    """

    def __init__(
        self,
        mesh: StructuredMesh,
        D: float,
        order: int = 4,
        safety_factor: float = 0.4,
    ):
        """
        Initialize high-order diffusion solver.

        Args:
            mesh: Computational mesh
            D: Diffusion coefficient
            order: Spatial accuracy order (2, 4, or 6)
            safety_factor: CFL safety factor (default 0.4)
        """
        if order not in (2, 4, 6):
            raise ValueError("Order must be 2, 4, or 6")
        if D <= 0:
            raise ValueError("Diffusion coefficient must be positive")

        self.mesh = mesh
        self.D = D
        self.order = order
        self.safety_factor = safety_factor

        # Extract mesh info
        self.nx = mesh.nx()
        self.dx = mesh.dx()
        self.is_1d = mesh.is_1d  # Python binding uses snake_case property

        if not self.is_1d:
            self.ny = mesh.ny()
            self.dy = mesh.dy()
        else:
            self.ny = 0
            self.dy = self.dx

        # Default boundary conditions (Dirichlet = 0)
        self.bc_left = 0.0
        self.bc_right = 0.0
        self.bc_bottom = 0.0
        self.bc_top = 0.0

        # Select Laplacian function
        if order == 2:
            self._laplacian = laplacian_2nd_order
        elif order == 4:
            self._laplacian = laplacian_4th_order
        else:
            self._laplacian = laplacian_6th_order

    def set_boundary(
        self, boundary: Boundary, value: float
    ) -> "HighOrderDiffusionSolver":
        """Set Dirichlet boundary condition value."""
        if boundary == Boundary.Left:
            self.bc_left = value
        elif boundary == Boundary.Right:
            self.bc_right = value
        elif boundary == Boundary.Bottom:
            self.bc_bottom = value
        elif boundary == Boundary.Top:
            self.bc_top = value
        return self

    def compute_stable_dt(self) -> float:
        """Compute stable time step for explicit integration."""
        # Higher-order schemes have stricter stability requirements
        stability_factor = {2: 2.0, 4: 2.5, 6: 3.0}[self.order]

        if self.is_1d:
            dt = self.safety_factor * self.dx * self.dx / (stability_factor * self.D)
        else:
            dx2 = self.dx * self.dx
            dy2 = self.dy * self.dy
            dt = self.safety_factor / (
                stability_factor * self.D * (1.0 / dx2 + 1.0 / dy2)
            )

        return dt

    def solve(
        self,
        initial: np.ndarray,
        t_end: float,
        dt: Optional[float] = None,
        callback: Optional[Callable[[float, np.ndarray], None]] = None,
    ) -> HighOrderResult:
        """
        Solve the diffusion equation with high-order spatial accuracy.

        Args:
            initial: Initial condition
            t_end: End time
            dt: Time step (if None, computes stable dt)
            callback: Optional callback(t, u) called each step

        Returns:
            HighOrderResult with solution and statistics
        """
        # Compute stable timestep if not provided
        if dt is None:
            dt = self.compute_stable_dt()

        # Initialize solution
        if self.is_1d:
            u = np.array(initial, dtype=np.float64).copy()
        else:
            u = (
                np.array(initial, dtype=np.float64)
                .reshape(self.ny + 1, self.nx + 1)
                .copy()
            )

        t = 0.0
        step = 0

        while t < t_end:
            # Adjust final step
            if t + dt > t_end:
                dt = t_end - t

            # Compute Laplacian
            if self.is_1d:
                lap = self._laplacian(u, self.dx)
            else:
                lap = self._laplacian(u, self.dx, self.dy)

            # Forward Euler update
            u = u + self.D * dt * lap

            # Apply boundary conditions
            if self.is_1d:
                u[0] = self.bc_left
                u[-1] = self.bc_right
            else:
                u[:, 0] = self.bc_left
                u[:, -1] = self.bc_right
                u[0, :] = self.bc_bottom
                u[-1, :] = self.bc_top

            t += dt
            step += 1

            if callback is not None:
                callback(t, u)

        return HighOrderResult(
            solution=u.flatten() if not self.is_1d else u,
            time=t,
            steps=step,
            dt=dt,
            order=self.order,
        )


# =============================================================================
# Convenience Functions for Computing Derivatives
# =============================================================================


def d2dx2(u: np.ndarray, dx: float, order: int = 4) -> np.ndarray:
    """
    Compute second derivative d²u/dx² with specified accuracy order.

    Args:
        u: Solution field
        dx: Grid spacing
        order: Accuracy order (2, 4, or 6)

    Returns:
        Second derivative field
    """
    if order == 2:
        return laplacian_2nd_order(u, dx)
    elif order == 4:
        return laplacian_4th_order(u, dx)
    elif order == 6:
        return laplacian_6th_order(u, dx)
    else:
        raise ValueError(f"Unsupported order: {order}. Use 2, 4, or 6.")


def ddx(u: np.ndarray, dx: float, order: int = 4) -> np.ndarray:
    """
    Compute first derivative du/dx with specified accuracy order.

    Args:
        u: Solution field
        dx: Grid spacing
        order: Accuracy order (2 or 4)

    Returns:
        First derivative field
    """
    if order == 2:
        grad = np.zeros_like(u)
        grad[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        return grad
    elif order == 4:
        return gradient_4th_order(u, dx)
    else:
        raise ValueError(f"Unsupported order: {order}. Use 2 or 4.")


def verify_order_of_accuracy(
    solver_factory: Callable[[int], Callable[[np.ndarray], np.ndarray]],
    exact_solution: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float] = (0.0, 1.0),
    grid_sizes: Tuple[int, ...] = (20, 40, 80, 160),
) -> dict:
    """
    Verify the order of accuracy of a finite difference scheme.

    Uses Richardson extrapolation to compute the observed order of accuracy.

    Args:
        solver_factory: Function(n) -> solver that returns a solver for n grid points
        exact_solution: Function(x) -> u_exact that returns exact solution
        x_range: (x_min, x_max) domain
        grid_sizes: Tuple of grid sizes to test

    Returns:
        Dict with errors, observed_orders, and grid_sizes

    Example:
        >>> def factory(n):
        ...     x = np.linspace(0, 1, n+1)
        ...     return lambda u: laplacian_4th_order(u, 1.0/n)
        >>> exact = lambda x: np.sin(2*np.pi*x)
        >>> results = verify_order_of_accuracy(factory, exact)
        >>> print(f"Observed order: {results['observed_orders'][-1]:.2f}")
    """
    errors = []
    dxs = []

    for n in grid_sizes:
        x = np.linspace(x_range[0], x_range[1], n + 1)
        dx = (x_range[1] - x_range[0]) / n
        dxs.append(dx)

        # Get exact solution and apply numerical operator
        u_exact = exact_solution(x)
        solver = solver_factory(n)
        u_numerical = solver(u_exact)

        # For Laplacian, compute exact d²u/dx² for comparison
        # Assume exact_solution is smooth enough
        h = 1e-5
        u_plus = exact_solution(x + h)
        u_minus = exact_solution(x - h)
        d2u_exact = (u_plus - 2 * u_exact + u_minus) / (h * h)

        # Compute error in interior (skip boundaries)
        error = np.max(np.abs(u_numerical[2:-2] - d2u_exact[2:-2]))
        errors.append(error)

    # Compute observed orders
    observed_orders = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i - 1] > 0:
            order = np.log(errors[i - 1] / errors[i]) / np.log(dxs[i - 1] / dxs[i])
            observed_orders.append(order)
        else:
            observed_orders.append(float("nan"))

    return {
        "grid_sizes": list(grid_sizes),
        "dx": dxs,
        "errors": errors,
        "observed_orders": observed_orders,
    }
