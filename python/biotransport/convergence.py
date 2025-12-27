"""
Grid Convergence Study Utilities

Provides tools for rigorous verification of numerical solutions via:
- Grid convergence studies (Richardson extrapolation)
- Order of accuracy estimation
- Grid Convergence Index (GCI) for uncertainty quantification
- Automated mesh refinement studies

Based on the ASME V&V 20-2009 standard for verification and validation.

Example:
    >>> study = bt.GridConvergenceStudy()
    >>> study.add_solution(h=0.1, solution=u_coarse, error=0.05)
    >>> study.add_solution(h=0.05, solution=u_medium, error=0.02)
    >>> study.add_solution(h=0.025, solution=u_fine, error=0.008)
    >>> result = study.analyze()
    >>> print(f"Observed order: {result.observed_order:.2f}")
    >>> print(f"Richardson extrapolation: {result.richardson_estimate:.6f}")
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np


@dataclass
class ConvergenceResult:
    """Results from a grid convergence study.

    Attributes:
        observed_order: Estimated order of accuracy from Richardson extrapolation
        theoretical_order: Expected order (e.g., 2 for central differences)
        richardson_estimate: Extrapolated value at h→0
        gci_fine: Grid Convergence Index for finest grid (uncertainty estimate)
        gci_coarse: Grid Convergence Index for coarser grid
        asymptotic_ratio: Ratio indicating if asymptotic range is achieved (~1.0)
        mesh_sizes: Array of mesh sizes used
        errors: Array of errors (if analytical solution available)
        solutions: List of solution values at a point or norm
        is_asymptotic: Whether solution is in asymptotic convergence range
    """

    observed_order: float
    theoretical_order: float
    richardson_estimate: float
    gci_fine: float
    gci_coarse: float
    asymptotic_ratio: float
    mesh_sizes: np.ndarray
    errors: Optional[np.ndarray] = None
    solutions: Optional[np.ndarray] = None
    is_asymptotic: bool = False


@dataclass
class GridConvergenceStudy:
    """Performs grid convergence analysis using Richardson extrapolation.

    This class implements the Grid Convergence Index (GCI) method as
    recommended by the ASME V&V 20-2009 standard for CFD verification.

    The method requires solutions on at least 3 systematically refined grids.
    The refinement ratio r = h_coarse / h_fine should be constant (typically 2).

    Example:
        >>> study = GridConvergenceStudy(theoretical_order=2)
        >>> # Add solutions from coarse to fine
        >>> study.add_solution(h=0.04, value=1.234)
        >>> study.add_solution(h=0.02, value=1.256)
        >>> study.add_solution(h=0.01, value=1.261)
        >>> result = study.analyze()
        >>> print(f"Extrapolated: {result.richardson_estimate:.4f}")
    """

    theoretical_order: float = 2.0
    safety_factor: float = 1.25  # Fs = 1.25 for 3+ grids, 3.0 for 2 grids

    # Internal storage
    _mesh_sizes: List[float] = field(default_factory=list)
    _values: List[float] = field(default_factory=list)  # Solution value or norm
    _errors: List[float] = field(default_factory=list)  # Error vs analytical (optional)

    def __post_init__(self):
        self._mesh_sizes = []
        self._values = []
        self._errors = []

    def add_solution(
        self,
        h: float,
        value: float,
        error: Optional[float] = None,
    ) -> "GridConvergenceStudy":
        """Add a solution at a given mesh size.

        Args:
            h: Characteristic mesh size (dx, or 1/N, etc.)
            value: Solution value (can be a point value, norm, or QoI)
            error: Optional error vs analytical solution

        Returns:
            Self for method chaining
        """
        self._mesh_sizes.append(h)
        self._values.append(value)
        if error is not None:
            self._errors.append(error)
        return self

    def clear(self) -> "GridConvergenceStudy":
        """Clear all stored solutions."""
        self._mesh_sizes.clear()
        self._values.clear()
        self._errors.clear()
        return self

    def analyze(self) -> ConvergenceResult:
        """Perform Richardson extrapolation analysis.

        Returns:
            ConvergenceResult with observed order, Richardson estimate, and GCI

        Raises:
            ValueError: If fewer than 3 solutions are available
        """
        if len(self._mesh_sizes) < 3:
            raise ValueError(
                f"Need at least 3 grid levels, got {len(self._mesh_sizes)}"
            )

        # Sort by mesh size (coarsest to finest)
        idx = np.argsort(self._mesh_sizes)[::-1]
        h = np.array(self._mesh_sizes)[idx]
        f = np.array(self._values)[idx]

        # Use the three finest grids
        h1, h2, h3 = h[-3], h[-2], h[-1]  # h1 > h2 > h3 (coarse to fine)
        f1, f2, f3 = f[-3], f[-2], f[-1]

        # Refinement ratios
        r21 = h1 / h2
        r32 = h2 / h3

        # Estimate observed order using fixed-point iteration
        # p = ln((f1 - f2) / (f2 - f3)) / ln(r)  (for constant r)
        eps32 = f3 - f2
        eps21 = f2 - f1

        if abs(eps32) < 1e-15 or abs(eps21) < 1e-15:
            # Solutions are converged or oscillating
            observed_order = self.theoretical_order
            richardson_estimate = f3
        else:
            # Check for oscillatory convergence
            s = np.sign(eps32 / eps21)

            if s > 0:
                # Monotonic convergence - use fixed-point iteration for p
                observed_order = self._compute_order_fixed_point(
                    eps21, eps32, r21, r32
                )
            else:
                # Oscillatory convergence - use absolute values
                observed_order = abs(
                    np.log(abs(eps32 / eps21)) / np.log(r32)
                )

            # Richardson extrapolation: f_exact ≈ f3 + (f3 - f2) / (r32^p - 1)
            richardson_estimate = f3 + eps32 / (r32**observed_order - 1)

        # Grid Convergence Index (GCI)
        # GCI = Fs * |eps| / (r^p - 1)
        e_a_fine = abs((f3 - f2) / f3) if abs(f3) > 1e-15 else abs(f3 - f2)
        e_a_coarse = abs((f2 - f1) / f2) if abs(f2) > 1e-15 else abs(f2 - f1)

        gci_fine = self.safety_factor * e_a_fine / (r32**observed_order - 1)
        gci_coarse = self.safety_factor * e_a_coarse / (r21**observed_order - 1)

        # Asymptotic ratio (should be ≈ 1 if in asymptotic range)
        asymptotic_ratio = gci_coarse / (r21**observed_order * gci_fine)
        is_asymptotic = 0.95 <= asymptotic_ratio <= 1.05

        errors = np.array(self._errors)[idx] if self._errors else None

        return ConvergenceResult(
            observed_order=observed_order,
            theoretical_order=self.theoretical_order,
            richardson_estimate=richardson_estimate,
            gci_fine=gci_fine,
            gci_coarse=gci_coarse,
            asymptotic_ratio=asymptotic_ratio,
            mesh_sizes=h,
            errors=errors,
            solutions=f,
            is_asymptotic=is_asymptotic,
        )

    def _compute_order_fixed_point(
        self,
        eps21: float,
        eps32: float,
        r21: float,
        r32: float,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> float:
        """Compute observed order using fixed-point iteration.

        For non-constant refinement ratios, we solve:
            p = ln[(r21^p - s) / (r32^p - s)] / ln(r21)
        where s = sign(eps32/eps21).
        """
        # Initial guess from constant-r formula
        p = abs(np.log(abs(eps32 / eps21))) / np.log(r21)
        s = np.sign(eps32 / eps21)

        for _ in range(max_iter):
            p_new = abs(
                np.log(abs((r21**p - s) / (r32**p - s)) * abs(eps32 / eps21))
            ) / np.log(r21)

            if abs(p_new - p) < tol:
                return p_new
            p = p_new

        return p


def compute_order_of_accuracy(
    mesh_sizes: np.ndarray,
    errors: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute order of accuracy from error data.

    Uses least-squares fit to: log(error) = p * log(h) + log(C)

    Args:
        mesh_sizes: Array of characteristic mesh sizes
        errors: Array of corresponding errors

    Returns:
        Tuple of (order, coefficient, r_squared)
    """
    log_h = np.log(mesh_sizes)
    log_e = np.log(errors)

    # Linear fit
    coeffs = np.polyfit(log_h, log_e, 1)
    order = coeffs[0]
    C = np.exp(coeffs[1])

    # R-squared
    log_e_fit = np.polyval(coeffs, log_h)
    ss_res = np.sum((log_e - log_e_fit) ** 2)
    ss_tot = np.sum((log_e - np.mean(log_e)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return order, C, r_squared


def run_convergence_study(
    solve_func: Callable[[int], Tuple[float, float]],
    n_values: List[int],
    theoretical_order: float = 2.0,
    verbose: bool = True,
) -> ConvergenceResult:
    """Run an automated grid convergence study.

    Args:
        solve_func: Function that takes N (grid points) and returns (value, error)
                   where value is the QoI and error is vs analytical (or None)
        n_values: List of grid resolutions to test
        theoretical_order: Expected order of accuracy
        verbose: Whether to print progress

    Returns:
        ConvergenceResult from the analysis

    Example:
        >>> def solve(n):
        ...     mesh = bt.mesh_1d(n)
        ...     result = solver.run(problem, t_end)
        ...     error = np.max(np.abs(result.solution() - analytical))
        ...     return result.solution()[n//2], error  # midpoint value
        >>> result = run_convergence_study(solve, [25, 50, 100, 200])
    """
    study = GridConvergenceStudy(theoretical_order=theoretical_order)

    if verbose:
        print("=" * 60)
        print("Grid Convergence Study")
        print("=" * 60)
        print(f"{'N':>8} {'h':>12} {'Value':>16} {'Error':>14}")
        print("-" * 60)

    for n in sorted(n_values):
        h = 1.0 / n  # Characteristic mesh size
        result = solve_func(n)

        if isinstance(result, tuple):
            value, error = result
        else:
            value, error = result, None

        study.add_solution(h=h, value=value, error=error)

        if verbose:
            if error is not None:
                print(f"{n:>8} {h:>12.6f} {value:>16.8f} {error:>14.2e}")
            else:
                print(f"{n:>8} {h:>12.6f} {value:>16.8f} {'N/A':>14}")

    result = study.analyze()

    if verbose:
        print("-" * 60)
        print(f"\nRichardson Extrapolation Results:")
        print(f"  Observed order: {result.observed_order:.3f}")
        print(f"  Theoretical order: {result.theoretical_order:.1f}")
        print(f"  Richardson estimate: {result.richardson_estimate:.8f}")
        print(f"  GCI (fine): {result.gci_fine * 100:.2f}%")
        print(f"  GCI (coarse): {result.gci_coarse * 100:.2f}%")
        print(f"  Asymptotic ratio: {result.asymptotic_ratio:.3f}")
        print(f"  In asymptotic range: {'Yes' if result.is_asymptotic else 'No'}")

        if abs(result.observed_order - result.theoretical_order) < 0.3:
            print(f"\n✓ Order of accuracy VERIFIED (within 0.3 of theoretical)")
        else:
            print(f"\n⚠ Order deviation: {abs(result.observed_order - result.theoretical_order):.2f}")

    return result


def temporal_convergence_study(
    solve_func: Callable[[float], Tuple[float, float]],
    dt_values: List[float],
    theoretical_order: float = 1.0,
    verbose: bool = True,
) -> ConvergenceResult:
    """Run a temporal convergence study.

    Args:
        solve_func: Function that takes dt and returns (value, error)
        dt_values: List of time step sizes to test
        theoretical_order: Expected temporal order (1 for explicit, 2 for CN)
        verbose: Whether to print progress

    Returns:
        ConvergenceResult from the analysis
    """
    study = GridConvergenceStudy(theoretical_order=theoretical_order)

    if verbose:
        print("=" * 60)
        print("Temporal Convergence Study")
        print("=" * 60)
        print(f"{'dt':>12} {'Value':>16} {'Error':>14}")
        print("-" * 60)

    for dt in sorted(dt_values, reverse=True):  # Coarse to fine
        result = solve_func(dt)

        if isinstance(result, tuple):
            value, error = result
        else:
            value, error = result, None

        study.add_solution(h=dt, value=value, error=error)

        if verbose:
            if error is not None:
                print(f"{dt:>12.6f} {value:>16.8f} {error:>14.2e}")
            else:
                print(f"{dt:>12.6f} {value:>16.8f} {'N/A':>14}")

    result = study.analyze()

    if verbose:
        print("-" * 60)
        print(f"\nTemporal Richardson Extrapolation:")
        print(f"  Observed order: {result.observed_order:.3f}")
        print(f"  Richardson estimate: {result.richardson_estimate:.8f}")
        print(f"  GCI (fine): {result.gci_fine * 100:.2f}%")

    return result


def plot_convergence(
    result: ConvergenceResult,
    title: str = "Grid Convergence Study",
    xlabel: str = "Mesh size h",
    ax=None,
    show_richardson: bool = True,
    show_gci: bool = True,
):
    """Plot convergence study results.

    Args:
        result: ConvergenceResult from analyze()
        title: Plot title
        xlabel: X-axis label
        ax: Matplotlib axes (creates new figure if None)
        show_richardson: Show Richardson extrapolation estimate
        show_gci: Show GCI error bars
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    h = result.mesh_sizes
    f = result.solutions

    # Plot solutions
    ax.loglog(h, f, "bo-", markersize=10, linewidth=2, label="Computed")

    # Reference line with observed order
    h_ref = h[-1]
    f_ref = f[-1]
    h_line = np.array([h.min() / 2, h.max() * 2])
    f_line = f_ref + (result.richardson_estimate - f_ref) * (1 - (h_line / h_ref) ** result.observed_order)
    # This doesn't work well for log plots, use error-based approach instead

    if result.errors is not None:
        # Plot errors with order reference
        ax.loglog(h, result.errors, "rs-", markersize=10, linewidth=2, label="Error")

        # Reference lines
        h_ref = h[len(h) // 2]
        e_ref = result.errors[len(h) // 2]

        h_line = np.logspace(np.log10(h.min() / 1.5), np.log10(h.max() * 1.5), 50)
        e_theoretical = e_ref * (h_line / h_ref) ** result.theoretical_order
        e_observed = e_ref * (h_line / h_ref) ** result.observed_order

        ax.loglog(
            h_line, e_theoretical, "k--", alpha=0.5,
            label=f"O(h^{result.theoretical_order:.0f}) theoretical"
        )
        ax.loglog(
            h_line, e_observed, "g:", alpha=0.7, linewidth=2,
            label=f"O(h^{result.observed_order:.2f}) observed"
        )

    # Show Richardson estimate
    if show_richardson:
        ax.axhline(
            result.richardson_estimate, color="r", linestyle="--", alpha=0.5,
            label=f"Richardson: {result.richardson_estimate:.6f}"
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Value / Error", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    return ax
