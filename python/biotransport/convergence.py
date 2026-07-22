"""Numerical-convergence study utilities.

The module provides Richardson extrapolation, observed-order estimates, and
Grid Convergence Index (GCI)-style calculations for a user-selected scalar
quantity of interest (QoI).  These calculations are evidence about the tested
QoI and refinement sequence; they do not validate a physical model or verify a
library as a whole.

The terminology is consistent with common numerical-verification practice,
including concepts discussed by ASME V&V 20.  This convenience implementation
is not an ASME assessment and does not establish conformance with that standard.

Example:
    >>> study = bt.GridConvergenceStudy()
    >>> study.add_solution(h=0.1, value=u_coarse, error=0.05)
    >>> study.add_solution(h=0.05, value=u_medium, error=0.02)
    >>> study.add_solution(h=0.025, value=u_fine, error=0.008)
    >>> result = study.analyze()
    >>> print(f"Observed order: {result.observed_order:.2f}")
    >>> print(f"Richardson extrapolation: {result.richardson_estimate:.6f}")
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class ConvergenceResult:
    """Results from a grid convergence study.

    Attributes:
        observed_order: Order estimated from the three finest QoI values.
        theoretical_order: User-supplied comparison order; it is not inferred.
        richardson_estimate: QoI extrapolated to zero mesh or timestep size.
        gci_fine: GCI-style index for the finest-grid QoI pair.
        gci_coarse: GCI-style index for the next-coarser QoI pair.
        asymptotic_ratio: GCI ratio; values near one are consistent with the
            assumed asymptotic model for this three-level sequence.
        mesh_sizes: Mesh or timestep sizes, ordered coarse to fine.
        errors: User-supplied errors, if provided for every level.
        solutions: Scalar QoI values, ordered coarse to fine.
        is_asymptotic: Whether ``asymptotic_ratio`` falls in the module's fixed
            diagnostic window, ``[0.95, 1.05]``. This is not a proof that all
            discretization errors are asymptotic.
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

    The returned GCI-style diagnostics are intended for exploratory numerical
    verification. They are not a complete ASME V&V 20 procedure and do not
    represent ASME compliance.

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
    # Common three-grid GCI safety factor. Its default is not a compliance claim.
    safety_factor: float = 1.25

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
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError("h must be finite and greater than zero")
        if not np.isfinite(value):
            raise ValueError("value must be finite")
        if error is not None and (not np.isfinite(error) or error < 0.0):
            raise ValueError("error must be finite and nonnegative")

        self._mesh_sizes.append(float(h))
        self._values.append(float(value))
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
        """Analyze the three finest scalar QoI values.

        Returns:
            Scoped diagnostics for the supplied QoI and refinement sequence.

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

        if len(np.unique(h)) != len(h):
            raise ValueError("Mesh sizes must be unique for convergence analysis")

        # Refinement ratios
        r21 = h1 / h2
        r32 = h2 / h3

        # Estimate observed order using fixed-point iteration
        # p = ln((f1 - f2) / (f2 - f3)) / ln(r)  (for constant r)
        eps32 = f3 - f2
        eps21 = f2 - f1

        resolution = 100.0 * np.finfo(float).eps * max(1.0, abs(f1), abs(f2), abs(f3))
        if abs(eps32) <= resolution or abs(eps21) <= resolution:
            raise ValueError(
                "Observed order is indeterminate because successive solutions "
                "are identical within floating-point resolution. Verify that the "
                "refinement parameter actually reaches the solver."
            )

        # Check for oscillatory convergence
        s = np.sign(eps32 / eps21)

        if s > 0:
            # Monotonic convergence - use fixed-point iteration for p
            observed_order = self._compute_order_fixed_point(eps21, eps32, r21, r32)
        else:
            # Oscillatory convergence - use absolute values
            observed_order = abs(np.log(abs(eps32 / eps21)) / np.log(r32))

        if not np.isfinite(observed_order) or observed_order <= 0.0:
            raise ValueError(
                f"Observed order is not physically interpretable: {observed_order!r}"
            )

        extrapolation_denominator = r32**observed_order - 1.0
        if abs(extrapolation_denominator) <= np.finfo(float).eps:
            raise ValueError("Richardson extrapolation denominator is numerically zero")

        # Richardson extrapolation: f_exact ≈ f3 + (f3 - f2) / (r32^p - 1)
        richardson_estimate = f3 + eps32 / extrapolation_denominator

        # Grid Convergence Index (GCI)
        # GCI = Fs * |eps| / (r^p - 1)
        e_a_fine = abs((f3 - f2) / f3) if abs(f3) > 1e-15 else abs(f3 - f2)
        e_a_coarse = abs((f2 - f1) / f2) if abs(f2) > 1e-15 else abs(f2 - f1)

        gci_fine = self.safety_factor * e_a_fine / (r32**observed_order - 1)
        gci_coarse = self.safety_factor * e_a_coarse / (r21**observed_order - 1)

        # A ratio near one is consistent with the assumed asymptotic model.
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
        print("\nRichardson/GCI-style diagnostics for the scalar QoI:")
        print(f"  Observed order: {result.observed_order:.3f}")
        print(f"  User-supplied comparison order: {result.theoretical_order:.1f}")
        print(f"  Richardson estimate: {result.richardson_estimate:.8f}")
        print(
            f"  GCI-style index (fine): {result.gci_fine:.3e} "
            f"({result.gci_fine * 100:.3e}%)"
        )
        print(
            f"  GCI-style index (coarse): {result.gci_coarse:.3e} "
            f"({result.gci_coarse * 100:.3e}%)"
        )
        print(f"  Asymptotic ratio: {result.asymptotic_ratio:.3f}")
        print(
            "  Asymptotic-ratio window [0.95, 1.05]: "
            f"{'met' if result.is_asymptotic else 'not met'}"
        )

        order_deviation = abs(result.observed_order - result.theoretical_order)
        if order_deviation < 0.3:
            print(
                "\n[PASS] This QoI sequence met the configured observed-order "
                "criterion: |p_observed - p_expected| < 0.3."
            )
        else:
            print(
                "\n[FAIL] This QoI sequence did not meet the configured "
                "observed-order criterion: "
                f"deviation={order_deviation:.3f}, required < 0.3."
            )

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
        print("\nTemporal Richardson/GCI-style diagnostics for the scalar QoI:")
        print(f"  Observed order: {result.observed_order:.3f}")
        print(f"  Comparison order: {result.theoretical_order:.1f}")
        print(f"  Richardson estimate: {result.richardson_estimate:.8f}")
        print(
            f"  GCI-style index (fine): {result.gci_fine:.3e} "
            f"({result.gci_fine * 100:.3e}%)"
        )
        order_deviation = abs(result.observed_order - result.theoretical_order)
        print(
            "  Observed-order criterion "
            f"(|p_observed - p_expected| < 0.3): "
            f"{'PASS' if order_deviation < 0.3 else 'FAIL'} "
            f"(deviation={order_deviation:.3f})"
        )

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
            h_line,
            e_theoretical,
            "k--",
            alpha=0.5,
            label=f"O(h^{result.theoretical_order:.0f}) theoretical",
        )
        ax.loglog(
            h_line,
            e_observed,
            "g:",
            alpha=0.7,
            linewidth=2,
            label=f"O(h^{result.observed_order:.2f}) observed",
        )

    # Show Richardson estimate
    if show_richardson:
        ax.axhline(
            result.richardson_estimate,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Richardson: {result.richardson_estimate:.6f}",
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Value / Error", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    return ax
