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
from numbers import Integral, Real
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np


ConvergenceSolveResult = Union[float, Tuple[float, Optional[float]]]


def _finite_real(value: object, name: str) -> float:
    """Return one finite real scalar without silently discarding information."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _real_vector(values: object, name: str) -> np.ndarray:
    """Convert a one-dimensional real array without accepting complex/string data."""
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{name} must not contain masked values")
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a one-dimensional real array") from error
    if np.iscomplexobj(raw) or raw.dtype.kind in ("b", "S", "U"):
        raise TypeError(f"{name} must be a one-dimensional real array")
    if raw.dtype.kind == "O":
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in raw.flat
        ):
            raise TypeError(f"{name} must be a one-dimensional real array")
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a one-dimensional real array") from error
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
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

    def __post_init__(self) -> None:
        observed_order = _finite_real(self.observed_order, "observed_order")
        theoretical_order = _finite_real(self.theoretical_order, "theoretical_order")
        richardson_estimate = _finite_real(
            self.richardson_estimate, "richardson_estimate"
        )
        gci_fine = _finite_real(self.gci_fine, "gci_fine")
        gci_coarse = _finite_real(self.gci_coarse, "gci_coarse")
        asymptotic_ratio = _finite_real(self.asymptotic_ratio, "asymptotic_ratio")
        if observed_order <= 0.0:
            raise ValueError("observed_order must be greater than zero")
        if theoretical_order <= 0.0:
            raise ValueError("theoretical_order must be greater than zero")
        if gci_fine <= 0.0 or gci_coarse <= 0.0:
            raise ValueError("GCI-style indices must be greater than zero")
        if asymptotic_ratio <= 0.0:
            raise ValueError("asymptotic_ratio must be greater than zero")

        mesh_sizes = _real_vector(self.mesh_sizes, "mesh_sizes")
        if mesh_sizes.size < 3:
            raise ValueError("mesh_sizes must contain at least three levels")
        if not np.all(np.isfinite(mesh_sizes)) or np.any(mesh_sizes <= 0.0):
            raise ValueError("mesh_sizes must be finite and greater than zero")
        if mesh_sizes.size > 1 and np.any(np.diff(mesh_sizes) >= 0.0):
            raise ValueError("mesh_sizes must be unique and ordered coarse to fine")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            log_refinement_scale = observed_order * float(
                np.log(mesh_sizes[-2]) - np.log(mesh_sizes[-1])
            )
        log_expected_ratio = (
            float(np.log(gci_coarse)) - float(np.log(gci_fine)) - log_refinement_scale
        )
        log_supplied_ratio = float(np.log(asymptotic_ratio))
        log_scale = max(
            1.0,
            abs(float(np.log(gci_coarse))),
            abs(float(np.log(gci_fine))),
            abs(log_refinement_scale),
            abs(log_expected_ratio),
            abs(log_supplied_ratio),
        )
        ratio_tolerance = 512.0 * np.finfo(float).eps * log_scale
        if (
            not np.isfinite(log_expected_ratio)
            or abs(log_supplied_ratio - log_expected_ratio) > ratio_tolerance
        ):
            raise ValueError(
                "asymptotic_ratio is inconsistent with gci_coarse / "
                "((h2 / h3)**observed_order * gci_fine)"
            )

        errors = None
        if self.errors is not None:
            errors = _real_vector(self.errors, "errors")
            if errors.size != mesh_sizes.size:
                raise ValueError("errors length must match mesh_sizes")
            if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
                raise ValueError("errors must be finite and nonnegative")

        solutions = None
        if self.solutions is not None:
            solutions = _real_vector(self.solutions, "solutions")
            if solutions.size != mesh_sizes.size:
                raise ValueError("solutions length must match mesh_sizes")
            if not np.all(np.isfinite(solutions)):
                raise ValueError("solutions must be finite")

        if not isinstance(self.is_asymptotic, (bool, np.bool_)):
            raise TypeError("is_asymptotic must be a boolean")
        expected_asymptotic = 0.95 <= asymptotic_ratio <= 1.05
        if bool(self.is_asymptotic) != expected_asymptotic:
            raise ValueError(
                "is_asymptotic is inconsistent with the fixed [0.95, 1.05] "
                "asymptotic-ratio window"
            )

        object.__setattr__(self, "observed_order", observed_order)
        object.__setattr__(self, "theoretical_order", theoretical_order)
        object.__setattr__(self, "richardson_estimate", richardson_estimate)
        object.__setattr__(self, "gci_fine", gci_fine)
        object.__setattr__(self, "gci_coarse", gci_coarse)
        object.__setattr__(self, "asymptotic_ratio", asymptotic_ratio)
        object.__setattr__(self, "mesh_sizes", _readonly(mesh_sizes))
        object.__setattr__(
            self, "errors", None if errors is None else _readonly(errors)
        )
        object.__setattr__(
            self, "solutions", None if solutions is None else _readonly(solutions)
        )
        object.__setattr__(self, "is_asymptotic", bool(self.is_asymptotic))


@dataclass
class GridConvergenceStudy:
    """Performs grid convergence analysis using Richardson extrapolation.

    The returned GCI-style diagnostics are intended for exploratory numerical
    verification. They are not a complete ASME V&V 20 procedure and do not
    represent ASME compliance.

    The method requires solutions on at least 3 systematically refined grids.
    Constant refinement ratios are simplest, but unequal ratios are supported
    by solving the generalized three-level order equation.

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
    _mesh_sizes: List[float] = field(default_factory=list, init=False, repr=False)
    _values: List[float] = field(default_factory=list, init=False, repr=False)
    _errors: List[Optional[float]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        order = _finite_real(self.theoretical_order, "theoretical_order")
        if order <= 0.0:
            raise ValueError("theoretical_order must be greater than zero")
        factor = _finite_real(self.safety_factor, "safety_factor")
        if factor < 1.0:
            raise ValueError("safety_factor must be at least one")
        self.theoretical_order = order
        self.safety_factor = factor

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
        try:
            mesh_size = _finite_real(h, "h")
        except ValueError as error:
            raise ValueError("h must be finite and greater than zero") from error
        if mesh_size <= 0.0:
            raise ValueError("h must be finite and greater than zero")
        solution_value = _finite_real(value, "value")
        error_value = None if error is None else _finite_real(error, "error")
        if error_value is not None and error_value < 0.0:
            raise ValueError("error must be finite and nonnegative")

        self._mesh_sizes.append(mesh_size)
        self._values.append(solution_value)
        self._errors.append(error_value)
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
            ValueError: If fewer than three levels are available, the sequence
                is degenerate or oscillatory, analytical errors are only
                partially reported, or relative GCI normalization is undefined.
        """
        theoretical_order = _finite_real(self.theoretical_order, "theoretical_order")
        if theoretical_order <= 0.0:
            raise ValueError("theoretical_order must be greater than zero")
        safety_factor = _finite_real(self.safety_factor, "safety_factor")
        if safety_factor < 1.0:
            raise ValueError("safety_factor must be at least one")

        level_count = len(self._mesh_sizes)
        if level_count < 3:
            raise ValueError(f"Need at least 3 grid levels, got {level_count}")
        if len(self._values) != level_count or len(self._errors) != level_count:
            raise RuntimeError("convergence-study storage is internally inconsistent")

        # Sort by mesh size (coarsest to finest)
        idx = np.argsort(self._mesh_sizes)[::-1]
        h = np.asarray(self._mesh_sizes, dtype=np.float64)[idx]
        f = np.asarray(self._values, dtype=np.float64)[idx]

        # Use the three finest grids
        h1, h2, h3 = h[-3], h[-2], h[-1]  # h1 > h2 > h3 (coarse to fine)
        f1, f2, f3 = f[-3], f[-2], f[-1]

        if len(np.unique(h)) != len(h):
            raise ValueError("Mesh sizes must be unique for convergence analysis")

        # Refinement ratios
        r21 = h1 / h2
        r32 = h2 / h3
        log_r21 = float(np.log(r21))
        log_r32 = float(np.log(r32))
        ratio_resolution = 100.0 * np.finfo(float).eps
        if log_r21 <= ratio_resolution or log_r32 <= ratio_resolution:
            raise ValueError(
                "Successive mesh sizes are too close to define a numerically "
                "meaningful refinement ratio"
            )

        # Estimate observed order. For a constant ratio,
        # p = ln(|(f1 - f2) / (f2 - f3)|) / ln(r).
        with np.errstate(over="ignore", invalid="ignore"):
            eps32 = f3 - f2
            eps21 = f2 - f1
        if not np.isfinite(eps32) or not np.isfinite(eps21):
            raise ValueError(
                "Successive solution differences overflowed; rescale the "
                "quantity of interest"
            )

        solution_scale = max(abs(f1), abs(f2), abs(f3))
        resolution = 100.0 * abs(float(np.spacing(solution_scale)))
        if abs(eps32) <= resolution or abs(eps21) <= resolution:
            raise ValueError(
                "Observed order is indeterminate because successive solutions "
                "are identical within floating-point resolution. Verify that the "
                "refinement parameter actually reaches the solver."
            )

        if np.sign(eps32) != np.sign(eps21):
            raise ValueError(
                "The three finest solutions converge oscillatory; monotonic "
                "Richardson extrapolation and GCI-style indices are not valid "
                "for this sequence"
            )

        observed_order = self._compute_order_fixed_point(eps21, eps32, r21, r32)

        if not np.isfinite(observed_order) or observed_order <= 0.0:
            raise ValueError(
                f"Observed order is not physically interpretable: {observed_order!r}"
            )

        fine_exponent = observed_order * log_r32
        coarse_exponent = observed_order * log_r21
        if max(fine_exponent, coarse_exponent) >= np.log(np.finfo(float).max):
            raise ValueError(
                "Observed order makes the Richardson/GCI denominators overflow"
            )
        extrapolation_denominator = float(np.expm1(fine_exponent))
        coarse_denominator = float(np.expm1(coarse_exponent))
        if (
            extrapolation_denominator <= np.finfo(float).eps
            or coarse_denominator <= np.finfo(float).eps
        ):
            raise ValueError("Richardson/GCI denominator is numerically zero")

        # Richardson extrapolation: f_exact ≈ f3 + (f3 - f2) / (r32^p - 1)
        richardson_estimate = f3 + eps32 / extrapolation_denominator
        if not np.isfinite(richardson_estimate):
            raise ValueError("Richardson extrapolation produced a non-finite value")

        # Grid Convergence Index (GCI)
        # GCI = Fs * |eps| / (r^p - 1)
        if f3 == 0.0 or f2 == 0.0:
            raise ValueError(
                "Relative GCI-style indices are undefined when a fine or "
                "medium-grid quantity of interest is zero"
            )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            e_a_fine = abs((f3 - f2) / f3)
            e_a_coarse = abs((f2 - f1) / f2)
        if not np.isfinite(e_a_fine) or not np.isfinite(e_a_coarse):
            raise ValueError(
                "Relative grid differences overflowed; rescale the quantity "
                "of interest before computing GCI-style indices"
            )

        gci_fine = safety_factor * e_a_fine / extrapolation_denominator
        gci_coarse = safety_factor * e_a_coarse / coarse_denominator
        if not np.isfinite(gci_fine) or not np.isfinite(gci_coarse):
            raise ValueError("GCI-style calculation produced a non-finite value")

        # A ratio near one is consistent with the assumed asymptotic model.
        refinement_scale = float(np.exp(fine_exponent))
        asymptotic_ratio = gci_coarse / (refinement_scale * gci_fine)
        if not np.isfinite(asymptotic_ratio):
            raise ValueError("Asymptotic-ratio calculation produced a non-finite value")
        is_asymptotic = 0.95 <= asymptotic_ratio <= 1.05

        has_errors = [error is not None for error in self._errors]
        if any(has_errors) and not all(has_errors):
            raise ValueError(
                "Analytical errors must be provided for every grid level or "
                "omitted for every grid level"
            )
        errors = None
        if all(has_errors):
            errors = _readonly(np.asarray(self._errors, dtype=np.float64)[idx])

        return ConvergenceResult(
            observed_order=float(observed_order),
            theoretical_order=theoretical_order,
            richardson_estimate=float(richardson_estimate),
            gci_fine=float(gci_fine),
            gci_coarse=float(gci_coarse),
            asymptotic_ratio=float(asymptotic_ratio),
            mesh_sizes=_readonly(h),
            errors=errors,
            solutions=_readonly(f),
            is_asymptotic=bool(is_asymptotic),
        )

    def _compute_order_fixed_point(
        self,
        eps21: float,
        eps32: float,
        r21: float,
        r32: float,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> float:
        """Solve the generalized monotonic three-level order equation.

        The historical private-method name is retained for compatibility, but
        a bracketed solve is used so a failed iteration cannot silently return
        an unconverged estimate.
        """

        log_r21 = float(np.log(r21))
        log_r32 = float(np.log(r32))
        target = float(np.log(abs(eps21 / eps32)))

        def log_expm1(value: float) -> float:
            if value > 50.0:
                return value + float(np.log1p(-np.exp(-value)))
            return float(np.log(np.expm1(value)))

        def model_log_ratio(order: float) -> float:
            return (
                log_expm1(order * log_r21)
                + order * log_r32
                - log_expm1(order * log_r32)
            )

        zero_order_limit = float(np.log(log_r21 / log_r32))
        tolerance = tol * max(1.0, abs(target), abs(zero_order_limit))
        if target <= zero_order_limit + tolerance:
            raise ValueError(
                "The successive differences do not imply a positive observed order"
            )

        lower = 0.0
        upper = 1.0
        for _ in range(max_iter):
            if model_log_ratio(upper) >= target:
                break
            upper *= 2.0
        else:
            raise ValueError("Could not bracket a finite positive observed order")

        for _ in range(max_iter):
            midpoint = 0.5 * (lower + upper)
            residual = model_log_ratio(midpoint) - target
            if abs(residual) <= tolerance:
                return midpoint
            if residual < 0.0:
                lower = midpoint
            else:
                upper = midpoint
            if upper - lower <= tol * max(1.0, midpoint):
                return 0.5 * (lower + upper)

        raise ValueError("Observed-order solve did not converge")


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

    Raises:
        TypeError: If either input is not a real one-dimensional array.
        ValueError: If fewer than three levels are supplied, shapes differ,
            mesh sizes or errors are non-positive/non-finite, mesh sizes are
            duplicated or numerically indistinguishable, or the regression is
            otherwise degenerate.
    """
    h = _real_vector(mesh_sizes, "mesh_sizes")
    error_values = _real_vector(errors, "errors")
    if h.size != error_values.size:
        raise ValueError("mesh_sizes and errors must have the same length")
    if h.size < 3:
        raise ValueError("at least three refinement levels are required")
    if not np.all(np.isfinite(h)) or np.any(h <= 0.0):
        raise ValueError("mesh_sizes must be finite and greater than zero")
    if not np.all(np.isfinite(error_values)) or np.any(error_values <= 0.0):
        raise ValueError("errors must be finite and strictly positive")
    if np.unique(h).size != h.size:
        raise ValueError("mesh_sizes must be unique")

    log_h = np.log(h)
    log_e = np.log(error_values)
    centered_h = log_h - np.mean(log_h)
    centered_e = log_e - np.mean(log_e)
    h_energy = float(centered_h @ centered_h)
    h_resolution = (
        100.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(log_h)))) ** 2
    )
    if h_energy <= h_resolution:
        raise ValueError(
            "mesh_sizes are too close to condition an observed-order regression"
        )

    e_energy = float(centered_e @ centered_e)
    e_resolution = (
        100.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(log_e)))) ** 2
    )
    if e_energy <= e_resolution:
        raise ValueError(
            "observed order is undefined because the supplied errors are constant "
            "within floating-point resolution"
        )

    order = float((centered_h @ centered_e) / h_energy)
    intercept = float(np.mean(log_e) - order * np.mean(log_h))
    coefficient = float(np.exp(intercept))
    if not np.isfinite(order) or not np.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("order regression produced a non-finite coefficient")

    log_e_fit = order * log_h + intercept
    ss_res = np.sum((log_e - log_e_fit) ** 2)
    if not np.isfinite(ss_res):
        raise ValueError("order regression residuals are non-finite")
    r_squared = float(np.clip(1.0 - ss_res / e_energy, 0.0, 1.0))

    return order, coefficient, r_squared


def _grid_resolutions(values: Sequence[int]) -> List[int]:
    try:
        raw_values = list(values)
    except TypeError as error:
        raise TypeError("n_values must be an iterable of integers") from error
    if len(raw_values) < 3:
        raise ValueError("n_values must contain at least three refinement levels")

    resolutions: List[int] = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError("n_values must contain only integers")
        resolution = int(value)
        if resolution <= 0:
            raise ValueError("n_values must contain only positive integers")
        resolutions.append(resolution)
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("n_values must contain unique refinement levels")
    return sorted(resolutions)


def _time_steps(values: Sequence[float]) -> List[float]:
    steps = _real_vector(values, "dt_values")
    if steps.size < 3:
        raise ValueError("dt_values must contain at least three refinement levels")
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("dt_values must be finite and greater than zero")
    if np.unique(steps).size != steps.size:
        raise ValueError("dt_values must contain unique refinement levels")
    return sorted((float(step) for step in steps), reverse=True)


def _solve_result(raw_result: object, context: str) -> Tuple[float, Optional[float]]:
    if isinstance(raw_result, tuple):
        if len(raw_result) != 2:
            raise TypeError(
                f"solve_func must return a scalar or a (value, error) pair at {context}"
            )
        raw_value, raw_error = raw_result
    else:
        raw_value, raw_error = raw_result, None

    value = _finite_real(raw_value, f"solve_func value at {context}")
    if raw_error is None:
        return value, None
    error = _finite_real(raw_error, f"solve_func error at {context}")
    if error < 0.0:
        raise ValueError(f"solve_func error at {context} must be nonnegative")
    return value, error


def _grid_characteristic_sizes(
    input_resolutions: Sequence[int],
    resolutions: Sequence[int],
    characteristic_sizes: Optional[Sequence[float]],
    size_to_h: Optional[Callable[[int], float]],
) -> List[float]:
    """Resolve and preflight the physical refinement size for each resolution."""
    if characteristic_sizes is not None and size_to_h is not None:
        raise ValueError("provide either characteristic_sizes or size_to_h, not both")

    if characteristic_sizes is not None:
        raw_sizes = _real_vector(characteristic_sizes, "characteristic_sizes")
        if raw_sizes.size != len(input_resolutions):
            raise ValueError(
                "characteristic_sizes must have the same length as n_values"
            )
        size_by_resolution = {
            int(resolution): float(size)
            for resolution, size in zip(input_resolutions, raw_sizes)
        }
        sizes = [size_by_resolution[resolution] for resolution in resolutions]
    elif size_to_h is not None:
        if not callable(size_to_h):
            raise TypeError("size_to_h must be callable")
        sizes = []
        for resolution in resolutions:
            try:
                raw_size = size_to_h(resolution)
            except Exception as callback_error:
                raise RuntimeError(
                    f"size_to_h failed for N={resolution}"
                ) from callback_error
            sizes.append(_finite_real(raw_size, f"size_to_h result for N={resolution}"))
    else:
        # Backward-compatible convention: N is the number of cells on a unit domain.
        sizes = [1.0 / resolution for resolution in resolutions]

    if not np.all(np.isfinite(sizes)) or any(size <= 0.0 for size in sizes):
        raise ValueError("characteristic sizes must be finite and greater than zero")
    if len(set(sizes)) != len(sizes):
        raise ValueError("characteristic sizes must be unique")
    if any(coarse <= fine for coarse, fine in zip(sizes, sizes[1:])):
        raise ValueError("characteristic sizes must decrease as N increases")
    return sizes


def run_convergence_study(
    solve_func: Callable[[int], ConvergenceSolveResult],
    n_values: Sequence[int],
    theoretical_order: float = 2.0,
    verbose: bool = True,
    *,
    characteristic_sizes: Optional[Sequence[float]] = None,
    size_to_h: Optional[Callable[[int], float]] = None,
) -> ConvergenceResult:
    """Run an automated grid convergence study.

    Args:
        solve_func: Function that takes integer resolution ``N`` and returns a
            scalar QoI or ``(value, error)`` pair.
        n_values: Grid resolutions passed to ``solve_func``. By default ``N``
            means the number of cells on a unit-length domain and ``h=1/N``.
            If ``N`` instead counts grid points, supply
            ``size_to_h=lambda n: 1.0 / (n - 1)`` or explicit
            ``characteristic_sizes``.
        theoretical_order: Expected order of accuracy
        verbose: Whether to print progress
        characteristic_sizes: Optional physical ``h`` values paired with
            ``n_values`` in the caller's original order. This supports
            non-unit domains and conventions such as point counts.
        size_to_h: Optional function mapping each resolution ``N`` to its
            physical characteristic size. Mutually exclusive with
            ``characteristic_sizes``.

    Returns:
        ConvergenceResult from the analysis

    Raises:
        TypeError: If resolutions, callback, or callback values violate the
            scalar convergence-study contract.
        ValueError: If refinement levels, characteristic sizes, or returned
            values are invalid.
        RuntimeError: If ``solve_func`` raises; the failed resolution is
            included in the message and the original exception is chained.

    Example:
        >>> def solve(n):
        ...     mesh = bt.mesh_1d(n)
        ...     result = solver.run(problem, t_end)
        ...     error = np.max(np.abs(result.solution() - analytical))
        ...     return result.solution()[n//2], error  # midpoint value
        >>> result = run_convergence_study(solve, [25, 50, 100, 200])
    """
    if not callable(solve_func):
        raise TypeError("solve_func must be callable")
    try:
        input_resolutions = tuple(n_values)
    except TypeError as error:
        raise TypeError("n_values must be a sequence of integers") from error
    resolutions = _grid_resolutions(input_resolutions)
    study = GridConvergenceStudy(theoretical_order=theoretical_order)
    sizes = _grid_characteristic_sizes(
        input_resolutions,
        resolutions,
        characteristic_sizes,
        size_to_h,
    )

    if verbose:
        print("=" * 60)
        print("Grid Convergence Study")
        print("=" * 60)
        print(f"{'N':>8} {'h':>12} {'Value':>16} {'Error':>14}")
        print("-" * 60)

    for n, h in zip(resolutions, sizes):
        try:
            raw_result = solve_func(n)
        except Exception as callback_error:
            raise RuntimeError(f"solve_func failed for N={n}") from callback_error
        value, reported_error = _solve_result(raw_result, f"N={n}")

        study.add_solution(h=h, value=value, error=reported_error)

        if verbose:
            if reported_error is not None:
                print(f"{n:>8} {h:>12.6f} {value:>16.8f} {reported_error:>14.2e}")
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
    solve_func: Callable[[float], ConvergenceSolveResult],
    dt_values: Sequence[float],
    theoretical_order: float = 1.0,
    verbose: bool = True,
) -> ConvergenceResult:
    """Run a temporal convergence study.

    Args:
        solve_func: Function that takes ``dt`` and returns a scalar QoI or a
            ``(value, error)`` pair. The error may be ``None``.
        dt_values: List of time step sizes to test
        theoretical_order: Expected temporal order (1 for explicit, 2 for CN)
        verbose: Whether to print progress

    Returns:
        ConvergenceResult from the analysis

    Raises:
        TypeError: If time steps, callback, or callback values violate the
            scalar convergence-study contract.
        ValueError: If refinement levels or returned values are invalid.
        RuntimeError: If ``solve_func`` raises; the failed time step is
            included in the message and the original exception is chained.
    """
    if not callable(solve_func):
        raise TypeError("solve_func must be callable")
    steps = _time_steps(dt_values)
    study = GridConvergenceStudy(theoretical_order=theoretical_order)

    if verbose:
        print("=" * 60)
        print("Temporal Convergence Study")
        print("=" * 60)
        print(f"{'dt':>12} {'Value':>16} {'Error':>14}")
        print("-" * 60)

    for dt in steps:  # Coarse to fine
        try:
            raw_result = solve_func(dt)
        except Exception as callback_error:
            raise RuntimeError(
                f"solve_func failed for dt={dt:.17g}"
            ) from callback_error
        value, reported_error = _solve_result(raw_result, f"dt={dt:.17g}")

        study.add_solution(h=dt, value=value, error=reported_error)

        if verbose:
            if reported_error is not None:
                print(f"{dt:>12.6f} {value:>16.8f} {reported_error:>14.2e}")
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

    Raises:
        ValueError: If the result lacks plottable, finite, strictly positive
            data. A log-log plot cannot faithfully display zero or negative
            values.
    """
    import matplotlib.pyplot as plt

    if not isinstance(result, ConvergenceResult):
        raise TypeError("result must be a ConvergenceResult")
    if result.solutions is None:
        raise ValueError("result.solutions is required for convergence plotting")
    h = _real_vector(result.mesh_sizes, "result.mesh_sizes")
    f = _real_vector(result.solutions, "result.solutions")
    if h.size != f.size or h.size < 3:
        raise ValueError(
            "result mesh_sizes and solutions must have matching lengths of at least three"
        )
    if not np.all(np.isfinite(h)) or np.any(h <= 0.0):
        raise ValueError("result.mesh_sizes must be finite and strictly positive")
    if not np.all(np.isfinite(f)) or np.any(f <= 0.0):
        raise ValueError(
            "result.solutions must be finite and strictly positive for a log-log plot"
        )

    error_values = None
    if result.errors is not None:
        error_values = _real_vector(result.errors, "result.errors")
        if error_values.size != h.size:
            raise ValueError("result.errors length must match result.mesh_sizes")
        if not np.all(np.isfinite(error_values)) or np.any(error_values <= 0.0):
            raise ValueError(
                "result.errors must be finite and strictly positive for a log-log plot"
            )

    gci_fine = None
    if show_gci:
        gci_fine = _finite_real(result.gci_fine, "result.gci_fine")
        if gci_fine < 0.0:
            raise ValueError("result.gci_fine must be nonnegative")
        if gci_fine >= 1.0:
            raise ValueError(
                "fine-grid GCI interval reaches zero and cannot be shown on a log axis"
            )

    richardson = None
    if show_richardson:
        richardson = _finite_real(
            result.richardson_estimate, "result.richardson_estimate"
        )
        if richardson <= 0.0:
            raise ValueError(
                "result.richardson_estimate must be positive for a log-log plot"
            )

    observed_order = _finite_real(result.observed_order, "result.observed_order")
    theoretical_order = _finite_real(
        result.theoretical_order, "result.theoretical_order"
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))

    # Plot solutions
    ax.loglog(h, f, "bo-", markersize=10, linewidth=2, label="Computed")

    if error_values is not None:
        # Plot errors with order reference
        ax.loglog(h, error_values, "rs-", markersize=10, linewidth=2, label="Error")

        # Reference lines
        h_ref = h[len(h) // 2]
        e_ref = error_values[len(h) // 2]

        h_line = np.logspace(np.log10(h.min() / 1.5), np.log10(h.max() * 1.5), 50)
        e_theoretical = e_ref * (h_line / h_ref) ** theoretical_order
        e_observed = e_ref * (h_line / h_ref) ** observed_order
        if not np.all(np.isfinite(e_theoretical)) or not np.all(
            np.isfinite(e_observed)
        ):
            raise ValueError("convergence reference lines became non-finite")

        ax.loglog(
            h_line,
            e_theoretical,
            "k--",
            alpha=0.5,
            label=f"O(h^{theoretical_order:.0f}) theoretical",
        )
        ax.loglog(
            h_line,
            e_observed,
            "g:",
            alpha=0.7,
            linewidth=2,
            label=f"O(h^{observed_order:.2f}) observed",
        )

    if show_gci:
        assert gci_fine is not None
        ax.errorbar(
            [h[-1]],
            [f[-1]],
            yerr=[gci_fine * f[-1]],
            fmt="none",
            ecolor="tab:purple",
            capsize=5,
            label="Fine-grid GCI",
        )

    # Show Richardson estimate
    if show_richardson:
        assert richardson is not None
        ax.axhline(
            richardson,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Richardson: {richardson:.6f}",
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Value / Error", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    return ax


__all__ = [
    "ConvergenceResult",
    "ConvergenceSolveResult",
    "GridConvergenceStudy",
    "compute_order_of_accuracy",
    "plot_convergence",
    "run_convergence_study",
    "temporal_convergence_study",
]
