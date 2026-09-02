"""Legacy Python time-integration orchestration.

The standalone :func:`euler_step`, :func:`heun_step`, and :func:`rk4_step`
functions are generic ODE building blocks.  The :class:`HeunIntegrator` and
:class:`RK4Integrator` problem wrappers are deliberately narrower: they only
implement one-dimensional, uniform-diffusivity diffusion with fixed
Dirichlet values at the left and right boundaries.  They reject other
transport physics instead of silently dropping it.

For general transport problems, including reactions, sources, advection,
variable diffusivity, multidimensional meshes, or natural boundary
conditions, use :func:`biotransport.solve`.  The ``euler`` branch of
:func:`integrate` delegates to that canonical C++ solver.

Example:
    >>> mesh = bt.mesh_1d(100, 0.0, 1.0)
    >>> problem = bt.Problem(mesh).diffusivity(1e-5).initial_condition(u0)
    >>>
    >>> # RK4 integration
    >>> integrator = bt.RK4Integrator(problem)
    >>> result = integrator.solve(t_end=1.0, dt=0.01)
    >>> print(result.solution)
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Optional, cast

import numpy as np

from ._core import (
    BoundaryCondition,
    BoundaryType,
    StructuredMesh,
    TransportProblem,
)


# Largest binary64 value known to remain inside RK4's negative-real-axis
# stability interval.  The nearest decimal spelling above this value rounds
# one ULP past the exact root and is therefore not a safe ceiling.
_RK4_NEGATIVE_REAL_AXIS_RADIUS = float.fromhex("0x1.64847fde4ae0dp+1")
_MAXIMUM_LEGACY_STEPS = 10_000_000
_NUMERIC_ARRAY_KINDS = frozenset("iufc")
_PROBLEM_CONTRACT_METHODS = (
    "mesh",
    "diffusivity",
    "boundaries",
    "initial",
    "has_uniform_diffusivity",
    "has_reaction",
    "has_advection",
)


def _finite_real_scalar(value: object, name: str) -> float:
    """Return a finite real scalar without accepting boolean lookalikes."""
    if isinstance(value, (bool, np.bool_, str, bytes, complex, np.complexfloating)):
        raise TypeError(f"{name} must be a real number")
    try:
        resolved = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not np.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def _scaled_positive_ratio(
    numerators: tuple[float, ...],
    denominators: tuple[float, ...],
    *,
    binary_exponent: int = 0,
    conservative: bool = False,
) -> float:
    """Evaluate a non-negative product/ratio without intermediate range loss.

    All factors are expected to be finite.  The mantissa is normalized after
    every operation, so overflow or underflow is classified only when the
    final binary64 result itself is outside the representable range.
    """
    mantissa = 1.0
    exponent = binary_exponent

    for factor in numerators:
        if factor < 0.0 or not math.isfinite(factor):
            raise ValueError(
                "scaled-ratio numerator factors must be finite and non-negative"
            )
        if factor == 0.0:
            return 0.0
        factor_mantissa, factor_exponent = math.frexp(factor)
        mantissa *= factor_mantissa
        exponent += factor_exponent
        mantissa, shift = math.frexp(mantissa)
        exponent += shift

    for divisor in denominators:
        if divisor <= 0.0 or not math.isfinite(divisor):
            raise ValueError(
                "scaled-ratio denominator factors must be finite and positive"
            )
        divisor_mantissa, divisor_exponent = math.frexp(divisor)
        mantissa /= divisor_mantissa
        exponent -= divisor_exponent
        mantissa, shift = math.frexp(mantissa)
        exponent += shift

    try:
        result = math.ldexp(mantissa, exponent)
    except OverflowError:
        return float("inf")
    if conservative and result > 0.0:
        exact = Fraction(1)
        for factor in numerators:
            exact *= Fraction.from_float(factor)
        for divisor in denominators:
            exact /= Fraction.from_float(divisor)
        if binary_exponent >= 0:
            exact *= 1 << binary_exponent
        else:
            exact /= 1 << -binary_exponent
        if Fraction.from_float(result) > exact:
            result = math.nextafter(result, 0.0)
    return result


def _validated_ode_array(
    value: object,
    name: str,
    *,
    expected_shape: Optional[tuple[int, ...]] = None,
    nonfinite_error: type[Exception] = ValueError,
    allow_complex: bool = True,
) -> np.ndarray:
    """Copy a finite numeric ODE array and enforce an exact stage shape."""
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ValueError(f"{name} must not contain masked values")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array") from exc
    if array.dtype.kind not in _NUMERIC_ARRAY_KINDS:
        raise TypeError(f"{name} must contain numeric values")
    if not allow_complex and array.dtype.kind == "c":
        raise TypeError(f"{name} must contain real values")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            f"{name} shape {array.shape} does not match state shape {expected_shape}"
        )
    if not np.all(np.isfinite(array)):
        raise nonfinite_error(f"{name} must contain finite values only")
    return array.copy()


def _prepare_ode_step(
    u: object,
    rhs: Callable,
    t: object,
    dt: object,
) -> tuple[np.ndarray, float, float]:
    if not callable(rhs):
        raise TypeError("rhs must be callable")
    state = _validated_ode_array(u, "state")
    time = _finite_real_scalar(t, "t")
    step = _finite_real_scalar(dt, "dt")
    if step == 0.0:
        raise ValueError("dt must be nonzero")
    end_time = time + step
    if not np.isfinite(end_time) or end_time == time:
        raise ValueError(
            "t + dt must be a distinct finite floating-point time; shift the "
            "time origin or increase |dt|"
        )
    return state, time, step


def _evaluate_ode_rhs(
    rhs: Callable,
    state: np.ndarray,
    time: float,
    stage: str,
) -> np.ndarray:
    """Evaluate one stage without allowing the callback to alias solver state."""
    stage_time = _finite_real_scalar(time, f"{stage} time")
    derivative = rhs(state.copy(), stage_time)
    return _validated_ode_array(
        derivative,
        f"{stage} rhs result",
        expected_shape=state.shape,
        nonfinite_error=FloatingPointError,
    )


def _checked_ode_update(
    state: np.ndarray,
    stage: str,
    *weighted_derivatives: tuple[float, np.ndarray],
) -> np.ndarray:
    result_dtype = np.result_type(
        state,
        *(derivative for _, derivative in weighted_derivatives),
        *(weight for weight, _ in weighted_derivatives),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        increment = np.zeros(state.shape, dtype=result_dtype)
        for weight, derivative in weighted_derivatives:
            increment = increment + weight * derivative
        updated = state + increment
    if not np.all(np.isfinite(increment)) or not np.all(np.isfinite(updated)):
        raise FloatingPointError(f"{stage} produced a non-finite state")
    return updated


def _checked_scaled_derivative(
    derivative: np.ndarray,
    step: float,
    divisor: float,
    stage: str,
) -> np.ndarray:
    """Return ``step * derivative / divisor`` with balanced operation orders."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        direct = (step * derivative) / divisor
        divide_step_first = (step / divisor) * derivative
        divide_derivative_first = step * (derivative / divisor)

    increment = direct.copy()
    unusable = ~np.isfinite(increment)
    rescued = np.isfinite(divide_step_first)
    increment = np.where(unusable & rescued, divide_step_first, increment)
    unusable = ~np.isfinite(increment)
    rescued = np.isfinite(divide_derivative_first)
    increment = np.where(unusable & rescued, divide_derivative_first, increment)

    # A different association can preserve a representable subnormal that a
    # premature division rounded to zero.
    lost_nonzero = (increment == 0) & (derivative != 0) & (step != 0)
    increment = np.where(
        lost_nonzero & (divide_step_first != 0),
        divide_step_first,
        increment,
    )
    lost_nonzero = (increment == 0) & (derivative != 0) & (step != 0)
    increment = np.where(
        lost_nonzero & (divide_derivative_first != 0),
        divide_derivative_first,
        increment,
    )

    if not np.all(np.isfinite(increment)):
        raise FloatingPointError(f"{stage} produced a non-finite state")
    return increment


def _stage_derivative_average(
    stage: str,
    *weighted_derivatives: tuple[float, np.ndarray],
) -> np.ndarray:
    """Combine positive RK stage weights before applying the time step.

    Normalizing by the largest component keeps the convex combination in
    range and, importantly, avoids rounding each fractional contribution to
    zero before they are summed.
    """
    if not weighted_derivatives:
        raise ValueError("at least one stage derivative is required")
    total_weight = sum(weight for weight, _ in weighted_derivatives)
    if total_weight <= 0.0 or any(weight < 0.0 for weight, _ in weighted_derivatives):
        raise ValueError("stage derivative weights must be non-negative")

    derivatives = tuple(derivative for _, derivative in weighted_derivatives)
    result_dtype = np.result_type(*derivatives, float)
    scale = np.zeros(derivatives[0].shape, dtype=float)
    for derivative in derivatives:
        if np.iscomplexobj(derivative):
            magnitude = np.maximum(
                np.abs(np.real(derivative)),
                np.abs(np.imag(derivative)),
            )
        else:
            magnitude = np.abs(derivative)
        scale = np.maximum(scale, magnitude)

    _, scale_exponent = np.frexp(scale)
    normalized_sum = np.zeros(derivatives[0].shape, dtype=result_dtype)
    compensation = np.zeros(derivatives[0].shape, dtype=result_dtype)

    def scale_by_power_of_two(
        values: np.ndarray,
        exponents: np.ndarray,
    ) -> np.ndarray:
        if np.iscomplexobj(values):
            return np.ldexp(np.real(values), exponents) + 1j * np.ldexp(
                np.imag(values),
                exponents,
            )
        return np.ldexp(values, exponents)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for weight, derivative in weighted_derivatives:
            normalized_derivative = scale_by_power_of_two(
                derivative,
                -scale_exponent,
            )
            term = weight * normalized_derivative
            candidate = normalized_sum + term
            correction = np.where(
                np.abs(normalized_sum) >= np.abs(term),
                (normalized_sum - candidate) + term,
                (term - candidate) + normalized_sum,
            )
            compensation = compensation + correction
            normalized_sum = candidate

        normalized_average = (normalized_sum + compensation) / total_weight
        averaged = scale_by_power_of_two(normalized_average, scale_exponent)
    if not np.all(np.isfinite(averaged)):
        raise FloatingPointError(f"{stage} derivative average is not finite")
    return averaged


def _problem_flag(problem: TransportProblem, name: str, component: str) -> bool:
    """Read a native problem capability required for honest legacy dispatch."""
    accessor = getattr(TransportProblem, name, None)
    if accessor is None:
        raise RuntimeError(
            f"{component} requires a current biotransport native extension with "
            f"TransportProblem.{name}(). Rebuild the extension before using this "
            "legacy Python integrator."
        )
    return bool(accessor(problem))


def _reject_problem_contract_overrides(
    problem: TransportProblem,
    component: str,
) -> None:
    """Prevent Python overrides from weakening authoritative native checks."""
    if not isinstance(problem, TransportProblem):
        raise TypeError(f"{component} problem must be a TransportProblem")
    overridden: list[str] = []
    for problem_type in type(problem).__mro__:
        if problem_type is TransportProblem:
            break
        overridden.extend(
            name for name in _PROBLEM_CONTRACT_METHODS if name in problem_type.__dict__
        )
    if overridden:
        names = ", ".join(sorted(set(overridden)))
        raise TypeError(
            f"{component} cannot use Python overrides of TransportProblem "
            f"contract methods ({names}); configure the native problem directly"
        )


def _validate_legacy_diffusion_problem(
    problem: TransportProblem,
    component: str,
) -> tuple[StructuredMesh, float, BoundaryCondition, BoundaryCondition]:
    """Validate the intentionally narrow problem contract of Python steppers."""
    _reject_problem_contract_overrides(problem, component)
    mesh = TransportProblem.mesh(problem)
    if not mesh.is_1d():
        raise ValueError(
            f"{component} supports only 1D diffusion; use biotransport.solve "
            "for 2D or 3D transport."
        )
    if not _problem_flag(problem, "has_uniform_diffusivity", component):
        raise ValueError(
            f"{component} does not support variable diffusivity; "
            "use biotransport.solve."
        )
    if _problem_flag(problem, "has_reaction", component):
        raise ValueError(
            f"{component} does not support reactions or sources; "
            "use biotransport.solve."
        )
    if _problem_flag(problem, "has_advection", component):
        raise ValueError(
            f"{component} does not support advection; use biotransport.solve."
        )

    diffusivity = float(TransportProblem.diffusivity(problem))
    if not np.isfinite(diffusivity) or diffusivity < 0.0:
        raise ValueError(f"{component} requires a finite, non-negative diffusivity")

    boundaries = TransportProblem.boundaries(problem)
    left_bc = boundaries[0]
    right_bc = boundaries[1]
    if (
        left_bc.type != BoundaryType.DIRICHLET
        or right_bc.type != BoundaryType.DIRICHLET
    ):
        raise ValueError(
            f"{component} supports Dirichlet left/right boundaries only; "
            "use biotransport.solve for Neumann or Robin conditions."
        )
    if not np.isfinite(float(left_bc.value)) or not np.isfinite(float(right_bc.value)):
        raise ValueError(f"{component} requires finite Dirichlet boundary values")

    return mesh, diffusivity, left_bc, right_bc


@dataclass(frozen=True)
class _LegacyDiffusionSnapshot:
    """Owned data required by one validated legacy diffusion solve."""

    mesh: StructuredMesh
    diffusivity: float
    left_value: float
    right_value: float
    initial: np.ndarray


@dataclass(frozen=True)
class _ReadOnlyBoundaryCondition:
    """Immutable boundary metadata exposed by legacy integrator properties."""

    type: BoundaryType
    value: float


def _capture_legacy_diffusion_problem(
    problem: TransportProblem,
    component: str,
) -> _LegacyDiffusionSnapshot:
    """Revalidate a live problem and copy every value used by a solve."""
    mesh, diffusivity, left_bc, right_bc = _validate_legacy_diffusion_problem(
        problem, component
    )
    expected_shape = (int(mesh.nx()) + 1,)
    initial = _validated_ode_array(
        TransportProblem.initial(problem),
        "initial condition",
        expected_shape=expected_shape,
        allow_complex=False,
    ).astype(float, copy=False)
    initial.setflags(write=False)
    return _LegacyDiffusionSnapshot(
        mesh=mesh,
        diffusivity=diffusivity,
        left_value=float(left_bc.value),
        right_value=float(right_bc.value),
        initial=initial,
    )


def _validate_safety_factor(value: float, component: str) -> float:
    safety = _finite_real_scalar(value, f"{component} safety_factor")
    if not 0.0 < safety <= 1.0:
        raise ValueError(f"{component} safety_factor must be in (0, 1]")
    return safety


def _validate_solve_times(
    t_end: float, dt: Optional[float]
) -> tuple[float, Optional[float]]:
    final_time = _finite_real_scalar(t_end, "t_end")
    if final_time <= 0.0:
        raise ValueError("t_end must be finite and positive")
    if dt is None:
        return final_time, None

    requested_dt = _finite_real_scalar(dt, "dt")
    if requested_dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    return final_time, requested_dt


def _resolve_legacy_step(
    t_end: float,
    requested_dt: Optional[float],
    stable_dt: float,
    component: str,
) -> tuple[int, float]:
    """Resolve a stable nominal step and reject impractical Python loops.

    The returned step is used for every non-final interval.  A separately
    computed exact final remainder prevents a rounded uniform partition from
    numerically advancing farther than ``t_end``.
    """
    if not np.isfinite(stable_dt) and stable_dt != float("inf"):
        raise FloatingPointError(f"{component} stability limit is not finite")
    if stable_dt <= 0.0:
        raise FloatingPointError(f"{component} stability limit is not positive")

    if requested_dt is None:
        nominal_dt = min(t_end, stable_dt)
    else:
        # A requested dt is a ceiling.  A final interval shorter than that
        # ceiling is safe when it lies within the stability bound.
        nominal_dt = min(t_end, requested_dt)
        if np.isfinite(stable_dt) and nominal_dt > stable_dt:
            raise ValueError(
                f"{component} dt={requested_dt!r} exceeds the "
                f"safety-scaled stability limit {stable_dt!r}"
            )

    exact_ratio = Fraction.from_float(t_end) / Fraction.from_float(nominal_dt)
    num_steps = (
        exact_ratio.numerator + exact_ratio.denominator - 1
    ) // exact_ratio.denominator
    if num_steps > _MAXIMUM_LEGACY_STEPS:
        raise RuntimeError(
            f"{component} would require {num_steps} Python steps, exceeding "
            f"the {_MAXIMUM_LEGACY_STEPS} step limit; increase dt when stable "
            "or use biotransport.solve"
        )
    return num_steps, nominal_dt


def _exact_final_legacy_step(
    t_end: float,
    nominal_dt: float,
    num_steps: int,
    stable_dt: float,
    component: str,
) -> float:
    """Return a representable remainder whose exact durations sum to ``t_end``."""
    exact_final = Fraction.from_float(t_end) - (
        (num_steps - 1) * Fraction.from_float(nominal_dt)
    )
    if exact_final <= 0:
        raise FloatingPointError(
            f"{component} final time-step remainder is not positive"
        )
    final_dt = float(exact_final)
    if Fraction.from_float(final_dt) != exact_final:
        raise FloatingPointError(
            f"{component} final time-step remainder is not representable; "
            "choose a different dt"
        )
    if (
        not math.isfinite(final_dt)
        or final_dt <= 0.0
        or final_dt > nominal_dt
        or (math.isfinite(stable_dt) and final_dt > stable_dt)
    ):
        raise FloatingPointError(f"{component} final time-step remainder is not usable")
    return final_dt


@dataclass
class IntegrationResult:
    """Result of a time integration."""

    solution: np.ndarray
    """Final solution field."""

    time: float
    """Final simulation time reached."""

    stats: dict = field(default_factory=dict)
    """Statistics including steps, wall time, etc."""


def _compute_diffusion_rhs(
    mesh: StructuredMesh,
    u: np.ndarray,
    D: float,
) -> np.ndarray:
    """Compute the 1D uniform-diffusion right-hand side.

    The caller owns enforcement of the validated Dirichlet boundary values.

    Args:
        mesh: The computational mesh
        u: Current solution array
        D: Diffusion coefficient
    Returns:
        Array of du/dt values
    """
    n = len(u)
    dx = float(mesh.dx())
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("diffusion mesh spacing must be finite and positive")

    dudt = np.zeros(n, dtype=float)
    if D == 0.0:
        return dudt

    # Normalize the three-point stencil before applying D/dx².  This avoids
    # overflow in both dx² and the raw second difference while retaining a
    # finite final derivative whenever binary64 can represent it.
    for i in range(1, n - 1):
        left = float(u[i - 1])
        center = float(u[i])
        right = float(u[i + 1])
        field_magnitude = max(abs(left), abs(center), abs(right))
        if field_magnitude == 0.0:
            continue
        _, field_exponent = math.frexp(field_magnitude)
        normalized_laplacian = math.fsum(
            (
                math.ldexp(left, -field_exponent),
                -2.0 * math.ldexp(center, -field_exponent),
                math.ldexp(right, -field_exponent),
            )
        )
        if normalized_laplacian == 0.0:
            continue
        magnitude = _scaled_positive_ratio(
            (D, abs(normalized_laplacian)),
            (dx, dx),
            binary_exponent=field_exponent,
        )
        if not math.isfinite(magnitude):
            raise FloatingPointError(
                f"diffusion right-hand side at node {i} exceeds binary64 range"
            )
        if magnitude == 0.0:
            raise FloatingPointError(
                f"diffusion right-hand side at node {i} is below binary64 range"
            )
        dudt[i] = math.copysign(magnitude, normalized_laplacian)

    # Fixed Dirichlet nodes do not evolve.
    return dudt


def euler_step(u: np.ndarray, rhs: Callable, t: float, dt: float) -> np.ndarray:
    """Forward Euler step: u^{n+1} = u^n + dt * f(u^n, t^n).

    First-order accurate: O(dt).
    ``rhs`` receives an isolated state copy and must return finite numeric data
    with exactly the same shape.  The step rejects non-finite inputs or output.

    Args:
        u: Current state
        rhs: Function that computes du/dt given (u, t)
        t: Current time
        dt: Time step

    Returns:
        New state at t + dt
    """
    state, time, step = _prepare_ode_step(u, rhs, t, dt)
    derivative = _evaluate_ode_rhs(rhs, state, time, "Euler")
    return _checked_ode_update(state, "Euler step", (step, derivative))


def heun_step(u: np.ndarray, rhs: Callable, t: float, dt: float) -> np.ndarray:
    """Heun's method (improved Euler / RK2).

    Second-order accurate: O(dt²).

    k1 = f(u^n, t^n)
    k2 = f(u^n + dt*k1, t^n + dt)
    u^{n+1} = u^n + dt/2 * (k1 + k2)

    Each ``rhs`` call receives an isolated state copy and must return finite
    numeric data with exactly the same shape.  Non-finite stages are rejected.

    Args:
        u: Current state
        rhs: Function that computes du/dt given (u, t)
        t: Current time
        dt: Time step

    Returns:
        New state at t + dt
    """
    state, time, step = _prepare_ode_step(u, rhs, t, dt)
    k1 = _evaluate_ode_rhs(rhs, state, time, "Heun k1")
    predictor = _checked_ode_update(state, "Heun predictor", (step, k1))
    k2 = _evaluate_ode_rhs(rhs, predictor, time + step, "Heun k2")
    averaged_derivative = _stage_derivative_average(
        "Heun",
        (1.0, k1),
        (1.0, k2),
    )
    return _checked_ode_update(
        state,
        "Heun step",
        (step, averaged_derivative),
    )


def rk4_step(u: np.ndarray, rhs: Callable, t: float, dt: float) -> np.ndarray:
    """Classic 4th-order Runge-Kutta step.

    Fourth-order accurate: O(dt⁴).

    k1 = f(u^n, t^n)
    k2 = f(u^n + dt/2*k1, t^n + dt/2)
    k3 = f(u^n + dt/2*k2, t^n + dt/2)
    k4 = f(u^n + dt*k3, t^n + dt)
    u^{n+1} = u^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Each ``rhs`` call receives an isolated state copy and must return finite
    numeric data with exactly the same shape.  Non-finite stages are rejected.

    Args:
        u: Current state
        rhs: Function that computes du/dt given (u, t)
        t: Current time
        dt: Time step

    Returns:
        New state at t + dt
    """
    state, time, step = _prepare_ode_step(u, rhs, t, dt)
    midpoint = time + step / 2.0
    if not np.isfinite(midpoint) or midpoint == time or midpoint == time + step:
        raise ValueError(
            "RK4 midpoint time is not representable; shift the time origin or "
            "increase |dt|"
        )
    k1 = _evaluate_ode_rhs(rhs, state, time, "RK4 k1")
    stage_2_increment = _checked_scaled_derivative(k1, step, 2.0, "RK4 stage 2")
    stage_2 = _checked_ode_update(state, "RK4 stage 2", (1.0, stage_2_increment))
    k2 = _evaluate_ode_rhs(rhs, stage_2, midpoint, "RK4 k2")
    stage_3_increment = _checked_scaled_derivative(k2, step, 2.0, "RK4 stage 3")
    stage_3 = _checked_ode_update(state, "RK4 stage 3", (1.0, stage_3_increment))
    k3 = _evaluate_ode_rhs(rhs, stage_3, midpoint, "RK4 k3")
    stage_4 = _checked_ode_update(state, "RK4 stage 4", (step, k3))
    k4 = _evaluate_ode_rhs(rhs, stage_4, time + step, "RK4 k4")
    averaged_derivative = _stage_derivative_average(
        "RK4",
        (1.0, k1),
        (2.0, k2),
        (2.0, k3),
        (1.0, k4),
    )
    return _checked_ode_update(
        state,
        "RK4 step",
        (step, averaged_derivative),
    )


def _rk4_autonomous_step(
    u: np.ndarray,
    rhs: Callable,
    dt: float,
) -> np.ndarray:
    """RK4 step for an autonomous, internally owned right-hand side."""
    if not callable(rhs):
        raise TypeError("rhs must be callable")
    state = _validated_ode_array(u, "state")
    step = _finite_real_scalar(dt, "dt")
    if step <= 0.0:
        raise ValueError("dt must be positive")

    k1 = _evaluate_ode_rhs(rhs, state, 0.0, "RK4 k1")
    stage_2_increment = _checked_scaled_derivative(k1, step, 2.0, "RK4 stage 2")
    stage_2 = _checked_ode_update(state, "RK4 stage 2", (1.0, stage_2_increment))
    k2 = _evaluate_ode_rhs(rhs, stage_2, 0.0, "RK4 k2")
    stage_3_increment = _checked_scaled_derivative(k2, step, 2.0, "RK4 stage 3")
    stage_3 = _checked_ode_update(state, "RK4 stage 3", (1.0, stage_3_increment))
    k3 = _evaluate_ode_rhs(rhs, stage_3, 0.0, "RK4 k3")
    stage_4 = _checked_ode_update(state, "RK4 stage 4", (step, k3))
    k4 = _evaluate_ode_rhs(rhs, stage_4, 0.0, "RK4 k4")
    averaged_derivative = _stage_derivative_average(
        "RK4",
        (1.0, k1),
        (2.0, k2),
        (2.0, k3),
        (1.0, k4),
    )
    return _checked_ode_update(
        state,
        "RK4 step",
        (step, averaged_derivative),
    )


class _LegacyDiffusionIntegrator:
    """Read-only view over a live, solve-entry-validated problem."""

    def _initialize_legacy_problem(
        self,
        problem: TransportProblem,
        safety_factor: float,
        component: str,
    ) -> None:
        self._problem = problem
        self._component = component
        self._safety = _validate_safety_factor(safety_factor, component)
        self._last_snapshot = _capture_legacy_diffusion_problem(problem, component)

    def _refresh_snapshot(self) -> _LegacyDiffusionSnapshot:
        snapshot = _capture_legacy_diffusion_problem(
            self._problem,
            self._component,
        )
        self._last_snapshot = snapshot
        return snapshot

    @property
    def problem(self) -> TransportProblem:
        """The live problem revalidated at the beginning of every solve."""
        return self._problem

    @property
    def mesh(self) -> StructuredMesh:
        """Mesh from the most recently validated problem snapshot."""
        return self._last_snapshot.mesh

    @property
    def D(self) -> float:
        """Uniform diffusivity from the most recently validated snapshot."""
        return self._last_snapshot.diffusivity

    @property
    def safety(self) -> float:
        """Immutable safety factor used to compute the stability ceiling."""
        return self._safety

    @property
    def u0(self) -> np.ndarray:
        """Read-only view of the most recently validated initial field.

        Update :attr:`problem` with ``initial_condition(...)`` before the next
        solve instead of mutating this diagnostic view.
        """
        view = self._last_snapshot.initial.view()
        view.setflags(write=False)
        return view

    @property
    def left_bc(self) -> _ReadOnlyBoundaryCondition:
        """Immutable metadata for the last validated left boundary."""
        return _ReadOnlyBoundaryCondition(
            BoundaryType.DIRICHLET,
            self._last_snapshot.left_value,
        )

    @property
    def right_bc(self) -> _ReadOnlyBoundaryCondition:
        """Immutable metadata for the last validated right boundary."""
        return _ReadOnlyBoundaryCondition(
            BoundaryType.DIRICHLET,
            self._last_snapshot.right_value,
        )


class RK4Integrator(_LegacyDiffusionIntegrator):
    """Legacy RK4 wrapper for a narrow 1D diffusion problem.

    .. warning::
       This is a legacy compatibility wrapper with a Python time loop.  Prefer
       :func:`biotransport.solve` or ``integrate(method="euler")`` for native
       execution and the complete transport operator.  ``integrate`` requires
       ``method``, so the algorithm is always an explicit choice.

    This integrator uses method of lines: discretize in space first,
    then integrate the resulting ODE system in time using RK4.

    Only a uniform, non-negative diffusivity and Dirichlet conditions at both
    ends are supported.  General transport physics belongs in
    :func:`biotransport.solve`.

    Compared to Forward Euler (1st order), RK4 provides:
    - 4th-order time accuracy (error ~ O(dt⁴))
    - Better accuracy for the same time step
    - Allows larger stable time steps for some problems

    Note: RK4 requires 4 function evaluations per step vs 1 for Euler,
    but the improved accuracy often allows much larger time steps.

    Example:
        >>> mesh = bt.mesh_1d(50, 0.0, 1.0)
        >>> problem = (bt.Problem(mesh).diffusivity(0.01).initial_condition(u0)
        ...            .dirichlet(bt.Boundary.Left, 0.0)
        ...            .dirichlet(bt.Boundary.Right, 0.0))
        >>> integrator = bt.RK4Integrator(problem)
        >>> result = integrator.solve(t_end=1.0, dt=0.01)
    """

    def __init__(
        self,
        problem: TransportProblem,
        *,
        safety_factor: float = 0.5,
    ):
        """Initialize the RK4 integrator.

        Args:
            problem: The transport problem to solve
            safety_factor: Factor applied to CFL-based dt (default 0.5 for RK4)
        """
        self._initialize_legacy_problem(problem, safety_factor, "RK4Integrator")

    def max_stable_dt(self) -> float:
        """Compute the maximum stable time step for RK4.

        For centered 1D diffusion, the most negative semi-discrete eigenvalue
        approaches ``-4D/dx²``.  Classical RK4 is stable on the negative real
        axis through approximately ``-2.7852935634``, so
        ``dt <= 2.7852935634 * dx² / (4D)``.
        """
        return self._stable_dt(self._refresh_snapshot())

    def _stable_dt(self, snapshot: _LegacyDiffusionSnapshot) -> float:
        if snapshot.diffusivity == 0.0:
            return float("inf")
        dx = snapshot.mesh.dx()
        stability_limit = _scaled_positive_ratio(
            (
                self._safety,
                _RK4_NEGATIVE_REAL_AXIS_RADIUS,
                dx,
                dx,
            ),
            (4.0, snapshot.diffusivity),
            conservative=True,
        )
        if stability_limit == 0.0:
            raise FloatingPointError(
                "RK4Integrator stability limit is below binary64 range"
            )
        return stability_limit

    def solve(
        self,
        t_end: float,
        *,
        dt: Optional[float] = None,
        store_history: bool = False,
    ) -> IntegrationResult:
        """Solve the problem to t_end using RK4.

        Args:
            t_end: End time
            dt: Time-step ceiling (uses the stable limit if omitted). A value
                whose actual step would exceed the stability limit is rejected.
            store_history: If True, store solution at each step

        Returns:
            IntegrationResult with final solution and statistics
        """
        import time as time_module

        t_end, dt = _validate_solve_times(t_end, dt)
        requested_dt = dt
        if not isinstance(store_history, bool):
            raise TypeError("store_history must be bool")

        # Revalidate and own every live problem value before stepping.
        snapshot = self._refresh_snapshot()
        dt_max = self._stable_dt(snapshot)
        num_steps, nominal_dt = _resolve_legacy_step(
            t_end,
            dt,
            dt_max,
            "RK4Integrator",
        )
        final_dt = _exact_final_legacy_step(
            t_end,
            nominal_dt,
            num_steps,
            dt_max,
            "RK4Integrator",
        )

        # Initialize
        u = snapshot.initial.copy()
        u[0] = snapshot.left_value
        u[-1] = snapshot.right_value
        t = 0.0
        history = [u.copy()] if store_history else None

        # RHS function for diffusion
        def rhs(u_state: np.ndarray, _t_val: float) -> np.ndarray:
            return _compute_diffusion_rhs(
                snapshot.mesh,
                u_state,
                snapshot.diffusivity,
            )

        # Time integration loop
        start = time_module.perf_counter()

        for step_index in range(num_steps):
            step_dt = final_dt if step_index == num_steps - 1 else nominal_dt
            u = _rk4_autonomous_step(u, rhs, step_dt)

            # Apply boundary conditions
            u[0] = snapshot.left_value
            u[-1] = snapshot.right_value

            t = t_end if step_index == num_steps - 1 else t + step_dt

            if store_history:
                assert history is not None
                history.append(u.copy())

        elapsed = time_module.perf_counter() - start
        t = t_end

        # Build result
        stats: dict[str, object] = {
            "steps": num_steps,
            "dt": nominal_dt,
            "final_dt": final_dt,
            "requested_dt": requested_dt,
            "stability_limit": dt_max,
            "t_end": t,
            "wall_time_s": elapsed,
            "method": "rk4",
        }

        if store_history:
            stats["history"] = history

        return IntegrationResult(
            solution=u,
            time=t,
            stats=stats,
        )


class HeunIntegrator(_LegacyDiffusionIntegrator):
    """Legacy Heun wrapper for a narrow 1D diffusion problem.

    .. warning::
       This is a legacy compatibility wrapper with a Python time loop.  Prefer
       :func:`biotransport.solve` or ``integrate(method="euler")`` for native
       execution and the complete transport operator.  ``integrate`` requires
       ``method``, so the algorithm is always an explicit choice.

    Within its supported uniform-diffusion contract, this offers:
    - 2nd-order time accuracy (error ~ O(dt²))
    - Only 2 function evaluations per step (vs 4 for RK4)

    Reactions, sources, advection, variable diffusivity, multidimensional
    meshes, and non-Dirichlet conditions are rejected.  Use
    :func:`biotransport.solve` for those problems.

    Example:
        >>> integrator = bt.HeunIntegrator(problem)
        >>> result = integrator.solve(t_end=1.0, dt=0.01)
    """

    def __init__(
        self,
        problem: TransportProblem,
        *,
        safety_factor: float = 0.8,
    ):
        """Initialize the Heun integrator.

        Args:
            problem: The transport problem to solve
            safety_factor: Factor applied to CFL-based dt
        """
        self._initialize_legacy_problem(problem, safety_factor, "HeunIntegrator")

    def max_stable_dt(self) -> float:
        """Return the safety-scaled centered-diffusion stability limit.

        Explicit Heun/RK2 reaches ``-2`` on the negative real axis, exactly
        the same limit as Forward Euler for this semi-discrete operator.
        """
        return self._stable_dt(self._refresh_snapshot())

    def _stable_dt(self, snapshot: _LegacyDiffusionSnapshot) -> float:
        if snapshot.diffusivity == 0.0:
            return float("inf")
        dx = snapshot.mesh.dx()
        stability_limit = _scaled_positive_ratio(
            (self._safety, dx, dx),
            (2.0, snapshot.diffusivity),
            conservative=True,
        )
        if stability_limit == 0.0:
            raise FloatingPointError(
                "HeunIntegrator stability limit is below binary64 range"
            )
        return stability_limit

    def solve(
        self,
        t_end: float,
        *,
        dt: Optional[float] = None,
        store_history: bool = False,
    ) -> IntegrationResult:
        """Solve the problem to t_end using Heun's method.

        ``dt`` is a ceiling and is rejected when the resulting actual step
        would exceed the safety-scaled diffusion stability limit.
        """
        import time as time_module

        t_end, dt = _validate_solve_times(t_end, dt)
        requested_dt = dt
        if not isinstance(store_history, bool):
            raise TypeError("store_history must be bool")
        snapshot = self._refresh_snapshot()
        dt_max = self._stable_dt(snapshot)
        num_steps, nominal_dt = _resolve_legacy_step(
            t_end,
            dt,
            dt_max,
            "HeunIntegrator",
        )
        final_dt = _exact_final_legacy_step(
            t_end,
            nominal_dt,
            num_steps,
            dt_max,
            "HeunIntegrator",
        )

        u = snapshot.initial.copy()
        u[0] = snapshot.left_value
        u[-1] = snapshot.right_value
        t = 0.0
        history = [u.copy()] if store_history else None

        def rhs(u_state: np.ndarray, _t_val: float) -> np.ndarray:
            return _compute_diffusion_rhs(
                snapshot.mesh,
                u_state,
                snapshot.diffusivity,
            )

        start = time_module.perf_counter()

        for step_index in range(num_steps):
            step_dt = final_dt if step_index == num_steps - 1 else nominal_dt
            # The supported diffusion RHS is autonomous.  Using a local clock
            # avoids falsely rejecting an exact tiny final remainder after
            # accumulated display-time roundoff.
            u = heun_step(u, rhs, 0.0, step_dt)

            u[0] = snapshot.left_value
            u[-1] = snapshot.right_value

            t = t_end if step_index == num_steps - 1 else t + step_dt

            if store_history:
                assert history is not None
                history.append(u.copy())

        elapsed = time_module.perf_counter() - start
        t = t_end

        stats: dict[str, object] = {
            "steps": num_steps,
            "dt": nominal_dt,
            "final_dt": final_dt,
            "requested_dt": requested_dt,
            "stability_limit": dt_max,
            "t_end": t,
            "wall_time_s": elapsed,
            "method": "heun",
        }

        if store_history:
            stats["history"] = history

        return IntegrationResult(
            solution=u,
            time=t,
            stats=stats,
        )


def integrate(
    problem: TransportProblem,
    t_end: float,
    *,
    method: str,
    dt: Optional[float] = None,
) -> IntegrationResult:
    """Dispatch to canonical Euler or limited legacy Python integrators.

    ``method`` is required so the algorithm is always chosen explicitly.
    ``method="euler"`` uses the canonical C++ transport solver and preserves
    every configured problem term.  ``"heun"`` and ``"rk4"`` are legacy Python
    reference integrators that accept only 1D uniform diffusion with Dirichlet
    conditions at both ends.

    Args:
        problem: The transport problem to solve
        t_end: End time
        method: ``"euler"`` (canonical native), ``"heun"`` or ``"rk4"``
            (legacy Python reference).
        dt: Time step (uses method-specific stable dt if not provided)

    Returns:
        IntegrationResult with solution and statistics

    Example:
        >>> result = bt.integrate(problem, t_end=1.0, method="euler")
        >>> assert result.stats["method"] == "euler"
        >>> print(f"Final solution: {result.solution}")
    """
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    normalized_method = method.strip().lower()
    if normalized_method not in {"euler", "heun", "rk4"}:
        raise ValueError(
            f"Unknown integration method: {method}. Choose from: 'euler', 'heun', 'rk4'"
        )

    if normalized_method == "euler":
        import time as time_module

        from .run import solve

        t_end, dt = _validate_solve_times(t_end, dt)
        start = time_module.perf_counter()
        run_result = solve(
            problem,
            end_time=t_end,
            time_step=dt,
            method="explicit",
        )
        elapsed = time_module.perf_counter() - start
        diagnostics = run_result.diagnostics
        return IntegrationResult(
            solution=np.asarray(run_result.concentration, dtype=float).copy(),
            time=float(run_result.time),
            stats={
                "steps": diagnostics.steps,
                "dt": diagnostics.maximum_time_step,
                "dt_min": diagnostics.minimum_time_step,
                "t_end": run_result.time,
                "wall_time_s": elapsed,
                "method": "euler",
                "diagnostics": diagnostics,
            },
        )

    if normalized_method == "heun":
        heun_integrator = HeunIntegrator(problem)
        return heun_integrator.solve(t_end, dt=dt)

    rk4_integrator = RK4Integrator(problem)
    return rk4_integrator.solve(t_end, dt=dt)
