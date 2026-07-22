"""Validated high-order finite differences and explicit time integration.

The expensive spatial loops and diffusion time loop in this module run in C++.
The stated fourth- and sixth-order accuracies apply where the full centered
stencil fits.  Boundary nodes have no derivative value and nearby nodes use a
second-order closure (with a fourth-order transition for the sixth-order
stencil).  Consequently, a full boundary-value problem is not automatically
globally fourth- or sixth-order accurate.

``HighOrderDiffusionSolver`` uses Forward Euler in time.  Its temporal order is
one even when a higher-order spatial stencil is selected.  The separate Heun
and classical RK4 adapter is intended for carefully validated Python ODE
callbacks; callback execution still crosses the Python GIL at every stage, so
it is a correctness convenience rather than a callback-acceleration claim.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, cast

import numpy as np

from ._core import Boundary, StructuredMesh
from ._core._core import (
    _high_order_gradient_1d,
    _high_order_laplacian_1d,
    _high_order_laplacian_2d,
    _high_order_stable_dt,
    _integrate_explicit_runge_kutta,
    _solve_high_order_diffusion,
)


_NUMERIC_KINDS = frozenset("iuf")


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes, complex, np.complexfloating)):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_scalar(value: object, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _order(value: object, allowed: Tuple[int, ...], name: str = "order") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        choices = ", ".join(str(item) for item in allowed)
        raise ValueError(f"{name} must be one of {choices}")
    result = int(value)
    if result not in allowed:
        choices = ", ".join(str(item) for item in allowed)
        raise ValueError(f"{name} must be one of {choices}")
    return result


def _field_array(
    value: object,
    name: str = "field",
    allowed_dimensions: Optional[Tuple[int, ...]] = (1, 2),
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in _NUMERIC_KINDS:
        raise TypeError(f"{name} must contain real numeric values")
    if allowed_dimensions is None and raw.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if allowed_dimensions is not None and raw.ndim not in allowed_dimensions:
        dimensions = " or ".join(f"{dimension}D" for dimension in allowed_dimensions)
        raise ValueError(f"{name} must be a {dimensions} array")
    if raw.size == 0:
        raise ValueError(f"{name} must not be empty")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values only")
    return result


def _validate_mesh(mesh: object) -> StructuredMesh:
    if not isinstance(mesh, StructuredMesh):
        raise TypeError("mesh must be a StructuredMesh")
    return mesh


def _spacing_matches(actual: float, supplied: float, name: str) -> None:
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(actual), abs(supplied))
    if abs(actual - supplied) > tolerance:
        raise ValueError(
            f"{name}={supplied!r} does not match the mesh spacing {actual!r}"
        )


def _operator_grid(
    field_or_mesh: object,
    spacing_or_field: object,
    dy: Optional[float],
    mesh: Optional[StructuredMesh],
    operation: str,
) -> Tuple[np.ndarray, int, int, float, float]:
    """Resolve preferred ``(field, dx)`` and legacy ``(mesh, field)`` calls."""
    mesh_object: Optional[StructuredMesh]
    legacy_mesh = isinstance(field_or_mesh, StructuredMesh)
    if legacy_mesh:
        if mesh is not None:
            raise TypeError(f"{operation} received mesh twice")
        mesh_object = _validate_mesh(field_or_mesh)
        if spacing_or_field is None:
            raise TypeError(f"{operation}(mesh, field) is missing field")
        field = _field_array(spacing_or_field)
        if dy is not None:
            raise TypeError("dy is inferred from mesh in the legacy call form")
        supplied_dx = None
    else:
        field = _field_array(field_or_mesh)
        mesh_object = _validate_mesh(mesh) if mesh is not None else None
        supplied_dx = (
            None
            if spacing_or_field is None
            else _positive_scalar(spacing_or_field, "dx")
        )

    if mesh_object is None:
        if supplied_dx is None:
            raise TypeError(f"{operation}(field, dx) is missing dx")
        dx = supplied_dx
        if field.ndim == 1:
            if dy is not None:
                raise ValueError("dy is only valid for a two-dimensional field")
            return field, field.size - 1, 0, dx, dx
        resolved_dy = dx if dy is None else _positive_scalar(dy, "dy")
        ny, nx = field.shape[0] - 1, field.shape[1] - 1
        return field, nx, ny, dx, resolved_dy

    dx = _positive_scalar(mesh_object.dx(), "mesh.dx()")
    if supplied_dx is not None:
        _spacing_matches(dx, supplied_dx, "dx")

    nx = int(mesh_object.nx())
    if mesh_object.is_1d():
        if field.ndim != 1 or field.size != nx + 1:
            raise ValueError(f"field must have shape ({nx + 1},) for this 1D mesh")
        if dy is not None:
            raise ValueError("dy is only valid for a two-dimensional mesh")
        return field, nx, 0, dx, dx

    ny = int(mesh_object.ny())
    expected_shape = (ny + 1, nx + 1)
    if field.ndim == 2 and field.shape != expected_shape:
        raise ValueError(f"field must have shape {expected_shape} for this 2D mesh")
    if field.ndim == 1 and field.size != (nx + 1) * (ny + 1):
        raise ValueError("flat field size must equal (mesh.nx() + 1) * (mesh.ny() + 1)")
    resolved_dy = _positive_scalar(mesh_object.dy(), "mesh.dy()")
    if dy is not None:
        supplied_dy = _positive_scalar(dy, "dy")
        _spacing_matches(resolved_dy, supplied_dy, "dy")
    return field, nx, ny, dx, resolved_dy


def _laplacian(
    field_or_mesh: object,
    spacing_or_field: object,
    order: int,
    dy: Optional[float] = None,
    mesh: Optional[StructuredMesh] = None,
) -> np.ndarray:
    field, nx, ny, dx, resolved_dy = _operator_grid(
        field_or_mesh,
        spacing_or_field,
        dy,
        mesh,
        f"laplacian_{order}th_order",
    )
    original_shape = field.shape
    flat = field.reshape(-1)
    if ny == 0:
        result = _high_order_laplacian_1d(flat, dx, order)
    else:
        if order == 6:
            raise ValueError("sixth-order Laplacian is available for 1D fields only")
        result = _high_order_laplacian_2d(flat, nx, ny, dx, resolved_dy, order)
    return np.asarray(result, dtype=np.float64).reshape(original_shape)


def laplacian_2nd_order(
    field_or_mesh: object,
    spacing_or_field: object = None,
    dy: Optional[float] = None,
    *,
    mesh: Optional[StructuredMesh] = None,
) -> np.ndarray:
    """Return the second-order centered Laplacian.

    The preferred forms are ``laplacian_2nd_order(field, dx)`` for 1D and
    ``laplacian_2nd_order(field_2d, dx, dy)`` for 2D.  Supplying
    ``mesh=mesh`` validates both dimensions and spacing.  The historical
    ``laplacian_2nd_order(mesh, field)`` form is also accepted.

    Boundary entries are zero because no derivative is inferred outside the
    domain.
    """
    return _laplacian(field_or_mesh, spacing_or_field, 2, dy, mesh)


def laplacian_4th_order(
    field_or_mesh: object,
    spacing_or_field: object = None,
    dy: Optional[float] = None,
    *,
    mesh: Optional[StructuredMesh] = None,
) -> np.ndarray:
    """Return a fourth-order centered Laplacian in the stencil interior.

    Prefer ``laplacian_4th_order(field, dx[, dy])``.  The legacy
    ``laplacian_4th_order(mesh, field)`` call remains supported.  A
    second-order closure is used next to the boundary, so fourth order is an
    interior claim, not an unconditional global convergence claim.
    """
    return _laplacian(field_or_mesh, spacing_or_field, 4, dy, mesh)


def laplacian_6th_order(
    field_or_mesh: object,
    spacing_or_field: object = None,
    *,
    mesh: Optional[StructuredMesh] = None,
) -> np.ndarray:
    """Return a sixth-order centered 1D Laplacian in the stencil interior.

    Prefer ``laplacian_6th_order(field, dx)``.  The legacy
    ``laplacian_6th_order(mesh, field)`` call remains supported.  The two
    transition layers use fourth- and second-order closures.
    """
    return _laplacian(field_or_mesh, spacing_or_field, 6, None, mesh)


def gradient_4th_order(
    field_or_mesh: object,
    spacing_or_field: object = None,
    *,
    mesh: Optional[StructuredMesh] = None,
) -> np.ndarray:
    """Return a fourth-order centered 1D first derivative in the interior.

    Prefer ``gradient_4th_order(field, dx)``.  The legacy
    ``gradient_4th_order(mesh, field)`` form remains supported.  Boundary
    entries are zero and boundary-adjacent entries use a second-order closure.
    """
    field, nx, ny, dx, _ = _operator_grid(
        field_or_mesh,
        spacing_or_field,
        None,
        mesh,
        "gradient_4th_order",
    )
    if ny != 0 or field.ndim != 1:
        raise ValueError("gradient_4th_order requires a one-dimensional field")
    if field.size != nx + 1:
        raise ValueError("field size does not match the one-dimensional grid")
    return np.asarray(_high_order_gradient_1d(field, dx, 4), dtype=np.float64)


@dataclass(frozen=True)
class HighOrderResult:
    """Result from the native explicit diffusion solver.

    ``order`` is the formal centered-stencil interior order,
    ``boundary_order`` describes the lowest-order near-boundary closure, and
    ``temporal_order`` is the Forward Euler time order.  ``dt`` is the nominal
    step while ``last_dt`` is the possibly shortened final step.
    """

    solution: np.ndarray
    time: float
    steps: int
    dt: float
    last_dt: float
    order: int
    boundary_order: int = 2
    temporal_order: int = 1

    @property
    def interior_order(self) -> int:
        """Alias spelling out what ``order`` represents."""
        return self.order


class HighOrderDiffusionSolver:
    """Native explicit solver for ``du/dt = D * Laplacian(u)``.

    The solution loop runs in C++ and ends exactly at ``t_end`` by shortening
    the final step.  Spatial orders 2, 4, and 6 are available in 1D; 2 and 4
    are available in 2D.  All boundaries are constant Dirichlet values.

    The method is first-order Forward Euler in time.  Selecting a fourth- or
    sixth-order stencil does not change that temporal order, and the
    near-boundary closure remains second order.
    """

    def __init__(
        self,
        mesh: StructuredMesh,
        D: float,
        order: int = 4,
        safety_factor: float = 0.4,
    ) -> None:
        self.mesh = _validate_mesh(mesh)
        self.D = _positive_scalar(D, "D")
        self.order = _order(order, (2, 4, 6))
        self.safety_factor = _finite_scalar(safety_factor, "safety_factor")
        if not 0.0 < self.safety_factor <= 1.0:
            raise ValueError("safety_factor must be in (0, 1]")

        self.nx = int(self.mesh.nx())
        self.dx = _positive_scalar(self.mesh.dx(), "mesh.dx()")
        self.is_1d = bool(self.mesh.is_1d())
        if self.is_1d:
            self.ny = 0
            self.dy = self.dx
        else:
            self.ny = int(self.mesh.ny())
            self.dy = _positive_scalar(self.mesh.dy(), "mesh.dy()")
            if self.order == 6:
                raise ValueError(
                    "sixth-order diffusion is available for 1D meshes only"
                )

        if self.nx < self.order or (not self.is_1d and self.ny < self.order):
            raise ValueError("mesh is too small for the requested centered stencil")

        self.bc_left = 0.0
        self.bc_right = 0.0
        self.bc_bottom = 0.0
        self.bc_top = 0.0

    def set_boundary(
        self, boundary: Boundary, value: float
    ) -> "HighOrderDiffusionSolver":
        """Set one constant Dirichlet boundary and return ``self``.

        In 2D, bottom/top values own the four corners when adjacent boundary
        values disagree.  Bottom and top are invalid for a 1D mesh.
        """
        resolved = _finite_scalar(value, "boundary value")
        if boundary == Boundary.Left:
            self.bc_left = resolved
        elif boundary == Boundary.Right:
            self.bc_right = resolved
        elif boundary == Boundary.Bottom:
            if self.is_1d:
                raise ValueError("Bottom is not a boundary of a 1D mesh")
            self.bc_bottom = resolved
        elif boundary == Boundary.Top:
            if self.is_1d:
                raise ValueError("Top is not a boundary of a 1D mesh")
            self.bc_top = resolved
        else:
            raise ValueError("boundary must be Left, Right, Bottom, or Top")
        return self

    def compute_stable_dt(self) -> float:
        """Return the safety-scaled Forward Euler diffusion limit.

        The exact centered-stencil spectral radii are used: 4, 16/3, and
        272/45 for spatial orders 2, 4, and 6 respectively.
        """
        return float(
            _high_order_stable_dt(
                self.D,
                self.dx,
                self.dy,
                self.order,
                self.safety_factor,
                not self.is_1d,
            )
        )

    def solve(
        self,
        initial: object,
        t_end: float,
        dt: Optional[float] = None,
        callback: Optional[Callable[[float, np.ndarray], None]] = None,
    ) -> HighOrderResult:
        """Advance the diffusion equation from time zero to ``t_end``.

        ``dt=None`` uses :meth:`compute_stable_dt`.  A supplied step larger
        than that safety-scaled bound is rejected.  ``callback(time, state)``
        receives an isolated copy after every accepted step; mutating it cannot
        alter the native solution.  Because a Python callback reacquires the
        GIL once per step, omit it for maximum throughput.
        """
        end_time = _finite_scalar(t_end, "t_end")
        if end_time < 0.0:
            raise ValueError("t_end must be nonnegative")
        requested_dt = None if dt is None else _positive_scalar(dt, "dt")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")

        field = _field_array(initial)
        expected_shape: Tuple[int, ...]
        output_shape: Tuple[int, ...]
        callback_shape: Tuple[int, ...]
        if self.is_1d:
            expected_shape = (self.nx + 1,)
            if field.shape != expected_shape:
                raise ValueError(f"initial must have shape {expected_shape}")
            output_shape = expected_shape
            callback_shape = expected_shape
        else:
            expected_shape = (self.ny + 1, self.nx + 1)
            expected_size = expected_shape[0] * expected_shape[1]
            if field.ndim == 2 and field.shape != expected_shape:
                raise ValueError(f"initial must have shape {expected_shape}")
            if field.ndim == 1 and field.size != expected_size:
                raise ValueError(
                    "flat initial size must equal (mesh.nx() + 1) * (mesh.ny() + 1)"
                )
            output_shape = field.shape
            callback_shape = expected_shape

        native_callback = None
        if callback is not None:

            def native_callback(time: float, state: np.ndarray) -> None:
                callback(float(time), np.asarray(state).reshape(callback_shape))

        native_result = _solve_high_order_diffusion(
            field.reshape(-1),
            self.nx,
            self.ny,
            self.dx,
            self.dy,
            self.D,
            self.order,
            self.safety_factor,
            end_time,
            requested_dt,
            self.bc_left,
            self.bc_right,
            self.bc_bottom,
            self.bc_top,
            native_callback,
        )
        solution = np.asarray(native_result["solution"], dtype=np.float64).reshape(
            output_shape
        )
        return HighOrderResult(
            solution=solution,
            time=float(native_result["time"]),
            steps=int(native_result["steps"]),
            dt=float(native_result["dt"]),
            last_dt=float(native_result["last_dt"]),
            order=int(native_result["order"]),
            boundary_order=int(native_result["boundary_order"]),
        )


def d2dx2(
    u: object,
    dx: float,
    order: int = 4,
    mesh: Optional[StructuredMesh] = None,
) -> np.ndarray:
    """Return the 1D second derivative with the selected interior order."""
    field = _field_array(u, "u", (1,))
    resolved_order = _order(order, (2, 4, 6))
    if mesh is not None and not _validate_mesh(mesh).is_1d():
        raise ValueError("d2dx2 requires a one-dimensional mesh")
    if resolved_order == 2:
        return laplacian_2nd_order(field, dx, mesh=mesh)
    if resolved_order == 4:
        return laplacian_4th_order(field, dx, mesh=mesh)
    return laplacian_6th_order(field, dx, mesh=mesh)


def ddx(
    u: object,
    dx: float,
    order: int = 4,
    mesh: Optional[StructuredMesh] = None,
) -> np.ndarray:
    """Return the 1D first derivative with interior order two or four."""
    field = _field_array(u, "u", (1,))
    resolved_order = _order(order, (2, 4))
    if mesh is not None and not _validate_mesh(mesh).is_1d():
        raise ValueError("ddx requires a one-dimensional mesh")
    resolved_dx = _positive_scalar(dx, "dx")
    if mesh is not None:
        mesh_dx = _positive_scalar(mesh.dx(), "mesh.dx()")
        _spacing_matches(mesh_dx, resolved_dx, "dx")
        if field.size != int(mesh.nx()) + 1:
            raise ValueError("u size does not match the one-dimensional mesh")
    return np.asarray(
        _high_order_gradient_1d(field, resolved_dx, resolved_order),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class RungeKuttaResult:
    """Result from the validated explicit Runge--Kutta adapter."""

    solution: np.ndarray
    initial_time: float
    time: float
    steps: int
    dt: float
    last_dt: float
    method: str
    order: int


def _canonical_runge_kutta_method(method: object) -> str:
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    normalized = method.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "heun": "heun",
        "rk2": "heun",
        "improved_euler": "heun",
        "rk4": "rk4",
        "classical_rk4": "rk4",
        "classic_rk4": "rk4",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError("method must be 'heun'/'rk2' or 'rk4'") from exc


def integrate_explicit_runge_kutta(
    initial: object,
    rhs: Callable[..., object],
    t_end: float,
    dt: float,
    *,
    t_start: float = 0.0,
    method: str = "rk4",
    autonomous: bool = False,
    maximum_steps: int = 10_000_000,
) -> RungeKuttaResult:
    """Integrate a non-stiff ODE with validated Heun or classical RK4.

    With the default ``autonomous=False``, ``rhs(state, time)`` is called at
    the mathematically correct stage times.  Set ``autonomous=True`` to call
    ``rhs(state)`` explicitly.  The callback must return the same shape as the
    initial state and finite values.  Stage arrays never alias the accepted
    state, the caller's initial array is not mutated, and the final step is
    shortened to end exactly at ``t_end``.

    The stages and vector updates are orchestrated in C++, but a Python
    callback still executes under the GIL two times per Heun step or four times
    per RK4 step.  Use a fully native model solver when callback overhead is a
    performance concern.
    """
    if not callable(rhs):
        raise TypeError("rhs must be callable")
    if not isinstance(autonomous, bool):
        raise TypeError("autonomous must be bool")
    if (
        isinstance(maximum_steps, bool)
        or not isinstance(maximum_steps, Integral)
        or int(maximum_steps) <= 0
    ):
        raise ValueError("maximum_steps must be a positive integer")

    state = _field_array(initial, "initial state", None)
    shape = state.shape
    initial_time = _finite_scalar(t_start, "t_start")
    end_time = _finite_scalar(t_end, "t_end")
    if end_time < initial_time:
        raise ValueError("t_end must not precede t_start")
    step = _positive_scalar(dt, "dt")
    canonical_method = _canonical_runge_kutta_method(method)

    def checked_rhs(flat_state: np.ndarray, *time: float) -> np.ndarray:
        shaped_state = np.asarray(flat_state, dtype=np.float64).reshape(shape)
        value = rhs(shaped_state) if autonomous else rhs(shaped_state, float(time[0]))
        derivative = _field_array(value, "rhs result", (len(shape),))
        if derivative.shape != shape:
            raise ValueError(
                f"rhs result shape {derivative.shape} does not match state shape {shape}"
            )
        return derivative.reshape(-1).copy()

    native_result = _integrate_explicit_runge_kutta(
        state.reshape(-1),
        checked_rhs,
        initial_time,
        end_time,
        step,
        canonical_method,
        autonomous,
        int(maximum_steps),
    )
    return RungeKuttaResult(
        solution=np.asarray(native_result["solution"], dtype=np.float64).reshape(shape),
        initial_time=float(native_result["initial_time"]),
        time=float(native_result["time"]),
        steps=int(native_result["steps"]),
        dt=float(native_result["dt"]),
        last_dt=float(native_result["last_dt"]),
        method=canonical_method,
        order=2 if canonical_method == "heun" else 4,
    )


def verify_order_of_accuracy(
    operator_factory: Callable[[int], Callable[[np.ndarray], np.ndarray]],
    field: Callable[[np.ndarray], object],
    exact_derivative: Callable[[np.ndarray], object],
    x_range: Tuple[float, float] = (0.0, 1.0),
    grid_sizes: Sequence[int] = (20, 40, 80, 160),
    interior_margin: int = 3,
) -> Dict[str, object]:
    """Measure spatial convergence against an independently known derivative.

    ``field(x)`` supplies the function sampled on each mesh and
    ``exact_derivative(x)`` supplies the exact quantity returned by the
    operator.  Requiring the exact derivative avoids circular verification by
    a second finite-difference approximation.  The infinity norm is measured
    after excluding ``interior_margin`` nodes at each boundary.
    """
    if not callable(operator_factory):
        raise TypeError("operator_factory must be callable")
    if not callable(field):
        raise TypeError("field must be callable")
    if not callable(exact_derivative):
        raise TypeError("exact_derivative must be callable")
    if not isinstance(interior_margin, Integral) or isinstance(interior_margin, bool):
        raise ValueError("interior_margin must be a nonnegative integer")
    margin = int(interior_margin)
    if margin < 0:
        raise ValueError("interior_margin must be a nonnegative integer")

    if len(x_range) != 2:
        raise ValueError("x_range must contain exactly two bounds")
    x_min = _finite_scalar(x_range[0], "x_range[0]")
    x_max = _finite_scalar(x_range[1], "x_range[1]")
    if x_max <= x_min:
        raise ValueError("x_range upper bound must exceed its lower bound")

    sizes = list(grid_sizes)
    if len(sizes) < 2:
        raise ValueError("grid_sizes must contain at least two grids")
    validated_sizes = []
    previous = 0
    for value in sizes:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("grid_sizes must contain integers")
        cells = int(value)
        if cells <= previous:
            raise ValueError("grid_sizes must be strictly increasing")
        if cells + 1 <= 2 * margin:
            raise ValueError("a grid is too small for the requested interior_margin")
        validated_sizes.append(cells)
        previous = cells

    errors = []
    spacings = []
    for cells in validated_sizes:
        x = np.linspace(x_min, x_max, cells + 1)
        spacing = (x_max - x_min) / cells
        sampled_field = _field_array(field(x), "field(x)", (1,))
        exact = _field_array(exact_derivative(x), "exact_derivative(x)", (1,))
        if sampled_field.shape != x.shape or exact.shape != x.shape:
            raise ValueError("field and exact_derivative must preserve the grid shape")

        operator = operator_factory(cells)
        if not callable(operator):
            raise TypeError("operator_factory must return a callable")
        numerical = _field_array(
            operator(sampled_field.copy()), "operator result", (1,)
        )
        if numerical.shape != x.shape:
            raise ValueError("operator result must preserve the grid shape")

        interior = slice(margin, -margin) if margin else slice(None)
        errors.append(float(np.max(np.abs(numerical[interior] - exact[interior]))))
        spacings.append(float(spacing))

    observed_orders = []
    for coarse_error, fine_error, coarse_dx, fine_dx in zip(
        errors[:-1], errors[1:], spacings[:-1], spacings[1:]
    ):
        if coarse_error == 0.0 and fine_error == 0.0:
            observed_orders.append(float("nan"))
        elif fine_error == 0.0:
            observed_orders.append(float("inf"))
        elif coarse_error == 0.0:
            observed_orders.append(float("-inf"))
        else:
            observed_orders.append(
                float(np.log(coarse_error / fine_error) / np.log(coarse_dx / fine_dx))
            )

    return {
        "grid_sizes": validated_sizes,
        "dx": spacings,
        "errors": errors,
        "observed_orders": observed_orders,
        "norm": "L_inf",
        "interior_margin": margin,
    }


__all__ = [
    "HighOrderDiffusionSolver",
    "HighOrderResult",
    "RungeKuttaResult",
    "d2dx2",
    "ddx",
    "gradient_4th_order",
    "integrate_explicit_runge_kutta",
    "laplacian_2nd_order",
    "laplacian_4th_order",
    "laplacian_6th_order",
    "verify_order_of_accuracy",
]
