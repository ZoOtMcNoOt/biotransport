"""Exact solutions to compare your numbers against.

This is the module you reach for when the question is "did I get this right?".
Everything here is a closed-form or series solution, evaluated in Python, with no
solver involved -- so agreement between one of these and a
:func:`biotransport.solve` result is real evidence that both are right.

Two things make these easy to use:

* **They take arrays.** Pass a whole coordinate vector and get a whole field
  back. The underlying C++ helpers are scalar-only; the wrappers here broadcast
  for you, and still return a plain float when you hand them plain floats.
* **They plug straight into** :meth:`biotransport.Solution.compare`::

      sol = bt.solve(problem, end_time=t)
      print(sol.compare(lambda x: bt.analytical.slab(x, t, D=D, L=L, c_surface=1.0)))

The finite-domain solutions are truncated series. They converge fast at moderate
and long times and slowly at very short times, so each one takes a ``terms``
argument and warns when the truncation is not trustworthy at the time you asked
for.

Sign and geometry conventions are stated per function. Every one of them uses a
single consistent unit system -- whatever you feed in.
"""

from __future__ import annotations

import math
import re
from typing import Any
import warnings

import numpy as np

from ._core import analytical as _native

__all__ = [
    # Transient, finite domains (the textbook series solutions)
    "slab",
    "sphere",
    "cylinder",
    # Transient, unbounded
    "semi_infinite",
    "instantaneous_source",
    # Steady, with reaction
    "steady_slab_first_order",
    "steady_radial_first_order",
    "thiele_modulus",
    "effectiveness_factor",
]


# ---------------------------------------------------------------------------
# Re-export every native helper, but broadcasting over arrays
# ---------------------------------------------------------------------------


def _broadcasting(function: Any, name: str) -> Any:
    """Wrap a scalar C++ helper so it also accepts arrays.

    Returns a plain float for all-scalar input, so existing scalar calls behave
    exactly as they did before.
    """

    vectorized = np.vectorize(function, otypes=[np.float64])

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError(
                f"{name}() takes positional arguments only; pass them in the order "
                f"given in its docstring"
            )
        if all(np.ndim(argument) == 0 for argument in args):
            return float(function(*args))
        return vectorized(*args)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    original = getattr(function, "__doc__", None) or ""
    # The native docstrings use maths notation like |G*| for a magnitude, which
    # reStructuredText reads as a substitution reference and then fails to
    # resolve. Render those as inline literals so the docs build cleanly.
    original = re.sub(r"\|([^|\s][^|]*?)\|", r"``|\1|``", original)
    wrapper.__doc__ = (
        f"{original}\n\nAccepts scalars or NumPy arrays; arrays broadcast together."
    ).strip()
    return wrapper


_native_names = [name for name in dir(_native) if not name.startswith("_")]
for _name in _native_names:
    globals()[_name] = _broadcasting(getattr(_native, _name), _name)
__all__ = sorted(set(__all__) | set(_native_names))
del _name, _native_names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive, finite number, got {value!r}")
    return number


def _check_series_convergence(fourier: float, terms: int, where: str) -> None:
    """Warn when a truncated series cannot resolve the time requested.

    Term ``n`` decays like ``exp(-n^2 pi^2 Fo)``. When ``Fo`` is very small the
    high terms are still alive, so a truncated sum is simply wrong -- most
    visibly as ripples near a sharp initial front.
    """

    if fourier <= 0.0:
        return
    smallest = math.exp(-(terms**2) * math.pi**2 * fourier)
    if smallest > 1.0e-6:
        needed = max(
            terms + 1,
            int(math.ceil(math.sqrt(math.log(1.0e6) / (math.pi**2 * fourier)))),
        )
        warnings.warn(
            f"{where}: Fourier number {fourier:.3g} is small, so {terms} series terms "
            f"leave the last term at {smallest:.2g} of the first and the result will "
            f"ripple. Pass terms={needed} or compare at a later time.",
            RuntimeWarning,
            stacklevel=3,
        )


def _as_array(values: Any, name: str) -> tuple[np.ndarray, bool]:
    scalar = np.ndim(values) == 0
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return np.atleast_1d(array), scalar


def _restore(result: np.ndarray, scalar: bool) -> Any:
    return float(result[0]) if scalar else result


# ---------------------------------------------------------------------------
# Transient solutions on finite domains
# ---------------------------------------------------------------------------


def slab(
    x: Any,
    t: float,
    *,
    D: float,
    L: float,
    c_surface: float,
    c_initial: float = 0.0,
    terms: int = 100,
) -> Any:
    """Transient diffusion in a slab held at a fixed value on both faces.

    Solves ``dc/dt = D d2c/dx2`` on ``0 <= x <= L``, starting uniform at
    ``c_initial``, with both faces held at ``c_surface`` from ``t = 0``. This is
    the standard separation-of-variables result::

        (c - c_s)/(c_0 - c_s) = sum_{n odd} (4/(n pi)) sin(n pi x / L) exp(-n^2 pi^2 Fo)

    with ``Fo = D t / L^2``.

    A slab that is sealed on one face and exposed on the other is the same
    problem of twice the thickness: model thickness ``a`` sealed at ``x = a`` by
    calling this with ``L = 2a`` and reading ``0 <= x <= a``.

    Args:
        x: Position, or an array of positions, in ``[0, L]``.
        t: Time. ``t = 0`` returns the initial condition.
        D: Diffusivity.
        L: Slab thickness.
        c_surface: Value held at both faces.
        c_initial: Uniform starting value.
        terms: Number of series terms.

    Returns:
        Concentration at ``x``; an array if ``x`` was an array.

    Example:
        >>> mesh = bt.mesh_1d(200, 0.0, 0.01)
        >>> problem = (bt.Problem(mesh).diffusivity(1e-9).initial(0.0)
        ...            .dirichlet("left", 1.0).dirichlet("right", 1.0))
        >>> sol = bt.solve(problem, end_time=20.0)
        >>> print(sol.compare(lambda x: bt.analytical.slab(
        ...     x, 20.0, D=1e-9, L=0.01, c_surface=1.0)))
    """

    positions, scalar = _as_array(x, "x")
    diffusivity = _positive(D, "D")
    thickness = _positive(L, "L")
    duration = float(t)
    if duration < 0.0:
        raise ValueError("t must be non-negative")
    if terms < 1:
        raise ValueError("terms must be at least 1")

    if duration == 0.0:
        return _restore(np.full(positions.shape, float(c_initial)), scalar)

    fourier = diffusivity * duration / (thickness * thickness)
    _check_series_convergence(fourier, terms, "analytical.slab")

    odd = 2.0 * np.arange(terms) + 1.0  # 1, 3, 5, ...
    theta = np.zeros(positions.shape, dtype=np.float64)
    for n in odd:
        theta += (
            (4.0 / (n * math.pi))
            * np.sin(n * math.pi * positions / thickness)
            * math.exp(-(n**2) * math.pi**2 * fourier)
        )
    result = float(c_surface) + (float(c_initial) - float(c_surface)) * theta
    return _restore(result, scalar)


def sphere(
    r: Any,
    t: float,
    *,
    D: float,
    R: float,
    c_surface: float,
    c_initial: float = 0.0,
    terms: int = 100,
) -> Any:
    """Transient diffusion in a sphere with its surface held at a fixed value.

    Crank's series for ``0 <= r <= R``, uniform ``c_initial`` at ``t = 0`` and
    ``c = c_surface`` on the surface::

        (c - c_s)/(c_0 - c_s) = (2R/(pi r)) sum_n ((-1)^(n+1)/n)
                                sin(n pi r / R) exp(-n^2 pi^2 D t / R^2)

    The ``r = 0`` centre is evaluated from its limit, so passing a mesh that
    includes the origin is fine.

    This is the shape for a spherical cell, a tumour spheroid, or a drug-loaded
    microsphere.

    Args:
        r: Radius, or an array of radii, in ``[0, R]``.
        t: Time.
        D: Diffusivity.
        R: Sphere radius.
        c_surface: Value held at the surface.
        c_initial: Uniform starting value.
        terms: Number of series terms.
    """

    radii, scalar = _as_array(r, "r")
    diffusivity = _positive(D, "D")
    outer = _positive(R, "R")
    duration = float(t)
    if duration < 0.0:
        raise ValueError("t must be non-negative")
    if np.any(radii < 0.0) or np.any(radii > outer * (1.0 + 1.0e-12)):
        raise ValueError(f"r must lie in [0, R] with R = {outer:g}")

    if duration == 0.0:
        return _restore(np.full(radii.shape, float(c_initial)), scalar)

    fourier = diffusivity * duration / (outer * outer)
    _check_series_convergence(fourier, terms, "analytical.sphere")

    theta = np.zeros(radii.shape, dtype=np.float64)
    at_centre = radii <= outer * 1.0e-12
    inner = ~at_centre
    for index in range(1, terms + 1):
        decay = math.exp(-(index**2) * math.pi**2 * fourier)
        sign = -1.0 if index % 2 == 0 else 1.0
        if np.any(inner):
            theta[inner] += (
                (2.0 * outer / (math.pi * radii[inner]))
                * (sign / index)
                * np.sin(index * math.pi * radii[inner] / outer)
                * decay
            )
        if np.any(at_centre):
            # lim r->0 of sin(n pi r/R)/r is n pi / R
            theta[at_centre] += 2.0 * sign * decay

    result = float(c_surface) + (float(c_initial) - float(c_surface)) * theta
    return _restore(result, scalar)


def cylinder(
    r: Any,
    t: float,
    *,
    D: float,
    R: float,
    c_surface: float,
    c_initial: float = 0.0,
    terms: int = 60,
) -> Any:
    """Transient radial diffusion in a long cylinder held at a fixed surface value.

    Bessel series for ``0 <= r <= R``, uniform ``c_initial`` at ``t = 0``::

        (c - c_s)/(c_0 - c_s) = 2 sum_n [J0(a_n r/R) / (a_n J1(a_n))]
                                exp(-a_n^2 D t / R^2)

    where ``a_n`` is the n-th zero of ``J0``. This is the radial part of the
    Krogh tissue cylinder and of any long fibre or vessel problem.

    Args:
        r: Radius, or an array of radii, in ``[0, R]``.
        t: Time.
        D: Diffusivity.
        R: Cylinder radius.
        c_surface: Value held at the surface.
        c_initial: Uniform starting value.
        terms: Number of Bessel terms.
    """

    from scipy.special import j0, j1, jn_zeros

    radii, scalar = _as_array(r, "r")
    diffusivity = _positive(D, "D")
    outer = _positive(R, "R")
    duration = float(t)
    if duration < 0.0:
        raise ValueError("t must be non-negative")
    if np.any(radii < 0.0) or np.any(radii > outer * (1.0 + 1.0e-12)):
        raise ValueError(f"r must lie in [0, R] with R = {outer:g}")

    if duration == 0.0:
        return _restore(np.full(radii.shape, float(c_initial)), scalar)

    fourier = diffusivity * duration / (outer * outer)
    roots = jn_zeros(0, int(terms))
    smallest = math.exp(-(roots[-1] ** 2) * fourier)
    if smallest > 1.0e-6:
        warnings.warn(
            f"analytical.cylinder: Fourier number {fourier:.3g} is small, so "
            f"{terms} Bessel terms leave the last term at {smallest:.2g} and the "
            f"result will ripple. Raise terms or compare at a later time.",
            RuntimeWarning,
            stacklevel=2,
        )

    theta = np.zeros(radii.shape, dtype=np.float64)
    for root in roots:
        theta += (
            2.0
            * j0(root * radii / outer)
            / (root * j1(root))
            * math.exp(-(root**2) * fourier)
        )

    result = float(c_surface) + (float(c_initial) - float(c_surface)) * theta
    return _restore(result, scalar)


# ---------------------------------------------------------------------------
# Transient solutions on unbounded domains
# ---------------------------------------------------------------------------


def semi_infinite(
    x: Any,
    t: float,
    *,
    D: float,
    c_surface: float,
    c_initial: float = 0.0,
) -> Any:
    """Diffusion into a half-space whose surface is held at a fixed value.

    ``c = c_s + (c_0 - c_s) erf(x / (2 sqrt(D t)))``.

    Valid while the penetration depth stays well inside your domain -- roughly
    ``4 sqrt(D t) < L``. Past that, use :func:`slab` instead, because the far
    boundary has started to matter.

    Args:
        x: Depth from the surface, or an array of depths.
        t: Time.
        D: Diffusivity.
        c_surface: Value held at the surface.
        c_initial: Uniform starting value.
    """

    from scipy.special import erf

    depths, scalar = _as_array(x, "x")
    diffusivity = _positive(D, "D")
    duration = float(t)
    if duration < 0.0:
        raise ValueError("t must be non-negative")
    if duration == 0.0:
        return _restore(np.full(depths.shape, float(c_initial)), scalar)

    surface = float(c_surface)
    start = float(c_initial)
    result = surface + (start - surface) * erf(
        depths / (2.0 * math.sqrt(diffusivity * duration))
    )
    return _restore(result, scalar)


def instantaneous_source(
    x: Any,
    t: float,
    *,
    D: float,
    amount: float = 1.0,
    center: float = 0.0,
) -> Any:
    """Spreading of a spike of material released at one instant.

    ``c = amount / sqrt(4 pi D t) * exp(-(x - center)^2 / (4 D t))``, the
    fundamental solution in one dimension. The integral over all ``x`` stays
    equal to ``amount`` for all time, which makes this a good check on whether a
    solver conserves mass.

    Args:
        x: Position, or an array of positions.
        t: Time since release. Must be positive.
        D: Diffusivity.
        amount: Total amount released per unit area.
        center: Release position.
    """

    positions, scalar = _as_array(x, "x")
    diffusivity = _positive(D, "D")
    duration = _positive(t, "t")
    spread = 4.0 * diffusivity * duration
    result = (
        float(amount)
        / math.sqrt(math.pi * spread)
        * np.exp(-((positions - float(center)) ** 2) / spread)
    )
    return _restore(result, scalar)


# ---------------------------------------------------------------------------
# Steady solutions with reaction
# ---------------------------------------------------------------------------


def steady_slab_first_order(
    x: Any,
    *,
    D: float,
    k: float,
    L: float,
    c_surface: float,
) -> Any:
    """Steady diffusion with first-order consumption in a slab sealed on one side.

    Solves ``D c'' - k c = 0`` on ``0 <= x <= L`` with ``c(0) = c_surface`` and
    no flux at ``x = L``::

        c = c_surface * cosh(m (L - x)) / cosh(m L),    m = sqrt(k / D)

    This is the workhorse steady reaction-diffusion profile: a nutrient entering
    tissue from one surface and being consumed as it goes. The group ``m L`` is
    the Thiele modulus -- see :func:`thiele_modulus`.

    Args:
        x: Depth from the exposed face, or an array of depths.
        D: Diffusivity.
        k: First-order rate constant.
        L: Slab thickness (exposed at ``x = 0``, sealed at ``x = L``).
        c_surface: Value at the exposed face.
    """

    depths, scalar = _as_array(x, "x")
    diffusivity = _positive(D, "D")
    rate = _positive(k, "k")
    thickness = _positive(L, "L")
    surface = float(c_surface)
    if not math.isfinite(surface):
        raise ValueError("c_surface must be finite")
    endpoint_tolerance = 8.0 * np.finfo(np.float64).eps * thickness
    if np.any(depths < -endpoint_tolerance) or np.any(
        depths - thickness > endpoint_tolerance
    ):
        raise ValueError("x must lie in [0, L]")
    # A native uniform mesh can place its last node one ulp beyond L after
    # multiplying spacing by the cell count. Accept only endpoint roundoff.
    depths = np.clip(depths, 0.0, thickness)
    # Taking the square roots separately avoids an overflowing k/D ratio even
    # when the inverse reaction length itself is representable.
    modulus = math.sqrt(rate) / math.sqrt(diffusivity)
    if not math.isfinite(modulus):
        raise ValueError("sqrt(k/D) is not representable; choose different units")
    if surface == 0.0:
        return _restore(np.zeros_like(depths), scalar)
    # cosh(m(L-x))/cosh(mL) = exp(-mx) *
    # (1 + exp(-2m(L-x))) / (1 + exp(-2mL)). All exponentials are bounded
    # on [0,L]. Work in log concentration to preserve representable results
    # even when the unscaled ratio would underflow before multiplying c_surface.
    with np.errstate(over="ignore", under="ignore"):
        log_ratio = (
            -modulus * depths
            + np.log1p(np.exp(-2.0 * (modulus * (thickness - depths))))
            - math.log1p(math.exp(-2.0 * (modulus * thickness)))
        )
        result = np.copysign(np.exp(math.log(abs(surface)) + log_ratio), surface)
    result = np.where(depths == 0.0, surface, result)
    return _restore(result, scalar)


def steady_radial_first_order(
    r: Any,
    *,
    D: float,
    k: float,
    R: float,
    c_surface: float,
    geometry: str = "sphere",
) -> Any:
    """Steady diffusion and linear consumption in a solid cylinder or sphere.

    Solves ``D div(grad(c)) - k c = 0`` on ``0 <= r <= R`` with symmetry
    at the origin and ``c(R) = c_surface``. ``D`` must be constant and positive;
    ``k`` may be zero. Annular domains have different boundary conditions and
    are outside this reference's scope.

    With ``m = sqrt(k/D)``, the cylindrical solution is
    ``c/c_surface = I0(m r)/I0(m R)``. The spherical solution is
    ``c/c_surface = R sinh(m r)/(r sinh(m R))``, evaluated by its finite limit
    at the origin. Exponentially scaled formulas remain finite at large
    Thiele modulus, including when the centre concentration underflows to zero.

    Args:
        r: Radius or array of radii in ``[0, R]``.
        D: Uniform diffusivity.
        k: Nonnegative first-order consumption rate.
        R: Outer radius.
        c_surface: Fixed outer concentration.
        geometry: ``"sphere"``/``"spherical"`` or
            ``"cylinder"``/``"cylindrical"``.
    """

    radii, scalar = _as_array(r, "r")
    diffusivity = _positive(D, "D")
    outer = _positive(R, "R")
    rate = float(k)
    surface = float(c_surface)
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError("k must be finite and non-negative")
    if not math.isfinite(surface):
        raise ValueError("c_surface must be finite")
    kind = geometry.strip().casefold()
    if kind not in ("sphere", "spherical", "cylinder", "cylindrical"):
        raise ValueError("geometry must be 'sphere' or 'cylinder'")
    endpoint_tolerance = 8.0 * np.finfo(np.float64).eps * outer
    if np.any(radii < -endpoint_tolerance) or np.any(
        radii - outer > endpoint_tolerance
    ):
        raise ValueError("r must lie in [0, R]")
    radii = np.clip(radii, 0.0, outer)
    if rate == 0.0 or surface == 0.0:
        return _restore(np.full_like(radii, surface), scalar)
    phi = thiele_modulus(D=diffusivity, k=rate, length=outer)
    if phi < 1.0e-8:
        return _restore(np.full_like(radii, surface), scalar)
    argument = phi * (radii / outer)
    # Form the small distance to the surface before multiplying by phi;
    # subtracting the two large Bessel/hyperbolic arguments loses this detail.
    distance = phi * ((outer - radii) / outer)

    if kind in ("cylinder", "cylindrical"):
        from scipy.special import i0e

        log_ratio = -distance + np.log(i0e(argument)) - math.log(i0e(phi))
    else:

        def log_scaled_sinhc(values: np.ndarray) -> np.ndarray:
            # log(exp(-z) * sinh(z)/z), with the analytic value zero at z=0.
            result = np.empty_like(values)
            small = values < 0.1
            squared = values[small] ** 2
            series = squared * (
                1.0 / 6.0
                + squared
                * (
                    1.0 / 120.0
                    + squared
                    * (1.0 / 5040.0 + squared * (1.0 / 362880.0 + squared / 39916800.0))
                )
            )
            result[small] = np.log1p(series) - values[small]
            large = values[~small]
            with np.errstate(over="ignore", under="ignore"):
                result[~small] = (
                    np.log(-np.expm1(-2.0 * large)) - math.log(2.0) - np.log(large)
                )
            return result

        log_ratio = (
            -distance
            + log_scaled_sinhc(argument)
            - log_scaled_sinhc(np.array([phi]))[0]
        )
    # Combine in log concentration to retain representable tiny values when
    # the dimensionless ratio alone would underflow before scaling by c_surface.
    with np.errstate(over="ignore", under="ignore"):
        magnitude = np.minimum(
            np.exp(math.log(abs(surface)) + np.minimum(log_ratio, 0.0)), abs(surface)
        )
    result = np.copysign(magnitude, surface)
    result = np.where(radii == outer, surface, result)
    return _restore(result, scalar)


def thiele_modulus(*, D: float, k: float, length: float) -> float:
    """``sqrt(k / D) * length`` -- reaction rate against diffusion rate.

    Small means diffusion wins and the interior sees nearly the surface value.
    Large means reaction wins and the interior is starved. The crossover is
    around 1.

    ``length`` is the characteristic length of the geometry: slab thickness for a
    slab sealed on one face, radius for a cylinder or sphere.
    """

    root_rate = math.sqrt(_positive(k, "k"))
    root_diffusivity = math.sqrt(_positive(D, "D"))
    span = _positive(length, "length")
    inverse_length = root_rate / root_diffusivity
    value = (
        inverse_length * span
        if math.isfinite(inverse_length)
        else (root_rate * span) / root_diffusivity
    )
    if not math.isfinite(value):
        raise ValueError("the Thiele modulus is not representable in float64")
    return value


def effectiveness_factor(modulus: float, geometry: str = "slab") -> float:
    """What fraction of the maximum possible reaction rate you actually get.

    An effectiveness factor of 1 means every point reacts as fast as it would at
    the surface value -- diffusion is not limiting. Well below 1 means the
    interior is starved and making the region bigger buys you almost nothing.

    Args:
        modulus: Thiele modulus, from :func:`thiele_modulus`.
        geometry: ``"slab"``, ``"cylinder"`` or ``"sphere"``.

    Returns:
        The effectiveness factor, between 0 and 1.
    """

    phi = float(modulus)
    if not math.isfinite(phi) or phi < 0.0:
        raise ValueError("the Thiele modulus must be finite and non-negative")
    kind = geometry.strip().casefold()
    if kind not in ("slab", "cylinder", "sphere"):
        raise ValueError("geometry must be 'slab', 'cylinder' or 'sphere'")
    if phi < 1.0e-8:
        # All three geometries differ from one by O(phi^2), below one ulp
        # here. This also avoids forming half a subnormal Bessel argument.
        return 1.0

    if kind == "slab":
        return math.tanh(phi) / phi
    if kind == "cylinder":
        from scipy.special import i0e, i1e

        # Both exponentially scaled Bessel functions remain finite at large
        # phi; their common exp(-phi) factor cancels in the ratio.
        return float((2.0 * (i1e(phi) / i0e(phi))) / phi)
    if phi < 0.1:
        # Taylor expansion of 3*(phi*coth(phi)-1)/phi^2 avoids subtracting
        # nearly equal numbers. The omitted term is below float64 precision
        # over this interval.
        squared = phi * phi
        return 1.0 + squared * (
            -1.0 / 15.0
            + squared
            * (
                2.0 / 315.0
                + squared
                * (
                    -1.0 / 1575.0
                    + squared * (2.0 / 31185.0 - squared * 1382.0 / 212837625.0)
                )
            )
        )
    # Divide before multiplying, so phi^2 cannot overflow at large phi.
    return (3.0 / phi) * (1.0 / math.tanh(phi) - 1.0 / phi)
