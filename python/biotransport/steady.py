"""Solving for the steady state directly.

A lot of transport homework asks for the *final* answer: the oxygen profile in a
tissue slab that has settled down, the flux through a membrane at steady state,
the concentration in a pellet where diffusion and reaction have balanced. You
can get there by integrating forward until nothing changes, but that means
taking thousands of tiny explicit steps to watch a transient you do not care
about -- and on a 2D grid it can take minutes.

This module solves the steady equation directly with Newton's method instead::

    0 = div(D grad c) + R(c)

which usually converges in a handful of iterations. It reuses the problem you
already built, reads the reaction you already declared, and supplies the exact
analytic Jacobian for the built-in kinetics so you never have to differentiate
anything by hand.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any
import warnings

import numpy as np

from ._core import Boundary, BoundaryType, Geometry, StructuredMesh
from .newton_raphson import (
    NewtonEvaluationError,
    NewtonLineSearchError,
    NonlinearDiffusionSolver,
    _real_array,
)
from .solution import Solution

__all__ = ["solve_steady"]


_SIDE_LABEL = {
    Boundary.Left: "left",
    Boundary.Right: "right",
    Boundary.Bottom: "bottom",
    Boundary.Top: "top",
}


def _sides_for(mesh: Any) -> tuple[Boundary, ...]:
    if bool(mesh.is_1d()):
        return (Boundary.Left, Boundary.Right)
    return (Boundary.Left, Boundary.Right, Boundary.Bottom, Boundary.Top)


def _boundary_conditions(problem: Any, mesh: Any) -> dict[Boundary, Any]:
    """Read the configured boundary conditions off the native problem."""

    raw = problem.boundaries()
    conditions: dict[Boundary, Any] = {}
    for side in _sides_for(mesh):
        conditions[side] = raw[int(side.value)]
    return conditions


def _reaction_terms(recipe: Any, *, concentration_scale: float, rate_scale: float):
    """Translate the recorded reaction stack into the steady solver's sign convention.

    The transient problem is ``dc/dt = div(D grad c) + R(c)``. The steady solver
    is written as ``-div(D grad u) + G(u) = S``, so a source in ``R`` becomes a
    right-hand side ``S``, and everything else enters as ``G = -R``.

    The returned terms use ``u = c / concentration_scale`` and dimensionless
    time ``t * rate_scale``. Scaling the parameters before evaluation avoids
    first computing products in a physical unit system that can underflow.
    """

    pieces: list[Any] = []
    derivatives: list[Any] = []
    source = 0.0

    for term in recipe.reactions:
        if term.mode == "replace":
            pieces.clear()
            derivatives.clear()
            source = 0.0

        if term.kind == "linear_decay":
            k = float(term.args[0]) / rate_scale
            pieces.append(lambda u, k=k: k * u)
            derivatives.append(lambda u, k=k: np.full_like(np.asarray(u, float), k))
        elif term.kind == "constant_source":
            source += float(term.args[0]) / concentration_scale / rate_scale
        elif term.kind == "michaelis_menten":
            vmax = float(term.args[0]) / concentration_scale / rate_scale
            km = float(term.args[1]) / concentration_scale

            def uptake(u, v=vmax, m=km):
                positive = np.maximum(u, 0.0)
                scale = np.maximum(positive, m)
                substrate, affinity = positive / scale, m / scale
                return v * (substrate / (substrate + affinity))

            def uptake_derivative(u, v=vmax, m=km):
                positive = np.maximum(u, 0.0)
                scale = np.maximum(positive, m)
                substrate, affinity = positive / scale, m / scale
                slope = (v / scale) * (affinity / (substrate + affinity) ** 2)
                # Native kinetics consume nothing below zero. At zero take
                # the nonnegative-side derivative so Newton can leave zero.
                return np.where(u >= 0.0, slope, 0.0)

            pieces.append(uptake)
            derivatives.append(uptake_derivative)
        elif term.kind == "logistic_growth":
            r = float(term.args[0]) / rate_scale
            capacity = float(term.args[1]) / concentration_scale
            pieces.append(lambda u, r=r, K=capacity: -r * u * (1.0 - u / K))
            derivatives.append(lambda u, r=r, K=capacity: -r * (1.0 - 2.0 * u / K))
        else:
            raise ValueError(
                "a steady solve needs to differentiate the reaction, which it can "
                "only do for the built-in kinetics (linear_decay, constant_source, "
                "michaelis_menten, logistic_growth). For a custom reaction, either "
                "integrate forward with bt.solve(problem, end_time=...) until it "
                "settles, or drive bt.NonlinearDiffusionSolver directly and supply "
                "your own dR/dc."
            )

    if not pieces:
        return None, None, source

    def reaction(u: np.ndarray) -> np.ndarray:
        values = np.asarray(u, dtype=np.float64)
        total = np.zeros_like(values)
        for piece in pieces:
            total = total + piece(values)
        return total

    def derivative(u: np.ndarray) -> np.ndarray:
        values = np.asarray(u, dtype=np.float64)
        total = np.zeros_like(values)
        for slope in derivatives:
            total = total + slope(values)
        return total

    return reaction, derivative, source


def _scaled_geometry(mesh: Any) -> tuple[Any, float]:
    """Use the smallest spacing as a length unit for the discrete operator.

    This keeps diffusion coefficients of order one after dividing diffusivity
    by its maximum. Scaling the boundaries too is essential: a
    residual vector containing concentrations, gradients and rates otherwise
    compares quantities with different dimensions during Newton's line search.
    """

    dx = float(mesh.dx())
    nx = int(mesh.nx())
    if bool(mesh.is_1d()):
        geometry = mesh.geometry()
        if geometry != Geometry.CARTESIAN:
            inner = float(mesh.x(0)) / dx
            return StructuredMesh(nx, inner, inner + nx, geometry), dx
        return StructuredMesh(nx, 0.0, float(nx)), dx
    dy = float(mesh.dy())
    ny = int(mesh.ny())
    spacing = min(dx, dy)
    return (
        StructuredMesh(nx, ny, 0.0, nx * (dx / spacing), 0.0, ny * (dy / spacing)),
        spacing,
    )


def _concentration_scale(
    recipe: Any,
    mesh: Any,
    conditions: dict[Boundary, Any],
    starting: np.ndarray,
    reference_diffusivity: float,
) -> float:
    """Choose a field unit from boundary data, a source, or the initial state.

    A poor Newton guess must not make a prescribed concentration insignificant.
    Consequently boundary data take precedence over the initial guess. A
    gradient contributes concentration over the domain length, not its raw
    numeric value, which would change this scale on converting metres to mm.
    """

    span = float(mesh.dx()) * int(mesh.nx())
    if not bool(mesh.is_1d()):
        span = max(span, float(mesh.dy()) * int(mesh.ny()))
    magnitude = 0.0
    for condition in conditions.values():
        if condition.type == BoundaryType.DIRICHLET:
            magnitude = max(magnitude, abs(float(condition.value)))
        elif condition.type == BoundaryType.NEUMANN:
            magnitude = max(magnitude, abs(float(condition.value)) * span)
    if magnitude == 0.0:
        source = 0.0
        for term in recipe.reactions:
            if term.mode == "replace":
                source = 0.0
            if term.kind == "constant_source":
                source += float(term.args[0])
        # S L^2 / D is the scale of the Poisson solution with zero boundaries.
        if source != 0.0:
            magnitude = (abs(source) / reference_diffusivity * span) * span
    if magnitude == 0.0 and starting.size:
        magnitude = float(np.max(np.abs(starting)))
    if not math.isfinite(magnitude):
        raise ValueError(
            "the concentration scale is not representable in float64; "
            "choose units that keep boundary and source magnitudes finite"
        )
    return magnitude if magnitude > 0.0 else 1.0


def _check_2d_cost(mesh: Any) -> None:
    """Warn before a 2D steady solve large enough to be slow.

    The 2D Jacobian is assembled analytically and sparsely, so this is no longer
    the hard wall it once was -- a few tens of thousands of unknowns solve in
    seconds. Past that, the sparse factorization starts to dominate and it is
    worth saying so before the wait rather than after.
    """

    if bool(mesh.is_1d()):
        return
    nodes = int(mesh.num_nodes())
    if nodes <= 40_000:
        return
    warnings.warn(
        f"this 2D steady solve has {nodes} unknowns. The Jacobian is sparse, so "
        f"this will finish, but the factorization will take a while. A coarser "
        f"mesh usually gives nearly the same steady field much faster.",
        RuntimeWarning,
        stacklevel=3,
    )


def solve_steady(
    problem: Any,
    *,
    guess: Any = None,
    tol: float | None = None,
    max_iterations: int = 50,
    verbose: bool = False,
) -> Solution:
    """Solve for the steady state of a problem, without marching through time.

    Solves ``0 = div(D grad c) + R(c)`` on the problem's mesh, using Newton's
    method with the analytic Jacobian of whichever built-in reaction you
    configured.

    Args:
        problem: A :class:`biotransport.Problem`. Its diffusivity, reaction and
            boundary conditions are read directly.
        guess: Starting field for Newton. Defaults to the problem's initial
            condition, which is usually a good guess.
        tol: Dimensionless convergence tolerance on both the residual norm and
            the Newton correction norm; defaults to ``1e-10``. The mesh spacing,
            diffusivity and concentration are scaled before solving, including
            boundary gradients. Equivalent unit systems therefore give the
            same convergence decision. This tolerance is never loosened on a
            retry. It controls the algebraic solve, not mesh discretization error.
        max_iterations: Give up after this many Newton steps.
        verbose: Print the residual at each iteration.

    Returns:
        A :class:`~biotransport.Solution` holding the steady field. It plots and
        compares exactly like a transient one; :attr:`Solution.newton` carries
        the convergence record and :attr:`Solution.steady` is ``True``. Newton's
        residual norms are dimensionless; its solution and correction norms are
        returned in the original concentration units.

    Raises:
        ValueError: If the problem is outside what the steady solver covers --
            advection, a Robin boundary, a custom reaction, a spatially varying
            diffusivity in 2D, radial geometry in 2D, or a Neumann condition in
            2D. The message says
            which, and what to do instead.

    Example:
        Oxygen in a tissue slab, consumed by Michaelis-Menten kinetics, fed from
        one face:

        >>> mesh = bt.mesh_1d(100, 0, 100e-6)
        >>> problem = (
        ...     bt.Problem(mesh)
        ...     .diffusivity(2e-9)
        ...     .michaelis_menten(Vmax=1e-3, Km=1e-3)
        ...     .initial(0.05)
        ...     .dirichlet("left", 0.05)
        ...     .sealed("right")
        ... )
        >>> steady = bt.solve_steady(problem)
        >>> steady.plot()

    Note:
        In 1D, slabs, cylinders and spheres use the same conservative
        control-volume balance as the transient solver, including reaction and
        source terms at Neumann boundaries and harmonic face diffusivity.
        Annuli support fixed concentrations or outward-normal gradients at
        either wall. At the radial origin only symmetry (zero Neumann) is
        valid, and is the default. Advection is not included.
    """

    mesh = problem.mesh()
    recipe = getattr(problem, "_recipe", None)
    if recipe is None:
        raise TypeError(
            "solve_steady needs a problem built with bt.Problem(mesh) so it can "
            "read back the reaction it has to differentiate."
        )

    if problem.has_advection():
        raise ValueError(
            "the steady solver handles diffusion and reaction but not advection. "
            "Integrate forward instead: bt.solve(problem, end_time=...), which "
            "reaches steady state once the Fourier number passes about 1."
        )

    is_1d = bool(mesh.is_1d())
    if not is_1d and mesh.is_radial():
        raise ValueError(
            "the steady solver supports radial geometry in 1D only. For a "
            "2D axisymmetric problem, integrate forward with "
            "bt.solve(problem, end_time=...)."
        )

    if recipe.diffusivity_field is not None:
        if not is_1d:
            raise ValueError(
                "a steady solve with a spatially varying diffusivity is only "
                "supported in 1D. In 2D, either use a uniform diffusivity or "
                "integrate forward with bt.solve(problem, end_time=...)."
            )
        diffusivity: Any = np.asarray(recipe.diffusivity_field, dtype=np.float64)
    else:
        diffusivity = recipe.diffusivity
        if diffusivity is None:
            diffusivity = float(problem.diffusivity())
        if diffusivity <= 0.0:
            raise ValueError(
                f"a steady solve needs a positive diffusivity, got {diffusivity:g}. "
                f"With D = 0 there is nothing to balance the reaction against."
            )

    _check_2d_cost(mesh)

    conditions = _boundary_conditions(problem, mesh)
    if guess is None:
        starting = _real_array(problem.initial(), "Initial condition").reshape(-1)
        if starting.size != int(mesh.num_nodes()):
            starting = np.zeros(int(mesh.num_nodes()))
    else:
        starting = _real_array(guess, "Initial guess").reshape(-1)

    normalized_mesh, length_scale = _scaled_geometry(mesh)
    reference_diffusivity = float(np.max(np.asarray(diffusivity, dtype=np.float64)))
    rate_scale = (reference_diffusivity / length_scale) / length_scale
    if not math.isfinite(rate_scale) or rate_scale <= 0.0:
        raise ValueError(
            "the diffusion rate D / spacing^2 is not representable in float64; "
            "choose units that keep diffusivity and mesh spacing finite"
        )
    concentration_scale = _concentration_scale(
        recipe, mesh, conditions, starting, reference_diffusivity
    )
    normalized_diffusivity = (
        np.asarray(diffusivity, dtype=np.float64) / reference_diffusivity
    )
    solver = NonlinearDiffusionSolver(normalized_mesh, normalized_diffusivity)

    for side, condition in conditions.items():
        kind = condition.type
        label = _SIDE_LABEL[side]
        if kind == BoundaryType.DIRICHLET:
            solver.set_boundary(
                side, float(condition.value) / concentration_scale, "dirichlet"
            )
        elif kind == BoundaryType.NEUMANN:
            if not is_1d:
                raise ValueError(
                    f"the steady solver does not support a Neumann (zero-gradient) "
                    f"condition in 2D, and {label} is one. Either make every 2D side "
                    f"a fixed value, or integrate forward with "
                    f"bt.solve(problem, end_time=...)."
                )
            solver.set_boundary(
                side,
                float(condition.value) / concentration_scale * length_scale,
                "neumann",
            )
        else:
            raise ValueError(
                f"the steady solver supports fixed-value and fixed-gradient "
                f"boundaries, but {label} is a Robin condition. Integrate forward "
                f"with bt.solve(problem, end_time=...) instead."
            )

    reaction, derivative, source = _reaction_terms(
        recipe, concentration_scale=concentration_scale, rate_scale=rate_scale
    )
    if reaction is not None:
        active = []
        for term in recipe.reactions:
            if term.mode == "replace":
                active.clear()
            active.append(term)
        nonnegative_uptake = (
            any(t.kind == "michaelis_menten" and t.args[0] > 0 for t in active)
            and all(
                t.kind in {"michaelis_menten", "linear_decay", "constant_source"}
                for t in active
            )
            and source >= 0
            and all(
                (bc.type == BoundaryType.DIRICHLET and bc.value >= 0)
                or (bc.type == BoundaryType.NEUMANN and bc.value == 0)
                for bc in conditions.values()
            )
        )
        if nonnegative_uptake:
            # These data admit a nonnegative concentration. Reject trial states
            # outside that domain so Armijo backtracks instead of accepting the
            # flat, clipped branch of native uptake and a singular Jacobian.
            # Only the numerical guess is projected; the model is unchanged.
            starting = np.maximum(starting, 0.0)

            def physical_reaction(u):
                if np.any(u < 0):
                    raise NewtonEvaluationError(
                        "uptake trial concentration is negative"
                    )
                return reaction(u)

            solver.set_reaction(physical_reaction, derivative)
        else:
            solver.set_reaction(reaction, derivative)
    if source != 0.0:
        solver.set_source(np.full(int(mesh.num_nodes()), float(source)))

    solver.set_parameters(
        max_iterations=max_iterations,
        tol=1.0e-10 if tol is None else tol,
        verbose=verbose,
    )
    try:
        result = solver.solve(starting / concentration_scale)
    except (NewtonEvaluationError, NewtonLineSearchError) as error:
        raise RuntimeError(
            f"the steady solve did not converge: {error}. Pass a better starting "
            f"field with guess=, or set a dimensionless tol= appropriate for the "
            f"requested accuracy."
        ) from error
    if not result.converged:
        raise RuntimeError(
            f"the steady solve did not converge after {result.iterations} "
            f"iterations (dimensionless residual {result.residual_norm:.3g}). "
            f"Pass a better starting field with guess= or raise max_iterations."
        )

    result = replace(
        result,
        solution=np.asarray(result.solution) * concentration_scale,
        update_norm=result.update_norm * concentration_scale,
        applied_update_norm=result.applied_update_norm * concentration_scale,
    )
    field = np.asarray(result.solution, dtype=np.float64).reshape(-1)
    return Solution(
        mesh=mesh,
        times=[0.0],
        fields=[field],
        diagnostics=(),
        problem=problem,
        total_steps=int(result.iterations),
        steady=True,
        newton=result,
    )
