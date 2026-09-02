"""Bitwise golden fixtures for every native BioTransport solver.

This module defines a small, deterministic problem for each compiled solver
entry point exposed by ``biotransport._core`` and records every output the
solver reports (fields, times, step counts, numeric diagnostics).  The
fixtures are consumed by ``python/tests/test_native_goldens.py`` which
re-runs each problem and demands bitwise equality, so later refactors can
prove that no numerics moved.

Regenerate the fixtures (only when a numerics change is intended) with::

    python python/tests/golden/capture_goldens.py

Design rules that keep the fixtures trustworthy:

* Every case uses literal parameters and literal time steps; nothing is
  derived from wall-clock time, randomness, or environment.
* Only ``biotransport._core`` is exercised so Python-layer refactors cannot
  hide a native change behind an adapter.
* Wall-clock diagnostics (``SolverStats.wall_time_s``) are never recorded.
* Arrays are stored with the dtype the binding returns; Python scalars are
  stored as 0-d arrays and compared exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

import biotransport
from biotransport import _core as core
from biotransport.contracts import list_native_solver_symbols

GOLDEN_DIR = Path(__file__).resolve().parent

Entries = dict[str, Any]


@dataclass(frozen=True)
class GoldenCase:
    """One deterministic solver problem and the native symbols it exercises."""

    name: str
    symbols: tuple[str, ...]
    run: Callable[[], Entries]
    requires_sparse: bool = False

    @property
    def fixture_path(self) -> Path:
        return GOLDEN_DIR / f"{self.name}.npz"

    def available(self) -> bool:
        return not self.requires_sparse or bool(core.sparse_matrix_available())


CASES: dict[str, GoldenCase] = {}


def _case(
    name: str, *symbols: str, requires_sparse: bool = False
) -> Callable[[Callable[[], Entries]], Callable[[], Entries]]:
    def register(run: Callable[[], Entries]) -> Callable[[], Entries]:
        if name in CASES:
            raise ValueError(f"duplicate golden case {name!r}")
        CASES[name] = GoldenCase(name, tuple(symbols), run, requires_sparse)
        return run

    return register


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def _x_nodes(mesh: core.StructuredMesh) -> np.ndarray:
    return np.array([mesh.x(i) for i in range(mesh.nx() + 1)], dtype=np.float64)


def _xy_grid(mesh: core.StructuredMesh) -> tuple[np.ndarray, np.ndarray]:
    x = _x_nodes(mesh)
    y = np.array([mesh.y(0, j) for j in range(mesh.ny() + 1)], dtype=np.float64)
    return np.meshgrid(x, y)


def _gaussian_1d(mesh: core.StructuredMesh, center: float, width: float) -> np.ndarray:
    x = _x_nodes(mesh)
    return np.exp(-((x - center) ** 2) / (2.0 * width * width))


def _gaussian_2d(
    mesh: core.StructuredMesh, cx: float, cy: float, w: float
) -> np.ndarray:
    X, Y = _xy_grid(mesh)
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * w * w)).ravel(order="C")


def _gaussian_3d(mesh: core.StructuredMesh3D, w: float) -> np.ndarray:
    values = np.empty(mesh.num_nodes(), dtype=np.float64)
    cx = 0.5 * (mesh.xmin() + mesh.xmax())
    cy = 0.5 * (mesh.ymin() + mesh.ymax())
    cz = 0.5 * (mesh.zmin() + mesh.zmax())
    for k in range(mesh.nz() + 1):
        for j in range(mesh.ny() + 1):
            for i in range(mesh.nx() + 1):
                r2 = (
                    (mesh.x(i) - cx) ** 2
                    + (mesh.y(j) - cy) ** 2
                    + (mesh.z(k) - cz) ** 2
                )
                values[mesh.index(i, j, k)] = math.exp(-r2 / (2.0 * w * w))
    return values


def _transport_diagnostics(diagnostics: core.SolveDiagnostics) -> Entries:
    names = (
        "steps",
        "requested_final_time",
        "final_time",
        "requested_time_step",
        "minimum_time_step",
        "maximum_time_step",
        "transport_stable_time_step",
        "certified_stable_time_step",
        "maximum_transport_loss_rate",
        "reaction_rate_bound",
        "automatic_time_step",
        "reaction_stability_bound_known",
        "initial_mass",
        "final_mass",
        "mass_change",
        "initial_minimum",
        "initial_maximum",
        "final_minimum",
        "final_maximum",
    )
    return {f"diag_{name}": getattr(diagnostics, name) for name in names}


def _transport_entries(result: core.TransportResult) -> Entries:
    entries: Entries = {
        "concentration": result.concentration,
        "time": result.time,
    }
    entries.update(_transport_diagnostics(result.diagnostics))
    return entries


def _solver_stats(stats: core.SolverStats) -> Entries:
    # wall_time_s is deliberately excluded: it is not a numerical output.
    names = (
        "dt",
        "steps",
        "t_end",
        "mass_initial",
        "mass_final",
        "mass_abs_drift",
        "mass_rel_drift",
        "u_min_initial",
        "u_max_initial",
        "u_min_final",
        "u_max_final",
    )
    return {f"stats_{name}": getattr(stats, name) for name in names}


def _implicit_result(prefix: str, result: core.ImplicitSolveResult) -> Entries:
    return {
        f"{prefix}_steps": result.steps,
        f"{prefix}_total_time": result.total_time,
        f"{prefix}_residual": result.residual,
        f"{prefix}_success": result.success,
    }


def _adi_result(prefix: str, result: core.ADISolveResult) -> Entries:
    return {
        f"{prefix}_steps": result.steps,
        f"{prefix}_substeps": result.substeps,
        f"{prefix}_time": result.time,
        f"{prefix}_total_time": result.total_time,
        f"{prefix}_success": result.success,
    }


def _membrane_result(result: core.MembraneDiffusionResult) -> Entries:
    return {
        "x": result.x(),
        "concentration": result.concentration(),
        "flux": result.flux,
        "permeability": result.permeability,
        "effective_diffusivity": result.effective_diffusivity,
    }


# ---------------------------------------------------------------------------
# solve_transport (canonical unified core)
# ---------------------------------------------------------------------------


@_case("transport_1d_uniform_auto_dt", "solve_transport")
def _transport_1d_uniform_auto_dt() -> Entries:
    mesh = core.StructuredMesh(32, 0.0, 1.0)
    problem = core.TransportProblem(mesh)
    problem.diffusivity(0.01).initial_condition(_gaussian_1d(mesh, 0.5, 0.1))
    problem.dirichlet(core.Boundary.Left, 0.0).neumann(core.Boundary.Right, 0.0)
    return _transport_entries(
        core.solve_transport(problem, core.SolveOptions.until(0.2))
    )


@_case("transport_1d_uniform_fixed_dt", "solve_transport")
def _transport_1d_uniform_fixed_dt() -> Entries:
    mesh = core.StructuredMesh(32, 0.0, 1.0)
    problem = core.TransportProblem(mesh)
    problem.diffusivity(0.01).initial_condition(_gaussian_1d(mesh, 0.5, 0.1))
    problem.dirichlet(core.Boundary.Left, 1.0).dirichlet(core.Boundary.Right, 0.0)
    options = core.SolveOptions.until(0.1)
    options.time_step = 0.003
    return _transport_entries(core.solve_transport(problem, options))


@_case("transport_1d_variable_d", "solve_transport")
def _transport_1d_variable_d() -> Entries:
    mesh = core.StructuredMesh(40, 0.0, 1.0)
    x = _x_nodes(mesh)
    d_field = 0.005 + 0.02 * x
    problem = core.TransportProblem(mesh)
    problem.diffusivity_field(d_field).initial_condition(_gaussian_1d(mesh, 0.4, 0.08))
    problem.neumann(core.Boundary.Left, 0.0).neumann(core.Boundary.Right, 0.0)
    return _transport_entries(
        core.solve_transport(problem, core.SolveOptions.until(0.15))
    )


@_case("transport_1d_advection_upwind", "solve_transport")
def _transport_1d_advection_upwind() -> Entries:
    mesh = core.StructuredMesh(48, 0.0, 1.0)
    problem = core.TransportProblem(mesh)
    problem.diffusivity(0.002).velocity(0.5).advection_scheme(
        core.AdvectionScheme.UPWIND
    )
    problem.initial_condition(_gaussian_1d(mesh, 0.25, 0.05))
    problem.dirichlet(core.Boundary.Left, 0.0).neumann(core.Boundary.Right, 0.0)
    return _transport_entries(
        core.solve_transport(problem, core.SolveOptions.until(0.4))
    )


@_case("transport_1d_linear_decay", "solve_transport")
def _transport_1d_linear_decay() -> Entries:
    mesh = core.StructuredMesh(32, 0.0, 1.0)
    problem = core.TransportProblem(mesh)
    problem.diffusivity(0.01).linear_decay(0.7).initial_condition(1.0)
    problem.dirichlet(core.Boundary.Left, 1.0).neumann(core.Boundary.Right, 0.0)
    return _transport_entries(
        core.solve_transport(problem, core.SolveOptions.until(0.3))
    )


@_case("transport_1d_robin", "solve_transport")
def _transport_1d_robin() -> Entries:
    mesh = core.StructuredMesh(32, 0.0, 1.0)
    problem = core.TransportProblem(mesh)
    problem.diffusivity(0.02).initial_condition(0.5)
    problem.robin(core.Boundary.Left, 1.0, 0.5, 1.0)
    problem.boundary(core.Boundary.Right, core.BoundaryCondition.robin(2.0, -1.0, 0.0))
    return _transport_entries(
        core.solve_transport(problem, core.SolveOptions.until(0.25))
    )


@_case("transport_1d_custom_reaction_bound", "solve_transport")
def _transport_1d_custom_reaction_bound() -> Entries:
    mesh = core.StructuredMesh(32, 0.0, 1.0)
    problem = core.TransportProblem(mesh)
    problem.diffusivity(0.01).initial_condition(_gaussian_1d(mesh, 0.5, 0.15))
    problem.reaction(lambda c, x, y, t: -0.8 * c + 0.1 * x, 0.8)
    problem.add_reaction(lambda c, x, y, t: 0.05 * math.sin(3.0 * t), 0.0)
    problem.neumann(core.Boundary.Left, 0.0).neumann(core.Boundary.Right, 0.0)
    return _transport_entries(
        core.solve_transport(problem, core.SolveOptions.until(0.3))
    )


@_case("transport_1d_custom_reaction_unbounded_fixed_dt", "solve_transport")
def _transport_1d_custom_reaction_unbounded_fixed_dt() -> Entries:
    mesh = core.StructuredMesh(24, 0.0, 1.0)
    problem = core.TransportProblem(mesh)
    problem.diffusivity(0.01).initial_condition(0.2)
    problem.reaction(lambda c, x, y, t: 0.3 * c * (1.0 - c))
    problem.dirichlet(core.Boundary.Left, 0.2).dirichlet(core.Boundary.Right, 0.2)
    options = core.SolveOptions.until(0.2)
    options.time_step = 0.01
    return _transport_entries(core.solve_transport(problem, options))


@_case("transport_1d_builtin_kinetics", "solve_transport")
def _transport_1d_builtin_kinetics() -> Entries:
    mesh = core.StructuredMesh(24, 0.0, 1.0)
    problem = core.TransportProblem(mesh)
    problem.diffusivity(0.01).initial_condition(0.5)
    problem.michaelis_menten(0.4, 0.3).add_constant_source(0.05)
    problem.add_logistic_growth(0.2, 2.0)
    problem.neumann(core.Boundary.Left, 0.0).dirichlet(core.Boundary.Right, 0.5)
    # Logistic growth has no certified derivative bound, so the step is explicit.
    options = core.SolveOptions.until(0.3)
    options.time_step = 0.01
    return _transport_entries(core.solve_transport(problem, options))


@_case("transport_2d_uniform", "solve_transport")
def _transport_2d_uniform() -> Entries:
    mesh = core.StructuredMesh(12, 10, 0.0, 1.2, 0.0, 1.0)
    problem = core.TransportProblem(mesh)
    problem.diffusivity(0.01).initial_condition(_gaussian_2d(mesh, 0.6, 0.5, 0.15))
    problem.dirichlet(core.Boundary.Left, 0.0).dirichlet(core.Boundary.Right, 0.0)
    problem.neumann(core.Boundary.Bottom, 0.0).neumann(core.Boundary.Top, 0.0)
    return _transport_entries(
        core.solve_transport(problem, core.SolveOptions.until(0.2))
    )


@_case("transport_2d_variable_d_advection_decay", "solve_transport")
def _transport_2d_variable_d_advection_decay() -> Entries:
    mesh = core.StructuredMesh(12, 8, 0.0, 1.0, 0.0, 0.8)
    X, Y = _xy_grid(mesh)
    d_field = (0.004 + 0.006 * X + 0.003 * Y).ravel(order="C")
    problem = core.TransportProblem(mesh)
    problem.diffusivity_field(d_field).velocity(0.2, 0.1).linear_decay(0.3)
    problem.advection_scheme(core.AdvectionScheme.UPWIND)
    problem.initial_condition(_gaussian_2d(mesh, 0.3, 0.4, 0.1))
    problem.dirichlet(core.Boundary.Left, 0.0).neumann(core.Boundary.Right, 0.0)
    problem.neumann(core.Boundary.Bottom, 0.0).neumann(core.Boundary.Top, 0.0)
    return _transport_entries(
        core.solve_transport(problem, core.SolveOptions.until(0.3))
    )


@_case("transport_2d_velocity_field_robin", "solve_transport")
def _transport_2d_velocity_field_robin() -> Entries:
    mesh = core.StructuredMesh(10, 10, 0.0, 1.0, 0.0, 1.0)
    X, Y = _xy_grid(mesh)
    vx = (0.1 * (1.0 - Y)).ravel(order="C")
    vy = (0.05 * X).ravel(order="C")
    problem = core.TransportProblem(mesh)
    problem.diffusivity(0.01).velocity_field(vx, vy).initial_condition(0.3)
    problem.robin(core.Boundary.Left, 1.0, 0.2, 1.0).neumann(core.Boundary.Right, 0.0)
    problem.dirichlet(core.Boundary.Bottom, 0.3).neumann(core.Boundary.Top, 0.0)
    return _transport_entries(
        core.solve_transport(problem, core.SolveOptions.until(0.2))
    )


# ---------------------------------------------------------------------------
# ExplicitFD legacy facade
# ---------------------------------------------------------------------------


@_case("explicit_fd_1d_decay", "ExplicitFD")
def _explicit_fd_1d_decay() -> Entries:
    mesh = core.StructuredMesh(32, 0.0, 1.0)
    problem = core.TransportProblem(mesh)
    problem.diffusivity(0.01).linear_decay(0.5)
    problem.initial_condition(_gaussian_1d(mesh, 0.5, 0.1))
    problem.dirichlet(core.Boundary.Left, 0.0).neumann(core.Boundary.Right, 0.0)
    result = core.ExplicitFD().safety_factor(0.8).run(problem, 0.2)
    entries: Entries = {"solution": result.solution()}
    entries.update(_solver_stats(result.stats))
    return entries


@_case("explicit_fd_2d_logistic", "ExplicitFD")
def _explicit_fd_2d_logistic() -> Entries:
    mesh = core.StructuredMesh(10, 8, 0.0, 1.0, 0.0, 0.8)
    problem = core.TransportProblem(mesh)
    problem.diffusivity(0.01).logistic_growth(0.4, 1.0)
    problem.initial_condition(_gaussian_2d(mesh, 0.5, 0.4, 0.15))
    problem.neumann(core.Boundary.Left, 0.0).neumann(core.Boundary.Right, 0.0)
    problem.neumann(core.Boundary.Bottom, 0.0).neumann(core.Boundary.Top, 0.0)
    result = core.ExplicitFD().run(problem, 0.15)
    entries: Entries = {"solution": result.solution()}
    entries.update(_solver_stats(result.stats))
    return entries


# ---------------------------------------------------------------------------
# Explicit diffusion / reaction-diffusion / advection-diffusion classes
# ---------------------------------------------------------------------------


@_case("diffusion_solver_1d", "DiffusionSolver")
def _diffusion_solver_1d() -> Entries:
    mesh = core.StructuredMesh(32, 0.0, 1.0)
    solver = core.DiffusionSolver(mesh, 0.01)
    solver.set_initial_condition(_gaussian_1d(mesh, 0.5, 0.1))
    solver.set_dirichlet_boundary(core.Boundary.Left, 0.0)
    solver.set_neumann_boundary(core.Boundary.Right, 0.0)
    solver.solve(0.02, 25)
    return {"solution": solver.solution()}


@_case("diffusion_solver_2d", "DiffusionSolver")
def _diffusion_solver_2d() -> Entries:
    mesh = core.StructuredMesh(12, 10, 0.0, 1.2, 0.0, 1.0)
    solver = core.DiffusionSolver(mesh, 0.01)
    solver.set_initial_condition(_gaussian_2d(mesh, 0.6, 0.5, 0.15))
    solver.set_dirichlet_boundary(core.Boundary.Left, 1.0)
    solver.set_dirichlet_boundary(core.Boundary.Right, 0.0)
    solver.set_boundary_condition(
        core.Boundary.Bottom, core.BoundaryCondition.neumann(0.0)
    )
    solver.set_neumann_boundary(core.Boundary.Top, 0.5)
    solver.solve(0.1, 20)
    return {"solution": solver.solution()}


@_case("constant_source_rd_1d", "ConstantSourceReactionDiffusionSolver")
def _constant_source_rd_1d() -> Entries:
    mesh = core.StructuredMesh(32, 0.0, 1.0)
    solver = core.ConstantSourceReactionDiffusionSolver(mesh, 0.01, 0.2)
    solver.set_initial_condition(np.zeros(mesh.num_nodes()))
    solver.set_boundary(core.Boundary.Left, core.BoundaryCondition.dirichlet(0.0))
    solver.set_boundary(core.Boundary.Right, core.BoundaryCondition.dirichlet(0.0))
    solver.solve(0.02, 25)
    return {"solution": solver.solution()}


@_case("linear_rd_1d", "LinearReactionDiffusionSolver")
def _linear_rd_1d() -> Entries:
    mesh = core.StructuredMesh(32, 0.0, 1.0)
    solver = core.LinearReactionDiffusionSolver(mesh, 0.01, 0.5)
    solver.set_initial_condition(_gaussian_1d(mesh, 0.5, 0.1))
    solver.set_boundary(core.Boundary.Left, core.BoundaryCondition.neumann(0.0))
    solver.set_boundary(core.Boundary.Right, core.BoundaryCondition.dirichlet(0.0))
    solver.solve(0.02, 25)
    return {"solution": solver.solution()}


@_case("linear_rd_2d", "LinearReactionDiffusionSolver")
def _linear_rd_2d() -> Entries:
    mesh = core.StructuredMesh(10, 8, 0.0, 1.0, 0.0, 0.8)
    solver = core.LinearReactionDiffusionSolver(mesh, 0.01, 0.3)
    solver.set_initial_condition(_gaussian_2d(mesh, 0.5, 0.4, 0.15))
    solver.set_boundary(core.Boundary.Left, core.BoundaryCondition.dirichlet(0.0))
    solver.set_boundary(core.Boundary.Right, core.BoundaryCondition.dirichlet(0.0))
    solver.set_boundary(core.Boundary.Bottom, core.BoundaryCondition.neumann(0.0))
    solver.set_boundary(core.Boundary.Top, core.BoundaryCondition.neumann(0.0))
    solver.solve(0.1, 15)
    return {"solution": solver.solution()}


@_case("logistic_rd_1d", "LogisticReactionDiffusionSolver")
def _logistic_rd_1d() -> Entries:
    mesh = core.StructuredMesh(32, 0.0, 1.0)
    solver = core.LogisticReactionDiffusionSolver(mesh, 0.01, 0.5, 1.0)
    solver.set_initial_condition(0.2 * _gaussian_1d(mesh, 0.5, 0.1))
    solver.set_boundary(core.Boundary.Left, core.BoundaryCondition.neumann(0.0))
    solver.set_boundary(core.Boundary.Right, core.BoundaryCondition.neumann(0.0))
    solver.solve(0.02, 25)
    return {"solution": solver.solution()}


@_case("michaelis_menten_rd_1d", "MichaelisMentenReactionDiffusionSolver")
def _michaelis_menten_rd_1d() -> Entries:
    mesh = core.StructuredMesh(32, 0.0, 1.0)
    solver = core.MichaelisMentenReactionDiffusionSolver(mesh, 0.01, 0.4, 0.3)
    solver.set_initial_condition(np.ones(mesh.num_nodes()))
    solver.set_boundary(core.Boundary.Left, core.BoundaryCondition.dirichlet(1.0))
    solver.set_boundary(core.Boundary.Right, core.BoundaryCondition.neumann(0.0))
    solver.solve(0.02, 25)
    return {"solution": solver.solution()}


@_case("masked_michaelis_menten_rd_2d", "MaskedMichaelisMentenReactionDiffusionSolver")
def _masked_michaelis_menten_rd_2d() -> Entries:
    mesh = core.StructuredMesh(10, 8, 0.0, 1.0, 0.0, 0.8)
    X, Y = _xy_grid(mesh)
    mask = (((X - 0.5) ** 2 + (Y - 0.4) ** 2) <= 0.25**2).astype(np.int64)
    solver = core.MaskedMichaelisMentenReactionDiffusionSolver(
        mesh, 0.01, 0.5, 0.2, mask.ravel(order="C").tolist(), 1.0
    )
    solver.set_initial_condition(np.ones(mesh.num_nodes()))
    solver.set_boundary(core.Boundary.Left, core.BoundaryCondition.dirichlet(1.0))
    solver.set_boundary(core.Boundary.Right, core.BoundaryCondition.dirichlet(1.0))
    solver.set_boundary(core.Boundary.Bottom, core.BoundaryCondition.dirichlet(1.0))
    solver.set_boundary(core.Boundary.Top, core.BoundaryCondition.dirichlet(1.0))
    solver.solve(0.1, 15)
    return {"solution": solver.solution()}


@_case("reaction_diffusion_callback_1d", "ReactionDiffusionSolver")
def _reaction_diffusion_callback_1d() -> Entries:
    mesh = core.StructuredMesh(24, 0.0, 1.0)
    solver = core.ReactionDiffusionSolver(
        mesh, 0.01, lambda u, x, y, t: -0.5 * u + 0.1 * x * math.cos(t)
    )
    solver.set_initial_condition(_gaussian_1d(mesh, 0.5, 0.15))
    solver.set_dirichlet_boundary(core.Boundary.Left, 0.0)
    solver.set_neumann_boundary(core.Boundary.Right, 0.0)
    solver.solve(0.02, 20)
    return {"solution": solver.solution()}


@_case("advection_diffusion_upwind_1d", "AdvectionDiffusionSolver")
def _advection_diffusion_upwind_1d() -> Entries:
    mesh = core.StructuredMesh(40, 0.0, 1.0)
    solver = core.AdvectionDiffusionSolver(
        mesh, 0.002, 0.5, 0.0, core.AdvectionScheme.UPWIND
    )
    solver.set_initial_condition(_gaussian_1d(mesh, 0.25, 0.05))
    solver.set_boundary(core.Boundary.Left, core.BoundaryCondition.dirichlet(0.0))
    solver.set_boundary(core.Boundary.Right, core.BoundaryCondition.neumann(0.0))
    solver.solve(0.01, 30)
    return {
        "solution": solver.solution(),
        "cell_peclet": solver.cell_peclet(),
        "max_time_step": solver.max_time_step(0.9),
        "is_scheme_stable": solver.is_scheme_stable(),
    }


@_case("advection_diffusion_upwind_2d_field", "AdvectionDiffusionSolver")
def _advection_diffusion_upwind_2d_field() -> Entries:
    mesh = core.StructuredMesh(12, 10, 0.0, 1.2, 0.0, 1.0)
    X, Y = _xy_grid(mesh)
    vx = (0.3 * (1.0 - Y) * Y * 4.0).ravel(order="C")
    vy = (0.05 * X).ravel(order="C")
    solver = core.AdvectionDiffusionSolver(
        mesh, 0.005, vx, vy, core.AdvectionScheme.UPWIND
    )
    solver.set_initial_condition(_gaussian_2d(mesh, 0.3, 0.5, 0.1))
    solver.set_boundary(core.Boundary.Left, core.BoundaryCondition.dirichlet(0.0))
    solver.set_boundary(core.Boundary.Right, core.BoundaryCondition.neumann(0.0))
    solver.set_boundary(core.Boundary.Bottom, core.BoundaryCondition.neumann(0.0))
    solver.set_boundary(core.Boundary.Top, core.BoundaryCondition.neumann(0.0))
    solver.solve(0.02, 20)
    return {"solution": solver.solution(), "cell_peclet": solver.cell_peclet()}


# ---------------------------------------------------------------------------
# 3D explicit solvers
# ---------------------------------------------------------------------------


@_case("diffusion_3d", "DiffusionSolver3D")
def _diffusion_3d() -> Entries:
    mesh = core.StructuredMesh3D(5, 4, 3, 0.0, 1.0, 0.0, 0.8, 0.0, 0.6)
    solver = core.DiffusionSolver3D(mesh, 0.01)
    solver.set_initial_condition(_gaussian_3d(mesh, 0.15))
    solver.set_dirichlet_boundary(core.Boundary3D.XMin, 0.0)
    solver.set_dirichlet_boundary(core.Boundary3D.XMax, 0.0)
    solver.set_neumann_boundary(core.Boundary3D.YMin, 0.0)
    solver.set_neumann_boundary(core.Boundary3D.YMax, 0.0)
    solver.set_neumann_boundary(core.Boundary3D.ZMin, 0.2)
    solver.set_dirichlet_boundary(core.Boundary3D.ZMax, 0.0)
    solver.solve(0.2, 12)
    return {
        "solution": solver.solution(),
        "time": solver.time(),
        "max_stable_time_step": solver.max_stable_time_step(),
    }


@_case("linear_rd_3d", "LinearReactionDiffusionSolver3D")
def _linear_rd_3d() -> Entries:
    mesh = core.StructuredMesh3D(4, 0.8)
    solver = core.LinearReactionDiffusionSolver3D(mesh, 0.01, 0.4)
    solver.set_initial_condition(_gaussian_3d(mesh, 0.2))
    solver.set_neumann_boundary(core.Boundary3D.XMin, 0.0)
    solver.set_neumann_boundary(core.Boundary3D.XMax, 0.0)
    solver.set_dirichlet_boundary(core.Boundary3D.YMin, 0.0)
    solver.set_dirichlet_boundary(core.Boundary3D.YMax, 0.0)
    solver.set_neumann_boundary(core.Boundary3D.ZMin, 0.0)
    solver.set_neumann_boundary(core.Boundary3D.ZMax, 0.0)
    solver.solve(0.25, 12)
    return {
        "solution": solver.solution(),
        "time": solver.time(),
        "decay_rate": solver.decay_rate(),
        "max_stable_time_step": solver.max_stable_time_step(),
    }


# ---------------------------------------------------------------------------
# Implicit / semi-implicit diffusion
# ---------------------------------------------------------------------------


@_case("crank_nicolson_1d", "CrankNicolsonDiffusion")
def _crank_nicolson_1d() -> Entries:
    mesh = core.StructuredMesh(32, 0.0, 1.0)
    solver = core.CrankNicolsonDiffusion(mesh, 0.01)
    solver.set_initial_condition(_gaussian_1d(mesh, 0.5, 0.1))
    solver.set_dirichlet_boundary(core.Boundary.Left, 0.0)
    solver.set_neumann_boundary(core.Boundary.Right, 0.0)
    solver.set_tolerance(1e-12).set_max_iterations(2000)
    step = solver.step(0.05)
    solver.solve(0.05, 9)
    return {
        "solution": solver.solution(),
        "time": solver.time(),
        "step_iterations": step.iterations,
        "step_residual": step.residual,
        "step_converged": step.converged,
    }


@_case("crank_nicolson_2d", "CrankNicolsonDiffusion")
def _crank_nicolson_2d() -> Entries:
    mesh = core.StructuredMesh(10, 8, 0.0, 1.0, 0.0, 0.8)
    solver = core.CrankNicolsonDiffusion(mesh, 0.01)
    solver.set_initial_condition(_gaussian_2d(mesh, 0.5, 0.4, 0.15))
    solver.set_dirichlet_boundary(core.Boundary.Left, 0.0)
    solver.set_dirichlet_boundary(core.Boundary.Right, 0.0)
    solver.set_neumann_boundary(core.Boundary.Bottom, 0.0)
    solver.set_neumann_boundary(core.Boundary.Top, 0.0)
    solver.set_tolerance(1e-12).set_max_iterations(5000)
    step = solver.step(0.1)
    solver.solve(0.1, 4)
    return {
        "solution": solver.solution(),
        "time": solver.time(),
        "step_iterations": step.iterations,
        "step_residual": step.residual,
        "step_converged": step.converged,
    }


@_case("adi_2d", "ADIDiffusion2D")
def _adi_2d() -> Entries:
    mesh = core.StructuredMesh(12, 10, 0.0, 1.2, 0.0, 1.0)
    solver = core.ADIDiffusion2D(mesh, 0.01)
    solver.set_initial_condition(_gaussian_2d(mesh, 0.6, 0.5, 0.15))
    solver.set_dirichlet_boundary(core.Boundary.Left, 0.0)
    solver.set_dirichlet_boundary(core.Boundary.Right, 0.5)
    solver.set_neumann_boundary(core.Boundary.Bottom, 0.0)
    solver.set_neumann_boundary(core.Boundary.Top, 0.0)
    entries: Entries = {}
    entries.update(_adi_result("step", solver.step(0.1)))
    entries.update(_adi_result("solve", solver.solve(0.1, 9)))
    entries["solution"] = solver.solution()
    entries["time"] = solver.time()
    return entries


@_case("adi_3d", "ADIDiffusion3D")
def _adi_3d() -> Entries:
    mesh = core.StructuredMesh3D(5, 4, 3, 0.0, 1.0, 0.0, 0.8, 0.0, 0.6)
    solver = core.ADIDiffusion3D(mesh, 0.01)
    solver.set_initial_condition(_gaussian_3d(mesh, 0.15))
    solver.set_dirichlet_boundary(core.Boundary3D.XMin, 0.0)
    solver.set_dirichlet_boundary(core.Boundary3D.XMax, 0.0)
    solver.set_neumann_boundary(core.Boundary3D.YMin, 0.0)
    solver.set_neumann_boundary(core.Boundary3D.YMax, 0.0)
    solver.set_neumann_boundary(core.Boundary3D.ZMin, 0.0)
    solver.set_dirichlet_boundary(core.Boundary3D.ZMax, 0.0)
    entries: Entries = {}
    entries.update(_adi_result("step", solver.step(0.2)))
    entries.update(_adi_result("solve", solver.solve(0.2, 5)))
    entries["solution"] = solver.solution()
    entries["time"] = solver.time()
    return entries


@_case("implicit_2d_source", "ImplicitDiffusion2D", requires_sparse=True)
def _implicit_2d_source() -> Entries:
    mesh = core.StructuredMesh(10, 8, 0.0, 1.0, 0.0, 0.8)
    X, Y = _xy_grid(mesh)
    d_field = (0.005 + 0.01 * X * Y).ravel(order="C")
    solver = core.ImplicitDiffusion2D(mesh, d_field)
    solver.set_initial_condition(_gaussian_2d(mesh, 0.5, 0.4, 0.15))
    solver.set_dirichlet_boundary(core.Boundary.Left, 0.0)
    solver.set_neumann_boundary(core.Boundary.Right, 0.0)
    solver.set_neumann_boundary(core.Boundary.Bottom, 0.0)
    solver.set_neumann_boundary(core.Boundary.Top, 0.1)
    solver.set_source_term(lambda x, y, t: 0.1 * x * (1.0 + t))
    solver.set_solver_type(core.SparseSolverType.SparseLU)
    entries: Entries = {}
    entries.update(_implicit_result("step", solver.step(0.1)))
    entries.update(_implicit_result("solve", solver.solve(0.1, 4)))
    entries["solution"] = solver.solution()
    entries["diffusivity"] = solver.diffusivity()
    entries["time"] = solver.time()
    return entries


@_case("implicit_2d_uniform_cg", "ImplicitDiffusion2D", requires_sparse=True)
def _implicit_2d_uniform_cg() -> Entries:
    mesh = core.StructuredMesh(8, 8, 0.0, 1.0, 0.0, 1.0)
    solver = core.ImplicitDiffusion2D(mesh, 0.02)
    solver.set_initial_condition(_gaussian_2d(mesh, 0.5, 0.5, 0.15))
    solver.set_dirichlet_boundary(core.Boundary.Left, 0.0)
    solver.set_dirichlet_boundary(core.Boundary.Right, 0.0)
    solver.set_dirichlet_boundary(core.Boundary.Bottom, 0.0)
    solver.set_dirichlet_boundary(core.Boundary.Top, 0.0)
    solver.set_solver_type(core.SparseSolverType.ConjugateGradient)
    solver.set_tolerance(1e-12)
    solver.set_max_iterations(2000)
    entries: Entries = {}
    entries.update(_implicit_result("solve", solver.solve(0.05, 6)))
    entries["solution"] = solver.solution()
    entries["time"] = solver.time()
    return entries


@_case("implicit_3d", "ImplicitDiffusion3D", requires_sparse=True)
def _implicit_3d() -> Entries:
    mesh = core.StructuredMesh3D(4, 3, 3, 0.0, 0.8, 0.0, 0.6, 0.0, 0.6)
    solver = core.ImplicitDiffusion3D(mesh, 0.01)
    solver.set_initial_condition(_gaussian_3d(mesh, 0.15))
    solver.set_dirichlet_boundary(core.Boundary3D.XMin, 0.0)
    solver.set_dirichlet_boundary(core.Boundary3D.XMax, 0.0)
    solver.set_neumann_boundary(core.Boundary3D.YMin, 0.0)
    solver.set_neumann_boundary(core.Boundary3D.YMax, 0.0)
    solver.set_neumann_boundary(core.Boundary3D.ZMin, 0.0)
    solver.set_neumann_boundary(core.Boundary3D.ZMax, 0.0)
    solver.set_source_term(lambda x, y, z, t: 0.05 * z)
    entries: Entries = {}
    entries.update(_implicit_result("step", solver.step(0.2)))
    entries.update(_implicit_result("solve", solver.solve(0.2, 3)))
    entries["solution"] = solver.solution()
    entries["diffusivity"] = solver.diffusivity()
    entries["time"] = solver.time()
    return entries


# ---------------------------------------------------------------------------
# Multi-species reaction-diffusion
# ---------------------------------------------------------------------------


@_case("multi_species_lotka_volterra", "MultiSpeciesSolver")
def _multi_species_lotka_volterra() -> Entries:
    mesh = core.StructuredMesh(20, 0.0, 10.0)
    solver = core.MultiSpeciesSolver(mesh, [0.01, 0.02])
    solver.set_reaction_model(core.LotkaVolterraReaction(1.0, 0.1, 0.1, 0.02, 100.0))
    prey = 40.0 + 5.0 * _gaussian_1d(mesh, 5.0, 1.0)
    solver.set_initial_condition(0, prey)
    solver.set_uniform_initial_condition(1, 9.0)
    solver.set_all_species_neumann(core.Boundary.Left, 0.0)
    solver.set_all_species_neumann(core.Boundary.Right, 0.0)
    solver.solve(0.01, 20)
    solver.solve_until(0.5, 0.05)
    solutions = solver.all_solutions()
    return {
        "prey": solutions[0],
        "predator": solutions[1],
        "time": solver.time(),
        "max_stable_time_step": solver.max_stable_time_step(),
        "total_mass_prey": solver.total_mass(0),
        "total_mass_predator": solver.total_mass(1),
        "solution_norm_prey": solver.solution_norm(0),
    }


@_case("multi_species_callback_2d", "MultiSpeciesSolver")
def _multi_species_callback_2d() -> Entries:
    mesh = core.StructuredMesh(6, 5, 0.0, 1.2, 0.0, 1.0)
    solver = core.MultiSpeciesSolver(mesh, [0.01, 0.005, 0.0])

    def reaction(rates, concentrations, x, y, t):
        a, b, c = concentrations
        rates[0] = -0.5 * a * b
        rates[1] = -0.5 * a * b
        rates[2] = 0.5 * a * b + 0.1 * x
        return None

    solver.set_reaction_function(reaction)
    solver.set_initial_condition(0, _gaussian_2d(mesh, 0.4, 0.5, 0.2))
    solver.set_uniform_initial_condition(1, 1.0)
    solver.set_uniform_initial_condition(2, 0.0)
    solver.set_dirichlet_boundary(0, core.Boundary.Left, 1.0)
    solver.set_neumann_boundary(0, core.Boundary.Right, 0.0)
    solver.set_all_species_neumann(core.Boundary.Bottom, 0.0)
    solver.set_all_species_neumann(core.Boundary.Top, 0.0)
    solver.set_dirichlet_boundary(1, 2, 1.0)  # Bottom by integer id
    solver.solve(0.05, 10)
    solutions = solver.all_solutions()
    return {
        "species_0": solutions[0],
        "species_1": solutions[1],
        "species_2": solutions[2],
        "time": solver.time(),
        "total_concentration_center": solver.total_concentration(mesh.index(3, 2)),
    }


# ---------------------------------------------------------------------------
# Electrochemistry
# ---------------------------------------------------------------------------


@_case("nernst_planck_uniform_field", "NernstPlanckSolver")
def _nernst_planck_uniform_field() -> Entries:
    mesh = core.StructuredMesh(16, 0.0, 1e-3)
    ion = core.IonSpecies("Na+", 1, 1.33e-9)
    solver = core.NernstPlanckSolver(mesh, ion, 310.0)
    solver.set_initial_condition(100.0 + 40.0 * _gaussian_1d(mesh, 5e-4, 1.5e-4))
    solver.set_uniform_field(1000.0)
    solver.set_dirichlet_boundary(core.Boundary.Left, 100.0)
    solver.set_outward_flux_boundary(core.Boundary.Right, 0.0)
    solver.solve(0.1, 20)
    return {
        "concentration": solver.solution(),
        "potential": solver.potential(),
        "current_density": solver.compute_current_density(),
        "time": solver.time(),
        "thermal_voltage": solver.thermal_voltage(),
        "electrical_mobility": solver.electrical_mobility(),
        "maximum_stable_time_step": solver.maximum_stable_time_step(),
        "recommended_time_step": solver.recommended_time_step(0.9),
    }


@_case("nernst_planck_potential_field_2d", "NernstPlanckSolver")
def _nernst_planck_potential_field_2d() -> Entries:
    mesh = core.StructuredMesh(8, 6, 0.0, 1e-3, 0.0, 0.75e-3)
    X, Y = _xy_grid(mesh)
    solver = core.NernstPlanckSolver(mesh, core.IonSpecies("Cl-", -1, 2.03e-9), 298.15)
    solver.set_initial_condition(np.full(mesh.num_nodes(), 10.0))
    solver.set_potential_field(
        (0.02 * X / 1e-3 + 0.01 * (Y / 0.75e-3) ** 2).ravel(order="C")
    )
    solver.set_dirichlet_boundary(0, 10.0)
    solver.set_dirichlet_boundary(1, 12.0)
    solver.set_outward_flux_boundary(core.Boundary.Bottom, 0.0)
    solver.set_outward_flux_boundary(core.Boundary.Top, 0.0)
    solver.solve(0.2, 15)
    return {
        "concentration": solver.solution(),
        "potential": solver.potential(),
        "current_density": solver.compute_current_density(),
        "time": solver.time(),
    }


@_case("multi_ion_uniform_field", "MultiIonSolver")
def _multi_ion_uniform_field() -> Entries:
    mesh = core.StructuredMesh(16, 0.0, 1e-3)
    ions = [
        core.IonSpecies("Na+", 1, 1.33e-9),
        core.IonSpecies("Cl-", -1, 2.03e-9),
    ]
    solver = core.MultiIonSolver(mesh, ions, 310.0)
    solver.set_initial_condition(0, 140.0 + 10.0 * _gaussian_1d(mesh, 5e-4, 1.5e-4))
    solver.set_initial_condition(1, np.full(mesh.num_nodes(), 140.0))
    solver.set_uniform_field(500.0)
    solver.set_dirichlet_boundary(0, core.Boundary.Left, 140.0)
    solver.set_dirichlet_boundary(0, 1, 140.0)
    solver.set_dirichlet_boundary(1, core.Boundary.Left, 140.0)
    solver.set_outward_flux_boundary(1, core.Boundary.Right, 0.0)
    solver.solve(0.1, 20)
    return {
        "sodium": solver.concentration(0),
        "chloride": solver.concentration(1),
        "potential": solver.potential(),
        "charge_density": solver.charge_density(),
        "time": solver.time(),
        "maximum_stable_time_step": solver.maximum_stable_time_step(),
        "recommended_time_step": solver.recommended_time_step(0.9),
        "electrical_mobility_0": solver.electrical_mobility(0),
        "electrical_mobility_1": solver.electrical_mobility(1),
    }


# ---------------------------------------------------------------------------
# Nonuniform 1D diffusion
# ---------------------------------------------------------------------------


@_case("nonuniform_diffusion_1d", "NonuniformDiffusion1D")
def _nonuniform_diffusion_1d() -> Entries:
    nodes = [0.0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0]
    mesh = core.NonuniformMesh1D(nodes)
    solver = core.NonuniformDiffusion1D(mesh, [0.1, 0.1, 0.2, 0.2, 0.05, 0.05, 0.1])
    solver.set_initial_condition([1.0, 0.8, 0.5, 0.2, 0.1, 0.0, 0.0])
    solver.set_dirichlet_boundary(core.Boundary.Left, 1.0)
    solver.set_neumann_boundary(core.Boundary.Right, 0.0)
    solver.step(0.001)
    solver.solve(0.001, 4)
    solver.solve_until(0.05, 0.002)
    diagnostics = solver.diagnostics()
    return {
        "solution": solver.solution(),
        "diffusivity": solver.diffusivity(),
        "face_diffusivities": solver.face_diffusivities(),
        "face_fluxes": solver.face_fluxes(),
        "time": solver.time(),
        "steps": solver.steps(),
        "total_mass": solver.total_mass(),
        "max_stable_time_step": solver.max_stable_time_step(),
        "left_outward_flux": solver.boundary_outward_flux(core.Boundary.Left),
        "right_outward_flux": solver.boundary_outward_flux(core.Boundary.Right),
        "diag_steps": diagnostics.steps,
        "diag_reference_time": diagnostics.reference_time,
        "diag_time": diagnostics.time,
        "diag_stability_limit": diagnostics.stability_limit,
        "diag_reference_mass": diagnostics.reference_mass,
        "diag_total_mass": diagnostics.total_mass,
        "diag_cumulative_boundary_input": diagnostics.cumulative_boundary_input,
        "diag_mass_balance_error": diagnostics.mass_balance_error,
        "diag_minimum_concentration": diagnostics.minimum_concentration,
        "diag_maximum_concentration": diagnostics.maximum_concentration,
        "diag_left_outward_flux": diagnostics.left_outward_flux,
        "diag_right_outward_flux": diagnostics.right_outward_flux,
    }


# ---------------------------------------------------------------------------
# Pattern formation
# ---------------------------------------------------------------------------


@_case("gray_scott_small", "GrayScottSolver")
def _gray_scott_small() -> Entries:
    mesh = core.StructuredMesh(16, 12, 0.0, 1.6, 0.0, 1.2)
    nx, ny = mesh.nx(), mesh.ny()
    u0 = np.ones((ny, nx), dtype=np.float64)
    v0 = np.zeros((ny, nx), dtype=np.float64)
    u0[4:8, 6:10] = 0.5
    v0[4:8, 6:10] = 0.25
    v0[5:7, 7:9] = 0.35
    solver = core.GrayScottSolver(mesh, 0.16, 0.08, 0.035, 0.065)
    result = solver.simulate(
        u0.ravel(order="C"),
        v0.ravel(order="C"),
        total_steps=40,
        dt=0.01,
        steps_between_frames=10,
        check_interval=1000,
        stable_tol=1e-4,
        min_frames_before_early_stop=6,
    )
    return {
        "u_frames": result.u_frames(),
        "v_frames": result.v_frames(),
        "frame_steps": np.asarray(result.frame_steps, dtype=np.int64),
        "steps_run": result.steps_run,
        "frames": result.frames,
        "final_time": result.final_time,
        "nx": result.nx,
        "ny": result.ny,
    }


# ---------------------------------------------------------------------------
# Application solvers (configs supply documented illustrative defaults)
# ---------------------------------------------------------------------------


@_case("tumor_drug_delivery_small", "TumorDrugDeliverySolver")
def _tumor_drug_delivery_small() -> Entries:
    config = biotransport.TumorDrugDeliveryConfig(
        domain_size=5e-3,
        tumor_radius=1.25e-3,
        rim_thickness=0.5e-3,
        nx=8,
        ny=8,
    )
    L = config.domain_size
    mesh = core.StructuredMesh(config.nx, config.ny, 0.0, L, 0.0, L)
    assert config.tumor_center is not None
    cx, cy = config.tumor_center
    X, Y = _xy_grid(mesh)
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask_tumor = dist <= config.tumor_radius
    mask_rim = mask_tumor & (dist > (config.tumor_radius - config.rim_thickness))
    mask_core = mask_tumor & ~mask_rim

    surface_area = np.full(X.shape, config.vascular_surface_area_normal)
    surface_area[mask_rim] = config.vascular_surface_area_tumor_rim
    surface_area[mask_core] = config.vascular_surface_area_tumor_core
    permeability = np.full(X.shape, config.P_vessel_normal)
    permeability[mask_tumor] = config.P_vessel_tumor
    diffusivity = np.full(X.shape, config.D_drug_normal)
    diffusivity[mask_tumor] = config.D_drug_tumor
    conductivity = np.full(X.shape, config.K_hydraulic_normal)
    conductivity[mask_tumor] = config.K_hydraulic_tumor

    solver = core.TumorDrugDeliverySolver(
        mesh,
        mask_tumor.astype(np.uint8).ravel(order="C").tolist(),
        conductivity.ravel(order="C"),
        config.IFP_normal_Pa,
        config.IFP_tumor_Pa,
    )
    pressure = solver.solve_pressure_sor(max_iter=20000, tol=1e-10, omega=1.5)
    saved = solver.simulate(
        pressure,
        diffusivity.ravel(order="C"),
        permeability.ravel(order="C"),
        surface_area.ravel(order="C"),
        config.k_binding,
        config.k_uptake,
        config.C_plasma,
        1.0,
        20,
        [0.0, 7.5, 20.0],
    )
    return {
        "pressure": np.asarray(pressure, dtype=np.float64),
        "free": saved.free(),
        "bound": saved.bound(),
        "cellular": saved.cellular(),
        "total": saved.total(),
        "times_s": np.asarray(saved.times_s, dtype=np.float64),
        "frames": saved.frames,
        "final_time_s": saved.final_time_s,
        "stability_limit_s": saved.stability_limit_s,
        "free_amount_per_depth": np.asarray(saved.free_amount_per_depth),
        "bound_amount_per_depth": np.asarray(saved.bound_amount_per_depth),
        "cellular_amount_per_depth": np.asarray(saved.cellular_amount_per_depth),
        "total_amount_per_depth": np.asarray(saved.total_amount_per_depth),
        "cumulative_net_vascular_exchange_per_depth": np.asarray(
            saved.cumulative_net_vascular_exchange_per_depth
        ),
        "cumulative_boundary_outflow_per_depth": np.asarray(
            saved.cumulative_boundary_outflow_per_depth
        ),
        "mass_balance_error_per_depth": np.asarray(saved.mass_balance_error_per_depth),
    }


@_case("bioheat_cryotherapy_small", "BioheatCryotherapySolver")
def _bioheat_cryotherapy_small() -> Entries:
    config = biotransport.BioheatCryotherapyConfig(
        domain_size_x=0.04,
        domain_size_y=0.04,
        nx=8,
        ny=8,
        probe_radius=2.5e-3,
        tumor_radius=0.01,
        dt=0.05,
    )
    config.validate()
    mesh = core.StructuredMesh(
        config.nx, config.ny, 0.0, config.domain_size_x, 0.0, config.domain_size_y
    )
    X, Y = _xy_grid(mesh)
    assert config.probe_position is not None
    assert config.tumor_center is not None
    px, py = config.probe_position
    tx, ty = config.tumor_center
    probe_mask = (X - px) ** 2 + (Y - py) ** 2 <= config.probe_radius**2
    tumor_mask = (X - tx) ** 2 + (Y - ty) ** 2 <= config.tumor_radius**2
    perfusion = np.where(tumor_mask, config.w_b_tumor, config.w_b_normal)
    q_met = np.where(tumor_mask, config.q_met_tumor, config.q_met_normal)
    solver = config.create_solver(
        mesh,
        probe_mask=probe_mask.astype(np.uint8).ravel(order="C").tolist(),
        perfusion_map=perfusion.astype(np.float64).ravel(order="C").tolist(),
        q_met_map=q_met.astype(np.float64).ravel(order="C").tolist(),
    )
    saved = solver.simulate(config.dt, 20, [0.0, 0.5, 1.0])
    return {
        "temperature_K": saved.temperature_K(),
        "damage": saved.damage(),
        "frozen_fraction": saved.frozen_fraction(),
        "times_s": np.asarray(saved.times_s, dtype=np.float64),
        "minimum_temperature_K": np.asarray(saved.minimum_temperature_K),
        "maximum_temperature_K": np.asarray(saved.maximum_temperature_K),
        "maximum_stable_dt_s": saved.maximum_stable_dt_s,
        "frames": saved.frames,
        "probe_mask_count": int(probe_mask.sum()),
        "solver_max_stable_dt_s": solver.maximum_stable_time_step_s(),
    }


# ---------------------------------------------------------------------------
# Flow solvers
# ---------------------------------------------------------------------------


@_case("darcy_uniform_kappa", "DarcyFlowSolver")
def _darcy_uniform_kappa() -> Entries:
    mesh = core.StructuredMesh(8, 6, 0.0, 0.01, 0.0, 0.006)
    solver = core.DarcyFlowSolver(mesh, 1e-12)
    solver.set_dirichlet(core.Boundary.Left, 1000.0)
    solver.set_dirichlet(core.Boundary.Right, 0.0)
    solver.set_outward_pressure_gradient(core.Boundary.Bottom, 0.0)
    solver.set_neumann(core.Boundary.Top, 0.0)
    solver.set_omega(1.5).set_tolerance(1e-10).set_max_iterations(20000)
    result = solver.solve()
    return {
        "pressure": result.pressure(),
        "vx": result.vx(),
        "vy": result.vy(),
        "converged": result.converged,
        "iterations": result.iterations,
        "residual": result.residual,
    }


@_case("darcy_heterogeneous_kappa_internal_pressure", "DarcyFlowSolver")
def _darcy_heterogeneous_kappa_internal_pressure() -> Entries:
    mesh = core.StructuredMesh(8, 8, 0.0, 0.01, 0.0, 0.01)
    X, Y = _xy_grid(mesh)
    kappa = np.where(X < 0.005, 1e-12, 4e-12).ravel(order="C")
    mask = (((X - 0.005) ** 2 + (Y - 0.005) ** 2) <= 0.0015**2).astype(np.uint8)
    solver = core.DarcyFlowSolver(mesh, kappa)
    solver.set_dirichlet(core.Boundary.Left, 0.0)
    solver.set_dirichlet(core.Boundary.Right, 0.0)
    solver.set_dirichlet(core.Boundary.Bottom, 0.0)
    solver.set_dirichlet(core.Boundary.Top, 0.0)
    solver.set_internal_pressure(mask.ravel(order="C").tolist(), 500.0)
    solver.set_initial_guess(np.zeros(mesh.num_nodes()))
    solver.set_omega(1.6).set_tolerance(1e-10).set_max_iterations(20000)
    result = solver.solve()
    return {
        "kappa": solver.kappa(),
        "pressure": result.pressure(),
        "vx": result.vx(),
        "vy": result.vy(),
        "converged": result.converged,
        "iterations": result.iterations,
        "residual": result.residual,
    }


@_case("stokes_poiseuille_body_force", "StokesSolver")
def _stokes_poiseuille_body_force() -> Entries:
    mesh = core.StructuredMesh(8, 6, 0.0, 1.0, 0.0, 0.1)
    solver = core.StokesSolver(mesh, 0.001)
    solver.set_velocity_bc(core.Boundary.Bottom, core.VelocityBC.no_slip())
    solver.set_velocity_bc(core.Boundary.Top, core.VelocityBC.no_slip())
    solver.set_velocity_bc(core.Boundary.Left, core.VelocityBC.outflow())
    solver.set_velocity_bc(core.Boundary.Right, core.VelocityBC.outflow())
    solver.set_body_force(1000.0, 0.0)
    solver.set_tolerance(1e-6).set_max_iterations(20000)
    result = solver.solve()
    return {
        "u": result.u(),
        "v": result.v(),
        "pressure": result.pressure(),
        "divergence": result.divergence,
        "converged": result.converged,
        "iterations": result.iterations,
        "residual": result.residual,
        "reynolds": solver.reynolds(0.1, 1.0, 1000.0),
    }


@_case("stokes_lid_driven", "StokesSolver")
def _stokes_lid_driven() -> Entries:
    mesh = core.StructuredMesh(6, 6, 0.0, 1.0, 0.0, 1.0)
    solver = core.StokesSolver(mesh, 0.1)
    solver.set_velocity_bc(core.Boundary.Bottom, core.VelocityBC.no_slip())
    solver.set_velocity_bc(core.Boundary.Left, core.VelocityBC.no_slip())
    solver.set_velocity_bc(core.Boundary.Right, core.VelocityBC.no_slip())
    solver.set_velocity_bc(core.Boundary.Top, core.VelocityBC.dirichlet(1.0, 0.0))
    solver.set_body_force(0.0, -1.0)
    solver.set_velocity_relaxation(0.7).set_pressure_relaxation(0.3)
    solver.set_tolerance(1e-6).set_max_iterations(5000)
    result = solver.solve()
    return {
        "u": result.u(),
        "v": result.v(),
        "pressure": result.pressure(),
        "divergence": result.divergence,
        "converged": result.converged,
        "iterations": result.iterations,
        "residual": result.residual,
    }


def _navier_stokes_entries(result: core.NavierStokesResult) -> Entries:
    return {
        "u": result.u(),
        "v": result.v(),
        "pressure": result.pressure(),
        "time": result.time,
        "time_steps": result.time_steps,
        "reynolds": result.reynolds,
        "max_velocity": result.max_velocity,
        "pressure_iterations": result.pressure_iterations,
        "pressure_residual": result.pressure_residual,
        "divergence": result.divergence,
        "stable": result.stable,
    }


@_case("navier_stokes_lid_driven_duration", "NavierStokesSolver")
def _navier_stokes_lid_driven_duration() -> Entries:
    mesh = core.StructuredMesh(6, 5, 0.0, 1.0, 0.0, 0.8)
    solver = core.NavierStokesSolver(mesh, 1.0, 0.05)
    solver.set_velocity_bc(core.Boundary.Top, core.VelocityBC.dirichlet(0.1, 0.0))
    solver.set_velocity_bc(core.Boundary.Bottom, core.VelocityBC.no_slip())
    solver.set_velocity_bc(core.Boundary.Left, core.VelocityBC.no_slip())
    solver.set_velocity_bc(core.Boundary.Right, core.VelocityBC.no_slip())
    solver.set_convection_scheme(core.ConvectionScheme.UPWIND)
    solver.set_time_step(0.002).set_pressure_tolerance(1e-10)
    solver.set_max_pressure_iterations(5000)
    return _navier_stokes_entries(solver.solve(0.011))


@_case("navier_stokes_uniform_steps", "NavierStokesSolver")
def _navier_stokes_uniform_steps() -> Entries:
    mesh = core.StructuredMesh(7, 5, -0.4, 1.4, 0.2, 1.25)
    solver = core.NavierStokesSolver(mesh, 1.3, 0.07)
    boundary = core.VelocityBC.dirichlet(0.23, -0.17)
    for side in (
        core.Boundary.Left,
        core.Boundary.Right,
        core.Boundary.Bottom,
        core.Boundary.Top,
    ):
        solver.set_velocity_bc(side, boundary)
    n = mesh.num_nodes()
    solver.set_initial_velocity(np.full(n, 0.23), np.full(n, -0.17))
    solver.set_body_force(0.0, -0.5)
    solver.set_convection_scheme(core.ConvectionScheme.CENTRAL)
    solver.set_time_step(0.001).set_pressure_tolerance(1e-12)
    return _navier_stokes_entries(solver.solve_steps(5))


# ---------------------------------------------------------------------------
# Membrane transport
# ---------------------------------------------------------------------------


@_case("membrane_1d_hindered", "MembraneDiffusion1DSolver")
def _membrane_1d_hindered() -> Entries:
    solver = (
        core.MembraneDiffusion1DSolver()
        .set_membrane_thickness(1e-4)
        .set_diffusivity(1e-10)
        .set_partition_coefficient(0.6)
        .set_left_concentration(10.0)
        .set_right_concentration(1.0)
        .set_num_nodes(41)
        .set_hindered_diffusion(1e-9, 5e-9)
    )
    entries = _membrane_result(solver.solve())
    entries["compute_flux"] = solver.compute_flux()
    entries["compute_permeability"] = solver.compute_permeability()
    entries["lambda_ratio"] = solver.lambda_ratio()
    return entries


@_case("membrane_1d_bulk", "MembraneDiffusion1DSolver")
def _membrane_1d_bulk() -> Entries:
    solver = (
        core.MembraneDiffusion1DSolver()
        .set_membrane_thickness(5e-5)
        .set_diffusivity(2e-10)
        .set_partition_coefficient(1.0)
        .set_left_concentration(1.0)
        .set_right_concentration(0.0)
        .set_num_nodes(21)
    )
    return _membrane_result(solver.solve())


@_case("membrane_multilayer", "MultiLayerMembraneSolver")
def _membrane_multilayer() -> Entries:
    solver = core.MultiLayerMembraneSolver()
    solver.add_layer(1e-5, 1e-11, 0.5)
    solver.add_layer(5e-5, 1e-10, 1.0)
    solver.add_layer(2e-5, 5e-11)
    solver.set_left_concentration(1.0).set_right_concentration(0.1)
    entries = _membrane_result(solver.solve())
    entries["num_layers"] = solver.num_layers()
    entries["total_thickness"] = solver.total_thickness()
    return entries


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def normalize_entries(entries: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Convert every recorded value into an array with a stable dtype."""

    normalized: dict[str, np.ndarray] = {}
    for key, value in entries.items():
        if isinstance(value, np.ndarray):
            normalized[key] = value
        elif isinstance(value, bool):
            normalized[key] = np.asarray(value, dtype=np.bool_)
        elif isinstance(value, int):
            normalized[key] = np.asarray(value, dtype=np.int64)
        elif isinstance(value, float):
            normalized[key] = np.asarray(value, dtype=np.float64)
        else:
            normalized[key] = np.asarray(value)
    return normalized


def run_case(name: str) -> dict[str, np.ndarray]:
    """Run one registered case and return normalized entries."""

    return normalize_entries(CASES[name].run())


def covered_symbols() -> frozenset[str]:
    return frozenset(symbol for case in CASES.values() for symbol in case.symbols)


def missing_symbols() -> tuple[str, ...]:
    """Native solver symbols from the contract registry with no golden case."""

    return tuple(sorted(set(list_native_solver_symbols()) - covered_symbols()))


def unknown_symbols() -> tuple[str, ...]:
    """Golden-case symbols that the contract registry does not know about."""

    return tuple(sorted(covered_symbols() - set(list_native_solver_symbols())))


def check_coverage() -> None:
    missing = missing_symbols()
    unknown = unknown_symbols()
    if missing or unknown:
        raise AssertionError(
            "golden cases do not match the native solver registry: "
            f"missing={missing} unknown={unknown}"
        )


def capture(out_dir: Path | None = None) -> list[Path]:
    """Run every available case and write ``<case>.npz`` into ``out_dir``."""

    check_coverage()
    target = GOLDEN_DIR if out_dir is None else Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for case in CASES.values():
        if not case.available():
            print(f"skip   {case.name} (sparse matrix support unavailable)")
            continue
        entries = run_case(case.name)
        path = target / f"{case.name}.npz"
        np.savez(path, **entries)
        written.append(path)
        print(f"wrote  {path.name} ({len(entries)} entries)")
    return written


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    out_dir = Path(args[0]) if args else None
    written = capture(out_dir)
    total = sum(path.stat().st_size for path in written)
    print(f"{len(written)} fixtures, {total / 1024:.1f} KiB total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
