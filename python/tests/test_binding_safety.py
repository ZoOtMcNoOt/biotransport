"""Regression tests for safety guarantees at the Python/C++ boundary."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from biotransport import _core as core


def _assert_keeps_mesh_alive(
    mesh_factory: Callable[[], object], factory: Callable[[object], object]
) -> None:
    # Construct here so CPython 3.9's caller evaluation stack cannot retain the
    # temporary mesh while this helper verifies its lifetime.
    mesh = mesh_factory()
    mesh_reference = weakref.ref(mesh)
    owner = factory(mesh)

    del mesh
    gc.collect()
    assert mesh_reference() is not None

    del owner
    gc.collect()
    assert mesh_reference() is None


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda mesh: core.DiffusionSolver(mesh, 0.01), id="explicit"),
        pytest.param(
            lambda mesh: core.CrankNicolsonDiffusion(mesh, 0.01),
            id="crank-nicolson",
        ),
        pytest.param(lambda mesh: core.ADIDiffusion2D(mesh, 0.01), id="adi"),
        pytest.param(
            lambda mesh: core.LinearReactionDiffusionSolver(mesh, 0.01, 0.1),
            id="reaction-diffusion",
        ),
        pytest.param(
            lambda mesh: core.AdvectionDiffusionSolver(mesh, 0.01, 0.1),
            id="advection-diffusion",
        ),
        pytest.param(lambda mesh: core.TransportProblem(mesh), id="problem"),
        pytest.param(
            lambda mesh: core.MultiSpeciesSolver(mesh, [0.01, 0.02]),
            id="multi-species",
        ),
        pytest.param(
            lambda mesh: core.NernstPlanckSolver(mesh, core.ions.sodium()),
            id="nernst-planck",
        ),
        pytest.param(
            lambda mesh: core.ImplicitDiffusion2D(mesh, 0.01),
            id="implicit",
        ),
        pytest.param(lambda mesh: core.StencilOps(mesh), id="stencil-ops"),
    ],
)
def test_2d_native_objects_keep_their_mesh_alive(
    factory: Callable[[object], object],
) -> None:
    _assert_keeps_mesh_alive(
        lambda: core.StructuredMesh(4, 4, 0.0, 1.0, 0.0, 1.0), factory
    )


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda mesh: core.DiffusionSolver3D(mesh, 0.01), id="explicit"),
        pytest.param(lambda mesh: core.ADIDiffusion3D(mesh, 0.01), id="adi"),
        pytest.param(
            lambda mesh: core.LinearReactionDiffusionSolver3D(mesh, 0.01, 0.1),
            id="reaction-diffusion",
        ),
        pytest.param(
            lambda mesh: core.ImplicitDiffusion3D(mesh, 0.01),
            id="implicit",
        ),
    ],
)
def test_3d_native_objects_keep_their_mesh_alive(
    factory: Callable[[object], object],
) -> None:
    _assert_keeps_mesh_alive(lambda: core.StructuredMesh3D(2, 1.0), factory)


def test_invalid_2d_integer_boundaries_raise_value_error() -> None:
    mesh = core.StructuredMesh(4, 0.0, 1.0)
    solver = core.DiffusionSolver(mesh, 0.01)
    condition = core.BoundaryCondition.dirichlet(1.0)

    with pytest.raises(ValueError, match="between 0 .* and 3"):
        solver.set_dirichlet_boundary(-1, 0.0)
    with pytest.raises(ValueError, match="between 0 .* and 3"):
        solver.set_neumann_boundary(4, 0.0)
    with pytest.raises(ValueError, match="between 0 .* and 3"):
        solver.set_boundary_condition(99, condition)

    # The strongly typed API remains available.
    solver.set_dirichlet_boundary(core.Boundary.Left, 1.0)


def test_invalid_3d_integer_boundaries_raise_value_error() -> None:
    mesh = core.StructuredMesh3D(2, 1.0)
    solver = core.DiffusionSolver3D(mesh, 0.01)

    with pytest.raises(ValueError, match="between 0 .* and 5"):
        solver.set_dirichlet_boundary(-1, 0.0)
    with pytest.raises(ValueError, match="between 0 .* and 5"):
        solver.set_neumann_boundary(6, 0.0)

    solver.set_dirichlet_boundary(core.Boundary3D.XMin, 1.0)


def test_invalid_boundary_is_checked_for_coupled_solver_overloads() -> None:
    mesh = core.StructuredMesh(4, 0.0, 1.0)
    multi_species = core.MultiSpeciesSolver(mesh, [0.01, 0.02])
    nernst_planck = core.NernstPlanckSolver(mesh, core.ions.sodium())

    with pytest.raises(ValueError, match="between 0 .* and 3"):
        multi_species.set_dirichlet_boundary(0, 4, 0.0)
    with pytest.raises(ValueError, match="between 0 .* and 3"):
        nernst_planck.set_neumann_boundary(-1, 0.0)


def test_robin_boundary_metadata_is_exposed_without_implying_solver_support() -> None:
    condition = core.BoundaryCondition.robin(a=2.0, b=3.0, c=4.0)

    assert condition.type == core.BoundaryType.ROBIN
    assert condition.value == 0.0
    assert condition.a == 2.0
    assert condition.b == 3.0
    assert condition.c == 4.0


def test_transport_problem_exposes_reaction_replacement_and_composition() -> None:
    mesh = core.StructuredMesh(4, 0.0, 1.0)
    problem = core.TransportProblem(mesh)

    problem.reaction(lambda c, x, y, t: -c, 1.0)
    assert problem.has_reaction()
    assert problem.reaction_stability_bound_known()
    assert problem.reaction_stability_rate_bound() == 1.0

    problem.add_constant_source(2.0).add_linear_decay(0.5)
    assert problem.reaction_stability_rate_bound() == 1.5

    problem.add_logistic_growth(0.1, 10.0)
    assert not problem.reaction_stability_bound_known()

    problem.clear_reaction()
    assert not problem.has_reaction()
    assert problem.reaction_stability_bound_known()
    assert problem.reaction_stability_rate_bound() == 0.0


def test_solver_solution_arrays_are_owned_snapshots() -> None:
    mesh = core.StructuredMesh(8, 0.0, 1.0)
    solver = core.DiffusionSolver(mesh, 0.01)
    initial = np.linspace(0.0, 1.0, mesh.num_nodes())
    solver.set_initial_condition(initial)

    snapshot = solver.solution()
    expected = snapshot.copy()
    assert snapshot.flags.owndata

    snapshot[0] = -123.0
    assert solver.solution()[0] != -123.0

    snapshot[:] = expected
    solver.solve(0.01, 1)
    np.testing.assert_array_equal(snapshot, expected)

    del solver
    gc.collect()
    np.testing.assert_array_equal(snapshot, expected)


def _read_vtk_scalar_values(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    data_start = lines.index("LOOKUP_TABLE default") + 1
    return np.asarray([float(value) for value in lines[data_start:]])


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.arange(10.0)[::2], id="positive-stride"),
        pytest.param(np.arange(5.0)[::-1], id="negative-stride"),
    ],
)
def test_native_vtk_export_preserves_strided_array_values(
    tmp_path: Path, values: np.ndarray
) -> None:
    mesh = core.StructuredMesh(4, 0.0, 1.0)
    output = tmp_path / "strided.vtk"

    core.write_vtk(mesh, values, str(output), "concentration")

    np.testing.assert_array_equal(_read_vtk_scalar_values(output), values)


def test_native_vtk_series_preserves_each_array_stride(tmp_path: Path) -> None:
    mesh = core.StructuredMesh(4, 0.0, 1.0)
    sliced = np.arange(10.0)[::2]
    reversed_values = np.arange(5.0)[::-1]
    prefix = tmp_path / "series"

    core.write_vtk_series(
        mesh,
        [sliced, reversed_values],
        [0.0, 1.0],
        str(prefix),
        "concentration",
    )

    np.testing.assert_array_equal(
        _read_vtk_scalar_values(tmp_path / "series_0000.vtk"), sliced
    )
    np.testing.assert_array_equal(
        _read_vtk_scalar_values(tmp_path / "series_0001.vtk"), reversed_values
    )
