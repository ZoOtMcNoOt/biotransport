"""Science and API checks for conservative nonuniform 1D diffusion."""

from __future__ import annotations

import math

import numpy as np
import pytest

import biotransport as bt


def _stretched_nodes(cells: int) -> np.ndarray:
    coordinate = np.linspace(0.0, 1.0, cells + 1)
    return coordinate + 0.25 * np.sin(2.0 * np.pi * coordinate) / (2.0 * np.pi)


def _manufactured_error(cells: int) -> float:
    diffusivity = 0.1
    final_time = 0.05
    nodes = _stretched_nodes(cells)
    mesh = bt.NonuniformMesh1D(nodes.tolist())
    solver = bt.NonuniformDiffusion1D(mesh, diffusivity)
    solver.set_dirichlet_boundary(bt.Boundary.Left, 0.0)
    solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)
    solver.set_initial_condition(np.sin(np.pi * nodes).tolist())
    solver.solve_until(final_time, 0.2 * solver.max_stable_time_step())

    exact = np.sin(np.pi * nodes) * math.exp(-diffusivity * np.pi**2 * final_time)
    error = solver.solution() - exact
    return float(np.sqrt(np.sum(mesh.control_volumes() * error**2) / mesh.length()))


def test_mesh_geometry_is_validated_and_arrays_are_owned() -> None:
    mesh = bt.NonuniformMesh1D([0.0, 0.1, 0.4, 1.0])
    np.testing.assert_allclose(mesh.control_volumes(), [0.05, 0.20, 0.45, 0.30])
    assert mesh.num_nodes() == 4
    assert mesh.num_cells() == 3
    assert np.sum(mesh.control_volumes()) == pytest.approx(mesh.length())

    coordinates = mesh.nodes()
    coordinates[0] = 99.0
    assert mesh.x(0) == 0.0

    for invalid in ([0.0], [0.0, 0.5, 0.5], [0.0, 0.7, 0.6], [0.0, np.nan, 1.0]):
        with pytest.raises(ValueError):
            bt.NonuniformMesh1D(invalid)


def test_uniform_grid_parity_and_exact_final_time() -> None:
    nodes = np.linspace(0.0, 1.0, 11)
    initial = 0.5 + 0.4 * np.sin(np.pi * nodes)
    diffusivity = 0.2
    dt = 0.01
    spacing = 0.1
    fourier = diffusivity * dt / spacing**2

    solver = bt.NonuniformDiffusion1D(bt.NonuniformMesh1D(nodes.tolist()), diffusivity)
    solver.set_initial_condition(initial.tolist())
    solver.step(dt)

    expected = initial.copy()
    expected[0] += 2.0 * fourier * (initial[1] - initial[0])
    expected[-1] += 2.0 * fourier * (initial[-2] - initial[-1])
    expected[1:-1] += fourier * (initial[:-2] - 2.0 * initial[1:-1] + initial[2:])
    np.testing.assert_allclose(solver.solution(), expected, rtol=2e-15, atol=3e-15)

    owned_solution = solver.solution()
    owned_solution[:] = -1.0
    assert np.all(solver.solution() >= 0.0)

    solver.solve_until(0.037, 0.9 * solver.max_stable_time_step())
    assert solver.time() == 0.037


def test_manufactured_solution_is_second_order_on_stretched_meshes() -> None:
    coarse = _manufactured_error(20)
    medium = _manufactured_error(40)
    assert coarse / medium > 3.2


def test_discontinuous_diffusivity_uses_one_harmonic_face_flux() -> None:
    nodes = np.array([0.0, 0.12, 0.30, 0.55, 0.78, 1.0])
    diffusivity = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
    mesh = bt.NonuniformMesh1D(nodes.tolist())
    solver = bt.NonuniformDiffusion1D(mesh, diffusivity)
    solver.set_dirichlet_boundary(bt.Boundary.Left, 1.0)
    solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)

    face_diffusivity = solver.face_diffusivities()
    assert face_diffusivity[2] == pytest.approx(2.0 / 11.0)
    resistance = np.diff(nodes) / face_diffusivity
    steady = 1.0 - np.concatenate(([0.0], np.cumsum(resistance))) / np.sum(resistance)
    solver.set_initial_condition(steady.tolist())

    np.testing.assert_allclose(
        solver.face_fluxes(),
        np.full(nodes.size - 1, 1.0 / np.sum(resistance)),
        rtol=2e-14,
    )
    solver.step(0.5 * solver.max_stable_time_step())
    np.testing.assert_allclose(solver.solution(), steady, rtol=2e-14, atol=2e-14)


def test_closed_mass_balance_and_neumann_sign_are_explicit() -> None:
    nodes = [0.0, 0.03, 0.11, 0.26, 0.50, 0.72, 1.0]
    diffusivity = [0.20, 0.18, 0.16, 0.13, 0.10, 0.08, 0.07]
    initial = [0.2, 1.1, 0.4, 0.9, 0.3, 0.7, 0.5]
    solver = bt.NonuniformDiffusion1D(bt.NonuniformMesh1D(nodes), diffusivity)
    solver.set_initial_condition(initial)
    initial_mass = solver.total_mass()
    solver.solve_until(0.1, 0.8 * solver.max_stable_time_step())
    diagnostics = solver.diagnostics()
    assert diagnostics.total_mass == pytest.approx(initial_mass, abs=5e-14)
    assert diagnostics.mass_balance_error == pytest.approx(0.0, abs=6e-14)
    assert diagnostics.cumulative_boundary_input == 0.0

    source = bt.NonuniformDiffusion1D(bt.NonuniformMesh1D([0.0, 0.2, 0.5, 1.0]), 0.2)
    source.set_neumann_boundary(bt.Boundary.Left, 0.5)
    source.set_uniform_initial_condition(1.0)
    source.step(1e-3)
    diagnostics = source.diagnostics()
    assert diagnostics.left_outward_flux == pytest.approx(-0.1)
    assert diagnostics.cumulative_boundary_input == pytest.approx(1e-4)
    assert diagnostics.mass_balance_error == pytest.approx(0.0, abs=2e-15)


def test_invalid_material_boundary_and_unstable_steps_fail_loudly() -> None:
    mesh = bt.NonuniformMesh1D([0.0, 0.2, 0.6, 1.0])
    with pytest.raises(ValueError):
        bt.NonuniformDiffusion1D(mesh, [0.1, 0.2])
    with pytest.raises(ValueError):
        bt.NonuniformDiffusion1D(mesh, -0.1)

    solver = bt.NonuniformDiffusion1D(mesh, 0.1)
    with pytest.raises(ValueError):
        solver.set_dirichlet_boundary(bt.Boundary.Bottom, 1.0)
    with pytest.raises(ValueError):
        solver.set_boundary_condition(
            bt.Boundary.Left, bt.BoundaryCondition.robin(1.0, 1.0, 1.0)
        )

    solver.set_initial_condition([0.1, 0.8, 0.3, 0.6])
    before = solver.solution()
    unstable = np.nextafter(solver.max_stable_time_step(), np.inf)
    assert not solver.check_stability(float(unstable))
    with pytest.raises(ValueError):
        solver.step(float(unstable))
    np.testing.assert_array_equal(solver.solution(), before)
    assert solver.time() == 0.0
