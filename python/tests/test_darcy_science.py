"""Independent science checks for the public Darcy-flow interface."""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np
import pytest

import biotransport as bt


LENGTH_M = 1.0
INTERFACE_M = 0.5
PRESSURE_LEFT_PA = 6000.0
PRESSURE_RIGHT_PA = 1000.0
KAPPA_LEFT_M2_PER_PA_S = 2.0e-10
KAPPA_RIGHT_M2_PER_PA_S = 8.0e-10


def _layered_flux() -> float:
    resistance = (
        INTERFACE_M / KAPPA_LEFT_M2_PER_PA_S
        + (LENGTH_M - INTERFACE_M) / KAPPA_RIGHT_M2_PER_PA_S
    )
    return (PRESSURE_LEFT_PA - PRESSURE_RIGHT_PA) / resistance


def _layered_pressure(x_m: float) -> float:
    flux_m_per_s = _layered_flux()
    if x_m <= INTERFACE_M:
        return PRESSURE_LEFT_PA - flux_m_per_s * x_m / KAPPA_LEFT_M2_PER_PA_S
    pressure_at_interface = (
        PRESSURE_LEFT_PA - flux_m_per_s * INTERFACE_M / KAPPA_LEFT_M2_PER_PA_S
    )
    return pressure_at_interface - (
        flux_m_per_s * (x_m - INTERFACE_M) / KAPPA_RIGHT_M2_PER_PA_S
    )


def _field(mesh: bt.StructuredMesh, value_at_x: Callable[[float], float]) -> np.ndarray:
    values = np.empty(mesh.num_nodes(), dtype=np.float64)
    for j in range(mesh.ny() + 1):
        for i in range(mesh.nx() + 1):
            values[mesh.index(i, j)] = value_at_x(mesh.x(i))
    return values


def _layered_mobility(
    mesh: bt.StructuredMesh, *, interface_node_is_left: bool
) -> np.ndarray:
    def mobility(x_m: float) -> float:
        is_left = x_m < INTERFACE_M or (interface_node_is_left and x_m == INTERFACE_M)
        return KAPPA_LEFT_M2_PER_PA_S if is_left else KAPPA_RIGHT_M2_PER_PA_S

    return _field(mesh, mobility)


def _solve_node_aligned_interface(cells: int) -> tuple[float, float]:
    mesh = bt.StructuredMesh(cells, 2, 0.0, LENGTH_M, 0.0, 0.125)
    kappa = _layered_mobility(mesh, interface_node_is_left=True)
    solver = bt.DarcyFlowSolver(mesh, kappa.tolist())
    solver.set_dirichlet(bt.Boundary.Left, PRESSURE_LEFT_PA)
    solver.set_dirichlet(bt.Boundary.Right, PRESSURE_RIGHT_PA)
    solver.set_outward_pressure_gradient(bt.Boundary.Bottom, 0.0)
    solver.set_outward_pressure_gradient(bt.Boundary.Top, 0.0)
    solver.set_initial_guess(_field(mesh, _layered_pressure).tolist())
    solver.set_omega(1.6)
    solver.set_tolerance(1.0e-10)
    solver.set_max_iterations(20_000)
    pressure = np.asarray(solver.solve().pressure())
    exact = _field(mesh, _layered_pressure)
    return mesh.dx(), float(np.max(np.abs(pressure - exact)))


def test_uniform_dirichlet_drop_matches_pressure_and_velocity_solution() -> None:
    length_m = 0.04
    pressure_left_pa = 3200.0
    pressure_right_pa = 800.0
    kappa_m2_per_pa_s = 2.5e-10
    gradient_pa_per_m = (pressure_right_pa - pressure_left_pa) / length_m
    velocity_m_per_s = -kappa_m2_per_pa_s * gradient_pa_per_m

    mesh = bt.StructuredMesh(20, 6, 0.0, length_m, 0.0, 0.012)
    solver = bt.DarcyFlowSolver(mesh, kappa_m2_per_pa_s)
    solver.set_dirichlet(bt.Boundary.Left, pressure_left_pa)
    solver.set_dirichlet(bt.Boundary.Right, pressure_right_pa)
    solver.set_outward_pressure_gradient(bt.Boundary.Bottom, 0.0)
    solver.set_outward_pressure_gradient(bt.Boundary.Top, 0.0)
    solver.set_omega(1.65)
    solver.set_tolerance(1.0e-10)
    solver.set_max_iterations(20_000)
    result = solver.solve()

    exact_pressure = _field(
        mesh, lambda x_m: pressure_left_pa + gradient_pa_per_m * x_m
    )
    assert result.converged
    assert result.residual <= 1.0e-10
    assert np.max(np.abs(np.asarray(result.pressure()) - exact_pressure)) < 2.0e-8
    assert np.max(np.abs(np.asarray(result.vx()) - velocity_m_per_s)) < 2.0e-15
    assert np.max(np.abs(np.asarray(result.vy()))) < 2.0e-15


def test_neumann_value_is_outward_pressure_gradient_in_pa_per_m() -> None:
    length_m = 0.02
    fixed_pressure_pa = 1200.0
    gradient_pa_per_m = 25_000.0
    kappa_m2_per_pa_s = 4.0e-10
    expected_pressure_pa = fixed_pressure_pa + gradient_pa_per_m * length_m
    speed_m_per_s = kappa_m2_per_pa_s * gradient_pa_per_m
    mesh = bt.StructuredMesh(10, 4, 0.0, length_m, 0.0, 0.008)

    right_solver = bt.DarcyFlowSolver(mesh, kappa_m2_per_pa_s)
    right_solver.set_dirichlet(bt.Boundary.Left, fixed_pressure_pa)
    right_solver.set_outward_pressure_gradient(bt.Boundary.Right, gradient_pa_per_m)
    right_solver.set_outward_pressure_gradient(bt.Boundary.Bottom, 0.0)
    right_solver.set_outward_pressure_gradient(bt.Boundary.Top, 0.0)
    right_solver.set_initial_guess(
        _field(mesh, lambda x_m: fixed_pressure_pa + gradient_pa_per_m * x_m).tolist()
    )
    right_solver.set_tolerance(1.0e-11)
    right_solver.set_max_iterations(10)
    right = right_solver.solve()

    left_solver = bt.DarcyFlowSolver(mesh, kappa_m2_per_pa_s)
    left_solver.set_outward_pressure_gradient(bt.Boundary.Left, gradient_pa_per_m)
    left_solver.set_dirichlet(bt.Boundary.Right, fixed_pressure_pa)
    left_solver.set_outward_pressure_gradient(bt.Boundary.Bottom, 0.0)
    left_solver.set_outward_pressure_gradient(bt.Boundary.Top, 0.0)
    left_solver.set_initial_guess(
        _field(
            mesh,
            lambda x_m: fixed_pressure_pa + gradient_pa_per_m * (length_m - x_m),
        ).tolist()
    )
    left_solver.set_tolerance(1.0e-11)
    left_solver.set_max_iterations(10)
    left = left_solver.solve()

    right_mid = mesh.index(mesh.nx(), mesh.ny() // 2)
    left_mid = mesh.index(0, mesh.ny() // 2)
    assert np.asarray(right.pressure())[right_mid] == pytest.approx(
        expected_pressure_pa, abs=2.0e-11
    )
    assert np.asarray(left.pressure())[left_mid] == pytest.approx(
        expected_pressure_pa, abs=2.0e-11
    )
    assert np.asarray(right.vx())[right_mid] == pytest.approx(
        -speed_m_per_s, abs=2.0e-15
    )
    assert np.asarray(left.vx())[left_mid] == pytest.approx(speed_m_per_s, abs=2.0e-15)


def test_face_aligned_two_material_solution_has_continuous_normal_flux() -> None:
    mesh = bt.StructuredMesh(31, 4, 0.0, LENGTH_M, 0.0, 0.125)
    kappa = _layered_mobility(mesh, interface_node_is_left=False)
    solver = bt.DarcyFlowSolver(mesh, kappa.tolist())
    solver.set_dirichlet(bt.Boundary.Left, PRESSURE_LEFT_PA)
    solver.set_dirichlet(bt.Boundary.Right, PRESSURE_RIGHT_PA)
    solver.set_outward_pressure_gradient(bt.Boundary.Bottom, 0.0)
    solver.set_outward_pressure_gradient(bt.Boundary.Top, 0.0)
    solver.set_initial_guess(_field(mesh, _layered_pressure).tolist())
    solver.set_tolerance(1.0e-10)
    solver.set_max_iterations(100)
    result = solver.solve()
    pressure = np.asarray(result.pressure())
    expected_pressure = _field(mesh, _layered_pressure)

    row = mesh.ny() // 2
    face_fluxes = []
    for i in range(mesh.nx()):
        west = mesh.index(i, row)
        east = mesh.index(i + 1, row)
        face_kappa = 2.0 * kappa[west] * kappa[east] / (kappa[west] + kappa[east])
        face_fluxes.append(-face_kappa * (pressure[east] - pressure[west]) / mesh.dx())
    face_fluxes_array = np.asarray(face_fluxes)
    expected_flux = _layered_flux()
    relative_flux_spread = float(np.ptp(face_fluxes_array) / expected_flux)

    left_i = (mesh.nx() - 1) // 2
    right_i = left_i + 1
    interface_pressure_from_left = pressure[mesh.index(left_i, row)] - (
        expected_flux * (INTERFACE_M - mesh.x(left_i)) / KAPPA_LEFT_M2_PER_PA_S
    )
    interface_pressure_from_right = pressure[mesh.index(right_i, row)] + (
        expected_flux * (mesh.x(right_i) - INTERFACE_M) / KAPPA_RIGHT_M2_PER_PA_S
    )
    expected_interface_pressure = (
        PRESSURE_LEFT_PA - expected_flux * INTERFACE_M / KAPPA_LEFT_M2_PER_PA_S
    )

    assert np.max(np.abs(pressure - expected_pressure)) < 2.0e-7
    assert relative_flux_spread < 2.0e-9
    assert interface_pressure_from_left == pytest.approx(
        expected_interface_pressure, abs=2.0e-7
    )
    assert interface_pressure_from_right == pytest.approx(
        expected_interface_pressure, abs=2.0e-7
    )


def test_node_aligned_discontinuous_interface_has_measured_first_order_error() -> None:
    measurements = [_solve_node_aligned_interface(cells) for cells in (16, 32, 64)]
    errors = [measurement[1] for measurement in measurements]
    orders = [math.log(errors[i] / errors[i + 1], 2.0) for i in range(2)]

    assert errors[2] < errors[1] < errors[0]
    assert all(0.8 < order < 1.2 for order in orders)


def test_unanchored_and_forced_unconverged_systems_raise() -> None:
    mesh = bt.StructuredMesh(8, 4, 0.0, 1.0, 0.0, 0.5)

    with pytest.raises(ValueError, match="unanchored"):
        bt.DarcyFlowSolver(mesh, 1.0e-10).solve()

    exhausted = bt.DarcyFlowSolver(mesh, 1.0e-10)
    exhausted.set_dirichlet(bt.Boundary.Left, 2000.0)
    exhausted.set_dirichlet(bt.Boundary.Right, 1000.0)
    exhausted.set_outward_pressure_gradient(bt.Boundary.Bottom, 0.0)
    exhausted.set_outward_pressure_gradient(bt.Boundary.Top, 0.0)
    exhausted.set_tolerance(1.0e-15)
    exhausted.set_max_iterations(1)
    with pytest.raises(RuntimeError, match="did not converge"):
        exhausted.solve()
