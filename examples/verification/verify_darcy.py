"""Headless verification of the public steady Darcy-flow solver.

The checks use SI units and independently evaluate analytical pressure,
Darcy velocity, harmonic-face flux, refinement behavior, and fail-loud
contracts. ``set_neumann`` is exercised as outward ``dp/dn`` [Pa/m], not as a
Darcy flux. The discontinuous node-aligned refinement check is expected to be
first order because the nodal material label shifts the represented interface
by half a cell; this script does not claim second-order interface convergence.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np

import biotransport as bt


LENGTH_M = 1.0
INTERFACE_M = 0.5
PRESSURE_LEFT_PA = 6000.0
PRESSURE_RIGHT_PA = 1000.0
KAPPA_LEFT_M2_PER_PA_S = 2.0e-10
KAPPA_RIGHT_M2_PER_PA_S = 8.0e-10


def field(mesh: bt.StructuredMesh, value_at_x: Callable[[float], float]) -> np.ndarray:
    """Create a flat row-major nodal field from a one-dimensional function."""
    values = np.empty(mesh.num_nodes(), dtype=np.float64)
    for j in range(mesh.ny() + 1):
        for i in range(mesh.nx() + 1):
            values[mesh.index(i, j)] = value_at_x(mesh.x(i))
    return values


def layered_flux() -> float:
    """Analytical Darcy velocity through two layers in series [m/s]."""
    resistance = (
        INTERFACE_M / KAPPA_LEFT_M2_PER_PA_S
        + (LENGTH_M - INTERFACE_M) / KAPPA_RIGHT_M2_PER_PA_S
    )
    return (PRESSURE_LEFT_PA - PRESSURE_RIGHT_PA) / resistance


def layered_pressure(x_m: float) -> float:
    """Analytical piecewise-linear pressure [Pa]."""
    flux_m_per_s = layered_flux()
    if x_m <= INTERFACE_M:
        return PRESSURE_LEFT_PA - flux_m_per_s * x_m / KAPPA_LEFT_M2_PER_PA_S
    interface_pressure_pa = (
        PRESSURE_LEFT_PA - flux_m_per_s * INTERFACE_M / KAPPA_LEFT_M2_PER_PA_S
    )
    return interface_pressure_pa - (
        flux_m_per_s * (x_m - INTERFACE_M) / KAPPA_RIGHT_M2_PER_PA_S
    )


def layered_mobility(
    mesh: bt.StructuredMesh, *, interface_node_is_left: bool
) -> np.ndarray:
    """Create the flat nodal mobility field used by the public API."""

    def mobility(x_m: float) -> float:
        is_left = x_m < INTERFACE_M or (interface_node_is_left and x_m == INTERFACE_M)
        return KAPPA_LEFT_M2_PER_PA_S if is_left else KAPPA_RIGHT_M2_PER_PA_S

    return field(mesh, mobility)


def verify_uniform_solution() -> bool:
    """Check a linear Dirichlet solution and its full velocity field."""
    length_m = 0.04
    pressure_left_pa = 3200.0
    pressure_right_pa = 800.0
    kappa_m2_per_pa_s = 2.5e-10
    gradient_pa_per_m = (pressure_right_pa - pressure_left_pa) / length_m
    expected_velocity_m_per_s = -kappa_m2_per_pa_s * gradient_pa_per_m

    mesh = bt.StructuredMesh(20, 6, 0.0, length_m, 0.0, 0.012)
    solver = bt.DarcyFlowSolver(mesh, kappa_m2_per_pa_s)
    solver.set_dirichlet(bt.Boundary.Left, pressure_left_pa)
    solver.set_dirichlet(bt.Boundary.Right, pressure_right_pa)
    solver.set_neumann(bt.Boundary.Bottom, 0.0)
    solver.set_neumann(bt.Boundary.Top, 0.0)
    solver.set_omega(1.65)
    solver.set_tolerance(1.0e-10)
    solver.set_max_iterations(20_000)
    result = solver.solve()

    exact_pressure = field(mesh, lambda x_m: pressure_left_pa + gradient_pa_per_m * x_m)
    pressure_error_pa = float(
        np.max(np.abs(np.asarray(result.pressure()) - exact_pressure))
    )
    velocity_error_m_per_s = float(
        np.max(np.abs(np.asarray(result.vx()) - expected_velocity_m_per_s))
    )
    max_vy_m_per_s = float(np.max(np.abs(np.asarray(result.vy()))))

    print("\nUniform slab: left/right pressure + zero top/bottom dp/dn")
    print(f"  pressure L_inf error:    {pressure_error_pa:.10e} Pa")
    print(f"  x-velocity L_inf error:  {velocity_error_m_per_s:.10e} m/s")
    print(f"  max |vy|:                {max_vy_m_per_s:.10e} m/s")
    print(f"  iterations:              {result.iterations}")
    print(f"  pressure defect:         {result.residual:.10e} Pa")
    return bool(
        result.converged
        and result.residual <= 1.0e-10
        and pressure_error_pa < 2.0e-8
        and velocity_error_m_per_s < 2.0e-15
        and max_vy_m_per_s < 2.0e-15
    )


def verify_outward_gradient_sign() -> bool:
    """Check right and left outward-normal gradient signs and SI scaling."""
    length_m = 0.02
    fixed_pressure_pa = 1200.0
    gradient_pa_per_m = 25_000.0
    kappa_m2_per_pa_s = 4.0e-10
    expected_pressure_pa = fixed_pressure_pa + gradient_pa_per_m * length_m
    expected_speed_m_per_s = kappa_m2_per_pa_s * gradient_pa_per_m
    mesh = bt.StructuredMesh(10, 4, 0.0, length_m, 0.0, 0.008)

    right_solver = bt.DarcyFlowSolver(mesh, kappa_m2_per_pa_s)
    right_solver.set_dirichlet(bt.Boundary.Left, fixed_pressure_pa)
    right_solver.set_neumann(bt.Boundary.Right, gradient_pa_per_m)
    right_solver.set_neumann(bt.Boundary.Bottom, 0.0)
    right_solver.set_neumann(bt.Boundary.Top, 0.0)
    right_solver.set_initial_guess(
        field(mesh, lambda x_m: fixed_pressure_pa + gradient_pa_per_m * x_m).tolist()
    )
    right_solver.set_tolerance(1.0e-11)
    right_solver.set_max_iterations(10)
    right = right_solver.solve()

    left_solver = bt.DarcyFlowSolver(mesh, kappa_m2_per_pa_s)
    left_solver.set_neumann(bt.Boundary.Left, gradient_pa_per_m)
    left_solver.set_dirichlet(bt.Boundary.Right, fixed_pressure_pa)
    left_solver.set_neumann(bt.Boundary.Bottom, 0.0)
    left_solver.set_neumann(bt.Boundary.Top, 0.0)
    left_solver.set_initial_guess(
        field(
            mesh,
            lambda x_m: fixed_pressure_pa + gradient_pa_per_m * (length_m - x_m),
        ).tolist()
    )
    left_solver.set_tolerance(1.0e-11)
    left_solver.set_max_iterations(10)
    left = left_solver.solve()

    right_mid = mesh.index(mesh.nx(), mesh.ny() // 2)
    left_mid = mesh.index(0, mesh.ny() // 2)
    right_pressure_pa = float(np.asarray(right.pressure())[right_mid])
    left_pressure_pa = float(np.asarray(left.pressure())[left_mid])
    right_outward_velocity_m_per_s = float(np.asarray(right.vx())[right_mid])
    left_outward_velocity_m_per_s = float(-np.asarray(left.vx())[left_mid])

    print("\nNeumann semantics: prescribed value is outward dp/dn")
    print(f"  prescribed gradient:     {gradient_pa_per_m:.10e} Pa/m")
    print(f"  expected pressure:       {expected_pressure_pa:.10e} Pa")
    print(f"  right boundary pressure: {right_pressure_pa:.10e} Pa")
    print(f"  left boundary pressure:  {left_pressure_pa:.10e} Pa")
    print(f"  right outward velocity:  {right_outward_velocity_m_per_s:.10e} m/s")
    print(f"  left outward velocity:   {left_outward_velocity_m_per_s:.10e} m/s")
    return bool(
        abs(right_pressure_pa - expected_pressure_pa) < 2.0e-11
        and abs(left_pressure_pa - expected_pressure_pa) < 2.0e-11
        and abs(right_outward_velocity_m_per_s + expected_speed_m_per_s) < 2.0e-15
        and abs(left_outward_velocity_m_per_s + expected_speed_m_per_s) < 2.0e-15
    )


def verify_layered_interface() -> bool:
    """Check series resistance and continuous harmonic-face Darcy flux."""
    # With 31 cells, x=0.5 is the face between nodes 15 and 16.
    mesh = bt.StructuredMesh(31, 4, 0.0, LENGTH_M, 0.0, 0.125)
    kappa = layered_mobility(mesh, interface_node_is_left=False)
    solver = bt.DarcyFlowSolver(mesh, kappa.tolist())
    solver.set_dirichlet(bt.Boundary.Left, PRESSURE_LEFT_PA)
    solver.set_dirichlet(bt.Boundary.Right, PRESSURE_RIGHT_PA)
    solver.set_neumann(bt.Boundary.Bottom, 0.0)
    solver.set_neumann(bt.Boundary.Top, 0.0)
    solver.set_initial_guess(field(mesh, layered_pressure).tolist())
    solver.set_tolerance(1.0e-10)
    solver.set_max_iterations(100)
    result = solver.solve()
    pressure = np.asarray(result.pressure())
    expected_pressure = field(mesh, layered_pressure)
    expected_flux_m_per_s = layered_flux()

    row = mesh.ny() // 2
    face_fluxes = []
    for i in range(mesh.nx()):
        west = mesh.index(i, row)
        east = mesh.index(i + 1, row)
        face_kappa = 2.0 * kappa[west] * kappa[east] / (kappa[west] + kappa[east])
        face_fluxes.append(-face_kappa * (pressure[east] - pressure[west]) / mesh.dx())
    face_fluxes_array = np.asarray(face_fluxes)
    relative_flux_spread = float(np.ptp(face_fluxes_array) / expected_flux_m_per_s)
    pressure_error_pa = float(np.max(np.abs(pressure - expected_pressure)))

    left_i = (mesh.nx() - 1) // 2
    right_i = left_i + 1
    pressure_from_left_pa = pressure[mesh.index(left_i, row)] - (
        expected_flux_m_per_s * (INTERFACE_M - mesh.x(left_i)) / KAPPA_LEFT_M2_PER_PA_S
    )
    pressure_from_right_pa = pressure[mesh.index(right_i, row)] + (
        expected_flux_m_per_s
        * (mesh.x(right_i) - INTERFACE_M)
        / KAPPA_RIGHT_M2_PER_PA_S
    )
    expected_interface_pressure_pa = (
        PRESSURE_LEFT_PA - expected_flux_m_per_s * INTERFACE_M / KAPPA_LEFT_M2_PER_PA_S
    )

    print("\nTwo-material, face-aligned interface")
    print(f"  analytical Darcy flux:   {expected_flux_m_per_s:.10e} m/s")
    print(f"  relative flux spread:    {relative_flux_spread:.10e}")
    print(f"  pressure L_inf error:    {pressure_error_pa:.10e} Pa")
    print(f"  interface p from left:   {pressure_from_left_pa:.10e} Pa")
    print(f"  interface p from right:  {pressure_from_right_pa:.10e} Pa")
    return bool(
        pressure_error_pa < 2.0e-7
        and relative_flux_spread < 2.0e-9
        and abs(pressure_from_left_pa - expected_interface_pressure_pa) < 2.0e-7
        and abs(pressure_from_right_pa - expected_interface_pressure_pa) < 2.0e-7
    )


def node_aligned_interface_error(cells: int) -> tuple[float, float]:
    """Return spacing and pressure error for a node-labeled material jump."""
    mesh = bt.StructuredMesh(cells, 2, 0.0, LENGTH_M, 0.0, 0.125)
    kappa = layered_mobility(mesh, interface_node_is_left=True)
    solver = bt.DarcyFlowSolver(mesh, kappa.tolist())
    solver.set_dirichlet(bt.Boundary.Left, PRESSURE_LEFT_PA)
    solver.set_dirichlet(bt.Boundary.Right, PRESSURE_RIGHT_PA)
    solver.set_neumann(bt.Boundary.Bottom, 0.0)
    solver.set_neumann(bt.Boundary.Top, 0.0)
    solver.set_initial_guess(field(mesh, layered_pressure).tolist())
    solver.set_omega(1.6)
    solver.set_tolerance(1.0e-10)
    solver.set_max_iterations(20_000)
    pressure = np.asarray(solver.solve().pressure())
    error_pa = float(np.max(np.abs(pressure - field(mesh, layered_pressure))))
    return mesh.dx(), error_pa


def verify_refinement() -> bool:
    """Measure, rather than assume, the discontinuous-interface order."""
    cells_sequence = (16, 32, 64)
    measurements = [node_aligned_interface_error(cells) for cells in cells_sequence]
    errors_pa = [measurement[1] for measurement in measurements]
    orders = [math.log(errors_pa[i] / errors_pa[i + 1], 2.0) for i in range(2)]

    print("\nNode-aligned discontinuous-interface refinement")
    print("  cells       h [m]       pressure L_inf error [Pa]")
    for cells, (spacing_m, error_pa) in zip(cells_sequence, measurements):
        print(f"  {cells:5d}  {spacing_m:.10e}  {error_pa:.10e}")
    print(f"  observed order 16 -> 32: {orders[0]:.10f}")
    print(f"  observed order 32 -> 64: {orders[1]:.10f}")
    print("  interpretation: first-order interface-location error (not second order)")
    return bool(
        errors_pa[2] < errors_pa[1] < errors_pa[0]
        and all(0.8 < order < 1.2 for order in orders)
    )


def verify_fail_loud_contracts() -> bool:
    """Check gauge rejection and forced iteration exhaustion."""
    mesh = bt.StructuredMesh(8, 4, 0.0, 1.0, 0.0, 0.5)

    gauge_rejected = False
    try:
        bt.DarcyFlowSolver(mesh, 1.0e-10).solve()
    except ValueError as error:
        gauge_rejected = "unanchored" in str(error)

    exhausted = bt.DarcyFlowSolver(mesh, 1.0e-10)
    exhausted.set_dirichlet(bt.Boundary.Left, 2000.0)
    exhausted.set_dirichlet(bt.Boundary.Right, 1000.0)
    exhausted.set_neumann(bt.Boundary.Bottom, 0.0)
    exhausted.set_neumann(bt.Boundary.Top, 0.0)
    exhausted.set_tolerance(1.0e-15)
    exhausted.set_max_iterations(1)
    nonconvergence_rejected = False
    try:
        exhausted.solve()
    except RuntimeError as error:
        nonconvergence_rejected = "did not converge" in str(error)

    print("\nFail-loud contracts")
    print(f"  missing pressure gauge rejected: {gauge_rejected}")
    print(f"  forced nonconvergence rejected:  {nonconvergence_rejected}")
    return gauge_rejected and nonconvergence_rejected


def main() -> int:
    checks = [
        ("uniform analytical solution", verify_uniform_solution),
        ("outward-gradient sign and units", verify_outward_gradient_sign),
        ("two-material interface", verify_layered_interface),
        ("measured refinement", verify_refinement),
        ("fail-loud contracts", verify_fail_loud_contracts),
    ]

    print("Darcy-flow scientific verification (SI units)")
    print("Equation: div(kappa grad(p)) = 0; v = -kappa grad(p)")
    outcomes: list[bool] = []
    for name, check in checks:
        passed = check()
        outcomes.append(passed)
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")

    passed_count = sum(outcomes)
    print(f"\nSUMMARY: {passed_count}/{len(outcomes)} checks passed")
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
