#!/usr/bin/env python3
"""Symmetric alternating-direction Crank--Nicolson diffusion examples.

``ADIDiffusion2D`` uses the symmetric composition x/2--y--x/2 (three
directional solves per step). ``ADIDiffusion3D`` uses
x/2--y/2--z--y/2--x/2 (five solves per step). These are not the historical
Peaceman--Rachford or Douglas--Gunn formulas.

Each directional linear-diffusion solve is unconditionally stable, and the
symmetric composition is second-order for smooth solutions with
time-independent boundary data. That statement does not make arbitrarily large
steps accurate or positivity-preserving. The checks below use separable sine
modes with known solutions and report how error changes with the time step.
"""

from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt
import numpy as np

import biotransport as bt

EXAMPLE_NAME = "adi_diffusion"


def relative_l2(numerical: np.ndarray, reference: np.ndarray) -> float:
    """Return a relative discrete L2 error."""
    return float(np.linalg.norm(numerical - reference) / np.linalg.norm(reference))


def set_zero_dirichlet_2d(solver: bt.ADIDiffusion2D) -> None:
    for boundary in (
        bt.Boundary.Left,
        bt.Boundary.Right,
        bt.Boundary.Bottom,
        bt.Boundary.Top,
    ):
        solver.set_dirichlet_boundary(boundary, 0.0)


def set_zero_dirichlet_3d(solver: bt.ADIDiffusion3D) -> None:
    for boundary in (
        bt.Boundary3D.XMin,
        bt.Boundary3D.XMax,
        bt.Boundary3D.YMin,
        bt.Boundary3D.YMax,
        bt.Boundary3D.ZMin,
        bt.Boundary3D.ZMax,
    ):
        solver.set_dirichlet_boundary(boundary, 0.0)


def run_2d() -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Check a 2D separable eigenmode and return data for plotting."""
    cells = 64
    length = 1.0
    diffusivity = 0.01
    final_time = 0.2
    mesh = bt.StructuredMesh(cells, cells, 0.0, length, 0.0, length)
    x = np.linspace(0.0, length, cells + 1)
    y = np.linspace(0.0, length, cells + 1)
    xx, yy = np.meshgrid(x, y)
    initial = np.sin(np.pi * xx / length) * np.sin(np.pi * yy / length)
    reference = initial * np.exp(
        -2.0 * diffusivity * (np.pi / length) ** 2 * final_time
    )

    explicit_limit = 1.0 / (
        2.0 * diffusivity * (1.0 / mesh.dx() ** 2 + 1.0 / mesh.dy() ** 2)
    )
    requested_dt = 8.0 * explicit_limit
    steps = math.ceil(final_time / requested_dt)
    dt = final_time / steps

    solver = bt.ADIDiffusion2D(mesh, diffusivity)
    solver.set_initial_condition(initial.ravel())
    set_zero_dirichlet_2d(solver)
    start = time.perf_counter()
    result = solver.solve(dt=dt, num_steps=steps)
    wall_time = time.perf_counter() - start
    numerical = np.asarray(solver.solution()).reshape(initial.shape)
    error = relative_l2(numerical, reference)
    boundary_max = float(
        max(
            np.max(np.abs(numerical[0, :])),
            np.max(np.abs(numerical[-1, :])),
            np.max(np.abs(numerical[:, 0])),
            np.max(np.abs(numerical[:, -1])),
        )
    )

    checks = {
        "finite field": bool(np.all(np.isfinite(numerical))),
        "three directional solves per step": result.substeps == 3 * steps,
        "requested time reached": math.isclose(
            result.total_time, final_time, rel_tol=0.0, abs_tol=1.0e-14
        ),
        "Dirichlet boundary enforced": boundary_max < 1.0e-13,
        "relative L2 error < 5e-4": error < 5.0e-4,
    }

    print("\n2D x/2--y--x/2 composition")
    print(f"  grid                         {cells + 1} x {cells + 1} nodes")
    print(f"  dt / explicit limit         {dt / explicit_limit:.2f}")
    print(f"  steps / directional solves  {result.steps} / {result.substeps}")
    print(f"  relative L2 error           {error:.3e}")
    print(f"  wall time                   {1.0e3 * wall_time:.2f} ms")
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    return all(checks.values()), xx, yy, initial, numerical


def run_3d() -> bool:
    """Check the five-sweep 3D composition with correctly ordered flat data."""
    cells = 18
    length = 1.0
    diffusivity = 0.1
    final_time = 0.05
    mesh = bt.StructuredMesh3D(
        cells,
        cells,
        cells,
        0.0,
        length,
        0.0,
        length,
        0.0,
        length,
    )
    x = np.linspace(0.0, length, cells + 1)
    y = np.linspace(0.0, length, cells + 1)
    z = np.linspace(0.0, length, cells + 1)

    # Shape is (z, y, x), so C-order flattening matches mesh.index(i, j, k).
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    initial = (
        np.sin(np.pi * xx / length)
        * np.sin(np.pi * yy / length)
        * np.sin(np.pi * zz / length)
    )
    reference = initial * np.exp(
        -3.0 * diffusivity * (np.pi / length) ** 2 * final_time
    )

    explicit_limit = 1.0 / (
        2.0
        * diffusivity
        * (1.0 / mesh.dx() ** 2 + 1.0 / mesh.dy() ** 2 + 1.0 / mesh.dz() ** 2)
    )
    steps = math.ceil(final_time / (4.0 * explicit_limit))
    dt = final_time / steps

    solver = bt.ADIDiffusion3D(mesh, diffusivity)
    solver.set_initial_condition(initial.ravel())
    set_zero_dirichlet_3d(solver)
    start = time.perf_counter()
    result = solver.solve(dt=dt, num_steps=steps)
    wall_time = time.perf_counter() - start
    numerical = np.asarray(solver.solution()).reshape(initial.shape)
    error = relative_l2(numerical, reference)

    boundary_max = float(
        max(
            np.max(np.abs(numerical[0, :, :])),
            np.max(np.abs(numerical[-1, :, :])),
            np.max(np.abs(numerical[:, 0, :])),
            np.max(np.abs(numerical[:, -1, :])),
            np.max(np.abs(numerical[:, :, 0])),
            np.max(np.abs(numerical[:, :, -1])),
        )
    )
    checks = {
        "finite field": bool(np.all(np.isfinite(numerical))),
        "five directional solves per step": result.substeps == 5 * steps,
        "requested time reached": math.isclose(
            result.total_time, final_time, rel_tol=0.0, abs_tol=1.0e-14
        ),
        "Dirichlet boundary enforced": boundary_max < 1.0e-13,
        "relative L2 error < 3e-3": error < 3.0e-3,
    }

    print("\n3D x/2--y/2--z--y/2--x/2 composition")
    print(f"  grid                         {(cells + 1) ** 3} nodes")
    print(f"  dt / explicit limit         {dt / explicit_limit:.2f}")
    print(f"  steps / directional solves  {result.steps} / {result.substeps}")
    print(f"  relative L2 error           {error:.3e}")
    print(f"  wall time                   {1.0e3 * wall_time:.2f} ms")
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    return all(checks.values())


def time_step_accuracy() -> tuple[bool, np.ndarray, np.ndarray]:
    """Show that stable ADI steps still need temporal refinement for accuracy."""
    cells = 40
    mode = 5
    length = 1.0
    diffusivity = 0.1
    final_time = 0.02
    mesh = bt.StructuredMesh(cells, cells, 0.0, length, 0.0, length)
    x = np.linspace(0.0, length, cells + 1)
    y = np.linspace(0.0, length, cells + 1)
    xx, yy = np.meshgrid(x, y)
    initial = np.sin(mode * np.pi * xx) * np.sin(mode * np.pi * yy)
    # Compare with the exact solution of the *semi-discrete* five-point system,
    # which isolates temporal error instead of mixing it with spatial truncation.
    spatial_rate = (
        8.0 * diffusivity * np.sin(mode * np.pi / (2.0 * cells)) ** 2 / mesh.dx() ** 2
    )
    reference = initial * np.exp(-spatial_rate * final_time)
    explicit_limit = 1.0 / (
        2.0 * diffusivity * (1.0 / mesh.dx() ** 2 + 1.0 / mesh.dy() ** 2)
    )

    step_counts = np.array([40, 20, 10, 5, 2])
    time_steps = final_time / step_counts
    errors = []
    minima = []
    for steps, dt in zip(step_counts, time_steps):
        solver = bt.ADIDiffusion2D(mesh, diffusivity)
        solver.set_initial_condition(initial.ravel())
        set_zero_dirichlet_2d(solver)
        solver.solve(dt=float(dt), num_steps=int(steps))
        numerical = np.asarray(solver.solution()).reshape(initial.shape)
        errors.append(relative_l2(numerical, reference))
        minima.append(float(np.min(numerical)))

    errors_array = np.asarray(errors)
    checks = {
        "all runs remain finite": bool(np.all(np.isfinite(errors_array))),
        "refinement reduces error": bool(errors_array[0] < errors_array[-1]),
        "finest-step relative error < 1e-4": errors_array[0] < 1.0e-4,
    }

    print("\nStable does not mean accurate (semi-discrete reference)")
    print("  dt [s]      dt / explicit limit    relative L2      field minimum")
    for dt, error, minimum in zip(time_steps, errors_array, minima):
        print(
            f"  {dt:9.3e}   {dt / explicit_limit:10.2f}           "
            f"{error:9.3e}      {minimum: .3e}"
        )
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print(
        "  Negative values are not automatically a failure for this signed sine mode. "
        "For nonnegative concentration data, independently check positivity when using "
        "large Crank--Nicolson directional steps."
    )
    return all(checks.values()), time_steps, errors_array


def positivity_stress_check() -> bool:
    """Demonstrate that an extreme stable step need not preserve positivity."""
    cells = 40
    diffusivity = 0.1
    mesh = bt.StructuredMesh(cells, cells, 0.0, 1.0, 0.0, 1.0)
    pulse = np.zeros((cells + 1, cells + 1))
    pulse[cells // 2, cells // 2] = 1.0
    explicit_limit = 1.0 / (
        2.0 * diffusivity * (1.0 / mesh.dx() ** 2 + 1.0 / mesh.dy() ** 2)
    )
    dt = 1000.0 * explicit_limit

    solver = bt.ADIDiffusion2D(mesh, diffusivity)
    solver.set_initial_condition(pulse.ravel())
    set_zero_dirichlet_2d(solver)
    solver.solve(dt=dt, num_steps=1)
    field = np.asarray(solver.solution())
    minimum = float(np.min(field))
    norm_ratio = float(np.linalg.norm(field) / np.linalg.norm(pulse))
    checks = {
        "extreme step remains finite": bool(np.all(np.isfinite(field))),
        "field remains bounded in L2": norm_ratio <= 1.0 + 1.0e-10,
        "nonnegative data develop an undershoot": minimum < -1.0e-3,
    }

    print("\nPositivity stress check")
    print(f"  dt / explicit limit          {dt / explicit_limit:.0f}")
    print(f"  final minimum                {minimum:.3e}")
    print(f"  final/initial L2 norm        {norm_ratio:.3f}")
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    return all(checks.values())


def save_plots(
    xx: np.ndarray,
    yy: np.ndarray,
    initial: np.ndarray,
    final: np.ndarray,
    time_steps: np.ndarray,
    errors: np.ndarray,
) -> str:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, field, title in (
        (axes[0], initial, "Initial 2D sine mode"),
        (axes[1], final, "ADI solution"),
    ):
        image = axis.pcolormesh(xx, yy, field, shading="auto", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_aspect("equal")
        figure.colorbar(image, ax=axis)

    axes[2].loglog(time_steps, errors, "o-")
    axes[2].set_title("Time-step accuracy")
    axes[2].set_xlabel("dt [s]")
    axes[2].set_ylabel("relative L2 error")
    axes[2].grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    output = bt.get_result_path("symmetric_adi_checks.png", EXAMPLE_NAME)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def main() -> int:
    plt.switch_backend("Agg")
    print("Symmetric alternating-direction Crank--Nicolson diffusion")
    passed_2d, xx, yy, initial, final = run_2d()
    passed_3d = run_3d()
    passed_accuracy, time_steps, errors = time_step_accuracy()
    passed_positivity = positivity_stress_check()
    output = save_plots(xx, yy, initial, final, time_steps, errors)
    print(f"\nFigure: {output}")
    print(
        "Conclusion: the directional solves remove the explicit diffusion CFL "
        "restriction for linear stability, but dt must still resolve the physics."
    )
    return 0 if passed_2d and passed_3d and passed_accuracy and passed_positivity else 1


if __name__ == "__main__":
    raise SystemExit(main())
