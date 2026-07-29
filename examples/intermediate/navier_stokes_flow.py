#!/usr/bin/env python3
"""Bounded, incompressible Navier--Stokes examples.

The compatible :class:`biotransport.NavierStokesSolver` advances a Newtonian,
laminar flow on a two-dimensional MAC (staggered) grid.  It currently supports
closed domains with no-slip or constant Dirichlet velocity boundaries.  Open
inlets/outlets and prescribed-pressure or traction boundaries need a different
pressure-boundary model and are deliberately rejected.

These examples therefore use wall-driven cavities.  They demonstrate transient
momentum diffusion, oscillatory wall forcing, and Reynolds-number sensitivity;
they are not models of an open blood vessel.  Pressure is the cell-centred
projection pressure with a zero-mean gauge, so only pressure differences are
physical.  The solver has no turbulence or non-Newtonian closure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import biotransport as bt


EXAMPLE_NAME = "navier_stokes"


def _set_closed_cavity_boundaries(
    solver: bt.NavierStokesSolver, lid_velocity: float
) -> None:
    """Apply flux-compatible walls with a tangentially moving top lid."""
    solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.dirichlet(lid_velocity, 0.0))


def _cell_centered_fields(
    result: bt.NavierStokesResult, nx: int, ny: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate packed MAC velocities to cells and remove display padding."""
    shape = (ny + 1, nx + 1)
    u_faces = np.asarray(result.u()).reshape(shape)
    v_faces = np.asarray(result.v()).reshape(shape)
    pressure_packed = np.asarray(result.pressure()).reshape(shape)

    u_cells = 0.5 * (u_faces[:ny, :nx] + u_faces[:ny, 1 : nx + 1])
    v_cells = 0.5 * (v_faces[:ny, :nx] + v_faces[1 : ny + 1, :nx])
    pressure_cells = pressure_packed[:ny, :nx]
    return u_cells, v_cells, pressure_cells


def _require_stable(result: bt.NavierStokesResult, label: str) -> None:
    """Stop an example instead of plotting a failed numerical state."""
    arrays = (result.u(), result.v(), result.pressure())
    if not result.stable or not all(np.all(np.isfinite(values)) for values in arrays):
        raise RuntimeError(f"{label} did not return a finite projection-stable state")


def example_lid_driven_cavity() -> bt.NavierStokesResult:
    """Show transient momentum transfer from a moving wall into a closed cavity."""
    print("=" * 68)
    print("Example 1: Transient Lid-Driven Cavity")
    print("=" * 68)

    # SI values give a low-Reynolds-number, blood-viscosity-scale demonstration.
    length = 1.0e-3
    density = 1060.0
    viscosity = 3.5e-3
    lid_velocity = 5.0e-2
    reynolds = density * lid_velocity * length / viscosity
    print(f"  Lid Reynolds number rho*U*L/mu: {reynolds:.2f}")

    nx = ny = 20
    mesh = bt.StructuredMesh(nx, ny, 0.0, length, 0.0, length)
    solver = bt.NavierStokesSolver(mesh, density, viscosity)
    solver.set_convection_scheme(bt.ConvectionScheme.CENTRAL)
    solver.set_pressure_tolerance(1.0e-9)
    _set_closed_cavity_boundaries(solver, lid_velocity)

    # Adaptive stepping enforces the explicit convective/diffusive stability bound.
    result = solver.solve_steps(100)
    _require_stable(result, "lid-driven cavity")
    u, v, pressure = _cell_centered_fields(result, nx, ny)
    speed = np.hypot(u, v)

    x = (np.arange(nx) + 0.5) * length / nx
    y = (np.arange(ny) + 0.5) * length / ny
    x_mm = x * 1.0e3
    y_mm = y * 1.0e3

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    image = axes[0, 0].imshow(
        speed,
        extent=[0.0, length * 1.0e3, 0.0, length * 1.0e3],
        origin="lower",
        aspect="equal",
        cmap="viridis",
    )
    axes[0, 0].set_title("Cell-Centred Speed")
    axes[0, 0].set_xlabel("x (mm)")
    axes[0, 0].set_ylabel("y (mm)")
    figure.colorbar(image, ax=axes[0, 0], label="m/s")

    pressure_image = axes[0, 1].imshow(
        pressure,
        extent=[0.0, length * 1.0e3, 0.0, length * 1.0e3],
        origin="lower",
        aspect="equal",
        cmap="coolwarm",
    )
    axes[0, 1].set_title("Projection Pressure (Zero-Mean Gauge)")
    axes[0, 1].set_xlabel("x (mm)")
    axes[0, 1].set_ylabel("y (mm)")
    figure.colorbar(pressure_image, ax=axes[0, 1], label="Pa")

    axes[1, 0].streamplot(
        x_mm,
        y_mm,
        u,
        v,
        color=speed,
        cmap="viridis",
        density=1.4,
        linewidth=0.9,
    )
    axes[1, 0].set_title("Cell-Centred Streamlines")
    axes[1, 0].set_xlabel("x (mm)")
    axes[1, 0].set_ylabel("y (mm)")
    axes[1, 0].set_aspect("equal")

    centre_column = nx // 2
    axes[1, 1].plot(u[:, centre_column] / lid_velocity, y_mm, linewidth=2)
    axes[1, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Transient Horizontal-Velocity Profile")
    axes[1, 1].set_xlabel("u / U_lid")
    axes[1, 1].set_ylabel("y (mm)")
    axes[1, 1].grid(True, alpha=0.3)

    figure.suptitle(
        f"Closed cavity after {result.time * 1.0e3:.2f} ms "
        f"({result.time_steps} adaptive steps)"
    )
    figure.tight_layout()
    output = bt.get_result_path("lid_driven_cavity.png", EXAMPLE_NAME)
    figure.savefig(output, dpi=150)
    plt.close(figure)

    print(f"  Simulated time: {result.time * 1.0e3:.3f} ms")
    print(f"  Maximum cell-centred speed: {result.max_velocity:.5f} m/s")
    print(f"  Maximum discrete divergence: {result.divergence:.3e} 1/s")
    print(f"  Pressure residual: {result.pressure_residual:.3e}")
    print(f"  Plot saved to: {output}")
    return result


def example_oscillatory_lid() -> bt.NavierStokesResult:
    """Drive a normalized closed cavity with a sinusoidally moving top wall.

    The frequency is selected to keep this explicit demonstration short.  This
    is an oscillatory-shear problem, not a physiological cardiac waveform.
    """
    print("\n" + "=" * 68)
    print("Example 2: Oscillatory Wall-Driven Cavity")
    print("=" * 68)

    length = 1.0
    density = 1.0
    viscosity = 2.0e-2
    amplitude = 2.0e-1
    frequency = 4.0
    period = 1.0 / frequency
    nx = ny = 16
    steps_per_cycle = 64
    cycles = 2
    dt = period / steps_per_cycle

    reynolds = density * amplitude * length / viscosity
    stokes_layer_depth = np.sqrt(viscosity / (density * np.pi * frequency))
    print(f"  Oscillatory Reynolds number rho*U*L/mu: {reynolds:.1f}")
    print(f"  Stokes-layer scale sqrt(nu/(pi*f)): {stokes_layer_depth:.3f}")
    print(f"  Time step: {dt:.6f}; samples per cycle: {steps_per_cycle}")

    mesh = bt.StructuredMesh(nx, ny, 0.0, length, 0.0, length)
    solver = bt.NavierStokesSolver(mesh, density, viscosity)
    solver.set_convection_scheme(bt.ConvectionScheme.UPWIND)
    solver.set_time_step(dt)
    solver.set_pressure_tolerance(1.0e-8)

    packed_size = (nx + 1) * (ny + 1)
    u_state = np.zeros(packed_size)
    v_state = np.zeros(packed_size)
    times: list[float] = []
    lid_history: list[float] = []
    centre_history: list[float] = []
    result: bt.NavierStokesResult | None = None

    for step in range(cycles * steps_per_cycle):
        time = step * dt
        lid_velocity = amplitude * np.sin(2.0 * np.pi * frequency * time)
        _set_closed_cavity_boundaries(solver, lid_velocity)
        solver.set_initial_velocity(u_state, v_state)
        result = solver.solve_steps(1)
        _require_stable(result, "oscillatory cavity")

        u_state = np.asarray(result.u()).copy()
        v_state = np.asarray(result.v()).copy()
        u_cells, _, _ = _cell_centered_fields(result, nx, ny)
        times.append(time + dt)
        lid_history.append(lid_velocity)
        centre_history.append(float(u_cells[ny // 2, nx // 2]))

    assert result is not None
    u, v, _ = _cell_centered_fields(result, nx, ny)
    speed = np.hypot(u, v)
    times_array = np.asarray(times)
    lid_array = np.asarray(lid_history)
    centre_array = np.asarray(centre_history)
    y = (np.arange(ny) + 0.5) / ny

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(times_array, lid_array, linewidth=2, label="moving lid")
    axes[0, 0].plot(times_array, centre_array, linewidth=2, label="cavity centre")
    axes[0, 0].set_title("Imposed and Interior Horizontal Velocity")
    axes[0, 0].set_xlabel("normalized time")
    axes[0, 0].set_ylabel("velocity")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    last_cycle = slice(steps_per_cycle, 2 * steps_per_cycle)
    phase = np.linspace(0.0, 360.0, steps_per_cycle, endpoint=False)
    axes[0, 1].plot(phase, lid_array[last_cycle], linewidth=2, label="moving lid")
    axes[0, 1].plot(phase, centre_array[last_cycle], linewidth=2, label="cavity centre")
    axes[0, 1].set_title("Second-Cycle Phase Response")
    axes[0, 1].set_xlabel("phase (degrees)")
    axes[0, 1].set_ylabel("velocity")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    speed_image = axes[1, 0].imshow(
        speed,
        extent=[0.0, 1.0, 0.0, 1.0],
        origin="lower",
        aspect="equal",
        cmap="viridis",
    )
    axes[1, 0].set_title("Final Cell-Centred Speed")
    axes[1, 0].set_xlabel("x / L")
    axes[1, 0].set_ylabel("y / L")
    figure.colorbar(speed_image, ax=axes[1, 0], label="normalized velocity")

    axes[1, 1].plot(u[:, nx // 2] / amplitude, y, linewidth=2)
    axes[1, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Final Centreline Response")
    axes[1, 1].set_xlabel("u / U_amplitude")
    axes[1, 1].set_ylabel("y / L")
    axes[1, 1].grid(True, alpha=0.3)

    figure.tight_layout()
    output = bt.get_result_path("oscillatory_cavity.png", EXAMPLE_NAME)
    figure.savefig(output, dpi=150)
    plt.close(figure)

    print(f"  Maximum final speed: {result.max_velocity:.5f}")
    print(f"  Maximum final divergence: {result.divergence:.3e}")
    print(f"  Plot saved to: {output}")
    return result


def example_reynolds_comparison() -> list[bt.NavierStokesResult]:
    """Compare equal-duration lid-driven transients at several Reynolds numbers.

    This is a qualitative laminar-grid comparison, not evidence of a transition
    threshold.  The coarse grid and lack of turbulence closure preclude a
    turbulence claim.
    """
    print("\n" + "=" * 68)
    print("Example 3: Bounded-Cavity Reynolds-Number Comparison")
    print("=" * 68)

    length = 1.0
    density = 1.0
    lid_velocity = 1.0
    reynolds_values = (1.0, 10.0, 50.0, 100.0)
    duration = 0.08 * length / lid_velocity
    nx = ny = 16
    cell_width = length / nx
    largest_kinematic_viscosity = lid_velocity * length / min(reynolds_values)
    time_step = 0.04 * cell_width**2 / largest_kinematic_viscosity
    print(f"  Common time step for a fair transient comparison: {time_step:.6f}")

    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    results: list[bt.NavierStokesResult] = []

    for axis, reynolds in zip(axes.flat, reynolds_values, strict=True):
        viscosity = density * lid_velocity * length / reynolds
        mesh = bt.StructuredMesh(nx, ny, 0.0, length, 0.0, length)
        solver = bt.NavierStokesSolver(mesh, density, viscosity)
        solver.set_convection_scheme(bt.ConvectionScheme.UPWIND)
        solver.set_time_step(time_step)
        solver.set_pressure_tolerance(1.0e-8)
        _set_closed_cavity_boundaries(solver, lid_velocity)

        print(f"  Solving Re = {reynolds:.0f}...")
        result = solver.solve(duration)
        _require_stable(result, f"Re={reynolds:.0f} cavity")
        results.append(result)
        u, v, _ = _cell_centered_fields(result, nx, ny)
        speed = np.hypot(u, v)

        image = axis.imshow(
            speed / lid_velocity,
            extent=[0.0, 1.0, 0.0, 1.0],
            origin="lower",
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        axis.streamplot(
            (np.arange(nx) + 0.5) / nx,
            (np.arange(ny) + 0.5) / ny,
            u,
            v,
            color="white",
            density=0.8,
            linewidth=0.55,
        )
        axis.set_title(f"Re = {reynolds:.0f}; {result.time_steps} fixed steps")
        axis.set_xlabel("x / L")
        axis.set_ylabel("y / L")
        figure.colorbar(image, ax=axis, label="|u| / U_lid")
        print(
            f"    max speed={result.max_velocity:.4f}, "
            f"max divergence={result.divergence:.2e}"
        )

    figure.suptitle(
        "Equal-Duration Laminar Cavity Transients "
        f"(t U_lid / L = {duration * lid_velocity / length:.2f})"
    )
    figure.tight_layout()
    output = bt.get_result_path("reynolds_comparison.png", EXAMPLE_NAME)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"  Plot saved to: {output}")
    return results


if __name__ == "__main__":
    print("Bounded Navier--Stokes examples")
    print("Closed, wall-driven laminar flow with a compatible MAC projection\n")

    example_lid_driven_cavity()
    example_oscillatory_lid()
    example_reynolds_comparison()

    print("\n" + "=" * 68)
    print("Navier--Stokes examples completed")
    print("=" * 68)
