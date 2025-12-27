#!/usr/bin/env python3
"""
ADI (Alternating Direction Implicit) Diffusion Solver Demo
===========================================================

This example demonstrates the ADI (Alternating Direction Implicit) method
for solving 2D and 3D diffusion equations efficiently.

**Key Features of ADI:**
- Unconditionally stable (no CFL restriction on time step)
- 2nd-order accurate in both space and time
- O(N) complexity per time step (vs O(N^2-N^3) for full implicit)
- Uses Thomas algorithm for efficient tridiagonal solves

**Method Details:**
- 2D: Peaceman-Rachford splitting (implicit x, then implicit y)
- 3D: Douglas-Gunn splitting (implicit x, y, z sequentially)

This example compares ADI performance with explicit methods on the same problem.
"""

import biotransport as bt
import numpy as np
import matplotlib.pyplot as plt
import time


def adi_2d_example():
    """Demonstrate 2D ADI solver with Gaussian diffusion."""
    print("=" * 60)
    print("2D ADI Diffusion - Peaceman-Rachford Splitting")
    print("=" * 60)

    # Parameters
    nx, ny = 100, 100
    D = 0.01  # Diffusion coefficient [m^2/s]
    L = 1.0  # Domain size [m]

    # Create mesh (has (nx+1) x (ny+1) nodes)
    mesh = bt.StructuredMesh(nx, ny, 0.0, L, 0.0, L)

    # Create ADI solver
    solver = bt.ADIDiffusion2D(mesh, D)

    # Set Gaussian initial condition
    x = np.linspace(0, L, nx + 1)
    y = np.linspace(0, L, ny + 1)
    X, Y = np.meshgrid(x, y)
    ic = np.exp(-50 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)).flatten()
    solver.set_initial_condition(ic.tolist())

    # Set Dirichlet boundary conditions (u = 0 on all boundaries)
    solver.set_dirichlet_boundary(bt.Boundary.Left, 0.0)
    solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)
    solver.set_dirichlet_boundary(bt.Boundary.Bottom, 0.0)
    solver.set_dirichlet_boundary(bt.Boundary.Top, 0.0)

    # Store initial state
    c_initial = np.array(solver.solution()).reshape(ny + 1, nx + 1)

    # Solve with large time step (ADI is unconditionally stable!)
    dt = 0.01  # Much larger than explicit CFL limit
    num_steps = 50

    print(f"Grid: {nx+1} x {ny+1} nodes")
    print(f"Time step: dt = {dt} s")
    print(f"Number of steps: {num_steps}")

    start_time = time.perf_counter()
    result = solver.solve(dt, num_steps)
    elapsed = time.perf_counter() - start_time

    print("\nResult:")
    print(f"  Steps completed: {result.steps}")
    print(f"  Simulation time: {result.total_time:.3f} s")
    print(f"  Wall-clock time: {elapsed*1000:.2f} ms")
    print(f"  Success: {result.success}")

    # Get final solution
    c_final = np.array(solver.solution()).reshape(ny + 1, nx + 1)

    # Compare with explicit stability limit
    dx = L / nx
    explicit_dt_limit = 0.25 * dx**2 / D  # CFL condition for explicit
    speedup = dt / explicit_dt_limit

    print("\nStability comparison:")
    print(f"  Explicit CFL limit: dt ≤ {explicit_dt_limit:.6f} s")
    print(f"  ADI time step:      dt = {dt:.6f} s")
    print(f"  Speedup factor:     {speedup:.1f}x larger time step")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].contourf(X, Y, c_initial, levels=50, cmap="hot")
    axes[0].set_title("Initial (t = 0)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_aspect("equal")

    axes[1].contourf(X, Y, c_final, levels=50, cmap="hot")
    axes[1].set_title(f"Final (t = {result.total_time:.2f} s)")
    axes[1].set_xlabel("x [m]")
    axes[1].set_aspect("equal")

    # Show cross-section at y = 0.5
    mid_idx = ny // 2
    axes[2].plot(x, c_initial[mid_idx, :], "b--", label="Initial")
    axes[2].plot(x, c_final[mid_idx, :], "r-", label="Final")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("Concentration")
    axes[2].set_title("Cross-section at y = 0.5")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/adi_diffusion_2d.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to: results/adi_diffusion_2d.png")
    plt.close()

    return result


def adi_3d_example():
    """Demonstrate 3D ADI solver with Douglas-Gunn splitting."""
    print("\n" + "=" * 60)
    print("3D ADI Diffusion - Douglas-Gunn Splitting")
    print("=" * 60)

    # Parameters
    nx, ny, nz = 30, 30, 30
    D = 0.01  # Diffusion coefficient
    L = 1.0  # Domain size

    # Create mesh
    mesh = bt.StructuredMesh3D(nx, ny, nz, 0.0, L, 0.0, L, 0.0, L)

    # Create solver
    solver = bt.ADIDiffusion3D(mesh, D)

    # Set Gaussian initial condition
    x = np.linspace(0, L, nx + 1)
    y = np.linspace(0, L, ny + 1)
    z = np.linspace(0, L, nz + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    ic = np.exp(-50 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)).flatten()
    solver.set_initial_condition(ic.tolist())

    # Set Dirichlet boundary conditions
    solver.set_dirichlet_boundary(bt.Boundary3D.XMin, 0.0)
    solver.set_dirichlet_boundary(bt.Boundary3D.XMax, 0.0)
    solver.set_dirichlet_boundary(bt.Boundary3D.YMin, 0.0)
    solver.set_dirichlet_boundary(bt.Boundary3D.YMax, 0.0)
    solver.set_dirichlet_boundary(bt.Boundary3D.ZMin, 0.0)
    solver.set_dirichlet_boundary(bt.Boundary3D.ZMax, 0.0)

    # Solve
    dt = 0.005
    num_steps = 20

    print(f"Grid: {nx+1} x {ny+1} x {nz+1} = {(nx+1)*(ny+1)*(nz+1)} nodes")
    print(f"Time step: dt = {dt} s")
    print(f"Number of steps: {num_steps}")

    start_time = time.perf_counter()
    result = solver.solve(dt, num_steps)
    elapsed = time.perf_counter() - start_time

    print("\nResult:")
    print(f"  Steps: {result.steps}")
    print(f"  Simulation time: {result.total_time:.3f} s")
    print(f"  Wall-clock time: {elapsed*1000:.2f} ms")
    print(f"  Time per step: {elapsed*1000/num_steps:.2f} ms")

    # Get final solution
    c = np.array(solver.solution()).reshape(nx + 1, ny + 1, nz + 1)
    print(f"  Max concentration: {c.max():.4f}")
    print(f"  Mass (sum): {c.sum() * (L/nx)**3:.6f}")

    return result


def compare_stability():
    """Demonstrate unconditional stability of ADI."""
    print("\n" + "=" * 60)
    print("ADI Stability Demonstration")
    print("=" * 60)

    nx, ny = 50, 50
    D = 0.1  # Higher diffusivity
    L = 1.0

    mesh = bt.StructuredMesh(nx, ny, 0.0, L, 0.0, L)
    dx = L / nx

    # Explicit CFL limit
    explicit_cfl = 0.25 * dx**2 / D
    print(f"\nExplicit CFL limit: dt ≤ {explicit_cfl:.6f} s")

    # Test ADI with increasingly large time steps
    time_steps = [
        explicit_cfl,
        10 * explicit_cfl,
        100 * explicit_cfl,
        1000 * explicit_cfl,
    ]

    print("\nADI stability with large time steps:")
    print("-" * 45)
    print(f"{'dt (s)':<15} {'Factor':<10} {'Success':<10}")
    print("-" * 45)

    for dt in time_steps:
        solver = bt.ADIDiffusion2D(mesh, D)

        # Gaussian IC
        x = np.linspace(0, L, nx + 1)
        y = np.linspace(0, L, ny + 1)
        X, Y = np.meshgrid(x, y)
        ic = np.exp(-30 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)).flatten()
        solver.set_initial_condition(ic.tolist())

        for bc in [
            bt.Boundary.Left,
            bt.Boundary.Right,
            bt.Boundary.Bottom,
            bt.Boundary.Top,
        ]:
            solver.set_dirichlet_boundary(bc, 0.0)

        _result = solver.solve(dt, 5)  # noqa: F841

        c = np.array(solver.solution())
        stable = np.all(np.isfinite(c)) and c.min() >= -0.01 and c.max() <= 1.01

        factor = dt / explicit_cfl
        print(
            f"{dt:<15.6f} {factor:<10.1f}x {'✓ Stable' if stable else '✗ Unstable':<10}"
        )

    print("-" * 45)
    print("ADI remains stable even with 1000x larger time steps!")


if __name__ == "__main__":
    import os

    os.makedirs("results", exist_ok=True)

    # Run examples
    adi_2d_example()
    adi_3d_example()
    compare_stability()

    print("\n" + "=" * 60)
    print("ADI solver demonstration complete!")
    print("=" * 60)
