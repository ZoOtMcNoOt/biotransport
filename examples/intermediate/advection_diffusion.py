#!/usr/bin/env python3
"""
Advection-Diffusion Example: Drug Transport in Blood Vessel

This example demonstrates the advection-diffusion solver for modeling
solute transport in flowing blood. The governing equation is:

    ∂C/∂t + v·∇C = D∇²C

where:
- C is the drug concentration
- v is the blood velocity (assumed uniform for simplicity)
- D is the drug diffusivity in blood

The example shows:
1. A bolus injection (Gaussian pulse) being carried downstream
2. The effect of Peclet number on transport regime
3. Comparison of upwind vs central differencing schemes

BMEN 341 Reference: Weeks 5-6 (Convection-Diffusion, Peclet Number)
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt


def run_advection_diffusion_1d():
    """Run 1D advection-diffusion simulation."""
    print("=" * 60)
    print("1D Drug Bolus Transport in Blood Vessel")
    print("=" * 60)

    # Domain: 10 cm vessel segment
    L = 0.1  # 10 cm in meters
    nx = 200
    mesh = bt.mesh_1d(nx, x_max=L)

    # Physical parameters
    D = 1e-9  # Drug diffusivity in blood (typical small molecule) [m²/s]
    v_blood = 0.01  # Blood velocity (slow capillary flow) [m/s]

    # Calculate Peclet number (dimensionless ratio of advection to diffusion)
    Pe = bt.dimensionless.peclet(v_blood, L, D)
    Pe_cell = v_blood * mesh.dx() / D
    print("\nPhysical Parameters:")
    print(f"  Vessel length: {L * 100:.1f} cm")
    print(f"  Blood velocity: {v_blood * 100:.1f} cm/s")
    print(f"  Drug diffusivity: {D:.2e} m²/s")
    print(f"  Domain Peclet: {Pe:.2e}")
    print(f"  Cell Peclet: {Pe_cell:.1f}")

    # Initial condition: Gaussian bolus at x = 2 cm
    # Using the gaussian helper with center and width in domain coordinates
    x0 = 0.02
    sigma = 0.005  # 5 mm spread
    ic = np.array(bt.gaussian(mesh, center=x0, width=sigma))

    # Simulation time
    t_end = 5.0  # 5 seconds

    # Compare schemes
    schemes = [
        ("Upwind", bt.AdvectionScheme.UPWIND),
        ("Hybrid", bt.AdvectionScheme.HYBRID),
    ]

    results = {}
    for name, scheme in schemes:
        problem = (
            bt.Problem(mesh)
            .diffusivity(D)
            .velocity(v_blood)
            .advection_scheme(scheme)
            .initial_condition(ic.tolist())
            .dirichlet(bt.Boundary.Left, 0.0)
            .neumann(bt.Boundary.Right, 0.0)
        )

        result = bt.solve(problem, t=t_end, safety_factor=0.4)

        results[name] = result
        print(f"\n{name} scheme:")
        print(f"  Steps: {result.stats.steps}")
        print(f"  dt: {result.stats.dt:.2e} s")
        print(f"  Wall time: {result.stats.wall_time_s:.3f} s")

    # Expected final position
    expected_x = x0 + v_blood * t_end
    print(f"\nExpected bolus center after {t_end}s: {expected_x * 100:.1f} cm")

    # Plot results
    x = bt.x_nodes(mesh)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Initial condition
    axes[0].plot(x * 100, ic, "k--", linewidth=2, label="Initial (t=0)")
    for name, result in results.items():
        sol = result.solution()
        axes[0].plot(x * 100, sol, linewidth=2, label=f"{name} (t={t_end}s)")
    axes[0].axvline(
        expected_x * 100, color="gray", linestyle=":", label="Expected center"
    )
    axes[0].set_xlabel("Position (cm)")
    axes[0].set_ylabel("Concentration (normalized)")
    axes[0].set_title("Drug Bolus Transport")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Zoom on the bolus
    axes[1].plot(x * 100, ic, "k--", linewidth=2, label="Initial")
    for name, result in results.items():
        sol = result.solution()
        axes[1].plot(x * 100, sol, linewidth=2, label=f"{name}")
    axes[1].set_xlim([expected_x * 100 - 3, expected_x * 100 + 3])
    axes[1].set_xlabel("Position (cm)")
    axes[1].set_ylabel("Concentration")
    axes[1].set_title("Zoomed View at Final Position")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        bt.get_result_path("bolus_transport_1d.png", "advection_diffusion"), dpi=150
    )
    print(
        f"\nSaved: {bt.get_result_path('bolus_transport_1d.png', 'advection_diffusion')}"
    )
    plt.show()


def run_peclet_comparison():
    """Compare transport at different Peclet numbers."""
    print("\n" + "=" * 60)
    print("Effect of Peclet Number on Transport")
    print("=" * 60)

    L = 0.1
    nx = 100
    mesh = bt.mesh_1d(nx, x_max=L)

    # Different Peclet regimes:
    # Pe < 1: diffusion dominates (spreading)
    # Pe > 1: advection dominates (translation)
    # Pe >> 1: strongly convective (minimal spreading)
    cases = [
        {"D": 1e-4, "v": 0.001, "label": "Diffusion-dominated (Pe~1)"},
        {"D": 1e-6, "v": 0.01, "label": "Convection-dominated (Pe~100)"},
        {"D": 1e-8, "v": 0.05, "label": "Strongly convective (Pe~5000)"},
    ]

    x0 = 0.02
    sigma = 0.005
    ic = np.array(bt.gaussian(mesh, center=x0, width=sigma))
    t_end = 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    x = bt.x_nodes(mesh)
    ax.plot(x * 100, ic, "k--", linewidth=2, label="Initial")

    for case in cases:
        Pe = bt.dimensionless.peclet(case["v"], L, case["D"])

        problem = (
            bt.Problem(mesh)
            .diffusivity(case["D"])
            .velocity(case["v"])
            .advection_scheme(bt.AdvectionScheme.HYBRID)
            .initial_condition(ic.tolist())
            .dirichlet(bt.Boundary.Left, 0.0)
            .neumann(bt.Boundary.Right, 0.0)
        )

        result = bt.solve(problem, t=t_end)
        sol = result.solution()
        ax.plot(x * 100, sol, linewidth=2, label=f"{case['label']}")

        print(f"{case['label']}: Pe = {Pe:.0f}, steps = {result.stats.steps}")

    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Concentration")
    ax.set_title(f"Advection-Diffusion at Different Peclet Numbers (t={t_end}s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        bt.get_result_path("peclet_comparison.png", "advection_diffusion"), dpi=150
    )
    print(
        f"\nSaved: {bt.get_result_path('peclet_comparison.png', 'advection_diffusion')}"
    )
    plt.show()


def run_2d_transport():
    """2D advection-diffusion with varying velocity field."""
    print("\n" + "=" * 60)
    print("2D Drug Transport (Channel Flow)")
    print("=" * 60)

    # 2D channel: 8 cm × 2 cm
    nx, ny = 80, 40
    Lx, Ly = 0.08, 0.02
    mesh = bt.mesh_2d(nx, ny, x_max=Lx, y_max=Ly)

    D = 1e-7  # m²/s
    v_mean = 0.01  # m/s mean flow velocity

    # Initial: circular bolus near inlet
    # Using circle() helper would give sharp boundaries; we want smooth Gaussian
    X, Y = bt.xy_grid(mesh)
    x0, y0 = 0.01, 0.01  # Center of bolus
    r0 = 0.005  # Characteristic radius
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    ic = np.exp(-r2 / (r0**2)).reshape(-1)

    # Uniform velocity in +x direction (Poiseuille profile could be added)
    problem = (
        bt.Problem(mesh)
        .diffusivity(D)
        .velocity(v_mean, 0.0)  # vx, vy
        .advection_scheme(bt.AdvectionScheme.HYBRID)
        .initial_condition(ic.tolist())
        .dirichlet(bt.Boundary.Left, 0.0)
        .neumann(bt.Boundary.Right, 0.0)
        .neumann(bt.Boundary.Bottom, 0.0)
        .neumann(bt.Boundary.Top, 0.0)
    )

    # Time snapshots
    times = [0.0, 1.0, 2.0, 4.0]
    snapshots = [ic.copy()]
    current_t = 0.0

    for t in times[1:]:
        dt = t - current_t
        result = bt.solve(problem, t=dt, safety_factor=0.3)
        snapshots.append(result.solution().copy())
        # Update IC for next segment
        problem.initial_condition(result.solution().tolist())
        current_t = t
        print(f"t = {t:.1f}s: steps = {result.stats.steps}")

    # Plot snapshots
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes = axes.flatten()

    x = bt.x_nodes(mesh)
    y = bt.y_nodes(mesh)
    X, Y = np.meshgrid(x * 100, y * 100)

    for ax, t, sol in zip(axes, times, snapshots):
        Z = np.array(sol).reshape(ny + 1, nx + 1)
        c = ax.pcolormesh(X, Y, Z, cmap="viridis", shading="auto")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title(f"t = {t:.1f} s")
        ax.set_aspect("equal")
        plt.colorbar(c, ax=ax, label="C")

    plt.suptitle("2D Drug Transport in Channel Flow", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        bt.get_result_path("channel_transport_2d.png", "advection_diffusion"), dpi=150
    )
    print(
        f"\nSaved: {bt.get_result_path('channel_transport_2d.png', 'advection_diffusion')}"
    )
    plt.show()


if __name__ == "__main__":
    run_advection_diffusion_1d()
    run_peclet_comparison()
    run_2d_transport()

    print("\n" + "=" * 60)
    print("All advection-diffusion examples completed!")
    print("=" * 60)
