#!/usr/bin/env python3
"""
Darcy Flow Example: Flow Through Porous Media
==============================================

This example demonstrates Darcy flow simulation for porous media transport,
applicable to:
- Interstitial fluid flow in biological tissues
- Drug delivery through tumor microenvironment
- Flow through tissue engineering scaffolds
- Groundwater flow (analogous physics)

Darcy's Law: v = -kappa * grad(p)
Where:
  - v is Darcy velocity [m/s]
  - kappa = K/mu is hydraulic conductivity [m^2/(Pa*s)]
  - K is permeability [m^2]
  - mu is dynamic viscosity [Pa*s]
  - p is pressure [Pa]

The pressure field satisfies: div(kappa * grad(p)) = 0 (steady-state)
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt


def run_uniform_flow():
    """
    Demonstrate uniform Darcy flow through a porous slab.

    Setup:
    - Rectangular domain with high pressure on left, low on right
    - No-flux (impermeable) boundaries on top and bottom
    - Uniform permeability

    Expected: Linear pressure drop, uniform horizontal velocity
    """
    print("=" * 60)
    print("Uniform Darcy Flow Through Porous Slab")
    print("=" * 60)

    # Domain: 10 cm x 5 cm (typical tissue sample)
    Lx, Ly = 0.1, 0.05  # meters
    nx, ny = 50, 25

    mesh = bt.StructuredMesh(nx, ny, 0.0, Lx, 0.0, Ly)

    # Tissue hydraulic conductivity (typical for soft tissue)
    # kappa = K/mu where K ~ 1e-14 m^2, mu ~ 1e-3 Pa*s
    kappa = 1e-11  # m^2/(Pa*s)

    # Pressure difference (physiological interstitial pressure)
    p_left = 1000.0  # Pa (~ 7.5 mmHg)
    p_right = 0.0  # Pa

    print("\nPhysical Parameters:")
    print(f"  Domain: {Lx*100:.1f} cm x {Ly*100:.1f} cm")
    print(f"  Grid: {nx} x {ny}")
    print(f"  Hydraulic conductivity: {kappa:.2e} m^2/(Pa*s)")
    print(f"  Pressure drop: {p_left:.0f} Pa over {Lx*100:.1f} cm")

    # Create solver with boundary conditions
    solver = (
        bt.DarcyFlowSolver(mesh, kappa)
        .set_dirichlet(bt.Boundary.Left, p_left)
        .set_dirichlet(bt.Boundary.Right, p_right)
        .set_neumann(bt.Boundary.Top, 0.0)  # No-flux
        .set_neumann(bt.Boundary.Bottom, 0.0)  # No-flux
        .set_omega(1.6)
        .set_tolerance(1e-8)
    )

    # Solve
    result = solver.solve()

    print("\nSolver Results:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final residual: {result.residual:.2e}")

    # Get fields
    p = result.pressure().reshape(ny + 1, nx + 1)
    vx = result.vx().reshape(ny + 1, nx + 1)
    vy = result.vy().reshape(ny + 1, nx + 1)
    v_mag = np.sqrt(vx**2 + vy**2)

    # Expected analytical solution
    dp_dx = (p_right - p_left) / Lx
    v_expected = -kappa * dp_dx  # Darcy velocity

    print("\nVelocity Analysis:")
    print(f"  Expected velocity: {v_expected*1e6:.2f} um/s")
    print(f"  Computed mean |v|: {np.mean(v_mag)*1e6:.2f} um/s")
    print(f"  Max |v|: {np.max(v_mag)*1e6:.2f} um/s")

    # Plot
    x = np.linspace(0, Lx * 100, nx + 1)  # cm
    y = np.linspace(0, Ly * 100, ny + 1)  # cm
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Pressure field
    c1 = axes[0].contourf(X, Y, p, levels=20, cmap="coolwarm")
    axes[0].set_xlabel("x (cm)")
    axes[0].set_ylabel("y (cm)")
    axes[0].set_title("Pressure Field (Pa)")
    axes[0].set_aspect("equal")
    plt.colorbar(c1, ax=axes[0])

    # Velocity magnitude
    c2 = axes[1].contourf(X, Y, v_mag * 1e6, levels=20, cmap="viridis")
    axes[1].set_xlabel("x (cm)")
    axes[1].set_ylabel("y (cm)")
    axes[1].set_title("Velocity Magnitude (um/s)")
    axes[1].set_aspect("equal")
    plt.colorbar(c2, ax=axes[1])

    # Streamlines
    axes[2].streamplot(X, Y, vx, vy, color=v_mag * 1e6, cmap="plasma", linewidth=1.5)
    axes[2].set_xlabel("x (cm)")
    axes[2].set_ylabel("y (cm)")
    axes[2].set_title("Flow Streamlines")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(bt.get_result_path("darcy_flow/uniform_flow.png"), dpi=150)
    print(f"\nSaved: {bt.get_result_path('darcy_flow/uniform_flow.png')}")
    plt.show()


def run_heterogeneous_medium():
    """
    Demonstrate Darcy flow through heterogeneous porous medium.

    Setup:
    - Variable permeability (low in center, higher on edges)
    - Models scenarios like:
      - Tumor with dense core surrounded by loose tissue
      - Layered tissue structures
    """
    print("\n" + "=" * 60)
    print("Heterogeneous Porous Medium")
    print("=" * 60)

    Lx, Ly = 0.1, 0.1  # 10 cm x 10 cm
    nx, ny = 50, 50

    mesh = bt.StructuredMesh(nx, ny, 0.0, Lx, 0.0, Ly)

    # Create heterogeneous permeability field
    # Low permeability in center (e.g., dense tumor core)
    kappa_base = 1e-11  # m^2/(Pa*s)
    kappa_low = 1e-13  # m^2/(Pa*s) - 100x lower in core

    X, Y = bt.xy_grid(mesh)

    # Gaussian low-permeability region in center
    cx, cy = Lx / 2, Ly / 2
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    sigma = 0.02  # 2 cm radius

    kappa_field = kappa_base - (kappa_base - kappa_low) * np.exp(
        -(r**2) / (2 * sigma**2)
    )
    kappa_list = kappa_field.flatten().tolist()

    print("\nPermeability Field:")
    print(f"  Background kappa: {kappa_base:.2e} m^2/(Pa*s)")
    print(f"  Core kappa: {kappa_low:.2e} m^2/(Pa*s)")
    print(f"  Ratio: {kappa_base/kappa_low:.0f}x")

    # Boundary conditions
    p_left = 1000.0
    p_right = 0.0

    solver = (
        bt.DarcyFlowSolver(mesh, kappa_list)
        .set_dirichlet(bt.Boundary.Left, p_left)
        .set_dirichlet(bt.Boundary.Right, p_right)
        .set_neumann(bt.Boundary.Top, 0.0)
        .set_neumann(bt.Boundary.Bottom, 0.0)
        .set_omega(1.5)
        .set_tolerance(1e-8)
        .set_max_iterations(20000)
    )

    result = solver.solve()

    print("\nSolver Results:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")

    p = result.pressure().reshape(ny + 1, nx + 1)
    vx = result.vx().reshape(ny + 1, nx + 1)
    vy = result.vy().reshape(ny + 1, nx + 1)
    v_mag = np.sqrt(vx**2 + vy**2)

    print("\nVelocity Statistics:")
    print(f"  Min |v|: {np.min(v_mag)*1e6:.2f} um/s")
    print(f"  Max |v|: {np.max(v_mag)*1e6:.2f} um/s")
    print(f"  Mean |v|: {np.mean(v_mag)*1e6:.2f} um/s")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    X_cm = X * 100
    Y_cm = Y * 100

    # Permeability field
    c1 = axes[0, 0].contourf(
        X_cm, Y_cm, np.log10(kappa_field), levels=20, cmap="RdYlBu_r"
    )
    axes[0, 0].set_xlabel("x (cm)")
    axes[0, 0].set_ylabel("y (cm)")
    axes[0, 0].set_title("log10(Permeability) [m^2/(Pa*s)]")
    axes[0, 0].set_aspect("equal")
    plt.colorbar(c1, ax=axes[0, 0])

    # Pressure field
    c2 = axes[0, 1].contourf(X_cm, Y_cm, p, levels=20, cmap="coolwarm")
    axes[0, 1].set_xlabel("x (cm)")
    axes[0, 1].set_ylabel("y (cm)")
    axes[0, 1].set_title("Pressure Field (Pa)")
    axes[0, 1].set_aspect("equal")
    plt.colorbar(c2, ax=axes[0, 1])

    # Velocity magnitude
    c3 = axes[1, 0].contourf(X_cm, Y_cm, v_mag * 1e6, levels=20, cmap="viridis")
    axes[1, 0].set_xlabel("x (cm)")
    axes[1, 0].set_ylabel("y (cm)")
    axes[1, 0].set_title("Velocity Magnitude (um/s)")
    axes[1, 0].set_aspect("equal")
    plt.colorbar(c3, ax=axes[1, 0])

    # Streamlines with quiver overlay
    skip = 3
    axes[1, 1].streamplot(X_cm, Y_cm, vx, vy, color="gray", linewidth=0.5, density=1.5)
    axes[1, 1].quiver(
        X_cm[::skip, ::skip],
        Y_cm[::skip, ::skip],
        vx[::skip, ::skip],
        vy[::skip, ::skip],
        v_mag[::skip, ::skip] * 1e6,
        cmap="plasma",
    )
    axes[1, 1].set_xlabel("x (cm)")
    axes[1, 1].set_ylabel("y (cm)")
    axes[1, 1].set_title("Flow Field (Streamlines + Vectors)")
    axes[1, 1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(bt.get_result_path("darcy_flow/heterogeneous_medium.png"), dpi=150)
    print(f"\nSaved: {bt.get_result_path('darcy_flow/heterogeneous_medium.png')}")
    plt.show()


def run_tumor_flow():
    """
    Simulate interstitial fluid pressure (IFP) in tumor microenvironment.

    This demonstrates a key biotransport phenomenon:
    - Elevated interstitial pressure in tumors impedes drug delivery
    - Fluid flows from high-pressure tumor core to surrounding tissue

    Setup:
    - Circular tumor region with elevated pressure (Dirichlet)
    - Normal tissue with lower pressure at boundary
    - Demonstrates radial pressure distribution
    """
    print("\n" + "=" * 60)
    print("Tumor Interstitial Fluid Pressure (IFP)")
    print("=" * 60)

    Lx, Ly = 0.04, 0.04  # 4 cm x 4 cm domain
    nx, ny = 80, 80

    mesh = bt.StructuredMesh(nx, ny, 0.0, Lx, 0.0, Ly)

    # Uniform permeability for simplicity
    kappa = 1e-11  # m^2/(Pa*s)

    # Create tumor mask (circular region in center)
    tumor_radius = 0.008  # 8 mm tumor
    cx, cy = Lx / 2, Ly / 2

    X, Y = bt.xy_grid(mesh)
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    tumor_mask = (r <= tumor_radius).astype(int).ravel().tolist()

    # Pressures
    p_tumor = 2000.0  # Pa (~ 15 mmHg, typical elevated IFP)
    p_boundary = 500.0  # Pa (~ 3.75 mmHg, normal tissue)

    print("\nTumor Model Parameters:")
    print(f"  Domain: {Lx*100:.1f} cm x {Ly*100:.1f} cm")
    print(f"  Tumor radius: {tumor_radius*1000:.1f} mm")
    print(f"  Tumor IFP: {p_tumor:.0f} Pa ({p_tumor/133.322:.1f} mmHg)")
    print(f"  Normal tissue IFP: {p_boundary:.0f} Pa ({p_boundary/133.322:.1f} mmHg)")

    solver = (
        bt.DarcyFlowSolver(mesh, kappa)
        .set_internal_pressure(tumor_mask, p_tumor)
        .set_dirichlet(bt.Boundary.Left, p_boundary)
        .set_dirichlet(bt.Boundary.Right, p_boundary)
        .set_dirichlet(bt.Boundary.Top, p_boundary)
        .set_dirichlet(bt.Boundary.Bottom, p_boundary)
        .set_omega(1.7)
        .set_tolerance(1e-9)
        .set_max_iterations(50000)
    )

    result = solver.solve()

    print("\nSolver Results:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")

    p = result.pressure().reshape(ny + 1, nx + 1)
    vx = result.vx().reshape(ny + 1, nx + 1)
    vy = result.vy().reshape(ny + 1, nx + 1)
    v_mag = np.sqrt(vx**2 + vy**2)
    tumor_mask_2d = np.array(tumor_mask).reshape(ny + 1, nx + 1)

    print("\nFlow Analysis:")
    print(f"  Max velocity at tumor edge: {np.max(v_mag)*1e6:.2f} um/s")
    print(
        f"  Mean velocity in tissue: {np.mean(v_mag[tumor_mask_2d == 0])*1e6:.2f} um/s"
    )

    # Plot
    x = np.linspace(0, Lx * 100, nx + 1)  # cm
    y = np.linspace(0, Ly * 100, ny + 1)  # cm
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Pressure field with tumor outline
    c1 = axes[0].contourf(X, Y, p / 133.322, levels=20, cmap="hot")  # Convert to mmHg
    axes[0].contour(X, Y, tumor_mask_2d, levels=[0.5], colors="white", linewidths=2)
    axes[0].set_xlabel("x (cm)")
    axes[0].set_ylabel("y (cm)")
    axes[0].set_title("Interstitial Fluid Pressure (mmHg)")
    axes[0].set_aspect("equal")
    plt.colorbar(c1, ax=axes[0], label="IFP (mmHg)")

    # Velocity magnitude
    c2 = axes[1].contourf(X, Y, v_mag * 1e6, levels=20, cmap="viridis")
    axes[1].contour(X, Y, tumor_mask_2d, levels=[0.5], colors="white", linewidths=2)
    axes[1].set_xlabel("x (cm)")
    axes[1].set_ylabel("y (cm)")
    axes[1].set_title("IFV Magnitude (um/s)")
    axes[1].set_aspect("equal")
    plt.colorbar(c2, ax=axes[1], label="Velocity (um/s)")

    # Velocity vectors
    skip = 5
    axes[2].quiver(
        X[::skip, ::skip],
        Y[::skip, ::skip],
        vx[::skip, ::skip],
        vy[::skip, ::skip],
        v_mag[::skip, ::skip] * 1e6,
        cmap="plasma",
    )
    axes[2].contour(X, Y, tumor_mask_2d, levels=[0.5], colors="black", linewidths=2)
    axes[2].set_xlabel("x (cm)")
    axes[2].set_ylabel("y (cm)")
    axes[2].set_title("Interstitial Fluid Velocity Vectors")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(bt.get_result_path("darcy_flow/tumor_ifp.png"), dpi=150)
    print(f"\nSaved: {bt.get_result_path('darcy_flow/tumor_ifp.png')}")
    plt.show()

    # Radial profile plot
    fig2, ax = plt.subplots(figsize=(8, 5))

    # Sample pressure along centerline
    mid_j = ny // 2
    x_line = x
    p_line = p[mid_j, :]

    ax.plot(x_line, p_line / 133.322, "b-", linewidth=2, label="Centerline pressure")
    ax.axhline(p_tumor / 133.322, color="r", linestyle="--", label="Tumor IFP")
    ax.axhline(p_boundary / 133.322, color="g", linestyle="--", label="Boundary IFP")
    ax.axvline((cx - tumor_radius) * 100, color="gray", linestyle=":", alpha=0.5)
    ax.axvline((cx + tumor_radius) * 100, color="gray", linestyle=":", alpha=0.5)
    ax.fill_betweenx(
        [0, p_tumor / 133.322 + 1],
        (cx - tumor_radius) * 100,
        (cx + tumor_radius) * 100,
        alpha=0.2,
        color="red",
        label="Tumor region",
    )

    ax.set_xlabel("x (cm)")
    ax.set_ylabel("Interstitial Fluid Pressure (mmHg)")
    ax.set_title("IFP Profile Across Tumor")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(bt.get_result_path("darcy_flow/tumor_ifp_profile.png"), dpi=150)
    print(f"Saved: {bt.get_result_path('darcy_flow/tumor_ifp_profile.png')}")
    plt.show()


if __name__ == "__main__":
    # Create output directory
    import os

    os.makedirs(bt.get_result_path("darcy_flow"), exist_ok=True)

    # Run demonstrations
    run_uniform_flow()
    run_heterogeneous_medium()
    run_tumor_flow()

    print("\n" + "=" * 60)
    print("All Darcy flow examples completed!")
    print("=" * 60)
