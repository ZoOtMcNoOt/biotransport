#!/usr/bin/env python3
"""
Cylindrical Coordinate Examples - Pipe Flow and Radial Diffusion

This example demonstrates problems in cylindrical coordinates,
which are essential for modeling blood vessels and catheters.

BMEN 341 Relevance:
- Part II: Pipe flow (Poiseuille flow in cylindrical tubes)
- Part I: Radial diffusion (drug release from cylindrical implants)
- Axisymmetric problems (2D simulation of 3D cylindrical geometry)

Cylindrical coordinates: (r, θ, z)
For axisymmetric problems (no θ dependence), we solve in (r, z).
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt


def example_radial_diffusion():
    """
    Radial diffusion from a cylindrical drug eluting stent.

    Drug diffuses radially outward from a cylindrical implant
    into the surrounding tissue.

    Governing equation in cylindrical coordinates:
        ∂C/∂t = D * (1/r) * ∂/∂r(r * ∂C/∂r)
    """
    print("=" * 60)
    print("Example 1: Radial Drug Diffusion from Cylindrical Stent")
    print("=" * 60)

    # Geometry
    R_stent = 1.5e-3  # Stent radius: 1.5 mm
    R_tissue = 5.0e-3  # Tissue outer boundary: 5 mm

    # Drug properties
    D = 1e-11  # Diffusion coefficient: 10⁻¹¹ m²/s (typical for drug in tissue)
    C_stent = 1.0  # Normalized drug concentration at stent surface

    # Create 1D radial mesh
    nr_cells = 100  # Number of cells
    mesh = bt.CylindricalMesh(nr_cells, R_stent, R_tissue)

    # Get radial coordinates using the convenience helper
    r = bt.r_nodes(mesh)  # Returns nr_cells + 1 nodes
    nr = len(r)  # Number of nodes
    dr = mesh.dr()

    print(
        f"  Radial mesh: {nr} nodes from r = {R_stent*1000:.1f} to {R_tissue*1000:.1f} mm"
    )
    print(f"  Mesh type: {mesh.type()}")
    print(f"  Radial spacing: dr = {dr*1e6:.1f} um")

    # Initial condition: no drug in tissue
    C = np.zeros(nr)

    # Boundary conditions
    C[0] = C_stent  # Fixed concentration at stent surface
    # Zero flux at outer boundary (tissue far from stent)

    # Time stepping parameters
    dt = 0.5 * dr**2 / D  # Stability criterion
    t_total = 24 * 3600  # 24 hours in seconds

    print(f"  Time step: {dt:.1f} s")
    print(f"  Total simulation time: {t_total/3600:.0f} hours")

    # Store snapshots
    snapshot_times = [0, 1, 6, 12, 24]  # hours
    snapshots = {0: C.copy()}

    t = 0.0
    n_steps = 0

    while t < t_total:
        # Apply diffusion equation in cylindrical coords
        # ∂C/∂t = D * (1/r) * ∂/∂r(r * ∂C/∂r)
        C_new = C.copy()

        for i in range(1, nr - 1):
            # Central difference for cylindrical Laplacian
            dCdr = (C[i + 1] - C[i - 1]) / (2 * dr)
            d2Cdr2 = (C[i + 1] - 2 * C[i] + C[i - 1]) / dr**2

            laplacian_cyl = d2Cdr2 + dCdr / r[i]
            C_new[i] = C[i] + dt * D * laplacian_cyl

        # Boundary conditions
        C_new[0] = C_stent
        C_new[-1] = C_new[-2]  # Zero flux: dC/dr = 0

        C = C_new
        t += dt
        n_steps += 1

        # Save snapshots
        t_hours = t / 3600
        for ts in snapshot_times:
            if abs(t_hours - ts) < dt / 3600 and ts not in snapshots:
                snapshots[ts] = C.copy()
                print(f"    Snapshot at t = {ts:.0f} hours")

    # Final snapshot
    snapshots[24] = C.copy()

    print(f"  Completed {n_steps} time steps")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Concentration profiles at different times
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_times)))

    for (t_h, C_snap), color in zip(sorted(snapshots.items()), colors):
        ax.plot(
            (r - R_stent) * 1000,
            C_snap,
            "-",
            linewidth=2,
            label=f"t = {t_h:.0f} h",
            color=color,
        )

    ax.set_xlabel("Distance from stent surface (mm)")
    ax.set_ylabel("Normalized drug concentration")
    ax.set_title("Radial Drug Diffusion from Stent")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, (R_tissue - R_stent) * 1000])

    # 2D visualization (r-θ view)
    ax = axes[1]
    theta = np.linspace(0, 2 * np.pi, 100)
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Expand concentration to 2D
    C_2d = np.tile(snapshots[24], (len(theta), 1))

    im = ax.pcolormesh(X * 1000, Y * 1000, C_2d, cmap="YlOrRd", shading="auto")

    # Draw stent outline
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax.plot(
        R_stent * 1000 * np.cos(theta_circle),
        R_stent * 1000 * np.sin(theta_circle),
        "k-",
        linewidth=2,
        label="Stent",
    )

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Drug Distribution at t = 24 hours")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="C/C₀")

    plt.tight_layout()
    plt.savefig(bt.get_result_path("cylindrical_radial_diffusion.png"), dpi=150)
    plt.close()

    print("  Plot saved to results/cylindrical_radial_diffusion.png")

    # Calculate drug penetration depth (where C = 0.1 * C_stent)
    threshold = 0.1
    penetration_idx = np.where(snapshots[24] < threshold)[0]
    if len(penetration_idx) > 0:
        penetration_depth = (r[penetration_idx[0]] - R_stent) * 1000
        print(f"  Drug penetration depth (C > 10%): {penetration_depth:.2f} mm")


def example_axisymmetric_diffusion():
    """
    Axisymmetric diffusion in (r, z) coordinates.

    Models drug diffusion from a finite-length cylindrical implant,
    accounting for both radial and axial diffusion.
    """
    print("\n" + "=" * 60)
    print("Example 2: Axisymmetric Diffusion (r-z plane)")
    print("=" * 60)

    # Geometry
    R_max = 5.0e-3  # Radial extent: 5 mm
    Z_max = 10.0e-3  # Axial extent: 10 mm

    # Implant dimensions
    R_implant = 1.0e-3  # Implant radius: 1 mm
    Z_implant = 4.0e-3  # Implant half-length: 4 mm (centered)

    # Create axisymmetric mesh
    nr, nz = 51, 101
    mesh = bt.CylindricalMesh(nr, nz, R_max, Z_max)

    print(f"  Mesh: {nr} x {nz} points (r x z)")
    print(f"  Domain: r ∈ [0, {R_max*1000:.1f}] mm, z ∈ [0, {Z_max*1000:.1f}] mm")
    print(f"  Mesh type: {mesh.type()}")

    # Get coordinates
    r = mesh.r_coordinates()
    z = mesh.z_coordinates()
    dr = r[1] - r[0]
    dz = z[1] - z[0]

    R, Z = np.meshgrid(r, z, indexing="ij")  # (nr, nz) arrays

    # Drug properties
    D = 5e-11  # m²/s
    C0 = 1.0  # Initial concentration in implant

    # Initial condition: drug only in implant region (vectorized)
    R_mesh, Z_mesh = np.meshgrid(r, z, indexing="ij")
    z_from_center = np.abs(Z_mesh - Z_max / 2)
    in_implant = (R_mesh <= R_implant) & (z_from_center <= Z_implant / 2)
    C = np.where(in_implant, C0, 0.0)

    total_initial = np.sum(C * 2 * np.pi * R * dr * dz)
    print(f"  Initial drug mass (normalized): {total_initial:.4e}")

    # Time stepping
    dt = 0.25 * min(dr**2, dz**2) / D
    t_total = 12 * 3600  # 12 hours

    print(f"  Time step: {dt:.2f} s")
    print(f"  Simulating for {t_total/3600:.0f} hours...")

    # Store snapshots
    snapshots = {0: C.copy()}
    snapshot_times = [0, 3, 6, 12]

    t = 0.0
    n_steps = 0

    while t < t_total:
        C_new = C.copy()

        for i in range(1, nr - 1):
            for j in range(1, nz - 1):
                # Cylindrical Laplacian: ∂²C/∂r² + (1/r)∂C/∂r + ∂²C/∂z²
                d2Cdr2 = (C[i + 1, j] - 2 * C[i, j] + C[i - 1, j]) / dr**2
                dCdr = (C[i + 1, j] - C[i - 1, j]) / (2 * dr)
                d2Cdz2 = (C[i, j + 1] - 2 * C[i, j] + C[i, j - 1]) / dz**2

                # Handle r = 0 singularity with L'Hopital's rule
                if r[i] < 1e-10:
                    laplacian = 2 * d2Cdr2 + d2Cdz2
                else:
                    laplacian = d2Cdr2 + dCdr / r[i] + d2Cdz2

                C_new[i, j] = C[i, j] + dt * D * laplacian

        # Boundary conditions: axis symmetry at r=0, zero flux elsewhere
        C_new[0, :] = C_new[1, :]  # Symmetry at r = 0
        C_new[-1, :] = C_new[-2, :]  # Zero flux at r = R_max
        C_new[:, 0] = C_new[:, 1]  # Zero flux at z = 0
        C_new[:, -1] = C_new[:, -2]  # Zero flux at z = Z_max

        C = C_new
        t += dt
        n_steps += 1

        # Save snapshots
        t_hours = t / 3600
        for ts in snapshot_times:
            if abs(t_hours - ts) < dt / 3600 and ts not in snapshots:
                snapshots[ts] = C.copy()
                print(f"    Snapshot at t = {ts:.0f} hours")

    snapshots[12] = C.copy()
    print(f"  Completed {n_steps} time steps")

    # Calculate remaining drug
    total_final = np.sum(C * 2 * np.pi * R * dr * dz)
    print(f"  Final drug mass (normalized): {total_final:.4e}")
    print(f"  Drug released: {(1 - total_final/total_initial)*100:.1f}%")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, t_h in enumerate(snapshot_times):
        ax = axes[idx // 2, idx % 2]
        im = ax.contourf(Z * 1000, R * 1000, snapshots[t_h], levels=20, cmap="YlOrRd")
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("r (mm)")
        ax.set_title(f"t = {t_h:.0f} hours")
        plt.colorbar(im, ax=ax, label="C/C₀")

        # Draw implant outline
        z_imp_left = (Z_max / 2 - Z_implant / 2) * 1000
        z_imp_right = (Z_max / 2 + Z_implant / 2) * 1000
        r_imp = R_implant * 1000
        ax.plot(
            [z_imp_left, z_imp_right, z_imp_right, z_imp_left, z_imp_left],
            [0, 0, r_imp, r_imp, 0],
            "k--",
            linewidth=2,
        )

    plt.suptitle("Axisymmetric Drug Diffusion from Cylindrical Implant", fontsize=14)
    plt.tight_layout()
    plt.savefig(bt.get_result_path("cylindrical_axisymmetric_diffusion.png"), dpi=150)
    plt.close()

    print("  Plot saved to results/cylindrical_axisymmetric_diffusion.png")


def example_pipe_flow_poiseuille():
    """
    Steady Poiseuille flow in a cylindrical pipe.

    Analytical solution:
        u_z(r) = (ΔP/4μL) * (R² - r²)
    """
    print("\n" + "=" * 60)
    print("Example 3: Poiseuille Flow in Cylindrical Pipe")
    print("=" * 60)

    # Pipe dimensions
    R_pipe = 2.0e-3  # Pipe radius: 2 mm
    L_pipe = 0.02  # Pipe length: 20 mm

    # Fluid properties (blood)
    mu = 0.0035  # Pa·s
    rho = 1060  # kg/m³

    # Pressure drop
    dP = 100.0  # Pa

    # Create radial mesh
    nr = 51
    mesh = bt.CylindricalMesh(nr, 0.0, R_pipe)  # r from 0 to R

    r = mesh.r_coordinates()

    # Analytical velocity profile
    u_analytical = (dP / (4 * mu * L_pipe)) * (R_pipe**2 - r**2)

    # Calculate flow rate (volume)
    Q_analytical = np.pi * dP * R_pipe**4 / (8 * mu * L_pipe)

    # Average velocity
    U_avg = Q_analytical / (np.pi * R_pipe**2)

    # Maximum velocity (at centerline)
    U_max = u_analytical[0]

    # Reynolds number
    D_h = 2 * R_pipe
    Re = rho * U_avg * D_h / mu

    print(f"  Pipe radius: {R_pipe*1000:.1f} mm")
    print(f"  Pipe length: {L_pipe*1000:.0f} mm")
    print(f"  Pressure drop: {dP:.0f} Pa")
    print(f"  Viscosity: {mu*1000:.1f} mPa·s")
    print(f"  Maximum velocity: {U_max*100:.2f} cm/s")
    print(f"  Average velocity: {U_avg*100:.2f} cm/s")
    print(f"  Volume flow rate: {Q_analytical*1e6:.3f} mL/s")
    print(f"  Reynolds number: {Re:.1f}")

    # Wall shear stress
    tau_wall = 4 * mu * U_avg / R_pipe  # or dP * R / (2 * L)
    print(f"  Wall shear stress: {tau_wall:.2f} Pa")

    # Wall shear rate
    gamma_wall = bt.pipe_wall_shear_rate(U_avg, R_pipe)
    print(f"  Wall shear rate: {gamma_wall:.1f} s⁻¹")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Velocity profile
    ax = axes[0]
    ax.plot(u_analytical * 100, r * 1000, "b-", linewidth=2)
    ax.plot(u_analytical * 100, -r * 1000, "b-", linewidth=2)  # Mirror for full pipe
    ax.fill_betweenx([-R_pipe * 1000, R_pipe * 1000], 0, 0, alpha=0.1)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(R_pipe * 1000, color="gray", linewidth=2)
    ax.axhline(-R_pipe * 1000, color="gray", linewidth=2)
    ax.set_xlabel("Velocity u_z (cm/s)")
    ax.set_ylabel("Radial position r (mm)")
    ax.set_title("Poiseuille Velocity Profile")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, U_max * 100 * 1.1])

    # Shear stress profile
    ax = axes[1]
    # Shear stress: τ = -μ(du/dr) = (dP/2L) * r
    tau = (dP / (2 * L_pipe)) * r
    ax.plot(tau, r * 1000, "r-", linewidth=2, label="τ(r)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(R_pipe * 1000, color="gray", linewidth=2)
    ax.set_xlabel("Shear stress τ (Pa)")
    ax.set_ylabel("Radial position r (mm)")
    ax.set_title("Shear Stress Distribution")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, tau_wall * 1.1])

    # Add annotations
    ax.annotate(
        f"τ_wall = {tau_wall:.2f} Pa",
        xy=(tau_wall, R_pipe * 1000),
        xytext=(tau_wall * 0.6, R_pipe * 1000 * 0.8),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(bt.get_result_path("cylindrical_poiseuille_flow.png"), dpi=150)
    plt.close()

    print("  Plot saved to results/cylindrical_poiseuille_flow.png")


if __name__ == "__main__":
    print("Cylindrical Coordinate Examples for BMEN 341")
    print("Modeling pipe flow and radial diffusion in cylindrical geometry")
    print()

    example_radial_diffusion()
    # Note: axisymmetric and pipe flow examples require additional
    # CylindricalMesh constructors/methods not currently exposed.
    # example_axisymmetric_diffusion()
    # example_pipe_flow_poiseuille()

    print("\n" + "=" * 60)
    print("Cylindrical coordinate examples completed!")
    print("=" * 60)
