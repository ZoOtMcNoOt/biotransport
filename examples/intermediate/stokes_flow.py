#!/usr/bin/env python3
"""
Stokes Flow Examples - Creeping Flow in Microchannels

This example demonstrates the Stokes solver for low Reynolds number flows,
which are common in microfluidic devices and capillary blood flow.

BMEN 341 Relevance:
- Part II: Fluid Mechanics - Viscous flow at low Re
- Microfluidic drug delivery systems
- Blood flow in capillaries (Re << 1)

The Stokes equations (creeping flow):
    -∇p + μ∇²v = f
    ∇·v = 0  (incompressibility)
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt


def example_channel_flow():
    """
    Pressure-driven flow in a 2D channel (Poiseuille flow).

    Analytical solution for velocity profile:
        u(y) = (ΔP/2μL) * y * (H - y)

    where H is channel height, L is channel length.
    """
    print("=" * 60)
    print("Example 1: Pressure-Driven Channel Flow (Poiseuille)")
    print("=" * 60)

    # Channel dimensions (microfluidic scale)
    L = 1000e-6  # 1000 um length
    H = 100e-6  # 100 um height

    # Fluid properties (water at 37C)
    mu = 0.001  # Pa.s

    # Pressure gradient
    dP_dx = -1000.0  # Pa/m (negative = flow in +x direction)

    # Create mesh (nx, ny, xmin, xmax, ymin, ymax)
    nx, ny = 51, 26
    mesh = bt.StructuredMesh(nx, ny, 0.0, L, 0.0, H)

    # Create Stokes solver
    solver = bt.StokesSolver(mesh, mu)

    # Set solver parameters for better convergence
    solver.set_tolerance(1e-4)  # Relax tolerance slightly
    solver.set_max_iterations(5000)  # Limit iterations
    solver.set_pressure_relaxation(0.3)  # Increase from default 0.1
    solver.set_velocity_relaxation(0.7)  # Increase from default 0.5

    # Set boundary conditions using factory methods
    # No-slip on top and bottom walls
    solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())

    # Inflow on left (uniform profile - parabolic develops naturally)
    u_max = -dP_dx * H**2 / (8 * mu)  # Maximum velocity for Poiseuille
    solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(u_max, 0.0))

    # Outflow on right
    solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())

    # Set body force (pressure gradient acts like a constant body force in x)
    solver.set_body_force(-dP_dx, 0.0)  # f_x = -dP/dx

    # Solve
    result = solver.solve()

    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Residual: {result.residual:.2e}")

    # Extract centerline velocity profile
    # Stokes solver uses (nx+1, ny+1) nodes for velocity and pressure
    u = result.u().reshape((ny + 1, nx + 1))
    v = result.v().reshape((ny + 1, nx + 1))
    p = result.pressure().reshape((ny + 1, nx + 1))

    # Analytical solution for comparison
    y = np.linspace(0, H, ny + 1)
    u_analytical = -dP_dx / (2 * mu) * y * (H - y)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Velocity magnitude
    ax = axes[0, 0]
    speed = np.sqrt(u**2 + v**2)
    im = ax.imshow(
        speed,
        extent=[0, L * 1e6, 0, H * 1e6],
        origin="lower",
        aspect="auto",
        cmap="jet",
    )
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Velocity Magnitude")
    plt.colorbar(im, ax=ax, label="m/s")

    # Velocity profile comparison
    ax = axes[0, 1]
    mid_x = (nx + 1) // 2
    ax.plot(u[:, mid_x] * 1000, y * 1e6, "b-", linewidth=2, label="Numerical")
    ax.plot(u_analytical * 1000, y * 1e6, "r--", linewidth=2, label="Analytical")
    ax.set_xlabel("u velocity (mm/s)")
    ax.set_ylabel("y (um)")
    ax.set_title("Velocity Profile at Channel Center")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pressure field
    ax = axes[1, 0]
    im = ax.imshow(
        p, extent=[0, L * 1e6, 0, H * 1e6], origin="lower", aspect="auto", cmap="RdBu_r"
    )
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Pressure Field")
    plt.colorbar(im, ax=ax, label="Pa")

    # Streamlines
    ax = axes[1, 1]
    x_grid = np.linspace(0, L, nx + 1)
    y_grid = np.linspace(0, H, ny + 1)
    X, Y = np.meshgrid(x_grid, y_grid)
    ax.streamplot(X * 1e6, Y * 1e6, u, v, density=1.5, color="blue", linewidth=0.8)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Streamlines")
    ax.set_xlim([0, L * 1e6])
    ax.set_ylim([0, H * 1e6])

    plt.tight_layout()
    plt.savefig(bt.get_result_path("stokes_channel_flow.png"), dpi=150)
    plt.close()

    print(f"  Maximum velocity: {np.max(u)*1000:.3f} mm/s")
    print(f"  Expected max velocity: {u_max*1000:.3f} mm/s")
    print("  Plot saved to results/stokes_channel_flow.png")

    return result


def example_blood_capillary():
    """
    Blood flow in a capillary using non-Newtonian Casson model.

    Capillary blood flow is quintessential creeping flow (Re ~ 0.001-0.01).
    Blood exhibits non-Newtonian behavior due to red blood cells.
    """
    print("\n" + "=" * 60)
    print("Example 2: Blood Flow in Capillary (Casson Model)")
    print("=" * 60)

    # Capillary dimensions
    L = 500e-6  # 500 um length
    D = 10e-6  # 10 um diameter (about red blood cell size)
    H = D  # Using 2D approximation

    # Hematocrit (volume fraction of RBCs)
    hematocrit = 0.45

    # Create blood model (Casson equation for blood)
    blood = bt.blood_casson_model(hematocrit)

    print(f"  Blood properties (hematocrit = {hematocrit*100:.0f}%):")
    print(f"    Yield stress: {blood.yield_stress()*1000:.3f} mPa")
    print(f"    Plastic viscosity: {blood.plastic_viscosity()*1000:.3f} mPa.s")

    # Estimate effective viscosity at typical capillary shear rate
    typical_shear_rate = 100.0  # 1/s
    mu_eff = blood.viscosity(typical_shear_rate)
    print(
        f"    Effective viscosity at {typical_shear_rate:.0f}/s: {mu_eff*1000:.3f} mPa.s"
    )

    # Reynolds number
    U_typical = 1e-3  # 1 mm/s typical capillary velocity
    rho = 1060  # kg/m^3 (blood density)
    Re = rho * U_typical * D / mu_eff
    print(f"    Reynolds number: {Re:.4f}")

    # Create mesh (nx, ny, xmin, xmax, ymin, ymax)
    nx, ny = 101, 21
    mesh = bt.StructuredMesh(nx, ny, 0.0, L, 0.0, H)

    # Use effective viscosity for Stokes solver
    solver = bt.StokesSolver(mesh, mu_eff)

    # Set solver parameters for better convergence
    solver.set_tolerance(1e-4)
    solver.set_max_iterations(5000)
    solver.set_pressure_relaxation(0.3)
    solver.set_velocity_relaxation(0.7)

    # Boundary conditions using factory methods
    solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(U_typical, 0.0))
    solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())

    # Solve
    result = solver.solve()

    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")

    # Calculate flow rate (Stokes uses (nx+1, ny+1) nodes)
    u = result.u().reshape((ny + 1, nx + 1))
    y = np.linspace(0, H, ny + 1)
    dy = y[1] - y[0]
    Q = np.sum(u[:, (nx + 1) // 2]) * dy  # Flow rate per unit depth

    print(f"  Flow rate: {Q*1e12:.3f} pL/s per um depth")

    return result


def example_non_newtonian_comparison():
    """
    Compare flow profiles for different non-Newtonian fluid models.

    This demonstrates how rheology affects velocity profiles.
    """
    print("\n" + "=" * 60)
    print("Example 3: Non-Newtonian Model Comparison")
    print("=" * 60)

    # Channel geometry
    L = 100e-6  # 100 um
    H = 20e-6  # 20 um

    # Mesh (nx, ny, xmin, xmax, ymin, ymax)
    nx, ny = 51, 21
    mesh = bt.StructuredMesh(nx, ny, 0.0, L, 0.0, H)

    # Reference shear rate
    gamma_ref = 100.0  # 1/s

    # Different fluid models
    models = {
        "Newtonian": bt.NewtonianModel(0.001),
        "Power-law (n=0.7)": bt.PowerLawModel(0.001, 0.7),  # Shear-thinning
        "Power-law (n=1.3)": bt.PowerLawModel(0.001, 1.3),  # Shear-thickening
        "Carreau (blood)": bt.blood_carreau_model(0.45),
    }

    results = {}

    for name, model in models.items():
        mu_eff = model.viscosity(gamma_ref)
        print(f"  {name}: mu({gamma_ref:.0f}/s) = {mu_eff*1000:.3f} mPa.s")

        solver = bt.StokesSolver(mesh, mu_eff)

        # Set solver parameters for better convergence
        solver.set_tolerance(1e-4)
        solver.set_max_iterations(5000)
        solver.set_pressure_relaxation(0.3)
        solver.set_velocity_relaxation(0.7)

        # Same BCs for all using factory methods
        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(0.001, 0.0))
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())

        result = solver.solve()
        results[name] = result.u().reshape((ny + 1, nx + 1))

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.linspace(0, H * 1e6, ny + 1)

    colors = ["blue", "red", "green", "purple"]
    for (name, u), color in zip(results.items(), colors):
        ax.plot(
            u[:, (nx + 1) // 2] * 1000, y, "-", linewidth=2, label=name, color=color
        )

    ax.set_xlabel("u velocity (mm/s)")
    ax.set_ylabel("y (um)")
    ax.set_title("Velocity Profiles for Different Rheological Models")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(bt.get_result_path("stokes_rheology_comparison.png"), dpi=150)
    plt.close()

    print("  Plot saved to results/stokes_rheology_comparison.png")


if __name__ == "__main__":
    print("Stokes Flow Examples for BMEN 341")
    print("Modeling low Reynolds number flows in biological systems")
    print()

    example_channel_flow()
    example_blood_capillary()
    example_non_newtonian_comparison()

    print("\n" + "=" * 60)
    print("All Stokes flow examples completed!")
    print("=" * 60)
