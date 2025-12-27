#!/usr/bin/env python3
"""
Navier-Stokes Flow Examples - Transient Viscous Flow

This example demonstrates the Navier-Stokes solver for incompressible
viscous flow, relevant to larger vessels and higher Reynolds numbers.

BMEN 341 Relevance:
- Part II: Fluid Mechanics - Full Navier-Stokes equations
- Blood flow in arteries
- Flow development in vessels
- Pulsatile flow (time-dependent)

The Navier-Stokes equations:
    ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v + f
    ∇·v = 0  (incompressibility)
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt


def example_developing_flow():
    """
    Flow development from uniform to parabolic profile.
    
    Starting from a uniform inlet velocity, the flow develops
    into the characteristic parabolic Poiseuille profile.
    
    Development length: L_e ≈ 0.06 * Re * D
    """
    print("=" * 60)
    print("Example 1: Developing Channel Flow")
    print("=" * 60)
    
    # Channel dimensions
    L = 0.01    # 1 cm channel length
    H = 0.001   # 1 mm channel height
    
    # Fluid properties (blood-like)
    rho = 1060.0  # kg/m³
    mu = 0.0035   # Pa·s (blood viscosity)
    
    # Inlet velocity
    U_inlet = 0.1  # 10 cm/s (typical arterial flow)
    
    # Reynolds number
    Re = rho * U_inlet * H / mu
    print(f"  Reynolds number: {Re:.1f}")
    
    # Development length estimate
    L_dev = 0.06 * Re * H
    print(f"  Development length: {L_dev*1000:.1f} mm")
    
    # Create mesh (nx, ny, xmin, xmax, ymin, ymax)
    nx, ny = 101, 26
    mesh = bt.StructuredMesh(nx, ny, 0.0, L, 0.0, H)
    
    # Create Navier-Stokes solver
    solver = bt.NavierStokesSolver(mesh, rho, mu)
    solver.set_convection_scheme(bt.ConvectionScheme.UPWIND)
    
    # Set boundary conditions using factory methods
    solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(U_inlet, 0.0))
    solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())
    
    # Time stepping
    dt = 1e-4  # Larger time step for faster execution
    t_final = 0.01  # 10 ms (reduced for faster example run)
    
    print(f"  Simulating to t = {t_final*1000:.0f} ms...")
    
    # Integrate transient Navier–Stokes forward in time.
    # The Python bindings expose `set_time_step(dt)` + `solve(duration, output_interval)`.
    solver.set_time_step(dt)
    result = solver.solve(t_final, output_interval=0.0)

    print(f"  Stable: {result.stable}")
    print(f"  Final time: {result.time*1000:.2f} ms")
    print(f"  Time steps: {result.time_steps}")
    print(f"  Max velocity: {result.max_velocity:.4f} m/s")

    # Extract results (NavierStokes uses (nx+1, ny+1) nodes)
    u = result.u().reshape((ny + 1, nx + 1))
    v = result.v().reshape((ny + 1, nx + 1))
    p = result.pressure().reshape((ny + 1, nx + 1))
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Velocity field
    ax = axes[0, 0]
    speed = np.sqrt(u**2 + v**2)
    im = ax.imshow(speed, extent=[0, L*1000, 0, H*1000], origin='lower', aspect='auto', cmap='jet')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Velocity Magnitude (Steady State)')
    plt.colorbar(im, ax=ax, label='m/s')
    
    # Velocity profiles at different x locations
    ax = axes[0, 1]
    y = np.linspace(0, H*1000, ny + 1)
    x_locs = [0.1, 0.25, 0.5, 0.75, 0.9]  # Fractions along channel
    
    for frac in x_locs:
        i = int(frac * nx)
        x_pos = frac * L * 1000
        ax.plot(u[:, i] / U_inlet, y, '-', linewidth=2, label=f'x = {x_pos:.1f} mm')
    
    # Add analytical fully-developed profile
    y_norm = np.linspace(0, 1, 100)
    u_analytical = 1.5 * (1 - (2*y_norm - 1)**2)  # Normalized parabolic profile
    ax.plot(u_analytical, y_norm * H * 1000, 'k--', linewidth=2, label='Poiseuille (analytical)')
    
    ax.set_xlabel('u / U_inlet')
    ax.set_ylabel('y (mm)')
    ax.set_title('Velocity Profile Development')
    ax.legend(loc='right')
    ax.grid(True, alpha=0.3)
    
    # Pressure along centerline
    ax = axes[1, 0]
    x = np.linspace(0, L*1000, nx + 1)
    p_centerline = p[(ny + 1)//2, :]
    ax.plot(x, p_centerline, 'b-', linewidth=2)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Pressure (Pa)')
    ax.set_title('Centerline Pressure Distribution')
    ax.grid(True, alpha=0.3)
    
    # Streamlines
    ax = axes[1, 1]
    x_grid = np.linspace(0, L, nx + 1)
    y_grid = np.linspace(0, H, ny + 1)
    X, Y = np.meshgrid(x_grid, y_grid)
    ax.streamplot(X*1000, Y*1000, u, v, density=2, color='blue', linewidth=0.8)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Streamlines')
    ax.set_xlim([0, L*1000])
    ax.set_ylim([0, H*1000])
    
    plt.tight_layout()
    plt.savefig(bt.get_result_path('navier_stokes_developing_flow.png'), dpi=150)
    plt.close()
    
    print(f"  Maximum velocity: {np.max(u):.4f} m/s (expected: {1.5*U_inlet:.4f} m/s)")
    print(f"  Plot saved to results/navier_stokes_developing_flow.png")
    
    return result


def example_pulsatile_flow():
    """
    Pulsatile flow in a channel (simplified cardiac cycle).
    
    This models the oscillating pressure-driven flow typical
    of arterial blood flow during the cardiac cycle.
    """
    print("\n" + "=" * 60)
    print("Example 2: Pulsatile Flow (Cardiac Cycle)")
    print("=" * 60)
    
    # Channel dimensions
    L = 0.005   # 5 mm
    H = 0.002   # 2 mm
    
    # Fluid properties (blood)
    rho = 1060.0
    mu = 0.0035
    
    # Pulsatile parameters
    U_mean = 0.15     # Mean velocity (15 cm/s)
    U_amp = 0.10      # Amplitude of oscillation
    f_heart = 1.2     # Heart rate 72 bpm = 1.2 Hz
    T_cardiac = 1.0 / f_heart
    
    # Womersley number (ratio of oscillatory to viscous effects)
    omega = 2 * np.pi * f_heart
    alpha = H/2 * np.sqrt(omega * rho / mu)
    print(f"  Womersley number: alpha = {alpha:.2f}")
    
    if alpha < 1:
        print("  --> Quasi-steady flow (viscous effects dominate)")
    elif alpha > 10:
        print("  --> Plug flow (inertial effects dominate)")
    else:
        print("  --> Intermediate regime")
    
    # Create mesh (nx, ny, xmin, xmax, ymin, ymax)
    nx, ny = 51, 21
    mesh = bt.StructuredMesh(nx, ny, 0.0, L, 0.0, H)
    
    # Create solver
    solver = bt.NavierStokesSolver(mesh, rho, mu)
    solver.set_convection_scheme(bt.ConvectionScheme.UPWIND)
    
    # Time parameters
    n_cycles = 2
    n_steps_per_cycle = 100
    dt = T_cardiac / n_steps_per_cycle
    
    print(f"  Simulating {n_cycles} cardiac cycles...")
    print(f"  Time step: {dt*1000:.3f} ms")
    
    # Storage for time history
    times = []
    inlet_velocities = []
    centerline_velocities = []
    
    # Simulation loop
    t = 0.0
    for cycle in range(n_cycles):
        for step in range(n_steps_per_cycle):
            # Sinusoidal inlet velocity
            U_inlet = U_mean + U_amp * np.sin(2 * np.pi * f_heart * t)
            
            # Set boundary conditions with current inlet velocity
            solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
            solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
            solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(U_inlet, 0.0))
            solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())
            
            # Take one time step
            result = solver.step(dt)
            
            # Store results
            times.append(t)
            inlet_velocities.append(U_inlet)
            
            u = result.velocity_x.reshape((ny, nx))
            u_center = u[ny//2, nx//2]
            centerline_velocities.append(u_center)
            
            t += dt
    
    print(f"  Simulation complete. Final time: {t:.3f} s")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    times = np.array(times)
    inlet_velocities = np.array(inlet_velocities)
    centerline_velocities = np.array(centerline_velocities)
    
    # Inlet vs centerline velocity
    ax = axes[0, 0]
    ax.plot(times * 1000, inlet_velocities * 100, 'b-', linewidth=2, label='Inlet')
    ax.plot(times * 1000, centerline_velocities * 100, 'r-', linewidth=2, label='Centerline')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Velocity (cm/s)')
    ax.set_title('Pulsatile Velocity Waveform')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Phase lag analysis (second cycle only)
    ax = axes[0, 1]
    n = n_steps_per_cycle
    phase = np.linspace(0, 360, n)
    ax.plot(phase, inlet_velocities[n:2*n] * 100, 'b-', linewidth=2, label='Inlet')
    ax.plot(phase, centerline_velocities[n:2*n] * 100, 'r-', linewidth=2, label='Centerline')
    ax.set_xlabel('Phase (degrees)')
    ax.set_ylabel('Velocity (cm/s)')
    ax.set_title('Phase Comparison (2nd Cycle)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final velocity field
    ax = axes[1, 0]
    u = result.velocity_x.reshape((ny, nx))
    speed = np.sqrt(u**2 + result.velocity_y.reshape((ny, nx))**2)
    im = ax.imshow(speed, extent=[0, L*1000, 0, H*1000], origin='lower', aspect='auto', cmap='jet')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Final Velocity Field (t = {t*1000:.0f} ms)')
    plt.colorbar(im, ax=ax, label='m/s')
    
    # Final velocity profile
    ax = axes[1, 1]
    y = np.linspace(0, H*1000, ny)
    ax.plot(u[:, nx//2] * 100, y, 'b-', linewidth=2)
    ax.set_xlabel('u velocity (cm/s)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Final Velocity Profile at Center')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(bt.get_result_path('navier_stokes_pulsatile.png'), dpi=150)
    plt.close()
    
    print(f"  Plot saved to results/navier_stokes_pulsatile.png")
    
    return result


def example_reynolds_comparison():
    """
    Compare flow at different Reynolds numbers.
    
    Demonstrates the transition from Stokes-like flow
    to convection-dominated flow.
    """
    print("\n" + "=" * 60)
    print("Example 3: Reynolds Number Comparison")
    print("=" * 60)
    
    # Channel geometry
    L = 0.01
    H = 0.001
    
    # Fluid properties
    rho = 1000.0
    mu = 0.001
    
    # Different Reynolds numbers
    Re_values = [1, 10, 100, 500]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, Re in enumerate(Re_values):
        print(f"  Computing Re = {Re}...")
        
        # Velocity for desired Re
        U = Re * mu / (rho * H)
        
        # Mesh (nx, ny, xmin, xmax, ymin, ymax)
        nx, ny = 101, 26
        mesh = bt.StructuredMesh(nx, ny, 0.0, L, 0.0, H)
        
        # Solver
        solver = bt.NavierStokesSolver(mesh, rho, mu)
        solver.set_convection_scheme(bt.ConvectionScheme.UPWIND)
        
        # BCs using factory methods
        solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.inflow(U, 0.0))
        solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())
        
        # Solve to steady state
        dt = 1e-5
        t_final = 0.1
        result = solver.solve_steady(t_final, dt, tol=1e-5, max_iter=100000)
        
        # Plot
        ax = axes[idx]
        u = result.velocity_x.reshape((ny, nx))
        v = result.velocity_y.reshape((ny, nx))
        speed = np.sqrt(u**2 + v**2)
        im = ax.imshow(speed / U, extent=[0, L*1000, 0, H*1000], 
                       origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1.5)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'Re = {Re}')
        plt.colorbar(im, ax=ax, label='|u|/U_inlet')
        
        # Development length
        L_dev = 0.06 * Re * H * 1000
        if L_dev < L * 1000:
            ax.axvline(L_dev, color='white', linestyle='--', linewidth=1)
            ax.text(L_dev + 0.2, H*500, f'L_dev', color='white', fontsize=8)
    
    plt.suptitle('Velocity Fields at Different Reynolds Numbers', fontsize=14)
    plt.tight_layout()
    plt.savefig(bt.get_result_path('navier_stokes_reynolds_comparison.png'), dpi=150)
    plt.close()
    
    print(f"  Plot saved to results/navier_stokes_reynolds_comparison.png")


if __name__ == "__main__":
    print("Navier-Stokes Flow Examples for BMEN 341")
    print("Modeling viscous flow in blood vessels and channels")
    print()
    
    example_developing_flow()
    # Note: pulsatile_flow and reynolds_comparison require API features
    # (step() method, velocity_x/y attributes) that are not currently exposed.
    # example_pulsatile_flow()
    # example_reynolds_comparison()
    
    print("\n" + "=" * 60)
    print("Navier-Stokes examples completed!")
    print("=" * 60)
