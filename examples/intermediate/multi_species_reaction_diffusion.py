#!/usr/bin/env python3
"""
Multi-Species Reaction-Diffusion Framework Example
===================================================

This example demonstrates the N-species reaction-diffusion framework with
several biologically and chemically relevant models:

1. Lotka-Volterra (Predator-Prey) System
2. SIR Epidemiological Model with Spatial Spread
3. Brusselator Chemical Oscillator

The general form solved is:
    ∂u_i/∂t = D_i ∇²u_i + R_i(u_1, ..., u_N, x, y, t)

where u_i are species concentrations, D_i are diffusion coefficients,
and R_i are reaction kinetics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure we can import biotransport
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import biotransport as bt


def example_lotka_volterra():
    """
    Lotka-Volterra Predator-Prey System
    =====================================

    Classic 2-species model with spatial diffusion:
        du/dt = D_u ∇²u + α·u - β·u·v      (prey)
        dv/dt = D_v ∇²v + δ·u·v - γ·v      (predator)

    Parameters:
        α (alpha) = prey growth rate
        β (beta)  = predation rate
        γ (gamma) = predator death rate
        δ (delta) = predator reproduction rate

    When prey diffuses faster than predators, we can see spatial patterns.
    """
    print("\n" + "=" * 60)
    print("Example 2: Lotka-Volterra Predator-Prey System")
    print("=" * 60)

    # Domain: 10x10 units, 30x30 grid (coarser for stability)
    Lx, Ly = 10.0, 10.0
    nx, ny = 30, 30
    mesh = bt.StructuredMesh(nx, ny, 0.0, Lx, 0.0, Ly)

    # Diffusion coefficients: minimal for stability with explicit solver
    D_prey = 0.005  # Very small prey diffusivity
    D_pred = 0.002  # Even smaller predator diffusivity

    # Create multi-species solver
    solver = bt.MultiSpeciesSolver(mesh, [D_prey, D_pred])

    # Set up Lotka-Volterra reaction model with carrying capacity
    # This prevents unbounded prey growth and stabilizes the explicit solver
    alpha = 1.0  # Prey growth rate
    beta = 0.1  # Predation rate
    gamma = 0.5  # Predator death rate
    delta = 0.05  # Predator reproduction rate
    K = 50.0  # Prey carrying capacity

    reaction_model = bt.LotkaVolterraReaction(alpha, beta, gamma, delta, K)
    solver.set_reaction_model(reaction_model)
    print(
        f"Lotka-Volterra parameters: α={alpha}, β={beta}, γ={gamma}, δ={delta}, K={K}"
    )

    # Initial conditions: prey everywhere, predator in center
    prey_ic = np.full(mesh.num_nodes(), 10.0)  # Uniform prey
    pred_ic = np.zeros(mesh.num_nodes())

    # Predator starts in center patch
    for j in range(ny + 1):
        for i in range(nx + 1):
            idx = mesh.index(i, j)
            x_coord = mesh.x(i)
            y_coord = mesh.y(i, j)
            r = np.sqrt((x_coord - Lx / 2) ** 2 + (y_coord - Ly / 2) ** 2)
            if r < 2.0:
                pred_ic[idx] = 5.0

    solver.set_initial_condition(0, prey_ic)
    solver.set_initial_condition(1, pred_ic)

    # Neumann (no-flux) boundaries for closed ecosystem
    solver.set_all_species_neumann(bt.Boundary.Left, 0.0)
    solver.set_all_species_neumann(bt.Boundary.Right, 0.0)
    solver.set_all_species_neumann(bt.Boundary.Bottom, 0.0)
    solver.set_all_species_neumann(bt.Boundary.Top, 0.0)

    # Time integration - use smaller timestep for reaction-diffusion stability
    dt = 0.01  # Fixed small timestep for coupled stability
    T_final = 30.0  # Shorter simulation for speed
    num_steps = int(T_final / dt)

    print(f"Time step: {dt:.4f}")
    print(f"Running for {num_steps} steps (T = {T_final})...")

    # Store snapshots for visualization
    times = [0]
    prey_history = [solver.solution(0).copy()]
    pred_history = [solver.solution(1).copy()]

    steps_per_snapshot = num_steps // 10
    for snapshot in range(10):
        solver.solve(dt, steps_per_snapshot)
        times.append(solver.time())
        prey_history.append(solver.solution(0).copy())
        pred_history.append(solver.solution(1).copy())
        print(
            f"  t = {solver.time():.1f}: prey_total = {solver.total_mass(0):.1f}, "
            f"pred_total = {solver.total_mass(1):.1f}"
        )

    # Plot results
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    # Show prey at different times
    for idx, t_idx in enumerate([0, 3, 6, 10]):
        prey_2d = np.array(prey_history[t_idx]).reshape(ny + 1, nx + 1)
        im = axes[0, idx].imshow(
            prey_2d,
            origin="lower",
            cmap="Greens",
            extent=[0, Lx, 0, Ly],
            vmin=0,
            vmax=15,
        )
        axes[0, idx].set_title(f"Prey (t = {times[t_idx]:.1f})")
        plt.colorbar(im, ax=axes[0, idx], shrink=0.8)

    # Show predator at different times
    for idx, t_idx in enumerate([0, 3, 6, 10]):
        pred_2d = np.array(pred_history[t_idx]).reshape(ny + 1, nx + 1)
        im = axes[1, idx].imshow(
            pred_2d, origin="lower", cmap="Reds", extent=[0, Lx, 0, Ly], vmin=0, vmax=10
        )
        axes[1, idx].set_title(f"Predator (t = {times[t_idx]:.1f})")
        plt.colorbar(im, ax=axes[1, idx], shrink=0.8)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent.parent / "results" / "multi_species"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "lotka_volterra.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_dir / 'lotka_volterra.png'}")
    plt.close()

    return True


def example_sir_epidemic():
    """
    SIR Epidemiological Model with Spatial Spread
    ===============================================

    Models disease spread with diffusion:
        dS/dt = D_S ∇²S - β·S·I/N
        dI/dt = D_I ∇²I + β·S·I/N - γ·I
        dR/dt = D_R ∇²R + γ·I

    The diffusion models population movement.
    Infected individuals may move less (D_I < D_S).
    """
    print("\n" + "=" * 60)
    print("Example 3: SIR Epidemic with Spatial Spread")
    print("=" * 60)

    # Domain: 20x20 units (e.g., 20km x 20km region)
    Lx, Ly = 20.0, 20.0
    nx, ny = 60, 60
    mesh = bt.StructuredMesh(nx, ny, 0.0, Lx, 0.0, Ly)

    # Diffusion coefficients (mobility)
    D_S = 0.5  # Susceptible mobility
    D_I = 0.1  # Infected mobility (reduced - stay home when sick)
    D_R = 0.5  # Recovered mobility

    # Create solver with 3 species
    solver = bt.MultiSpeciesSolver(mesh, [D_S, D_I, D_R])

    # SIR parameters
    # R₀ = β/γ determines epidemic behavior:
    #   R₀ > 1: epidemic spreads
    #   R₀ < 1: epidemic dies out
    beta = 0.4  # Transmission rate
    gamma = 0.1  # Recovery rate (1/gamma = 10 days average infection)
    N = 100.0  # Reference population density (matches S_ic)

    reaction_model = bt.SIRReaction(beta, gamma, N)
    solver.set_reaction_model(reaction_model)
    print(f"SIR parameters: β={beta}, γ={gamma}, N={N}")
    print(f"Basic reproduction number R₀ = {reaction_model.R0:.2f}")

    # Initial conditions: susceptible population everywhere
    # Small infected cluster in center (patient zero)
    S_ic = np.full(mesh.num_nodes(), 100.0)  # Susceptible density
    I_ic = np.zeros(mesh.num_nodes())  # Infected
    R_ic = np.zeros(mesh.num_nodes())  # Recovered

    # Initial infection at center
    for j in range(ny + 1):
        for i in range(nx + 1):
            idx = mesh.index(i, j)
            x_coord = mesh.x(i)
            y_coord = mesh.y(i, j)
            r = np.sqrt((x_coord - Lx / 2) ** 2 + (y_coord - Ly / 2) ** 2)
            if r < 1.0:
                I_ic[idx] = 10.0
                S_ic[idx] -= 10.0  # Convert some S to I

    solver.set_initial_condition(0, S_ic)
    solver.set_initial_condition(1, I_ic)
    solver.set_initial_condition(2, R_ic)

    # No-flux boundaries (closed region)
    for boundary in [
        bt.Boundary.Left,
        bt.Boundary.Right,
        bt.Boundary.Bottom,
        bt.Boundary.Top,
    ]:
        solver.set_all_species_neumann(boundary, 0.0)

    # Time integration
    dt = solver.max_stable_time_step()
    T_final = 100.0
    num_steps = int(T_final / dt)

    print(f"Time step: {dt:.4f}")
    print(f"Running for {num_steps} steps (T = {T_final})...")

    # Store time series of total populations
    t_history = [0]
    S_total = [np.sum(S_ic)]
    I_total = [np.sum(I_ic)]
    R_total = [np.sum(R_ic)]

    # Store spatial snapshots
    I_snapshots = [I_ic.copy()]
    snapshot_times = [0]

    steps_per_record = num_steps // 200
    for record in range(200):
        solver.solve(dt, steps_per_record)
        t_history.append(solver.time())
        S_total.append(solver.total_mass(0))
        I_total.append(solver.total_mass(1))
        R_total.append(solver.total_mass(2))

        # Save snapshots at key times
        if record in [25, 50, 100, 199]:
            I_snapshots.append(solver.solution(1).copy())
            snapshot_times.append(solver.time())
            print(
                f"  t = {solver.time():.1f}: S={S_total[-1]:.0f}, "
                f"I={I_total[-1]:.0f}, R={R_total[-1]:.0f}"
            )

    # Plot results
    fig = plt.figure(figsize=(14, 10))

    # Time series plot
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(t_history, S_total, "b-", label="Susceptible", linewidth=2)
    ax1.plot(t_history, I_total, "r-", label="Infected", linewidth=2)
    ax1.plot(t_history, R_total, "g-", label="Recovered", linewidth=2)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Total Population")
    ax1.set_title(f"SIR Dynamics (R₀ = {reaction_model.R0:.2f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Spatial snapshots of infected
    for idx, (t, I_data) in enumerate(zip(snapshot_times, I_snapshots)):
        ax = fig.add_subplot(2, 3, idx + 2)
        I_2d = np.array(I_data).reshape(ny + 1, nx + 1)
        im = ax.imshow(I_2d, origin="lower", cmap="Reds", extent=[0, Lx, 0, Ly], vmin=0)
        ax.set_title(f"Infected (t = {t:.0f})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent.parent / "results" / "multi_species"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "sir_epidemic.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_dir / 'sir_epidemic.png'}")
    plt.close()

    return True


def example_brusselator():
    """
    Brusselator Chemical Oscillator with Turing Patterns
    ======================================================

    Classic autocatalytic reaction system:
        dX/dt = D_X ∇²X + A - (B+1)·X + X²·Y
        dY/dt = D_Y ∇²Y + B·X - X²·Y

    For B > 1 + A², the system exhibits oscillations.
    With appropriate diffusion ratios, Turing patterns emerge.
    """
    print("\n" + "=" * 60)
    print("Example 1: Brusselator Turing Patterns")
    print("=" * 60)

    # Domain
    Lx, Ly = 100.0, 100.0
    nx, ny = 100, 100
    mesh = bt.StructuredMesh(nx, ny, 0.0, Lx, 0.0, Ly)

    # Diffusion coefficients (Turing instability requires D_Y >> D_X)
    D_X = 1.0
    D_Y = 8.0  # Inhibitor diffuses faster

    # Create solver
    solver = bt.MultiSpeciesSolver(mesh, [D_X, D_Y])

    # Brusselator parameters
    A = 4.5
    B = 9.0  # B > 1 + A² = 21.25 is needed for oscillation, but
    # lower B can give Turing patterns with diffusion

    reaction_model = bt.BrusselatorReaction(A, B)
    solver.set_reaction_model(reaction_model)
    print(f"Brusselator parameters: A={A}, B={B}")
    print(f"Oscillatory (homogeneous): {reaction_model.is_oscillatory}")

    # Steady state: X_ss = A, Y_ss = B/A
    X_ss = A
    Y_ss = B / A
    print(f"Steady state: X* = {X_ss:.2f}, Y* = {Y_ss:.2f}")

    # Initial conditions: small perturbations around steady state
    np.random.seed(42)
    X_ic = X_ss + 0.1 * np.random.randn(mesh.num_nodes())
    Y_ic = Y_ss + 0.1 * np.random.randn(mesh.num_nodes())

    # Ensure non-negative concentrations
    X_ic = np.maximum(X_ic, 0.01)
    Y_ic = np.maximum(Y_ic, 0.01)

    solver.set_initial_condition(0, X_ic)
    solver.set_initial_condition(1, Y_ic)

    # Neumann (no-flux) boundaries
    for boundary in [
        bt.Boundary.Left,
        bt.Boundary.Right,
        bt.Boundary.Bottom,
        bt.Boundary.Top,
    ]:
        solver.set_all_species_neumann(boundary, 0.0)

    # Time integration
    dt = 0.8 * solver.max_stable_time_step()  # Smaller for stability
    T_final = 50.0  # Reduced for faster demo
    num_steps = int(T_final / dt)

    print(f"Time step: {dt:.4f}")
    print(f"Running for {num_steps} steps (T = {T_final})...")

    # Store snapshots
    X_snapshots = [X_ic.copy()]
    snapshot_times = [0]

    steps_per_snapshot = num_steps // 5
    for snapshot in range(5):
        solver.solve(dt, steps_per_snapshot)
        X_snapshots.append(solver.solution(0).copy())
        snapshot_times.append(solver.time())
        print(
            f"  t = {solver.time():.1f}: X_mean = {np.mean(solver.solution(0)):.3f}, "
            f"Y_mean = {np.mean(solver.solution(1)):.3f}"
        )

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for idx, (t, X_data) in enumerate(zip(snapshot_times, X_snapshots)):
        ax = axes.flat[idx]
        X_2d = np.array(X_data).reshape(ny + 1, nx + 1)
        im = ax.imshow(X_2d, origin="lower", cmap="viridis", extent=[0, Lx, 0, Ly])
        ax.set_title(f"X (t = {t:.0f})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(f"Brusselator: A={A}, B={B}, D_X={D_X}, D_Y={D_Y}", fontsize=14)
    plt.tight_layout()

    output_dir = Path(__file__).parent.parent.parent / "results" / "multi_species"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "brusselator.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_dir / 'brusselator.png'}")
    plt.close()

    return True


def example_custom_reaction():
    """
    Custom Reaction Function Example
    =================================

    Demonstrates how to define a custom reaction function
    for systems not covered by the built-in models.

    Example: 3-species chain reaction
        A -> B -> C
    with Michaelis-Menten kinetics.
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Reaction Function")
    print("=" * 60)

    # 1D domain for simplicity
    Lx = 10.0
    nx = 100
    mesh = bt.StructuredMesh(nx, 0.0, Lx)

    # Different diffusivities for each species
    D_A = 0.1
    D_B = 0.05
    D_C = 0.01

    solver = bt.MultiSpeciesSolver(mesh, [D_A, D_B, D_C])

    # Use built-in EnzymeCascadeReaction for A -> B -> C
    # Parameters: Vmax list, Km list, and kdeg list for each species
    Vmax_list = [0.5, 0.3]  # A->B, B->C rates
    Km_list = [1.0, 1.0]  # Michaelis constants
    kdeg_list = [0.0, 0.0, 0.0]  # No degradation for A, B, C

    reaction_model = bt.EnzymeCascadeReaction(Vmax_list, Km_list, kdeg_list)
    solver.set_reaction_model(reaction_model)
    print("Chain reaction: A -> B -> C (EnzymeCascadeReaction)")
    print(f"Step 1: Vmax={Vmax_list[0]}, Km={Km_list[0]}")
    print(f"Step 2: Vmax={Vmax_list[1]}, Km={Km_list[1]}")

    # Initial condition: A at left boundary
    A_ic = np.zeros(mesh.num_nodes())
    for i in range(nx + 1):
        x = mesh.x(i)
        if x < 2.0:
            A_ic[i] = 5.0 * (1 - x / 2.0)

    solver.set_initial_condition(0, A_ic)
    solver.set_uniform_initial_condition(1, 0.0)
    solver.set_uniform_initial_condition(2, 0.0)

    # Dirichlet on left (continuous source of A), Neumann on right
    solver.set_dirichlet_boundary(0, bt.Boundary.Left, 5.0)  # A source
    solver.set_neumann_boundary(0, bt.Boundary.Right, 0.0)
    solver.set_neumann_boundary(1, bt.Boundary.Left, 0.0)
    solver.set_neumann_boundary(1, bt.Boundary.Right, 0.0)
    solver.set_neumann_boundary(2, bt.Boundary.Left, 0.0)
    solver.set_neumann_boundary(2, bt.Boundary.Right, 0.0)

    # Time integration
    dt = solver.max_stable_time_step()
    T_final = 100.0
    num_steps = int(T_final / dt)

    print(f"Time step: {dt:.4f}")
    print(f"Running for {num_steps} steps...")

    # Store profiles
    x_coords = [mesh.x(i) for i in range(nx + 1)]
    profiles = {"A": [], "B": [], "C": [], "t": []}

    snapshot_times = [0, 10, 25, 50, 100]
    for t_target in snapshot_times:
        if t_target > 0:
            steps_to_run = int((t_target - solver.time()) / dt)
            if steps_to_run > 0:
                solver.solve(dt, steps_to_run)

        profiles["t"].append(solver.time())
        profiles["A"].append(solver.solution(0).copy())
        profiles["B"].append(solver.solution(1).copy())
        profiles["C"].append(solver.solution(2).copy())
        print(
            f"  t = {solver.time():.1f}: A_total = {solver.total_mass(0):.2f}, "
            f"B_total = {solver.total_mass(1):.2f}, C_total = {solver.total_mass(2):.2f}"
        )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_times)))

    for idx, species in enumerate(["A", "B", "C"]):
        ax = axes[idx]
        for i, (t, profile) in enumerate(zip(profiles["t"], profiles[species])):
            ax.plot(x_coords, profile, color=colors[i], label=f"t={t:.0f}")
        ax.set_xlabel("x")
        ax.set_ylabel(f"[{species}]")
        ax.set_title(f"Species {species}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Chain Reaction: A -> B -> C", fontsize=14)
    plt.tight_layout()

    output_dir = Path(__file__).parent.parent.parent / "results" / "multi_species"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "custom_reaction.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_dir / 'custom_reaction.png'}")
    plt.close()

    return True


def main():
    """Run all multi-species examples."""
    print("=" * 60)
    print("Multi-Species Reaction-Diffusion Framework Examples")
    print("=" * 60)

    # Note: Order matters due to C++ solver state issues
    # Run Brusselator first (most sensitive to numerical precision)
    examples = [
        ("Brusselator", example_brusselator),
        ("Lotka-Volterra", example_lotka_volterra),
        ("SIR Epidemic", example_sir_epidemic),
        ("Custom Reaction", example_custom_reaction),
    ]

    results = []
    for name, func in examples:
        try:
            success = func()
            results.append((name, success))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {name}: {status}")

    return all(r[1] for r in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
