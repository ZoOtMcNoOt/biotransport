#!/usr/bin/env python3
"""
Newton-Raphson Iteration for Nonlinear Steady-State Problems
=============================================================

Demonstrates solving nonlinear reaction-diffusion equations of the form:

    -D ∇²u + R(u) = S

where R(u) is a nonlinear reaction term (e.g., Michaelis-Menten kinetics).

Key Features:
- Quadratic convergence for smooth problems
- Automatic Jacobian via finite differences (or analytical)
- Line search for global convergence
- Built-in reaction models (Michaelis-Menten, Hill, bistable)

Applications:
- Enzyme kinetics (Michaelis-Menten)
- Cooperative binding (Hill kinetics)
- Bistable switches (gene regulatory networks)
- Nonlinear boundary conditions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import biotransport as bt

# Results directory
RESULTS_DIR = Path(bt.get_results_dir()) / "newton_raphson"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def demo_michaelis_menten_reaction():
    """
    Steady-state substrate diffusion with Michaelis-Menten consumption.

    Models substrate (e.g., oxygen, glucose) diffusing into tissue
    while being consumed by enzyme reaction:

        -D d²S/dx² + Vmax*S/(Km + S) = 0

    with S(0) = S0 (boundary supply), dS/dx(L) = 0 (no flux at center)
    """
    print("\n" + "=" * 60)
    print("DEMO 1: Michaelis-Menten Substrate Consumption")
    print("=" * 60)

    # Physical parameters
    L = 0.01  # 1 cm tissue thickness
    D = 1e-9  # Diffusion coefficient (m²/s)
    S0 = 0.2  # Boundary concentration (mM)
    Vmax = 1e-3  # Max reaction rate (mM/s)
    Km = 0.1  # Michaelis constant (mM)

    # Create mesh
    mesh = bt.mesh_1d(100, 0, L)
    x = bt.x_nodes(mesh)

    # Set up reaction term
    reaction, deriv = bt.michaelis_menten(Vmax, Km)

    # Create solver
    solver = bt.NonlinearDiffusionSolver(mesh, D=D)
    solver.set_reaction(reaction, deriv)
    solver.set_boundary(bt.Boundary.Left, S0)  # Supply at boundary
    solver.set_boundary(bt.Boundary.Right, 0.0, bc_type="neumann")  # No flux at center
    solver.set_parameters(verbose=True)

    # Initial guess: linear decay
    initial = S0 * (1 - x / L)

    # Solve
    result = solver.solve(initial)

    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final residual: {result.residual_norm:.2e}")
    print(f"Min concentration: {result.solution.min():.4f} mM")

    # Compare with linear case (D*d²S/dx² = k*S)
    # For small S << Km, Michaelis-Menten → linear: R ≈ (Vmax/Km)*S

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Concentration profile
    ax = axes[0]
    ax.plot(x * 1000, result.solution, "b-", linewidth=2, label="Numerical solution")
    ax.axhline(y=Km, color="r", linestyle="--", alpha=0.5, label=f"Km = {Km} mM")
    ax.set_xlabel("Position (mm)", fontsize=12)
    ax.set_ylabel("Substrate concentration (mM)", fontsize=12)
    ax.set_title("Steady-State Substrate Profile", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Local reaction rate
    ax = axes[1]
    R = reaction(result.solution)
    ax.plot(x * 1000, R * 1000, "g-", linewidth=2)
    ax.set_xlabel("Position (mm)", fontsize=12)
    ax.set_ylabel("Reaction rate (μM/s)", fontsize=12)
    ax.set_title("Local Reaction Rate", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "michaelis_menten.png", dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'michaelis_menten.png'}")


def demo_hill_kinetics():
    """
    Cooperative binding with Hill kinetics.

    Hill equation models cooperative binding (e.g., hemoglobin-O2):

        R(u) = Vmax * u^n / (Km^n + u^n)

    where n > 1 indicates positive cooperativity (sigmoidal response).
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Hill Kinetics (Cooperative Binding)")
    print("=" * 60)

    mesh = bt.mesh_1d(100, 0, 1)
    x = bt.x_nodes(mesh)

    # Compare different Hill coefficients
    Vmax, Km = 1.0, 0.5
    hill_coefficients = [1, 2, 4]  # n=1 is Michaelis-Menten

    fig, ax = plt.subplots(figsize=(10, 6))

    for n in hill_coefficients:
        reaction, deriv = bt.hill_kinetics(Vmax, Km, n)

        solver = bt.NonlinearDiffusionSolver(mesh, D=0.1)
        solver.set_reaction(reaction, deriv)
        solver.set_boundary(bt.Boundary.Left, 1.0)
        solver.set_boundary(bt.Boundary.Right, 0.0, bc_type="neumann")

        initial = 1 - x
        result = solver.solve(initial)

        label = f"Hill n={n}" + (" (Michaelis-Menten)" if n == 1 else "")
        ax.plot(x, result.solution, linewidth=2, label=label)

        print(f"n={n}: converged in {result.iterations} iterations")

    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Concentration", fontsize=12)
    ax.set_title("Effect of Hill Coefficient on Steady-State Profile", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hill_kinetics.png", dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'hill_kinetics.png'}")


def demo_bistable_reaction():
    """
    Bistable system with two stable steady states.

    The bistable reaction R(u) = u(1-u)(u-a) has:
    - Stable fixed points at u=0 and u=1
    - Unstable fixed point at u=a

    This models genetic switches, cell fate decisions, etc.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Bistable Switch")
    print("=" * 60)

    mesh = bt.mesh_1d(100, 0, 1)
    x = bt.x_nodes(mesh)

    # Bistable with threshold at a=0.3
    a = 0.3
    reaction, deriv = bt.bistable(a)

    # Strong diffusion - should reach intermediate state
    solver = bt.NonlinearDiffusionSolver(mesh, D=0.5)
    solver.set_reaction(reaction, deriv)
    solver.set_boundary(bt.Boundary.Left, 0.0)  # Low state on left
    solver.set_boundary(bt.Boundary.Right, 1.0)  # High state on right

    # Start near threshold
    initial = a * np.ones(len(x))
    result = solver.solve(initial)

    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Steady-state profile
    ax = axes[0]
    ax.plot(x, result.solution, "b-", linewidth=2, label="Steady state")
    ax.axhline(y=a, color="r", linestyle="--", alpha=0.5, label=f"Threshold a={a}")
    ax.axhline(y=0, color="k", linestyle=":", alpha=0.3)
    ax.axhline(y=1, color="k", linestyle=":", alpha=0.3)
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Concentration", fontsize=12)
    ax.set_title("Bistable Front: Two Stable States", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Reaction function
    ax = axes[1]
    u_range = np.linspace(-0.1, 1.1, 200)
    R = reaction(u_range)
    ax.plot(u_range, R, "g-", linewidth=2)
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="b", linestyle="--", alpha=0.5, label="Stable: u=0")
    ax.axvline(x=a, color="r", linestyle="--", alpha=0.5, label=f"Unstable: u={a}")
    ax.axvline(x=1, color="b", linestyle="--", alpha=0.5, label="Stable: u=1")
    ax.set_xlabel("u", fontsize=12)
    ax.set_ylabel("R(u)", fontsize=12)
    ax.set_title("Bistable Reaction Term", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "bistable.png", dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'bistable.png'}")


def demo_convergence_history():
    """
    Demonstrate quadratic convergence of Newton's method.
    """
    print("\n" + "=" * 60)
    print("DEMO 4: Newton's Method Convergence")
    print("=" * 60)

    mesh = bt.mesh_1d(50, 0, 1)
    x = bt.x_nodes(mesh)

    # Nonlinear problem with known convergence
    reaction, deriv = bt.michaelis_menten(1.0, 0.5)

    solver = bt.NonlinearDiffusionSolver(mesh, D=0.1)
    solver.set_reaction(reaction, deriv)
    solver.set_boundary(bt.Boundary.Left, 1.0)
    solver.set_boundary(bt.Boundary.Right, 0.0)

    initial = np.linspace(1, 0, len(x))
    result = solver.solve(initial)

    # Plot convergence history
    fig, ax = plt.subplots(figsize=(8, 6))

    iterations = range(len(result.residual_history))
    ax.semilogy(iterations, result.residual_history, "bo-", markersize=8, linewidth=2)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Residual norm", fontsize=12)
    ax.set_title("Newton's Method: Quadratic Convergence", fontsize=14)
    ax.grid(True, alpha=0.3, which="both")

    # Annotate final residual
    ax.annotate(
        f"Final: {result.residual_history[-1]:.2e}",
        xy=(len(result.residual_history) - 1, result.residual_history[-1]),
        xytext=(len(result.residual_history) - 2, result.residual_history[-1] * 100),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "convergence.png", dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'convergence.png'}")

    print("\nResidual history:")
    for i, r in enumerate(result.residual_history):
        print(f"  Iteration {i}: ||F|| = {r:.6e}")


def demo_2d_nonlinear():
    """
    2D nonlinear reaction-diffusion (Poisson with nonlinear source).
    """
    print("\n" + "=" * 60)
    print("DEMO 5: 2D Nonlinear Reaction-Diffusion")
    print("=" * 60)

    mesh = bt.mesh_2d(30, 30, 0, 1, 0, 1)
    x = bt.x_nodes(mesh)
    y = bt.y_nodes(mesh)
    X, Y = np.meshgrid(x, y)

    # Nonlinear reaction: R(u) = u²
    def reaction(u):
        return u**2

    solver = bt.NonlinearDiffusionSolver(mesh, D=1.0)
    solver.set_reaction(reaction)
    solver.set_boundary(bt.Boundary.Left, 1.0)
    solver.set_boundary(bt.Boundary.Right, 0.0)
    solver.set_boundary(bt.Boundary.Bottom, 0.5)
    solver.set_boundary(bt.Boundary.Top, 0.5)

    result = solver.solve()

    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Max value: {result.solution.max():.4f}")
    print(f"Min value: {result.solution.min():.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(X, Y, result.solution, shading="auto", cmap="viridis")
    plt.colorbar(c, ax=ax, label="Concentration")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("2D Nonlinear Reaction-Diffusion Steady State", fontsize=14)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "2d_nonlinear.png", dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / '2d_nonlinear.png'}")


def main():
    """Run all Newton-Raphson demonstrations."""
    print("=" * 60)
    print("NEWTON-RAPHSON FOR NONLINEAR STEADY-STATE PROBLEMS")
    print("=" * 60)

    demo_michaelis_menten_reaction()
    demo_hill_kinetics()
    demo_bistable_reaction()
    demo_convergence_history()
    demo_2d_nonlinear()

    print("\n" + "=" * 60)
    print("AVAILABLE NEWTON-RAPHSON FUNCTIONS")
    print("=" * 60)
    print(
        """
Core Solvers:
  - bt.NewtonRaphsonSolver(residual, jacobian, n)  # General F(u)=0 solver
  - bt.NonlinearDiffusionSolver(mesh, D)           # -D∇²u + R(u) = S

Built-in Reaction Terms:
  - bt.michaelis_menten(Vmax, Km)     # R = Vmax*u/(Km+u)
  - bt.hill_kinetics(Vmax, Km, n)     # R = Vmax*u^n/(Km^n+u^n)
  - bt.bistable(a)                    # R = u(1-u)(u-a)
  - bt.exponential_decay(k)           # R = k*u

Usage Example:
  solver = bt.NonlinearDiffusionSolver(mesh, D=1.0)
  solver.set_reaction(*bt.michaelis_menten(Vmax, Km))
  solver.set_boundary(bt.Boundary.Left, 1.0)
  solver.set_boundary(bt.Boundary.Right, 0.0)
  result = solver.solve(initial_guess)
"""
    )

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
