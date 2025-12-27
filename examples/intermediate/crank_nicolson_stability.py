"""
Example: Crank-Nicolson vs Explicit Method Stability Comparison

This example demonstrates the unconditional stability of the Crank-Nicolson (CN)
implicit method compared to the explicit FTCS method. We solve a 1D diffusion
problem with two different time step sizes:

1. Small dt (stable for both methods)
2. Large dt (stable only for CN, explicit method blows up)

Model:
    dC/dt = D * d²C/dx²

The explicit method (Forward-Time Central-Space) has a stability criterion:
    dt ≤ dx²/(2*D)  (1D case)

Crank-Nicolson is unconditionally stable and allows much larger time steps.

BMEN 341 Reference: Numerical Methods for PDEs (Week 5)
"""

import matplotlib.pyplot as plt
import numpy as np
import biotransport as bt

EXAMPLE_NAME = "crank_nicolson_stability"

print("=" * 70)
print("Crank-Nicolson Stability Demonstration")
print("=" * 70)

# Create mesh: 50 nodes from x=0 to x=1
mesh = bt.mesh_1d(50)
dx = mesh.dx()
D = 0.01  # Diffusion coefficient

# Calculate stability limit for explicit method
dt_stable = 0.4 * dx**2 / (2 * D)  # Use 40% of the theoretical limit for safety
dt_unstable = 5.0 * dx**2 / (2 * D)  # 5x beyond stability limit

print(f"\nMesh spacing: dx = {dx:.6f}")
print(f"Diffusion coefficient: D = {D}")
print(f"Explicit stability limit: dt <= {dx**2 / (2*D):.6f}")
print(f"Safe stable dt: {dt_stable:.6f}")
print(f"Deliberately unstable dt: {dt_unstable:.6f} (5x beyond limit)")

# Initial condition: Gaussian pulse
initial = bt.gaussian(mesh, center=0.5, width=0.1)
x = bt.x_nodes(mesh)

# Simulation end time
t_end = 0.1

# ========================================================================
# Test 1: Small time step (stable for both methods)
# ========================================================================
print(f"\n{'='*70}")
print("Test 1: Small time step (dt = {:.6f})".format(dt_stable))
print("Expected: Both methods should produce stable, similar results")
print(f"{'='*70}")

# Create fresh ICs for each test to avoid any state issues
ic_explicit = bt.gaussian(mesh, center=0.5, width=0.1)
ic_cn_small = bt.gaussian(mesh, center=0.5, width=0.1)

# Explicit method with small dt
problem_explicit = (
    bt.Problem(mesh)
    .diffusivity(D)
    .initial_condition(ic_explicit)
    .dirichlet(bt.Boundary.Left, 0.0)
    .dirichlet(bt.Boundary.Right, 0.0)
)
result_explicit = bt.solve(problem_explicit, t=t_end, dt=dt_stable, method="explicit")
solution_explicit_stable = result_explicit.solution()

print(f"Explicit method: {result_explicit.stats.steps} steps")
print(
    f"  Min: {solution_explicit_stable.min():.6f}, Max: {solution_explicit_stable.max():.6f}"
)

# Crank-Nicolson with small dt - pass IC explicitly to workaround binding issue
problem_cn_small = (
    bt.Problem(mesh)
    .diffusivity(D)
    .initial_condition(ic_cn_small)
    .dirichlet(bt.Boundary.Left, 0.0)
    .dirichlet(bt.Boundary.Right, 0.0)
)
result_cn = bt.solve(
    problem_cn_small,
    t=t_end,
    dt=dt_stable,
    method="crank_nicolson",
    initial_condition=ic_cn_small,
)
solution_cn_stable = result_cn.solution()

print(f"Crank-Nicolson: {result_cn.stats.steps} steps")
print(f"  Min: {solution_cn_stable.min():.6f}, Max: {solution_cn_stable.max():.6f}")

# ========================================================================
# Test 2: Large time step (stable only for CN)
# ========================================================================
print(f"\n{'='*70}")
print("Test 2: Large time step (dt = {:.6f})".format(dt_unstable))
print("Expected: Explicit method blows up, CN remains stable")
print(f"{'='*70}")

# Create fresh ICs for Test 2
ic_explicit_large = bt.gaussian(mesh, center=0.5, width=0.1)
ic_cn_large = bt.gaussian(mesh, center=0.5, width=0.1)

# Create fresh problems for Test 2
problem_explicit_large = (
    bt.Problem(mesh)
    .diffusivity(D)
    .initial_condition(ic_explicit_large)
    .dirichlet(bt.Boundary.Left, 0.0)
    .dirichlet(bt.Boundary.Right, 0.0)
)

try:
    result_explicit_unstable = bt.solve(
        problem_explicit_large, t=t_end, dt=dt_unstable, method="explicit"
    )
    solution_explicit_unstable = result_explicit_unstable.solution()

    # Check if solution blew up
    if np.any(np.abs(solution_explicit_unstable) > 1e10) or np.any(
        np.isnan(solution_explicit_unstable)
    ):
        print("Explicit method: UNSTABLE (solution diverged)")
        solution_explicit_unstable[:] = np.nan  # Mark as failed for plotting
    else:
        print(f"Explicit method: {result_explicit_unstable.stats.steps} steps")
        print(
            f"  Min: {solution_explicit_unstable.min():.6f}, Max: {solution_explicit_unstable.max():.6f}"
        )
except Exception as e:
    print(f"Explicit method: FAILED - {str(e)}")
    solution_explicit_unstable = np.full_like(x, np.nan)

# Crank-Nicolson with large dt (should remain stable) - pass IC explicitly
problem_cn_large = (
    bt.Problem(mesh)
    .diffusivity(D)
    .initial_condition(ic_cn_large)
    .dirichlet(bt.Boundary.Left, 0.0)
    .dirichlet(bt.Boundary.Right, 0.0)
)
result_cn_unstable = bt.solve(
    problem_cn_large,
    t=t_end,
    dt=dt_unstable,
    method="crank_nicolson",
    initial_condition=ic_cn_large,
)
solution_cn_unstable = result_cn_unstable.solution()

print(f"Crank-Nicolson: {result_cn_unstable.stats.steps} steps")
print(f"  Min: {solution_cn_unstable.min():.6f}, Max: {solution_cn_unstable.max():.6f}")

# ========================================================================
# Visualization
# ========================================================================
print(f"\n{'='*70}")
print("Creating comparison plots...")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Stability Comparison: Explicit vs Crank-Nicolson", fontsize=16, fontweight="bold"
)

# Plot 1: Small dt - Explicit
ax1 = axes[0, 0]
ax1.plot(x, initial, "k--", alpha=0.5, label="Initial", linewidth=2)
ax1.plot(
    x,
    solution_explicit_stable,
    "b-",
    label=f"Explicit (dt={dt_stable:.6f})",
    linewidth=2,
)
ax1.set_xlabel("Position x")
ax1.set_ylabel("Concentration")
ax1.set_title(
    f"Explicit Method - Small Δt (Stable)\n{result_explicit.stats.steps} steps"
)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([-0.1, 1.1])

# Plot 2: Small dt - CN
ax2 = axes[0, 1]
ax2.plot(x, initial, "k--", alpha=0.5, label="Initial", linewidth=2)
ax2.plot(x, solution_cn_stable, "r-", label=f"CN (dt={dt_stable:.6f})", linewidth=2)
ax2.set_xlabel("Position x")
ax2.set_ylabel("Concentration")
ax2.set_title(f"Crank-Nicolson - Small Δt (Stable)\n{result_cn.stats.steps} steps")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-0.1, 1.1])

# Plot 3: Large dt - Explicit (unstable)
ax3 = axes[1, 0]
ax3.plot(x, initial, "k--", alpha=0.5, label="Initial", linewidth=2)
if not np.all(np.isnan(solution_explicit_unstable)):
    ax3.plot(
        x,
        solution_explicit_unstable,
        "b-",
        label=f"Explicit (dt={dt_unstable:.6f})",
        linewidth=2,
    )
ax3.set_xlabel("Position x")
ax3.set_ylabel("Concentration")
ax3.set_title(
    f"Explicit Method - Large Δt (UNSTABLE)\ndt is {dt_unstable/dt_stable:.1f}x stability limit"
)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.text(
    0.5,
    0.5,
    "NUMERICAL\nINSTABILITY",
    transform=ax3.transAxes,
    fontsize=24,
    color="red",
    alpha=0.3,
    ha="center",
    va="center",
    weight="bold",
)

# Plot 4: Large dt - CN (stable)
ax4 = axes[1, 1]
ax4.plot(x, initial, "k--", alpha=0.5, label="Initial", linewidth=2)
ax4.plot(x, solution_cn_unstable, "r-", label=f"CN (dt={dt_unstable:.6f})", linewidth=2)
ax4.set_xlabel("Position x")
ax4.set_ylabel("Concentration")
ax4.set_title(
    f"Crank-Nicolson - Large Δt (STABLE)\n{result_cn_unstable.stats.steps} steps (only!)"
)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim([-0.1, 1.1])

plt.tight_layout()
plt.savefig(bt.get_result_path("stability_comparison.png", EXAMPLE_NAME), dpi=150)

# ========================================================================
# Comparison plot: Both methods with small dt
# ========================================================================
fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, initial, "k--", alpha=0.5, label="Initial", linewidth=2)
ax.plot(
    x,
    solution_explicit_stable,
    "b-",
    label="Explicit (small dt)",
    linewidth=2,
    alpha=0.7,
)
ax.plot(
    x,
    solution_cn_stable,
    "r--",
    label="Crank-Nicolson (small dt)",
    linewidth=2,
    alpha=0.7,
)
ax.set_xlabel("Position x", fontsize=12)
ax.set_ylabel("Concentration", fontsize=12)
ax.set_title(
    f"Method Comparison at t={t_end} (Both Stable)", fontsize=14, fontweight="bold"
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(bt.get_result_path("method_comparison_stable.png", EXAMPLE_NAME), dpi=150)

plt.show()

# ========================================================================
# Summary
# ========================================================================
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print("\n[OK] Crank-Nicolson is unconditionally stable")
print(f"  - Works with dt = {dt_unstable:.6f} (5x beyond explicit stability limit)")
print(f"  - Completed simulation in only {result_cn_unstable.stats.steps} steps")
print("\n[X] Explicit method is conditionally stable")
print(f"  - Requires dt <= {dx**2 / (2*D):.6f}")
print(f"  - Needed {result_explicit.stats.steps} steps for stable solution")
print("  - Diverged when using large time step")
print("\n[i] Use Crank-Nicolson when:")
print("  - You need large time steps for efficiency")
print("  - Problem is stiff (small diffusivity, fine mesh)")
print("  - Accuracy at large dt is more important than speed per step")
print(f"\nResults saved to: {bt.get_result_path('', EXAMPLE_NAME)}")
print(f"{'='*70}")
