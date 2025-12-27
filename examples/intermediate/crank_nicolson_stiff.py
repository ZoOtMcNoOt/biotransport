"""
Example: Crank-Nicolson for Stiff Problems

This example demonstrates the power of Crank-Nicolson for "stiff" problems
where explicit methods would require prohibitively small time steps.

A problem is "stiff" when:
1. It has very different time scales (fast and slow processes)
2. Small diffusivity requires fine spatial resolution
3. Explicit stability criterion Δt ≤ Δx²/(2D) becomes very restrictive

We solve three scenarios:
1. Small diffusivity (D = 1e-6) - Very stiff
2. Fine mesh (nx = 500) - Stiff due to small Δx
3. 2D problem - Stiffness compounds in multiple dimensions

BMEN 341 Reference: Stiff Differential Equations (Week 6)
"""

import matplotlib.pyplot as plt
import numpy as np
import biotransport as bt
import time as time_module

EXAMPLE_NAME = "crank_nicolson_stiff"

print("=" * 80)
print("Crank-Nicolson for Stiff Problems")
print("=" * 80)

# ========================================================================
# Scenario 1: Very Small Diffusivity (Stiff in Time)
# ========================================================================
print(f"\n{'='*80}")
print("Scenario 1: Small Diffusivity Problem")
print(f"{'='*80}")

# Create mesh
mesh_1 = bt.mesh_1d(100, x_min=0.0, x_max=1.0)
dx_1 = mesh_1.dx()

# Very small diffusivity - typical of drug diffusion in dense tissue
D_small = 1e-6  # m²/s (very slow diffusion)

# Calculate stability limits
dt_explicit_limit = 0.5 * dx_1**2 / (2 * D_small)
dt_cn_large = 100 * dt_explicit_limit  # Use 100x the explicit limit!

print("\nProblem setup:")
print(f"  Domain: 1D, [0, 1] with {mesh_1.num_nodes()} nodes")
print(f"  Diffusivity: D = {D_small:.2e} m²/s (very small)")
print(f"  Mesh spacing: Δx = {dx_1:.6f} m")
print(f"  Explicit stability limit: Δt ≤ {dt_explicit_limit:.6e} s")
print(f"  CN time step (100x limit): Δt = {dt_cn_large:.6e} s")

# Initial condition: Localized source
initial_1 = bt.gaussian(mesh_1, center=0.5, width=0.05)

problem_1 = (
    bt.Problem(mesh_1)
    .diffusivity(D_small)
    .initial_condition(initial_1)
    .dirichlet(bt.Boundary.Left, 0.0)
    .dirichlet(bt.Boundary.Right, 0.0)
)

t_end_1 = 1.0  # seconds

print(f"\nSimulation time: {t_end_1} s")
print(f"Steps with explicit (safe dt): {int(t_end_1 / dt_explicit_limit):,}")
print(f"Steps with CN (large dt): {int(np.ceil(t_end_1 / dt_cn_large))}")

# Time the simulations
print("\nTiming comparison:")

# Explicit with small dt (safe)
start = time_module.time()
result_explicit_1 = bt.solve(
    problem_1, t=t_end_1, dt=dt_explicit_limit, method="explicit"
)
time_explicit_1 = time_module.time() - start
print(f"  Explicit: {time_explicit_1:.3f} s ({result_explicit_1.stats.steps} steps)")

# Crank-Nicolson with large dt
start = time_module.time()
result_cn_1 = bt.solve(problem_1, t=t_end_1, dt=dt_cn_large, method="crank_nicolson")
time_cn_1 = time_module.time() - start
print(f"  Crank-Nicolson: {time_cn_1:.3f} s ({result_cn_1.stats.steps} steps)")
print(f"  ⚡ Speedup: {time_explicit_1/time_cn_1:.1f}x faster with CN")

# ========================================================================
# Scenario 2: Fine Mesh (Stiff in Space)
# ========================================================================
print(f"\n{'='*80}")
print("Scenario 2: Fine Mesh Problem")
print(f"{'='*80}")

# Very fine mesh - typical for resolving sharp gradients
mesh_2 = bt.mesh_1d(500, x_min=0.0, x_max=1.0)
dx_2 = mesh_2.dx()
D_2 = 0.01  # Moderate diffusivity

dt_explicit_limit_2 = 0.5 * dx_2**2 / (2 * D_2)
dt_cn_large_2 = 50 * dt_explicit_limit_2

print("\nProblem setup:")
print(f"  Domain: 1D, [0, 1] with {mesh_2.num_nodes()} nodes (fine resolution)")
print(f"  Diffusivity: D = {D_2} m²/s")
print(f"  Mesh spacing: Δx = {dx_2:.6f} m (very small)")
print(f"  Explicit stability limit: Δt ≤ {dt_explicit_limit_2:.6e} s")
print(f"  CN time step (50x limit): Δt = {dt_cn_large_2:.6e} s")

initial_2 = bt.gaussian(mesh_2, center=0.5, width=0.05)

problem_2 = (
    bt.Problem(mesh_2)
    .diffusivity(D_2)
    .initial_condition(initial_2)
    .dirichlet(bt.Boundary.Left, 0.0)
    .dirichlet(bt.Boundary.Right, 0.0)
)

t_end_2 = 0.1

print(f"\nSimulation time: {t_end_2} s")
print(f"Steps with explicit: {int(t_end_2 / dt_explicit_limit_2):,}")
print(f"Steps with CN: {int(np.ceil(t_end_2 / dt_cn_large_2))}")

print("\nTiming comparison:")

# Explicit
start = time_module.time()
result_explicit_2 = bt.solve(
    problem_2, t=t_end_2, dt=dt_explicit_limit_2, method="explicit"
)
time_explicit_2 = time_module.time() - start
print(f"  Explicit: {time_explicit_2:.3f} s ({result_explicit_2.stats.steps} steps)")

# Crank-Nicolson
start = time_module.time()
result_cn_2 = bt.solve(problem_2, t=t_end_2, dt=dt_cn_large_2, method="crank_nicolson")
time_cn_2 = time_module.time() - start
print(f"  Crank-Nicolson: {time_cn_2:.3f} s ({result_cn_2.stats.steps} steps)")
print(f"  ⚡ Speedup: {time_explicit_2/time_cn_2:.1f}x faster with CN")

# ========================================================================
# Scenario 3: 2D Problem (Compound Stiffness)
# ========================================================================
print(f"\n{'='*80}")
print("Scenario 3: 2D Problem (Compound Stiffness)")
print(f"{'='*80}")

# 2D mesh with moderate resolution
mesh_3 = bt.mesh_2d(100, 100, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
dx_3 = mesh_3.dx()
dy_3 = mesh_3.dy()
D_3 = 1e-5  # Small diffusivity

# 2D stability limit is more restrictive: dt ≤ dx²*dy²/(2*D*(dx²+dy²))
# For square mesh: dt ≤ dx²/(4*D)
dt_explicit_limit_3 = 0.5 * dx_3**2 / (4 * D_3)
dt_cn_large_3 = 20 * dt_explicit_limit_3

print("\nProblem setup:")
print(f"  Domain: 2D, [0,1]×[0,1] with {mesh_3.num_nodes()} nodes")
print(f"  Diffusivity: D = {D_3:.2e} m²/s")
print(f"  Mesh spacing: Δx = Δy = {dx_3:.6f} m")
print(f"  Explicit 2D stability limit: Δt ≤ {dt_explicit_limit_3:.6e} s")
print(f"  CN time step (20x limit): Δt = {dt_cn_large_3:.6e} s")

# Initial condition: Central Gaussian source
initial_3 = bt.gaussian(mesh_3, center=0.5, width=0.1)

problem_3 = (
    bt.Problem(mesh_3)
    .diffusivity(D_3)
    .initial_condition(initial_3)
    .neumann(bt.Boundary.Left, 0.0)
    .neumann(bt.Boundary.Right, 0.0)
    .neumann(bt.Boundary.Bottom, 0.0)
    .neumann(bt.Boundary.Top, 0.0)
)

t_end_3 = 0.5

print(f"\nSimulation time: {t_end_3} s")
print(f"Steps with explicit: {int(t_end_3 / dt_explicit_limit_3):,}")
print(f"Steps with CN: {int(np.ceil(t_end_3 / dt_cn_large_3))}")

print("\nTiming comparison:")

# Explicit (may take a while!)
start = time_module.time()
result_explicit_3 = bt.solve(
    problem_3, t=t_end_3, dt=dt_explicit_limit_3, method="explicit"
)
time_explicit_3 = time_module.time() - start
print(f"  Explicit: {time_explicit_3:.3f} s ({result_explicit_3.stats.steps} steps)")

# Crank-Nicolson
start = time_module.time()
result_cn_3 = bt.solve(problem_3, t=t_end_3, dt=dt_cn_large_3, method="crank_nicolson")
time_cn_3 = time_module.time() - start
print(f"  Crank-Nicolson: {time_cn_3:.3f} s ({result_cn_3.stats.steps} steps)")
print(f"  ⚡ Speedup: {time_explicit_3/time_cn_3:.1f}x faster with CN")

# ========================================================================
# Visualizations
# ========================================================================
print(f"\n{'='*80}")
print("Creating comparison plots...")
print(f"{'='*80}")

# Figure 1: Scenario 1 - Small diffusivity
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
x_1 = bt.x_nodes(mesh_1)

axes1[0].plot(x_1, initial_1, "k--", alpha=0.5, label="Initial", linewidth=2)
axes1[0].plot(x_1, result_explicit_1.solution(), "b-", label="Explicit", linewidth=2)
axes1[0].set_xlabel("Position x", fontsize=12)
axes1[0].set_ylabel("Concentration", fontsize=12)
axes1[0].set_title(
    f"Explicit Method\nD={D_small:.1e}, {result_explicit_1.stats.steps} steps, {time_explicit_1:.2f}s",
    fontsize=11,
)
axes1[0].legend()
axes1[0].grid(True, alpha=0.3)

axes1[1].plot(x_1, initial_1, "k--", alpha=0.5, label="Initial", linewidth=2)
axes1[1].plot(x_1, result_cn_1.solution(), "r-", label="Crank-Nicolson", linewidth=2)
axes1[1].set_xlabel("Position x", fontsize=12)
axes1[1].set_ylabel("Concentration", fontsize=12)
axes1[1].set_title(
    f"Crank-Nicolson\nD={D_small:.1e}, {result_cn_1.stats.steps} steps, {time_cn_1:.2f}s",
    fontsize=11,
)
axes1[1].legend()
axes1[1].grid(True, alpha=0.3)

fig1.suptitle(
    "Scenario 1: Small Diffusivity (Very Stiff)", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    bt.get_result_path("scenario1_small_diffusivity.png", EXAMPLE_NAME), dpi=150
)

# Figure 2: Scenario 2 - Fine mesh
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
x_2 = bt.x_nodes(mesh_2)

axes2[0].plot(x_2, initial_2, "k--", alpha=0.5, label="Initial", linewidth=2)
axes2[0].plot(x_2, result_explicit_2.solution(), "b-", label="Explicit", linewidth=1)
axes2[0].set_xlabel("Position x", fontsize=12)
axes2[0].set_ylabel("Concentration", fontsize=12)
axes2[0].set_title(
    f"Explicit Method\n{mesh_2.num_nodes()} nodes, {result_explicit_2.stats.steps} steps, {time_explicit_2:.2f}s",
    fontsize=11,
)
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

axes2[1].plot(x_2, initial_2, "k--", alpha=0.5, label="Initial", linewidth=2)
axes2[1].plot(x_2, result_cn_2.solution(), "r-", label="Crank-Nicolson", linewidth=1)
axes2[1].set_xlabel("Position x", fontsize=12)
axes2[1].set_ylabel("Concentration", fontsize=12)
axes2[1].set_title(
    f"Crank-Nicolson\n{mesh_2.num_nodes()} nodes, {result_cn_2.stats.steps} steps, {time_cn_2:.2f}s",
    fontsize=11,
)
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)

fig2.suptitle("Scenario 2: Fine Mesh (Spatially Stiff)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(bt.get_result_path("scenario2_fine_mesh.png", EXAMPLE_NAME), dpi=150)

# Figure 3: Scenario 3 - 2D problem
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

bt.plot(
    mesh_3,
    result_explicit_3.solution(),
    ax=axes3[0],
    title=f"Explicit\n{result_explicit_3.stats.steps} steps, {time_explicit_3:.2f}s",
    colorbar=True,
    cmap="hot",
)

bt.plot(
    mesh_3,
    result_cn_3.solution(),
    ax=axes3[1],
    title=f"Crank-Nicolson\n{result_cn_3.stats.steps} steps, {time_cn_3:.2f}s",
    colorbar=True,
    cmap="hot",
)

fig3.suptitle(f"Scenario 3: 2D Problem (D={D_3:.1e})", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(bt.get_result_path("scenario3_2d_problem.png", EXAMPLE_NAME), dpi=150)

# Figure 4: Performance summary
fig4, ax4 = plt.subplots(figsize=(10, 6))
scenarios = ["Small D\n(1D)", "Fine Mesh\n(1D)", "2D Problem"]
speedups = [
    time_explicit_1 / time_cn_1,
    time_explicit_2 / time_cn_2,
    time_explicit_3 / time_cn_3,
]
colors = ["#2E86AB", "#A23B72", "#F18F01"]

bars = ax4.bar(
    scenarios, speedups, color=colors, alpha=0.7, edgecolor="black", linewidth=2
)
ax4.axhline(1, color="k", linestyle="--", alpha=0.5, label="No speedup")
ax4.set_ylabel("Speedup Factor (CN vs Explicit)", fontsize=12)
ax4.set_title(
    "Crank-Nicolson Performance Advantage\nfor Stiff Problems",
    fontsize=14,
    fontweight="bold",
)
ax4.grid(True, axis="y", alpha=0.3)

# Add value labels on bars
for bar, speedup in zip(bars, speedups):
    height = bar.get_height()
    ax4.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{speedup:.1f}x",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

ax4.legend()
plt.tight_layout()
plt.savefig(bt.get_result_path("performance_summary.png", EXAMPLE_NAME), dpi=150)

plt.show()

# ========================================================================
# Summary
# ========================================================================
print(f"\n{'='*80}")
print("SUMMARY: When to Use Crank-Nicolson for Stiff Problems")
print(f"{'='*80}")

avg_speedup = np.mean(speedups)

print("\n⚡ Performance Results:")
print(f"  Scenario 1 (Small D): {speedups[0]:.1f}x speedup")
print(f"  Scenario 2 (Fine Mesh): {speedups[1]:.1f}x speedup")
print(f"  Scenario 3 (2D): {speedups[2]:.1f}x speedup")
print(f"  Average: {avg_speedup:.1f}x speedup")

print("\n✓ Use Crank-Nicolson when:")
print("  • Diffusivity is very small (D < 1e-5)")
print("  • Mesh spacing is fine (Δx < 0.01)")
print("  • Working in 2D/3D where stability limits compound")
print("  • Explicit stability criterion requires tiny time steps")
print("  • Long-time simulations are needed")
print("  • Accuracy is more important than speed per step")

print("\n✗ Explicit methods may be better when:")
print("  • Problem is not stiff (moderate D, coarse mesh)")
print("  • Very short simulation times")
print("  • Per-step computational cost is critical")
print("  • Simplicity is valued over efficiency")

print("\n📁 Results saved to:")
print(f"  {bt.get_result_path('', EXAMPLE_NAME)}")
print(f"{'='*80}")
