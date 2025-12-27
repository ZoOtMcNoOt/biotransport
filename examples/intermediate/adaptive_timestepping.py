"""
Adaptive Time-Stepping Example

This example demonstrates the benefits of adaptive time-stepping for
transport simulations. We compare:

1. Fixed time-stepping (very small dt for accuracy)
2. Adaptive time-stepping (error-controlled)

The test problem is 1D diffusion with a sharp Gaussian initial condition.
Early in the simulation, the solution changes rapidly and needs small steps.
Later, as it smooths out, larger steps suffice. Adaptive stepping captures
this automatically.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import biotransport as bt

# =============================================================================
# Problem Setup
# =============================================================================

# Domain and mesh
L = 1.0  # 1 meter domain
nx = 100  # 100 cells
mesh = bt.mesh_1d(nx, 0.0, L)

# Material properties
D = 1e-4  # Diffusivity [m²/s]

# Initial condition: Very sharp Gaussian peak (challenging case)
x = np.array([mesh.x(i) for i in range(nx + 1)])
sigma = 0.02  # Very narrow peak (2 cm width) - needs fine resolution early
u0 = np.exp(-((x - L / 2) ** 2) / (2 * sigma**2))

# Boundary conditions: Zero flux (Neumann) on both ends
problem = (
    bt.Problem(mesh)
    .diffusivity(D)
    .initial_condition(u0.tolist())
    .boundary(bt.Boundary.Left, bt.BoundaryCondition.neumann(0.0))
    .boundary(bt.Boundary.Right, bt.BoundaryCondition.neumann(0.0))
)

# Simulation time
t_end = 1.0  # seconds

print("=" * 60)
print("Adaptive Time-Stepping Demonstration")
print("=" * 60)
print(f"Domain: [0, {L}] m with {nx} cells")
print(f"Diffusivity: D = {D:.2e} m²/s")
print(f"Initial: Sharp Gaussian peak (σ = {sigma} m)")
print(f"Simulation time: {t_end} s")
print()

# =============================================================================
# Method 1: Fixed Time-Stepping (ExplicitFD with default CFL)
# =============================================================================

print("Running with FIXED time-stepping (CFL-based)...")
start = time.perf_counter()
result_fixed = bt.ExplicitFD().run(problem, t_end)
time_fixed = time.perf_counter() - start
u_fixed = np.array(result_fixed.solution())

print(f"  Steps: {result_fixed.stats.steps}")
print(f"  dt: {result_fixed.stats.dt:.6e} s")
print(f"  Wall time: {time_fixed:.3f} s")
print()

# =============================================================================
# Method 2: Adaptive Time-Stepping (loose tolerance)
# =============================================================================

print("Running with ADAPTIVE time-stepping (tol=1e-3)...")
start = time.perf_counter()
stepper = bt.AdaptiveTimeStepper(problem, tol=1e-3, verbose=False)
result_adaptive = stepper.solve(t_end)
time_adaptive = time.perf_counter() - start
u_adaptive = result_adaptive.solution

stats = result_adaptive.stats
print(f"  Steps: {stats['steps']}")
print(f"  Rejections: {stats['rejections']}")
print(f"  dt range: [{stats['dt_min_used']:.6e}, {stats['dt_max_used']:.6e}] s")
print(f"  dt average: {stats['dt_avg']:.6e} s")
print(f"  CFL limit: {stats['cfl_limit']:.6e} s")
print(f"  Wall time: {time_adaptive:.3f} s")
print()

# =============================================================================
# Method 3: Tight tolerance (high accuracy)
# =============================================================================

print("Running with ADAPTIVE (tol=1e-5, high accuracy)...")
start = time.perf_counter()
stepper_tight = bt.AdaptiveTimeStepper(problem, tol=1e-5, verbose=False)
result_tight = stepper_tight.solve(t_end)
time_tight = time.perf_counter() - start
u_tight = result_tight.solution

stats_tight = result_tight.stats
print(f"  Steps: {stats_tight['steps']}")
print(f"  Rejections: {stats_tight['rejections']}")
print(
    f"  dt range: [{stats_tight['dt_min_used']:.6e}, {stats_tight['dt_max_used']:.6e}] s"
)
print(f"  Wall time: {time_tight:.3f} s")
print()

# =============================================================================
# Comparison
# =============================================================================

print("=" * 60)
print("COMPARISON")
print("=" * 60)

# Compute differences using tight tolerance as reference
diff_adaptive = np.max(np.abs(u_adaptive - u_tight))
diff_fixed = np.max(np.abs(u_fixed - u_tight))

print(f"Max difference (adaptive tol=1e-3 vs 1e-5 reference): {diff_adaptive:.2e}")
print(f"Max difference (fixed CFL vs 1e-5 reference): {diff_fixed:.2e}")
print()

print("Key insight: Adaptive time-stepping provides:")
print("  1. Automatic error control (user specifies tolerance)")
print("  2. Smaller steps early when solution changes rapidly")
print("  3. Larger steps later when solution is smooth")
print("  4. Robust handling of unknown dynamics")
print()

# =============================================================================
# Visualization
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Solutions comparison
ax1 = axes[0, 0]
ax1.plot(x * 100, u0, "k--", label="Initial", linewidth=2)
ax1.plot(x * 100, u_fixed, "b-", label="Fixed (CFL)", linewidth=2)
ax1.plot(x * 100, u_adaptive, "r--", label="Adaptive (tol=1e-3)", linewidth=2)
ax1.plot(x * 100, u_tight, "g:", label="Adaptive (tol=1e-5)", linewidth=2)
ax1.set_xlabel("Position (cm)")
ax1.set_ylabel("Concentration")
ax1.set_title(f"Solution Comparison at t = {t_end} s")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Difference from reference
ax2 = axes[0, 1]
ax2.plot(x * 100, np.abs(u_fixed - u_tight), "b-", label="Fixed (CFL)")
ax2.plot(x * 100, np.abs(u_adaptive - u_tight), "r-", label="Adaptive (tol=1e-3)")
ax2.set_xlabel("Position (cm)")
ax2.set_ylabel("Absolute Difference from Reference")
ax2.set_title("Error vs Reference (tol=1e-5)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Time step history
ax3 = axes[1, 0]
dt_history = stats["dt_history"]
if dt_history:
    step_times = np.cumsum(dt_history)
    ax3.semilogy(
        step_times, dt_history, "r-", linewidth=1.5, label="Adaptive (tol=1e-3)"
    )
dt_history_tight = stats_tight["dt_history"]
if dt_history_tight:
    step_times_tight = np.cumsum(dt_history_tight)
    ax3.semilogy(
        step_times_tight,
        dt_history_tight,
        "g-",
        alpha=0.7,
        linewidth=1,
        label="Adaptive (tol=1e-5)",
    )
ax3.axhline(
    result_fixed.stats.dt, color="b", linestyle="--", linewidth=2, label="Fixed (CFL)"
)
ax3.axhline(stats["cfl_limit"], color="k", linestyle=":", label="CFL stability limit")
ax3.set_xlabel("Simulation Time (s)")
ax3.set_ylabel("Time Step (s)")
ax3.set_title("Time Step Evolution")
ax3.legend(loc="upper left")
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, t_end)

# Plot 4: Summary comparison
ax4 = axes[1, 1]
methods = ["Fixed\n(CFL)", "Adaptive\n(tol=1e-3)", "Adaptive\n(tol=1e-5)"]
step_counts = [result_fixed.stats.steps, stats["steps"], stats_tight["steps"]]
colors = ["blue", "red", "green"]
bars = ax4.bar(methods, step_counts, color=colors, alpha=0.7, edgecolor="black")
ax4.set_ylabel("Number of Steps")
ax4.set_title("Computational Cost")

for bar, count in zip(bars, step_counts):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(step_counts) * 0.02,
        str(count),
        ha="center",
        va="bottom",
        fontweight="bold",
    )

ax4.set_ylim(0, max(step_counts) * 1.15)
ax4.grid(True, alpha=0.3, axis="y")

plt.suptitle("Adaptive vs Fixed Time-Stepping", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(bt.get_result_path("adaptive_timestepping.png", "adaptive"), dpi=150)
plt.show()

print("✅ Adaptive time-stepping example completed!")
print(
    f"   Plot saved to: {bt.get_result_path('adaptive_timestepping.png', 'adaptive')}"
)
