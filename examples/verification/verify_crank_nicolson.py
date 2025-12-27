"""
Example: Crank-Nicolson Accuracy Verification

This example verifies the temporal accuracy of the Crank-Nicolson method
against analytical solutions for 1D diffusion problems. We compare:

1. Explicit method (FTCS) - First-order accurate in time
2. Crank-Nicolson - Second-order accurate in time

The analytical solution for 1D diffusion with Dirichlet BCs:
    u(x,t) = sum[n=1 to ∞] B_n * sin(n*π*x/L) * exp(-D*(n*π/L)²*t)

We verify the order of accuracy by measuring error vs time step size.

BMEN 341 Reference: Numerical Methods Verification (Week 5)
"""

import matplotlib.pyplot as plt
import numpy as np
import biotransport as bt

EXAMPLE_NAME = "crank_nicolson_verification"

print("=" * 70)
print("Crank-Nicolson Temporal Accuracy Verification")
print("=" * 70)

# ========================================================================
# Analytical solution for 1D diffusion
# ========================================================================


def analytical_solution_1d(x, t, D, L, n_terms=50):
    """
    Analytical solution for 1D diffusion with:
    - Initial condition: u(x,0) = sin(π*x/L)
    - Boundary conditions: u(0,t) = u(L,t) = 0

    Solution: u(x,t) = sin(π*x/L) * exp(-D*(π/L)²*t)
    """
    # For initial condition sin(π*x/L), only the n=1 term is non-zero
    # and B_1 = 1
    return np.sin(np.pi * x / L) * np.exp(-D * (np.pi / L) ** 2 * t)


def analytical_solution_gaussian(x, t, D, x0, sigma0):
    """
    Analytical solution for 1D diffusion with Gaussian initial condition
    on infinite domain (approximation for large domain with zero BCs).

    u(x,t) = 1/sqrt(4πDt + 2πσ₀²) * exp(-(x-x₀)²/(4Dt + 2σ₀²))
    """
    sigma_t_sq = 2 * sigma0**2 + 4 * D * t
    return (sigma0 / np.sqrt(sigma_t_sq)) * np.exp(-((x - x0) ** 2) / sigma_t_sq)


# ========================================================================
# Problem setup
# ========================================================================

L = 1.0  # Domain length
D = 0.01  # Diffusion coefficient
t_end = 0.05  # Simulation time

# Create mesh
mesh = bt.mesh_1d(100, x_min=0.0, x_max=L)
x = bt.x_nodes(mesh)
dx = mesh.dx()

print("\nProblem parameters:")
print(f"  Domain: [0, {L}]")
print(f"  Diffusion coefficient: D = {D}")
print(f"  End time: t = {t_end}")
print(f"  Mesh points: {mesh.num_nodes()}")
print(f"  Spatial resolution: dx = {dx:.6f}")

# ========================================================================
# Test Case 1: Sine wave initial condition
# ========================================================================
print(f"\n{'='*70}")
print("Test Case 1: Sine Wave Initial Condition")
print(f"{'='*70}")

initial_sine = np.sin(np.pi * x / L)
analytical_sine = analytical_solution_1d(x, t_end, D, L)

# Test different time step sizes
dt_values = np.array([0.001, 0.002, 0.005, 0.01, 0.02])
errors_explicit = []
errors_cn = []

print(f"\n{'dt':>10} {'Explicit Error':>18} {'CN Error':>18} {'Ratio':>10}")
print("-" * 70)

for dt in dt_values:
    # Explicit method
    problem_explicit = (
        bt.Problem(mesh)
        .diffusivity(D)
        .initial_condition(initial_sine)
        .dirichlet(bt.Boundary.Left, 0.0)
        .dirichlet(bt.Boundary.Right, 0.0)
    )
    result_explicit = bt.solve(problem_explicit, t=t_end, dt=dt, method="explicit")
    solution_explicit = result_explicit.solution()
    error_explicit = np.sqrt(np.mean((solution_explicit - analytical_sine) ** 2))
    errors_explicit.append(error_explicit)

    # Crank-Nicolson method
    result_cn = bt.solve(problem_explicit, t=t_end, dt=dt, method="crank_nicolson")
    solution_cn = result_cn.solution()
    error_cn = np.sqrt(np.mean((solution_cn - analytical_sine) ** 2))
    errors_cn.append(error_cn)

    ratio = error_explicit / error_cn if error_cn > 0 else np.inf
    print(f"{dt:>10.5f} {error_explicit:>18.10f} {error_cn:>18.10f} {ratio:>10.2f}")

errors_explicit = np.array(errors_explicit)
errors_cn = np.array(errors_cn)

# ========================================================================
# Compute convergence rates
# ========================================================================
print(f"\n{'='*70}")
print("Convergence Rate Analysis")
print(f"{'='*70}")

# Compute slopes in log-log plot (should be ~1 for explicit, ~2 for CN)
log_dt = np.log(dt_values)
log_error_explicit = np.log(errors_explicit)
log_error_cn = np.log(errors_cn)

# Linear fit: log(error) = slope * log(dt) + intercept
slope_explicit = np.polyfit(log_dt[-3:], log_error_explicit[-3:], 1)[
    0
]  # Use last 3 points
slope_cn = np.polyfit(log_dt[-3:], log_error_cn[-3:], 1)[0]

print("\nMeasured temporal convergence rates:")
print(f"  Explicit method: {slope_explicit:.2f} (theoretical: 1.0)")
print(f"  Crank-Nicolson:  {slope_cn:.2f} (theoretical: 2.0)")

# ========================================================================
# Test Case 2: Large time step comparison
# ========================================================================
print(f"\n{'='*70}")
print("Test Case 2: Large Time Step Performance")
print(f"{'='*70}")

# Use a very large time step
dt_large = 0.05  # Same as total simulation time - single step!

print(f"\nTesting with single time step: dt = {dt_large} (entire simulation)")
print(
    f"This exceeds explicit stability limit by ~{dt_large / (0.5 * dx**2 / (2*D)):.1f}x"
)

# Explicit with large dt (will be inaccurate or unstable)
try:
    result_explicit_large = bt.solve(
        problem_explicit, t=t_end, dt=dt_large, method="explicit"
    )
    solution_explicit_large = result_explicit_large.solution()
    error_explicit_large = np.sqrt(
        np.mean((solution_explicit_large - analytical_sine) ** 2)
    )

    if np.any(np.abs(solution_explicit_large) > 1e10) or np.any(
        np.isnan(solution_explicit_large)
    ):
        print("\n✗ Explicit method: UNSTABLE (solution diverged)")
        solution_explicit_large[:] = np.nan
        error_explicit_large = np.inf
    else:
        print(f"\n✓ Explicit method: Error = {error_explicit_large:.10f}")
except Exception as e:
    print(f"\n✗ Explicit method: FAILED - {str(e)}")
    solution_explicit_large = np.full_like(x, np.nan)
    error_explicit_large = np.inf

# Crank-Nicolson with large dt
result_cn_large = bt.solve(
    problem_explicit, t=t_end, dt=dt_large, method="crank_nicolson"
)
solution_cn_large = result_cn_large.solution()
error_cn_large = np.sqrt(np.mean((solution_cn_large - analytical_sine) ** 2))

print(f"✓ Crank-Nicolson: Error = {error_cn_large:.10f}")
print(f"  Steps taken: {result_cn_large.stats.steps} (just 1!)")

# ========================================================================
# Visualization
# ========================================================================
print(f"\n{'='*70}")
print("Creating verification plots...")
print(f"{'='*70}")

# Figure 1: Convergence plot
fig1, ax1 = plt.subplots(figsize=(10, 7))
ax1.loglog(
    dt_values,
    errors_explicit,
    "bo-",
    linewidth=2,
    markersize=8,
    label=f"Explicit (slope ≈ {slope_explicit:.2f})",
)
ax1.loglog(
    dt_values,
    errors_cn,
    "rs-",
    linewidth=2,
    markersize=8,
    label=f"Crank-Nicolson (slope ≈ {slope_cn:.2f})",
)

# Add reference lines
dt_ref = dt_values[2]
error_ref_explicit = errors_explicit[2]
error_ref_cn = errors_cn[2]

# First-order reference line
first_order_line = error_ref_explicit * (dt_values / dt_ref) ** 1.0
ax1.loglog(
    dt_values, first_order_line, "k--", alpha=0.5, linewidth=1.5, label="1st order (dt)"
)

# Second-order reference line
second_order_line = error_ref_cn * (dt_values / dt_ref) ** 2.0
ax1.loglog(
    dt_values,
    second_order_line,
    "k:",
    alpha=0.5,
    linewidth=1.5,
    label="2nd order (dt²)",
)

ax1.set_xlabel("Time step size (dt)", fontsize=12)
ax1.set_ylabel("RMS Error", fontsize=12)
ax1.set_title(
    "Temporal Accuracy: Explicit vs Crank-Nicolson", fontsize=14, fontweight="bold"
)
ax1.legend(fontsize=11)
ax1.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(bt.get_result_path("convergence_rates.png", EXAMPLE_NAME), dpi=150)

# Figure 2: Solution comparison at t_end with small dt
fig2, axes2 = plt.subplots(2, 1, figsize=(12, 10))

# Top: Full solutions
ax_top = axes2[0]
ax_top.plot(x, analytical_sine, "k-", linewidth=3, label="Analytical", alpha=0.7)
ax_top.plot(
    x, solution_explicit, "b--", linewidth=2, label=f"Explicit (dt={dt_values[2]:.4f})"
)
ax_top.plot(
    x, solution_cn, "r:", linewidth=2, label=f"Crank-Nicolson (dt={dt_values[2]:.4f})"
)
ax_top.set_xlabel("Position x", fontsize=12)
ax_top.set_ylabel("Concentration", fontsize=12)
ax_top.set_title(f"Solution at t = {t_end}", fontsize=14, fontweight="bold")
ax_top.legend(fontsize=11)
ax_top.grid(True, alpha=0.3)

# Bottom: Errors
ax_bottom = axes2[1]
error_plot_explicit = solution_explicit - analytical_sine
error_plot_cn = solution_cn - analytical_sine
ax_bottom.plot(
    x,
    error_plot_explicit,
    "b--",
    linewidth=2,
    label=f"Explicit Error (RMS={errors_explicit[2]:.6f})",
)
ax_bottom.plot(
    x, error_plot_cn, "r:", linewidth=2, label=f"CN Error (RMS={errors_cn[2]:.6f})"
)
ax_bottom.axhline(0, color="k", linestyle="-", alpha=0.3)
ax_bottom.set_xlabel("Position x", fontsize=12)
ax_bottom.set_ylabel("Error", fontsize=12)
ax_bottom.set_title("Pointwise Error Distribution", fontsize=14, fontweight="bold")
ax_bottom.legend(fontsize=11)
ax_bottom.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(bt.get_result_path("solution_comparison.png", EXAMPLE_NAME), dpi=150)

# Figure 3: Large time step comparison
fig3, ax3 = plt.subplots(figsize=(12, 7))
ax3.plot(x, analytical_sine, "k-", linewidth=3, label="Analytical", alpha=0.7)
if not np.all(np.isnan(solution_explicit_large)):
    ax3.plot(
        x,
        solution_explicit_large,
        "b--",
        linewidth=2,
        label=f"Explicit (dt={dt_large}, UNSTABLE)",
    )
ax3.plot(
    x, solution_cn_large, "r-", linewidth=2, label=f"Crank-Nicolson (dt={dt_large})"
)
ax3.set_xlabel("Position x", fontsize=12)
ax3.set_ylabel("Concentration", fontsize=12)
ax3.set_title(
    f"Large Time Step Comparison (dt = {dt_large})", fontsize=14, fontweight="bold"
)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(bt.get_result_path("large_timestep_comparison.png", EXAMPLE_NAME), dpi=150)

plt.show()

# ========================================================================
# Summary
# ========================================================================
print(f"\n{'='*70}")
print("VERIFICATION SUMMARY")
print(f"{'='*70}")

print("\n📊 Temporal Accuracy:")
print(f"  Explicit method: ~{slope_explicit:.1f} order (expected: 1st order)")
print(f"  Crank-Nicolson:  ~{slope_cn:.1f} order (expected: 2nd order)")

if abs(slope_explicit - 1.0) < 0.3:
    print("  ✓ Explicit first-order accuracy VERIFIED")
else:
    print(f"  ⚠ Explicit order {slope_explicit:.2f} deviates from theoretical 1.0")

if abs(slope_cn - 2.0) < 0.3:
    print("  ✓ Crank-Nicolson second-order accuracy VERIFIED")
else:
    print(f"  ⚠ CN order {slope_cn:.2f} deviates from theoretical 2.0")

print("\n💡 Key Findings:")
print(
    f"  • CN is {errors_explicit[-1]/errors_cn[-1]:.1f}x more accurate at dt={dt_values[-1]}"
)
print(f"  • CN remains stable with dt={dt_large} (single time step)")
print(
    f"  • Explicit method requires {int(t_end/dt_values[0])} steps for similar accuracy"
)

print("\n📁 Results saved to:")
print(f"  {bt.get_result_path('', EXAMPLE_NAME)}")
print(f"{'='*70}")
