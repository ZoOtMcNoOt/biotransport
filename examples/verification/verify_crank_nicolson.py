"""Temporal-order evidence for one semidiscrete 1D diffusion problem.

The spatially discretized sine mode has a known exact evolution.  Using that
method-of-lines reference isolates time-integration error from spatial error.
The script checks whether explicit Euler and Crank-Nicolson meet declared
observed-order criteria over one timestep sequence, plus one explicitly scoped
single-step error criterion.  It does not establish accuracy or stability for
other meshes, equations, timesteps, or the library as a whole.

BMEN 341 Reference: Numerical Methods Verification (Week 5)
"""

import biotransport as bt
import matplotlib.pyplot as plt
import numpy as np

EXAMPLE_NAME = "crank_nicolson_verification"

print("=" * 70)
print("Crank-Nicolson Temporal-Order Evidence")
print("=" * 70)

# ========================================================================
# Analytical solution for 1D diffusion
# ========================================================================


def semidiscrete_sine_reference(
    x: np.ndarray,
    t: float,
    diffusivity: float,
    length: float,
    spacing: float,
) -> np.ndarray:
    """Exact evolution of the first sine mode under the discrete Laplacian."""
    eigenvalue = (
        -4.0 * diffusivity / spacing**2 * np.sin(np.pi * spacing / (2.0 * length)) ** 2
    )
    return np.sin(np.pi * x / length) * np.exp(eigenvalue * t)


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
print("  Error reference: exact first eigenmode of the discrete Laplacian")

# ========================================================================
# Test Case 1: Sine wave initial condition
# ========================================================================
print(f"\n{'=' * 70}")
print("Test Case 1: Sine Wave Initial Condition")
print(f"{'=' * 70}")

initial_sine = np.sin(np.pi * x / L)
reference_sine = semidiscrete_sine_reference(x, t_end, D, L, dx)

# Coarse-to-fine steps; each divides t_end and all satisfy the 1D explicit limit.
dt_values = np.array([0.005, 0.0025, 0.00125, 0.000625, 0.0003125])
errors_explicit: list[float] = []
errors_cn: list[float] = []

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
    result_explicit = bt.solve(
        problem_explicit,
        end_time=t_end,
        time_step=dt,
        method="explicit",
    )
    solution_explicit = np.asarray(result_explicit.concentration)
    error_explicit = np.sqrt(np.mean((solution_explicit - reference_sine) ** 2))
    errors_explicit.append(error_explicit)

    # Specialized C++ Crank-Nicolson solver.
    num_steps = int(round(t_end / dt))
    if not np.isclose(num_steps * dt, t_end, rtol=0.0, atol=1e-14):
        raise ValueError(f"dt={dt} does not divide t_end={t_end}")
    solver_cn = bt.CrankNicolsonDiffusion(mesh, D)
    solver_cn.set_initial_condition(initial_sine)
    solver_cn.set_dirichlet_boundary(bt.Boundary.Left, 0.0)
    solver_cn.set_dirichlet_boundary(bt.Boundary.Right, 0.0)
    solver_cn.solve(dt=dt, num_steps=num_steps)
    solution_cn = np.asarray(solver_cn.solution())
    error_cn = np.sqrt(np.mean((solution_cn - reference_sine) ** 2))
    errors_cn.append(error_cn)

    ratio = error_explicit / error_cn if error_cn > 0 else np.inf
    print(f"{dt:>10.5f} {error_explicit:>18.10f} {error_cn:>18.10f} {ratio:>10.2f}")

errors_explicit_array = np.asarray(errors_explicit)
errors_cn_array = np.asarray(errors_cn)

if (
    np.any(~np.isfinite(errors_explicit_array))
    or np.any(~np.isfinite(errors_cn_array))
    or np.any(errors_explicit_array <= 0.0)
    or np.any(errors_cn_array <= 0.0)
):
    raise RuntimeError("Temporal-order fit requires finite, positive RMS errors")

# ========================================================================
# Compute convergence rates
# ========================================================================
print(f"\n{'=' * 70}")
print("Convergence Rate Analysis")
print(f"{'=' * 70}")

# Compute slopes in a log-log least-squares fit over the stated sequence.
log_dt = np.log(dt_values)
log_error_explicit = np.log(errors_explicit_array)
log_error_cn = np.log(errors_cn_array)

# Linear fit: log(error) = slope * log(dt) + intercept
slope_explicit = np.polyfit(log_dt, log_error_explicit, 1)[0]
slope_cn = np.polyfit(log_dt, log_error_cn, 1)[0]

print("\nObserved temporal orders from all five RMS-error values:")
print(f"  Explicit method: {slope_explicit:.3f} (comparison value: 1.0)")
print(f"  Crank-Nicolson:  {slope_cn:.3f} (comparison value: 2.0)")

# ========================================================================
# Test Case 2: Large time step comparison
# ========================================================================
print(f"\n{'=' * 70}")
print("Test Case 2: One Configured Large-Step Observation")
print(f"{'=' * 70}")

# Use a very large time step
dt_large = 0.05  # Same as total simulation time - single step!

print(f"\nTesting with single time step: dt = {dt_large} (entire simulation)")
explicit_stability_limit = dx**2 / (2 * D)
print(
    "Requested dt / 1D explicit diffusion limit = "
    f"{dt_large / explicit_stability_limit:.1f}"
)

# Record exactly how the explicit solver handles this out-of-limit request.
try:
    result_explicit_large = bt.solve(
        problem_explicit,
        end_time=t_end,
        time_step=dt_large,
        method="explicit",
    )
    solution_explicit_large = np.asarray(result_explicit_large.concentration)
    error_explicit_large = np.sqrt(
        np.mean((solution_explicit_large - reference_sine) ** 2)
    )
    explicit_large_status = (
        "completed; "
        f"finite={bool(np.all(np.isfinite(solution_explicit_large)))}, "
        f"RMS error={error_explicit_large:.3e}"
    )
except Exception as e:
    explicit_large_status = f"rejected with {type(e).__name__}: {e}"
    solution_explicit_large = np.full_like(x, np.nan)
    error_explicit_large = np.inf
print(f"\nExplicit out-of-limit request: {explicit_large_status}")

# Crank-Nicolson with one large step.
solver_cn_large = bt.CrankNicolsonDiffusion(mesh, D)
solver_cn_large.set_initial_condition(initial_sine)
solver_cn_large.set_dirichlet_boundary(bt.Boundary.Left, 0.0)
solver_cn_large.set_dirichlet_boundary(bt.Boundary.Right, 0.0)
solver_cn_large.solve(dt=dt_large, num_steps=1)
solution_cn_large = np.asarray(solver_cn_large.solution())
error_cn_large = np.sqrt(np.mean((solution_cn_large - reference_sine) ** 2))

print(f"Crank-Nicolson one-step RMS error: {error_cn_large:.3e}")
print(f"Crank-Nicolson reported final time: {solver_cn_large.time():.6f}")

ORDER_TOLERANCE = 0.3
SINGLE_STEP_RMS_LIMIT = 1e-6
explicit_order_passed = (
    np.isfinite(slope_explicit) and abs(slope_explicit - 1.0) < ORDER_TOLERANCE
)
cn_order_passed = np.isfinite(slope_cn) and abs(slope_cn - 2.0) < ORDER_TOLERANCE
cn_single_step_passed = (
    np.all(np.isfinite(solution_cn_large))
    and error_cn_large < SINGLE_STEP_RMS_LIMIT
    and np.isclose(solver_cn_large.time(), t_end, rtol=0.0, atol=1e-14)
)

# ========================================================================
# Visualization
# ========================================================================
print(f"\n{'=' * 70}")
print("Creating scoped numerical-evidence plots...")
print(f"{'=' * 70}")

# Figure 1: Convergence plot
fig1, ax1 = plt.subplots(figsize=(10, 7))
ax1.loglog(
    dt_values,
    errors_explicit_array,
    "bo-",
    linewidth=2,
    markersize=8,
    label=f"Explicit (slope ≈ {slope_explicit:.2f})",
)
ax1.loglog(
    dt_values,
    errors_cn_array,
    "rs-",
    linewidth=2,
    markersize=8,
    label=f"Crank-Nicolson (slope ≈ {slope_cn:.2f})",
)

# Add reference lines
dt_ref = dt_values[2]
error_ref_explicit = errors_explicit_array[2]
error_ref_cn = errors_cn_array[2]

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
    "Temporal RMS Error vs Timestep: Explicit and Crank-Nicolson",
    fontsize=14,
    fontweight="bold",
)
ax1.legend(fontsize=11)
ax1.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(bt.get_result_path("convergence_rates.png", EXAMPLE_NAME), dpi=150)

# Figure 2: Solution comparison at t_end with the finest tested dt.
fig2, axes2 = plt.subplots(2, 1, figsize=(12, 10))

# Top: Full solutions
ax_top = axes2[0]
ax_top.plot(
    x,
    reference_sine,
    "k-",
    linewidth=3,
    label="Semidiscrete exact reference",
    alpha=0.7,
)
ax_top.plot(
    x,
    solution_explicit,
    "b--",
    linewidth=2,
    label=f"Explicit (dt={dt_values[-1]:.7f})",
)
ax_top.plot(
    x,
    solution_cn,
    "r:",
    linewidth=2,
    label=f"Crank-Nicolson (dt={dt_values[-1]:.7f})",
)
ax_top.set_xlabel("Position x", fontsize=12)
ax_top.set_ylabel("Concentration", fontsize=12)
ax_top.set_title(f"Solution at t = {t_end}", fontsize=14, fontweight="bold")
ax_top.legend(fontsize=11)
ax_top.grid(True, alpha=0.3)

# Bottom: Errors
ax_bottom = axes2[1]
error_plot_explicit = solution_explicit - reference_sine
error_plot_cn = solution_cn - reference_sine
ax_bottom.plot(
    x,
    error_plot_explicit,
    "b--",
    linewidth=2,
    label=f"Explicit Error (RMS={errors_explicit_array[-1]:.3e})",
)
ax_bottom.plot(
    x,
    error_plot_cn,
    "r:",
    linewidth=2,
    label=f"CN Error (RMS={errors_cn_array[-1]:.3e})",
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
ax3.plot(
    x,
    reference_sine,
    "k-",
    linewidth=3,
    label="Semidiscrete exact reference",
    alpha=0.7,
)
if not np.all(np.isnan(solution_explicit_large)):
    ax3.plot(
        x,
        solution_explicit_large,
        "b--",
        linewidth=2,
        label=f"Explicit requested dt={dt_large}",
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
print(f"\n{'=' * 70}")
print("NUMERICAL CHECK SUMMARY")
print(f"{'=' * 70}")

print("\nChecked reference and norm:")
print("  Exact first eigenmode of the discrete 1D Laplacian; RMS field error")
print("\nObserved-order criteria over the five stated dt values:")
print(
    "  Explicit: "
    f"p={slope_explicit:.3f}, |p-1|={abs(slope_explicit - 1.0):.3f}, "
    f"required < {ORDER_TOLERANCE:.1f}: "
    f"{'PASS' if explicit_order_passed else 'FAIL'}"
)
print(
    "  Crank-Nicolson: "
    f"p={slope_cn:.3f}, |p-2|={abs(slope_cn - 2.0):.3f}, "
    f"required < {ORDER_TOLERANCE:.1f}: "
    f"{'PASS' if cn_order_passed else 'FAIL'}"
)

print("\nSingle configured CN step criterion:")
print(
    f"  dt={dt_large}, RMS error={error_cn_large:.3e}, "
    f"required < {SINGLE_STEP_RMS_LIMIT:.1e}, final time={solver_cn_large.time():.6f}: "
    f"{'PASS' if cn_single_step_passed else 'FAIL'}"
)

print("\nDescriptive comparisons (not additional acceptance checks):")
print(f"  Explicit out-of-limit request: {explicit_large_status}")
print(
    f"  At dt={dt_values[-1]:.7f}, explicit/CN RMS-error ratio="
    f"{errors_explicit_array[-1] / errors_cn_array[-1]:.3e}"
)
print("  No conclusion is claimed outside the configured problem and sequences.")

checks_passed = explicit_order_passed and cn_order_passed and cn_single_step_passed
print(f"\nOverall declared checks: {'PASS' if checks_passed else 'FAIL'}")

print("\nResults saved to:")
print(f"  {bt.get_result_path('', EXAMPLE_NAME)}")
print(f"{'=' * 70}")

if not checks_passed:
    raise SystemExit(1)
