"""
Adaptive Time-Stepping Example

This example asks whether error-controlled adaptive stepping is worth its
overhead on a simple diffusion problem, and answers it with measured numbers
rather than assertion. We compare:

1. Fixed stepping at the CFL ceiling (the cheapest stable run)
2. Adaptive stepping (error-controlled)
3. Fixed stepping at a step size *matched to the adaptive run's accuracy*

Comparison (3) is the one that decides the question. Comparing an adaptive run
against a 3-step CFL-ceiling run is not a fair fight: the fixed run is cheaper
only because it is far less accurate. Both are measured against a common
reference solution -- the same spatial discretization marched with a very small
fixed step, so that only the *time* discretization differs.

The cost unit is a native diffusion sweep, not an accepted step. Step doubling
takes three sweeps per attempt (one step of dt, plus two of dt/2), and rejected
attempts are paid for too, so an adaptive run of N accepted steps with R
rejections costs 3*(N + R) sweeps.

The test problem is 1D uniform diffusion with a sharp Gaussian initial
condition and homogeneous Dirichlet boundaries. Early in the simulation, the
solution changes rapidly and needs small steps. Later, as it smooths out,
larger steps suffice.

``AdaptiveTimeStepper`` is a deliberately limited Python teaching reference,
not the high-throughput production path. It rejects reactions, advection,
variable diffusivity, multidimensional meshes, and non-Dirichlet boundaries.
Use ``bt.solve`` for the canonical native C++ transport solver.
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
x = bt.x_nodes(mesh)
sigma = 0.02  # Very narrow peak (2 cm width) - needs fine resolution early
u0 = np.exp(-((x - L / 2) ** 2) / (2 * sigma**2))

# Homogeneous Dirichlet boundaries match the Gaussian to machine precision and
# are the boundary semantics implemented by the legacy adaptive reference path.
problem = (
    bt.Problem(mesh)
    .diffusivity(D)
    .initial(u0)
    .dirichlet("left", 0.0)
    .dirichlet("right", 0.0)
)

# Simulation time
t_end = 1.0  # seconds

cfl_limit = problem.stable_time_step(1.0)

print("=" * 60)
print("Adaptive Time-Stepping Demonstration")
print("=" * 60)
print(f"Domain: [0, {L}] m with {nx} cells")
print(f"Diffusivity: D = {D:.2e} m²/s")
print(f"Initial: Sharp Gaussian peak (sigma = {sigma} m)")
print(f"Simulation time: {t_end} s")
print(f"Explicit stability limit: dt <= {cfl_limit:.6e} s")
print()

# =============================================================================
# Reference solution: same spatial operator, time discretization converged
# =============================================================================

print("Building the reference (fixed dt = CFL/8000)...")
reference_run = bt.solve(problem, end_time=t_end, time_step=cfl_limit / 8000.0)
u_reference = reference_run.c
coarser_reference = bt.solve(problem, end_time=t_end, time_step=cfl_limit / 4000.0)
reference_drift = float(np.max(np.abs(coarser_reference.c - u_reference)))
print(f"  Steps: {reference_run.steps}")
print(f"  Change on halving the reference step: {reference_drift:.2e}")
print("  Every error below is measured against this field, so anything")
print(f"  much larger than {reference_drift:.0e} is genuine time-stepping error.")
print()


def error_against_reference(field):
    """Max-norm difference from the time-converged reference."""

    return float(np.max(np.abs(np.asarray(field) - u_reference)))


# =============================================================================
# Method 1: Fixed Time-Stepping at the CFL ceiling (cheapest stable run)
# =============================================================================

print("Running with FIXED time-stepping at the CFL ceiling...")
start = time.perf_counter()
result_fixed = bt.ExplicitFD().run(problem, t_end)
time_fixed = time.perf_counter() - start
u_fixed = np.array(result_fixed.solution())
error_fixed = error_against_reference(u_fixed)

print(f"  Steps: {result_fixed.stats.steps}")
print(f"  dt: {result_fixed.stats.dt:.6e} s")
print(f"  Error vs reference: {error_fixed:.3e}")
print(f"  Wall time: {time_fixed:.4f} s")
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
error_adaptive = error_against_reference(u_adaptive)

stats = result_adaptive.stats
sweeps_adaptive = 3 * (stats["steps"] + stats["rejections"])
print(f"  Steps: {stats['steps']}")
print(f"  Rejections: {stats['rejections']}")
print(
    f"  Native sweeps: 3 x ({stats['steps']} + {stats['rejections']}) "
    f"= {sweeps_adaptive}"
)
print(f"  dt range: [{stats['dt_min_used']:.6e}, {stats['dt_max_used']:.6e}] s")
print(f"  dt average: {stats['dt_avg']:.6e} s")
print(f"  CFL limit: {stats['cfl_limit']:.6e} s")
print(f"  Error vs reference: {error_adaptive:.3e}")
print(f"  Wall time: {time_adaptive:.4f} s")
print()

# =============================================================================
# Method 3: Fixed Time-Stepping matched to the adaptive run's accuracy
# =============================================================================

print(f"Searching for the fixed dt that matches error <= {error_adaptive:.3e}...")


def fixed_run(step_count):
    """Fixed-step run with ``step_count`` uniform steps over ``t_end``."""

    return bt.solve(problem, end_time=t_end, time_step=t_end / step_count)


search_runs = 0
lower = max(2, int(np.ceil(t_end / cfl_limit)))  # smallest stable step count
upper = lower
result_probe = fixed_run(upper)
search_runs += 1
while error_against_reference(result_probe.c) > error_adaptive:
    lower = upper
    upper *= 2
    result_probe = fixed_run(upper)
    search_runs += 1

# Bisect for the smallest step count that reaches the adaptive accuracy.
while upper - lower > 1:
    middle = (lower + upper) // 2
    if error_against_reference(fixed_run(middle).c) <= error_adaptive:
        upper = middle
    else:
        lower = middle
    search_runs += 1

start = time.perf_counter()
result_matched = fixed_run(upper)
time_matched = time.perf_counter() - start
u_matched = result_matched.c
error_matched = error_against_reference(u_matched)
dt_matched = t_end / upper

print(f"  Trial runs used by the search: {search_runs}")
print(f"  Steps: {result_matched.steps}")
print(f"  dt: {dt_matched:.6e} s")
print(f"  Error vs reference: {error_matched:.3e}")
print(f"  Wall time: {time_matched:.4f} s")
print()

# =============================================================================
# Method 4: Tight tolerance (shows the tolerance knob actually works)
# =============================================================================

print("Running with ADAPTIVE (tol=1e-5, high accuracy)...")
start = time.perf_counter()
stepper_tight = bt.AdaptiveTimeStepper(problem, tol=1e-5, verbose=False)
result_tight = stepper_tight.solve(t_end)
time_tight = time.perf_counter() - start
u_tight = result_tight.solution
error_tight = error_against_reference(u_tight)

stats_tight = result_tight.stats
sweeps_tight = 3 * (stats_tight["steps"] + stats_tight["rejections"])
print(f"  Steps: {stats_tight['steps']}")
print(f"  Rejections: {stats_tight['rejections']}")
print(f"  Native sweeps: {sweeps_tight}")
print(
    f"  dt range: [{stats_tight['dt_min_used']:.6e}, {stats_tight['dt_max_used']:.6e}] s"
)
print(f"  Error vs reference: {error_tight:.3e}")
print(f"  Wall time: {time_tight:.4f} s")
print()

# =============================================================================
# Comparison
# =============================================================================

print("=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"{'method':<28} {'sweeps':>8} {'error':>11} {'wall (s)':>10}")
print(
    f"{'fixed, CFL ceiling':<28} {result_fixed.stats.steps:>8} "
    f"{error_fixed:>11.3e} {time_fixed:>10.4f}"
)
print(
    f"{'fixed, accuracy-matched':<28} {result_matched.steps:>8} "
    f"{error_matched:>11.3e} {time_matched:>10.4f}"
)
print(
    f"{'adaptive tol=1e-3':<28} {sweeps_adaptive:>8} "
    f"{error_adaptive:>11.3e} {time_adaptive:>10.4f}"
)
print(
    f"{'adaptive tol=1e-5':<28} {sweeps_tight:>8} "
    f"{error_tight:>11.3e} {time_tight:>10.4f}"
)
print()

cost_ratio = sweeps_adaptive / result_matched.steps
break_even = stats["dt_avg"] / dt_matched

print("What the numbers say:")
print(
    f"  The CFL-ceiling run looks {sweeps_adaptive / result_fixed.stats.steps:.0f}x "
    f"cheaper than adaptive, but it is {error_fixed / error_adaptive:.0f}x less "
    f"accurate."
)
print(
    f"  Matched on accuracy, fixed stepping needs {result_matched.steps} sweeps "
    f"against adaptive's {sweeps_adaptive}."
)
if cost_ratio > 1.0:
    print(
        f"  So adaptive costs {cost_ratio:.2f}x MORE work here, and "
        f"{time_adaptive / time_matched:.0f}x more wall time (the controller runs "
        f"in Python around native steps)."
    )
else:
    print(f"  So adaptive costs {cost_ratio:.2f}x the work of matched fixed stepping.")
print()
print("Why adaptive loses on this problem:")
print(
    "  Step doubling costs 3 sweeps per attempt, so it only pays off when the "
    "average step it finds is more than 3x the fixed step you would otherwise "
    "need."
)
print(
    f"  Here that ratio is only {stats['dt_avg']:.3e} / {dt_matched:.3e} = "
    f"{break_even:.2f}, which is below 3."
)
print(
    f"  The step size varies just "
    f"{stats['dt_max_used'] / stats['dt_min_used']:.0f}x across the run, and the "
    f"CFL ceiling of {cfl_limit:.3g} s caps how large it may grow anyway."
)
print()
print("When adaptive stepping does pay off:")
print(
    "  1. When you do not know the right dt in advance. The matched fixed run "
    f"above took {search_runs} trial solves to find; adaptive hit its tolerance "
    "on the first try."
)
print(
    "  2. When the required step varies by orders of magnitude, not by "
    f"{stats['dt_max_used'] / stats['dt_min_used']:.0f}x -- stiff reactions, "
    "ignition, or a moving front."
)
print(
    "  3. With an implicit integrator, where no CFL ceiling caps step growth "
    "and the smooth tail can be crossed in a handful of steps."
)
print("  4. When you need a certified error bound rather than a plausible answer.")
print("  Production transport remains on the canonical C++ bt.solve path.")
print()

# =============================================================================
# Visualization
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Solutions comparison
ax1 = axes[0, 0]
ax1.plot(x * 100, u0, "k--", label="Initial", linewidth=2)
ax1.plot(x * 100, u_reference, "k-", label="Reference (dt = CFL/8000)", linewidth=2)
ax1.plot(x * 100, u_fixed, "b-", label="Fixed (CFL ceiling)", linewidth=2)
ax1.plot(
    x * 100,
    u_matched,
    "m-.",
    label=f"Fixed (matched, {result_matched.steps} steps)",
    linewidth=2,
)
ax1.plot(x * 100, u_adaptive, "r--", label="Adaptive (tol=1e-3)", linewidth=2)
ax1.set_xlabel("Position (cm)")
ax1.set_ylabel("Concentration")
ax1.set_title(f"Solution Comparison at t = {t_end} s")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Difference from reference
ax2 = axes[0, 1]
ax2.plot(x * 100, np.abs(u_fixed - u_reference), "b-", label="Fixed (CFL ceiling)")
ax2.plot(
    x * 100,
    np.abs(u_matched - u_reference),
    "m-.",
    label=f"Fixed (matched, {result_matched.steps} steps)",
)
ax2.plot(x * 100, np.abs(u_adaptive - u_reference), "r-", label="Adaptive (tol=1e-3)")
ax2.plot(x * 100, np.abs(u_tight - u_reference), "g-", label="Adaptive (tol=1e-5)")
ax2.set_yscale("log")
ax2.set_ylim(bottom=1e-12)
ax2.set_xlabel("Position (cm)")
ax2.set_ylabel("Absolute Difference from Reference")
ax2.set_title("Error vs Time-Converged Reference")
ax2.legend(fontsize=8)
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
    result_fixed.stats.dt,
    color="b",
    linestyle="--",
    linewidth=2,
    label="Fixed (CFL ceiling)",
)
ax3.axhline(
    dt_matched, color="m", linestyle="-.", linewidth=2, label="Fixed (accuracy-matched)"
)
ax3.axhline(stats["cfl_limit"], color="k", linestyle=":", label="CFL stability limit")
ax3.set_xlabel("Simulation Time (s)")
ax3.set_ylabel("Time Step (s)")
ax3.set_title("Time Step Evolution")
ax3.legend(loc="lower right", fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, t_end)

# Plot 4: Cost at comparable accuracy
ax4 = axes[1, 1]
methods = [
    "Fixed\n(CFL ceiling)",
    "Fixed\n(matched)",
    "Adaptive\n(tol=1e-3)",
    "Adaptive\n(tol=1e-5)",
]
sweep_counts = [
    result_fixed.stats.steps,
    result_matched.steps,
    sweeps_adaptive,
    sweeps_tight,
]
errors = [error_fixed, error_matched, error_adaptive, error_tight]
colors = ["blue", "magenta", "red", "green"]
bars = ax4.bar(methods, sweep_counts, color=colors, alpha=0.7, edgecolor="black")
ax4.set_ylabel("Native diffusion sweeps")
ax4.set_title("Computational Cost (labels show error)")

for bar, count, err in zip(bars, sweep_counts, errors):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(sweep_counts) * 0.02,
        f"{count}\nerr {err:.1e}",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )

ax4.set_ylim(0, max(sweep_counts) * 1.25)
ax4.grid(True, alpha=0.3, axis="y")

plt.suptitle("Adaptive vs Fixed Time-Stepping", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(bt.get_result_path("adaptive_timestepping.png", "adaptive"), dpi=150)
plt.show()

print("[OK] Adaptive time-stepping example completed!")
print(
    f"   Plot saved to: {bt.get_result_path('adaptive_timestepping.png', 'adaptive')}"
)
