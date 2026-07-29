"""Scoped grid- and timestep-convergence evidence for one diffusion case.

The example measures a midpoint quantity of interest and an L2 error against a
known solution.  It exercises Richardson extrapolation, a GCI-style diagnostic,
and explicit observed-order acceptance checks.  Passing these checks supports
only the stated discretization behavior for this problem and refinement
sequence; it does not validate the physical model or verify the whole library.

The calculations use terminology found in numerical-verification practice,
including ASME V&V 20, but this script is not an ASME assessment and makes no
claim of compliance with that standard.

BMEN 341 Reference: Numerical Methods Verification (Week 5)
"""

import matplotlib.pyplot as plt
import numpy as np

import biotransport as bt

EXAMPLE_NAME = "grid_convergence"

print("=" * 70)
print("Grid and Timestep Convergence Evidence - 1D Diffusion")
print("=" * 70)

# =============================================================================
# Problem Setup: 1D Diffusion with Analytical Solution
# =============================================================================

# Physical parameters
L = 1.0  # Domain length [m]
D = 0.01  # Diffusivity [m²/s]
t_end = 0.1  # Simulation time [s]


def analytical_solution(x: np.ndarray, t: float, n_terms: int = 50) -> np.ndarray:
    """Analytical solution for 1D diffusion with sin(πx) initial condition.

    u(x,t) = sin(πx) * exp(-D*π²*t)
    """
    return np.sin(np.pi * x / L) * np.exp(-D * (np.pi / L) ** 2 * t)


def semidiscrete_solution(x: np.ndarray, t: float, spacing: float) -> np.ndarray:
    """Exact first-mode evolution under the centered discrete Laplacian."""
    eigenvalue = -4.0 * D / spacing**2 * np.sin(np.pi * spacing / (2.0 * L)) ** 2
    return np.sin(np.pi * x / L) * np.exp(eigenvalue * t)


print("\nProblem: 1D diffusion with sin(pi*x/L) initial condition")
print(f"  Domain: [0, {L}] m")
print(f"  Diffusivity: D = {D} m^2/s")
print(f"  End time: t = {t_end} s")
print("  Continuum reference: u(x,t) = sin(pi*x/L) * exp(-D*(pi/L)^2*t)")

# =============================================================================
# Part 1: Spatial Convergence Study
# =============================================================================

print(f"\n{'=' * 70}")
print("PART 1: GRID REFINEMENT WITH AN AUTOMATIC EXPLICIT TIMESTEP")
print(f"{'=' * 70}")


def solve_spatial(n: int) -> tuple[float, float]:
    """Solve diffusion problem on mesh with n cells and return (midpoint, L2_error)."""
    mesh = bt.mesh_1d(n, 0.0, L)
    x = bt.x_nodes(mesh)

    # Initial condition
    u0 = np.sin(np.pi * x / L)

    problem = (
        bt.Problem(mesh)
        .diffusivity(D)
        .initial_condition(u0.tolist())
        .dirichlet(bt.Boundary.Left, 0.0)
        .dirichlet(bt.Boundary.Right, 0.0)
    )

    # The canonical C++ solver selects a stable timestep for each mesh.
    result = bt.solve(problem, end_time=t_end)
    u_numerical = np.asarray(result.concentration)
    u_analytical = analytical_solution(x, t_end)

    # L2 error norm
    error = np.sqrt(np.mean((u_numerical - u_analytical) ** 2))

    # Return midpoint value as the quantity of interest
    return u_numerical[n // 2], error


# Run spatial convergence study
n_values = [10, 20, 40, 80, 160]
spatial_result = bt.run_convergence_study(
    solve_func=solve_spatial,
    n_values=n_values,
    # Central space is second order and the selected explicit dt scales as dx^2,
    # so the coupled refinement is expected to approach order two.
    theoretical_order=2.0,
    verbose=True,
)

# =============================================================================
# Part 2: Temporal Convergence Study
# =============================================================================

print(f"\n{'=' * 70}")
print("PART 2: TEMPORAL CONVERGENCE STUDY")
print(f"{'=' * 70}")

# Use fine spatial mesh to minimize spatial error
n_fine = 200
mesh_fine = bt.mesh_1d(n_fine, 0.0, L)
x_fine = bt.x_nodes(mesh_fine)
dx_fine = mesh_fine.dx()
u_semidiscrete_fine = semidiscrete_solution(x_fine, t_end, dx_fine)
print("Temporal-error reference: exact first mode of the discrete Laplacian")


def solve_temporal(dt: float) -> tuple[float, float]:
    """Return midpoint QoI and RMS temporal error for one explicit run."""
    u0 = np.sin(np.pi * x_fine / L)

    problem = (
        bt.Problem(mesh_fine)
        .diffusivity(D)
        .initial_condition(u0.tolist())
        .dirichlet(bt.Boundary.Left, 0.0)
        .dirichlet(bt.Boundary.Right, 0.0)
    )

    # Use bt.solve with explicit method and specified dt
    result = bt.solve(
        problem,
        end_time=t_end,
        time_step=dt,
        method="explicit",
    )
    u_numerical = np.asarray(result.concentration)
    error = np.sqrt(np.mean((u_numerical - u_semidiscrete_fine) ** 2))
    return u_numerical[n_fine // 2], error


# Time steps are fixed fractions of the 1D forward-Euler diffusion limit.
dt_explicit_limit = dx_fine**2 / (2 * D)
dt_values = [dt_explicit_limit * f for f in [0.8, 0.4, 0.2, 0.1, 0.05]]

temporal_result = bt.temporal_convergence_study(
    solve_func=solve_temporal,
    dt_values=dt_values,
    theoretical_order=1.0,  # Forward Euler is 1st order
    verbose=True,
)

# =============================================================================
# Part 3: Crank-Nicolson vs Explicit Comparison
# =============================================================================

print(f"\n{'=' * 70}")
print("PART 3: CRANK-NICOLSON vs EXPLICIT COMPARISON")
print(f"{'=' * 70}")
print("This section reports three same-mesh RMS errors; it has no pass criterion.")
print("It is not an additional temporal-order or general-stability check.")

# Compare errors at same dt
n_compare = 100
mesh_compare = bt.mesh_1d(n_compare, 0.0, L)
x_compare = bt.x_nodes(mesh_compare)
dx_compare = mesh_compare.dx()

# Use half of the 1D explicit diffusion limit so that both configured
# Crank-Nicolson step sizes divide the requested end time exactly.
dt_explicit_limit_compare = dx_compare**2 / (2 * D)
dt_compare = 0.5 * dt_explicit_limit_compare
dt_compare_large = 10.0 * dt_compare

u0_compare = np.sin(np.pi * x_compare / L)
u_semidiscrete_compare = semidiscrete_solution(x_compare, t_end, dx_compare)

problem_compare = (
    bt.Problem(mesh_compare)
    .diffusivity(D)
    .initial_condition(u0_compare.tolist())
    .dirichlet(bt.Boundary.Left, 0.0)
    .dirichlet(bt.Boundary.Right, 0.0)
)

# Explicit at half its 1D diffusion limit.
result_explicit = bt.solve(
    problem_compare,
    end_time=t_end,
    time_step=dt_compare,
    method="explicit",
)
error_explicit = np.sqrt(
    np.mean((np.asarray(result_explicit.concentration) - u_semidiscrete_compare) ** 2)
)


def run_crank_nicolson(dt: float) -> np.ndarray:
    """Run the specialized C++ CN solver to exactly ``t_end``."""
    num_steps = int(round(t_end / dt))
    if not np.isclose(num_steps * dt, t_end, rtol=0.0, atol=1e-14):
        raise ValueError("Crank-Nicolson comparison timestep must divide t_end")

    solver = bt.CrankNicolsonDiffusion(mesh_compare, D)
    solver.set_initial_condition(u0_compare)
    solver.set_dirichlet_boundary(bt.Boundary.Left, 0.0)
    solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)
    solver.solve(dt=dt, num_steps=num_steps)
    return np.asarray(solver.solution())


# CN at the same dt.
solution_cn = run_crank_nicolson(dt_compare)
error_cn = np.sqrt(np.mean((solution_cn - u_semidiscrete_compare) ** 2))

# CN at 10x the comparison dt.
solution_cn_large = run_crank_nicolson(dt_compare_large)
error_cn_large = np.sqrt(np.mean((solution_cn_large - u_semidiscrete_compare) ** 2))

print(f"\nComparison at n={n_compare} mesh points:")
print(f"  Explicit (dt={dt_compare:.6f}): error = {error_explicit:.2e}")
print(f"  CN (dt={dt_compare:.6f}):       error = {error_cn:.2e}")
print(f"  CN (dt={dt_compare_large:.6f}):    error = {error_cn_large:.2e}")
print(
    "  Observed CN-large/explicit error ratio for this case: "
    f"{error_cn_large / error_explicit:.3f}"
)

# =============================================================================
# Visualization
# =============================================================================

print(f"\n{'=' * 70}")
print("Creating convergence-evidence plots...")
print(f"{'=' * 70}")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: coupled grid/timestep refinement
ax1 = axes[0, 0]
h_spatial = 1.0 / np.array(n_values)
if spatial_result.errors is None:
    raise RuntimeError("spatial convergence study did not return an error sequence")
errors_spatial = np.asarray(spatial_result.errors)

ax1.loglog(
    h_spatial, errors_spatial, "bo-", markersize=10, linewidth=2, label="Computed"
)

# Reference lines
h_ref = h_spatial[2]
e_ref = errors_spatial[2]
h_line = np.logspace(
    np.log10(h_spatial.min() / 1.5), np.log10(h_spatial.max() * 1.5), 50
)
e_2nd = e_ref * (h_line / h_ref) ** 2

ax1.loglog(h_line, e_2nd, "k--", alpha=0.5, label="O(h²) reference")
ax1.set_xlabel("Mesh size h = 1/N", fontsize=12)
ax1.set_ylabel("L² Error", fontsize=12)
ax1.set_title(
    "Grid Refinement (explicit dt scales as dx^2)\n"
    f"Observed order: {spatial_result.observed_order:.2f} (comparison: 2.0)",
    fontsize=12,
    fontweight="bold",
)
ax1.legend()
ax1.grid(True, which="both", alpha=0.3)

# Plot 2: Temporal convergence (Explicit)
ax2 = axes[0, 1]
dt_arr = np.array(dt_values)
if temporal_result.errors is None:
    raise RuntimeError("temporal convergence study did not return an error sequence")
errors_temporal = np.asarray(temporal_result.errors)

ax2.loglog(
    dt_arr, errors_temporal, "rs-", markersize=10, linewidth=2, label="Explicit FD"
)

# Reference line
dt_ref = dt_arr[2]
e_ref = errors_temporal[2]
dt_line = np.logspace(np.log10(dt_arr.min() / 1.5), np.log10(dt_arr.max() * 1.5), 50)
e_1st = e_ref * (dt_line / dt_ref) ** 1

ax2.loglog(dt_line, e_1st, "k--", alpha=0.5, label="O(dt) reference")
ax2.set_xlabel("Time step dt [s]", fontsize=12)
ax2.set_ylabel("L² Error", fontsize=12)
ax2.set_title(
    f"Temporal Convergence (Explicit)\nObserved order: {temporal_result.observed_order:.2f} (expected: 1.0)",
    fontsize=12,
    fontweight="bold",
)
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)

# Plot 3: Explicit vs CN comparison
ax3 = axes[1, 0]
methods = ["Explicit\n(CFL dt)", "CN\n(same dt)", "CN\n(10x dt)"]
errors_compare = [error_explicit, error_cn, error_cn_large]
colors = ["blue", "green", "darkgreen"]
bars = ax3.bar(methods, errors_compare, color=colors, alpha=0.7, edgecolor="black")
ax3.set_ylabel("L² Error", fontsize=12)
ax3.set_title(
    "Same-mesh RMS errors for three configured runs",
    fontsize=12,
    fontweight="bold",
)
ax3.set_yscale("log")
ax3.grid(True, axis="y", alpha=0.3)

# Add value labels
for bar, err in zip(bars, errors_compare):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        err * 1.5,
        f"{err:.2e}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

# Plot 4: Summary table
ax4 = axes[1, 1]
ax4.axis("off")

summary_text = f"""
NUMERICAL EVIDENCE SUMMARY
===========================================================

GRID REFINEMENT (central space; automatic explicit dt)
  * Observed order:     {spatial_result.observed_order:.3f}
  * Comparison order:   {spatial_result.theoretical_order:.1f}
  * Richardson estimate: {spatial_result.richardson_estimate:.8f}
  * GCI-style (fine):   {spatial_result.gci_fine:.3e}
  * In asymptotic range: {"Yes" if spatial_result.is_asymptotic else "No"}
  * Order criterion:    {"PASS" if abs(spatial_result.observed_order - 2.0) < 0.3 else "FAIL"}
                        |p - 2| < 0.3

TEMPORAL CONVERGENCE (Explicit Euler)
  * Observed order:     {temporal_result.observed_order:.3f}
  * Comparison order:   {temporal_result.theoretical_order:.1f}
  * GCI-style (fine):   {temporal_result.gci_fine:.3e}
  * Order criterion:    {"PASS" if abs(temporal_result.observed_order - 1.0) < 0.3 else "FAIL"}
                        |p - 1| < 0.3

CRANK-NICOLSON vs EXPLICIT
  * Explicit error:     {error_explicit:.2e}
  * CN (same dt):       {error_cn:.2e}
  * CN (10x dt):        {error_cn_large:.2e}
  * Descriptive comparison only; no acceptance criterion

===========================================================
"""

ax4.text(
    0.05,
    0.95,
    summary_text,
    transform=ax4.transAxes,
    fontsize=10,
    fontfamily="monospace",
    verticalalignment="top",
)

plt.suptitle("Grid and Timestep Convergence Evidence", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(bt.get_result_path("grid_convergence.png", EXAMPLE_NAME), dpi=150)
plt.show()

# =============================================================================
# Final Summary
# =============================================================================

print(f"\n{'=' * 70}")
print("FINAL NUMERICAL CHECKS")
print(f"{'=' * 70}")

all_passed = True

print("\nGrid-refinement midpoint QoI (space and explicit dt co-refined):")
print(f"   Observed: {spatial_result.observed_order:.3f}, comparison value: 2.0")
if abs(spatial_result.observed_order - 2.0) < 0.3:
    print("   [PASS] |p_observed - 2.0| < 0.3 for this grid sequence")
else:
    print(f"   [FAIL] Order deviation: {abs(spatial_result.observed_order - 2.0):.3f}")
    all_passed = False

print("\nTemporal midpoint-QoI convergence (explicit Euler):")
print(f"   Observed: {temporal_result.observed_order:.3f}, comparison value: 1.0")
if abs(temporal_result.observed_order - 1.0) < 0.3:
    print("   [PASS] |p_observed - 1.0| < 0.3 for this timestep sequence")
else:
    print(f"   [FAIL] Order deviation: {abs(temporal_result.observed_order - 1.0):.3f}")
    all_passed = False

print("\nCrank-Nicolson comparison:")
print("   This plot is a method comparison, not an additional verification claim.")
print("   Temporal accuracy must be judged from the convergence sequence above.")

print(f"\n{'=' * 70}")
if all_passed:
    print("[PASS] The two stated observed-order criteria passed")
else:
    print("[FAIL] One or more stated observed-order criteria failed")
print(f"{'=' * 70}")

print(f"\nPlot saved to: {bt.get_result_path('grid_convergence.png', EXAMPLE_NAME)}")

if not all_passed:
    raise SystemExit(1)
