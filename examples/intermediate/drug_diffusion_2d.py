"""
Example of 2D drug diffusion in tissue with VTK export.

This example simulates the diffusion of a drug from a central source
into surrounding tissue, with a first-order reaction term representing
drug metabolism or degradation.

Model:
    dC/dt = D * ∇²C - k * C

where:
- D is the diffusion coefficient
- k is the first-order decay/metabolism rate

This example demonstrates:
1. Time-resolved simulation with multiple snapshots
2. Matplotlib visualization for quick analysis
3. VTK export for advanced ParaView visualization

Notes:
- This example treats the spatial coordinates as centimeters (cm), so `D` is in cm²/s.
- `k` is a first-order decay rate (1/s).

BMEN 341 Reference: Week 3 (Drug Delivery, Reaction-Diffusion)
"""

import matplotlib.pyplot as plt
import numpy as np
import biotransport as bt

EXAMPLE_NAME = "drug_diffusion_2d"

# Create a 2D mesh representing tissue domain
# Coordinates in cm: 2cm x 2cm centered at origin
mesh = bt.mesh_2d(50, 50, x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0)

# Physical parameters
# Units: cm, seconds
D = 1e-6  # Drug diffusivity in tissue (cm²/s)
illustrative_half_life_hours = 24.0
decay_rate = np.log(2.0) / (illustrative_half_life_hours * 3600.0)
# This half-life is an explicit teaching assumption, not a calibrated drug value.

# Initial condition: Gaussian drug bolus in center
# gaussian() creates a symmetric 2D Gaussian centered at (center, center)
initial_condition = bt.gaussian(mesh, center=0.0, width=0.15, amplitude=1.0)

# Setup problem with no-flux boundaries (drug stays in domain)
problem = (
    bt.Problem(mesh)
    .diffusivity(D)
    .linear_decay(decay_rate)
    .initial_condition(initial_condition)
    .neumann(bt.Boundary.Left, 0.0)
    .neumann(bt.Boundary.Right, 0.0)
    .neumann(bt.Boundary.Bottom, 0.0)
    .neumann(bt.Boundary.Top, 0.0)
)

# Plot initial condition
bt.plot(mesh, initial_condition, title="Initial Drug Distribution")
plt.savefig(bt.get_result_path("drug_initial.png", EXAMPLE_NAME))
plt.show()

# ========================================================================
# Time-resolved simulation with multiple snapshots
# ========================================================================
print("=" * 70)
print("Simulating drug diffusion with time snapshots...")
print("=" * 70)

# Define time points to capture (in hours)
snapshot_hours = [0, 2, 6, 12, 24, 48, 72]
snapshot_times = [t * 3600 for t in snapshot_hours]  # Convert to seconds

solutions = []
times_s = []

# Start with initial condition
current_solution = initial_condition.copy()
solutions.append(current_solution.copy())
times_s.append(0.0)

print(f"\nCapturing snapshots at: {snapshot_hours} hours")
print(f"{'Time (h)':>10} {'Min Conc':>15} {'Max Conc':>15} {'Inventory':>15}")
print("-" * 70)

initial_inventory = None
final_inventory = None

for i in range(1, len(snapshot_times)):
    # Time interval for this segment
    dt_segment = snapshot_times[i] - snapshot_times[i - 1]

    # Update problem with current state as initial condition
    problem_segment = (
        bt.Problem(mesh)
        .diffusivity(D)
        .linear_decay(decay_rate)
        .initial_condition(current_solution)
        .neumann(bt.Boundary.Left, 0.0)
        .neumann(bt.Boundary.Right, 0.0)
        .neumann(bt.Boundary.Bottom, 0.0)
        .neumann(bt.Boundary.Top, 0.0)
    )

    # Advance the full diffusion-and-decay model with the verified conservative
    # C++ solver. CrankNicolsonDiffusion is intentionally not used here because
    # that specialized API solves diffusion only and would omit metabolism.
    result = bt.solve(
        problem_segment,
        end_time=dt_segment,
        time_step=100.0,
    )
    current_solution = result.concentration
    if initial_inventory is None:
        initial_inventory = result.diagnostics.initial_mass
    final_inventory = result.diagnostics.final_mass

    # Store snapshot
    solutions.append(current_solution.copy())
    times_s.append(snapshot_times[i])

    # Print statistics
    print(
        f"{snapshot_hours[i]:>10.1f} {current_solution.min():>15.6e} "
        f"{current_solution.max():>15.6e} "
        f"{final_inventory / initial_inventory:>15.3f}"
    )

print(f"\n[OK] Captured {len(solutions)} time snapshots")

# ========================================================================
# VTK Export for ParaView visualization
# ========================================================================
print(f"\n{'=' * 70}")
print("Exporting VTK files for ParaView...")
print(f"{'=' * 70}")

# Export time series
series_prefix = bt.get_result_path("drug_diffusion_series", EXAMPLE_NAME)
time_fields = [
    (time_s, {"drug_concentration": solution})
    for time_s, solution in zip(times_s, solutions)
]
bt.write_vtk_series(mesh, time_fields, series_prefix)

print("\n[OK] VTK time series exported:")
print(f"  Prefix: {series_prefix}")
print(f"  Files: drug_diffusion_series_*.vtk ({len(solutions)} files)")
print("\nTo visualize in ParaView:")
print("  1. Open ParaView")
print("  2. File -> Open -> Select all 'drug_diffusion_series_*.vtk' files")
print("  3. Click 'Apply' in Properties panel")
print("  4. Press 'Play' button to animate")

# Also export final state as single file for quick viewing
final_vtk = bt.get_result_path("drug_final.vtk", EXAMPLE_NAME)
bt.write_vtk(mesh, {"drug_concentration": solutions[-1]}, final_vtk)
print(f"\n[OK] Final snapshot exported: {final_vtk}")

# ========================================================================
# Matplotlib visualization of snapshots
# ========================================================================
print(f"\n{'=' * 70}")
print("Creating matplotlib visualization...")
print(f"{'=' * 70}")

# Create multi-panel plot
n_snapshots = len(solutions)
n_cols = 3
n_rows = (n_snapshots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten() if n_snapshots > 1 else [axes]

for i, (solution, t_s) in enumerate(zip(solutions, times_s)):
    t_h = t_s / 3600
    bt.plot(
        mesh,
        solution,
        ax=axes[i],
        title=f"t = {t_h:.0f} hours\nMax = {solution.max():.4f}",
        colorbar_label="Drug concentration",
    )

# Hide unused subplots
for i in range(len(solutions), len(axes)):
    axes[i].axis("off")

fig.suptitle("Drug Diffusion in Tissue Over Time", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(bt.get_result_path("drug_time_series.png", EXAMPLE_NAME), dpi=150)

# Create 3D surface plot of final state
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection="3d")
bt.plot(
    mesh,
    solutions[-1],
    ax=ax_3d,
    kind="surface",
    title=f"Drug Concentration at {snapshot_hours[-1]} hours",
    zlabel="Drug concentration",
)
plt.tight_layout()
plt.savefig(bt.get_result_path("drug_final_3d.png", EXAMPLE_NAME), dpi=150)

plt.show()

# ========================================================================
# Summary
# ========================================================================
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")

final_solution = solutions[-1]
if initial_inventory is None or final_inventory is None:
    raise RuntimeError("the time integration produced no inventory diagnostics")
retention = final_inventory / initial_inventory * 100

print(f"\nSimulation completed for {snapshot_hours[-1]} hours")
print(f"  Initial peak concentration: {initial_condition.max():.6f}")
print(f"  Final peak concentration: {final_solution.max():.6f}")
print(f"  Drug retention: {retention:.1f}% (first-order metabolism)")
print(f"  Reduction factor: {initial_condition.max() / final_solution.max():.1f}x")

print("\nVisualization files created:")
print("  - Matplotlib time series: drug_time_series.png")
print("  - 3D surface plot: drug_final_3d.png")
print("  - VTK series for ParaView: drug_diffusion_series_*.vtk")
print("  - VTK final snapshot: drug_final.vtk")

print("\nNext steps:")
print("  - Open VTK files in ParaView for interactive 3D visualization")
print("  - Try different decay rates to see metabolism effects")
print("  - Modify initial condition to simulate injection sites")
print("  - Add heterogeneous diffusivity for tissue layers")

print(f"\nAll results saved to: {bt.get_result_path('', EXAMPLE_NAME)}")
print(f"{'=' * 70}")
