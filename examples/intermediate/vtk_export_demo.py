"""
Example: VTK Export for ParaView Visualization

This example demonstrates how to export simulation results to VTK format
for visualization in ParaView. We solve a 2D diffusion problem and export
both single snapshots and time series animations showing how an initial
concentration pattern diffuses over time.

VTK (Visualization Toolkit) files can be opened in ParaView, a powerful
open-source visualization tool that provides:
- Interactive 3D visualization
- Advanced rendering and colormaps
- Animation capabilities
- Quantitative analysis tools

Two export methods are demonstrated:
1. write_vtk() - Export a single timestep snapshot
2. write_vtk_series() - Export a time series for animation

BMEN 341 Reference: Data Visualization and Analysis (Week 8)
"""

import matplotlib.pyplot as plt
import numpy as np
import biotransport as bt
import os

EXAMPLE_NAME = "vtk_export_demo"

print("=" * 70)
print("VTK Export Demonstration for ParaView")
print("=" * 70)

# ========================================================================
# Example 1: 2D Diffusion with Initial Pattern
# ========================================================================
print("\nSetting up 2D diffusion problem...")

# Create a fine 2D mesh for interesting patterns
mesh = bt.mesh_2d(100, 100, x_min=0.0, x_max=2.5, y_min=0.0, y_max=2.5)

# Diffusion parameters
D = 2e-5  # Diffusivity

# mesh.y() takes (i, j), so mesh.y(ny) would read column ny of row 0 and
# report 0.0. Query the corners explicitly to get the real extents.
x_min, x_max = mesh.x(0), mesh.x(mesh.nx())
y_min, y_max = mesh.y(0, 0), mesh.y(0, mesh.ny())
extent = [x_min, x_max, y_min, y_max]

print(f"Mesh: {mesh.nx() + 1} x {mesh.ny() + 1} nodes")
print(f"Domain: x in [{x_min:.3f}, {x_max:.3f}], y in [{y_min:.3f}, {y_max:.3f}]")
print(f"Extents: {x_max - x_min:.3f} x {y_max - y_min:.3f}")
print(f"Cell size: dx = {mesh.dx():.4f}, dy = {mesh.dy():.4f}")

# Initial condition: Background with localized spots
u0 = np.ones(mesh.num_nodes())

# Add localized spots that will diffuse over time
nx, ny = mesh.nx() + 1, mesh.ny() + 1
for i in range(3):
    for j in range(3):
        center_x = 0.5 + i * 0.75
        center_y = 0.5 + j * 0.75
        spot = np.array(
            bt.gaussian(mesh, center=center_x, width=0.1, center_y=center_y)
        )
        u0 -= 0.5 * spot  # Create spots by lowering concentration

# Setup diffusion problem
# The initial pattern will diffuse toward a uniform state over time

problem = (
    bt.Problem(mesh)
    .diffusivity(D)
    .initial_condition(u0)
    .neumann(bt.Boundary.Left, 0.0)
    .neumann(bt.Boundary.Right, 0.0)
    .neumann(bt.Boundary.Bottom, 0.0)
    .neumann(bt.Boundary.Top, 0.0)
)

# ========================================================================
# Simulate and capture multiple time points
# ========================================================================
print("\nSimulating diffusion of initial pattern...")

time_points = [0.0, 500.0, 1000.0, 2000.0, 4000.0]  # Time points to save
solutions = []
times = []

current_solution = u0.copy()
for i, t_target in enumerate(time_points):
    if i == 0:
        solutions.append(current_solution.copy())
        times.append(0.0)
    else:
        dt_segment = t_target - time_points[i - 1]

        # Update problem with current state
        problem = (
            bt.Problem(mesh)
            .diffusivity(D)
            .initial_condition(current_solution)
            .neumann(bt.Boundary.Left, 0.0)
            .neumann(bt.Boundary.Right, 0.0)
            .neumann(bt.Boundary.Bottom, 0.0)
            .neumann(bt.Boundary.Top, 0.0)
        )

        # This demonstration uses the canonical conservative C++ diffusion
        # solver and lets it choose a certified explicit step automatically.
        result = bt.solve(problem, end_time=dt_segment)
        current_solution = result.concentration
        solutions.append(current_solution.copy())
        times.append(t_target)

    print(
        f"  Time = {t_target:7.1f}: Min = {current_solution.min():.4f}, "
        f"Max = {current_solution.max():.4f}, "
        f"Range = {np.ptp(current_solution):.4f}, "
        f"Std = {current_solution.std():.4f}"
    )

# ========================================================================
# VTK Export Method 1: Single snapshot
# ========================================================================
print(f"\n{'=' * 70}")
print("VTK Export Method 1: Single Snapshot")
print(f"{'=' * 70}")

# Export the snapshot that still carries the pattern. By t = 4000 the spots
# have merged into a near-uniform field, so exporting the last frame would
# ship a picture with almost nothing in it. Pick the latest snapshot that
# still holds at least 25% of the initial spatial contrast.
initial_range = float(np.ptp(solutions[0]))
contrast_floor = 0.25 * initial_range
export_idx = max(
    i for i, sol in enumerate(solutions) if float(np.ptp(sol)) >= contrast_floor
)
export_solution = solutions[export_idx]
export_time = times[export_idx]

print(f"\nInitial spatial contrast (max - min): {initial_range:.4f}")
print(f"Contrast floor for export (25% of initial): {contrast_floor:.4f}")
print(
    f"Final frame t = {times[-1]:.0f} contrast: {np.ptp(solutions[-1]):.4f} (too flat to export)"
)
print(f"Selected snapshot: t = {export_time:.0f}")
print(
    f"  Min = {export_solution.min():.4f}, Max = {export_solution.max():.4f}, "
    f"Range = {np.ptp(export_solution):.4f}, Std = {export_solution.std():.4f}"
)

vtk_single_path = bt.get_result_path(
    f"pattern_t{int(export_time):05d}.vtk", EXAMPLE_NAME
)
bt.write_vtk(mesh, {"concentration": export_solution}, vtk_single_path)

print("\n[OK] Single snapshot exported to:")
print(f"  {vtk_single_path}")
print("\nTo visualize in ParaView:")
print("  1. Open ParaView")
print(f"  2. File -> Open -> Select 'pattern_t{int(export_time):05d}.vtk'")
print("  3. Click 'Apply' in the Properties panel")
print("  4. Use 'Surface' or 'Surface with Edges' representation")

# ========================================================================
# VTK Export Method 2: Time series for animation
# ========================================================================
print(f"\n{'=' * 70}")
print("VTK Export Method 2: Time Series Animation")
print(f"{'=' * 70}")

# Export all time points as a series
series_prefix = bt.get_result_path("pattern_series", EXAMPLE_NAME)

# Create list of (time, fields_dict) tuples
time_fields = [(t, {"concentration": sol}) for t, sol in zip(times, solutions)]
bt.write_vtk_series(mesh, time_fields, series_prefix)

print("\n[OK] Time series exported to:")
print(f"  {os.path.dirname(series_prefix)}")
print(f"  Files: pattern_series_*.vtk ({len(solutions)} files)")
print("\nTo animate in ParaView:")
print("  1. Open ParaView")
print("  2. File -> Open -> Select 'pattern_series.pvd' (the collection file)")
print("  3. Click 'Apply'")
print("  4. Use the animation controls (play button) to animate")
print("  5. Optional: File -> Save Animation to create video")

# ========================================================================
# Also create matplotlib visualizations for comparison
# ========================================================================
print(f"\n{'=' * 70}")
print("Creating matplotlib comparison plots...")
print(f"{'=' * 70}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    "2D Diffusion: Pattern Spreading Over Time", fontsize=16, fontweight="bold"
)

for idx, ax in enumerate(axes.flat):
    if idx < len(solutions):
        # Manual 2D plot since bt.plot doesn't support all kwargs
        nx, ny = mesh.nx() + 1, mesh.ny() + 1
        Z = solutions[idx].reshape((ny, nx))
        im = ax.imshow(Z, origin="lower", cmap="viridis", aspect="equal", extent=extent)
        ax.set_title(f"t = {times[idx]:.0f} (range {np.ptp(Z):.3f})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax)
    else:
        ax.axis("off")

plt.tight_layout()
plt.savefig(
    bt.get_result_path("pattern_evolution_matplotlib.png", EXAMPLE_NAME), dpi=150
)
plt.close()

# Create individual high-quality matplotlib plots
for i, (solution, t) in enumerate(zip(solutions, times)):
    fig, ax = plt.subplots(figsize=(8, 7))
    nx, ny = mesh.nx() + 1, mesh.ny() + 1
    Z = solution.reshape((ny, nx))
    im = ax.imshow(Z, origin="lower", cmap="viridis", aspect="equal", extent=extent)
    ax.set_title(f"Concentration at t = {t:.0f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(
        bt.get_result_path(f"snapshot_t{int(t):05d}.png", EXAMPLE_NAME), dpi=150
    )
    plt.close()

print(f"  [OK] Matplotlib plots saved ({len(solutions)} snapshots)")

# Note: plt.show() commented out for non-interactive execution
# Uncomment if you want to display plots interactively:
# plt.show()

# ========================================================================
# Export additional fields (example with multiple variables)
# ========================================================================
print(f"\n{'=' * 70}")
print("Advanced: Exporting Multiple Fields")
print(f"{'=' * 70}")

# Calculate derived quantities from the same snapshot that was exported, so
# the gradient field describes a field that still has structure.
gradient_magnitude = np.zeros_like(export_solution)

# Simple gradient approximation
for j in range(1, mesh.ny()):
    for i in range(1, mesh.nx()):
        idx = mesh.index(i, j)
        idx_w = mesh.index(i - 1, j)
        idx_s = mesh.index(i, j - 1)

        dx_val = (export_solution[idx] - export_solution[idx_w]) / mesh.dx()
        dy_val = (export_solution[idx] - export_solution[idx_s]) / mesh.dy()
        gradient_magnitude[idx] = np.sqrt(dx_val**2 + dy_val**2)

# Export with multiple fields (requires custom VTK writing - placeholder example)
print("\nFor multi-field export, you can:")
print("  1. Export each field separately with different filenames")
print("  2. Use bt.write_vtk() multiple times with different field names in the dict")
print("  3. In ParaView, load all files and compare side-by-side")

vtk_grad_path = bt.get_result_path("gradient_magnitude.vtk", EXAMPLE_NAME)
bt.write_vtk(mesh, {"gradient": gradient_magnitude}, vtk_grad_path)

print("\n[OK] Gradient field exported to:")
print(f"  {vtk_grad_path}")
print(
    f"  (gradient of the t = {export_time:.0f} snapshot) "
    f"Min = {gradient_magnitude.min():.4f}, Max = {gradient_magnitude.max():.4f}, "
    f"Std = {gradient_magnitude.std():.4f}"
)

# ========================================================================
# Summary and Tips
# ========================================================================
print(f"\n{'=' * 70}")
print("SUMMARY: VTK Export Best Practices")
print(f"{'=' * 70}")

print("\nWhen to use VTK export:")
print("  - 2D and 3D problems requiring detailed spatial visualization")
print("  - Time series animations of dynamic processes")
print("  - Publishing-quality figures with advanced rendering")
print("  - Quantitative analysis (line plots, volume integration, etc.)")

print("\nWhen to use matplotlib:")
print("  - Quick exploratory analysis during development")
print("  - 1D problems and simple 2D plots")
print("  - Direct integration with Python analysis workflow")
print("  - Subplots and multi-panel figures")

print("\nParaView Tips:")
print("  - Use 'Filters -> Warp By Scalar' for 3D surface plots of 2D data")
print("  - 'Filters -> Contour' to extract isolines/isosurfaces")
print("  - 'Filters -> Calculator' to compute derived quantities")
print("  - 'Tools -> Python Shell' for scripting and batch processing")
print("  - Save state (.pvsm) to preserve visualization settings")

print("\nAll results saved to:")
print(f"  {bt.get_result_path('', EXAMPLE_NAME)}")

print(f"\n{'=' * 70}")
print("Example complete!")
print(f"{'=' * 70}")
