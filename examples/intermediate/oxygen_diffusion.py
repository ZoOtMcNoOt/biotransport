"""
Example of oxygen diffusion and consumption in tissue.

This example simulates oxygen diffusion from blood vessels into surrounding tissue,
with oxygen consumption by the tissue. This is a classic biotransport problem
representative of oxygen delivery in biological tissues.

Notes:
- Concentration is normalized (dimensionless) in this demo.
- Consumption uses Michaelis-Menten kinetics with an effective rate scale (1/s).

BMEN 341 Reference: Weeks 7-8 (Oxygen Transport, Michaelis-Menten Kinetics)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import biotransport as bt

# Create results subdirectory for this example
EXAMPLE_NAME = "oxygen_diffusion"

# Physical / model parameters
# This example uses *normalized* oxygen concentration (dimensionless). The sink term is
# modeled with Michaelis-Menten kinetics using an effective rate scale (1/s).
D_oxygen = 2e-9  # Oxygen diffusion coefficient in tissue (m²/s)
M_oxygen = 2e-2  # Max consumption rate scale (1/s) in normalized units
C_blood = 1.0  # Normalized oxygen concentration in blood vessels

# Michaelis-Menten kinetics for oxygen consumption
# R = -M * C / (C + K_m)
# where K_m is the concentration at half-maximal consumption rate
K_m = 0.1  # Michaelis constant (normalized)

# Create a 2D mesh representing tissue domain
# Coarser grid reduces expensive reaction-callback overhead substantially while
# still resolving ~0.08–0.10 mm vessels in a 1 mm domain.
nx, ny = 50, 50
Lx, Ly = 1e-3, 1e-3  # Domain size in meters (1mm x 1mm)
mesh = bt.mesh_2d(nx, ny, x_max=Lx, y_max=Ly)

# Define blood vessel locations (position_x, position_y, radius)
blood_vessels = [
    (0.25e-3, 0.25e-3, 0.1e-3),  # Bottom left vessel
    (0.75e-3, 0.75e-3, 0.1e-3),  # Top right vessel
    (0.25e-3, 0.75e-3, 0.08e-3),  # Top left vessel
    (0.75e-3, 0.25e-3, 0.08e-3),  # Bottom right vessel
]

# Vessel mask on nodes
x = bt.x_nodes(mesh)
y = bt.y_nodes(mesh)
Xn, Yn = bt.xy_grid(mesh)
vessel_mask = np.logical_or.reduce(
    [((Xn - vx) ** 2 + (Yn - vy) ** 2) < (r**2) for vx, vy, r in blood_vessels]
)

# Create masked Michaelis–Menten solver with pinned vessel concentration
mask_flat = vessel_mask.astype(np.uint8).ravel(order="C").tolist()
solver = bt.MaskedMichaelisMentenReactionDiffusionSolver(
    mesh, D_oxygen, M_oxygen, K_m, mask_flat, C_blood
)

# Initial condition: pinned vessels at C_blood
initial_condition_2d = np.zeros((ny + 1, nx + 1), dtype=np.float64)
initial_condition_2d[vessel_mask] = C_blood
initial_condition = initial_condition_2d.ravel(order="C")
solver.set_initial_condition(initial_condition.tolist())

# Set boundary conditions (no oxygen flux at boundaries)
solver.set_boundary(bt.Boundary.Left, bt.BoundaryCondition.neumann(0.0))
solver.set_boundary(bt.Boundary.Right, bt.BoundaryCondition.neumann(0.0))
solver.set_boundary(bt.Boundary.Bottom, bt.BoundaryCondition.neumann(0.0))
solver.set_boundary(bt.Boundary.Top, bt.BoundaryCondition.neumann(0.0))


# Create a custom plot with blood vessels marked
def plot_with_vessels(solution_flat, title):
    Z = bt.as_2d(mesh, solution_flat)
    fig, ax = plt.subplots(figsize=(10, 8))
    cf = ax.contourf(Xn * 1e3, Yn * 1e3, Z, 50, cmap="viridis")
    plt.colorbar(cf, label="Oxygen Concentration (normalized)")
    for vx, vy, r in blood_vessels:
        ax.add_patch(
            patches.Circle(
                (vx * 1e3, vy * 1e3), r * 1e3, fill=False, edgecolor="red", linewidth=2
            )
        )
    ax.set(title=title, xlabel="X (mm)", ylabel="Y (mm)")
    return fig


# Plot initial condition
initial_fig = plot_with_vessels(initial_condition, "Initial Oxygen Distribution")
initial_fig.savefig(bt.get_result_path("initial_oxygen.png", EXAMPLE_NAME))

# Solve the diffusion equation
# Explicit stability guideline: dt <= dx^2 / (4 D)
dt_max = (mesh.dx() * mesh.dx()) / (4.0 * D_oxygen)
dt = min(0.04, 0.8 * dt_max)
time_points = [0.2, 1.0, 5.0, 10.0, 30.0, 60.0]  # seconds
current_time = 0.0

for target_time in time_points:
    steps_needed = int((target_time - current_time) / dt)
    if steps_needed <= 0:
        continue

    print(f"Simulating from t={current_time:.1f}s to t={target_time:.1f}s...")
    solver.solve(dt, steps_needed)
    current_time = target_time

    if target_time in (5.0, 60.0):
        plot_with_vessels(
            solver.solution(),
            f"Oxygen Concentration at t = {current_time:.1f}s",
        ).savefig(bt.get_result_path(f"oxygen_t{current_time:.1f}s.png", EXAMPLE_NAME))

# Get the final steady-state solution
solution = np.asarray(solver.solution(), dtype=np.float64)

# Plot the final steady-state solution
final_fig = plot_with_vessels(
    solution, f"Steady-State Oxygen Distribution (t = {current_time:.1f}s)"
)
final_fig.savefig(bt.get_result_path("oxygen_steady_state.png", EXAMPLE_NAME))

# Create 3D surface plot of the steady-state solution
surface_fig = bt.plot_field(
    mesh,
    solution,
    title=f"Oxygen Concentration Profile (t = {current_time:.1f}s)",
    kind="surface",
    zlabel="Concentration",
)
surface_fig.savefig(bt.get_result_path("oxygen_3d.png", EXAMPLE_NAME))

# Plot a cross-section through the middle of the domain
mid_y = ny // 2
Z = bt.as_2d(mesh, solution)
mid_y_values = Z[mid_y, :]
x_values = x

plt.figure(figsize=(10, 6))
plt.plot(x_values * 1e3, mid_y_values, "b-", linewidth=2)
plt.grid(True)
plt.title(f"Oxygen Concentration at y = {mesh.y(0, mid_y) * 1e3:.2f} mm")
plt.xlabel("X (mm)")
plt.ylabel("Oxygen Concentration")
plt.savefig(bt.get_result_path("oxygen_cross_section.png", EXAMPLE_NAME))

# Show all plots
plt.show()

results_dir = bt.get_result_path("", EXAMPLE_NAME)
print(f"Simulation complete. Results saved to '{results_dir}'.")
