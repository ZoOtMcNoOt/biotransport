"""
Example of oxygen diffusion and consumption in tissue.

This example simulates oxygen diffusion from blood vessels into surrounding tissue,
with oxygen consumption by the tissue. This is a classic biotransport problem
representative of oxygen delivery in biological tissues.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from biotransport import StructuredMesh, ReactionDiffusionSolver
from biotransport.utils import get_result_path
from biotransport.visualization import plot_2d_solution, plot_2d_surface

# Create results subdirectory for this example
EXAMPLE_NAME = "oxygen_diffusion"

# Physical parameters
D_oxygen = 2e-9  # Oxygen diffusion coefficient in tissue (m²/s)
M_oxygen = 1e-3  # Oxygen consumption rate (kg/(m³·s))
C_blood = 1.0    # Normalized oxygen concentration in blood vessels

# Michaelis-Menten kinetics for oxygen consumption
# R = -M * C / (C + K_m)
# where K_m is the concentration at half-maximal consumption rate
K_m = 0.1  # Michaelis constant (normalized)

# Define oxygen consumption as a reaction function (Michaelis-Menten kinetics)
def oxygen_consumption(C, x, y, t):
    """
    Oxygen consumption by tissue following Michaelis-Menten kinetics.

    Args:
        C: Oxygen concentration
        x, y: Position coordinates
        t: Time

    Returns:
        Consumption rate (negative for consumption)
    """
    # Check if we're in a blood vessel
    for vessel_x, vessel_y, vessel_radius in blood_vessels:
        dist = np.sqrt((x - vessel_x)**2 + (y - vessel_y)**2)
        if dist < vessel_radius:
            return 0.0  # No consumption inside blood vessels

    # Michaelis-Menten kinetics
    return -M_oxygen * C / (C + K_m)

# Create a 2D mesh representing tissue domain
nx, ny = 100, 100
Lx, Ly = 1e-3, 1e-3  # Domain size in meters (1mm x 1mm)
mesh = StructuredMesh(nx, ny, 0.0, Lx, 0.0, Ly)

# Define blood vessel locations (position_x, position_y, radius)
blood_vessels = [
    (0.25e-3, 0.25e-3, 0.1e-3),  # Bottom left vessel
    (0.75e-3, 0.75e-3, 0.1e-3),  # Top right vessel
    (0.25e-3, 0.75e-3, 0.08e-3), # Top left vessel
    (0.75e-3, 0.25e-3, 0.08e-3)  # Bottom right vessel
]

# Create reaction-diffusion solver
solver = ReactionDiffusionSolver(mesh, D_oxygen, oxygen_consumption)

# Set up initial condition (zero oxygen everywhere except blood vessels)
initial_condition = np.zeros(mesh.num_nodes())
for j in range(mesh.ny() + 1):
    for i in range(mesh.nx() + 1):
        x = mesh.x(i)
        y = mesh.y(i, j)
        idx = mesh.index(i, j)

        # Set initial oxygen concentration in blood vessels
        for vessel_x, vessel_y, vessel_radius in blood_vessels:
            dist = np.sqrt((x - vessel_x)**2 + (y - vessel_y)**2)
            if dist < vessel_radius:
                initial_condition[idx] = C_blood
                break

solver.set_initial_condition(initial_condition)

# Set boundary conditions (no oxygen flux at boundaries)
solver.set_neumann_boundary(0, 0.0)  # left
solver.set_neumann_boundary(1, 0.0)  # right
solver.set_neumann_boundary(2, 0.0)  # bottom
solver.set_neumann_boundary(3, 0.0)  # top

# Create a custom plot with blood vessels marked
def plot_with_vessels(mesh, solution, title=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create grid for plotting
    x = np.array([mesh.x(i) for i in range(mesh.nx() + 1)])
    y = np.array([mesh.y(0, j) for j in range(mesh.ny() + 1)])
    X, Y = np.meshgrid(x, y)

    # Reshape solution to 2D array
    Z = np.zeros((mesh.ny() + 1, mesh.nx() + 1))
    for j in range(mesh.ny() + 1):
        for i in range(mesh.nx() + 1):
            idx = mesh.index(i, j)
            Z[j, i] = solution[idx]

    # Plot oxygen concentration
    contour = ax.contourf(X*1e3, Y*1e3, Z, 50, cmap='viridis')
    plt.colorbar(contour, label='Oxygen Concentration (normalized)')

    # Mark blood vessels with circles
    for vessel_x, vessel_y, vessel_radius in blood_vessels:
        circle = patches.Circle((vessel_x*1e3, vessel_y*1e3), vessel_radius*1e3,
                                fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(circle)

    if title:
        ax.set_title(title)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')

    return fig

# Plot initial condition
initial_fig = plot_with_vessels(mesh, initial_condition,
                                title='Initial Oxygen Distribution')
initial_fig.savefig(get_result_path('initial_oxygen.png', EXAMPLE_NAME))

# Solve the diffusion equation
dt = 0.01  # time step (seconds)
time_points = [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]  # seconds
current_time = 0.0

# Store intermediate results
solutions = [(current_time, initial_condition.copy())]

for target_time in time_points:
    steps_needed = int((target_time - current_time) / dt)
    if steps_needed <= 0:
        continue

    print(f"Simulating from t={current_time:.1f}s to t={target_time:.1f}s...")
    solver.solve(dt, steps_needed)
    current_time = target_time

    # Get and store the solution
    solution = solver.solution()
    solutions.append((current_time, solution.copy()))

    # Plot at specific time points
    if target_time in [0.5, 5.0, 60.0]:
        fig = plot_with_vessels(mesh, solution,
                                title=f'Oxygen Concentration at t = {current_time:.1f}s')
        fig.savefig(get_result_path(f'oxygen_t{current_time:.1f}s.png', EXAMPLE_NAME))

# Get the final steady-state solution
solution = solver.solution()

# Plot the final steady-state solution
final_fig = plot_with_vessels(mesh, solution,
                              title=f'Steady-State Oxygen Distribution (t = {current_time:.1f}s)')
final_fig.savefig(get_result_path('oxygen_steady_state.png', EXAMPLE_NAME))

# Create 3D surface plot of the steady-state solution
surface_fig = plot_2d_surface(mesh, solution,
                              title=f'Oxygen Concentration Profile (t = {current_time:.1f}s)',
                              zlabel='Concentration')
surface_fig.savefig(get_result_path('oxygen_3d.png', EXAMPLE_NAME))

# Plot a cross-section through the middle of the domain
mid_y = ny // 2
mid_y_values = np.array([solution[mesh.index(i, mid_y)] for i in range(nx + 1)])
x_values = np.array([mesh.x(i) for i in range(nx + 1)])

plt.figure(figsize=(10, 6))
plt.plot(x_values*1e3, mid_y_values, 'b-', linewidth=2)
plt.grid(True)
plt.title(f'Oxygen Concentration at y = {mesh.y(0, mid_y)*1e3:.2f} mm')
plt.xlabel('X (mm)')
plt.ylabel('Oxygen Concentration')
plt.savefig(get_result_path('oxygen_cross_section.png', EXAMPLE_NAME))

# Show all plots
plt.show()

results_dir = get_result_path('', EXAMPLE_NAME)
print(f"Simulation complete. Results saved to '{results_dir}'.")