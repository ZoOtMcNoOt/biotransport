"""
Example of 1D heat conduction in a rod.

This example simulates heat diffusion in a rod with fixed temperatures
at both ends. The thermal diffusivity equation is equivalent to the
diffusion equation, just with different physical interpretation.
"""

import numpy as np
import matplotlib.pyplot as plt
from biotransport import StructuredMesh, DiffusionSolver
from biotransport.utils import get_result_path
from biotransport.visualization import plot_1d_solution

# Create results subdirectory for this example
EXAMPLE_NAME = "heat_conduction"

# Physical parameters
length = 0.1  # Length of rod in meters
T_left = 100   # Temperature at left end (°C)
T_right = 20   # Temperature at right end (°C)
T_initial = 20  # Initial temperature (°C)
thermal_diffusivity = 1e-5  # Thermal diffusivity (m²/s) for typical metal

# Create a 1D mesh
nx = 100
mesh = StructuredMesh(nx, 0.0, length)

# Set up the diffusion solver (using thermal diffusivity)
solver = DiffusionSolver(mesh, thermal_diffusivity)

# Initial condition (uniform temperature)
initial_temperature = np.ones(mesh.num_nodes()) * T_initial
solver.set_initial_condition(initial_temperature)

# Set boundary conditions (fixed temperatures at ends)
solver.set_dirichlet_boundary(0, T_left)   # left boundary
solver.set_dirichlet_boundary(1, T_right)  # right boundary

# Get x coordinates for plotting
x = np.array([mesh.x(i) for i in range(mesh.nx() + 1)])

# Plot initial condition
plt.figure(figsize=(10, 6))
plt.plot(x, initial_temperature, 'b-', label='Initial')
plt.grid(True)
plt.title('Initial Temperature Distribution')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.savefig(get_result_path('initial_temperature.png', EXAMPLE_NAME))

# Solve the heat equation for different time points
# We'll save solutions at multiple times to see the evolution
dt = 0.01  # time step (seconds)
intervals = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]  # seconds
solutions = []

temperature = initial_temperature.copy()
current_time = 0.0

plt.figure(figsize=(12, 8))
plt.plot(x, temperature, 'k--', label=f't = {current_time:.1f}s')

for target_time in intervals:
    # Calculate number of steps needed
    steps_needed = int((target_time - current_time) / dt)
    if steps_needed <= 0:
        continue

    print(f"Simulating from t={current_time:.1f}s to t={target_time:.1f}s...")
    solver.solve(dt, steps_needed)
    current_time = target_time

    # Get and store the solution
    temperature = solver.solution()
    solutions.append((current_time, temperature.copy()))

    # Plot this time point
    plt.plot(x, temperature, label=f't = {current_time:.1f}s')

# Add steady-state analytical solution for comparison
steady_state = T_left + (T_right - T_left) * x / length
plt.plot(x, steady_state, 'r--', linewidth=2, label='Steady State (Analytical)')

plt.grid(True)
plt.title('Heat Conduction in a Rod')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.savefig(get_result_path('temperature_evolution.png', EXAMPLE_NAME))

# Plot the final solution with a different visualization
plt.figure(figsize=(10, 6))
plt.plot(x, temperature, 'b-', linewidth=2, label='Numerical')
plt.plot(x, steady_state, 'r--', linewidth=2, label='Analytical')
plt.grid(True)
plt.title(f'Temperature Distribution at t = {current_time:.1f}s')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.savefig(get_result_path('final_temperature.png', EXAMPLE_NAME))

# Display the plots
plt.show()

results_dir = get_result_path('', EXAMPLE_NAME)
print(f"Simulation complete. Results saved to '{results_dir}'.")