"""
Example of 1D diffusion simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from biotransport import StructuredMesh, DiffusionSolver
from biotransport.visualization import plot_1d_solution
from biotransport.utils import get_result_path

# Create results subdirectory for this example
EXAMPLE_NAME = "1d_diffusion"

# Create a 1D mesh
nx = 100
mesh = StructuredMesh(nx, 0.0, 1.0)

# Set up the diffusion solver
D = 0.01  # diffusion coefficient
solver = DiffusionSolver(mesh, D)

# Initial condition (Gaussian pulse)
x = np.array([mesh.x(i) for i in range(mesh.nx() + 1)])
initial_condition = np.exp(-100 * (x - 0.5)**2)
solver.set_initial_condition(initial_condition)

# Set boundary conditions
solver.set_dirichlet_boundary(0, 0.0)  # left boundary
solver.set_dirichlet_boundary(1, 0.0)  # right boundary

# Plot initial condition
plt.figure(figsize=(10, 6))
plt.plot(x, initial_condition, 'b-', label='Initial')
plt.grid(True)
plt.title('Initial Condition')
plt.xlabel('Position')
plt.ylabel('Concentration')
plt.savefig(get_result_path('initial_condition.png', EXAMPLE_NAME))

# Solve the diffusion equation
dt = 0.0001  # time step
num_steps = 1000
total_time = dt * num_steps

print(f"Solving 1D diffusion for {total_time} time units...")
solver.solve(dt, num_steps)

# Get the solution
solution = solver.solution()

# Plot the solution
plot_1d_solution(mesh, solution,
                 title=f'Diffusion after t = {total_time}',
                 xlabel='Position',
                 ylabel='Concentration')
plt.plot(x, initial_condition, 'b--', alpha=0.5, label='Initial')
plt.legend()
plt.savefig(get_result_path('diffusion_result.png', EXAMPLE_NAME))
plt.show()

results_dir = get_result_path('', EXAMPLE_NAME)
print(f"Simulation complete. Results saved to '{results_dir}'.")