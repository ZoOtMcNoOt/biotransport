"""
Example of 2D drug diffusion in tissue.

This example simulates the diffusion of a drug from a central source
into surrounding tissue, with a first-order reaction term representing
drug metabolism or degradation.
"""

import numpy as np
import matplotlib.pyplot as plt
from biotransport import StructuredMesh, ReactionDiffusionSolver, BoundaryType
from biotransport.visualization import plot_2d_solution, plot_2d_surface

# Create a 2D mesh representing tissue domain
nx, ny = 50, 50
mesh = StructuredMesh(nx, ny, -1.0, 1.0, -1.0, 1.0)

# Diffusion coefficient (cm²/s)
D = 1e-6

# Define reaction function (first-order degradation)
# k * C where k is decay rate (1/s)
decay_rate = 1e-3

def reaction(u, x, y, t):
    return -decay_rate * u

# Create reaction-diffusion solver
solver = ReactionDiffusionSolver(mesh, D, reaction)

# Set up initial condition (drug concentrated in center)
initial_condition = np.zeros(mesh.num_nodes())
for j in range(mesh.ny() + 1):
    for i in range(mesh.nx() + 1):
        x = mesh.x(i)
        y = mesh.y(i, j)
        r = np.sqrt(x*x + y*y)  # Distance from center
        idx = mesh.index(i, j)
        
        # Gaussian initial distribution
        if r < 0.2:
            initial_condition[idx] = 1.0  # Normalized concentration

solver.set_initial_condition(initial_condition)

# Set boundary conditions (zero flux at boundaries)
solver.set_neumann_boundary(0, 0.0)  # left
solver.set_neumann_boundary(1, 0.0)  # right
solver.set_neumann_boundary(2, 0.0)  # bottom
solver.set_neumann_boundary(3, 0.0)  # top

# Plot initial condition
initial_plot = plot_2d_solution(mesh, initial_condition, 
                              title='Initial Drug Distribution',
                              colorbar_label='Concentration')
plt.savefig('drug_initial.png')

# Solve the diffusion equation
dt = 10.0  # time step (seconds)
num_steps = 1000
total_time = dt * num_steps  # total simulation time (seconds)

print(f"Simulating drug diffusion for {total_time} seconds...")
solver.solve(dt, num_steps)

# Get the solution
solution = solver.solution()

# Plot the results
conc_plot = plot_2d_solution(mesh, solution, 
                          title=f'Drug Concentration after {total_time/3600:.1f} hours',
                          colorbar_label='Concentration')
plt.savefig('drug_concentration.png')

# 3D surface plot
surface_plot = plot_2d_surface(mesh, solution, 
                             title=f'Drug Concentration Profile after {total_time/3600:.1f} hours',
                             zlabel='Concentration')
plt.savefig('drug_concentration_3d.png')

plt.show()

print("Simulation complete. Results saved to image files.")