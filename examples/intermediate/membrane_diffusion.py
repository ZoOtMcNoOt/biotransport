"""
Example of diffusion through a membrane.

This example simulates diffusion of a solute through a membrane separating
two compartments. The membrane has a different diffusion coefficient than
the surrounding medium, creating a barrier to transport.
"""

import numpy as np
import matplotlib.pyplot as plt
from biotransport import StructuredMesh, ReactionDiffusionSolver
from biotransport.utils import get_result_path
from biotransport.visualization import plot_1d_solution

# Create results subdirectory for this example
EXAMPLE_NAME = "membrane_diffusion"

# Physical parameters
L = 1.0e-3  # Domain length (m)
D_medium = 1e-9  # Diffusion coefficient in medium (m²/s)
D_membrane = 1e-11  # Diffusion coefficient in membrane (m²/s)
membrane_pos = 0.5e-3  # Membrane position (m)
membrane_width = 0.05e-3  # Membrane width (m)

# Initial concentrations
C_left = 1.0  # Initial concentration in left compartment
C_right = 0.0  # Initial concentration in right compartment

# Create a 1D mesh
nx = 200
mesh = StructuredMesh(nx, 0.0, L)

# Determine diffusion coefficient at each position
def get_diffusion_coefficient(x):
    """
    Return the diffusion coefficient at position x.
    Lower in the membrane region.
    """
    # Check if position is within membrane
    if abs(x - membrane_pos) <= membrane_width/2:
        return D_membrane
    else:
        return D_medium

# Modified reaction term to simulate spatially-varying diffusion coefficient
# This is a workaround since our current library doesn't directly support
# spatially-varying diffusion coefficients.
def spatially_varying_diffusion(u, x, y, t):
    """
    Implement spatially-varying diffusion as a reaction term.

    The true PDE with variable D is:
        ∂u/∂t = ∇·(D(x)∇u)

    We're splitting it into a constant diffusion with base coefficient D_medium:
        ∂u/∂t = D_medium ∇²u + ∇·((D(x) - D_medium)∇u)

    The second term becomes our "reaction" term:
        R(u,x) = ∇·((D(x) - D_medium)∇u)

    For 1D with numeric approximation:
        R(u,x) ≈ ((D(x+dx) - D_medium) * (u(x+dx) - u(x)) -
                  (D(x-dx) - D_medium) * (u(x) - u(x-dx))) / dx²
    """
    # Get indices for x-dx, x, x+dx
    # Note: This is approximated for the example and might not be accurate at boundaries
    dx = mesh.dx()
    i = int(x / dx)

    # Handle boundary cases
    if i == 0:
        u_left = u
        x_left = x
    else:
        x_left = mesh.x(i-1)
        u_left = u  # We need to get from solution vector, but can't in this callback

    if i >= nx:
        u_right = u
        x_right = x
    else:
        x_right = mesh.x(i+1)
        u_right = u  # Same issue as above

    # Get diffusion coefficients
    D_x = get_diffusion_coefficient(x)
    D_left = get_diffusion_coefficient(x_left)
    D_right = get_diffusion_coefficient(x_right)

    # For simplicity, return zero - we'll handle this differently
    # The reaction method isn't ideal for implementing variable diffusion
    return 0.0

# Create diffusion solver with base diffusion coefficient
solver = ReactionDiffusionSolver(mesh, D_medium, spatially_varying_diffusion)

# Position vector
x = np.array([mesh.x(i) for i in range(mesh.nx() + 1)])

# Initial condition (step function)
initial_condition = np.zeros_like(x)
for i in range(len(x)):
    if x[i] < membrane_pos - membrane_width/2:
        initial_condition[i] = C_left
    elif x[i] > membrane_pos + membrane_width/2:
        initial_condition[i] = C_right
    else:
        # Linear interpolation within membrane
        fraction = (x[i] - (membrane_pos - membrane_width/2)) / membrane_width
        initial_condition[i] = C_left * (1 - fraction) + C_right * fraction

solver.set_initial_condition(initial_condition)

# Set boundary conditions
solver.set_dirichlet_boundary(0, C_left)   # Fixed concentration on left
solver.set_dirichlet_boundary(1, C_right)  # Fixed concentration on right

# Plot diffusion coefficient profile
plt.figure(figsize=(10, 4))
D_profile = np.array([get_diffusion_coefficient(pos) for pos in x])
plt.semilogy(x*1e3, D_profile, 'k-', linewidth=2)
plt.grid(True)
plt.title('Diffusion Coefficient Profile')
plt.xlabel('Position (mm)')
plt.ylabel('Diffusion Coefficient (m²/s)')
plt.tight_layout()
plt.savefig(get_result_path('diffusion_coefficient.png', EXAMPLE_NAME))

# Plot initial condition
plt.figure(figsize=(10, 6))
plt.plot(x*1e3, initial_condition, 'b-', linewidth=2, label='Initial')

# Define alternate method for variable diffusion
# Instead of using the reaction term, we'll implement a custom solver to handle
# the variable diffusion coefficient properly
def solve_variable_diffusion(initial, dx, dt, num_steps, times_to_save=None):
    """
    Solve the diffusion equation with variable diffusion coefficient.

    Args:
        initial: Initial concentration profile
        dx: Spatial step size
        dt: Time step size
        num_steps: Number of time steps
        times_to_save: Optional list of times to save solutions

    Returns:
        Dictionary of saved solutions at specified times
    """
    u = initial.copy()
    saved_solutions = {}
    current_time = 0.0

    # Precompute diffusion coefficients at each point
    D = np.array([get_diffusion_coefficient(pos) for pos in x])

    # Stability check
    max_dt = 0.5 * dx**2 / np.max(D)
    if dt > max_dt:
        print(f"Warning: Time step may be too large for stability. Recommended dt <= {max_dt:.2e}")

    # Time stepping loop
    for step in range(num_steps):
        # Create a copy for updating
        u_new = u.copy()

        # Update interior points
        for i in range(1, len(u)-1):
            # Calculate diffusion coefficients at interfaces
            D_left = 0.5 * (D[i-1] + D[i])   # Left interface
            D_right = 0.5 * (D[i] + D[i+1])  # Right interface

            # Calculate fluxes
            flux_left = D_left * (u[i] - u[i-1]) / dx
            flux_right = D_right * (u[i+1] - u[i]) / dx

            # Update using explicit scheme
            u_new[i] = u[i] + dt * (flux_right - flux_left) / dx

        # Apply boundary conditions (Dirichlet)
        u_new[0] = C_left
        u_new[-1] = C_right

        # Update solution
        u = u_new

        current_time += dt

        # Save solution at specified times
        if times_to_save is not None and any(abs(t - current_time) < 0.5*dt for t in times_to_save):
            saved_solutions[current_time] = u.copy()

    return u, saved_solutions

# Solve using custom variable diffusion solver
dt = 1e-3  # Time step (s)
times_to_save = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]  # Times to save (s)
solution, saved_solutions = solve_variable_diffusion(
    initial_condition, mesh.dx(), dt, 100000, times_to_save)

# Plot solutions at different times
for t, sol in sorted(saved_solutions.items()):
    plt.plot(x*1e3, sol, label=f't = {t:.1f}s')

# Analytical steady-state solution for comparison
# For variable D, the steady state satisfies D(x)·∂c/∂x = constant
# This gives a piecewise linear profile with different slopes
x_analytical = np.linspace(0, L, 1000)
c_analytical = np.zeros_like(x_analytical)

flux = (C_left - C_right) / (
        (membrane_pos - 0) / D_medium +
        membrane_width / D_membrane +
        (L - (membrane_pos + membrane_width/2)) / D_medium
)

for i, pos in enumerate(x_analytical):
    if pos < membrane_pos - membrane_width/2:
        c_analytical[i] = C_left - flux * pos / D_medium
    elif pos <= membrane_pos + membrane_width/2:
        c_mem_left = C_left - flux * (membrane_pos - membrane_width/2) / D_medium
        c_analytical[i] = c_mem_left - flux * (pos - (membrane_pos - membrane_width/2)) / D_membrane
    else:
        c_mem_right = c_analytical[np.abs(x_analytical - (membrane_pos + membrane_width/2)).argmin()]
        c_analytical[i] = c_mem_right - flux * (pos - (membrane_pos + membrane_width/2)) / D_medium

plt.plot(x_analytical*1e3, c_analytical, 'r--', linewidth=2, label='Steady State (Analytical)')

# Add membrane position indicator
plt.axvspan((membrane_pos - membrane_width/2)*1e3,
            (membrane_pos + membrane_width/2)*1e3,
            color='gray', alpha=0.3, label='Membrane')

plt.grid(True)
plt.title('Diffusion Through a Membrane')
plt.xlabel('Position (mm)')
plt.ylabel('Concentration')
plt.legend()
plt.tight_layout()
plt.savefig(get_result_path('membrane_diffusion.png', EXAMPLE_NAME))

# Plot concentration profile near the membrane with higher resolution
plt.figure(figsize=(10, 6))
plt.plot(x*1e3, solution, 'b-', linewidth=2, label='Numerical')
plt.plot(x_analytical*1e3, c_analytical, 'r--', linewidth=2, label='Analytical')

# Add membrane position indicator
plt.axvspan((membrane_pos - membrane_width/2)*1e3,
            (membrane_pos + membrane_width/2)*1e3,
            color='gray', alpha=0.3, label='Membrane')

plt.grid(True)
plt.title('Steady-State Concentration Profile')
plt.xlabel('Position (mm)')
plt.ylabel('Concentration')
plt.xlim((membrane_pos - 0.2e-3)*1e3, (membrane_pos + 0.2e-3)*1e3)  # Zoom on membrane
plt.legend()
plt.tight_layout()
plt.savefig(get_result_path('membrane_zoom.png', EXAMPLE_NAME))

# Show all plots
plt.show()

results_dir = get_result_path('', EXAMPLE_NAME)
print(f"Simulation complete. Results saved to '{results_dir}'.")