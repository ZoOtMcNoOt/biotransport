"""
Example of 2D drug diffusion in tissue.

This example simulates the diffusion of a drug from a central source
into surrounding tissue, with a first-order reaction term representing
drug metabolism or degradation.

Model:
    dC/dt = D * ∇²C - k * C

where:
- D is the diffusion coefficient
- k is the first-order decay/metabolism rate

Notes:
- This example treats the spatial coordinates as centimeters (cm), so `D` is in cm²/s.
- `k` is a first-order decay rate (1/s).

BMEN 341 Reference: Week 3 (Drug Delivery, Reaction-Diffusion)
"""

import matplotlib.pyplot as plt
import biotransport as bt

EXAMPLE_NAME = "drug_diffusion_2d"

# Create a 2D mesh representing tissue domain
# Coordinates in cm: 2cm x 2cm centered at origin
mesh = bt.mesh_2d(50, 50, x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0)

# Physical parameters
# Units: cm, seconds
D = 1e-6  # Drug diffusivity in tissue (cm²/s)
decay_rate = 1e-3  # First-order metabolism rate (1/s)

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

# Simulate for ~28 hours
total_time = 10.0 * 10000  # seconds
print(f"Simulating drug diffusion for {total_time / 3600:.1f} hours...")
result = bt.solve(problem, t=total_time)
solution = result.solution()

# Plot final concentration
bt.plot(mesh, solution, title=f"Drug Concentration after {total_time / 3600:.1f} hours")
plt.savefig(bt.get_result_path("drug_concentration.png", EXAMPLE_NAME))

# 3D surface plot for visualization
bt.plot(
    mesh,
    solution,
    kind="surface",
    title=f"Drug Concentration Profile after {total_time / 3600:.1f} hours",
)
plt.savefig(bt.get_result_path("drug_concentration_3d.png", EXAMPLE_NAME))

plt.show()

print(
    f"Simulation complete. Results saved to '{bt.get_result_path('', EXAMPLE_NAME)}'."
)
