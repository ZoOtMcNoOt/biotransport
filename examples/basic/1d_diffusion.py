"""Example: 1D diffusion.

This is a minimal diffusion demo solving:
    dC/dt = D * d²C/dx²

Notes:
- Units are treated as *dimensionless* here (domain length = 1, time in arbitrary units).
- Stability depends on the explicit time step; keep `dt` small enough for the chosen `D` and grid.

BMEN 341 Reference: Weeks 1-2 (Fick's Laws)
"""

import matplotlib.pyplot as plt
import biotransport as bt

EXAMPLE_NAME = "1d_diffusion"

# Create mesh: 100 nodes from x=0 to x=1
mesh = bt.mesh_1d(100)

# Setup problem with Gaussian initial condition
# The gaussian() helper creates exp(-((x - center)^2) / (2 * width^2))
problem = (
    bt.Problem(mesh)
    .diffusivity(0.01)
    .initial_condition(bt.gaussian(mesh, center=0.5, width=0.1))
    .dirichlet(bt.Boundary.Left, 0.0)
    .dirichlet(bt.Boundary.Right, 0.0)
)

# Solve using the simplified API
print("Solving 1D diffusion for t_end=0.1...")
result = bt.solve(problem, t=0.1)

# Plot result with initial condition overlay
x = bt.x_nodes(mesh)
bt.plot(mesh, result.solution(), title=f"Diffusion after t = {result.stats.t_end}")
plt.plot(x, bt.gaussian(mesh, center=0.5, width=0.1), "b--", alpha=0.5, label="Initial")
plt.legend()
plt.savefig(bt.get_result_path("diffusion_result.png", EXAMPLE_NAME))
plt.show()

print(
    f"Simulation complete. Results saved to '{bt.get_result_path('', EXAMPLE_NAME)}'."
)
