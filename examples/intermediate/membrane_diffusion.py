"""
Example of diffusion through a membrane.

This example simulates diffusion of a solute through a membrane separating
two compartments. The membrane has a different diffusion coefficient than
the surrounding medium, creating a barrier to transport.

Notes:
- Units are SI (meters, seconds, m^2/s).
- The PDE is dC/dt = ∇·(D(x)∇C) with a low-D membrane region.
- Uses the new variable diffusivity support in the C++ solver.

BMEN 341 Reference: Week 4 (Membrane Transport)
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt

EXAMPLE_NAME = "membrane_diffusion"

# Physical parameters
L = 1e-3  # Domain length (m)
D_medium = 1e-9  # Diffusivity in medium (m²/s)
D_membrane = 1e-11  # Diffusivity in membrane (m²/s)
membrane_pos = 0.5e-3  # Membrane center (m)
membrane_width = 0.05e-3  # Membrane thickness (m)
C_left, C_right = 1.0, 0.0  # Boundary concentrations

# Setup mesh using convenience function
mesh = bt.mesh_1d(200, x_max=L)
x = bt.x_nodes(mesh)

# Build spatially-varying diffusivity using SpatialField builder
mem_lo = membrane_pos - membrane_width / 2
mem_hi = membrane_pos + membrane_width / 2
D_field = list(
    bt.SpatialField(mesh)
    .default(D_medium)
    .region_box(mem_lo, mem_hi, value=D_membrane)
    .build()
)

# Initial condition: linear ramp across membrane
frac = np.clip((x - mem_lo) / membrane_width, 0.0, 1.0)
initial = np.where(
    x < mem_lo,
    C_left,
    np.where(x > mem_hi, C_right, C_left * (1 - frac) + C_right * frac),
)

# Build the problem with variable diffusivity
problem = (
    bt.Problem(mesh)
    .diffusivity_field(D_field)
    .initial_condition(list(initial))
    .dirichlet(bt.Boundary.Left, C_left)
    .dirichlet(bt.Boundary.Right, C_right)
)

# Run simulation at multiple time snapshots
times_to_save = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
saved_solutions = {0.0: initial.copy()}

t = 0.0
for t_target in times_to_save:
    print(f"Simulating from t={t:.1f}s to t={t_target:.1f}s...")
    result = bt.solve(problem, t_target - t)
    sol = np.array(result.solution())  # solution() is a method
    problem = problem.initial_condition(list(sol))
    saved_solutions[t_target] = sol
    t = t_target

# Analytical steady state for comparison
x_analytical = np.linspace(0, L, 1000)
# Steady-state flux through series resistances
flux = (C_left - C_right) / (
    (mem_lo / D_medium) + (membrane_width / D_membrane) + ((L - mem_hi) / D_medium)
)
c_mem_left = C_left - flux * mem_lo / D_medium
c_mem_right = c_mem_left - flux * membrane_width / D_membrane
c_analytical = np.where(
    x_analytical < mem_lo,
    C_left - flux * x_analytical / D_medium,
    np.where(
        x_analytical <= mem_hi,
        c_mem_left - flux * (x_analytical - mem_lo) / D_membrane,
        c_mem_right - flux * (x_analytical - mem_hi) / D_medium,
    ),
)

# Plot time evolution
plt.figure(figsize=(10, 6))
for t, sol in sorted(saved_solutions.items()):
    plt.plot(x * 1e3, sol, label=f"t = {t:.1f}s")
plt.plot(x_analytical * 1e3, c_analytical, "r--", linewidth=2, label="Steady State")
plt.axvspan(mem_lo * 1e3, mem_hi * 1e3, color="gray", alpha=0.3, label="Membrane")
plt.grid(True)
plt.title("Diffusion Through a Membrane")
plt.xlabel("Position (mm)")
plt.ylabel("Concentration")
plt.legend()
plt.tight_layout()
plt.savefig(bt.get_result_path("membrane_diffusion.png", EXAMPLE_NAME))
plt.show()

print(
    f"Simulation complete. Results saved to '{bt.get_result_path('', EXAMPLE_NAME)}'."
)
