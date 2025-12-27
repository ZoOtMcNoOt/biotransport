"""
Example of 1D heat conduction in a rod.

This example simulates heat diffusion in a rod with fixed temperatures
at both ends. The thermal diffusivity equation is equivalent to the
diffusion equation, just with different physical interpretation.

    ∂T/∂t = α ∂²T/∂x²

where α is the thermal diffusivity (m²/s).

Notes:
- This uses the same numerical diffusion solver; interpret the field as temperature.

BMEN 341 Reference: Weeks 1-2 (Heat Transfer Analogy)
"""

import matplotlib.pyplot as plt
import biotransport as bt

EXAMPLE_NAME = "heat_conduction"

# Physical parameters
length = 0.1  # Rod length: 10 cm
T_left = 100  # Left boundary: 100°C
T_right = 20  # Right boundary: 20°C
T_initial = 20  # Initial temperature: 20°C (room temp)
thermal_diffusivity = 1e-5  # m²/s (typical for metal)

# Create mesh
mesh = bt.mesh_1d(100, x_max=length)
x = bt.x_nodes(mesh)

# Solve at multiple time points to show evolution
times = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
solutions = {0.0: bt.uniform(mesh, T_initial)}

for t_target in times:
    t_current = list(solutions.keys())[-1]
    print(f"Simulating from t={t_current:.1f}s to t={t_target:.1f}s...")

    # Create problem starting from previous solution
    problem = (
        bt.Problem(mesh)
        .diffusivity(thermal_diffusivity)
        .initial_condition(solutions[t_current])
        .dirichlet(bt.Boundary.Left, T_left)
        .dirichlet(bt.Boundary.Right, T_right)
    )

    result = bt.solve(problem, t=t_target - t_current, safety_factor=0.9)
    solutions[t_target] = list(result.solution())

# Plot evolution
plt.figure(figsize=(12, 8))
for t, temp in sorted(solutions.items()):
    plt.plot(x, temp, label=f"t = {t:.1f}s")

# Analytical steady state: linear temperature profile
steady_state = T_left + (T_right - T_left) * x / length
plt.plot(x, steady_state, "r--", linewidth=2, label="Steady State")

plt.grid(True)
plt.title("Heat Conduction in a Rod")
plt.xlabel("Position (m)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.savefig(bt.get_result_path("temperature_evolution.png", EXAMPLE_NAME))
plt.show()

print(
    f"Simulation complete. Results saved to '{bt.get_result_path('', EXAMPLE_NAME)}'."
)
