"""
Example of 1D heat conduction in a rod.

This example simulates heat diffusion in a rod with fixed temperatures
at both ends. The thermal diffusivity equation is equivalent to the
diffusion equation, just with different physical interpretation.

    ∂T/∂t = α ∂²T/∂x²

where α is the thermal diffusivity (m²/s).

The clock that matters is the Fourier number, Fo = α t / L². It is the elapsed
time measured in units of the rod's own diffusion time L²/α. Heat has only
crossed the rod once Fo is of order 1, so the run has to reach Fo ≈ 1 before
the linear steady-state profile is anything more than a promise. This example
prints Fo and the gap to steady state at every saved time so the approach is
something you can read off the numbers, not just believe from the picture.

Notes:
- This uses the same numerical diffusion solver; interpret the field as temperature.

BMEN 341 Reference: Weeks 1-2 (Heat Transfer Analogy)
"""

import matplotlib.pyplot as plt
import numpy as np

import biotransport as bt

EXAMPLE_NAME = "heat_conduction"

# Physical parameters
length = 0.1  # Rod length: 10 cm
T_left = 100  # Left boundary: 100°C
T_right = 20  # Right boundary: 20°C
T_initial = 20  # Initial temperature: 20°C (room temp)
thermal_diffusivity = 1e-5  # m²/s (typical for metal)

# Diffusion time of the rod: how long heat needs to cross it end to end.
diffusion_time = length**2 / thermal_diffusivity  # 1000 s here

# Create mesh
mesh = bt.mesh_1d(100, x_max=length)
x = bt.x_nodes(mesh)

# Analytical steady state: linear temperature profile
steady_state = T_left + (T_right - T_left) * x / length

# Save at times spanning Fo = 0.01 (barely started) to Fo = 1 (heat has crossed).
times = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]  # in units of the diffusion time
save_at = [fraction * diffusion_time for fraction in times]

problem = (
    bt.Problem(mesh)
    .diffusivity(thermal_diffusivity)
    .initial(T_initial)
    .dirichlet("left", T_left)
    .dirichlet("right", T_right)
)

print(problem.describe())
print()
print(f"Rod diffusion time L^2/alpha: {diffusion_time:.0f} s")
print(f"Integrating to t = {save_at[-1]:.0f} s, i.e. Fourier number Fo = {times[-1]:g}")
print()

solution = bt.solve(problem, end_time=save_at[-1], save_at=save_at)
print(f"Steps taken: {solution.steps}")
print()

# How far each saved profile still is from the steady state it is heading for.
print(f"{'t (s)':>8}  {'Fo = alpha t / L^2':>18}  {'max|T - T_steady| (degC)':>26}")
for t in solution.times:
    fourier = thermal_diffusivity * t / length**2
    gap = float(np.max(np.abs(solution.at(t) - steady_state)))
    print(f"{t:8.1f}  {fourier:18.3f}  {gap:26.4f}")

final_gap = float(np.max(np.abs(solution.c - steady_state)))
print()
print(
    f"At Fo = {times[-1]:g} the final profile is within {final_gap:.4f} degC of the "
    f"steady state everywhere."
)
print(
    "For a slab the transient decays like exp(-pi^2 Fo), so Fo = 0.05 (the old "
    "stopping point) still leaves tens of degrees on the table."
)

# Plot evolution
plt.figure(figsize=(12, 8))
for t in solution.times:
    fourier = thermal_diffusivity * t / length**2
    plt.plot(x, solution.at(t), label=f"t = {t:.0f} s (Fo = {fourier:g})")

plt.plot(x, steady_state, "r--", linewidth=2, label="Steady state (linear)")

plt.grid(True)
plt.title("Heat Conduction in a Rod: Approach to Steady State")
plt.xlabel("Position (m)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.savefig(bt.get_result_path("temperature_evolution.png", EXAMPLE_NAME))
plt.show()

print(
    f"Simulation complete. Results saved to '{bt.get_result_path('', EXAMPLE_NAME)}'."
)
