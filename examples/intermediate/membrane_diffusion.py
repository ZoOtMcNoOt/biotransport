"""
Example of diffusion through a membrane.

This example simulates diffusion of a solute through a membrane separating
two compartments. The membrane has a different diffusion coefficient than
the surrounding medium, creating a barrier to transport.

Notes:
- Units are SI (meters, seconds, m^2/s).
- The PDE is dC/dt = ∇·(D(x)∇C) with a low-D membrane region.
- Uses the new variable diffusivity support in the C++ solver.
- The solute starts in the left compartment only; the membrane is initially
  solute-free, so the run shows the barrier filling and then breaking through.
- Snapshots are log-spaced over the membrane diffusion time
  tau_mem = membrane_width^2 / D_membrane, which is the slow scale here.

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
D_field = np.asarray(
    bt.SpatialField(mesh)
    .default(D_medium)
    .region_box(mem_lo, mem_hi, value=D_membrane)
    .build(),
    dtype=np.float64,
)

# Initial condition: solute fills the left compartment, the membrane and the
# right compartment start empty.
initial = np.where(x < mem_lo, C_left, C_right)

# Build the problem with variable diffusivity
problem = (
    bt.Problem(mesh)
    .diffusivity_field(D_field)
    .initial(initial)
    .dirichlet(bt.Boundary.Left, C_left)
    .dirichlet(bt.Boundary.Right, C_right)
)

# Snapshot times: log-spaced across the membrane diffusion time so the curves
# are actually separated instead of piling up at one end of the transient.
tau_mem = membrane_width**2 / D_membrane  # s
times_to_save = list(tau_mem * np.logspace(np.log10(0.002), np.log10(6.0), 7))

print(f"Membrane diffusion time tau_mem = {tau_mem:.1f} s")
print(f"Certified stable time step: {problem.stable_time_step():.4g} s")
print(
    f"Running to t = {times_to_save[-1]:.1f} s = {times_to_save[-1] / tau_mem:.1f} tau_mem"
)

result = bt.solve(problem, end_time=times_to_save[-1], save_at=times_to_save)
saved_solutions = {0.0: initial.copy()}
for t_target in times_to_save:
    saved_solutions[t_target] = np.asarray(result.at(t_target)).copy()

# Steady state of this exact discretization (what the transient converges to)
c_steady = np.asarray(bt.solve_steady(problem).concentration)

# Analytical sharp-interface steady state (series resistances) for comparison
x_analytical = np.linspace(0, L, 1000)
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

# Report how far apart the plotted curves actually are
plotted_times = sorted(saved_solutions)
spreads = [
    float(np.max(np.abs(saved_solutions[t2] - saved_solutions[t1])))
    for t1, t2 in zip(plotted_times[:-1], plotted_times[1:])
]
print("\nSeparation between consecutive plotted curves (max |dC|):")
for (t1, t2), s in zip(zip(plotted_times[:-1], plotted_times[1:]), spreads):
    print(f"  t = {t1:7.1f} s -> {t2:7.1f} s : {s:.4f}")
print(f"  smallest consecutive spread: {min(spreads):.4f}")
print(f"  largest consecutive spread:  {max(spreads):.4f}")

final_gap = float(np.max(np.abs(saved_solutions[plotted_times[-1]] - c_steady)))
mesh_gap = float(np.max(np.abs(c_steady - np.interp(x, x_analytical, c_analytical))))
print("\nApproach to steady state:")
print(f"  max |C(t_end) - C_steady(numerical)| = {final_gap:.2e}")
print(
    f"  max |C_steady(numerical) - C_steady(sharp interface)| = {mesh_gap:.4f}\n"
    f"  (the interface is only resolved to +/- dx/2 = {0.5 * mesh.dx() * 1e6:.1f} um of a "
    f"{membrane_width * 1e6:.0f} um membrane, an O(dx) offset in the membrane resistance)"
)

# Plot time evolution
plt.figure(figsize=(10, 6))
for t in plotted_times:
    plt.plot(x * 1e3, saved_solutions[t], label=f"t = {t:.1f}s")
plt.plot(
    x * 1e3,
    c_steady,
    "k--",
    linewidth=2,
    label="Steady state (numerical)",
)
plt.plot(
    x_analytical * 1e3,
    c_analytical,
    "r:",
    linewidth=2,
    label="Steady state (sharp interface)",
)
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
    f"\nSimulation complete. Results saved to '{bt.get_result_path('', EXAMPLE_NAME)}'."
)
