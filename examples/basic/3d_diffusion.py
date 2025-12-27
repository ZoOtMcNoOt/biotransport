"""
3D Diffusion Example: Heat Conduction in a Cubic Block

This example demonstrates 3D diffusion simulation using the StructuredMesh3D class.
It simulates heat diffusion in a unit cube with:
- Initial hot region in the center
- Zero temperature (Dirichlet) on all boundaries

This is a canonical test case for verifying 3D diffusion solvers.
"""

import numpy as np
import biotransport as bt

# =============================================================================
# Problem Setup
# =============================================================================

# Material properties (thermal diffusivity of aluminum, m²/s)
D = 8.4e-5  # m²/s

# Domain: 1 cm x 1 cm x 1 cm cube
L = 0.01  # m
nx, ny, nz = 20, 20, 20  # 20 cells in each direction

# Create 3D mesh
mesh = bt.StructuredMesh3D(nx, ny, nz, 0.0, L, 0.0, L, 0.0, L)

print("3D Mesh created:")
print(f"  Nodes: {mesh.num_nodes()} ({nx+1} x {ny+1} x {nz+1})")
print(f"  Cells: {mesh.num_cells()} ({nx} x {ny} x {nz})")
print(
    f"  Grid spacing: dx={mesh.dx()*1e3:.3f} mm, dy={mesh.dy()*1e3:.3f} mm, dz={mesh.dz()*1e3:.3f} mm"
)

# =============================================================================
# Initial Condition: Hot sphere in center
# =============================================================================

u0 = np.zeros(mesh.num_nodes())

# Create a hot region (sphere) in the center
center = (L / 2, L / 2, L / 2)
radius = L / 4  # quarter of the domain size

for k in range(nz + 1):
    for j in range(ny + 1):
        for i in range(nx + 1):
            x, y, z = mesh.x(i), mesh.y(j), mesh.z(k)
            r = np.sqrt(
                (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
            )

            idx = mesh.index(i, j, k)
            if r <= radius:
                u0[idx] = 100.0  # Hot core (100°C above ambient)
            else:
                u0[idx] = 0.0  # Ambient

print(
    f"\nInitial condition: Hot sphere (T=100°C) at center, radius={radius*1e3:.2f} mm"
)

# =============================================================================
# Solver Setup
# =============================================================================

solver = bt.DiffusionSolver3D(mesh, D)
solver.set_initial_condition(u0.tolist())

# Dirichlet BCs: zero temperature on all boundaries
for boundary_id in range(6):  # XMin=0, XMax=1, YMin=2, YMax=3, ZMin=4, ZMax=5
    solver.set_dirichlet_boundary(boundary_id, 0.0)

# Time stepping
dt_max = solver.max_stable_time_step()
dt = 0.8 * dt_max  # Use 80% of max stable step for safety

print("\nSolver configuration:")
print(f"  Diffusivity: D = {D:.2e} m²/s")
print(f"  Max stable dt: {dt_max:.4e} s")
print(f"  Using dt: {dt:.4e} s")
print(f"  CFL stability check: {solver.check_stability(dt)}")

# =============================================================================
# Time Integration
# =============================================================================

t_end = 0.01  # seconds
num_steps = int(t_end / dt)

print("\nRunning simulation:")
print(f"  Total time: {t_end*1000:.1f} ms")
print(f"  Time steps: {num_steps}")

# Run the simulation
solver.solve(dt, num_steps)

print(f"  Simulation complete, t = {solver.time()*1000:.1f} ms")

# =============================================================================
# Results Analysis
# =============================================================================

solution = np.array(solver.solution())

# Find temperature at center and corners
center_idx = mesh.index(nx // 2, ny // 2, nz // 2)
corner_idx = mesh.index(0, 0, 0)

T_center = solution[center_idx]
T_corner = solution[corner_idx]
T_max = solution.max()
T_min = solution.min()
T_avg = solution.mean()

print("\nResults:")
print(f"  Center temperature: {T_center:.2f}°C")
print(f"  Corner temperature: {T_corner:.4f}°C")  # Should be ~0 (boundary)
print(f"  Max temperature: {T_max:.2f}°C")
print(f"  Min temperature: {T_min:.4f}°C")
print(f"  Average temperature: {T_avg:.2f}°C")

# Check conservation (total thermal energy should decrease due to BC)
initial_energy = u0.sum() * mesh.dx() * mesh.dy() * mesh.dz()
final_energy = solution.sum() * mesh.dx() * mesh.dy() * mesh.dz()
energy_ratio = final_energy / initial_energy

print("\nEnergy analysis:")
print(f"  Initial energy (arbitrary units): {initial_energy:.2e}")
print(f"  Final energy (arbitrary units): {final_energy:.2e}")
print(f"  Energy ratio: {energy_ratio:.2%}")
print("  (Energy decrease expected with Dirichlet T=0 boundaries)")

# =============================================================================
# Line Profile (along z-axis through center)
# =============================================================================

print("\nTemperature profile along z-axis through center:")
print(f"{'z (mm)':<10} {'T (°C)':<10}")
print("-" * 20)
for k in range(0, nz + 1, max(1, nz // 5)):
    idx = mesh.index(nx // 2, ny // 2, k)
    z_mm = mesh.z(k) * 1000
    print(f"{z_mm:<10.2f} {solution[idx]:<10.2f}")

print("\n✅ 3D diffusion example completed successfully!")
