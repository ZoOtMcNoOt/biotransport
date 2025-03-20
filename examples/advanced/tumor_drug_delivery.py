"""
Example of drug delivery to a tumor with interstitial pressure effects.

This advanced example simulates the delivery of a drug to a solid tumor,
accounting for:
1. Elevated interstitial fluid pressure (IFP) in the tumor
2. Convection and diffusion of the drug
3. Heterogeneous vascular permeability
4. Drug binding to tissue and cellular uptake

This model demonstrates key barriers to effective drug delivery in cancer treatment.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from biotransport import StructuredMesh, ReactionDiffusionSolver
from biotransport.utils import get_result_path

# Create results subdirectory for this example
EXAMPLE_NAME = "tumor_drug_delivery"

# Physical parameters
L = 5e-3  # Domain size (m) - 5mm tissue region
R_tumor = 2e-3  # Tumor radius (m)
center_x, center_y = L/2, L/2  # Tumor center position

# Transport parameters
D_drug_normal = 5e-11  # Drug diffusion coefficient in normal tissue (m²/s)
D_drug_tumor = 2e-11   # Drug diffusion coefficient in tumor (m²/s) - often lower due to dense ECM
k_binding = 1e-3       # Drug binding rate to tissue (1/s)
k_uptake = 5e-4        # Cellular uptake rate (1/s)

# Vasculature parameters
MVD_normal = 100       # Microvascular density in normal tissue (vessels/mm²)
MVD_tumor_core = 20    # Microvascular density in tumor core - hypoxic, fewer vessels
MVD_tumor_rim = 200    # Microvascular density in tumor periphery - angiogenic rim

# Blood vessel permeability
P_normal = 1e-7        # Vessel permeability in normal tissue (m/s)
P_tumor = 5e-7         # Vessel permeability in tumor (m/s) - enhanced due to leaky vessels
C_plasma = 1.0         # Normalized drug concentration in plasma

# Interstitial fluid pressure (IFP) parameters
IFP_normal = 0.0       # Normal tissue IFP (mmHg)
IFP_tumor = 20.0       # Tumor IFP (mmHg) - elevated in tumors
# Convert pressure to Pa
IFP_normal_pa = IFP_normal * 133.322
IFP_tumor_pa = IFP_tumor * 133.322

# Hydraulic conductivity
K_normal = 5e-7        # Hydraulic conductivity in normal tissue (m²/(Pa·s))
K_tumor = 2.5e-7       # Hydraulic conductivity in tumor (m²/(Pa·s))

# Create a 2D mesh
nx, ny = 100, 100
mesh = StructuredMesh(nx, ny, 0.0, L, 0.0, L)
dx, dy = mesh.dx(), mesh.dy()

# Helper function to determine if a point is in the tumor
def in_tumor(x, y, rim_size=0.5e-3):
    """
    Check if a point is in the tumor.
    Returns:
    - 0 for normal tissue
    - 1 for tumor rim
    - 2 for tumor core
    """
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    if dist > R_tumor:
        return 0  # Normal tissue
    elif dist > R_tumor - rim_size:
        return 1  # Tumor rim
    else:
        return 2  # Tumor core

# Initialize spatial maps
tissue_map = np.zeros((ny+1, nx+1))
vessel_density = np.zeros((ny+1, nx+1))
permeability = np.zeros((ny+1, nx+1))
diffusivity = np.zeros((ny+1, nx+1))

# Create tissue type, vessel density and permeability maps
for j in range(ny+1):
    for i in range(nx+1):
        x, y = mesh.x(i), mesh.y(0, j)
        tissue_type = in_tumor(x, y)
        tissue_map[j, i] = tissue_type

        # Set vessel density
        if tissue_type == 0:  # Normal tissue
            vessel_density[j, i] = MVD_normal
            permeability[j, i] = P_normal
            diffusivity[j, i] = D_drug_normal
        elif tissue_type == 1:  # Tumor rim
            vessel_density[j, i] = MVD_tumor_rim
            permeability[j, i] = P_tumor
            diffusivity[j, i] = D_drug_tumor
        else:  # Tumor core
            vessel_density[j, i] = MVD_tumor_core
            permeability[j, i] = P_tumor
            diffusivity[j, i] = D_drug_tumor

# Solve for interstitial fluid pressure (IFP)
# The IFP follows Darcy's Law: ∇·(K∇p) = 0
# We'll solve this with a simple iterative method
def solve_pressure(max_iter=10000, tol=1e-6):
    """
    Solve for the interstitial fluid pressure distribution.
    Uses iterative method (Gauss-Seidel) with fixed pressure at boundaries.
    """
    # Initialize pressure with boundary and initial guesses
    pressure = np.ones((ny+1, nx+1)) * IFP_normal_pa

    # Set initial guess for tumor region
    for j in range(ny+1):
        for i in range(nx+1):
            x, y = mesh.x(i), mesh.y(0, j)
            if in_tumor(x, y) > 0:
                pressure[j, i] = IFP_tumor_pa

    # Calculate spatially varying hydraulic conductivity
    K = np.zeros_like(pressure)
    for j in range(ny+1):
        for i in range(nx+1):
            if in_tumor(mesh.x(i), mesh.y(0, j)) > 0:
                K[j, i] = K_tumor
            else:
                K[j, i] = K_normal

    # Iterative solution (Gauss-Seidel)
    for iter in range(max_iter):
        p_old = pressure.copy()

        # Update interior points
        for j in range(1, ny):
            for i in range(1, nx):
                # Get hydraulic conductivity at cell interfaces
                K_left = 0.5 * (K[j, i] + K[j, i-1])
                K_right = 0.5 * (K[j, i] + K[j, i+1])
                K_bottom = 0.5 * (K[j, i] + K[j-1, i])
                K_top = 0.5 * (K[j, i] + K[j+1, i])

                # Discrete Laplace operator with variable coefficients
                pressure[j, i] = (K_left * pressure[j, i-1] +
                                  K_right * pressure[j, i+1] +
                                  K_bottom * pressure[j-1, i] +
                                  K_top * pressure[j+1, i]) / (K_left + K_right + K_bottom + K_top)

        # Enforce boundary conditions (fixed pressure)
        pressure[0, :] = IFP_normal_pa
        pressure[-1, :] = IFP_normal_pa
        pressure[:, 0] = IFP_normal_pa
        pressure[:, -1] = IFP_normal_pa

        # Check convergence
        rel_error = np.max(np.abs(pressure - p_old)) / (np.max(np.abs(pressure)) + 1e-10)
        if rel_error < tol:
            print(f"Pressure solution converged in {iter+1} iterations")
            break

    if iter == max_iter-1:
        print(f"Warning: Pressure solution did not converge within {max_iter} iterations")

    return pressure

# Solve for pressure field
pressure = solve_pressure()

# Calculate pressure gradient
pressure_grad_x = np.zeros_like(pressure)
pressure_grad_y = np.zeros_like(pressure)

for j in range(1, ny):
    for i in range(1, nx):
        pressure_grad_x[j, i] = (pressure[j, i+1] - pressure[j, i-1]) / (2 * dx)
        pressure_grad_y[j, i] = (pressure[j+1, i] - pressure[j-1, i]) / (2 * dy)

# Calculate velocity field from pressure gradient
velocity_x = np.zeros_like(pressure)
velocity_y = np.zeros_like(pressure)

for j in range(ny+1):
    for i in range(nx+1):
        if in_tumor(mesh.x(i), mesh.y(0, j)) > 0:
            k = K_tumor
        else:
            k = K_normal

        velocity_x[j, i] = -k * pressure_grad_x[j, i]
        velocity_y[j, i] = -k * pressure_grad_y[j, i]

# Set up the reaction-diffusion solver
# For the drug transport, we need to account for:
# 1. Diffusion (position-dependent)
# 2. Convection due to interstitial fluid flow
# 3. Drug binding and uptake
# 4. Source terms due to vascular delivery

# Since our library is limited to reaction-diffusion without convection,
# we'll implement a custom solver for this complex model

def solve_drug_transport(num_steps=10000, dt=0.1, times_to_save=None):
    """
    Solve the drug transport equation considering diffusion, convection,
    binding, uptake, and vascular sources.

    The equation is:
        ∂C/∂t = ∇·(D∇C) - ∇·(v*C) - k_binding*C - k_uptake*C + Source

    where Source = P * SA/V * (C_plasma - C) and SA/V is the surface area to
    volume ratio of blood vessels, proportional to vessel density.
    """
    # Initialize concentrations
    C_free = np.zeros((ny+1, nx+1))  # Free drug concentration
    C_bound = np.zeros((ny+1, nx+1))  # Bound drug concentration
    C_cell = np.zeros((ny+1, nx+1))   # Intracellular drug concentration

    # Surface area to volume ratio (SA/V) is proportional to vessel density
    # This determines rate of drug extravasation from vessels
    SA_V = vessel_density / np.max(vessel_density)

    # Storage for saved solutions
    saved_solutions = {}
    current_time = 0.0

    # Time stepping loop
    for step in range(num_steps):
        # Create arrays for updated values
        C_free_new = C_free.copy()

        # Update interior points
        for j in range(1, ny):
            for i in range(1, nx):
                # Diffusion term (central difference)
                diff_x = diffusivity[j, i] * (C_free[j, i+1] - 2*C_free[j, i] + C_free[j, i-1]) / (dx*dx)
                diff_y = diffusivity[j, i] * (C_free[j+1, i] - 2*C_free[j, i] + C_free[j-1, i]) / (dy*dy)
                diffusion = diff_x + diff_y

                # Convection term (upwind scheme for stability)
                if velocity_x[j, i] > 0:
                    conv_x = velocity_x[j, i] * (C_free[j, i] - C_free[j, i-1]) / dx
                else:
                    conv_x = velocity_x[j, i] * (C_free[j, i+1] - C_free[j, i]) / dx

                if velocity_y[j, i] > 0:
                    conv_y = velocity_y[j, i] * (C_free[j, i] - C_free[j-1, i]) / dy
                else:
                    conv_y = velocity_y[j, i] * (C_free[j+1, i] - C_free[j, i]) / dy

                convection = conv_x + conv_y

                # Binding and uptake (sink terms)
                binding = k_binding * C_free[j, i]
                uptake = k_uptake * C_free[j, i]

                # Source term (vascular delivery)
                source = permeability[j, i] * SA_V[j, i] * (C_plasma - C_free[j, i])

                # Update free drug concentration
                C_free_new[j, i] = C_free[j, i] + dt * (diffusion - convection - binding - uptake + source)

        # Update bound and cellular drug concentrations
        C_bound += dt * k_binding * C_free
        C_cell += dt * k_uptake * C_free

        # Apply boundary conditions (no flux)
        C_free_new[0, :] = C_free_new[1, :]
        C_free_new[-1, :] = C_free_new[-2, :]
        C_free_new[:, 0] = C_free_new[:, 1]
        C_free_new[:, -1] = C_free_new[:, -2]

        # Update solution
        C_free = C_free_new
        current_time += dt

        # Save solution at specified times
        if times_to_save is not None and any(abs(t - current_time) < 0.5*dt for t in times_to_save):
            idx = np.argmin(np.abs(np.array(times_to_save) - current_time))
            t_save = times_to_save[idx]
            saved_solutions[t_save] = {
                'free': C_free.copy(),
                'bound': C_bound.copy(),
                'cellular': C_cell.copy(),
                'total': C_free.copy() + C_bound.copy() + C_cell.copy()
            }
            print(f"Saved solution at t = {t_save} hours")

        # Print progress
        if step % 1000 == 0:
            print(f"Step {step}/{num_steps}, t = {current_time/3600:.2f} hours")

    return {
        'free': C_free,
        'bound': C_bound,
        'cellular': C_cell,
        'total': C_free + C_bound + C_cell
    }, saved_solutions

# Plot tissue structure and vascular parameters
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot tissue map
im0 = axes[0, 0].imshow(tissue_map, origin='lower', extent=[0, L*1e3, 0, L*1e3],
                        cmap='viridis', vmin=0, vmax=2)
axes[0, 0].set_title('Tissue Map')
axes[0, 0].set_xlabel('x (mm)')
axes[0, 0].set_ylabel('y (mm)')
cbar0 = plt.colorbar(im0, ax=axes[0, 0])
cbar0.set_ticks([0, 1, 2])
cbar0.set_ticklabels(['Normal', 'Tumor Rim', 'Tumor Core'])

# Plot vessel density
im1 = axes[0, 1].imshow(vessel_density, origin='lower', extent=[0, L*1e3, 0, L*1e3],
                        cmap='plasma')
axes[0, 1].set_title('Vessel Density (vessels/mm²)')
axes[0, 1].set_xlabel('x (mm)')
axes[0, 1].set_ylabel('y (mm)')
plt.colorbar(im1, ax=axes[0, 1])

# Plot interstitial fluid pressure
im2 = axes[1, 0].imshow(pressure/133.322, origin='lower', extent=[0, L*1e3, 0, L*1e3],
                        cmap='coolwarm')  # Convert Pa to mmHg
axes[1, 0].set_title('Interstitial Fluid Pressure (mmHg)')
axes[1, 0].set_xlabel('x (mm)')
axes[1, 0].set_ylabel('y (mm)')
plt.colorbar(im2, ax=axes[1, 0])

# Plot velocity field (using arrows)
# Downsample for clarity
skip = 5
x_grid = np.linspace(0, L, nx+1)
y_grid = np.linspace(0, L, ny+1)
X, Y = np.meshgrid(x_grid, y_grid)
speed = np.sqrt(velocity_x**2 + velocity_y**2)

axes[1, 1].quiver(X[::skip, ::skip]*1e3, Y[::skip, ::skip]*1e3,
                  velocity_x[::skip, ::skip], velocity_y[::skip, ::skip],
                  speed[::skip, ::skip], cmap='viridis', scale=5e-7)
axes[1, 1].set_title('Interstitial Fluid Velocity')
axes[1, 1].set_xlabel('x (mm)')
axes[1, 1].set_ylabel('y (mm)')

# Add tumor boundary circle
for ax in axes.flat:
    circle = plt.Circle((center_x*1e3, center_y*1e3), R_tumor*1e3,
                        fill=False, color='red', linestyle='--', linewidth=2)
    ax.add_patch(circle)

plt.tight_layout()
plt.savefig(get_result_path('tumor_structure.png', EXAMPLE_NAME))

# Solve for drug transport at different time points
# Modified: Reduced time points to match what the simulation can reach
times_to_save = [1, 2, 6]  # hours
solution, saved_solutions = solve_drug_transport(
    num_steps=50000, dt=0.5, times_to_save=[t*3600 for t in times_to_save])

# Custom colormap for drug concentration
drug_cmap = LinearSegmentedColormap.from_list(
    'drug_concentration', ['#ffffff', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c']
)

# Plot drug concentration at different time points
plt.figure(figsize=(16, 12))

# Get all saved time points
available_times = sorted([t/3600 for t in saved_solutions.keys()])  # Convert to hours
print(f"Available time points (hours): {available_times}")

# Use the available times for plotting
for i, t in enumerate(available_times):
    plt.subplot(2, 3, i+1)

    # Get drug concentration data (t is in hours, saved_solutions keys are in seconds)
    t_seconds = t * 3600
    conc = saved_solutions[t_seconds]['total']

    # Plot concentration
    im = plt.imshow(conc, origin='lower', extent=[0, L*1e3, 0, L*1e3],
                    cmap=drug_cmap, vmin=0, vmax=np.max([np.max(saved_solutions[tt]['total'])
                                                         for tt in saved_solutions.keys()]))

    # Add tumor boundary
    circle = plt.Circle((center_x*1e3, center_y*1e3), R_tumor*1e3,
                        fill=False, color='red', linestyle='--', linewidth=2)
    plt.gca().add_patch(circle)

    plt.title(f't = {t:.1f} hours')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')

    # Stop if we've filled all subplots
    if i >= 5:  # 2x3 grid has 6 spots
        break

plt.tight_layout()
cbar_ax = plt.axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('Total Drug Concentration (normalized)')
plt.savefig(get_result_path('drug_concentration_time.png', EXAMPLE_NAME))

# Plot drug penetration profiles
plt.figure(figsize=(10, 8))

# Extract concentrations along the centerline
centerline_idx_j = ny // 2
r_values = np.array([np.sqrt((mesh.x(i) - center_x)**2 + (mesh.y(0, centerline_idx_j) - center_y)**2)
                     for i in range(nx+1)])

for t_seconds in sorted(saved_solutions.keys()):
    t_hours = t_seconds / 3600
    conc = saved_solutions[t_seconds]['total']
    centerline_conc = conc[centerline_idx_j, :]

    # Sort by distance from tumor center
    sort_idx = np.argsort(r_values)
    r_sorted = r_values[sort_idx]
    conc_sorted = centerline_conc[sort_idx]

    plt.plot(r_sorted*1e3, conc_sorted, label=f't = {t_hours:.1f} hours')

# Add vertical line for tumor boundary
plt.axvline(x=R_tumor*1e3, color='red', linestyle='--', linewidth=2, label='Tumor Boundary')

plt.grid(True)
plt.xlabel('Distance from Tumor Center (mm)')
plt.ylabel('Total Drug Concentration (normalized)')
plt.title('Drug Penetration Profile')
plt.legend()
plt.savefig(get_result_path('drug_penetration.png', EXAMPLE_NAME))

# Plot individual drug compartments at final time point
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Free drug
im0 = axes[0, 0].imshow(solution['free'], origin='lower', extent=[0, L*1e3, 0, L*1e3],
                        cmap=drug_cmap)
axes[0, 0].set_title('Free Drug')
axes[0, 0].set_xlabel('x (mm)')
axes[0, 0].set_ylabel('y (mm)')
plt.colorbar(im0, ax=axes[0, 0])

# Bound drug
im1 = axes[0, 1].imshow(solution['bound'], origin='lower', extent=[0, L*1e3, 0, L*1e3],
                        cmap=drug_cmap)
axes[0, 1].set_title('Tissue-Bound Drug')
axes[0, 1].set_xlabel('x (mm)')
axes[0, 1].set_ylabel('y (mm)')
plt.colorbar(im1, ax=axes[0, 1])

# Cellular drug
im2 = axes[1, 0].imshow(solution['cellular'], origin='lower', extent=[0, L*1e3, 0, L*1e3],
                        cmap=drug_cmap)
axes[1, 0].set_title('Cellular Drug')
axes[1, 0].set_xlabel('x (mm)')
axes[1, 0].set_ylabel('y (mm)')
plt.colorbar(im2, ax=axes[1, 0])

# Total drug
im3 = axes[1, 1].imshow(solution['total'], origin='lower', extent=[0, L*1e3, 0, L*1e3],
                        cmap=drug_cmap)
axes[1, 1].set_title('Total Drug')
axes[1, 1].set_xlabel('x (mm)')
axes[1, 1].set_ylabel('y (mm)')
plt.colorbar(im3, ax=axes[1, 1])

# Add tumor boundary to all plots
for ax in axes.flat:
    circle = plt.Circle((center_x*1e3, center_y*1e3), R_tumor*1e3,
                        fill=False, color='red', linestyle='--', linewidth=2)
    ax.add_patch(circle)

plt.tight_layout()
plt.savefig(get_result_path('drug_compartments.png', EXAMPLE_NAME))

# Show all plots
plt.show()

results_dir = get_result_path('', EXAMPLE_NAME)
print(f"Simulation complete. Results saved to '{results_dir}'.")