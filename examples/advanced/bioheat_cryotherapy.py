"""
Example of bioheat transfer with tissue cryotherapy.

This advanced example simulates heat transfer in biological tissue during
cryotherapy (freezing treatment) with:
1. Pennes bioheat equation for tissue temperature
2. Blood perfusion cooling/warming
3. Metabolic heat generation
4. Phase change effects during freezing
5. Temperature-dependent tissue properties
6. Cell death model based on thermal history

This is relevant for cryosurgery planning and other thermal treatments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from scipy.special import erf
from biotransport import StructuredMesh
from biotransport.utils import get_result_path

# Create results subdirectory for this example
EXAMPLE_NAME = "bioheat_cryotherapy"

# Physical parameters for tissue
rho_tissue = 1050.0  # Tissue density (kg/m³)
c_tissue_unfrozen = 3600.0  # Specific heat of unfrozen tissue (J/(kg·K))
c_tissue_frozen = 1800.0   # Specific heat of frozen tissue (J/(kg·K))
k_tissue_unfrozen = 0.5  # Thermal conductivity of unfrozen tissue (W/(m·K))
k_tissue_frozen = 2.0   # Thermal conductivity of frozen tissue (W/(m·K))

# Blood perfusion parameters
rho_blood = 1060.0  # Blood density (kg/m³)
c_blood = 3800.0  # Specific heat of blood (J/(kg·K))
w_b_normal = 0.0005  # Blood perfusion rate (1/s) - baseline
w_b_tumor = 0.002   # Blood perfusion rate in tumor (1/s) - often higher

# Cryoprobe parameters
T_probe = -150.0 + 273.15  # Cryoprobe temperature (K)
r_probe = 1.5e-3  # Cryoprobe radius (m)
probe_x, probe_y = 0.025, 0.025  # Probe position (m)

# Metabolic heat generation
q_met_normal = 420.0  # Metabolic heat in normal tissue (W/m³)
q_met_tumor = 840.0   # Metabolic heat in tumor tissue (W/m³)

# Phase change parameters
T_freeze = 273.15 - 1.0  # Freezing temperature of tissue (K)
T_freeze_range = 2.0  # Temperature range over which phase change occurs (K)
L_fusion = 333000.0  # Latent heat of fusion for water (J/kg)

# Cell death model parameters (Arrhenius)
E_a = 2.0e5   # Activation energy (J/mol)
A = 7.39e29   # Frequency factor (1/s)
R_gas = 8.314 # Gas constant (J/(mol·K))

# Simulation parameters
T_body = 310.15  # Body temperature (K) = 37°C
T_ambient = 293.15  # Ambient temperature (K) = 20°C

# Domain parameters
L_x, L_y = 0.05, 0.05  # Domain size (m) - 5cm × 5cm
nx, ny = 100, 100  # Number of grid points

# Tumor parameters
R_tumor = 0.01  # Tumor radius (m)
tumor_x, tumor_y = 0.025, 0.025  # Tumor center position (m)

# Create a 2D mesh
mesh = StructuredMesh(nx, ny, 0.0, L_x, 0.0, L_y)
dx, dy = mesh.dx(), mesh.dy()

# Time stepping parameters
dt = 0.1  # Time step (s)
total_time = 600.0  # Total simulation time (s) = 10 minutes
num_steps = int(total_time / dt)

# Helper functions
def in_tumor(x, y):
    """Check if a point is within the tumor."""
    dist = np.sqrt((x - tumor_x)**2 + (y - tumor_y)**2)
    return dist <= R_tumor

def in_probe(x, y):
    """Check if a point is within the cryoprobe."""
    dist = np.sqrt((x - probe_x)**2 + (y - probe_y)**2)
    return dist <= r_probe

def temperature_dependent_properties(T):
    """
    Calculate temperature-dependent tissue properties.

    Args:
        T: Temperature (K)

    Returns:
        Dictionary of properties
    """
    # Smooth transition for phase change using error function
    phase_fraction = 0.5 * (1 + erf((T_freeze - T) / (T_freeze_range / np.sqrt(2))))

    # Interpolate properties between frozen and unfrozen states
    k = k_tissue_unfrozen * (1 - phase_fraction) + k_tissue_frozen * phase_fraction
    c = c_tissue_unfrozen * (1 - phase_fraction) + c_tissue_frozen * phase_fraction

    # Effective specific heat including latent heat effect
    c_effective = c + L_fusion * rho_tissue * np.exp(-(T - T_freeze)**2 / (2 * (T_freeze_range/2)**2)) / (np.sqrt(2 * np.pi) * (T_freeze_range/2))

    # Blood perfusion decreases in frozen tissue
    w_b_factor = 1.0 - phase_fraction

    return {
        'k': k,
        'c': c,
        'c_effective': c_effective,
        'w_b_factor': w_b_factor
    }

def calculate_cell_death(T, time_at_temp):
    """
    Calculate cell damage using Arrhenius model.

    Args:
        T: Temperature (K)
        time_at_temp: Time spent at each temperature (s)

    Returns:
        Damage integral Ω
    """
    # For temperatures above freezing, use standard Arrhenius equation
    if T > T_freeze:
        return A * time_at_temp * np.exp(-E_a / (R_gas * T))
    else:
        # For freezing, cell death is enhanced - simplified model
        # Real models are more complex with multiple mechanisms
        freezing_factor = 10.0 * (1.0 - T / T_freeze) if T < T_freeze else 0
        return A * time_at_temp * np.exp(-E_a / (R_gas * T)) * (1.0 + freezing_factor)

# Initialize fields
T = np.ones((ny+1, nx+1)) * T_body  # Temperature field (K)
damage = np.zeros((ny+1, nx+1))  # Cell damage field

# Initialize tissue properties
tumor_map = np.zeros((ny+1, nx+1))  # 1 for tumor, 0 for normal tissue
perfusion_map = np.zeros((ny+1, nx+1))  # Blood perfusion map
q_met_map = np.zeros((ny+1, nx+1))  # Metabolic heat map

for j in range(ny+1):
    for i in range(nx+1):
        x, y = mesh.x(i), mesh.y(0, j)

        # Set tumor and probe regions
        if in_tumor(x, y):
            tumor_map[j, i] = 1
            perfusion_map[j, i] = w_b_tumor
            q_met_map[j, i] = q_met_tumor
        else:
            perfusion_map[j, i] = w_b_normal
            q_met_map[j, i] = q_met_normal

        # Set cryoprobe
        if in_probe(x, y):
            T[j, i] = T_probe

# Time stepping loop
times_to_save = [0, 60, 120, 300, 600]  # seconds
saved_results = {0: {'T': T.copy(), 'damage': damage.copy()}}

for step in range(1, num_steps+1):
    current_time = step * dt

    # Create arrays for updated values
    T_new = T.copy()

    # Update interior points
    for j in range(1, ny):
        for i in range(1, nx):
            x, y = mesh.x(i), mesh.y(0, j)

            # Skip probe points (fixed temperature)
            if in_probe(x, y):
                continue

            # Get temperature-dependent properties
            props = temperature_dependent_properties(T[j, i])
            k = props['k']
            c_effective = props['c_effective']
            w_b_factor = props['w_b_factor']

            # Heat diffusion (central difference)
            d2T_dx2 = (T[j, i+1] - 2*T[j, i] + T[j, i-1]) / (dx*dx)
            d2T_dy2 = (T[j+1, i] - 2*T[j, i] + T[j-1, i]) / (dy*dy)
            diffusion = k * (d2T_dx2 + d2T_dy2)

            # Blood perfusion (heat sink/source)
            perfusion = rho_blood * c_blood * perfusion_map[j, i] * w_b_factor * (T_body - T[j, i])

            # Metabolic heat generation
            metabolism = q_met_map[j, i] * w_b_factor  # Metabolic heat decreases when frozen

            # Update temperature
            dT_dt = (diffusion + perfusion + metabolism) / (rho_tissue * c_effective)
            T_new[j, i] = T[j, i] + dt * dT_dt

            # Update cell damage
            damage[j, i] += calculate_cell_death(T[j, i], dt)

    # Apply boundary conditions (fixed temperature at domain boundaries)
    # Dirichlet boundary condition at all edges
    T_new[0, :] = T_body
    T_new[-1, :] = T_body
    T_new[:, 0] = T_body
    T_new[:, -1] = T_body

    # Fix cryoprobe temperature
    for j in range(ny+1):
        for i in range(nx+1):
            if in_probe(mesh.x(i), mesh.y(0, j)):
                T_new[j, i] = T_probe

    # Update temperature field
    T = T_new

    # Save results at specified times
    if current_time in times_to_save:
        saved_results[current_time] = {
            'T': T.copy(),
            'damage': damage.copy()
        }
        print(f"Saved results at t = {current_time} seconds")

    # Print progress
    if step % 100 == 0:
        print(f"Step {step}/{num_steps}, t = {current_time:.1f} seconds")

# Create visualizations
# Custom temperature colormap
temp_cmap = LinearSegmentedColormap.from_list(
    'temperature',
    [(0, '#313695'),   # Dark blue (very cold)
     (0.25, '#4575b4'), # Blue (cold)
     (0.4, '#74add1'),  # Light blue (cool)
     (0.5, '#abd9e9'),  # Very light blue
     (0.55, '#fee090'), # Light orange
     (0.75, '#fdae61'), # Orange
     (0.9, '#f46d43'),  # Dark orange
     (1.0, '#d73027')]  # Red (very warm)
)

# Cell damage visualization
damage_cmap = LinearSegmentedColormap.from_list(
    'cell_damage',
    [(0, '#ffffff'),     # White (no damage)
     (0.5, '#fee8c8'),   # Light orange (some damage)
     (0.8, '#fdbb84'),   # Orange (moderate damage)
     (0.9, '#e34a33'),   # Red (severe damage)
     (1.0, '#b30000')]   # Dark red (complete necrosis)
)

# Plot temperature evolution
plt.figure(figsize=(15, 10))

for i, t in enumerate(times_to_save):
    plt.subplot(2, 3, i+1)

    # Temperature in °C for display
    T_celsius = saved_results[t]['T'] - 273.15

    # Plot temperature
    im = plt.imshow(T_celsius, origin='lower', extent=[0, L_x*1000, 0, L_y*1000],
                    cmap=temp_cmap, vmin=-60, vmax=40)

    # Add contours for freezing front
    contour = plt.contour(np.linspace(0, L_x*1000, nx+1),
                          np.linspace(0, L_y*1000, ny+1),
                          T_celsius, levels=[0], colors='white', linewidths=2)

    # Add probe and tumor boundaries
    probe_circle = plt.Circle((probe_x*1000, probe_y*1000), r_probe*1000,
                              fill=True, color='black', alpha=0.7)
    tumor_circle = plt.Circle((tumor_x*1000, tumor_y*1000), R_tumor*1000,
                              fill=False, color='black', linestyle='--', linewidth=2)
    plt.gca().add_patch(probe_circle)
    plt.gca().add_patch(tumor_circle)

    plt.title(f't = {t} seconds')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')

plt.tight_layout()
cbar_ax = plt.axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('Temperature (°C)')
plt.savefig(get_result_path('temperature_evolution.png', EXAMPLE_NAME))

# Plot cell damage
plt.figure(figsize=(15, 10))

for i, t in enumerate(times_to_save):
    plt.subplot(2, 3, i+1)

    # Cell survival fraction (convert damage to probability)
    # Ω = 1 corresponds to 63% probability of cell death, Ω = 4 to 98%
    survival = np.exp(-saved_results[t]['damage'])
    death_probability = 1 - survival

    # Plot cell damage
    im = plt.imshow(death_probability, origin='lower',
                    extent=[0, L_x*1000, 0, L_y*1000],
                    cmap=damage_cmap, vmin=0, vmax=1)

    # Add contours for different damage thresholds
    contour = plt.contour(np.linspace(0, L_x*1000, nx+1),
                          np.linspace(0, L_y*1000, ny+1),
                          death_probability,
                          levels=[0.63, 0.95], colors=['black', 'red'],
                          linewidths=2, linestyles=['--', '-'])
    plt.clabel(contour, inline=1, fontsize=10, fmt='%.2f')

    # Add probe and tumor boundaries
    probe_circle = plt.Circle((probe_x*1000, probe_y*1000), r_probe*1000,
                              fill=True, color='black', alpha=0.7)
    tumor_circle = plt.Circle((tumor_x*1000, tumor_y*1000), R_tumor*1000,
                              fill=False, color='black', linestyle='--', linewidth=2)
    plt.gca().add_patch(probe_circle)
    plt.gca().add_patch(tumor_circle)

    plt.title(f't = {t} seconds')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')

plt.tight_layout()
cbar_ax = plt.axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('Cell Death Probability')
plt.savefig(get_result_path('cell_damage.png', EXAMPLE_NAME))

# Plot final temperature profiles
plt.figure(figsize=(10, 8))

# Plot temperature along x-axis through probe center
j_center = int(probe_y / dy)
x_values = np.array([mesh.x(i) for i in range(nx+1)])
T_profile_x = saved_results[total_time]['T'][j_center, :]

plt.plot(x_values*1000, T_profile_x - 273.15, 'b-', linewidth=2, label='Final Temperature')

# Plot initial temperature
T_initial = np.ones_like(T_profile_x) * T_body
plt.plot(x_values*1000, T_initial - 273.15, 'r--', linewidth=2, label='Initial Temperature')

# Add vertical lines for probe and tumor boundaries
plt.axvline(x=(probe_x - r_probe)*1000, color='black', linestyle='-', linewidth=1)
plt.axvline(x=(probe_x + r_probe)*1000, color='black', linestyle='-', linewidth=1)
plt.axvline(x=(tumor_x - R_tumor)*1000, color='black', linestyle='--', linewidth=1)
plt.axvline(x=(tumor_x + R_tumor)*1000, color='black', linestyle='--', linewidth=1)

# Add horizontal line for freezing temperature
plt.axhline(y=T_freeze - 273.15, color='blue', linestyle='--', linewidth=1, label='Freezing Temperature')

plt.grid(True)
plt.xlabel('x (mm)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Profile Along x-axis')
plt.legend()
plt.savefig(get_result_path('temperature_profile.png', EXAMPLE_NAME))

# Plot tissue properties vs temperature
plt.figure(figsize=(12, 10))

# Temperature range for plotting
T_range = np.linspace(T_probe, T_body, 200)
k_values = []
c_values = []
c_eff_values = []
w_b_factor_values = []

for T_val in T_range:
    props = temperature_dependent_properties(T_val)
    k_values.append(props['k'])
    c_values.append(props['c'])
    c_eff_values.append(props['c_effective'])
    w_b_factor_values.append(props['w_b_factor'])

# Plot thermal conductivity
plt.subplot(2, 2, 1)
plt.plot(T_range - 273.15, k_values, 'b-', linewidth=2)
plt.grid(True)
plt.xlabel('Temperature (°C)')
plt.ylabel('Thermal Conductivity (W/(m·K))')
plt.title('Thermal Conductivity vs Temperature')
plt.axvline(x=T_freeze - 273.15, color='black', linestyle='--')

# Plot specific heat
plt.subplot(2, 2, 2)
plt.plot(T_range - 273.15, c_values, 'g-', linewidth=2)
plt.grid(True)
plt.xlabel('Temperature (°C)')
plt.ylabel('Specific Heat (J/(kg·K))')
plt.title('Specific Heat vs Temperature')
plt.axvline(x=T_freeze - 273.15, color='black', linestyle='--')

# Plot effective specific heat (with phase change)
plt.subplot(2, 2, 3)
plt.plot(T_range - 273.15, c_eff_values, 'r-', linewidth=2)
plt.grid(True)
plt.xlabel('Temperature (°C)')
plt.ylabel('Effective Specific Heat (J/(kg·K))')
plt.title('Effective Specific Heat vs Temperature')
plt.axvline(x=T_freeze - 273.15, color='black', linestyle='--')

# Plot blood perfusion factor
plt.subplot(2, 2, 4)
plt.plot(T_range - 273.15, w_b_factor_values, 'm-', linewidth=2)
plt.grid(True)
plt.xlabel('Temperature (°C)')
plt.ylabel('Blood Perfusion Factor')
plt.title('Blood Perfusion Factor vs Temperature')
plt.axvline(x=T_freeze - 273.15, color='black', linestyle='--')

plt.tight_layout()
plt.savefig(get_result_path('tissue_properties.png', EXAMPLE_NAME))

# Calculate treatment effectiveness
final_damage = saved_results[total_time]['damage']
tumor_death_fraction = 0.0
normal_death_fraction = 0.0
tumor_count = 0
normal_count = 0

for j in range(ny+1):
    for i in range(nx+1):
        if in_tumor(mesh.x(i), mesh.y(0, j)):
            tumor_count += 1
            tumor_death_fraction += 1 - np.exp(-final_damage[j, i])
        else:
            normal_count += 1
            normal_death_fraction += 1 - np.exp(-final_damage[j, i])

tumor_death_fraction /= max(tumor_count, 1)
normal_death_fraction /= max(normal_count, 1)

# Plot treatment statistics
plt.figure(figsize=(10, 6))
plt.bar(['Tumor Tissue', 'Normal Tissue'],
        [tumor_death_fraction * 100, normal_death_fraction * 100],
        color=['red', 'blue'])
plt.ylabel('Cell Death Percentage (%)')
plt.title('Treatment Effectiveness')
plt.grid(True, axis='y')
plt.ylim(0, 100)

# Add text labels
plt.text(0, tumor_death_fraction * 100 + 2, f'{tumor_death_fraction*100:.1f}%',
         ha='center', va='bottom')
plt.text(1, normal_death_fraction * 100 + 2, f'{normal_death_fraction*100:.1f}%',
         ha='center', va='bottom')

plt.tight_layout()
plt.savefig(get_result_path('treatment_effectiveness.png', EXAMPLE_NAME))

# Calculate isotherms and cell death contours at final time
T_final = saved_results[total_time]['T'] - 273.15  # Convert to Celsius
damage_final = saved_results[total_time]['damage']
cell_death_prob = 1 - np.exp(-damage_final)

plt.figure(figsize=(12, 10))

# Plot final temperature field
plt.subplot(2, 1, 1)
im1 = plt.imshow(T_final, origin='lower', extent=[0, L_x*1000, 0, L_y*1000],
                 cmap=temp_cmap, vmin=-60, vmax=40)
plt.colorbar(im1, label='Temperature (°C)')

# Add isotherms
contour_levels = [-40, -20, -10, -5, 0, 10, 20, 30]
contour1 = plt.contour(np.linspace(0, L_x*1000, nx+1),
                       np.linspace(0, L_y*1000, ny+1),
                       T_final, levels=contour_levels, colors='black', linewidths=1)
plt.clabel(contour1, inline=1, fontsize=8, fmt='%d°C')

# Add probe and tumor boundaries
probe_circle = plt.Circle((probe_x*1000, probe_y*1000), r_probe*1000,
                          fill=True, color='black', alpha=0.7)
tumor_circle = plt.Circle((tumor_x*1000, tumor_y*1000), R_tumor*1000,
                          fill=False, color='black', linestyle='--', linewidth=2)
plt.gca().add_patch(probe_circle)
plt.gca().add_patch(tumor_circle)

plt.title('Final Temperature Field')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')

# Plot cell death probability
plt.subplot(2, 1, 2)
im2 = plt.imshow(cell_death_prob, origin='lower', extent=[0, L_x*1000, 0, L_y*1000],
                 cmap=damage_cmap, vmin=0, vmax=1)
plt.colorbar(im2, label='Cell Death Probability')

# Add cell death contours
damage_levels = [0.5, 0.63, 0.8, 0.9, 0.95, 0.99]
contour2 = plt.contour(np.linspace(0, L_x*1000, nx+1),
                       np.linspace(0, L_y*1000, ny+1),
                       cell_death_prob, levels=damage_levels,
                       colors='black', linewidths=1)
plt.clabel(contour2, inline=1, fontsize=8, fmt='%.2f')

# Add probe and tumor boundaries
probe_circle = plt.Circle((probe_x*1000, probe_y*1000), r_probe*1000,
                          fill=True, color='black', alpha=0.7)
tumor_circle = plt.Circle((tumor_x*1000, tumor_y*1000), R_tumor*1000,
                          fill=False, color='black', linestyle='--', linewidth=2)
plt.gca().add_patch(probe_circle)
plt.gca().add_patch(tumor_circle)

plt.title('Cell Death Probability')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')

plt.tight_layout()
plt.savefig(get_result_path('final_results.png', EXAMPLE_NAME))

# Show all plots
plt.show()

results_dir = get_result_path('', EXAMPLE_NAME)
print(f"Simulation complete. Results saved to '{results_dir}'.")
print(f"Treatment effectiveness: {tumor_death_fraction*100:.1f}% tumor cell death, {normal_death_fraction*100:.1f}% normal tissue damage.")