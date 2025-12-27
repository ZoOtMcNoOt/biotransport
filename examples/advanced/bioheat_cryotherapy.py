"""Bioheat cryotherapy with phase change + Arrhenius damage (C++).

This example models a simplified cryotherapy treatment via:
- Pennes bioheat (conduction + perfusion + metabolic heat)
- phase change (freezing) via an effective heat capacity
- an Arrhenius-type damage integral converted to death probability

Notes:
- Units are SI internally (m, s, K, W/m^3, ...), plots are in mm and °C.
- The Python code is primarily setup + plotting; the time stepping is in C++.

Configuration:
    This example supports optional use of BioheatCryotherapyConfig for parameter
    management. Set USE_CONFIG=True to use the config dataclass, or False to use
    inline parameters (default behavior for backward compatibility).

    Example with config:
        from biotransport import BioheatCryotherapyConfig
        config = bt.BioheatCryotherapyConfig(
            T_probe=-180.0,
            cooling_rate=100.0,
            blood_perfusion_rate=0.001,
        )
        print(config.describe())  # View all parameters
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import biotransport as bt

EXAMPLE_NAME = "bioheat_cryotherapy"

# ============================================================================
# Configuration Mode
# ============================================================================
# Set to True to use BioheatCryotherapyConfig dataclass for parameter management
# Set to False to use inline parameters (backward compatible)
USE_CONFIG = False

if USE_CONFIG:
    # Use configuration dataclass for organized parameter management
    config = bt.BioheatCryotherapyConfig(
        domain_length=0.05,
        domain_width=0.05,
        probe_x=0.025,
        probe_y=0.025,
        probe_radius=0.0015,
        rho_tissue=1050.0,
        c_tissue=3600.0,
        k_tissue_unfrozen=0.5,
        k_tissue_frozen=2.0,
        blood_perfusion_rate=0.0005,
        rho_blood=1060.0,
        c_blood=3800.0,
        T_arterial=37.0,
        Q_metabolic=420.0,
        T_probe=-150.0,
        cooling_rate=50.0,
        T_freeze_start=-1.0,
        T_freeze_end=-3.0,
        latent_heat=333000.0,
        T_initial=37.0,
        T_boundary=37.0,
        nx=100,
        ny=100,
        t_end=600.0,
        dt=0.1,
    )

    # Extract parameters from config
    rho_tissue = config.rho_tissue
    c_tissue_unfrozen = config.c_tissue
    c_tissue_frozen = 1800.0  # Frozen tissue specific heat (hardcoded ratio)
    k_tissue_unfrozen = config.k_tissue_unfrozen
    k_tissue_frozen = config.k_tissue_frozen
    rho_blood = config.rho_blood
    c_blood = config.c_blood
    w_b_normal = config.blood_perfusion_rate
    w_b_tumor = 0.002  # Tumor perfusion (often higher)
    T_probe = config.T_probe_kelvin
    r_probe = config.probe_radius
    probe_x, probe_y = config.probe_x, config.probe_y
    q_met_normal = config.Q_metabolic
    q_met_tumor = 840.0  # Higher metabolic heat in tumor
    T_freeze = 273.15 + config.T_freeze_start
    T_freeze_range = abs(config.T_freeze_start - config.T_freeze_end)
    L_fusion = config.latent_heat
    E_a = config.E_a_damage
    A = config.A_damage
    R_gas = 8.314
    T_body = config.T_initial + 273.15
    T_ambient = 293.15
    L_x, L_y = config.domain_length, config.domain_width
    nx, ny = config.nx, config.ny
    dt = config.dt
    total_time = config.t_end
    num_steps = int(total_time / dt)

    # Display configuration summary
    print("=" * 60)
    print("Bioheat Cryotherapy Simulation")
    print("=" * 60)
    print(config.describe())
    print("=" * 60)

else:
    # Physical parameters for tissue
    rho_tissue = 1050.0  # Tissue density (kg/m³)
    c_tissue_unfrozen = 3600.0  # Specific heat of unfrozen tissue (J/(kg·K))
    c_tissue_frozen = 1800.0  # Specific heat of frozen tissue (J/(kg·K))
    k_tissue_unfrozen = 0.5  # Thermal conductivity of unfrozen tissue (W/(m·K))
    k_tissue_frozen = 2.0  # Thermal conductivity of frozen tissue (W/(m·K))

    # Blood perfusion parameters
    rho_blood = 1060.0  # Blood density (kg/m³)
    c_blood = 3800.0  # Specific heat of blood (J/(kg·K))
    w_b_normal = 0.0005  # Blood perfusion rate (1/s) - baseline
    w_b_tumor = 0.002  # Blood perfusion rate in tumor (1/s) - often higher

    # Cryoprobe parameters
    T_probe = -150.0 + 273.15  # Cryoprobe temperature (K)
    r_probe = 1.5e-3  # Cryoprobe radius (m)
    probe_x, probe_y = 0.025, 0.025  # Probe position (m)

    # Metabolic heat generation
    q_met_normal = 420.0  # Metabolic heat in normal tissue (W/m³)
    q_met_tumor = 840.0  # Metabolic heat in tumor tissue (W/m³)

    # Phase change parameters
    T_freeze = 273.15 - 1.0  # Freezing temperature of tissue (K)
    T_freeze_range = 2.0  # Temperature range over which phase change occurs (K)
    L_fusion = 333000.0  # Latent heat of fusion for water (J/kg)

    # Cell death model parameters (Arrhenius)
    E_a = 2.0e5  # Activation energy (J/mol)
    A = 7.39e29  # Frequency factor (1/s)
    R_gas = 8.314  # Gas constant (J/(mol·K))

    # Simulation parameters
    T_body = 310.15  # Body temperature (K) = 37°C
    T_ambient = 293.15  # Ambient temperature (K) = 20°C

    # Domain parameters
    L_x, L_y = 0.05, 0.05  # Domain size (m) - 5cm × 5cm
    nx, ny = 100, 100  # Number of grid points

    # Time stepping parameters
    dt = 0.1  # Time step (s)
    total_time = 600.0  # Total simulation time (s) = 10 minutes
    num_steps = int(total_time / dt)

# Tumor parameters (shared between both modes)
R_tumor = 0.01  # Tumor radius (m)
tumor_x, tumor_y = 0.025, 0.025  # Tumor center position (m)

# Create a 2D mesh
mesh = bt.StructuredMesh(nx, ny, 0.0, L_x, 0.0, L_y)
dx, dy = mesh.dx(), mesh.dy()
dx2 = dx * dx
dy2 = dy * dy

# Time stepping parameters
dt = 0.1  # Time step (s)
total_time = 600.0  # Total simulation time (s) = 10 minutes
num_steps = int(total_time / dt)

# Precompute coordinate grids + region masks for performance
x_coords = bt.x_nodes(mesh)
y_coords = bt.y_nodes(mesh)
X, Y = bt.xy_grid(mesh)

x_mm = x_coords * 1e3
y_mm = y_coords * 1e3
extent_mm = [0, L_x * 1e3, 0, L_y * 1e3]


def add_probe_and_tumor_overlays(ax):
    """Add probe (filled) and tumor (dashed) outlines to an axes (in mm)."""

    ax.add_patch(
        plt.Circle(
            (probe_x * 1e3, probe_y * 1e3),
            r_probe * 1e3,
            fill=True,
            color="black",
            alpha=0.7,
        )
    )
    ax.add_patch(
        plt.Circle(
            (tumor_x * 1e3, tumor_y * 1e3),
            R_tumor * 1e3,
            fill=False,
            color="black",
            linestyle="--",
            linewidth=2,
        )
    )


tumor_mask = ((X - tumor_x) ** 2 + (Y - tumor_y) ** 2) <= (R_tumor**2)
probe_mask = ((X - probe_x) ** 2 + (Y - probe_y) ** 2) <= (r_probe**2)


def temperature_dependent_properties(T_val):
    """Scalar (non-vectorized) helper for plotting tissue properties vs temperature."""

    # Smooth transition for phase change using error function
    phase_fraction = 0.5 * (
        1.0 + math.erf((T_freeze - float(T_val)) / (T_freeze_range / math.sqrt(2.0)))
    )

    k = k_tissue_unfrozen * (1.0 - phase_fraction) + k_tissue_frozen * phase_fraction
    c = c_tissue_unfrozen * (1.0 - phase_fraction) + c_tissue_frozen * phase_fraction

    sigma = T_freeze_range / 2.0
    z = (float(T_val) - T_freeze) / sigma
    c_effective = c + (L_fusion * rho_tissue) * math.exp(-0.5 * z * z) / (
        math.sqrt(2.0 * math.pi) * sigma
    )

    w_b_factor = 1.0 - phase_fraction
    return {
        "k": k,
        "c": c,
        "c_effective": c_effective,
        "w_b_factor": w_b_factor,
    }


# Tissue property maps
perfusion_map = np.where(tumor_mask, w_b_tumor, w_b_normal).astype(np.float64)
q_met_map = np.where(tumor_mask, q_met_tumor, q_met_normal).astype(np.float64)

times_to_save = [0, 60, 120, 300, 600]  # seconds

solver = bt.BioheatCryotherapySolver(
    mesh,
    probe_mask.astype(np.uint8).ravel(order="C").tolist(),
    perfusion_map.ravel(order="C").tolist(),
    q_met_map.ravel(order="C").tolist(),
    rho_tissue,
    rho_blood,
    c_blood,
    k_tissue_unfrozen,
    k_tissue_frozen,
    c_tissue_unfrozen,
    c_tissue_frozen,
    T_body,
    T_probe,
    T_freeze,
    T_freeze_range,
    L_fusion,
    A,
    E_a,
    R_gas,
)

saved = solver.simulate(dt, num_steps, [float(t) for t in times_to_save])
T_stack = saved.temperature_K()
damage_stack = saved.damage()

saved_results = {}
for frame_idx, t_save in enumerate(saved.times_s):
    t_key = int(round(t_save))
    saved_results[t_key] = {
        "T": T_stack[frame_idx].copy(),
        "damage": damage_stack[frame_idx].copy(),
    }
    print(f"Saved results at t = {t_key} seconds")

T = saved_results[times_to_save[-1]]["T"]
damage = saved_results[times_to_save[-1]]["damage"]

# Custom temperature colormap
temp_cmap = LinearSegmentedColormap.from_list(
    "temperature",
    [
        (0, "#313695"),  # Dark blue (very cold)
        (0.25, "#4575b4"),  # Blue (cold)
        (0.4, "#74add1"),  # Light blue (cool)
        (0.5, "#abd9e9"),  # Very light blue
        (0.55, "#fee090"),  # Light orange
        (0.75, "#fdae61"),  # Orange
        (0.9, "#f46d43"),  # Dark orange
        (1.0, "#d73027"),
    ],  # Red (very warm)
)

# Cell damage visualization
damage_cmap = LinearSegmentedColormap.from_list(
    "cell_damage",
    [
        (0, "#ffffff"),  # White (no damage)
        (0.5, "#fee8c8"),  # Light orange (some damage)
        (0.8, "#fdbb84"),  # Orange (moderate damage)
        (0.9, "#e34a33"),  # Red (severe damage)
        (1.0, "#b30000"),
    ],  # Dark red (complete necrosis)
)

plt.figure(figsize=(15, 10))

for i, t in enumerate(times_to_save):
    plt.subplot(2, 3, i + 1)

    T_celsius = saved_results[t]["T"] - 273.15

    im = plt.imshow(
        T_celsius, origin="lower", extent=extent_mm, cmap=temp_cmap, vmin=-60, vmax=40
    )

    plt.contour(x_mm, y_mm, T_celsius, levels=[0], colors="white", linewidths=2)

    add_probe_and_tumor_overlays(plt.gca())

    plt.title(f"t = {t} seconds")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")

plt.tight_layout()
cbar_ax = plt.axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label("Temperature (°C)")
plt.savefig(bt.get_result_path("temperature_evolution.png", EXAMPLE_NAME))

plt.figure(figsize=(15, 10))

for i, t in enumerate(times_to_save):
    plt.subplot(2, 3, i + 1)

    survival = np.exp(-saved_results[t]["damage"])
    death_probability = 1 - survival

    im = plt.imshow(
        death_probability,
        origin="lower",
        extent=extent_mm,
        cmap=damage_cmap,
        vmin=0,
        vmax=1,
    )

    contour = plt.contour(
        x_mm,
        y_mm,
        death_probability,
        levels=[0.63, 0.95],
        colors=["black", "red"],
        linewidths=2,
        linestyles=["--", "-"],
    )
    plt.clabel(contour, inline=1, fontsize=10, fmt="%.2f")

    add_probe_and_tumor_overlays(plt.gca())

    plt.title(f"t = {t} seconds")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")

plt.tight_layout()
cbar_ax = plt.axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label("Cell Death Probability")
plt.savefig(bt.get_result_path("cell_damage.png", EXAMPLE_NAME))

plt.figure(figsize=(10, 8))

# Plot temperature along x-axis through probe center
j_center = int(probe_y / dy)
x_values = bt.x_nodes(mesh)
T_profile_x = saved_results[total_time]["T"][j_center, :]

plt.plot(
    x_values * 1000, T_profile_x - 273.15, "b-", linewidth=2, label="Final Temperature"
)

# Plot initial temperature
T_initial = np.ones_like(T_profile_x) * T_body
plt.plot(
    x_values * 1000, T_initial - 273.15, "r--", linewidth=2, label="Initial Temperature"
)

# Add vertical lines for probe and tumor boundaries
plt.axvline(x=(probe_x - r_probe) * 1000, color="black", linestyle="-", linewidth=1)
plt.axvline(x=(probe_x + r_probe) * 1000, color="black", linestyle="-", linewidth=1)
plt.axvline(x=(tumor_x - R_tumor) * 1000, color="black", linestyle="--", linewidth=1)
plt.axvline(x=(tumor_x + R_tumor) * 1000, color="black", linestyle="--", linewidth=1)

# Add horizontal line for freezing temperature
plt.axhline(
    y=T_freeze - 273.15,
    color="blue",
    linestyle="--",
    linewidth=1,
    label="Freezing Temperature",
)

plt.grid(True)
plt.xlabel("x (mm)")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Profile Along x-axis")
plt.legend()
plt.savefig(bt.get_result_path("temperature_profile.png", EXAMPLE_NAME))

plt.figure(figsize=(12, 10))

# Temperature range for plotting
T_range = np.linspace(T_probe, T_body, 200)
props = [temperature_dependent_properties(T_val) for T_val in T_range]
k_values = [p["k"] for p in props]
c_values = [p["c"] for p in props]
c_eff_values = [p["c_effective"] for p in props]
w_b_factor_values = [p["w_b_factor"] for p in props]

# Plot thermal conductivity
plt.subplot(2, 2, 1)
plt.plot(T_range - 273.15, k_values, "b-", linewidth=2)
plt.grid(True)
plt.xlabel("Temperature (°C)")
plt.ylabel("Thermal Conductivity (W/(m·K))")
plt.title("Thermal Conductivity vs Temperature")
plt.axvline(x=T_freeze - 273.15, color="black", linestyle="--")

# Plot specific heat
plt.subplot(2, 2, 2)
plt.plot(T_range - 273.15, c_values, "g-", linewidth=2)
plt.grid(True)
plt.xlabel("Temperature (°C)")
plt.ylabel("Specific Heat (J/(kg·K))")
plt.title("Specific Heat vs Temperature")
plt.axvline(x=T_freeze - 273.15, color="black", linestyle="--")

# Plot effective specific heat (with phase change)
plt.subplot(2, 2, 3)
plt.plot(T_range - 273.15, c_eff_values, "r-", linewidth=2)
plt.grid(True)
plt.xlabel("Temperature (°C)")
plt.ylabel("Effective Specific Heat (J/(kg·K))")
plt.title("Effective Specific Heat vs Temperature")
plt.axvline(x=T_freeze - 273.15, color="black", linestyle="--")

# Plot blood perfusion factor
plt.subplot(2, 2, 4)
plt.plot(T_range - 273.15, w_b_factor_values, "m-", linewidth=2)
plt.grid(True)
plt.xlabel("Temperature (°C)")
plt.ylabel("Blood Perfusion Factor")
plt.title("Blood Perfusion Factor vs Temperature")
plt.axvline(x=T_freeze - 273.15, color="black", linestyle="--")

plt.tight_layout()
plt.savefig(bt.get_result_path("tissue_properties.png", EXAMPLE_NAME))

# Calculate treatment effectiveness
final_damage = saved_results[total_time]["damage"]
death_prob = 1.0 - np.exp(-final_damage)
tumor_death_fraction = (
    float(np.mean(death_prob[tumor_mask])) if np.any(tumor_mask) else 0.0
)
normal_death_fraction = (
    float(np.mean(death_prob[~tumor_mask])) if np.any(~tumor_mask) else 0.0
)

# Plot treatment statistics
plt.figure(figsize=(10, 6))
plt.bar(
    ["Tumor Tissue", "Normal Tissue"],
    [tumor_death_fraction * 100, normal_death_fraction * 100],
    color=["red", "blue"],
)
plt.ylabel("Cell Death Percentage (%)")
plt.title("Treatment Effectiveness")
plt.grid(True, axis="y")
plt.ylim(0, 100)

# Add text labels
plt.text(
    0,
    tumor_death_fraction * 100 + 2,
    f"{tumor_death_fraction * 100:.1f}%",
    ha="center",
    va="bottom",
)
plt.text(
    1,
    normal_death_fraction * 100 + 2,
    f"{normal_death_fraction * 100:.1f}%",
    ha="center",
    va="bottom",
)

plt.tight_layout()
plt.savefig(bt.get_result_path("treatment_effectiveness.png", EXAMPLE_NAME))

T_final = saved_results[total_time]["T"] - 273.15  # Convert to Celsius
damage_final = saved_results[total_time]["damage"]
cell_death_prob = 1 - np.exp(-damage_final)

plt.figure(figsize=(12, 10))

# Plot final temperature field
plt.subplot(2, 1, 1)
im1 = plt.imshow(
    T_final,
    origin="lower",
    extent=[0, L_x * 1000, 0, L_y * 1000],
    cmap=temp_cmap,
    vmin=-60,
    vmax=40,
)
plt.colorbar(im1, label="Temperature (°C)")

# Add isotherms
contour_levels = [-40, -20, -10, -5, 0, 10, 20, 30]
contour1 = plt.contour(
    x_mm, y_mm, T_final, levels=contour_levels, colors="black", linewidths=1
)
plt.clabel(contour1, inline=1, fontsize=8, fmt="%d°C")

add_probe_and_tumor_overlays(plt.gca())

plt.title("Final Temperature Field")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")

# Plot cell death probability
plt.subplot(2, 1, 2)
im2 = plt.imshow(
    cell_death_prob,
    origin="lower",
    extent=[0, L_x * 1000, 0, L_y * 1000],
    cmap=damage_cmap,
    vmin=0,
    vmax=1,
)
plt.colorbar(im2, label="Cell Death Probability")

# Add cell death contours
damage_levels = [0.5, 0.63, 0.8, 0.9, 0.95, 0.99]
contour2 = plt.contour(
    x_mm, y_mm, cell_death_prob, levels=damage_levels, colors="black", linewidths=1
)
plt.clabel(contour2, inline=1, fontsize=8, fmt="%.2f")

add_probe_and_tumor_overlays(plt.gca())

plt.title("Cell Death Probability")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")

plt.tight_layout()
plt.savefig(bt.get_result_path("final_results.png", EXAMPLE_NAME))

# Show all plots
plt.show()

results_dir = bt.get_result_path("", EXAMPLE_NAME)
print(f"Simulation complete. Results saved to '{results_dir}'.")
print(
    f"Treatment effectiveness: {tumor_death_fraction * 100:.1f}% tumor cell death, {normal_death_fraction * 100:.1f}% normal tissue damage."
)
