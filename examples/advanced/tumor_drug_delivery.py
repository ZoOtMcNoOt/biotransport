"""Tumor drug delivery with interstitial pressure and transport (C++).

This example couples:
- elevated tumor interstitial fluid pressure (IFP)
- Darcy interstitial flow from pressure gradients
- drug convection + diffusion
- vascular source (permeability × vessel density)
- binding + cellular uptake

Units:
- length in meters (plots in mm)
- pressure specified in mmHg (converted to Pa)

Configuration:
    This example supports optional use of TumorDrugDeliveryConfig for parameter
    management. Set USE_CONFIG=True to use the config dataclass, or False to use
    inline parameters (default behavior for backward compatibility).

    Example with config:
        from biotransport import TumorDrugDeliveryConfig
        config = bt.TumorDrugDeliveryConfig(
            domain_size=0.005,
            tumor_radius=0.002,
            D_drug_tumor=1e-11,  # lower diffusivity in dense tumor
        )
        print(config.describe())  # View all parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import biotransport as bt

EXAMPLE_NAME = "tumor_drug_delivery"

# ============================================================================
# Configuration Mode
# ============================================================================
# Set to True to use TumorDrugDeliveryConfig dataclass for parameter management
# Set to False to use inline parameters (backward compatible)
USE_CONFIG = False

if USE_CONFIG:
    # Use configuration dataclass for organized parameter management
    config = bt.TumorDrugDeliveryConfig(
        domain_size=5e-3,
        tumor_radius=2e-3,
        D_drug_normal=5e-11,
        D_drug_tumor=2e-11,
        k_binding=1e-3,
        k_uptake=5e-4,
        MVD_normal=100.0,
        MVD_tumor=200.0,  # Rim value; core computed below
        P_vessel_normal=1e-7,
        P_vessel_tumor=5e-7,
        IFP_normal=0.0,
        IFP_tumor=20.0,
        K_hydraulic_normal=5e-12,
        K_hydraulic_tumor=2.5e-12,
        C_plasma=1.0,
        nx=100,
        ny=100,
    )

    # Extract parameters from config
    L = config.domain_size
    R_tumor = config.tumor_radius
    center_x, center_y = L / 2, L / 2
    D_drug_normal = config.D_drug_normal
    D_drug_tumor = config.D_drug_tumor
    k_binding = config.k_binding
    k_uptake = config.k_uptake
    MVD_normal = config.MVD_normal
    MVD_tumor_rim = config.MVD_tumor
    MVD_tumor_core = 20  # Hypoxic core has fewer vessels
    P_normal = config.P_vessel_normal
    P_tumor = config.P_vessel_tumor
    C_plasma = config.C_plasma
    IFP_normal_pa = config.IFP_normal_Pa
    IFP_tumor_pa = config.IFP_tumor_Pa
    K_normal = config.K_hydraulic_normal
    K_tumor = config.K_hydraulic_tumor
    nx, ny = config.nx, config.ny

    # Display configuration summary
    print("=" * 60)
    print("Tumor Drug Delivery Simulation")
    print("=" * 60)
    print(config.describe())
    print("=" * 60)

else:
    # Inline parameters (original behavior)
    L = 5e-3  # Domain size (m) - 5mm tissue region
    R_tumor = 2e-3  # Tumor radius (m)
    center_x, center_y = L / 2, L / 2  # Tumor center position

    D_drug_normal = 5e-11  # Drug diffusion coefficient in normal tissue (m²/s)
    D_drug_tumor = 2e-11  # Drug diffusion coefficient in tumor (m²/s) - often lower due to dense ECM
    k_binding = 1e-3  # Drug binding rate to tissue (1/s)
    k_uptake = 5e-4  # Cellular uptake rate (1/s)

    MVD_normal = 100  # Microvascular density in normal tissue (vessels/mm²)
    MVD_tumor_core = 20  # Microvascular density in tumor core - hypoxic, fewer vessels
    MVD_tumor_rim = 200  # Microvascular density in tumor periphery - angiogenic rim

    P_normal = 1e-7  # Vessel permeability in normal tissue (m/s)
    P_tumor = 5e-7  # Vessel permeability in tumor (m/s) - enhanced due to leaky vessels
    C_plasma = 1.0  # Normalized drug concentration in plasma

    IFP_normal = 0.0  # Normal tissue IFP (mmHg)
    IFP_tumor = 20.0  # Tumor IFP (mmHg) - elevated in tumors
    IFP_normal_pa = IFP_normal * 133.322
    IFP_tumor_pa = IFP_tumor * 133.322

    K_normal = 5e-12  # Hydraulic conductivity in normal tissue (m²/(Pa·s))
    K_tumor = 2.5e-12  # Hydraulic conductivity in tumor (m²/(Pa·s))

    nx, ny = 100, 100
mesh = bt.StructuredMesh(nx, ny, 0.0, L, 0.0, L)
dx, dy = mesh.dx(), mesh.dy()

rim_size = 0.5e-3
extent_mm = (0, L * 1e3, 0, L * 1e3)


def add_tumor_outline(ax):
    """Draw the tumor boundary on an axes (in mm)."""

    ax.add_patch(
        plt.Circle(
            (center_x * 1e3, center_y * 1e3),
            R_tumor * 1e3,
            fill=False,
            color="red",
            linestyle="--",
            linewidth=2,
        )
    )


# Precompute coordinate grids for fast, vectorized spatial maps
x_coords = bt.x_nodes(mesh)
y_coords = bt.y_nodes(mesh)
X, Y = bt.xy_grid(mesh)
dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

mask_tumor = dist <= R_tumor
mask_rim = mask_tumor & (dist > (R_tumor - rim_size))
mask_core = mask_tumor & ~mask_rim

# Initialize spatial maps (vectorized)
tissue_map = np.zeros((ny + 1, nx + 1), dtype=np.int8)
tissue_map[mask_rim] = 1
tissue_map[mask_core] = 2

vessel_density = np.full((ny + 1, nx + 1), MVD_normal, dtype=np.float64)
vessel_density[mask_rim] = MVD_tumor_rim
vessel_density[mask_core] = MVD_tumor_core

permeability = np.full((ny + 1, nx + 1), P_normal, dtype=np.float64)
permeability[mask_tumor] = P_tumor

diffusivity = np.full((ny + 1, nx + 1), D_drug_normal, dtype=np.float64)
diffusivity[mask_tumor] = D_drug_tumor

K = np.full((ny + 1, nx + 1), K_normal, dtype=np.float64)
K[mask_tumor] = K_tumor

tumor_mask_flat = mask_tumor.astype(np.uint8).ravel(order="C").tolist()
K_flat = K.ravel(order="C").tolist()
solver = bt.TumorDrugDeliverySolver(
    mesh, tumor_mask_flat, K_flat, IFP_normal_pa, IFP_tumor_pa
)

pressure_flat = solver.solve_pressure_sor(max_iter=20000, tol=1e-10, omega=1.8)
pressure = bt.as_2d(mesh, pressure_flat)

pressure_grad_x = np.zeros_like(pressure)
pressure_grad_y = np.zeros_like(pressure)
pressure_grad_x[1:-1, 1:-1] = (pressure[1:-1, 2:] - pressure[1:-1, :-2]) / (2 * dx)
pressure_grad_y[1:-1, 1:-1] = (pressure[2:, 1:-1] - pressure[:-2, 1:-1]) / (2 * dy)

velocity_x = -K * pressure_grad_x
velocity_y = -K * pressure_grad_y


def solve_drug_transport(num_steps=10000, dt=0.1, times_to_save=None):
    """Run the C++ tumor drug delivery simulation."""

    if times_to_save is None:
        times_to_save_s = []
    else:
        times_to_save_s = sorted(times_to_save)

    saved = solver.simulate(
        pressure_flat,
        diffusivity.ravel(order="C").tolist(),
        permeability.ravel(order="C").tolist(),
        vessel_density.ravel(order="C").tolist(),
        k_binding,
        k_uptake,
        C_plasma,
        dt,
        num_steps,
        times_to_save_s,
    )

    free_stack = saved.free()
    bound_stack = saved.bound()
    cell_stack = saved.cellular()
    total_stack = saved.total()

    saved_solutions = {}
    for frame_idx, t_save in enumerate(saved.times_s):
        saved_solutions[float(t_save)] = {
            "free": free_stack[frame_idx].copy(),
            "bound": bound_stack[frame_idx].copy(),
            "cellular": cell_stack[frame_idx].copy(),
            "total": total_stack[frame_idx].copy(),
        }
        print(f"Saved solution at t = {t_save / 3600:.2f} hours")

    if len(saved.times_s) > 0:
        solution = saved_solutions[float(saved.times_s[-1])]
    else:
        solution = {
            "free": np.zeros((ny + 1, nx + 1), dtype=np.float64),
            "bound": np.zeros((ny + 1, nx + 1), dtype=np.float64),
            "cellular": np.zeros((ny + 1, nx + 1), dtype=np.float64),
            "total": np.zeros((ny + 1, nx + 1), dtype=np.float64),
        }

    return solution, saved_solutions


fig, axes = plt.subplots(2, 2, figsize=(16, 14))

im0 = axes[0, 0].imshow(
    tissue_map, origin="lower", extent=extent_mm, cmap="viridis", vmin=0, vmax=2
)
axes[0, 0].set_title("Tissue Map")
axes[0, 0].set_xlabel("x (mm)")
axes[0, 0].set_ylabel("y (mm)")
cbar0 = plt.colorbar(im0, ax=axes[0, 0])
cbar0.set_ticks([0, 1, 2])
cbar0.set_ticklabels(["Normal", "Tumor Rim", "Tumor Core"])

im1 = axes[0, 1].imshow(vessel_density, origin="lower", extent=extent_mm, cmap="plasma")
axes[0, 1].set_title("Vessel Density (vessels/mm²)")
axes[0, 1].set_xlabel("x (mm)")
axes[0, 1].set_ylabel("y (mm)")
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[1, 0].imshow(
    pressure / 133.322, origin="lower", extent=extent_mm, cmap="coolwarm"
)
axes[1, 0].set_title("Interstitial Fluid Pressure (mmHg)")
axes[1, 0].set_xlabel("x (mm)")
axes[1, 0].set_ylabel("y (mm)")
plt.colorbar(im2, ax=axes[1, 0])

skip = 5
speed = np.sqrt(velocity_x**2 + velocity_y**2)
axes[1, 1].quiver(
    X[::skip, ::skip] * 1e3,
    Y[::skip, ::skip] * 1e3,
    velocity_x[::skip, ::skip],
    velocity_y[::skip, ::skip],
    speed[::skip, ::skip],
    cmap="viridis",
    scale=5e-7,
)
axes[1, 1].set_title("Interstitial Fluid Velocity")
axes[1, 1].set_xlabel("x (mm)")
axes[1, 1].set_ylabel("y (mm)")

for ax in axes.flat:
    add_tumor_outline(ax)

plt.tight_layout()
plt.savefig(bt.get_result_path("tumor_structure.png", EXAMPLE_NAME))

times_to_save = [1, 2, 6]  # hours
solution, saved_solutions = solve_drug_transport(
    num_steps=50000, dt=0.5, times_to_save=[t * 3600 for t in times_to_save]
)

# Custom colormap for drug concentration
drug_cmap = LinearSegmentedColormap.from_list(
    "drug_concentration",
    ["#ffffff", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"],
)

plt.figure(figsize=(16, 12))

available_times = sorted([t / 3600 for t in saved_solutions.keys()])  # Convert to hours
print(f"Available time points (hours): {available_times}")
vmax = (
    max(np.max(saved_solutions[t]["total"]) for t in saved_solutions.keys())
    if saved_solutions
    else 1.0
)

im = None
for i, t in enumerate(available_times):
    plt.subplot(2, 3, i + 1)

    # Get drug concentration data (t is in hours, saved_solutions keys are in seconds)
    t_seconds = t * 3600
    conc = saved_solutions[t_seconds]["total"]

    im = plt.imshow(
        conc, origin="lower", extent=extent_mm, cmap=drug_cmap, vmin=0, vmax=vmax
    )
    add_tumor_outline(plt.gca())

    plt.title(f"t = {t:.1f} hours")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")

    # Stop if we've filled all subplots
    if i >= 5:  # 2x3 grid has 6 spots
        break

plt.tight_layout()
cbar_ax = plt.axes((0.92, 0.15, 0.02, 0.7))
if im is not None:
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label("Total Drug Concentration (normalized)")
plt.savefig(bt.get_result_path("drug_concentration_time.png", EXAMPLE_NAME))

# Plot drug penetration profiles
plt.figure(figsize=(10, 8))

centerline_idx_j = ny // 2
y_center = y_coords[centerline_idx_j]
r_values = np.sqrt((x_coords - center_x) ** 2 + (y_center - center_y) ** 2)

for t_seconds in sorted(saved_solutions.keys()):
    t_hours = t_seconds / 3600
    conc = saved_solutions[t_seconds]["total"]
    centerline_conc = conc[centerline_idx_j, :]

    # Sort by distance from tumor center
    sort_idx = np.argsort(r_values)
    r_sorted = r_values[sort_idx]
    conc_sorted = centerline_conc[sort_idx]

    plt.plot(r_sorted * 1e3, conc_sorted, label=f"t = {t_hours:.1f} hours")

plt.axvline(
    x=R_tumor * 1e3, color="red", linestyle="--", linewidth=2, label="Tumor Boundary"
)

plt.grid(True)
plt.xlabel("Distance from Tumor Center (mm)")
plt.ylabel("Total Drug Concentration (normalized)")
plt.title("Drug Penetration Profile")
plt.legend()
plt.savefig(bt.get_result_path("drug_penetration.png", EXAMPLE_NAME))

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

for ax, (key, title) in zip(
    axes.flat,
    [
        ("free", "Free Drug"),
        ("bound", "Tissue-Bound Drug"),
        ("cellular", "Cellular Drug"),
        ("total", "Total Drug"),
    ],
):
    im = ax.imshow(solution[key], origin="lower", extent=extent_mm, cmap=drug_cmap)
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax)

for ax in axes.flat:
    add_tumor_outline(ax)

plt.tight_layout()
plt.savefig(bt.get_result_path("drug_compartments.png", EXAMPLE_NAME))

# Show all plots
plt.show()

results_dir = bt.get_result_path("", EXAMPLE_NAME)
print(f"Simulation complete. Results saved to '{results_dir}'.")
