"""Pennes bioheat cryotherapy example with an apparent-heat-capacity model.

The C++ solver advances the temperature, phase fraction, and Arrhenius
*heat-injury* integral. The latter is intentionally not presented as a
cryogenic cell-death probability: a calibrated tissue-specific cryoinjury law
would require additional biological parameters and validation data.

All solver inputs are SI and all temperatures passed to C++ are kelvin.
The probe is represented by embedded fixed-temperature grid nodes; this
example does not model probe contact resistance or coolant dynamics.
"""

import matplotlib.pyplot as plt
import numpy as np

import biotransport as bt


EXAMPLE_NAME = "bioheat_cryotherapy"

config = bt.BioheatCryotherapyConfig.from_celsius(
    probe_C=-150.0,
    freeze_C=-1.0,
    initial_C=37.0,
    arterial_C=37.0,
    boundary_C=37.0,
    nx=80,
    ny=80,
    dt=0.05,
)
config.validate()
print(config.describe())

mesh = bt.StructuredMesh(
    config.nx,
    config.ny,
    0.0,
    config.domain_size_x,
    0.0,
    config.domain_size_y,
)
x_nodes = bt.x_nodes(mesh)
y_nodes = bt.y_nodes(mesh)
X, Y = bt.xy_grid(mesh)

assert config.probe_position is not None
assert config.tumor_center is not None
probe_x, probe_y = config.probe_position
tumor_x, tumor_y = config.tumor_center

probe_mask = (X - probe_x) ** 2 + (Y - probe_y) ** 2 <= config.probe_radius**2
tumor_mask = (X - tumor_x) ** 2 + (Y - tumor_y) ** 2 <= config.tumor_radius**2
perfusion_map = np.where(tumor_mask, config.w_b_tumor, config.w_b_normal)
q_met_map = np.where(tumor_mask, config.q_met_tumor, config.q_met_normal)

solver = config.create_solver(
    mesh,
    probe_mask=probe_mask.astype(np.uint8).ravel(order="C").tolist(),
    perfusion_map=perfusion_map.astype(np.float64).ravel(order="C").tolist(),
    q_met_map=q_met_map.astype(np.float64).ravel(order="C").tolist(),
)

total_time_s = 300.0
save_times_s = [0.0, 30.0, 120.0, total_time_s]
result = solver.simulate(
    config.dt,
    round(total_time_s / config.dt),
    save_times_s,
)

temperature_K = result.temperature_K()
frozen_fraction = result.frozen_fraction()
arrhenius_damage = result.damage()

extent_mm = [
    0.0,
    config.domain_size_x * 1e3,
    0.0,
    config.domain_size_y * 1e3,
]
x_mm = x_nodes * 1e3
y_mm = y_nodes * 1e3


def add_geometry(ax: plt.Axes) -> None:
    """Overlay the embedded probe and modeled tumor region."""

    ax.add_patch(
        plt.Circle(
            (probe_x * 1e3, probe_y * 1e3),
            config.probe_radius * 1e3,
            color="black",
            alpha=0.75,
        )
    )
    ax.add_patch(
        plt.Circle(
            (tumor_x * 1e3, tumor_y * 1e3),
            config.tumor_radius * 1e3,
            fill=False,
            color="black",
            linestyle="--",
        )
    )


figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
for frame, (time_s, ax) in enumerate(zip(result.times_s, axes.flat)):
    temperature_C = temperature_K[frame] - 273.15
    image = ax.imshow(
        temperature_C,
        origin="lower",
        extent=extent_mm,
        cmap="coolwarm",
        vmin=-80.0,
        vmax=40.0,
    )
    ax.contour(x_mm, y_mm, frozen_fraction[frame], levels=[0.5], colors="white")
    add_geometry(ax)
    ax.set(title=f"Temperature at t = {time_s:g} s", xlabel="x (mm)", ylabel="y (mm)")
figure.colorbar(image, ax=axes, label="Temperature (degC)")
figure.savefig(bt.get_result_path("temperature_evolution.png", EXAMPLE_NAME), dpi=180)

final_temperature_C = temperature_K[-1] - 273.15
final_frozen_fraction = frozen_fraction[-1]
final_heat_injury_probability = -np.expm1(-arrhenius_damage[-1])

figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
fields = (
    (final_temperature_C, "coolwarm", "Temperature (degC)", -80.0, 40.0),
    (final_frozen_fraction, "Blues", "Apparent frozen fraction", 0.0, 1.0),
    (
        final_heat_injury_probability,
        "inferno",
        "Arrhenius heat-injury probability",
        0.0,
        1.0,
    ),
)
for ax, (field, cmap, label, lower, upper) in zip(axes, fields):
    image = ax.imshow(
        field,
        origin="lower",
        extent=extent_mm,
        cmap=cmap,
        vmin=lower,
        vmax=upper,
    )
    add_geometry(ax)
    ax.set(xlabel="x (mm)", ylabel="y (mm)")
    figure.colorbar(image, ax=ax, label=label)
axes[0].set_title("Final temperature")
axes[1].set_title("Phase-change diagnostic")
axes[2].set_title("Heat injury only; not cryoinjury")
figure.savefig(bt.get_result_path("final_diagnostics.png", EXAMPLE_NAME), dpi=180)

tumor_frozen_fraction = float(np.mean(final_frozen_fraction[tumor_mask]))
minimum_temperature_C = float(np.min(final_temperature_C))
print(f"Minimum final temperature: {minimum_temperature_C:.2f} degC")
print(f"Mean apparent frozen fraction in tumor mask: {tumor_frozen_fraction:.3f}")
print(
    "No cryogenic cell-death percentage is reported without a calibrated injury model."
)

plt.show()
