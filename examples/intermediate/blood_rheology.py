#!/usr/bin/env python3
"""Blood rheology with explicit constitutive-model scope.

This example compares generalized-Newtonian viscosity laws and the library's
bounded hematocrit helpers. It does *not* model red-cell migration,
viscoelasticity, thixotropy, patient-specific plasma chemistry, or the
Fahraeus-Lindqvist effect. Those omissions matter in small vessels and at very
low shear rates.

Parameter sources
-----------------
- Cho & Kensey, Biorheology 28 (1991), doi:10.3233/BIR-1991-283-415
  for the widely reused reference Carreau fit. The helper anchors this fit at
  45% hematocrit as an explicitly documented educational surrogate.
- Mouza et al., Fluids 3 (2018) 75, doi:10.3390/fluids3040075
  for the Merrill hematocrit parameterization used by ``blood_casson_model``.

All calculations use SI units: Pa, Pa s, m, s, and m^3/s.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import biotransport as bt


def example_constitutive_curves() -> None:
    """Compare viscosity and signed constitutive stress over a shear-rate sweep."""
    shear_rates = np.logspace(-1, 3, 160)
    models = {
        "Newtonian (3.45 mPa s)": bt.NewtonianModel(0.00345),
        "Power law (K=0.017, n=0.7)": bt.PowerLawModel(0.017, 0.7),
        "Carreau (45% Hct reference)": bt.blood_carreau_model(0.45),
        "Casson (45% Hct correlation)": bt.blood_casson_model(0.45),
    }

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, model in models.items():
        viscosity = np.array([model.viscosity(rate) for rate in shear_rates])
        stress = np.array([model.shear_stress(rate) for rate in shear_rates])
        axes[0].loglog(shear_rates, 1e3 * viscosity, linewidth=2, label=name)
        axes[1].loglog(shear_rates, stress, linewidth=2, label=name)

    axes[0].set(
        xlabel=r"Shear-rate magnitude $\dot{\gamma}$ (s$^{-1}$)",
        ylabel=r"Apparent viscosity $\mu$ (mPa s)",
        title="Generalized-Newtonian viscosity laws",
    )
    axes[1].set(
        xlabel=r"Shear-rate magnitude $\dot{\gamma}$ (s$^{-1}$)",
        ylabel=r"Constitutive shear stress $\tau$ (Pa)",
        title="Constitutive stress response",
    )
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)

    figure.tight_layout()
    path = bt.get_result_path("blood_rheology_models.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Saved: {path}")


def example_hematocrit_parameterization() -> None:
    """Inspect the Casson correlation over the source study's 35--55% range."""
    hematocrit = np.linspace(0.35, 0.55, 61)
    shear_rates = (10.0, 100.0, 500.0)
    models = [bt.blood_casson_model(float(value)) for value in hematocrit]

    yield_stress = np.array([model.yield_stress() for model in models])
    high_shear_viscosity = np.array([model.plastic_viscosity() for model in models])

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for rate in shear_rates:
        apparent = np.array([model.viscosity(rate) for model in models])
        axes[0].plot(100.0 * hematocrit, 1e3 * apparent, label=f"{rate:g} s^-1")
    axes[0].set(
        xlabel="Hematocrit (%)",
        ylabel="Apparent viscosity (mPa s)",
        title="Casson correlation within its bounded domain",
    )
    axes[0].legend(title="Shear rate")

    axes[1].plot(
        100.0 * hematocrit,
        1e3 * yield_stress,
        color="tab:red",
        label="Yield-stress parameter (mPa)",
    )
    axes[1].plot(
        100.0 * hematocrit,
        1e3 * high_shear_viscosity,
        color="tab:blue",
        label="High-shear viscosity (mPa s)",
    )
    axes[1].set(
        xlabel="Hematocrit (%)",
        title="Merrill parameterization",
    )
    axes[1].legend()

    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    path = bt.get_result_path("blood_hematocrit_parameterization.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Saved: {path}")

    model_45 = bt.blood_casson_model(0.45)
    print("\n45% hematocrit Casson parameters")
    print(f"  yield-stress parameter: {model_45.yield_stress():.6g} Pa")
    print(f"  high-shear viscosity:   {model_45.plastic_viscosity():.6g} Pa s")


def _milliliters_per_minute_to_m3_per_second(value: float) -> float:
    return value * 1e-6 / 60.0


def example_nominal_vessel_shear_rates() -> None:
    """Evaluate constitutive models at illustrative nominal wall shear rates.

    The rate ``4 |Q|/(pi R^3)`` is exact for Newtonian fully developed pipe
    flow. Here it is only a transparent comparison rate for the non-Newtonian
    models; it is not a solved patient-specific wall-shear-stress field. The
    vessel dimensions and average flows below are illustrative, not reference
    intervals or patient data.
    """
    vessels = {
        "Aorta": (25e-3, _milliliters_per_minute_to_m3_per_second(5000.0)),
        "Carotid": (6e-3, _milliliters_per_minute_to_m3_per_second(360.0)),
        "Femoral": (8e-3, _milliliters_per_minute_to_m3_per_second(240.0)),
        "Coronary": (3e-3, _milliliters_per_minute_to_m3_per_second(60.0)),
    }
    casson = bt.blood_casson_model(0.45)
    carreau = bt.blood_carreau_model(0.45)
    newtonian = bt.NewtonianModel(0.00345)

    print("\nConstitutive stress at nominal wall shear rate")
    print(
        f"  {'Vessel':<10} {'D (mm)':>8} {'Q (mL/min)':>12} "
        f"{'gamma_nom (1/s)':>16} {'Newtonian (Pa)':>15} "
        f"{'Casson (Pa)':>12} {'Carreau (Pa)':>13}"
    )
    for name, (diameter, flow_rate) in vessels.items():
        radius = 0.5 * diameter
        nominal_rate = bt.pipe_wall_shear_rate(flow_rate, radius)
        flow_ml_min = flow_rate * 60.0 / 1e-6
        print(
            f"  {name:<10} {1e3 * diameter:8.2f} {flow_ml_min:12.0f} "
            f"{nominal_rate:16.2f} {newtonian.shear_stress(nominal_rate):15.4f} "
            f"{casson.shear_stress(nominal_rate):12.4f} "
            f"{carreau.shear_stress(nominal_rate):13.4f}"
        )

    print(
        "\nThese are constitutive comparisons at a nominal rate. Geometry, "
        "pulsatility, velocity gradients, and a compatible flow solve are needed "
        "for predictive wall shear stress."
    )


def main() -> None:
    print("Blood rheology: bounded generalized-Newtonian examples")
    example_constitutive_curves()
    example_hematocrit_parameterization()
    example_nominal_vessel_shear_rates()


if __name__ == "__main__":
    main()
