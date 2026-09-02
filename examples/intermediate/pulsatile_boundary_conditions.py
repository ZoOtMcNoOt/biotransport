#!/usr/bin/env python3
"""Inspect time-varying physiological and dosing waveform generators.

``PulsatileBC`` objects generate values; their names do not give those values
PDE semantics. An ``ArterialPressureBC`` therefore produces a pressure-shaped
signal in mmHg, but it does not impose fluid traction, solve pressure
propagation, or define inlet velocity. Likewise, a concentration protocol is
not a transport solution.

The native C++ transport API does not yet couple time-varying boundary values
into its internal time-stepping loop. This example deliberately stops at
sampling and plotting the protocols instead of presenting the legacy Python
``solve_pulsatile`` loop as a performance or physics-equivalent native solve.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import biotransport as bt


RESULTS_DIR = Path(bt.get_results_dir()) / "pulsatile_bc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_waveform_gallery() -> None:
    """Plot waveform values without assigning them unintended PDE semantics."""
    figure, axes = plt.subplots(3, 2, figsize=(12, 10))

    sinusoid = bt.reference.SinusoidalBC(mean=1.0, amplitude=0.2, frequency=1.2)
    time, value = bt.reference.sample_waveform(sinusoid, t_end=2.0, num_points=240)
    axes[0, 0].plot(time, value)
    axes[0, 0].set(title="Dimensionless sinusoid (1.2 Hz)", ylabel="Value")

    arterial = bt.reference.ArterialPressureBC(
        systolic=120.0, diastolic=80.0, heart_rate=72.0
    )
    time, value = bt.reference.sample_waveform(
        arterial, t_end=2.0 * arterial.period(), num_points=240
    )
    axes[0, 1].plot(time, value, color="tab:red")
    axes[0, 1].set(title="Arterial pressure-shaped signal", ylabel="Pressure (mmHg)")

    venous = bt.reference.VenousPressureBC(
        mean_pressure=8.0, amplitude=4.0, heart_rate=72.0
    )
    time, value = bt.reference.sample_waveform(
        venous, t_end=2.0 * venous.period(), num_points=240
    )
    axes[1, 0].plot(time, value, color="tab:purple")
    axes[1, 0].set(title="Venous pressure-shaped signal", ylabel="Pressure (mmHg)")

    cardiac_output = bt.reference.CardiacOutputBC(
        mean_flow=5.0, peak_flow=25.0, heart_rate=72.0
    )
    time, value = bt.reference.sample_waveform(
        cardiac_output, t_end=2.0 * cardiac_output.period(), num_points=240
    )
    axes[1, 1].plot(time, value, color="tab:green")
    axes[1, 1].set(title="Cardiac-output-shaped signal", ylabel="Flow (L/min)")

    respiratory = bt.reference.RespiratoryBC(
        mean=0.0, amplitude=1.0, respiratory_rate=12.0
    )
    time, value = bt.reference.sample_waveform(
        respiratory, t_end=2.0 * respiratory.period(), num_points=240
    )
    axes[2, 0].plot(time, value, color="tab:cyan")
    axes[2, 0].set(title="Respiratory signal", ylabel="Relative value")

    infusion = bt.reference.DrugInfusionBC(
        bolus_concentration=1.0,
        maintenance_concentration=0.1,
        bolus_duration=60.0,
        infusion_start=0.0,
    )
    time = np.linspace(0.0, 300.0, 300)
    value = np.array([infusion(float(t)) for t in time])
    axes[2, 1].plot(time, value, color="tab:orange")
    axes[2, 1].set(
        title="Bolus plus maintenance concentration", ylabel="Relative concentration"
    )

    for axis in axes.flat:
        axis.set_xlabel("Time (s)")
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    path = RESULTS_DIR / "waveform_gallery.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Saved: {path}")


def main() -> None:
    print("Pulsatile waveform generators")
    print(
        "These classes provide values and units; they do not by themselves "
        "create scalar, traction, or velocity boundary conditions."
    )
    plot_waveform_gallery()
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
