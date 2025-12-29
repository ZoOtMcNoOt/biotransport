#!/usr/bin/env python3
"""
Pulsatile Boundary Conditions Example
=====================================

Demonstrates time-varying boundary conditions for simulating
physiological processes with cardiac cycle variations.

This example shows:
1. Basic sinusoidal BC for oscillating conditions
2. Arterial pressure waveform at vessel inlet
3. Venous pressure and cardiac output waveforms
4. Composite BCs with respiratory modulation
5. Drug infusion with bolus and maintenance phases
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import biotransport as bt

# Results directory
RESULTS_DIR = Path(bt.get_results_dir()) / "pulsatile_bc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_waveform_gallery():
    """
    Plot a gallery of available pulsatile BC waveforms.
    """
    print("\n" + "=" * 60)
    print("PULSATILE WAVEFORM GALLERY")
    print("=" * 60)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # 1. Sinusoidal BC
    ax = axes[0, 0]
    bc = bt.SinusoidalBC(mean=100, amplitude=20, frequency=1.2)
    t, v = bt.sample_waveform(bc, t_end=2.0, num_points=200)
    ax.plot(t, v, "b-", linewidth=2)
    ax.set_title("Sinusoidal BC (1.2 Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)

    # 2. Arterial Pressure
    ax = axes[0, 1]
    bc = bt.ArterialPressureBC(systolic=120, diastolic=80, heart_rate=72)
    period = bc.period()
    t, v = bt.sample_waveform(bc, t_end=2 * period, num_points=200)
    ax.plot(t, v, "r-", linewidth=2)
    ax.axhline(y=120, color="r", linestyle="--", alpha=0.5, label="Systolic")
    ax.axhline(y=80, color="b", linestyle="--", alpha=0.5, label="Diastolic")
    ax.set_title(f"Arterial Pressure (HR=72 bpm, T={period:.2f}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (mmHg)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 3. Venous Pressure
    ax = axes[1, 0]
    bc = bt.VenousPressureBC(mean_pressure=8.0, amplitude=4.0, heart_rate=72)
    t, v = bt.sample_waveform(bc, t_end=2 * bc.period(), num_points=200)
    ax.plot(t, v, "purple", linewidth=2)
    ax.set_title("Venous Pressure (A, C, V waves)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (mmHg)")
    ax.grid(True, alpha=0.3)

    # 4. Cardiac Output
    ax = axes[1, 1]
    bc = bt.CardiacOutputBC(mean_flow=5.0, peak_flow=25.0, heart_rate=72)
    t, v = bt.sample_waveform(bc, t_end=2 * bc.period(), num_points=200)
    ax.plot(t, v, "g-", linewidth=2)
    ax.fill_between(t, 0, v, alpha=0.3, color="g")
    ax.set_title("Cardiac Output (Pulsatile Flow)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flow (L/min)")
    ax.grid(True, alpha=0.3)

    # 5. Respiratory
    ax = axes[2, 0]
    bc = bt.RespiratoryBC(mean=0, amplitude=1, respiratory_rate=12)
    t, v = bt.sample_waveform(bc, t_end=2 * bc.period(), num_points=200)
    ax.plot(t, v, "c-", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_title("Respiratory Waveform (12 breaths/min)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lung Volume (relative)")
    ax.grid(True, alpha=0.3)

    # 6. Drug Infusion
    ax = axes[2, 1]
    bc = bt.DrugInfusionBC(
        bolus_concentration=1.0,
        maintenance_concentration=0.1,
        bolus_duration=60,
        infusion_start=0,
    )
    t = np.linspace(0, 300, 300)
    v = [bc(ti) for ti in t]
    ax.plot(t, v, "orange", linewidth=2)
    ax.axvline(x=60, color="gray", linestyle="--", alpha=0.5, label="Bolus ends")
    ax.set_title("Drug Infusion (Bolus + Maintenance)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Concentration (relative)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "waveform_gallery.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'waveform_gallery.png'}")


def demo_sinusoidal_diffusion():
    """
    Simple diffusion with oscillating inlet concentration.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: Sinusoidal Inlet Concentration")
    print("=" * 60)

    # Domain setup
    L = 0.01  # 1 cm
    mesh = bt.mesh_1d(100, 0, L)
    D = 1e-9  # Diffusion coefficient (m^2/s)

    # Initial condition: uniform low concentration
    initial = np.zeros(101)

    problem = bt.Problem(mesh).diffusivity(D).initial_condition(initial)

    # Sinusoidal BC at inlet (frequency = 0.1 Hz, period = 10s)
    bc_inlet = bt.SinusoidalBC(mean=1.0, amplitude=0.5, frequency=0.1)

    # Simulate for 3 periods
    t_end = 30.0
    result = bt.solve_pulsatile(
        problem,
        t_end=t_end,
        pulsatile_bcs={bt.Boundary.Left: bc_inlet},
        save_every=500,
    )

    print(f"Simulated {t_end:.1f}s with {result.stats['steps']} steps")
    print(f"BC at inlet oscillated between {0.5:.1f} and {1.5:.1f}")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: BC history
    ax = axes[0]
    t_bc = result.time_history
    bc_vals = [bc_inlet(t) for t in t_bc]
    ax.plot(t_bc, bc_vals, "b-", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Inlet Concentration")
    ax.set_title("Time-Varying Inlet BC")
    ax.grid(True, alpha=0.3)

    # Right: Final concentration profile
    ax = axes[1]
    x = bt.x_nodes(mesh) * 1000  # Convert to mm
    ax.plot(x, result.solution, "r-", linewidth=2)
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("Concentration")
    ax.set_title(f"Concentration Profile at t={t_end:.0f}s")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sinusoidal_diffusion.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'sinusoidal_diffusion.png'}")


def demo_arterial_transport():
    """
    Arterial pressure-driven transport simulation.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Arterial Pressure-Driven Transport")
    print("=" * 60)

    # Vessel segment
    L = 0.05  # 5 cm vessel segment
    mesh = bt.mesh_1d(100, 0, L)
    D = 1e-6  # Higher diffusivity for faster dynamics

    # Initial: uniform at diastolic pressure
    initial = np.ones(101) * 80.0

    problem = bt.Problem(mesh).diffusivity(D).initial_condition(initial)

    # Arterial pressure waveform
    arterial = bt.ArterialPressureBC(systolic=120, diastolic=80, heart_rate=72)
    period = arterial.period()

    print(f"Cardiac period: {period:.3f}s ({72} bpm)")

    # Simulate 3 cardiac cycles
    t_end = 3 * period
    result = bt.solve_pulsatile(
        problem,
        t_end=t_end,
        pulsatile_bcs={bt.Boundary.Left: arterial},
        save_every=100,
    )

    print(f"Simulated {t_end:.2f}s ({3} cardiac cycles)")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Top left: Arterial waveform
    ax = axes[0, 0]
    t_plot = np.linspace(0, t_end, 500)
    p_plot = [arterial(t) for t in t_plot]
    ax.plot(t_plot, p_plot, "r-", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (mmHg)")
    ax.set_title("Arterial Pressure at Inlet")
    ax.axhline(y=120, color="r", linestyle="--", alpha=0.3)
    ax.axhline(y=80, color="b", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    # Top right: Inlet tracking
    ax = axes[0, 1]
    inlet_vals = [s[0] for s in result.solution_history]
    ax.plot(result.time_history, inlet_vals, "b-", linewidth=1.5, label="Simulated")
    ax.plot(
        result.time_history,
        [arterial(t) for t in result.time_history],
        "r--",
        linewidth=1,
        label="Target BC",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (mmHg)")
    ax.set_title("Inlet Boundary Tracking")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom left: Spatial profiles at different times
    ax = axes[1, 0]
    x = bt.x_nodes(mesh) * 100  # Convert to cm
    n_profiles = min(5, len(result.solution_history))
    indices = np.linspace(0, len(result.solution_history) - 1, n_profiles, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))
    for i, idx in enumerate(indices):
        ax.plot(
            x,
            result.solution_history[idx],
            color=colors[i],
            linewidth=1.5,
            label=f"t={result.time_history[idx]:.2f}s",
        )
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Pressure (mmHg)")
    ax.set_title("Pressure Profiles Over Time")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom right: Final profile
    ax = axes[1, 1]
    ax.plot(x, result.solution, "b-", linewidth=2)
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Pressure (mmHg)")
    ax.set_title(f"Final Pressure Profile (t={t_end:.2f}s)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "arterial_transport.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'arterial_transport.png'}")


def demo_composite_waveform():
    """
    Composite BC: cardiac + respiratory modulation.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Composite Waveform (Cardiac + Respiratory)")
    print("=" * 60)

    # Create composite waveform
    cardiac = bt.SinusoidalBC(mean=100, amplitude=20, frequency=1.2)  # ~72 bpm
    respiratory = bt.SinusoidalBC(
        mean=1.0, amplitude=0.05, frequency=0.2
    )  # 12 breaths/min

    # Multiply: respiratory modulates cardiac amplitude
    composite = bt.CompositeBC(components=[cardiac, respiratory], operation="multiply")

    # Sample and plot
    t = np.linspace(0, 10, 1000)  # 10 seconds
    cardiac_vals = [cardiac(ti) for ti in t]
    composite_vals = [composite(ti) for ti in t]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, cardiac_vals, "b-", alpha=0.5, linewidth=1, label="Cardiac only")
    ax.plot(t, composite_vals, "r-", linewidth=1.5, label="Cardiac + Respiratory")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (mmHg)")
    ax.set_title("Respiratory Modulation of Cardiac Waveform")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "composite_waveform.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'composite_waveform.png'}")


def demo_drug_infusion_transport():
    """
    Drug infusion with bolus and maintenance phases.
    """
    print("\n" + "=" * 60)
    print("DEMO 4: Drug Infusion Transport")
    print("=" * 60)

    # Tissue domain
    L = 0.005  # 5mm tissue thickness
    mesh = bt.mesh_1d(50, 0, L)
    D = 1e-10  # Drug diffusivity in tissue

    # Initial: no drug
    initial = np.zeros(51)

    problem = bt.Problem(mesh).diffusivity(D).initial_condition(initial)

    # Drug infusion BC
    infusion = bt.DrugInfusionBC(
        bolus_concentration=1.0,
        maintenance_concentration=0.2,
        bolus_duration=30.0,  # 30 second bolus
        infusion_start=0.0,
    )

    # Simulate for 5 minutes
    t_end = 300.0
    result = bt.solve_pulsatile(
        problem,
        t_end=t_end,
        pulsatile_bcs={bt.Boundary.Left: infusion},
        save_every=1000,
    )

    print(f"Simulated {t_end:.0f}s drug infusion")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Infusion profile
    ax = axes[0]
    t_plot = np.linspace(0, t_end, 500)
    c_plot = [infusion(t) for t in t_plot]
    ax.plot(t_plot, c_plot, "orange", linewidth=2)
    ax.axvline(x=30, color="gray", linestyle="--", alpha=0.5)
    ax.text(35, 0.8, "Bolus ends", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Drug Concentration (relative)")
    ax.set_title("Drug Infusion Protocol")
    ax.grid(True, alpha=0.3)

    # Right: Tissue concentration profiles
    ax = axes[1]
    x = bt.x_nodes(mesh) * 1000  # mm
    n_profiles = min(6, len(result.solution_history))
    indices = np.linspace(0, len(result.solution_history) - 1, n_profiles, dtype=int)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, n_profiles))
    for i, idx in enumerate(indices):
        ax.plot(
            x,
            result.solution_history[idx],
            color=colors[i],
            linewidth=1.5,
            label=f"t={result.time_history[idx]:.0f}s",
        )
    ax.set_xlabel("Tissue Depth (mm)")
    ax.set_ylabel("Drug Concentration")
    ax.set_title("Drug Penetration into Tissue")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "drug_infusion.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'drug_infusion.png'}")


def demo_custom_waveform():
    """
    Custom waveform from user-defined function.
    """
    print("\n" + "=" * 60)
    print("DEMO 5: Custom Waveform")
    print("=" * 60)

    # Define a custom double-peaked waveform
    def biphasic_waveform(t):
        """Biphasic waveform with two peaks per cycle."""
        phase = t % 1.0  # 1 Hz
        # Two Gaussian peaks
        peak1 = np.exp(-((phase - 0.2) ** 2) / 0.01)
        peak2 = 0.6 * np.exp(-((phase - 0.6) ** 2) / 0.02)
        return 50 + 30 * (peak1 + peak2)

    custom_bc = bt.CustomBC(func=biphasic_waveform, T=1.0)

    # Sample and plot
    t = np.linspace(0, 3, 500)
    v = [custom_bc(ti) for ti in t]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, v, "purple", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.set_title("Custom Biphasic Waveform")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "custom_waveform.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'custom_waveform.png'}")


def main():
    """Run all pulsatile BC demonstrations."""
    print("=" * 60)
    print("PULSATILE BOUNDARY CONDITIONS")
    print("Time-Varying BCs for Cardiac and Respiratory Simulations")
    print("=" * 60)

    # Show available waveforms
    plot_waveform_gallery()

    # Run demonstrations
    demo_sinusoidal_diffusion()
    demo_arterial_transport()
    demo_composite_waveform()
    demo_drug_infusion_transport()
    demo_custom_waveform()

    print("\n" + "=" * 60)
    print("AVAILABLE PULSATILE BC TYPES")
    print("=" * 60)
    print(
        """
Basic Waveforms:
  - ConstantBC(value)           : Time-invariant
  - SinusoidalBC(mean, amp, f)  : Sine wave oscillation
  - RampBC(start, end, dur)     : Linear transition
  - StepBC(before, after, t)    : Instantaneous step
  - SquareWaveBC(high, low, f)  : Square wave

Cardiac Waveforms:
  - ArterialPressureBC(sys, dia, hr)  : Arterial pressure waveform
  - VenousPressureBC(mean, amp, hr)   : Venous pressure (A/C/V waves)
  - CardiacOutputBC(mean, peak, hr)   : Pulsatile blood flow

Other Physiological:
  - RespiratoryBC(mean, amp, rr)      : Breathing waveform
  - DrugInfusionBC(bolus, maint, dur) : IV drug administration

Advanced:
  - CompositeBC(components, op)       : Combine multiple waveforms
  - CustomBC(func, period)            : User-defined function
"""
    )

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
