#!/usr/bin/env python3
"""
Nernst-Planck Ion Transport Examples
=====================================

This example demonstrates the Nernst-Planck equation for electrochemical
transport of charged species under concentration gradients and electric fields.

The Nernst-Planck equation is:
    ∂c/∂t = D∇²c + (zFD/RT) ∇·(c ∇φ)

where:
    c = ion concentration [mol/m³]
    D = diffusion coefficient [m²/s]
    z = ion valence (charge number)
    F = Faraday constant (96485 C/mol)
    R = gas constant (8.314 J/(mol·K))
    T = temperature [K]
    φ = electric potential [V]

Applications include:
    - Ion channels and membrane transport
    - Neural action potentials
    - Battery electrolytes
    - Electrophoresis and iontophoresis

Examples:
    1. Single ion electromigration (Na+ in electric field)
    2. Multi-ion system (Na+/K+/Cl- in neural context)
    3. Nernst potential calculation
    4. Electrophoretic separation

Author: BioTransport Development Team
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt

# Create output directory
import os

os.makedirs("results/nernst_planck", exist_ok=True)


def example_1_single_ion_electromigration():
    """
    Example 1: Single Ion Electromigration
    ======================================

    Sodium ions (Na+) transported by both diffusion and an applied
    electric field in a 1D domain (e.g., microfluidic channel).

    The applied field drives cations (positive ions) toward the cathode.
    """
    print("\n" + "=" * 60)
    print("Example 1: Single Ion Electromigration (Na+)")
    print("=" * 60)

    # Domain: 1 mm channel
    L = 1e-3  # 1 mm
    nx = 100
    mesh = bt.StructuredMesh(nx, 0.0, L)

    # Sodium ion properties
    Na = bt.ions.sodium()  # D = 1.33e-9 m²/s, z = +1
    print(f"Ion: {Na.name}, valence z = {Na.valence}, D = {Na.diffusivity:.2e} m²/s")
    print(f"Mobility μ = {Na.mobility:.2e} m²/(V·s)")

    # Create solver at body temperature
    T = 310.0  # 37°C
    solver = bt.NernstPlanckSolver(mesh, Na, temperature=T)

    # Thermal voltage
    Vt = solver.thermal_voltage()
    print(f"Thermal voltage V_T = RT/F = {Vt*1000:.2f} mV")

    # Initial condition: Gaussian pulse in center
    x = np.array([mesh.x(i) for i in range(nx + 1)])
    c0 = 10.0 * np.exp(-((x - L / 2) ** 2) / (0.1 * L) ** 2)  # mM peak
    solver.set_initial_condition(c0.tolist())

    # Apply uniform electric field (1 kV/m = 1 V/mm)
    E_field = 1000.0  # V/m
    solver.set_uniform_field(Ex=E_field)
    print(f"Applied field E = {E_field/1000:.1f} kV/m")

    # Drift velocity: v = z*D/Vt * E = μ*E
    v_drift = Na.mobility * E_field
    print(f"Drift velocity v = {v_drift*1e6:.2f} μm/s")

    # Zero-flux boundaries (insulating walls)
    solver.set_neumann_boundary(bt.Boundary.Left, 0.0)
    solver.set_neumann_boundary(bt.Boundary.Right, 0.0)

    # Time stepping
    dt = 1e-5  # 10 μs
    t_end = 0.5  # 500 ms
    steps_per_frame = 1000
    num_frames = 10

    # Store results
    times = [0.0]
    profiles = [np.array(solver.solution())]

    print(f"\nSimulating for {t_end} s with dt = {dt*1e6:.1f} μs...")

    for frame in range(num_frames):
        solver.solve(dt, steps_per_frame)
        times.append(solver.time())
        profiles.append(np.array(solver.solution()))
        print(
            f"  t = {solver.time()*1000:.1f} ms, "
            f"max(c) = {np.max(profiles[-1]):.3f} mM"
        )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Concentration profiles at different times
    colors = plt.cm.viridis(np.linspace(0, 1, len(profiles)))
    for i, (t, c) in enumerate(zip(times, profiles)):
        ax1.plot(x * 1e3, c, color=colors[i], label=f"t = {t*1000:.0f} ms")
    ax1.set_xlabel("Position x [mm]")
    ax1.set_ylabel("Concentration c [mM]")
    ax1.set_title(f"Na+ Transport in E = {E_field/1000:.0f} kV/m Field")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Peak position vs time (shows drift)
    peak_positions = [x[np.argmax(c)] * 1e3 for c in profiles]
    ax2.plot(np.array(times) * 1000, peak_positions, "bo-", markersize=8)
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("Peak position [mm]")
    ax2.set_title("Na+ Peak Migration")
    ax2.grid(True, alpha=0.3)

    # Expected drift line
    t_array = np.array(times)
    expected_pos = L / 2 * 1e3 + v_drift * 1e3 * t_array
    ax2.plot(t_array * 1000, expected_pos, "r--", label="Expected drift")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("results/nernst_planck/single_ion_electromigration.png", dpi=150)
    plt.close()

    print("\n✓ Saved: results/nernst_planck/single_ion_electromigration.png")

    return True


def example_2_multi_ion_system():
    """
    Example 2: Multi-Ion System (Na+/K+/Cl-)
    ========================================

    Simulates three physiologically important ions with different
    initial distributions, mimicking ionic gradients across a membrane.
    """
    print("\n" + "=" * 60)
    print("Example 2: Multi-Ion System (Na+/K+/Cl-)")
    print("=" * 60)

    # Domain: 100 μm (approximate membrane + cytoplasm scale)
    L = 100e-6  # 100 μm
    nx = 50
    mesh = bt.StructuredMesh(nx, 0.0, L)

    # Three ion species
    ions = [
        bt.ions.sodium(),  # Na+, z=+1
        bt.ions.potassium(),  # K+, z=+1
        bt.ions.chloride(),  # Cl-, z=-1
    ]

    print("Ion species:")
    for ion in ions:
        print(f"  {ion.name}: z={ion.valence:+d}, D={ion.diffusivity:.2e} m²/s")

    # Create multi-ion solver
    solver = bt.MultiIonSolver(mesh, ions, temperature=310.0)

    # Initial conditions - mimicking extracellular (left) to intracellular (right)
    x = np.array([mesh.x(i) for i in range(nx + 1)])
    x_norm = x / L  # Normalized position [0, 1]

    # Extracellular concentrations (left): high Na, low K
    # Intracellular concentrations (right): low Na, high K
    # Using smooth transition
    Na_ic = 140.0 * (1 - x_norm) + 14.0 * x_norm  # mM: 140→14
    K_ic = 5.0 * (1 - x_norm) + 140.0 * x_norm  # mM: 5→140
    Cl_ic = 120.0 * (1 - x_norm) + 4.0 * x_norm  # mM: 120→4

    solver.set_initial_condition(0, Na_ic.tolist())  # Na+
    solver.set_initial_condition(1, K_ic.tolist())  # K+
    solver.set_initial_condition(2, Cl_ic.tolist())  # Cl-

    # Fixed concentration boundaries (Dirichlet)
    # Left = extracellular
    solver.set_dirichlet_boundary(0, bt.Boundary.Left, 140.0)  # Na+ = 140 mM
    solver.set_dirichlet_boundary(1, bt.Boundary.Left, 5.0)  # K+ = 5 mM
    solver.set_dirichlet_boundary(2, bt.Boundary.Left, 120.0)  # Cl- = 120 mM

    # Right = intracellular
    solver.set_dirichlet_boundary(0, bt.Boundary.Right, 14.0)  # Na+ = 14 mM
    solver.set_dirichlet_boundary(1, bt.Boundary.Right, 140.0)  # K+ = 140 mM
    solver.set_dirichlet_boundary(2, bt.Boundary.Right, 4.0)  # Cl- = 4 mM

    # Apply modest electric field
    E_field = 100.0  # V/m (100 mV over 1 mm)
    solver.set_uniform_field(Ex=E_field)
    print(f"Applied field E = {E_field} V/m")

    # Simulate to steady state
    dt = 1e-7  # 0.1 μs (faster diffusion in small domain)
    t_end = 1e-3  # 1 ms
    steps_per_frame = 100
    num_frames = 10

    # Store results
    times = [0.0]
    Na_profiles = [np.array(solver.concentration(0))]
    K_profiles = [np.array(solver.concentration(1))]
    Cl_profiles = [np.array(solver.concentration(2))]
    charge_densities = [np.array(solver.charge_density())]

    print(f"\nSimulating for {t_end*1000:.2f} ms...")

    for frame in range(num_frames):
        solver.solve(dt, steps_per_frame)
        times.append(solver.time())
        Na_profiles.append(np.array(solver.concentration(0)))
        K_profiles.append(np.array(solver.concentration(1)))
        Cl_profiles.append(np.array(solver.concentration(2)))
        charge_densities.append(np.array(solver.charge_density()))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x_um = x * 1e6  # Convert to μm

    # Initial vs final concentrations
    ax = axes[0, 0]
    ax.plot(x_um, Na_profiles[0], "r--", alpha=0.5, label="Na+ (initial)")
    ax.plot(x_um, Na_profiles[-1], "r-", linewidth=2, label="Na+ (final)")
    ax.plot(x_um, K_profiles[0], "b--", alpha=0.5, label="K+ (initial)")
    ax.plot(x_um, K_profiles[-1], "b-", linewidth=2, label="K+ (final)")
    ax.plot(x_um, Cl_profiles[0], "g--", alpha=0.5, label="Cl- (initial)")
    ax.plot(x_um, Cl_profiles[-1], "g-", linewidth=2, label="Cl- (final)")
    ax.set_xlabel("Position [μm]")
    ax.set_ylabel("Concentration [mM]")
    ax.set_title("Ion Concentration Profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Na+ time evolution
    ax = axes[0, 1]
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(Na_profiles)))
    for i, (t, c) in enumerate(zip(times, Na_profiles)):
        ax.plot(x_um, c, color=colors[i])
    ax.set_xlabel("Position [μm]")
    ax.set_ylabel("Na+ [mM]")
    ax.set_title("Na+ Evolution")
    ax.grid(True, alpha=0.3)

    # K+ time evolution
    ax = axes[1, 0]
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(K_profiles)))
    for i, (t, c) in enumerate(zip(times, K_profiles)):
        ax.plot(x_um, c, color=colors[i])
    ax.set_xlabel("Position [μm]")
    ax.set_ylabel("K+ [mM]")
    ax.set_title("K+ Evolution")
    ax.grid(True, alpha=0.3)

    # Charge density
    ax = axes[1, 1]
    colors = plt.cm.RdBu(np.linspace(0, 1, len(charge_densities)))
    for i, (t, rho) in enumerate(zip(times, charge_densities)):
        ax.plot(x_um, rho, color=colors[i])
    ax.set_xlabel("Position [μm]")
    ax.set_ylabel("Charge density [C/m³]")
    ax.set_title("Net Charge Density")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/nernst_planck/multi_ion_system.png", dpi=150)
    plt.close()

    print("\n✓ Saved: results/nernst_planck/multi_ion_system.png")

    return True


def example_3_nernst_equilibrium():
    """
    Example 3: Nernst Equilibrium Potentials
    ========================================

    Demonstrates calculation of Nernst equilibrium potentials for
    various ions at physiological conditions (37°C).
    """
    print("\n" + "=" * 60)
    print("Example 3: Nernst Equilibrium Potentials")
    print("=" * 60)

    # Typical mammalian neuron concentrations (mM → mol/m³ for calculation)
    ions_data = {
        "K+": {"z": 1, "c_in": 140e-3, "c_out": 5e-3},
        "Na+": {"z": 1, "c_in": 14e-3, "c_out": 140e-3},
        "Cl-": {"z": -1, "c_in": 4e-3, "c_out": 120e-3},
        "Ca2+": {"z": 2, "c_in": 0.1e-6, "c_out": 2e-3},
    }

    print("\nNernst equilibrium potentials at T = 37°C:")
    print("-" * 50)
    print(f"{'Ion':<8} {'[in] mM':<12} {'[out] mM':<12} {'E (mV)':<10}")
    print("-" * 50)

    nernst_potentials = {}
    for ion_name, data in ions_data.items():
        E = bt.ghk.nernst_potential(
            z=data["z"], c_in=data["c_in"], c_out=data["c_out"], temperature=310.0
        )
        nernst_potentials[ion_name] = E * 1000  # Convert to mV
        print(
            f"{ion_name:<8} {data['c_in']*1000:<12.4f} {data['c_out']*1000:<12.4f} "
            f"{E*1000:<+10.1f}"
        )

    # Goldman-Hodgkin-Katz voltage equation
    print("\n\nGoldman-Hodgkin-Katz Membrane Potential:")
    print("-" * 50)

    # Resting state: high K permeability
    P_K, P_Na, P_Cl = 1.0, 0.04, 0.45
    V_rest = bt.ghk.ghk_voltage(
        P_K=P_K,
        K_in=140e-3,
        K_out=5e-3,
        P_Na=P_Na,
        Na_in=14e-3,
        Na_out=140e-3,
        P_Cl=P_Cl,
        Cl_in=4e-3,
        Cl_out=120e-3,
    )
    print("Resting state (P_K:P_Na:P_Cl = 1:0.04:0.45):")
    print(f"  V_rest = {V_rest*1000:.1f} mV")

    # Action potential peak: high Na permeability
    P_K, P_Na, P_Cl = 1.0, 20.0, 0.45
    V_peak = bt.ghk.ghk_voltage(
        P_K=P_K,
        K_in=140e-3,
        K_out=5e-3,
        P_Na=P_Na,
        Na_in=14e-3,
        Na_out=140e-3,
        P_Cl=P_Cl,
        Cl_in=4e-3,
        Cl_out=120e-3,
    )
    print("\nAP peak (P_K:P_Na:P_Cl = 1:20:0.45):")
    print(f"  V_peak = {V_peak*1000:+.1f} mV")

    # Undershoot: very high K permeability
    P_K, P_Na, P_Cl = 10.0, 0.04, 0.45
    V_under = bt.ghk.ghk_voltage(
        P_K=P_K,
        K_in=140e-3,
        K_out=5e-3,
        P_Na=P_Na,
        Na_in=14e-3,
        Na_out=140e-3,
        P_Cl=P_Cl,
        Cl_in=4e-3,
        Cl_out=120e-3,
    )
    print("\nUndershoot (P_K:P_Na:P_Cl = 10:0.04:0.45):")
    print(f"  V_under = {V_under*1000:.1f} mV")

    # Plot Nernst potentials
    fig, ax = plt.subplots(figsize=(8, 6))

    ions_list = list(nernst_potentials.keys())
    E_values = list(nernst_potentials.values())
    colors = ["blue", "red", "green", "orange"]

    bars = ax.barh(ions_list, E_values, color=colors, edgecolor="black", height=0.6)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.axvline(
        x=V_rest * 1000,
        color="purple",
        linestyle="--",
        linewidth=2,
        label=f"V_rest = {V_rest*1000:.0f} mV",
    )

    ax.set_xlabel("Equilibrium Potential [mV]")
    ax.set_title("Nernst Equilibrium Potentials (37°C)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, val in zip(bars, E_values):
        ax.text(
            val + (5 if val > 0 else -5),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.0f} mV",
            va="center",
            ha="left" if val > 0 else "right",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("results/nernst_planck/nernst_potentials.png", dpi=150)
    plt.close()

    print("\n✓ Saved: results/nernst_planck/nernst_potentials.png")

    return True


def example_4_electrophoretic_separation():
    """
    Example 4: Electrophoretic Separation
    =====================================

    Simulates separation of two ion species with different mobilities
    in an applied electric field (principle of capillary electrophoresis).
    """
    print("\n" + "=" * 60)
    print("Example 4: Electrophoretic Separation")
    print("=" * 60)

    # Domain: 2 mm capillary
    L = 2e-3  # 2 mm
    nx = 200
    mesh = bt.StructuredMesh(nx, 0.0, L)

    # Two species with different mobilities
    # Using Na+ and a "large ion" with lower diffusivity
    Na = bt.ions.sodium()  # Fast
    large_ion = bt.IonSpecies(
        "Protein+", valence=2, diffusivity=0.5e-9
    )  # Slow, +2 charge

    ions = [Na, large_ion]

    print("Ion species for separation:")
    for ion in ions:
        print(
            f"  {ion.name}: z={ion.valence:+d}, D={ion.diffusivity:.2e} m²/s, "
            f"μ={ion.mobility:.2e} m²/(V·s)"
        )

    # Create solver
    solver = bt.MultiIonSolver(mesh, ions, temperature=298.0)  # Room temperature

    # Initial condition: both species start as Gaussian peaks at same position
    x = np.array([mesh.x(i) for i in range(nx + 1)])
    x0 = 0.3 * L  # Start at 30% of channel
    sigma = 0.05 * L

    c_init = 10.0 * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
    solver.set_initial_condition(0, c_init.tolist())  # Na+
    solver.set_initial_condition(1, c_init.tolist())  # Large ion

    # Zero-flux boundaries
    for s in range(2):
        solver.set_neumann_boundary(s, bt.Boundary.Left, 0.0)
        solver.set_neumann_boundary(s, bt.Boundary.Right, 0.0)

    # Strong electric field for separation
    E_field = 5000.0  # 5 kV/m
    solver.set_uniform_field(Ex=E_field)

    # Calculate expected velocities
    v_Na = Na.mobility * E_field
    v_large = large_ion.mobility * E_field

    print(f"\nApplied field: E = {E_field/1000:.1f} kV/m")
    print("Expected velocities:")
    print(f"  Na+: v = {v_Na*1e3:.3f} mm/s")
    print(f"  {large_ion.name}: v = {v_large*1e3:.3f} mm/s")
    print(f"  Separation factor: {v_large/v_Na:.2f}")

    # Time stepping
    dt = 5e-6  # 5 μs
    t_end = 0.2  # 200 ms
    steps_per_frame = 1000
    num_frames = 8

    # Store results
    times = [0.0]
    Na_profiles = [np.array(solver.concentration(0))]
    large_profiles = [np.array(solver.concentration(1))]

    print(f"\nSimulating for {t_end*1000:.0f} ms...")

    for frame in range(num_frames):
        solver.solve(dt, steps_per_frame)
        times.append(solver.time())
        Na_profiles.append(np.array(solver.concentration(0)))
        large_profiles.append(np.array(solver.concentration(1)))

        # Track peak positions
        Na_peak = x[np.argmax(Na_profiles[-1])]
        large_peak = x[np.argmax(large_profiles[-1])]
        sep = abs(large_peak - Na_peak)
        print(
            f"  t = {solver.time()*1000:.0f} ms: "
            f"Na+ peak = {Na_peak*1e3:.2f} mm, "
            f"{large_ion.name} peak = {large_peak*1e3:.2f} mm, "
            f"separation = {sep*1e3:.2f} mm"
        )

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x_mm = x * 1e3

    # Evolution of Na+
    ax = axes[0, 0]
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(times)))
    for i, (t, c) in enumerate(zip(times, Na_profiles)):
        ax.plot(x_mm, c, color=colors[i], label=f"{t*1000:.0f} ms")
    ax.set_xlabel("Position [mm]")
    ax.set_ylabel("Concentration")
    ax.set_title("Na+ Migration")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Evolution of large ion
    ax = axes[0, 1]
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(times)))
    for i, (t, c) in enumerate(zip(times, large_profiles)):
        ax.plot(x_mm, c, color=colors[i], label=f"{t*1000:.0f} ms")
    ax.set_xlabel("Position [mm]")
    ax.set_ylabel("Concentration")
    ax.set_title(f"{large_ion.name} Migration")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Both species at final time
    ax = axes[1, 0]
    ax.fill_between(x_mm, 0, Na_profiles[-1], alpha=0.5, color="red", label="Na+")
    ax.fill_between(
        x_mm, 0, large_profiles[-1], alpha=0.5, color="blue", label=large_ion.name
    )
    ax.set_xlabel("Position [mm]")
    ax.set_ylabel("Concentration")
    ax.set_title(f"Separation at t = {times[-1]*1000:.0f} ms")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Peak positions vs time
    ax = axes[1, 1]
    Na_peaks = [x_mm[np.argmax(c)] for c in Na_profiles]
    large_peaks = [x_mm[np.argmax(c)] for c in large_profiles]
    t_ms = np.array(times) * 1000

    ax.plot(t_ms, Na_peaks, "ro-", label="Na+ peak", markersize=6)
    ax.plot(t_ms, large_peaks, "bs-", label=f"{large_ion.name} peak", markersize=6)

    # Expected trajectories
    ax.plot(t_ms, x0 * 1e3 + v_Na * t_ms, "r--", alpha=0.5, label="Expected Na+")
    ax.plot(
        t_ms,
        x0 * 1e3 + v_large * t_ms,
        "b--",
        alpha=0.5,
        label=f"Expected {large_ion.name}",
    )

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Peak position [mm]")
    ax.set_title("Electrophoretic Separation")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/nernst_planck/electrophoretic_separation.png", dpi=150)
    plt.close()

    print("\n✓ Saved: results/nernst_planck/electrophoretic_separation.png")

    return True


def main():
    """Run all Nernst-Planck examples."""
    print("=" * 60)
    print("NERNST-PLANCK ION TRANSPORT EXAMPLES")
    print("=" * 60)
    print("\nPhysical constants:")
    print(f"  Faraday constant F = {bt.constants.FARADAY:.2f} C/mol")
    print(f"  Gas constant R = {bt.constants.GAS_CONSTANT:.4f} J/(mol·K)")
    print(
        f"  Thermal voltage at 37°C: V_T = {bt.IonSpecies.thermal_voltage(310)*1000:.2f} mV"
    )

    results = {}

    try:
        results["ex1"] = example_1_single_ion_electromigration()
    except Exception as e:
        print(f"Example 1 failed: {e}")
        results["ex1"] = False

    try:
        results["ex2"] = example_2_multi_ion_system()
    except Exception as e:
        print(f"Example 2 failed: {e}")
        results["ex2"] = False

    try:
        results["ex3"] = example_3_nernst_equilibrium()
    except Exception as e:
        print(f"Example 3 failed: {e}")
        results["ex3"] = False

    try:
        results["ex4"] = example_4_electrophoretic_separation()
    except Exception as e:
        print(f"Example 4 failed: {e}")
        results["ex4"] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All Nernst-Planck examples completed successfully!")
    else:
        print("\n⚠ Some examples failed. Check output above.")

    return all_passed


if __name__ == "__main__":
    main()
