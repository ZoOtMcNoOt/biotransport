"""
Example of steady-state membrane diffusion with partition coefficients.

This example demonstrates the MembraneDiffusion1DSolver for calculating
steady-state flux and concentration profiles across biological membranes.

Applications:
- Blood-brain barrier transport
- Drug release from polymer matrices
- Cell membrane permeation
- Dialysis membranes

Physics:
- Steady-state Fick's first law: j = -D * dC/dx
- Partition coefficient at interfaces: C_membrane = Phi * C_solution
- Hindered diffusion for large solutes (Renkin equation)

Key equation:
    j = D * Phi * (C_left - C_right) / L
    P = D * Phi / L  (membrane permeability)
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt

EXAMPLE_NAME = "steady_membrane_diffusion"


def run_simple_membrane():
    """
    Example 1: Simple membrane diffusion - Blood-Brain Barrier analog.

    Model a drug crossing a lipid membrane barrier similar to BBB.
    """
    print("\n" + "=" * 60)
    print("Example 1: Simple Membrane Diffusion (BBB analog)")
    print("=" * 60)

    # Membrane parameters (typical for lipophilic drug crossing BBB)
    L = 5e-9  # Membrane thickness ~ 5 nm (lipid bilayer)
    D = 1e-12  # Diffusion in membrane (m^2/s) - slower than in water
    Phi = 0.5  # Partition coefficient - drug is moderately lipophilic

    # Concentrations
    C_plasma = 10.0  # Drug concentration in blood plasma (arbitrary units)
    C_brain = 1.0  # Drug concentration in brain tissue

    # Create solver with fluent API
    solver = (
        bt.MembraneDiffusion1DSolver()
        .set_membrane_thickness(L)
        .set_diffusivity(D)
        .set_partition_coefficient(Phi)
        .set_left_concentration(C_plasma)
        .set_right_concentration(C_brain)
        .set_num_nodes(51)
    )

    # Solve
    result = solver.solve()

    # Print results
    print("\nMembrane Properties:")
    print(f"  Thickness: {L * 1e9:.1f} nm")
    print(f"  Diffusivity: {D:.2e} m^2/s")
    print(f"  Partition coefficient: {Phi:.2f}")

    print("\nBoundary Concentrations:")
    print(f"  Plasma (left): {C_plasma:.1f} units")
    print(f"  Brain (right): {C_brain:.1f} units")

    print("\nResults:")
    print(f"  Steady-state flux: {result.flux:.4e} units/(m^2*s)")
    print(f"  Membrane permeability: {result.permeability:.4e} m/s")
    print(f"  Permeability: {result.permeability * 100:.4f} cm/s")

    # Analytical verification
    j_analytical = D * Phi * (C_plasma - C_brain) / L
    P_analytical = D * Phi / L
    print("\nAnalytical verification:")
    print(f"  j = D*Phi*deltaC/L = {j_analytical:.4e} units/(m^2*s)")
    print(f"  P = D*Phi/L = {P_analytical:.4e} m/s")
    print(f"  Match: {np.isclose(result.flux, j_analytical)}")

    # Plot concentration profile
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Profile inside membrane
    x_nm = result.x() * 1e9
    ax1 = axes[0]
    ax1.plot(x_nm, result.concentration(), "b-", linewidth=2)
    ax1.set_xlabel("Position in membrane (nm)")
    ax1.set_ylabel("Concentration (units)")
    ax1.set_title("Concentration Profile Inside Membrane")
    ax1.grid(True, alpha=0.3)

    # Add annotations for partition
    ax1.axhline(
        y=Phi * C_plasma,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Phi*C_plasma = {Phi * C_plasma:.1f}",
    )
    ax1.axhline(
        y=Phi * C_brain,
        color="g",
        linestyle="--",
        alpha=0.5,
        label=f"Phi*C_brain = {Phi * C_brain:.1f}",
    )
    ax1.legend()

    # Full picture including external solutions
    ax2 = axes[1]
    x_full = [-2, 0] + list(x_nm) + [x_nm[-1], x_nm[-1] + 2]
    C_full = [C_plasma, C_plasma] + list(result.concentration()) + [C_brain, C_brain]
    ax2.plot(x_full, C_full, "b-", linewidth=2)
    ax2.axvspan(0, L * 1e9, alpha=0.2, color="gray", label="Membrane")
    ax2.set_xlabel("Position (nm)")
    ax2.set_ylabel("Concentration (units)")
    ax2.set_title("Full Concentration Profile with Partition Jump")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(bt.get_result_path("simple_membrane.png", EXAMPLE_NAME))
    plt.show()

    return result


def run_hindered_diffusion():
    """
    Example 2: Hindered diffusion of proteins through porous membrane.

    Model albumin diffusion through dialysis membrane with hindered transport.
    """
    print("\n" + "=" * 60)
    print("Example 2: Hindered Diffusion (Dialysis Membrane)")
    print("=" * 60)

    # Membrane properties
    L = 50e-6  # 50 micron thick membrane
    D_bulk = 6.0e-11  # Albumin diffusivity in water (m^2/s)
    Phi = 0.8  # Partition coefficient
    pore_radius = 10e-9  # 10 nm pores

    # Albumin hydrodynamic radius ~ 3.5 nm
    solute_radius = 3.5e-9

    # Concentrations
    C_blood = 40.0  # g/L (typical serum albumin)
    C_dialysate = 0.0  # Pure dialysate

    # Calculate hindrance factor
    lambda_ratio = solute_radius / pore_radius
    H = bt.renkin_hindrance(lambda_ratio)
    print("\nHindrance calculation:")
    print(f"  Solute radius: {solute_radius * 1e9:.1f} nm")
    print(f"  Pore radius: {pore_radius * 1e9:.1f} nm")
    print(f"  Lambda (r/R): {lambda_ratio:.3f}")
    print(f"  Renkin hindrance H: {H:.4f}")

    # Compare with and without hindrance
    solver_bulk = (
        bt.MembraneDiffusion1DSolver()
        .set_membrane_thickness(L)
        .set_diffusivity(D_bulk)
        .set_partition_coefficient(Phi)
        .set_left_concentration(C_blood)
        .set_right_concentration(C_dialysate)
    )
    result_bulk = solver_bulk.solve()

    solver_hindered = (
        bt.MembraneDiffusion1DSolver()
        .set_membrane_thickness(L)
        .set_diffusivity(D_bulk)
        .set_partition_coefficient(Phi)
        .set_left_concentration(C_blood)
        .set_right_concentration(C_dialysate)
        .set_hindered_diffusion(solute_radius, pore_radius)
    )
    result_hindered = solver_hindered.solve()

    print("\nResults comparison:")
    print("  Without hindrance:")
    print(f"    D_eff = {result_bulk.effective_diffusivity:.4e} m^2/s")
    print(f"    Flux = {result_bulk.flux:.4e} g/(m^2*s)")
    print("  With hindrance:")
    print(f"    D_eff = {result_hindered.effective_diffusivity:.4e} m^2/s")
    print(f"    Flux = {result_hindered.flux:.4e} g/(m^2*s)")
    print(
        f"  Flux reduction: {100 * (1 - result_hindered.flux / result_bulk.flux):.1f}%"
    )

    # Plot hindrance curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Hindrance factor vs lambda
    ax1 = axes[0]
    lambdas = np.linspace(0, 0.95, 100)
    H_values = [bt.renkin_hindrance(lam) for lam in lambdas]
    ax1.plot(lambdas, H_values, "b-", linewidth=2)
    ax1.axvline(
        x=lambda_ratio,
        color="r",
        linestyle="--",
        label=f"Albumin (lambda={lambda_ratio:.2f})",
    )
    ax1.axhline(y=H, color="r", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Lambda (solute radius / pore radius)")
    ax1.set_ylabel("Hindrance factor H")
    ax1.set_title("Renkin Hindrance Factor")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Concentration profiles
    ax2 = axes[1]
    x_um = result_bulk.x() * 1e6
    ax2.plot(
        x_um, result_bulk.concentration(), "b-", linewidth=2, label="Bulk diffusion"
    )
    ax2.plot(
        x_um,
        result_hindered.concentration(),
        "r--",
        linewidth=2,
        label="Hindered diffusion",
    )
    ax2.set_xlabel("Position in membrane (um)")
    ax2.set_ylabel("Albumin concentration (g/L)")
    ax2.set_title("Concentration Profiles")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(bt.get_result_path("hindered_diffusion.png", EXAMPLE_NAME))
    plt.show()

    return result_hindered


def run_multilayer_skin():
    """
    Example 3: Drug permeation through multi-layer skin.

    Model transdermal drug delivery through stratum corneum,
    epidermis, and dermis layers.
    """
    print("\n" + "=" * 60)
    print("Example 3: Multi-Layer Skin Permeation")
    print("=" * 60)

    # Skin layer properties (simplified model)
    # Layer 1: Stratum corneum - main barrier
    L_sc = 15e-6  # 15 microns
    D_sc = 1e-13  # Very low diffusivity (lipophilic barrier)
    Phi_sc = 0.01  # Low partition for hydrophilic drug

    # Layer 2: Viable epidermis
    L_epi = 100e-6  # 100 microns
    D_epi = 1e-10  # Higher diffusivity (aqueous)
    Phi_epi = 0.8  # Better partition

    # Layer 3: Dermis
    L_dermis = 1e-3  # 1 mm
    D_dermis = 5e-10  # Similar to water
    Phi_dermis = 1.0  # No barrier

    # Concentrations
    C_patch = 100.0  # Drug in patch (mg/mL)
    C_blood = 0.0  # Systemic blood (sink condition)

    # Create multi-layer solver
    solver = (
        bt.MultiLayerMembraneSolver()
        .add_layer(L_sc, D_sc, Phi_sc)
        .add_layer(L_epi, D_epi, Phi_epi)
        .add_layer(L_dermis, D_dermis, Phi_dermis)
        .set_left_concentration(C_patch)
        .set_right_concentration(C_blood)
    )

    result = solver.solve()

    print("\nSkin layer structure:")
    print(f"  Stratum corneum: {L_sc * 1e6:.0f} um, D={D_sc:.1e} m^2/s, Phi={Phi_sc}")
    print(f"  Epidermis: {L_epi * 1e6:.0f} um, D={D_epi:.1e} m^2/s, Phi={Phi_epi}")
    print(
        f"  Dermis: {L_dermis * 1e3:.1f} mm, D={D_dermis:.1e} m^2/s, Phi={Phi_dermis}"
    )
    print(f"  Total thickness: {solver.total_thickness() * 1e3:.2f} mm")

    print("\nResults:")
    print(f"  Steady-state flux: {result.flux:.4e} mg/(m^2*s)")
    print(f"  Flux: {result.flux * 3600 * 1e-4:.4f} mg/(cm^2*hr)")
    print(f"  Overall permeability: {result.permeability:.4e} m/s")

    # Calculate individual layer resistances
    R_sc = L_sc / (D_sc * Phi_sc)
    R_epi = L_epi / (D_epi * Phi_epi)
    R_dermis = L_dermis / (D_dermis * Phi_dermis)
    R_total = R_sc + R_epi + R_dermis

    print("\nLayer resistances (s/m):")
    print(f"  Stratum corneum: {R_sc:.2e} ({100*R_sc/R_total:.1f}%)")
    print(f"  Epidermis: {R_epi:.2e} ({100*R_epi/R_total:.1f}%)")
    print(f"  Dermis: {R_dermis:.2e} ({100*R_dermis/R_total:.1f}%)")
    print(f"  Total: {R_total:.2e}")

    # Plot concentration profile through all layers
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_um = np.array(result.x()) * 1e6
    conc = result.concentration()

    ax1 = axes[0]
    ax1.plot(x_um, conc, "b-", linewidth=2)
    ax1.axvspan(0, L_sc * 1e6, alpha=0.3, color="red", label="Stratum corneum")
    ax1.axvspan(
        L_sc * 1e6, (L_sc + L_epi) * 1e6, alpha=0.3, color="green", label="Epidermis"
    )
    ax1.axvspan(
        (L_sc + L_epi) * 1e6,
        solver.total_thickness() * 1e6,
        alpha=0.3,
        color="blue",
        label="Dermis",
    )
    ax1.set_xlabel("Position in skin (um)")
    ax1.set_ylabel("Drug concentration (mg/mL)")
    ax1.set_title("Concentration Profile Through Skin Layers")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Resistance pie chart
    ax2 = axes[1]
    labels = ["Stratum corneum", "Epidermis", "Dermis"]
    sizes = [R_sc, R_epi, R_dermis]
    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1"]
    ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax2.set_title("Resistance Distribution")

    plt.tight_layout()
    plt.savefig(bt.get_result_path("multilayer_skin.png", EXAMPLE_NAME))
    plt.show()

    return result


def run_permeability_study():
    """
    Example 4: Parametric study of membrane permeability.

    Explore how permeability depends on membrane properties.
    """
    print("\n" + "=" * 60)
    print("Example 4: Permeability Parametric Study")
    print("=" * 60)

    # Base parameters
    L_base = 100e-6  # 100 um
    D_base = 1e-10  # 10^-10 m^2/s
    Phi_base = 0.5

    # Vary each parameter
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Permeability vs thickness
    ax1 = axes[0]
    L_values = np.logspace(-7, -4, 50)  # 0.1 um to 100 um
    P_vs_L = []
    for L in L_values:
        solver = bt.MembraneDiffusion1DSolver()
        solver.set_membrane_thickness(L).set_diffusivity(
            D_base
        ).set_partition_coefficient(Phi_base)
        P_vs_L.append(solver.compute_permeability())

    ax1.loglog(L_values * 1e6, P_vs_L, "b-", linewidth=2)
    ax1.set_xlabel("Membrane thickness (um)")
    ax1.set_ylabel("Permeability (m/s)")
    ax1.set_title("P = D*Phi/L")
    ax1.grid(True, alpha=0.3, which="both")

    # 2. Permeability vs diffusivity
    ax2 = axes[1]
    D_values = np.logspace(-13, -9, 50)
    P_vs_D = []
    for D in D_values:
        solver = bt.MembraneDiffusion1DSolver()
        solver.set_membrane_thickness(L_base).set_diffusivity(
            D
        ).set_partition_coefficient(Phi_base)
        P_vs_D.append(solver.compute_permeability())

    ax2.loglog(D_values, P_vs_D, "g-", linewidth=2)
    ax2.set_xlabel("Diffusivity (m^2/s)")
    ax2.set_ylabel("Permeability (m/s)")
    ax2.set_title("P = D*Phi/L")
    ax2.grid(True, alpha=0.3, which="both")

    # 3. Permeability vs partition coefficient
    ax3 = axes[2]
    Phi_values = np.linspace(0.01, 2.0, 50)
    P_vs_Phi = []
    for Phi in Phi_values:
        solver = bt.MembraneDiffusion1DSolver()
        solver.set_membrane_thickness(L_base).set_diffusivity(
            D_base
        ).set_partition_coefficient(Phi)
        P_vs_Phi.append(solver.compute_permeability())

    ax3.plot(Phi_values, P_vs_Phi, "r-", linewidth=2)
    ax3.set_xlabel("Partition coefficient Phi")
    ax3.set_ylabel("Permeability (m/s)")
    ax3.set_title("P = D*Phi/L")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(bt.get_result_path("permeability_study.png", EXAMPLE_NAME))
    plt.show()

    print("Permeability scales as P = D*Phi/L:")
    print("  - Inversely with thickness (P ~ 1/L)")
    print("  - Linearly with diffusivity (P ~ D)")
    print("  - Linearly with partition coefficient (P ~ Phi)")


if __name__ == "__main__":
    print("=" * 60)
    print("Steady-State Membrane Diffusion Examples")
    print("=" * 60)

    run_simple_membrane()
    run_hindered_diffusion()
    run_multilayer_skin()
    run_permeability_study()

    print("\n" + "=" * 60)
    print("All membrane diffusion examples completed!")
    print("=" * 60)
