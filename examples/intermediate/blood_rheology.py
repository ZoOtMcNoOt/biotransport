#!/usr/bin/env python3
"""
Blood Rheology Examples - Non-Newtonian Flow Models

This example demonstrates the non-Newtonian viscosity models
essential for accurate blood flow simulation.

BMEN 341 Relevance:
- Part II: Non-Newtonian fluid mechanics
- Blood as a complex fluid (red blood cell effects)
- Yield stress behavior (rouleaux formation)
- Shear-thinning in blood vessels

Blood exhibits non-Newtonian behavior:
1. Shear-thinning: viscosity decreases with shear rate
2. Yield stress: blood "gels" at very low shear (rouleaux)
3. Viscoelasticity (not modeled here)
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt


def example_viscosity_models():
    """
    Compare different non-Newtonian viscosity models.
    
    Demonstrates how viscosity varies with shear rate for:
    - Newtonian (constant viscosity)
    - Power-law (u = K.gammȧ^(n-1))
    - Carreau (smooth transition between low and high shear)
    - Casson (yield stress model for blood)
    """
    print("=" * 60)
    print("Example 1: Non-Newtonian Viscosity Model Comparison")
    print("=" * 60)
    
    # Shear rate range (physiological for blood: ~1 to ~1000 s^-1)
    gamma_dot = np.logspace(-1, 3, 100)  # 0.1 to 1000 s^-1
    
    # Reference parameters (blood-like)
    mu_inf = 0.0035   # High shear viscosity (Pa.s)
    mu_0 = 0.056      # Low shear viscosity (Pa.s)
    
    # Create models
    models = {
        'Newtonian': bt.NewtonianModel(mu_inf),
        'Power-law (n=0.7)': bt.PowerLawModel(0.017, 0.7),  # K, n (shear-thinning)
        'Carreau': bt.CarreauModel(mu_0, mu_inf, 3.313, 0.3568),  # u₀, u∞, λ, n
        'Casson': bt.CassonModel(0.004, 0.003),  # tau_y, u_p
    }
    
    # Calculate viscosities
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    for name, model in models.items():
        mu = np.array([model.viscosity(g) for g in gamma_dot])
        ax.loglog(gamma_dot, mu * 1000, '-', linewidth=2, label=name)
    
    ax.set_xlabel('Shear rate gammȧ (s^-1)')
    ax.set_ylabel('Apparent viscosity u (mPa.s)')
    ax.set_title('Viscosity vs Shear Rate')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0.1, 1000])
    ax.set_ylim([1, 100])
    
    # Add physiological shear rate regions
    ax.axvspan(1, 10, alpha=0.2, color='red', label='Venous')
    ax.axvspan(100, 500, alpha=0.2, color='blue', label='Arterial')
    ax.text(3, 2, 'Venous\nflow', fontsize=8, ha='center')
    ax.text(200, 2, 'Arterial\nflow', fontsize=8, ha='center')
    
    # Shear stress vs shear rate
    ax = axes[1]
    for name, model in models.items():
        tau = np.array([model.viscosity(g) * g for g in gamma_dot])
        ax.loglog(gamma_dot, tau, '-', linewidth=2, label=name)
    
    ax.set_xlabel('Shear rate gammȧ (s^-1)')
    ax.set_ylabel('Shear stress tau (Pa)')
    ax.set_title('Shear Stress vs Shear Rate')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(bt.get_result_path('blood_rheology_models.png'), dpi=150)
    plt.close()
    
    # Print model parameters
    print("\n  Model Parameters:")
    print("  " + "-" * 50)
    
    for name, model in models.items():
        mu_low = model.viscosity(1.0)
        mu_high = model.viscosity(100.0)
        print(f"  {name}:")
        print(f"    u(1/s) = {mu_low*1000:.2f} mPa.s")
        print(f"    u(100/s) = {mu_high*1000:.2f} mPa.s")
        print(f"    Ratio = {mu_low/mu_high:.2f}")
    
    print(f"\n  Plot saved to results/blood_rheology_models.png")


def example_hematocrit_effect():
    """
    Effect of hematocrit on blood viscosity.
    
    Hematocrit (H) = volume fraction of red blood cells
    Normal range: 38-54% for males, 36-48% for females
    """
    print("\n" + "=" * 60)
    print("Example 2: Hematocrit Effect on Blood Viscosity")
    print("=" * 60)
    
    # Hematocrit range
    hct_values = np.linspace(0.30, 0.60, 7)
    
    # Shear rates to evaluate
    gamma_rates = [10, 50, 100, 500]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Casson model viscosity vs hematocrit
    ax = axes[0]
    
    for gamma in gamma_rates:
        mu_values = []
        for hct in hct_values:
            blood = bt.blood_casson_model(hct)
            mu_values.append(blood.viscosity(gamma) * 1000)  # mPa.s
        ax.plot(hct_values * 100, mu_values, 'o-', linewidth=2, 
                markersize=8, label=f'gammȧ = {gamma} s^-1')
    
    ax.set_xlabel('Hematocrit (%)')
    ax.set_ylabel('Apparent viscosity (mPa.s)')
    ax.set_title('Viscosity vs Hematocrit (Casson Model)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Normal hematocrit range
    ax.axvspan(36, 54, alpha=0.2, color='green')
    ax.text(45, ax.get_ylim()[1]*0.9, 'Normal\nrange', fontsize=10, ha='center')
    
    # Yield stress vs hematocrit
    ax = axes[1]
    tau_y_values = []
    mu_p_values = []
    
    for hct in hct_values:
        blood = bt.blood_casson_model(hct)
        tau_y_values.append(blood.yield_stress() * 1000)  # mPa
        mu_p_values.append(blood.plastic_viscosity() * 1000)  # mPa.s
    
    ax.plot(hct_values * 100, tau_y_values, 'ro-', linewidth=2, 
            markersize=8, label='Yield stress tau_y')
    ax.set_xlabel('Hematocrit (%)')
    ax.set_ylabel('Yield stress (mPa)', color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(hct_values * 100, mu_p_values, 'bs-', linewidth=2, 
             markersize=8, label='Plastic viscosity u_p')
    ax2.set_ylabel('Plastic viscosity (mPa.s)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax.set_title('Casson Parameters vs Hematocrit')
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(bt.get_result_path('blood_hematocrit_effect.png'), dpi=150)
    plt.close()
    
    print("  Hematocrit effects on Casson blood model:")
    print("  " + "-" * 50)
    print(f"  {'Hct (%)':>10} {'tau_y (mPa)':>12} {'u_p (mPa.s)':>14} {'u_eff @100/s':>14}")
    print("  " + "-" * 50)
    
    for hct in hct_values:
        blood = bt.blood_casson_model(hct)
        print(f"  {hct*100:>10.0f} {blood.yield_stress()*1000:>12.3f} "
              f"{blood.plastic_viscosity()*1000:>14.3f} {blood.viscosity(100)*1000:>14.3f}")
    
    print(f"\n  Plot saved to results/blood_hematocrit_effect.png")


def example_vessel_size_effect():
    """
    Fahraeus-Lindqvist effect: blood viscosity depends on vessel diameter.
    
    In small vessels (< 300 um), apparent viscosity decreases due to:
    1. Cell-free layer near walls
    2. RBC alignment in flow direction
    """
    print("\n" + "=" * 60)
    print("Example 3: Fahraeus-Lindqvist Effect (Vessel Size)")
    print("=" * 60)
    
    # Vessel diameters
    diameters = np.array([10, 20, 50, 100, 200, 500, 1000, 5000])  # um
    
    # Hematocrit (systemic)
    H_sys = 0.45
    
    # Empirical correlations for tube hematocrit and relative viscosity
    # (Pries et al., 1992 correlations)
    
    def tube_hematocrit(D, H_d):
        """Hematocrit in tube vs discharge hematocrit."""
        if D >= 300:
            return H_d
        # Fahraeus effect
        x = np.log10(D)
        H_t = H_d * (H_d + (1-H_d) * (1 + 1.7*np.exp(-0.35*D) - 0.6*np.exp(-0.01*D)))
        return min(H_t, H_d)
    
    def relative_viscosity_pries(D, H_d):
        """Relative viscosity (u/u_plasma) using Pries correlation."""
        if D > 1000:
            D = 1000  # Use large vessel limit
        
        # Parameters
        mu_45 = 220 * np.exp(-1.3 * D) + 3.2 - 2.44 * np.exp(-0.06 * D**0.645)
        C = (0.8 + np.exp(-0.075 * D)) * (-1 + 1 / (1 + 1e-11 * D**12)) + 1 / (1 + 1e-11 * D**12)
        
        mu_rel = 1 + (mu_45 - 1) * ((1 - H_d)**C - 1) / ((1 - 0.45)**C - 1)
        return max(mu_rel, 1.0)
    
    # Calculate viscosities
    mu_plasma = 0.0012  # Pa.s
    mu_bulk_blood = bt.blood_casson_model(H_sys).viscosity(100)  # At typical shear
    
    mu_apparent = []
    H_tube = []
    
    for D in diameters:
        H_t = tube_hematocrit(D, H_sys)
        mu_rel = relative_viscosity_pries(D, H_sys)
        
        H_tube.append(H_t)
        mu_apparent.append(mu_plasma * mu_rel)
    
    mu_apparent = np.array(mu_apparent)
    H_tube = np.array(H_tube)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Apparent viscosity vs vessel diameter
    ax = axes[0]
    ax.semilogx(diameters, mu_apparent * 1000, 'bo-', linewidth=2, markersize=10)
    ax.axhline(mu_bulk_blood * 1000, color='red', linestyle='--', 
               linewidth=2, label='Bulk blood viscosity')
    ax.set_xlabel('Vessel diameter (um)')
    ax.set_ylabel('Apparent viscosity (mPa.s)')
    ax.set_title('Fahraeus-Lindqvist Effect')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Add vessel type annotations
    ax.axvspan(5, 20, alpha=0.2, color='red')
    ax.axvspan(20, 100, alpha=0.2, color='orange')
    ax.axvspan(100, 1000, alpha=0.2, color='yellow')
    ax.axvspan(1000, 10000, alpha=0.2, color='green')
    ax.text(10, ax.get_ylim()[1]*0.95, 'Capillaries', fontsize=8, ha='center')
    ax.text(50, ax.get_ylim()[1]*0.95, 'Arterioles', fontsize=8, ha='center')
    ax.text(300, ax.get_ylim()[1]*0.95, 'Small\narteries', fontsize=8, ha='center')
    
    # Tube hematocrit vs vessel diameter
    ax = axes[1]
    ax.semilogx(diameters, H_tube * 100, 'go-', linewidth=2, markersize=10)
    ax.axhline(H_sys * 100, color='red', linestyle='--', 
               linewidth=2, label='Systemic hematocrit')
    ax.set_xlabel('Vessel diameter (um)')
    ax.set_ylabel('Tube hematocrit (%)')
    ax.set_title('Fahraeus Effect')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(bt.get_result_path('blood_fahraeus_lindqvist.png'), dpi=150)
    plt.close()
    
    print("  Apparent viscosity vs vessel diameter:")
    print("  " + "-" * 50)
    print(f"  {'Diameter (um)':>15} {'u_app (mPa.s)':>15} {'H_tube (%)':>12}")
    print("  " + "-" * 50)
    
    for D, mu, H in zip(diameters, mu_apparent, H_tube):
        print(f"  {D:>15.0f} {mu*1000:>15.2f} {H*100:>12.1f}")
    
    print(f"\n  Bulk blood viscosity (at 100 s^-1): {mu_bulk_blood*1000:.2f} mPa.s")
    print(f"  Plot saved to results/blood_fahraeus_lindqvist.png")


def example_wall_shear_stress():
    """
    Wall shear stress in blood vessels with non-Newtonian blood.
    
    WSS is critical for:
    - Endothelial cell mechanotransduction
    - Atherosclerosis development (low/oscillatory WSS)
    - Thrombus formation
    """
    print("\n" + "=" * 60)
    print("Example 4: Wall Shear Stress in Blood Vessels")
    print("=" * 60)
    
    # Vessel parameters (typical)
    vessels = {
        'Aorta': {'D': 25e-3, 'Q': 5e-6},          # 25 mm, 5 L/min = 5e-6 m³/s at peak
        'Carotid': {'D': 6e-3, 'Q': 6e-6/60},      # 6 mm, 360 mL/min
        'Femoral': {'D': 8e-3, 'Q': 4e-6/60},      # 8 mm, 240 mL/min
        'Coronary': {'D': 3e-3, 'Q': 1e-6/60},     # 3 mm, 60 mL/min
        'Arteriole': {'D': 50e-6, 'Q': 1e-12},     # 50 um
    }
    
    # Blood properties
    hct = 0.45
    blood_casson = bt.blood_casson_model(hct)
    blood_carreau = bt.blood_carreau_model(hct)
    mu_newtonian = 0.0035  # Constant viscosity assumption
    
    print("  Wall shear stress comparison (Newtonian vs Non-Newtonian):")
    print("  " + "-" * 80)
    print(f"  {'Vessel':>12} {'D (mm)':>8} {'gamma_wall (1/s)':>12} "
          f"{'tau_Newt (Pa)':>12} {'tau_Casson (Pa)':>14} {'tau_Carreau (Pa)':>14}")
    print("  " + "-" * 80)
    
    results = []
    
    for name, params in vessels.items():
        D = params['D']
        Q = params['Q']
        R = D / 2
        
        # Average velocity
        U_avg = Q / (np.pi * R**2)
        
        # Wall shear rate (for Newtonian Poiseuille flow)
        gamma_wall = 8 * U_avg / D
        
        # Wall shear stress for different models
        tau_newtonian = mu_newtonian * gamma_wall
        
        mu_casson = blood_casson.viscosity(gamma_wall)
        tau_casson = mu_casson * gamma_wall
        
        mu_carreau = blood_carreau.viscosity(gamma_wall)
        tau_carreau = mu_carreau * gamma_wall
        
        results.append({
            'name': name,
            'D': D,
            'gamma_wall': gamma_wall,
            'tau_newt': tau_newtonian,
            'tau_casson': tau_casson,
            'tau_carreau': tau_carreau
        })
        
        print(f"  {name:>12} {D*1000:>8.2f} {gamma_wall:>12.1f} "
              f"{tau_newtonian:>12.3f} {tau_casson:>14.3f} {tau_carreau:>14.3f}")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [r['name'] for r in results]
    x = np.arange(len(names))
    width = 0.25
    
    tau_newt = [r['tau_newt'] for r in results]
    tau_casson = [r['tau_casson'] for r in results]
    tau_carreau = [r['tau_carreau'] for r in results]
    
    ax.bar(x - width, tau_newt, width, label='Newtonian', color='blue', alpha=0.7)
    ax.bar(x, tau_casson, width, label='Casson', color='red', alpha=0.7)
    ax.bar(x + width, tau_carreau, width, label='Carreau', color='green', alpha=0.7)
    
    ax.set_ylabel('Wall Shear Stress (Pa)')
    ax.set_title('Wall Shear Stress: Effect of Blood Rheology Model')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add physiological WSS range
    ax.axhspan(1, 7, alpha=0.2, color='green')
    ax.text(len(names)-0.5, 4, 'Normal arterial\nWSS range', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(bt.get_result_path('blood_wall_shear_stress.png'), dpi=150)
    plt.close()
    
    print(f"\n  Normal arterial WSS range: 1-7 Pa")
    print(f"  Low WSS (<0.4 Pa) associated with atherosclerosis")
    print(f"\n  Plot saved to results/blood_wall_shear_stress.png")


if __name__ == "__main__":
    print("Blood Rheology Examples for BMEN 341")
    print("Non-Newtonian fluid models for blood flow simulation")
    print()
    
    example_viscosity_models()
    example_hematocrit_effect()
    example_vessel_size_effect()
    example_wall_shear_stress()
    
    print("\n" + "=" * 60)
    print("All blood rheology examples completed!")
    print("=" * 60)
