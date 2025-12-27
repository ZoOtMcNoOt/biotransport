"""Verification: Poiseuille flow analytical solution.

Compares the analytical Poiseuille velocity profile against itself
(self-consistency check) and demonstrates the parabolic profile.

BMEN 341 Reference: Weeks 3-4 (Viscous Flow, Hagen-Poiseuille equation)
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt

EXAMPLE_NAME = "verification/poiseuille"


def verify_poiseuille_profile():
    """Verify Poiseuille velocity profile is parabolic with correct max."""
    print("=" * 60)
    print("Poiseuille Flow Verification")
    print("=" * 60)

    # Blood vessel parameters
    R = 1.0e-3  # Radius: 1 mm
    dp_dz = -1000.0  # Pressure gradient: -1000 Pa/m (flow in +z)
    mu = 3.5e-3  # Blood viscosity: 3.5 mPa*s

    # Radial positions
    r = np.linspace(0, R, 100)

    # Compute velocity profile
    v = np.array([bt.analytical.poiseuille_velocity(ri, R, dp_dz, mu) for ri in r])

    # Analytical maximum velocity at centerline
    v_max_analytical = bt.analytical.poiseuille_max_velocity(R, dp_dz, mu)
    v_max_numerical = v[0]  # At r=0

    # Wall shear stress
    tau_w = bt.analytical.poiseuille_wall_shear(R, dp_dz)

    # Flow rate
    Q = bt.analytical.poiseuille_flow_rate(R, dp_dz, mu)

    print("\nParameters:")
    print(f"  Radius R = {R*1000:.2f} mm")
    print(f"  Pressure gradient = {dp_dz:.0f} Pa/m")
    print(f"  Viscosity mu = {mu*1000:.2f} mPa.s")

    print("\nResults:")
    print(f"  Max velocity (centerline) = {v_max_analytical*100:.4f} cm/s")
    print(f"  Wall shear stress = {tau_w:.4f} Pa")
    print(f"  Volume flow rate = {Q*1e9:.4f} mm^3/s")

    # Verification checks
    error_vmax = abs(v_max_numerical - v_max_analytical) / v_max_analytical
    error_wall = abs(v[-1]) / v_max_analytical  # Should be zero at wall

    print("\nVerification:")
    print(f"  v_max error: {error_vmax*100:.6f}%")
    print(f"  v(R)/v_max: {error_wall:.2e} (should be ~0)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Velocity profile
    ax1 = axes[0]
    ax1.plot(r * 1000, v * 100, "b-", linewidth=2, label="Analytical")
    ax1.axhline(
        y=v_max_analytical * 100,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"v_max = {v_max_analytical*100:.3f} cm/s",
    )
    ax1.set_xlabel("Radial Position r (mm)")
    ax1.set_ylabel("Velocity v(r) (cm/s)")
    ax1.set_title("Poiseuille Velocity Profile")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, R * 1000)

    # Normalized profile (should be parabolic: 1 - (r/R)²)
    ax2 = axes[1]
    v_normalized = v / v_max_analytical
    r_normalized = r / R
    parabola = 1 - r_normalized**2

    ax2.plot(r_normalized, v_normalized, "b-", linewidth=2, label="v(r)/v_max")
    ax2.plot(r_normalized, parabola, "r--", linewidth=2, alpha=0.7, label="1 - (r/R)²")
    ax2.set_xlabel("Normalized Radius r/R")
    ax2.set_ylabel("Normalized Velocity v/v_max")
    ax2.set_title("Parabolic Profile Verification")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(bt.get_result_path("poiseuille_verification.png", EXAMPLE_NAME))
    plt.show()

    # Return verification status
    passed = error_vmax < 1e-10 and error_wall < 1e-10
    print(f"\n{'OK PASSED' if passed else 'X FAILED'}: Poiseuille verification")
    return passed


def verify_flow_rate_formula():
    """Verify Q = piR^4|dp/dz| / (8mu) (Hagen-Poiseuille equation)."""
    print("\n" + "=" * 60)
    print("Hagen-Poiseuille Flow Rate Verification")
    print("=" * 60)

    R = 0.5e-3  # 0.5 mm capillary
    dp_dz = -500.0  # Pa/m
    mu = 1.0e-3  # Water viscosity

    Q_analytical = bt.analytical.poiseuille_flow_rate(R, dp_dz, mu)
    Q_formula = np.pi * R**4 * abs(dp_dz) / (8 * mu)

    error = abs(Q_analytical - Q_formula) / Q_formula

    print(f"\nQ from biotransport.analytical: {Q_analytical:.6e} m^3/s")
    print(f"Q from piR^4|dp/dz|/(8mu):         {Q_formula:.6e} m^3/s")
    print(f"Relative error: {error*100:.6e}%")

    passed = error < 1e-10
    print(f"\n{'OK PASSED' if passed else 'X FAILED'}: Flow rate formula")
    return passed


if __name__ == "__main__":
    results = []
    results.append(verify_poiseuille_profile())
    results.append(verify_flow_rate_formula())

    print("\n" + "=" * 60)
    print(f"SUMMARY: {sum(results)}/{len(results)} verifications passed")
    print("=" * 60)
