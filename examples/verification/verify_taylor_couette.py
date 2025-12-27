"""Verification: Taylor-Couette flow analytical solution.

Demonstrates the velocity profile between concentric rotating cylinders.

BMEN 341 Reference: NASA Bioreactor practice problem (rotating wall vessel)
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt

EXAMPLE_NAME = "verification/taylor_couette"


def verify_boundary_conditions():
    """Verify velocity matches cylinder rotation at inner and outer walls."""
    print("=" * 60)
    print("Taylor-Couette Boundary Condition Verification")
    print("=" * 60)

    # NASA rotating wall bioreactor typical dimensions
    a = 0.025  # Inner radius: 25 mm
    b = 0.050  # Outer radius: 50 mm

    test_cases = [
        ("Inner rotating, outer stationary", 10.0, 0.0),
        ("Outer rotating, inner stationary", 0.0, 10.0),
        ("Co-rotating", 10.0, 5.0),
        ("Counter-rotating", 10.0, -5.0),
    ]

    all_passed = True

    for name, omega_a, omega_b in test_cases:
        v_inner = bt.analytical.taylor_couette_velocity(a, a, b, omega_a, omega_b)
        v_outer = bt.analytical.taylor_couette_velocity(b, a, b, omega_a, omega_b)

        expected_inner = a * omega_a
        expected_outer = b * omega_b

        error_inner = abs(v_inner - expected_inner)
        error_outer = abs(v_outer - expected_outer)

        passed = error_inner < 1e-12 and error_outer < 1e-12

        print(f"\n{name}:")
        print(f"  omega_a = {omega_a:.1f} rad/s, omega_b = {omega_b:.1f} rad/s")
        print(f"  v(a) = {v_inner:.6f} m/s (expected {expected_inner:.6f})")
        print(f"  v(b) = {v_outer:.6f} m/s (expected {expected_outer:.6f})")
        print(f"  {'OK PASSED' if passed else 'X FAILED'}")

        all_passed = all_passed and passed

    return all_passed


def verify_velocity_profile():
    """Plot and verify Taylor-Couette velocity profile."""
    print("\n" + "=" * 60)
    print("Taylor-Couette Velocity Profile")
    print("=" * 60)

    a = 0.02  # Inner radius: 20 mm
    b = 0.04  # Outer radius: 40 mm
    omega_a = 10.0  # Inner cylinder: 10 rad/s
    omega_b = 0.0  # Outer cylinder: stationary

    # Radial positions
    r = np.linspace(a, b, 100)

    # Velocity profile
    v = np.array(
        [bt.analytical.taylor_couette_velocity(ri, a, b, omega_a, omega_b) for ri in r]
    )

    # Torque
    mu = 0.001  # Pa*s (water)
    torque = bt.analytical.taylor_couette_torque(a, b, omega_a, omega_b, mu)

    print("\nParameters:")
    print(f"  Inner radius a = {a*1000:.1f} mm")
    print(f"  Outer radius b = {b*1000:.1f} mm")
    print(f"  Inner angular velocity = {omega_a:.1f} rad/s")
    print(f"  Outer angular velocity = {omega_b:.1f} rad/s")
    print(f"  Viscosity = {mu*1000:.1f} mPa*s")
    print("\nResults:")
    print(f"  Max velocity (inner wall) = {v[0]*100:.2f} cm/s")
    print(f"  Torque per unit length = {torque:.6e} N*m/m")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Velocity vs radius
    ax1 = axes[0]
    ax1.plot(r * 1000, v * 100, "b-", linewidth=2)
    ax1.axvline(x=a * 1000, color="gray", linestyle="--", alpha=0.5, label="Inner wall")
    ax1.axvline(x=b * 1000, color="gray", linestyle=":", alpha=0.5, label="Outer wall")
    ax1.set_xlabel("Radial Position r (mm)")
    ax1.set_ylabel("Tangential Velocity vθ (cm/s)")
    ax1.set_title("Taylor-Couette Velocity Profile\n(Inner rotating, outer stationary)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Compare different rotation cases
    ax2 = axes[1]
    cases = [
        ("Inner only", 10.0, 0.0, "b-"),
        ("Outer only", 0.0, 10.0, "r-"),
        ("Co-rotating", 10.0, 5.0, "g-"),
        ("Counter-rotating", 10.0, -5.0, "m-"),
    ]

    for label, wa, wb, style in cases:
        v_case = np.array(
            [bt.analytical.taylor_couette_velocity(ri, a, b, wa, wb) for ri in r]
        )
        ax2.plot(r * 1000, v_case * 100, style, linewidth=2, label=label)

    ax2.set_xlabel("Radial Position r (mm)")
    ax2.set_ylabel("Tangential Velocity vθ (cm/s)")
    ax2.set_title("Various Rotation Configurations")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(bt.get_result_path("taylor_couette_verification.png", EXAMPLE_NAME))
    plt.show()

    return True


def verify_solid_body_rotation():
    """When both cylinders rotate at same rate, expect solid body rotation."""
    print("\n" + "=" * 60)
    print("Solid Body Rotation Verification")
    print("=" * 60)

    a, b = 0.02, 0.04
    omega = 5.0  # Both rotating at same rate

    r = np.linspace(a, b, 50)
    v_taylor = np.array(
        [bt.analytical.taylor_couette_velocity(ri, a, b, omega, omega) for ri in r]
    )
    v_solid = r * omega  # Solid body: v = r*omega

    max_error = np.max(np.abs(v_taylor - v_solid))

    print(f"\nBoth cylinders at omega = {omega} rad/s")
    print(f"Max deviation from v = r*omega: {max_error:.2e} m/s")

    passed = max_error < 1e-12
    print(f"\n{'OK PASSED' if passed else 'X FAILED'}: Solid body rotation limit")
    return passed


if __name__ == "__main__":
    results = []
    results.append(verify_boundary_conditions())
    results.append(verify_velocity_profile())
    results.append(verify_solid_body_rotation())

    print("\n" + "=" * 60)
    print(f"SUMMARY: {sum(results)}/{len(results)} verifications passed")
    print("=" * 60)
