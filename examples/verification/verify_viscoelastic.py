"""Verification: Viscoelastic model analytical solutions.

Demonstrates stress relaxation and creep for Maxwell, Kelvin-Voigt,
Standard Linear Solid (SLS), and Burgers models.

BMEN 341 Reference: HW6 Problem 5 (Viscoelasticity)
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt

EXAMPLE_NAME = "verification/viscoelastic"


def verify_maxwell_model():
    """Verify Maxwell stress relaxation: sigma(t) = E*eps0*exp(-t/tau)."""
    print("=" * 60)
    print("Maxwell Model Verification")
    print("=" * 60)

    E = 1000.0  # Pa (spring modulus)
    eta = 100.0  # Pa*s (dashpot viscosity)
    eps0 = 0.01  # Initial strain (1%)

    tau = bt.analytical.maxwell_relaxation_time(E, eta)
    print("\nParameters:")
    print(f"  E = {E:.0f} Pa")
    print(f"  eta = {eta:.0f} Pa*s")
    print(f"  eps0 = {eps0*100:.1f}%")
    print(f"  tau = eta/E = {tau:.3f} s")

    # Time array (0 to 5tau)
    t = np.linspace(0, 5 * tau, 200)

    # Stress from biotransport
    sigma = np.array([bt.analytical.maxwell_relaxation(E, eta, eps0, ti) for ti in t])

    # Expected: sigma(t) = E*eps0*exp(-t/tau)
    sigma_expected = E * eps0 * np.exp(-t / tau)

    max_error = np.max(np.abs(sigma - sigma_expected))

    print("\nVerification:")
    print(f"  sigma(0) = {sigma[0]:.4f} Pa (expected {E*eps0:.4f})")
    print(
        f"  sigma(tau) = {bt.analytical.maxwell_relaxation(E, eta, eps0, tau):.4f} Pa "
        f"(expected {E*eps0/np.e:.4f})"
    )
    print(f"  Max error: {max_error:.2e} Pa")

    passed = max_error < 1e-10
    print(f"\n{'OK PASSED' if passed else 'X FAILED'}: Maxwell relaxation")
    return passed, t / tau, sigma / (E * eps0), "Maxwell"


def verify_kelvin_voigt_model():
    """Verify Kelvin-Voigt creep: eps(t) = (sigma0/E)*(1 - exp(-t/tau))."""
    print("\n" + "=" * 60)
    print("Kelvin-Voigt Model Verification")
    print("=" * 60)

    E = 1000.0
    eta = 100.0
    sigma0 = 10.0  # Applied stress (Pa)

    tau = eta / E
    print("\nParameters:")
    print(f"  E = {E:.0f} Pa")
    print(f"  eta = {eta:.0f} Pa*s")
    print(f"  sigma0 = {sigma0:.1f} Pa")
    print(f"  tau = eta/E = {tau:.3f} s")

    t = np.linspace(0, 5 * tau, 200)
    eps = np.array([bt.analytical.kelvin_voigt_creep(E, eta, sigma0, ti) for ti in t])
    eps_expected = (sigma0 / E) * (1 - np.exp(-t / tau))

    max_error = np.max(np.abs(eps - eps_expected))
    eps_inf = sigma0 / E

    print("\nVerification:")
    print(f"  eps(0) = {eps[0]:.6f} (expected 0)")
    print(f"  eps(inf) = {eps[-1]:.6f} (expected {eps_inf:.6f})")
    print(f"  Max error: {max_error:.2e}")

    passed = max_error < 1e-10
    print(f"\n{'OK PASSED' if passed else 'X FAILED'}: Kelvin-Voigt creep")
    return passed, t / tau, eps / eps_inf, "Kelvin-Voigt"


def verify_sls_model():
    """Verify Standard Linear Solid stress relaxation."""
    print("\n" + "=" * 60)
    print("Standard Linear Solid (SLS) Verification")
    print("=" * 60)

    E1 = 500.0  # Equilibrium modulus
    E2 = 500.0  # Relaxation modulus
    eta = 100.0  # Dashpot
    eps0 = 0.01

    tau = eta / E2
    print("\nParameters:")
    print(f"  E1 = {E1:.0f} Pa (equilibrium spring)")
    print(f"  E2 = {E2:.0f} Pa (Maxwell spring)")
    print(f"  eta = {eta:.0f} Pa*s")
    print(f"  tau = eta/E2 = {tau:.3f} s")

    t = np.linspace(0, 5 * tau, 200)
    sigma = np.array([bt.analytical.sls_relaxation(E1, E2, eta, eps0, ti) for ti in t])

    # Expected: sigma(t) = eps0*(E1 + E2*exp(-t/tau))
    sigma_expected = eps0 * (E1 + E2 * np.exp(-t / tau))

    sigma_0 = eps0 * (E1 + E2)
    sigma_inf = eps0 * E1

    print("\nVerification:")
    print(f"  sigma(0) = {sigma[0]:.4f} Pa (expected {sigma_0:.4f})")
    print(f"  sigma(inf) = {sigma[-1]:.4f} Pa (expected {sigma_inf:.4f})")

    max_error = np.max(np.abs(sigma - sigma_expected))
    print(f"  Max error: {max_error:.2e} Pa")

    passed = max_error < 1e-10
    print(f"\n{'OK PASSED' if passed else 'X FAILED'}: SLS relaxation")
    return passed, t / tau, sigma / sigma_0, "SLS"


def verify_burgers_model():
    """Verify Burgers 4-parameter creep model."""
    print("\n" + "=" * 60)
    print("Burgers Model Verification")
    print("=" * 60)

    E1 = 1000.0  # Maxwell spring
    mu1 = 1000.0  # Maxwell dashpot
    E2 = 500.0  # Kelvin-Voigt spring
    mu2 = 50.0  # Kelvin-Voigt dashpot
    sigma0 = 10.0

    tau2 = mu2 / E2
    print("\nParameters:")
    print(f"  E1 = {E1:.0f} Pa, mu1 = {mu1:.0f} Pa*s (Maxwell arm)")
    print(f"  E2 = {E2:.0f} Pa, mu2 = {mu2:.0f} Pa*s (Kelvin-Voigt arm)")
    print(f"  sigma0 = {sigma0:.1f} Pa")
    print(f"  tau2 = mu2/E2 = {tau2:.3f} s")

    t = np.linspace(0, 10 * tau2, 200)
    eps = np.array(
        [bt.analytical.burgers_creep(E1, mu1, E2, mu2, sigma0, ti) for ti in t]
    )
    J = np.array([bt.analytical.burgers_compliance(E1, mu1, E2, mu2, ti) for ti in t])

    # Expected compliance: J(t) = 1/E1 + t/mu1 + (1/E2)*(1 - exp(-t/tau2))
    J_expected = 1 / E1 + t / mu1 + (1 / E2) * (1 - np.exp(-t / tau2))

    max_error = np.max(np.abs(J - J_expected))

    print("\nVerification:")
    print(f"  eps(0) = {eps[0]:.6f} (expected {sigma0/E1:.6f})")
    print(f"  J(0) = {J[0]:.6f} Pa^-1 (expected {1/E1:.6f})")
    print(f"  Max compliance error: {max_error:.2e} Pa^-1")

    passed = max_error < 1e-10
    print(f"\n{'OK PASSED' if passed else 'X FAILED'}: Burgers creep")
    return passed, t / tau2, eps / (sigma0 / E1), "Burgers"


def verify_complex_modulus():
    """Verify complex modulus utilities."""
    print("\n" + "=" * 60)
    print("Complex Modulus Utilities Verification")
    print("=" * 60)

    # Test with 3-4-5 triangle
    G1, G2 = 3.0, 4.0

    G_star = bt.analytical.complex_modulus_magnitude(G1, G2)
    tan_delta = bt.analytical.loss_tangent(G1, G2)
    delta = bt.analytical.phase_angle(G1, G2)

    print(f"\nTest case: G1 = {G1}, G2 = {G2}")
    print(f"  |G*| = sqrt(G1² + G2²) = {G_star:.6f} (expected 5.0)")
    print(f"  tan(delta) = G2/G1 = {tan_delta:.6f} (expected {4/3:.6f})")
    print(
        f"  delta = atan2(G2, G1) = {np.degrees(delta):.4f}° "
        f"(expected {np.degrees(np.arctan2(4, 3)):.4f}°)"
    )

    passed = (
        abs(G_star - 5.0) < 1e-10
        and abs(tan_delta - 4 / 3) < 1e-10
        and abs(delta - np.arctan2(4, 3)) < 1e-10
    )

    print(f"\n{'OK PASSED' if passed else 'X FAILED'}: Complex modulus utilities")
    return passed


def plot_all_models():
    """Create comparison plots for all viscoelastic models."""
    print("\n" + "=" * 60)
    print("Generating Comparison Plots")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Relaxation models (Maxwell, SLS)
    ax1 = axes[0, 0]
    E, eta = 1000.0, 100.0
    eps0 = 0.01
    tau = eta / E
    t = np.linspace(0, 5 * tau, 200)

    # Maxwell
    sigma_maxwell = np.array(
        [bt.analytical.maxwell_relaxation(E, eta, eps0, ti) for ti in t]
    )
    ax1.plot(t / tau, sigma_maxwell / (E * eps0), "b-", linewidth=2, label="Maxwell")

    # SLS
    E1, E2 = 500.0, 500.0
    sigma_sls = np.array(
        [bt.analytical.sls_relaxation(E1, E2, eta, eps0, ti) for ti in t]
    )
    ax1.plot(t / tau, sigma_sls / (eps0 * (E1 + E2)), "r-", linewidth=2, label="SLS")

    ax1.set_xlabel("Normalized Time t/tau")
    ax1.set_ylabel("Normalized Stress sigma/sigma0")
    ax1.set_title("Stress Relaxation Models")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Creep models (Kelvin-Voigt, Burgers)
    ax2 = axes[0, 1]
    sigma0 = 10.0

    # Kelvin-Voigt
    eps_kv = np.array(
        [bt.analytical.kelvin_voigt_creep(E, eta, sigma0, ti) for ti in t]
    )
    ax2.plot(t / tau, eps_kv / (sigma0 / E), "g-", linewidth=2, label="Kelvin-Voigt")

    # Burgers
    E1, mu1, E2, mu2 = 1000.0, 1000.0, 500.0, 50.0
    t_burgers = np.linspace(0, 1.0, 200)
    eps_burgers = np.array(
        [bt.analytical.burgers_creep(E1, mu1, E2, mu2, sigma0, ti) for ti in t_burgers]
    )
    ax2.plot(t_burgers, eps_burgers * 100, "m-", linewidth=2, label="Burgers")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Strain (%)")
    ax2.set_title("Creep Models")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Model comparison: relaxation
    ax3 = axes[1, 0]
    t_norm = np.linspace(0, 5, 200)

    # Normalized relaxation functions
    ax3.plot(t_norm, np.exp(-t_norm), "b-", linewidth=2, label="Maxwell: exp(-t/tau)")
    ax3.plot(
        t_norm,
        0.5 + 0.5 * np.exp(-t_norm),
        "r-",
        linewidth=2,
        label="SLS: E1/(E1+E2) + E2/(E1+E2)*exp(-t/tau)",
    )

    ax3.set_xlabel("Normalized Time t/tau")
    ax3.set_ylabel("G(t) / G(0)")
    ax3.set_title("Relaxation Modulus Comparison")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)

    # Model schematics (text)
    ax4 = axes[1, 1]
    ax4.axis("off")
    schematic = """
    VISCOELASTIC MODEL SUMMARY

    Maxwell (Series):
      ─[E]─[eta]─
      Stress relaxation: sigma(t) = E*eps0*exp(-t/tau)
      tau = eta/E

    Kelvin-Voigt (Parallel):
        ┌─[E]─┐
      ──┤     ├──
        └─[eta]─┘
      Creep: eps(t) = (sigma0/E)*(1 - exp(-t/tau))

    SLS (3-parameter):
          ┌─[E2]─[eta]─┐
      ──[E1]─┤         ├──
             └─────────┘
      Relaxation: sigma(t) = eps0*(E1 + E2*exp(-t/tau))

    Burgers (4-parameter):
        ┌─[E2]─[mu2]─┐
      ──┤           ├──[E1]──[mu1]──
        └───────────┘
      Creep: eps(t) = sigma0*J(t)
      J(t) = 1/E1 + t/mu1 + (1/E2)*(1 - exp(-t/tau2))
    """
    ax4.text(
        0.05,
        0.95,
        schematic,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )
    ax4.set_title("Model Schematics")

    plt.tight_layout()
    plt.savefig(bt.get_result_path("viscoelastic_verification.png", EXAMPLE_NAME))
    plt.show()


if __name__ == "__main__":
    results = []
    results.append(verify_maxwell_model()[0])
    results.append(verify_kelvin_voigt_model()[0])
    results.append(verify_sls_model()[0])
    results.append(verify_burgers_model()[0])
    results.append(verify_complex_modulus())

    plot_all_models()

    print("\n" + "=" * 60)
    print(f"SUMMARY: {sum(results)}/{len(results)} verifications passed")
    print("=" * 60)
