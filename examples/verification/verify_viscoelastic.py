"""Verification: Viscoelastic model analytical solutions.

Demonstrates stress relaxation and creep for Maxwell, Kelvin-Voigt,
Standard Linear Solid (SLS), and Burgers models.

Every library formula is checked twice, and neither check reuses the library:

1. against a closed form written out here from the spring-and-dashpot
   definition. It is built from the raw moduli and viscosities (never from the
   library's relaxation time) and uses ``expm1`` where the algebra allows, so
   the two evaluations follow different floating-point routes and cannot agree
   by construction.
2. against a fixed-step RK4 integration of the constitutive ODE that defines
   the model. That check shares no algebra at all with the closed form, so a
   wrong sign, a wrong time constant or a wrong prefactor in the library would
   show up rather than cancel.

Both are reported as relative errors against tolerances that a real
discrepancy would exceed.

BMEN 341 Reference: HW6 Problem 5 (Viscoelasticity)
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt

EXAMPLE_NAME = "verification/viscoelastic"

# Tolerances. The closed-form check compares two different roundings of the
# same exact quantity, so it lives near machine precision. The RK4 check
# carries the integrator's own truncation error on top of that.
CLOSED_FORM_TOL = 1e-13
ODE_TOL = 1e-9

# Steps used per RK4 checkpoint.
ODE_STEPS = 4000


def relative_error(computed, reference):
    """Max absolute deviation divided by the peak magnitude of the reference."""
    computed = np.asarray(computed, dtype=float)
    reference = np.asarray(reference, dtype=float)
    scale = np.max(np.abs(reference))
    return float(np.max(np.abs(computed - reference)) / scale)


def integrate_ode(rate, y0, t_end, steps=ODE_STEPS):
    """Fixed-step RK4 for dy/dt = rate(y). Written here, not taken from the library."""
    h = t_end / steps
    y = float(y0)
    for _ in range(steps):
        k1 = rate(y)
        k2 = rate(y + 0.5 * h * k1)
        k3 = rate(y + 0.5 * h * k2)
        k4 = rate(y + h * k3)
        y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y


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
    print(f"  eps0 = {eps0 * 100:.1f}%")
    print(f"  tau = eta/E = {tau:.3f} s")

    # Time array (0 to 5tau)
    t = np.linspace(0, 5 * tau, 200)

    # Stress from biotransport
    sigma = np.array([bt.analytical.maxwell_relaxation(E, eta, eps0, ti) for ti in t])

    # Independent closed form. Spring and dashpot in series share the stress,
    # and total strain rate is the sum of the two: deps/dt = sigmadot/E + sigma/eta.
    # Holding the strain at eps0 for t > 0 makes deps/dt = 0, so
    #   sigmadot = -(E/eta)*sigma,  sigma(0+) = E*eps0
    #   ==> sigma(t) = E*eps0*exp(-(E/eta)*t)
    # Written with E/eta rather than 1/tau so the arithmetic differs.
    sigma_closed = E * eps0 * np.exp(-(E / eta) * t)
    closed_error = relative_error(sigma, sigma_closed)

    # Independent numerical check: integrate the constitutive ODE itself.
    checkpoints = np.array([tau, 3.0 * tau, 5.0 * tau])
    sigma_ode = np.array(
        [
            integrate_ode(lambda s: -(E / eta) * s, E * eps0, float(tc))
            for tc in checkpoints
        ]
    )
    sigma_lib = np.array(
        [
            bt.analytical.maxwell_relaxation(E, eta, eps0, float(tc))
            for tc in checkpoints
        ]
    )
    ode_error = relative_error(sigma_lib, sigma_ode)

    print("\nSpot values:")
    print(f"  sigma(0) = {sigma[0]:.4f} Pa (expected {E * eps0:.4f})")
    print(
        f"  sigma(tau) = {bt.analytical.maxwell_relaxation(E, eta, eps0, tau):.4f} Pa "
        f"(expected {E * eps0 / np.e:.4f})"
    )
    print(
        f"  sigma(5 tau) = {sigma[-1]:.4f} Pa = {100 * np.exp(-5.0):.3f}% of sigma(0); "
        "the t -> inf limit is 0"
    )

    print("\nIndependent comparisons:")
    print(f"  vs closed form written from the model: {closed_error:.2e} (rel)")
    print(f"  vs RK4 of sigmadot = -(E/eta) sigma:   {ode_error:.2e} (rel)")

    passed = closed_error < CLOSED_FORM_TOL and ode_error < ODE_TOL
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

    # Independent closed form. Spring and dashpot in parallel share the strain
    # and their stresses add: sigma0 = E*eps + eta*epsdot, with eps(0) = 0, so
    #   eps(t) = (sigma0/E)*(1 - exp(-(E/eta)*t)) = -(sigma0/E)*expm1(-(E/eta)*t)
    # expm1 is a different evaluation path from the library's 1 - exp form.
    eps_closed = -(sigma0 / E) * np.expm1(-(E / eta) * t)
    closed_error = relative_error(eps, eps_closed)

    # Independent numerical check on the same constitutive ODE.
    checkpoints = np.array([tau, 3.0 * tau, 5.0 * tau])
    eps_ode = np.array(
        [
            integrate_ode(lambda e: (sigma0 - E * e) / eta, 0.0, float(tc))
            for tc in checkpoints
        ]
    )
    eps_lib = np.array(
        [
            bt.analytical.kelvin_voigt_creep(E, eta, sigma0, float(tc))
            for tc in checkpoints
        ]
    )
    ode_error = relative_error(eps_lib, eps_ode)

    eps_inf = sigma0 / E
    eps_5tau = bt.analytical.kelvin_voigt_creep(E, eta, sigma0, 5 * tau)
    eps_40tau = bt.analytical.kelvin_voigt_creep(E, eta, sigma0, 40 * tau)

    print("\nSpot values:")
    print(f"  eps(0) = {eps[0]:.6f} (expected 0)")
    print("\nApproach to the t -> inf limit eps_inf = sigma0/E:")
    print(f"  eps_inf     = {eps_inf:.9f}")
    print(
        f"  eps(5 tau)  = {eps_5tau:.9f} = {100 * eps_5tau / eps_inf:.3f}% of eps_inf "
        f"(short by exp(-5) = {100 * np.exp(-5.0):.3f}%)"
    )
    print(
        f"  eps(40 tau) = {eps_40tau:.9f}, which matches eps_inf to "
        f"{abs(eps_40tau - eps_inf):.2e}"
    )

    print("\nIndependent comparisons:")
    print(f"  vs closed form written from the model: {closed_error:.2e} (rel)")
    print(f"  vs RK4 of eta*epsdot = sigma0 - E*eps: {ode_error:.2e} (rel)")

    passed = closed_error < CLOSED_FORM_TOL and ode_error < ODE_TOL
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

    # Independent closed form. The equilibrium spring carries E1*eps0 for all
    # time. The Maxwell arm carries sigma2, which relaxes exactly as the Maxwell
    # model does with its own spring E2:
    #   sigma2dot = -(E2/eta)*sigma2,  sigma2(0+) = E2*eps0
    #   ==> sigma(t) = E1*eps0 + E2*eps0*exp(-(E2/eta)*t)
    sigma_closed = E1 * eps0 + E2 * eps0 * np.exp(-(E2 / eta) * t)
    closed_error = relative_error(sigma, sigma_closed)

    # Independent numerical check: integrate the Maxwell arm and add the spring.
    checkpoints = np.array([tau, 3.0 * tau, 5.0 * tau])
    sigma_ode = np.array(
        [
            E1 * eps0 + integrate_ode(lambda s: -(E2 / eta) * s, E2 * eps0, float(tc))
            for tc in checkpoints
        ]
    )
    sigma_lib = np.array(
        [
            bt.analytical.sls_relaxation(E1, E2, eta, eps0, float(tc))
            for tc in checkpoints
        ]
    )
    ode_error = relative_error(sigma_lib, sigma_ode)

    sigma_0 = eps0 * (E1 + E2)
    sigma_inf = eps0 * E1
    sigma_5tau = bt.analytical.sls_relaxation(E1, E2, eta, eps0, 5 * tau)
    sigma_40tau = bt.analytical.sls_relaxation(E1, E2, eta, eps0, 40 * tau)

    print("\nSpot values:")
    print(f"  sigma(0) = {sigma[0]:.4f} Pa (expected {sigma_0:.4f})")
    print("\nApproach to the t -> inf limit sigma_inf = eps0*E1:")
    print(f"  sigma_inf     = {sigma_inf:.9f} Pa")
    print(
        f"  sigma(5 tau)  = {sigma_5tau:.9f} Pa = {100 * sigma_5tau / sigma_inf:.3f}% "
        f"of sigma_inf (excess E2*eps0*exp(-5) = {eps0 * E2 * np.exp(-5.0):.6f} Pa)"
    )
    print(
        f"  sigma(40 tau) = {sigma_40tau:.9f} Pa, which matches sigma_inf to "
        f"{abs(sigma_40tau - sigma_inf):.2e} Pa"
    )

    print("\nIndependent comparisons:")
    print(f"  vs closed form written from the model:  {closed_error:.2e} (rel)")
    print(f"  vs RK4 of the relaxing Maxwell arm:     {ode_error:.2e} (rel)")

    passed = closed_error < CLOSED_FORM_TOL and ode_error < ODE_TOL
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

    # Independent closed form. A Maxwell arm in series with a Kelvin-Voigt arm
    # means the two strains add under a common stress sigma0:
    #   Maxwell arm       eps_M = sigma0/E1 + sigma0*t/mu1
    #   Kelvin-Voigt arm  mu2*epsdot + E2*eps = sigma0, eps(0) = 0
    #                     ==> eps_KV = -(sigma0/E2)*expm1(-(E2/mu2)*t)
    # so J = eps/sigma0 = 1/E1 + t/mu1 - (1/E2)*expm1(-(E2/mu2)*t).
    J_closed = 1.0 / E1 + t / mu1 - (1.0 / E2) * np.expm1(-(E2 / mu2) * t)
    closed_error = relative_error(J, J_closed)

    # The two library entry points must agree with each other as well.
    consistency_error = relative_error(eps, sigma0 * J)

    # Independent numerical check on the Kelvin-Voigt arm; the Maxwell arm of a
    # Burgers element is algebraic in t and carries no ODE.
    checkpoints = np.array([tau2, 3.0 * tau2, 10.0 * tau2])
    J_ode = np.array(
        [
            1.0 / E1
            + float(tc) / mu1
            + integrate_ode(lambda e: (1.0 - E2 * e) / mu2, 0.0, float(tc))
            for tc in checkpoints
        ]
    )
    J_lib = np.array(
        [
            bt.analytical.burgers_compliance(E1, mu1, E2, mu2, float(tc))
            for tc in checkpoints
        ]
    )
    ode_error = relative_error(J_lib, J_ode)

    # Long-time behaviour: the Kelvin-Voigt arm saturates and the Maxwell
    # dashpot keeps flowing, so dJ/dt must approach 1/mu1 (no finite limit).
    slope_late = (
        bt.analytical.burgers_compliance(E1, mu1, E2, mu2, 41.0 * tau2)
        - bt.analytical.burgers_compliance(E1, mu1, E2, mu2, 40.0 * tau2)
    ) / tau2
    slope_error = abs(slope_late - 1.0 / mu1) * mu1

    print("\nSpot values:")
    print(f"  eps(0) = {eps[0]:.6f} (expected {sigma0 / E1:.6f})")
    print(f"  J(0) = {J[0]:.6f} Pa^-1 (expected {1 / E1:.6f})")
    print("\nLong-time behaviour (Burgers creep is unbounded, so there is no eps_inf):")
    print(
        f"  dJ/dt near 40 tau2 = {slope_late:.9e} Pa^-1/s "
        f"(expected 1/mu1 = {1 / mu1:.9e}, relative gap {slope_error:.2e})"
    )

    print("\nIndependent comparisons:")
    print(
        f"  compliance vs closed form written from the model: {closed_error:.2e} (rel)"
    )
    print(f"  compliance vs RK4 of the Kelvin-Voigt arm:        {ode_error:.2e} (rel)")
    print(
        f"  burgers_creep vs sigma0*burgers_compliance:       {consistency_error:.2e} (rel)"
    )

    passed = (
        closed_error < CLOSED_FORM_TOL
        and ode_error < ODE_TOL
        and consistency_error < CLOSED_FORM_TOL
        and slope_error < 1e-9
    )
    print(f"\n{'OK PASSED' if passed else 'X FAILED'}: Burgers creep")
    return passed, t / tau2, eps / (sigma0 / E1), "Burgers"


def verify_complex_modulus():
    """Verify complex modulus utilities."""
    print("\n" + "=" * 60)
    print("Complex Modulus Utilities Verification")
    print("=" * 60)

    # Two cases: a 3-4-5 triangle with an exactly representable magnitude, and
    # an off-lattice case whose answers are not round numbers.
    cases = [(3.0, 4.0), (1.7, 0.45)]

    worst = 0.0
    for G1, G2 in cases:
        G_star = bt.analytical.complex_modulus_magnitude(G1, G2)
        tan_delta = bt.analytical.loss_tangent(G1, G2)
        delta = bt.analytical.phase_angle(G1, G2)

        # Independent closed forms from the definition G* = G1 + i*G2.
        G_star_expected = np.hypot(G1, G2)
        tan_expected = G2 / G1
        delta_expected = np.arctan(G2 / G1)  # first quadrant, so arctan suffices

        # Independent structural identities: the three outputs must describe the
        # same complex number, so projecting |G*| back onto the axes must return
        # the inputs. This catches a mismatch that agreeing formulas would hide.
        G1_back = G_star * np.cos(delta)
        G2_back = G_star * np.sin(delta)

        errors = {
            "|G*|": abs(G_star - G_star_expected) / G_star_expected,
            "tan(delta)": abs(tan_delta - tan_expected) / tan_expected,
            "delta": abs(delta - delta_expected) / delta_expected,
            "|G*|cos(delta) -> G1": abs(G1_back - G1) / G1,
            "|G*|sin(delta) -> G2": abs(G2_back - G2) / G2,
        }
        worst = max(worst, max(errors.values()))

        print(f"\nTest case: G1 = {G1}, G2 = {G2}")
        print(f"  |G*| = {G_star:.9f} (expected {G_star_expected:.9f})")
        print(f"  tan(delta) = {tan_delta:.9f} (expected {tan_expected:.9f})")
        print(
            f"  delta = {np.degrees(delta):.6f} deg "
            f"(expected {np.degrees(delta_expected):.6f} deg)"
        )
        print(f"  round trip |G*|cos(delta) = {G1_back:.9f} (input G1 = {G1})")
        print(f"  round trip |G*|sin(delta) = {G2_back:.9f} (input G2 = {G2})")
        for name, value in errors.items():
            print(f"    {name:>22}: {value:.2e} (rel)")

    print(f"\nWorst relative deviation across both cases: {worst:.2e}")
    passed = worst < CLOSED_FORM_TOL

    print(f"\n{'OK PASSED' if passed else 'X FAILED'}: Complex modulus utilities")
    return passed


def plot_all_models():
    """Create comparison plots for all viscoelastic models."""
    print("\n" + "=" * 60)
    print("Generating Comparison Plots")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Relaxation models (Maxwell, SLS). Each model is plotted against its own
    # relaxation time, which is not the same number for the two of them.
    ax1 = axes[0, 0]
    E, eta = 1000.0, 100.0
    eps0 = 0.01
    tau_maxwell = eta / E

    t_maxwell = np.linspace(0, 5 * tau_maxwell, 200)
    sigma_maxwell = np.array(
        [bt.analytical.maxwell_relaxation(E, eta, eps0, ti) for ti in t_maxwell]
    )
    ax1.plot(
        t_maxwell / tau_maxwell,
        sigma_maxwell / (E * eps0),
        "b-",
        linewidth=2,
        label=f"Maxwell (tau = {tau_maxwell:.2f} s)",
    )

    # SLS
    E1, E2 = 500.0, 500.0
    tau_sls = eta / E2
    t_sls = np.linspace(0, 5 * tau_sls, 200)
    sigma_sls = np.array(
        [bt.analytical.sls_relaxation(E1, E2, eta, eps0, ti) for ti in t_sls]
    )
    ax1.plot(
        t_sls / tau_sls,
        sigma_sls / (eps0 * (E1 + E2)),
        "r-",
        linewidth=2,
        label=f"SLS (tau = {tau_sls:.2f} s)",
    )

    ax1.set_xlabel("Time in units of each model's own tau")
    ax1.set_ylabel("Normalized Stress sigma/sigma0")
    ax1.set_title("Stress Relaxation Models")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Creep models (Kelvin-Voigt, Burgers), both against time in seconds.
    ax2 = axes[0, 1]
    sigma0 = 10.0
    t_creep = np.linspace(0, 1.0, 200)

    eps_kv = np.array(
        [bt.analytical.kelvin_voigt_creep(E, eta, sigma0, ti) for ti in t_creep]
    )
    ax2.plot(t_creep, eps_kv * 100, "g-", linewidth=2, label="Kelvin-Voigt")

    # Burgers
    E1b, mu1, E2b, mu2 = 1000.0, 1000.0, 500.0, 50.0
    eps_burgers = np.array(
        [bt.analytical.burgers_creep(E1b, mu1, E2b, mu2, sigma0, ti) for ti in t_creep]
    )
    ax2.plot(t_creep, eps_burgers * 100, "m-", linewidth=2, label="Burgers")

    ax2.axhline(
        y=100 * sigma0 / E,
        color="g",
        linestyle=":",
        alpha=0.7,
        label="Kelvin-Voigt limit sigma0/E",
    )

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
    if not all(results):
        raise SystemExit(1)
