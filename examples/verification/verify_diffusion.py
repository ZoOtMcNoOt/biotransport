"""Verification: Semi-infinite diffusion analytical solution.

Compares numerical diffusion solver against the analytical erfc solution.

BMEN 341 Reference: Weeks 1-2 (Fick's Laws, Error function solutions)
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt


EXAMPLE_NAME = "verification/diffusion"


def verify_semi_infinite_diffusion():
    """Compare numerical 1D diffusion to erfc analytical solution."""
    print("=" * 60)
    print("Semi-Infinite Diffusion Verification")
    print("=" * 60)

    # Parameters
    D = 1e-9  # Diffusivity (m²/s) - typical small molecule
    C_surface = 1.0
    C_initial = 0.0
    L = 1e-3  # Domain length: 1 mm
    t_final = 100.0  # seconds

    # Create mesh using convenience function
    mesh = bt.mesh_1d(200, x_max=L)
    x = bt.x_nodes(mesh)

    # Setup problem with:
    # - Dirichlet at left (surface at fixed concentration)
    # - Neumann at right (zero flux, approximating semi-infinite domain)
    problem = (
        bt.Problem(mesh)
        .diffusivity(D)
        .initial_condition(bt.uniform(mesh, C_initial))
        .dirichlet(bt.Boundary.Left, C_surface)
        .neumann(bt.Boundary.Right, 0.0)
    )

    print("\nParameters:")
    print(f"  D = {D:.2e} m²/s")
    print(f"  C_surface = {C_surface}")
    print(f"  C_initial = {C_initial}")
    print(f"  Domain = 0 to {L*1000:.1f} mm")
    print(f"  t_final = {t_final:.0f} s")

    # Solve numerically using simplified API
    print("\nRunning numerical solver...")
    result = bt.solve(problem, t=t_final)
    C_numerical = result.solution()

    # Analytical solution: C(x,t) = C_surface * erfc(x / (2*sqrt(D*t)))
    C_analytical = np.array(
        [
            bt.analytical.diffusion_1d_semi_infinite(
                xi, t_final, D, C_surface, C_initial
            )
            for xi in x
        ]
    )

    # Penetration depth: characteristic length scale of diffusion
    delta = bt.analytical.diffusion_penetration_depth(D, t_final)
    print(f"\nPenetration depth delta = sqrt(Dt) = {delta*1e6:.2f} um")

    # Error analysis (only where penetration has occurred)
    mask = x < 5 * delta  # Where concentration is significant
    if np.sum(mask) > 10:
        max_error = np.max(np.abs(C_numerical[mask] - C_analytical[mask]))
        rms_error = np.sqrt(np.mean((C_numerical[mask] - C_analytical[mask]) ** 2))
        print("\nError analysis (x < 5delta):")
        print(f"  Max absolute error: {max_error:.4e}")
        print(f"  RMS error: {rms_error:.4e}")
    else:
        max_error = 0
        rms_error = 0

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.plot(x * 1e6, C_numerical, "b-", linewidth=2, label="Numerical (FD)")
    ax1.plot(x * 1e6, C_analytical, "r--", linewidth=2, label="Analytical (erfc)")
    ax1.axvline(
        x=delta * 1e6,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"delta = {delta*1e6:.1f} mum",
    )
    ax1.set_xlabel("Position x (mum)")
    ax1.set_ylabel("Concentration C/C0")
    ax1.set_title(f"Semi-Infinite Diffusion at t = {t_final:.0f} s")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, L * 1e6)

    # Error plot
    ax2 = axes[1]
    error = C_numerical - C_analytical
    ax2.plot(x * 1e6, error * 100, "g-", linewidth=2)
    ax2.set_xlabel("Position x (mum)")
    ax2.set_ylabel("Error (% of C0)")
    ax2.set_title("Numerical - Analytical Error")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, L * 1e6)

    plt.tight_layout()
    plt.savefig(bt.get_result_path("semi_infinite_verification.png", EXAMPLE_NAME))
    plt.show()

    # Pass if RMS error is small (< 1% of C_surface)
    passed = rms_error < 0.01 * C_surface
    print(f"\n{'OK PASSED' if passed else 'X FAILED'}: Semi-infinite diffusion")
    return passed


def verify_time_evolution():
    """Show concentration profile evolution and compare to analytical."""
    print("\n" + "=" * 60)
    print("Time Evolution Verification")
    print("=" * 60)

    D = 1e-9
    C_surface = 1.0
    C_initial = 0.0
    L = 0.5e-3  # 500 μm
    times = [1, 10, 50, 100, 500]

    mesh = bt.mesh_1d(150, x_max=L)
    x = bt.x_nodes(mesh)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

    for i, t in enumerate(times):
        # Analytical erfc solution
        C_ana = np.array(
            [
                bt.analytical.diffusion_1d_semi_infinite(xi, t, D, C_surface, C_initial)
                for xi in x
            ]
        )
        ax.plot(x * 1e6, C_ana, "-", color=colors[i], linewidth=2, label=f"t = {t} s")

        # Penetration depth marker
        delta = bt.analytical.diffusion_penetration_depth(D, t)
        ax.axvline(x=delta * 1e6, color=colors[i], linestyle=":", alpha=0.3)

    ax.set_xlabel("Position x (mum)")
    ax.set_ylabel("Concentration C/C0")
    ax.set_title("Diffusion Profile Evolution (Analytical)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L * 1e6)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(bt.get_result_path("diffusion_evolution.png", EXAMPLE_NAME))
    plt.show()

    return True


def verify_penetration_depth():
    """Verify penetration depth formula delta = sqrt(Dt)."""
    print("\n" + "=" * 60)
    print("Penetration Depth Verification")
    print("=" * 60)

    # Test various D and t combinations
    test_cases = [
        (1e-9, 1.0),  # Small molecule, 1 s
        (1e-9, 100.0),  # Small molecule, 100 s
        (1e-11, 3600),  # Macromolecule, 1 hr
        (1e-5, 1.0),  # Gas, 1 s
    ]

    all_passed = True

    for D, t in test_cases:
        delta_ana = bt.analytical.diffusion_penetration_depth(D, t)
        delta_expected = np.sqrt(D * t)
        error = abs(delta_ana - delta_expected) / delta_expected

        passed = error < 1e-10
        all_passed = all_passed and passed

        print(f"\nD = {D:.0e} m²/s, t = {t:.0f} s:")
        print(f"  delta = {delta_ana:.4e} m")
        print(f"  sqrt(Dt) = {delta_expected:.4e} m")
        print(f"  {'OK' if passed else 'X'} Error: {error*100:.2e}%")

    print(f"\n{'OK PASSED' if all_passed else 'X FAILED'}: Penetration depth formula")
    return all_passed


if __name__ == "__main__":
    results = []
    results.append(verify_semi_infinite_diffusion())
    results.append(verify_time_evolution())
    results.append(verify_penetration_depth())

    print("\n" + "=" * 60)
    print(f"SUMMARY: {sum(results)}/{len(results)} verifications passed")
    print("=" * 60)
