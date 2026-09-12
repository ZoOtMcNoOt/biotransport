"""Verification: 1D diffusion against closed-form solutions.

Compares the numerical diffusion solver against the half-space erf solution
and against the finite-slab series.

The half-space solution only describes a finite mesh while the diffusing front
has not reached the far wall. The condition is 4*sqrt(D*t) < L, and this
example prints that ratio next to every comparison so the reader can see
whether the reference is entitled to be used at all. Run at a time where the
condition holds, the remaining gap is the solver's discretization error and
falls at second order under mesh refinement; run past it, the gap is dominated
by the far boundary and says nothing about the solver. Both regimes are shown.

BMEN 341 Reference: Weeks 1-2 (Fick's Laws, Error function solutions)
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt


EXAMPLE_NAME = "verification/diffusion"

# Shared physical setup: a small molecule entering a 1 mm slab that is sealed
# at its far face.
D = 1e-9  # Diffusivity (m²/s)
C_SURFACE = 1.0
C_INITIAL = 0.0
L = 1e-3  # Domain length: 1 mm


def solve_numerically(n_cells, t_final):
    """Run the solver on an n_cells mesh and return (x, C)."""
    mesh = bt.mesh_1d(n_cells, 0.0, L)
    x = bt.x_nodes(mesh)
    problem = (
        bt.Problem(mesh)
        .diffusivity(D)
        .initial(C_INITIAL)
        .dirichlet("left", C_SURFACE)
        .sealed("right")
    )
    return x, np.asarray(bt.solve(problem, end_time=t_final).c)


def erf_reference(x, t):
    """Half-space solution: C = C_s + (C_0 - C_s)*erf(x / (2 sqrt(D t)))."""
    return bt.analytical.semi_infinite(
        x, t, D=D, c_surface=C_SURFACE, c_initial=C_INITIAL
    )


def slab_reference(x, t):
    """Finite-domain series, exact for a slab exposed at x=0 and sealed at x=L.

    A slab sealed on one face is the symmetric half of a slab of twice the
    thickness held at C_surface on both faces, so the series is evaluated with
    thickness 2L and read on 0 <= x <= L.
    """
    return bt.analytical.slab(
        x, t, D=D, L=2.0 * L, c_surface=C_SURFACE, c_initial=C_INITIAL
    )


def report_validity(t_final):
    """Print the penetration depth against the domain length, and return it."""
    delta = bt.analytical.diffusion_length(D, t_final)
    penetration = 4.0 * delta
    print(f"  sqrt(D t)         = {delta * 1e6:8.2f} um")
    print(f"  4 sqrt(D t)       = {penetration * 1e6:8.2f} um")
    print(f"  domain length L   = {L * 1e6:8.2f} um")
    print(f"  4 sqrt(D t) / L   = {penetration / L:8.3f}", end="  ")
    if penetration < L:
        print("(< 1: the half-space solution applies)")
    else:
        print("(>= 1: the far wall matters, the half-space solution does not apply)")
    return delta, penetration


def verify_semi_infinite_diffusion():
    """Compare the solver to the erf solution at a time where it is valid."""
    print("=" * 60)
    print("Semi-Infinite Diffusion Verification")
    print("=" * 60)

    n_cells = 200
    t_final = 10.0  # s; chosen so 4 sqrt(D t) stays well inside the domain

    print("\nParameters:")
    print(f"  D = {D:.2e} m²/s")
    print(f"  C_surface = {C_SURFACE}")
    print(f"  C_initial = {C_INITIAL}")
    print(f"  Domain = 0 to {L * 1000:.1f} mm, {n_cells} cells")
    print(f"  t_final = {t_final:.0f} s")

    print("\nValidity of the half-space reference:")
    delta, _ = report_validity(t_final)

    print("\nRunning numerical solver...")
    x, C_numerical = solve_numerically(n_cells, t_final)
    C_analytical = erf_reference(x, t_final)

    # The far wall has not been reached, so the sealed boundary the solver sees
    # and the infinite half-space the reference assumes agree to well below the
    # discretization error. The finite-slab series confirms it.
    C_slab = slab_reference(x, t_final)
    reference_gap = float(np.max(np.abs(C_analytical - C_slab)))
    print(f"\n  C_analytical at the far wall: {C_analytical[-1]:.3e}")
    print(f"  max |erf reference - finite-slab reference|: {reference_gap:.3e}")

    # Error analysis over the whole domain; there is no longer any region that
    # has to be masked out to make the comparison hold.
    error = C_numerical - C_analytical
    max_error = float(np.max(np.abs(error)))
    rms_error = float(np.sqrt(np.mean(error**2)))
    print("\nError analysis (whole domain, 0 <= x <= L):")
    print(f"  Max absolute error: {max_error:.4e}")
    print(f"  RMS error: {rms_error:.4e}")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.plot(x * 1e6, C_numerical, "b-", linewidth=2, label="Numerical (FD)")
    ax1.plot(x * 1e6, C_analytical, "r--", linewidth=2, label="Analytical (erf)")
    ax1.axvline(
        x=4 * delta * 1e6,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"4 sqrt(Dt) = {4 * delta * 1e6:.0f} mum",
    )
    ax1.set_xlabel("Position x (mum)")
    ax1.set_ylabel("Concentration C/C0")
    ax1.set_title(f"Semi-Infinite Diffusion at t = {t_final:.0f} s")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, L * 1e6)

    # Error plot
    ax2 = axes[1]
    ax2.plot(x * 1e6, error * 100, "g-", linewidth=2)
    ax2.axvline(x=4 * delta * 1e6, color="gray", linestyle=":", alpha=0.7)
    ax2.set_xlabel("Position x (mum)")
    ax2.set_ylabel("Error (% of C0)")
    ax2.set_title(f"Numerical - Analytical Error (RMS = {rms_error:.2e})")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, L * 1e6)

    plt.tight_layout()
    plt.savefig(bt.get_result_path("semi_infinite_verification.png", EXAMPLE_NAME))
    plt.show()

    # The reference must be valid, the two independent references must agree,
    # and the residual must sit at the discretization level for this mesh.
    passed = (
        4.0 * delta < L
        and reference_gap < 1e-10
        and rms_error < 2.0e-4
        and max_error < 5.0e-4
    )
    print(f"\n{'OK PASSED' if passed else 'X FAILED'}: Semi-infinite diffusion")
    return passed


def verify_domain_truncation():
    """Show that running past 4 sqrt(D t) = L measures the domain, not the solver.

    Same mesh, same solver, later time. The erf comparison degrades by orders of
    magnitude while the finite-slab comparison does not, which locates the
    discrepancy in the reference rather than in the numerics.
    """
    print("\n" + "=" * 60)
    print("Domain-Truncation Check")
    print("=" * 60)

    n_cells = 200
    rows = []
    for t_final in (10.0, 100.0):
        print(f"\nt = {t_final:.0f} s")
        _, penetration = report_validity(t_final)

        x, C_numerical = solve_numerically(n_cells, t_final)
        erf_rms = float(
            np.sqrt(np.mean((C_numerical - erf_reference(x, t_final)) ** 2))
        )
        slab_rms = float(
            np.sqrt(np.mean((C_numerical - slab_reference(x, t_final)) ** 2))
        )
        print(f"  RMS vs erf (half-space):   {erf_rms:.4e}")
        print(f"  RMS vs slab (finite):      {slab_rms:.4e}")
        print(f"  ratio erf/slab:            {erf_rms / slab_rms:8.1f}")
        rows.append((t_final, penetration / L, erf_rms, slab_rms, x, C_numerical))

    (_, _, erf_valid, slab_valid, _, _) = rows[0]
    (t_late, _, erf_late, slab_late, x_late, C_late) = rows[1]

    print("\nReading:")
    print(
        f"  At 4 sqrt(Dt)/L = {rows[0][1]:.2f} the two references agree with each "
        f"other and with the solver ({erf_valid:.2e} vs {slab_valid:.2e})."
    )
    print(
        f"  At 4 sqrt(Dt)/L = {rows[1][1]:.2f} the erf residual grows to "
        f"{erf_late:.2e} while the finite-slab residual stays at {slab_late:.2e}."
    )
    print("  The growth is domain truncation in the reference, not solver error.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.plot(x_late * 1e6, C_late, "b-", linewidth=2, label="Numerical (FD)")
    ax1.plot(
        x_late * 1e6,
        erf_reference(x_late, t_late),
        "r--",
        linewidth=2,
        label="erf (half-space, invalid here)",
    )
    ax1.plot(
        x_late * 1e6,
        slab_reference(x_late, t_late),
        "k:",
        linewidth=2,
        label="finite-slab series (valid)",
    )
    ax1.set_xlabel("Position x (mum)")
    ax1.set_ylabel("Concentration C/C0")
    ax1.set_title(
        f"t = {t_late:.0f} s: 4 sqrt(Dt) = {rows[1][1]:.2f} L\n"
        "half-space reference has run off the end of the mesh"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, L * 1e6)

    ax2 = axes[1]
    labels = [f"t = {row[0]:.0f} s\n4sqrt(Dt)/L = {row[1]:.2f}" for row in rows]
    width = 0.35
    positions = np.arange(len(rows))
    ax2.bar(
        positions - width / 2,
        [row[2] for row in rows],
        width,
        label="vs erf (half-space)",
        color="firebrick",
        alpha=0.8,
    )
    ax2.bar(
        positions + width / 2,
        [row[3] for row in rows],
        width,
        label="vs slab (finite domain)",
        color="steelblue",
        alpha=0.8,
    )
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels)
    ax2.set_yscale("log")
    ax2.set_ylabel("RMS error")
    ax2.set_title("Which reference is entitled to be used")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(bt.get_result_path("domain_truncation.png", EXAMPLE_NAME))
    plt.show()

    # Inside the valid window the two references are interchangeable; outside
    # it only the finite-domain one still tracks the solver.
    passed = erf_valid < 10.0 * slab_valid and erf_late > 100.0 * slab_late
    print(f"\n{'OK PASSED' if passed else 'X FAILED'}: Domain-truncation diagnosis")
    return passed


def verify_grid_refinement():
    """Refine the mesh and check the error falls at the expected second-order rate."""
    print("\n" + "=" * 60)
    print("Grid Refinement Verification")
    print("=" * 60)

    t_final = 10.0
    n_values = [50, 100, 200, 400, 800]

    print(f"\nt_final = {t_final:.0f} s, reference = half-space erf solution")
    print("Validity at this time:")
    report_validity(t_final)
    print(
        "\nCentral differences are second order in dx and the automatically "
        "selected\nexplicit timestep scales as dx^2, so the co-refined error is "
        "expected at\norder 2."
    )

    def solve_at(n_cells):
        x, C_numerical = solve_numerically(n_cells, t_final)
        rms = float(np.sqrt(np.mean((C_numerical - erf_reference(x, t_final)) ** 2)))
        probe = int(round(0.1 * n_cells))  # QoI: C at x = 0.1 L, a node for every n
        return float(C_numerical[probe]), rms

    result = bt.run_convergence_study(
        solve_func=solve_at,
        n_values=n_values,
        theoretical_order=2.0,
        verbose=False,
        size_to_h=lambda n: L / n,
    )

    errors = np.asarray(result.errors)
    h_values = np.asarray([L / n for n in n_values])

    print(f"\n{'N':>6} {'dx (um)':>10} {'C(0.1L)':>14} {'RMS error':>12} {'ratio':>8}")
    solutions = np.asarray(result.solutions)
    for i, n_cells in enumerate(n_values):
        ratio = "" if i == 0 else f"{errors[i - 1] / errors[i]:8.3f}"
        print(
            f"{n_cells:6d} {h_values[i] * 1e6:10.3f} {solutions[i]:14.10f} "
            f"{errors[i]:12.4e} {ratio:>8}"
        )

    order, _, r_squared = bt.compute_order_of_accuracy(h_values, errors)
    print(
        f"\n  Least-squares order over all 5 levels: {order:.3f} (R^2 = {r_squared:.6f})"
    )
    print(f"  Three-finest-QoI observed order:       {result.observed_order:.3f}")
    print(f"  Richardson extrapolated C(0.1L):       {result.richardson_estimate:.10f}")
    print("  Expected order for this scheme:        2.0")
    print("  Error ratios above should approach 2^2 = 4 as the mesh refines.")

    passed = abs(order - 2.0) < 0.15 and abs(result.observed_order - 2.0) < 0.3

    # Convergence plot
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.loglog(
        h_values * 1e6, errors, "bo-", markersize=9, linewidth=2, label="RMS error"
    )
    h_ref = h_values[2]
    e_ref = errors[2]
    h_line = np.logspace(
        np.log10(h_values.min() / 1.5), np.log10(h_values.max() * 1.5), 50
    )
    ax.loglog(
        h_line * 1e6,
        e_ref * (h_line / h_ref) ** 2,
        "k--",
        alpha=0.6,
        label="O(dx^2) reference",
    )
    ax.set_xlabel("Mesh spacing dx (mum)")
    ax.set_ylabel("RMS error vs erf solution")
    ax.set_title(
        f"Grid Refinement at t = {t_final:.0f} s\nObserved order {order:.2f} (expected 2.0)"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(bt.get_result_path("grid_refinement.png", EXAMPLE_NAME))
    plt.show()

    print(f"\n{'OK PASSED' if passed else 'X FAILED'}: Second-order grid refinement")
    return passed


def verify_time_evolution():
    """Show concentration profile evolution, flagging where the erf form applies."""
    print("\n" + "=" * 60)
    print("Time Evolution (Analytical Profiles)")
    print("=" * 60)

    window = 0.5e-3  # 500 μm viewing window
    times = [1, 10, 50, 100, 500]

    mesh = bt.mesh_1d(150, 0.0, window)
    x = bt.x_nodes(mesh)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

    print(f"\nViewing window: 0 to {window * 1e6:.0f} um")
    print(f"\n{'t (s)':>8} {'4 sqrt(Dt) (um)':>18} {'within window?':>16}")
    for i, t in enumerate(times):
        C_ana = erf_reference(x, float(t))
        penetration = 4.0 * bt.analytical.diffusion_length(D, float(t))
        inside = penetration < window
        print(f"{t:8d} {penetration * 1e6:18.1f} {'yes' if inside else 'no':>16}")
        suffix = "" if inside else "  (front past the window)"
        ax.plot(
            x * 1e6,
            C_ana,
            "-" if inside else "--",
            color=colors[i],
            linewidth=2,
            label=f"t = {t} s{suffix}",
        )
        ax.axvline(x=penetration * 1e6, color=colors[i], linestyle=":", alpha=0.3)

    print(
        "\nThese are half-space profiles drawn on a 500 um window. Dashed curves\n"
        "are past 4 sqrt(D t) = 500 um, so they would not describe a 500 um slab\n"
        "with a sealed far face; only the solid ones would."
    )

    ax.set_xlabel("Position x (mum)")
    ax.set_ylabel("Concentration C/C0")
    ax.set_title(
        "Half-Space Diffusion Profiles (dotted lines mark 4 sqrt(Dt))\n"
        "dashed profiles have outgrown the plotted window"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, window * 1e6)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(bt.get_result_path("diffusion_evolution.png", EXAMPLE_NAME))
    plt.show()

    return True


def verify_diffusion_length():
    """Check the library's characteristic length against its own definition sqrt(Dt).

    This is a definitional identity, not an independent verification: both sides
    evaluate the same one-line formula. It guards against a wiring mistake in the
    binding, nothing more.
    """
    print("\n" + "=" * 60)
    print("Characteristic Diffusion Length (definitional check)")
    print("=" * 60)

    # Test various D and t combinations
    test_cases = [
        (1e-9, 1.0),  # Small molecule, 1 s
        (1e-9, 100.0),  # Small molecule, 100 s
        (1e-11, 3600),  # Macromolecule, 1 hr
        (1e-5, 1.0),  # Gas, 1 s
    ]

    all_passed = True

    for diffusivity, t in test_cases:
        delta_ana = bt.analytical.diffusion_length(diffusivity, t)
        delta_expected = np.sqrt(diffusivity * t)
        error = abs(delta_ana - delta_expected) / delta_expected

        passed = error < 1e-10
        all_passed = all_passed and passed

        print(f"\nD = {diffusivity:.0e} m²/s, t = {t:.0f} s:")
        print(f"  delta = {delta_ana:.4e} m")
        print(f"  sqrt(Dt) = {delta_expected:.4e} m")
        print(f"  {'OK' if passed else 'X'} Error: {error * 100:.2e}%")

    print(f"\n{'OK PASSED' if all_passed else 'X FAILED'}: Diffusion length formula")
    return all_passed


if __name__ == "__main__":
    results = []
    results.append(verify_semi_infinite_diffusion())
    results.append(verify_domain_truncation())
    results.append(verify_grid_refinement())
    verify_time_evolution()  # Plot-only demonstration; not counted as a check.
    results.append(verify_diffusion_length())

    print("\n" + "=" * 60)
    print(f"SUMMARY: {sum(results)}/{len(results)} verifications passed")
    print("=" * 60)
    if not all(results):
        raise SystemExit(1)
