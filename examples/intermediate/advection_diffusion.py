#!/usr/bin/env python3
"""
Advection-Diffusion Example: Drug Transport in Blood Vessel

This example demonstrates the advection-diffusion solver for modeling
solute transport in flowing blood. The governing equation is:

    ∂C/∂t + v·∇C = D∇²C

where:
- C is the drug concentration
- v is the blood velocity (assumed uniform for simplicity)
- D is the drug diffusivity in blood

The example shows:
1. A bolus injection (Gaussian pulse) being carried downstream
2. Where the spreading you see actually comes from: a grid-refinement study
   against the exact advected Gaussian
3. The effect of Peclet number on transport regime

About the spreading. ``bt.solve`` advects with first-order upwinding, and
upwinding is not free: its truncation error acts exactly like an extra
diffusivity,

    D_num = v Δx (1 - C) / 2,     C = v Δt / Δx  (Courant number)

On a teaching grid with a small-molecule diffusivity this numerical term is
*hundreds to thousands of times larger* than the physical D, so a plot of "drug
spreading in blood" is really a plot of the discretization. The cell Peclet
number Pe_cell = v Δx / D is the warning sign: it equals 2 D_num / D at zero
Courant number, so anything much above 2 means the grid, not the physics, is
setting the pulse width. The refinement study below makes that measurable
rather than rhetorical.

BMEN 341 Reference: Weeks 5-6 (Convection-Diffusion, Peclet Number)
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import biotransport as bt


def advected_gaussian(x, t, *, x0, sigma, velocity, diffusivity):
    """Exact solution for a Gaussian pulse in a uniform flow (unbounded domain).

    The pulse translates at ``velocity`` and its variance grows as
    ``sigma^2 + 2 D t``. Pass ``diffusivity = D + D_num`` to get the profile the
    discrete scheme is actually solving for.
    """

    variance = sigma**2 + 2.0 * diffusivity * t
    return (sigma / np.sqrt(variance)) * np.exp(
        -((x - x0 - velocity * t) ** 2) / (2.0 * variance)
    )


def solve_bolus(nx, *, L, D, v, x0, sigma, t_end):
    """Advect a Gaussian bolus on an ``nx``-cell grid with the canonical solver."""

    mesh = bt.mesh_1d(nx, x_max=L)
    problem = (
        bt.Problem(mesh)
        .diffusivity(D)
        .velocity(v)
        .initial(bt.gaussian(mesh, center=x0, width=sigma))
        .dirichlet("left", 0.0)
        .sealed("right")
    )
    start = time.perf_counter()
    solution = bt.solve(problem, end_time=t_end)
    elapsed = time.perf_counter() - start
    return solution, elapsed


def run_advection_diffusion_1d():
    """Run 1D advection-diffusion and separate physical from numerical spreading."""
    print("=" * 60)
    print("1D Drug Bolus Transport in Blood Vessel")
    print("=" * 60)

    # Domain: 10 cm vessel segment
    L = 0.1  # 10 cm in meters

    # Physical parameters
    D = 1e-9  # Drug diffusivity in blood (typical small molecule) [m²/s]
    v_blood = 0.01  # Blood velocity (slow capillary flow) [m/s]

    # Initial condition: Gaussian bolus at x = 2 cm
    x0 = 0.02
    sigma = 0.005  # 5 mm standard deviation
    t_end = 5.0  # 5 seconds

    Pe = bt.dimensionless.peclet(v_blood, L, D)
    print("\nPhysical Parameters:")
    print(f"  Vessel length: {L * 100:.1f} cm")
    print(f"  Blood velocity: {v_blood * 100:.1f} cm/s")
    print(f"  Drug diffusivity: {D:.2e} m²/s")
    print(f"  Domain Peclet: {Pe:.2e}")

    # Physical spreading over the whole run, for scale. It adds in quadrature to
    # sigma, so 100 um against a 5000 um pulse is invisible.
    physical_spread = np.sqrt(2.0 * D * t_end)
    print(f"  Physical diffusive spread sqrt(2 D t): {physical_spread * 1e6:.1f} um")
    print(f"  Initial pulse width sigma: {sigma * 1e6:.0f} um")
    print("  So physical diffusion cannot visibly widen this pulse in 5 s.")

    # Same problem on three grids. Only dx changes.
    resolutions = [200, 800, 3200]
    runs = {}

    print(
        f"\n{'nx':>6} {'dx (m)':>10} {'Pe_cell':>9} {'D_num (m2/s)':>13} "
        f"{'D_num/D':>9} {'peak':>7} {'max err':>9} {'steps':>7} {'wall (s)':>9}"
    )

    for nx in resolutions:
        solution, wall_time = solve_bolus(
            nx, L=L, D=D, v=v_blood, x0=x0, sigma=sigma, t_end=t_end
        )
        x = solution.x
        dx = float(x[1] - x[0])
        dt = float(solution.diagnostics.maximum_time_step)
        courant = v_blood * dt / dx

        pe_cell = v_blood * dx / D
        d_num = v_blood * dx * (1.0 - courant) / 2.0

        physical = advected_gaussian(
            x, t_end, x0=x0, sigma=sigma, velocity=v_blood, diffusivity=D
        )
        error = float(np.max(np.abs(solution.c - physical)))

        runs[nx] = {
            "x": x,
            "c": solution.c,
            "dx": dx,
            "dt": dt,
            "courant": courant,
            "pe_cell": pe_cell,
            "d_num": d_num,
            "error": error,
        }

        print(
            f"{nx:6d} {dx:10.3e} {pe_cell:9.1f} {d_num:13.3e} {d_num / D:9.3e} "
            f"{float(solution.c.max()):7.4f} {error:9.4f} "
            f"{solution.steps:7d} {wall_time:9.3f}"
        )

    # Two different grids give two genuinely different answers. Compare them on
    # the coarse grid's nodes so the numbers are directly subtractable.
    coarse, fine = runs[resolutions[0]], runs[resolutions[-1]]
    fine_on_coarse = np.interp(coarse["x"], fine["x"], fine["c"])
    grid_gap = float(np.max(np.abs(coarse["c"] - fine_on_coarse)))
    print(
        f"\nmax|nx={resolutions[0]} - nx={resolutions[-1]}| = {grid_gap:.4f} "
        f"(peak amplitude is 1.0), so the two grids do not agree."
    )
    print(
        f"Error against the exact physical solution falls "
        f"{coarse['error']:.4f} -> {fine['error']:.4f} as dx shrinks "
        f"{coarse['dx'] / fine['dx']:.0f}x: the difference between the curves was "
        f"discretization error, not physics."
    )

    # The decisive check: does the modified equation account for the spreading?
    print("\nIs the observed width physical or numerical?")
    for nx in resolutions:
        run = runs[nx]
        modified = advected_gaussian(
            run["x"],
            t_end,
            x0=x0,
            sigma=sigma,
            velocity=v_blood,
            diffusivity=D + run["d_num"],
        )
        residual = float(np.max(np.abs(run["c"] - modified)))
        print(
            f"  nx={nx:5d}: max|computed - Gaussian spread by (D + D_num)| = "
            f"{residual:.4f}   vs {run['error']:.4f} against physical D alone"
        )
    print("  The computed pulse matches the numerically-widened Gaussian, not the")
    print("  physical one. The spreading in this plot is the upwind scheme.")

    # Expected final position
    expected_x = x0 + v_blood * t_end
    print(f"\nExpected bolus center after {t_end}s: {expected_x * 100:.1f} cm")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_ref = runs[resolutions[-1]]["x"]
    ic_ref = advected_gaussian(
        x_ref, 0.0, x0=x0, sigma=sigma, velocity=v_blood, diffusivity=D
    )
    exact_ref = advected_gaussian(
        x_ref, t_end, x0=x0, sigma=sigma, velocity=v_blood, diffusivity=D
    )

    for ax in axes:
        ax.plot(x_ref * 100, ic_ref, "k--", linewidth=1.5, label="Initial (t=0)")
        ax.plot(
            x_ref * 100,
            exact_ref,
            color="black",
            linewidth=2,
            label=f"Exact, physical D (t={t_end}s)",
        )
        for nx in resolutions:
            run = runs[nx]
            ax.plot(
                run["x"] * 100,
                run["c"],
                linewidth=1.8,
                label=f"Upwind nx={nx} (Pe_cell={run['pe_cell']:.0f})",
            )
        ax.set_xlabel("Position (cm)")
        ax.grid(True, alpha=0.3)

    axes[0].axvline(
        expected_x * 100, color="gray", linestyle=":", label="Expected center"
    )
    axes[0].set_ylabel("Concentration (normalized)")
    axes[0].set_title("Drug Bolus Transport")
    axes[0].legend(fontsize=8)

    axes[1].set_xlim([expected_x * 100 - 3, expected_x * 100 + 3])
    axes[1].set_ylabel("Concentration")
    axes[1].set_title("Refining the grid removes the numerical spreading")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(
        bt.get_result_path("bolus_transport_1d.png", "advection_diffusion"), dpi=150
    )
    print(
        f"\nSaved: {bt.get_result_path('bolus_transport_1d.png', 'advection_diffusion')}"
    )
    plt.show()


def run_peclet_comparison():
    """Compare transport at different Peclet numbers."""
    print("\n" + "=" * 60)
    print("Effect of Peclet Number on Transport")
    print("=" * 60)

    L = 0.1
    nx = 100
    mesh = bt.mesh_1d(nx, x_max=L)
    dx = L / nx

    # Different Peclet regimes:
    # Pe < 1: diffusion dominates (spreading)
    # Pe > 1: advection dominates (translation)
    # Pe >> 1: strongly convective (minimal spreading)
    cases = [
        {"D": 1e-4, "v": 0.001, "label": "Balanced domain transport"},
        {"D": 1e-6, "v": 0.01, "label": "Advection-dominated"},
        {"D": 1e-8, "v": 0.05, "label": "Strongly advection-dominated"},
    ]

    x0 = 0.02
    sigma = 0.005
    t_end = 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    x = bt.x_nodes(mesh)
    ax.plot(
        x * 100,
        bt.gaussian(mesh, center=x0, width=sigma),
        "k--",
        linewidth=2,
        label="Initial",
    )

    print(
        "Cell Peclet also tells you how much of the width on this fixed 100-cell\n"
        "grid is numerical. D_num/D >> 1 means you are plotting the scheme.\n"
    )
    for case in cases:
        Pe = bt.dimensionless.peclet(case["v"], L, case["D"])

        problem = (
            bt.Problem(mesh)
            .diffusivity(case["D"])
            .velocity(case["v"])
            .initial(bt.gaussian(mesh, center=x0, width=sigma))
            .dirichlet("left", 0.0)
            .sealed("right")
        )
        solution = bt.solve(problem, end_time=t_end)
        ax.plot(x * 100, solution.c, linewidth=2, label=f"{case['label']}")

        dt = float(solution.diagnostics.maximum_time_step)
        courant = case["v"] * dt / dx
        cell_peclet = abs(case["v"]) * dx / case["D"]
        d_num = abs(case["v"]) * dx * (1.0 - courant) / 2.0
        print(
            f"{case['label']}: domain Pe = {Pe:.0f}, "
            f"cell Pe = {cell_peclet:.2g}, D_num/D = {d_num / case['D']:.2g}, "
            f"steps = {solution.steps}"
        )

    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Concentration")
    ax.set_title(
        f"Advection-Diffusion at Different Peclet Numbers (t={t_end}s, {nx} cells)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        bt.get_result_path("peclet_comparison.png", "advection_diffusion"), dpi=150
    )
    print(
        f"\nSaved: {bt.get_result_path('peclet_comparison.png', 'advection_diffusion')}"
    )
    plt.show()


def run_2d_transport():
    """2D advection-diffusion with a uniform velocity field."""
    print("\n" + "=" * 60)
    print("2D Drug Transport (Channel Flow)")
    print("=" * 60)

    # 2D channel: 8 cm × 2 cm
    nx, ny = 80, 40
    Lx, Ly = 0.08, 0.02
    mesh = bt.mesh_2d(nx, ny, x_max=Lx, y_max=Ly)

    D = 1e-7  # m²/s
    v_mean = 0.01  # m/s mean flow velocity

    # Initial: circular bolus near inlet
    # Using circle() helper would give sharp boundaries; we want smooth Gaussian
    X, Y = bt.xy_grid(mesh)
    x0, y0 = 0.01, 0.01  # Center of bolus
    r0 = 0.005  # Characteristic radius
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    ic = np.exp(-r2 / (r0**2)).reshape(-1)

    times = [1.0, 2.0, 4.0]

    problem = (
        bt.Problem(mesh)
        .diffusivity(D)
        .velocity(v_mean, 0.0)
        .initial(ic)
        .dirichlet("left", 0.0)
        .sealed("right")
        .sealed("bottom")
        .sealed("top")
    )

    solution = bt.solve(problem, end_time=times[-1], save_at=times)
    print(f"Steps: {solution.steps}")

    dx = Lx / nx
    dt = float(solution.diagnostics.maximum_time_step)
    courant = v_mean * dt / dx
    d_num = v_mean * dx * (1.0 - courant) / 2.0
    print(f"Cell Peclet in x: {v_mean * dx / D:.1f}")
    print(f"D_num/D = {d_num / D:.1f}: streamwise smearing here is mostly numerical.")

    # Plot snapshots
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes = axes.flatten()

    x = bt.x_nodes(mesh)
    y = bt.y_nodes(mesh)
    X, Y = np.meshgrid(x * 100, y * 100)

    for ax, t in zip(axes, solution.times):
        Z = solution.at(t)
        c = ax.pcolormesh(X, Y, Z, cmap="viridis", shading="auto")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title(f"t = {t:.1f} s")
        ax.set_aspect("equal")
        plt.colorbar(c, ax=ax, label="C")

    plt.suptitle("2D Drug Transport in Channel Flow", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        bt.get_result_path("channel_transport_2d.png", "advection_diffusion"), dpi=150
    )
    print(
        f"\nSaved: {bt.get_result_path('channel_transport_2d.png', 'advection_diffusion')}"
    )
    plt.show()


if __name__ == "__main__":
    run_advection_diffusion_1d()
    run_peclet_comparison()
    run_2d_transport()

    print("\n" + "=" * 60)
    print("All advection-diffusion examples completed!")
    print("=" * 60)
