"""Check a numerical answer against theory, the way you would check homework.

Run this end to end and it does four things, in the order you should do them
yourself:

1. Solves transient diffusion in a slab and compares it against the Fourier
   series solution.
2. Refines the grid and measures the observed order of accuracy, which is a much
   stronger statement about correctness than any single error number.
3. Solves the steady reaction-diffusion problem two ways -- marching the
   transient, and Newton on the steady operator -- and checks they agree with the
   closed-form cosh profile.
4. Verifies that a sealed domain conserves its contents.

If all four pass, the solver is doing what it says. If one fails, the printout
tells you which.

    python examples/verification/check_your_homework.py
"""

from __future__ import annotations

import numpy as np

import biotransport as bt
from biotransport.utils import get_result_path

EXAMPLE = "check_your_homework"


def heading(text: str) -> None:
    print()
    print(text)
    print("=" * len(text))


# ---------------------------------------------------------------------------
# 1. Transient diffusion in a slab, against the Fourier series
# ---------------------------------------------------------------------------


def transient_slab() -> bool:
    heading("1. Transient diffusion in a slab")

    # A 1 cm slab of water-like medium, initially empty, with both faces raised
    # to c = 1 at t = 0. Textbook separation of variables applies.
    diffusivity = 1.0e-9  # m^2/s
    thickness = 1.0e-2  # m
    surface = 1.0  # whatever concentration unit you like

    mesh = bt.mesh_1d(400, 0.0, thickness)
    problem = (
        bt.Problem(mesh)
        .diffusivity(diffusivity)
        .initial(0.0)
        .dirichlet("left", surface)
        .dirichlet("right", surface)
    )

    print(problem.describe())

    # The diffusion time scale. Comparing at a fraction of it keeps the problem
    # in its interesting transient rather than at equilibrium.
    tau = thickness**2 / diffusivity
    end_time = 0.02 * tau

    sol = bt.solve(problem, end_time=end_time, save_every=end_time / 5.0)

    def exact(x, t):
        return bt.analytical.slab(
            x, t, D=diffusivity, L=thickness, c_surface=surface, c_initial=0.0
        )

    report = sol.compare(exact)
    print()
    print(report)

    axes = sol.plot(
        times=list(sol.times),
        title="Slab diffusion: numerical (lines) vs series (dots)",
        xlabel="position (m)",
        ylabel="concentration",
    )
    for when in sol.times:
        axes.plot(sol.x[::25], exact(sol.x[::25], when), "k.", markersize=4)
    axes.figure.savefig(
        get_result_path("slab_vs_series.png", EXAMPLE), dpi=150, bbox_inches="tight"
    )

    passed = report.rel_l2 is not None and report.rel_l2 < 1.0e-3
    print(f"  -> {'PASS' if passed else 'FAIL'}: relative L2 below 0.1%")
    return passed


# ---------------------------------------------------------------------------
# 2. Grid refinement: is it converging at the right rate?
# ---------------------------------------------------------------------------


def spatial_order() -> bool:
    heading("2. Observed order of accuracy in space")

    diffusivity = 1.0e-9
    thickness = 1.0e-2
    end_time = 0.02 * thickness**2 / diffusivity

    # Hold the time step fixed across refinements. Otherwise each run picks its
    # own stable step and you measure space and time error mixed together.
    time_step = 0.05

    print(f"  fixed dt = {time_step:g} s so this isolates spatial error")
    print()
    print(f"  {'cells':>6}  {'dx':>10}  {'L2 error':>12}  {'ratio':>7}  {'order':>6}")

    previous: float | None = None
    orders: list[float] = []
    for cells in (50, 100, 200, 400):
        mesh = bt.mesh_1d(cells, 0.0, thickness)
        problem = (
            bt.Problem(mesh)
            .diffusivity(diffusivity)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 1.0)
        )
        sol = bt.solve(problem, end_time=end_time, time_step=time_step)
        error = sol.compare(
            lambda x, t: bt.analytical.slab(
                x, t, D=diffusivity, L=thickness, c_surface=1.0
            )
        ).l2

        spacing = thickness / cells
        if previous is None:
            print(f"  {cells:6d}  {spacing:10.3e}  {error:12.4e}  {'':>7}  {'':>6}")
        else:
            ratio = previous / error
            order = np.log2(ratio)
            orders.append(order)
            print(
                f"  {cells:6d}  {spacing:10.3e}  {error:12.4e}  {ratio:7.2f}  {order:6.2f}"
            )
        previous = error

    # The scheme is second order in space, so halving dx should quarter the error
    # and the measured order should sit near 2.
    finest = orders[-1] if orders else 0.0
    passed = 1.7 < finest < 2.3
    print()
    print(f"  finest-pair order {finest:.2f}")
    print(f"  -> {'PASS' if passed else 'FAIL'}: second-order spatial convergence")
    return passed


# ---------------------------------------------------------------------------
# 3. Steady reaction-diffusion, three ways
# ---------------------------------------------------------------------------


def steady_reaction_diffusion() -> bool:
    heading("3. Steady diffusion with first-order consumption")

    diffusivity = 1.0e-9
    rate = 5.0e-4  # 1/s
    thickness = 1.0e-2
    surface = 1.0

    mesh = bt.mesh_1d(300, 0.0, thickness)
    problem = (
        bt.Problem(mesh)
        .diffusivity(diffusivity)
        .linear_decay(rate)
        .initial(surface)
        .dirichlet("left", surface)
        .sealed("right")
    )

    def exact(x):
        return bt.analytical.steady_slab_first_order(
            x, D=diffusivity, k=rate, L=thickness, c_surface=surface
        )

    # (a) Newton on the steady operator.
    steady = bt.solve_steady(problem)
    steady_error = steady.compare(exact)
    print(f"  Newton:    {steady.newton.iterations} iterations")
    print(f"             relative L2 vs exact = {steady_error.rel_l2:.3e}")

    # (b) March the transient far past the reaction time scale and land in the
    #     same place. This is the slow route, shown for comparison.
    transient = bt.solve(problem, end_time=20.0 / rate)
    transient_error = transient.compare(exact)
    print(f"  transient: {transient.steps} steps")
    print(f"             relative L2 vs exact = {transient_error.rel_l2:.3e}")

    difference = float(np.max(np.abs(steady.c - transient.c)))
    print(f"  the two numerical routes differ by {difference:.3e}")

    # (c) The dimensionless reading of the same problem.
    modulus = bt.analytical.thiele_modulus(D=diffusivity, k=rate, length=thickness)
    effectiveness = bt.analytical.effectiveness_factor(modulus, "slab")
    print()
    print(f"  Thiele modulus       {modulus:.3f}")
    print(f"  effectiveness factor {effectiveness:.4f}")
    print(
        f"  reading: only {100.0 * effectiveness:.0f}% of the maximum possible "
        f"reaction rate is achieved,"
    )
    print("           because the interior is starved of solute.")

    axes = steady.plot(
        label="steady solver",
        title="Steady reaction-diffusion",
        xlabel="depth into the slab (m)",
        ylabel="concentration",
    )
    axes.plot(steady.x, exact(steady.x), "k--", lw=1, label="exact cosh profile")
    axes.legend()
    axes.figure.savefig(
        get_result_path("steady_reaction.png", EXAMPLE), dpi=150, bbox_inches="tight"
    )

    passed = (
        steady_error.rel_l2 < 1.0e-4
        and transient_error.rel_l2 < 1.0e-3
        and difference < 1.0e-5
    )
    print(f"  -> {'PASS' if passed else 'FAIL'}: both routes match the exact profile")
    return passed


# ---------------------------------------------------------------------------
# 4. Conservation on a closed domain
# ---------------------------------------------------------------------------


def conservation() -> bool:
    heading("4. Conservation in a sealed domain")

    mesh = bt.mesh_1d(200, 0.0, 1.0e-2)
    problem = (
        bt.Problem(mesh)
        .diffusivity(1.0e-9)
        .initial(bt.gaussian(mesh, center=0.005, width=0.0005))
        .sealed("left")
        .sealed("right")
    )

    sol = bt.solve(problem, end_time=600.0, save_every=120.0)

    totals = [sol.total(t) for t in sol.times]
    drift = max(totals) - min(totals)
    print(f"  total at each saved time: {', '.join(f'{v:.10g}' for v in totals)}")
    print(f"  drift across the whole run: {drift:.3e}")
    print(f"  as a fraction of the total: {drift / totals[0]:.3e}")

    # Nothing enters or leaves, so the only change allowed is roundoff.
    passed = drift / totals[0] < 1.0e-12
    print(f"  -> {'PASS' if passed else 'FAIL'}: conserved to roundoff")
    return passed


# ---------------------------------------------------------------------------


def main() -> int:
    print(__doc__.strip().split("\n\n")[0])

    checks = {
        "transient slab vs Fourier series": transient_slab(),
        "second-order spatial convergence": spatial_order(),
        "steady reaction-diffusion vs cosh": steady_reaction_diffusion(),
        "conservation in a sealed domain": conservation(),
    }

    heading("Summary")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    failures = sum(1 for passed in checks.values() if not passed)
    print()
    if failures:
        print(f"{failures} of {len(checks)} checks failed.")
    else:
        print(f"All {len(checks)} checks passed.")
        print(f"Figures written to {get_result_path('', EXAMPLE)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
