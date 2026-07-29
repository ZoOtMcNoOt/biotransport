#!/usr/bin/env python3
"""Scoped temporal-order comparison for three explicit integrators.

The problem is one-dimensional uniform diffusion of a single discrete sine
mode with homogeneous Dirichlet data. Errors are measured against the exact
evolution of that *semi-discrete* mode, so fixed spatial truncation error does
not hide the temporal orders:

* Forward Euler: order 1
* Heun / explicit trapezoid: order 2
* classical RK4: order 4

This is numerical-verification evidence for one smooth linear problem. It is
not a blanket performance ranking or permission to exceed a method's stability
limit. ``bt.integrate(..., method="euler")`` uses the canonical C++ solver;
Heun and RK4 are deliberately limited Python teaching adapters.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import biotransport as bt


METHODS = ("euler", "heun", "rk4")
EXPECTED_ORDERS = {"euler": 1.0, "heun": 2.0, "rk4": 4.0}
COLORS = {"euler": "C0", "heun": "C1", "rk4": "C2"}


def relative_rms(numerical: np.ndarray, reference: np.ndarray) -> float:
    """Return RMS error normalized by the reference RMS."""
    difference_rms = np.sqrt(np.mean((numerical - reference) ** 2))
    reference_rms = np.sqrt(np.mean(reference**2))
    return float(difference_rms / reference_rms)


def main() -> int:
    length = 1.0
    cells = 10
    diffusivity = 0.1
    end_time = 0.48
    time_steps = np.array([0.03, 0.015, 0.0075, 0.00375])

    mesh = bt.mesh_1d(cells, 0.0, length)
    x = np.asarray(bt.x_nodes(mesh))
    initial = np.sin(np.pi * x / length)
    problem = (
        bt.Problem(mesh)
        .diffusivity(diffusivity)
        .initial_condition(initial)
        .dirichlet(bt.Boundary.Left, 0.0)
        .dirichlet(bt.Boundary.Right, 0.0)
    )

    # Exact eigenvalue of the centered second-difference operator for sin(pi*x/L).
    spacing = mesh.dx()
    discrete_eigenvalue = (
        -4.0 * diffusivity / spacing**2 * np.sin(np.pi * spacing / (2.0 * length)) ** 2
    )
    reference = initial * np.exp(discrete_eigenvalue * end_time)

    print("=" * 72)
    print("Explicit temporal convergence on one semi-discrete diffusion mode")
    print("=" * 72)
    print(f"Cells / spacing:            {cells} / {spacing:.6f} m")
    print(f"Diffusivity:                {diffusivity:.6f} m^2/s")
    print(f"End time:                   {end_time:.6f} s")
    print(f"Discrete modal eigenvalue:  {discrete_eigenvalue:.9f} 1/s")
    print("Reference: exact exponential of the centered spatial operator")

    solutions: dict[str, np.ndarray] = {}
    errors: dict[str, np.ndarray] = {}
    steps: dict[str, int] = {}

    print("\nRelative RMS temporal error")
    print(f"{'dt [s]':>12} {'Euler':>14} {'Heun':>14} {'RK4':>14}")
    print("-" * 58)
    rows: list[dict[str, float]] = []
    for dt in time_steps:
        row: dict[str, float] = {"dt": float(dt)}
        for method in METHODS:
            result = bt.integrate(
                problem,
                t_end=end_time,
                method=method,
                dt=float(dt),
            )
            if not np.isclose(result.stats["dt"], dt, rtol=0.0, atol=1.0e-14):
                raise RuntimeError(
                    f"{method} did not use the requested exactly dividing timestep"
                )
            row[method] = relative_rms(result.solution, reference)
            if dt == time_steps[0]:
                solutions[method] = result.solution
                steps[method] = int(result.stats["steps"])
        rows.append(row)
        print(f"{dt:12.6f} {row['euler']:14.6e} {row['heun']:14.6e} {row['rk4']:14.6e}")

    for method in METHODS:
        errors[method] = np.asarray([row[method] for row in rows])

    observed_orders = {
        method: np.log(errors[method][:-1] / errors[method][1:]) / np.log(2.0)
        for method in METHODS
    }
    checks: dict[str, bool] = {}
    print("\nObserved orders between successive refinements")
    for method in METHODS:
        order_text = ", ".join(f"{value:.4f}" for value in observed_orders[method])
        expected = EXPECTED_ORDERS[method]
        checks[f"{method} final order within 0.1 of {expected:g}"] = bool(
            abs(observed_orders[method][-1] - expected) < 0.1
        )
        checks[f"{method} errors decrease monotonically"] = bool(
            np.all(np.diff(errors[method]) < 0.0)
        )
        print(f"  {method:5s}: {order_text} (expected {expected:g})")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(x, reference, "k-", linewidth=2, label="semi-discrete exact")
    for method in METHODS:
        axes[0].plot(
            x,
            solutions[method],
            "--",
            color=COLORS[method],
            label=f"{method.upper()} ({errors[method][0]:.2e})",
        )
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("u")
    axes[0].set_title(f"Solutions at t={end_time:g} s, dt={time_steps[0]:g} s")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for method in METHODS:
        axes[1].semilogy(
            x,
            np.abs(solutions[method] - reference) + np.finfo(float).tiny,
            color=COLORS[method],
            label=method.upper(),
        )
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("absolute temporal error")
    axes[1].set_title("Error against semi-discrete reference")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for method in METHODS:
        axes[2].loglog(
            time_steps,
            errors[method],
            "o-",
            color=COLORS[method],
            label=f"{method.upper()} measured",
        )
        order = EXPECTED_ORDERS[method]
        axes[2].loglog(
            time_steps,
            errors[method][0] * (time_steps / time_steps[0]) ** order,
            ":",
            color=COLORS[method],
            alpha=0.65,
            label=f"O(dt^{order:g})",
        )
    axes[2].set_xlabel("dt [s]")
    axes[2].set_ylabel("relative RMS temporal error")
    axes[2].set_title("Temporal refinement")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, which="both", alpha=0.3)

    figure.tight_layout()
    output_dir = Path(bt.get_results_dir("time_integration_comparison"))
    output = output_dir / "time_integration_methods.png"
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print("\nDeclared checks")
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    coarse_improvements = {
        method: errors["euler"][0] / errors[method][0] for method in METHODS[1:]
    }
    print("\nScoped interpretation")
    print(
        f"  At dt={time_steps[0]:g} s on this mode, Heun and RK4 reduce "
        f"Euler's temporal error by factors of {coarse_improvements['heun']:.1f} "
        f"and {coarse_improvements['rk4']:.3g}, respectively."
    )
    print(
        "  Those ratios do not include the different RHS work per step and do "
        "not generalize to nonlinear, stiff, or nonsmooth problems."
    )
    print(
        "  Every method remains explicit and must obey its own stability "
        "contract; a stable step can still be inaccurate."
    )
    print(f"  Coarse-run step counts: {steps}")
    print(f"  Figure: {output}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
