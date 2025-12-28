#!/usr/bin/env python3
"""
Time Integration Methods Comparison
====================================

This example compares three explicit time integration methods:
1. Forward Euler (1st order)
2. Heun's method (2nd order, improved Euler)
3. Classic RK4 (4th order)

For the same time step, higher-order methods provide better accuracy.
Alternatively, higher-order methods can use larger time steps for the same accuracy.

Physics: 1D diffusion of a sinusoidal profile
    du/dt = D d2u/dx2

with homogeneous Dirichlet BCs: u(0,t) = u(L,t) = 0

Initial condition: u(x,0) = sin(pi*x/L)

Analytical solution: u(x,t) = sin(pi*x/L) * exp(-pi^2*D*t/L^2)

Key takeaways:
- RK4 achieves ~10-100x better accuracy than Euler for same dt
- RK4 allows ~3-4x larger dt for same accuracy (saving compute time)
- Use RK4 for long-time integration or when accuracy is critical
"""

import numpy as np
import matplotlib.pyplot as plt

import biotransport as bt


def analytical_solution(x, t, D, L=1.0):
    """Analytical solution for decaying sinusoidal diffusion."""
    return np.sin(np.pi * x / L) * np.exp(-(np.pi**2) * D * t / L**2)


def main():
    print("=" * 70)
    print("Time Integration Methods: Euler vs Heun vs RK4")
    print("=" * 70)

    # Problem setup
    L = 1.0  # Domain length [m]
    n = 51  # Number of nodes
    D = 0.1  # Diffusivity [m2/s]
    t_end = 0.5  # Simulation time [s]

    mesh = bt.mesh_1d(n, 0.0, L)
    x = np.array(bt.x_nodes(mesh))

    # Initial condition: sin(pi*x)
    ic = [np.sin(np.pi * xi / L) for xi in x]

    # Build problem
    problem = (
        bt.Problem(mesh)
        .diffusivity(D)
        .initial_condition(ic)
        .dirichlet(bt.Boundary.Left, 0.0)
        .dirichlet(bt.Boundary.Right, 0.0)
    )

    # Analytical solution at t_end
    u_exact = analytical_solution(x, t_end, D, L)

    print("\nProblem: 1D diffusion of sin(pi*x)")
    print("  Domain: [0, {}] m".format(L))
    print("  Nodes: {}".format(n))
    print("  Diffusivity: {} m^2/s".format(D))
    print("  End time: {} s".format(t_end))

    # =========================================================================
    # Part 1: Same timestep comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 1: Same Time Step Comparison")
    print("=" * 70)

    # Use a small timestep that's stable for all methods
    dx = L / (n - 1)
    dt_stable = 0.5 * dx * dx / (2 * D)  # Conservative CFL
    print("\nUsing dt = {:.6f} s for all methods".format(dt_stable))

    results = {}
    methods = ["euler", "heun", "rk4"]

    for method in methods:
        result = bt.integrate(problem, t_end=t_end, method=method, dt=dt_stable)
        rmse = np.sqrt(np.mean((result.solution - u_exact) ** 2))
        results[method] = {
            "solution": result.solution,
            "steps": result.stats["steps"],
            "error": rmse,
            "time": result.stats.get("wall_time_s", 0),
        }

    print("\n" + "-" * 66)
    print("SAME DT COMPARISON")
    print("-" * 66)
    print(
        "{:15} {:>10} {:>15} {:>15}".format("Method", "Steps", "RMSE Error", "vs Euler")
    )
    print("-" * 66)

    euler_error = results["euler"]["error"]
    for method in methods:
        r = results[method]
        improvement = euler_error / r["error"] if r["error"] > 0 else float("inf")
        print(
            "{:15} {:>10} {:>15.2e} {:>14.1f}x".format(
                method.upper(), r["steps"], r["error"], improvement
            )
        )

    print("-" * 66)

    # =========================================================================
    # Part 2: Convergence order verification
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 2: Convergence Order Verification")
    print("=" * 70)

    # Use successively smaller dt values
    dts = [dt_stable, dt_stable / 2, dt_stable / 4, dt_stable / 8]

    print("\n" + "-" * 70)
    print(
        "{:>15} {:>12} {:>12} {:>12} {:>15}".format(
            "dt", "Euler", "Heun", "RK4", "Order"
        )
    )
    print("-" * 70)

    errors = {m: [] for m in methods}

    for dt_val in dts:
        row = "{:>15.6f}".format(dt_val)
        for method in methods:
            result = bt.integrate(problem, t_end=t_end, method=method, dt=dt_val)
            error = np.sqrt(np.mean((result.solution - u_exact) ** 2))
            errors[method].append(error)
            row += " {:>11.2e}".format(error)
        row += "     (1, 2, 4)"
        print(row)

    print("-" * 70)

    # Compute and print orders
    orders = {m: [] for m in methods}
    for method in methods:
        for i in range(len(dts) - 1):
            if errors[method][i] > 0 and errors[method][i + 1] > 0:
                order = np.log(errors[method][i] / errors[method][i + 1]) / np.log(2)
                orders[method].append(order)

    if all(len(orders[m]) > 0 for m in methods):
        avg_order = {m: np.mean(orders[m]) for m in methods}
        print(
            "{:>15} {:>12.2f} {:>12.2f} {:>12.2f}".format(
                "Measured Order",
                avg_order["euler"],
                avg_order["heun"],
                avg_order["rk4"],
            )
        )
        print("-" * 70)

    # =========================================================================
    # Visualization
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Solutions comparison
    ax1 = axes[0, 0]
    ax1.plot(x, u_exact, "k-", linewidth=2, label="Analytical")
    colors = {"euler": "C0", "heun": "C1", "rk4": "C2"}
    for method in methods:
        ax1.plot(
            x,
            results[method]["solution"],
            "--",
            color=colors[method],
            label="{} (RMSE={:.2e})".format(method.upper(), results[method]["error"]),
        )
    ax1.set_xlabel("Position x [m]")
    ax1.set_ylabel("Concentration u")
    ax1.set_title("Solution at t = {} s (dt = {:.5f} s)".format(t_end, dt_stable))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error vs position
    ax2 = axes[0, 1]
    for method in methods:
        error = np.abs(results[method]["solution"] - u_exact)
        ax2.semilogy(x, error + 1e-16, color=colors[method], label=method.upper())
    ax2.set_xlabel("Position x [m]")
    ax2.set_ylabel("Absolute Error")
    ax2.set_title("Spatial Error Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Convergence study
    ax3 = axes[1, 0]
    for method in methods:
        ax3.loglog(
            dts, errors[method], "o-", color=colors[method], label=method.upper()
        )

    # Reference lines
    dt_ref = np.array(dts)
    ax3.loglog(
        dt_ref, errors["euler"][0] * (dt_ref / dts[0]), "k:", alpha=0.5, label="O(dt)"
    )
    ax3.loglog(
        dt_ref,
        errors["heun"][0] * (dt_ref / dts[0]) ** 2,
        "k--",
        alpha=0.5,
        label="O(dt^2)",
    )
    ax3.loglog(
        dt_ref,
        errors["rk4"][0] * (dt_ref / dts[0]) ** 4,
        "k-.",
        alpha=0.5,
        label="O(dt^4)",
    )

    ax3.set_xlabel("Time step dt [s]")
    ax3.set_ylabel("RMSE Error")
    ax3.set_title("Convergence Order Verification")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Steps comparison bar chart
    ax4 = axes[1, 1]
    steps_per_method = [results[m]["steps"] for m in methods]
    bars = ax4.bar(
        [m.upper() for m in methods],
        steps_per_method,
        color=[colors[m] for m in methods],
    )
    ax4.set_ylabel("Steps Used")
    ax4.set_title("Number of Steps (dt = {:.5f} s)".format(dt_stable))

    # Add error annotations
    for i, (method, bar) in enumerate(zip(methods, bars)):
        ax4.annotate(
            "Error: {:.2e}".format(results[method]["error"]),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    # Save the figure
    from pathlib import Path

    output_dir = Path(bt.get_results_dir("time_integration_comparison"))
    fig.savefig(
        output_dir / "time_integration_methods.png", dpi=150, bbox_inches="tight"
    )
    print("\nFigure saved to: {}".format(output_dir / "time_integration_methods.png"))

    plt.show()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    rk4_improvement = (
        euler_error / results["rk4"]["error"] if results["rk4"]["error"] > 0 else 0
    )
    heun_improvement = (
        euler_error / results["heun"]["error"] if results["heun"]["error"] > 0 else 0
    )

    print(
        """
For the same time step (dt = {:.5f} s):
  * RK4 is ~{:.0f}x more accurate than Euler
  * Heun is ~{:.0f}x more accurate than Euler

Recommendations:
  - Use RK4 for long-time integration or accuracy-critical applications
  - Use Heun as a good balance between speed and accuracy
  - Use Euler only for quick prototyping or when stability is the priority
""".format(dt_stable, rk4_improvement, heun_improvement)
    )


if __name__ == "__main__":
    main()
