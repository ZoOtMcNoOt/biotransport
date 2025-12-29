#!/usr/bin/env python3
"""
Higher-Order Finite Difference Schemes Example
===============================================

Demonstrates the improved accuracy of 4th and 6th-order finite difference
schemes compared to standard 2nd-order central differences.

Key Concepts:
- Standard 2nd-order: O(dx²) truncation error
- 4th-order: O(dx⁴) truncation error (16x improvement when dx halves)
- 6th-order: O(dx⁶) truncation error (64x improvement when dx halves)

When to use higher-order schemes:
- Smooth solutions without discontinuities
- High accuracy requirements (research, verification)
- Moderate grid sizes where higher-order reduces computational cost
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import biotransport as bt

# Results directory
RESULTS_DIR = Path(bt.get_results_dir()) / "high_order"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def demo_laplacian_accuracy():
    """
    Compare accuracy of 2nd, 4th, and 6th-order Laplacian stencils.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: Laplacian Accuracy Comparison")
    print("=" * 60)

    # Test function: sin(2πx), exact Laplacian = -4π²sin(2πx)
    k = 2 * np.pi

    def u_exact(x):
        return np.sin(k * x)

    def laplacian_exact(x):
        return -(k**2) * np.sin(k * x)

    # Test on different grid sizes
    grid_sizes = [10, 20, 40, 80, 160]
    errors_2nd = []
    errors_4th = []
    errors_6th = []
    dxs = []

    for n in grid_sizes:
        x = np.linspace(0, 1, n + 1)
        dx = 1.0 / n
        dxs.append(dx)

        u = u_exact(x)
        exact = laplacian_exact(x)

        # Compute numerical Laplacians
        lap2 = bt.laplacian_2nd_order(u, dx)
        lap4 = bt.laplacian_4th_order(u, dx)
        lap6 = bt.laplacian_6th_order(u, dx)

        # Compute max errors in interior (skip boundary cells)
        errors_2nd.append(np.max(np.abs(lap2[2:-2] - exact[2:-2])))
        errors_4th.append(np.max(np.abs(lap4[3:-3] - exact[3:-3])))
        errors_6th.append(np.max(np.abs(lap6[3:-3] - exact[3:-3])))

    # Compute convergence rates
    print("\nGrid Size |    2nd-order    |    4th-order    |    6th-order")
    print("-" * 65)
    for i, n in enumerate(grid_sizes):
        print(
            f"   {n:4d}   | {errors_2nd[i]:13.2e}   | {errors_4th[i]:13.2e}   | {errors_6th[i]:13.2e}"
        )

    print("\nConvergence Rates (error reduction when dx halves):")
    print("-" * 65)
    for i in range(1, len(grid_sizes)):
        rate_2nd = errors_2nd[i - 1] / errors_2nd[i]
        rate_4th = errors_4th[i - 1] / errors_4th[i]
        rate_6th = errors_6th[i - 1] / errors_6th[i]
        print(
            f"  {grid_sizes[i - 1]:3d}→{grid_sizes[i]:3d}  |     {rate_2nd:5.1f}x       |     {rate_4th:5.1f}x       |     {rate_6th:5.1f}x"
        )

    print("\nExpected rates: 2nd→4x, 4th→16x, 6th→64x")

    # Plot convergence
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(
        dxs, errors_2nd, "o-", linewidth=2, markersize=8, label="2nd-order (O(dx²))"
    )
    ax.loglog(
        dxs, errors_4th, "s-", linewidth=2, markersize=8, label="4th-order (O(dx⁴))"
    )
    ax.loglog(
        dxs, errors_6th, "^-", linewidth=2, markersize=8, label="6th-order (O(dx⁶))"
    )

    # Reference slopes
    dx_ref = np.array([dxs[0], dxs[-1]])
    ax.loglog(
        dx_ref,
        0.5 * errors_2nd[0] * (dx_ref / dx_ref[0]) ** 2,
        "k--",
        alpha=0.5,
        label="slope=2",
    )
    ax.loglog(
        dx_ref,
        0.5 * errors_4th[0] * (dx_ref / dx_ref[0]) ** 4,
        "k-.",
        alpha=0.5,
        label="slope=4",
    )
    ax.loglog(
        dx_ref,
        0.5 * errors_6th[0] * (dx_ref / dx_ref[0]) ** 6,
        "k:",
        alpha=0.5,
        label="slope=6",
    )

    ax.set_xlabel("Grid spacing (dx)", fontsize=12)
    ax.set_ylabel("Maximum error", fontsize=12)
    ax.set_title("Convergence of Finite Difference Laplacian", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "laplacian_convergence.png", dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'laplacian_convergence.png'}")


def demo_diffusion_solver():
    """
    Solve diffusion equation with different spatial orders.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Diffusion Solver Comparison")
    print("=" * 60)

    # Problem setup
    L = 1.0
    D = 0.01
    t_end = 1.0

    # Exact solution for diffusion of sine wave:
    # u(x,t) = exp(-D*k²*t) * sin(k*x)
    k = 2 * np.pi

    def exact_solution(x, t):
        return np.exp(-D * k**2 * t) * np.sin(k * x)

    # Test with different grid sizes
    grid_sizes = [20, 40, 80]

    for n in grid_sizes:
        mesh = bt.mesh_1d(n, 0, L)
        x = bt.x_nodes(mesh)

        # Initial condition: sin(2πx)
        initial = np.sin(k * x)
        exact = exact_solution(x, t_end)

        # Solve with 2nd and 4th order
        solver2 = bt.HighOrderDiffusionSolver(mesh, D=D, order=2)
        solver4 = bt.HighOrderDiffusionSolver(mesh, D=D, order=4)

        result2 = solver2.solve(initial, t_end=t_end)
        result4 = solver4.solve(initial, t_end=t_end)

        error2 = np.max(np.abs(result2.solution[1:-1] - exact[1:-1]))
        error4 = np.max(np.abs(result4.solution[1:-1] - exact[1:-1]))

        print(
            f"n={n:3d}: 2nd-order error = {error2:.2e}, 4th-order error = {error4:.2e}"
        )
        print(
            f"        2nd-order steps = {result2.steps}, 4th-order steps = {result4.steps}"
        )

    # Detailed comparison at n=80
    mesh = bt.mesh_1d(80, 0, L)
    x = bt.x_nodes(mesh)
    initial = np.sin(k * x)
    exact = exact_solution(x, t_end)

    solver2 = bt.HighOrderDiffusionSolver(mesh, D=D, order=2)
    solver4 = bt.HighOrderDiffusionSolver(mesh, D=D, order=4)

    result2 = solver2.solve(initial, t_end=t_end)
    result4 = solver4.solve(initial, t_end=t_end)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Solutions
    ax = axes[0]
    ax.plot(x, initial, "k--", alpha=0.5, linewidth=1, label="Initial")
    ax.plot(x, exact, "g-", linewidth=2, label="Exact")
    ax.plot(
        x,
        result2.solution,
        "b--",
        linewidth=1.5,
        label=f"2nd-order (steps={result2.steps})",
    )
    ax.plot(
        x,
        result4.solution,
        "r:",
        linewidth=2,
        label=f"4th-order (steps={result4.steps})",
    )
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("u(x, t)", fontsize=12)
    ax.set_title(f"Diffusion of sin(2πx) at t={t_end}", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: Error profiles
    ax = axes[1]
    error2_profile = result2.solution - exact
    error4_profile = result4.solution - exact
    ax.plot(
        x,
        error2_profile,
        "b-",
        linewidth=2,
        label=f"2nd-order (max={np.max(np.abs(error2_profile)):.2e})",
    )
    ax.plot(
        x,
        error4_profile,
        "r-",
        linewidth=2,
        label=f"4th-order (max={np.max(np.abs(error4_profile)):.2e})",
    )
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title("Error Profiles (n=80)", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "diffusion_comparison.png", dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'diffusion_comparison.png'}")


def demo_gradient_accuracy():
    """
    Compare accuracy of 2nd and 4th-order gradient stencils.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Gradient Accuracy Comparison")
    print("=" * 60)

    # Test function: sin(2πx), exact gradient = 2π*cos(2πx)
    k = 2 * np.pi

    grid_sizes = [10, 20, 40, 80, 160]
    errors_2nd = []
    errors_4th = []
    dxs = []

    for n in grid_sizes:
        x = np.linspace(0, 1, n + 1)
        dx = 1.0 / n
        dxs.append(dx)

        u = np.sin(k * x)
        exact = k * np.cos(k * x)

        # Compute gradients
        grad2 = bt.ddx(u, dx, order=2)
        grad4 = bt.ddx(u, dx, order=4)

        errors_2nd.append(np.max(np.abs(grad2[2:-2] - exact[2:-2])))
        errors_4th.append(np.max(np.abs(grad4[2:-2] - exact[2:-2])))

    print("\nGrid Size |    2nd-order    |    4th-order")
    print("-" * 50)
    for i, n in enumerate(grid_sizes):
        print(f"   {n:4d}   | {errors_2nd[i]:13.2e}   | {errors_4th[i]:13.2e}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(
        dxs, errors_2nd, "o-", linewidth=2, markersize=8, label="2nd-order (O(dx²))"
    )
    ax.loglog(
        dxs, errors_4th, "s-", linewidth=2, markersize=8, label="4th-order (O(dx⁴))"
    )

    ax.set_xlabel("Grid spacing (dx)", fontsize=12)
    ax.set_ylabel("Maximum error", fontsize=12)
    ax.set_title("Convergence of Finite Difference Gradient", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gradient_convergence.png", dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'gradient_convergence.png'}")


def demo_when_to_use_higher_order():
    """
    Demonstrate when higher-order schemes provide the most benefit.
    """
    print("\n" + "=" * 60)
    print("DEMO 4: When to Use Higher-Order Schemes")
    print("=" * 60)

    print("""
Higher-order schemes are BEST for:
  ✓ Smooth solutions (no discontinuities)
  ✓ High accuracy requirements (research verification)
  ✓ Wave propagation problems
  ✓ When computational cost matters (coarser grid → same accuracy)

Higher-order schemes may NOT help for:
  ✗ Solutions with discontinuities (shocks, interfaces)
  ✗ Rough initial conditions
  ✗ Problems dominated by time integration error
  ✗ Very coarse grids (stencil may be too wide)

Cost-Accuracy Trade-off:
  • To achieve error ε with 2nd-order: need n ∝ ε^(-1/2) grid points
  • To achieve error ε with 4th-order: need n ∝ ε^(-1/4) grid points
  • Example: For ε = 10⁻⁸, 2nd-order needs 10⁴ points, 4th-order needs 10² points

Stability Considerations:
  • Higher-order schemes have stricter CFL conditions
  • 4th-order: dt ≤ C·dx²/(2.5·D) vs 2nd-order: dt ≤ C·dx²/(2·D)
  • This means more time steps for the same dt, but fewer spatial points
""")


def main():
    """Run all higher-order scheme demonstrations."""
    print("=" * 60)
    print("HIGHER-ORDER FINITE DIFFERENCE SCHEMES")
    print("4th and 6th-order Spatial Accuracy for Research Applications")
    print("=" * 60)

    demo_laplacian_accuracy()
    demo_diffusion_solver()
    demo_gradient_accuracy()
    demo_when_to_use_higher_order()

    print("\n" + "=" * 60)
    print("AVAILABLE HIGH-ORDER FUNCTIONS")
    print("=" * 60)
    print("""
Stencil Functions:
  - bt.laplacian_2nd_order(u, dx)  : Standard 2nd-order ∇²
  - bt.laplacian_4th_order(u, dx)  : 4th-order ∇² (O(dx⁴))
  - bt.laplacian_6th_order(u, dx)  : 6th-order ∇² (O(dx⁶))
  - bt.gradient_4th_order(u, dx)   : 4th-order du/dx

Convenience Wrappers:
  - bt.d2dx2(u, dx, order=4)       : Select order for d²u/dx²
  - bt.ddx(u, dx, order=4)         : Select order for du/dx

Solver:
  - bt.HighOrderDiffusionSolver(mesh, D, order=4)
    .set_boundary(bt.Boundary.Left, value)
    .solve(initial, t_end)

Verification:
  - bt.verify_order_of_accuracy(solver_factory, exact_solution)
""")

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
