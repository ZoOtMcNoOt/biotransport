#!/usr/bin/env python3
"""Science-first tour of BioTransport's high-order numerical kernels.

This example deliberately separates spatial-stencil verification from a full
time-dependent PDE.  A fourth-order spatial stencil does not make a solver
fourth order in time, and the current Dirichlet boundary closure is second
order.  Those distinctions matter when interpreting convergence results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import biotransport as bt
from biotransport.high_order import (
    HighOrderDiffusionSolver,
    integrate_explicit_runge_kutta,
    laplacian_2nd_order,
    laplacian_4th_order,
    laplacian_6th_order,
    verify_order_of_accuracy,
)


RESULTS_DIR = Path(bt.get_results_dir()) / "high_order"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def demonstrate_spatial_convergence() -> None:
    """Verify formal interior order against an analytical derivative."""
    wave_number = 2.0 * np.pi

    def field(x):
        return np.sin(wave_number * x)

    def exact_laplacian(x):
        return -(wave_number**2) * np.sin(wave_number * x)

    grid_sizes = (20, 40, 80, 160)

    schemes = (
        ("2nd-order interior", laplacian_2nd_order, 1, 2),
        ("4th-order interior", laplacian_4th_order, 2, 4),
        ("6th-order interior", laplacian_6th_order, 3, 6),
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    print("\nSpatial Laplacian convergence against d2/dx2 sin(2*pi*x)")
    print("scheme                 final observed order       finest L_inf error")
    print("-" * 72)
    for label, operator, margin, formal_order in schemes:
        verification = verify_order_of_accuracy(
            lambda cells, selected=operator: (
                lambda values: selected(values, 1.0 / cells)
            ),
            field,
            exact_laplacian,
            grid_sizes=grid_sizes,
            interior_margin=margin,
        )
        print(
            f"{label:24s} {verification['observed_orders'][-1]:20.3f}"
            f" {verification['errors'][-1]:24.3e}"
        )
        axis.loglog(
            verification["dx"],
            verification["errors"],
            "o-",
            label=f"{label} (formal p={formal_order})",
        )

    axis.set_xlabel("grid spacing dx")
    axis.set_ylabel("interior L_inf error")
    axis.set_title("Centered Laplacian convergence")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    destination = RESULTS_DIR / "laplacian_convergence.png"
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    print(f"Saved {destination}")


def demonstrate_native_diffusion() -> None:
    """Run the C++ PDE loop and inspect its explicit accuracy metadata."""
    mesh = bt.StructuredMesh(80, 0.0, 1.0)
    x = np.linspace(0.0, 1.0, mesh.nx() + 1)
    diffusivity = 0.01
    end_time = 0.25
    initial = np.sin(np.pi * x)

    solver = HighOrderDiffusionSolver(mesh, D=diffusivity, order=4)
    result = solver.solve(initial, end_time)
    exact = np.exp(-diffusivity * np.pi**2 * end_time) * np.sin(np.pi * x)
    error = np.max(np.abs(result.solution[1:-1] - exact[1:-1]))

    print("\nNative explicit diffusion")
    print(f"  final time:                 {result.time:g}")
    print(f"  accepted steps:             {result.steps}")
    print(f"  nominal / final dt:         {result.dt:.3e} / {result.last_dt:.3e}")
    print(f"  centered interior order:    {result.interior_order}")
    print(f"  lowest boundary closure:    {result.boundary_order}")
    print(f"  temporal order:             {result.temporal_order}")
    print(f"  interior error at t_end:     {error:.3e}")
    print(
        "  Interpretation: this full solve is first order in dt; use the spatial\n"
        "  convergence experiment above to verify the stencil independently."
    )


def demonstrate_nonautonomous_rk4() -> None:
    """Show correct stage-time handling for y' = -2*t*y."""
    result = integrate_explicit_runge_kutta(
        [1.0],
        lambda state, time: -2.0 * time * state,
        t_end=1.0,
        dt=0.1,
        method="rk4",
    )
    exact = np.exp(-1.0)
    print("\nValidated nonautonomous RK4")
    print(f"  numerical y(1): {result.solution[0]:.10f}")
    print(f"  analytical y(1): {exact:.10f}")
    print(f"  absolute error: {abs(result.solution[0] - exact):.3e}")
    print(
        "  Note: Python RHS callbacks cross the GIL four times per RK4 step.\n"
        "  This adapter is convenient and safe, but fully native model solvers are\n"
        "  the high-throughput path."
    )


def print_method_scope() -> None:
    print(
        """
Method scope
------------
* Interior order 4 or 6 assumes a smooth, uniformly spaced field.
* Boundary derivative entries are zero; near-boundary closure order is 2.
* Discontinuities and unresolved interfaces generally defeat formal high order.
* The diffusion solver is Forward Euler and enforces its spectral stability bound.
* User-supplied dt values above the safety-scaled limit are rejected.
"""
    )


def main() -> None:
    print("BioTransport high-order numerics: claims verified in their proper scope")
    demonstrate_spatial_convergence()
    demonstrate_native_diffusion()
    demonstrate_nonautonomous_rk4()
    print_method_scope()


if __name__ == "__main__":
    main()
