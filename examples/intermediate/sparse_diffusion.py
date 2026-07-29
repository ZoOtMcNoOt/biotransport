#!/usr/bin/env python3
"""Sparse Backward Euler diffusion and linear-system checks.

The native ``ImplicitDiffusion2D`` solver uses a conservative finite-volume
spatial operator and Backward Euler in time. For linear diffusion, Backward
Euler is L-stable, so there is no explicit diffusion CFL restriction for linear
stability and stiff modes are strongly damped. It is only first-order accurate
in time, however, so a stable large step can still be inaccurate.

This example checks direct and iterative implicit solves, solves a manufactured
Poisson problem through the sparse-matrix API, and measures the expected
first-order temporal convergence against an exact semi-discrete eigenmode.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

import biotransport as bt

EXAMPLE_NAME = "sparse_diffusion"


def relative_l2(numerical: np.ndarray, reference: np.ndarray) -> float:
    """Return a relative discrete L2 error."""
    return float(np.linalg.norm(numerical - reference) / np.linalg.norm(reference))


def set_zero_dirichlet(solver) -> None:
    for boundary in (
        bt.Boundary.Left,
        bt.Boundary.Right,
        bt.Boundary.Bottom,
        bt.Boundary.Top,
    ):
        solver.set_dirichlet_boundary(boundary, 0.0)


def solve_implicit(
    mesh: bt.StructuredMesh,
    diffusivity: float,
    initial: np.ndarray,
    dt: float,
    steps: int,
    solver_type: bt.SparseSolverType,
) -> tuple[np.ndarray, bt.ImplicitSolveResult, float]:
    solver = bt.ImplicitDiffusion2D(mesh, diffusivity)
    solver.set_initial_condition(initial.ravel())
    set_zero_dirichlet(solver)
    solver.set_solver_type(solver_type)
    solver.set_tolerance(1.0e-11)
    solver.set_max_iterations(5000)
    start = time.perf_counter()
    result = solver.solve(dt=dt, num_steps=steps)
    elapsed = time.perf_counter() - start
    return np.asarray(solver.solution()).reshape(initial.shape), result, elapsed


def implicit_solver_comparison() -> tuple[
    bool,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    length = 1.0
    cells = 50
    diffusivity = 0.01
    final_time = 0.5
    dt = 0.05
    steps = round(final_time / dt)
    mesh = bt.StructuredMesh(cells, cells, 0.0, length, 0.0, length)
    x = np.linspace(0.0, length, cells + 1)
    y = np.linspace(0.0, length, cells + 1)
    xx, yy = np.meshgrid(x, y)
    sigma = 0.1
    initial = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (2.0 * sigma**2))
    # Match the configured homogeneous Dirichlet boundary exactly.
    initial[[0, -1], :] = 0.0
    initial[:, [0, -1]] = 0.0

    explicit_limit = 1.0 / (
        2.0 * diffusivity * (1.0 / mesh.dx() ** 2 + 1.0 / mesh.dy() ** 2)
    )
    solutions: dict[str, np.ndarray] = {}
    results: dict[str, bt.ImplicitSolveResult] = {}

    print("\nConservative Backward Euler diffusion")
    print(f"  dt / explicit diffusion limit       {dt / explicit_limit:.1f}")
    for solver_type, label in (
        (bt.SparseSolverType.SparseLU, "SparseLU"),
        (bt.SparseSolverType.ConjugateGradient, "ConjugateGradient"),
    ):
        solution, result, elapsed = solve_implicit(
            mesh, diffusivity, initial, dt, steps, solver_type
        )
        solutions[label] = solution
        results[label] = result
        print(f"  {label:20s} residual={result.residual:.3e}, wall={elapsed:.3f} s")

    direct = solutions["SparseLU"]
    iterative = solutions["ConjugateGradient"]
    solver_difference = relative_l2(iterative, direct)
    boundary_max = float(
        max(
            np.max(np.abs(direct[0, :])),
            np.max(np.abs(direct[-1, :])),
            np.max(np.abs(direct[:, 0])),
            np.max(np.abs(direct[:, -1])),
        )
    )

    # Compare a different A-stable, second-order algorithm at the same dt. The
    # difference is a discretization comparison, not a performance benchmark.
    adi = bt.ADIDiffusion2D(mesh, diffusivity)
    adi.set_initial_condition(initial.ravel())
    set_zero_dirichlet(adi)
    adi_result = adi.solve(dt=dt, num_steps=steps)
    adi_solution = np.asarray(adi.solution()).reshape(initial.shape)
    adi_difference = relative_l2(adi_solution, direct)

    checks = {
        "both implicit solves report success": all(
            result.success for result in results.values()
        ),
        "both algebraic residuals < 1e-8": all(
            result.residual < 1.0e-8 for result in results.values()
        ),
        "direct and CG fields agree": solver_difference < 1.0e-8,
        "homogeneous Dirichlet data are exact": boundary_max < 1.0e-13,
        "Backward Euler preserves this nonnegative field": float(np.min(direct))
        >= -1.0e-12,
        "ADI completed the requested steps": adi_result.steps == steps,
    }

    print(f"  direct/CG relative difference        {solver_difference:.3e}")
    print(f"  ADI/Backward Euler relative difference {adi_difference:.3e}")
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print("  Timings above are illustrative only; they are not a controlled benchmark.")
    return all(checks.values()), xx, yy, initial, direct, adi_solution


def sparse_poisson_check() -> bool:
    """Solve -Laplacian(u)=f for a manufactured sine solution."""
    cells = 20
    spacing = 1.0 / cells
    coordinates = np.linspace(0.0, 1.0, cells + 1)
    xx, yy = np.meshgrid(coordinates, coordinates)
    exact = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    rhs = 2.0 * np.pi**2 * exact
    rhs[[0, -1], :] = 0.0
    rhs[:, [0, -1]] = 0.0

    matrix = bt.build_2d_laplacian(cells, cells, spacing, spacing)
    errors = {}
    residuals = {}
    for solver_type in (
        bt.SparseSolverType.SparseLU,
        bt.SparseSolverType.BiCGSTAB,
    ):
        solution = np.asarray(matrix.solve(rhs.ravel().tolist(), solver_type)).reshape(
            exact.shape
        )
        residual = np.asarray(matrix.multiply(solution.ravel().tolist())) - rhs.ravel()
        errors[solver_type.name] = relative_l2(solution, exact)
        residuals[solver_type.name] = float(
            np.linalg.norm(residual) / np.linalg.norm(rhs)
        )

    checks = {
        "matrix dimensions match nodal field": matrix.rows == matrix.cols == exact.size,
        "matrix is sparse": matrix.nnz < 0.02 * matrix.rows * matrix.cols,
        "manufactured-solution relative error < 3e-3": max(errors.values()) < 3.0e-3,
        "relative linear residual < 1e-8": max(residuals.values()) < 1.0e-8,
    }

    print("\nSparse matrix manufactured Poisson problem")
    print(
        f"  shape / nonzeros                  {matrix.rows} x {matrix.cols} / {matrix.nnz}"
    )
    for name in errors:
        print(
            f"  {name:20s} field error={errors[name]:.3e}, "
            f"linear residual={residuals[name]:.3e}"
        )
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    # Also exercise the Backward Euler matrix builder using physically matched
    # spacing. Its boundary rows encode homogeneous Dirichlet data through the RHS.
    implicit_matrix = bt.build_implicit_diffusion_2d(
        cells, cells, spacing, spacing, alpha=0.01, dt=0.02
    )
    print(
        f"  Backward Euler matrix             {implicit_matrix.rows} x "
        f"{implicit_matrix.cols}, {implicit_matrix.nnz} nonzeros"
    )
    return all(checks.values())


def backward_euler_temporal_order() -> tuple[bool, np.ndarray, np.ndarray]:
    """Measure time accuracy against the exact semi-discrete sine mode."""
    cells = 20
    length = 1.0
    diffusivity = 0.1
    final_time = 0.2
    mesh = bt.StructuredMesh(cells, cells, 0.0, length, 0.0, length)
    x = np.linspace(0.0, length, cells + 1)
    y = np.linspace(0.0, length, cells + 1)
    xx, yy = np.meshgrid(x, y)
    initial = np.sin(np.pi * xx) * np.sin(np.pi * yy)

    # Exact decay rate for this eigenmode of the *discrete* five-point operator.
    spatial_rate = (
        8.0 * diffusivity * np.sin(np.pi / (2.0 * cells)) ** 2 / mesh.dx() ** 2
    )
    reference = initial * np.exp(-spatial_rate * final_time)
    step_counts = np.array([4, 8, 16, 32])
    time_steps = final_time / step_counts
    errors = []

    for steps, dt in zip(step_counts, time_steps):
        numerical, _, _ = solve_implicit(
            mesh,
            diffusivity,
            initial,
            float(dt),
            int(steps),
            bt.SparseSolverType.SparseLU,
        )
        errors.append(relative_l2(numerical, reference))

    errors_array = np.asarray(errors)
    orders = np.log(errors_array[:-1] / errors_array[1:]) / np.log(2.0)
    checks = {
        "error decreases under refinement": bool(np.all(np.diff(errors_array) < 0.0)),
        "observed final orders are first-order": bool(
            np.all((orders[-2:] > 0.9) & (orders[-2:] < 1.1))
        ),
    }

    print("\nBackward Euler temporal convergence (semi-discrete reference)")
    for dt, error in zip(time_steps, errors_array):
        print(f"  dt={dt:.5f} s, relative L2={error:.3e}")
    print(f"  observed orders: {', '.join(f'{order:.3f}' for order in orders)}")
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    return all(checks.values()), time_steps, errors_array


def save_figure(
    xx: np.ndarray,
    yy: np.ndarray,
    initial: np.ndarray,
    backward_euler: np.ndarray,
    adi: np.ndarray,
    time_steps: np.ndarray,
    errors: np.ndarray,
) -> str:
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, field, title in (
        (axes[0], initial, "Initial"),
        (axes[1], backward_euler, "Backward Euler"),
        (axes[2], adi, "Symmetric ADI"),
    ):
        image = axis.pcolormesh(xx, yy, field, shading="auto", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_aspect("equal")
        figure.colorbar(image, ax=axis)

    axes[3].loglog(time_steps, errors, "o-", label="measured")
    axes[3].loglog(
        time_steps,
        errors[-1] * time_steps / time_steps[-1],
        "k--",
        label="first-order slope",
    )
    axes[3].set_title("Backward Euler time error")
    axes[3].set_xlabel("dt [s]")
    axes[3].set_ylabel("relative L2 error")
    axes[3].grid(True, which="both", alpha=0.3)
    axes[3].legend()
    figure.tight_layout()
    output = bt.get_result_path("implicit_diffusion_checks.png", EXAMPLE_NAME)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def main() -> int:
    plt.switch_backend("Agg")
    if not bt.sparse_matrix_available():
        print(
            "This example requires a build with Eigen sparse support; nothing was run."
        )
        return 0

    print("Sparse implicit diffusion: stability, residuals, and accuracy")
    implicit_ok, xx, yy, initial, backward_euler, adi = implicit_solver_comparison()
    poisson_ok = sparse_poisson_check()
    order_ok, time_steps, errors = backward_euler_temporal_order()
    output = save_figure(xx, yy, initial, backward_euler, adi, time_steps, errors)
    print(f"\nFigure: {output}")
    print(
        "Conclusion: Backward Euler is L-stable for linear diffusion, but its "
        "first-order temporal error still requires a time-step study."
    )
    return 0 if implicit_ok and poisson_ok and order_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
