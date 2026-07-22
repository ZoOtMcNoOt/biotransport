#!/usr/bin/env python3
"""Headless accuracy and stability checks for native Crank--Nicolson diffusion.

Mesh refinement or large diffusivity makes the high-frequency modes of a
semi-discrete diffusion problem fast relative to resolved physical time scales.
Forward Euler must resolve those modes for stability. Crank--Nicolson is
A-stable, so its linear modes remain bounded beyond that explicit limit, but it
is not L-stable and an oversized step can retain a sign-alternating stiff mode.

This executable example makes no timing or speedup claim. It checks the native
``CrankNicolsonDiffusion`` API against the exact uniform-grid eigenmode and
checks second-order temporal convergence, algebraic convergence, boundedness,
and the expected non-L-stable amplification of a stiff mode. The checks raise
explicitly and therefore remain active under ``python -O``.
"""

from __future__ import annotations

import math

import numpy as np

import biotransport as bt


def require(condition: object, message: str) -> None:
    """Raise when a verification condition is not met, including under ``-O``."""

    if not bool(condition):
        raise RuntimeError(message)


def relative_l2(numerical: np.ndarray, reference: np.ndarray) -> float:
    """Return the relative discrete L2 error."""

    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        raise ValueError("Reference norm must be non-zero")
    return float(np.linalg.norm(numerical - reference) / denominator)


def run_native_cn(
    mesh: bt.StructuredMesh,
    diffusivity: float,
    initial: np.ndarray,
    dt: float,
    steps: int,
) -> tuple[np.ndarray, bt.CrankNicolsonDiffusion, int]:
    """Advance with the native solver and verify every algebraic solve."""

    solver = bt.CrankNicolsonDiffusion(mesh, diffusivity)
    solver.set_initial_condition(initial.tolist())
    solver.set_dirichlet_boundary(bt.Boundary.Left, 0.0)
    solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)
    solver.set_tolerance(1.0e-12)

    total_iterations = 0
    for step_index in range(steps):
        result = solver.step(dt)
        require(
            result.converged,
            f"native linear solve failed at step {step_index}; "
            f"relative residual={result.residual:.3e}",
        )
        require(
            np.isfinite(result.residual),
            f"native linear solve returned a non-finite residual at step {step_index}",
        )
        total_iterations += result.iterations

    solution = np.asarray(solver.solution(), dtype=np.float64)
    require(np.all(np.isfinite(solution)), "native solution contains non-finite values")
    return solution, solver, total_iterations


def main() -> int:
    length = 1.0
    cells = 64
    diffusivity = 1.0
    final_time = 0.1
    mesh = bt.StructuredMesh(cells, 0.0, length)
    x = np.linspace(0.0, length, cells + 1)
    dx = mesh.dx()

    explicit_limit = dx**2 / (2.0 * diffusivity)
    smooth_initial = np.sin(np.pi * x / length)
    smooth_eigenvalue = -4.0 * diffusivity * np.sin(np.pi / (2.0 * cells)) ** 2 / dx**2
    semidiscrete_reference = smooth_initial * np.exp(smooth_eigenvalue * final_time)

    coarse_steps = 5
    fine_steps = 10
    coarse_dt = final_time / coarse_steps
    fine_dt = final_time / fine_steps
    coarse, coarse_solver, coarse_iterations = run_native_cn(
        mesh, diffusivity, smooth_initial, coarse_dt, coarse_steps
    )
    fine, fine_solver, fine_iterations = run_native_cn(
        mesh, diffusivity, smooth_initial, fine_dt, fine_steps
    )
    coarse_error = relative_l2(coarse, semidiscrete_reference)
    fine_error = relative_l2(fine, semidiscrete_reference)
    temporal_ratio = coarse_error / fine_error

    # The highest nontrivial Dirichlet eigenmode is stiff on this mesh. For a
    # mode du/dt=lambda*u, Crank--Nicolson has amplification
    # (1 + lambda*dt/2)/(1 - lambda*dt/2), which approaches -1 as stiffness grows.
    stiff_mode = cells - 1
    stiff_initial = np.sin(stiff_mode * np.pi * x / length)
    stiff_eigenvalue = (
        -4.0 * diffusivity * np.sin(stiff_mode * np.pi / (2.0 * cells)) ** 2 / dx**2
    )
    stiff_dt = 20.0 * explicit_limit
    stiff_solution, stiff_solver, stiff_iterations = run_native_cn(
        mesh, diffusivity, stiff_initial, stiff_dt, 1
    )
    expected_factor = (1.0 + 0.5 * stiff_eigenvalue * stiff_dt) / (
        1.0 - 0.5 * stiff_eigenvalue * stiff_dt
    )
    measured_factor = float(
        np.dot(stiff_solution, stiff_initial) / np.dot(stiff_initial, stiff_initial)
    )
    modal_error = relative_l2(stiff_solution, expected_factor * stiff_initial)

    require(coarse_dt > explicit_limit, "coarse step is not beyond the explicit limit")
    require(fine_dt > explicit_limit, "fine step is not beyond the explicit limit")
    require(fine_error < coarse_error, "temporal refinement did not reduce error")
    require(
        3.7 < temporal_ratio < 4.3,
        f"temporal error ratio {temporal_ratio:.6g} is not second-order",
    )
    require(
        fine_error < 1.0e-3,
        f"fine-grid temporal error {fine_error:.6g} is too large",
    )
    require(
        math.isclose(coarse_solver.time(), final_time, rel_tol=0.0, abs_tol=1.0e-14),
        "coarse solve did not reach the requested time",
    )
    require(
        math.isclose(fine_solver.time(), final_time, rel_tol=0.0, abs_tol=1.0e-14),
        "fine solve did not reach the requested time",
    )
    require(
        np.linalg.norm(stiff_solution)
        <= np.linalg.norm(stiff_initial) * (1.0 + 1.0e-10),
        "stiff-mode step increased the discrete L2 norm",
    )
    require(
        expected_factor < 0.0,
        "stiff mode did not exercise negative Crank--Nicolson amplification",
    )
    require(
        math.isclose(measured_factor, expected_factor, rel_tol=0.0, abs_tol=2.0e-9),
        "measured stiff-mode amplification does not match the discrete theory",
    )
    require(
        modal_error < 2.0e-9,
        f"stiff modal error {modal_error:.6g} is too large",
    )
    require(
        math.isclose(stiff_solver.time(), stiff_dt, rel_tol=0.0, abs_tol=1.0e-14),
        "stiff solve did not reach the requested time",
    )

    print("Native Crank--Nicolson stiff-mode verification")
    print(f"  cells                                  {cells}")
    print(f"  explicit Forward Euler limit           {explicit_limit:.6e} s")
    print(f"  coarse dt / explicit limit             {coarse_dt / explicit_limit:.1f}")
    print(f"  fine dt / explicit limit               {fine_dt / explicit_limit:.1f}")
    print(f"  coarse semidiscrete relative L2 error  {coarse_error:.6e}")
    print(f"  fine semidiscrete relative L2 error    {fine_error:.6e}")
    print(f"  temporal error ratio                   {temporal_ratio:.4f}")
    print(
        f"  smooth-mode PCG iterations             {coarse_iterations + fine_iterations}"
    )
    print(f"  stiff dt / explicit limit              {stiff_dt / explicit_limit:.1f}")
    print(f"  stiff amplification, expected          {expected_factor:.8f}")
    print(f"  stiff amplification, measured          {measured_factor:.8f}")
    print(f"  stiff modal relative error             {modal_error:.3e}")
    print(f"  stiff-step PCG iterations              {stiff_iterations}")
    print("  all accuracy, stability, and algebraic checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
