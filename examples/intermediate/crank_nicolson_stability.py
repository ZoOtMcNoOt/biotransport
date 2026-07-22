#!/usr/bin/env python3
"""Crank--Nicolson stability, accuracy, and damping demonstration.

For linear diffusion, Crank--Nicolson is A-stable: every eigenmode of the
semi-discrete diffusion operator remains bounded for every positive time step.
It is *not* L-stable.  A very stiff mode has an amplification factor approaching
-1 rather than 0, so an oversized step can leave a bounded, sign-alternating
mode and can violate positivity.  Linear stability is therefore not an accuracy
or monotonicity guarantee.

This script checks a smooth Dirichlet eigenmode against its analytical solution,
shows that the conservative explicit API rejects a step beyond its certified CFL
limit, and measures the large-step Crank--Nicolson amplification of a stiff mode.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

import biotransport as bt

EXAMPLE_NAME = "crank_nicolson_stability"


def relative_l2(numerical: np.ndarray, reference: np.ndarray) -> float:
    """Return the relative discrete L2 error."""
    return float(np.linalg.norm(numerical - reference) / np.linalg.norm(reference))


def zero_dirichlet(problem_or_solver):
    """Apply homogeneous Dirichlet data to a 1D problem or specialized solver."""
    problem_or_solver.set_dirichlet_boundary(bt.Boundary.Left, 0.0)
    problem_or_solver.set_dirichlet_boundary(bt.Boundary.Right, 0.0)
    return problem_or_solver


def run_cn(
    mesh: bt.StructuredMesh,
    diffusivity: float,
    initial: np.ndarray,
    dt: float,
    steps: int,
) -> tuple[np.ndarray, bt.CrankNicolsonDiffusion]:
    """Advance the specialized native Crank--Nicolson solver."""
    solver = bt.CrankNicolsonDiffusion(mesh, diffusivity)
    solver.set_initial_condition(initial)
    zero_dirichlet(solver)
    solver.solve(dt=dt, num_steps=steps)
    return np.asarray(solver.solution()), solver


def main() -> int:
    plt.switch_backend("Agg")
    length = 1.0
    cells = 80
    diffusivity = 0.01
    final_time = 0.1
    mesh = bt.StructuredMesh(cells, 0.0, length)
    x = np.linspace(0.0, length, cells + 1)
    dx = mesh.dx()

    # The standard 1D Forward Euler bound for this uniform diffusion problem.
    explicit_limit = dx**2 / (2.0 * diffusivity)
    small_target_dt = 0.4 * explicit_limit
    small_steps = math.ceil(final_time / small_target_dt)
    small_dt = final_time / small_steps

    smooth_initial = np.sin(np.pi * x / length)
    smooth_reference = smooth_initial * np.exp(
        -diffusivity * (np.pi / length) ** 2 * final_time
    )

    explicit_problem = (
        bt.Problem(mesh)
        .diffusivity(diffusivity)
        .initial_condition(smooth_initial)
        .dirichlet(bt.Boundary.Left, 0.0)
        .dirichlet(bt.Boundary.Right, 0.0)
    )
    explicit_result = bt.solve(
        explicit_problem,
        end_time=final_time,
        time_step=small_dt,
    )
    explicit_solution = np.asarray(explicit_result.concentration)
    cn_small, cn_small_solver = run_cn(
        mesh, diffusivity, smooth_initial, small_dt, small_steps
    )

    explicit_error = relative_l2(explicit_solution, smooth_reference)
    cn_error = relative_l2(cn_small, smooth_reference)

    # Use a high-frequency discrete sine eigenmode.  For u_t = lambda*u, the
    # Crank--Nicolson factor is (1 + lambda*dt/2)/(1 - lambda*dt/2).
    mode = cells - 1
    stiff_initial = np.sin(mode * np.pi * x / length)
    large_dt = 20.0 * explicit_limit
    stiff_cn, _ = run_cn(mesh, diffusivity, stiff_initial, large_dt, 1)
    laplacian_eigenvalue = (
        -4.0 * diffusivity * np.sin(mode * np.pi / (2.0 * cells)) ** 2 / dx**2
    )
    expected_factor = (1.0 + 0.5 * large_dt * laplacian_eigenvalue) / (
        1.0 - 0.5 * large_dt * laplacian_eigenvalue
    )
    measured_factor = float(
        np.dot(stiff_cn, stiff_initial) / np.dot(stiff_initial, stiff_initial)
    )
    modal_residual = relative_l2(stiff_cn, expected_factor * stiff_initial)

    # A-stability does not imply a discrete maximum principle. A concentrated,
    # nonnegative field develops an undershoot under the same oversized step.
    pulse_initial = np.zeros_like(x)
    pulse_initial[cells // 2] = 1.0
    pulse_cn, _ = run_cn(mesh, diffusivity, pulse_initial, large_dt, 1)
    pulse_minimum = float(np.min(pulse_cn))

    # The friendly explicit API refuses a user-supplied step outside its
    # certified range instead of silently producing an unstable trajectory.
    rejected_unsafe_explicit_step = False
    unsafe_problem = (
        bt.Problem(mesh)
        .diffusivity(diffusivity)
        .initial_condition(stiff_initial)
        .dirichlet(bt.Boundary.Left, 0.0)
        .dirichlet(bt.Boundary.Right, 0.0)
    )
    try:
        bt.solve(unsafe_problem, end_time=large_dt, time_step=large_dt)
    except ValueError:
        rejected_unsafe_explicit_step = True

    checks = {
        "explicit smooth-mode relative L2 error < 2e-4": explicit_error < 2.0e-4,
        "CN smooth-mode relative L2 error < 2e-4": cn_error < 2.0e-4,
        "CN solver reached the requested time": math.isclose(
            cn_small_solver.time(), final_time, rel_tol=0.0, abs_tol=1.0e-14
        ),
        "large-step CN mode is bounded": np.linalg.norm(stiff_cn)
        <= np.linalg.norm(stiff_initial) * (1.0 + 1.0e-10),
        "large-step CN amplification matches theory": modal_residual < 1.0e-8,
        "large-step CN mode changes sign (not L-stable)": measured_factor < 0.0,
        "oversized CN step demonstrates loss of positivity": pulse_minimum < -1.0e-3,
        "unsafe explicit step is rejected": rejected_unsafe_explicit_step,
    }

    print("Crank--Nicolson: stability is not accuracy")
    print(f"  grid spacing                         {dx:.6g} m")
    print(f"  explicit diffusion limit             {explicit_limit:.6g} s")
    print(f"  resolved comparison step             {small_dt:.6g} s")
    print(f"  explicit smooth-mode relative L2     {explicit_error:.3e}")
    print(f"  CN smooth-mode relative L2           {cn_error:.3e}")
    print(f"  oversized step / explicit limit      {large_dt / explicit_limit:.1f}")
    print(f"  stiff-mode amplification, predicted  {expected_factor:.6f}")
    print(f"  stiff-mode amplification, measured   {measured_factor:.6f}")
    print(f"  stiff-mode shape residual             {modal_residual:.3e}")
    print(f"  nonnegative-pulse minimum after CN    {pulse_minimum:.6f}")

    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(x, smooth_reference, "k-", label="analytical")
    axes[0].plot(x, explicit_solution, "--", label="explicit")
    axes[0].plot(x, cn_small, ":", linewidth=2.2, label="Crank--Nicolson")
    axes[0].set_title("Resolved smooth mode")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("field")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, stiff_initial, "k--", alpha=0.65, label="before step")
    axes[1].plot(x, stiff_cn, color="tab:red", label="after oversized CN step")
    axes[1].set_title("Bounded sign reversal: CN is not L-stable")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("field")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(x, pulse_initial, "k--", alpha=0.65, label="nonnegative initial")
    axes[2].plot(x, pulse_cn, color="tab:purple", label="after oversized CN step")
    axes[2].axhline(0.0, color="0.4", linewidth=0.8)
    axes[2].set_title("Linear stability does not ensure positivity")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("field")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    figure.tight_layout()
    output = bt.get_result_path("stability_and_damping.png", EXAMPLE_NAME)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"  figure                               {output}")
    print(
        "\nInterpretation: A-stability prevents unbounded linear growth. It does "
        "not make an oversized step accurate, strongly damped, or positivity-preserving."
    )

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
