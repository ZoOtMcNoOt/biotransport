#!/usr/bin/env python3
"""Verified, deliberately scoped examples for the steady Stokes solver.

The native solver advances the two-dimensional, incompressible Newtonian
equations

    -grad(p) + mu*laplacian(v) + f = 0,    div(v) = 0.

The examples use configurations with direct analytical checks.  In
particular, ``VelocityBC.outflow()`` is only a zero-normal-gradient velocity
condition; it is not a pressure or traction outlet.
"""

import matplotlib.pyplot as plt
import numpy as np

import biotransport as bt


def verified_poiseuille_flow() -> bt.StokesResult:
    """Solve a body-force-driven channel and check the exact parabola."""
    length = 2.0
    height = 1.0
    viscosity = 1.0
    force_x = 8.0
    nx, ny = 24, 12

    mesh = bt.StructuredMesh(nx, ny, 0.0, length, 0.0, height)
    solver = bt.StokesSolver(mesh, viscosity)
    solver.set_velocity_bc(bt.Boundary.Bottom, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Top, bt.VelocityBC.no_slip())
    solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.outflow())
    solver.set_velocity_bc(bt.Boundary.Right, bt.VelocityBC.outflow())
    solver.set_body_force(force_x, 0.0)
    solver.set_tolerance(1e-5)
    solver.set_max_iterations(2000)

    result = solver.solve()
    u = result.u().reshape(ny + 1, nx + 1)
    v = result.v().reshape(ny + 1, nx + 1)
    y = np.linspace(0.0, height, ny + 1)
    exact = force_x * y * (height - y) / (2.0 * viscosity)

    velocity_error = float(np.max(np.abs(u - exact[:, None])))
    transverse_velocity = float(np.max(np.abs(v)))
    if velocity_error >= 5e-3 or transverse_velocity >= 1e-5:
        raise RuntimeError(
            "Poiseuille verification failed: "
            f"max velocity error={velocity_error:.6g}, "
            f"max transverse velocity={transverse_velocity:.6g}"
        )

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(u[:, nx // 2], y, label="native solver")
    axis.plot(exact, y, "--", label="analytical")
    axis.set(xlabel="u", ylabel="y", title="Verified plane Poiseuille profile")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(bt.get_result_path("stokes_verified_poiseuille.png"), dpi=150)
    plt.close(fig)

    print("Poiseuille verification")
    print(f"  iterations: {result.iterations}")
    print(f"  max velocity error: {velocity_error:.6g}")
    print(f"  max transverse velocity: {transverse_velocity:.6g}")
    print(f"  max divergence: {result.divergence:.6g}")
    return result


def verified_hydrostatic_equilibrium() -> bt.StokesResult:
    """Check that a conservative force in a sealed cavity causes no flow."""
    nx, ny = 10, 8
    force_x, force_y = 3.0, -2.0
    mesh = bt.StructuredMesh(nx, ny, 0.0, 2.0, 0.0, 1.0)
    solver = bt.StokesSolver(mesh, viscosity=1.0)
    solver.set_body_force(force_x, force_y)
    result = solver.solve()

    u = result.u().reshape(ny + 1, nx + 1)
    v = result.v().reshape(ny + 1, nx + 1)
    pressure = result.pressure().reshape(ny + 1, nx + 1)
    x = np.linspace(0.0, 2.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(x, y)
    exact_pressure = force_x * xx + force_y * yy
    exact_pressure -= np.mean(exact_pressure)

    velocity_magnitude = float(np.max(np.hypot(u, v)))
    pressure_error = float(np.max(np.abs(pressure - exact_pressure)))
    if velocity_magnitude >= 1e-13 or pressure_error >= 1e-12:
        raise RuntimeError(
            "Hydrostatic verification failed: "
            f"max speed={velocity_magnitude:.6g}, pressure error={pressure_error:.6g}"
        )

    print("Hydrostatic equilibrium verification")
    print(f"  max speed: {velocity_magnitude:.6g}")
    print(f"  max pressure error: {pressure_error:.6g}")
    return result


def apparent_viscosity_snapshot() -> None:
    """Compare rheology values without claiming a coupled non-Newtonian solve."""
    shear_rate = 100.0
    models = {
        "Newtonian": bt.NewtonianModel(0.001),
        "Power law n=0.7": bt.PowerLawModel(0.001, 0.7),
        "Power law n=1.3": bt.PowerLawModel(0.001, 1.3),
        "Carreau blood demo": bt.blood_carreau_model(0.45),
    }

    names = list(models)
    viscosities = np.array([models[name].viscosity(shear_rate) for name in names])
    if not np.all(np.isfinite(viscosities)) or np.any(viscosities <= 0.0):
        raise RuntimeError("Rheology model returned an invalid apparent viscosity")

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(names, viscosities * 1e3)
    axis.set_ylabel("apparent viscosity at 100 1/s (mPa s)")
    axis.set_title("Constitutive-model snapshot (not a coupled flow solution)")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(bt.get_result_path("stokes_apparent_viscosity_snapshot.png"), dpi=150)
    plt.close(fig)

    print("Apparent-viscosity snapshot")
    for name, viscosity in zip(names, viscosities):
        print(f"  {name}: {viscosity * 1e3:.6g} mPa s")
    print("  The Stokes solver itself remains Newtonian with constant viscosity.")


def main() -> None:
    verified_poiseuille_flow()
    verified_hydrostatic_equilibrium()
    apparent_viscosity_snapshot()
    print("All scoped Stokes examples passed.")


if __name__ == "__main__":
    main()
