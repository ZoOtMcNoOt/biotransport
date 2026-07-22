#!/usr/bin/env python3
"""Science-first cylindrical-coordinate examples.

The examples use :class:`biotransport.CylindricalMesh` for metric factors,
indexing, control-volume measures, and differential operators.  A full 3-D
mesh stores ``ntheta`` unique periodic nodes; it does not duplicate ``2*pi``.

All geometry is in SI units.  Array fields below use shape ``(nz + 1,
nr + 1)`` so C-order flattening is radial-fastest, matching the C++ storage
contract.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

import biotransport as bt


def radial_drug_diffusion() -> None:
    """Diffuse drug from a fixed-concentration cylindrical stent surface.

    The C++ mesh evaluates

    ``dC/dt = D * (1/r) * d/dr(r * dC/dr)``.

    The inner surface is Dirichlet and the remote outer surface has zero
    outward derivative.  This is a reservoir-release model, so total drug in
    the tissue is not expected to be conserved.
    """

    print("\nRadial drug diffusion from a cylindrical stent")
    stent_radius_m = 1.5e-3
    tissue_radius_m = 5.0e-3
    diffusivity_m2_s = 1.0e-11
    surface_concentration = 1.0

    mesh = bt.CylindricalMesh(100, stent_radius_m, tissue_radius_m)
    radius = bt.r_nodes(mesh)
    concentration = np.zeros(mesh.num_nodes())
    concentration[0] = surface_concentration

    # For the second-order explicit radial operator, D*dt/dr^2 < 1/2 is a
    # conservative stability choice on this annulus.
    nominal_dt_s = 0.40 * mesh.dr() ** 2 / diffusivity_m2_s
    final_time_s = 24.0 * 3600.0
    requested_hours = (0.0, 1.0, 6.0, 12.0, 24.0)
    snapshots: dict[float, np.ndarray] = {0.0: concentration.copy()}
    pending = list(requested_hours[1:])

    time_s = 0.0
    steps = 0
    while time_s < final_time_s:
        dt_s = min(nominal_dt_s, final_time_s - time_s)
        laplacian = np.asarray(mesh.laplacian(concentration))
        updated = concentration + dt_s * diffusivity_m2_s * laplacian

        updated[0] = surface_concentration
        # Second-order one-sided enforcement of dC/dr = 0 at r = R.
        updated[-1] = (4.0 * updated[-2] - updated[-3]) / 3.0
        concentration = updated
        time_s += dt_s
        steps += 1

        while pending and time_s >= pending[0] * 3600.0:
            snapshots[pending.pop(0)] = concentration.copy()

    print(f"  {mesh.nr()} radial cells, dr = {mesh.dr() * 1e6:.1f} um")
    print(
        f"  {steps} C++ Laplacian evaluations, exact final time = {time_s / 3600:.1f} h"
    )

    fig = plt.figure(figsize=(12, 5))
    profile_axis = fig.add_subplot(1, 2, 1)
    polar_axis = fig.add_subplot(1, 2, 2, projection="polar")
    for hour in requested_hours:
        profile_axis.plot(
            (radius - stent_radius_m) * 1e3,
            snapshots[hour],
            linewidth=2,
            label=f"{hour:g} h",
        )
    profile_axis.set(
        xlabel="Distance from stent surface (mm)",
        ylabel="Normalized concentration",
        title="Reservoir-driven radial diffusion",
    )
    profile_axis.grid(alpha=0.3)
    profile_axis.legend()

    theta = np.linspace(0.0, 2.0 * math.pi, 181)
    radial_grid, theta_grid = np.meshgrid(radius, theta)
    image = polar_axis.pcolormesh(
        theta_grid,
        radial_grid * 1e3,
        np.tile(concentration, (theta.size, 1)),
        shading="auto",
        cmap="YlOrRd",
    )
    polar_axis.set_title("Concentration after 24 h")
    polar_axis.set_ylabel("r (mm)")
    fig.colorbar(image, ax=polar_axis, label="C/C_surface")
    fig.tight_layout()
    fig.savefig(bt.get_result_path("cylindrical_radial_diffusion.png"), dpi=150)
    plt.close(fig)

    below_threshold = np.flatnonzero(concentration < 0.1 * surface_concentration)
    penetration_m = (
        tissue_radius_m - stent_radius_m
        if below_threshold.size == 0
        else radius[below_threshold[0]] - stent_radius_m
    )
    print(f"  10% concentration penetration = {penetration_m * 1e3:.2f} mm")


def verify_axisymmetric_operator() -> None:
    """Verify the cylindrical Laplacian against a manufactured solution.

    For ``phi(r,z) = r^2 + 3 z^2``, the axisymmetric cylindrical
    Laplacian is exactly ``4 + 6 = 10``, including at ``r=0``.
    """

    print("\nAxisymmetric manufactured-solution verification")
    mesh = bt.CylindricalMesh(40, 50, 0.0, 5.0e-3, -5.0e-3, 5.0e-3)
    radius = bt.r_nodes(mesh)
    axial = bt.z_nodes(mesh)
    radial_grid, axial_grid = np.meshgrid(radius, axial)
    phi = radial_grid**2 + 3.0 * axial_grid**2

    numerical = np.asarray(mesh.laplacian(phi.ravel())).reshape(phi.shape)
    max_error = float(np.max(np.abs(numerical - 10.0)))

    integrated_volume = sum(
        mesh.cell_volume(i, 0, k)
        for k in range(mesh.nz() + 1)
        for i in range(mesh.nr() + 1)
    )
    exact_volume = math.pi * mesh.rmax() ** 2 * (mesh.zmax() - mesh.zmin())

    print(f"  max |Laplacian(phi) - 10| = {max_error:.3e}")
    print(
        f"  nodal-volume integration relative error = {abs(integrated_volume / exact_volume - 1):.3e}"
    )
    if max_error > 1e-8:
        raise RuntimeError("cylindrical manufactured-solution verification failed")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.pcolormesh(
        axial_grid * 1e3,
        radial_grid * 1e3,
        numerical,
        shading="auto",
        cmap="viridis",
    )
    ax.set(xlabel="z (mm)", ylabel="r (mm)", title="Numerical cylindrical Laplacian")
    fig.colorbar(image, ax=ax, label="Laplacian(phi)")
    fig.tight_layout()
    fig.savefig(bt.get_result_path("cylindrical_manufactured_solution.png"), dpi=150)
    plt.close(fig)


def poiseuille_pipe_flow() -> None:
    """Evaluate steady Newtonian Poiseuille flow in a circular pipe."""

    print("\nPoiseuille flow in a cylindrical vessel")
    pipe_radius_m = 2.0e-3
    pipe_length_m = 2.0e-2
    viscosity_pa_s = 3.5e-3
    density_kg_m3 = 1060.0
    pressure_drop_pa = 100.0

    mesh = bt.CylindricalMesh(80, 0.0, pipe_radius_m)
    radius = bt.r_nodes(mesh)
    velocity = (
        pressure_drop_pa
        * (pipe_radius_m**2 - radius**2)
        / (4.0 * viscosity_pa_s * pipe_length_m)
    )

    exact_flow_m3_s = (
        math.pi
        * pressure_drop_pa
        * pipe_radius_m**4
        / (8.0 * viscosity_pa_s * pipe_length_m)
    )
    nodal_flow_m3_s = sum(velocity[i] * mesh.cell_area(i) for i in range(mesh.nr() + 1))
    average_velocity_m_s = exact_flow_m3_s / mesh.cross_section_area()
    reynolds = (
        density_kg_m3 * average_velocity_m_s * (2.0 * pipe_radius_m) / viscosity_pa_s
    )
    wall_shear_pa = pressure_drop_pa * pipe_radius_m / (2.0 * pipe_length_m)

    print(f"  Q = {exact_flow_m3_s * 1e6:.4f} mL/s")
    print(
        f"  nodal quadrature relative error = {abs(nodal_flow_m3_s / exact_flow_m3_s - 1):.3e}"
    )
    print(f"  Reynolds number = {reynolds:.1f}")
    print(f"  wall shear stress = {wall_shear_pa:.2f} Pa")
    print(
        f"  wall shear rate = {bt.pipe_wall_shear_rate(exact_flow_m3_s, pipe_radius_m):.1f} 1/s"
    )

    shear_stress = pressure_drop_pa * radius / (2.0 * pipe_length_m)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(velocity * 100.0, radius * 1e3, linewidth=2)
    axes[0].set(
        xlabel="Axial velocity (cm/s)",
        ylabel="Radius (mm)",
        title="Velocity profile",
    )
    axes[1].plot(shear_stress, radius * 1e3, linewidth=2, color="tab:red")
    axes[1].set(
        xlabel="Shear stress (Pa)",
        ylabel="Radius (mm)",
        title="Shear-stress magnitude",
    )
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(bt.get_result_path("cylindrical_poiseuille_flow.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    radial_drug_diffusion()
    verify_axisymmetric_operator()
    poiseuille_pipe_flow()
    print("\nCylindrical-coordinate examples completed.")
