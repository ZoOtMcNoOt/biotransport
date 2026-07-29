"""Contract tests for the bounded, staggered-grid Navier--Stokes solver."""

from __future__ import annotations

import math

import numpy as np
import pytest

import biotransport as bt


def _solver(nx: int = 8, ny: int = 6) -> tuple[bt.StructuredMesh, object]:
    mesh = bt.StructuredMesh(nx, ny, 0.0, 1.0, 0.0, ny / nx)
    return mesh, bt.NavierStokesSolver(mesh, density=1.0, viscosity=0.05)


def _mac_divergence(mesh: bt.StructuredMesh, u: np.ndarray, v: np.ndarray) -> float:
    nx = mesh.nx()
    ny = mesh.ny()
    stride = nx + 1
    u_grid = np.asarray(u).reshape(ny + 1, stride)
    v_grid = np.asarray(v).reshape(ny + 1, stride)
    divergence = (u_grid[:ny, 1 : nx + 1] - u_grid[:ny, :nx]) / mesh.dx() + (
        v_grid[1 : ny + 1, :nx] - v_grid[:ny, :nx]
    ) / mesh.dy()
    return float(np.max(np.abs(divergence)))


class TestBoundedBoundaryContract:
    @pytest.mark.parametrize(
        "condition",
        [
            pytest.param(bt.VelocityBC.inflow(0.1), id="inflow"),
            pytest.param(bt.VelocityBC.outflow(), id="outflow"),
            pytest.param(bt.VelocityBC.stress_free(), id="traction"),
        ],
    )
    def test_open_and_traction_boundaries_fail_loudly(self, condition: object) -> None:
        _, solver = _solver()

        with pytest.raises(ValueError, match="supports NOSLIP and DIRICHLET"):
            solver.set_velocity_bc(bt.Boundary.Left, condition)

    def test_nonfinite_prescribed_velocity_is_rejected(self) -> None:
        _, solver = _solver()

        with pytest.raises(ValueError, match="finite"):
            solver.set_velocity_bc(
                bt.Boundary.Top, bt.VelocityBC.dirichlet(math.nan, 0.0)
            )

    def test_incompatible_closed_domain_flux_is_rejected(self) -> None:
        _, solver = _solver()
        solver.set_velocity_bc(bt.Boundary.Left, bt.VelocityBC.dirichlet(0.1, 0.0))

        with pytest.raises(ValueError, match="nonzero net boundary flux"):
            solver.solve_steps(1)


class TestResultContract:
    @pytest.mark.parametrize(
        "scheme", [bt.ConvectionScheme.UPWIND, bt.ConvectionScheme.CENTRAL]
    )
    def test_uniform_velocity_is_an_exact_navier_stokes_solution(
        self, scheme: object
    ) -> None:
        """A spatially uniform flow has zero convection, diffusion, and pressure gradient."""
        nx, ny = 9, 7
        mesh = bt.StructuredMesh(nx, ny, -0.4, 1.4, 0.2, 1.25)
        solver = bt.NavierStokesSolver(mesh, density=1.3, viscosity=0.07)
        exact_u = 0.23
        exact_v = -0.17
        boundary = bt.VelocityBC.dirichlet(exact_u, exact_v)
        for side in (
            bt.Boundary.Left,
            bt.Boundary.Right,
            bt.Boundary.Bottom,
            bt.Boundary.Top,
        ):
            solver.set_velocity_bc(side, boundary)

        u0 = np.full(mesh.num_nodes(), exact_u)
        v0 = np.full(mesh.num_nodes(), exact_v)
        solver.set_initial_velocity(u0, v0)
        solver.set_convection_scheme(scheme)
        solver.set_time_step(0.001)
        solver.set_pressure_tolerance(1e-12)

        result = solver.solve(0.031)

        np.testing.assert_allclose(result.u(), exact_u, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(result.v(), exact_v, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(result.pressure(), 0.0, atol=2e-15, rtol=0.0)
        assert result.time_steps == 31
        assert result.max_velocity == pytest.approx(
            math.hypot(exact_u, exact_v), abs=2e-15
        )
        assert result.divergence == pytest.approx(0.0, abs=2e-15)
        assert result.pressure_residual == pytest.approx(0.0, abs=2e-15)
        assert result.stable

    def test_exact_time_shapes_padding_and_diagnostics(self) -> None:
        nx, ny = 6, 5
        mesh, solver = _solver(nx, ny)
        lid_velocity = 0.1
        solver.set_velocity_bc(
            bt.Boundary.Top, bt.VelocityBC.dirichlet(lid_velocity, 0.0)
        )
        solver.set_time_step(0.002)

        result = solver.solve(0.005)

        assert result.time == pytest.approx(0.005, abs=1e-15)
        assert result.time_steps == 3
        assert result.stable
        assert result.max_velocity >= 0.0
        assert result.reynolds >= 0.0
        assert result.pressure_iterations >= 0
        assert 0.0 <= result.pressure_residual <= 1e-8
        assert result.divergence >= 0.0

        expected_shape = ((nx + 1) * (ny + 1),)
        u = result.u()
        v = result.v()
        pressure = result.pressure()
        assert u.shape == expected_shape
        assert v.shape == expected_shape
        assert pressure.shape == expected_shape
        assert np.all(np.isfinite(u))
        assert np.all(np.isfinite(v))
        assert np.all(np.isfinite(pressure))

        u_grid = u.reshape(ny + 1, nx + 1)
        v_grid = v.reshape(ny + 1, nx + 1)
        pressure_grid = pressure.reshape(ny + 1, nx + 1)
        np.testing.assert_allclose(u_grid[-1, :], lid_velocity, atol=0.0)
        np.testing.assert_allclose(v_grid[:, -1], 0.0, atol=0.0)
        np.testing.assert_allclose(
            pressure_grid[:ny, -1], pressure_grid[:ny, -2], atol=0.0
        )
        np.testing.assert_allclose(pressure_grid[-1, :], pressure_grid[-2, :], atol=0.0)

    def test_projection_reports_quantitatively_small_divergence(self) -> None:
        nx, ny = 8, 6
        mesh, solver = _solver(nx, ny)
        stride = nx + 1
        u0 = np.zeros(mesh.num_nodes())
        v0 = np.zeros(mesh.num_nodes())
        for j in range(ny):
            for i in range(1, nx):
                u0[j * stride + i] = 0.05 * np.sin(2.0 * np.pi * i / nx)

        solver.set_initial_velocity(u0, v0)
        solver.set_pressure_tolerance(1e-9)
        solver.set_time_step(0.002)
        result = solver.solve_steps(1)

        measured_divergence = _mac_divergence(mesh, result.u(), result.v())
        assert result.stable
        assert result.pressure_iterations > 0
        assert result.pressure_residual <= 1e-9
        assert result.divergence == pytest.approx(measured_divergence, abs=1e-13)
        assert result.divergence < 5e-8

    @pytest.mark.parametrize("num_steps", [0, 1, 4])
    def test_solve_steps_completes_exactly_requested_steps(
        self, num_steps: int
    ) -> None:
        _, solver = _solver(4, 4)
        dt = 0.002
        solver.set_time_step(dt)

        result = solver.solve_steps(num_steps)

        assert result.time_steps == num_steps
        assert result.time == pytest.approx(num_steps * dt, abs=1e-15)
        assert result.stable
        assert result.divergence == 0.0


class TestStabilityAndInputValidation:
    def test_max_time_step_matches_the_diffusive_bound_at_rest(self) -> None:
        mesh, solver = _solver(4, 2)
        zeros = np.zeros(mesh.num_nodes())
        expected = 0.25 * 0.5 / (0.05 * (1.0 / mesh.dx() ** 2 + 1.0 / mesh.dy() ** 2))

        assert solver.max_time_step(zeros, zeros) == pytest.approx(expected)

    @pytest.mark.parametrize("cfl", [0.0, -0.1, 1.01, math.nan, math.inf])
    def test_invalid_cfl_is_rejected(self, cfl: float) -> None:
        _, solver = _solver()

        with pytest.raises(ValueError, match="CFL"):
            solver.set_cfl(cfl)

    @pytest.mark.parametrize(
        "scheme", [bt.ConvectionScheme.QUICK, bt.ConvectionScheme.HYBRID]
    )
    def test_unimplemented_convection_schemes_are_rejected(
        self, scheme: object
    ) -> None:
        _, solver = _solver()

        with pytest.raises(ValueError, match="not implemented"):
            solver.set_convection_scheme(scheme)

    @pytest.mark.parametrize(
        "scheme", [bt.ConvectionScheme.UPWIND, bt.ConvectionScheme.CENTRAL]
    )
    def test_implemented_convection_schemes_are_accepted(self, scheme: object) -> None:
        _, solver = _solver()

        assert solver.set_convection_scheme(scheme) is solver

    def test_initial_velocity_requires_exact_finite_packed_fields(self) -> None:
        mesh, solver = _solver()
        valid = np.zeros(mesh.num_nodes())
        wrong_size = np.zeros(mesh.num_nodes() - 1)
        nonfinite = valid.copy()
        nonfinite[3] = math.nan

        with pytest.raises(ValueError, match="exactly"):
            solver.set_initial_velocity(wrong_size, valid)
        with pytest.raises(ValueError, match="exactly"):
            solver.set_initial_velocity(valid, wrong_size)
        with pytest.raises(ValueError, match="finite"):
            solver.set_initial_velocity(nonfinite, valid)
        with pytest.raises(ValueError, match="finite"):
            solver.max_time_step(valid, nonfinite)

    @pytest.mark.parametrize(
        ("force_x", "force_y"),
        [(math.nan, 0.0), (0.0, math.inf), (-math.inf, 0.0)],
    )
    def test_nonfinite_constant_body_force_is_rejected(
        self, force_x: float, force_y: float
    ) -> None:
        _, solver = _solver()

        with pytest.raises(ValueError, match="finite"):
            solver.set_body_force(force_x, force_y)

    def test_fixed_step_above_stability_limit_is_rejected(self) -> None:
        mesh, solver = _solver(4, 4)
        zeros = np.zeros(mesh.num_nodes())
        solver.set_time_step(1.01 * solver.max_time_step(zeros, zeros))

        with pytest.raises(ValueError, match="exceeds the explicit stability bound"):
            solver.solve_steps(1)

    @pytest.mark.parametrize("duration", [-1.0, math.nan, math.inf])
    def test_invalid_duration_is_rejected(self, duration: float) -> None:
        _, solver = _solver()

        with pytest.raises(ValueError, match="duration"):
            solver.solve(duration)

    def test_unsupported_snapshot_interval_and_negative_steps_are_rejected(
        self,
    ) -> None:
        _, solver = _solver()

        with pytest.raises(ValueError, match="not implemented"):
            solver.solve(0.1, output_interval=0.01)
        with pytest.raises(ValueError, match="nonnegative"):
            solver.solve_steps(-1)
