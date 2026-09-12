"""Tests for reconstructed fluxes, transfer rates, and the balance statement.

The valuable checks here are the ones with an independent answer: a linear
profile whose flux is exactly ``D*dc/L``, a reacting slab whose wall flux has a
closed form, and a closed domain where every rate must be zero.
"""

from __future__ import annotations

import numpy as np
import pytest

import biotransport as bt


DIFFUSIVITY = 2.0e-9
LENGTH = 1.0e-2


def _linear_problem(cells: int = 200):
    """Fixed values at both ends, no reaction: the steady field is a straight line."""
    mesh = bt.mesh_1d(cells, 0.0, LENGTH)
    return (
        bt.Problem(mesh)
        .diffusivity(DIFFUSIVITY)
        .initial(0.0)
        .dirichlet("left", 1.0)
        .dirichlet("right", 0.0)
    )


def _reacting_problem(cells: int = 400, rate: float = 5.0e-4):
    """Fed from the left, sealed at the right, first-order consumption."""
    mesh = bt.mesh_1d(cells, 0.0, LENGTH)
    return (
        bt.Problem(mesh)
        .diffusivity(DIFFUSIVITY)
        .linear_decay(rate)
        .initial(1.0)
        .dirichlet("left", 1.0)
        .sealed("right")
    )


class TestInteriorFlux:
    def test_linear_profile_has_uniform_exact_flux(self):
        sol = bt.solve_steady(_linear_problem())
        flux = sol.flux()

        expected = DIFFUSIVITY * 1.0 / LENGTH
        assert flux.shape == (sol.mesh.nx(),)
        np.testing.assert_allclose(flux, expected, rtol=1.0e-12)

    def test_flux_points_down_the_gradient(self):
        """c falls left to right, so J points along +x."""
        sol = bt.solve_steady(_linear_problem())
        assert np.all(sol.flux() > 0.0)

    def test_2d_returns_both_components(self):
        mesh = bt.mesh_2d(20, 12, 0.0, 1.0, 0.0, 0.5)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-3)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 0.0)
            .sealed("bottom")
            .sealed("top")
        )
        sol = bt.solve(problem, end_time=3000.0)
        flux_x, flux_y = sol.flux()

        assert flux_x.shape == (mesh.ny() + 1, mesh.nx())
        assert flux_y.shape == (mesh.ny(), mesh.nx() + 1)
        np.testing.assert_allclose(flux_x, 1.0e-3, rtol=1.0e-9)
        # Nothing drives transport across y.
        np.testing.assert_allclose(flux_y, 0.0, atol=1.0e-18)

    def test_zero_diffusivity_layer_blocks_flux(self):
        """Harmonic face averaging must make an impermeable layer impermeable."""
        mesh = bt.mesh_1d(20, 0.0, 1.0)
        field = np.zeros(21)
        field[:10] = 1.0
        diffusivity = np.full(21, 1.0e-9)
        diffusivity[10] = 0.0

        problem = (
            bt.Problem(mesh)
            .diffusivity(diffusivity)
            .initial(field)
            .sealed("left")
            .sealed("right")
        )
        flux = bt.solve(problem, end_time=0.0).flux()
        assert flux[9] == 0.0
        assert flux[10] == 0.0


class TestBoundaryFlux:
    def test_dirichlet_wall_flux_is_exact_for_a_linear_profile(self):
        sol = bt.solve_steady(_linear_problem())
        expected = DIFFUSIVITY * 1.0 / LENGTH

        # Outward: negative at the left (entering), positive at the right.
        assert sol.flux_at("left") == pytest.approx(-expected, rel=1.0e-12)
        assert sol.flux_at("right") == pytest.approx(expected, rel=1.0e-12)

    def test_reacting_wall_flux_matches_the_closed_form(self):
        """J(0) = D*cs*m*tanh(m L) with m = sqrt(k/D)."""
        rate = 5.0e-4
        sol = bt.solve_steady(_reacting_problem(rate=rate))

        modulus = np.sqrt(rate / DIFFUSIVITY)
        expected = DIFFUSIVITY * 1.0 * modulus * np.tanh(modulus * LENGTH)
        assert sol.flux_at("left") == pytest.approx(-expected, rel=1.0e-3)

    def test_sealed_wall_carries_no_flux(self):
        sol = bt.solve_steady(_reacting_problem())
        assert sol.flux_at("right") == 0.0

    def test_neumann_wall_reports_its_declared_gradient(self):
        """A prescribed gradient g gives an outward flux of exactly -D*g."""
        mesh = bt.mesh_1d(50, 0.0, LENGTH)
        gradient = -3.0
        problem = (
            bt.Problem(mesh)
            .diffusivity(DIFFUSIVITY)
            .initial(1.0)
            .neumann("left", gradient)
            .dirichlet("right", 1.0)
        )
        sol = bt.solve(problem, end_time=1.0)
        assert sol.flux_at("left") == pytest.approx(-DIFFUSIVITY * gradient)

    def test_1d_mesh_has_no_top_or_bottom(self):
        sol = bt.solve_steady(_linear_problem())
        with pytest.raises(ValueError, match="only 'left' and 'right'"):
            sol.flux_at("top")

    def test_unknown_side_is_rejected(self):
        sol = bt.solve_steady(_linear_problem())
        with pytest.raises(ValueError, match="not a boundary"):
            sol.flux_at("sideways")


class TestRates:
    def test_1d_rate_equals_the_flux(self):
        sol = bt.solve_steady(_linear_problem())
        assert sol.rate("left") == pytest.approx(sol.flux_at("left"))

    def test_2d_rate_integrates_along_the_edge(self):
        height = 0.5
        mesh = bt.mesh_2d(20, 12, 0.0, 1.0, 0.0, height)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-3)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 0.0)
            .sealed("bottom")
            .sealed("top")
        )
        sol = bt.solve(problem, end_time=3000.0)

        expected = 1.0e-3 * 1.0 / 1.0 * height
        assert sol.rate("left") == pytest.approx(-expected, rel=1.0e-6)
        assert sol.rate("right") == pytest.approx(expected, rel=1.0e-6)
        assert sol.rate("bottom") == pytest.approx(0.0, abs=1.0e-18)

    def test_uptake_balances_what_enters_at_steady_state(self):
        sol = bt.solve_steady(_reacting_problem())
        entering = -sol.rate("left")
        assert sol.uptake() == pytest.approx(-entering, rel=1.0e-6)

    def test_uptake_is_zero_without_a_reaction(self):
        assert bt.solve_steady(_linear_problem()).uptake() == 0.0


class TestBalance:
    def test_steady_balance_closes(self):
        report = bt.solve_steady(_reacting_problem()).balance()

        assert report.steady is True
        scale = max(abs(report.entered), abs(report.produced))
        assert abs(report.residual) / scale < 1.0e-8
        assert "steady state" in str(report)

    def test_steady_balance_names_every_side(self):
        report = bt.solve_steady(_reacting_problem()).balance()
        assert set(report.by_side) == {"left", "right"}
        assert "entering" in str(report)

    def test_sealed_domain_transfers_nothing(self):
        mesh = bt.mesh_1d(200, 0.0, LENGTH)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-9)
            .initial(bt.gaussian(mesh, center=0.005, width=0.0005))
            .sealed("left")
            .sealed("right")
        )
        report = bt.solve(problem, end_time=60.0, frames=8).balance()

        assert report.entered == pytest.approx(0.0, abs=1.0e-18)
        assert report.produced == 0.0
        assert report.closure_error == pytest.approx(0.0, abs=1.0e-15)

    def test_transient_closure_improves_with_resolution(self):
        """Refining frames and the time step tightens the integrated balance."""
        problem = (
            bt.Problem(bt.mesh_1d(200, 0.0, LENGTH))
            .diffusivity(1.0e-9)
            .linear_decay(0.02)
            .initial(1.0)
            .dirichlet("left", 1.0)
            .sealed("right")
        )
        coarse = bt.solve(problem, end_time=100.0, frames=4, time_step=0.01).balance()
        fine = bt.solve(problem, end_time=100.0, frames=64, time_step=0.01).balance()

        coarse_error = abs(coarse.closure_error / coarse.accumulated)
        fine_error = abs(fine.closure_error / fine.accumulated)
        assert fine_error < coarse_error
        assert fine_error < 1.0e-4

    def test_single_frame_declines_to_invent_a_check(self):
        report = bt.solve(_reacting_problem(), end_time=10.0).balance()
        assert report.residual is None
        assert report.closure_error is None
        assert "save at least three frames" in str(report)

    def _square_report(self, cells: int):
        mesh = bt.mesh_2d(cells, cells, 0.0, 1.0, 0.0, 1.0)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-3)
            .linear_decay(0.5)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 1.0)
            .dirichlet("bottom", 1.0)
            .dirichlet("top", 1.0)
        )
        return bt.solve_steady(problem).balance()

    def test_2d_balance_covers_four_sides(self):
        report = self._square_report(16)
        assert set(report.by_side) == {"left", "right", "bottom", "top"}
        assert report.steady is True

    def test_2d_balance_closes_at_second_order(self):
        """Corner nodes own two walls and one balance, so 2D closure is
        discretization-limited rather than exact. It must still converge."""
        errors = []
        for cells in (8, 16, 32, 64):
            report = self._square_report(cells)
            scale = max(abs(report.entered), abs(report.produced))
            errors.append(abs(report.residual) / scale)

        assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
        # Second order means each halving of dx quarters the residual.
        ratios = [earlier / later for earlier, later in zip(errors, errors[1:])]
        assert ratios[-1] > 3.0
        assert errors[-1] < 5.0e-3

    def test_2d_exact_case_is_exact(self):
        """With a one-dimensional field and sealed sides there is no corner
        ambiguity, so the rates must be exact."""
        height = 0.5
        mesh = bt.mesh_2d(20, 12, 0.0, 1.0, 0.0, height)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-3)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 0.0)
            .sealed("bottom")
            .sealed("top")
        )
        sol = bt.solve(problem, end_time=3000.0)
        expected = 1.0e-3 * height
        assert sol.rate("left") == pytest.approx(-expected, rel=1.0e-9)
        assert sol.rate("right") == pytest.approx(expected, rel=1.0e-9)

    def test_repr_is_informative(self):
        report = bt.solve_steady(_reacting_problem()).balance()
        assert "FluxReport" in repr(report)
        assert "stored" in repr(report)


class TestTwoDimensionalSteady:
    """The sparse analytic Jacobian is what makes these solvable at all."""

    def test_uniform_boundaries_give_a_uniform_field(self):
        mesh = bt.mesh_2d(30, 30, 0.0, 1.0, 0.0, 1.0)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-3)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 1.0)
            .dirichlet("bottom", 1.0)
            .dirichlet("top", 1.0)
        )
        sol = bt.solve_steady(problem)
        np.testing.assert_allclose(sol.c, 1.0, atol=1.0e-12)

    def test_agrees_with_the_transient_it_replaces(self):
        mesh = bt.mesh_2d(30, 30, 0.0, 1.0, 0.0, 1.0)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-2)
            .linear_decay(0.5)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 1.0)
            .dirichlet("bottom", 1.0)
            .dirichlet("top", 1.0)
        )
        steady = bt.solve_steady(problem)
        transient = bt.solve(problem, end_time=400.0)
        np.testing.assert_allclose(steady.c, transient.c, atol=1.0e-9)

    def test_uses_a_sparse_solve(self):
        mesh = bt.mesh_2d(40, 40, 0.0, 1.0, 0.0, 1.0)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-3)
            .michaelis_menten(Vmax=2.0e-4, Km=0.5)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 1.0)
            .dirichlet("bottom", 1.0)
            .dirichlet("top", 1.0)
        )
        sol = bt.solve_steady(problem)
        assert sol.newton.linear_solver == "sparse_direct"
        assert sol.newton.converged
