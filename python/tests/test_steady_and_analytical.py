"""Tests for the steady solver and the exact-solution catalog.

The most valuable checks here are the cross-validations: a steady numerical
solve against a closed-form profile, and a transient solve against a series
solution. Agreement is evidence for both sides at once.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import biotransport as bt


# ---------------------------------------------------------------------------
# Steady solver
# ---------------------------------------------------------------------------


class TestSteadyAgainstExact:
    def test_first_order_reaction_matches_the_cosh_profile(self):
        """-D c'' + k c = 0, fixed at one face, sealed at the other."""
        diffusivity, rate, length, surface = 1.0e-9, 5.0e-4, 1.0e-2, 1.0
        mesh = bt.mesh_1d(300, 0.0, length)
        problem = (
            bt.Problem(mesh)
            .diffusivity(diffusivity)
            .linear_decay(rate)
            .initial(surface)
            .dirichlet("left", surface)
            .sealed("right")
        )

        steady = bt.solve_steady(problem)
        report = steady.compare(
            lambda x: bt.analytical.steady_slab_first_order(
                x, D=diffusivity, k=rate, L=length, c_surface=surface
            )
        )
        assert report.rel_l2 < 1.0e-4

    def test_pure_diffusion_between_fixed_faces_is_linear(self):
        mesh = bt.mesh_1d(50, 0.0, 2.0)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-3)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 3.0)
        )
        steady = bt.solve_steady(problem)
        report = steady.compare(lambda x: 1.0 + x)
        assert report.max_abs < 1.0e-9

    def test_agrees_with_a_long_transient(self):
        diffusivity, length = 2.0e-9, 1.0e-4
        mesh = bt.mesh_1d(100, 0.0, length)
        problem = (
            bt.Problem(mesh)
            .diffusivity(diffusivity)
            .michaelis_menten(Vmax=1.0e-3, Km=1.0e-3)
            .initial(0.05)
            .dirichlet("left", 0.05)
            .sealed("right")
        )

        steady = bt.solve_steady(problem)
        transient = bt.solve(problem, end_time=25.0)

        np.testing.assert_allclose(steady.c, transient.c, atol=1.0e-6)


class TestSteadyResultShape:
    def test_reports_itself_as_steady(self):
        mesh = bt.mesh_1d(20)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-3)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 0.0)
        )
        steady = bt.solve_steady(problem)

        assert steady.steady is True
        assert steady.diagnostics is None
        assert steady.newton.converged
        assert "steady state" in repr(steady)
        assert "Steady solution summary" in steady.summary()
        assert "state" in steady._repr_html_()

    def test_solve_with_steady_flag_matches_solve_steady(self):
        mesh = bt.mesh_1d(20)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-3)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 0.0)
        )
        np.testing.assert_allclose(
            bt.solve(problem, steady=True).c, bt.solve_steady(problem).c
        )

    def test_steady_rejects_an_end_time(self):
        mesh = bt.mesh_1d(20)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-3)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 0.0)
        )
        with pytest.raises(TypeError, match="no end time"):
            bt.solve(problem, end_time=1.0, steady=True)


class TestSteadyRefusals:
    """Each refusal must name the alternative, not just say no."""

    def _problem(self, mesh):
        return (
            bt.Problem(mesh)
            .diffusivity(1.0e-3)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 0.0)
        )

    def test_advection_is_refused_with_a_pointer_to_the_transient(self):
        mesh = bt.mesh_1d(20)
        problem = self._problem(mesh).velocity(0.1)
        with pytest.raises(ValueError, match="not advection"):
            bt.solve_steady(problem)

    def test_custom_reaction_is_refused_because_it_cannot_be_differentiated(self):
        mesh = bt.mesh_1d(20)
        problem = self._problem(mesh)
        problem.reaction(lambda c, x, y, t: -(c**3), max_abs_dc=3.0)
        with pytest.raises(ValueError, match="built-in kinetics"):
            bt.solve_steady(problem)

    def test_robin_boundary_is_refused(self):
        mesh = bt.mesh_1d(20)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-3)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .robin("right", 1.0, 1.0, 0.0)
        )
        with pytest.raises(ValueError, match="Robin"):
            bt.solve_steady(problem)

    def test_moderately_large_2d_grid_now_solves(self):
        """The 2D Jacobian is sparse and analytic, so this is no longer refused."""
        mesh = bt.mesh_2d(60, 60)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-3)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 1.0)
            .dirichlet("bottom", 1.0)
            .dirichlet("top", 1.0)
        )
        steady = bt.solve_steady(problem)
        assert steady.newton.converged
        assert steady.newton.linear_solver == "sparse_direct"
        np.testing.assert_allclose(steady.c, 1.0, atol=1.0e-12)

    def test_inconsistent_corner_says_so_plainly(self):
        """A configuration error must not be buried under a convergence message."""
        mesh = bt.mesh_2d(8, 8)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-3)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 0.0)
            .dirichlet("bottom", 0.0)
            .dirichlet("top", 0.0)
        )
        with pytest.raises(ValueError, match="corner"):
            bt.solve_steady(problem)

    def test_raw_native_problem_is_refused(self):
        mesh = bt.mesh_1d(20)
        native = bt.TransportProblem(mesh)
        native.diffusivity(1.0e-3)
        with pytest.raises(TypeError, match="bt.Problem"):
            bt.solve_steady(native)


# ---------------------------------------------------------------------------
# Analytical catalog
# ---------------------------------------------------------------------------


class TestNativeHelpersStillWork:
    def test_scalar_calls_return_plain_floats(self):
        value = bt.analytical.diffusion_length(1.0e-9, 100.0)
        assert isinstance(value, float)
        assert value == pytest.approx(math.sqrt(1.0e-9 * 100.0))

    def test_the_same_helpers_now_broadcast(self):
        times = np.array([1.0, 10.0, 100.0])
        values = bt.analytical.diffusion_length(1.0e-9, times)
        assert isinstance(values, np.ndarray)
        np.testing.assert_allclose(values, np.sqrt(1.0e-9 * times))

    def test_module_is_the_python_layer(self):
        assert bt.analytical.__file__.endswith("analytical.py")
        assert hasattr(bt.analytical, "slab")
        assert hasattr(bt.analytical, "poiseuille_velocity")


class TestSeriesSolutions:
    def test_slab_matches_a_numerical_solve(self):
        diffusivity, length = 1.0e-9, 1.0e-2
        mesh = bt.mesh_1d(400, 0.0, length)
        problem = (
            bt.Problem(mesh)
            .diffusivity(diffusivity)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 1.0)
        )
        sol = bt.solve(problem, end_time=2000.0)
        report = sol.compare(
            lambda x, t: bt.analytical.slab(
                x, t, D=diffusivity, L=length, c_surface=1.0
            )
        )
        assert report.max_abs < 1.0e-4

    def test_slab_starts_at_the_initial_condition(self):
        values = bt.analytical.slab(
            np.linspace(0.0, 1.0, 11), 0.0, D=1.0, L=1.0, c_surface=1.0, c_initial=0.25
        )
        np.testing.assert_allclose(values, 0.25)

    def test_slab_pins_both_faces_to_the_surface_value(self):
        edges = bt.analytical.slab(
            np.array([0.0, 1.0]), 0.05, D=1.0, L=1.0, c_surface=2.0, c_initial=0.0
        )
        np.testing.assert_allclose(edges, 2.0, atol=1.0e-12)

    def test_slab_warns_when_the_series_cannot_resolve_the_time(self):
        with pytest.warns(RuntimeWarning, match="Fourier number"):
            bt.analytical.slab(0.5, 1.0e-8, D=1.0, L=1.0, c_surface=1.0, terms=5)

    def test_semi_infinite_matches_a_deep_domain_solve(self):
        diffusivity = 1.0e-9
        mesh = bt.mesh_1d(600, 0.0, 0.02)
        problem = (
            bt.Problem(mesh)
            .diffusivity(diffusivity)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .sealed("right")
        )
        sol = bt.solve(problem, end_time=200.0)
        report = sol.compare(
            lambda x, t: bt.analytical.semi_infinite(x, t, D=diffusivity, c_surface=1.0)
        )
        assert report.rel_l2 < 1.0e-3

    @pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
    def test_radial_solutions_decay_and_pin_the_surface(self, geometry):
        solution = getattr(bt.analytical, geometry)
        radius = 1.0e-3
        centre = [
            solution(0.0, t, D=1.0e-9, R=radius, c_surface=0.0, c_initial=1.0)
            for t in (10.0, 100.0, 300.0, 1000.0)
        ]
        assert all(a >= b for a, b in zip(centre, centre[1:]))
        assert centre[0] == pytest.approx(1.0, abs=1.0e-6)
        surface = solution(
            radius, 100.0, D=1.0e-9, R=radius, c_surface=0.0, c_initial=1.0
        )
        assert abs(surface) < 1.0e-9

    def test_radial_solutions_reject_a_radius_outside_the_body(self):
        with pytest.raises(ValueError, match=r"\[0, R\]"):
            bt.analytical.sphere(2.0, 1.0, D=1.0, R=1.0, c_surface=0.0)

    def test_instantaneous_source_conserves_the_amount_released(self):
        positions = np.linspace(-0.01, 0.01, 20001)
        field = bt.analytical.instantaneous_source(
            positions, 100.0, D=1.0e-9, amount=2.5
        )
        assert float(np.trapezoid(field, positions)) == pytest.approx(2.5, rel=1.0e-8)

    def test_instantaneous_source_needs_a_positive_time(self):
        with pytest.raises(ValueError, match="positive"):
            bt.analytical.instantaneous_source(0.0, 0.0, D=1.0)


class TestThieleAndEffectiveness:
    def test_thiele_modulus_definition(self):
        assert bt.analytical.thiele_modulus(D=2.0, k=8.0, length=3.0) == pytest.approx(
            math.sqrt(8.0 / 2.0) * 3.0
        )

    def test_effectiveness_tends_to_one_for_slow_reaction(self):
        assert bt.analytical.effectiveness_factor(1.0e-6, "slab") == pytest.approx(
            1.0, abs=1.0e-9
        )

    def test_slab_effectiveness_is_tanh_over_phi(self):
        phi = 2.0
        assert bt.analytical.effectiveness_factor(phi, "slab") == pytest.approx(
            math.tanh(phi) / phi
        )

    def test_sphere_effectiveness_matches_the_closed_form(self):
        phi = 3.0
        expected = (3.0 / phi**2) * (phi / math.tanh(phi) - 1.0)
        assert bt.analytical.effectiveness_factor(phi, "sphere") == pytest.approx(
            expected
        )

    def test_geometry_ordering_at_a_fixed_modulus(self):
        """For the same modulus, a sphere is more effective than a slab."""
        phi = 5.0
        slab = bt.analytical.effectiveness_factor(phi, "slab")
        cylinder = bt.analytical.effectiveness_factor(phi, "cylinder")
        sphere = bt.analytical.effectiveness_factor(phi, "sphere")
        assert slab < cylinder < sphere

    def test_unknown_geometry_is_rejected(self):
        with pytest.raises(ValueError, match="slab"):
            bt.analytical.effectiveness_factor(1.0, "cube")
