"""Tests for the Solution object returned by solve()."""

from __future__ import annotations

import math

import numpy as np
import pytest

import biotransport as bt


matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


def _pulse_problem(cells: int = 40):
    mesh = bt.mesh_1d(cells, 0.0, 1.0)
    problem = (
        bt.Problem(mesh)
        .diffusivity(1.0e-2)
        .initial(bt.gaussian(mesh, center=0.5, width=0.1))
        .sealed("left")
        .sealed("right")
    )
    return mesh, problem


class TestNativeCompatibility:
    """Solution must stand in for the native TransportResult."""

    def test_exposes_native_result_attributes(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.05)

        assert isinstance(sol, bt.Solution)
        assert sol.time == pytest.approx(0.05)
        assert sol.concentration.shape == (41,)
        np.testing.assert_allclose(sol.solution, sol.concentration)
        assert sol.diagnostics.steps > 0

    def test_concentration_is_a_writable_copy(self):
        """Matching native semantics: callers may scribble on what they get."""
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.05)

        first = sol.concentration
        first[0] = 12345.0
        assert sol.concentration[0] != 12345.0

    def test_t_and_dt_aliases_still_work(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, t=0.02, dt=1.0e-3)
        assert sol.time == pytest.approx(0.02)


class TestGeometryAwareness:
    def test_carries_its_own_mesh_and_coordinates(self):
        mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.01)

        # pybind11 hands back a fresh Python wrapper for the same C++ mesh on
        # every call, so compare what the mesh *is* rather than object identity.
        assert sol.mesh.nx() == mesh.nx()
        assert sol.mesh.num_nodes() == mesh.num_nodes()
        assert sol.is_1d
        np.testing.assert_allclose(sol.x, bt.x_nodes(mesh))

    def test_2d_fields_come_back_shaped(self):
        mesh = bt.mesh_2d(8, 5, 0.0, 1.0, 0.0, 2.0)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-3)
            .initial(bt.circle(mesh, 0.5, 1.0, 0.3))
            .dirichlet("left", 0.0)
        )
        sol = bt.solve(problem, end_time=0.01)

        assert sol.c.shape == (mesh.ny() + 1, mesh.nx() + 1)
        assert sol.concentration.shape == (mesh.num_nodes(),)
        assert not sol.is_1d

    def test_initial_frame_is_recorded(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.01)

        assert sol.times[0] == 0.0
        assert float(np.max(sol.c0)) == pytest.approx(1.0)


class TestSavedFrames:
    def test_save_every_lands_on_requested_times(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.10, save_every=0.02)

        assert sol.times == pytest.approx((0.0, 0.02, 0.04, 0.06, 0.08, 0.10))
        assert len(sol) == 6
        assert sol.history.shape == (6, 41)

    def test_save_at_sorts_and_ends_exactly(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.08, save_at=[0.04, 0.01])

        assert sol.times == pytest.approx((0.0, 0.01, 0.04, 0.08))

    def test_frames_gives_evenly_spaced_snapshots(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.10, frames=4)

        assert sol.times == pytest.approx((0.0, 0.025, 0.05, 0.075, 0.10))

    def test_at_snaps_to_the_nearest_saved_time(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.10, save_every=0.05)

        assert sol.nearest_time(0.049) == pytest.approx(0.05)
        np.testing.assert_allclose(sol.at(0.049), sol.at(0.05))
        np.testing.assert_allclose(sol[0.10], sol.c)

    def test_saving_does_not_disturb_the_problem(self):
        """A saved run must leave the problem exactly as it was configured."""
        mesh, problem = _pulse_problem()
        before = np.asarray(problem.initial()).copy()

        bt.solve(problem, end_time=0.05, save_every=0.01)

        np.testing.assert_allclose(np.asarray(problem.initial()), before)
        assert problem.has_reaction() is False

    def test_saved_and_one_shot_agree_with_a_matched_step(self):
        _mesh, problem = _pulse_problem()
        step = 1.0e-4
        one_shot = bt.solve(problem, end_time=0.05, time_step=step)
        segmented = bt.solve(problem, end_time=0.05, time_step=step, save_every=0.01)

        np.testing.assert_allclose(segmented.c, one_shot.c, rtol=1.0e-10, atol=1.0e-12)

    def test_conflicting_save_options_are_rejected(self):
        _mesh, problem = _pulse_problem()
        with pytest.raises(TypeError, match="only one of"):
            bt.solve(problem, end_time=0.1, save_every=0.01, frames=5)

    def test_save_at_beyond_end_time_is_rejected(self):
        _mesh, problem = _pulse_problem()
        with pytest.raises(ValueError, match="beyond end_time"):
            bt.solve(problem, end_time=0.05, save_at=[0.01, 0.9])


class TestCustomReactionClock:
    """Segmented runs must not restart a time-dependent reaction's clock."""

    def test_time_dependent_reaction_sees_absolute_time(self):
        mesh = bt.mesh_1d(20)
        problem = bt.Problem(mesh).diffusivity(1.0e-4).initial(0.0)
        # Adds material only while t < 0.5, so the total added is 0.5 either way.
        problem.reaction(lambda c, x, y, t: 1.0 if t < 0.5 else 0.0, max_abs_dc=0.0)

        one_shot = bt.solve(problem, end_time=1.0, time_step=1.0e-3)
        segmented = bt.solve(problem, end_time=1.0, time_step=1.0e-3, save_every=0.1)

        assert one_shot.mean() == pytest.approx(0.5, abs=1.0e-9)
        assert segmented.mean() == pytest.approx(one_shot.mean(), abs=1.0e-9)

    def test_composed_reactions_survive_segmentation(self):
        mesh = bt.mesh_1d(20)
        problem = bt.Problem(mesh).diffusivity(1.0e-3).initial(1.0).linear_decay(1.0)
        problem.add_constant_source(0.5)

        one_shot = bt.solve(problem, end_time=0.5, time_step=1.0e-4)
        segmented = bt.solve(problem, end_time=0.5, time_step=1.0e-4, save_every=0.1)

        assert segmented.mean() == pytest.approx(one_shot.mean(), rel=1.0e-12)
        assert [term.kind for term in problem._recipe.reactions] == [
            "linear_decay",
            "constant_source",
        ]


class TestReductions:
    def test_weights_sum_to_the_domain_length(self):
        mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.01)
        assert float(np.sum(sol.weights)) == pytest.approx(1.0)

    def test_total_matches_the_native_mass_diagnostic(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.02)
        assert sol.total() == pytest.approx(sol.diagnostics.final_mass, rel=1.0e-12)

    def test_sealed_domain_conserves_its_total(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.05, save_every=0.01)
        totals = [sol.total(t) for t in sol.times]
        assert max(totals) - min(totals) < 1.0e-12

    def test_trace_follows_a_point_through_time(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.05, save_every=0.01)

        history = sol.trace(at=0.5)
        assert history.shape == (len(sol.times),)
        # The peak of a spreading sealed pulse only decays.
        assert all(a >= b for a, b in zip(history, history[1:]))

    def test_trace_needs_exactly_one_locator(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.01)
        with pytest.raises(TypeError, match="exactly one"):
            sol.trace(index=0, at=0.5)


class TestCompare:
    def test_exact_decay_is_recovered(self):
        mesh = bt.mesh_1d(30)
        rate = 3.0
        problem = bt.Problem(mesh).diffusivity(0.0).linear_decay(rate).initial(1.0)
        sol = bt.solve(problem, end_time=0.2, time_step=1.0e-5)

        report = sol.compare(lambda x, t: math.exp(-rate * t))
        assert report.max_abs < 1.0e-4
        assert report.n == 31
        assert "largest absolute error" in str(report)

    def test_relative_errors_are_undefined_for_a_flat_reference(self):
        mesh = bt.mesh_1d(10)
        problem = bt.Problem(mesh).diffusivity(1.0e-3).initial(1.0)
        sol = bt.solve(problem, end_time=0.01)

        report = sol.compare(np.ones(11))
        assert report.rel_l2 is None
        assert "constant" in str(report)

    def test_accepts_an_array_reference(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.01)
        report = sol.compare(sol.concentration)
        assert report.max_abs == 0.0

    def test_wrong_sized_reference_is_rejected(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.01)
        with pytest.raises(ValueError, match="has 3 values"):
            sol.compare(np.zeros(3))

    def test_l2_uses_control_volume_weights(self):
        """A uniform error of e must give an L2 of exactly e."""
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.01)
        offset = 0.25
        report = sol.compare(sol.concentration - offset)
        assert report.l2 == pytest.approx(offset)
        assert report.max_abs == pytest.approx(offset)


class TestPlotting:
    def test_plot_returns_axes_and_does_not_show(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.01)
        ax = sol.plot(label="run")
        assert ax.get_legend() is not None
        assert ax.figure is not None

    def test_overlaying_times_labels_each_curve(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.04, save_every=0.01)
        ax = sol.plot(times=[0.0, 0.02, 0.04])
        assert len(ax.lines) == 3
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "t = 0",
            "t = 0.02",
            "t = 0.04",
        ]

    def test_2d_kinds(self):
        mesh = bt.mesh_2d(6, 6)
        problem = bt.Problem(mesh).diffusivity(1.0e-3).initial(bt.circle(mesh))
        sol = bt.solve(problem, end_time=0.01)
        for kind in ("contour", "heatmap", "surface"):
            assert sol.plot(kind=kind) is not None

    def test_1d_rejects_a_2d_kind(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.01)
        with pytest.raises(ValueError, match="cannot be drawn"):
            sol.plot(kind="contour")

    def test_animate_needs_more_than_one_frame(self):
        """A zero-length run has only the initial frame, so there is nothing to animate."""
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.0)
        assert len(sol) == 1
        with pytest.raises(ValueError, match="more than one saved frame"):
            sol.animate()

    def test_a_plain_solve_keeps_the_initial_and_final_frames(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.01)
        assert len(sol) == 2

    def test_animate_builds_from_saved_frames(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.04, save_every=0.01)
        animation = sol.animate()
        assert animation is not None
        # Draw once so Matplotlib does not warn about an unrendered animation.
        animation._init_draw()


class TestReporting:
    def test_summary_mentions_grid_stability_and_conservation(self):
        _mesh, problem = _pulse_problem()
        text = bt.solve(problem, end_time=0.02).summary()

        for expected in ("grid", "step size", "stability", "conservation", "range"):
            assert expected in text

    def test_summary_reports_dimensionless_numbers(self):
        mesh = bt.mesh_1d(40)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0e-2)
            .velocity(0.15)
            .linear_decay(0.2)
            .initial(1.0)
        )
        sol = bt.solve(problem, end_time=0.05)
        numbers = sol.dimensionless()

        assert "Peclet" in numbers
        assert "grid Peclet" in numbers
        assert "Damkohler" in numbers
        assert numbers["Peclet"][0] == pytest.approx(0.15 * 1.0 / 1.0e-2)
        assert "Dimensionless numbers" in sol.summary()

    def test_repr_and_html_are_informative(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.02)

        assert "Solution" in repr(sol)
        html = sol._repr_html_()
        assert "biotransport.Solution" in html
        assert "final time" in html

    def test_solution_is_read_only(self):
        _mesh, problem = _pulse_problem()
        sol = bt.solve(problem, end_time=0.01)
        with pytest.raises(AttributeError):
            sol.times = ()


class TestTeachingErrors:
    def test_unstable_step_explains_which_process_binds(self):
        mesh = bt.mesh_1d(100, 0.0, 1.0e-4)
        problem = bt.Problem(mesh).diffusivity(2.0e-9).initial(1.0)

        with pytest.raises(ValueError) as caught:
            bt.solve(problem, end_time=1.0, time_step=1.0)

        message = str(caught.value)
        assert "you asked for dt = 1" in message
        assert "diffusion" in message
        assert "<-- smallest" in message
        assert "solve_steady" in message

    def test_advection_limited_step_names_advection(self):
        mesh = bt.mesh_1d(100, 0.0, 1.0)
        problem = bt.Problem(mesh).diffusivity(1.0e-4).velocity(2.0).initial(1.0)
        with pytest.raises(ValueError) as caught:
            bt.solve(problem, end_time=1.0, time_step=0.5)

        message = str(caught.value)
        assert "advection" in message
        assert "<-- smallest" in message

    def test_unverified_method_names_the_alternatives(self):
        _mesh, problem = _pulse_problem()
        with pytest.raises(ValueError, match="CrankNicolsonDiffusion"):
            bt.solve(problem, end_time=0.01, method="crank_nicolson")
