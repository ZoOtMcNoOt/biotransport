"""Tests for cylindrical and spherical geometry on the canonical path.

The load-bearing check is convergence against the series solutions, because
those references know nothing about the discretization. Everything else
guards the ways a curved metric can be quietly dropped: mass weights, boundary
areas, and solvers that only implement slabs.
"""

from __future__ import annotations

import numpy as np
import pytest

import biotransport as bt


RADIUS = 1.0e-3
DIFFUSIVITY = 1.0e-9


def _decaying_sphere(cells: int):
    """Uniform inside, surface held at zero -- Crank's canonical problem."""
    mesh = bt.mesh_1d(cells, 0.0, RADIUS, "spherical")
    problem = (
        bt.Problem(mesh).diffusivity(DIFFUSIVITY).initial(1.0).dirichlet("right", 0.0)
    )
    return mesh, problem


class TestMeshGeometry:
    def test_default_is_cartesian(self):
        mesh = bt.mesh_1d(10, 0.0, 1.0)
        assert mesh.is_radial() is False
        assert mesh.geometry() == bt.Geometry.CARTESIAN

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("cartesian", bt.Geometry.CARTESIAN),
            ("slab", bt.Geometry.CARTESIAN),
            ("cylindrical", bt.Geometry.CYLINDRICAL),
            ("cylinder", bt.Geometry.CYLINDRICAL),
            ("spherical", bt.Geometry.SPHERICAL),
            ("sphere", bt.Geometry.SPHERICAL),
        ],
    )
    def test_names_resolve(self, name, expected):
        assert bt.mesh_1d(4, 0.0, 1.0, name).geometry() == expected

    def test_geometry_value_is_accepted_directly(self):
        mesh = bt.mesh_1d(4, 0.0, 1.0, bt.Geometry.SPHERICAL)
        assert mesh.is_radial()

    def test_unknown_name_is_rejected(self):
        with pytest.raises(ValueError, match="not a geometry"):
            bt.mesh_1d(4, 0.0, 1.0, "toroidal")

    @pytest.mark.parametrize(
        ("geometry", "measure"),
        [("cartesian", 2.0), ("cylindrical", 2.0), ("spherical", 8.0 / 3.0)],
    )
    def test_control_volumes_sum_to_the_exact_measure(self, geometry, measure):
        """R, R^2/2 and R^3/3 for a radius of 2."""
        mesh = bt.mesh_1d(64, 0.0, 2.0, geometry)
        total = sum(mesh.control_volume(i) for i in range(mesh.nx() + 1))
        expected = measure if geometry != "cylindrical" else 2.0
        assert total == pytest.approx(expected, rel=1.0e-13)

    def test_area_factors(self):
        assert bt.mesh_1d(4, 0.0, 4.0, "cartesian").area_factor(3.0) == 1.0
        assert bt.mesh_1d(4, 0.0, 4.0, "cylindrical").area_factor(3.0) == 3.0
        assert bt.mesh_1d(4, 0.0, 4.0, "spherical").area_factor(3.0) == 9.0

    def test_centre_face_has_no_area(self):
        """This is why symmetry at r = 0 needs no boundary condition."""
        for geometry in ("cylindrical", "spherical"):
            assert bt.mesh_1d(8, 0.0, 1.0, geometry).lower_face_area(0) == 0.0

    def test_negative_inner_radius_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative inner radius"):
            bt.mesh_1d(8, -1.0, 1.0, "spherical")

    def test_an_annulus_is_allowed(self):
        """A cylindrical shell that does not reach the axis is legitimate."""
        mesh = bt.mesh_1d(16, 0.5, 1.0, "cylindrical")
        assert mesh.lower_face_area(0) == pytest.approx(0.5)


class TestAgainstSeriesSolutions:
    """The references here are independent of the discretization."""

    @pytest.mark.parametrize(
        ("geometry", "reference"),
        [("spherical", "sphere"), ("cylindrical", "cylinder")],
    )
    def test_matches_the_series(self, geometry, reference):
        exact = getattr(bt.analytical, reference)
        mesh = bt.mesh_1d(200, 0.0, RADIUS, geometry)
        problem = (
            bt.Problem(mesh)
            .diffusivity(DIFFUSIVITY)
            .initial(1.0)
            .dirichlet("right", 0.0)
        )
        sol = bt.solve(problem, end_time=100.0)
        report = sol.compare(
            lambda r, t: exact(
                r, t, D=DIFFUSIVITY, R=RADIUS, c_surface=0.0, c_initial=1.0
            )
        )
        assert report.max_abs < 5.0e-5

    @pytest.mark.parametrize("geometry", ["spherical", "cylindrical"])
    def test_converges_at_second_order(self, geometry):
        exact = getattr(
            bt.analytical, "sphere" if geometry == "spherical" else "cylinder"
        )
        errors = []
        for cells in (100, 200, 400):
            mesh = bt.mesh_1d(cells, 0.0, RADIUS, geometry)
            problem = (
                bt.Problem(mesh)
                .diffusivity(DIFFUSIVITY)
                .initial(1.0)
                .dirichlet("right", 0.0)
            )
            sol = bt.solve(problem, end_time=100.0)
            reference = exact(
                sol.x, 100.0, D=DIFFUSIVITY, R=RADIUS, c_surface=0.0, c_initial=1.0
            )
            errors.append(float(np.max(np.abs(sol.concentration - reference))))

        orders = [
            np.log2(earlier / later) for earlier, later in zip(errors, errors[1:])
        ]
        assert all(1.8 < order < 2.2 for order in orders), orders


class TestConservationAndWeights:
    def test_mass_weights_are_shell_measures(self):
        mesh, _problem = _decaying_sphere(200)
        sealed = (
            bt.Problem(mesh)
            .diffusivity(DIFFUSIVITY)
            .initial(1.0)
            .sealed("left")
            .sealed("right")
        )
        sol = bt.solve(sealed, end_time=10.0)
        # A uniform field of 1 integrates to the sphere measure R^3/3.
        assert sol.total() == pytest.approx(RADIUS**3 / 3.0, rel=1.0e-12)

    def test_sealed_sphere_conserves(self):
        mesh = bt.mesh_1d(150, 0.0, RADIUS, "spherical")
        problem = (
            bt.Problem(mesh)
            .diffusivity(DIFFUSIVITY)
            .initial(bt.gaussian(mesh, center=0.0, width=0.3 * RADIUS))
            .sealed("left")
            .sealed("right")
        )
        sol = bt.solve(problem, end_time=20.0, frames=5)
        totals = [sol.total(t) for t in sol.times]
        assert (max(totals) - min(totals)) / totals[0] < 1.0e-12

    def test_curved_operator_annihilates_a_constant(self):
        for geometry in ("cylindrical", "spherical"):
            mesh = bt.mesh_1d(40, 0.0, 1.0, geometry)
            problem = (
                bt.Problem(mesh)
                .diffusivity(1.0e-3)
                .initial(3.0)
                .dirichlet("right", 3.0)
            )
            sol = bt.solve(problem, end_time=50.0)
            np.testing.assert_allclose(sol.c, 3.0, atol=1.0e-12)


class TestFluxesAreAreaWeighted:
    def test_centre_carries_no_flux(self):
        mesh, problem = _decaying_sphere(100)
        sol = bt.solve(problem, end_time=50.0)
        assert sol.flux_at("left") == pytest.approx(0.0, abs=1.0e-30)

    def test_surface_rate_uses_the_face_area(self):
        """A rate is a flux times an area, and the sphere's area factor is R^2."""
        mesh, problem = _decaying_sphere(200)
        sol = bt.solve(problem, end_time=50.0)
        flux = sol.flux_at("right")
        assert sol.rate("right") == pytest.approx(flux * RADIUS**2, rel=1.0e-12)

    def test_transient_radial_balance_closes(self):
        """A settled radial transient must balance influx against consumption."""
        mesh = bt.mesh_1d(200, 0.0, RADIUS, "spherical")
        problem = (
            bt.Problem(mesh)
            .diffusivity(DIFFUSIVITY)
            .linear_decay(1.0e-3)
            .initial(1.0)
            .dirichlet("right", 1.0)
        )
        sol = bt.solve(problem, end_time=20.0 * RADIUS**2 / DIFFUSIVITY)
        entering = -sol.rate("right")
        assert entering == pytest.approx(-sol.uptake(), rel=1.0e-6)

    def test_steady_radial_balance_closes(self):
        """Direct equilibration must balance influx against consumption."""
        mesh = bt.mesh_1d(200, 0.0, RADIUS, "spherical")
        problem = (
            bt.Problem(mesh)
            .diffusivity(DIFFUSIVITY)
            .linear_decay(1.0e-3)
            .initial(1.0)
            .dirichlet("right", 1.0)
        )
        sol = bt.solve_steady(problem)
        entering = -sol.rate("right")
        assert entering == pytest.approx(-sol.uptake(), rel=1.0e-10)


class TestAxisymmetric:
    """A 2D (r, z) mesh: radial weights in x, plain Cartesian in z."""

    HEIGHT = 4.0e-4

    def _ring(self, radial_cells=60, axial_cells=7):
        return bt.mesh_2d(
            radial_cells, axial_cells, 0.0, RADIUS, 0.0, self.HEIGHT, "axisymmetric"
        )

    def test_names_resolve(self):
        for name in ("axisymmetric", "rz", "cylindrical"):
            mesh = bt.mesh_2d(4, 4, 0.0, 1.0, 0.0, 1.0, name)
            assert mesh.is_radial()

    def test_two_dimensional_spherical_is_refused(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            bt.mesh_2d(4, 4, 0.0, 1.0, 0.0, 1.0, "spherical")

    def test_reduces_to_the_one_dimensional_radial_problem(self):
        """Nothing drives z, so the answer must be the 1D radial one.

        Both runs are pinned to the same step: left to choose, the 2D mesh takes
        a smaller one for its extra axial constraint, and first-order time
        integration would separate the answers by O(dt).
        """
        step = 0.01
        line = bt.mesh_1d(60, 0.0, RADIUS, "cylindrical")
        flat = (
            bt.Problem(line)
            .diffusivity(DIFFUSIVITY)
            .initial(1.0)
            .dirichlet("right", 0.0)
        )
        one_d = bt.solve(flat, end_time=60.0, time_step=step)

        ring = self._ring()
        wedge = (
            bt.Problem(ring)
            .diffusivity(DIFFUSIVITY)
            .initial(1.0)
            .dirichlet("right", 0.0)
            .sealed("bottom")
            .sealed("top")
        )
        two_d = bt.solve(wedge, end_time=60.0, time_step=step)

        expected = np.broadcast_to(one_d.c, two_d.c.shape)
        np.testing.assert_allclose(two_d.c, expected, atol=1.0e-14)

    def test_volume_is_the_true_annulus(self):
        ring = self._ring()
        problem = (
            bt.Problem(ring)
            .diffusivity(DIFFUSIVITY)
            .initial(1.0)
            .sealed("left")
            .sealed("right")
            .sealed("bottom")
            .sealed("top")
        )
        sol = bt.solve(problem, end_time=1.0)
        assert sol.total() == pytest.approx(RADIUS**2 / 2.0 * self.HEIGHT, rel=1.0e-12)

    def test_sealed_domain_conserves(self):
        ring = self._ring()
        problem = (
            bt.Problem(ring)
            .diffusivity(DIFFUSIVITY)
            .initial(
                bt.gaussian(
                    ring, center_x=0.0, center_y=self.HEIGHT / 2, width=0.3 * RADIUS
                )
            )
            .sealed("left")
            .sealed("right")
            .sealed("bottom")
            .sealed("top")
        )
        sol = bt.solve(problem, end_time=20.0, frames=5)
        totals = [sol.total(t) for t in sol.times]
        assert (max(totals) - min(totals)) / totals[0] < 1.0e-12

    def test_axis_carries_no_flux(self):
        ring = self._ring()
        problem = (
            bt.Problem(ring)
            .diffusivity(DIFFUSIVITY)
            .initial(1.0)
            .dirichlet("right", 0.0)
            .sealed("bottom")
            .sealed("top")
        )
        sol = bt.solve(problem, end_time=20.0)
        assert sol.rate("left") == pytest.approx(0.0, abs=1.0e-30)

    def test_outer_rate_uses_the_annulus_area(self):
        """The wall area per unit angle is R times the axial height."""
        ring = self._ring()
        problem = (
            bt.Problem(ring)
            .diffusivity(DIFFUSIVITY)
            .initial(1.0)
            .dirichlet("right", 0.0)
            .sealed("bottom")
            .sealed("top")
        )
        sol = bt.solve(problem, end_time=20.0)
        flux = sol.flux_at("right")
        # Uniform along z here, so the integral is flux * R * H.
        assert sol.rate("right") == pytest.approx(
            float(flux[0]) * RADIUS * self.HEIGHT, rel=1.0e-9
        )

    def test_describe_names_the_axes(self):
        problem = bt.Problem(self._ring(8, 4)).diffusivity(DIFFUSIVITY).initial(1.0)
        assert "axisymmetric" in problem.describe()

    def test_cartesian_2d_describe_is_unchanged(self):
        problem = bt.Problem(bt.mesh_2d(8, 4)).diffusivity(DIFFUSIVITY).initial(1.0)
        assert "axisymmetric" not in problem.describe()

    def test_cartesian_2d_rates_are_unchanged(self):
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
        assert sol.rate("left") == pytest.approx(-1.0e-3 * height, rel=1.0e-9)


class TestSlabOnlySolversRefuse:
    """A curved mesh must never be silently treated as a slab."""

    def test_crank_nicolson_refuses(self):
        mesh = bt.mesh_1d(20, 0.0, 1.0, "spherical")
        with pytest.raises(ValueError, match="slab geometry only"):
            bt.CrankNicolsonDiffusion(mesh, 1.0e-9)

    def test_explicit_diffusion_solver_refuses(self):
        mesh = bt.mesh_1d(20, 0.0, 1.0, "cylindrical")
        with pytest.raises(ValueError, match="slab geometry only"):
            bt.DiffusionSolver(mesh, 1.0e-9)

    def test_steady_solver_uses_spherical_diffusion(self):
        """The radial Newton operator agrees with the spherical exact profile."""
        mesh = bt.mesh_1d(50, 0.0, RADIUS, "spherical")
        problem = (
            bt.Problem(mesh)
            .diffusivity(DIFFUSIVITY)
            .linear_decay(1.0e-3)
            .initial(1.0)
            .dirichlet("right", 1.0)
        )
        sol = bt.solve_steady(problem)
        rho = sol.x / RADIUS
        exact = np.ones_like(rho) / np.sinh(1.0)
        exact[1:] = np.sinh(rho[1:]) / (rho[1:] * np.sinh(1.0))
        np.testing.assert_allclose(sol.c, exact, atol=2e-5, rtol=0)
        assert sol.newton.converged

    def test_cartesian_meshes_are_unaffected(self):
        mesh = bt.mesh_1d(20, 0.0, 1.0)
        assert bt.CrankNicolsonDiffusion(mesh, 1.0e-9) is not None
        assert bt.DiffusionSolver(mesh, 1.0e-9) is not None
        problem = (
            bt.Problem(bt.mesh_1d(20, 0.0, 1.0))
            .diffusivity(1.0e-3)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 0.0)
        )
        assert bt.solve_steady(problem).newton.converged


class TestReporting:
    def test_describe_names_the_geometry(self):
        mesh = bt.mesh_1d(20, 0.0, RADIUS, "spherical")
        problem = bt.Problem(mesh).diffusivity(DIFFUSIVITY).initial(1.0)
        assert "spherical" in problem.describe()

    def test_cartesian_describe_is_unchanged(self):
        problem = bt.Problem(bt.mesh_1d(20)).diffusivity(1.0e-3).initial(1.0)
        assert "spherical" not in problem.describe()
        assert "cylindrical" not in problem.describe()
