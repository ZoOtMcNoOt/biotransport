"""Independent exact profiles and balances for the common 1D steady operator."""

import numpy as np
import pytest
from scipy.special import i0

import biotransport as bt


@pytest.mark.parametrize(
    "geometry, dimension", [("cartesian", 1), ("cylindrical", 2), ("spherical", 3)]
)
@pytest.mark.parametrize("cells", [1, 7, 40])
def test_uniform_production_reproduces_quadratic_and_integral_balance(
    geometry, dimension, cells
):
    radius, diffusivity, source, surface = 0.3, 0.4, 1.2, 0.7
    mesh = bt.mesh_1d(cells, 0.0, radius, geometry)
    problem = (
        bt.Problem(mesh)
        .diffusivity(diffusivity)
        .constant_source(source)
        .initial(0.0)
        .sealed("left")
        .dirichlet("right", surface)
    )
    sol = bt.solve_steady(problem)
    exact = surface + source * (radius**2 - sol.x**2) / (2 * dimension * diffusivity)
    np.testing.assert_allclose(sol.c, exact, atol=2e-13, rtol=0)
    # Integrate the source independently: integral_0^R S r^(d-1) dr.
    expected_outflow = source * radius**dimension / dimension
    assert sol.rate("right") == pytest.approx(expected_outflow, rel=2e-11)
    assert abs(sol.balance().residual) < 2e-11 * expected_outflow


@pytest.mark.parametrize("geometry", ["cartesian", "cylindrical", "spherical"])
def test_linear_consumption_has_second_order_refinement_and_closes_balance(geometry):
    errors = []
    for cells in (20, 40, 80):
        mesh = bt.mesh_1d(cells, 0.0, 1.0, geometry)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0)
            .linear_decay(1.0)
            .initial(0.0)
            .sealed("left")
            .dirichlet("right", 1.0)
        )
        sol = bt.solve_steady(problem)
        if geometry == "cartesian":
            exact = np.cosh(sol.x) / np.cosh(1.0)
        elif geometry == "cylindrical":
            exact = i0(sol.x) / i0(1.0)
        else:
            exact = np.ones_like(sol.x) / np.sinh(1.0)
            exact[1:] = np.sinh(sol.x[1:]) / (sol.x[1:] * np.sinh(1.0))
        errors.append(np.max(np.abs(sol.c - exact)))
        assert abs(sol.balance().residual) < 2e-11 * abs(sol.uptake())
        assert sol.newton.linear_solver == "sparse_direct"
    assert 3.8 < errors[0] / errors[1] < 4.3
    assert 3.8 < errors[1] / errors[2] < 4.3
    assert errors[-1] < 1.2e-5


@pytest.mark.parametrize("geometry, dimension", [("cylindrical", 2), ("spherical", 3)])
@pytest.mark.parametrize("neumann_side", ["left", "right"])
def test_annulus_outward_gradient_with_source_matches_quadratic(
    geometry, dimension, neumann_side
):
    inner, outer, diffusivity, source = 0.2, 0.7, 0.3, 0.9
    mesh = bt.mesh_1d(30, inner, outer, geometry)
    problem = (
        bt.Problem(mesh).diffusivity(diffusivity).constant_source(source).initial(0.0)
    )

    def profile(r):
        return 2.0 - source * r**2 / (2 * dimension * diffusivity)

    if neumann_side == "left":
        problem.neumann("left", source * inner / (dimension * diffusivity))
        problem.dirichlet("right", profile(outer))
    else:
        problem.dirichlet("left", profile(inner))
        problem.neumann("right", -source * outer / (dimension * diffusivity))
    sol = bt.solve_steady(problem)
    np.testing.assert_allclose(sol.c, profile(sol.x), atol=3e-12, rtol=0)
    assert abs(sol.balance().residual) < 2e-11 * abs(sol.uptake())


@pytest.mark.parametrize("geometry, power", [("cylindrical", 1), ("spherical", 2)])
@pytest.mark.parametrize("variable_diffusivity", [False, True])
def test_annulus_exact_resistance_refines_and_face_transfer_is_constant(
    geometry, power, variable_diffusivity
):
    inner, outer = 0.3, 1.1
    errors = []
    for cells in (30, 60, 120):
        mesh = bt.mesh_1d(cells, inner, outer, geometry)
        r = bt.x_nodes(mesh)
        diffusivity = r if variable_diffusivity else np.ones_like(r)
        problem = (
            bt.Problem(mesh)
            .diffusivity_field(diffusivity)
            .initial(0.0)
            .dirichlet("left", 1.0)
            .dirichlet("right", 2.0)
        )
        sol = bt.solve_steady(problem)
        # A(r) D(r) c'(r) is constant. Its primitive is log(r) when
        # A*D=r, and -r^(1-p)/(p-1) for total power p>1.
        total_power = power + int(variable_diffusivity)
        if total_power == 1:
            primitive = np.log(r / inner)
            span = np.log(outer / inner)
        else:
            primitive = r ** (1 - total_power) - inner ** (1 - total_power)
            span = outer ** (1 - total_power) - inner ** (1 - total_power)
        exact = 1.0 + primitive / span
        errors.append(np.max(np.abs(sol.c - exact)))
        face_d = 2.0 / (1.0 / diffusivity[:-1] + 1.0 / diffusivity[1:])
        face_r = (r[:-1] + r[1:]) / 2
        transfer = -(face_r**power) * face_d * np.diff(sol.c) / mesh.dx()
        np.testing.assert_allclose(transfer, transfer[0], rtol=2e-11, atol=0)
        assert abs(sol.balance().residual) < 2e-11 * abs(sol.rate("right"))
    if geometry == "cylindrical" and variable_diffusivity:
        # Harmonic D for D(r)=r makes A_face*D_face=r_i*r_(i+1),
        # so the inverse-radius profile has exactly constant discrete flux.
        assert max(errors) < 2e-12
    else:
        assert 3.8 < errors[0] / errors[1] < 4.2
        assert 3.8 < errors[1] / errors[2] < 4.2
    assert errors[-1] < 5e-5


@pytest.mark.parametrize("geometry", ["cylindrical", "spherical"])
@pytest.mark.parametrize(
    "diffusivity, concentration, radius",
    [(1e-20, 1e-20, 1.0), (1e20, 1e20, 1.0), (2e-9, 0.05, 1e-4), (1e-6, 1e-12, 1e3)],
)
def test_radial_profiles_preserve_equivalent_physical_units(
    geometry, diffusivity, concentration, radius
):
    mesh = bt.mesh_1d(80, 0.0, radius, geometry)
    problem = (
        bt.Problem(mesh)
        .diffusivity(diffusivity)
        .linear_decay(diffusivity / radius**2)
        .initial(0.0)
        .dirichlet("right", concentration)
    )
    sol = bt.solve_steady(problem)
    rho = sol.x / radius
    if geometry == "cylindrical":
        exact = i0(rho) / i0(1.0)
    else:
        exact = np.ones_like(rho) / np.sinh(1.0)
        exact[1:] = np.sinh(rho[1:]) / (rho[1:] * np.sinh(1.0))
    np.testing.assert_allclose(sol.c / concentration, exact, atol=1.2e-5, rtol=0)
    assert abs(sol.balance().residual) < 5e-11 * abs(sol.uptake())


@pytest.mark.parametrize("geometry", ["cartesian", "cylindrical", "spherical"])
def test_nonlinear_boundary_balance_jacobian_matches_directional_derivative(geometry):
    mesh = bt.mesh_1d(20, 0.3, 1.0, geometry)
    r = bt.x_nodes(mesh)
    solver = bt.NonlinearDiffusionSolver(mesh, D=np.where(r < 0.6, 0.2, 0.9))
    solver.set_boundary(bt.Boundary.Left, 0.3, "neumann")
    solver.set_boundary(bt.Boundary.Right, -0.1, "neumann")
    solver.set_reaction(lambda u: u**3, lambda u: 3 * u**2)
    u, direction = 0.4 + r / 3, np.cos(2 * r)
    delta = 1e-5
    expected = (
        solver._residual_1d(u + delta * direction)
        - solver._residual_1d(u - delta * direction)
    ) / (2 * delta)
    matrix = solver._jacobian_1d(u)
    assert matrix.nnz <= 3 * u.size
    np.testing.assert_allclose(matrix @ direction, expected, rtol=2e-7, atol=1e-8)


@pytest.mark.parametrize("geometry", ["cylindrical", "spherical"])
def test_pure_neumann_radial_problem_preserves_singularity_contract(geometry):
    solver = bt.NonlinearDiffusionSolver(bt.mesh_1d(31, 0.0, 1.0, geometry), D=1.0)
    solver.set_boundary(bt.Boundary.Right, 0.0, "neumann")
    with pytest.raises(bt.NewtonLinearSolveError, match="singular"):
        solver.solve(np.linspace(0.0, 1.0, 32))


@pytest.mark.parametrize("geometry", ["cylindrical", "spherical"])
def test_sealed_radial_reaction_source_has_unique_uniform_equilibrium(geometry):
    problem = (
        bt.Problem(bt.mesh_1d(25, 0.0, 1.0, geometry))
        .diffusivity(1.0)
        .linear_decay(0.2)
        .add_constant_source(0.6)
        .initial(0.0)
        .sealed("right")
    )
    sol = bt.solve_steady(problem)
    np.testing.assert_allclose(sol.c, 3.0, atol=1e-11, rtol=0)


@pytest.mark.parametrize("kind, value", [("dirichlet", 1.0), ("neumann", 0.1)])
def test_origin_rejects_conditions_without_a_physical_surface(kind, value):
    solver = bt.NonlinearDiffusionSolver(bt.mesh_1d(10, 0.0, 1.0, "spherical"), D=1.0)
    with pytest.raises(ValueError, match="origin only symmetry"):
        solver.set_boundary(bt.Boundary.Left, value, kind)


def test_axisymmetric_2d_is_rejected_in_both_steady_interfaces():
    mesh = bt.mesh_2d(4, 3, 0.0, 1.0, 0.0, 1.0, "axisymmetric")
    with pytest.raises(ValueError, match="radial geometry"):
        bt.NonlinearDiffusionSolver(mesh, D=1.0)
    with pytest.raises(ValueError, match="radial geometry in 1D only"):
        bt.solve_steady(bt.Problem(mesh).diffusivity(1.0).initial(0.0))


@pytest.mark.parametrize("geometry", ["cartesian", "cylindrical", "spherical"])
@pytest.mark.parametrize("guess", [-0.5, -1.0, 0.0])
def test_michaelis_menten_negative_trials_reach_nonnegative_equilibrium(
    geometry, guess
):
    problem = (
        bt.Problem(bt.mesh_1d(40, 0.0, 1.0, geometry))
        .diffusivity(0.1)
        .michaelis_menten(Vmax=0.4, Km=0.5)
        .add_constant_source(0.4 / 1.5)
        .initial(guess)
        .sealed("left")
        .dirichlet("right", 1.0)
    )
    sol = bt.solve_steady(problem)
    np.testing.assert_allclose(sol.c, 1.0, atol=2e-10, rtol=0)
    assert sol.newton.converged
