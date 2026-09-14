"""Independent mathematical and ownership checks for coupled transport."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from scipy.optimize import brentq

import biotransport as bt


def two_compartments():
    return (
        bt.CoupledModel(["A", "B"])
        .compartment("left", volume=2.0, initial={"A": 1.2, "B": 0.1})
        .compartment("right", volume=3.0, initial={"A": 0.1, "B": 0.5})
        .membrane(
            "wall",
            "left",
            "right",
            area=0.4,
            permeability={"A": 0.7},
            partition={"A": 2.5},
        )
        .conserve("A total", {"A": 1.0})
    )


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_partition_and_unequal_volumes_against_exact_exchange(method):
    solution = two_compartments().solve(
        40.0, frames=80, method=method, rtol=2e-11, atol=1e-13
    )
    v1, v2, partition, conductance = 2.0, 3.0, 2.5, 0.4 * 0.7
    total = v1 * 1.2 + v2 * 0.1
    equilibrium = total / (v1 + partition * v2)
    decay = np.exp(-conductance * (1 / v1 + 1 / (partition * v2)) * solution.times)
    c1 = equilibrium + (1.2 - equilibrium) * decay
    c2 = (total - v1 * c1) / v2
    assert_allclose(solution.history("left", "A")[:, 0], c1, rtol=2e-9, atol=2e-11)
    assert_allclose(solution.history("right", "A")[:, 0], c2, rtol=2e-9, atol=2e-11)
    assert_allclose(
        solution.interface_rate("wall", "A"),
        conductance * (c1 - c2 / partition),
        atol=2e-10,
    )
    assert_allclose(solution.amount("A"), total, atol=2e-14)
    assert_allclose(solution.field("left", "B"), [0.1])
    assert_allclose(solution.field("right", "B"), [0.5])
    assert solution.balance("A total").relative_drift < 2e-14


def test_equilibrium_is_stationary_even_with_partition_and_stiff_exchange():
    m = (
        bt.CoupledModel(["A"])
        .compartment("a", volume=1e-12, initial={"A": 2})
        .compartment("b", volume=1e-9, initial={"A": 6})
        .membrane("m", "a", "b", area=1e-4, permeability=1e-3, partition=3)
    )
    compiled = m.compile()
    assert_allclose(compiled.rhs(0, compiled.initial_state), 0, atol=1e-10)
    assert_allclose(m.solve(0.1).field("b", "A"), [6], rtol=1e-12)


def test_reaction_chain_and_linear_operator_against_exact_kinetics():
    k1, k2 = 1.3, 0.3
    m = (
        bt.CoupledModel(["A", "B", "C"])
        .compartment("reactor", volume=0.7, initial={"A": 2})
        .mass_action("reactor", reactants={"A": 1}, products={"B": 1}, rate_constant=k1)
        .mass_action("reactor", reactants={"B": 1}, products={"C": 1}, rate_constant=k2)
        .conserve("total", {"A": 1, "B": 1, "C": 1})
    )
    result = m.solve(8, rtol=1e-10, atol=1e-12)
    a = 2 * np.exp(-k1 * result.times)
    b = 2 * k1 / (k2 - k1) * (np.exp(-k1 * result.times) - np.exp(-k2 * result.times))
    assert_allclose(result.history("reactor", "A")[:, 0], a, atol=2e-9)
    assert_allclose(result.history("reactor", "B")[:, 0], b, atol=2e-9)
    assert_allclose(result.history("reactor", "C")[:, 0], 2 - a - b, atol=2e-9)
    assert result.diagnostics.jacobian_evaluations == 0  # constant sparse Jacobian
    assert result.balance("total").relative_drift < 3e-15
    assert m.compile().transport_matrix.nnz == 0


def test_bimolecular_stoichiometry_rate_convention_and_zero_state_jacobian():
    k = 0.6
    m = (
        bt.CoupledModel(["A", "B"])
        .compartment("reactor", volume=2, initial={"A": 1.7})
        .mass_action("reactor", reactants={"A": 2}, products={"B": 1}, rate_constant=k)
        .conserve("A units", {"A": 1, "B": 2})
    )
    result = m.solve(5, rtol=1e-10, atol=1e-12)
    a = 1.7 / (1 + 2 * k * 1.7 * result.times)
    assert_allclose(result.history("reactor", "A")[:, 0], a, rtol=5e-9)
    assert_allclose(result.history("reactor", "B")[:, 0], (1.7 - a) / 2, atol=2e-9)
    assert result.balance("A units").relative_drift < 3e-15
    assert_allclose(m.compile().jacobian(0, np.zeros(2)).toarray(), np.zeros((2, 2)))


def test_reversible_binding_conserves_both_moieties_and_reaches_equilibrium():
    m = (
        bt.CoupledModel(["drug", "site", "bound"])
        .compartment("tissue", volume=1e-6, initial={"drug": 1.0, "site": 2.0})
        .mass_action(
            "tissue",
            reactants={"drug": 1, "site": 1},
            products={"bound": 1},
            rate_constant=3,
        )
        .mass_action(
            "tissue",
            reactants={"bound": 1},
            products={"drug": 1, "site": 1},
            rate_constant=0.4,
        )
        .conserve("drug", {"drug": 1, "bound": 1})
        .conserve("sites", {"site": 1, "bound": 1})
    )
    result = m.solve(20, rtol=1e-10, atol=1e-12)
    bound = float(result.field("tissue", "bound")[0])
    # At equilibrium 3*(1-b)*(2-b) = 0.4*b, the root in [0,1].
    root = ((9 + 0.4) - np.sqrt((9 + 0.4) ** 2 - 72)) / 6
    assert bound == pytest.approx(root, abs=1e-10)
    assert result.balance("drug").relative_drift < 2e-14
    assert result.balance("sites").relative_drift < 2e-14
    assert result.diagnostics.minimum_concentration >= 0.0


@pytest.mark.parametrize(
    "geometry,kwargs,total_volume",
    [
        ("cartesian", {"cross_section": 0.7}, 0.7 * (0.8 - 0.2)),
        ("cylindrical", {"axial_length": 0.7}, math.pi * 0.7 * (0.8**2 - 0.2**2)),
        ("spherical", {}, 4 * math.pi / 3 * (0.8**3 - 0.2**3)),
    ],
)
def test_physical_volumes_conservation_and_transport_signs(
    geometry, kwargs, total_volume
):
    mesh = bt.mesh_1d(17, 0.2, 0.8, geometry=geometry)
    m = bt.CoupledModel(["A", "B"]).domain(
        "tissue",
        mesh,
        **kwargs,
        initial={"A": 2},
        diffusivity={"A": np.linspace(0.1, 2, 18), "B": 0.2},
    )
    compiled = m.compile()
    weights = compiled.state_volumes
    assert sum(weights[:18]) == pytest.approx(total_volume, rel=5e-15)
    matrix = compiled.transport_matrix.toarray()
    residual = np.abs(weights @ matrix)
    rounding_budget = 8 * np.finfo(float).eps * (weights @ np.abs(matrix))
    assert np.all(residual <= rounding_budget)
    assert_allclose(matrix @ np.ones(matrix.shape[0]), 0.0, atol=5e-12)
    assert np.all(np.diag(matrix) <= 0)
    np.fill_diagonal(matrix, 0)
    assert np.all(matrix >= 0)
    assert m.solve(0).amount("A")[0] == pytest.approx(2 * total_volume)


def test_closed_slab_diffusion_second_order_against_continuum_eigenmode():
    errors = []
    for n in [12, 24, 48]:
        mesh = bt.mesh_1d(n)
        x = bt.x_nodes(mesh)
        initial = 1 + 0.5 * np.cos(np.pi * x)
        m = bt.CoupledModel(["A"]).domain(
            "slab",
            mesh,
            cross_section=0.4,
            initial={"A": initial},
            diffusivity={"A": 0.2},
        )
        result = m.solve(0.4, frames=1, rtol=2e-11, atol=1e-12)
        exact = 1 + 0.5 * np.cos(np.pi * x) * np.exp(-0.2 * np.pi**2 * 0.4)
        errors.append(np.max(np.abs(result.field("slab", "A") - exact)))
        assert_allclose(result.amount("A"), 0.4, atol=2e-14)
    assert np.all(np.log2(np.asarray(errors[:-1]) / errors[1:]) > 1.99)


@pytest.mark.parametrize(
    "geometry,kwargs",
    [
        ("cartesian", {"cross_section": 0.3}),
        ("cylindrical", {"axial_length": 0.5}),
        ("spherical", {}),
    ],
)
def test_spatial_diffusion_agrees_with_native_conservative_engine(geometry, kwargs):
    mesh = bt.mesh_1d(12, 0, 1, geometry=geometry)
    initial = 0.2 + np.cos(bt.x_nodes(mesh) * np.pi) ** 2
    m = bt.CoupledModel(["A"]).domain(
        "tissue", mesh, **kwargs, diffusivity={"A": 0.3}, initial={"A": initial}
    )
    result = m.solve(0.1, frames=1, rtol=1e-10, atol=1e-12)
    native = bt.solve(
        bt.Problem(mesh)
        .diffusivity(0.3)
        .initial(initial)
        .sealed("left")
        .sealed("right"),
        end_time=0.1,
        time_step=1e-6,
    )
    assert_allclose(result.field("tissue", "A"), native.c, atol=4e-6)


def test_two_spatial_domains_against_independently_assembled_dense_exponential():
    # Each one-cell slab has two half-volume boundary nodes. The two fields
    # remain separate at the membrane; no shared-node concentration constraint.
    m = (
        bt.CoupledModel(["A"])
        .domain(
            "slab1",
            bt.mesh_1d(1, 0, 2),
            cross_section=3,
            diffusivity={"A": 0.7},
            initial={"A": [2, 1]},
        )
        .domain(
            "slab2",
            bt.mesh_1d(1, 0, 4),
            cross_section=2,
            diffusivity={"A": 0.2},
            initial={"A": [0.5, 0]},
        )
        .membrane(
            "wall",
            ("slab1", "right"),
            ("slab2", "left"),
            area=1.5,
            permeability=0.4,
            partition=3,
        )
    )
    volumes = np.array([3, 3, 4, 4])
    balance = np.zeros((4, 4))
    # Independent amount balance matrix, then divide each row by its volume.
    for a, b, g, k in [
        (0, 1, 3 * 0.7 / 2, 1),
        (1, 2, 1.5 * 0.4, 3),
        (2, 3, 2 * 0.2 / 4, 1),
    ]:
        balance[a, a] -= g
        balance[b, a] += g
        balance[a, b] += g / k
        balance[b, b] -= g / k
    operator = balance / volumes[:, None]
    result = m.solve(10, frames=10, rtol=1e-10, atol=1e-12)
    observed = np.hstack([result.history("slab1", "A"), result.history("slab2", "A")])
    expected = np.array([expm(t * operator) @ [2, 1, 0.5, 0] for t in result.times])
    assert_allclose(observed, expected, atol=1e-9)
    assert_allclose(result.amount("A"), 11, atol=2e-13)


def test_harmonic_diffusivity_and_impermeable_internal_face():
    mesh = bt.mesh_1d(2)
    compiled = (
        bt.CoupledModel(["A"])
        .domain(
            "slab",
            mesh,
            cross_section=2,
            diffusivity={"A": [0.2, 0.8, 0]},
            initial={"A": [2, 1, 3]},
        )
        .compile()
    )
    # D_face = 2*.2*.8/(.2+.8) = .32; dx=.5; area=2; V_left=.5.
    assert_allclose(compiled.rhs(0, compiled.initial_state), [-2.56, 1.28, 0])


def nonlinear_spatial(custom=False):
    mesh = bt.mesh_1d(5)
    model = bt.CoupledModel(["A", "B", "C"]).domain(
        "slab",
        mesh,
        cross_section=0.4,
        initial={"A": 1, "B": 2},
        diffusivity={"A": 0.1, "B": 0.2},
    )
    if custom:
        model.reaction(
            "slab",
            stoichiometry={"A": -1, "C": 1},
            rate=lambda t, c: (1 + t) * c["A"] / (0.3 + c["A"]),
            derivative=lambda t, c: {"A": (1 + t) * 0.3 / (0.3 + c["A"]) ** 2},
        )
    else:
        model.mass_action(
            "slab", reactants={"A": 1, "B": 1}, products={"C": 1}, rate_constant=0.7
        )
    return model


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("state_kind", ["positive", "zero"])
def test_analytic_sparse_jacobian_directional_derivative(custom, state_kind):
    compiled = nonlinear_spatial(custom).compile()
    rng = np.random.default_rng(17)
    state = (
        rng.uniform(0.2, 2, compiled.initial_state.size)
        if state_kind == "positive"
        else np.zeros_like(compiled.initial_state)
    )
    direction = rng.normal(size=state.size)
    eps = 1e-6
    finite_difference = (
        compiled.rhs(0.4, state + eps * direction)
        - compiled.rhs(0.4, state - eps * direction)
    ) / (2 * eps)
    assert_allclose(
        compiled.jacobian(0.4, state) @ direction,
        finite_difference,
        rtol=1e-8,
        atol=2e-8,
    )


def test_time_dependent_custom_reaction_integrates_once_across_output_frames():
    m = bt.CoupledModel(["A", "B"]).compartment("r", volume=1, initial={"A": 2})
    m.reaction(
        "r",
        stoichiometry={"A": -1, "B": 1},
        rate=lambda t, c: t * c["A"],
        derivative=lambda t, c: {"A": t},
    ).conserve("total", {"A": 1, "B": 1})
    many = m.solve(3, frames=60, rtol=1e-10, atol=1e-12)
    few = m.solve(3, frames=1, rtol=1e-10, atol=1e-12)
    assert_allclose(
        many.history("r", "A")[:, 0], 2 * np.exp(-(many.times**2) / 2), atol=3e-9
    )
    assert_allclose(few.field("r", "A"), many.field("r", "A"), rtol=0, atol=0)
    assert few.diagnostics.rhs_evaluations == many.diagnostics.rhs_evaluations
    assert many.balance("total").relative_drift < 1e-14


def test_callback_cannot_mutate_integrator_state_even_if_it_changes_its_copy():
    def rate(t, c):
        c["A"].setflags(write=True)
        c["A"][:] = 999
        return 0.0

    model = bt.CoupledModel(["A"]).compartment("r", volume=1, initial={"A": 2})
    model.reaction("r", stoichiometry={"A": 1}, rate=rate, derivative=lambda t, c: {})
    assert_allclose(model.solve(1).field("r", "A"), [2])


def test_compile_and_results_own_input_arrays_and_ignore_later_builder_edits():
    initial, diffusion = np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3])
    m = bt.CoupledModel(["A"]).domain(
        "slab",
        bt.mesh_1d(2),
        cross_section=2,
        initial={"A": initial},
        diffusivity={"A": diffusion},
    )
    compiled = m.compile()
    result = compiled.solve(0.1)
    before = result.field("slab", "A")
    initial[:] = 999
    diffusion[:] = 0
    compiled.initial_state[:] = 999
    compiled.state_volumes[:] = 999
    compiled.transport_matrix.data[:] = 999
    result.field("slab", "A")[:] = 999
    result.history("slab", "A")[:] = 999
    result.times[:] = 999
    m.mass_action("slab", reactants={"A": 1}, products={}, rate_constant=100)
    m.compartment("new", volume=1)
    assert_allclose(compiled.solve(0.1).field("slab", "A"), before, rtol=0, atol=0)
    assert_allclose(result.field("slab", "A"), before, rtol=0, atol=0)
    assert compiled.domains == ("slab",)
    assert compiled.coordinates("slab").tolist() == [0, 0.5, 1]


def test_invalid_invariant_and_reaction_edits_are_atomic():
    m = bt.CoupledModel(["A", "B"]).compartment("r", volume=1, initial={"A": 1})
    m.mass_action("r", reactants={"A": 1}, products={"B": 1}, rate_constant=1)
    with pytest.raises(ValueError, match="violates"):
        m.conserve("total", {"A": 1})
    m.conserve("total", {"A": 1, "B": 1})
    before = m.compile().rhs(0, np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="violates"):
        m.mass_action("r", reactants={"B": 1}, products={}, rate_constant=1)
    assert_allclose(m.compile().rhs(0, np.array([1.0, 0.0])), before)
    assert m.solve(1).balance("total").relative_drift < 1e-14


def test_membrane_area_budget_and_zero_radius_are_enforced_atomically():
    m = (
        bt.CoupledModel(["A"])
        .compartment("bath", volume=1)
        .domain("sphere", bt.mesh_1d(5, 0, 0.5, geometry="spherical"))
    )
    with pytest.raises(ValueError, match="physical area"):
        m.membrane("wall", "bath", ("sphere", "left"), area=1, permeability=0.1)
    m.membrane(
        "wall", "bath", ("sphere", "right"), area=math.pi * 0.75, permeability=0.1
    )
    with pytest.raises(ValueError, match="physical area"):
        m.membrane(
            "wall2", "bath", ("sphere", "right"), area=math.pi * 0.5, permeability=0.1
        )
    m.membrane(
        "wall2", "bath", ("sphere", "right"), area=math.pi * 0.25, permeability=0.1
    )


@pytest.mark.parametrize("values", [["A", "A"], [], [""], [" A"], [4]])
def test_invalid_species(values):
    with pytest.raises(ValueError):
        bt.CoupledModel(values)


@pytest.mark.parametrize("volume", [0, -1, float("nan"), float("inf"), True, "2"])
def test_invalid_domain_edits_are_atomic(volume):
    m = bt.CoupledModel(["A"])
    with pytest.raises((ValueError, TypeError)):
        m.compartment("r", volume=volume)
    m.compartment("r", volume=1)
    assert m.compile().domains == ("r",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"diffusivity": {"A": -1}},
        {"initial": {"unknown": 2}},
        {"initial": {"A": [1, 2]}},
        {"initial": {"A": [1, float("nan"), 2]}},
        {"initial": {"A": [1 + 2j, 1, 2]}},
        {"diffusivity": {"A": True}},
    ],
)
def test_invalid_spatial_fields_leave_builder_usable(kwargs):
    m = bt.CoupledModel(["A"])
    with pytest.raises((TypeError, ValueError)):
        m.domain("r", bt.mesh_1d(2), cross_section=1, **kwargs)
    m.domain("r", bt.mesh_1d(2), cross_section=1)
    assert m.compile().initial_state.size == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"permeability": -1},
        {"permeability": {"unknown": 2}},
        {"partition": 0},
        {"partition": float("nan")},
        {"area": 0},
    ],
)
def test_invalid_membrane_parameters_are_atomic(kwargs):
    m = bt.CoupledModel(["A"]).compartment("a", volume=1).compartment("b", volume=1)
    args = dict(area=1, permeability=0.1)
    args.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        m.membrane("wall", "a", "b", **args)
    m.membrane("wall", "a", "b", area=1, permeability=0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reactants": {"A": 0}},
        {"reactants": {"A": 0.5}},
        {"reactants": {"A": True}},
        {"products": {"B": -1}},
        {"rate_constant": float("nan")},
    ],
)
def test_invalid_mass_action(kwargs):
    m = bt.CoupledModel(["A", "B"]).compartment("r", volume=1)
    args = dict(reactants={"A": 1}, products={"B": 1}, rate_constant=1)
    args.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        m.mass_action("r", **args)
    assert_allclose(m.compile().rhs(0, np.ones(2)), 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"end_time": -1},
        {"end_time": float("inf")},
        {"frames": 0},
        {"frames": True},
        {"frames": 2.5},
        {"frames": 2, "save_at": [0.5]},
        {"save_at": [2]},
        {"save_at": [0]},
        {"rtol": 0},
        {"rtol": 1e-18},
        {"atol": {"A": 1e-9}},
        {"atol": -1},
        {"max_step": 0},
        {"method": "explicit"},
    ],
)
def test_invalid_solver_controls(kwargs):
    model = two_compartments()
    args = dict(end_time=1)
    args.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        model.solve(**args)


def test_zero_duration_and_requested_times_and_species_tolerances():
    m = two_compartments()
    zero = m.solve(0)
    assert zero.times.tolist() == [0]
    assert zero.diagnostics.rhs_evaluations == 0
    result = m.solve(1, save_at=[0.7, 0.2, 0.2], atol={"A": 1e-12, "B": 1e-9})
    assert result.times.tolist() == [0, 0.2, 0.7, 1]
    assert result.diagnostics.minimum_atol == 1e-12
    assert result.diagnostics.maximum_atol == 1e-9
    with pytest.raises(ValueError):
        m.solve(0, frames=-1)


def test_supported_quantities_convert_and_wrong_dimensions_fail():
    model = (
        bt.CoupledModel(["A"])
        .compartment("blood", volume=1e-6, initial={"A": bt.quantity(1, "mM")})
        .domain(
            "tissue",
            bt.mesh_1d(4, 0, 1e-3),
            cross_section=1e-4,
            diffusivity={"A": bt.quantity(1e-9, "m^2/s")},
        )
        .membrane(
            "wall",
            "blood",
            ("tissue", "left"),
            area=1e-4,
            permeability=bt.quantity(1e-6, "m/s"),
        )
    )
    assert model.solve(bt.quantity(1, "s")).times[-1] == 1
    with pytest.raises(bt.units.DimensionError):
        bt.CoupledModel(["A"]).compartment(
            "r", volume=1, initial={"A": bt.quantity(2, "s")}
        )


def test_integrator_failure_is_not_returned_as_a_success(monkeypatch):
    import biotransport._coupled_integration as module

    monkeypatch.setattr(
        module,
        "solve_ivp",
        lambda *a, **kw: SimpleNamespace(success=False, message="test failure"),
    )
    with pytest.raises(RuntimeError, match="test failure"):
        two_compartments().solve(1)


def test_custom_nonfinite_or_malformed_rate_is_rejected():
    for rate in [lambda t, c: float("nan"), lambda t, c: [1, 2]]:
        m = bt.CoupledModel(["A"]).compartment("r", volume=1)
        m.reaction("r", stoichiometry={"A": 1}, rate=rate, derivative=lambda t, c: {})
        with pytest.raises(ValueError):
            m.solve(1)


def test_large_model_stays_sparse():
    compiled = (
        bt.CoupledModel(["A", "B"])
        .domain(
            "tissue",
            bt.mesh_1d(10000),
            cross_section=1,
            diffusivity={"A": 0.1, "B": 0.2},
        )
        .mass_action("tissue", reactants={"A": 1}, products={"B": 1}, rate_constant=0.3)
        .compile()
    )
    n = compiled.initial_state.size
    assert compiled.transport_matrix.nnz < 3 * n
    assert compiled.jacobian(0, compiled.initial_state).nnz < 4 * n
    assert compiled.transport_matrix.data.nbytes < 24 * n


def test_zero_initial_conserved_quantity_has_no_relative_error():
    result = (
        bt.CoupledModel(["A"])
        .compartment("r", volume=1)
        .conserve("A", {"A": 1})
        .solve(1)
    )
    report = result.balance("A")
    assert report.relative_drift is None
    assert report.maximum_absolute_drift == 0


def test_production_and_loss_against_exact_forced_kinetics():
    model = (
        bt.CoupledModel(["A"])
        .compartment("reactor", volume=3.0)
        .mass_action("reactor", reactants={}, products={"A": 1}, rate_constant=0.6)
        .mass_action("reactor", reactants={"A": 1}, products={}, rate_constant=0.2)
    )
    solution = model.solve(10, rtol=1e-11, atol=1e-13)
    exact = 3.0 * (1 - np.exp(-0.2 * solution.times))
    assert_allclose(solution.history("reactor", "A")[:, 0], exact, atol=5e-10)
    assert_allclose(solution.amount("A"), 3.0 * exact, atol=2e-9)


def test_closed_sphere_diffusion_second_order_against_continuum_eigenmode():
    # phi(r) = sin(q*r)/(q*r); its derivative at r=1 vanishes when tan(q)=q.
    # Regularity fixes phi(0)=1 and the radial Laplacian is -q^2*phi.
    q = brentq(lambda z: np.tan(z) - z, 4.0, 4.6)
    errors = []
    for n in (16, 32, 64):
        mesh = bt.mesh_1d(n, 0, 1, geometry="spherical")
        phi = np.sinc(q * bt.x_nodes(mesh) / np.pi)
        model = bt.CoupledModel(["A"]).domain(
            "sphere", mesh, initial={"A": 1 + 0.1 * phi}, diffusivity={"A": 0.3}
        )
        solution = model.solve(0.08, frames=1, rtol=1e-11, atol=1e-13)
        exact = 1 + 0.1 * phi * np.exp(-0.3 * q**2 * 0.08)
        errors.append(np.max(np.abs(solution.field("sphere", "A") - exact)))
    assert np.all(np.log2(np.asarray(errors[:-1]) / errors[1:]) > 1.95)


def test_large_stoichiometry_cannot_overflow_the_invariant_check():
    model = bt.CoupledModel(["A", "B"]).compartment("r", volume=1)
    model.reaction(
        "r",
        stoichiometry={"A": 1e200, "B": -1e200},
        rate=lambda t, c: 0.0,
        derivative=lambda t, c: {},
    )
    with pytest.raises(ValueError, match="violates"):
        model.conserve("total", {"A": 1e200, "B": 2e200})
    model.conserve("total", {"A": 1e200, "B": 1e200})


@pytest.mark.parametrize(
    "derivative",
    [lambda t, c: None, lambda t, c: {"A": [1, 2]}, lambda t, c: {"A": float("nan")}],
)
def test_malformed_custom_derivative_is_rejected(derivative):
    compiled = (
        bt.CoupledModel(["A"])
        .compartment("r", volume=1)
        .reaction(
            "r", stoichiometry={"A": 1}, rate=lambda t, c: 0.0, derivative=derivative
        )
        .compile()
    )
    with pytest.raises((ValueError, TypeError)):
        compiled.jacobian(0, compiled.initial_state)
