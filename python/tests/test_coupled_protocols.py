"""Exact open-system references, protocol edges and external amount balances."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from scipy.integrate import quad

import biotransport as bt


def bath_model(*, volume=1.0, value=1.4, initial=0.2, target=False, partition=2.5):
    model = bt.CoupledModel(["A"]).compartment(
        "tissue", volume=volume, initial={"A": initial}
    )
    model.bath("bath", concentration={"A": value})
    source, destination = ("tissue", "bath") if target else ("bath", "tissue")
    model.membrane(
        "wall",
        source,
        destination,
        area=1.0,
        permeability=0.7 * volume,
        partition=partition,
    )
    return model.conserve("A", {"A": 1})


@pytest.mark.parametrize("method", ["BDF", "Radau"])
@pytest.mark.parametrize("target", [False, True])
@pytest.mark.parametrize("volume", [1e-18, 1e-9, 1.0])
def test_constant_bath_partition_orientation_and_tiny_volume_accounting(
    method, target, volume
):
    model = bath_model(volume=volume, target=target)
    result = model.solve(8, method=method, frames=20, rtol=1e-10, atol=1e-12)
    rate = 0.7 if target else 0.7 / 2.5
    equilibrium = 1.4 / 2.5 if target else 1.4 * 2.5
    exact = equilibrium + (0.2 - equilibrium) * np.exp(-rate * result.times)
    assert_allclose(result.history("tissue", "A")[:, 0], exact, atol=2e-9)
    # Normalize by volume to expose errors hidden by an absolute tolerance in moles.
    assert_allclose(result.external_amount("A") / volume, exact - 0.2, atol=2e-9)
    signed = (-1 if target else 1) * volume * rate * (equilibrium - exact)
    assert_allclose(
        result.interface_rate("wall", "A") / volume, signed / volume, atol=2e-9
    )
    report = result.balance("A")
    assert report.relative_drift < 3e-14
    assert report.expected_final == pytest.approx(report.final, rel=3e-14)
    assert report.external_change / volume == pytest.approx(exact[-1] - 0.2, abs=2e-9)
    compiled = model.compile()
    assert compiled.baths == ("bath",)
    assert compiled.domains == ("tissue",)
    assert compiled.initial_state.size == 1
    assert compiled.external_rates(0, compiled.initial_state)[
        "wall", "A"
    ] / volume == pytest.approx(rate * (equilibrium - 0.2))


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_narrow_pulse_cannot_be_skipped_between_output_times(method):
    start, stop = 1.234567, 1.234568
    protocol = bt.ConcentrationSchedule([0, start, stop], [0, 1, 0])
    model = bath_model(value=protocol, initial=0, partition=1)
    solution = model.solve(4, save_at=[4], method=method, rtol=1e-10, atol=1e-14)
    assert solution.times.tolist() == [0, start, stop, 4]
    peak = -np.expm1(-0.7 * (stop - start))
    exact = [0, 0, peak, peak * np.exp(-0.7 * (4 - stop))]
    assert_allclose(solution.history("tissue", "A")[:, 0], exact, atol=4e-14, rtol=2e-7)
    assert_allclose(solution.external_amount("A"), exact, atol=4e-14, rtol=2e-7)
    assert solution.bath_history("bath", "A").tolist() == [0, 1, 0, 0]
    assert solution.balance("A").relative_drift < 2e-14


def test_final_state_uses_left_limit_but_instantaneous_rate_uses_new_bath_value():
    model = bath_model(
        value=bt.ConcentrationSchedule([0, 1], [0, 100]), initial=0, partition=1
    )
    solution = model.solve(1, frames=1)
    assert_allclose(solution.field("tissue", "A"), 0)
    assert_allclose(solution.external_amount("A"), 0)
    assert solution.bath_history("bath", "A").tolist() == [0, 100]
    assert solution.interface_rate("wall", "A")[-1] == pytest.approx(70)


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_linear_ramps_against_piecewise_exact_affine_forcing(method):
    knots, values = [0.0, 1.0, 3.0, 4.0], [0.2, 1.7, 0.0, 0.8]
    model = bath_model(
        value=bt.ConcentrationSchedule(knots, values, interpolation="linear"),
        initial=0.6,
        partition=1,
    )
    result = model.solve(6, frames=60, method=method, rtol=1e-11, atol=1e-13)

    def exact(time):
        c = 0.6
        for i, start in enumerate(knots):
            stop = min(time, knots[i + 1] if i + 1 < len(knots) else time)
            if stop <= start:
                break
            duration = stop - start
            slope = (
                (values[i + 1] - values[i]) / (knots[i + 1] - start)
                if i + 1 < len(knots)
                else 0
            )
            decay = np.exp(-0.7 * duration)
            c = (
                values[i]
                + (c - values[i]) * decay
                + slope * (duration + np.expm1(-0.7 * duration) / 0.7)
            )
        return c

    reference = np.asarray([exact(t) for t in result.times])
    assert_allclose(result.history("tissue", "A")[:, 0], reference, atol=4e-9)
    assert_allclose(result.external_amount("A"), reference - 0.6, atol=4e-9)
    assert result.balance("A").relative_drift < 5e-14


def test_opposing_baths_track_gross_exchange_even_when_state_is_stationary():
    model = bt.CoupledModel(["A"]).compartment("cell", volume=1, initial={"A": 0.5})
    model.bath("donor", concentration={"A": 1}).bath("sink", concentration={"A": 0})
    model.membrane("entry", "donor", "cell", area=1, permeability=1)
    model.membrane("exit", "cell", "sink", area=1, permeability=1)
    model.conserve("A", {"A": 1})
    result = model.solve(5, frames=5)
    assert_allclose(result.history("cell", "A"), 0.5, atol=1e-15)
    assert_allclose(
        result.external_amount("A", membrane="entry"), 0.5 * result.times, atol=1e-14
    )
    assert_allclose(
        result.external_amount("A", membrane="exit"), -0.5 * result.times, atol=1e-14
    )
    assert_allclose(result.external_amount("A"), 0, atol=1e-14)
    assert result.balance("A").relative_drift < 5e-14


def test_external_accounting_with_reactions_against_exact_open_kinetics():
    volume, rate, reaction = 1e-15, 0.4, 0.2
    model = bt.CoupledModel(["A", "B"]).compartment("r", volume=volume)
    model.bath("donor", concentration={"A": 1})
    model.membrane("wall", "donor", "r", area=1, permeability={"A": rate * volume})
    model.mass_action(
        "r", reactants={"A": 1}, products={"B": 2}, rate_constant=reaction
    )
    model.conserve("equivalents", {"A": 1, "B": 0.5})
    result = model.solve(10, rtol=1e-11, atol={"A": 1e-13, "B": 1e-12})
    lam, equilibrium = rate + reaction, rate / (rate + reaction)
    a = equilibrium * -np.expm1(-lam * result.times)
    b = (
        2
        * reaction
        * equilibrium
        * (result.times + np.expm1(-lam * result.times) / lam)
    )
    assert_allclose(result.history("r", "A")[:, 0], a, atol=2e-10)
    assert_allclose(result.history("r", "B")[:, 0], b, atol=3e-10)
    assert_allclose(result.external_amount("A") / volume, a + b / 2, atol=3e-10)
    assert_allclose(result.external_amount("B"), 0)
    assert result.balance("equivalents").relative_drift < 3e-14


def test_spatial_open_balance_against_independent_augmented_matrix_exponential():
    model = bt.CoupledModel(["A"]).domain(
        "slab",
        bt.mesh_1d(1, 0, 2),
        cross_section=1,
        diffusivity={"A": 0.3},
        initial={"A": [0.2, 0.1]},
    )
    model.bath("donor", concentration={"A": 1})
    model.membrane(
        "wall", "donor", ("slab", "left"), area=1, permeability=0.4, partition=2
    )
    model.conserve("A", {"A": 1})
    # The last coordinate is a constant for the affine forcing, and the third
    # coordinate integrates signed boundary moles. Each half-cell volume is 1.
    operator = np.array(
        [[-0.35, 0.15, 0, 0.4], [0.15, -0.15, 0, 0], [-0.2, 0, 0, 0.4], [0, 0, 0, 0]]
    )
    result = model.solve(8, frames=20, rtol=1e-11, atol=1e-13)
    expected = np.array([expm(t * operator) @ [0.2, 0.1, 0, 1] for t in result.times])
    assert_allclose(result.history("slab", "A"), expected[:, :2], atol=3e-10)
    assert_allclose(result.external_amount("A"), expected[:, 2], atol=3e-10)
    assert result.balance("A").relative_drift < 3e-14


def test_open_nonlinear_rhs_and_sparse_jacobian_agree_by_directional_derivative():
    model = bt.CoupledModel(["A", "B"]).compartment("r", volume=0.3, initial={"A": 1})
    model.bath(
        "donor",
        concentration={
            "A": bt.ConcentrationSchedule([0, 1], [0.2, 1], interpolation="linear")
        },
    )
    model.membrane("wall", "r", "donor", area=0.2, permeability={"A": 0.3}, partition=2)
    model.mass_action("r", reactants={"A": 2}, products={"B": 1}, rate_constant=0.5)
    compiled = model.compile()
    state, direction, eps = np.array([0.7, 0.1]), np.array([0.4, -0.8]), 1e-6
    numerical = (
        compiled.rhs(0.4, state + eps * direction)
        - compiled.rhs(0.4, state - eps * direction)
    ) / (2 * eps)
    assert_allclose(
        compiled.jacobian(0.4, state) @ direction, numerical, rtol=1e-8, atol=1e-10
    )
    assert compiled.jacobian(0.4, state).shape == (2, 2)


def test_restarts_keep_custom_reaction_time_absolute():
    model = bt.CoupledModel(["A"]).compartment("r", volume=1, initial={"A": 1})
    model.bath(
        "sink", concentration={"A": bt.ConcentrationSchedule([0, 0.5, 1], [0, 0, 0])}
    )
    model.membrane("wall", "r", "sink", area=1, permeability=0.7)
    model.reaction(
        "r",
        stoichiometry={"A": -1},
        rate=lambda t, c: t * c["A"],
        derivative=lambda t, c: {"A": t},
    )
    result = model.solve(3, rtol=1e-11, atol=1e-13)
    assert_allclose(
        result.history("r", "A")[:, 0],
        np.exp(-0.7 * result.times - result.times**2 / 2),
        atol=4e-10,
    )


def test_output_density_does_not_change_integration_or_ledger():
    model = bath_model(value=bt.ConcentrationSchedule([0, 0.3, 1.7], [0, 1, 0]))
    sparse, dense = model.solve(5, frames=1), model.solve(5, frames=100)
    assert_allclose(
        sparse.field("tissue", "A"), dense.field("tissue", "A"), rtol=0, atol=0
    )
    assert sparse.external_amount("A")[-1] == dense.external_amount("A")[-1]
    assert sparse.diagnostics.rhs_evaluations == dense.diagnostics.rhs_evaluations


def test_bath_configuration_and_schedules_are_owned_and_edits_are_atomic():
    times, values = [0, 1], [0, 2]
    schedule = bt.ConcentrationSchedule(times, values)
    model = bath_model(value=schedule)
    compiled = model.compile()
    times[1], values[1] = 100, 999
    with pytest.raises(FrozenInstanceError):
        schedule.times = (0, 100)
    assert compiled.breakpoints == (1,)
    assert compiled.bath_concentration("bath", "A", 1) == 2
    with pytest.raises(ValueError, match="already exists"):
        model.compartment("bath", volume=1)
    with pytest.raises(ValueError, match="unknown species"):
        model.bath("other", concentration={"missing": 1})
    model.bath("other", concentration={"A": 1})
    assert compiled.baths == ("bath",)


def test_inactive_schedules_do_not_add_breakpoints():
    model = bt.CoupledModel(["A", "B"]).compartment("r", volume=1)
    model.bath(
        "unused", concentration={"A": bt.ConcentrationSchedule([0, 0.2], [0, 1])}
    )
    model.bath("donor", concentration={"B": bt.ConcentrationSchedule([0, 0.5], [0, 1])})
    model.membrane("wall", "donor", "r", area=1, permeability={"A": 1})
    assert model.compile().breakpoints == ()
    result = model.solve(1, frames=1)
    assert result.times.tolist() == [0, 1]
    assert_allclose(result.external_amount("B", membrane="wall"), 0)


def test_multiple_baths_union_schedule_knots_and_preserve_zero_duration():
    model = bath_model(value=bt.ConcentrationSchedule([0, 0.2, 0.7], [0, 1, 0]))
    model.bath(
        "second",
        concentration={"A": bt.ConcentrationSchedule([0, 0.4, 0.7, 2], [1, 0, 1, 0])},
    )
    model.membrane("other", "second", "tissue", area=1, permeability=0.2)
    assert model.compile().breakpoints == (0.2, 0.4, 0.7, 2)
    zero = model.solve(0)
    assert zero.times.tolist() == [0]
    assert_allclose(zero.external_amount("A"), 0)
    assert model.solve(1, frames=1).times.tolist() == [0, 0.2, 0.4, 0.7, 1]


def test_baths_require_a_finite_domain_and_obey_spatial_face_area_budget():
    model = (
        bt.CoupledModel(["A"])
        .bath("a", concentration={"A": 1})
        .bath("b", concentration={})
    )
    with pytest.raises(ValueError, match="modeled domain"):
        model.membrane("wall", "a", "b", area=1, permeability=1)
    with pytest.raises(ValueError, match="at least one"):
        model.compile()
    model.domain("slab", bt.mesh_1d(2), cross_section=1)
    with pytest.raises(ValueError, match="physical area"):
        model.membrane("wall", "a", ("slab", "left"), area=2, permeability=1)
    model.membrane("wall", "a", ("slab", "left"), area=1, permeability=1)
    with pytest.raises(ValueError, match="physical area"):
        model.membrane("other", "b", ("slab", "left"), area=0.1, permeability=1)


@pytest.mark.parametrize(
    "times,values",
    [
        ([], []),
        ([0, 1], [0]),
        ([1], [0]),
        ([0, 0], [0, 1]),
        ([0, -1], [0, 1]),
        ([0, float("nan")], [0, 1]),
        ([0, float("inf")], [0, 1]),
        ([False], [1]),
        ([0], [-1]),
        ([0], [float("nan")]),
        ([0], [True]),
        ([0], [1j]),
        ([0], ["2"]),
    ],
)
def test_invalid_schedules_fail_before_model_edits(times, values):
    with pytest.raises((ValueError, TypeError)):
        bt.ConcentrationSchedule(times, values)


def test_schedule_quantities_interpolation_and_dimension_checks():
    schedule = bt.ConcentrationSchedule(
        [bt.quantity(0, "s"), bt.quantity(1, "min")],
        [bt.quantity(0, "mM"), bt.quantity(2, "mM")],
        interpolation="linear",
    )
    assert schedule.at(30) == 1
    assert schedule.at(bt.quantity(2, "min")) == 2
    with pytest.raises(bt.units.DimensionError):
        bt.ConcentrationSchedule([0], [bt.quantity(1, "s")])
    with pytest.raises(ValueError):
        schedule.at(-1)


def test_bad_external_queries_fail_clearly_and_returned_arrays_are_owned():
    result = bath_model().solve(1)
    before = result.external_amount("A")
    result.external_amount("A")[:] = 999
    result.bath_history("bath", "A")[:] = 999
    assert_allclose(result.external_amount("A"), before)
    with pytest.raises(ValueError, match="unknown species"):
        result.external_amount("typo")
    with pytest.raises(ValueError, match="external bath"):
        result.external_amount("A", membrane="typo")
    with pytest.raises(ValueError, match="unknown bath/species"):
        result.bath_history("bath", "typo")
    with pytest.raises(ValueError):
        result.amount("A", domain="bath")


def test_interleaved_step_and_ramp_baths_against_independent_quadrature():
    model = bt.CoupledModel(["A"]).compartment("r", volume=1, initial={"A": 0.3})
    model.bath(
        "step", concentration={"A": bt.ConcentrationSchedule([0, 0.7, 2.1], [0, 1, 0])}
    )
    model.bath(
        "ramp",
        concentration={
            "A": bt.ConcentrationSchedule(
                [0, 1.2, 3], [0.2, 1.4, 0.5], interpolation="linear"
            )
        },
    )
    model.membrane("one", "step", "r", area=1, permeability=0.7)
    model.membrane("two", "ramp", "r", area=1, permeability=0.4)
    model.conserve("A", {"A": 1})
    result = model.solve(4, frames=8, rtol=1e-10, atol=1e-12)
    for t, c in zip(result.times[1:], result.history("r", "A")[1:, 0]):
        knots = [v for v in (0.7, 1.2, 2.1, 3) if v < t]
        integral = quad(
            lambda tau: (
                np.exp(-1.1 * (t - tau))
                * (
                    0.7 * (0.7 <= tau < 2.1)
                    + 0.4 * np.interp(tau, [0, 1.2, 3], [0.2, 1.4, 0.5])
                )
            ),
            0,
            t,
            points=knots,
            epsabs=1e-12,
        )[0]
        assert c == pytest.approx(0.3 * np.exp(-1.1 * t) + integral, abs=3e-9)
    assert result.balance("A").relative_drift < 2e-14


def test_research_rhs_exposes_both_time_limits_without_shifting_reaction_time():
    schedule = bt.ConcentrationSchedule([0, 1], [0, 2])
    model = bath_model(value=schedule, initial=0, partition=1)
    compiled = model.compile()
    assert schedule.at(1, side="left") == 0
    assert schedule.at(1, side="right") == 2
    assert_allclose(compiled.rhs(1, compiled.initial_state, bath_side="left"), 0)
    assert_allclose(compiled.rhs(1, compiled.initial_state), 1.4)
    assert (
        compiled.external_rates(1, compiled.initial_state, bath_side="left")[
            "wall", "A"
        ]
        == 0
    )
    assert compiled.bath_concentration("bath", "A", 1, side="left") == 0
    with pytest.raises(ValueError, match="side"):
        compiled.rhs(1, compiled.initial_state, bath_side="typo")
    with pytest.raises(ValueError, match="side"):
        schedule.at(1, side="typo")


def test_unsupported_schedule_interpolation_is_explicit():
    with pytest.raises(ValueError, match="interpolation"):
        bt.ConcentrationSchedule([0], [1], interpolation="cubic")
