"""Analytic contracts for sensitivity and uncertainty orchestration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from biotransport.analysis import (
    EvaluationFailure,
    LocalSensitivityResult,
    ModelEvaluationError,
    ParameterRange,
    ParameterSweepResult,
    RegressionScreeningResult,
    SampleDesign,
    UncertaintyResult,
    latin_hypercube,
    local_sensitivity,
    parameter_sweep,
    propagate_uncertainty,
    standardized_regression_coefficients,
)


def test_parameter_range_validates_bounds_nominal_and_distribution() -> None:
    with pytest.raises(ValueError, match="lower must be less"):
        ParameterRange("D", nominal=1.0, lower=2.0, upper=1.0)
    with pytest.raises(ValueError, match="nominal must lie"):
        ParameterRange("D", nominal=3.0, lower=1.0, upper=2.0)
    with pytest.raises(ValueError, match="positive"):
        ParameterRange(
            "D", nominal=1.0, lower=0.0, upper=2.0, distribution="log_uniform"
        )
    with pytest.raises(ValueError, match="distribution"):
        ParameterRange(  # type: ignore[arg-type]
            "D", nominal=1.0, lower=0.5, upper=2.0, distribution="normal"
        )
    with pytest.raises(ValueError, match="width must be finite"):
        ParameterRange("extreme", nominal=0.0, lower=-1.0e308, upper=1.0e308)


def test_parameter_names_are_unique_and_ordered() -> None:
    parameters = [
        ParameterRange("beta", 2.0, 1.0, 3.0),
        ParameterRange("alpha", 1.0, 0.0, 2.0),
    ]
    observed_keys = []

    def model(values):
        observed_keys.append(tuple(values))
        return values["alpha"] + values["beta"]

    result = parameter_sweep(model, parameters, "alpha", [0.5, 1.5])

    assert result.parameter_names == ("beta", "alpha")
    assert observed_keys == [("beta", "alpha"), ("beta", "alpha")]

    with pytest.raises(ValueError, match="unique"):
        latin_hypercube([parameters[0], parameters[0]], 4)


def test_parameter_sweep_preserves_values_and_holds_other_inputs_nominal() -> None:
    parameters = [
        ParameterRange("diffusivity", 2.0, 1.0, 4.0),
        ParameterRange("source", 3.0, 0.0, 5.0),
    ]
    values = np.array([4.0, 1.0, 2.5])

    result = parameter_sweep(
        lambda p: 2.0 * p["diffusivity"] + p["source"],
        parameters,
        "diffusivity",
        values,
    )

    np.testing.assert_array_equal(result.swept_values, values)
    np.testing.assert_array_equal(result.samples[:, 1], 3.0)
    np.testing.assert_array_equal(result.outputs, [11.0, 5.0, 8.0])
    with pytest.raises(ValueError, match="read-only"):
        result.outputs[0] = 0.0


@pytest.mark.parametrize("values", [[0.5], [4.5], [np.nan]])
def test_parameter_sweep_rejects_out_of_range_or_nonfinite_values(values) -> None:
    parameters = [ParameterRange("D", 2.0, 1.0, 4.0)]
    with pytest.raises(ValueError):
        parameter_sweep(lambda p: p["D"], parameters, "D", values)


def test_parameter_sweep_rejects_complex_values_without_casting() -> None:
    parameters = [ParameterRange("D", 2.0, 1.0, 4.0)]
    with pytest.raises(TypeError, match="real numeric array"):
        parameter_sweep(
            lambda p: p["D"],
            parameters,
            "D",
            np.array([1.5 + 0.25j, 2.5 + 0.5j]),
        )


def test_parameter_sweep_rejects_active_masks() -> None:
    parameters = [ParameterRange("D", 2.0, 1.0, 4.0)]
    values = np.ma.array([1.5, 2.5], mask=[False, True])

    with pytest.raises(ValueError, match="masked"):
        parameter_sweep(lambda p: p["D"], parameters, "D", values)


def test_parameter_sweep_wraps_model_errors_with_sample_context() -> None:
    parameters = [ParameterRange("D", 2.0, 1.0, 4.0)]

    def broken(_parameters):
        raise ArithmeticError("solver diverged")

    with pytest.raises(ModelEvaluationError, match="sweep sample 0") as caught:
        parameter_sweep(broken, parameters, "D", [1.5])

    assert caught.value.exception_type == "ArithmeticError"
    assert caught.value.parameters == {"D": 1.5}


def test_model_must_return_one_finite_real_scalar() -> None:
    parameters = [ParameterRange("D", 2.0, 1.0, 4.0)]
    with pytest.raises(ModelEvaluationError, match="one real scalar"):
        parameter_sweep(lambda _p: np.array([1.0]), parameters, "D", [2.0])
    with pytest.raises(ModelEvaluationError, match="NonFiniteOutput"):
        parameter_sweep(lambda _p: np.nan, parameters, "D", [2.0])


@pytest.mark.parametrize("output", [1.0 + 2.0j, np.complex128(3.0 + 4.0j)])
def test_parameter_sweep_rejects_complex_model_outputs_without_casting(output) -> None:
    parameters = [ParameterRange("D", 2.0, 1.0, 4.0)]

    with pytest.raises(ModelEvaluationError, match="one real scalar"):
        parameter_sweep(lambda _p: output, parameters, "D", [2.0])


@pytest.mark.parametrize(
    "output",
    ["1.25", b"1.25", np.str_("1.25"), np.bytes_("1.25")],
)
def test_parameter_sweep_rejects_text_model_outputs(output) -> None:
    parameters = [ParameterRange("D", 2.0, 1.0, 4.0)]

    with pytest.raises(ModelEvaluationError, match="one real scalar"):
        parameter_sweep(lambda _p: output, parameters, "D", [2.0])


def test_central_local_elasticities_match_power_law_exponents() -> None:
    parameters = [
        ParameterRange("k", 2.0, 1.0, 3.0),
        ParameterRange("c", 3.0, 2.0, 4.0),
    ]

    result = local_sensitivity(
        lambda p: p["k"] ** 2 * p["c"], parameters, relative_step=1.0e-5
    )

    assert result.baseline_output == pytest.approx(12.0)
    assert result.derivative_by_parameter["k"] == pytest.approx(12.0, rel=1e-10)
    assert result.derivative_by_parameter["c"] == pytest.approx(4.0, rel=1e-10)
    assert result.normalized_by_parameter == pytest.approx({"k": 2.0, "c": 1.0})


def test_range_normalization_handles_zero_nominal_explicitly() -> None:
    parameters = [ParameterRange("offset", 0.0, -1.0, 1.0)]
    result = local_sensitivity(
        lambda p: p["offset"] + 2.0,
        parameters,
        normalization="range",
    )
    assert result.derivatives[0] == pytest.approx(1.0)
    assert result.normalized_sensitivities[0] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="nonzero nominal"):
        local_sensitivity(lambda p: p["offset"] + 2.0, parameters)


def test_local_sensitivity_does_not_substitute_one_sided_difference() -> None:
    parameters = [ParameterRange("x", 0.0, 0.0, 1.0)]
    with pytest.raises(ValueError, match="central step"):
        local_sensitivity(
            lambda p: p["x"] + 1.0,
            parameters,
            normalization="range",
        )


def test_local_normalization_rejects_zero_baseline() -> None:
    parameters = [ParameterRange("x", 1.0, 0.0, 2.0)]
    with pytest.raises(ValueError, match="zero baseline"):
        local_sensitivity(lambda p: p["x"] - 1.0, parameters)


def test_local_sensitivity_rejects_nonfinite_finite_difference() -> None:
    parameters = [ParameterRange("x", 1.0, 0.5, 1.5)]

    def extreme_jump(p):
        return 1.0e308 if p["x"] > 1.0 else -1.0e308

    with pytest.raises(ValueError, match="overflowed or became non-finite"):
        local_sensitivity(
            extreme_jump,
            parameters,
            absolute_steps={"x": 0.1},
        )


def test_latin_hypercube_is_reproducible_and_marginally_stratified() -> None:
    parameters = [
        ParameterRange("x", 0.5, 0.0, 1.0),
        ParameterRange("y", 15.0, 10.0, 20.0),
    ]
    first = latin_hypercube(parameters, 32, seed=1928)
    second = latin_hypercube(parameters, 32, seed=1928)
    different = latin_hypercube(parameters, 32, seed=1929)

    assert first.method == "latin_hypercube"
    np.testing.assert_array_equal(first.samples, second.samples)
    assert not np.array_equal(first.samples, different.samples)
    for column in range(2):
        occupied = np.floor(first.unit_samples[:, column] * 32).astype(int)
        np.testing.assert_array_equal(np.sort(occupied), np.arange(32))


def test_log_uniform_latin_hypercube_stratifies_log_probability() -> None:
    parameter = ParameterRange(
        "permeability",
        nominal=1.0e-12,
        lower=1.0e-14,
        upper=1.0e-10,
        distribution="log_uniform",
    )
    design = latin_hypercube([parameter], 20, seed=4)
    reconstructed = (np.log(design.samples[:, 0]) - np.log(parameter.lower)) / (
        np.log(parameter.upper) - np.log(parameter.lower)
    )

    np.testing.assert_allclose(reconstructed, design.unit_samples[:, 0])
    occupied = np.floor(reconstructed * 20).astype(int)
    np.testing.assert_array_equal(np.sort(occupied), np.arange(20))


def test_latin_hypercube_rejects_unrepresentable_float64_strata() -> None:
    upper = np.nextafter(1.0, 2.0)
    parameter = ParameterRange("narrow", nominal=1.0, lower=1.0, upper=upper)

    with pytest.raises(ValueError, match="distinct Latin-hypercube strata"):
        latin_hypercube([parameter], 8, seed=3)


def test_sample_design_validates_shape_range_and_unit_coordinates() -> None:
    parameters = (ParameterRange("x", 0.5, 0.0, 1.0),)

    with pytest.raises(ValueError, match="matching shapes"):
        SampleDesign(
            parameters=parameters,
            samples=np.ones((3, 1)),
            unit_samples=np.ones((2, 1)),
            seed=0,
        )
    with pytest.raises(ValueError, match="declared range"):
        SampleDesign(
            parameters=parameters,
            samples=np.array([[0.25], [1.25]]),
            unit_samples=np.array([[0.25], [0.75]]),
            seed=0,
        )
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        SampleDesign(
            parameters=parameters,
            samples=np.array([[0.25], [0.75]]),
            unit_samples=np.array([[-0.1], [1.1]]),
            seed=0,
        )


def test_sample_design_defensively_copies_and_freezes_arrays() -> None:
    parameters = (ParameterRange("x", 0.5, 0.0, 1.0),)
    samples = np.array([[0.25], [0.75]])
    unit_samples = samples.copy()

    design = SampleDesign(
        parameters=parameters,
        samples=samples,
        unit_samples=unit_samples,
        seed=0,
    )
    samples[0, 0] = 0.9
    unit_samples[0, 0] = 0.9

    assert design.samples[0, 0] == pytest.approx(0.25)
    assert design.unit_samples[0, 0] == pytest.approx(0.25)
    assert design.samples.flags.writeable is False
    assert design.unit_samples.flags.writeable is False
    assert design.method == "user_supplied"


def test_sample_design_values_at_requires_a_genuine_integer_index() -> None:
    parameter = ParameterRange("x", 0.5, 0.0, 1.0)
    design = SampleDesign(
        parameters=(parameter,),
        samples=np.array([[0.25], [0.75]]),
        unit_samples=np.array([[0.25], [0.75]]),
        seed=0,
    )

    assert design.values_at(np.int64(1)) == {"x": 0.75}
    for invalid in (True, np.bool_(False), 1.0, "1"):
        with pytest.raises(TypeError, match="integer"):
            design.values_at(invalid)  # type: ignore[arg-type]


def test_uncertainty_summary_is_reproducible_and_close_to_analytic_mean() -> None:
    parameters = [
        ParameterRange("x", 0.5, 0.0, 1.0),
        ParameterRange("y", 0.0, -1.0, 1.0),
    ]

    def model(p):
        return 2.0 * p["x"] + p["y"]

    first = propagate_uncertainty(model, parameters, 256, seed=71)
    second = propagate_uncertainty(model, parameters, 256, seed=71)

    np.testing.assert_array_equal(first.outputs, second.outputs)
    np.testing.assert_array_equal(first.quantile_values, second.quantile_values)
    assert first.mean == pytest.approx(1.0, abs=0.01)
    assert first.standard_deviation == pytest.approx(np.sqrt(2.0 / 3.0), rel=0.08)
    assert first.quantiles[0.025] < first.quantiles[0.5] < first.quantiles[0.975]
    assert first.n_attempted == first.n_successful == 256
    assert first.n_failed == 0


def test_uncertainty_failures_raise_by_default() -> None:
    parameters = [ParameterRange("x", 0.5, 0.0, 1.0)]

    def model(p):
        if p["x"] > 0.5:
            raise RuntimeError("outside solver domain")
        return p["x"]

    with pytest.raises(ModelEvaluationError, match="outside solver domain"):
        propagate_uncertainty(model, parameters, 10, seed=2)


@pytest.mark.parametrize("output", [1.0 + 2.0j, np.complex128(3.0 + 4.0j)])
def test_uncertainty_rejects_complex_model_outputs_without_casting(output) -> None:
    parameters = [ParameterRange("x", 0.5, 0.0, 1.0)]

    with pytest.raises(ModelEvaluationError, match="one real scalar"):
        propagate_uncertainty(lambda _p: output, parameters, 8, seed=2)


@pytest.mark.parametrize(
    "output",
    ["1.25", b"1.25", np.str_("1.25"), np.bytes_("1.25")],
)
def test_uncertainty_rejects_text_model_outputs(output) -> None:
    parameters = [ParameterRange("x", 0.5, 0.0, 1.0)]

    with pytest.raises(ModelEvaluationError, match="one real scalar"):
        propagate_uncertainty(lambda _p: output, parameters, 8, seed=2)


def test_uncertainty_rejects_single_sample_before_model_evaluation() -> None:
    parameters = [ParameterRange("x", 0.5, 0.0, 1.0)]
    calls = []

    def model(p):
        calls.append(p["x"])
        return p["x"]

    with pytest.raises(ValueError, match="at least two requested"):
        propagate_uncertainty(model, parameters, 1)

    assert calls == []


def test_record_policy_accounts_for_every_failure_without_imputation() -> None:
    parameters = [ParameterRange("x", 0.5, 0.0, 1.0)]

    def model(p):
        if p["x"] > 0.5:
            return np.inf
        return p["x"]

    result = propagate_uncertainty(
        model, parameters, 20, seed=2, failure_policy="record"
    )

    assert result.n_attempted == 20
    assert result.n_successful == 10
    assert result.n_failed == 10
    assert result.failure_fraction == 0.5
    assert result.outputs.shape == (10,)
    assert np.all(np.isfinite(result.outputs))
    assert all(
        failure.exception_type == "NonFiniteOutput" for failure in result.failures
    )
    assert {failure.sample_index for failure in result.failures}.isdisjoint(
        result.successful_indices.tolist()
    )


def test_uncertainty_rejects_nonfinite_summary_statistics() -> None:
    parameters = [ParameterRange("x", 0.5, 0.0, 1.0)]
    limit = np.finfo(float).max

    def extreme_output(p):
        return limit if p["x"] > 0.5 else -limit

    with pytest.raises(OverflowError, match="summary statistics"):
        propagate_uncertainty(extreme_output, parameters, 8, seed=0)


def test_uncertainty_statistics_are_stable_at_tiny_scale() -> None:
    parameters = [ParameterRange("x", 0.5, 0.0, 1.0)]
    result = propagate_uncertainty(
        lambda p: 1.0e-200 * (1.0 + p["x"]), parameters, 32, seed=7
    )

    assert np.isfinite(result.standard_deviation)
    assert result.standard_deviation > 0.0
    assert result.mean > 1.0e-200


def test_uncertainty_statistics_are_stable_at_huge_scale() -> None:
    parameters = [ParameterRange("x", 0.5, 0.0, 1.0)]
    result = propagate_uncertainty(
        lambda p: 1.0e308 * (p["x"] - 0.5), parameters, 32, seed=7
    )

    assert np.isfinite(result.mean)
    assert np.isfinite(result.standard_deviation)
    assert result.standard_deviation > 1.0e307


def test_uncertainty_quantiles_are_stable_across_extreme_signs() -> None:
    parameters = [ParameterRange("x", 0.5, 0.0, 1.0)]
    result = propagate_uncertainty(
        lambda p: 1.0e308 if p["x"] > 0.5 else -1.0e308,
        parameters,
        32,
        seed=7,
    )

    assert np.all(np.isfinite(result.quantile_values))
    np.testing.assert_allclose(
        result.quantile_values,
        [-1.0e308, 0.0, 1.0e308],
        rtol=1.0e-15,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "quantiles",
    [[], [0.5, 0.5], [0.9, 0.1], [-0.1, 0.5], [0.5, np.nan]],
)
def test_uncertainty_quantiles_are_explicit_and_validated(quantiles) -> None:
    parameters = [ParameterRange("x", 0.5, 0.0, 1.0)]
    with pytest.raises(ValueError):
        propagate_uncertainty(lambda p: p["x"], parameters, 8, quantiles=quantiles)


def test_standardized_regression_recovers_exact_linear_screening_model() -> None:
    parameters = [
        ParameterRange("x", 0.0, -1.0, 1.0),
        ParameterRange("z", 0.0, -1.0, 1.0),
        ParameterRange("k", 2.0, 1.0, 4.0, distribution="log_uniform"),
    ]
    design = latin_hypercube(parameters, 128, seed=5)
    transformed = design.samples.copy()
    transformed[:, 2] = np.log(transformed[:, 2])
    outputs = (
        3.0 * transformed[:, 0] - 2.0 * transformed[:, 1] + 0.5 * transformed[:, 2]
    )

    result = standardized_regression_coefficients(design, outputs)

    input_std = np.std(transformed, axis=0, ddof=1)
    output_std = np.std(outputs, ddof=1)
    expected = np.array([3.0, -2.0, 0.5]) * input_std / output_std
    np.testing.assert_allclose(result.coefficients, expected, rtol=1.0e-12)
    assert result.r_squared == pytest.approx(1.0, abs=1.0e-14)
    assert result.adjusted_r_squared == pytest.approx(1.0, abs=1.0e-14)
    assert result.standardized_rmse < 1.0e-13
    assert result.design_rank == 3
    assert result.log_transformed == (False, False, True)
    assert set(result.absolute_ranking) == {"x", "z", "k"}


def test_standardized_regression_accepts_uncertainty_result() -> None:
    parameters = [
        ParameterRange("x", 0.5, 0.0, 1.0),
        ParameterRange("y", 0.5, 0.0, 1.0),
    ]
    uncertainty = propagate_uncertainty(
        lambda p: p["x"] - 4.0 * p["y"], parameters, 32, seed=18
    )

    result = standardized_regression_coefficients(uncertainty)

    assert result.r_squared == pytest.approx(1.0)
    assert result.coefficient_by_parameter["x"] > 0.0
    assert result.coefficient_by_parameter["y"] < 0.0


@pytest.mark.parametrize(
    ("scale", "slope"),
    [(1.0e-200, 3.0), (1.0e307, 1.5)],
    ids=["tiny", "huge"],
)
def test_standardized_regression_is_stable_across_scales(scale, slope) -> None:
    samples = scale * np.linspace(1.0, 9.0, 16)
    parameter = ParameterRange(
        "x",
        nominal=5.0 * scale,
        lower=scale,
        upper=9.0 * scale,
    )
    design = SampleDesign(
        parameters=(parameter,),
        samples=samples[:, None],
        unit_samples=np.linspace(0.0, 1.0, samples.size)[:, None],
        seed=0,
    )

    result = standardized_regression_coefficients(design, slope * samples)

    assert result.coefficients[0] == pytest.approx(1.0, rel=1.0e-12)
    assert result.r_squared == pytest.approx(1.0, abs=1.0e-14)


@pytest.mark.parametrize("noise_scale", [1.0e-6, 1.0e-14])
def test_standardized_regression_accepts_near_perfect_noisy_fits(
    noise_scale,
) -> None:
    design = latin_hypercube(
        [
            ParameterRange("x", 0.5, 0.0, 1.0),
            ParameterRange("y", 0.5, 0.0, 1.0),
        ],
        64,
        seed=29,
    )
    outputs = (
        2.0 * design.samples[:, 0]
        - 0.5 * design.samples[:, 1]
        + noise_scale * np.sin(np.arange(design.n_samples))
    )

    result = standardized_regression_coefficients(design, outputs)

    assert result.r_squared > 0.999999999
    assert np.isfinite(result.standardized_rmse)


def test_standardized_regression_rejects_rank_deficiency() -> None:
    parameters = (
        ParameterRange("x", 0.5, 0.0, 1.0),
        ParameterRange("duplicate_x", 0.5, 0.0, 1.0),
    )
    x = np.linspace(0.1, 0.9, 8)
    samples = np.column_stack([x, x])
    design = SampleDesign(
        parameters=parameters,
        samples=samples,
        unit_samples=samples,
        seed=0,
    )

    with pytest.raises(ValueError, match="rank deficient"):
        standardized_regression_coefficients(design, x)


def test_standardized_regression_rejects_ill_conditioned_design() -> None:
    parameters = (
        ParameterRange("x", 0.5, 0.0, 1.0),
        ParameterRange("almost_x", 0.5, 0.0, 1.0),
    )
    x = np.linspace(0.1, 0.9, 20)
    perturbation = np.sin(np.arange(20)) * 1.0e-8
    samples = np.column_stack([x, x + perturbation])
    design = SampleDesign(
        parameters=parameters,
        samples=samples,
        unit_samples=samples,
        seed=0,
    )

    with pytest.raises(ValueError, match="ill-conditioned"):
        standardized_regression_coefficients(
            design, x + perturbation, max_condition_number=1.0e6
        )


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ([1.0, 2.0], "length"),
        (np.ones(8), "constant outputs"),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan], "finite"),
    ],
)
def test_standardized_regression_rejects_invalid_outputs(outputs, message) -> None:
    design = latin_hypercube([ParameterRange("x", 0.5, 0.0, 1.0)], 8)
    with pytest.raises(ValueError, match=message):
        standardized_regression_coefficients(design, outputs)


def test_standardized_regression_rejects_complex_outputs_without_casting() -> None:
    design = latin_hypercube([ParameterRange("x", 0.5, 0.0, 1.0)], 8)
    outputs = np.arange(8, dtype=np.float64) + 1.0j

    with pytest.raises(TypeError, match="real numeric array"):
        standardized_regression_coefficients(design, outputs)


def test_standardized_regression_rejects_active_output_masks() -> None:
    design = latin_hypercube([ParameterRange("x", 0.5, 0.0, 1.0)], 8)
    outputs = np.ma.array(np.arange(8.0), mask=[False] * 7 + [True])

    with pytest.raises(ValueError, match="masked"):
        standardized_regression_coefficients(design, outputs)


class TestScientificResultRecords:
    def test_parameter_sweep_result_copies_freezes_and_validates(self) -> None:
        samples = np.array([[1.0, 2.0], [2.0, 2.0]])
        outputs = np.array([3.0, 4.0])
        result = ParameterSweepResult(("x", "y"), "x", samples, outputs)
        samples[0, 0] = 99.0
        outputs[0] = 99.0

        assert result.samples[0, 0] == pytest.approx(1.0)
        assert result.outputs[0] == pytest.approx(3.0)
        assert result.samples.flags.writeable is False
        assert result.outputs.flags.writeable is False
        with pytest.raises(FrozenInstanceError):
            result.swept_parameter = "y"
        with pytest.raises(ValueError, match="outputs length"):
            ParameterSweepResult(("x", "y"), "x", np.ones((2, 2)), np.ones(3))

    def test_local_sensitivity_result_copies_freezes_and_validates(self) -> None:
        steps = np.array([0.1, 0.2])
        derivatives = np.array([1.0, 2.0])
        normalized = np.array([0.5, 1.0])
        result = LocalSensitivityResult(
            ("x", "y"),
            2.0,
            steps,
            derivatives,
            normalized,
            "elasticity",
        )
        steps[0] = 99.0
        derivatives[0] = 99.0

        assert result.step_sizes[0] == pytest.approx(0.1)
        assert result.derivatives[0] == pytest.approx(1.0)
        assert result.step_sizes.flags.writeable is False
        with pytest.raises(ValueError, match="step_sizes"):
            LocalSensitivityResult(
                ("x", "y"),
                2.0,
                np.array([0.1]),
                np.ones(2),
                np.ones(2),
                "elasticity",
            )

    def test_evaluation_failure_validates_metadata(self) -> None:
        failure = EvaluationFailure(1, (0.25, 2.0), "RuntimeError", "failed")

        assert failure.parameters(("x", "y")) == {"x": 0.25, "y": 2.0}
        with pytest.raises(ValueError, match="nonnegative"):
            EvaluationFailure(-1, (0.25,), "RuntimeError", "failed")
        with pytest.raises(TypeError, match="real scalar"):
            EvaluationFailure(0, ("0.25",), "RuntimeError", "failed")

    def test_uncertainty_result_copies_freezes_and_validates(self) -> None:
        parameter = ParameterRange("x", 0.5, 0.0, 1.0)
        design = SampleDesign(
            (parameter,),
            np.array([[0.1], [0.5], [0.9]]),
            np.array([[0.1], [0.5], [0.9]]),
            seed=0,
        )
        indices = np.array([0, 2])
        outputs = np.array([1.0, 3.0])
        probabilities = np.array([0.5])
        quantile_values = np.array([2.0])
        failure = EvaluationFailure(1, (0.5,), "RuntimeError", "failed")
        result = UncertaintyResult(
            design,
            indices,
            outputs,
            probabilities,
            quantile_values,
            mean=2.0,
            standard_deviation=np.sqrt(2.0),
            failures=(failure,),
        )
        indices[0] = 1
        outputs[0] = 99.0
        quantile_values[0] = 99.0

        np.testing.assert_array_equal(result.successful_indices, [0, 2])
        np.testing.assert_array_equal(result.outputs, [1.0, 3.0])
        np.testing.assert_array_equal(result.quantile_values, [2.0])
        assert result.outputs.flags.writeable is False
        with pytest.raises(ValueError, match="mean is inconsistent"):
            UncertaintyResult(
                design,
                np.array([0, 2]),
                np.array([1.0, 3.0]),
                np.array([0.5]),
                np.array([2.0]),
                mean=5.0,
                standard_deviation=np.sqrt(2.0),
                failures=(failure,),
            )

    def test_regression_result_copies_freezes_and_validates(self) -> None:
        coefficients = np.array([0.5, -0.25])
        singular_values = np.array([2.0, 1.0])
        result = RegressionScreeningResult(
            ("x", "y"),
            coefficients,
            r_squared=0.9,
            adjusted_r_squared=0.7,
            standardized_rmse=np.sqrt(0.075),
            design_rank=2,
            condition_number=2.0,
            singular_values=singular_values,
            n_samples=4,
            log_transformed=(False, True),
        )
        coefficients[0] = 99.0
        singular_values[0] = 99.0

        np.testing.assert_array_equal(result.coefficients, [0.5, -0.25])
        np.testing.assert_array_equal(result.singular_values, [2.0, 1.0])
        assert result.coefficients.flags.writeable is False
        with pytest.raises(ValueError, match="condition_number"):
            RegressionScreeningResult(
                ("x", "y"),
                np.array([0.5, -0.25]),
                r_squared=0.9,
                adjusted_r_squared=0.7,
                standardized_rmse=np.sqrt(0.075),
                design_rank=2,
                condition_number=3.0,
                singular_values=np.array([2.0, 1.0]),
                n_samples=4,
                log_transformed=(False, True),
            )

    @pytest.mark.parametrize(
        ("adjusted_r_squared", "standardized_rmse", "message"),
        [
            (0.8, np.sqrt(0.075), "adjusted_r_squared is inconsistent"),
            (0.7, 0.1, "standardized_rmse is inconsistent"),
        ],
    )
    def test_regression_result_rejects_impossible_summaries(
        self, adjusted_r_squared, standardized_rmse, message
    ) -> None:
        with pytest.raises(ValueError, match=message):
            RegressionScreeningResult(
                ("x", "y"),
                np.array([0.5, -0.25]),
                r_squared=0.9,
                adjusted_r_squared=adjusted_r_squared,
                standardized_rmse=standardized_rmse,
                design_rank=2,
                condition_number=2.0,
                singular_values=np.array([2.0, 1.0]),
                n_samples=4,
                log_transformed=(False, True),
            )
