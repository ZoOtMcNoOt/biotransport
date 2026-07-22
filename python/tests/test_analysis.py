"""Analytic contracts for sensitivity and uncertainty orchestration."""

from __future__ import annotations

import numpy as np
import pytest

from biotransport.analysis import (
    ModelEvaluationError,
    ParameterRange,
    SampleDesign,
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


def test_latin_hypercube_is_reproducible_and_marginally_stratified() -> None:
    parameters = [
        ParameterRange("x", 0.5, 0.0, 1.0),
        ParameterRange("y", 15.0, 10.0, 20.0),
    ]
    first = latin_hypercube(parameters, 32, seed=1928)
    second = latin_hypercube(parameters, 32, seed=1928)
    different = latin_hypercube(parameters, 32, seed=1929)

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
