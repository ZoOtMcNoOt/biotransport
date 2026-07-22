"""Reproducible sensitivity screening and uncertainty propagation.

The functions in this module orchestrate scalar quantities of interest around
user-supplied model callables.  A callable can construct and run one of
BioTransport's native C++ solvers, then reduce the solution to a scalar.  The
resulting diagnostics are conditional on the supplied model, parameter ranges,
and sampling assumptions.  They are not evidence of model validation,
causality, or compliance with an uncertainty standard.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Model = Callable[[Mapping[str, float]], float]
Distribution = Literal["uniform", "log_uniform"]
Normalization = Literal["elasticity", "range"]
FailurePolicy = Literal["raise", "record"]


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _seed(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("seed must be an integer")
    result = int(value)
    if not 0 <= result < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    return result


def _readonly_float(values: ArrayLike) -> FloatArray:
    result = np.ascontiguousarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def _readonly_int(values: ArrayLike) -> IntArray:
    result = np.ascontiguousarray(values, dtype=np.int64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ParameterRange:
    """Nominal value and independent sampling range for one model input.

    ``distribution="uniform"`` samples uniformly in the physical value.
    ``distribution="log_uniform"`` samples uniformly in the natural logarithm
    and therefore requires positive bounds.  A distribution is a user-supplied
    modeling assumption, not a distribution inferred from data.
    """

    name: str
    nominal: float
    lower: float
    upper: float
    distribution: Distribution = "uniform"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("parameter name must be a string")
        if not self.name or self.name != self.name.strip():
            raise ValueError("parameter name must be nonempty without outer whitespace")

        nominal = _finite_real(self.nominal, f"{self.name}.nominal")
        lower = _finite_real(self.lower, f"{self.name}.lower")
        upper = _finite_real(self.upper, f"{self.name}.upper")
        if not lower < upper:
            raise ValueError(f"{self.name}: lower must be less than upper")
        if not lower <= nominal <= upper:
            raise ValueError(f"{self.name}: nominal must lie within [lower, upper]")
        if self.distribution not in ("uniform", "log_uniform"):
            raise ValueError(
                f"{self.name}: distribution must be 'uniform' or 'log_uniform'"
            )
        if self.distribution == "log_uniform" and lower <= 0.0:
            raise ValueError(f"{self.name}: log_uniform bounds must be positive")

        object.__setattr__(self, "nominal", nominal)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


def _parameters(values: Sequence[ParameterRange]) -> Tuple[ParameterRange, ...]:
    result = tuple(values)
    if not result:
        raise ValueError("at least one parameter is required")
    if any(not isinstance(parameter, ParameterRange) for parameter in result):
        raise TypeError("parameters must contain only ParameterRange instances")
    names = tuple(parameter.name for parameter in result)
    if len(set(names)) != len(names):
        raise ValueError("parameter names must be unique")
    return result


def _parameter_mapping(
    parameters: Sequence[ParameterRange], values: Iterable[float]
) -> Dict[str, float]:
    return {
        parameter.name: float(value) for parameter, value in zip(parameters, values)
    }


class ModelEvaluationError(RuntimeError):
    """A model callable raised or returned a non-finite/non-scalar value."""

    def __init__(
        self,
        context: str,
        parameter_names: Sequence[str],
        values: Iterable[float],
        exception_type: str,
        reason: str,
    ) -> None:
        self.context = context
        self.parameter_names = tuple(parameter_names)
        self.values = tuple(float(value) for value in values)
        self.exception_type = exception_type
        self.reason = reason
        formatted = ", ".join(
            f"{name}={value:.17g}"
            for name, value in zip(self.parameter_names, self.values)
        )
        super().__init__(
            f"model evaluation failed at {context} ({formatted}): "
            f"{exception_type}: {reason}"
        )

    @property
    def parameters(self) -> Dict[str, float]:
        """Parameter values at the failed evaluation, in declared order."""
        return dict(zip(self.parameter_names, self.values))


def _evaluate_model(
    model: Model,
    parameters: Sequence[ParameterRange],
    values: Iterable[float],
    context: str,
) -> float:
    if not callable(model):
        raise TypeError("model must be callable")
    names = tuple(parameter.name for parameter in parameters)
    value_tuple = tuple(float(value) for value in values)
    try:
        raw_value = model(_parameter_mapping(parameters, value_tuple))
    except Exception as error:
        raise ModelEvaluationError(
            context, names, value_tuple, type(error).__name__, str(error)
        ) from error

    if isinstance(raw_value, (bool, np.bool_)) or not np.isscalar(raw_value):
        raise ModelEvaluationError(
            context,
            names,
            value_tuple,
            type(raw_value).__name__,
            "model must return one real scalar quantity of interest",
        )
    try:
        result = float(cast(Any, raw_value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelEvaluationError(
            context,
            names,
            value_tuple,
            type(raw_value).__name__,
            "model must return one real scalar quantity of interest",
        ) from error
    if not np.isfinite(result):
        raise ModelEvaluationError(
            context,
            names,
            value_tuple,
            "NonFiniteOutput",
            f"model returned {result!r}",
        )
    return result


@dataclass(frozen=True)
class ParameterSweepResult:
    """Outputs from a deterministic one-at-a-time parameter sweep."""

    parameter_names: Tuple[str, ...]
    swept_parameter: str
    samples: FloatArray
    outputs: FloatArray

    @property
    def swept_values(self) -> FloatArray:
        """Physical values used for the swept parameter."""
        column = self.parameter_names.index(self.swept_parameter)
        return self.samples[:, column]


def parameter_sweep(
    model: Model,
    parameters: Sequence[ParameterRange],
    parameter_name: str,
    values: ArrayLike,
) -> ParameterSweepResult:
    """Evaluate ``model`` in caller-provided order while varying one input.

    Every non-swept input remains at its declared nominal value.  Duplicate
    sweep values are allowed; shape, finiteness, range, and model errors raise.
    """

    specs = _parameters(parameters)
    names = tuple(parameter.name for parameter in specs)
    if parameter_name not in names:
        raise KeyError(f"unknown parameter {parameter_name!r}; expected one of {names}")
    swept_index = names.index(parameter_name)
    try:
        sweep_values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("values must be a one-dimensional real array") from error
    if sweep_values.ndim != 1 or sweep_values.size == 0:
        raise ValueError("values must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(sweep_values)):
        raise ValueError("sweep values must be finite")
    swept_spec = specs[swept_index]
    if np.any(sweep_values < swept_spec.lower) or np.any(
        sweep_values > swept_spec.upper
    ):
        raise ValueError(
            f"sweep values for {parameter_name!r} must lie within "
            f"[{swept_spec.lower}, {swept_spec.upper}]"
        )

    samples = np.tile(
        np.asarray([parameter.nominal for parameter in specs], dtype=np.float64),
        (sweep_values.size, 1),
    )
    samples[:, swept_index] = sweep_values
    outputs = np.empty(sweep_values.size, dtype=np.float64)
    for index, row in enumerate(samples):
        outputs[index] = _evaluate_model(
            model, specs, row, f"parameter sweep sample {index}"
        )

    return ParameterSweepResult(
        parameter_names=names,
        swept_parameter=parameter_name,
        samples=_readonly_float(samples),
        outputs=_readonly_float(outputs),
    )


@dataclass(frozen=True)
class LocalSensitivityResult:
    """Central finite-difference derivatives and normalized sensitivities."""

    parameter_names: Tuple[str, ...]
    baseline_output: float
    step_sizes: FloatArray
    derivatives: FloatArray
    normalized_sensitivities: FloatArray
    normalization: Normalization

    @property
    def derivative_by_parameter(self) -> Dict[str, float]:
        return dict(zip(self.parameter_names, self.derivatives.tolist()))

    @property
    def normalized_by_parameter(self) -> Dict[str, float]:
        return dict(zip(self.parameter_names, self.normalized_sensitivities.tolist()))


def local_sensitivity(
    model: Model,
    parameters: Sequence[ParameterRange],
    *,
    relative_step: float = 1.0e-4,
    absolute_steps: Optional[Mapping[str, float]] = None,
    normalization: Normalization = "elasticity",
) -> LocalSensitivityResult:
    """Compute central local sensitivities at the nominal parameter vector.

    ``elasticity`` reports ``(x_i / y) * dy/dx_i``.  ``range`` reports
    ``((upper_i - lower_i) / y) * dy/dx_i``.  Both are dimensionless and both
    require a nonzero baseline output.  Elasticity additionally requires
    nonzero nominal inputs.  No one-sided fallback is silently substituted.
    """

    specs = _parameters(parameters)
    names = tuple(parameter.name for parameter in specs)
    step_fraction = _finite_real(relative_step, "relative_step")
    if step_fraction <= 0.0:
        raise ValueError("relative_step must be greater than zero")
    if normalization not in ("elasticity", "range"):
        raise ValueError("normalization must be 'elasticity' or 'range'")

    requested_steps = dict(absolute_steps or {})
    unknown_steps = set(requested_steps).difference(names)
    if unknown_steps:
        raise KeyError(f"absolute_steps contains unknown parameters: {unknown_steps}")

    nominal = np.asarray([parameter.nominal for parameter in specs], dtype=np.float64)
    baseline = _evaluate_model(model, specs, nominal, "nominal baseline")
    if baseline == 0.0:
        raise ValueError(
            "normalized local sensitivity is undefined for a zero baseline output"
        )
    if normalization == "elasticity":
        zero_names = [parameter.name for parameter in specs if parameter.nominal == 0.0]
        if zero_names:
            raise ValueError(
                "elasticity normalization requires nonzero nominal values; "
                f"use normalization='range' for {zero_names}"
            )

    steps = np.empty(len(specs), dtype=np.float64)
    derivatives = np.empty(len(specs), dtype=np.float64)
    normalized = np.empty(len(specs), dtype=np.float64)

    for index, parameter in enumerate(specs):
        if parameter.name in requested_steps:
            step = _finite_real(
                requested_steps[parameter.name],
                f"absolute_steps[{parameter.name!r}]",
            )
            if step <= 0.0:
                raise ValueError(
                    f"absolute step for {parameter.name!r} must be greater than zero"
                )
        else:
            scale = max(
                abs(parameter.nominal), 0.5 * (parameter.upper - parameter.lower)
            )
            step = step_fraction * scale

        lower_value = parameter.nominal - step
        upper_value = parameter.nominal + step
        if lower_value < parameter.lower or upper_value > parameter.upper:
            distance = min(
                parameter.nominal - parameter.lower,
                parameter.upper - parameter.nominal,
            )
            raise ValueError(
                f"central step {step:.17g} for {parameter.name!r} exceeds its "
                f"range around the nominal value; maximum admissible step is {distance:.17g}"
            )
        if lower_value == parameter.nominal or upper_value == parameter.nominal:
            raise ValueError(
                f"central step for {parameter.name!r} is below floating-point resolution"
            )

        lower_sample = nominal.copy()
        upper_sample = nominal.copy()
        lower_sample[index] = lower_value
        upper_sample[index] = upper_value
        lower_output = _evaluate_model(
            model, specs, lower_sample, f"{parameter.name} lower perturbation"
        )
        upper_output = _evaluate_model(
            model, specs, upper_sample, f"{parameter.name} upper perturbation"
        )
        derivative = (upper_output - lower_output) / (2.0 * step)
        scale = (
            parameter.nominal
            if normalization == "elasticity"
            else parameter.upper - parameter.lower
        )
        steps[index] = step
        derivatives[index] = derivative
        normalized[index] = derivative * scale / baseline

    return LocalSensitivityResult(
        parameter_names=names,
        baseline_output=baseline,
        step_sizes=_readonly_float(steps),
        derivatives=_readonly_float(derivatives),
        normalized_sensitivities=_readonly_float(normalized),
        normalization=normalization,
    )


@dataclass(frozen=True)
class SampleDesign:
    """A reproducible sample matrix with its parameter metadata."""

    parameters: Tuple[ParameterRange, ...]
    samples: FloatArray
    unit_samples: FloatArray
    seed: int
    method: str = "latin_hypercube"

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    def values_at(self, index: int) -> Dict[str, float]:
        """Return one physical sample as an insertion-ordered dictionary."""
        if not -self.n_samples <= index < self.n_samples:
            raise IndexError("sample index out of range")
        return _parameter_mapping(self.parameters, self.samples[index])


def latin_hypercube(
    parameters: Sequence[ParameterRange],
    n_samples: int,
    *,
    seed: int = 0,
) -> SampleDesign:
    """Generate an independent-marginal Latin-hypercube design.

    Each parameter has exactly one point in every equal-probability stratum.
    The fixed default seed and explicit PCG64 generator make repeated calls
    reproducible.  This routine does not model correlation between parameters.
    """

    specs = _parameters(parameters)
    count = _positive_integer(n_samples, "n_samples")
    seed_value = _seed(seed)
    generator = np.random.Generator(np.random.PCG64(seed_value))
    unit = np.empty((count, len(specs)), dtype=np.float64)
    samples = np.empty_like(unit)

    strata = np.arange(count, dtype=np.float64)
    for column, parameter in enumerate(specs):
        coordinate = (strata + generator.random(count)) / count
        generator.shuffle(coordinate)
        unit[:, column] = coordinate
        if parameter.distribution == "uniform":
            samples[:, column] = parameter.lower + coordinate * (
                parameter.upper - parameter.lower
            )
        else:
            log_lower = np.log(parameter.lower)
            samples[:, column] = np.exp(
                log_lower + coordinate * (np.log(parameter.upper) - log_lower)
            )

    return SampleDesign(
        parameters=specs,
        samples=_readonly_float(samples),
        unit_samples=_readonly_float(unit),
        seed=seed_value,
    )


@dataclass(frozen=True)
class EvaluationFailure:
    """One explicitly recorded failed model evaluation."""

    sample_index: int
    parameter_values: Tuple[float, ...]
    exception_type: str
    message: str

    def parameters(self, names: Sequence[str]) -> Dict[str, float]:
        if len(names) != len(self.parameter_values):
            raise ValueError("names must match the recorded parameter count")
        return dict(zip(names, self.parameter_values))


@dataclass(frozen=True)
class UncertaintyResult:
    """Finite output sample and explicit evaluation-failure accounting."""

    design: SampleDesign
    successful_indices: IntArray
    outputs: FloatArray
    quantile_probabilities: FloatArray
    quantile_values: FloatArray
    mean: float
    standard_deviation: float
    failures: Tuple[EvaluationFailure, ...]
    quantile_method: str = "linear"

    @property
    def n_attempted(self) -> int:
        return self.design.n_samples

    @property
    def n_successful(self) -> int:
        return int(self.outputs.size)

    @property
    def n_failed(self) -> int:
        return len(self.failures)

    @property
    def failure_fraction(self) -> float:
        return self.n_failed / self.n_attempted

    @property
    def successful_samples(self) -> FloatArray:
        result = self.design.samples[self.successful_indices].copy()
        result.setflags(write=False)
        return result

    @property
    def quantiles(self) -> Dict[float, float]:
        return dict(
            zip(self.quantile_probabilities.tolist(), self.quantile_values.tolist())
        )


def _quantile_probabilities(values: ArrayLike) -> FloatArray:
    try:
        probabilities = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("quantiles must be a one-dimensional real array") from error
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("quantiles must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("quantiles must be finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("quantiles must lie within [0, 1]")
    if np.any(np.diff(probabilities) <= 0.0):
        raise ValueError("quantiles must be strictly increasing")
    return _readonly_float(probabilities)


def propagate_uncertainty(
    model: Model,
    parameters: Sequence[ParameterRange],
    n_samples: int,
    *,
    seed: int = 0,
    quantiles: ArrayLike = (0.025, 0.5, 0.975),
    failure_policy: FailurePolicy = "raise",
) -> UncertaintyResult:
    """Propagate declared independent input distributions through ``model``.

    The returned mean, sample standard deviation (``ddof=1``), and linearly
    interpolated empirical quantiles summarize the finite successful outputs.
    By default the first model failure raises.  ``failure_policy="record"`` is
    an explicit opt-in to continue and retain every failure with its sample
    index and parameter values.  At least two finite outputs are always
    required; failed evaluations are never silently imputed.
    """

    if failure_policy not in ("raise", "record"):
        raise ValueError("failure_policy must be 'raise' or 'record'")
    probabilities = _quantile_probabilities(quantiles)
    design = latin_hypercube(parameters, n_samples, seed=seed)
    outputs = []
    successful_indices = []
    failures = []

    for index, row in enumerate(design.samples):
        try:
            output = _evaluate_model(
                model, design.parameters, row, f"uncertainty sample {index}"
            )
        except ModelEvaluationError as error:
            if failure_policy == "raise":
                raise
            failures.append(
                EvaluationFailure(
                    sample_index=index,
                    parameter_values=tuple(float(value) for value in row),
                    exception_type=error.exception_type,
                    message=error.reason,
                )
            )
        else:
            successful_indices.append(index)
            outputs.append(output)

    if len(outputs) < 2:
        raise RuntimeError(
            "uncertainty propagation requires at least two finite successful "
            f"outputs; got {len(outputs)} of {design.n_samples}"
        )

    output_array = np.asarray(outputs, dtype=np.float64)
    quantile_values = np.quantile(output_array, probabilities, method="linear")
    return UncertaintyResult(
        design=design,
        successful_indices=_readonly_int(successful_indices),
        outputs=_readonly_float(output_array),
        quantile_probabilities=probabilities,
        quantile_values=_readonly_float(quantile_values),
        mean=float(np.mean(output_array)),
        standard_deviation=float(np.std(output_array, ddof=1)),
        failures=tuple(failures),
    )


@dataclass(frozen=True)
class RegressionScreeningResult:
    """Standardized linear-regression coefficients and adequacy diagnostics."""

    parameter_names: Tuple[str, ...]
    coefficients: FloatArray
    r_squared: float
    adjusted_r_squared: float
    standardized_rmse: float
    design_rank: int
    condition_number: float
    singular_values: FloatArray
    n_samples: int
    log_transformed: Tuple[bool, ...]

    @property
    def coefficient_by_parameter(self) -> Dict[str, float]:
        return dict(zip(self.parameter_names, self.coefficients.tolist()))

    @property
    def absolute_ranking(self) -> Tuple[str, ...]:
        order = np.argsort(-np.abs(self.coefficients), kind="stable")
        return tuple(self.parameter_names[int(index)] for index in order)


ScreeningData = Union[SampleDesign, UncertaintyResult]


def standardized_regression_coefficients(
    data: ScreeningData,
    outputs: Optional[ArrayLike] = None,
    *,
    max_condition_number: float = 1.0e8,
) -> RegressionScreeningResult:
    """Fit standardized regression coefficients as a global screening metric.

    Pass either a ``SampleDesign`` plus its finite scalar outputs, or an
    ``UncertaintyResult`` (whose successful rows and outputs are used).  Inputs
    declared ``log_uniform`` are regressed in log coordinates.  Rank-deficient,
    underdetermined, constant-output, non-finite, and excessively ill-conditioned
    problems raise instead of producing unstable rankings.

    Coefficient magnitude is a linear-association screening metric conditional
    on the design.  ``r_squared`` describes the in-sample linear surrogate fit;
    it is not a validation score and coefficients do not establish causality.
    """

    limit = _finite_real(max_condition_number, "max_condition_number")
    if limit <= 1.0:
        raise ValueError("max_condition_number must be greater than one")

    if isinstance(data, UncertaintyResult):
        if outputs is not None:
            raise TypeError("outputs must be omitted when data is UncertaintyResult")
        parameters = data.design.parameters
        samples = np.asarray(data.successful_samples, dtype=np.float64)
        output_array = np.asarray(data.outputs, dtype=np.float64)
    elif isinstance(data, SampleDesign):
        if outputs is None:
            raise TypeError("outputs are required when data is SampleDesign")
        parameters = data.parameters
        samples = np.asarray(data.samples, dtype=np.float64)
        try:
            output_array = np.asarray(outputs, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError("outputs must be a one-dimensional real array") from error
    else:
        raise TypeError("data must be a SampleDesign or UncertaintyResult")

    if samples.ndim != 2 or samples.shape[1] != len(parameters):
        raise ValueError("sample matrix shape does not match parameter metadata")
    if output_array.ndim != 1:
        raise ValueError("outputs must be one-dimensional")
    if output_array.size != samples.shape[0]:
        raise ValueError("outputs length must equal the number of sample rows")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(output_array)):
        raise ValueError("samples and outputs must be finite")

    n_rows, n_parameters = samples.shape
    if n_rows < n_parameters + 2:
        raise ValueError(
            "standardized regression requires at least p + 2 rows "
            f"(got n={n_rows}, p={n_parameters})"
        )

    transformed = samples.copy()
    log_transformed = tuple(
        parameter.distribution == "log_uniform" for parameter in parameters
    )
    for column, use_log in enumerate(log_transformed):
        if use_log:
            if np.any(transformed[:, column] <= 0.0):
                raise ValueError("log-transformed samples must be positive")
            transformed[:, column] = np.log(transformed[:, column])

    input_std = np.std(transformed, axis=0, ddof=1)
    if np.any(input_std == 0.0):
        constant = [
            parameters[index].name
            for index in np.flatnonzero(input_std == 0.0).tolist()
        ]
        raise ValueError(f"constant sampled parameters cannot be screened: {constant}")
    output_std = float(np.std(output_array, ddof=1))
    if output_std == 0.0:
        raise ValueError("standardized regression is undefined for constant outputs")

    standardized_inputs = (transformed - np.mean(transformed, axis=0)) / input_std
    standardized_outputs = (output_array - np.mean(output_array)) / output_std
    singular_values = np.linalg.svd(standardized_inputs, compute_uv=False)
    rank = int(np.linalg.matrix_rank(standardized_inputs))
    if rank != n_parameters:
        raise ValueError(
            f"standardized design is rank deficient (rank={rank}, p={n_parameters})"
        )
    condition_number = float(singular_values[0] / singular_values[-1])
    if not np.isfinite(condition_number) or condition_number > limit:
        raise ValueError(
            "standardized design is too ill-conditioned for stable screening "
            f"(condition_number={condition_number:.6g}, limit={limit:.6g})"
        )

    coefficients, _residuals, fitted_rank, _fit_singular = np.linalg.lstsq(
        standardized_inputs, standardized_outputs, rcond=None
    )
    if int(fitted_rank) != n_parameters:
        raise RuntimeError("least-squares solver reported an unexpected rank loss")
    fitted = standardized_inputs @ coefficients
    residual = standardized_outputs - fitted
    residual_sum_squares = float(residual @ residual)
    total_sum_squares = float(standardized_outputs @ standardized_outputs)
    r_squared = min(1.0, 1.0 - residual_sum_squares / total_sum_squares)
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n_rows - 1) / (
        n_rows - n_parameters - 1
    )
    standardized_rmse = float(np.sqrt(np.mean(residual**2)))

    return RegressionScreeningResult(
        parameter_names=tuple(parameter.name for parameter in parameters),
        coefficients=_readonly_float(coefficients),
        r_squared=r_squared,
        adjusted_r_squared=adjusted_r_squared,
        standardized_rmse=standardized_rmse,
        design_rank=rank,
        condition_number=condition_number,
        singular_values=_readonly_float(singular_values),
        n_samples=n_rows,
        log_transformed=log_transformed,
    )


__all__ = [
    "EvaluationFailure",
    "LocalSensitivityResult",
    "Model",
    "ModelEvaluationError",
    "ParameterRange",
    "ParameterSweepResult",
    "RegressionScreeningResult",
    "SampleDesign",
    "UncertaintyResult",
    "latin_hypercube",
    "local_sensitivity",
    "parameter_sweep",
    "propagate_uncertainty",
    "standardized_regression_coefficients",
]
