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
    Callable,
    Dict,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    Union,
    cast,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
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


def _real_array(values: ArrayLike, name: str) -> np.ndarray:
    """Convert real numeric data without silently dropping imaginary parts."""
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{name} must not contain masked values")
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a real numeric array") from error
    if np.iscomplexobj(raw) or raw.dtype.kind in ("b", "S", "U"):
        raise TypeError(f"{name} must be a real numeric array")
    if raw.dtype.kind == "O":
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in raw.flat
        ):
            raise TypeError(f"{name} must be a real numeric array")
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a real numeric array") from error


def _readonly_float(values: ArrayLike) -> FloatArray:
    result = np.array(values, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _readonly_int(values: ArrayLike) -> IntArray:
    result = np.array(values, dtype=np.int64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _integer_array(values: ArrayLike, name: str) -> np.ndarray:
    """Convert genuine integer data without truncation or mask loss."""
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{name} must not contain masked values")
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be an integer array") from error
    if raw.dtype.kind in ("i", "u"):
        pass
    elif raw.dtype.kind == "O" and all(
        isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        for value in raw.flat
    ):
        pass
    else:
        raise TypeError(f"{name} must be an integer array")
    try:
        return np.asarray(values, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be an integer array") from error


def _parameter_name_tuple(
    values: object, name: str = "parameter_names"
) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    try:
        result: Tuple[object, ...] = tuple(cast(Iterable[object], values))
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of strings") from error
    if not result:
        raise ValueError(f"{name} must not be empty")
    for value in result:
        if not isinstance(value, str):
            raise TypeError(f"{name} must contain only strings")
        if not value or value != value.strip():
            raise ValueError(
                f"{name} entries must be nonempty without outer whitespace"
            )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    return cast(Tuple[str, ...], result)


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty without outer whitespace")
    return value


def _scaled_mean_and_sample_std(values: np.ndarray) -> Tuple[float, float]:
    """Compute finite float64 sample statistics without squaring raw magnitudes."""
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0, 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = values / scale
        mean = float(np.mean(scaled) * scale)
        standard_deviation = float(np.std(scaled, ddof=1) * scale)
    if not np.isfinite(mean) or not np.isfinite(standard_deviation):
        raise OverflowError(
            "uncertainty summary statistics became non-finite; rescale the "
            "quantity of interest"
        )
    return mean, standard_deviation


def _scaled_quantiles(
    values: FloatArray,
    probabilities: FloatArray,
    *,
    method: Literal["linear"],
) -> FloatArray:
    """Compute empirical quantiles without interpolating raw extreme magnitudes."""
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return np.zeros(probabilities.shape, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        quantiles = (
            np.asarray(
                np.quantile(values / scale, probabilities, method=method),
                dtype=np.float64,
            )
            * scale
        )
    if not np.all(np.isfinite(quantiles)):
        raise OverflowError(
            "uncertainty summary statistics became non-finite; rescale the "
            "quantity of interest"
        )
    return quantiles


def _scaled_standardize_columns(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize matrix columns without under/overflow in raw variance."""
    scales = np.max(np.abs(values), axis=0)
    safe_scales = np.where(scales == 0.0, 1.0, scales)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = values / safe_scales
        means = np.mean(scaled, axis=0)
        centered = scaled - means
        standard_deviations = np.sqrt(
            np.sum(centered * centered, axis=0) / (values.shape[0] - 1)
        )
        standardized = centered / standard_deviations
    return standardized, standard_deviations


def _scaled_standardize_vector(values: np.ndarray) -> Tuple[np.ndarray, float]:
    """Standardize one vector without under/overflow in raw variance."""
    scale = float(np.max(np.abs(values)))
    safe_scale = 1.0 if scale == 0.0 else scale
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = values / safe_scale
        centered = scaled - np.mean(scaled)
        standard_deviation = float(
            np.sqrt(np.sum(centered * centered) / (values.size - 1))
        )
        standardized = centered / standard_deviation
    return standardized, standard_deviation


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
        if not np.isfinite(upper - lower):
            raise ValueError(f"{self.name}: parameter range width must be finite")
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
    try:
        result = tuple(values)
    except TypeError as error:
        raise TypeError(
            "parameters must be a sequence of ParameterRange values"
        ) from error
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
    value_tuple = tuple(values)
    if len(value_tuple) != len(parameters):
        raise ValueError("parameter values must match the parameter metadata")
    return {
        parameter.name: float(value)
        for parameter, value in zip(parameters, value_tuple)
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

    if isinstance(raw_value, (bool, np.bool_)) or not isinstance(raw_value, Real):
        raise ModelEvaluationError(
            context,
            names,
            value_tuple,
            type(raw_value).__name__,
            "model must return one real scalar quantity of interest",
        )
    try:
        result = float(raw_value)
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

    def __post_init__(self) -> None:
        names = _parameter_name_tuple(self.parameter_names)
        swept_parameter = _nonempty_text(self.swept_parameter, "swept_parameter")
        if swept_parameter not in names:
            raise ValueError("swept_parameter must be present in parameter_names")
        samples = _real_array(self.samples, "samples")
        outputs = _real_array(self.outputs, "outputs")
        if samples.ndim != 2:
            raise ValueError("samples must be two-dimensional")
        if outputs.ndim != 1:
            raise ValueError("outputs must be one-dimensional")
        if samples.shape[0] == 0 or samples.shape[1] != len(names):
            raise ValueError("samples shape must match nonempty parameter metadata")
        if outputs.size != samples.shape[0]:
            raise ValueError("outputs length must equal the number of sample rows")
        if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(outputs)):
            raise ValueError("samples and outputs must be finite")

        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "swept_parameter", swept_parameter)
        object.__setattr__(self, "samples", _readonly_float(samples))
        object.__setattr__(self, "outputs", _readonly_float(outputs))

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
    sweep_values = _real_array(values, "values")
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

    def __post_init__(self) -> None:
        names = _parameter_name_tuple(self.parameter_names)
        baseline = _finite_real(self.baseline_output, "baseline_output")
        if baseline == 0.0:
            raise ValueError("baseline_output must be nonzero")
        step_sizes = _real_array(self.step_sizes, "step_sizes")
        derivatives = _real_array(self.derivatives, "derivatives")
        normalized = _real_array(
            self.normalized_sensitivities, "normalized_sensitivities"
        )
        for name, values in (
            ("step_sizes", step_sizes),
            ("derivatives", derivatives),
            ("normalized_sensitivities", normalized),
        ):
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if values.size != len(names):
                raise ValueError(f"{name} length must match parameter_names")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
        if np.any(step_sizes <= 0.0):
            raise ValueError("step_sizes must be greater than zero")
        if self.normalization not in ("elasticity", "range"):
            raise ValueError("normalization must be 'elasticity' or 'range'")

        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "baseline_output", baseline)
        object.__setattr__(self, "step_sizes", _readonly_float(step_sizes))
        object.__setattr__(self, "derivatives", _readonly_float(derivatives))
        object.__setattr__(
            self, "normalized_sensitivities", _readonly_float(normalized)
        )

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
        normalized_value = derivative * scale / baseline
        if not np.isfinite(derivative) or not np.isfinite(normalized_value):
            raise ValueError(
                f"local sensitivity for {parameter.name!r} overflowed or became "
                "non-finite; rescale the model output or parameter"
            )
        steps[index] = step
        derivatives[index] = derivative
        normalized[index] = normalized_value

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
    method: str = "user_supplied"

    def __post_init__(self) -> None:
        specs = _parameters(self.parameters)
        samples = _real_array(self.samples, "samples")
        unit_samples = _real_array(self.unit_samples, "unit_samples")
        if samples.ndim != 2 or unit_samples.ndim != 2:
            raise ValueError("samples and unit_samples must be two-dimensional")
        if samples.shape != unit_samples.shape:
            raise ValueError("samples and unit_samples must have matching shapes")
        if samples.shape[0] == 0 or samples.shape[1] != len(specs):
            raise ValueError(
                "sample matrix shape must match nonempty parameter metadata"
            )
        if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(unit_samples)):
            raise ValueError("samples and unit_samples must be finite")
        if np.any(unit_samples < 0.0) or np.any(unit_samples > 1.0):
            raise ValueError("unit_samples must lie within [0, 1]")
        for column, parameter in enumerate(specs):
            if np.any(samples[:, column] < parameter.lower) or np.any(
                samples[:, column] > parameter.upper
            ):
                raise ValueError(
                    f"samples for {parameter.name!r} must lie within its declared range"
                )
        if not isinstance(self.method, str):
            raise TypeError("method must be a string")
        if not self.method or self.method != self.method.strip():
            raise ValueError("method must be nonempty without outer whitespace")

        object.__setattr__(self, "parameters", specs)
        object.__setattr__(self, "samples", _readonly_float(samples))
        object.__setattr__(self, "unit_samples", _readonly_float(unit_samples))
        object.__setattr__(self, "seed", _seed(self.seed))

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    def values_at(self, index: int) -> Dict[str, float]:
        """Return one physical sample as an insertion-ordered dictionary."""
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
            raise TypeError("index must be an integer")
        position = int(index)
        if not -self.n_samples <= position < self.n_samples:
            raise IndexError("sample index out of range")
        return _parameter_mapping(self.parameters, self.samples[position])


def latin_hypercube(
    parameters: Sequence[ParameterRange],
    n_samples: int,
    *,
    seed: int = 0,
) -> SampleDesign:
    """Generate an independent-marginal Latin-hypercube design.

    Each parameter has exactly one point in every equal-probability stratum.
    The fixed default seed and explicit PCG64 generator make repeated calls
    reproducible. Every stratum must also map to a distinct float64 physical
    value; an unrepresentably narrow range raises. This routine does not model
    correlation between parameters.
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
        if np.unique(samples[:, column]).size != count:
            raise ValueError(
                f"{parameter.name}: range cannot represent {count} distinct "
                "Latin-hypercube strata in float64"
            )

    return SampleDesign(
        parameters=specs,
        samples=_readonly_float(samples),
        unit_samples=_readonly_float(unit),
        seed=seed_value,
        method="latin_hypercube",
    )


@dataclass(frozen=True)
class EvaluationFailure:
    """One explicitly recorded failed model evaluation."""

    sample_index: int
    parameter_values: Tuple[float, ...]
    exception_type: str
    message: str

    def __post_init__(self) -> None:
        if isinstance(self.sample_index, (bool, np.bool_)) or not isinstance(
            self.sample_index, Integral
        ):
            raise TypeError("sample_index must be an integer")
        sample_index = int(self.sample_index)
        if sample_index < 0:
            raise ValueError("sample_index must be nonnegative")
        try:
            raw_values = tuple(self.parameter_values)
        except TypeError as error:
            raise TypeError("parameter_values must be a sequence") from error
        if not raw_values:
            raise ValueError("parameter_values must not be empty")
        parameter_values = tuple(
            _finite_real(value, f"parameter_values[{index}]")
            for index, value in enumerate(raw_values)
        )
        exception_type = _nonempty_text(self.exception_type, "exception_type")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")

        object.__setattr__(self, "sample_index", sample_index)
        object.__setattr__(self, "parameter_values", parameter_values)
        object.__setattr__(self, "exception_type", exception_type)

    def parameters(self, names: Sequence[str]) -> Dict[str, float]:
        validated_names = _parameter_name_tuple(names, "names")
        if len(validated_names) != len(self.parameter_values):
            raise ValueError("names must match the recorded parameter count")
        return dict(zip(validated_names, self.parameter_values))


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

    def __post_init__(self) -> None:
        if not isinstance(self.design, SampleDesign):
            raise TypeError("design must be a SampleDesign")
        successful_indices = _integer_array(
            self.successful_indices, "successful_indices"
        )
        outputs = _real_array(self.outputs, "outputs")
        probabilities = _real_array(
            self.quantile_probabilities, "quantile_probabilities"
        )
        quantile_values = _real_array(self.quantile_values, "quantile_values")
        if successful_indices.ndim != 1:
            raise ValueError("successful_indices must be one-dimensional")
        if outputs.ndim != 1:
            raise ValueError("outputs must be one-dimensional")
        if outputs.size < 2:
            raise ValueError("outputs must contain at least two successful values")
        if successful_indices.size != outputs.size:
            raise ValueError("successful_indices length must equal outputs length")
        if not np.all(np.isfinite(outputs)):
            raise ValueError("outputs must be finite")
        if np.any(successful_indices < 0) or np.any(
            successful_indices >= self.design.n_samples
        ):
            raise ValueError("successful_indices are outside the sample design")
        if successful_indices.size > 1 and np.any(np.diff(successful_indices) <= 0):
            raise ValueError("successful_indices must be unique and increasing")

        if probabilities.ndim != 1 or probabilities.size == 0:
            raise ValueError(
                "quantile_probabilities must be a nonempty one-dimensional array"
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("quantile_probabilities must be finite")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError("quantile_probabilities must lie within [0, 1]")
        if np.any(np.diff(probabilities) <= 0.0):
            raise ValueError("quantile_probabilities must be strictly increasing")
        if quantile_values.ndim != 1 or quantile_values.size != probabilities.size:
            raise ValueError(
                "quantile_values must be one-dimensional and match "
                "quantile_probabilities"
            )
        if not np.all(np.isfinite(quantile_values)):
            raise ValueError("quantile_values must be finite")

        mean = _finite_real(self.mean, "mean")
        standard_deviation = _finite_real(self.standard_deviation, "standard_deviation")
        if standard_deviation < 0.0:
            raise ValueError("standard_deviation must be nonnegative")
        quantile_method = _nonempty_text(self.quantile_method, "quantile_method")
        if quantile_method != "linear":
            raise ValueError("quantile_method must be 'linear'")
        linear_quantile_method = cast(Literal["linear"], quantile_method)

        try:
            failures = tuple(self.failures)
        except TypeError as error:
            raise TypeError("failures must be a sequence") from error
        if any(not isinstance(failure, EvaluationFailure) for failure in failures):
            raise TypeError("failures must contain only EvaluationFailure records")
        failed_indices = [failure.sample_index for failure in failures]
        if len(set(failed_indices)) != len(failed_indices):
            raise ValueError("failure sample indices must be unique")
        if set(failed_indices).intersection(successful_indices.tolist()):
            raise ValueError("successful and failed sample indices must be disjoint")
        if set(failed_indices).union(successful_indices.tolist()) != set(
            range(self.design.n_samples)
        ):
            raise ValueError(
                "successful and failed indices must account for every design row"
            )
        for failure in failures:
            if len(failure.parameter_values) != len(self.design.parameters):
                raise ValueError(
                    "failure parameter_values must match design parameter metadata"
                )
            expected_values = tuple(
                float(value) for value in self.design.samples[failure.sample_index]
            )
            if failure.parameter_values != expected_values:
                raise ValueError(
                    "failure parameter_values must match its design sample row"
                )

        expected_mean, expected_std = _scaled_mean_and_sample_std(outputs)
        if not np.isclose(mean, expected_mean, rtol=1.0e-12, atol=0.0):
            raise ValueError("mean is inconsistent with outputs")
        if not np.isclose(standard_deviation, expected_std, rtol=1.0e-12, atol=0.0):
            raise ValueError("standard_deviation is inconsistent with outputs")
        expected_quantiles = _scaled_quantiles(
            outputs, probabilities, method=linear_quantile_method
        )
        if not np.allclose(quantile_values, expected_quantiles, rtol=1.0e-12, atol=0.0):
            raise ValueError("quantile_values are inconsistent with outputs")

        object.__setattr__(
            self, "successful_indices", _readonly_int(successful_indices)
        )
        object.__setattr__(self, "outputs", _readonly_float(outputs))
        object.__setattr__(
            self, "quantile_probabilities", _readonly_float(probabilities)
        )
        object.__setattr__(self, "quantile_values", _readonly_float(quantile_values))
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", standard_deviation)
        object.__setattr__(self, "failures", failures)
        object.__setattr__(self, "quantile_method", linear_quantile_method)

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
    probabilities = _real_array(values, "quantiles")
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
    required, so fewer than two requested samples fail before model evaluation.
    Failed evaluations are never silently imputed. Summary-statistic overflow
    raises with a rescaling recommendation.
    """

    if failure_policy not in ("raise", "record"):
        raise ValueError("failure_policy must be 'raise' or 'record'")
    probabilities = _quantile_probabilities(quantiles)
    count = _positive_integer(n_samples, "n_samples")
    if count < 2:
        raise ValueError(
            "uncertainty propagation requires at least two requested samples"
        )
    design = latin_hypercube(parameters, count, seed=seed)
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
    mean, standard_deviation = _scaled_mean_and_sample_std(output_array)
    quantile_values = _scaled_quantiles(output_array, probabilities, method="linear")
    return UncertaintyResult(
        design=design,
        successful_indices=_readonly_int(successful_indices),
        outputs=_readonly_float(output_array),
        quantile_probabilities=probabilities,
        quantile_values=_readonly_float(quantile_values),
        mean=mean,
        standard_deviation=standard_deviation,
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

    def __post_init__(self) -> None:
        names = _parameter_name_tuple(self.parameter_names)
        coefficients = _real_array(self.coefficients, "coefficients")
        singular_values = _real_array(self.singular_values, "singular_values")
        if coefficients.ndim != 1 or coefficients.size != len(names):
            raise ValueError(
                "coefficients must be one-dimensional and match parameter_names"
            )
        if singular_values.ndim != 1 or singular_values.size != len(names):
            raise ValueError(
                "singular_values must be one-dimensional and match parameter_names"
            )
        if not np.all(np.isfinite(coefficients)) or not np.all(
            np.isfinite(singular_values)
        ):
            raise ValueError("coefficients and singular_values must be finite")
        if np.any(singular_values <= 0.0):
            raise ValueError("singular_values must be greater than zero")
        if np.any(np.diff(singular_values) > 0.0):
            raise ValueError("singular_values must be ordered largest to smallest")

        r_squared = _finite_real(self.r_squared, "r_squared")
        adjusted_r_squared = _finite_real(self.adjusted_r_squared, "adjusted_r_squared")
        standardized_rmse = _finite_real(self.standardized_rmse, "standardized_rmse")
        condition_number = _finite_real(self.condition_number, "condition_number")
        if not 0.0 <= r_squared <= 1.0:
            raise ValueError("r_squared must lie within [0, 1]")
        if adjusted_r_squared > 1.0:
            raise ValueError("adjusted_r_squared must not exceed one")
        if standardized_rmse < 0.0:
            raise ValueError("standardized_rmse must be nonnegative")
        if condition_number < 1.0:
            raise ValueError("condition_number must be at least one")

        design_rank = _positive_integer(self.design_rank, "design_rank")
        n_samples = _positive_integer(self.n_samples, "n_samples")
        if design_rank != len(names):
            raise ValueError("design_rank must equal the parameter count")
        if n_samples < len(names) + 2:
            raise ValueError("n_samples must be at least parameter count plus two")
        expected_adjusted = 1.0 - (1.0 - r_squared) * (n_samples - 1) / (
            n_samples - len(names) - 1
        )
        expected_rmse_squared = (1.0 - r_squared) * (n_samples - 1) / n_samples
        supplied_rmse_squared = standardized_rmse * standardized_rmse
        summary_atol = 64.0 * np.finfo(float).eps
        if not np.isclose(
            adjusted_r_squared,
            expected_adjusted,
            rtol=1.0e-10,
            atol=summary_atol,
        ):
            raise ValueError(
                "adjusted_r_squared is inconsistent with r_squared, "
                "n_samples, and parameter count"
            )
        if not np.isclose(
            supplied_rmse_squared,
            expected_rmse_squared,
            rtol=1.0e-10,
            atol=summary_atol,
        ):
            raise ValueError(
                "standardized_rmse is inconsistent with r_squared and n_samples"
            )
        expected_condition = float(singular_values[0] / singular_values[-1])
        if not np.isclose(condition_number, expected_condition, rtol=1.0e-12, atol=0.0):
            raise ValueError("condition_number is inconsistent with singular_values")

        try:
            raw_log_transformed = tuple(self.log_transformed)
        except TypeError as error:
            raise TypeError("log_transformed must be a sequence of booleans") from error
        if len(raw_log_transformed) != len(names) or any(
            not isinstance(value, (bool, np.bool_)) for value in raw_log_transformed
        ):
            raise ValueError("log_transformed must contain one boolean per parameter")

        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "coefficients", _readonly_float(coefficients))
        object.__setattr__(self, "r_squared", r_squared)
        object.__setattr__(self, "adjusted_r_squared", adjusted_r_squared)
        object.__setattr__(self, "standardized_rmse", standardized_rmse)
        object.__setattr__(self, "design_rank", design_rank)
        object.__setattr__(self, "condition_number", condition_number)
        object.__setattr__(self, "singular_values", _readonly_float(singular_values))
        object.__setattr__(self, "n_samples", n_samples)
        object.__setattr__(
            self,
            "log_transformed",
            tuple(bool(value) for value in raw_log_transformed),
        )

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
        output_array = _real_array(outputs, "outputs")
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

    standardized_inputs, input_std = _scaled_standardize_columns(transformed)
    standardized_outputs, output_std = _scaled_standardize_vector(output_array)
    if not np.all(np.isfinite(input_std)) or not np.isfinite(output_std):
        raise ValueError(
            "standardization statistics became non-finite; rescale samples or outputs"
        )
    if np.any(input_std == 0.0):
        constant = [
            parameters[index].name
            for index in np.flatnonzero(input_std == 0.0).tolist()
        ]
        raise ValueError(f"constant sampled parameters cannot be screened: {constant}")
    if output_std == 0.0:
        raise ValueError("standardized regression is undefined for constant outputs")

    if not np.all(np.isfinite(standardized_inputs)) or not np.all(
        np.isfinite(standardized_outputs)
    ):
        raise ValueError("standardized samples or outputs became non-finite")
    singular_values = np.linalg.svd(standardized_inputs, compute_uv=False)
    if not np.all(np.isfinite(singular_values)):
        raise ValueError("standardized design singular values are non-finite")
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
    if (
        not np.all(np.isfinite(coefficients))
        or not np.isfinite(residual_sum_squares)
        or not np.isfinite(total_sum_squares)
        or total_sum_squares <= 0.0
    ):
        raise ValueError("standardized regression produced non-finite diagnostics")
    r_squared = float(np.clip(1.0 - residual_sum_squares / total_sum_squares, 0.0, 1.0))
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
