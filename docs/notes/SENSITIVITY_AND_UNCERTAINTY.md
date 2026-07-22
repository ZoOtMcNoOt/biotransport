# Sensitivity and uncertainty screening

`biotransport.analysis` provides a reproducible orchestration layer around a
user-defined scalar quantity of interest (QoI). The model callable can build and
run a native C++ BioTransport solver, then return one finite scalar such as total
mass, outlet flux, maximum temperature, or concentration at a stated location.

These workflows answer conditional questions such as “how does this QoI change
over these declared parameter ranges?” They do **not** validate a biological
model, establish causality, infer parameter distributions from data, or certify
compliance with JCGM or ASME procedures.

## Minimal workflow

```python
from typing import Mapping

from biotransport.analysis import (
    ParameterRange,
    local_sensitivity,
    propagate_uncertainty,
    standardized_regression_coefficients,
)

parameters = [
    ParameterRange("D", nominal=1.0e-9, lower=0.7e-9, upper=1.3e-9),
    ParameterRange("k", nominal=0.2, lower=0.1, upper=0.4),
]

def quantity_of_interest(values: Mapping[str, float]) -> float:
    # Construct a BioTransport problem, run its native solver, and reduce the
    # returned field to one explicitly defined scalar QoI.
    return run_model(D=values["D"], k=values["k"])

local = local_sensitivity(quantity_of_interest, parameters)
uncertainty = propagate_uncertainty(
    quantity_of_interest,
    parameters,
    n_samples=256,
    seed=2026,
    quantiles=(0.025, 0.5, 0.975),
)
screening = standardized_regression_coefficients(uncertainty)
```

Parameter order is the order supplied by the caller and is retained in every
sample matrix and result. Names must be unique. Bounds, nominal values, model
outputs, step sizes, seeds, shapes, ranks, and conditioning are checked before a
result is reported.

## What each operation computes

### Deterministic parameter sweep

`parameter_sweep` changes one named parameter through the caller-provided values
in their original order. Every other parameter remains at its nominal value.
The operation is one-at-a-time and local to that baseline: it does not capture
interactions among simultaneously changing inputs.

### Central local sensitivity

For parameter (x_i), the numerical derivative is

\[
\frac{\partial y}{\partial x_i}
\approx
\frac{y(x_i+h_i)-y(x_i-h_i)}{2h_i}.
\]

The default step is (10^{-4}) times the larger of the nominal magnitude and
half the declared range. `absolute_steps` can override individual steps. Both
perturbations must remain inside the declared bounds. A nominal value on a bound
therefore raises; the routine never silently substitutes a one-sided formula.

Two dimensionless normalizations are available:

- `normalization="elasticity"` (default):
  ((x_i/y)(\partial y/\partial x_i)). For a local power law it is the local
  exponent. It requires nonzero nominal inputs and a nonzero baseline QoI.
- `normalization="range"`:
  (((x_{i,max}-x_{i,min})/y)(\partial y/\partial x_i)). This supports a zero
  nominal input but still requires a nonzero baseline QoI.

Sensitivity to the chosen step should be checked for expensive, noisy, or
nonsmooth models. A derivative at one nominal point is not a global importance
measure.

### Latin-hypercube design

`latin_hypercube` uses an explicit NumPy PCG64 generator. The default seed is
fixed at zero, and scientific artifacts should still state the seed explicitly.
For (N) samples, every parameter has one randomly jittered point in each of
its (N) equal-probability marginal strata, followed by an independent
permutation. Repeating the same call with the same inputs and seed reproduces the
same design.

`distribution="uniform"` is uniform in the physical parameter.
`distribution="log_uniform"` is uniform in its natural logarithm and requires
positive bounds. These distributions are assumptions supplied by the user. The
implementation does not estimate them from observations, and it assumes the
parameter marginals are independent. Do not use this sampler when known input
correlations materially affect the QoI.

Latin-hypercube stratification is marginal, not a guarantee of uniform coverage
of every multidimensional projection. A seed controls the design; it does not
remove finite-sample error. Repeat designs or increase the sample count when
stability of a conclusion matters.

### Uncertainty propagation

`propagate_uncertainty` evaluates a Latin-hypercube design serially and reports:

- the arithmetic sample mean;
- the sample standard deviation with `ddof=1`;
- caller-selected empirical quantiles using linear interpolation;
- attempted, successful, and failed evaluation counts; and
- the seed and complete sample design.

The default `failure_policy="raise"` stops at the first exception, non-scalar
output, or non-finite output and includes the sample index and ordered parameter
values in `ModelEvaluationError`. `failure_policy="record"` is an explicit opt-in
to continue. It retains the type, message, index, and input vector for every
failure and summarizes only successful finite outputs; failed values are never
imputed. At least two successful outputs are required.

If failures are related to parameter values, successful-only summaries can be
selection-biased. Always report the failure fraction and investigate its
location in parameter space. A low failure count is not permission to hide it.

The propagated output distribution is conditional on all of the following:

1. the model equations, geometry, initial and boundary conditions;
2. the numerical method and discretization;
3. the QoI definition;
4. the parameter ranges, marginal distributions, and independence assumption;
5. the sampling seed and finite sample count.

It does not automatically include discretization error, parameter-estimation
error, experimental measurement uncertainty, model-form discrepancy, or
population variability unless those sources are represented explicitly and
defensibly by the caller.

### Standardized regression screening

`standardized_regression_coefficients` fits a multiple linear regression to
standardized inputs and standardized output. Inputs declared `log_uniform` are
first transformed with the natural logarithm. For standardized column (z_i),
the fitted screening model is

\[
z_y = \sum_i \beta_i z_i + \epsilon.
\]

The returned β values are standardized regression coefficients (SRCs). Their
signs describe fitted linear association within this design, and their absolute
magnitudes can screen relative linear influence. The result also reports:

- matrix rank and singular values;
- the standardized design condition number;
- in-sample (R^2) and adjusted (R^2); and
- standardized residual root-mean-square error.

The fit requires at least (p+2) successful rows for (p) parameters, nonzero
input and output variance, full column rank, and a condition number no larger
than `max_condition_number` (default (10^8)). Violations raise rather than
returning an unstable ranking.

SRCs are only defensible as a **linear-association screening metric** for the
sampled design. A low (R^2) indicates that the linear surrogate is inadequate;
small SRCs can then coexist with strong nonlinear or interaction effects. A high
(R^2) does not validate the transport model, and no SRC establishes causality.

## Reporting checklist

For a reproducible scientific artifact, record:

- the code revision, solver, compiler/build options, and platform;
- the QoI definition, location, time, and units;
- every parameter's name, units, nominal value, bounds, distribution, and
  provenance;
- assumptions about independence or correlation;
- mesh/time-step convergence and available conservation residuals;
- local finite-difference steps and normalization;
- sample method, count, seed, and requested quantiles;
- all failed-evaluation counts and reasons;
- SRC rank, condition number, (R^2), and residual diagnostic; and
- the exact scope of any conclusion.

The runnable, plotting-free example is
`examples/verification/sensitivity_and_uncertainty.py`.

## Explicit exclusions

This first workflow does not implement correlated or conditional input
distributions, Sobol indices, Morris screening, Bayesian calibration, surrogate
validation, confidence/coverage claims, automatic combination with numerical
uncertainty, or model discrepancy. Those require additional assumptions and
evidence and must not be inferred from this API.

## Framing references

- Joint Committee for Guides in Metrology, [JCGM 100:2008, *Evaluation of
  measurement data—Guide to the expression of uncertainty in measurement*](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf).
  It provides broad measurement-uncertainty concepts and reporting context. This
  BioTransport sampling workflow is not a GUM evaluation or compliance claim.
- ASME, [V&V 20—Standard for Verification and Validation in Computational Fluid
  Dynamics and Heat Transfer](https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer).
  Its stated scope concerns quantified comparison of specified simulation and
  experimental variables at validation points. This module performs no such
  experimental comparison and therefore makes no ASME validation claim.
