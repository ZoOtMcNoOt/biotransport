"""Headless sensitivity screening around the native C++ transport solver.

The scalar quantity of interest (QoI) is midpoint concentration after diffusion
and first-order decay in a one-dimensional slab.  All statistics are conditional
on the illustrative independent parameter distributions below.  This example
does not validate the physical model or establish causal effects.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

import biotransport as bt
from biotransport.analysis import (
    ParameterRange,
    local_sensitivity,
    parameter_sweep,
    propagate_uncertainty,
    standardized_regression_coefficients,
)


def main() -> None:
    mesh = bt.mesh_1d(41, x_max=1.0)
    x = bt.x_nodes(mesh)
    midpoint = len(x) // 2
    end_time = 0.05

    parameters = [
        ParameterRange(
            "diffusivity",
            nominal=0.010,
            lower=0.005,
            upper=0.020,
            distribution="uniform",
        ),
        ParameterRange(
            "decay_rate",
            nominal=0.30,
            lower=0.10,
            upper=0.60,
            distribution="uniform",
        ),
        ParameterRange(
            "initial_amplitude",
            nominal=1.0,
            lower=0.8,
            upper=1.2,
            distribution="uniform",
        ),
    ]

    def midpoint_concentration(values: Mapping[str, float]) -> float:
        """Run the native solver and reduce its field to one scalar QoI."""
        initial = values["initial_amplitude"] * np.sin(np.pi * x)
        problem = (
            bt.Problem(mesh)
            .diffusivity(values["diffusivity"])
            .linear_decay(values["decay_rate"])
            .initial_condition(initial)
            .dirichlet(bt.Boundary.Left, 0.0)
            .dirichlet(bt.Boundary.Right, 0.0)
        )
        result = bt.solve(problem, end_time=end_time)
        return float(result.concentration[midpoint])

    sweep = parameter_sweep(
        midpoint_concentration,
        parameters,
        "diffusivity",
        np.linspace(0.005, 0.020, 5),
    )
    local = local_sensitivity(midpoint_concentration, parameters)
    uncertainty = propagate_uncertainty(
        midpoint_concentration,
        parameters,
        n_samples=96,
        seed=341,
        quantiles=(0.05, 0.5, 0.95),
    )
    screening = standardized_regression_coefficients(uncertainty)

    print("Sensitivity and uncertainty screening")
    print("-------------------------------------")
    print("Diffusivity sweep (D, midpoint concentration):")
    for value, output in zip(sweep.swept_values, sweep.outputs):
        print(f"  {value:.5f}  {output:.8f}")

    print("\nLocal elasticities at nominal inputs:")
    for name, value in local.normalized_by_parameter.items():
        print(f"  {name:>18}: {value:+.6f}")

    print(f"\nSeeded LHS attempts: {uncertainty.n_attempted}")
    print(f"Finite outputs:       {uncertainty.n_successful}")
    print(f"Failed evaluations:   {uncertainty.n_failed}")
    print(f"Sample mean:          {uncertainty.mean:.8f}")
    print(f"Sample std. dev.:     {uncertainty.standard_deviation:.8f}")
    for probability, value in uncertainty.quantiles.items():
        print(f"q={probability:0.3f}:             {value:.8f}")

    print("\nStandardized regression screening coefficients:")
    for name, value in screening.coefficient_by_parameter.items():
        print(f"  {name:>18}: {value:+.6f}")
    print(f"Linear-surrogate R^2: {screening.r_squared:.6f}")
    print(f"Design rank:          {screening.design_rank}")
    print(f"Condition number:     {screening.condition_number:.6f}")

    # These are numerical behavior checks for this example, not validation.
    derivatives = local.derivative_by_parameter
    if derivatives["diffusivity"] >= 0.0 or derivatives["decay_rate"] >= 0.0:
        raise RuntimeError("expected diffusion and decay to reduce the midpoint QoI")
    if derivatives["initial_amplitude"] <= 0.0:
        raise RuntimeError("expected initial amplitude to increase the midpoint QoI")
    if uncertainty.n_failed != 0 or screening.r_squared < 0.99:
        raise RuntimeError(
            "illustrative screening diagnostics did not meet expectations"
        )

    print("\nExample checks passed (screening only; no validation claim).")


if __name__ == "__main__":
    main()
