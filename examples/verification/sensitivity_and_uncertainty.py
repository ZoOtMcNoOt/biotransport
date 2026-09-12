"""Headless sensitivity screening around the native C++ transport solver.

The scalar quantity of interest (QoI) is the steady flux entering a
one-dimensional slab that is held at a fixed surface concentration on one face,
sealed on the other, and consumes the solute by first-order reaction.  Its
closed form is

    N = -D dc/dx|_0 = c_s sqrt(D k) tanh(phi),   phi = L sqrt(k / D)

so the QoI depends on diffusivity and decay rate through the Thiele modulus and
does not collapse to a scale factor on any one input.  The nominal point is set
at phi = 1, the crossover between diffusion control and reaction control, where
the three elasticities are all distinct.  All statistics are conditional on the
illustrative independent parameter distributions below.  This example does not
validate the physical model or establish causal effects.
"""

from __future__ import annotations

import math
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

SLAB_THICKNESS = 1.0


def main() -> None:
    mesh = bt.mesh_1d(80, 0.0, SLAB_THICKNESS)
    spacing = mesh.dx()

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
            nominal=0.010,
            lower=0.005,
            upper=0.020,
            distribution="uniform",
        ),
        ParameterRange(
            "surface_concentration",
            nominal=1.0,
            lower=0.8,
            upper=1.2,
            distribution="uniform",
        ),
    ]

    def surface_flux(values: Mapping[str, float]) -> float:
        """Run the native solver and reduce its field to one scalar QoI."""
        diffusivity = values["diffusivity"]
        problem = (
            bt.Problem(mesh)
            .diffusivity(diffusivity)
            .linear_decay(values["decay_rate"])
            .initial(0.0)
            .dirichlet("left", values["surface_concentration"])
            .sealed("right")
        )
        field = np.asarray(bt.solve_steady(problem).c)
        # Second-order one-sided gradient at the exposed face.
        gradient = (-3.0 * field[0] + 4.0 * field[1] - field[2]) / (2.0 * spacing)
        return float(-diffusivity * gradient)

    def exact_flux(values: Mapping[str, float]) -> float:
        """Closed form for the same QoI, used only to check the reduction."""
        modulus = bt.analytical.thiele_modulus(
            D=values["diffusivity"], k=values["decay_rate"], length=SLAB_THICKNESS
        )
        return (
            values["surface_concentration"]
            * math.sqrt(values["diffusivity"] * values["decay_rate"])
            * math.tanh(modulus)
        )

    nominal = {parameter.name: parameter.nominal for parameter in parameters}
    nominal_modulus = bt.analytical.thiele_modulus(
        D=nominal["diffusivity"], k=nominal["decay_rate"], length=SLAB_THICKNESS
    )
    nominal_flux = surface_flux(nominal)
    nominal_exact = exact_flux(nominal)
    qoi_error = abs(nominal_flux - nominal_exact) / nominal_exact

    # The same flux read as a consumption rate: N = k*c_s*L*effectiveness(phi).
    effectiveness = bt.analytical.effectiveness_factor(nominal_modulus, "slab")
    flux_from_effectiveness = (
        nominal["decay_rate"]
        * nominal["surface_concentration"]
        * SLAB_THICKNESS
        * effectiveness
    )

    sweeps = {
        name: parameter_sweep(
            surface_flux, parameters, name, np.linspace(0.005, 0.020, 5)
        )
        for name in ("diffusivity", "decay_rate")
    }
    local = local_sensitivity(surface_flux, parameters)
    uncertainty = propagate_uncertainty(
        surface_flux,
        parameters,
        n_samples=96,
        seed=341,
        quantiles=(0.05, 0.5, 0.95),
    )
    screening = standardized_regression_coefficients(uncertainty)

    print("Sensitivity and uncertainty screening")
    print("-------------------------------------")
    print(f"QoI: steady surface flux into a slab of thickness {SLAB_THICKNESS:.1f}")
    print(f"Nominal Thiele modulus phi = L sqrt(k/D): {nominal_modulus:.6f}")
    print(f"Solver QoI at nominal inputs:  {nominal_flux:.8f}")
    print(f"Closed form at nominal inputs: {nominal_exact:.8f}")
    print(f"Relative QoI discretization error: {qoi_error:.3e}")
    print(
        f"Same flux as k*c_s*L*effectiveness({nominal_modulus:.3f}) = "
        f"{flux_from_effectiveness:.8f} "
        f"(effectiveness factor {effectiveness:.6f})"
    )

    for name, sweep in sweeps.items():
        other = "decay_rate" if name == "diffusivity" else "diffusivity"
        print(f"\n{name} sweep ({name}, phi, surface flux) at nominal {other}:")
        for value, output in zip(sweep.swept_values, sweep.outputs):
            modulus = bt.analytical.thiele_modulus(
                D=value if name == "diffusivity" else nominal["diffusivity"],
                k=value if name == "decay_rate" else nominal["decay_rate"],
                length=SLAB_THICKNESS,
            )
            print(f"  {value:.5f}  {modulus:.5f}  {output:.8f}")
        span = max(sweep.outputs) / min(sweep.outputs)
        print(f"  QoI varies by a factor of {span:.3f} across this sweep")

    print("\nLocal elasticities at nominal inputs:")
    for name, value in local.normalized_by_parameter.items():
        print(f"  {name:>22}: {value:+.6f}")
    print("Closed-form elasticities at phi = 1 for comparison:")
    tanh_term = nominal_modulus / (
        math.tanh(nominal_modulus) * math.cosh(nominal_modulus) ** 2
    )
    print(f"  {'diffusivity':>22}: {0.5 - 0.5 * tanh_term:+.6f}")
    print(f"  {'decay_rate':>22}: {0.5 + 0.5 * tanh_term:+.6f}")
    print(f"  {'surface_concentration':>22}: {1.0:+.6f}")

    print(f"\nSeeded LHS attempts: {uncertainty.n_attempted}")
    print(f"Finite outputs:       {uncertainty.n_successful}")
    print(f"Failed evaluations:   {uncertainty.n_failed}")
    print(f"Sample mean:          {uncertainty.mean:.8f}")
    print(f"Sample std. dev.:     {uncertainty.standard_deviation:.8f}")
    for probability, value in uncertainty.quantiles.items():
        print(f"q={probability:0.3f}:             {value:.8f}")

    print("\nStandardized regression screening coefficients:")
    for name, value in screening.coefficient_by_parameter.items():
        print(f"  {name:>22}: {value:+.6f}")
    print(f"Linear-surrogate R^2: {screening.r_squared:.6f}")
    print(f"Design rank:          {screening.design_rank}")
    print(f"Condition number:     {screening.condition_number:.6f}")

    magnitudes = {
        name: abs(value) for name, value in screening.coefficient_by_parameter.items()
    }
    ranking = sorted(magnitudes.items(), key=lambda item: item[1], reverse=True)
    spread = ranking[0][1] / ranking[-1][1]
    print(
        "Screening ranking: "
        + " > ".join(f"{name} ({value:.3f})" for name, value in ranking)
    )
    print(f"Largest/smallest coefficient ratio: {spread:.3f}")
    print(
        f"Unexplained variance 1 - R^2 = {1.0 - screening.r_squared:.6f}, "
        "the part of the response the linear surrogate cannot carry"
    )

    # These are numerical behavior checks for this example, not validation.
    derivatives = local.derivative_by_parameter
    if min(derivatives.values()) <= 0.0:
        raise RuntimeError(
            "expected faster diffusion, faster reaction and a higher surface "
            "value to each raise the surface flux"
        )
    elasticities = local.normalized_by_parameter
    if abs(elasticities["diffusivity"] - (0.5 - 0.5 * tanh_term)) > 1e-3:
        raise RuntimeError("diffusivity elasticity does not match the closed form")
    if abs(elasticities["decay_rate"] - (0.5 + 0.5 * tanh_term)) > 1e-3:
        raise RuntimeError("decay_rate elasticity does not match the closed form")
    if qoi_error > 1e-3:
        raise RuntimeError("the flux reduction is not resolving the closed form")
    # The screening has to separate the inputs rather than return one dominant
    # scale factor with two negligible partners.
    if spread < 2.0:
        raise RuntimeError("screening coefficients are too close to rank the inputs")
    if ranking[-1][1] < 0.1:
        raise RuntimeError("an input carries no detectable share of the variance")
    if uncertainty.n_failed != 0 or screening.r_squared < 0.90:
        raise RuntimeError(
            "illustrative screening diagnostics did not meet expectations"
        )

    print("\nExample checks passed (screening only; no validation claim).")


if __name__ == "__main__":
    main()
