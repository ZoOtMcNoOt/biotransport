"""JSON presentation of engine results, independent of HTTP and the browser."""

from __future__ import annotations

import math
import warnings

import numpy as np

from .. import analytical


def _reference(document: dict, solution) -> dict | None:
    """Offer an exact reference only when its assumptions match the model."""
    geometry = document["domain"]["geometry"]
    parts = document["components"]
    allowed = {
        "diffusion",
        "initial.uniform",
        "boundary.fixed",
        "boundary.sealed",
        "reaction.decay",
    }
    if any(part["type"] not in allowed for part in parts):
        return None
    fixed = [p["parameters"] for p in parts if p["type"] == "boundary.fixed"]
    if geometry == "cartesian":
        if len(fixed) != 2 or fixed[0]["value"] != fixed[1]["value"]:
            return None
    else:
        sealed = [p["parameters"] for p in parts if p["type"] == "boundary.sealed"]
        if (
            len(fixed) != 1
            or fixed[0]["side"] != "right"
            or len(sealed) != 1
            or sealed[0]["side"] != "left"
        ):
            return None
    diffusivity = next(
        p["parameters"]["coefficient"] for p in parts if p["type"] == "diffusion"
    )
    if diffusivity <= 0:
        return None
    length = document["domain"]["length"]
    surface = fixed[0]["value"]
    initial = next(
        p["parameters"]["value"] for p in parts if p["type"] == "initial.uniform"
    )
    rate = sum(p["parameters"]["rate"] for p in parts if p["type"] == "reaction.decay")
    if solution.steady:
        # Stable cosh ratio for equal face concentrations; avoids overflow at
        # large Thiele modulus. This is the exact homogeneous linear ODE.
        if geometry == "cartesian":
            b = (
                (math.sqrt(rate) * (length / 2) / math.sqrt(diffusivity))
                if rate
                else 0.0
            )
            if not math.isfinite(b):
                return None
            z = 2 * b * np.abs(solution.x / length - 0.5)
            values = surface * (np.exp(z - b) + np.exp(-z - b)) / (1 + math.exp(-2 * b))
        else:
            values = analytical.steady_radial_first_order(
                solution.x,
                D=diffusivity,
                k=rate,
                R=length,
                c_surface=surface,
                geometry=geometry,
            )
        fields = [values.tolist()]
        label = "Exact steady diffusion–reaction profile"
    elif rate:
        return None
    else:
        reference_function = {
            "cartesian": analytical.slab,
            "cylindrical": analytical.cylinder,
            "spherical": analytical.sphere,
        }[geometry]
        dimensions = {"L" if geometry == "cartesian" else "R": length}
        with warnings.catch_warnings():
            # An unconverged truncated reference must never look certified.
            warnings.simplefilter("error", RuntimeWarning)
            try:
                fields = [
                    np.asarray(
                        reference_function(
                            solution.x,
                            t,
                            D=diffusivity,
                            **dimensions,
                            c_surface=surface,
                            c_initial=initial,
                        )
                    ).tolist()
                    for t in solution.times
                ]
            except RuntimeWarning:
                return None
        label = f"Exact {geometry} diffusion series"
    report = solution.compare(np.asarray(fields[-1]))
    return {
        "label": label,
        "fields": fields,
        "max_abs_error": report.max_abs,
        "l2_error": report.l2,
    }


def result_payload(document: dict, solution, elapsed: float) -> dict:
    """Serialize actual saved fields and measured diagnostics without resampling."""
    history = solution.history
    if not np.all(np.isfinite(history)):
        raise ValueError(
            "The solver produced a non-finite field; try a better resolved model."
        )
    totals = history @ solution.weights
    dimensionless = [
        {
            "name": name,
            "value": value if math.isfinite(value) else None,
            "description": description,
        }
        for name, (value, description) in solution.dimensionless().items()
        if not solution.steady or name not in {"Fourier", "sqrt(D t) / L"}
    ]
    return {
        "experiment": document,
        "solution": {
            "x": solution.x.tolist(),
            "times": list(solution.times),
            "fields": history.tolist(),
            "totals": totals.tolist(),
            "minimum": float(history.min()),
            "maximum": float(history.max()),
            "steps": solution.steps,
            "steady": solution.steady,
            "elapsed_seconds": elapsed,
            "summary": solution.summary(),
            "dimensionless": dimensionless,
        },
        "reference": _reference(document, solution),
    }
