#!/usr/bin/env python3
"""Create a deterministic publication manifest for a diffusion check.

The numerical example uses a cosine eigenmode with homogeneous Neumann data,
records a coarse-to-fine error table and one integral-balance residual per grid,
and writes canonical JSON.  It is numerical-verification evidence for this test
case, not validation of a biological model.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import biotransport as bt
import biotransport.reproducibility as repro

EXAMPLE_NAME = "verification/reproducible_artifact"
RANDOM_SEED = 20260722


def relative_l2(numerical: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(numerical - reference) / np.linalg.norm(reference))


def run_study():
    length_m = 1.0
    diffusivity_m2_per_s = 0.05
    final_time_s = 0.1
    cells_by_level = [20, 40, 80]

    # The seed controls a sampled amplitude while preserving the analytical
    # cosine eigenmode. The actual sampled value is frozen in the configuration.
    generator = np.random.default_rng(RANDOM_SEED)
    amplitude = float(generator.uniform(0.08, 0.12))

    rows = []
    balances = []
    errors = []
    final_result = None
    final_field = None
    previous_h = None
    previous_error = None

    for level, cells in enumerate(cells_by_level):
        mesh = bt.StructuredMesh(cells, 0.0, length_m)
        x = np.linspace(0.0, length_m, cells + 1)
        initial = 1.0 + amplitude * np.cos(np.pi * x / length_m)
        problem = (
            bt.Problem(mesh)
            .diffusivity(diffusivity_m2_per_s)
            .initial_condition(initial)
            .neumann(bt.Boundary.Left, 0.0)
            .neumann(bt.Boundary.Right, 0.0)
        )
        result = bt.solve(problem, end_time=final_time_s)
        numerical = np.asarray(result.concentration)
        reference = 1.0 + amplitude * np.cos(np.pi * x / length_m) * np.exp(
            -diffusivity_m2_per_s * (np.pi / length_m) ** 2 * final_time_s
        )
        error = relative_l2(numerical, reference)
        h = mesh.dx()
        observed_order = None
        if previous_h is not None and previous_error is not None:
            observed_order = math.log(previous_error / error) / math.log(previous_h / h)

        rows.append(
            {
                "level": level,
                "cells": cells,
                "h_m": h,
                "relative_l2_error": error,
                "observed_order": observed_order,
                "steps": result.diagnostics.steps,
            }
        )
        balances.append(
            repro.balance_residual(
                f"field inventory, {cells} cells",
                initial_inventory=result.diagnostics.initial_mass,
                final_inventory=result.diagnostics.final_mass,
                units="field_unit*m",
                relative_tolerance=5.0e-13,
            )
        )
        errors.append(error)
        previous_h = h
        previous_error = error
        final_result = result
        final_field = numerical

    if final_result is None or final_field is None:
        raise RuntimeError("the convergence study did not produce a finest-grid result")
    convergence = repro.convergence_table(
        rows,
        refinement_parameter="h_m",
        quantity="relative_l2_error",
        expected_order=2.0,
        units={"h_m": "m", "relative_l2_error": "1"},
        study="Neumann cosine diffusion eigenmode",
    )
    passed = bool(
        rows[-1]["observed_order"] is not None
        and rows[-1]["observed_order"] > 1.8
        and all(record["within_tolerance"] for record in balances)
    )

    config = {
        "length_m": length_m,
        "diffusivity_m2_per_s": diffusivity_m2_per_s,
        "final_time_s": final_time_s,
        "cells_by_level": cells_by_level,
        "initial_condition": {
            "equation": "1 + amplitude*cos(pi*x/length)",
            "sampled_amplitude": amplitude,
            "random_seed": RANDOM_SEED,
            "numpy_bit_generator": "PCG64",
        },
        "boundary_conditions": {
            "left_outward_normal_derivative": 0.0,
            "right_outward_normal_derivative": 0.0,
        },
    }
    method = repro.method_metadata(
        "conservative explicit finite-volume diffusion",
        implementation="BioTransport C++ canonical transport solver",
        parameters={
            "time_integration": "Forward Euler",
            "automatic_time_step_safety_factor": 0.8,
            "spatial_operator": "vertex-centred conservative diffusion",
        },
        random_seed=RANDOM_SEED,
        deterministic=True,
    )
    results = {
        "passed": passed,
        "finest_relative_l2_error": errors[-1],
        "finest_observed_order": rows[-1]["observed_order"],
        "finest_steps": final_result.diagnostics.steps,
        "finest_field_canonical_json_sha256": repro.sha256_fingerprint(final_field),
        "maximum_absolute_balance_residual": max(
            record["absolute_residual"] for record in balances
        ),
    }
    return config, method, results, convergence, balances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(bt.get_result_path("manifest.json", EXAMPLE_NAME)),
        help="Destination JSON file (default: results directory)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing manifest atomically",
    )
    parser.add_argument(
        "--include-volatile",
        action="store_true",
        help="Include a timestamp and random run ID; output will not be byte-stable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, method, results, convergence, balances = run_study()
    manifest = repro.create_manifest(
        "Neumann cosine diffusion convergence",
        config=config,
        method=method,
        results=results,
        convergence=[convergence],
        balances=balances,
        notes=[
            "This artifact records numerical-verification evidence for one test case.",
            "It does not establish biological calibration, model validation, or FAIR compliance.",
        ],
        include_volatile=args.include_volatile,
    )
    output = repro.write_manifest(args.output, manifest, overwrite=args.overwrite)
    loaded = repro.load_manifest(output)

    print("Reproducible numerical artifact")
    print(f"  study passed                 {results['passed']}")
    print(f"  finest relative L2 error     {results['finest_relative_l2_error']:.3e}")
    print(f"  finest observed order        {results['finest_observed_order']:.3f}")
    print(
        f"  configuration SHA-256        {loaded['configuration']['fingerprint']['value']}"
    )
    print(f"  manifest content SHA-256     {loaded['content_fingerprint']['value']}")
    print(f"  native metadata available    {loaded['software']['native']['available']}")
    print(f"  output                        {output}")
    if args.include_volatile:
        print("  volatile run metadata         included (bytes vary between runs)")
    else:
        print("  volatile run metadata         omitted (byte-stable content)")
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
