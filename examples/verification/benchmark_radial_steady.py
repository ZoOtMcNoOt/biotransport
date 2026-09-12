"""Measure direct radial equilibration with exact-profile and balance gates.

The 200-cell sphere is the same D=1e-9, R=1e-3, k=1e-3 problem previously
equilibrated to Fourier number 20 in test_radial_geometry.py. This script times
the complete direct solve without repeating that millions-of-steps transient.
An optional smaller-mesh comparison times both paths on identical equations;
it is bounded to 40 cells and Fourier number 3. Imports and warmups are excluded.

    python examples/verification/benchmark_radial_steady.py --transient-cells 20 \
        --output results/radial-steady-benchmark.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[200, 2000, 10_000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--transient-cells", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(cells < 2 for cells in args.cells):
        parser.error("repeats must be positive and cells must be at least 2")
    if args.transient_cells is not None and not 2 <= args.transient_cells <= 40:
        parser.error("transient-cells must be between 2 and 40")
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[variable] = "1"

    import numpy as np
    import scipy
    from scipy.special import i0

    import biotransport as bt

    diffusivity, radius, rate = 1e-9, 1e-3, 1e-3

    def problem_for(cells, geometry):
        return (
            bt.Problem(bt.mesh_1d(cells, 0.0, radius, geometry))
            .diffusivity(diffusivity)
            .linear_decay(rate)
            .initial(1.0)
            .dirichlet("right", 1.0)
        )

    def measure(cells, geometry):
        problem = problem_for(cells, geometry)
        bt.solve_steady(problem)
        seconds = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            solution = bt.solve_steady(problem)
            seconds.append(time.perf_counter() - start)
        rho = solution.x / radius
        if geometry == "cylindrical":
            exact = i0(rho) / i0(1.0)
        else:
            exact = np.ones_like(rho) / np.sinh(1.0)
            exact[1:] = np.sinh(rho[1:]) / (rho[1:] * np.sinh(1.0))
        error = float(np.max(np.abs(solution.c - exact)))
        relative_balance = abs(solution.balance().residual / solution.uptake())
        if error > 0.1 / cells**2 or relative_balance > 2e-7:
            raise RuntimeError(
                f"{geometry}, {cells} cells: accuracy/balance gate failed: "
                f"max error {error:.3g}, relative balance {relative_balance:.3g}"
            )
        entry = {
            "geometry": geometry,
            "cells": cells,
            "seconds": seconds,
            "median_seconds": statistics.median(seconds),
            "newton_iterations": solution.newton.iterations,
            "dimensionless_residual": solution.newton.residual_norm,
            "max_absolute_error": error,
            "relative_balance_residual": relative_balance,
        }
        print(
            f"{geometry:>11}, {cells:>5} cells: {entry['median_seconds']:.6f}s, "
            f"max error {error:.3e}, relative balance {relative_balance:.3e}",
            flush=True,
        )
        return entry, solution

    direct = [
        measure(cells, geometry)[0]
        for geometry in ("cylindrical", "spherical")
        for cells in args.cells
    ]
    comparisons = []
    if args.transient_cells is not None:
        cells = args.transient_cells
        end_time = 3 * radius**2 / diffusivity
        for geometry in ("cylindrical", "spherical"):
            steady, solution = measure(cells, geometry)
            problem = problem_for(cells, geometry)
            bt.solve(problem, end_time=end_time)
            seconds = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                transient = bt.solve(problem, end_time=end_time)
                seconds.append(time.perf_counter() - start)
            difference = float(np.max(np.abs(solution.c - transient.c)))
            if difference > 2e-8:
                raise RuntimeError(f"steady/transient comparison failed: {difference}")
            comparisons.append(
                {
                    "geometry": geometry,
                    "cells": cells,
                    "fourier_number": 3.0,
                    "direct": steady,
                    "transient_seconds": seconds,
                    "transient_median_seconds": statistics.median(seconds),
                    "transient_steps": transient.steps,
                    "max_steady_transient_difference": difference,
                    "median_speedup": statistics.median(seconds)
                    / steady["median_seconds"],
                }
            )

    package = Path(bt.__file__).parent
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "threads_requested": 1,
            "package": str(package),
        },
        "source_sha256": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in ("steady.py", "newton_raphson.py", "analytical.py")
        },
        "problem": {
            "D": diffusivity,
            "R": radius,
            "k": rate,
            "surface": 1.0,
            "initial": 1.0,
        },
        "method": "warm complete direct solves, default dimensionless Newton tolerance; exact profile and balance gates",
        "historical_case": "200-cell sphere matches former Fourier-number-20 equilibration test; that transient is not rerun or included in speedup claims",
        "repeats": args.repeats,
        "direct": direct,
        "bounded_transient_comparisons": comparisons,
    }
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Report: {args.output}")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
