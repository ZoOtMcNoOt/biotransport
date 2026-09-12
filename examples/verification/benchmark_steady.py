"""Reproduce steady-solver timings and accuracy on one machine.

Run from an environment where BioTransport is installed::

    python examples/verification/benchmark_steady.py --output results/steady-benchmark.json

Both compared paths solve the same scaled equations with the same stopping
criteria. The dense reference converts the analytic sparse stencil to an array,
reproducing the former dense rank-check/solve path. It is a benchmark reference,
not a supported engine backend. Reported storage covers the Jacobian arrays,
not total process or factorization peak memory. Imports and warmups are excluded.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from unittest.mock import patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[100, 300, 600])
    parser.add_argument("--large-cells", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1 or args.large_cells < 2:
        parser.error(
            "repeats and threads must be positive; large-cells must be at least 2"
        )
    if any(cells < 2 or cells > 2_000 for cells in args.cells):
        parser.error("dense comparison cells must be between 2 and 2000")

    # Set before loading NumPy/SciPy so BLAS comparisons use the requested
    # thread count. These settings affect this benchmark process only.
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    for name in thread_variables:
        os.environ[name] = str(args.threads)

    import numpy as np
    import scipy
    import biotransport as bt

    original_jacobian = bt.NonlinearDiffusionSolver._jacobian_1d

    def dense_jacobian(solver, state):
        return original_jacobian(solver, state).toarray()

    def measure(cells, dense=False):
        mesh = bt.mesh_1d(cells)
        problem = (
            bt.Problem(mesh)
            .diffusivity(1.0)
            .linear_decay(1.0)
            .initial(1.0)
            .dirichlet("left", 1.0)
            .sealed("right")
        )
        context = (
            patch.object(bt.NonlinearDiffusionSolver, "_jacobian_1d", dense_jacobian)
            if dense
            else nullcontext()
        )
        with context:
            bt.solve_steady(problem)  # Warm imports and solver dispatch.
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                solution = bt.solve_steady(problem)
                samples.append(time.perf_counter() - start)

        exact = np.cosh(1.0 - solution.x) / np.cosh(1.0)
        error = float(np.max(np.abs(solution.c - exact)))
        if not solution.newton.converged or error > 0.03 / cells**2:
            raise RuntimeError(
                f"accuracy gate failed for {cells} cells: max error {error}"
            )
        assembly = bt.NonlinearDiffusionSolver(mesh, D=1.0)
        assembly.set_reaction(lambda u: u, lambda u: np.ones_like(u))
        assembly.set_boundary(bt.Boundary.Left, 1.0)
        assembly.set_boundary(bt.Boundary.Right, 0.0, "neumann")
        matrix = original_jacobian(assembly, np.ones(cells + 1))
        storage = (
            8 * (cells + 1) ** 2
            if dense
            else matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        )
        return {
            "cells": cells,
            "unknowns": cells + 1,
            "linear_solver": solution.newton.linear_solver,
            "seconds": samples,
            "median_seconds": statistics.median(samples),
            "max_absolute_error": error,
            "iterations": solution.newton.iterations,
            "dimensionless_residual": solution.newton.residual_norm,
            "jacobian_array_bytes": int(storage),
            "jacobian_nonzeros": int(matrix.nnz),
        }

    comparisons = []
    for cells in args.cells:
        sparse_result = measure(cells)
        dense_result = measure(cells, dense=True)
        comparisons.append(
            {
                "sparse": sparse_result,
                "dense_reference": dense_result,
                "median_speedup": dense_result["median_seconds"]
                / sparse_result["median_seconds"],
            }
        )
        print(
            f"{cells:>5} cells: sparse {sparse_result['median_seconds']:.6f}s; "
            f"dense {dense_result['median_seconds']:.6f}s; "
            f"max error {sparse_result['max_absolute_error']:.3e}",
            flush=True,
        )
    large_result = measure(args.large_cells)
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "threads_requested": args.threads,
        },
        "method": "warm complete solve; identical scaled equations and stopping criteria",
        "problem": "-c'' + c = 0 on [0,1], c(0)=1, c'(1)=0; initial c=1",
        "repeats": args.repeats,
        "comparisons": comparisons,
        "large_sparse": large_result,
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
