"""Measure coupled transport and reversible binding with conservation gates.

    python examples/verification/benchmark_coupled.py --output results/coupled.json

Reports timings, sparse storage, solver work, conservation and agreement with
a tighter-tolerance run on the same mesh. Imports are excluded; each size is
warmed up. This is a reproducible workload, not a comparison to another engine.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 2 for n in args.cells):
        parser.error("repeats must be positive and each cell count must be at least 2")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"

    import numpy as np
    import scipy
    import biotransport as bt
    import biotransport.coupled as coupled

    def model_for(n):
        radius = 1e-3
        model = bt.CoupledModel(["drug", "site", "bound"])
        model.compartment("reservoir", volume=5e-9, initial={"drug": 1})
        model.domain(
            "tissue",
            bt.mesh_1d(n, 0, radius, geometry="spherical"),
            diffusivity={"drug": 1e-9},
            initial={"site": 1},
        )
        model.membrane(
            "wall",
            "reservoir",
            ("tissue", "right"),
            area=4 * np.pi * radius**2,
            permeability={"drug": 2e-6},
            partition={"drug": 2},
        )
        model.mass_action(
            "tissue",
            reactants={"drug": 1, "site": 1},
            products={"bound": 1},
            rate_constant=0.1,
        )
        model.mass_action(
            "tissue",
            reactants={"bound": 1},
            products={"drug": 1, "site": 1},
            rate_constant=0.01,
        )
        model.conserve("drug", {"drug": 1, "bound": 1})
        model.conserve("sites", {"site": 1, "bound": 1})
        return model

    results = []
    for n in args.cells:
        model = model_for(n)
        compiled = model.compile()
        compiled.solve(300, frames=40)
        assembly_seconds, solve_seconds = [], []
        for _ in range(args.repeats):
            start = time.perf_counter()
            compiled = model.compile()
            assembly_seconds.append(time.perf_counter() - start)
            start = time.perf_counter()
            solution = compiled.solve(300, frames=40)
            solve_seconds.append(time.perf_counter() - start)
        reference = compiled.solve(300, frames=1, rtol=1e-10, atol=1e-13)
        difference = max(
            float(np.max(np.abs(solution.field(d, s) - reference.field(d, s))))
            for d in compiled.domains
            for s in compiled.species
        )
        drug, sites = solution.balance("drug"), solution.balance("sites")
        assert drug.relative_drift is not None and drug.relative_drift < 2e-9
        assert sites.relative_drift is not None and sites.relative_drift < 2e-9
        assert difference < 2e-6
        assert solution.diagnostics.minimum_concentration >= -1e-9
        jacobian = compiled.jacobian(0, compiled.initial_state)
        entry = {
            "cells": n,
            "state_size": compiled.initial_state.size,
            "assembly_seconds": assembly_seconds,
            "solve_seconds": solve_seconds,
            "median_assembly_seconds": statistics.median(assembly_seconds),
            "median_solve_seconds": statistics.median(solve_seconds),
            "initial_jacobian_nonzeros": jacobian.nnz,
            "initial_jacobian_storage_bytes": jacobian.data.nbytes
            + jacobian.indices.nbytes
            + jacobian.indptr.nbytes,
            "saved_frames": solution.times.size,
            "final_max_abs_difference_to_tighter_run": difference,
            "drug_balance": asdict(drug),
            "site_balance": asdict(sites),
            "diagnostics": asdict(solution.diagnostics),
        }
        results.append(entry)
        print(
            f"{n:>5} cells / {entry['state_size']:>5} states: "
            f"compile {entry['median_assembly_seconds']:.4f}s, solve {entry['median_solve_seconds']:.4f}s, "
            f"reference difference {difference:.3g}, relative balance drift {drug.relative_drift:.3g}"
        )
    source = Path(coupled.__file__).resolve()
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "native_build": bt.native_build_info(),
        "blas_threads": 1,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "workload": "closed reservoir / spherical tissue / drug + site <-> bound",
        "end_time_s": 300,
        "rtol": 1e-7,
        "atol": 1e-10,
        "reference_rtol": 1e-10,
        "reference_atol": 1e-13,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
