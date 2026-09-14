"""Measure scheduled bath transport with independently checked amount balances.

    python examples/verification/benchmark_protocols.py --output results/protocols.json

The bath delivers a 20-second pulse to a sphere with reversible binding. Output
requests only the final frame; protocol knots must still be reached and saved.
Each mesh is warmed up before three timed solves and a tighter-tolerance check.
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
        parser.error("repeats must be positive and cells must be at least 2")
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

    def build(cells):
        model = bt.CoupledModel(["drug", "site", "bound"])
        model.domain(
            "tissue",
            bt.mesh_1d(cells, 0, 1e-3, geometry="spherical"),
            initial={"site": 1},
            diffusivity={"drug": 1e-9},
        )
        model.bath(
            "dose",
            concentration={"drug": bt.ConcentrationSchedule([0, 20, 40], [0, 1, 0])},
        )
        model.membrane(
            "wall",
            "dose",
            ("tissue", "right"),
            area=4 * np.pi * 1e-6,
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
        return model.compile()

    measurements = []
    for cells in args.cells:
        compiled = build(cells)
        compiled.solve(120, frames=1)
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            solution = compiled.solve(120, frames=1)
            timings.append(time.perf_counter() - start)
        reference = compiled.solve(120, frames=1, rtol=1e-10, atol=1e-13)
        field_difference = max(
            float(
                np.max(
                    np.abs(solution.field("tissue", s) - reference.field("tissue", s))
                )
            )
            for s in compiled.species
        )
        amount_difference = float(
            np.max(
                np.abs(
                    solution.external_amount("drug") - reference.external_amount("drug")
                )
            )
        )
        drug, sites = solution.balance("drug"), solution.balance("sites")
        assert solution.times.tolist() == [0, 20, 40, 120]
        assert field_difference < 2e-6
        assert drug.relative_drift is not None and drug.relative_drift < 2e-8
        assert sites.relative_drift is not None and sites.relative_drift < 2e-9
        assert solution.diagnostics.minimum_concentration >= -1e-9
        row = {
            "cells": cells,
            "concentration_states": compiled.initial_state.size,
            "external_ledger_states": len(
                compiled.external_rates(0, compiled.initial_state)
            ),
            "solve_seconds": timings,
            "median_solve_seconds": statistics.median(timings),
            "saved_times": solution.times.tolist(),
            "max_final_concentration_difference_to_tighter_run": field_difference,
            "max_external_moles_difference_to_tighter_run": amount_difference,
            "drug_balance": asdict(drug),
            "site_balance": asdict(sites),
            "diagnostics": asdict(solution.diagnostics),
        }
        measurements.append(row)
        print(
            f"{cells:>5} cells: {row['median_solve_seconds']:.4f}s; "
            f"field difference {field_difference:.3g}; balance residual {drug.relative_drift:.3g}"
        )

    package = Path(bt.__file__).resolve().parent
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "native_build": bt.native_build_info(),
        "blas_threads_requested": 1,
        "source_sha256": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in ("coupled.py", "protocols.py", "_coupled_integration.py")
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "workload": "prescribed bath pulse / spherical tissue / drug + site <-> bound",
        "end_time_s": 120,
        "rtol": 1e-7,
        "atol": 1e-10,
        "reference_rtol": 1e-10,
        "reference_atol": 1e-13,
        "results": measurements,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
