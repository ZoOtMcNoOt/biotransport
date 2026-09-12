"""Reproduce a bounded native transport timing, optionally against an old install.

Run in separate processes when comparing package builds. This measures the
same 200-cell sphere for 100 physical seconds, not time to equilibrium.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.package_root:
        sys.path.insert(0, str(args.package_root.resolve()))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"

    import numpy as np
    import biotransport as bt

    mesh = bt.mesh_1d(200, 0, 1e-3, "spherical")
    problem = (
        bt.Problem(mesh)
        .diffusivity(1e-9)
        .linear_decay(1e-3)
        .initial(1)
        .dirichlet("right", 1)
    )
    options = bt.SolveOptions.until(100)
    bt.solve_transport(problem, options)
    seconds = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = bt.solve_transport(problem, options)
        seconds.append(time.perf_counter() - start)
    field = np.asarray(result.concentration)
    if not np.all(np.isfinite(field)) or not np.all((field >= 0) & (field <= 1)):
        raise RuntimeError("field failed the physical range check")
    native = next((Path(bt.__file__).parent / "_core").glob("_core*.pyd"), None)
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "package": bt.__file__,
        "build": bt.native_build_info(),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest()
        if native
        else None,
        "threads_requested": 1,
        "repeats": args.repeats,
        "case": {
            "geometry": "spherical",
            "cells": 200,
            "R": 1e-3,
            "D": 1e-9,
            "k": 1e-3,
            "initial": 1,
            "surface": 1,
            "duration": 100,
        },
        "steps": result.diagnostics.steps,
        "selected_step": result.diagnostics.maximum_time_step,
        "seconds": seconds,
        "median_seconds": statistics.median(seconds),
        "final_field": field.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{report['steps']} steps, median {report['median_seconds']:.6f}s; {args.output}"
    )


if __name__ == "__main__":
    main()
