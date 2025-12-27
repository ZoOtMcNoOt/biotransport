"""Run all example scripts headlessly and report runtimes.

This is a lightweight regression check: it executes each file under `examples/`
with `MPLBACKEND=Agg` so Matplotlib doesn't require a GUI.

Usage:
  python run_examples.py
  python run_examples.py --timeout 1200

Exit code is non-zero if any example fails.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def iter_example_scripts(examples_dir: Path) -> list[Path]:
    scripts: list[Path] = []
    for path in sorted(examples_dir.rglob("*.py")):
        if any(part == "__pycache__" for part in path.parts):
            continue
        scripts.append(path)
    return scripts


def run_one(script: Path, repo_root: Path, timeout_s: int) -> tuple[int, float, str]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed = time.perf_counter() - started
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, elapsed, output
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        output = (exc.stdout or "") + (exc.stderr or "")
        return 124, elapsed, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-example timeout in seconds (default: 1800).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    examples_dir = repo_root / "examples"
    if not examples_dir.exists():
        print(f"ERROR: missing examples directory: {examples_dir}")
        return 2

    scripts = iter_example_scripts(examples_dir)
    if not scripts:
        print("No example scripts found.")
        return 0

    print(f"Found {len(scripts)} example scripts.")

    failures: list[tuple[Path, int]] = []
    timings: list[tuple[float, Path, int]] = []

    for script in scripts:
        rel = script.relative_to(repo_root)
        print(f"\n=== Running {rel} ===")
        code, elapsed, output = run_one(script, repo_root, timeout_s=args.timeout)
        timings.append((elapsed, rel, code))

        print(f"Exit code: {code}  Time: {elapsed:.2f}s")
        if code != 0:
            failures.append((rel, code))
            if output.strip():
                print("--- output (tail) ---")
                tail = output.splitlines()[-80:]
                print("\n".join(tail))
        else:
            if output.strip():
                # Print a small tail for progress visibility.
                tail = output.splitlines()[-10:]
                print("--- output (tail) ---")
                print("\n".join(tail))

    print("\n=== Summary (slowest first) ===")
    for elapsed, rel, code in sorted(timings, key=lambda x: x[0], reverse=True):
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"{elapsed:8.2f}s  {status:9s}  {rel}")

    if failures:
        print("\n=== Failures ===")
        for rel, code in failures:
            print(f"- {rel}: exit {code}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
