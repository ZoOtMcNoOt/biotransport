"""Bitwise regression guard for every native solver.

Each case defined in ``python/tests/golden/capture_goldens.py`` is re-run and
every recorded entry must match the stored ``<case>.npz`` fixture byte for
byte (same shape, same dtype, identical buffer).  A mismatch means the native
numerics moved; regenerate the fixtures only when that change is intended::

    python python/tests/golden/capture_goldens.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from biotransport.contracts import list_native_solver_symbols

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_CAPTURE_SCRIPT = _GOLDEN_DIR / "capture_goldens.py"
_REGENERATE_HINT = (
    "run `python python/tests/golden/capture_goldens.py` to (re)generate the "
    "golden fixtures"
)


def _load_capture_module() -> ModuleType:
    module_name = "biotransport_golden_capture"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _CAPTURE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


capture = _load_capture_module()
CASE_NAMES = tuple(capture.CASES)


def _describe_mismatch(name: str, expected: np.ndarray, actual: np.ndarray) -> str:
    if expected.shape != actual.shape:
        return f"{name}: shape {actual.shape} != golden {expected.shape}"
    if expected.dtype != actual.dtype:
        return f"{name}: dtype {actual.dtype} != golden {expected.dtype}"
    if expected.dtype.kind == "f":
        with np.errstate(invalid="ignore"):
            diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        finite = np.isfinite(diff)
        max_diff = float(diff[finite].max()) if finite.any() else float("nan")
        nan_mismatch = int(np.count_nonzero(np.isnan(actual) != np.isnan(expected)))
        return (
            f"{name}: bitwise mismatch (max |diff| over finite entries = "
            f"{max_diff!r}, NaN-pattern mismatches = {nan_mismatch})"
        )
    return f"{name}: bitwise mismatch (expected {expected!r}, got {actual!r})"


def test_golden_cases_cover_every_native_solver_symbol() -> None:
    assert capture.missing_symbols() == ()
    assert capture.unknown_symbols() == ()
    assert set(capture.covered_symbols()) == set(list_native_solver_symbols())


def test_case_table_has_stable_fixture_names() -> None:
    for name, case in capture.CASES.items():
        assert case.name == name
        assert case.fixture_path == _GOLDEN_DIR / f"{name}.npz"
        assert case.symbols, f"{name} declares no native symbol"


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_native_solver_matches_golden_bitwise(case_name: str) -> None:
    case = capture.CASES[case_name]
    if not case.available():
        pytest.skip(f"{case_name} requires sparse matrix support (Eigen)")
    if not case.fixture_path.is_file():
        pytest.fail(f"missing golden fixture {case.fixture_path}; {_REGENERATE_HINT}")

    actual = capture.run_case(case_name)
    with np.load(case.fixture_path, allow_pickle=False) as archive:
        expected = {key: archive[key] for key in archive.files}

    assert set(actual) == set(expected), (
        f"{case_name}: recorded entries changed "
        f"(new={sorted(set(actual) - set(expected))}, "
        f"gone={sorted(set(expected) - set(actual))}); {_REGENERATE_HINT}"
    )

    problems = [
        _describe_mismatch(key, expected[key], actual[key])
        for key in sorted(actual)
        if not (
            expected[key].shape == actual[key].shape
            and expected[key].dtype == actual[key].dtype
            and expected[key].tobytes() == actual[key].tobytes()
        )
    ]
    assert not problems, (
        f"{case_name}: native numerics moved; "
        + "; ".join(problems)
        + f". {_REGENERATE_HINT}"
    )
