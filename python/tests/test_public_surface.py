"""Guard: the public import surface is snapshotted and diffed.

The snapshot lives in ``python/tests/data/public_surface.json``. Any change to
``biotransport.__all__``, a namespace module's ``__all__``, or
``biotransport._core.__all__`` fails this test with an added/removed listing.

To accept an intentional change, regenerate the snapshot::

    BIOTRANSPORT_UPDATE_SNAPSHOTS=1 python -m pytest python/tests/test_public_surface.py
"""

from __future__ import annotations

import json
import os
from importlib import import_module
from pathlib import Path

import biotransport as bt
from biotransport._deprecation import ROOT_DEPRECATED, ROOT_LAZY
import biotransport._core as core_pkg

SNAPSHOT_PATH = Path(__file__).with_name("data") / "public_surface.json"
UPDATE_ENV_VAR = "BIOTRANSPORT_UPDATE_SNAPSHOTS"

NAMESPACES = (
    "diffusion",
    "electrochem",
    "flow",
    "applications",
    "analysis",
    "contracts",
    "provenance",
    "reproducibility",
    "units",
    "balance",
    "reference",
    "stepping",
)


def current_surface() -> dict[str, object]:
    namespaces = {
        name: sorted(import_module(f"biotransport.{name}").__all__)
        for name in NAMESPACES
    }
    return {
        "root_all": sorted(bt.__all__),
        "root_lazy": sorted(ROOT_LAZY),
        "root_deprecated": sorted(ROOT_DEPRECATED),
        "namespaces": namespaces,
        "core_all": sorted(core_pkg.__all__),
    }


def _write_snapshot(surface: dict[str, object]) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(
        json.dumps(surface, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _diff_lists(label: str, expected: list[str], actual: list[str]) -> list[str]:
    added = sorted(set(actual) - set(expected))
    removed = sorted(set(expected) - set(actual))
    lines: list[str] = []
    if added:
        lines.append(f"  {label}: added   {added}")
    if removed:
        lines.append(f"  {label}: removed {removed}")
    return lines


def _describe_diff(expected: dict[str, object], actual: dict[str, object]) -> str:
    lines: list[str] = []
    lines += _diff_lists("root_all", expected.get("root_all", []), actual["root_all"])
    lines += _diff_lists("core_all", expected.get("core_all", []), actual["core_all"])
    exp_ns = expected.get("namespaces", {})
    act_ns = actual["namespaces"]
    assert isinstance(exp_ns, dict)
    assert isinstance(act_ns, dict)
    for ns in sorted(set(exp_ns) | set(act_ns)):
        if ns not in exp_ns:
            lines.append(f"  namespaces.{ns}: new namespace {sorted(act_ns[ns])}")
        elif ns not in act_ns:
            lines.append(f"  namespaces.{ns}: namespace removed")
        else:
            lines += _diff_lists(f"namespaces.{ns}", exp_ns[ns], act_ns[ns])
    return "\n".join(lines)


def test_surface_entries_are_unique_and_public() -> None:
    surface = current_surface()
    for label, names in (("root_all", bt.__all__), ("core_all", core_pkg.__all__)):
        assert len(names) == len(set(names)), f"{label} has duplicates"
        # ``__version__`` is the one dunder the package deliberately exports.
        assert all(not n.startswith("_") or n == "__version__" for n in names), (
            f"{label} exports private names"
        )
    for ns, names in surface["namespaces"].items():  # type: ignore[union-attr]
        assert all(not n.startswith("_") for n in names), f"{ns} exports private names"


def test_public_surface_matches_snapshot() -> None:
    actual = current_surface()

    if os.environ.get(UPDATE_ENV_VAR) == "1":
        _write_snapshot(actual)
        return

    assert SNAPSHOT_PATH.is_file(), (
        f"missing snapshot {SNAPSHOT_PATH}; run with {UPDATE_ENV_VAR}=1 to create it"
    )
    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    if expected != actual:
        diff = _describe_diff(expected, actual)
        raise AssertionError(
            "public API surface changed relative to "
            f"{SNAPSHOT_PATH.relative_to(Path(__file__).parents[2])}:\n{diff}\n"
            f"If intentional, regenerate with {UPDATE_ENV_VAR}=1."
        )
