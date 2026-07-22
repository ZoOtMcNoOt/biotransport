"""Tests for deterministic publication manifests and build provenance."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from biotransport.reproducibility import (
    BALANCE_SCHEMA,
    MANIFEST_SCHEMA,
    ReproducibilityError,
    balance_residual,
    canonical_json,
    canonical_json_bytes,
    convergence_table,
    create_manifest,
    freeze_config,
    load_manifest,
    method_metadata,
    sha256_fingerprint,
    software_metadata,
    verify_manifest,
    write_manifest,
)


class Scheme(Enum):
    CONSERVATIVE = "conservative"


@dataclass
class ExampleConfig:
    diffusivity: float
    cells: int
    scheme: Scheme


def example_evidence():
    method = method_metadata(
        "finite-volume Forward Euler",
        parameters={"safety_factor": 0.8},
        random_seed=0,
        deterministic=True,
    )
    convergence = convergence_table(
        [
            {"h_m": 0.1, "relative_l2_error": 0.04, "observed_order": None},
            {"h_m": 0.05, "relative_l2_error": 0.01, "observed_order": 2.0},
            {"h_m": 0.025, "relative_l2_error": 0.0025, "observed_order": 2.0},
        ],
        refinement_parameter="h_m",
        quantity="relative_l2_error",
        expected_order=2.0,
        units={"h_m": "m", "relative_l2_error": "1"},
        study="smooth diffusion eigenmode",
    )
    balance = balance_residual(
        "amount",
        initial_inventory=1.0,
        final_inventory=0.9,
        time_integrated_boundary_outflow=0.1,
        units="mol",
        relative_tolerance=1.0e-12,
    )
    return method, convergence, balance


def example_manifest(**volatile):
    method, convergence, balance = example_evidence()
    return create_manifest(
        "deterministic test",
        config={"D_m2_per_s": 1.0e-9, "cells": [20, 40, 80]},
        method=method,
        results={"passed": True, "finest_error": 0.0025},
        convergence=[convergence],
        balances=[balance],
        notes=["Numerical evidence only; no physical-validation claim."],
        **volatile,
    )


def test_canonical_json_has_one_deterministic_representation() -> None:
    first = {
        "z": np.array([1.0, -0.0]),
        "config": ExampleConfig(1.0e-9, 40, Scheme.CONSERVATIVE),
        "é": "e\u0301",
    }
    second = {
        "é": "é",
        "config": {
            "scheme": "conservative",
            "cells": 40,
            "diffusivity": 1.0e-9,
        },
        "z": [1.0, 0.0],
    }

    assert canonical_json(first) == canonical_json(second)
    assert canonical_json(first) == canonical_json(first)
    assert json.loads(canonical_json(first))["z"] == [1.0, 0.0]
    assert len(sha256_fingerprint(first)) == 64
    assert sha256_fingerprint(first) == sha256_fingerprint(second)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_fail_loudly_at_any_depth(bad: float) -> None:
    with pytest.raises(ReproducibilityError, match="finite"):
        canonical_json({"outer": [1.0, {"bad": bad}]})


@pytest.mark.parametrize(
    "bad",
    [Path("private/output.json"), {1, 2}, object(), {1: "non-string key"}],
)
def test_ambiguous_or_sensitive_value_types_are_rejected(bad) -> None:
    with pytest.raises(ReproducibilityError):
        canonical_json({"value": bad})


def test_reference_cycles_are_rejected() -> None:
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ReproducibilityError, match="cycle"):
        canonical_json(cyclic)


def test_freeze_config_detaches_values_and_fingerprints_payload() -> None:
    array = np.array([1.0, 2.0])
    frozen = freeze_config({"field": array})
    original_fingerprint = frozen["fingerprint"]["value"]
    array[0] = 99.0

    assert frozen["values"] == {"field": [1.0, 2.0]}
    assert original_fingerprint == sha256_fingerprint(frozen["values"])


def test_stable_manifests_are_byte_for_byte_deterministic(tmp_path: Path) -> None:
    first = example_manifest()
    second = example_manifest()

    assert first == second
    assert canonical_json_bytes(first, trailing_newline=True) == canonical_json_bytes(
        second, trailing_newline=True
    )
    assert verify_manifest(first)

    first_path = write_manifest(tmp_path / "first.json", first)
    second_path = write_manifest(tmp_path / "second.json", second)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_bytes().endswith(b"\n")
    assert load_manifest(first_path) == first


def test_volatile_metadata_is_explicitly_and_honestly_variable() -> None:
    stable = example_manifest()
    first = example_manifest(
        include_volatile=True,
        created_utc="2026-07-22T12:00:00.000000Z",
        run_id="run-a",
    )
    second = example_manifest(
        include_volatile=True,
        created_utc="2026-07-22T12:00:01.000000Z",
        run_id="run-b",
    )

    assert "run" not in stable["software"]
    assert first["software"]["run"] != second["software"]["run"]
    assert canonical_json_bytes(first) != canonical_json_bytes(second)
    assert (
        first["configuration"]["fingerprint"] == second["configuration"]["fingerprint"]
    )
    assert verify_manifest(first) and verify_manifest(second)


def test_volatile_values_cannot_be_silently_ignored() -> None:
    with pytest.raises(ReproducibilityError, match="include_volatile"):
        software_metadata(created_utc="2026-07-22T00:00:00Z")


def test_volatile_timestamp_requires_and_normalizes_utc() -> None:
    metadata = software_metadata(
        include_volatile=True,
        created_utc="2026-07-22T07:00:00-05:00",
        run_id="run",
    )
    assert metadata["run"]["created_utc"] == "2026-07-22T12:00:00.000000Z"

    with pytest.raises(ReproducibilityError, match="UTC offset"):
        software_metadata(
            include_volatile=True,
            created_utc="2026-07-22T12:00:00",
            run_id="run",
        )


def test_manifest_fingerprint_detects_any_result_mutation(tmp_path: Path) -> None:
    manifest = example_manifest()
    tampered = copy.deepcopy(manifest)
    tampered["results"]["finest_error"] = 1.0

    assert not verify_manifest(tampered)
    with pytest.raises(ReproducibilityError, match="untampered"):
        write_manifest(tmp_path / "tampered.json", tampered)


def test_nested_configuration_fingerprint_is_independently_verified() -> None:
    manifest = example_manifest()
    tampered = copy.deepcopy(manifest)
    tampered["configuration"]["values"]["D_m2_per_s"] = 2.0e-9
    payload = dict(tampered)
    payload.pop("content_fingerprint")
    tampered["content_fingerprint"] = {
        "algorithm": "sha256",
        "value": sha256_fingerprint(payload),
    }

    assert not verify_manifest(tampered)


def test_writer_refuses_clobber_and_leaves_original_intact(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "manifest.json", example_manifest())
    original = path.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_manifest(path, example_manifest())
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_loader_rejects_duplicate_keys_and_nonfinite_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(ReproducibilityError, match="duplicate key"):
        load_manifest(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ReproducibilityError, match="non-finite"):
        load_manifest(nonfinite)


def test_balance_residual_uses_documented_outflow_sign() -> None:
    record = balance_residual(
        "mass",
        initial_inventory=10.0,
        final_inventory=9.4,
        time_integrated_source=0.2,
        time_integrated_boundary_outflow=0.8,
        units="kg",
        relative_tolerance=1.0e-14,
    )

    assert record["schema"] == BALANCE_SCHEMA
    assert record["signed_residual"] == pytest.approx(0.0, abs=1.0e-15)
    assert record["relative_residual"] == pytest.approx(0.0, abs=1.0e-15)
    assert record["within_tolerance"] is True


def test_zero_balance_has_defined_zero_relative_residual() -> None:
    record = balance_residual(
        "zero inventory",
        initial_inventory=0.0,
        final_inventory=0.0,
        units="mol",
    )
    assert record["normalization"] == 0.0
    assert record["relative_residual"] == 0.0


def test_balance_arithmetic_overflow_fails_loudly() -> None:
    with pytest.raises(ReproducibilityError, match="overflowed"):
        balance_residual(
            "overflow",
            initial_inventory=-1.0e308,
            final_inventory=1.0e308,
            units="mol",
        )


def test_convergence_table_requires_ordered_finite_evidence() -> None:
    with pytest.raises(ReproducibilityError, match="strictly decreasing"):
        convergence_table(
            [{"h": 0.1, "error": 0.01}, {"h": 0.2, "error": 0.02}],
            refinement_parameter="h",
            quantity="error",
        )
    with pytest.raises(ReproducibilityError, match="finite"):
        convergence_table(
            [{"h": 0.1, "error": float("nan")}],
            refinement_parameter="h",
            quantity="error",
        )


def test_method_seed_is_explicit_and_portable() -> None:
    record = method_metadata("seeded method", random_seed=0, deterministic=True)
    assert record["random_seed"] == 0
    assert record["deterministic"] is True

    with pytest.raises(ReproducibilityError, match="random_seed"):
        method_metadata("bad seed", random_seed=-1)
    with pytest.raises(ReproducibilityError, match="random_seed"):
        method_metadata("bad seed", random_seed=2**64)


def test_software_metadata_excludes_sensitive_location_fields() -> None:
    metadata = software_metadata()
    encoded = canonical_json(metadata).lower()
    for forbidden_key in (
        "hostname",
        "username",
        "home_directory",
        "current_directory",
        "executable_path",
        "command_line",
        "environment_variables",
    ):
        assert forbidden_key not in encoded

    assert set(metadata) == {"packages", "python", "platform", "native"}


def test_native_build_metadata_is_structured_when_binding_is_available() -> None:
    native = software_metadata()["native"]
    assert isinstance(native["available"], bool)
    if not native["available"]:
        assert native["status"] in {
            "binding_unavailable",
            "invalid_binding_result",
            "query_failed",
        }
        return

    assert native["status"] == "reported_by_loaded_extension"
    assert native["compiler"]["id"]
    assert native["compiler"]["version"]
    assert native["cxx"]["standard"] >= 201703
    assert native["cxx"]["standard_name"] == "C++17"
    assert isinstance(native["features"]["eigen"]["compile_definition"], bool)
    assert isinstance(native["features"]["eigen"]["enabled"], bool)
    assert isinstance(native["features"]["openmp"]["compile_definition"], bool)
    assert isinstance(native["features"]["openmp"]["enabled"], bool)


def test_manifest_has_declared_schema_and_valid_content_fingerprint() -> None:
    manifest = example_manifest()
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["content_fingerprint"]["algorithm"] == "sha256"
    assert len(manifest["content_fingerprint"]["value"]) == 64
    assert verify_manifest(manifest)
