"""Contract checks for machine-readable native benchmark evidence."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "cpp" / "benchmarks" / "performance_schema.json"
EVIDENCE_DIRECTORY = PROJECT_ROOT / "build" / "performance-evidence"


def _assert_json_schema(value: Any, schema: dict[str, Any], path: str = "root") -> None:
    """Validate the JSON-Schema features used by performance_schema.json."""
    if "const" in schema:
        assert value == schema["const"], f"const mismatch at {path}"
    if "enum" in schema:
        assert value in schema["enum"], f"enum mismatch at {path}"

    expected_type = schema.get("type")
    if expected_type == "object":
        assert isinstance(value, dict), f"expected object at {path}"
        missing = set(schema.get("required", [])).difference(value)
        assert not missing, f"missing {sorted(missing)} at {path}"
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                _assert_json_schema(item, properties[key], f"{path}.{key}")
            elif additional is False:
                raise AssertionError(f"unexpected property {path}.{key}")
            elif isinstance(additional, dict):
                _assert_json_schema(item, additional, f"{path}.{key}")
    elif expected_type == "array":
        assert isinstance(value, list), f"expected array at {path}"
        assert len(value) >= schema.get("minItems", 0), f"too few items at {path}"
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _assert_json_schema(item, item_schema, f"{path}[{index}]")
    elif expected_type == "string":
        assert isinstance(value, str), f"expected string at {path}"
        assert len(value) >= schema.get("minLength", 0), f"string too short at {path}"
    elif expected_type == "integer":
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"expected integer at {path}"
        )
    elif expected_type == "number":
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"expected number at {path}"
        )
    elif expected_type == "boolean":
        assert isinstance(value, bool), f"expected boolean at {path}"

    if "minimum" in schema:
        assert value >= schema["minimum"], f"value below minimum at {path}"
    if "exclusiveMinimum" in schema:
        assert value > schema["exclusiveMinimum"], (
            f"value below exclusive minimum at {path}"
        )


def _assert_finite_json_numbers(value: Any, path: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        assert math.isfinite(value), f"non-finite number at {path}"
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json_numbers(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_json_numbers(item, f"{path}.{key}")
        return
    raise AssertionError(f"unsupported JSON value at {path}: {type(value).__name__}")


def validate_performance_evidence(document: dict[str, Any]) -> None:
    """Validate the semantic subset required for publishable timing evidence."""
    _assert_finite_json_numbers(document)
    assert document["schema_version"] == "biotransport.performance.v1"
    assert document["program"]

    provenance = document["provenance"]
    for field in ("compiler", "build", "platform", "openmp"):
        assert isinstance(provenance[field], dict)
    assert provenance["compiler"]["id"]
    assert provenance["compiler"]["version"]
    assert provenance["build"]["type"]
    assert provenance["build"]["configured_cxx_flags"] is not None
    assert provenance["platform"]["cpu_model"]
    assert provenance["platform"]["logical_threads"] >= 0
    assert provenance["openmp"]["observed_threads"] >= 1
    if provenance["revision_discoverable"]:
        revision = provenance["revision"]
        assert len(revision) >= 7
        assert all(character in "0123456789abcdefABCDEF" for character in revision)

    configuration = document["configuration"]
    assert configuration["timed_runs"] >= 2
    assert configuration["warmup_runs"] >= 0
    assert configuration["requested_steps"] >= 1
    assert configuration["requested_sizes_cells_per_axis"]

    results = document["results"]
    assert results
    for result in results:
        assert result["name"]
        assert result["implementation"]
        assert result["timed_scope"]
        workload = result["workload"]
        assert workload["node_count"] > 0
        assert workload["cell_count"] > 0
        assert workload["species_count"] > 0
        assert workload["step_count"] == configuration["requested_steps"]
        assert workload["maximum_time_step"] > 0.0
        assert workload["final_time"] > 0.0

        parallel = result["parallel"]
        assert parallel["observed_threads"] == provenance["openmp"]["observed_threads"]
        if parallel["openmp_effective_for_workload"]:
            assert provenance["openmp"]["enabled"]
            assert parallel["kernel"]

        correctness = result["correctness"]
        assert correctness["status"] == "pass"
        assert correctness["invariant"]
        assert correctness["tolerance_basis"] == "relative_error"
        assert correctness["absolute_error"] >= 0.0
        assert 0.0 <= correctness["relative_error"] <= correctness["tolerance"]
        assert correctness["repeatable_across_runs"] is True
        assert correctness["maximum_checksum_difference"] == 0.0
        assert correctness["checksum_definition"]

        timing = result["timing_ms"]
        samples = timing["samples"]
        assert len(samples) == configuration["timed_runs"]
        assert all(sample > 0.0 for sample in samples)
        assert timing["mean"] == pytest.approx(statistics.fmean(samples), rel=1.0e-12)
        assert timing["population_stddev"] == pytest.approx(
            statistics.pstdev(samples), rel=1.0e-12, abs=1.0e-15
        )
        assert timing["minimum"] == min(samples)
        assert timing["maximum"] == max(samples)
        assert timing["median"] == statistics.median(samples)

        throughput = result["throughput"]["node_species_steps_per_second"]
        expected_throughput = (
            workload["node_count"]
            * workload["species_count"]
            * workload["step_count"]
            / (timing["median"] / 1000.0)
        )
        assert throughput == pytest.approx(expected_throughput, rel=1.0e-12)


def representative_document() -> dict[str, Any]:
    samples = [1.0, 1.2, 0.8]
    return {
        "schema_version": "biotransport.performance.v1",
        "generated_utc": "2026-07-22T00:00:00Z",
        "program": "contract_fixture",
        "provenance": {
            "project_version": "0.1.0",
            "revision": "unknown",
            "revision_discoverable": False,
            "revision_dirty": False,
            "compiler": {
                "id": "test",
                "version": "1",
                "cpp_standard": 201703,
                "cpp_standard_name": "C++17",
            },
            "build": {
                "type": "Release",
                "generator": "test",
                "configured_cxx_flags": "-O3",
                "openmp_cxx_flags": "",
                "benchmark_target_flags": "",
                "flags_scope": "fixture",
                "assertions_enabled": False,
                "eigen_enabled": False,
            },
            "platform": {
                "operating_system": "test",
                "architecture": "test",
                "cpu_model": "test cpu",
                "logical_threads": 1,
            },
            "openmp": {
                "compile_definition": False,
                "enabled": False,
                "specification_date": 0,
                "maximum_threads": 1,
                "observed_threads": 1,
                "thread_limit": 1,
                "dynamic_teams": False,
                "omp_num_threads_environment": "",
                "omp_proc_bind_environment": "",
                "omp_places_environment": "",
            },
        },
        "configuration": {
            "warmup_runs": 0,
            "timed_runs": 3,
            "requested_steps": 10,
            "requested_sizes_cells_per_axis": [8],
        },
        "results": [
            {
                "name": "fixture",
                "description": "fixture",
                "implementation": "fixture",
                "timed_scope": "fixture",
                "parallel": {
                    "kernel": "none",
                    "openmp_effective_for_workload": False,
                    "observed_threads": 1,
                },
                "workload": {
                    "cells_x": 8,
                    "cells_y": 8,
                    "cell_count": 64,
                    "node_count": 81,
                    "species_count": 1,
                    "step_count": 10,
                    "maximum_time_step": 0.01,
                    "final_time": 0.1,
                    "parameters": {},
                },
                "correctness": {
                    "status": "pass",
                    "invariant": "fixture invariant",
                    "tolerance_basis": "relative_error",
                    "reference_value": 1.0,
                    "observed_value": 1.0,
                    "absolute_error": 0.0,
                    "relative_error": 0.0,
                    "tolerance": 1.0e-12,
                    "solution_checksum": 1.0,
                    "checksum_definition": "fixture",
                    "repeatable_across_runs": True,
                    "maximum_checksum_difference": 0.0,
                },
                "timing_ms": {
                    "clock": "fixture",
                    "samples": samples,
                    "mean": statistics.fmean(samples),
                    "population_stddev": statistics.pstdev(samples),
                    "minimum": min(samples),
                    "maximum": max(samples),
                    "median": statistics.median(samples),
                },
                "throughput": {
                    "basis": "median elapsed time",
                    "node_species_steps_per_second": 810_000.0,
                },
            }
        ],
    }


def test_json_schema_declares_the_versioned_contract() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == (
        "biotransport.performance.v1"
    )
    assert schema["additionalProperties"] is False
    assert {"provenance", "configuration", "results"}.issubset(schema["required"])


def test_semantic_validator_accepts_complete_finite_passing_evidence() -> None:
    document = representative_document()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    _assert_json_schema(document, schema)
    validate_performance_evidence(document)


def test_semantic_validator_rejects_nonfinite_timing() -> None:
    document = representative_document()
    document["results"][0]["timing_ms"]["mean"] = math.inf
    with pytest.raises(AssertionError, match="non-finite"):
        validate_performance_evidence(document)


def test_semantic_validator_rejects_failed_correctness() -> None:
    document = representative_document()
    document["results"][0]["correctness"]["status"] = "fail"
    with pytest.raises(AssertionError):
        validate_performance_evidence(document)


def test_generated_benchmark_evidence_when_present() -> None:
    evidence_files = sorted(EVIDENCE_DIRECTORY.glob("*.json"))
    if not evidence_files:
        pytest.skip(
            "native benchmark evidence has not been generated in this build tree"
        )
    for evidence_file in evidence_files:
        document = json.loads(evidence_file.read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        _assert_json_schema(document, schema)
        validate_performance_evidence(document)
