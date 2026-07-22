"""Deterministic, machine-readable records for numerical publications.

This module records enough information to identify a numerical result without
silently claiming that the underlying physical model has been validated.  Its
canonical JSON format rejects non-finite and unsupported values, configuration
and manifest payloads receive SHA-256 fingerprints, and writes use a temporary
file followed by an atomic replacement on the destination filesystem.

Stable manifests omit timestamps and run IDs by default.  Set
``include_volatile=True`` only when provenance for an individual execution is
more important than byte-for-byte reproducibility.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import importlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import unicodedata
import uuid
from datetime import datetime, timezone
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

JSONScalar = Union[None, bool, int, float, str]
JSONValue = Union[JSONScalar, List["JSONValue"], Dict[str, "JSONValue"]]

CANONICAL_JSON_SCHEMA = "biotransport.canonical-json/v1"
FROZEN_CONFIG_SCHEMA = "biotransport.frozen-config/v1"
METHOD_SCHEMA = "biotransport.method/v1"
CONVERGENCE_SCHEMA = "biotransport.convergence-table/v1"
BALANCE_SCHEMA = "biotransport.balance-residual/v1"
MANIFEST_SCHEMA = "biotransport.result-manifest/v1"


class ReproducibilityError(ValueError):
    """Raised when data cannot be represented without ambiguity or data loss."""


def _location_child(location: str, key: Union[str, int]) -> str:
    if isinstance(key, int):
        return f"{location}[{key}]"
    return f"{location}.{key}"


def _normalize(
    value: Any, location: str = "$", active: Optional[Set[int]] = None
) -> JSONValue:
    """Convert supported values to a detached, strict JSON value."""
    if active is None:
        active = set()

    if isinstance(value, Enum):
        return _normalize(value.value, location, active)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ReproducibilityError(f"{location} must be finite, got {number!r}")
        # Canonicalize signed zero so equivalent numerical inputs have one encoding.
        return 0.0 if number == 0.0 else number
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Path):
        raise ReproducibilityError(
            f"{location} is a filesystem path; paths are intentionally not collected. "
            "Record a portable identifier or content fingerprint instead."
        )
    if isinstance(value, np.ndarray):
        identity = id(value)
        if identity in active:
            raise ReproducibilityError(f"{location} contains a reference cycle")
        active.add(identity)
        try:
            return _normalize(value.tolist(), location, active)
        finally:
            active.remove(identity)
    if isinstance(value, np.generic):
        return _normalize(value.item(), location, active)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        identity = id(value)
        if identity in active:
            raise ReproducibilityError(f"{location} contains a reference cycle")
        active.add(identity)
        try:
            dataclass_record = {
                field.name: _normalize(
                    getattr(value, field.name),
                    _location_child(location, field.name),
                    active,
                )
                for field in dataclasses.fields(value)
            }
            return dataclass_record
        finally:
            active.remove(identity)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ReproducibilityError(f"{location} contains a reference cycle")
        active.add(identity)
        try:
            mapping_record: Dict[str, JSONValue] = {}
            for raw_key, raw_value in value.items():
                if not isinstance(raw_key, str):
                    raise ReproducibilityError(
                        f"{location} has a non-string key {raw_key!r}; JSON object keys "
                        "must be strings"
                    )
                key = unicodedata.normalize("NFC", raw_key)
                if key in mapping_record:
                    raise ReproducibilityError(
                        f"{location} contains duplicate key {key!r} after Unicode normalization"
                    )
                mapping_record[key] = _normalize(
                    raw_value, _location_child(location, key), active
                )
            return mapping_record
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ReproducibilityError(f"{location} contains a reference cycle")
        active.add(identity)
        try:
            return [
                _normalize(item, _location_child(location, index), active)
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)

    raise ReproducibilityError(
        f"{location} has unsupported type {type(value).__name__}; convert it to explicit "
        "JSON data first"
    )


def canonical_json(value: Any) -> str:
    """Return deterministic compact JSON for supported finite data.

    Object keys are sorted, Unicode strings are normalized to NFC, tuples and
    NumPy arrays become JSON arrays, and negative zero becomes ``0.0``.  Sets,
    paths, callbacks, non-string mapping keys, reference cycles, NaN, and
    infinities are rejected instead of being serialized ambiguously.
    """
    normalized = _normalize(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_bytes(value: Any, *, trailing_newline: bool = False) -> bytes:
    """Return the UTF-8 bytes used for fingerprints and manifest files."""
    encoded = canonical_json(value).encode("utf-8")
    return encoded + (b"\n" if trailing_newline else b"")


def sha256_fingerprint(value: Any) -> str:
    """Return the lowercase SHA-256 digest of a value's canonical JSON bytes."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fingerprint_record(value: Any) -> Dict[str, JSONValue]:
    return {"algorithm": "sha256", "value": sha256_fingerprint(value)}


def freeze_config(config: Any) -> Dict[str, JSONValue]:
    """Detach and fingerprint a finite JSON-compatible configuration snapshot."""
    values = _normalize(config, "$.configuration")
    if not isinstance(values, dict):
        raise ReproducibilityError("configuration must normalize to a JSON object")
    return {
        "schema": FROZEN_CONFIG_SCHEMA,
        "values": values,
        "fingerprint": _fingerprint_record(values),
    }


def _distribution_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _native_build_metadata() -> Dict[str, JSONValue]:
    try:
        core = importlib.import_module("._core", package=__package__)
        collector = getattr(core, "native_build_info", None)
        if collector is None or not callable(collector):
            return {"available": False, "status": "binding_unavailable"}
        data = _normalize(collector(), "$.software.native")
        if not isinstance(data, dict):
            return {"available": False, "status": "invalid_binding_result"}
        data["available"] = True
        data["status"] = "reported_by_loaded_extension"
        return data
    except Exception:
        # Exception text can contain local paths or other host-specific details.
        return {"available": False, "status": "query_failed"}


def _canonical_utc_timestamp(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReproducibilityError("created_utc must be a non-empty UTC timestamp")
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise ReproducibilityError(
            "created_utc must be an ISO 8601 timestamp with a UTC offset"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReproducibilityError("created_utc must include a UTC offset")
    return (
        parsed.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def software_metadata(
    *,
    include_volatile: bool = False,
    created_utc: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, JSONValue]:
    """Collect portable software and build provenance without host/user paths.

    Stable metadata includes package versions, Python implementation/version,
    operating-system family/release/machine architecture, and metadata reported
    by the loaded C++ extension.  It deliberately excludes username, hostname,
    current directory, executable path, environment variables, and command line.

    ``created_utc`` and ``run_id`` are accepted only when ``include_volatile``
    is true.  Supplying them explicitly is useful for deterministic tests or an
    externally managed run identifier.
    """
    if not include_volatile and (created_utc is not None or run_id is not None):
        raise ReproducibilityError(
            "created_utc and run_id require include_volatile=True"
        )

    metadata: Dict[str, JSONValue] = {
        "packages": {
            "biotransport": _distribution_version("biotransport"),
            "matplotlib": _distribution_version("matplotlib"),
            "numpy": _distribution_version("numpy"),
            "scipy": _distribution_version("scipy"),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag or "unknown",
        },
        "platform": {
            "system": platform.system() or "unknown",
            "release": platform.release() or "unknown",
            "machine": platform.machine() or "unknown",
        },
        "native": _native_build_metadata(),
    }

    if include_volatile:
        timestamp = created_utc or datetime.now(timezone.utc).isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z")
        identifier = run_id or uuid.uuid4().hex
        if not isinstance(identifier, str) or not identifier.strip():
            raise ReproducibilityError("run_id must be a non-empty string")
        metadata["run"] = {
            "created_utc": _canonical_utc_timestamp(timestamp),
            "run_id": unicodedata.normalize("NFC", identifier),
        }

    return metadata


def _nonempty_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReproducibilityError(f"{name} must be a non-empty string")
    return unicodedata.normalize("NFC", value.strip())


def method_metadata(
    name: str,
    *,
    parameters: Optional[Mapping[str, Any]] = None,
    random_seed: Optional[int] = None,
    deterministic: Optional[bool] = None,
    implementation: str = "BioTransport C++",
) -> Dict[str, JSONValue]:
    """Create explicit numerical-method and random-seed metadata.

    ``deterministic=None`` records that no determinism claim was supplied.  A
    seed is restricted to the portable unsigned 64-bit range and is recorded
    even when it is zero.
    """
    if random_seed is not None:
        if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
            raise ReproducibilityError("random_seed must be an integer or None")
        if random_seed < 0 or random_seed >= 2**64:
            raise ReproducibilityError("random_seed must be in [0, 2**64)")
    if deterministic is not None and not isinstance(deterministic, bool):
        raise ReproducibilityError("deterministic must be bool or None")

    normalized_parameters = _normalize(parameters or {}, "$.method.parameters")
    if not isinstance(normalized_parameters, dict):
        raise ReproducibilityError("method parameters must normalize to a JSON object")
    return {
        "schema": METHOD_SCHEMA,
        "name": _nonempty_text(name, "method name"),
        "implementation": _nonempty_text(implementation, "implementation"),
        "parameters": normalized_parameters,
        "random_seed": int(random_seed) if random_seed is not None else None,
        "deterministic": deterministic,
    }


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ReproducibilityError(f"{location} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ReproducibilityError(f"{location} must be finite")
    return 0.0 if number == 0.0 else number


def convergence_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    refinement_parameter: str,
    quantity: str,
    expected_order: Optional[float] = None,
    units: Optional[Mapping[str, str]] = None,
    study: Optional[str] = None,
) -> Dict[str, JSONValue]:
    """Validate and record a coarse-to-fine numerical convergence table.

    Every row must contain the named refinement parameter and quantity.  The
    refinement values must be finite, positive, unique, and strictly decreasing.
    Optional ``error`` values must be nonnegative, and optional
    ``observed_order`` values must be finite or ``None``.  The function records
    supplied evidence; it does not infer model validity or standards compliance.
    """
    refinement_name = _nonempty_text(refinement_parameter, "refinement_parameter")
    quantity_name = _nonempty_text(quantity, "quantity")
    if not rows:
        raise ReproducibilityError("convergence rows must not be empty")

    normalized_rows: List[JSONValue] = []
    refinements: List[float] = []
    columns: Set[str] = set()
    for index, raw_row in enumerate(rows):
        row = _normalize(raw_row, f"$.convergence.rows[{index}]")
        if not isinstance(row, dict):
            raise ReproducibilityError(f"convergence row {index} must be a JSON object")
        if refinement_name not in row or quantity_name not in row:
            raise ReproducibilityError(
                f"convergence row {index} must contain {refinement_name!r} and "
                f"{quantity_name!r}"
            )
        refinement = _finite_number(
            row[refinement_name], f"$.convergence.rows[{index}].{refinement_name}"
        )
        if refinement <= 0.0:
            raise ReproducibilityError("convergence refinement values must be positive")
        row[refinement_name] = refinement
        row[quantity_name] = _finite_number(
            row[quantity_name], f"$.convergence.rows[{index}].{quantity_name}"
        )
        if "error" in row:
            error = _finite_number(row["error"], f"$.convergence.rows[{index}].error")
            if error < 0.0:
                raise ReproducibilityError("convergence errors must be nonnegative")
            row["error"] = error
        if "observed_order" in row and row["observed_order"] is not None:
            row["observed_order"] = _finite_number(
                row["observed_order"],
                f"$.convergence.rows[{index}].observed_order",
            )
        refinements.append(refinement)
        columns.update(row.keys())
        normalized_rows.append(row)

    if any(left <= right for left, right in zip(refinements, refinements[1:])):
        raise ReproducibilityError(
            "convergence refinement values must be strictly decreasing (coarse to fine)"
        )

    order: Optional[float] = None
    if expected_order is not None:
        order = _finite_number(expected_order, "$.convergence.expected_order")
        if order <= 0.0:
            raise ReproducibilityError("expected_order must be positive")

    normalized_units = _normalize(units or {}, "$.convergence.units")
    if not isinstance(normalized_units, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in normalized_units.items()
    ):
        raise ReproducibilityError("convergence units must map column names to strings")

    return {
        "schema": CONVERGENCE_SCHEMA,
        "study": _nonempty_text(study or quantity_name, "convergence study"),
        "refinement_parameter": refinement_name,
        "quantity": quantity_name,
        "expected_order": order,
        "units": normalized_units,
        "columns": _normalize(sorted(columns)),
        "rows": normalized_rows,
    }


def balance_residual(
    name: str,
    *,
    initial_inventory: float,
    final_inventory: float,
    time_integrated_source: float = 0.0,
    time_integrated_boundary_outflow: float = 0.0,
    units: str = "unspecified",
    normalization: Optional[float] = None,
    relative_tolerance: Optional[float] = None,
) -> Dict[str, JSONValue]:
    """Record a signed integral-balance residual.

    The convention is

    ``residual = final - initial - integrated_source + integrated_outflow``.

    Thus a positive source adds inventory and a positive outward boundary flux
    removes it.  Integrals must already include their time/area factors.
    """
    initial = _finite_number(initial_inventory, "$.balance.initial_inventory")
    final = _finite_number(final_inventory, "$.balance.final_inventory")
    source = _finite_number(time_integrated_source, "$.balance.time_integrated_source")
    outflow = _finite_number(
        time_integrated_boundary_outflow,
        "$.balance.time_integrated_boundary_outflow",
    )
    residual = final - initial - source + outflow
    if not math.isfinite(residual):
        raise ReproducibilityError(
            "balance arithmetic overflowed; rescale the inventory and integrated terms"
        )

    if normalization is None:
        scale = max(abs(initial), abs(final), abs(source), abs(outflow))
    else:
        scale = _finite_number(normalization, "$.balance.normalization")
        if scale <= 0.0:
            raise ReproducibilityError("balance normalization must be positive")
    relative = 0.0 if scale == 0.0 else residual / scale

    tolerance: Optional[float] = None
    within_tolerance: Optional[bool] = None
    if relative_tolerance is not None:
        tolerance = _finite_number(relative_tolerance, "$.balance.relative_tolerance")
        if tolerance < 0.0:
            raise ReproducibilityError("relative_tolerance must be nonnegative")
        within_tolerance = abs(relative) <= tolerance

    return {
        "schema": BALANCE_SCHEMA,
        "name": _nonempty_text(name, "balance name"),
        "units": _nonempty_text(units, "balance units"),
        "sign_convention": (
            "residual = final_inventory - initial_inventory - "
            "time_integrated_source + time_integrated_boundary_outflow"
        ),
        "initial_inventory": initial,
        "final_inventory": final,
        "time_integrated_source": source,
        "time_integrated_boundary_outflow": outflow,
        "signed_residual": residual,
        "absolute_residual": abs(residual),
        "normalization": scale,
        "relative_residual": relative,
        "relative_tolerance": tolerance,
        "within_tolerance": within_tolerance,
    }


def create_manifest(
    study_name: str,
    *,
    config: Any,
    method: Mapping[str, Any],
    results: Mapping[str, Any],
    convergence: Sequence[Mapping[str, Any]] = (),
    balances: Sequence[Mapping[str, Any]] = (),
    notes: Sequence[str] = (),
    include_volatile: bool = False,
    created_utc: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, JSONValue]:
    """Create a fingerprinted result manifest from explicit evidence records."""
    normalized_method = _normalize(method, "$.method")
    normalized_results = _normalize(results, "$.results")
    normalized_convergence = _normalize(convergence, "$.evidence.convergence")
    normalized_balances = _normalize(balances, "$.evidence.balances")
    normalized_notes = _normalize(notes, "$.notes")
    if not isinstance(normalized_method, dict):
        raise ReproducibilityError("method must normalize to a JSON object")
    if normalized_method.get("schema") != METHOD_SCHEMA:
        raise ReproducibilityError("method must be created with method_metadata()")
    if not isinstance(normalized_results, dict):
        raise ReproducibilityError("results must normalize to a JSON object")
    if not isinstance(normalized_convergence, list) or not all(
        isinstance(record, dict) and record.get("schema") == CONVERGENCE_SCHEMA
        for record in normalized_convergence
    ):
        raise ReproducibilityError(
            "convergence entries must be created with convergence_table()"
        )
    if not isinstance(normalized_balances, list) or not all(
        isinstance(record, dict) and record.get("schema") == BALANCE_SCHEMA
        for record in normalized_balances
    ):
        raise ReproducibilityError(
            "balance entries must be created with balance_residual()"
        )
    if not isinstance(normalized_notes, list) or not all(
        isinstance(note, str) for note in normalized_notes
    ):
        raise ReproducibilityError("notes must be a sequence of strings")

    payload: Dict[str, JSONValue] = {
        "schema": MANIFEST_SCHEMA,
        "canonical_json_schema": CANONICAL_JSON_SCHEMA,
        "study": {"name": _nonempty_text(study_name, "study_name")},
        "configuration": freeze_config(config),
        "method": normalized_method,
        "software": software_metadata(
            include_volatile=include_volatile,
            created_utc=created_utc,
            run_id=run_id,
        ),
        "evidence": {
            "convergence": normalized_convergence,
            "balances": normalized_balances,
        },
        "results": normalized_results,
        "notes": normalized_notes,
    }
    manifest = dict(payload)
    manifest["content_fingerprint"] = _fingerprint_record(payload)
    return manifest


def verify_manifest(manifest: Mapping[str, Any]) -> bool:
    """Return whether a manifest is structurally valid and its fingerprint matches."""
    try:
        normalized = _normalize(manifest, "$")
    except ReproducibilityError:
        return False
    if not isinstance(normalized, dict):
        return False
    if normalized.get("schema") != MANIFEST_SCHEMA:
        return False
    if normalized.get("canonical_json_schema") != CANONICAL_JSON_SCHEMA:
        return False

    study = normalized.get("study")
    configuration = normalized.get("configuration")
    method = normalized.get("method")
    software = normalized.get("software")
    evidence = normalized.get("evidence")
    results = normalized.get("results")
    notes = normalized.get("notes")
    if not isinstance(study, dict) or not isinstance(study.get("name"), str):
        return False
    if not isinstance(configuration, dict):
        return False
    if configuration.get("schema") != FROZEN_CONFIG_SCHEMA:
        return False
    config_values = configuration.get("values")
    config_fingerprint = configuration.get("fingerprint")
    if not isinstance(config_values, dict) or not isinstance(config_fingerprint, dict):
        return False
    if config_fingerprint.get("algorithm") != "sha256":
        return False
    config_digest = config_fingerprint.get("value")
    if not isinstance(config_digest, str) or not hmac.compare_digest(
        config_digest, sha256_fingerprint(config_values)
    ):
        return False
    if not isinstance(method, dict) or method.get("schema") != METHOD_SCHEMA:
        return False
    if not isinstance(software, dict) or not isinstance(results, dict):
        return False
    if not isinstance(evidence, dict):
        return False
    convergence = evidence.get("convergence")
    balances = evidence.get("balances")
    if not isinstance(convergence, list) or not all(
        isinstance(record, dict) and record.get("schema") == CONVERGENCE_SCHEMA
        for record in convergence
    ):
        return False
    if not isinstance(balances, list) or not all(
        isinstance(record, dict) and record.get("schema") == BALANCE_SCHEMA
        for record in balances
    ):
        return False
    if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
        return False

    fingerprint = normalized.pop("content_fingerprint", None)
    if not isinstance(fingerprint, dict):
        return False
    if fingerprint.get("algorithm") != "sha256":
        return False
    recorded = fingerprint.get("value")
    if not isinstance(recorded, str):
        return False
    expected = sha256_fingerprint(normalized)
    return hmac.compare_digest(recorded, expected)


def _reject_duplicate_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReproducibilityError(f"manifest JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def load_manifest(path: Union[str, os.PathLike[str]]) -> Dict[str, JSONValue]:
    """Load strict JSON from disk and verify its content fingerprint."""
    target = Path(path)

    def reject_constant(value: str) -> None:
        raise ReproducibilityError(f"manifest JSON contains non-finite value {value}")

    try:
        raw = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReproducibilityError(
            "could not read a valid UTF-8 JSON manifest"
        ) from error
    normalized = _normalize(raw)
    if not isinstance(normalized, dict) or not verify_manifest(normalized):
        raise ReproducibilityError("manifest fingerprint does not match its content")
    return normalized


def write_manifest(
    path: Union[str, os.PathLike[str]],
    manifest: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a verified canonical manifest with one trailing newline.

    The temporary file is created beside the destination, flushed, and replaced
    atomically.  ``overwrite=False`` performs a no-clobber preflight check; the
    standard library cannot make that check and replacement one indivisible
    operation on every supported platform.
    """
    normalized = _normalize(manifest)
    if not isinstance(normalized, dict) or not verify_manifest(normalized):
        raise ReproducibilityError(
            "write_manifest requires an untampered manifest from create_manifest()"
        )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing manifest: {target.name}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(normalized, trailing_newline=True))
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite existing manifest: {target.name}"
            )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


__all__ = [
    "BALANCE_SCHEMA",
    "CANONICAL_JSON_SCHEMA",
    "CONVERGENCE_SCHEMA",
    "FROZEN_CONFIG_SCHEMA",
    "MANIFEST_SCHEMA",
    "METHOD_SCHEMA",
    "ReproducibilityError",
    "balance_residual",
    "canonical_json",
    "canonical_json_bytes",
    "convergence_table",
    "create_manifest",
    "freeze_config",
    "load_manifest",
    "method_metadata",
    "sha256_fingerprint",
    "software_metadata",
    "verify_manifest",
    "write_manifest",
]
