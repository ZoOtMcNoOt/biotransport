"""Machine-readable provenance records for scientific model parameters.

The records in this module describe where a parameter value came from and the
conditions under which it may be used.  They support traceability; they do not
by themselves establish that a model, dataset, or workflow is FAIR, calibrated,
validated, or suitable for a patient-specific decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional, Tuple, Union
from urllib.parse import urlparse


ParameterValue = Union[int, float, str, Tuple[float, ...]]
"""JSON-compatible scalar or numeric tuple stored with a parameter record."""


class EvidenceLevel(str, Enum):
    """Strength and kind of evidence associated with a parameter value."""

    UNPROVENANCED = "unprovenanced"
    PRIMARY_MEASUREMENT = "primary_measurement"
    REVIEW = "review"
    CONSENSUS_STANDARD = "consensus_standard"
    CALIBRATED = "calibrated"
    VALIDATED = "validated"


class ParameterStatus(str, Enum):
    """Whether the library merely illustrates or recommends a value."""

    ILLUSTRATIVE = "illustrative"
    RECOMMENDED = "recommended"


class UncertaintyKind(str, Enum):
    """Supported machine-readable uncertainty representations."""

    NOT_REPORTED = "not_reported"
    RANGE = "range"
    STANDARD_DEVIATION = "standard_deviation"
    CONFIDENCE_INTERVAL = "confidence_interval"
    EXACT = "exact"


def _finite_optional(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number when provided")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite when provided")
    return converted


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _string_field(
    data: Mapping[str, Any], key: str, *, default: Optional[str] = None
) -> str:
    if key not in data:
        if default is not None:
            return default
        raise ValueError(f"missing required string field {key!r}")
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(f"field {key!r} must be a string")
    return value


def _mapping_field(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"field {key!r} must be an object")
    return value


def _normalise_value(value: Any) -> ParameterValue:
    if isinstance(value, bool):
        raise TypeError("parameter values cannot be bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("numeric parameter values must be finite")
        return value
    if isinstance(value, str):
        return _nonempty(value, "value")
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("tuple parameter values cannot be empty")
        converted = tuple(float(item) for item in value)
        if not all(math.isfinite(item) for item in converted):
            raise ValueError("tuple parameter values must be finite")
        return converted
    raise TypeError("parameter value must be an int, float, string, or numeric tuple")


@dataclass(frozen=True)
class TemperatureContext:
    """Temperature at which a parameter was measured or characterized.

    ``value`` may be omitted only when ``description`` explains why the
    temperature is unknown or not applicable.  Kelvin is preferred for
    numeric values, but the explicit ``unit`` is retained for source fidelity.
    """

    value: Optional[float]
    unit: str
    description: str

    def __post_init__(self) -> None:
        value = _finite_optional(self.value, "temperature")
        unit = _nonempty(self.unit, "temperature unit")
        if value is not None and unit.lower() in {"k", "kelvin"} and value <= 0.0:
            raise ValueError("absolute temperature must be greater than zero kelvin")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(
            self,
            "description",
            _nonempty(self.description, "temperature description"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "description": self.description,
            "unit": self.unit,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TemperatureContext:
        """Construct from :meth:`to_dict` output."""

        return cls(
            value=data.get("value"),
            unit=_string_field(data, "unit"),
            description=_string_field(data, "description"),
        )


@dataclass(frozen=True)
class ValidityRange:
    """Claimed range over which a parameter value is applicable."""

    lower: Optional[float]
    upper: Optional[float]
    unit: str
    description: str

    def __post_init__(self) -> None:
        lower = _finite_optional(self.lower, "validity lower bound")
        upper = _finite_optional(self.upper, "validity upper bound")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("validity lower bound cannot exceed upper bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "unit", _nonempty(self.unit, "validity unit"))
        object.__setattr__(
            self,
            "description",
            _nonempty(self.description, "validity description"),
        )

    def contains(self, value: float) -> bool:
        """Return whether a finite scalar falls inside the inclusive range."""

        converted = float(value)
        if not math.isfinite(converted):
            return False
        return (self.lower is None or converted >= self.lower) and (
            self.upper is None or converted <= self.upper
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "description": self.description,
            "lower": self.lower,
            "unit": self.unit,
            "upper": self.upper,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ValidityRange:
        """Construct from :meth:`to_dict` output."""

        return cls(
            lower=data.get("lower"),
            upper=data.get("upper"),
            unit=_string_field(data, "unit"),
            description=_string_field(data, "description"),
        )


@dataclass(frozen=True)
class Uncertainty:
    """Machine-readable uncertainty attached to a parameter value."""

    kind: UncertaintyKind
    unit: str
    description: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    standard_deviation: Optional[float] = None
    confidence_level: Optional[float] = None

    def __post_init__(self) -> None:
        try:
            kind = UncertaintyKind(self.kind)
        except ValueError as error:
            raise ValueError(f"unsupported uncertainty kind: {self.kind!r}") from error
        lower = _finite_optional(self.lower, "uncertainty lower bound")
        upper = _finite_optional(self.upper, "uncertainty upper bound")
        standard_deviation = _finite_optional(
            self.standard_deviation, "uncertainty standard deviation"
        )
        confidence_level = _finite_optional(
            self.confidence_level, "uncertainty confidence level"
        )
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("uncertainty lower bound cannot exceed upper bound")
        if standard_deviation is not None and standard_deviation < 0.0:
            raise ValueError("uncertainty standard deviation must be nonnegative")
        if confidence_level is not None and not 0.0 < confidence_level < 1.0:
            raise ValueError("uncertainty confidence level must lie in (0, 1)")
        if kind is UncertaintyKind.RANGE and (lower is None or upper is None):
            raise ValueError("range uncertainty requires lower and upper bounds")
        if kind is UncertaintyKind.STANDARD_DEVIATION and standard_deviation is None:
            raise ValueError("standard-deviation uncertainty requires a value")
        if kind is UncertaintyKind.CONFIDENCE_INTERVAL:
            if lower is None or upper is None or confidence_level is None:
                raise ValueError(
                    "confidence-interval uncertainty requires bounds and confidence level"
                )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "unit", _nonempty(self.unit, "uncertainty unit"))
        object.__setattr__(
            self,
            "description",
            _nonempty(self.description, "uncertainty description"),
        )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "standard_deviation", standard_deviation)
        object.__setattr__(self, "confidence_level", confidence_level)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "confidence_level": self.confidence_level,
            "description": self.description,
            "kind": self.kind.value,
            "lower": self.lower,
            "standard_deviation": self.standard_deviation,
            "unit": self.unit,
            "upper": self.upper,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Uncertainty:
        """Construct from :meth:`to_dict` output."""

        return cls(
            kind=UncertaintyKind(_string_field(data, "kind")),
            unit=_string_field(data, "unit"),
            description=_string_field(data, "description"),
            lower=data.get("lower"),
            upper=data.get("upper"),
            standard_deviation=data.get("standard_deviation"),
            confidence_level=data.get("confidence_level"),
        )


@dataclass(frozen=True)
class ParameterProvenance:
    """Provenance, applicability, and uncertainty for one parameter value."""

    parameter_name: str
    value: ParameterValue
    unit: str
    source_identifier: str
    citation: str
    url: Optional[str]
    population_or_material: str
    temperature: TemperatureContext
    measurement_method: str
    validity_range: ValidityRange
    uncertainty: Uncertainty
    evidence_level: EvidenceLevel
    status: ParameterStatus
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parameter_name", _nonempty(self.parameter_name, "parameter_name")
        )
        object.__setattr__(self, "value", _normalise_value(self.value))
        object.__setattr__(self, "unit", _nonempty(self.unit, "parameter unit"))
        object.__setattr__(
            self,
            "source_identifier",
            _nonempty(self.source_identifier, "source_identifier"),
        )
        object.__setattr__(self, "citation", _nonempty(self.citation, "citation"))
        object.__setattr__(
            self,
            "population_or_material",
            _nonempty(self.population_or_material, "population_or_material"),
        )
        object.__setattr__(
            self,
            "measurement_method",
            _nonempty(self.measurement_method, "measurement_method"),
        )
        try:
            evidence_level = EvidenceLevel(self.evidence_level)
            status = ParameterStatus(self.status)
        except ValueError as error:
            raise ValueError(
                "unsupported evidence level or parameter status"
            ) from error
        object.__setattr__(self, "evidence_level", evidence_level)
        object.__setattr__(self, "status", status)
        if not isinstance(self.temperature, TemperatureContext):
            raise TypeError("temperature must be a TemperatureContext")
        if not isinstance(self.validity_range, ValidityRange):
            raise TypeError("validity_range must be a ValidityRange")
        if not isinstance(self.uncertainty, Uncertainty):
            raise TypeError("uncertainty must be an Uncertainty")
        if self.validity_range.unit != self.unit:
            raise ValueError("validity-range unit must match the parameter unit")
        if self.uncertainty.unit != self.unit:
            raise ValueError("uncertainty unit must match the parameter unit")
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")

        if self.url is not None:
            url = _nonempty(self.url, "url")
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("url must be an absolute HTTP(S) URL")
            object.__setattr__(self, "url", url)

        if status is ParameterStatus.RECOMMENDED:
            self._validate_recommended_completeness()

    def _validate_recommended_completeness(self) -> None:
        missing = []
        source_text = f"{self.source_identifier} {self.citation}".lower()
        context_text = (
            f"{self.population_or_material} {self.measurement_method}"
        ).lower()
        missing_markers = (
            "unprovenanced",
            "not reported",
            "unspecified",
            "unknown",
            "no external source",
            "not available",
        )
        if any(marker in source_text for marker in missing_markers):
            missing.append("source identifier/citation")
        if self.url is None:
            missing.append("URL")
        if any(marker in context_text for marker in missing_markers):
            missing.append("population/material or measurement method")
        temperature_text = self.temperature.description.lower()
        if self.temperature.value is None and "not applicable" not in temperature_text:
            missing.append("measurement temperature")
        if self.validity_range.lower is None or self.validity_range.upper is None:
            missing.append("finite validity range")
        if self.uncertainty.kind is UncertaintyKind.NOT_REPORTED:
            missing.append("uncertainty")
        if self.evidence_level is EvidenceLevel.UNPROVENANCED:
            missing.append("evidence level")
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"recommended parameter {self.parameter_name!r} has incomplete "
                f"provenance: {joined}"
            )
        scalar_values: Tuple[float, ...] = ()
        if isinstance(self.value, (int, float)):
            scalar_values = (float(self.value),)
        elif isinstance(self.value, tuple):
            scalar_values = self.value
        if any(not self.validity_range.contains(value) for value in scalar_values):
            raise ValueError(
                f"recommended parameter {self.parameter_name!r} lies outside its "
                "claimed validity range"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation with no hidden state."""

        value: Any = self.value
        if isinstance(value, tuple):
            value = list(value)
        return {
            "citation": self.citation,
            "evidence_level": self.evidence_level.value,
            "measurement_method": self.measurement_method,
            "notes": self.notes,
            "parameter_name": self.parameter_name,
            "population_or_material": self.population_or_material,
            "source_identifier": self.source_identifier,
            "status": self.status.value,
            "temperature": self.temperature.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "unit": self.unit,
            "url": self.url,
            "validity_range": self.validity_range.to_dict(),
            "value": value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParameterProvenance:
        """Construct from :meth:`to_dict` output."""

        return cls(
            parameter_name=_string_field(data, "parameter_name"),
            value=_normalise_value(data["value"]),
            unit=_string_field(data, "unit"),
            source_identifier=_string_field(data, "source_identifier"),
            citation=_string_field(data, "citation"),
            url=(None if data.get("url") is None else _string_field(data, "url")),
            population_or_material=_string_field(data, "population_or_material"),
            temperature=TemperatureContext.from_dict(
                _mapping_field(data, "temperature")
            ),
            measurement_method=_string_field(data, "measurement_method"),
            validity_range=ValidityRange.from_dict(
                _mapping_field(data, "validity_range")
            ),
            uncertainty=Uncertainty.from_dict(_mapping_field(data, "uncertainty")),
            evidence_level=EvidenceLevel(_string_field(data, "evidence_level")),
            status=ParameterStatus(_string_field(data, "status")),
            notes=_string_field(data, "notes", default=""),
        )


@dataclass(frozen=True)
class ParameterSetProvenance:
    """Deterministic, validated collection of parameter provenance records."""

    model_identifier: str
    records: Tuple[ParameterProvenance, ...]
    notes: str = ""
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_identifier",
            _nonempty(self.model_identifier, "model_identifier"),
        )
        object.__setattr__(
            self, "schema_version", _nonempty(self.schema_version, "schema_version")
        )
        if self.schema_version != "1.0":
            raise ValueError(
                f"unsupported parameter provenance schema {self.schema_version!r}"
            )
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")
        if not all(isinstance(record, ParameterProvenance) for record in self.records):
            raise TypeError("records must contain only ParameterProvenance objects")
        records = tuple(
            sorted(tuple(self.records), key=lambda item: item.parameter_name)
        )
        names = [record.parameter_name for record in records]
        if len(names) != len(set(names)):
            raise ValueError("parameter provenance names must be unique")
        if not records:
            raise ValueError("a parameter provenance set cannot be empty")
        object.__setattr__(self, "records", records)

    def record(self, parameter_name: str) -> ParameterProvenance:
        """Return one record or fail loudly when the name is absent."""

        for record in self.records:
            if record.parameter_name == parameter_name:
                return record
        raise KeyError(f"no provenance record for parameter {parameter_name!r}")

    def with_record(self, replacement: ParameterProvenance) -> ParameterSetProvenance:
        """Return a new set with one record inserted or replaced."""

        records = [
            record
            for record in self.records
            if record.parameter_name != replacement.parameter_name
        ]
        records.append(replacement)
        return ParameterSetProvenance(
            model_identifier=self.model_identifier,
            records=tuple(records),
            notes=self.notes,
            schema_version=self.schema_version,
        )

    def validate_parameter_values(
        self,
        values: Mapping[str, ParameterValue],
        *,
        require_exact_names: bool = True,
    ) -> None:
        """Ensure records match the named values used by a configuration."""

        expected_names = set(values)
        record_names = {record.parameter_name for record in self.records}
        missing = sorted(expected_names - record_names)
        unexpected = sorted(record_names - expected_names)
        if missing or (require_exact_names and unexpected):
            details = []
            if missing:
                details.append(f"missing records: {', '.join(missing)}")
            if require_exact_names and unexpected:
                details.append(f"unexpected records: {', '.join(unexpected)}")
            raise ValueError(
                "parameter provenance does not match configuration: "
                + "; ".join(details)
            )

        for name, value in values.items():
            expected = _normalise_value(value)
            actual = self.record(name).value
            if actual != expected:
                raise ValueError(
                    f"provenance value for {name!r} is stale: record has "
                    f"{actual!r}, configuration has {expected!r}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "model_identifier": self.model_identifier,
            "notes": self.notes,
            "records": [record.to_dict() for record in self.records],
            "schema_version": self.schema_version,
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Serialize deterministically; keys and records have stable ordering."""

        separators = (",", ":") if indent is None else None
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            separators=separators,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> ParameterSetProvenance:
        """Deserialize and fully revalidate a provenance collection."""

        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("parameter provenance JSON must contain an object")
        raw_records = data.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("parameter provenance JSON records must be a list")
        if not all(isinstance(item, dict) for item in raw_records):
            raise ValueError("each parameter provenance JSON record must be an object")
        return cls(
            model_identifier=_string_field(data, "model_identifier"),
            records=tuple(ParameterProvenance.from_dict(item) for item in raw_records),
            notes=_string_field(data, "notes", default=""),
            schema_version=_string_field(data, "schema_version"),
        )

    def fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of canonical compact JSON."""

        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def illustrative_parameter_set(
    model_identifier: str,
    values: Mapping[str, ParameterValue],
    units: Mapping[str, str],
    *,
    population_or_material: str,
    notes: str = "",
) -> ParameterSetProvenance:
    """Build explicit unprovenanced records for exploratory input values.

    This helper is intentionally conservative.  It never upgrades a value to
    ``recommended`` and never invents a literature source or measurement
    condition for a library default or user-supplied value.
    """

    missing_units = sorted(set(values) - set(units))
    if missing_units:
        raise ValueError(f"units are missing for: {', '.join(missing_units)}")
    records = []
    for name, value in values.items():
        unit = units[name]
        records.append(
            ParameterProvenance(
                parameter_name=name,
                value=value,
                unit=unit,
                source_identifier="biotransport:illustrative-unprovenanced",
                citation=(
                    "No external source is asserted; this is an illustrative "
                    "model input."
                ),
                url=None,
                population_or_material=population_or_material,
                temperature=TemperatureContext(
                    value=None,
                    unit="K",
                    description="Measurement temperature not reported.",
                ),
                measurement_method=(
                    "Not reported; value is an illustrative model input."
                ),
                validity_range=ValidityRange(
                    lower=None,
                    upper=None,
                    unit=unit,
                    description="No empirical validity range is asserted.",
                ),
                uncertainty=Uncertainty(
                    kind=UncertaintyKind.NOT_REPORTED,
                    unit=unit,
                    description="Uncertainty not reported.",
                ),
                evidence_level=EvidenceLevel.UNPROVENANCED,
                status=ParameterStatus.ILLUSTRATIVE,
            )
        )
    return ParameterSetProvenance(
        model_identifier=model_identifier,
        records=tuple(records),
        notes=notes,
    )


__all__ = [
    "EvidenceLevel",
    "ParameterProvenance",
    "ParameterSetProvenance",
    "ParameterStatus",
    "ParameterValue",
    "TemperatureContext",
    "Uncertainty",
    "UncertaintyKind",
    "ValidityRange",
    "illustrative_parameter_set",
]
