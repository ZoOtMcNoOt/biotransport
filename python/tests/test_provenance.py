"""Tests for machine-readable application-parameter provenance."""

from dataclasses import replace
import json

import pytest

from biotransport.config.parameter_ranges import (
    get_parameter_range_provenance,
    get_parameter_ranges,
)
from biotransport.provenance import (
    EvidenceLevel,
    ParameterProvenance,
    ParameterSetProvenance,
    ParameterStatus,
    TemperatureContext,
    Uncertainty,
    UncertaintyKind,
    ValidityRange,
    illustrative_parameter_set,
)


def _recommended_record(
    *, name: str = "coefficient", value: float = 1.0
) -> ParameterProvenance:
    """Create a complete synthetic record used only to exercise validation."""

    return ParameterProvenance(
        parameter_name=name,
        value=value,
        unit="model units",
        source_identifier="synthetic-test-fixture-v1",
        citation="Synthetic test fixture; not a scientific parameter source.",
        url="https://example.org/biotransport-test-fixture",
        population_or_material="synthetic homogeneous phantom",
        temperature=TemperatureContext(
            value=None,
            unit="K",
            description="Not applicable to this dimensionless synthetic fixture.",
        ),
        measurement_method="deterministic synthetic fixture construction",
        validity_range=ValidityRange(
            lower=0.0,
            upper=2.0,
            unit="model units",
            description="Closed test-fixture domain.",
        ),
        uncertainty=Uncertainty(
            kind=UncertaintyKind.EXACT,
            unit="model units",
            description="Exactly specified synthetic value.",
        ),
        evidence_level=EvidenceLevel.VALIDATED,
        status=ParameterStatus.RECOMMENDED,
        notes="Recommendation status is tested mechanically, not scientifically.",
    )


def test_illustrative_records_are_explicitly_unprovenanced() -> None:
    provenance = illustrative_parameter_set(
        "demo",
        {"D": 1e-9, "position": (0.0, 1.0)},
        {"D": "m^2/s", "position": "m (x, y)"},
        population_or_material="unspecified illustrative material",
    )

    diffusion = provenance.record("D")
    assert diffusion.status is ParameterStatus.ILLUSTRATIVE
    assert diffusion.evidence_level is EvidenceLevel.UNPROVENANCED
    assert diffusion.url is None
    assert diffusion.temperature.description
    assert diffusion.measurement_method
    assert diffusion.validity_range.description
    assert diffusion.uncertainty.kind is UncertaintyKind.NOT_REPORTED


def test_recommended_record_fails_loudly_when_provenance_is_incomplete() -> None:
    with pytest.raises(ValueError, match="recommended parameter.*incomplete"):
        ParameterProvenance(
            parameter_name="D",
            value=1e-9,
            unit="m^2/s",
            source_identifier="unprovenanced",
            citation="No external source is asserted.",
            url=None,
            population_or_material="unspecified",
            temperature=TemperatureContext(
                value=None,
                unit="K",
                description="Measurement temperature not reported.",
            ),
            measurement_method="not reported",
            validity_range=ValidityRange(
                lower=None,
                upper=None,
                unit="m^2/s",
                description="No range reported.",
            ),
            uncertainty=Uncertainty(
                kind=UncertaintyKind.NOT_REPORTED,
                unit="m^2/s",
                description="Not reported.",
            ),
            evidence_level=EvidenceLevel.UNPROVENANCED,
            status=ParameterStatus.RECOMMENDED,
        )


def test_recommended_value_must_lie_in_claimed_validity_range() -> None:
    with pytest.raises(ValueError, match="outside its claimed validity range"):
        replace(
            _recommended_record(value=1.0),
            value=3.0,
        )


def test_json_round_trip_and_fingerprint_are_deterministic() -> None:
    first = _recommended_record(name="zeta", value=1.5)
    second = replace(
        _recommended_record(name="alpha", value=0.5),
        uncertainty=Uncertainty(
            kind=UncertaintyKind.STANDARD_DEVIATION,
            unit="model units",
            description="Synthetic one-sigma uncertainty.",
            standard_deviation=0.1,
        ),
    )
    provenance = ParameterSetProvenance(
        model_identifier="synthetic_model",
        records=(first, second),
        notes="Test collection.",
    )
    reversed_order = ParameterSetProvenance(
        model_identifier="synthetic_model",
        records=(second, first),
        notes="Test collection.",
    )

    payload = provenance.to_json()
    restored = ParameterSetProvenance.from_json(payload)
    assert restored == provenance
    assert payload == reversed_order.to_json()
    assert restored.fingerprint() == provenance.fingerprint()
    assert len(provenance.fingerprint()) == 64
    assert json.loads(payload)["records"][0]["parameter_name"] == "alpha"


def test_json_loader_does_not_coerce_missing_metadata_to_strings() -> None:
    provenance = ParameterSetProvenance(
        model_identifier="synthetic_model",
        records=(_recommended_record(),),
    )
    data = json.loads(provenance.to_json())
    data["records"][0]["source_identifier"] = None

    with pytest.raises(ValueError, match="source_identifier.*string"):
        ParameterSetProvenance.from_json(json.dumps(data))

    data = json.loads(provenance.to_json())
    data["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="unsupported.*schema"):
        ParameterSetProvenance.from_json(json.dumps(data))


def test_record_rejects_mismatched_metadata_units() -> None:
    with pytest.raises(ValueError, match="uncertainty unit"):
        replace(
            _recommended_record(),
            uncertainty=Uncertainty(
                kind=UncertaintyKind.EXACT,
                unit="wrong units",
                description="Synthetic mismatch.",
            ),
        )

    with pytest.raises(ValueError, match="absolute temperature"):
        TemperatureContext(
            value=0.0,
            unit="K",
            description="Invalid absolute temperature.",
        )


def test_collection_validation_detects_missing_extra_and_stale_records() -> None:
    provenance = illustrative_parameter_set(
        "demo",
        {"a": 1.0, "b": 2.0},
        {"a": "1", "b": "1"},
        population_or_material="unspecified synthetic material",
    )

    provenance.validate_parameter_values({"a": 1.0, "b": 2.0})
    with pytest.raises(ValueError, match="missing records"):
        provenance.validate_parameter_values({"a": 1.0, "b": 2.0, "c": 3.0})
    with pytest.raises(ValueError, match="unexpected records"):
        provenance.validate_parameter_values({"a": 1.0})
    with pytest.raises(ValueError, match="stale"):
        provenance.validate_parameter_values({"a": 9.0, "b": 2.0})


def test_legacy_parameter_ranges_carry_honest_machine_records() -> None:
    ranges = get_parameter_ranges()
    provenance = get_parameter_range_provenance()

    assert set(ranges) == {record.parameter_name for record in provenance.records}
    for name, entry in ranges.items():
        record = entry["provenance"]
        assert record["parameter_name"] == name
        assert record["value"] == entry["typical"]
        assert record["status"] == "illustrative"
        assert record["evidence_level"] == "unprovenanced"
        assert record["validity_range"]["lower"] == entry["min"]
        assert record["validity_range"]["upper"] == entry["max"]
