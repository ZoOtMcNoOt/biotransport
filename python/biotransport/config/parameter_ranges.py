"""Illustrative parameter ranges with explicit provenance limitations.

The values in this module are legacy exploratory defaults. They are not
literature-derived recommendations and must not be treated as population,
material, drug, or patient-specific parameter estimates.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

from biotransport.provenance import (
    ParameterSetProvenance,
    ValidityRange,
    illustrative_parameter_set,
)


_PARAMETER_RANGES: Dict[str, Dict[str, Any]] = {
    "D_drug": {
        "min": 1e-12,
        "max": 1e-9,
        "typical": 1e-11,
        "unit": "m^2/s",
        "description": "Drug diffusion coefficient in tissue",
    },
    "D_oxygen": {
        "min": 1e-10,
        "max": 5e-9,
        "typical": 2e-9,
        "unit": "m^2/s",
        "description": "Oxygen diffusion coefficient in tissue",
    },
    "D_glucose": {
        "min": 1e-10,
        "max": 1e-9,
        "typical": 6e-10,
        "unit": "m^2/s",
        "description": "Glucose diffusion coefficient in tissue",
    },
    "k_tissue": {
        "min": 0.2,
        "max": 0.8,
        "typical": 0.5,
        "unit": "W/(m K)",
        "description": "Thermal conductivity of soft tissue",
    },
    "c_tissue": {
        "min": 3000,
        "max": 4000,
        "typical": 3600,
        "unit": "J/(kg K)",
        "description": "Specific heat of soft tissue",
    },
    "w_blood": {
        "min": 0.0001,
        "max": 0.01,
        "typical": 0.0005,
        "unit": "1/s",
        "description": "Blood perfusion rate",
    },
    "IFP_tumor": {
        "min": 5,
        "max": 60,
        "typical": 20,
        "unit": "mmHg",
        "description": "Interstitial fluid pressure in tumor",
    },
    "MVD": {
        "min": 10,
        "max": 400,
        "typical": 100,
        "unit": "vessels/mm^2",
        "description": "Microvascular density",
    },
}


def get_parameter_range_provenance() -> ParameterSetProvenance:
    """Return traceability records for the legacy illustrative range table."""

    base = illustrative_parameter_set(
        "illustrative_parameter_ranges",
        {name: entry["typical"] for name, entry in _PARAMETER_RANGES.items()},
        {name: str(entry["unit"]) for name, entry in _PARAMETER_RANGES.items()},
        population_or_material=(
            "Generic unspecified biological material; no population is asserted."
        ),
        notes=(
            "The table is retained for exploratory examples. Its bounds are not "
            "empirical applicability limits or recommended priors."
        ),
    )
    records = []
    for record in base.records:
        entry = _PARAMETER_RANGES[record.parameter_name]
        records.append(
            replace(
                record,
                validity_range=ValidityRange(
                    lower=float(entry["min"]),
                    upper=float(entry["max"]),
                    unit=str(entry["unit"]),
                    description=(
                        "Illustrative software range only; no empirical validity "
                        "claim is asserted."
                    ),
                ),
            )
        )
    return ParameterSetProvenance(
        model_identifier=base.model_identifier,
        records=tuple(records),
        notes=base.notes,
        schema_version=base.schema_version,
    )


def get_parameter_ranges() -> Dict[str, Dict[str, Any]]:
    """Return illustrative ranges plus machine-readable provenance.

    The historical ``min``, ``max``, ``typical``, ``unit``, and
    ``description`` entries remain compatible. Each entry now also contains a
    ``provenance`` dictionary. All bundled records are explicitly
    ``illustrative`` and ``unprovenanced``.

    Examples
    --------
    >>> ranges = get_parameter_ranges()
    >>> print(ranges["D_drug"]["typical"])
    1e-11
    >>> ranges["D_drug"]["provenance"]["status"]
    'illustrative'
    """

    provenance = get_parameter_range_provenance()
    result: Dict[str, Dict[str, Any]] = {}
    for name, entry in _PARAMETER_RANGES.items():
        result[name] = dict(entry)
        result[name]["provenance"] = provenance.record(name).to_dict()
    return result


__all__ = ["get_parameter_range_provenance", "get_parameter_ranges"]
