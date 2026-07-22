# Parameter provenance

Biotransport parameter values now have a machine-readable traceability model.
The model records what value entered a simulation, where it came from, the
measurement context in which it applies, and how uncertainty was represented.
It does **not** turn a library default into a patient-specific estimate or make
a model clinically validated.

The design supports the traceability goals associated with the FAIR principles
described by Wilkinson et al., *Scientific Data* **3**, 160018 (2016),
[doi:10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18). Providing
these records is not a claim that the library or a downstream workflow fully
complies with FAIR.

## Record contract

`biotransport.provenance.ParameterProvenance` stores:

- the parameter name, exact value, and unit;
- a source identifier, citation, and source URL;
- the population or material;
- the measurement temperature and temperature unit;
- the measurement or assay method;
- a claimed validity range;
- an uncertainty representation (range, standard deviation, confidence
  interval, exact value, or explicitly not reported);
- an evidence level; and
- either `illustrative` or `recommended` status.

`ParameterSetProvenance` groups uniquely named records for one model. Records
are sorted by parameter name, serialized to canonical JSON, and fingerprinted
with SHA-256. The fingerprint identifies the provenance manifest, not the
solver executable, mesh files, boundary data, or simulation outputs.

Any record marked `recommended` must include a non-placeholder source and URL,
a stated material/population and method, a measurement temperature or an
explicit explanation that temperature is not applicable, finite validity
bounds containing the value, a reported uncertainty representation, and an
evidence level other than `unprovenanced`. Construction fails when any of these
fields is incomplete. This is structural validation only: it cannot determine
whether a paper is trustworthy or whether its cohort and assay match a new
study.

## Current bundled values

The existing bioheat, cryotherapy, tumor-delivery, and generic parameter-range
values were introduced as demonstration inputs without a source ledger. The
library therefore labels them honestly as `illustrative` and `unprovenanced`.
Their records state that population, material, measurement method, temperature,
validity, and uncertainty were not reported. None is a recommended prior or a
patient-specific value.

The old `get_parameter_ranges()` keys remain available. Each entry also has a
`provenance` dictionary. Its displayed minimum and maximum are labeled as an
illustrative software range, not an empirical applicability interval.

## Configuration usage

Both application configurations expose a `provenance` property:

```python
import biotransport as bt

cfg = bt.BioheatCryotherapyConfig()
manifest = cfg.provenance

print(manifest.record("rho_tissue").status.value)  # illustrative
print(manifest.to_json(indent=2))
print(manifest.fingerprint())
```

Generated manifests always reflect the current configuration and remain
unprovenanced. A project can replace individual records with sourced records,
then attach the complete manifest:

```python
from dataclasses import replace

record = manifest.record("rho_tissue")
sourced_record = replace(
    record,
    source_identifier="project-dataset-accession",
    citation="Full source citation supplied by the project",
    url="https://repository.example.org/accession",
    # Also replace material, temperature, method, validity, uncertainty,
    # evidence_level, and status using the typed provenance objects.
)

cfg.attach_parameter_provenance(manifest.with_record(sourced_record))
```

The abbreviated replacement above intentionally remains `illustrative`; do not
change it to `recommended` until every required field is supplied. Attached
records are immutable claims. If a configuration value later changes, access
to the manifest and solver-side validation fail with a stale-value error rather
than silently relabeling the new value. Call
`reset_parameter_provenance_as_illustrative()` to discard the claims and create
honest unprovenanced records for the changed values.

A manifest can also be supplied directly to either factory:

```python
base = bt.TumorDrugDeliveryConfig()
manifest = base.provenance
cfg = bt.TumorDrugDeliveryConfig(parameter_provenance=manifest)
```

The model identifier, record names, and every recorded value must match the
configuration exactly. JSON round trips are revalidated on load.

## What a simulation still needs

Parameter traceability is only one part of scientific evidence. Research use
still requires an appropriate governing model, unit and boundary-condition
review, numerical verification, calibration independent from validation,
sensitivity and uncertainty propagation, applicability assessment, and a
reproducible link between the parameter manifest and complete simulation input
and output artifacts.
