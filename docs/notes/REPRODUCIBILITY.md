# Reproducible numerical artifacts

BioTransport can write deterministic JSON manifests that join a frozen model
configuration to numerical evidence, result summaries, method/seed metadata,
software versions, and the build provenance reported by the loaded C++
extension. The purpose is to make a reported calculation identifiable and
auditable. A manifest is not evidence that a physical model is calibrated or
validated.

The design is informed by the original FAIR principles paper:

> Wilkinson, M. D. et al. The FAIR Guiding Principles for scientific data
> management and stewardship. *Scientific Data* **3**, 160018 (2016).
> <https://doi.org/10.1038/sdata.2016.18>

These files do **not** establish FAIR compliance. Stable identifiers, explicit
schemas, machine-readable units and methods, content fingerprints, software
provenance, and reusable numerical evidence support parts of findability,
interoperability, and reuse. A real publication still needs durable repository
hosting, searchable metadata, an appropriate license, access policy, domain
standards, data documentation, and long-term stewardship.

## What a manifest records

The `biotransport.result-manifest/v1` payload contains:

- `configuration`: a detached configuration snapshot and its SHA-256 digest;
- `method`: method name, implementation, numerical parameters, determinism
  declaration, and the exact random seed or `null`;
- `software`: BioTransport, NumPy, SciPy, and Matplotlib versions; Python
  implementation/version; generic operating-system metadata; and native build
  information;
- `evidence.convergence`: explicit coarse-to-fine tables, not a blanket
  verification claim;
- `evidence.balances`: signed conservation/balance residual records with their
  convention, units, scale, and optional tolerance decision;
- `results`: caller-selected finite result values and content hashes;
- `notes`: scope and limitation statements; and
- `content_fingerprint`: SHA-256 of every preceding manifest field.

The native extension reports the compiler ID and version, effective C++
standard, assertion mode, Eigen enablement/version, and both the BioTransport
OpenMP compile definition and compiler OpenMP specification date. Those values
describe the extension that is actually loaded by Python; they are not inferred
from a source-tree CMake cache.

Build metadata deliberately omits compiler executable paths, source paths,
build directories, environment variables, command lines, username, and
hostname. Stable platform metadata is limited to operating-system family,
release, and machine architecture. If a project needs container or hardware
identifiers, record reviewed, portable identifiers explicitly rather than
capturing the host environment wholesale.

## Canonical JSON and fingerprints

`canonical_json()` uses a deliberately small data model:

- JSON null, booleans, integers, finite floats, and strings;
- mappings with string keys;
- lists and tuples;
- dataclass instances and enum values; and
- NumPy scalar values and arrays.

Object keys are sorted, whitespace is removed, text is normalized to Unicode
NFC, negative floating-point zero becomes `0.0`, and UTF-8 is used. NaN,
infinity, reference cycles, sets, callbacks, non-string keys, arbitrary objects,
and filesystem paths raise `ReproducibilityError`. This is BioTransport's
versioned canonical format, not a claim of conformance to an external JSON
canonicalization standard.

The strict treatment matters. Python's ordinary JSON encoder may emit NaN even
though it is not valid JSON, and converting a set to a list can create
nondeterministic byte order. Silently stringifying an object can also capture a
memory address or private path. Publication artifacts fail instead.

`freeze_config(config)` returns the normalized values plus:

```json
{
  "algorithm": "sha256",
  "value": "...64 lowercase hexadecimal characters..."
}
```

The configuration digest covers only `configuration.values`. The manifest's
`content_fingerprint` covers the complete payload except the fingerprint field
itself. `verify_manifest()` and `load_manifest()` recompute that digest and
reject modified content. SHA-256 identifies bytes; it is not a digital
signature and does not authenticate an author.

## Stable and volatile provenance

Manifests omit timestamps and run IDs by default. With the same input data and
software environment, two calls therefore produce the same canonical bytes.
This is useful for regression tests, content-addressed storage, and exact
artifact comparison.

Set `include_volatile=True` when each execution must carry an individual UTC
timestamp and random run ID. Those fields intentionally change the manifest
fingerprint. For workflow systems that already assign IDs, pass explicit
`created_utc` and `run_id` values to `create_manifest()`; they are accepted only
when volatile metadata is enabled, so they cannot be silently ignored.

Volatile metadata is provenance, not entropy for a simulation. A stochastic
calculation must separately record its algorithm and seed through
`method_metadata()` and should freeze sampled parameters in the configuration.

## Minimal workflow

```python
from pathlib import Path

import biotransport.reproducibility as repro

config = {
    "diffusivity_m2_per_s": 1.0e-9,
    "length_m": 1.0e-3,
    "cells": 100,
}

method = repro.method_metadata(
    "conservative explicit finite-volume transport",
    implementation="BioTransport C++ canonical transport solver",
    parameters={"safety_factor": 0.8},
    random_seed=None,
    deterministic=True,
)

balance = repro.balance_residual(
    "solute amount",
    initial_inventory=1.0,
    final_inventory=0.9999999999999,
    units="mol",
    relative_tolerance=1.0e-12,
)

manifest = repro.create_manifest(
    "closed-domain diffusion",
    config=config,
    method=method,
    results={"passed": balance["within_tolerance"]},
    balances=[balance],
    notes=["Numerical evidence only; physical validation is out of scope."],
)

repro.write_manifest(Path("results") / "manifest.json", manifest)
```

`write_manifest()` encodes one canonical JSON document with a final newline. It
writes and flushes a temporary file beside the destination before using the
operating system's atomic replacement operation. It refuses to overwrite by
default. The no-clobber preflight and replacement cannot be one indivisible
operation on every Python-supported platform, so concurrent writers should use
unique destination names or external coordination.

## Convergence evidence

`convergence_table()` accepts rows ordered from coarse to fine. Every row must
contain the named positive refinement parameter and finite quantity. Optional
`error` values must be nonnegative; optional `observed_order` values must be
finite or `null`.

```python
table = repro.convergence_table(
    [
        {"h_m": 0.10, "l2_error": 4.0e-2, "observed_order": None},
        {"h_m": 0.05, "l2_error": 1.0e-2, "observed_order": 2.0},
        {"h_m": 0.025, "l2_error": 2.5e-3, "observed_order": 2.0},
    ],
    refinement_parameter="h_m",
    quantity="l2_error",
    expected_order=2.0,
    units={"h_m": "m", "l2_error": "1"},
    study="manufactured diffusion solution",
)
```

The function validates and records supplied evidence. It does not decide that a
sequence is asymptotic, prove an implementation correct, perform an ASME
assessment, or validate a biological model. Record the equation, norm, domain,
boundary conditions, refinement policy, and reference solution in the frozen
configuration or accompanying publication.

## Balance residuals

`balance_residual()` uses one explicit sign convention:

```text
residual = final_inventory
         - initial_inventory
         - time_integrated_source
         + time_integrated_boundary_outflow
```

A positive source adds inventory; a positive outward boundary flux removes it.
Inputs named `time_integrated_*` must already include time and area/volume
integration. The default relative scale is the largest magnitude among the
four balance terms. An all-zero balance has a defined relative residual of
zero. A caller can supply a documented positive normalization instead.

The optional tolerance produces `within_tolerance`, but the manifest also keeps
the signed residual, absolute residual, normalization, and tolerance. Readers
can therefore audit the decision rather than receiving only a pass/fail label.
Choose tolerances from discretization and solver behavior; do not select them
after seeing the desired answer.

## Seed and method metadata

`method_metadata()` always writes the `random_seed` field. Use `null` when no
random seed was used. Seeds must fit an unsigned 64-bit integer. Also record the
generator family (for example, NumPy `PCG64`), sampling procedure, and every
sampled physical parameter in the frozen configuration. A seed alone does not
make results portable across different random-number algorithms or library
versions.

`deterministic=True` is a claim made by the caller. It should mean that the
recorded method and configuration reproduce the same numerical values under
the stated execution conditions. OpenMP or other parallel reductions may need
an explicit thread count and a validated deterministic algorithm. The build
metadata reports whether OpenMP was compiled, but it does not capture runtime
thread-environment variables because environment collection could expose
sensitive data. Record a reviewed thread count as a method parameter when it
matters.

## Complete example

Run:

```console
python examples/verification/reproducible_artifact.py
```

The example executes a seeded cosine-mode diffusion study on three grids,
records analytical errors and conservation residuals, fingerprints the finest
field, writes a stable manifest, reads it back, and exits nonzero if the scoped
checks fail. Re-running against an existing destination requires `--overwrite`.
Use `--include-volatile` only when per-run timestamp/ID provenance is wanted.

## Publication checklist

The manifest is one component of a reproducible release. Also preserve:

1. the exact manifest and any large field files named by their content digest;
2. source revision and dependency lock/container recipe;
3. governing equations, sign conventions, units, domain, and boundary/initial
   conditions;
4. parameter sources and calibration/validation datasets with licenses;
5. convergence and balance acceptance criteria chosen before analysis;
6. scripts that regenerate every table and figure from archived inputs;
7. runtime parallelism or accelerator settings when they affect results; and
8. a durable repository identifier, access conditions, and reuse license.

Never put credentials, access tokens, patient identifiers, private dataset
locations, usernames, home directories, or raw environment dumps in a
manifest. Content hashes are identifiers, not anonymization: a digest of a
small or guessable sensitive value may still disclose it by enumeration.
