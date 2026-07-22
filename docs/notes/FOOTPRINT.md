# BioTransport repository footprint

This document is a navigable map of the source tree and its scientific
responsibilities. Generated build products, caches, plots, and local virtual
environments are intentionally omitted.

BioTransport is alpha research and teaching software. This map describes where
capabilities live; it is not evidence that every class is equally verified or
that any biological model is clinically validated.

## Language and precision policy

Performance-critical canonical and specialized solver kernels are C++17 and
normally use `double`. NumPy-facing scientific fields therefore ordinarily use
`float64`. `GrayScottSolver` is a documented legacy exception: its
dimensionless periodic pattern fields use `float32`. Invariant and deterministic
execution tests exist, but there is no float-versus-double validation or timing
study supporting a broader precision/performance claim.

Python contains bindings and scientific workflow code. Unit conversion,
parameter provenance, sensitivity/uncertainty orchestration, reproducibility
manifests, plotting, and configuration are intentionally Python. Some older
numerical helpers also remain implemented in Python and do not automatically
inherit evidence from similarly named native solvers.

## Top-level layout

| Path | Responsibility |
|---|---|
| `CMakeLists.txt` | Top-level native build and feature options. |
| `cpp/` | Public C++ headers, implementations, tests, and engineering benchmarks. |
| `python/biotransport/` | Python package, type marker, workflow modules, and binding loader. |
| `python/bindings/` | Modular pybind11 registrations for the compiled `_core` extension. |
| `python/tests/` | Python API, workflow, numerical, and contract tests. |
| `examples/` | Runnable demonstrations and scoped verification workflows. |
| `docs/notes/` | Detailed scientific contracts, limitations, and workflow guides. |
| `docs/sphinx/` | Built user/API documentation. |
| `results/` | Default example output location; generated output is not source evidence. |
| `pyproject.toml` and `setup.py` | Python packaging and CMake-backed extension build. |
| `run_examples.py` | Headless example runner; example success is not blanket solver validation. |

## C++ library

### Core model and geometry

Public headers live under `cpp/include/biotransport/`; implementations live
under `cpp/src/`.

| Header area | Main responsibility |
|---|---|
| `core/problems/transport_problem.hpp` | Canonical scalar equation, coefficient fields, reactions, initial data, and boundaries. |
| `solvers/transport_solver.hpp` | Canonical conservative 1D/2D explicit solve, stability policy, exact final time, and diagnostics. |
| `core/mesh/structured_mesh.hpp` | Uniform Cartesian 1D/2D node mesh. |
| `core/mesh/structured_mesh_3d.hpp` | Uniform Cartesian 3D mesh used by specialized solvers. |
| `core/mesh/cylindrical_mesh.hpp` | Uniform radial/axisymmetric/full cylindrical coordinate operators. |
| `core/mesh/nonuniform_mesh_1d.hpp` | Validated strictly increasing fitted 1D nodes and node-centred control volumes. |
| `solvers/nonuniform_diffusion_1d.hpp` | Conservative nonuniform 1D diffusion with harmonic face coefficients and a local Forward-Euler limit. |
| `core/boundary.hpp` | Shared scalar boundary identifiers and data structures. Individual solver contracts still define whether a value is a derivative or physical flux. |
| `core/numerics/` | Stability helpers, time kernels, sparse/tridiagonal algebra, and iterative primitives. |
| `core/balance.hpp` | Unit-aware amount/energy/volume ledger arithmetic and transfer reconciliation. It does not couple solvers. |
| `core/build_info.hpp` | Native compiler/feature metadata exposed to reproducibility manifests. |
| `core/analytical.hpp` and `core/dimensionless.hpp` | Closed-form utilities and named dimensionless numbers. Applicability remains caller-owned. |

The nonuniform geometry is a deliberately narrow first slice. There is no
general nonuniform 2D/3D mesh, unstructured connectivity, moving mesh, finite
element framework, or adaptive mesh refinement.

### Native solver families

Solver headers under `cpp/include/biotransport/solvers/` and physics headers
under `cpp/include/biotransport/physics/` include:

- canonical conservative scalar transport and its compatibility facade;
- legacy and 3D explicit diffusion;
- Crank--Nicolson, ADI, and sparse backward-Euler diffusion;
- fitted nonuniform 1D diffusion;
- generic and specialized reaction--diffusion, multispecies systems, and
  Gray--Scott patterns;
- legacy scalar advection--diffusion;
- Nernst--Planck and prescribed-potential multi-ion transport;
- Darcy, Stokes, and bounded Navier--Stokes flow;
- steady single- and multilayer membrane transport;
- generalized-Newtonian constitutive laws;
- tumor drug-delivery and bioheat/cryotherapy application models; and
- high-order finite-difference/time-integration kernels.

The authoritative native public-solver inventory is not this prose list. It is
`python/biotransport/contracts.py`, where each entry records equation, units,
unknown placement, supported dimensions/terms/boundaries, method, stability,
evidence, exclusions, and warnings. `python/tests/test_solver_contracts.py`
mechanically compares that registry with the compiled runtime and checks every
evidence path/selector.

Important distinctions include:

- Stokes stores velocity components and pressure **collocated at structured
  mesh nodes** and uses centred gradients with a SIMPLE-like correction. It is
  not a staggered MAC discretization and has no Rhie--Chow-style checkerboard
  stabilization.
- Navier--Stokes has its own staggered/projection contract; its evidence does
  not transfer to Stokes.
- Electrochemical Neumann data are prescribed outward total molar fluxes,
  while canonical scalar and multispecies Neumann data are outward-normal
  concentration derivatives.
- Darcy's coefficient is hydraulic mobility/conductivity with unit
  `m^2/(Pa*s)`, not intrinsic permeability `m^2`, and its pressure needs a
  gauge/Dirichlet constraint.
- Application models are reduced mechanistic models. Their outputs are not
  patient-specific predictions.

See `docs/notes/SOLVER_CONTRACTS.md` and
`docs/notes/MODEL_SCOPE_AND_REFERENCES.md` for the evidence and model boundaries.

### Tests and benchmarks

`cpp/tests/` contains always-on native test programs for scoped API, balance,
analytical/manufactured, conservation, convergence, and failure claims. A test
supports only the claim and configuration it actually evaluates.

`cpp/benchmarks/` contains three bounded timing programs for selected kernels.
Their shared runner records workload parameters, correctness
invariants/checksums, compiler/build/CPU/OpenMP metadata, repeated timings, and
JSON output. They remain engineering tools until reviewed results are archived,
paired serial/OpenMP baselines and CI regression thresholds are established,
and broader representative workloads are covered. No generic or “massive”
speedup is claimed.

## Python package

### Native binding surface

`python/bindings/biotransport_bindings_new.cpp` assembles registrations split
across mesh, diffusion, transport, sparse, fluid, high-order, balance,
nonuniform, metadata, utility, and I/O binding files. The compiled module is
loaded by `python/biotransport/_core/__init__.py`; `_core.pyi` supplies static
typing information. Public convenience re-exports live in
`python/biotransport/__init__.py`.

The discoverable `biotransport.diffusion`, `biotransport.electrochem`,
`biotransport.flow`, and `biotransport.applications` modules organize native
objects. They do not create alternative solver kernels.

### Scientific workflow modules

| Module | Responsibility | Explicit non-claim |
|---|---|---|
| `biotransport.units` | Immutable semantic quantities, explicit conversion, affine temperatures, distinct permeability meanings, and perfusion-basis checks. | Raw C++ values remain untyped; unit correctness is not parameter validity. |
| `biotransport.provenance` | Immutable parameter records/sets with source, context, validity, uncertainty, status, JSON, and fingerprints. | Structural completeness cannot judge source quality or confer biological validity. |
| `biotransport.analysis` | Parameter sweeps, central local sensitivity, seeded independent-marginal Latin hypercubes, uncertainty propagation, and standardized-regression screening. | No correlated distributions, calibration, causality, Sobol/Morris indices, or model discrepancy. |
| `biotransport.contracts` | Immutable machine-readable native solver/evidence registry with lookup/filter/JSON helpers. | Evidence levels are scoped numerical claims, not experimental validation. |
| `biotransport.reproducibility` | Canonical JSON, frozen configs, SHA-256 fingerprints, method/seed/build metadata, convergence/balance records, and atomic manifest I/O. | A manifest is not authentication, durable archival, FAIR compliance, or validation. |
| `biotransport.config` | Validated application configuration dataclasses and parameter-provenance attachment. | Current bundled values remain illustrative/unprovenanced. |

Native `BalanceLedger` objects are top-level Python bindings because the same
accounting implementation is shared with C++. Callers must still integrate
fields, sources, and boundary fluxes over compatible domains before entering
amounts. Full automatic solver-result ledger coupling is not implemented.

### Convenience and legacy numerical modules

Mesh/reshape helpers, initial-condition builders, field builders, plotting,
VTK export, result paths, and the canonical `run`/`solve` wrappers are Python
convenience layers.

`adaptive.py`, `time_integrators.py`, `high_order.py`, `pulsatile.py`,
`convergence.py`, and `newton_raphson.py` contain orchestration or numerical
logic beyond a pure re-export. Some call native kernels; some are independent
Python implementations. Their existence does not make the package universally
thin and does not let them borrow another solver's evidence.

## Examples

Examples are grouped as:

- `examples/basic/`: introductory scalar problems and explicit unit conversion;
- `examples/intermediate/`: specialized methods and model components;
- `examples/advanced/`: multispecies/application demonstrations; and
- `examples/verification/`: scoped checks and auditable workflows, including
  grid convergence, sensitivity/uncertainty screening, and a reproducible
  manifest.

Examples document APIs and assumptions. A plotting example is not validation.
Where a script states acceptance criteria, it should exit nonzero when they
fail, and a research artifact should archive the exact configuration and
result rather than citing an unrecorded interactive run.

## Documentation map

| Note | Subject |
|---|---|
| `SCIENCE_FIRST_ARCHITECTURE.md` | Layer responsibilities and verification/performance policy. |
| `SOLVER_CONTRACTS.md` | Native contract schema, registry queries, evidence levels, and catalog. |
| `UNITS.md` | Python unit semantics and conversion boundaries. |
| `PARAMETER_PROVENANCE.md` | Traceability records and honest status of bundled parameters. |
| `SENSITIVITY_AND_UNCERTAINTY.md` | Scope and reporting of screening/propagation workflows. |
| `BALANCE_ACCOUNTING.md` | Ledger signs, units, transfer reconciliation, and non-coupling boundary. |
| `REPRODUCIBILITY.md` | Canonical manifests, fingerprints, build metadata, and publication checklist. |
| `NONUNIFORM_GEOMETRY.md` | Fitted 1D finite-volume equation, flux, stability, diagnostics, and exclusions. |
| `MODEL_SCOPE_AND_REFERENCES.md` | Application-model equations, references, and biological limits. |
| `PARAMETERS.md` | Parameter names, units, and current configuration conventions. |
| `GAP_ANALYSIS.md` | Closed/partial/open scientific-readiness matrix. |

## Build and packaging

- The top-level and `cpp/` CMake files build the native library, tests, and
  optional benchmarks.
- `python/bindings/CMakeLists.txt` builds the pybind11 `_core` extension.
- `pyproject.toml` declares Python dependencies and optional test/dev/docs
  environments.
- `setup.py` integrates the native CMake build with editable/wheel packaging.
- Sphinx sources live in `docs/sphinx/` and use autodoc, MyST, and the Furo
  theme.

Generated build directories and caches are disposable. Scientific outputs are
not: publication workflows should preserve frozen inputs, exact software/build
metadata, numerical evidence, selected field artifacts, and the scripts needed
to regenerate reported figures and tables.
