# BioTransport scientific readiness and gap analysis

This document tracks scoped evidence, not the mere presence of classes. A
solver is not research-ready because it compiles, produces a smooth plot, or
matches a formula implemented by the same code path. Likewise, workflow
infrastructure such as a units object, provenance record, balance ledger, or
JSON manifest does not validate a biological model.

BioTransport is alpha software. It is not a clinical device and is not
validated or authorized for patient-specific decisions.

## Status and evidence vocabulary

The status column below describes repository work:

| Status | Meaning |
|---|---|
| **Closed (scoped)** | The stated, deliberately bounded capability and its mechanical tests have landed. It is not a claim that adjacent physics or every scientific use is complete. |
| **Partial** | A useful implementation exists, but important integrations, evidence, or supported cases remain open. |
| **Open** | The acceptance evidence described here has not landed. |

Numerical evidence is described separately:

| Evidence label | Meaning |
|---|---|
| **Verified canonical** | The governing equation and signs are explicit, with independent analytical/manufactured cases, balances, scoped convergence evidence, and invalid/stability behavior for the canonical configuration under discussion. |
| **Separately benchmarked** | A specialized surface has direct evidence for a stated subset of geometry, coefficients, boundaries, and parameter ranges. Evidence does not transfer outside that subset. |
| **Experimental/API only** | An implementation or export exists, but its output still needs independent problem-specific verification before scientific use. |
| **Not modeled** | The library rejects or does not expose the requested physics. |

Biological validation is separate from numerical verification. It requires
traceable parameters, calibration data independent from validation data,
sensitivity/uncertainty analysis, and comparison with measurements relevant to
the intended use.

## Current numerical scope

### Canonical scalar path

The `Problem` / `solve` path advances one scalar field on uniform Cartesian 1D
and 2D meshes:

```text
dc/dt = div(D grad(c)) - div(v c) + R(c, x, y, t)
```

Its always-on evidence covers conservative finite-volume diffusion with
harmonic face coefficients, conservative first-order upwind advection,
Forward Euler stability enforcement, exact final-time landing, composable
reactions, boundary signs/corners, closed-domain conservation, manufactured
steady cases, and first-order reaction-time convergence.

`examples/verification/grid_convergence.py` additionally performs a scoped
spatial and temporal refinement study for one diffusion case. That example is
useful evidence when run and archived, but it is not an always-on CTest/pytest
order certificate for every canonical configuration. The canonical method does
not claim high-order advection, arbitrary-reaction monotonicity, or biological
validity.

### Specialized native surfaces

Specialized APIs cover 3D, Crank--Nicolson and ADI diffusion, sparse implicit
diffusion, membrane and electrochemical transport, multispecies systems,
Darcy/Stokes/Navier--Stokes flow, generalized-Newtonian constitutive laws,
bioheat cryotherapy, tumor drug delivery, and a fitted nonuniform 1D diffusion
slice.

`python/biotransport/contracts.py` is the machine-readable inventory for native
solver entry points. Each contract states its equation, unknown placement,
units, supported dimensions/terms/boundaries, numerical policy, exact evidence
references, exclusions, and warnings. Runtime tests fail when a native solver
is missing or a registered evidence path becomes stale. Evidence remains
claim-specific: an analytical test of one solver or configuration never
promotes a superficially similar surface.

The nonuniform slice is real node-centred finite volume on a fixed fitted 1D
mesh, with harmonic face diffusivity, a shared conservative flux, a local
Forward-Euler monotonicity bound, and automated smooth stretched-mesh spatial
convergence. It does not provide nonuniform Cartesian 2D/3D, cylindrical metric
terms on arbitrary coordinates, unstructured connectivity, moving meshes, or
adaptive refinement.

### Scientific workflow infrastructure

- `biotransport.units` provides immutable, dimension-checked conversion for a
  bounded set of quantities. Raw C++ solver APIs still accept plain doubles and
  require one caller-consistent unit system.
- `biotransport.provenance` provides immutable parameter/source/context/
  uncertainty records and fingerprints. Existing bundled application values
  are labeled `illustrative` and `unprovenanced`, not recommended priors.
- `biotransport.analysis` provides deterministic sweeps, central local
  sensitivities, seeded independent-marginal Latin hypercubes, uncertainty
  propagation, and standardized-regression screening for scalar quantities of
  interest. It does not perform calibration, infer distributions, or establish
  causality.
- `BalanceLedger` and `reconcile_balances` provide unit-aware arithmetic for
  amount, energy, and volume inventories and paired transfers. They are generic
  accounting objects; solvers do not yet automatically populate a unified
  cross-model result ledger or couple PDEs through it.
- `biotransport.reproducibility` writes canonical, fingerprinted manifests with
  frozen configuration, software/build metadata, declared method/seed,
  convergence tables, balances, and result summaries. A manifest is not a
  durable publication archive, a digital signature, or validation evidence by
  itself.

## Capability status matrix

| Priority | Capability | Status | What landed | What remains open |
|---|---|---|---|---|
| P0 | Per-native-solver scientific contracts | **Closed (scoped)** | Immutable registry, exact test references, runtime coverage/stale-reference gates, exclusions and warnings. | Keep contracts synchronized and strengthen low-evidence entries rather than inflating labels. Python-only numerical implementations need separate contracts or retirement. |
| P0 | Canonical conservative scalar contract | **Partial** | Strong equation/sign/boundary/stability/balance tests and scoped reaction-time convergence. | Move the spatial refinement acceptance sequence from an example into an always-on solver test and broaden coefficient/boundary combinations. |
| P0 | Independent specialized benchmarks | **Partial** | Focused analytical, manufactured, invariant, and some order tests exist for multiple specialized families. Darcy now has direct linear pressure/velocity, heterogeneous interface-flux, failure, and measured first-order discontinuous-interface evidence. | Broad spatial **and** temporal asymptotic evidence is absent for many dimensions/BCs; Navier--Stokes still lacks an independent velocity benchmark. |
| P0 | Fail-loud public numerical surfaces | **Partial** | Canonical transport plus audited legacy scalar/reaction/advection, specialized diffusion corners, sparse solves, Stokes, Darcy, and Python Newton paths reject their known unsupported, unstable, non-finite, singular, or non-converged cases. | Continue the same adversarial audit across remaining Python numerical helpers and add forced-failure evidence wherever a public convergence policy is not yet exercised. |
| P0 | Application parameter provenance | **Partial** | Typed immutable records, manifests, stale-value detection, and honest illustrative records for current defaults. | Supply defensible sourced records with material/population, method, temperature, applicability, and uncertainty before any value is called recommended. |
| P1 | Units at the API boundary | **Partial** | Python quantities distinguish temperature, pressure, concentration, diffusivity, permeability meanings, perfusion bases, and common dimensions. | Raw native C++ APIs remain untyped doubles; solver/config boundaries do not universally require `Quantity`; compound-specific mass-to-molar conversion remains caller-owned. |
| P1 | Sensitivity and uncertainty screening | **Closed (scoped)** | Seeded one-at-a-time/local/LHS/propagation/SRC workflow with failure and conditioning diagnostics. | Correlated inputs, Sobol/Morris methods, Bayesian calibration, surrogate validation, model discrepancy, and automatic numerical-uncertainty combination are not implemented. |
| P1 | Coupled balance accounting | **Partial** | Unit-aware amount/energy/volume ledgers and matched transfer reconciliation. | Full solver-result ledger integration is open: fields, sources, boundary fluxes, heat, and fluid volume are not automatically integrated into one coupled audit. The ledger does not advance or couple models. |
| P1 | Reproducible numerical artifacts | **Partial** | Deterministic canonical JSON, fingerprints, frozen configs, method/seed, build metadata, convergence/balance records, and atomic I/O. | Publication packaging still needs durable hosting/identifiers, archived field data, dependency/container capture, regenerated figures, and project-specific provenance/calibration/validation data. |
| P1 | Performance evidence | **Partial** | Three bounded native runners record workload parameters, correctness invariants/checksums, compiler/build/CPU/OpenMP metadata, repeated timings, and JSON output. A controlled same-binary one-versus-four-thread Release comparison records identical checksums and descriptive timings without a speedup threshold. | Archive reviewed baselines outside ignored build output, add carefully scoped CI/regression policy, and broaden representative workloads such as Gray--Scott. No generic speedup claim is supported. |
| P2 | Nonuniform geometry | **Partial** | Conservative fitted nonuniform 1D diffusion with geometry validation, interface conservation, diagnostics, and convergence evidence. | Nonuniform 2D/3D, general metric geometry, contact resistance, advection/reaction, AMR, moving meshes, unstructured finite volume, and finite elements remain open. |

## Persistent limitations

- Alpha/not clinical: no blanket research, regulatory, or patient-specific
  readiness claim is made.
- Raw native solver values have no runtime unit type. Use SI at solver
  boundaries where possible and record conversions explicitly.
- A stable implicit method can still be inaccurate. “Unconditionally stable”
  is never permission to skip a temporal study.
- General Python callbacks cross the Python/C++ boundary and can be slower than
  fully native kernels. No uncontrolled speedup claim should be inferred.
- The canonical scalar method is first order in time and first order for
  advection. Higher-order or specialized methods carry separate contracts.
- Stokes is a **collocated** centred-difference/SIMPLE-like implementation, not
  a staggered MAC method. Its pressure field lacks Rhie--Chow-style checkerboard
  stabilization.
- Fluid outflow/traction semantics are solver-specific. A zero-gradient
  outflow is not automatically a traction or pressure boundary.
- Cylindrical and 3D operators require dedicated axis, metric, balance, and
  convergence evidence for the intended application.
- Existing application defaults are demonstrations until sourced provenance
  and a calibration/applicability domain are supplied.
- Numerical verification, a closed balance, parameter traceability, and a
  reproducible file are necessary evidence components, not biological or
  clinical validation.

## Readiness by use case

| Use | Current position |
|---|---|
| Learning transport balances | Appropriate when the documented equation and assumptions match the exercise and the user checks grid/time sensitivity. |
| Developing numerical methods | Useful as an alpha C++ test bed; new methods need independent benchmarks and explicit scope. |
| Exploratory research modeling | Possible for a verified subset with project-specific parameter provenance, convergence, balances, sensitivity/uncertainty, and review. |
| Thesis/dissertation evidence | No blanket readiness claim. Suitability must be established per equation, solver, parameterization, artifact, and intended inference. |
| Clinical or patient-specific decisions | Not validated or authorized. |

## Definition of done for a new solver

A solver may move from experimental/API-only to separately benchmarked only
when:

1. the conservation law, fluxes, source signs, unknown locations, and units are
   written down;
2. every boundary value has unambiguous mathematical meaning;
3. parameter and field domains are checked before integration;
4. stability restrictions or nonlinear/linear convergence policies are
   enforced and exposed;
5. final-time and saved-state semantics are exact and tested;
6. at least one independent analytical or manufactured solution is automated;
7. a conservation or energy budget is automated where the PDE has one;
8. spatial and temporal convergence are measured over genuine asymptotic
   sequences when those claims are made;
9. invalid, unstable, singular, and non-converged cases are tests, not comments;
10. C++ and Python APIs describe the same equation and return the same data; and
11. its machine-readable contract names the evidence and explicit exclusions.

Feature count and test count are deliberately absent from readiness claims.
Evidence is specific to the tested scientific contract.
