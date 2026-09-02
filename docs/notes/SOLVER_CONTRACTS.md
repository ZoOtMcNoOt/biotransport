# Native solver and Python numerical contracts

BioTransport exposes a machine-readable scientific contract for every public
solver entry point implemented by the compiled extension. It also exposes a
separate machine-readable contract for governed public Python numerical and
workflow modules. The authoritative registries are both in
`python/biotransport/contracts.py`; the separation prevents a Python reference
implementation from being mistaken for a native solver or inheriting native
performance/evidence.

The registry answers these questions before a simulation starts:

- What equation is represented?
- What are the unknowns, where are they stored, and what units cross the API?
- Which dimensions, terms, and boundary conditions are actually implemented?
- Which numerical method and stability policy are used?
- Which exact automated tests support a claim?
- Which physics and interpretations are explicitly excluded?

It does not turn a solver into a validated biological model. Parameters,
constitutive assumptions, calibration data, and the intended inference remain
part of the user's scientific model.

## Python use

The registry is available without importing it into the top-level namespace:

```python
from biotransport.contracts import (
    filter_contracts,
    get_contract,
    get_python_numerical_contract,
    list_contracts,
    list_native_solver_symbols,
    list_python_numerical_contracts,
    python_registry_as_dict,
    registry_as_dict,
)

contract = get_contract("NernstPlanckSolver")
print(contract.equation)
print(contract.stability_policy)
print(contract.evidence_level.value)

# IDs are stable machine-facing names; native class/function names also work.
assert get_contract("electrochem.nernst_planck") is contract

# Exact, case-insensitive vocabulary filters.
three_dimensional = filter_contracts(dimension="3D")
stronger_evidence = filter_contracts(minimum_evidence="analytical")

# JSON-ready snapshot for reports, manifests, or a UI.
payload = registry_as_dict()
symbols = list_native_solver_symbols()
all_contracts = list_contracts()

python_surface = get_python_numerical_contract("NewtonRaphsonSolver")
print(python_surface.backend.value)
print(python_surface.disposition)
python_payload = python_registry_as_dict()
python_contracts = list_python_numerical_contracts()
```

`SolverContract` and `EvidenceRecord` are frozen dataclasses. Their collection
fields are tuples, and `SOLVER_CONTRACTS` is read-only. `to_dict()` and
`registry_as_dict()` deliberately return detached JSON-serializable data.
`PythonNumericalContract` and `PYTHON_NUMERICAL_CONTRACTS` have the same
immutability guarantees, with `python_registry_as_dict()` as their detached
JSON snapshot.

## Contract schema

| Field | Meaning |
|---|---|
| `contract_id` | Stable registry key such as `diffusion.adi_2d`. |
| `native_symbols` | Exact compiled class or function names covered by this contract. |
| `equation` | Governing equation or reduced model actually advanced. |
| `unknowns` and `locations` | State variables and their grid/control-volume placement. |
| `input_units` and `output_units` | Quantity-to-unit pairs; `1` means dimensionless. |
| `supported_dimensions` | Exact spatial dimensionality or steady-dimensional scope. |
| `supported_terms` | Terms and coefficient forms that are implemented. |
| `supported_boundary_conditions` | Boundary semantics, including derivative-versus-flux distinctions. |
| `numerical_method` | Spatial discretization, time method, splitting, or steady solve. |
| `stability_policy` | What the implementation checks and what remains the caller's responsibility. |
| `convergence_policy` | The narrow order/accuracy claim, or an explicit statement that none is registered. |
| `evidence` | Scoped claims with an evidence level and exact `path::selector` references. |
| `exclusions` and `warnings` | Unsupported physics and interpretation hazards. |

Units are part of the model contract, not a runtime unit-conversion system.
Generic scalar solvers accept any internally consistent length, time, and field
units. Application and electrochemical models state SI units where their
equations and constants require them.

## Python numerical-surface schema

Python records use a different schema because not every surface advances a
physical PDE:

| Field | Meaning |
|---|---|
| `contract_id` | Stable key such as `python.reference.newton`. |
| `module` and `public_symbols` | Exact governed module and top-level public names owned by this record. |
| `category` | Adapter, reference solver, legacy integrator, or workflow role. |
| `backend` | `native-adapter`, `mixed-python-native`, `python-reference`, or `python-workflow`. |
| `mathematical_scope` and `numerical_method` | Equation/task and algorithm actually implemented. |
| `failure_policy` | Inputs and unsuccessful states that raise or return an explicit diagnostic. |
| `evidence` | Claim-specific records using the same evidence vocabulary below. |
| `disposition` | Explicit retain, native-port, or eventual-deprecation decision. |
| `exclusions` and `warnings` | Unsupported meanings and interpretation/performance hazards. |

The registry owns every public object defined or explicitly listed in
``__all__`` by its governed modules. Result types, configuration types, and
waveform protocol objects are included because they are part of those module
surfaces, even when they do not perform a numerical update themselves. Some
advanced result and protocol types are intentionally accessed through their
module namespace rather than re-exported from the package root.

## Evidence levels

Evidence is deliberately claim-specific. The level shown for a contract is the
strongest record attached to it; it must not be read as blanket evidence for
every configuration that the class can construct.

| Level | Registry meaning |
|---|---|
| `untested` | No automated test is cited. This level is available for honest gaps. |
| `api` | Export, construction, or interface behavior only; no numerical result is established. |
| `behavior` | A qualitative update or exact discrete-operation regression is exercised. |
| `invariant` | A balance, conservation law, equilibrium, positivity rule, or projection invariant is checked. |
| `analytical` | A scoped result is compared with an independent analytical/manufactured value or closed-form balance. |
| `convergence` | An observed refinement/order claim is measured for the stated case. |

These labels are repository vocabulary. ASME V&V 20 describes an approach for
quantifying simulation accuracy by comparing a specified solution variable and
experimental data at a specified validation point while considering
uncertainties in both. This registry does not perform that assessment and makes
no claim of ASME compliance. See the
[official ASME V&V 20 scope](https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer).

Likewise, numerical verification of an equation is distinct from biological,
experimental, clinical, or device validation.

## Current native catalog

The catalog is one contract per native entry point so evidence from a 2D method
cannot silently be attributed to a 3D variant, and tests of a generic reaction
callback cannot be presented as direct evidence for an untested specialization.

| Contract ID | Native entry point | Strongest scoped evidence | Important boundary of the claim |
|---|---|---|---|
| `transport.canonical_explicit` | `solve_transport` | convergence | Smooth diffusion is second order in space, the heterogeneous mixed-boundary upwind case is first order in space, and the composed reaction case is first order in time; claims remain case-specific. |
| `transport.legacy_explicit_fd` | `ExplicitFD` | analytical | Compatibility facade; arbitrary reactions lack a complete automatic stability certificate. |
| `diffusion.forward_euler_1d_2d` | `DiffusionSolver` | behavior | Qualitative diffusion behavior only; no measured order is registered. |
| `diffusion.forward_euler_3d` | `DiffusionSolver3D` | analytical | Linear steady state and conservation, not a 3D convergence-order study. |
| `diffusion.nonuniform_forward_euler_1d` | `NonuniformDiffusion1D` | convergence | Second order for one smooth stretched-mesh case; no claim for nonsmooth meshes, material interfaces, or temporal order. |
| `diffusion.crank_nicolson` | `CrankNicolsonDiffusion` | convergence | Second order for a smooth 1D case; A-stability does not ensure positivity or L-stability. |
| `diffusion.adi_2d` | `ADIDiffusion2D` | convergence | Second order for a smooth, time-independent-boundary case. |
| `diffusion.adi_3d` | `ADIDiffusion3D` | analytical | Conservation/linear equilibrium only; no 3D order measurement. |
| `diffusion.backward_euler_2d` | `ImplicitDiffusion2D` | analytical | Discrete equilibrium and balance evidence; Eigen backend required. |
| `diffusion.backward_euler_3d` | `ImplicitDiffusion3D` | analytical | Linear Neumann equilibrium only; Eigen backend required. |
| `transport.legacy_advection_diffusion` | `AdvectionDiffusionSolver` | behavior | Legacy pointwise transport; QUICK is unsupported pending a genuine verified stencil. |
| `reaction.generic_explicit` | `ReactionDiffusionSolver` | convergence | First order for two specific uniform callbacks; arbitrary callbacks have no a priori certificate, but non-finite or negative concentration candidates fail transactionally. |
| `reaction.linear_imex_1d_2d` | `LinearReactionDiffusionSolver` | convergence | First-order IMEX method; explicit diffusion limit remains. |
| `reaction.linear_imex_3d` | `LinearReactionDiffusionSolver3D` | behavior | Exact repeated discrete update only; no 3D continuum-order claim. |
| `reaction.logistic_specialized` | `LogisticReactionDiffusionSolver` | behavior | Unsafe negative candidates fail without mutation; no measured order is registered. |
| `reaction.michaelis_menten_specialized` | `MichaelisMentenReactionDiffusionSolver` | behavior | Parameters, singular denominators, and unsafe candidates are checked; no measured order is registered. |
| `reaction.masked_michaelis_menten` | `MaskedMichaelisMentenReactionDiffusionSolver` | behavior | Unsafe unpinned updates fail atomically; pinned nodes are not a resolved vessel model. |
| `reaction.constant_source_specialized` | `ConstantSourceReactionDiffusionSolver` | behavior | A sink cannot silently create negative concentration; no measured order is registered. |
| `reaction.multispecies` | `MultiSpeciesSolver` | analytical | Diffusion/eigenmode and kinetic invariants; arbitrary reactions can impose a smaller step. |
| `reaction.gray_scott` | `GrayScottSolver` | invariant | Nondimensional periodic cell-centred model; no pattern-order or biological-mechanism claim. |
| `flow.darcy` | `DarcyFlowSolver` | convergence | Linear pressure/velocity and layered face flux are analytical; the registered discontinuous-interface pressure sequence is approximately first order. `kappa` is hydraulic mobility `K/mu`, and a pressure gauge is required. |
| `flow.stokes` | `StokesSolver` | analytical | Sealed uniform-force hydrostatics and plane Poiseuille are checked directly; the generic method is collocated 2D, not staggered MAC. |
| `flow.navier_stokes` | `NavierStokesSolver` | convergence | A smooth wall-compatible manufactured velocity/force case is approximately second order in space over three MAC grids with time error suppressed by `dt=O(h^3)`; no turbulence closure or blanket open-boundary claim. |
| `membrane.single_layer` | `MembraneDiffusion1DSolver` | analytical | Steady ideal-dilute resistance; no transient storage or solvent drag. |
| `membrane.multilayer` | `MultiLayerMembraneSolver` | analytical | Series resistance and interface activity; no interfacial kinetics. |
| `applications.tumor_drug_delivery` | `TumorDrugDeliverySolver` | invariant | Reduced balance model, not Starling filtration, systemic PK, or a clinical predictor. |
| `applications.bioheat_cryotherapy` | `BioheatCryotherapySolver` | convergence | A linear heat eigenmode measures second-order space with `dt=O(h^3)` and first-order Forward-Euler time against a semi-discrete reference; the full phase/Arrhenius model is not thereby validated. |
| `electrochem.nernst_planck` | `NernstPlanckSolver` | convergence | The zero-potential diffusion limit is second order in space for one Neumann eigenmode with time error suppressed by `dt=O(h^3)`; this remains prescribed-potential ideal-dilute transport, not Poisson--Nernst--Planck. |
| `electrochem.multi_ion` | `MultiIonSolver` | invariant | Ions share a potential but advance independently; electroneutral coupling is not implemented. |

For exact equations, units, test selectors, and every warning, inspect the
corresponding `SolverContract` rather than scraping this summary table.

## Current Python numerical catalog

| Contract ID | Governed module | Backend | Disposition boundary |
|---|---|---|---|
| `python.canonical.adapters` | `biotransport.run` | native adapter | Retain canonical `solve`/`run`; replace segmented checkpoints with native saved-time support when available. |
| `python.legacy.adaptive_diffusion` | `biotransport.adaptive` | mixed Python/native | Legacy 1D Dirichlet diffusion only; port the controller or deprecate after a native adaptive API. |
| `python.legacy.time_integrators` | `biotransport.time_integrators` | mixed Python/native | `integrate` requires `method`; `method="euler"` is native. Heun/RK4 diffusion and generic stages remain legacy/reference paths. |
| `python.native_backed.high_order` | `biotransport.high_order` | mixed Python/native | Retain compiled stencil, diffusion, and RK-stage orchestration; generic RHS callbacks still cross into Python and remain a reference surface. |
| `python.reference.newton` | `biotransport.newton_raphson` | Python reference | Retain with explicit backend; iteration exhaustion returns `converged=False` and must be checked. |
| `python.reference.pulsatile_diffusion` | `biotransport.pulsatile` | Python reference | Warning-emitting compatibility path pending native time-dependent boundaries, then deprecate. |
| `python.workflow.convergence` | `biotransport.convergence` | Python workflow | Retain as verification orchestration; it advances no PDE and rejects indeterminate evidence. |
| `python.workflow.analysis` | `biotransport.analysis` | Python workflow | Retain as scientific orchestration; callback assumptions and missing global/calibration methods remain explicit. |

Field/initial-condition builders, mesh reshaping, VTK, visualization, units,
provenance, and reproducibility are validated convenience, I/O, or record
layers rather than physical solvers. They are not silently inserted into the
numerical registry to inflate coverage.

## Mechanical integrity gates

`python/tests/test_solver_contracts.py` enforces the registry as an audit
surface rather than a best-effort document:

1. It discovers public compiled classes with `solve`, `simulate`, or `run`, plus
   the native `solve_transport` function.
2. It excludes only `SparseMatrix.solve`, which is a linear-algebra primitive;
   physical sparse diffusion is represented by `ImplicitDiffusion2D/3D`.
3. Runtime and registered symbol sets must match exactly. A newly exported
   solver fails until it has a contract, and a removed solver leaves a failing
   stale contract.
4. Every `path::selector` evidence reference must point to a current `.cpp` or
   `.py` test and the selector must still occur in that file.
5. Contract IDs and native symbols must have one owner, the records must be
   immutable and JSON serializable, and the module must parse as Python 3.9.

`python/tests/test_python_numerical_contracts.py` separately:

1. discovers every top-level public object defined by each governed Python
   module and requires exact registry ownership;
2. rejects duplicate contract IDs, modules, and public symbols;
3. resolves every evidence path and selector;
4. checks immutable/JSON-ready schema and ID/symbol lookup; and
5. prevents legacy Python solver dispositions from masquerading as native
   performance paths.

## Adding or strengthening a contract

When adding a native solver:

1. Add a distinct contract before exporting the class or function.
2. State the complete equation, unknown placement, units, implemented terms,
   boundary semantics, method, and stability behavior.
3. Cite only tests that directly exercise that implementation and claim.
4. Use `api` when only construction/export is tested. A test of a related class
   is not evidence for the new implementation.
5. Raise an evidence level only when a new test establishes that narrower
   claim. Do not infer convergence from stability, conservation from visual
   smoothness, or biological validity from numerical agreement.
6. Record missing physics in `exclusions` and interpretation hazards in
   `warnings`.
7. Run the registry tests, static typing, and formatting gates.

For a public Python numerical module, add or update its
`PythonNumericalContract` in the same change. State its actual backend,
mathematical task, failure behavior, and retain/port/deprecate disposition.
New top-level symbols defined by a governed module intentionally fail the exact
ownership test until assigned. If a new module is numerical, add it to the
governed registry rather than leaving it outside the audit surface.

The most useful registry is an honest one: visible gaps are work queues, while
inflated evidence labels hide the work that science still requires.
