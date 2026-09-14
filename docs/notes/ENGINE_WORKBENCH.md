# BioTransport engine implementation and verification record

This is the implementation and verification record for the shared experiment
API, local Studio, and steady solver improvements. The usage and extension guide
is [Build once, run from Python or Studio](../workbench.md).

## Revision and preserved work

The checkout is `ZoOtMcNoOt/biotransport`, branch `master`, based on
`12fcfb2c6f3e15ba6d7fff879998e8cc6683e035`. The local and remote commit agreed at
inspection. There was substantial uncommitted scientific API, geometry,
documentation and example work already present. Its 66 changed files were
snapshotted, with hashes and a patch, under
`build/workbench-validation/baseline/` before this increment. That work was
extended in place. The completed increment was committed and pushed to `master`
as `85ef2a556a7af0a846808cb78e47cfe5c4d93173` on 12 September 2026.

The follow-on extension began from a second snapshot of 90 changed files in
`build/extension-validation/baseline/`. Both snapshots and the first installed
wheel are retained.

## Current priority

The user has deferred further UI work. The current increment focuses on a simple
Python API for multiple physical domains, named species, conservative membrane
coupling and stoichiometric reactions, with inspectable sparse operators and
independent numerical verification.

The first pushed revision passed all native, Python and packaging CI jobs.
CI exposed typing checks masked by a local installed package and a Sphinx source
path that shadowed the installed wheel. The fixes require source-based typing
and real native imports for documentation. Locally, 416 affected Python tests,
Ruff, source-based mypy and a strict Sphinx build importing the installed wheel
pass. Evidence is in `build/extension-validation/ci-fix-pytest.log` and
`ci-docs-fix.log`. The repair was pushed as
`6cadff2d2a636b38ac7814e0648a480fa8f42ca7`; all CI jobs passed in
[run 34685185488](https://github.com/ZoOtMcNoOt/biotransport/actions/runs/34685185488).

## Coupled engine increment

- `CoupledModel` composes named species across well-mixed compartments and 1D
  Cartesian, cylindrical or spherical domains. Physical SI volumes and face
  areas determine concentration changes and amount accounting.
- A membrane contributes equal and opposite amount transfer to its endpoints,
  with explicit target/source partition convention. Domain boundaries retain
  separate concentration states; interface area cannot exceed a spatial face.
- Stoichiometric mass-action reactions support production, loss, conversion,
  binding and reversible chemistry. Custom local rates supply analytic partial
  derivatives. Declared chemical invariants reject incompatible reactions.
- `compile()` produces an owned model with named state slices, coordinates,
  physical volumes, a sparse transport matrix, RHS and analytic Jacobian.
  Linear reactions are assembled once; BDF and Radau integrate all domains
  together without restarting for saved frames. No new dependencies were added.
- `CoupledSolution` exposes named fields, histories, moles, signed membrane rates
  and conservation reports. Diagnostics report actual evaluations and sparse
  factorizations. Concentrations are never silently clipped.
- The [coupled engine guide](../coupled_transport.md) and README examples execute
  in the test suite. Sphinx documents the public API and resolves SciPy types.
  CI checks both source-only typing and NumPy-aware coupled-engine typing.

### Coupled engine verification

The coupled engine was pushed as `1b5388de1a014cfdef294ce9e79d90254c8b71ac`.
All 12 CI jobs passed in
[run 34686269172](https://github.com/ZoOtMcNoOt/biotransport/actions/runs/34686269172).
All **1,958 Python tests and 15 subtests** pass against a separately installed
wheel, with the six existing pulsatile-reference warnings, in 93.44 seconds.
This includes 77 coupled-engine checks and executable README, tutorial and
coupled-guide snippets. Tests independently verify exchange equilibria,
first- and second-order reaction laws, reversible binding, second-order slab
and spherical convergence, native radial agreement, dense matrix-exponential
references, sparse Jacobian directional derivatives and conservation.

Ruff, the source-only typing contract, the NumPy-aware coupled module check and
a fresh strict Sphinx build against the installed wheel pass. All 47 packaged
Python, typing and frontend source files match the wheel byte for byte.
Native sources are unchanged from the preceding green cross-platform CI run;
the repository CI also runs those gates for each push.

The final wheel is
`build/coupled-validation/final-dist/biotransport-0.1.0-cp314-cp314-win_amd64.whl`,
SHA-256 `863a584acdb49fef670b2a7d1064195e476209f12eb9e64aa0129b370756b8df`.
Logs are `build/coupled-validation/final-pytest.log`, `final-docs.log`,
`final-wheel.log` and `final-benchmark.log`.

A closed reservoir/spherical-tissue model with drug, binding sites and bound
drug was integrated for 300 seconds, saving 41 states. Three warm repetitions
against the installed wheel on Windows / Python 3.14.3 / SciPy 1.18.0, with one
BLAS thread requested, produced:

| Spatial cells | Concentration states | Median assembly | Median solve |
| ---: | ---: | ---: | ---: |
| 100 | 306 | 1.37 ms | 92.75 ms |
| 1,000 | 3,006 | 6.30 ms | 139.80 ms |
| 10,000 | 30,006 | 81.30 ms | 2,139.02 ms |

At 30,006 states, the maximum final concentration difference against a tighter
run was `6.55e-8 mol/m^3`. Maximum relative drug-amount drift was `8.17e-12`;
site-amount drift was `1.58e-15`. The initial sparse Jacobian had 80,009 nonzeros
and 1,080,136 bytes of array storage, excluding factorization and process memory.
These are workload measurements under shared machine load, not a universal
speedup or biological validation claim. Timing varied across local runs; no
best-run selection was used. The final raw samples, settings and source hashes
are in [coupled-transport-20260912.json](../benchmarks/coupled-transport-20260912.json).

## Open systems and dosing protocols — 13 September 2026

This engine increment follows `1b5388d`; its implementation is the commit
containing this section. UI work remains deferred.

- `bath` defines maintained external concentrations. `ConcentrationSchedule`
  supplies immutable step or linear protocols with dimension-checked values
  and explicit left/right limits. Names are shared with domain endpoints;
  existing membrane laws and spatial face-area constraints still apply.
- Integration is split at every connected, permeable schedule knot, even when
  it falls between requested output frames. The finishing interval sees the
  pre-jump bath and the next sees the new bath. State and cumulative transfers
  remain continuous; saved instantaneous bath values use the right limit.
- Each external membrane/species pair has a signed integrated amount ledger.
  Ledger states are divided by their connected physical volume so concentration
  tolerances remain meaningful for tiny modeled volumes. Sparse augmented
  Jacobians carry the same loss coefficients as the physical state equations.
- `external_amount`, `bath_history`, and expanded conservation reports separate
  stored moles from externally supplied or removed moles. `external_rates`,
  `breakpoints`, and the RHS's `bath_side` control support external integrators.
  Public state vectors contain concentrations; built-in solving manages the
  internal ledger automatically.
- The integration implementation is isolated in `_coupled_integration.py` and
  schedules in `protocols.py`. Closed networks call their existing local RHS
  directly. Interface-rate queries now read endpoint concentrations directly,
  avoiding a copy of a whole field history just to read one boundary node.

### Protocol verification

All **2,004 Python tests and 15 subtests** pass against the separately installed
wheel in 73.93 seconds, with six existing pulsatile-reference warnings. The 45
new protocol checks cover both membrane orientations, partition equilibrium,
one-microsecond pulses, linear ramps, interleaved baths, exact open reaction
kinetics, independent quadrature and matrix-exponential references, sparse
Jacobian derivatives, endpoint limits, state ownership and volumes down to
`1e-18 m^3`. The 77 earlier coupled checks and executable documentation also pass.

Ruff, source-only API typing, NumPy-aware typing for all three engine modules,
and strict Sphinx against the installed wheel pass. All 49 packaged source
files match the wheel byte for byte. Native sources and frontend assets are
unchanged; the push's CI performs the cross-platform release checks.

The retained wheel is
`build/protocol-validation/dist/biotransport-0.1.0-cp314-cp314-win_amd64.whl`,
SHA-256 `78194497018496f953253e650a1c90de9f87e107f246dc05427c66c502484d20`.
Evidence: `build/protocol-validation/installed-pytest.log`, `installed-docs.log`,
`wheel.log` and `installed-benchmark.log`.

The closed-network benchmark ran serially against the retained `1b5388d` wheel
and the new implementation, with identical equations, output frames and
tolerances. All integrator diagnostic values and tighter-reference differences
match at every size. At 30,006 states the three-run median was 1.6647 seconds
before and 1.6623 seconds after; these observations do not establish a speedup.
Raw samples and environment metadata are in
[protocol-closed-regression-20260913.json](../benchmarks/protocol-closed-regression-20260913.json).

The new installed-wheel benchmark applies a 20-second bath pulse to a sphere
with reversible binding and follows washout through 120 seconds. It requests
only a final output frame; the solver also saves both protocol changes.

| Spatial cells | Concentration states | External ledger states | Median solve |
| ---: | ---: | ---: | ---: |
| 100 | 303 | 1 | 70.49 ms |
| 1,000 | 3,003 | 1 | 268.83 ms |
| 10,000 | 30,003 | 1 | 3,005.84 ms |

At 10,000 cells, maximum final concentration difference against a tighter run
was `2.52e-9 mol/m^3`; the maximum external-accounted drug balance residual was
`2.00e-21 mol`, or `4.32e-12` relative. These are three warm repetitions on
Windows / Python 3.14.3 / SciPy 1.18.0 with one BLAS thread requested, under
shared host load. Raw values, settings and module hashes are in
[bath-protocols-20260913.json](../benchmarks/bath-protocols-20260913.json).
Reproduce with `examples/verification/benchmark_protocols.py`.

## Follow-on extension: completed

- Conservative steady solving now covers 1D Cartesian, cylindrical and spherical
  meshes, including annuli, harmonic diffusivity, sources and outward-normal
  Neumann conditions. Boundary reactions use the same control volumes as the
  native transient engine. Independent exact profiles, second-order refinement,
  flux balance and unit-scaling tests cover the extension.
- Stable `analytical.steady_radial_first_order` profiles support teaching and
  verification. Studio uses radial references only when their assumptions match;
  nonlinear uptake never inherits a linear reference overlay.
- Saturable uptake is a built-in component. Nonnegative uptake models use
  backtracking to avoid a negative Newton trial entering flat clipped kinetics
  and making a closed model's Jacobian singular. A closed steady experiment
  rejects production at or above its finite uptake capacity. Negative-field
  reaction accounting now follows native clipping.
- Native `plan_transport` and `TransportPlan` share the solver's preparation and
  exact endpoint-counting rule. `Experiment.plan()` includes every saved interval,
  reaction accuracy guards, the step budget and saved-array bytes without
  evaluating reactions or advancing time. A steady plan does not predict Newton
  convergence. `stable_time_step()` is documented as a stability ceiling, which
  can exceed the actual selected step.
- Studio previews the schedule after edits, explains field and cost errors, and
  disables incompatible palette entries from registry metadata. Larger valid
  Python documents can be imported and reduced to the interactive limits. All
  boundary cards remain selectable when duplicate sides need repair.
- Invalid `Problem` edits now leave both native state and the retained recipe
  intact; retained arrays are copied. The transaction regressions reproduced
  29 failures before the fix and passed all 41 cases afterward.
- Native radial shell volumes use factored polynomial differences. Six
  high-precision regression cases exposed cancellation in thin annuli before the
  fix; the rebuilt extension passes them within eight binary64 epsilons.
- Package builds refresh their generated CMake cache. The previous local Release
  cache had empty optimization flags and kept assertions enabled. The new build
  restores `/O2 /Ob2 /DNDEBUG`; CI checks Release assertion metadata.

### Current evidence

The installed rebuilt wheel passed **1,877 Python tests and 15 subtests**, with
six existing pulsatile-reference warnings, in 63.98 seconds. The original
200-cell, Fourier-number-20 transient balance regression was then retained
alongside the new direct steady check and passed separately in 35.71 seconds.
Thus 1,878 current Python cases were checked; the suite runtime is not a
before/after performance benchmark. Logs:
`build/extension-validation/installed-pytest.log` and `restored-transient.log`.

All **36 native Debug CTest executables** passed in 135.91 seconds. Seven Node
editor tests, Ruff, the typed native API smoke check and a fresh strict Sphinx
build also passed. The wheel has optimized native build metadata, and all 45
Python, stub and frontend source files match its contents byte for byte:

`build/extension-validation/dist/biotransport-0.1.0-cp314-cp314-win_amd64.whl`

SHA-256: `6e40801786d4710ab51baece1d68c4d137b5793c6f6584f4186c0f6bdb185b49`.

Browser acceptance ran both tissue-sphere examples, checked the radial exact
overlay and nonlinear no-overlay behavior, rejected an oversized transient,
repaired duplicate boundaries, and imported a 2,000-cell/200-interval model
before reducing and running it. The tested DOM had no page overflow at 320,
390, 1,280 and 1,440 CSS pixels. Console errors were absent. Browser screenshots
and the larger import fixture are in `build/extension-validation/`; the tab
viewport override did not affect the older workbench tab, so responsive checks
used that tab's documented development emulation controls instead.

### Current performance measurements

Three warm runs on the same 200-cell sphere, integrated for 100 seconds with
30,001 native steps, measured **1.558928 s before / 0.220059 s after** the build
repair (7.08×). Maximum field difference was `1.11e-16`. This is a comparison of
these specific local binaries under shared machine load, not a universal engine
speedup. Build metadata, binary hashes, all timings and full final fields are in
[before](../benchmarks/native-transport-before-20260912.json) and
[after](../benchmarks/native-transport-after-20260912.json). Reproduce each
isolated process with `examples/verification/benchmark_native_transport.py`.

Direct steady equilibration of the 200-cell sphere took **4.235 ms**, maximum
absolute error `3.134e-7` and relative balance residual `1.205e-13`. At 10,000
cells it took 59.51 ms, with error `1.254e-10`. Separate 20-cell comparisons
checked both direct and transient paths on identical equations: their fields
agreed within `3.40e-10` for the cylinder and `2.15e-14` for the sphere. The
[radial benchmark](../benchmarks/radial-steady-20260912.json) retains all samples,
source hashes and environment details; reproduce with
`examples/verification/benchmark_radial_steady.py`. Three warm repetitions were
used with one BLAS/OpenMP thread requested on Windows, Python 3.14.3, NumPy 2.5.1
and SciPy 1.18.0. The original transient is still covered independently.

## Delivered behavior

- A versioned, validated `Experiment` document compiles into `Problem` and runs
  through the public engine. Python and the browser share the same component
  definitions, validation, and numerical implementation. Documents and registry
  definitions are snapshotted rather than retaining mutable caller state.
- `ComponentRegistry`, `ComponentDefinition` and `Parameter` let trusted Python
  extensions describe their controls and supported geometry/solver scope once.
  Custom roles work without depending on a component's naming convention.
  JSON can select registered components; it cannot import or execute Python.
- Studio includes actual pointer-based drag and drop, keyboard-accessible add
  buttons, editable domains and components, five runnable examples, undo,
  validated JSON import/export, and full-resolution CSV export. Results identify
  when the model has changed since the last solve.
- Result views include profiles, space–time maps, saved-frame playback, a data
  table, physical interpretation and solver diagnostics. Color and profile
  limits stay fixed across frames. Exact overlays require matching assumptions;
  inventory change is only called a conservation check for a known closed,
  diffusion-only model. Steady results do not imply a simulated duration.
- The local HTTP adapter ships in the wheel with no frontend build or extra
  server dependency. It binds to loopback, checks request origins, restricts
  resource use and serializes solves. It is intended for a local user.
  Early rejection uses a staged connection close so a continuing upload does
  not discard the error response on Windows. Input draining is limited to one
  second and 512 KB, following the connection-close approach in
  [RFC 9112 section 9.6](https://www.rfc-editor.org/rfc/rfc9112.html#section-9.6).

## First increment accuracy and performance (historical)

The measurements below describe the first increment's operator and package,
before the conservative boundary extension and native build repair above.

The steady solver previously compared dimensional residuals directly, allowing
small diffusivities to appear converged before the physical solution was
reached. It now scales space, concentration and time internally and reports an
explicit dimensionless stopping residual. Physical fields and step norms retain
their original units. The default residual tolerance is `1e-10`; there is no
silent retry with a looser tolerance.

The 1D path now assembles a sparse Jacobian, reuses its factorization and checks
conditioning without a dense decomposition. Singular resonant modes still fail
explicitly; least-squares behavior must be requested. Tests cover tiny and large
coefficients, subnormal diffusivities, nonzero outward gradients, sources,
concentration scaling and fine grids.

Analytical helpers avoid hyperbolic/Bessel overflow and small-parameter
cancellation. Boundary evaluation accepts only machine-roundoff excursions from
native mesh coordinates. Independent high-precision references cover extreme
amplitudes and Thiele moduli.

Measured complete solves of `-c'' + c = 0`, with `c(0)=1` and `c'(1)=0`, used
identical scaled equations and stopping criteria for sparse and dense paths:

| Cells | Sparse median | Dense median | Dense / sparse |
| ---: | ---: | ---: | ---: |
| 100 | 10.87 ms | 8.07 ms | 0.74× |
| 300 | 8.64 ms | 62.32 ms | 7.21× |
| 600 | 14.65 ms | 649.61 ms | 44.35× |

At 600 cells, both paths had maximum absolute error `5.655609e-8`; Jacobian array
storage was 24,020 versus 2,889,608 bytes. This is array storage, not total process
or factorization memory. A 10,000-cell sparse solve took a median 78.64 ms with
maximum absolute error `2.055243e-10`.

These are three warm repetitions on Windows 11 / Python 3.14.3 / NumPy 2.5.1 /
SciPy 1.18.0, requesting one BLAS thread, on a machine under shared load. They are
not a universal speedup claim: sparse setup costs more on the smallest case.
The [raw measurements](../benchmarks/steady-engine-20260912.json) retain every
sample and environment metadata. Reproduce with
`examples/verification/benchmark_steady.py`.

## First increment verification (historical)

- The starting checkout passed 1,544 Python tests and 15 subtests.
- The final full run against the installed wheel passed **1,712 Python tests
  and 15 subtests**, with six warnings, in 348.73 seconds. The run requested one
  BLAS/OpenMP thread and used Matplotlib's headless backend. The log begins with
  the imported installation path and records the slowest cases:
  `build/workbench-validation/release-pytest.log`.
- Six Node tests cover custom component roles, replacement/additive semantics,
  missing-boundary handling and conservative labeling. HTTP tests exercise real
  requests and compare all bundled experiments against the public engine.
  Four new socket tests reproduced the Windows rejection bug before the fix;
  all 28 HTTP/workbench tests then passed. Independent review verified both
  the one-second abandoned-connection limit and the 524,288-byte drain limit.
- Ruff and the repository's typed API smoke check pass. A fresh strict Sphinx
  build, importing the actual native extension, completed with zero warnings.
  Public mesh/result types and native return signatures now resolve correctly.
- Browser acceptance exercised actual dragging, add buttons, parameter edits,
  rejected invalid input, undo, example selection, steady/transient results,
  frame navigation/playback and JSON import. Layouts at 320, 390 and 1,440 CSS
  pixels had no horizontal page overflow. Canvas/drop/footer text contrast was
  checked and improved. This is not a full accessibility certification.
- A downloaded JSON document matched its imported source, including a component
  named `domain`. All 4,961 downloaded CSV concentration values exactly matched
  the saved native-backed solution. Browser download-event notification timed
  out, so success was verified from the actual downloaded files.

Local raw logs and baseline files are in `build/workbench-validation/`. The
portable benchmark evidence is tracked under `docs/benchmarks/`.

The retained first-increment wheel is
`build/workbench-validation/release-dist/biotransport-0.1.0-cp314-cp314-win_amd64.whl`,
SHA-256 `0157a75fb8d7ec420d89a7a2f0339207ab43cc5fe84ed5e10439dc344cd3df00`.
All 44 Python and frontend source files match the wheel byte for byte. The only
change after the full-suite installation was the HTML label "Saved intervals",
clarifying that the initial state is included in addition to those intervals;
its browser control and packaged asset were checked separately. The native
binary and all executable Python/JavaScript were unchanged.

## Present boundary and next work

The experiment adapter covers one scalar field in 1D Cartesian, cylindrical or
spherical transient and steady diffusion/reaction transport. Studio allows up to
1,000 cells, 120 saved
intervals and 200,000 explicit steps. The Python experiment adapter allows 2,000
cells and 200 intervals; direct `Problem` users can work beyond those adapter
limits. Example values are illustrative, not calibrated biological data.

The separate coupled engine supports open and closed networks of compartments
and 1D spatial domains, multiple species, local reactions and prescribed bath
concentration protocols. Studio and the JSON
experiment adapter continue to cover their single-field scope. Further UI work
is deferred.

Long explicit radial transients still require a step count proportional to
resolution squared. The direct steady path now avoids that work when only the
final balance is needed; native planning exposes the cost when physical time
evolution is required. `CoupledModel` now provides sparse implicit transients
for its coupled-network scope, including radial domains and external baths.

Remaining engine boundaries are advection/flow coupling, time-varying membrane
permeability and partition coefficients, nonuniform or adaptive meshes, 2D/3D interface mappings,
coupled steady solving, parameter fitting and model serialization. Custom
reaction callbacks are local and must provide consistent derivatives. BDF and
Radau do not guarantee nonnegative states; inspect ranges, conserved quantities,
spatial refinement and tolerance convergence. Transient advection in the
existing native solver remains first-order upwind.
