# Engine and workbench increment — 12 September 2026

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
extended in place. No commit, push or deployment is part of this record.

The follow-on extension began from a second snapshot of 90 changed files in
`build/extension-validation/baseline/`. Both snapshots and the first installed
wheel are retained. The current work remains local on the same branch and HEAD.

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

The next architectural gap is an explicit coupling contract for multiple
domains, species and membranes: units, interface fluxes, conservation accounting
and numerical reference cases. A connected-node editor should follow that
contract. Arbitrary networks, 2D editing, mesh adaptation and adapters for the
specialized multiphysics solvers remain outside this increment. Transient flow
still uses first-order upwind spatial discretization; the improved steady path
and visual editor do not change that accuracy limit.

Long explicit radial transients still require a step count proportional to
resolution squared. The direct steady path now avoids that work when only the
final balance is needed; native planning exposes the cost when physical time
evolution is required. An implicit radial transient method, coupled domains and
species, and mesh refinement remain substantive future engine work.
