# Changelog

All notable changes to BioTransport are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) with the alpha caveat that minor
releases may retire spellings under the documented
[deprecation policy](docs/notes/DEPRECATION_POLICY.md).

## [Unreleased]

### Added
- `SolveOptions.save_times`: the canonical C++ solver records the field at
  requested absolute times in one call. Each save time partitions the step
  schedule so the field is captured exactly at that clock, and the reaction
  term always receives the absolute time. An empty list leaves the result
  bitwise unchanged (guarded by the golden fixtures).
- `TransportResult.mesh`, `snapshot_times` and `snapshot_fields`, so a result
  can be plotted, exported or reshaped on its own.
- `BoundaryType.OUTWARD_FLUX` / `BoundaryCondition.outward_flux(...)`: a
  distinct type for prescribed physical flux, and
  `NernstPlanckSolver.set_outward_flux_boundary` /
  `MultiIonSolver.set_outward_flux_boundary` as the unambiguous spelling for
  the molar-flux condition those solvers implement. `Problem` rejects the
  flux type loudly instead of reinterpreting it as a derivative.
- `time()`, `check_stability(dt)` on every explicit reaction-diffusion and
  advection-diffusion solver, and `max_stable_time_step()` on
  `DiffusionSolver`.
- `biotransport.BioTransportDeprecationWarning` and a table-driven deprecation
  mechanism (`biotransport/_deprecation.py`) with a written policy
  (`docs/notes/DEPRECATION_POLICY.md`).
- Bitwise golden fixtures for every native solver (`python/tests/golden/`) so
  refactors can prove the numerics did not move.
- Stub-versus-runtime parity test for `biotransport._core` and a public-surface
  snapshot test.

### Changed
- The Neumann keyword on `Problem.neumann`, `DiffusionSolver.set_neumann_boundary`
  and `ReactionDiffusionSolver.set_neumann_boundary` is now `normal_derivative`
  (it was `flux`, although the value is the outward-normal derivative). No
  caller passed it by keyword.
- `NernstPlanckSolver.solve` and `MultiIonSolver.solve` validate the stability
  bound before writing any Dirichlet trace into the field, so a rejected step
  leaves the exposed state untouched.
- The ten C++ tests that used `assert()` now use the `science_test` harness, so
  their checks also execute in Release builds, and the test CMake configuration
  rejects any test that includes `<cassert>`.
- The Sphinx configuration honours `BIOTRANSPORT_DOCS_OFFLINE=1` to skip
  intersphinx inventories when building without network access.

### Deprecated
- `TransportResult.solution`: use `TransportResult.concentration`.
- `NernstPlanckSolver.set_neumann_boundary` and
  `MultiIonSolver.set_neumann_boundary`: use `set_outward_flux_boundary`; the
  value is a physical molar flux, not a derivative.
- `biotransport.run(problem, t_end, ...)`: use `biotransport.solve(problem, end_time=...)`.
- `biotransport.DiffusionProblem`, `LinearReactionDiffusionProblem` and
  `AdvectionDiffusionProblem`: use `biotransport.Problem`.

## [0.1.0]

Initial science-first release: canonical conservative scalar transport on 1D/2D
structured meshes, specialized native solvers, units, provenance, sensitivity
and uncertainty screening, balance accounting, and reproducibility manifests.
