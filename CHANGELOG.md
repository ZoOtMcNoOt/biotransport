# Changelog

All notable changes to BioTransport are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) with the alpha caveat that minor
releases may retire spellings under the documented
[deprecation policy](docs/notes/DEPRECATION_POLICY.md).

## [Unreleased]

### Added
- `biotransport.Result` and `biotransport.Snapshots`: `bt.solve` now returns a
  `Result` carrying the field(s), exact time, step count, diagnostics, a copy
  of the mesh, the contract identifier, and `snapshots`; `result.plot()` and
  `result.as_grid()` work without the original mesh object.
- `bt.solve(..., save_times=[...])` records the field at absolute times in one
  call, preserving every configured term; `result.snapshots[t]` returns each
  snapshot. `bt.plot(result)` accepts a result on its own.
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
- `solver.solve_until(end_time, time_step=None, *, save_times=None)` on all 19
  transient native stepping solvers (`biotransport.stepping`), returning a
  `Result` with `StepDiagnostics`. The time step is chosen automatically only
  when the solver certifies a stability limit; otherwise `time_step` is
  required, and the solver refuses to step backwards or land off the clock.
- Fluent boundary verbs `dirichlet`, `neumann`, `robin`, `boundary` and
  `outward_flux` on the same solvers; each forwards to the native setter or
  refuses conditions the solver cannot honour, and returns the solver.
- `mesh()` on `DiffusionSolver`, the explicit reaction-diffusion family,
  `AdvectionDiffusionSolver`, `CrankNicolsonDiffusion`, `ADIDiffusion2D` and
  `ADIDiffusion3D`.
- `biotransport.BioTransportDeprecationWarning` and a table-driven deprecation
  mechanism (`biotransport/_deprecation.py`) with a written policy
  (`docs/notes/DEPRECATION_POLICY.md`).
- Bitwise golden fixtures for every native solver (`python/tests/golden/`) so
  refactors can prove the numerics did not move.
- Stub-versus-runtime parity test for `biotransport._core` and a public-surface
  snapshot test.

### Changed
- `bt.plot` no longer calls `plt.show()` by default (`show=False`); it returns
  the figure so callers can add to it before showing.
- `bt.integrate` requires `method=` (`"euler"`, `"heun"` or `"rk4"`). Omitting
  it previously selected legacy RK4 with a `FutureWarning`; the algorithm is
  now always an explicit choice.
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
- `solve_until(..., maximum_dt=...)` on `MultiSpeciesSolver` and
  `NonuniformDiffusion1D`: the keyword is now `time_step`.
- `bt.solve(problem, t=..., dt=...)`: use `end_time=` and `time_step=`.
- `bt.run_checkpoints(...)`: use `bt.solve(..., save_times=[...]).snapshots`.
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
