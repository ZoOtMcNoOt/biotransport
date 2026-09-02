# Changelog

All notable changes to BioTransport are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) with the alpha caveat that minor
releases may retire spellings under the documented
[deprecation policy](docs/notes/DEPRECATION_POLICY.md).

## [Unreleased]

### Added
- `biotransport.BioTransportDeprecationWarning` and a table-driven deprecation
  mechanism (`biotransport/_deprecation.py`) with a written policy
  (`docs/notes/DEPRECATION_POLICY.md`).
- Bitwise golden fixtures for every native solver (`python/tests/golden/`) so
  refactors can prove the numerics did not move.
- Stub-versus-runtime parity test for `biotransport._core` and a public-surface
  snapshot test.

### Changed
- The ten C++ tests that used `assert()` now use the `science_test` harness, so
  their checks also execute in Release builds, and the test CMake configuration
  rejects any test that includes `<cassert>`.
- The Sphinx configuration honours `BIOTRANSPORT_DOCS_OFFLINE=1` to skip
  intersphinx inventories when building without network access.

### Deprecated
- `biotransport.run(problem, t_end, ...)`: use `biotransport.solve(problem, end_time=...)`.
- `biotransport.DiffusionProblem`, `LinearReactionDiffusionProblem` and
  `AdvectionDiffusionProblem`: use `biotransport.Problem`.

## [0.1.0]

Initial science-first release: canonical conservative scalar transport on 1D/2D
structured meshes, specialized native solvers, units, provenance, sensitivity
and uncertainty screening, balance accounting, and reproducibility manifests.
