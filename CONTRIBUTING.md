# Contributing to BioTransport

BioTransport is science-first: the numerics must be inspectable, every public
surface must fail loudly on inputs it cannot honour, and no change may move a
verified result silently. The rules below exist to keep that true.

## Ground rules

1. **No silent numerics change.** `python/tests/golden/` holds bitwise fixtures
   for every native solver. A refactor must leave them green. When a numerical
   change is intended, regenerate the fixtures with
   `python python/tests/golden/capture_goldens.py`, explain the change in the
   commit and in `CHANGELOG.md`, and add the evidence that justifies it.
2. **No automatic time step without a certificate.** Only a solver that exposes
   its own stability bound may choose a step for the caller; everything else
   requires an explicit `time_step`.
3. **Deprecate, do not break.** Renames go through `biotransport/_deprecation.py`
   and follow `docs/notes/DEPRECATION_POLICY.md` (warn for one minor release,
   remove in the next). The test suite treats
   `BioTransportDeprecationWarning` as an error, so callers inside the repo
   must move to the new spelling in the same change.
4. **Contracts before claims.** A new native solver needs a `SolverContract`; a
   new Python numerical surface needs a `PythonNumericalContract`. Evidence
   selectors must point at real tests. See `docs/notes/SOLVER_CONTRACTS.md`
   and the definition of done in `docs/notes/GAP_ANALYSIS.md`.
5. **Tests use the `science_test` harness in C++**, never `assert()`, so they
   also execute in Release builds.

## Development loop

```bash
# Python extension (Release, written straight into the package tree)
cmake -S . -B build/py -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PYTHON_BINDINGS=ON -DBUILD_TESTING=OFF \
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$PWD/python/biotransport/_core/
cmake --build build/py --parallel

# Native tests (Debug, warnings as errors)
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON \
      -DBIOTRANSPORT_WERROR=ON
cmake --build build/native --parallel && ctest --test-dir build/native -C Debug

# Python tests, lint, typing, docs, examples
python -m pytest
pre-commit run --all-files
python -m mypy --no-incremental --ignore-missing-imports \
       python/biotransport/_core/_core.pyi python/tests/typing/api_smoke.py
BIOTRANSPORT_DOCS_OFFLINE=1 python -m sphinx -W --keep-going -E -b html docs/sphinx build/docs
MPLBACKEND=Agg python run_examples.py
```

`pre-commit install` runs ruff, clang-format, cmake-format and the file-hygiene
hooks on every commit.

## Changing the public API

- `python/biotransport/_core/_core.pyi` is hand-maintained and checked against
  the runtime by `python/tests/test_stub_parity.py`; update it with the
  bindings.
- `python/tests/data/public_surface.json` snapshots `biotransport.__all__`, the
  namespaces and the deprecation tables. When a change is intentional,
  regenerate it with `BIOTRANSPORT_UPDATE_SNAPSHOTS=1 python -m pytest
  python/tests/test_public_surface.py` and describe the change in the changelog.
- New root-level names belong to the canonical tier only (`Problem`, `solve`,
  `Result`, meshes, boundaries, field helpers, `plot`, VTK). Everything else
  lives in a namespace: `diffusion`, `electrochem`, `flow`, `applications`,
  `balance`, `reference`, `stepping`, or a workflow module.
- Record every user-visible change under `## [Unreleased]` in `CHANGELOG.md`.
