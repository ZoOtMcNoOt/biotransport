# Contributing to BioTransport

Thanks for wanting to help. This file covers how to get set up, what the tests
expect, and the one rule that matters more than the rest.

## The one rule

**A numerical path has to say what it claims, and the claim has to be backed by a
test.**

That means, for anything new that computes a number: state the equation it
solves, its sign and boundary conventions, which dimensions it supports, its
stability policy, and what evidence exists that it is correct. Then add the test
that produces that evidence.

Until the evidence exists, make the code refuse the configuration. A solver that
returns a plausible wrong answer is worse than one that stops, because you cannot
tell from the output which happened. That principle is why `bt.solve` rejects
unverified methods and unstable time steps instead of doing its best.

Registering a claim is not a formality — `biotransport.contracts` is a real
registry that users query, so add an entry there for a new solver.

## Getting set up

```bash
git clone https://github.com/ZoOtMcNoOt/biotransport.git
cd biotransport
python -m pip install -e ".[test,dev,docs]"
pre-commit install
```

You need a C++17 compiler, because installing builds the core. Windows: Visual
Studio Build Tools with "Desktop development with C++". macOS:
`xcode-select --install`. Linux: gcc.

## Running things

```bash
# Python tests
python -m pytest python/tests -q

# One file, verbosely, while you work
python -m pytest python/tests/test_solution_api.py -v

# C++ tests
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure

# Lint and format
ruff check python/biotransport
ruff format python/biotransport

# Types
python -m mypy --strict python/tests/typing/api_smoke.py

# Docs
python -m sphinx -b html docs/sphinx build/_docs
```

Set `MPLBACKEND=Agg` if a test opens a plot window on your machine.

## What good tests look like here

The suite leans hard on comparing against things that are independently known:

- **Against an exact solution.** `biotransport.analytical` has the series and
  closed-form solutions. `Solution.compare` gives you error norms directly. Quote
  the L2 norm, not the RMS — it is control-volume weighted, so it means the same
  thing across grids.
- **Against a convergence rate.** A single error number can be luck; an error
  that falls by 4× when you halve Δx is much harder to fake. `bt.convergence` has
  helpers for this.
- **Against a conservation law.** On a closed domain the total should not move.
  `sol.total()` and `diagnostics.mass_change` make that a one-line assertion.
- **Against a limit case.** Set a term to zero and check you recover the simpler
  problem you already trust.

Tests that assert an implementation detail rather than a behaviour tend to make
refactoring painful, so prefer the ones above.

## Style

- `ruff` decides formatting; don't argue with it.
- Docstrings are Google style. Say what a thing is *for* before you say what it
  does, and name the units or convention when there is any doubt.
- Write comments about constraints the code cannot express — a sign convention, a
  reason a fallback exists, a trap the next reader will hit. Skip comments that
  restate the line below them.
- Full words in names. `diffusivity`, not `D`, outside of a formula where `D` is
  the conventional symbol.

## Pull requests

Keep them focused; a PR that does one thing gets reviewed quickly. In the
description, say what changed and how you know it works — the test you added, the
convergence rate you measured, the case you checked against.

If you find a bug, a failing test that demonstrates it is a complete and very
welcome contribution on its own.

## Where things live

```
cpp/include/biotransport/   the numerical core, header-heavy
cpp/src/                    compiled implementations
cpp/tests/                  C++ tests, run by ctest
python/bindings/            pybind11 glue
python/biotransport/        the Python API
python/tests/               pytest suite
examples/                   runnable examples, basic through verification
docs/notes/                 working notes on scope, units, provenance, evidence
docs/sphinx/                the documentation site
```

The Python layer is deliberately thin for anything that computes: `solve()`
validates arguments and hands off to C++. Where Python *does* implement numerics
— units, provenance, sensitivity, and some older helpers — that is recorded in
`biotransport.contracts` so nobody has to guess.
