# Science-first architecture

BioTransport's canonical transport implementation and performance-critical
native solver kernels are C++17. Python provides bindings and deliberately
high-level scientific workflow tools. The package is **not** universally a thin
wrapper: unit conversion, parameter provenance, sensitivity/uncertainty
orchestration, reproducibility manifests, and some older numerical helpers are
implemented in Python.

The architectural rule is therefore narrower and testable:

> A public solver may not silently implement a different equation in Python
> while inheriting the name or evidence of a native C++ solver.

The canonical `Problem` / `solve` path is thin: Python normalizes arguments and
calls the compiled `solve_transport` entry point. Older Python numerical paths
remain separate surfaces and do not borrow the canonical claim. The separate
`PythonNumericalContract` registry now names their backend, mathematical scope,
failure policy, evidence, and disposition. Native adapters are retained;
workflow orchestration stays in Python; legacy stepping loops are explicit
port/deprecation candidates.

## Governing contract

The canonical scalar-transport problem is written in conservative form:

```text
dc/dt = div(D grad(c)) - div(v c) + R(c, x, y, t)
```

Every native solver contract must state:

- its equation, flux signs, unknowns, and storage locations;
- the units expected at the raw API boundary;
- supported dimensions, coefficient forms, terms, and boundary conditions;
- the spatial/temporal or steady numerical method;
- stability and convergence/failure policies;
- exact automated evidence references; and
- exclusions and interpretation warnings.

`python/biotransport/contracts.py` contains two machine-readable registries.
`SolverContract` covers compiled solver entry points. The deliberately separate
`PythonNumericalContract` covers governed public Python modules without
mislabeling them as native. Runtime tests require exact native entry-point
coverage, exact ownership of public symbols from governed Python modules, and
current evidence paths. An evidence level applies only to its recorded claim;
it is not biological validation or a blanket endorsement of every
configuration.

## Layer responsibilities

```text
Scientific records and orchestration (Python)
  units | parameter provenance | UQ screening | reproducibility manifests
  convergence | explicitly labeled reference algorithms
                              |
User configuration and native bindings (Python/pybind11)
                              |
Numerical kernels and solver diagnostics (C++17)
  canonical transport | specialized native solvers | balance arithmetic
```

The layers have explicit boundaries:

- `biotransport.units` converts a bounded set of quantities and checks semantic
  dimensions. Raw C++ solvers remain plain-double APIs and require a mutually
  consistent caller-selected unit system.
- `biotransport.provenance` records parameter source/context/uncertainty. It
  cannot judge source quality or make an illustrative default recommended.
- `biotransport.analysis` evaluates a caller-defined scalar quantity of
  interest. It can orchestrate native solves, but it does not calibrate a model,
  infer distributions, establish causality, or include model discrepancy by
  default.
- `BalanceLedger` accounts for caller-supplied inventories and exchanges. It
  does not integrate PDE fields, infer fluxes, choose a coupling algorithm, or
  advance coupled solvers. Automatic result-ledger coupling remains open work.
- `biotransport.reproducibility` freezes data and evidence into deterministic,
  fingerprinted JSON. A content hash is not authorship, durable archival, or
  validation.

## Public API principles

1. A user describes canonical scalar physics with a `TransportProblem` and
   numerical policy with `SolveOptions`.
2. `solve(problem, options)` returns values, exact final time, diagnostics, and
   the balance information that implementation actually computes.
3. Defaults are deterministic and conservative. Parallel or accelerated paths
   require equivalence evidence for their claimed configurations.
4. Boundary quantities have explicit mathematical definitions. A derivative
   and a physical flux must not share an ambiguous name.
5. Canonical time-step selection combines every active explicit operator.
   Each specialized solver contract states its own stability and exact-time
   policy.
6. Unsupported schemes, dimensions, constitutive domains, singular systems,
   and non-converged solves fail loudly or return an explicit non-converged
   diagnostic instead of falling through to another model or pretending
   success.
7. No method clips, normalizes, substitutes a theoretical result, or changes
   requested time/state semantics without reporting it.
8. Friendly APIs may reduce ceremony, but may not hide equation, units,
   evidence level, or exclusions.

## Geometry policy

Uniform Cartesian structured meshes remain the broad solver foundation. The
first nonuniform slice is intentionally separate: a fixed fitted 1D
node-centred finite-volume diffusion solver with harmonic face coefficients,
shared conservative fluxes, a local explicit stability certificate, and
smooth-mesh convergence evidence.

That slice is not an unstructured or AMR framework. Nonuniform 2D/3D, general
metric geometry, mesh motion, contact resistance, advection, reaction, and
adaptive refinement require their own discrete contracts and evidence.

## Verification standard

Unit-test counts are not scientific evidence. A production claim needs, as
applicable:

- an independent analytical or manufactured solution;
- measured spatial and temporal order over a genuine refinement sequence;
- mass, charge, volume, or energy balance checks;
- nonzero boundary-condition tests and explicit sign checks;
- discontinuous-coefficient interface tests;
- serial/parallel equivalence tests when parallel results are claimed; and
- rejection tests for invalid, unsupported, unstable, singular, and
  non-converged configurations.

Verification programs must return a failing process status when their stated
acceptance criteria are not met. Indeterminate order is reported as
indeterminate, never replaced with theoretical order. A convergence study that
exists only as a runnable example must be described as an example until it is
part of the always-on test gate.

The canonical test gate now contains smooth-diffusion spatial refinement,
heterogeneous mixed-boundary upwind refinement, and reaction-time refinement.
Specialized order evidence remains solver-specific: current examples include a
manufactured Navier--Stokes velocity field, bioheat spatial/time eigenmodes,
the Nernst--Planck diffusion limit, and cylindrical radial/angular operators.
The specialized spatial studies use ``dt=O(h^3)`` to suppress first-order time
error.

## Performance policy

Performance work follows discrete-model verification. Kernels stay in C++ when
their workload benefits from native execution, release the Python GIL when
safe, and operate on explicit data layouts. OpenMP, SIMD, sparse backends, and
accelerators must preserve the documented scientific result within stated
tolerances.

The friendly `solve()` path and explicit `integrate(method="euler")` execute
the canonical C++ solver. Omitting `integrate()`'s method temporarily warns and
preserves historical RK4 behavior. Python sensitivity/convergence
orchestration remains Python by design. Legacy adaptive, Heun/RK4 diffusion,
Newton, and pulsatile reference paths are discoverable in the Python numerical
registry and are not presented as native-performance APIs.

No “massive speedup” or generic performance claim is published without a
reproducible baseline containing the problem size, correctness criterion,
compiler and flags, hardware, thread count, repetitions, and raw timing
artifact. The current benchmark programs are engineering tools, not yet a
publication-grade performance evidence set.
