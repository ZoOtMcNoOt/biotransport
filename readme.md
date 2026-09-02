# BioTransport

<div align="center">

**A science-first library for conservative transport and auditable modeling workflows**

*C++17 numerical core · Python bindings and scientific workflows · research and teaching*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#testing-and-development-status)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://github.com/ZoOtMcNoOt/biotransport/blob/master/LICENSE)

</div>

BioTransport is being rebuilt around one explicit scientific contract: define a scalar transport
problem once, solve every configured term in the C++ core, and report what the numerical method can
and cannot certify. The canonical API currently targets conservative, explicit transport on 1D and
2D Cartesian structured meshes.

The repository also contains older and specialized solvers. They remain available for development,
but they do **not** automatically inherit the verification claims of the canonical `Problem` / `solve`
path.

Performance-critical canonical and native specialized solver kernels are implemented in C++17.
Python provides bindings, configuration, unit conversion, provenance, sensitivity/uncertainty
orchestration, and reproducibility tooling. Some older Python numerical helpers also remain public;
the package is therefore not universally a thin wrapper, and those helpers require their own
scientific evidence.

## Governing equation and sign convention

The canonical solver advances

$$
\frac{\partial c}{\partial t}
= \nabla\!\cdot(D\nabla c)
- \nabla\!\cdot(\mathbf{v}c)
+ R(c,\mathbf{x},t).
$$

Equivalently, with the physical transport flux

$$
\mathbf{J} = -D\nabla c + \mathbf{v}c,
\qquad
\frac{\partial c}{\partial t} = -\nabla\!\cdot\mathbf{J} + R.
$$

This is conservative advection: spatial variation in velocity is included through
$-\nabla\cdot(\mathbf{v}c)$, not approximated as only $-\mathbf{v}\cdot\nabla c$. A positive reaction
term adds concentration. All quantities must use one mutually consistent unit system; the library
does not attach units to raw C++ solver scalars. The optional `biotransport.units` Python layer
performs explicit, dimension-checked conversion to solver-ready SI values.

## Python quick start

This example solves 1D advection-diffusion with first-order decay. `mesh_1d(100, ...)` creates 100
cells and 101 nodes.

```python
import numpy as np
import biotransport as bt

mesh = bt.mesh_1d(100, x_min=0.0, x_max=1.0)
x = bt.x_nodes(mesh)
initial = np.exp(-((x - 0.35) / 0.07) ** 2)

problem = (
    bt.Problem(mesh)
    .diffusivity(1.0e-2)
    .velocity(0.15)
    .linear_decay(0.20)
    .initial_condition(initial)
    .dirichlet(bt.Boundary.Left, 0.0)
    .neumann(bt.Boundary.Right, 0.0)
)

# The C++ core selects a stable explicit step and shortens the last step so
# the returned time is exactly end_time.
result = bt.solve(problem, end_time=0.10)

print(f"time: {result.time}")
print(f"steps: {result.diagnostics.steps}")
print(f"mass change: {result.diagnostics.mass_change:.6g}")
print(result.concentration)  # NumPy array containing the final nodal field
```

To request a maximum step explicitly, use `time_step`:

```python
result = bt.solve(problem, end_time=0.10, time_step=1.0e-3)
```

The step is honored unless it exceeds the solver's explicit stability limit, in which case the
solve raises an exception. The older `t` and `dt` spellings still work but emit a
`BioTransportDeprecationWarning`; see `docs/notes/DEPRECATION_POLICY.md`.

`solve` returns a `Result` that carries the final field, the exact time, the step count, the
solver diagnostics, a copy of the mesh, and the identifier of the scientific contract that
produced it. To record the field at intermediate times in the same call, pass `save_times`; the
C++ solver partitions its step schedule so every snapshot lands exactly on the requested clock and
every configured term (including a time-dependent reaction) is preserved:

```python
result = bt.solve(problem, end_time=0.10, save_times=[0.02, 0.05, 0.10])
result.snapshots[0.05]        # NumPy array at t = 0.05
result.snapshots.stacked()    # (3, n_nodes) array
result.plot(title="t = 0.10") # the result knows its mesh
```

The specialized native stepping solvers share the same lifecycle. Configure the field and the
boundaries, then call `solve_until`; it returns a `Result` too, records `save_times` snapshots, and
picks a time step automatically only when the solver certifies its own stability limit (otherwise
`time_step=` is required and the call refuses to guess):

```python
solver = bt.DiffusionSolver(mesh, 1.0e-9)
solver.set_initial_condition(initial)
solver.dirichlet(bt.Boundary.Left, 1.0).neumann(bt.Boundary.Right, 0.0)
result = solver.solve_until(600.0, save_times=[60.0, 300.0])
```

Mesh helpers follow one pattern for every dimension: `mesh_1d`, `mesh_2d` and `mesh_3d` take cell
counts and bounds, `sides(mesh)` returns the boundary identifiers in canonical order, and the
initial-condition helpers (`gaussian`, `step`, `uniform`, `circle`, `sinusoidal`) return NumPy arrays
you can combine before handing them to a problem or solver. `bt.plot(mesh, values, save_to="c.png")`
renders any field and saves it in one call.

The root namespace advertises only the canonical path (`Problem`, `solve`, `Result`, `solve_until`,
meshes, boundaries, field helpers, `plot`, VTK writers) plus thin namespaces that organize the rest:
`bt.diffusion`, `bt.electrochem`, `bt.flow`, `bt.applications`, `bt.balance` (dimensioned balance
ledgers and residuals), `bt.reference` (Python reference and legacy numerics with their own
contracts), `bt.stepping`, and the workflow modules. Every specialized native class is still an
attribute of the root, so `bt.DiffusionSolver` keeps working; the namespaces only organize the API,
and numerical work still runs in the C++ core. Application configuration objects also provide
high-level factories, such as `BioheatCryotherapyConfig.create_solver(...)`, so ordinary users do
not need to call long low-level constructors directly.

## Scientific workflow and evidence APIs

The numerical result is only one part of a defensible workflow. BioTransport now exposes separate,
machine-readable tools for the assumptions and evidence around a run:

| Capability | Python surface | Scope boundary |
|---|---|---|
| Explicit unit conversion | `biotransport.units` | Converts and dimension-checks selected quantities; raw C++ solvers still receive plain, caller-consistent doubles. |
| Parameter traceability | `biotransport.provenance` | Records sources, context, validity, and uncertainty; bundled application values remain honestly illustrative and unprovenanced. |
| Sensitivity and uncertainty screening | `biotransport.analysis` | Seeded sweeps, local sensitivities, independent-marginal Latin hypercubes, propagation, and linear SRC screening—not calibration, causality, or model validation. |
| Numerical contracts | `biotransport.contracts` | Separately records native solver equations/units/evidence and governed Python numerical backends/dispositions; neither registry confers biological validation. |
| Balance accounting | `BalanceLedger` and `reconcile_balances` | Audits caller-supplied inventories and exchanges; it does not couple PDEs or infer fluxes from fields. |
| Reproducible artifacts | `biotransport.reproducibility` | Produces deterministic, fingerprinted JSON manifests; a manifest is not a publication repository or biological validation. |

For example, a units object fails before a length can be passed where diffusivity is expected:

```python
from biotransport import units

D = units.diffusivity(1.33e-5, "cm^2/s")
problem.diffusivity(D.require(units.Dimension.DIFFUSIVITY))  # 1.33e-9 m^2/s
```

The native registry can be queried by stable ID or native symbol, while the
separate Python registry makes reference/workflow backends explicit:

```python
from biotransport.contracts import get_contract, get_python_numerical_contract

contract = get_contract("NernstPlanckSolver")
print(contract.equation)
print(contract.evidence_level.value)
print(contract.exclusions)

python_surface = get_python_numerical_contract("NewtonRaphsonSolver")
print(python_surface.backend.value)  # python-reference
print(python_surface.disposition)
```

The first nonuniform-geometry slice is a separate native C++
`NonuniformDiffusion1D` solver. It uses a fitted node-centred finite-volume mesh, harmonic face
diffusivity, a conservative shared flux, and a checked local Forward-Euler limit. It does not add
nonuniform 2D/3D geometry, unstructured meshes, AMR, moving meshes, advection, or reaction.

## C++ quick start

```cpp
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/problems/transport_problem.hpp>
#include <biotransport/solvers/transport_solver.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    using namespace biotransport;

    StructuredMesh mesh(100, 0.0, 1.0);
    std::vector<double> initial(static_cast<std::size_t>(mesh.numNodes()));
    for (int i = 0; i <= mesh.nx(); ++i) {
        const double x = mesh.x(i);
        initial[static_cast<std::size_t>(mesh.index(i))] =
            std::exp(-std::pow((x - 0.35) / 0.07, 2));
    }

    TransportProblem problem(mesh);
    problem.diffusivity(1.0e-2)
        .velocity(0.15)
        .linearDecay(0.20)
        .initialCondition(initial)
        .dirichlet(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 0.0);

    SolveOptions options = SolveOptions::until(0.10);
    const TransportResult result = solve(problem, options);

    std::cout << "time: " << result.time << '\n';
    std::cout << "steps: " << result.diagnostics.steps << '\n';
    std::cout << "mass change: " << result.diagnostics.mass_change << '\n';
}
```

After installing the C++ package, a CMake consumer can link the exported target:

```cmake
find_package(biotransport CONFIG REQUIRED)
target_link_libraries(my_model PRIVATE biotransport::biotransport)
```

## Boundary semantics

The outward unit normal $\mathbf{n}$ defines every derivative boundary value.

| Builder | Mathematical meaning | Important detail |
|---|---|---|
| `.dirichlet(side, value)` | $c=\text{value}$ | Essential nodal value, imposed before the first stencil. |
| `.neumann(side, g)` | $\partial c/\partial n=g$ | `g` is an outward-normal **derivative**, not a premultiplied flux. The outward diffusive flux is $-Dg$. |
| `.robin(side, a, b, rhs)` | $a c+b\,\partial c/\partial n=rhs$ | `b == 0` is treated as an essential condition. |

Unspecified sides default to zero outward-normal derivative. That removes diffusive flux, but it is
not a no-transport wall when $\mathbf{v}\cdot\mathbf{n}\ne0$: advection can still carry material
through the boundary. Pure-advection inflow (`D == 0`) therefore requires concentration data through
Dirichlet or `b == 0` Robin data.

In 2D, essential conditions meeting at a corner must agree. Contradictory corner values are rejected
rather than resolved by an arbitrary side-ordering rule.

## Canonical API

### `TransportProblem`

`TransportProblem` (exported to Python as both `TransportProblem` and `Problem`) owns a mesh and a
complete scalar model. Its fluent builder supports:

- uniform or node-centred non-negative diffusivity;
- uniform or node-centred velocity in 1D/2D;
- fixed fields or scalar values for the initial condition;
- constant sources, linear decay, Michaelis-Menten consumption, logistic growth, and custom
  `R(c, x, y, t)` reactions;
- composable reactions through the `add...` methods; and
- Dirichlet, outward-derivative Neumann, and Robin boundaries.

Calling a replacement method such as `linear_decay` replaces the current reaction. Calling an
`add_...` method composes it with the existing reaction, making multi-process source terms explicit.

### `SolveOptions`

| C++ field | Python `solve` argument | Meaning |
|---|---|---|
| `final_time` | `end_time` | Non-negative physical duration. |
| `time_step` | `time_step` | Maximum explicit step; zero/omitted selects one automatically. |
| `safety_factor` | `safety_factor` | Fraction of the certified transport stability limit; default `0.8`. |
| `reaction_step_fraction` | `reaction_step_fraction` | Reaction-timescale accuracy guard; default `0.1`. |
| `max_steps` | `max_steps` | Guard against unexpectedly large explicit runs. |
| `check_finite` | `check_finite` | Reject non-finite reaction or solution values. |

Built-in bounded reactions provide a derivative bound for automatic stepping. For a custom reaction,
either declare `max_abs_dc` or choose `time_step` yourself:

```python
problem.reaction(
    lambda c, x, y, t: -0.4 * c * c,
    max_abs_dc=0.8,  # valid over the concentration range expected in this model
)
```

An explicit step for a custom reaction without a bound is allowed, but diagnostics mark the reaction
stability certificate as unknown.

### Results and diagnostics

`TransportResult` contains the final `concentration`, exact returned `time`, and `diagnostics`.
Diagnostics report the requested and used step sizes, transport and certified stability limits,
whether reaction stability was known, initial/final extrema, and trapezoidal mass change. They are
there to make numerical assumptions inspectable, not to replace problem-specific verification.

## Architecture

```text
Python: units/provenance/UQ/artifacts + configuration/convenience APIs
                              |
         canonical solve() normalization and pybind11 bindings
                              |
C++: validation -> conservative fluxes -> stable-step policy -> solvers -> diagnostics
```

The Python `solve()` function is deliberately thin: it validates aliases and method names, constructs
`SolveOptions`, and calls `solve_transport`. It does not duplicate the PDE operator or integrate a
different model in Python. The C++ `solve(problem, options)` function is the canonical implementation.
That statement is scoped to the canonical path; unit/provenance/UQ/artifact utilities and some legacy
numerical helpers are genuine Python implementations.

## Verified scope

The current science-first verification contract covers:

- one scalar field on 1D and 2D Cartesian `StructuredMesh` grids;
- node-centred finite-volume balances with half control volumes at physical boundaries;
- harmonic face averaging for variable diffusivity;
- conservative first-order upwind advection;
- first-order explicit Euler time integration with enforced transport/CFL limits;
- configured diffusion, advection, and reaction terms advanced together;
- Dirichlet, outward-normal Neumann, and Robin boundary behavior;
- conservation for closed variable-coefficient problems, manufactured steady cases, smooth
  second-order diffusion-space refinement, heterogeneous mixed-boundary first-order upwind
  refinement, first-order reaction-time refinement, exact final-time landing, and deterministic
  corner handling.

The runnable `examples/verification/grid_convergence.py` performs a scoped spatial and temporal
refinement report for one diffusion case. Smooth-diffusion spatial refinement is now also in the
always-on CTest gate, but neither sequence is a convergence certificate for every canonical
configuration. The nonuniform 1D solver separately has an automated smooth-mesh
spatial-convergence test; Navier--Stokes velocity, bioheat space/time, Nernst--Planck's diffusion
limit, and cylindrical operators now have their own bounded order evidence. The specialized spatial
studies suppress time error explicitly; exact claims are listed individually in the contract
registry.

This canonical contract does **not** extend automatically to 3D or cylindrical grids, implicit or
higher-order integration, central/QUICK/hybrid advection, coupled multi-species systems, fluid
dynamics, electrochemical transport, or application-scale models. Several specialized classes now
have their own focused conservation, manufactured-solution, stability, or limit-case tests. Those
are separate evidence scopes—not a blanket validation of every parameter regime, biological
closure, or clinical use. Review the model-specific documentation and validate the intended case
before drawing scientific conclusions.

## Fail-loud policy

The canonical API rejects unsupported or scientifically ambiguous configurations instead of silently
substituting another model. Examples include:

- `bt.solve(..., method="crank_nicolson")` or another unverified method;
- `AdvectionScheme.CENTRAL`, `HYBRID`, or `QUICK` in the canonical solver;
- an explicit step above the enforced stability ceiling;
- automatic stepping for a custom reaction without a derivative bound;
- non-finite fields or model evaluations;
- contradictory essential values at a 2D corner; and
- pure-advection inflow without prescribed concentration data.

At present, `method="conservative"`, `"explicit"`, and `"explicit_euler"` all name the same verified
canonical algorithm. Other algorithms must be selected through their specialized APIs, where their
own assumptions and validation apply.

## Installation

### Python development install

Python 3.9+ and a C++17 compiler are required to build the extension. The PEP 517
build declares and installs its CMake, Ninja, pybind11, and Eigen header dependencies;
it does not clone build dependencies during CMake configuration.

```bash
git clone https://github.com/ZoOtMcNoOt/biotransport.git
cd biotransport
python -m pip install -e ".[test]"
```

On Windows, install Visual Studio Build Tools with the Desktop development with C++ workload.

### C++ build

Direct C++ builds require CMake 3.16+ and a discoverable Eigen 3.4 package for the
default sparse backend. Configure with `-DBIOTRANSPORT_EIGEN=OFF` only when that
backend is intentionally unavailable.

```bash
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Testing and development status

```bash
# Python API and regression tests
python -m pytest python/tests -q

# C++ tests after configuring/building
ctest --test-dir build -C Release --output-on-failure
```

BioTransport is alpha software. Readiness is stated as a verified method/scope rather than a
hard-coded test count: the count changes as the suite grows and is not evidence that every legacy or
specialized module has the same validation depth. Check the current test run and review the relevant
verification cases for the physics you plan to model.

The implementation rationale and near-term boundaries are recorded in the
[`science-first architecture guide`](https://github.com/ZoOtMcNoOt/biotransport/blob/master/docs/notes/SCIENCE_FIRST_ARCHITECTURE.md).
Application equations, provenance, and biological-validity limits are recorded in the
[`model scope and references guide`](https://github.com/ZoOtMcNoOt/biotransport/blob/master/docs/notes/MODEL_SCOPE_AND_REFERENCES.md).
The current machine-readable solver evidence is summarized in the
[`solver contract guide`](https://github.com/ZoOtMcNoOt/biotransport/blob/master/docs/notes/SOLVER_CONTRACTS.md).
Workflow guides cover
[`units`](https://github.com/ZoOtMcNoOt/biotransport/blob/master/docs/notes/UNITS.md),
[`parameter provenance`](https://github.com/ZoOtMcNoOt/biotransport/blob/master/docs/notes/PARAMETER_PROVENANCE.md),
[`sensitivity and uncertainty`](https://github.com/ZoOtMcNoOt/biotransport/blob/master/docs/notes/SENSITIVITY_AND_UNCERTAINTY.md),
[`balance accounting`](https://github.com/ZoOtMcNoOt/biotransport/blob/master/docs/notes/BALANCE_ACCOUNTING.md),
[`reproducible artifacts`](https://github.com/ZoOtMcNoOt/biotransport/blob/master/docs/notes/REPRODUCIBILITY.md), and the
[`nonuniform 1D geometry slice`](https://github.com/ZoOtMcNoOt/biotransport/blob/master/docs/notes/NONUNIFORM_GEOMETRY.md).

## Contributing

Contributions are welcome. New numerical paths should state their equation, sign and boundary
conventions, supported dimensions, stability policy, and verification evidence. Unsupported choices
should fail explicitly until that evidence exists.

## License

BioTransport is available under the
[MIT License](https://github.com/ZoOtMcNoOt/biotransport/blob/master/LICENSE).

Built for transport-phenomena research and teaching at Texas A&M University.
