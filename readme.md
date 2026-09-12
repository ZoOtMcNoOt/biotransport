# BioTransport

<div align="center">

**Transport phenomena you can actually check.**

*A C++17 finite-volume core with a Python API built for learning and for prototyping*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

</div>

BioTransport solves diffusion, advection and reaction problems — the equations behind oxygen
reaching tissue, a drug crossing a membrane, heat moving through skin during cryotherapy. The
numerics run in C++. The part you touch is Python, and it is designed around one question:

**how do you know your answer is right?**

So the library ships the exact solutions to compare against, tells you which dimensionless numbers
govern what you built, and refuses to run a configuration it cannot vouch for rather than quietly
handing back a plausible-looking wrong number.

It was written for two people. One is a student working through a transport course who wants to
check a homework problem and understand why the profile looks the way it does. The other is a
researcher who wants to try an idea before lunch and get a figure out of it.

---

## Install

You need Python 3.9 or newer and a C++17 compiler, because the core gets built on install.

```bash
git clone https://github.com/ZoOtMcNoOt/biotransport.git
cd biotransport
python -m pip install -e ".[test]"
```

On Windows, install the Visual Studio Build Tools with the "Desktop development with C++" workload
first. On macOS, `xcode-select --install`. On Linux you almost certainly already have gcc.

**New here?** [`docs/tutorial.md`](docs/tutorial.md) walks from install to checking a real problem
against its textbook answer, in about half an hour. The rest of this page is the tour.

## Visual workbench and reusable experiments

Run `python -m biotransport.studio` and open `http://127.0.0.1:8766` to build a
1D transport experiment with components, editable parameters, saved-frame
plots, a space–time view and exact-reference checks. Add buttons provide the
same actions as dragging. Save a model as JSON and run it through
`bt.Experiment.from_dict(document).run()` in Python.

The browser and Python share the same component registry and solver API. New
registered components supply their own parameter controls without changes to
the frontend. See the [workbench and extension guide](docs/workbench.md) for
examples, present scope and the path toward coupled model networks.

---

## Your first solve

A pulse of something dissolved in water, sitting in a 1 cm channel, spreading out:

```python
import biotransport as bt

mesh = bt.mesh_1d(200, 0.0, 0.01)          # 200 cells across 1 cm

problem = (
    bt.Problem(mesh)
    .diffusivity(1e-9)                     # m^2/s, about right for a small solute in water
    .initial(bt.gaussian(mesh, center=0.005, width=0.0005))
    .sealed("left")                        # nothing escapes either end
    .sealed("right")
)

sol = bt.solve(problem, end_time=60.0)
sol.plot()
```

That's the whole program. A few things happened that are worth knowing about.

You never chose a time step. The core worked out the largest step that is provably stable for your
grid and diffusivity, then used 80% of it. You can pass `time_step=` if you want to, and it will
refuse anything unstable rather than let you produce garbage.

`sol` is not a bare array — it kept hold of the mesh it was computed on, which is why `sol.plot()`
knows what to draw without being told. `sol.x` gives you the node coordinates, `sol.c` the final
field, and `sol.concentration` the same thing under the name the C++ layer uses.

And the units are yours. The library never assumes SI or anything else; it just needs you to be
consistent. If you'd rather have that checked, `bt.units` will do it (see [Units](#units) below).

---

## Ask whether it's right

This is the part most PDE libraries leave to you. Diffusion into a slab held at a fixed
concentration on both faces has a textbook series solution, and you can compare against it in one
call:

```python
import biotransport as bt

D, L = 1e-9, 0.01
mesh = bt.mesh_1d(400, 0.0, L)

problem = (
    bt.Problem(mesh)
    .diffusivity(D)
    .initial(0.0)
    .dirichlet("left", 1.0)
    .dirichlet("right", 1.0)
)

sol = bt.solve(problem, end_time=2000.0)

print(sol.compare(lambda x, t: bt.analytical.slab(x, t, D=D, L=L, c_surface=1.0)))
```

```text
Compared 401 nodes against the reference field.
  largest absolute error  1.59457e-05  (at node 64)
  RMS error               9.599e-06
  L2 error                9.61099e-06
  as a fraction of the reference range: 0.00164% peak, 0.000986% L2
```

Sixteen parts per million at the worst node. That is the discretization error you would expect from
400 cells, and seeing it is the point: agreement between an exact solution and a numerical one is
evidence that *both* are right, and it is the fastest way to find out that you fed in the wrong
number.

Pass `plot=True` and you get the overlay and the difference plotted for you.

`bt.analytical` has the solutions coursework is actually set on — the finite slab, the sphere
(Crank's series, the one for a spheroid or a microsphere), the cylinder (Bessel series, the radial
part of a Krogh cylinder), the semi-infinite `erf` solution, an instantaneous point release, and the
steady `cosh` profile for a slab with first-order consumption. They all take arrays, and they all
tell you when you have asked for a time so early that the truncated series would ripple.

### The dimensionless numbers, worked out for you

`sol.summary()` reports what the solver did *and* what governs the problem:

```python
print(sol.summary())
```

```text
Solution summary
============================================================
  grid          400 cells on [0, 0.01], dx = 2.5e-05
  time          reached t = 2000 in 8000 steps
  step size     used dt = 0.25  (chosen automatically)
  stability     certified limit dt = 0.3125; this run used 80% of it
  conservation  total went from 2.5e-05 to 0.00319165
                net change +0.003167 (it started essentially empty, so material entered through the boundaries)
  range         0.0248358 to 1 (started 0 to 1)
  saved frames  2 between t = 0 and t = 2000

Dimensionless numbers for this problem
------------------------------------------------------------
  Fourier       0.02        early: diffusion has only reached a fraction of the domain
  sqrt(D t) / L 0.1414      diffusion has spread about 0.00141 into a domain of 0.01
```

A Fourier number of 0.02 says diffusion has crept about a seventh of the way in — so if you were
expecting a filled-in profile, your end time is too short, and now you know without having to squint
at a plot. Add a velocity and you also get the Péclet number and the **grid** Péclet number, which is
the one that tells you whether your mesh resolves a front or is quietly smearing it. Add a reaction
and you get the Damköhler number and the depth the solute actually reaches.

On a closed domain that conservation line is a genuine check rather than bookkeeping: the sealed
pulse from the first example reports a net change of `+1.084e-18`, which is machine precision on a
total of `0.00125331`. If that number is ever large, something is wrong with the model, not the
arithmetic.

---

## Watch it happen

A single final frame teaches you very little. Ask for snapshots and you can overlay them or animate:

```python
sol = bt.solve(problem, end_time=2000.0, save_every=200.0)

sol.plot(times=[0.0, 200.0, 600.0, 2000.0])   # overlaid, with a legend
anim = sol.animate(save="slab.gif")           # keep the reference or it won't render
```

`save_at=[...]` picks specific times and `frames=20` gives you twenty evenly spaced ones. You also
get `sol.times`, `sol.at(t)` for the field at a time, and `sol.trace(at=0.003)` for the history at
one point — the numerical equivalent of putting a probe in the domain.

This works for any problem you can build, reactions and advection included. Under the hood the run
is split into segments, and a reaction written as a function of time gets its clock shifted so it
still sees absolute time rather than restarting at zero in every segment.

---

## Skip the transient

Plenty of problems only want the end state: the steady oxygen profile in tissue, the steady flux
through a membrane. Marching there explicitly means tens of thousands of tiny steps to watch a
transient you don't care about. So don't:

```python
import biotransport as bt

L, D = 100e-6, 2e-9            # 100 um of tissue
mesh = bt.mesh_1d(100, 0.0, L)

problem = (
    bt.Problem(mesh)
    .diffusivity(D)
    .michaelis_menten(Vmax=1e-3, Km=1e-3)    # saturable uptake by the cells
    .initial(0.05)
    .dirichlet("left", 0.05)                 # fed from the capillary side
    .sealed("right")
)

steady = bt.solve_steady(problem)
print(f"converged in {steady.newton.iterations} Newton iterations")
print(f"oxygen at the far side: {steady.c[-1]:.5f}")
```

```text
converged in 2 Newton iterations
oxygen at the far side: 0.04755
```

Two iterations instead of 125,000 steps, and the two agree to eight decimal places. You never wrote
down a Jacobian — the built-in kinetics know their own derivatives.

`bt.solve(problem, steady=True)` is the same thing if you prefer one entry point. It handles
diffusion and reaction, and will tell you to march the transient instead if you have advection. It
works in 2D too — an 80×80 grid solves in about a quarter of a second, because the Jacobian is
assembled analytically and sparsely rather than by finite differences.

---

## Fluxes, and whether the books balance

Most transport questions ask for a **rate**, not a field. How much oxygen is actually reaching the
tissue? What fraction of it gets consumed? The solver computes those fluxes on every face of every
step, so you can have them:

```python
steady = bt.solve_steady(problem)

print(f"oxygen flux in at the capillary side: {steady.flux_at('left'):.4g}")
print(f"consumed by the tissue:               {steady.uptake():.4g}")
print(steady.balance())
```

```text
oxygen flux in at the capillary side: -9.797e-08
consumed by the tissue:               -9.797e-08

Transport balance
==============================================================
  stored now          4.8367363e-06

  rates at steady state (amount per unit time)
    left      -9.79739e-08   (entering)
    right     +0   (sealed)
    net in    +9.79739e-08
    reaction  -9.79739e-08

  nothing accumulates at steady state, so these must cancel:
    in + reaction     +5.193e-17
    relative to the flow through it: 5.3e-10
```

Boundary quantities are **outward**, so negative means entering. Everything that came in got
consumed, and the balance closes to five parts in ten billion — which is the point. That last number
is a real check on the whole discretization, and it is the first thing that moves if you have a sign
error in a boundary condition.

`sol.flux()` gives the flux on every interior face, `sol.rate(side)` integrates a boundary flux over
its area, and on a transient run `balance()` checks the integrated form over your saved frames
instead. These are reconstructed from the returned field using the solver's own face formulas, not a
fresh finite difference — exact in 1D, and second-order in 2D, where a corner node owns two walls but
only one balance.

---

## The equation, and the conventions that bite

The core advances

$$
\frac{\partial c}{\partial t}
= \nabla\!\cdot(D\nabla c)
- \nabla\!\cdot(\mathbf{v}c)
+ R(c,\mathbf{x},t)
$$

on 1D and 2D Cartesian grids, as a node-centred finite-volume balance with half control volumes at
the boundaries. Advection is conservative — spatial variation in velocity goes through
$-\nabla\cdot(\mathbf{v}c)$, not the shortcut $-\mathbf{v}\cdot\nabla c$. A positive reaction adds
material.

Boundary conditions are defined against the **outward** normal:

| You write | It means | Worth knowing |
|---|---|---|
| `.dirichlet(side, value)` | $c = \text{value}$ | Imposed before the first stencil. |
| `.neumann(side, g)` | $\partial c/\partial n = g$ | `g` is a *derivative*, not a flux. The outward diffusive flux is $-Dg$. |
| `.sealed(side)` | $\partial c/\partial n = 0$ | Shorthand for the common case. |
| `.robin(side, a, b, rhs)` | $ac + b\,\partial c/\partial n = rhs$ | For convective exchange, `a = h`, `b = D`, `rhs = h c_\infty`. `b = 0` is treated as a fixed value. |

Two traps worth stating outright. A side you don't mention defaults to zero gradient, which removes
the *diffusive* flux but is **not** a wall — if the velocity points through it, advection still
carries material out. And pure advection with `D = 0` needs real concentration data at the inflow,
so a zero-gradient inlet is rejected rather than silently treated as a wall.

Sides can be named `"left"`, `"right"`, `"bottom"`, `"top"` or given as `bt.Boundary.Left` and
friends. In 2D, fixed values that meet at a corner have to agree; contradictory ones are an error
rather than a coin flip on side ordering.

Before a long run, ask the problem what it thinks it is:

```python
print(problem.describe())
print(f"largest stable step: {problem.stable_time_step():.4g}")
```

```text
Transport problem
============================================================
  mesh          100 cells on [0, 0.0001] (101 nodes)
  diffusion     D = 2e-09 (uniform)
  reaction      Michaelis-Menten uptake, Vmax = 0.001, Km = 0.001
  boundaries    left: held at 0.05, right: sealed (no diffusive flux)
largest stable step: 0.0002499
```

That has caught more of my own mistakes than any error message — usually a boundary I meant to set
and didn't.

---

## When it refuses

The library would rather stop than solve a different problem than the one you described. If you ask
for a time step past the stability limit, it tells you which process is responsible:

```text
ValueError: time_step exceeds the certified explicit stability limit

  you asked for dt = 1
  the certified stable limit here is dt = 0.0002499

  what each process allows:
    diffusion  dx^2 / (2 D)                 = 0.00025  <-- the binding one
    reaction   1 / max|dR/dc|               = 1

  your options:
    - leave time_step out entirely and let the solver choose a stable step
    - use a coarser mesh: the diffusion limit scales as dx^2, so halving the cell count buys you 4x the step
    - use an implicit solver, which has no step limit: CrankNicolsonDiffusion (1D), ADIDiffusion2D or ImplicitDiffusion2D (2D)
    - if you only want the final steady answer, skip the transient with bt.solve_steady(problem)
```

It also refuses an unverified `method=`, the central/hybrid/QUICK advection schemes in the canonical
solver, automatic stepping for a custom reaction with no declared derivative bound, non-finite
values, and contradictory corner conditions. `method="conservative"`, `"explicit"` and
`"explicit_euler"` all name the same verified scheme.

For a custom reaction, declaring the bound is what keeps automatic stepping working:

```python
problem.reaction(lambda c, x, y, t: -0.4 * c * c, max_abs_dc=0.8)
```

---

## What's verified, and what isn't

This matters more than usual here, because the library covers a lot of ground and not all of it
carries the same evidence.

**The `Problem` / `solve` path is the verified one.** One scalar field on 1D and 2D Cartesian grids,
node-centred finite-volume balances, harmonic face averaging for variable diffusivity, conservative
first-order upwind advection, explicit Euler in time with enforced stability limits. The automated
tests cover conservation for closed variable-coefficient problems, manufactured steady solutions,
second-order spatial convergence for smooth diffusion, first-order convergence for upwind advection
and for reaction in time, landing exactly on the requested end time, and deterministic corner
handling.

**That contract does not extend by itself** to 3D, cylindrical grids, implicit or higher-order
integration, other advection schemes, coupled multi-species systems, fluid dynamics, electrochemical
transport, or the application-scale models. Several of those have their own focused tests — Navier–
Stokes velocity, bioheat in space and time, the Nernst–Planck diffusion limit, cylindrical
operators, the nonuniform 1D solver — and you can read exactly what each one claims:

```python
from biotransport.contracts import get_contract

contract = get_contract("NernstPlanckSolver")
print(contract.equation)
print(contract.evidence_level.value)
print(contract.exclusions)
```

**`solve_steady` is a different method with different evidence.** Newton's method on the steady
operator, with analytic Jacobians for the built-in kinetics. It doesn't inherit the transient
solver's verification, and it has no advection.

**Some of the Python modules are real implementations, not wrappers.** The units, provenance,
sensitivity and reproducibility layers are Python, as are a few older numerical helpers. Those need
their own evidence, and `biotransport.contracts` records which backend each one uses.

Finally, and most importantly: none of this is biological validation. Verified numerics mean the
code solves the equation you wrote. Whether that equation describes your tissue is a separate
question, and the bundled parameter values are illustrative rather than sourced —
`biotransport.provenance` exists so you can record where yours came from.

Known gaps, stated plainly: `solve_steady` implements slab geometry only, so a curved mesh has to be
marched through its transient rather than solved directly — it refuses rather than returning a slab
answer. It also covers diffusion and reaction but not advection, and in 2D it needs fixed-value
boundaries on every side. Boundary fluxes are exact in 1D and second-order accurate in 2D, where
corner nodes are shared between two walls. Spherical geometry is 1D only; a 2D `(r, θ)` operator is
not implemented.

---

## Cylinders and spheres

Biotransport is full of curved geometry — the Krogh tissue cylinder, a spherical tumour spheroid, a
drug-loaded microsphere, a cell taking up solute. Those are not slabs, and solving them as slabs
gives the wrong answer for reasons a student cannot easily see.

So the geometry is a property of the mesh:

```python
cell    = bt.mesh_1d(200, 0.0, 50e-6, "spherical")     # a 50 um cell
capillary = bt.mesh_1d(200, 0.0, 25e-6, "cylindrical") # a Krogh cylinder
shell   = bt.mesh_1d(200, 0.5, 1.0, "cylindrical")     # an annulus, not reaching the axis
```

Everything else is unchanged — same `Problem`, same `solve`, same `Solution`:

```python
D, R = 1e-9, 1e-3
mesh = bt.mesh_1d(200, 0.0, R, "spherical")

problem = (
    bt.Problem(mesh)
    .diffusivity(D)
    .initial(1.0)
    .dirichlet("right", 0.0)     # surface washed clean at t = 0
)

sol = bt.solve(problem, end_time=100.0)
print(sol.compare(lambda r, t: bt.analytical.sphere(
    r, t, D=D, R=R, c_surface=0.0, c_initial=1.0)))
```

```text
Compared 201 nodes against the reference field.
  largest absolute error  1.32213e-05  (at node 0)
  RMS error               8.9753e-06
  L2 error                5.23808e-06
  as a fraction of the reference range: 0.00187% peak, 0.000741% L2
```

Two details worth knowing. **You never write a symmetry condition at the centre.** The face area at
`r = 0` is zero, so no flux can cross it — symmetry falls out of the geometry rather than out of a
boundary condition you have to remember. And **integrals are shell measures, not widths**, so
`sol.total()` on a sphere gives you `∫c r² dr` and weights the outside more heavily than the centre,
as it should.

The discretization is verified the same way as everything else: against Crank's spherical series and
the Bessel cylinder series, converging at second order (measured 2.00 for both). Only the canonical
`Problem`/`solve` path understands a curved mesh; every other solver refuses one rather than quietly
returning a slab answer.

### Two dimensions, around an axis

A 2D mesh can be axisymmetric — an `(r, z)` slice through anything with rotational symmetry. A
capillary with axial flow, a cylindrical scaffold, a hollow fibre:

```python
vessel = bt.mesh_2d(60, 200, 0.0, 25e-6, 0.0, 500e-6, "axisymmetric")
```

`x` becomes `r` and `y` becomes `z`. Volumes are true annuli, so a cell at larger radius holds
proportionally more, and `rate("right")` integrates over the curved outer wall rather than a flat one.

The axial direction is *exactly* Cartesian here, and that is not an approximation — it falls out of
the algebra, because the axial face area and the control volume share the same radial measure and it
cancels. The suite pins that down: an axisymmetric solve with nothing driving `z` reproduces the 1D
radial answer **bitwise**.

`"spherical"` is 1D only. A 2D spherical mesh would be `(r, θ)`, which is a different operator, so
asking for one is an error rather than a silent approximation.

---

## Beyond the basics

The canonical path is deliberately small. Around it there is quite a lot more, grouped into
namespaces so `dir(bt)` isn't a wall of class names:

| | |
|---|---|
| `bt.diffusion` | Crank–Nicolson, ADI, implicit and sparse diffusion solvers; 3D |
| `bt.flow` | Stokes, Navier–Stokes, Darcy, and nine non-Newtonian viscosity models including blood |
| `bt.electrochem` | Nernst–Planck ion transport, GHK, standard ion properties |
| `bt.applications` | Tumour drug delivery, bioheat and cryotherapy, multi-species reaction systems |
| `bt.analysis` | Parameter sweeps, local sensitivity, Latin hypercubes, uncertainty propagation |
| `bt.units` | Dimension-checked conversion |
| `bt.provenance` | Where each parameter came from, and how much you trust it |
| `bt.reproducibility` | Deterministic, fingerprinted run manifests |
| `bt.contracts` | What each solver claims and what it excludes |

### Units

The units layer converts and dimension-checks, so a length can't be passed where a diffusivity
belongs:

```python
from biotransport import units

D = units.diffusivity(1.33e-5, "cm^2/s")
problem.diffusivity(D.require(units.Dimension.DIFFUSIVITY))   # 1.33e-9 m^2/s
```

Raw C++ solvers still take plain numbers — the checking happens on the way in.

---

## C++

The core is usable on its own:

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

    const TransportResult result = solve(problem, SolveOptions::until(0.10));

    std::cout << "time: " << result.time << '\n';
    std::cout << "steps: " << result.diagnostics.steps << '\n';
    std::cout << "mass change: " << result.diagnostics.mass_change << '\n';
}
```

Building it needs CMake 3.16+ and Eigen 3.4 for the sparse backend:

```bash
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Pass `-DBIOTRANSPORT_EIGEN=OFF` only if you deliberately want to go without that backend. Installed,
it exports a CMake target:

```cmake
find_package(biotransport CONFIG REQUIRED)
target_link_libraries(my_model PRIVATE biotransport::biotransport)
```

---

## Project status

Alpha. The canonical path is stable and tested; the surrounding solvers vary in maturity, and the
API may still move.

```bash
python -m pytest python/tests -q                          # Python
ctest --test-dir build -C Release --output-on-failure      # C++
```

Deliberately, no test count is quoted anywhere — it changes weekly and a big number is not evidence
that the module you care about has been checked. Run the suite, then read the verification cases for
the physics you're actually modelling.

Design rationale and near-term boundaries live in
[`docs/notes/SCIENCE_FIRST_ARCHITECTURE.md`](docs/notes/SCIENCE_FIRST_ARCHITECTURE.md). Model
equations and their biological limits are in
[`docs/notes/MODEL_SCOPE_AND_REFERENCES.md`](docs/notes/MODEL_SCOPE_AND_REFERENCES.md). Per-solver
evidence is in [`docs/notes/SOLVER_CONTRACTS.md`](docs/notes/SOLVER_CONTRACTS.md). There are also
workflow notes on [units](docs/notes/UNITS.md),
[parameter provenance](docs/notes/PARAMETER_PROVENANCE.md),
[sensitivity and uncertainty](docs/notes/SENSITIVITY_AND_UNCERTAINTY.md),
[balance accounting](docs/notes/BALANCE_ACCOUNTING.md),
[reproducible artifacts](docs/notes/REPRODUCIBILITY.md) and the
[nonuniform 1D geometry slice](docs/notes/NONUNIFORM_GEOMETRY.md).

## Contributing

Contributions are welcome. A new numerical path should say what equation it solves, its sign and
boundary conventions, which dimensions it supports, its stability policy, and what evidence backs
it. Until that evidence exists, it's better to fail loudly than to accept the configuration.

## License

MIT — see [LICENSE](LICENSE).

Built for transport-phenomena research and teaching at Texas A&M University.
