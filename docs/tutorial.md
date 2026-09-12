# Getting started with BioTransport

This walks you from a fresh install to checking a real transport problem against
its textbook answer. It assumes you know some Python and some NumPy, and that
you've seen a diffusion equation before. It should take about half an hour.

Everything here runs as written. If a snippet doesn't work for you, that's a bug
and worth reporting.

---

## 1. Install, and check it worked

```bash
git clone https://github.com/ZoOtMcNoOt/biotransport.git
cd biotransport
python -m pip install -e ".[test]"
```

The install compiles a C++ extension, so it takes a minute or two and needs a
compiler. Windows: Visual Studio Build Tools with "Desktop development with
C++". macOS: `xcode-select --install`. Linux: gcc, which you probably have.

Check it:

```python
import biotransport as bt
print(bt.__version__)
```

---

## 2. The three objects you need

Almost everything you do uses the same three pieces.

**A mesh** is where the answer lives. `bt.mesh_1d(200, 0.0, 0.01)` divides a
1 cm line into 200 cells. Cells and nodes are not the same thing: 200 cells have
201 nodes, because nodes sit at the cell edges including both ends. Field arrays
are node-length, so they'll have 201 entries. This trips up everyone once.

**A `Problem`** is the physics. You build it by chaining small statements, each
naming one thing about the model.

**A `Solution`** is what comes back. It holds the answer *and* the mesh it was
computed on, which is why it can plot and check itself.

```python
import biotransport as bt

mesh = bt.mesh_1d(200, 0.0, 0.01)

problem = (
    bt.Problem(mesh)
    .diffusivity(1e-9)
    .initial(bt.gaussian(mesh, center=0.005, width=0.0005))
    .sealed("left")
    .sealed("right")
)

sol = bt.solve(problem, end_time=60.0)
sol.plot()
```

### About units

The library has no opinion about units. It never converts anything behind your
back, and it never assumes SI. It just needs you to be consistent: if lengths
are in metres and time in seconds, then a diffusivity is in m²/s and you're
fine.

Being consistent is harder than it sounds, because diffusivities are usually
tabulated in cm²/s while domains get measured in µm. Two options. Convert by
hand and write down what you did, or let `bt.units` check you:

```python
from biotransport import units

D = units.diffusivity(1.33e-5, "cm^2/s")
print(D.require(units.Dimension.DIFFUSIVITY))   # 1.33e-09, now in m^2/s
```

`require` fails loudly if the quantity isn't a diffusivity, so you can't
accidentally pass a length where a diffusivity belongs.

---

## 3. Read the problem back before you solve it

The single most useful habit: print what you built.

```python
print(problem.describe())
```

```text
Transport problem
============================================================
  mesh          200 cells on [0, 0.01] (201 nodes)
  diffusion     D = 1e-09 (uniform)
  boundaries    left: sealed (no diffusive flux), right: sealed (no diffusive flux)
```

Most modelling mistakes are visible right there — a boundary you meant to set, a
diffusivity off by a thousand, a reaction that got replaced instead of added.
Catching it here is much cheaper than catching it in a plot.

---

## 4. Boundary conditions, and the one that bites

There are three kinds, and the second one causes most of the confusion.

```python
h, c_infinity = 2e-6, 0.2          # film coefficient, and the bath it exchanges with

problem.dirichlet("left", 1.0)     # hold this face at c = 1
problem.neumann("right", 0.0)      # fix the gradient at this face
problem.sealed("right")            # the same thing, named for what it does
problem.robin("right", h, 1e-9, h * c_infinity)   # exchange with a bath
```

`neumann` takes a **derivative**, not a flux. Passing `0.0` means
∂c/∂n = 0, which removes the diffusive flux — a sealed wall. If you want a
prescribed flux *J*, remember the outward diffusive flux is −D ∂c/∂n, so you
pass `-J/D`. Getting this backwards flips the sign of your answer, and the
result still looks plausible, which is why it's worth being careful.

And a warning worth repeating: a sealed side is only sealed against
**diffusion**. If you've set a velocity that points through it, advection still
carries material out. A genuinely closed domain needs no velocity through the
boundary.

Sides you don't mention default to sealed. That's usually what you want, but it's
a default rather than a decision, so `describe()` lists them explicitly.

---

## 5. Time steps, and why you usually shouldn't pick one

Explicit time stepping is only stable below a step size that depends on your grid
and your coefficients. For diffusion that limit is roughly Δx²/2D, which gets
brutal as you refine: halve the cell size and the allowed step drops by four.

So don't pick one. If you leave `time_step` out, the core computes the certified
limit and uses 80% of it:

```python
sol = bt.solve(problem, end_time=60.0)
print(f"took {sol.steps} steps of {sol.diagnostics.maximum_time_step:.4g}")
```

You can ask ahead of time what the limit is:

```python
print(problem.stable_time_step())
```

And if you do pass a step that's too big, the library tells you what's wrong
rather than producing oscillating nonsense:

```python
bt.solve(problem, end_time=60.0, time_step=10.0)
```

```text
ValueError: time_step exceeds the certified explicit stability limit

  you asked for dt = 10
  the certified stable limit here is dt = 1.25

  what each process allows:
    diffusion  dx^2 / (2 D)                 = 1.25  <-- the binding one
  ...
```

This is deliberate. A solver that silently accepts an unstable step is worse than
one that stops, because you can't tell from the output which happened.

---

## 6. Now check whether the answer is right

Here's the part that matters. A number on a screen isn't a result until you have
some reason to trust it.

Take a problem with a known answer: a slab of thickness *L*, starting empty, with
both faces suddenly raised to a fixed concentration. That's a standard
separation-of-variables problem, and `bt.analytical.slab` is the series solution.

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

report = sol.compare(
    lambda x, t: bt.analytical.slab(x, t, D=D, L=L, c_surface=1.0),
    plot=True,
)
print(report)
```

```text
Compared 401 nodes against the reference field.
  largest absolute error  1.59457e-05  (at node 64)
  RMS error               9.599e-06
  L2 error                9.61099e-06
  as a fraction of the reference range: 0.00164% peak, 0.000986% L2
```

About 16 parts per million. Quote the **L2** number in a convergence study, not
the RMS: it's weighted by control volume, so it means the same thing when you
change the grid.

### Watching the error shrink

If the code is right, refining the grid should reduce the error at a predictable
rate — second order in space here, so halving Δx should quarter the error:

```python
for cells in (50, 100, 200, 400):
    mesh = bt.mesh_1d(cells, 0.0, L)
    problem = (
        bt.Problem(mesh).diffusivity(D).initial(0.0)
        .dirichlet("left", 1.0).dirichlet("right", 1.0)
    )
    sol = bt.solve(problem, end_time=2000.0, time_step=0.05)
    error = sol.compare(
        lambda x, t: bt.analytical.slab(x, t, D=D, L=L, c_surface=1.0)
    )
    print(f"{cells:4d} cells   L2 = {error.l2:.3e}")
```

Watch the ratio between successive lines. Approaching 4 means second order, and
that's a much stronger statement about correctness than any single error value —
it says the scheme is converging to the right answer, not just landing near it.

Note the fixed `time_step`: without it each run picks its own step and you'd be
measuring space and time error mixed together.

### What else is in there

`bt.analytical` covers the solutions coursework actually uses:

| Function | Problem |
|---|---|
| `slab` | Finite slab, both faces fixed — Fourier series |
| `sphere` | Sphere with fixed surface — Crank's series (spheroid, microsphere) |
| `cylinder` | Long cylinder, fixed surface — Bessel series (Krogh cylinder) |
| `semi_infinite` | Half-space with a fixed surface — the `erf` solution |
| `instantaneous_source` | A spike released at one instant — the Gaussian |
| `steady_slab_first_order` | Steady diffusion with first-order consumption — `cosh` |
| `thiele_modulus`, `effectiveness_factor` | Reaction-diffusion competition in a catalyst or tissue |

Plus the flow and viscoelastic solutions from the C++ core (Poiseuille, Couette,
Taylor–Couette, Maxwell, Kelvin–Voigt and friends). All of them take arrays.

The series solutions are truncated sums, and they need many terms at very short
times. They'll warn you when the time you asked for is too early for the number
of terms, rather than silently returning something rippled.

---

## 7. Understanding *why* it looks like that

Getting the right numbers is half of it. The other half is knowing which physical
competition set the shape, and that's what dimensionless groups are for.
`summary()` works them out from what you configured:

```python
mesh = bt.mesh_1d(100, 0.0, 1.0)
problem = (
    bt.Problem(mesh)
    .diffusivity(1e-2)
    .velocity(0.15)
    .linear_decay(0.20)
    .initial(bt.gaussian(mesh, center=0.35, width=0.07))
    .dirichlet("left", 0.0)
    .sealed("right")
)
sol = bt.solve(problem, end_time=0.10)
print(sol.summary())
```

The interesting part of the output:

```text
Dimensionless numbers for this problem
------------------------------------------------------------
  Fourier       0.001       diffusion has barely started; the profile is still close to t = 0
  sqrt(D t) / L 0.03162     diffusion has spread about 0.0316 into a domain of 1
  Peclet        15          advection dominates; expect a sharp travelling front
  grid Peclet   0.15        the grid resolves the front (<= 2 is comfortable)
  Damkohler     20          reaction outpaces diffusion; solute reaches only about 22% of the way across
  sqrt(D/k) / L 0.2236      reaction consumes the solute within about 0.224 of the source
```

Read that as a sentence about the physics. Diffusion has barely moved (Fo ≈
0.001), so the pulse is still a pulse. Advection is the dominant transport
(Pe = 15), so it has travelled rather than spread. Reaction is fast (Da = 20), so
it's also shrinking, and material won't get more than about a fifth of the way
across before being consumed.

**Grid Péclet is the one to watch.** It's *v*Δx/*D* — advection versus diffusion
across a single cell, not across the domain. Above about 2, first-order upwinding
adds numerical diffusion comparable to the physical kind, and your front comes
out smeared. If you see a large grid Péclet, refine the mesh; the message tells
you roughly by how much.

---

## 8. Seeing the evolution

Ask for snapshots:

```python
sol = bt.solve(problem, end_time=0.10, save_every=0.01)

sol.plot(times=[0.0, 0.02, 0.05, 0.10])     # overlaid, labelled by time
print(sol.times)
print(sol.trace(at=0.5))                     # history at a single point
anim = sol.animate(save="pulse.gif")         # keep the reference!
```

`save_at=[...]` for specific times, `frames=20` for twenty evenly spaced ones.

One subtlety worth knowing. Saving splits the run into segments, and segment
boundaries force step boundaries, so a saved run can follow a slightly different
(equally valid) discrete path than the same run done in one shot. Pass an
explicit `time_step` that divides your save interval and they agree exactly.

---

## 9. When you only want the steady state

A great deal of coursework asks for the settled answer, not the journey. Consider
oxygen entering tissue and being consumed by cells — the classic
reaction-diffusion problem:

```python
import biotransport as bt

L, D = 100e-6, 2e-9        # 100 um of tissue
mesh = bt.mesh_1d(100, 0.0, L)

problem = (
    bt.Problem(mesh)
    .diffusivity(D)
    .michaelis_menten(Vmax=1e-3, Km=1e-3)
    .initial(0.05)
    .dirichlet("left", 0.05)
    .sealed("right")
)

steady = bt.solve_steady(problem)
print(steady.summary())
print(f"oxygen reaching the far side: {steady.c[-1]:.5f}")
```

Two Newton iterations, versus 125,000 explicit steps to march there — and the two
answers agree to eight decimal places. You never wrote a Jacobian; the built-in
kinetics know their own derivatives.

The steady solver covers diffusion and reaction. It will tell you to march the
transient if you have advection, and it warns before starting a 2D solve big
enough to be slow, because it builds a dense Jacobian in 2D.

### The answer is usually a rate

Notice what the question actually asks. "How much oxygen reaches the tissue" is a
flux, not a concentration field. You get those directly:

```python
print(f"flux in at the capillary side: {steady.flux_at('left'):.4g}")
print(f"consumed by the tissue:        {steady.uptake():.4g}")
print(steady.balance())
```

```text
flux in at the capillary side: -9.797e-08
consumed by the tissue:        -9.797e-08

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

Boundary quantities are reported **outward**, so a negative number means material
is entering. Here everything that came in got consumed, which is what steady
state means.

That last line is worth more than it looks. At steady state nothing accumulates,
so the influx and the reaction *must* cancel — and they do, to five parts in ten
billion. If you ever see that number get large, the model is wrong somewhere, and
a flipped sign on a boundary condition is the usual culprit. It is the cheapest
check available and it uses no extra information.

On a transient run there *is* accumulation, so `balance()` checks the integrated
form over your saved frames instead: the time integral of (what came in plus what
reacted) against the change in what is stored.

Other pieces: `sol.flux()` gives the flux on every interior face, and
`sol.rate(side)` integrates a boundary flux over its area, which in 2D is the
total delivery or clearance rate rather than a per-unit-area number.

### Cross-checking with the Thiele modulus

For *first-order* consumption there's a closed form, so you can check the steady
solver the same way you checked the transient one:

```python
D, k, L, cs = 1e-9, 5e-4, 1e-2, 1.0
mesh = bt.mesh_1d(300, 0.0, L)
problem = (
    bt.Problem(mesh).diffusivity(D).linear_decay(k).initial(cs)
    .dirichlet("left", cs).sealed("right")
)
steady = bt.solve_steady(problem)

print(steady.compare(
    lambda x: bt.analytical.steady_slab_first_order(x, D=D, k=k, L=L, c_surface=cs)
))

phi = bt.analytical.thiele_modulus(D=D, k=k, length=L)
print(f"Thiele modulus {phi:.3f}")
print(f"effectiveness  {bt.analytical.effectiveness_factor(phi, 'slab'):.4f}")
```

The effectiveness factor is the useful engineering number: the fraction of the
maximum possible reaction rate you actually get. Well below 1 means the interior
is starved, and making the region thicker buys you almost nothing — which is
exactly why tissue needs capillaries spaced the way it does.

---

## 10. Cylinders and spheres

Most of the geometry in biotransport is curved. A capillary feeding a sleeve of
tissue is a cylinder; a tumour spheroid, a microsphere, a cell are spheres.
Modelling them as slabs gets the wrong answer, because the amount of material a
shell holds grows with radius and a slab does not know that.

Geometry belongs to the mesh, and nothing else changes:

```python
import biotransport as bt

D, R = 1e-9, 1e-3
mesh = bt.mesh_1d(200, 0.0, R, "spherical")

problem = (
    bt.Problem(mesh)
    .diffusivity(D)
    .initial(1.0)                 # uniformly loaded
    .dirichlet("right", 0.0)      # surface washed clean at t = 0
)

sol = bt.solve(problem, end_time=100.0)
print(sol.compare(lambda r, t: bt.analytical.sphere(
    r, t, D=D, R=R, c_surface=0.0, c_initial=1.0)))
sol.plot(xlabel="radius (m)")
```

Note what you did *not* write: a symmetry condition at the centre. The face area
at `r = 0` is `0` for a cylinder and `0` for a sphere, so no flux can cross the
centre no matter what. Symmetry comes from the geometry rather than from a
boundary condition you have to remember — and `sol.flux_at("left")` returns
exactly zero, not approximately zero.

The other thing that changes quietly is integration. `sol.total()` on a sphere
computes ∫c r² dr, so a uniform field of 1 integrates to `R³/3`, not `R`. That
is what makes conservation and the flux balance mean the right thing.

`"cylindrical"` behaves the same way with an area factor of `r`, and you can
start a mesh away from the axis for an annulus:

```python
shell = bt.mesh_1d(200, 0.5e-3, 1e-3, "cylindrical")   # a hollow tube
```

A 2D mesh can be axisymmetric too — an `(r, z)` slice through anything with
rotational symmetry, which is what a capillary with axial flow or a cylindrical
scaffold actually is:

```python
vessel = bt.mesh_2d(60, 200, 0.0, 25e-6, 0.0, 500e-6, "axisymmetric")
```

`x` becomes `r`, `y` becomes `z`. The axial direction behaves exactly as it does
on a Cartesian mesh — that is not a simplification, it is what the algebra gives,
because the axial face area and the control volume share the same radial factor
and it cancels.

**One limitation to know.** `bt.solve_steady` implements slab geometry only. On a
curved mesh it refuses rather than handing back a slab answer, and tells you to
march the transient instead — which works, and reaches the same place once the
Fourier number passes about 1.

Only the canonical `Problem`/`solve` path understands curved geometry. The
specialized solvers refuse a curved mesh for the same reason.

---

## 11. Two dimensions

Same objects, one more coordinate:

```python
import biotransport as bt

mesh = bt.mesh_2d(60, 60, 0.0, 1.0, 0.0, 1.0)

problem = (
    bt.Problem(mesh)
    .diffusivity(2e-3)
    .velocity(0.05, 0.0)
    .initial(bt.circle(mesh, center_x=0.25, center_y=0.5, radius=0.1))
    .dirichlet("left", 0.0)
    .sealed("right")
    .sealed("bottom")
    .sealed("top")
)

sol = bt.solve(problem, end_time=4.0, frames=30)

sol.plot(title="drifting and spreading")
sol.plot(kind="surface")
anim = sol.animate(save="blob.gif")
```

2D fields come back shaped `(ny + 1, nx + 1)`, matching `sol.grid` so you can
hand them straight to matplotlib. `sol.concentration` is still the flat version
if you want it.

---

## 12. Where to go next

**More solvers.** The canonical `solve` path is explicit in time, which is
limiting for stiff problems. `bt.diffusion` has Crank–Nicolson, ADI, and implicit
sparse solvers with no step limit. `bt.flow` has Stokes, Navier–Stokes, Darcy and
nine non-Newtonian viscosity models including blood. `bt.electrochem` has
Nernst–Planck ion transport. `bt.applications` has tumour drug delivery and
bioheat/cryotherapy.

**Check what a solver claims** before you rely on it:

```python
from biotransport.contracts import get_contract, list_contracts

print(get_contract("NernstPlanckSolver").equation)
print(get_contract("NernstPlanckSolver").exclusions)
```

These aren't decoration. The library covers a lot of physics and the evidence
behind each piece varies, so it's recorded rather than implied.

**Explore parameter space.** `bt.analysis` has seeded sweeps, local
sensitivities, Latin hypercubes and uncertainty propagation.

**Make a run reproducible.** `bt.reproducibility` writes a fingerprinted
manifest of the configuration and results, and `bt.provenance` records where each
parameter came from and how much you trust it.

**Read the examples.** `examples/basic` through `examples/verification` are
runnable, and the verification ones show grid-convergence studies in full.

---

## Common problems

**"My result is completely flat."** Check the Fourier number in `summary()`. If
it's tiny, nothing has had time to diffuse — increase `end_time`. If it's much
bigger than 1, everything has equilibrated and you've integrated too far.

**"My initial condition is the wrong length."** *n* cells means *n* + 1 nodes.
The error message tells you the number the mesh wants.

**"It says my time step is unstable."** Leave `time_step` out and let it choose.
If you need a big step, use an implicit solver from `bt.diffusion`.

**"My front looks smeared."** Look at the grid Péclet number. Above ~2 the
upwind scheme is adding numerical diffusion; refine the mesh.

**"My custom reaction won't run automatically."** The solver sizes its step from
the fastest process and can't differentiate your function. Declare the bound:
`problem.reaction(f, max_abs_dc=...)`, or pass `time_step` yourself.

**"Material is escaping a sealed domain."** A sealed side blocks diffusion, not
advection. If a velocity points through the boundary, material leaves through it.
