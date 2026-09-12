# Changelog

Notable changes, newest first. This project is alpha and follows
[semantic versioning](https://semver.org/) loosely: until 1.0, minor versions may
change the API.

## Unreleased

### Added

- **Cylindrical and spherical geometry**, in the C++ core, on the canonical
  path. `bt.mesh_1d(n, 0, R, "spherical")` (or `"cylindrical"`) and everything
  else is unchanged — same `Problem`, same `solve`, same `Solution`. This is what
  the Krogh tissue cylinder, a tumour spheroid, a microsphere and a cell taking
  up solute actually are, and solving them as slabs was giving the wrong answer.

  The finite-volume balance now carries the face area (`r`, or `r²`) and the
  exact shell measure, so control volumes sum to `R²/2` and `R³/3` rather than to
  a quadrature of them. Two consequences worth knowing: there is no symmetry
  condition to write at the centre, because the area factor vanishes at `r = 0`
  and no flux can cross it; and integrals like `sol.total()` are shell measures,
  so they weight the outside of the domain more heavily than the middle.

  Verified against Crank's spherical series and the Bessel cylinder series at
  **second order — measured 2.000013** in the C++ suite — plus exact conservation
  on a sealed domain, and the curved operator annihilating a constant.

  **2D axisymmetric `(r, z)`** works the same way:
  `bt.mesh_2d(nr, nz, 0, R, 0, H, "axisymmetric")`. That case factorises — the
  radial direction carries the area weights, and the axial direction is
  identical to Cartesian, because the axial face area and the control volume
  share the same radial measure and it cancels. The test suite pins that down:
  an axisymmetric solve with nothing driving `z` reproduces the 1D radial answer
  **bitwise**, once both are given the same time step. (Left to choose, the 2D
  mesh takes a smaller step for its extra axial stability constraint, and
  first-order time integration then separates the two by O(dt) — which is worth
  knowing, because it looks like an operator bug and is not.)

  Spherical stays 1D. A 2D spherical mesh would be `(r, θ)`, a different
  operator, so asking for one is an error rather than a silent approximation.

  Only the canonical path understands a curved mesh. Every other solver
  (`CrankNicolsonDiffusion`, the explicit family, `MultiSpeciesSolver`,
  `NernstPlanckSolver`, and `solve_steady`) refuses one with a message naming the
  alternative, rather than silently returning a slab answer.
- **`Solution`, returned by `solve()`.** Results now carry the mesh they were
  computed on, so they can plot and check themselves. `sol.plot()`,
  `sol.summary()`, `sol.compare(exact)`, `sol.x`, `sol.c`, `sol.total()`,
  `sol.trace(at=...)` and `_repr_html_` for Jupyter. Everything the native result
  exposed — `concentration`, `solution`, `time`, `diagnostics` — still works and
  still means the same thing.
- **Time history.** `solve(..., save_every=)`, `save_at=[...]` or `frames=N`
  records intermediate fields, for any configured problem rather than pure
  diffusion only. `sol.at(t)`, `sol.times`, `sol.history` and `sol.animate()`
  follow from it. A time-dependent custom reaction gets its clock shifted per
  segment, so it sees absolute time rather than restarting at zero.
- **`solve_steady(problem)`**, also spelled `solve(problem, steady=True)`. Solves
  the steady equation with Newton's method instead of marching a transient,
  supplying analytic Jacobians for the built-in kinetics. On a 100-cell tissue
  slab with Michaelis-Menten uptake, two iterations instead of 125,000 explicit
  steps, agreeing to eight decimal places.
- **`Problem`** is now a Python class over the native `TransportProblem`. It
  records how you described the model, which is what lets the library report
  dimensionless numbers and keep the clock straight when saving frames. Adds
  `describe()`, `stable_time_step()`, `sealed(side)` and plain-string boundary
  names (`"left"`, `"right"`, `"bottom"`, `"top"`).
- **Dimensionless numbers, computed from the problem.** `sol.summary()` and
  `sol.dimensionless()` report Fourier, Péclet, grid Péclet and Damköhler with a
  plain-language reading of each. Grid Péclet in particular tells you whether the
  mesh resolves a front or is smearing it.
- **Exact solutions worth checking against.** `biotransport.analytical` gained the
  finite slab (Fourier series), sphere (Crank), cylinder (Bessel), semi-infinite
  `erf`, instantaneous source, the steady `cosh` profile for first-order
  consumption, and the Thiele modulus with effectiveness factors for slab,
  cylinder and sphere. Every function in the module — including the pre-existing
  native ones — now accepts NumPy arrays as well as scalars.
- **Fluxes and a conservation statement.** `sol.flux()` gives the transport flux
  on every interior face, `sol.flux_at(side)` the outward flux through a
  boundary, `sol.rate(side)` that integrated over the boundary's area,
  `sol.uptake()` the volume-integrated reaction, and `sol.balance()` a printable
  account of whether it all adds up. Most transport questions ask for one of
  these rather than for a field.

  These are reconstructed from the returned field using the core's own face
  formulas -- harmonic face diffusivity, upwinding, half control volumes -- not a
  fresh finite difference. Interior faces and 1D boundaries are exact; a 2D
  balance closes at second order, because a corner node owns two walls but only
  one control-volume balance.
- **A tutorial**, at `docs/tutorial.md`, going from install to checking a result
  against its textbook answer.
- **The documentation is now tested.** `test_documentation_runs.py` extracts and
  executes every code block in the readme and the tutorial, sharing a namespace so
  narrative snippets build on each other, and treats a block as expected-to-fail
  only when the document shows its error. It caught a broken snippet on its first
  run. Snippets are also checked against cp1252, because someone will paste them
  into a Windows console.
- `CONTRIBUTING.md`, `CHANGELOG.md` and `CITATION.cff`.

### Changed

- Stability errors now say what went wrong and what to do. They report the step
  you asked for, the certified limit, the indicative limit each process imposes,
  which is smallest, and the available remedies, instead of a single sentence. A
  custom reaction with no declared derivative bound gets its own message, because
  the advice there is the opposite.
- **2D steady solves are usable.** `NonlinearDiffusionSolver` assembled a *dense*
  finite-difference Jacobian in 2D, costing one full residual evaluation per
  unknown per iteration — an 80×80 grid did not finish a single iteration in two
  minutes. It now assembles the analytic five-point Jacobian sparsely and solves
  it directly: the same 80×80 case converges in 0.24 s, and 120×120 in about a
  second. `solve_steady` no longer refuses 2D grids on cost grounds; it warns
  only past about 40,000 unknowns.
- **`import biotransport` is about 3x faster** — roughly 1.09 s down to 0.34 s.
  Matplotlib and SciPy were both imported eagerly at package import; neither is
  loaded now until something actually plots or runs a sparse solve.
- **Initial-condition helpers return NumPy arrays** instead of lists. With lists,
  `2 * bt.gaussian(mesh)` silently doubled the *length* of the field rather than
  its amplitude, and adding two fields concatenated them.
- `bt.plot(sol)` works on a `Solution` with no second argument, since a Solution
  knows its own mesh. The old refusal message said results never carry one.
- `Solution.trace(at=...)` accepts an `(x, y)` pair on a 2D mesh; it was 1D-only.
- `Problem.stable_time_step()` defaults to reporting the certified limit itself.
  It previously accepted a `safety_factor` and silently ignored it.
- The readme was rewritten to lead with what the library does and how to check a
  result, with the scope and evidence boundaries stated once and plainly rather
  than repeated as disclaimers.
- `Problem.robin` names its right-hand side `rhs`. The native binding calls it
  `c`, which collides with the concentration in the equation it appears in; `c`
  still works as a keyword.
- The Sphinx build no longer mocks the compiled extension unconditionally. Mocking
  it made every native class render with no members while the build reported
  success; it is now mocked only when genuinely unimportable.

### Fixed

- **Examples that misled.** Several ran fine while demonstrating the wrong thing,
  and each is now checked by its own printed numbers:
  - `heat_conduction.py` drew a labelled "Steady State" line the run stopped 31.7
    degC short of. It now integrates to a Fourier number of 1 and lands within
    0.0026 degC of it.
  - `advection_diffusion.py` advertised a scheme comparison whose two curves were
    bitwise identical. It is now a grid-refinement study that *demonstrates*
    numerical diffusion: the computed pulse matches a Gaussian widened by
    `D + D_num` to 0.0011 while differing from the physical-`D` Gaussian by
    0.2095.
  - `adaptive_timestepping.py` concluded adaptive stepping was a win while its own
    output showed it costing 19-84x more against a 3-step reference. It now
    compares against a time-converged reference with an honest cost unit, and
    states the conclusion the numbers actually support.
  - `verify_viscoelastic.py` compared library formulas against themselves, so its
    "max error" was exactly zero and could never have failed. Each model is now
    checked against a closed form written out from the spring-and-dashpot
    definition *and* against RK4 of the constitutive ODE.
  - `verify_diffusion.py` ran its semi-infinite check where domain truncation
    dominated, so it could not have detected a solver error. It now runs where
    `4*sqrt(D t)` is 40% of the domain and prints that validity condition.
  - Also repaired: `membrane_diffusion.py` (eight indistinguishable curves),
    `steady_membrane_diffusion.py` (duplicate curve, single-wedge pie),
    `vtk_export_demo.py` (printed "Domain: 2.5 x 0.0", exported a vanished
    pattern), `multi_species_reaction_diffusion.py` (reported only spatial means,
    which never move), `sensitivity_and_uncertainty.py` (screened a quantity
    insensitive to both parameters), and `tumor_drug_delivery.py`.
- **Documentation that would not run.** `UNITS.md` asserted a floating-point
  equality that raised `AssertionError`; `SENSITIVITY_AND_UNCERTAINTY.md` called a
  `run_model()` it never defined. Fourteen more blocks across the notes were
  fragments that failed on copy-paste and are now self-contained, and four files
  used LaTeX delimiters GitHub does not render (14 display and 23 inline spans
  converted).
- `plot_field` wrote axis labels to `fig.axes[0]` instead of the axes it was given,
  so labels landed on the wrong panel of a multi-panel figure.
- `plot_2d_solution` added another colorbar on every call into the same axes,
  shrinking the plot each time. It takes `colorbar=False` now.
- The module docstring in `fields.py` showed `region_box(x_min, x_max, value)`
  positionally; `value` is keyword-only, so the documented example raised
  `TypeError`.
- The Fourier-number reading claimed a problem had "equilibrated" once `Fo > 1`,
  which is wrong for any reaction-limited case. It now describes diffusion only.
- `Solution.summary()` no longer prints a meaningless relative mass change when a
  domain starts essentially empty.
- `getting_started.rst` called `robin(..., rhs=0.0)`, which raised `TypeError`
  against the native argument name.
- `visualization.py` had `from .._core import ...` inside a `TYPE_CHECKING`
  block — one package level too far, so it would have failed had it ever run.
- `examples/advanced/tumor_drug_delivery.py` asked for 50,000 steps while saving
  nothing after step 43,200, computing and discarding 6,800 of them (about 14%).
  The figures it labels as the final state were always the 6 h frame.
- The Sphinx build reported zero warnings while hiding real broken references.
  It is now `nitpicky`, with genuinely unresolvable third-party and pybind11
  names on an explicit ignore list, so what remains is worth fixing. Maths
  notation like `|dR/dc|` in docstrings — including compiled ones that cannot be
  edited from Python — no longer breaks the parser.
