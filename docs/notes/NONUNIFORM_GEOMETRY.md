# Nonuniform geometry: first conservative slice

The first nonuniform-geometry capability is deliberately narrow: a fitted,
one-dimensional, node-centred finite-volume mesh and a native C++ diffusion
solver. It is intended for layered tissues, membranes, graded resolution near
an interface, and other 1D problems where uniform spacing wastes work or misses
an important length scale.

This is a real nonuniform discretization. It does not resample onto a hidden
uniform grid, and the Python API only calls the C++ implementation.

## Scientific contract

The solver advances

\[
  \frac{\partial c}{\partial t}
    = \frac{\partial}{\partial x}\left(D(x)\frac{\partial c}{\partial x}\right).
\]

`NonuniformMesh1D` accepts nodal coordinates \(x_0,\ldots,x_N\). Every
coordinate must be finite and the sequence must be strictly increasing. Face
spacing and node-centred control-volume width are

\[
  \Delta x_{i+1/2}=x_{i+1}-x_i,
\]

\[
  V_0=\frac{\Delta x_{1/2}}{2},\qquad
  V_i=\frac{\Delta x_{i-1/2}+\Delta x_{i+1/2}}{2},\qquad
  V_N=\frac{\Delta x_{N-1/2}}{2}.
\]

All \(V_i\) are positive and their sum is the domain length. Diffusivity is
supplied at nodes and must be finite and non-negative. A face uses the harmonic
mean

\[
  D_{i+1/2}=\frac{2D_iD_{i+1}}{D_i+D_{i+1}},
\]

with a zero face value if either adjacent value is zero. The single face flux
and conductance are

\[
  J_{i+1/2}=-D_{i+1/2}\frac{c_{i+1}-c_i}{\Delta x_{i+1/2}},\qquad
  K_{i+1/2}=\frac{D_{i+1/2}}{\Delta x_{i+1/2}}.
\]

Using the same \(J_{i+1/2}\) for both neighbouring control volumes makes the
interior update conservative, including across discontinuous material data.

### Boundary signs

Neumann data are the **outward-normal concentration derivative**
\(q=\partial c/\partial n\), not a flux. The physical Fickian outward flux is

\[
  J_{\mathrm{out}}=-Dq.
\]

Consequently, positive `q` adds integrated concentration to the domain at rate
\(Dq\), while negative `q` removes it. At the left boundary the outward normal
points toward decreasing \(x\); users do not manually reverse the sign.

Dirichlet boundary nodes are held exactly at their prescribed non-negative
concentration. Exchange with that reservoir is inferred from the adjacent
conservative face flux and included in the mass-balance diagnostics. Robin
conditions are not implemented in this slice and are rejected explicitly.

### Explicit stability

Forward Euler uses the local conductance/control-volume certificate

\[
  \Delta t \le
  \min_{i\ \mathrm{not\ Dirichlet}}
  \frac{V_i}{K_{i-1/2}+K_{i+1/2}},
\]

where a missing boundary face contributes zero. This is evaluated from the
actual local mesh and material coefficients; it is not a minimum-spacing
heuristic. `maxStableTimeStep()` returns the exact monotonicity limit and
`checkStability(dt)` checks it. A requested step above the limit throws before
changing the state. Non-finite inputs, negative concentrations, and
non-representable updates also fail loudly.

`solveUntil(final_time, maximum_dt)` treats `final_time` as an absolute time,
uses stable equal substeps no larger than `maximum_dt`, and lands exactly on the
requested final time. This explicit method is appropriate when its stability
limit is affordable; highly refined meshes or very large diffusivity contrasts
can make an implicit method preferable.

## Diagnostics

`diagnostics()` reports:

- the current and balance-reference times, accepted step count, and current CFL limit;
- reference and current integrated concentration \(\sum_i c_iV_i\);
- cumulative boundary input and the residual
  `total_mass - reference_mass - cumulative_boundary_input`;
- minimum and maximum concentration; and
- left and right physical outward Fickian fluxes.

In a unit-area interpretation, `total_mass` is mass per cross-sectional area.
If concentration has units \(M/L^3\), `total_mass` has units \(M/L^2\), while
diffusivity has units \(L^2/T\). The library does not attach units at runtime,
so one consistent unit system must be used throughout a simulation.

Calling `resetBalanceReference()` begins a new accounting interval without
changing the physical state. Setting an initial condition resets time and the
step count to zero. Changing a boundary condition starts a new balance interval
so a prescribed Dirichlet jump is not misreported as transported mass.

## API sketch

```cpp
using namespace biotransport;

NonuniformMesh1D mesh({0.0, 0.02, 0.08, 0.25, 1.0});
NonuniformDiffusion1D solver(mesh, {1e-9, 1e-9, 5e-10, 2e-10, 2e-10});
solver.setDirichletBoundary(Boundary::Left, 1.0)
    .setNeumannBoundary(Boundary::Right, 0.0)
    .setInitialCondition({1.0, 0.5, 0.1, 0.0, 0.0});
solver.solveUntil(3600.0, 0.9 * solver.maxStableTimeStep());
const auto report = solver.diagnostics();
```

```python
mesh = bt.NonuniformMesh1D([0.0, 0.02, 0.08, 0.25, 1.0])
solver = bt.NonuniformDiffusion1D(mesh, [1e-9, 1e-9, 5e-10, 2e-10, 2e-10])
solver.set_dirichlet_boundary(bt.Boundary.Left, 1.0)
solver.set_neumann_boundary(bt.Boundary.Right, 0.0)
solver.set_initial_condition([1.0, 0.5, 0.1, 0.0, 0.0])
solver.solve_until(3600.0, 0.9 * solver.max_stable_time_step())
report = solver.diagnostics()
```

Returned coordinate, solution, diffusivity, and flux arrays are owned NumPy
copies. Mutating them cannot corrupt solver state.

## Scope boundary

This capability models a fixed, fitted 1D line only. It does **not** currently
model:

- unstructured meshes or general finite-element/finite-volume connectivity;
- adaptive mesh refinement or coarsening;
- moving meshes, deformation, or remeshing;
- multidimensional nonuniform Cartesian grids;
- cylindrical/spherical metric factors on nonuniform coordinates;
- advection, reaction, nonlinear diffusion, tensor diffusivity, or contact resistance; or
- Robin boundary conditions.

Those require separate discretization, geometry, stability, and verification
contracts. They should not be inferred from the existence of
`NonuniformMesh1D`.
