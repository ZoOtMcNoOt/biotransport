# Nonlinear solver scope and failure contract

`NewtonRaphsonSolver` and `NonlinearDiffusionSolver` are validated Python
reference solvers for small and moderate nonlinear systems. They are useful for
model development, verification, and steady 1D/2D scalar problems. They are not
a native sparse nonlinear backend and should not be presented as the
high-performance path for large production meshes.

## Newton contract

`NewtonRaphsonSolver` solves a finite system \(F(u)=0\). The initial guess and
every callback result must be finite. The residual must have shape `(n,)`; a
user Jacobian must be a finite dense or sparse matrix with shape `(n, n)`.
Finite differences use a perturbation scaled to each component when no
Jacobian callback is supplied.

The default convergence criterion requires both the residual and Newton
correction norms to meet their configured tolerances. Residual-only and
update-only criteria remain explicit options. Reaching the iteration limit
returns `NewtonResult(converged=False, ...)`, with the residual history and last
update available for inspection.

The update criterion uses the norm of the undamped Newton correction, not the
applied displacement. Consequently, choosing a small damping factor or a short
line-search step cannot by itself manufacture convergence far from a root.
`NewtonResult.update_norm` reports that correction norm, while
`NewtonResult.applied_update_norm` reports the damped displacement separately.
The default `BOTH` criterion never accepts a merely scaled-small residual
without also checking its correction; an exactly zero residual is accepted
without evaluating a potentially singular Jacobian.

These numerical failures raise instead of silently continuing:

- singular or failed direct Jacobian solve;
- an Armijo line search that cannot find sufficient decrease;
- a non-finite initial guess, residual, Jacobian, step, or trial state;
- callback outputs with the wrong shape; and
- invalid tolerances, damping, iteration counts, or line-search settings.

### Least-squares opt-in

A singular Newton system does not generally define a unique Newton direction.
The previous implementation silently substituted `numpy.linalg.lstsq`, which
could make an underdetermined model appear solved. Least squares is now disabled
by default. It can be enabled only with `allow_least_squares=True` in the
constructor or `set_parameters`.

When used, `NewtonResult.used_least_squares` is true,
`NewtonResult.linear_solver` identifies the dense or sparse least-squares path,
and `NewtonResult.least_squares_rank` reports the dense numerical rank when
available. Opting in is appropriate only when a minimum-norm correction matches
the model's mathematical interpretation; it is not a generic singularity fix.
Sparse LSMR condition-limit and iteration-limit exits raise rather than being
reported as usable corrections.

## Steady nonlinear diffusion contract

`NonlinearDiffusionSolver` solves

\[
  -\nabla\!\cdot(D\nabla u) + R(u) = S
\]

on a uniform `StructuredMesh`. Scalar diffusivity must be finite and positive.
In 1D, a finite positive nodal diffusivity vector is also supported. Its face
coefficient is the harmonic mean

\[
  D_{i+1/2}=\frac{2D_iD_{i+1}}{D_i+D_{i+1}},
\]

and the discrete interior operator is the difference of the two face fluxes.
This gives one conservative flux through a material interface. It does not
silently approximate variable-coefficient diffusion as \(-D_i\nabla^2u\).

A boundary condition is required on every domain side. One-dimensional
Neumann values are the outward-normal derivative \(du/dn\): the left formula
therefore has the opposite sign from the positive-x derivative. The boundary
derivative uses a second-order one-sided stencil and requires at least three
nodes. Pure Neumann diffusion without a reaction or another gauge remains
singular and raises by default.

Two-dimensional Dirichlet data must agree at every corner. Conflicting traces
are rejected rather than silently averaged into a value that satisfies neither
boundary; values differing only by a tight, scale-aware floating-point
tolerance are treated as equal. Nodal diffusivity is exposed as a read-only
copy; assign `solver.D = new_value` to rebuild its harmonic face coefficients
safely. The corrected left-Neumann convention is a deliberate change from the
former positive-x interpretation.

Current explicit limitations are:

- variable diffusivity in 2D is rejected;
- 2D Neumann boundaries are rejected;
- only pointwise scalar reaction callbacks are modeled;
- no nonlinear boundary laws, coupled species, constraints, or continuation;
- no unstructured or nonuniform mesh support; and
- the 2D fallback Jacobian is dense finite difference, so this is not a
  large-mesh solver.

Reaction, derivative, source, and initial fields must match the mesh shape and
contain finite values. Unknown boundary types and missing boundary data fail
before Newton iteration begins.

Typed numerical failures are defined in `biotransport.newton_raphson` as
`NewtonSolverError`, `NewtonEvaluationError`, `NewtonLinearSolveError`, and
`NewtonLineSearchError`; they are also exported from the top-level
`biotransport` package.

## Stiff transient example

`examples/intermediate/crank_nicolson_stiff.py` is separate from the steady
Newton API. It runs headlessly through the native C++
`CrankNicolsonDiffusion` binding and raises on failed algebraic convergence,
second-order temporal accuracy for a resolved eigenmode, and bounded
large-step amplification. These checks remain active under `python -O`. The
example deliberately makes no timing or speedup claim.
Crank--Nicolson is A-stable but not L-stable, so stability alone is not an
accuracy, monotonicity, or positivity guarantee.
