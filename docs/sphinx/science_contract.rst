Scientific contract
===================

BioTransport records numerical scope and evidence; it does not confer
biological, experimental, clinical, or regulatory validation.  Always inspect
the contract for the exact solver being used.

Governing equation
------------------

The canonical :class:`biotransport.Problem` / :func:`biotransport.solve` path
advances

.. math::

   \frac{\partial c}{\partial t}
   = \nabla\!\cdot(D\nabla c)
     - \nabla\!\cdot(\mathbf{v}c)
     + R(c,\mathbf{x},t).

Thus the physical flux is

.. math::

   \mathbf{J}=-D\nabla c+\mathbf{v}c,
   \qquad \partial_t c=-\nabla\cdot\mathbf{J}+R.

A positive reaction value adds concentration.  Velocity is treated
conservatively, so a spatially varying velocity advances ``-div(v*c)`` rather
than only ``-v dot grad(c)``.

Units at the solver boundary
----------------------------

Native C++ solver APIs accept plain numeric values and do not carry runtime
unit types.  Generic scalar solvers require one mutually consistent unit
system; application and electrochemical contracts specify SI quantities where
their constants and equations require SI.

The optional ``biotransport.units`` Python module provides immutable semantic
quantities and explicit conversion before handoff to a native solver:

.. code-block:: python

   from biotransport import units

   D = units.diffusivity(1.33e-5, "cm^2/s")
   problem.diffusivity(D.require(units.Dimension.DIFFUSIVITY))

The module distinguishes absolute temperature from temperature difference,
molar concentration from mass concentration, three different permeability
meanings, and volumetric from mass-specific perfusion.  It does not wrap native
solver arrays, establish parameter provenance, or decide model applicability.
See :download:`the units guide <../notes/UNITS.md>`.

Boundary data are solver-specific
---------------------------------

For the canonical scalar path, all normal derivatives use the outward unit
normal.  A Neumann value ``g`` means ``dc/dn = g``; it is not a premultiplied
flux.  A Robin condition means ``a*c + b*dc/dn = rhs``.  Essential values that
disagree at a corner are rejected.

Do not transfer that wording blindly to another family.  Nernst--Planck
Neumann data prescribe an outward **total molar flux**.  The fitted nonuniform
1D diffusion solver again accepts ``dc/dn`` but rejects Robin conditions.  Flow
outflow and traction semantics are specific to each flow solver.

Machine-readable solver contracts
----------------------------------

``biotransport.contracts`` is the authoritative public registry for native
solver entry points.  A contract states the equation, unknowns and locations,
input/output units, supported dimensions/terms/boundaries, numerical method,
stability and convergence policy, exact test references, exclusions, and
warnings.

The registry uses these claim-specific evidence levels:

``untested``
   No automated numerical evidence is cited.

``api``
   Export, construction, or interface behavior only.

``behavior``
   A qualitative update or exact discrete-operation regression.

``invariant``
   A conservation law, balance, equilibrium, positivity rule, or projection
   invariant.

``analytical``
   A scoped comparison with an independent analytical, manufactured, or
   closed-form result.

``convergence``
   An observed refinement/order claim for the stated test case.

The strongest label on a solver is not blanket evidence for every
configuration.  Runtime tests require exact registry coverage and current
``path::selector`` references.  See
:download:`the solver-contract guide <../notes/SOLVER_CONTRACTS.md>`.

Current numerical evidence boundary
-----------------------------------

The canonical 1D/2D Cartesian path has always-on tests for manufactured steady
cases, conservative variable-coefficient balances, boundary signs and corners,
reaction-time convergence, exact final time, stability, and invalid inputs.
The runnable ``examples/verification/grid_convergence.py`` performs one scoped
spatial/time refinement study, but that example is not an always-on order
certificate for every canonical configuration.

Specialized solvers have separate evidence.  The fitted
:class:`biotransport.NonuniformDiffusion1D` slice, for example, has automated
smooth stretched-mesh spatial convergence, interface-flux, mass-balance, and
stability tests.  It remains fixed 1D diffusion: no nonuniform 2D/3D,
unstructured mesh, AMR, moving mesh, advection, or reaction is implied.  See
:download:`the nonuniform geometry guide <../notes/NONUNIFORM_GEOMETRY.md>`.

Stokes is a collocated centred-difference/SIMPLE-like solver, not a staggered
MAC method.  Darcy has scoped analytical tests for uniform linear pressure and
velocity plus interface-flux and measured first-order refinement evidence for
a face-aligned discontinuous-conductivity case.  Those cases do not validate
arbitrary heterogeneous tissue flow.  Always use the registry rather than
transferring evidence by similarity of names; see :download:`the Darcy
verification note <../notes/DARCY_VERIFICATION.md>`.

Scientific workflow records
---------------------------

Parameter provenance
~~~~~~~~~~~~~~~~~~~~

``biotransport.provenance`` records a parameter's value, unit, source,
material/population, method, temperature context, validity range, uncertainty,
and status.  Current bundled application defaults are explicitly
``illustrative`` and ``unprovenanced``.  Structural completeness cannot judge a
source or make a value patient-specific.  See
:download:`the provenance guide <../notes/PARAMETER_PROVENANCE.md>`.

Sensitivity and uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``biotransport.analysis`` supplies deterministic sweeps, central local
sensitivities, seeded independent-marginal Latin hypercubes, uncertainty
propagation, and standardized-regression screening for a caller-defined scalar
quantity of interest.  It does not infer distributions, model correlations,
calibrate parameters, establish causality, or include model discrepancy by
default.  See :download:`the screening guide
<../notes/SENSITIVITY_AND_UNCERTAINTY.md>`.

Balance accounting
~~~~~~~~~~~~~~~~~~

:class:`biotransport.BalanceLedger` and
:func:`biotransport.reconcile_balances` audit caller-supplied amount, energy,
and volume inventories and paired transfers.  They do **not** infer fluxes from
fields, integrate source terms, choose a coupling algorithm, or advance PDEs.
Full automatic solver-result ledger coupling remains open.  A closed ledger is
conservation evidence, not convergence or model validation.  See
:download:`the balance guide <../notes/BALANCE_ACCOUNTING.md>`.

Reproducible artifacts
~~~~~~~~~~~~~~~~~~~~~~

``biotransport.reproducibility`` can freeze configurations, record
method/seed/build metadata, attach convergence and balance records, and write a
deterministic fingerprinted JSON manifest.  A manifest is not a digital
signature, durable publication repository, FAIR-compliance certificate, or
physical validation.  See :download:`the reproducibility guide
<../notes/REPRODUCIBILITY.md>`.

Fail-loud policy
----------------

Unsupported discretizations, ambiguous boundary semantics, unstable explicit
steps, non-finite model evaluations, invalid physical parameter domains,
singular systems, and exhausted solver convergence are errors on surfaces that
claim the fail-loud policy.  The library must not silently substitute a
different equation or method.  Legacy and experimental surfaces with weaker
policies remain identified as gaps rather than inheriting this claim.
