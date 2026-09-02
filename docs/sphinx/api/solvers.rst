Solvers
=======

.. currentmodule:: biotransport

Canonical conservative scalar transport
---------------------------------------

Use :class:`Problem` and :func:`solve` for the primary 1D/2D Cartesian path.
It advances diffusion, conservative advection, and reaction together in C++ and
returns stability and balance diagnostics.  See :doc:`../science_contract` for
the exact verified scope.

The classes below are specialized numerical surfaces.  Their benchmarks apply
only to their stated equation, dimensions, coefficients, and boundary
conditions; they do not inherit the canonical solver's evidence automatically.

Friendly native adapters
------------------------

The primary :func:`solve` adapter and result diagnostics are documented on
:doc:`core`.

.. autofunction:: run_checkpoints

.. autoclass:: CheckpointResult
   :members:

``run_checkpoints`` partitions time into native solve segments and returns
per-segment diagnostics.  Automatic or shortened steps can therefore produce a
slightly different valid discrete trajectory than one one-shot solve.  A
requested step that divides every segment preserves the same partition.
This alpha API now returns an immutable mapping-like ``CheckpointResult`` rather
than a mutable ``dict``; indexing and ``dict(result)`` remain supported, while
callers that mutate the returned mapping must migrate to their own copy.

Python reference and legacy time surfaces
-----------------------------------------

These APIs have separate :class:`PythonNumericalContract` records and do not
claim native performance.  Explicit ``integrate(method="euler")`` uses the
canonical native Euler path.  Omitting ``method`` temporarily emits a
``FutureWarning`` and preserves historical RK4 behavior; explicit Heun/RK4
diffusion remains a legacy path.

.. autoclass:: AdaptiveTimeStepper
   :members:

.. autoclass:: AdaptiveTimeStepperConfig
   :members:

.. autofunction:: solve_adaptive

.. autoclass:: RK4Integrator
   :members:

.. autoclass:: HeunIntegrator
   :members:

.. autofunction:: integrate

.. autofunction:: integrate_explicit_runge_kutta

.. autoclass:: HighOrderDiffusionSolver
   :members:

.. autoclass:: NewtonRaphsonSolver
   :members:

.. autoclass:: NonlinearDiffusionSolver
   :members:

.. autofunction:: solve_pulsatile

Inspect :func:`get_python_numerical_contract` for each backend, failure policy,
evidence, and retain/port/deprecate disposition.  Newton iteration exhaustion
returns a result with ``converged=False`` rather than silently reporting
success.

Diffusion
---------

.. autoclass:: DiffusionSolver
   :members:

.. autoclass:: DiffusionSolver3D
   :members:

.. autoclass:: CrankNicolsonDiffusion
   :members:

.. autoclass:: ADIDiffusion2D
   :members:

.. autoclass:: ADIDiffusion3D
   :members:

.. autoclass:: ImplicitDiffusion2D
   :members:

.. autoclass:: ImplicitDiffusion3D
   :members:


Reaction-Diffusion
------------------

.. autoclass:: ReactionDiffusionSolver
   :members:

.. autoclass:: LinearReactionDiffusionSolver
   :members:

.. autoclass:: LogisticReactionDiffusionSolver
   :members:

.. autoclass:: MichaelisMentenReactionDiffusionSolver
   :members:

.. autoclass:: ConstantSourceReactionDiffusionSolver
   :members:

.. autoclass:: MaskedMichaelisMentenReactionDiffusionSolver
   :members:


Advection-Diffusion
-------------------

.. autoclass:: AdvectionDiffusionSolver
   :members:

.. autoclass:: AdvectionScheme
   :members:


Darcy Flow
----------

.. autoclass:: DarcyFlowSolver
   :members:

.. autoclass:: DarcyFlowResult
   :members:


Membrane Diffusion
------------------

.. autoclass:: MembraneDiffusion1DSolver
   :members:

.. autoclass:: MultiLayerMembraneSolver
   :members:

.. autoclass:: MembraneDiffusionResult
   :members:

The Renkin helper and its pore-model limitations are documented in
:doc:`applications`.


Electrochemical transport
-------------------------

.. autoclass:: IonSpecies
   :members:

.. autoclass:: NernstPlanckSolver
   :members:

.. autoclass:: MultiIonSolver
   :members:


Multi-species reaction--diffusion
---------------------------------

.. autoclass:: MultiSpeciesSolver
   :members:


Fluid Dynamics
--------------

Stokes Flow
~~~~~~~~~~~

.. autoclass:: StokesSolver
   :members:

.. autoclass:: StokesResult
   :members:

Navier-Stokes
~~~~~~~~~~~~~

.. autoclass:: NavierStokesSolver
   :members:

.. autoclass:: NavierStokesResult
   :members:

.. autoclass:: ConvectionScheme
   :members:

Velocity Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: VelocityBCType
   :members:

.. autoclass:: VelocityBC
   :members:


Non-Newtonian Fluid Models
--------------------------

.. autoclass:: ViscosityModel
   :members:

.. autoclass:: NewtonianModel
   :members:

.. autoclass:: PowerLawModel
   :members:

.. autoclass:: CarreauModel
   :members:

.. autoclass:: CarreauYasudaModel
   :members:

.. autoclass:: CrossModel
   :members:

.. autoclass:: BinghamModel
   :members:

.. autoclass:: HerschelBulkleyModel
   :members:

.. autoclass:: CassonModel
   :members:


Blood Rheology Utilities
~~~~~~~~~~~~~~~~~~~~~~~~

The blood-model helpers and their evidence ranges are documented in
:doc:`applications`.


Multi-Physics Solvers
---------------------

These are mechanistic application models, not clinical models.  Defaults are
demonstrations, and Arrhenius heat-injury output is not a cryogenic cell-death
law.  Record parameter provenance, calibration domain, uncertainty, grid/time
convergence, and balance residuals for any scientific result.

.. autoclass:: BioheatCryotherapySolver
   :members:

.. autoclass:: BioheatSaved
   :members:

.. autoclass:: TumorDrugDeliverySolver
   :members:

.. autoclass:: TumorDrugDeliverySaved
   :members:

.. autoclass:: GrayScottSolver
   :members:

.. autoclass:: GrayScottRunResult
   :members:
