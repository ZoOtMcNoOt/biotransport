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
