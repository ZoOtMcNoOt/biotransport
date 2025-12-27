Solvers
=======

.. currentmodule:: biotransport

Diffusion
---------

.. autoclass:: DiffusionSolver
   :members:
   :undoc-members:


Reaction-Diffusion
------------------

.. autoclass:: ReactionDiffusionSolver
   :members:
   :undoc-members:

.. autoclass:: LinearReactionDiffusionSolver
   :members:
   :undoc-members:

.. autoclass:: LogisticReactionDiffusionSolver
   :members:
   :undoc-members:

.. autoclass:: MichaelisMentenReactionDiffusionSolver
   :members:
   :undoc-members:

.. autoclass:: ConstantSourceReactionDiffusionSolver
   :members:
   :undoc-members:

.. autoclass:: MaskedMichaelisMentenReactionDiffusionSolver
   :members:
   :undoc-members:


Advection-Diffusion
-------------------

.. autoclass:: AdvectionDiffusionSolver
   :members:
   :undoc-members:

.. autoclass:: AdvectionScheme
   :members:
   :undoc-members:


Darcy Flow
----------

.. autoclass:: DarcyFlowSolver
   :members:
   :undoc-members:

.. autoclass:: DarcyFlowResult
   :members:
   :undoc-members:


Membrane Diffusion
------------------

.. autoclass:: MembraneDiffusion1DSolver
   :members:
   :undoc-members:

.. autoclass:: MultiLayerMembraneSolver
   :members:
   :undoc-members:

.. autoclass:: MembraneDiffusionResult
   :members:
   :undoc-members:

.. autofunction:: renkin_hindrance


Fluid Dynamics
--------------

Stokes Flow
~~~~~~~~~~~

.. autoclass:: StokesSolver
   :members:
   :undoc-members:

.. autoclass:: StokesResult
   :members:
   :undoc-members:

Navier-Stokes
~~~~~~~~~~~~~

.. autoclass:: NavierStokesSolver
   :members:
   :undoc-members:

.. autoclass:: NavierStokesResult
   :members:
   :undoc-members:

.. autoclass:: ConvectionScheme
   :members:
   :undoc-members:

Velocity Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: VelocityBCType
   :members:
   :undoc-members:

.. autoclass:: VelocityBC
   :members:
   :undoc-members:


Non-Newtonian Fluid Models
--------------------------

.. autoclass:: ViscosityModel
   :members:
   :undoc-members:

.. autoclass:: NewtonianModel
   :members:
   :undoc-members:

.. autoclass:: PowerLawModel
   :members:
   :undoc-members:

.. autoclass:: CarreauModel
   :members:
   :undoc-members:

.. autoclass:: CarreauYasudaModel
   :members:
   :undoc-members:

.. autoclass:: CrossModel
   :members:
   :undoc-members:

.. autoclass:: BinghamModel
   :members:
   :undoc-members:

.. autoclass:: HerschelBulkleyModel
   :members:
   :undoc-members:

.. autoclass:: CassonModel
   :members:
   :undoc-members:


Blood Rheology Utilities
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: blood_casson_model

.. autofunction:: blood_carreau_model

.. autofunction:: pipe_wall_shear_rate


Multi-Physics Solvers
---------------------

.. autoclass:: BioheatCryotherapySolver
   :members:
   :undoc-members:

.. autoclass:: BioheatSaved
   :members:
   :undoc-members:

.. autoclass:: TumorDrugDeliverySolver
   :members:
   :undoc-members:

.. autoclass:: TumorDrugDeliverySaved
   :members:
   :undoc-members:

.. autoclass:: GrayScottSolver
   :members:
   :undoc-members:

.. autoclass:: GrayScottRunResult
   :members:
   :undoc-members:
