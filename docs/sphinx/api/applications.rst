Application Models
==================

.. currentmodule:: biotransport

These APIs are mechanistic research models backed by compiled C++ kernels.
Their defaults demonstrate use; they are not universal physiological values,
clinical recommendations, or evidence that a model is valid for a particular
tissue, species, device, or patient.

Before using an output in a scientific claim, read the repository's
:download:`scientific parameter guide <../../notes/PARAMETERS.md>` and
:download:`model-scope and reference note <../../notes/MODEL_SCOPE_AND_REFERENCES.md>`.
Record parameter provenance, uncertainty, calibration scope, grid/time
convergence, and the available conservation or balance diagnostics.

Bioheat cryotherapy
-------------------

All solver-facing temperatures are absolute kelvin.  Use
:meth:`BioheatCryotherapyConfig.from_celsius` only when the input names make the
Celsius conversion explicit.  The Arrhenius output is a heat-injury diagnostic,
not a validated cryogenic cell-death law, and probe-mask nodes are fixed
temperatures rather than a conjugate probe model.

.. autoclass:: BioheatCryotherapyConfig
   :members:
   :no-index:

.. autoclass:: BioheatCryotherapySolver
   :members:
   :no-index:

.. autoclass:: BioheatSaved
   :members:
   :no-index:

Tumor drug delivery
-------------------

The tumor mask clamps interstitial pressure.  The model does not solve Starling
filtration or lymphatic drainage, and the fluid source implied by the clamp is
treated as solute-free.  Binding and uptake are irreversible first-order
compartments.  The vascular source requires vessel-wall solute permeability
``P`` [m/s] and perfused vascular surface area density ``S_v`` [1/m].

.. autoclass:: TumorDrugDeliveryConfig
   :members:
   :no-index:

.. autoclass:: TumorDrugDeliverySolver
   :members:
   :no-index:

.. autoclass:: TumorDrugDeliverySaved
   :members:
   :no-index:

Electrochemical transport
-------------------------

The Nernst--Planck solvers advance ideal dilute ions in a prescribed electric
potential.  They do not solve Poisson's equation or enforce
electroneutrality.  A Neumann value is outward **total molar flux**, unlike the
outward concentration derivative used by :class:`MultiSpeciesSolver`.

.. autoclass:: IonSpecies
   :members:
   :no-index:

.. autoclass:: NernstPlanckSolver
   :members:
   :no-index:

.. autoclass:: MultiIonSolver
   :members:
   :no-index:

Membrane transport
------------------

These are steady, one-dimensional, ideal-dilute resistance models.  They omit
external films, transient storage, reactions, active transport, and solvent
drag.  Renkin hindrance is a particular cylindrical-pore correlation, not a
universal correction for biological barriers.

.. autoclass:: MembraneDiffusion1DSolver
   :members:
   :no-index:

.. autoclass:: MultiLayerMembraneSolver
   :members:
   :no-index:

.. autoclass:: MembraneDiffusionResult
   :members:
   :no-index:

.. py:function:: renkin_hindrance(lambda_ratio)
   :no-index:

   Return the Renkin hindrance factor for a spherical solute in a cylindrical
   pore.  ``lambda_ratio`` is solute radius divided by pore radius and must be
   nonnegative; values at or above one return zero.

Multispecies reaction--diffusion
--------------------------------

The generic solver uses forward Euler.  Its reported maximum step is the
diffusion-only CFL ceiling; reactions can impose a smaller state-dependent
positivity limit.  The Gray--Scott specialization is a dimensionless,
single-precision, periodic pattern model, not a biological mechanism by
itself.

.. autoclass:: MultiSpeciesSolver
   :members:
   :no-index:

.. autoclass:: GrayScottSolver
   :members:
   :no-index:

.. autoclass:: GrayScottRunResult
   :members:
   :no-index:

Generalized-Newtonian rheology
------------------------------

These classes are instantaneous constitutive laws.  They do not model
viscoelastic memory, thixotropy, red-cell migration, vessel compliance, or
patient-specific blood without additional validated physics.

.. autoclass:: ViscosityModel
   :members:
   :no-index:

.. autoclass:: NewtonianModel
   :members:
   :no-index:

.. autoclass:: PowerLawModel
   :members:
   :no-index:

.. autoclass:: CarreauModel
   :members:
   :no-index:

.. autoclass:: CarreauYasudaModel
   :members:
   :no-index:

.. autoclass:: CrossModel
   :members:
   :no-index:

.. autoclass:: BinghamModel
   :members:
   :no-index:

.. autoclass:: HerschelBulkleyModel
   :members:
   :no-index:

.. autoclass:: CassonModel
   :members:
   :no-index:

.. py:function:: blood_casson_model(hematocrit)
   :no-index:

   Construct the supported Casson correlation for hematocrit in ``[0, 0.60]``.
   Values outside the source study's 35--55% range are extrapolations.

.. py:function:: blood_carreau_model(hematocrit)
   :no-index:

   Construct the stated educational Carreau surrogate, anchored at 45%
   hematocrit and supported on ``[0, 0.60]``.

.. py:function:: pipe_wall_shear_rate(Q, R)
   :no-index:

   Return the Newtonian nominal wall shear-rate magnitude
   ``4*abs(Q)/(pi*R**3)``.  It is not a non-Newtonian correction.

.. py:function:: apparent_viscosity_pipe(model, Q, R, pressure_gradient)
   :no-index:

   Infer apparent pipe viscosity using the model-based
   Rabinowitsch--Mooney correction.  Flow and pressure-gradient signs must be
   physically opposing.
