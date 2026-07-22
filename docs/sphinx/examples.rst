Examples
========

The ``examples/`` directory contains runnable examples organized by complexity.
Examples demonstrate APIs and assumptions; they do not validate a biological
model.  Scripts that state numerical acceptance criteria should return a
nonzero process status when those criteria fail.  A plotted example without an
archived configuration/result is not an always-on verification test.


Basic Examples
--------------

1D Diffusion
~~~~~~~~~~~~

Simple diffusion in a 1D domain with Dirichlet boundary conditions.

.. literalinclude:: ../../examples/basic/1d_diffusion.py
   :language: python
   :caption: examples/basic/1d_diffusion.py


Units and conversions
~~~~~~~~~~~~~~~~~~~~~

Explicit conversion and semantic checks before passing plain SI values to a
native solver.

.. literalinclude:: ../../examples/basic/units_and_conversions.py
   :language: python
   :caption: examples/basic/units_and_conversions.py


Intermediate Examples
---------------------

Advection-Diffusion
~~~~~~~~~~~~~~~~~~~

Convection-diffusion with a velocity field using upwind scheme.

See ``examples/intermediate/advection_diffusion.py``.


Advanced Examples
-----------------

Tumor drug delivery
~~~~~~~~~~~~~~~~~~~

Reduced interstitial-pressure, vascular-source, binding, uptake, and transport
demonstration.  It omits full Starling/lymphatic/systemic-PK coupling and is not
a clinical predictor.

See ``examples/advanced/tumor_drug_delivery.py``.


Bioheat cryotherapy
~~~~~~~~~~~~~~~~~~~

Pennes bioheat equation with phase change and Arrhenius tissue damage.

See ``examples/advanced/bioheat_cryotherapy.py``.


Verification and workflow examples
----------------------------------

Grid and time refinement
~~~~~~~~~~~~~~~~~~~~~~~~

``examples/verification/grid_convergence.py`` runs one scoped diffusion
refinement study and evaluates explicit observed-order criteria.  It is a
runnable artifact, not a blanket automated order claim for every solver.

:download:`Download the grid-convergence example
<../../examples/verification/grid_convergence.py>`.


Sensitivity and uncertainty screening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/verification/sensitivity_and_uncertainty.py`` wraps the native
canonical solver in a scalar quantity of interest, then demonstrates a
parameter sweep, local sensitivities, seeded Latin-hypercube propagation, and
standardized-regression screening.  Its declared ranges/distributions are
illustrative assumptions, not inferred biological uncertainty.

:download:`Download the screening example
<../../examples/verification/sensitivity_and_uncertainty.py>` and read
:download:`its scope guide <../notes/SENSITIVITY_AND_UNCERTAINTY.md>`.


Reproducible numerical artifact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/verification/reproducible_artifact.py`` runs a seeded diffusion
study, records convergence and balances, fingerprints the finest field, writes
a deterministic JSON manifest, verifies it on read, and exits nonzero when its
scoped criteria fail.

:download:`Download the artifact example
<../../examples/verification/reproducible_artifact.py>` and read
:download:`the manifest guide <../notes/REPRODUCIBILITY.md>`.


Darcy-flow verification
~~~~~~~~~~~~~~~~~~~~~~~

``examples/verification/verify_darcy.py`` checks uniform linear pressure,
Darcy velocity, outward-gradient signs, a face-aligned two-material flux, a
measured refinement sequence, gauge rejection, and forced nonconvergence.
The discontinuous-material sequence is documented as approximately
first-order; it is not a blanket order claim for smooth or arbitrary media.

:download:`Download the Darcy example <../../examples/verification/verify_darcy.py>`
and read :download:`its verification note <../notes/DARCY_VERIFICATION.md>`.


Note-only API sketches
----------------------

There is not currently a separate runnable example for the nonuniform 1D
solver, balance ledger, or contract registry.  Their reviewed API sketches are
in the :download:`nonuniform geometry <../notes/NONUNIFORM_GEOMETRY.md>`,
:download:`balance accounting <../notes/BALANCE_ACCOUNTING.md>`, and
:download:`solver contracts <../notes/SOLVER_CONTRACTS.md>` guides.
