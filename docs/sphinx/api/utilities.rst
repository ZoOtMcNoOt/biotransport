Scientific workflow and utility API
===================================

The modules on this page support auditable workflows around native solves.
They do not turn numerical evidence into biological or clinical validation.

Units
-----

Native C++ solvers accept plain numeric values.  ``biotransport.units`` is an
optional Python boundary layer for explicit conversion and semantic-dimension
checks before those values are passed to a solver.

.. automodule:: biotransport.units
   :members:
   :exclude-members: UNITS
   :show-inheritance:

``biotransport.units.UNITS`` is the read-only public symbol-to-unit registry;
use :func:`biotransport.units.get_unit` and
:func:`biotransport.units.available_units` for normal discovery.

See :download:`the units guide <../../notes/UNITS.md>`.


Parameter provenance
--------------------

Provenance records are immutable traceability claims.  Structural validation
does not assess source quality, applicability, or biological validity.  Use the
fully qualified ``biotransport.provenance.EvidenceLevel``; the solver-contract
module has a different enum with the same short name.

.. automodule:: biotransport.provenance
   :members:
   :show-inheritance:

See :download:`the provenance guide <../../notes/PARAMETER_PROVENANCE.md>`.


Sensitivity and uncertainty screening
-------------------------------------

The model callback returns one finite scalar quantity of interest.  These
operations describe behavior conditional on caller-declared ranges and
distributions; they are not calibration or validation.

.. automodule:: biotransport.analysis
   :members:
   :show-inheritance:

See :download:`the screening guide
<../../notes/SENSITIVITY_AND_UNCERTAINTY.md>`.


Reproducible artifacts
----------------------

The reproducibility module records frozen input/evidence metadata and writes
deterministic, fingerprinted JSON.  Fingerprints identify content; they do not
authenticate an author or replace durable publication archiving.

.. automodule:: biotransport.reproducibility
   :members:
   :show-inheritance:

.. currentmodule:: biotransport

.. py:function:: native_build_info()

   Return path-free metadata for the loaded native extension, including
   compiler identity/version, effective C++ standard, assertion mode, and
   Eigen/OpenMP build features.

See :download:`the reproducibility guide <../../notes/REPRODUCIBILITY.md>`.


Native and Python numerical contracts
-------------------------------------

The contract module contains separate authoritative inventories for native
solver equations/units and governed Python numerical backends/dispositions.
Its evidence labels are numerical and claim-specific.

.. automodule:: biotransport.contracts
   :members:
   :show-inheritance:

See :download:`the solver-contract guide <../../notes/SOLVER_CONTRACTS.md>`.


Balance accounting
------------------

Ledgers audit caller-supplied inventories and transfers.  They do not infer
fluxes from a solution field, integrate source terms, advance a PDE, or choose
a coupling algorithm.

.. currentmodule:: biotransport

.. autoclass:: BalanceDimension
   :members:

.. autoclass:: BalanceUnit
   :members:

.. autoclass:: BalanceLedger
   :members:

.. autoclass:: BalanceAudit
   :members:

.. autoclass:: BalanceReconciliation
   :members:

.. py:function:: convert_balance_value(value, from_unit, to_unit)

   Convert a finite balance quantity between compatible
   :class:`BalanceUnit` values.  Cross-dimension conversion is rejected.

.. py:function:: reconcile_balances(ledgers, relative_transfer_tolerance=1e-12, absolute_transfer_tolerance_base=0.0)

   Validate paired named transfers and aggregate compatible ledgers by
   dimension.  This reconciles accounting records; it does not couple or
   advance PDE solvers.

See :download:`the balance-accounting guide <../../notes/BALANCE_ACCOUNTING.md>`.


Nonuniform 1D geometry
----------------------

This is a separate fitted, fixed 1D finite-volume diffusion surface.  It does
not add nonuniform 2D/3D, unstructured geometry, moving meshes, or AMR to the
canonical :class:`Problem` builder.

.. autoclass:: NonuniformMesh1D
   :members:

.. autoclass:: NonuniformDiffusion1D
   :members:

.. autoclass:: NonuniformDiffusionDiagnostics
   :members:

See :download:`the nonuniform-geometry guide <../../notes/NONUNIFORM_GEOMETRY.md>`.


Dimensionless numbers
---------------------

``biotransport.dimensionless`` is the C++ utility submodule.  Its functions
include ``reynolds``, ``peclet``, ``schmidt``, ``sherwood``, ``biot``, and
``fourier``.  Characteristic length and material-property conventions remain
part of the model definition; a number is not meaningful without them.


Analytical solutions
--------------------

``biotransport.analytical`` exposes C++ closed forms used as independent
reference equations.  ``diffusion_length(D, t)`` means exactly ``sqrt(D*t)``;
it is not a threshold-defined penetration depth.  Plane Poiseuille and Couette
flow have separate, correctly named functions.


Visualization
-------------

.. currentmodule:: biotransport

.. autofunction:: plot

``plot_1d_solution``, ``plot_2d_solution``, ``plot_2d_surface``, ``plot_field``,
``plot_1d`` and ``plot_2d`` are deprecated spellings of :func:`plot`; each warns
and forwards, and they are removed in 0.4.0.


Mesh utilities
--------------

.. autofunction:: x_nodes

.. autofunction:: y_nodes

.. autofunction:: xy_grid

.. autofunction:: as_1d

.. autofunction:: as_2d

.. autofunction:: sides


File utilities
--------------

.. autofunction:: get_results_dir

.. autofunction:: get_result_path


Namespaces and tiers
--------------------

``biotransport.__all__`` names the canonical path only: :class:`Problem`,
:func:`solve`, :class:`Result`, :func:`solve_until`, the meshes, boundary and
field helpers, :func:`plot` and the VTK writers, plus the namespaces
``diffusion``, ``electrochem``, ``flow``, ``applications``, ``balance``,
``reference``, ``stepping``, ``analysis``, ``convergence``, ``contracts``,
``high_order``, ``provenance``, ``reproducibility`` and ``units``.  Every
specialized native class remains an attribute of the root (``bt.DiffusionSolver``
still works and tab-completes); the namespaces are the documented way to find
them.  Retired spellings resolve with a :class:`BioTransportDeprecationWarning`
and are listed in the changelog.

The balance-accounting objects documented above are also grouped in
:mod:`biotransport.balance` together with :func:`balance_residual`.

The deprecated ``run``, ``run_checkpoints`` and ``CheckpointResult`` helpers are
documented with their numerical semantics on :doc:`solvers`.
