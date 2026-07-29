API Reference
=============

.. toctree::
   :maxdepth: 2

   core
   solvers
   config
   applications
   utilities


Package Overview
----------------

.. currentmodule:: biotransport

The ``biotransport`` package provides:

- **Core classes**: :class:`StructuredMesh`, :class:`StructuredMesh3D`,
  :class:`CylindricalMesh`, :class:`NonuniformMesh1D`, and
  :class:`TransportProblem`.
- **Numerical contracts**: Physics-specific C++ solvers have native contracts,
  while governed Python adapters/reference/workflow modules have a separate
  backend/disposition registry in :mod:`biotransport.contracts`.
- **Config and provenance**: Validated application configurations plus
  :mod:`biotransport.provenance` records.  Bundled defaults remain
  illustrative unless a project supplies defensible provenance.
- **Units**: :mod:`biotransport.units` converts selected semantic quantities
  before handing plain SI values to native solvers.
- **Analysis**: :mod:`biotransport.analysis` provides scoped sensitivity and
  uncertainty screening for caller-defined scalar quantities of interest.
- **Accounting**: :class:`BalanceLedger` and :func:`reconcile_balances` audit
  caller-supplied inventories/transfers; they do not couple solvers.
- **Artifacts**: :mod:`biotransport.reproducibility` writes deterministic,
  fingerprinted run manifests.
- **Utilities**: Dimensionless numbers, analytical solutions, mesh helpers,
  visualization, and file helpers.

See :doc:`utilities` for the workflow-module API and
:doc:`../science_contract` for the scientific interpretation boundary.
