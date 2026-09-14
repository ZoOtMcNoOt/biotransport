Building, solving, interpreting
===============================

These are the objects you touch most: the problem builder, the solver entry
points, the result, and the exact solutions you check results against.

Everything here is documented at the path you actually import it from, so
``biotransport.Problem`` rather than ``biotransport.problem.Problem``.

.. currentmodule:: biotransport

Problem
-------

.. autoclass:: Problem
   :members:
   :undoc-members:
   :exclude-members: __init__

Solving
-------

.. autofunction:: solve

.. autofunction:: solve_steady

Coupled domains and species
---------------------------

See :doc:`../coupled_transport` for physical units, membrane conventions,
stoichiometric reactions, sparse numerical access and verified scope.

.. autoclass:: CoupledModel
   :members:

.. autoclass:: ConcentrationSchedule
   :members: at

.. autoclass:: CompiledModel
   :members:
   :exclude-members: __init__

.. autoclass:: CoupledSolution
   :members:
   :exclude-members: __init__

.. autoclass:: CoupledDiagnostics
   :members:

.. autoclass:: ConservationReport
   :members:

Reusable experiments
--------------------

An experiment is a validated JSON-compatible description compiled into
``Problem``. Studio uses the same registry and solver as Python.

.. autoclass:: Experiment
   :members: from_dict, to_dict, build, plan, run

.. autoclass:: ExperimentPlan
   :members:

.. autofunction:: plan_transport

.. autoclass:: TransportPlan
   :members:

.. autoclass:: ComponentRegistry
   :members: register, catalog

.. autoclass:: ComponentDefinition

.. autoclass:: Parameter

.. autoclass:: ExperimentValidationError

.. autofunction:: builtin_registry

Saving fields at chosen times is what ``solve(..., save_at=[...])`` is for. The
older :func:`run_checkpoints` and :class:`CheckpointResult` are documented under
:doc:`solvers`; they predate saved frames and only handle uniform diffusion.

Solution
--------

.. autoclass:: Solution
   :members:
   :undoc-members:

.. autoclass:: ErrorReport
   :members: worst_node

Fluxes and balances
-------------------

Most transport questions ask for a rate rather than a field.
:meth:`Solution.flux`, :meth:`Solution.flux_at`, :meth:`Solution.rate` and
:meth:`Solution.uptake` give you those, and :meth:`Solution.balance` checks that
they add up.

.. autoclass:: FluxReport
   :members:

Exact solutions
---------------

Every function here accepts NumPy arrays as well as scalars, and returns a plain
float when given plain floats. They pair naturally with
:meth:`Solution.compare`.

.. automodule:: biotransport.analytical
   :members: slab, sphere, cylinder, semi_infinite, instantaneous_source,
             steady_slab_first_order, steady_radial_first_order,
             thiele_modulus, effectiveness_factor
   :member-order: bysource
