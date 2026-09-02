BioTransport Documentation
===========================

BioTransport is a science-first C++17 numerical library with Python bindings
and scientific-workflow modules for biotransport models.  The primary API
makes the equation, sign convention, boundary semantics, stability policy, and
numerical diagnostics inspectable.  Python also supplies explicit unit
conversion, provenance, sensitivity/uncertainty screening, and reproducibility
tools; the package is not universally a thin wrapper.

BioTransport is alpha research and teaching software.  Numerical evidence is
not biological or clinical validation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   migration
   science_contract
   api/index
   examples

Scientific workflow guides
---------------------------

The detailed Markdown notes are distributed with the source documentation:

* :download:`Solver contracts and evidence registry <../notes/SOLVER_CONTRACTS.md>`
* :download:`Runtime units at the Python boundary <../notes/UNITS.md>`
* :download:`Parameter provenance <../notes/PARAMETER_PROVENANCE.md>`
* :download:`Sensitivity and uncertainty screening <../notes/SENSITIVITY_AND_UNCERTAINTY.md>`
* :download:`Balance accounting <../notes/BALANCE_ACCOUNTING.md>`
* :download:`Reproducible numerical artifacts <../notes/REPRODUCIBILITY.md>`
* :download:`Nonuniform 1D geometry <../notes/NONUNIFORM_GEOMETRY.md>`
* :download:`Darcy-flow verification <../notes/DARCY_VERIFICATION.md>`
* :download:`Performance evidence and reproducibility <../notes/PERFORMANCE_EVIDENCE.md>`
* :download:`Scientific readiness and open gaps <../notes/GAP_ANALYSIS.md>`

Quick Start
-----------

Install the library:

.. code-block:: bash

   pip install -e .

Solve conservative one-dimensional advection--diffusion with decay:

.. code-block:: python

   import numpy as np
   import biotransport as bt

   mesh = bt.mesh_1d(100, x_min=0.0, x_max=1.0)
   x = bt.x_nodes(mesh)
   problem = (
       bt.Problem(mesh)
       .diffusivity(1.0e-2)
       .velocity(0.15)
       .linear_decay(0.20)
       .initial_condition(np.exp(-((x - 0.35) / 0.07) ** 2))
       .dirichlet(bt.Boundary.Left, 0.0)
       .neumann(bt.Boundary.Right, 0.0)
   )
   result = bt.solve(problem, end_time=0.10)

   print(result.time, result.diagnostics.steps)
   print(result.concentration)


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
