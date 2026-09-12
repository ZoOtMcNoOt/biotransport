BioTransport
============

BioTransport solves diffusion, advection and reaction problems — the equations
behind oxygen reaching tissue, a drug crossing a membrane, heat moving through
skin during cryotherapy. A C++17 finite-volume core and sparse SciPy integration
support single-domain problems and coupled networks. The API
you touch is Python, and it is built around one question: **how do you know your
answer is right?**

So the library ships the exact solutions to compare against, reports the
dimensionless groups that govern whatever you configured, and refuses to run a
setup it cannot vouch for rather than returning a plausible-looking wrong number.

If you are new here, start with the :doc:`tutorial`. It goes from install to
checking a real problem against its textbook answer.

BioTransport is alpha research and teaching software. Verified numerics mean the
code solves the equation you wrote; whether that equation describes your biology
is a separate question, and this library does not answer it.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   tutorial
   coupled_transport
   science_contract
   api/index
   examples


Sixty seconds in
----------------

.. code-block:: bash

   python -m pip install -e ".[test]"

A pulse of solute spreading in a sealed 1 cm channel:

.. code-block:: python

   import biotransport as bt

   mesh = bt.mesh_1d(200, 0.0, 0.01)

   problem = (
       bt.Problem(mesh)
       .diffusivity(1.0e-9)
       .initial(bt.gaussian(mesh, center=0.005, width=0.0005))
       .sealed("left")
       .sealed("right")
   )

   sol = bt.solve(problem, end_time=60.0)
   sol.plot()
   print(sol.summary())

You never picked a time step — the core chose a provably stable one. ``sol`` kept
hold of its mesh, which is why it can plot itself. And ``summary()`` tells you
what the solver did and what physics governs the result.

Then ask whether it is right:

.. code-block:: python

   import biotransport as bt

   D, L = 1.0e-9, 0.01
   mesh = bt.mesh_1d(400, 0.0, L)
   problem = (
       bt.Problem(mesh)
       .diffusivity(D)
       .initial(0.0)
       .dirichlet("left", 1.0)
       .dirichlet("right", 1.0)
   )
   sol = bt.solve(problem, end_time=2000.0)

   print(sol.compare(lambda x, t: bt.analytical.slab(x, t, D=D, L=L, c_surface=1.0)))

.. code-block:: text

   Compared 401 nodes against the reference field.
     largest absolute error  1.59457e-05  (at node 64)
     RMS error               9.599e-06
     L2 error                9.61099e-06
     as a fraction of the reference range: 0.00164% peak, 0.000986% L2


Working notes
-------------

Longer-form notes ship with the source:

* :download:`Solver contracts and evidence registry <../notes/SOLVER_CONTRACTS.md>`
* :download:`Units at the Python boundary <../notes/UNITS.md>`
* :download:`Parameter provenance <../notes/PARAMETER_PROVENANCE.md>`
* :download:`Sensitivity and uncertainty screening <../notes/SENSITIVITY_AND_UNCERTAINTY.md>`
* :download:`Balance accounting <../notes/BALANCE_ACCOUNTING.md>`
* :download:`Reproducible numerical artifacts <../notes/REPRODUCIBILITY.md>`
* :download:`Nonuniform 1D geometry <../notes/NONUNIFORM_GEOMETRY.md>`
* :download:`Darcy-flow verification <../notes/DARCY_VERIFICATION.md>`
* :download:`Performance evidence <../notes/PERFORMANCE_EVIDENCE.md>`
* :download:`Open gaps and readiness <../notes/GAP_ANALYSIS.md>`


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
