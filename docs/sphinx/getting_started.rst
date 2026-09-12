Getting Started
===============

This page gets you installed and solving. For a longer walkthrough that ends with
checking a result against its textbook answer, read the
:doc:`tutorial <tutorial>`.

Installation
------------

You need Python 3.9 or newer and a C++17 compiler, because installing builds the
native core.

.. code-block:: bash

   git clone https://github.com/ZoOtMcNoOt/biotransport.git
   cd biotransport
   python -m pip install -e ".[test]"

On Windows install the Visual Studio Build Tools with the "Desktop development
with C++" workload. On macOS run ``xcode-select --install``. On Linux, gcc.

``pip`` fetches CMake, Ninja, pybind11 and Eigen into its isolated build
environment, so you do not install them yourself. A direct C++ build does need
CMake 3.16+ and Eigen 3.4 on the system, unless you deliberately pass
``-DBIOTRANSPORT_EIGEN=OFF``.


The shape of the API
--------------------

Three objects do almost everything.

A **mesh** says where the answer lives. A :class:`~biotransport.Problem` says what
physics to solve. A :class:`~biotransport.Solution` comes back holding both the
answer and the mesh it was computed on, which is why it can plot and check
itself.

.. code-block:: python

   import biotransport as bt

   mesh = bt.mesh_1d(200, x_min=0.0, x_max=0.01)   # 200 cells, so 201 nodes

   problem = (
       bt.Problem(mesh)
       .diffusivity(1.0e-9)
       .initial(bt.gaussian(mesh, center=0.005, width=0.0005))
       .sealed("left")
       .sealed("right")
   )

   sol = bt.solve(problem, end_time=60.0)
   sol.plot()

Cells and nodes are not the same thing. ``mesh_1d(200, ...)`` makes 200 cells and
201 nodes, because nodes sit on the cell edges including both ends. Field arrays
are node-length.

You did not choose a time step. The core computed the largest provably stable
explicit step for this grid and diffusivity and used 80% of it. Pass
``time_step=`` if you want to, and it will refuse anything unstable rather than
return oscillating nonsense.


Meshes
~~~~~~

.. code-block:: python

   import biotransport as bt

   line = bt.mesh_1d(100, x_min=0.0, x_max=1.0)
   rectangle = bt.mesh_2d(50, 40, x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.5)

Cylindrical coordinates, uniform Cartesian 3D, and a fitted nonuniform 1D
diffusion slice all have their own APIs. They do not extend
:class:`~biotransport.Problem` to arbitrary geometry:

.. code-block:: python

   import biotransport as bt

   mesh = bt.NonuniformMesh1D([0.0, 0.02, 0.08, 0.25, 1.0])
   solver = bt.NonuniformDiffusion1D(
       mesh,
       [1.0e-9, 1.0e-9, 5.0e-10, 2.0e-10, 2.0e-10],
   )
   solver.set_dirichlet_boundary(bt.Boundary.Left, 1.0)
   solver.set_neumann_boundary(bt.Boundary.Right, 0.0)
   solver.set_initial_condition([1.0, 0.5, 0.1, 0.0, 0.0])
   solver.solve_until(3600.0, 0.9 * solver.max_stable_time_step())

That one is diffusion-only and fixed 1D: no unstructured meshes, no AMR, no
moving meshes, no nonuniform 2D or 3D, no advection, no reaction. Read
:download:`its contract <../notes/NONUNIFORM_GEOMETRY.md>` first.


Units
~~~~~

The library has no opinion about units and converts nothing behind your back. It
needs you to be consistent, and that is genuinely easy to get wrong, because
diffusivities are tabulated in cm²/s while domains get measured in µm. Let
:mod:`biotransport.units` check you:

.. code-block:: python

   import biotransport as bt
   from biotransport import units

   line = bt.mesh_1d(100, x_min=0.0, x_max=1.0)

   D = units.diffusivity(1.33e-5, "cm^2/s")
   problem = bt.Problem(line).diffusivity(D.require(units.Dimension.DIFFUSIVITY))

``require`` fails if you hand it a length or a permeability where a diffusivity
belongs. It converts and dimension-checks at the boundary; it does not attach
unit metadata to fields inside C++, and it says nothing about whether a value is
physically appropriate. See :download:`the units guide <../notes/UNITS.md>`.


Boundary conditions
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import biotransport as bt

   problem = bt.Problem(bt.mesh_1d(100, 0.0, 1.0)).diffusivity(1.0e-9)

   problem.dirichlet("left", 1.0)         # hold this face at c = 1
   problem.neumann("right", 0.0)          # fix the outward gradient
   problem.sealed("right")                # the same thing, named for its effect
   problem.robin("right", a=1.0, b=0.2, rhs=0.0)

Sides accept plain strings or :class:`~biotransport.Boundary` values.

``neumann`` takes a **derivative**, not a flux. The outward diffusive flux that
results is ``-D * dc/dn``, so zero means no diffusive flux. For a prescribed flux
``J``, pass ``-J/D``.

Robin data mean ``a*c + b*dc/dn = rhs``, and ``b = 0`` reduces to a fixed value.

One trap worth stating outright: a sealed side blocks **diffusion** only. If a
velocity points through that boundary, advection still carries material across
it. Sides you never mention default to sealed.

Specialized solvers do not all share these meanings — electrochemical Neumann
data are outward total molar fluxes, and the nonuniform 1D solver rejects Robin
data entirely. Read :doc:`science_contract` before interpreting one.


Checking an answer
------------------

This is the part worth learning first. Compare against a known solution in one
call:

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

:mod:`biotransport.analytical` holds the solutions coursework is set on -- finite
slab, sphere, cylinder, semi-infinite, instantaneous source, and the steady
``cosh`` profile for first-order consumption -- and they all accept arrays.

``sol.summary()`` reports what the solver did and the dimensionless groups your
configuration implies, which is usually how you find out that an end time is too
short or a mesh too coarse.


Time history and steady states
------------------------------

Continuing with the slab ``problem`` built in the previous section:

.. code-block:: python

   sol = bt.solve(problem, end_time=2000.0, save_every=200.0)
   sol.plot(times=[0.0, 400.0, 2000.0])
   anim = sol.animate(save="slab.gif")

   steady = bt.solve_steady(problem)   # skip the transient entirely

``save_at=[...]`` picks specific times and ``frames=20`` gives evenly spaced ones.
:func:`~biotransport.solve_steady` solves the steady equation with Newton's method
instead of marching to it, which is usually a few iterations rather than tens of
thousands of steps.


Choosing a specialized solver
-----------------------------

Do not pick a solver from its class name. Ask the registry what it claims:

.. code-block:: python

   from biotransport.contracts import get_contract

   contract = get_contract("NernstPlanckSolver")
   print(contract.equation)
   print(contract.evidence_level.value)
   print(contract.exclusions)

A canonical transport test certifies nothing about a flow, electrochemical,
membrane or application solver. See
:download:`the complete registry guide <../notes/SOLVER_CONTRACTS.md>`.


From a run to something defensible
----------------------------------

A field on its own is not a result. For anything you intend to report, also
record:

* the unit conversions you applied, and the raw units the solver saw;
* where each parameter came from, through :mod:`biotransport.provenance`;
* grid and time refinement evidence, and any available balance residual;
* sensitivity or uncertainty screening over declared ranges, through
  :mod:`biotransport.analysis`; and
* a frozen, fingerprinted manifest, through :mod:`biotransport.reproducibility`.

:class:`~biotransport.BalanceLedger` reconciles amount, energy and volume
exchanges that you supply. It does not infer them from solver fields or couple
PDEs for you.

A closed ledger, a sourced parameter manifest and a reproducible JSON file are
good evidence that your calculation is what you say it is. None of them is
evidence that the model describes biology.
