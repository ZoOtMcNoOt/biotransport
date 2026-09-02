Getting Started
===============

Installation
------------

Prerequisites
~~~~~~~~~~~~~

- Python >= 3.9
- C++ compiler with C++17 support (MSVC, GCC, or Clang)

The Python build declares CMake, Ninja, pybind11, and Eigen as PEP 517 build
dependencies, so ``pip`` installs them in its isolated build environment. It
does not clone build dependencies during CMake configuration. Direct C++
builds require CMake >= 3.16 and a discoverable Eigen >= 3.4 package unless
``BIOTRANSPORT_EIGEN=OFF`` is selected deliberately.

From Source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/ZoOtMcNoOt/biotransport.git
   cd biotransport
   python -m pip install -e ".[test]"


Basic Concepts
--------------

Meshes
~~~~~~

The canonical :class:`Problem` builder uses uniform Cartesian 1D/2D meshes.
The convenience constructors make the cell count and physical bounds explicit:

.. code-block:: python

   import biotransport as bt

   # 1D mesh: 100 cells from 0 to 1 meter
   line = bt.mesh_1d(100, x_min=0.0, x_max=1.0)

   # 2D mesh: 50 by 40 cells
   rectangle = bt.mesh_2d(
       50, 40,
       x_min=0.0, x_max=1.0,
       y_min=0.0, y_max=0.5,
   )

Separate APIs exist for cylindrical coordinates, uniform Cartesian 3D, and a
fixed fitted nonuniform 1D finite-volume diffusion slice.  The nonuniform API
does not extend :class:`Problem` to arbitrary geometry:

.. code-block:: python

   mesh = bt.NonuniformMesh1D([0.0, 0.02, 0.08, 0.25, 1.0])
   solver = bt.NonuniformDiffusion1D(
       mesh,
       [1.0e-9, 1.0e-9, 5.0e-10, 2.0e-10, 2.0e-10],
   )
   solver.set_initial_condition([1.0, 0.5, 0.1, 0.0, 0.0])
   solver.dirichlet(bt.Boundary.Left, 1.0).neumann(bt.Boundary.Right, 0.0)
   result = solver.solve_until(3600.0)   # step chosen from the solver's own bound

This solver is diffusion-only and fixed 1D.  It does not provide unstructured
meshes, AMR, moving meshes, nonuniform 2D/3D, advection, or reaction.  Read
:download:`its contract <../notes/NONUNIFORM_GEOMETRY.md>` before use.


Units
~~~~~

Native solvers accept plain numbers.  Prefer SI at the solver boundary and use
``biotransport.units`` when an input needs explicit conversion or semantic
dimension checking:

.. code-block:: python

   from biotransport import units

   D = units.diffusivity(1.33e-5, "cm^2/s")
   D_m2_s = D.require(units.Dimension.DIFFUSIVITY)

   problem = bt.Problem(line).diffusivity(D_m2_s)

``require`` fails if, for example, a length or porous permeability is supplied
where diffusivity is required.  The units layer does not attach unit metadata
to fields inside C++ and does not establish parameter validity.  See
:download:`the units guide <../notes/UNITS.md>`.


Solvers
~~~~~~~

The canonical interface separates the physical problem from solve controls.
The C++ core advances every configured term together:

.. code-block:: python

   import numpy as np
   import biotransport as bt

   mesh = bt.mesh_1d(100, x_min=0.0, x_max=1.0)
   x = bt.x_nodes(mesh)
   problem = (
       bt.Problem(mesh)
       .diffusivity(1.0e-9)              # m²/s
       .initial_condition(np.exp(-((x - 0.5) / 0.08) ** 2))
       .neumann(bt.Boundary.Left, 0.0)   # outward derivative dc/dn
       .neumann(bt.Boundary.Right, 0.0)
   )
   result = bt.solve(problem, end_time=10.0)

``result.time`` is the exact requested final time.  ``result.concentration`` is
an owned NumPy copy of the returned C++ field, and ``result.diagnostics``
exposes stability, mass, and extrema information.  Pass ``save_times=[...]`` to
record the field at intermediate clocks in the same call
(``result.snapshots[t]``), and ``result.plot()`` or ``result.write_vtk(path)``
to look at it.


Stepping solvers share one lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The specialized native solvers (explicit and implicit diffusion, reaction--
diffusion, advection--diffusion, multi-species, Nernst--Planck, the nonuniform
1D slice) are configured object-by-object, but they all advance the same way:

.. code-block:: python

   solver = bt.DiffusionSolver(mesh, 1.0e-9)
   solver.set_initial_condition(bt.gaussian(mesh, center=0.5, width=0.08))
   for side in bt.sides(mesh):
       solver.neumann(side, 0.0)
   result = solver.solve_until(10.0, save_times=[2.0, 5.0])

``solve_until`` returns the same :class:`Result` as :func:`solve`.  It chooses
the time step only when the solver certifies its own stability limit; the
Crank--Nicolson, ADI, implicit and legacy reaction/advection classes require
``time_step=`` and never guess.  The fluent ``dirichlet``, ``neumann``,
``robin``, ``boundary`` and ``outward_flux`` verbs forward to each class's own
setters and refuse conditions the class does not implement.


Finding your way around
~~~~~~~~~~~~~~~~~~~~~~~

``biotransport.__all__`` names the canonical path and a handful of namespaces:
``bt.diffusion``, ``bt.electrochem``, ``bt.flow`` and ``bt.applications`` for
the specialized native solvers, ``bt.balance`` for dimensioned accounting,
``bt.reference`` for the Python reference numerics, and the workflow modules
``units``, ``provenance``, ``analysis``, ``convergence``, ``contracts`` and
``reproducibility``.  Every specialized class is still reachable directly from
the root (``bt.DiffusionSolver``); the namespaces only organize the API.  See
:doc:`migration` if you are upgrading from 0.1.


Choosing a specialized solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do not select a solver from its class name alone.  Query the native contract
registry for its equation, units, boundary meanings, evidence, and exclusions:

.. code-block:: python

   from biotransport.contracts import get_contract

   contract = get_contract("NernstPlanckSolver")
   print(contract.equation)
   print(contract.evidence_level.value)
   print(contract.warnings)

A canonical transport test does not certify a specialized flow,
electrochemical, membrane, or application solver.  See
:download:`the complete registry guide <../notes/SOLVER_CONTRACTS.md>`.


Boundary Conditions
~~~~~~~~~~~~~~~~~~~

Boundary conditions are fluent problem-builder calls:

.. code-block:: python

   problem.dirichlet(bt.Boundary.Left, 1.0)
   problem.neumann(bt.Boundary.Right, 0.0)
   problem.robin(bt.Boundary.Top, a=1.0, b=0.2, rhs=0.0)

Neumann data are outward-normal derivatives, ``dc/dn``.  The corresponding
outward diffusive flux is ``-D * dc/dn``.  Robin data mean
``a*c + b*dc/dn = rhs``.  See :doc:`science_contract` before interpreting a
specialized solver result: electrochemical Neumann data, for example, are
outward total molar fluxes, and the nonuniform 1D solver rejects Robin data.


From a run to an auditable artifact
-----------------------------------

For a quantitative report, the field alone is insufficient.  A defensible
workflow should also record:

* unit conversions and raw solver units;
* parameter source, material/population, method, applicability, and uncertainty
  through ``biotransport.provenance``;
* grid/time evidence and available balance residuals;
* sensitivity or uncertainty screening conditional on declared ranges and
  distributions through ``biotransport.analysis``; and
* a frozen, fingerprinted manifest through
  ``biotransport.reproducibility``.

The :class:`BalanceLedger` API (grouped with its helpers in
:mod:`biotransport.balance`) can reconcile caller-supplied amount, energy, and
volume exchanges, but it does not infer them from solver fields or
couple PDEs automatically.  A closed ledger, sourced parameter manifest, and
reproducible JSON file are useful evidence components; none is biological or
clinical validation.
