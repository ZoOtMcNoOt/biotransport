Core API
========

.. currentmodule:: biotransport

Problem construction
--------------------

.. autoclass:: TransportProblem
   :members:

This is the native builder. :class:`Problem` is a Python subclass of it that also
records how you described the model, which is what lets the library report
dimensionless groups and keep a correct clock across saved frames — see
:doc:`workflow`. Anything built with :class:`Problem` is accepted anywhere a
``TransportProblem`` is.

:meth:`TransportProblem.boundaries` returns a list of four boundary conditions,
ordered left, right, bottom, top. Only the first two are used on a 1D mesh.

:func:`solve` is documented under :doc:`workflow`, alongside the
:class:`Solution` it returns.

.. autoclass:: TransportResult
   :members:

The native result type. :func:`solve` wraps it in a :class:`Solution`, which
keeps these same attribute names.

.. autoclass:: SolveDiagnostics
   :members:

Meshes and fields
-----------------

.. autoclass:: StructuredMesh
   :members:

.. autoclass:: StructuredMesh3D
   :members:

:meth:`StructuredMesh3D.ijk` returns the three integer grid indices ``[i, j, k]``
for a flat node index.

.. autoclass:: CylindricalMesh
   :members:

.. autofunction:: mesh_1d

.. autofunction:: mesh_2d

.. autoclass:: SpatialField
   :members:

Boundary conditions
-------------------

.. autoclass:: Boundary
   :members:

.. autoclass:: BoundaryCondition
   :members:
