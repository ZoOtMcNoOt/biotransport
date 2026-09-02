Core API
========

.. currentmodule:: biotransport

Problem construction
--------------------

.. autoclass:: TransportProblem
   :members:

``Problem`` is the user-facing alias for ``TransportProblem``.

.. autofunction:: solve

.. autoclass:: biotransport.results.Result
   :members:

.. autoclass:: biotransport.results.Snapshots
   :members:

.. autoclass:: TransportResult
   :members:

.. autoclass:: SolveDiagnostics
   :members:

Meshes and fields
-----------------

.. autoclass:: StructuredMesh
   :members:

.. autofunction:: mesh_1d

.. autofunction:: mesh_2d

.. autofunction:: mesh_3d

.. autoclass:: SpatialField
   :members:

Boundary conditions
-------------------

.. autoclass:: Boundary
   :members:

.. autoclass:: BoundaryCondition
   :members:
