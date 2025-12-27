BioTransport Documentation
===========================

A high-performance C++ library with Python bindings for modeling biotransport
phenomena in biological systems.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api/index
   examples

Quick Start
-----------

Install the library:

.. code-block:: bash

   pip install -e .

Basic usage:

.. code-block:: python

   from biotransport import DiffusionSolver, DiffusionConfig, StructuredMesh

   # Create mesh
   mesh = StructuredMesh(nx=100, xmin=0.0, xmax=1.0)

   # Configure solver
   config = DiffusionConfig(
       D=1e-9,  # Diffusion coefficient [m²/s]
       dt=0.01,
       t_end=1.0,
   )

   # Run simulation
   solver = DiffusionSolver(mesh, config)
   result = solver.run()


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
