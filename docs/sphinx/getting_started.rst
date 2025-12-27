Getting Started
===============

Installation
------------

Prerequisites
~~~~~~~~~~~~~

- Python >= 3.9
- CMake >= 3.13
- C++ compiler with C++17 support (MSVC, GCC, or Clang)

From Source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/ZoOtMcNoOt/biotransport.git
   cd biotransport
   pip install -e .

Using Docker
~~~~~~~~~~~~

.. code-block:: bash

   docker build -t biotransport:latest .
   docker run -it -v $(pwd):/biotransport biotransport:latest
   ./dev.sh build && ./dev.sh install


Basic Concepts
--------------

Meshes
~~~~~~

BioTransport uses structured meshes for finite difference computations:

.. code-block:: python

   from biotransport import StructuredMesh

   # 1D mesh: 100 cells from 0 to 1 meter
   mesh_1d = StructuredMesh(nx=100, xmin=0.0, xmax=1.0)

   # 2D mesh: 50x50 cells
   mesh_2d = StructuredMesh(
       nx=50, ny=50,
       xmin=0.0, xmax=1.0,
       ymin=0.0, ymax=1.0
   )


Solvers
~~~~~~~

Each physics module provides a solver class with a configuration dataclass:

.. code-block:: python

   from biotransport import DiffusionSolver, DiffusionConfig

   config = DiffusionConfig(
       D=1e-9,       # Diffusion coefficient [m²/s]
       dt=0.01,      # Time step [s]
       t_end=10.0,   # End time [s]
   )

   solver = DiffusionSolver(mesh, config)
   result = solver.run()


Boundary Conditions
~~~~~~~~~~~~~~~~~~~

Boundary conditions are specified via enums:

.. code-block:: python

   from biotransport import BCType

   config = DiffusionConfig(
       D=1e-9,
       dt=0.01,
       t_end=1.0,
       bc_left=BCType.DIRICHLET,
       bc_right=BCType.NEUMANN,
       bc_left_value=1.0,  # Concentration = 1.0 at left
       bc_right_value=0.0, # Zero flux at right
   )
