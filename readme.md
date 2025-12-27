# BioTransport Library

A high-performance C++ library with Python bindings for modeling biotransport phenomena in biological systems.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

BioTransport provides a comprehensive framework for simulating transport processes in biological systems—diffusion, convection, reaction-diffusion, and fluid dynamics. Built with a high-performance C++ core and intuitive Python bindings, it bridges the gap between computational efficiency and ease of use.

## Features

### Core Infrastructure
- **C++ core (C++17)** for high-performance computations
- **Python bindings** via pybind11 for easy scripting and visualization
- **Structured meshes** — 1D, 2D Cartesian, and cylindrical coordinate systems
- **Boundary conditions** — Dirichlet, Neumann, and Robin types
- **Visualization tools** — Built-in plotting with `plot_field()` for 1D/2D solutions

### Mass Transport Solvers
- **Diffusion** — 1D/2D with uniform or spatially-varying coefficients
- **Reaction-diffusion** — Linear, logistic, Michaelis-Menten kinetics
- **Advection-diffusion** — Convection-diffusion with velocity fields (upwind scheme)
- **Membrane diffusion** — Steady-state with partition coefficients and hindered transport (Renkin)
- **Gray-Scott model** — Pattern formation (Turing patterns)

### Fluid Dynamics Solvers
- **Stokes flow** — Viscous incompressible creeping flow
- **Navier-Stokes** — Incompressible viscous flow with inertial effects
- **Darcy flow** — Porous media pressure/velocity field computation
- **Non-Newtonian models** — Power-law, Carreau, Casson, Cross models for blood rheology

### Multi-Physics Applications
- **Tumor drug delivery** — Coupled pressure-concentration with elevated IFP
- **Bioheat cryotherapy** — Pennes bioheat with phase change and Arrhenius damage

### Educational Utilities (BMEN 341)
- **Dimensionless numbers** — Reynolds, Schmidt, Péclet, Biot, Fourier, Sherwood, Damköhler
- **Analytical solutions** — Semi-infinite diffusion, Poiseuille flow, Taylor-Couette flow
- **Configuration dataclasses** — `TumorDrugDeliveryConfig`, `BioheatCryotherapyConfig` with documented parameters
- **Verification scripts** — Compare numerical results against analytical solutions

## Prerequisites

- CMake (>= 3.13)
- C++ compiler with C++17 support
- Python (>= 3.9)
- NumPy
- Matplotlib
- Docker (recommended for easy setup)

Notes:
- On Windows, you typically need "Visual Studio Build Tools" with the C++ workload so CMake can build the extension module.

## Getting Started

### Option 1: Using Docker (Recommended)

This is the easiest way to get started with BioTransport:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ZoOtMcNoOt/biotransport.git
   cd biotransport
   ```

2. **Build and run the Docker container**:
   ```bash
   docker build -t biotransport:latest .
   docker run -it -v $(pwd):/biotransport biotransport:latest
   ```
   On Windows PowerShell, use:
   ```powershell
   docker build -t biotransport:latest .
   docker run -it -v ${PWD}:/biotransport biotransport:latest
   ```

3. **Build and install the library**:
   ```bash
   # Inside the Docker container
   ./dev.sh build
   ./dev.sh install
   ```

4. **Run tests**:
   ```bash
   ./dev.sh test
   ```

5. **Run examples**:
   ```bash
   ./dev.sh run 1d_diffusion
   ```

### Option 2: Using Docker Compose

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ZoOtMcNoOt/biotransport.git
   cd biotransport
   ```

2. **Start the container**:
   ```bash
   docker-compose up -d
   docker-compose exec biotransport bash
   ```

3. **Build and install the library**:
   ```bash
   # Inside the Docker container
   ./dev.sh build
   ./dev.sh install
   ```

### Option 3: Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ZoOtMcNoOt/biotransport.git
   cd biotransport
   ```

2. **Create and activate a conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate biotransport
   ```

3. **Build and install the library**:
   ```bash
   ./dev.sh build
   ./dev.sh install
   ```

#### Alternative: pip editable install (no conda)

If you already have Python + CMake installed, you can install in editable mode:

```bash
python -m pip install -U pip
python -m pip install -e .
python -m pytest python/tests
```

## Using with an IDE

### CLion Setup with Docker

1. **Install Docker plugin**:
    - Go to File → Settings → Plugins
    - Install the Docker plugin

2. **Configure Docker in CLion**:
    - Go to File → Settings → Build, Execution, Deployment → Docker
    - Add Docker server

3. **Set up Docker toolchain**:
    - Go to File → Settings → Build, Execution, Deployment → Toolchains
    - Add a Docker toolchain
    - Select "biotransport:latest" as the image

4. **Configure CMake**:
    - Go to File → Settings → Build, Execution, Deployment → CMake
    - Add a profile with the Docker toolchain
    - Click Apply and reload the CMake project

## Running Examples

The project includes 19 examples organized by complexity:

### Basic Examples
```bash
python examples/basic/1d_diffusion.py       # Simple 1D diffusion
python examples/basic/heat_conduction.py    # 2D heat equation
```

### Intermediate Examples
```bash
python examples/intermediate/membrane_diffusion.py       # Transient membrane transport
python examples/intermediate/steady_membrane_diffusion.py # Steady-state with partitioning
python examples/intermediate/oxygen_diffusion.py         # O2 consumption in tissue
python examples/intermediate/drug_diffusion_2d.py        # 2D drug release
python examples/intermediate/advection_diffusion.py      # Convection-diffusion
python examples/intermediate/darcy_flow.py               # Porous media flow
python examples/intermediate/stokes_flow.py              # Creeping viscous flow
python examples/intermediate/navier_stokes_flow.py       # Inertial viscous flow
python examples/intermediate/blood_rheology.py           # Non-Newtonian blood models
python examples/intermediate/cylindrical_coordinates.py  # Axisymmetric problems
```

### Advanced Examples
```bash
python examples/advanced/tumor_drug_delivery.py   # Multi-physics drug transport
python examples/advanced/bioheat_cryotherapy.py   # Cryotherapy with tissue damage
python examples/advanced/turing_patterns.py       # Gray-Scott reaction-diffusion
```

### Verification Scripts
```bash
python examples/verification/verify_diffusion.py       # Against analytical solution
python examples/verification/verify_poiseuille.py      # Pipe flow validation
python examples/verification/verify_taylor_couette.py  # Rotating cylinders
python examples/verification/verify_viscoelastic.py    # Non-Newtonian validation
```

### Running All Examples

```bash
python run_examples.py
```

## API Overview

### C++ “Problem + run” façade (least LoC)

If you're using the C++ library directly, the façade API lets you run a conservative explicit solve without manually picking `dt`.

```cpp
#include <biotransport/solvers/explicit_fd.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>

using namespace biotransport;

StructuredMesh mesh(200, 0.0, 1.0);

std::vector<double> initial(mesh.numNodes(), 0.0);

auto problem = DiffusionProblem(mesh)
   .diffusivity(1e-2)
   .initialCondition(initial)
   .dirichlet(Boundary::Left, 0.0)
   .dirichlet(Boundary::Right, 0.0);

auto result = ExplicitFD().run(problem, /*t_end=*/0.1);

// result.solution -> final field
// result.stats.dt, result.stats.steps, result.stats.mass_rel_drift, ...
```

### Creating a Mesh

```python
from biotransport import StructuredMesh

# 1D mesh
mesh_1d = StructuredMesh(100, 0.0, 1.0)  # 100 cells from x=0 to x=1

# 2D mesh
mesh_2d = StructuredMesh(50, 50, -1.0, 1.0, -1.0, 1.0)  # 50x50 cells
```

### Setting Up a Diffusion Simulation

```python
from biotransport import DiffusionSolver, x_nodes
import numpy as np

# Create mesh
mesh = StructuredMesh(100, 0.0, 1.0)

# Set up diffusion solver
D = 0.01  # diffusion coefficient
solver = DiffusionSolver(mesh, D)

# Initial condition (Gaussian pulse)
x = x_nodes(mesh)
initial_condition = np.exp(-100 * (x - 0.5)**2)
solver.set_initial_condition(initial_condition)

# Set boundary conditions
solver.set_dirichlet_boundary(0, 0.0)  # left boundary
solver.set_dirichlet_boundary(1, 0.0)  # right boundary

# Solve
dt = 0.0001  # time step
num_steps = 1000
solver.solve(dt, num_steps)

# Get solution
solution = solver.solution()
```

### Visualizing Results

```python
from biotransport import plot_field

# Plot the solution
plot_field(
   mesh,
   solution,
   title="Diffusion Result",
   xlabel="Position",
   ylabel="Concentration",
)
```
### Configuration Dataclasses

For complex multi-physics simulations, use the configuration dataclasses to manage parameters:

```python
from biotransport import TumorDrugDeliveryConfig, BioheatCryotherapyConfig

# Create config with defaults or custom values
config = TumorDrugDeliveryConfig(
    domain_size=0.01,        # 10mm tissue region
    tumor_radius=0.003,      # 3mm tumor
    D_drug_tumor=1e-11,      # Lower diffusivity in tumor
    IFP_tumor=25.0,          # Elevated interstitial pressure (mmHg)
)

# View all parameters with units and descriptions
print(config.describe())

# Access derived quantities
print(f"IFP in Pascals: {config.IFP_tumor_Pa:.1f}")
print(f"Tumor area fraction: {config.tumor_area_fraction:.1%}")
```

See the `python/biotransport/config/` module for complete parameter definitions with units and ranges.

### Python “golden path” (least LoC)

For problems that support the façade API (`*Problem` + `ExplicitFD`), you can use the
top-level `run(...)` helper:

```python
from biotransport import StructuredMesh, DiffusionProblem, Boundary, run, plot_field
import numpy as np

mesh = StructuredMesh(200, 0.0, 1.0)
initial = np.zeros(mesh.num_nodes(), dtype=np.float64)

problem = (
   DiffusionProblem(mesh)
   .diffusivity(1e-2)
   .initial_condition(initial)
   .dirichlet(Boundary.Left, 0.0)
   .dirichlet(Boundary.Right, 0.0)
)

result = run(problem, t_end=0.1)
plot_field(mesh, result.solution(), title="Diffusion (façade)")
```

## Troubleshooting

### Python Version Mismatch

If you see an error like:
```
ImportError: Python version mismatch: module was compiled for Python 3.12, but the interpreter version is incompatible: 3.9.21
```

Solutions:
1. Make sure your conda environment is active
2. Rebuild the project with the correct Python version:
   ```bash
   rm -rf build/
   ./dev.sh build
   ./dev.sh install
   ```

### Windows build issues

- If compilation fails, confirm you have the MSVC compiler installed (Visual Studio / Build Tools) and that `cmake` can find it.
- Platform-specific compiled artifacts (like `.so`/`.pyd`) should not be committed to the repo.

### Docker Issues

If you encounter issues with Docker:
1. Ensure Docker is running
2. Try rebuilding the Docker image:
   ```bash
   docker build --no-cache -t biotransport:latest .
   ```
3. For CLion integration issues, make sure GDB is installed in your Docker image:
   ```bash
   docker run -it --name gdb-install biotransport:latest bash
   apt-get update && apt-get install -y gdb
   exit
   docker commit gdb-install biotransport:latest
   docker rm gdb-install
   ```

## Project Structure

```
biotransport/
├── cpp/                    # C++ core library
│   ├── include/biotransport/
│   │   ├── core/          # Mesh, numerics, analytical solutions
│   │   ├── physics/       # Fluid dynamics, mass transport, heat transfer
│   │   └── solvers/       # Diffusion, advection-diffusion, explicit FD
│   ├── src/               # Implementation files
│   ├── tests/             # C++ unit tests (Google Test)
│   └── benchmarks/        # Performance benchmarks
├── python/
│   ├── bindings/          # pybind11 bindings
│   ├── biotransport/      # Python package
│   │   ├── config/        # Configuration dataclasses
│   │   ├── visualization.py
│   │   └── ...
│   └── tests/             # Python tests (pytest)
├── examples/
│   ├── basic/             # Introductory examples
│   ├── intermediate/      # Standard physics problems
│   ├── advanced/          # Multi-physics simulations
│   └── verification/      # Validation against analytical solutions
└── results/               # Output directory for simulations
```

### Results Directory Behavior

Python examples write output to a `results/` folder. The location is chosen by this precedence:

1. **`BIOTRANSPORT_RESULTS_DIR`** environment variable (if set)
2. **`base_dir`** argument passed to `get_result_path(...)`
3. **Repo-root auto-detection** (walks up from script looking for `pyproject.toml`)
4. **Current working directory** as fallback

Set the environment variable for a predictable location when running from any CWD.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
