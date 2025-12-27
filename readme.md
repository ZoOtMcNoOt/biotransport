# BioTransport Library

A (work-in-progress) high-performance C++ library with Python bindings for modeling biotransport phenomena in biological systems.

## Overview

BioTransport is designed to simulate various transport processes that occur in biological systems, such as diffusion, convection, and reaction-diffusion. It provides a solid foundation for implementing custom biotransport models with a focus on performance and ease of use.

## Features

### Core Infrastructure
- C++ core (C++17) for high performance computations
- Python bindings for easy scripting and visualization
- Structured mesh generation for 1D and 2D domains
- Boundary condition handling (Dirichlet, Neumann)
- Visualization tools for solution analysis

### Solvers
- **Diffusion solvers** — 1D/2D diffusion with uniform or spatially-varying coefficients
- **Reaction-diffusion** — Linear reaction terms with diffusion
- **Advection-diffusion** — Convection-diffusion with velocity fields (upwind scheme)
- **Darcy flow** — Porous media pressure/velocity field computation
- **Membrane diffusion** — 1D steady-state with partition coefficients and hindered transport
- **Multi-physics** — Tumor drug delivery and bioheat cryotherapy solvers

### BMEN 341 Course Utilities
- **Dimensionless numbers** — Reynolds, Schmidt, Péclet, Biot, Fourier, Sherwood, Damköhler
- **Analytical solutions** — Semi-infinite diffusion, Poiseuille/Couette flow, etc.
- **Configuration dataclasses** — `TumorDrugDeliveryConfig`, `BioheatCryotherapyConfig` with documented parameters

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

The project includes several examples demonstrating different biotransport phenomena:

1. **1D Diffusion**:
   ```bash
   ./dev.sh run 1d_diffusion
   ```

2. **2D Drug Diffusion**:
   ```bash
   ./dev.sh run drug_diffusion_2d
   ```

3. **Advection-Diffusion**:
   ```bash
   python examples/intermediate/advection_diffusion.py
   ```

4. **Darcy Flow** (porous media):
   ```bash
   python examples/intermediate/darcy_flow.py
   ```

5. **Membrane Diffusion**:
   ```bash
   python examples/basic/membrane_diffusion.py
   ```

6. **Tumor Drug Delivery** (advanced multi-physics):
   ```bash
   python examples/advanced/tumor_drug_delivery.py
   ```

7. **Bioheat Cryotherapy** (phase change + Arrhenius damage):
   ```bash
   python examples/advanced/bioheat_cryotherapy.py
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

See [docs/PARAMETERS.md](docs/PARAMETERS.md) for complete parameter reference.
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

## Documentation

For deeper dives into the codebase:

- **[docs/FOOTPRINT.md](docs/FOOTPRINT.md)** — Full repository map + public API surface (C++ and Python).
- **[docs/PARAMETERS.md](docs/PARAMETERS.md)** — Multi-physics solver parameter reference with units and ranges.
- **[docs/PAIN_POINTS.md](docs/PAIN_POINTS.md)** — Known friction points and suggested remedies.
- **[docs/ROADMAP.md](docs/ROADMAP.md)** — Iteration history and upcoming priorities.
- **[docs/GAMEPLAN.md](docs/GAMEPLAN.md)** — Phased action plan for tech debt and new features.
- **[docs/BMEN341_BioTransport_Analysis.md](docs/BMEN341_BioTransport_Analysis.md)** — Course alignment analysis (Texas A&M BMEN 341).

### Results directory behavior

Python examples write output to a `results/` folder. The location is chosen by this precedence:

1. **`BIOTRANSPORT_RESULTS_DIR`** environment variable (if set).
2. **`base_dir`** argument passed to `get_result_path(...)`.
3. **Repo-root auto-detection** (walks up from script looking for `pyproject.toml`).
4. **Current working directory** as fallback.

Set the environment variable for a predictable location when running from any CWD.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
