# BioTransport Library

A high-performance C++ library with Python bindings for modeling biotransport phenomena in biological systems.

## Overview

BioTransport is designed to simulate various transport processes that occur in biological systems, such as diffusion, convection, and reaction-diffusion. It provides a solid foundation for implementing custom biotransport models with a focus on performance and ease of use.

## Features

- C++ core for high performance computations
- Python bindings for easy scripting and visualization
- Structured mesh generation for 1D and 2D domains
- Solvers for diffusion and reaction-diffusion equations
- Boundary condition handling (Dirichlet, Neumann)
- Visualization tools for solution analysis

## Prerequisites

- CMake (>= 3.13)
- C++ compiler with C++14 support
- Python 3.9
- NumPy
- Matplotlib
- Docker (recommended for easy setup)

## Getting Started

### Option 1: Using Docker (Recommended)

This is the easiest way to get started with BioTransport:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/biotransport.git
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
   git clone https://github.com/yourusername/biotransport.git
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
   git clone https://github.com/yourusername/biotransport.git
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

## API Overview

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
from biotransport import DiffusionSolver
import numpy as np

# Create mesh
mesh = StructuredMesh(100, 0.0, 1.0)

# Set up diffusion solver
D = 0.01  # diffusion coefficient
solver = DiffusionSolver(mesh, D)

# Initial condition (Gaussian pulse)
x = np.array([mesh.x(i) for i in range(mesh.nx() + 1)])
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
from biotransport.visualization import plot_1d_solution

# Plot the solution
plot_1d_solution(mesh, solution, 
                 title='Diffusion Result',
                 xlabel='Position',
                 ylabel='Concentration')
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

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.