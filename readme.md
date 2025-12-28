```
    ____  _       _____                                        _   
   | __ )(_) ___ |_   _| __ __ _ _ __  ___ _ __   ___  _ __ __| |_ 
   |  _ \| |/ _ \  | || '__/ _` | '_ \/ __| '_ \ / _ \| '__/ _` __|
   | |_) | | (_) | | || | | (_| | | | \__ \ |_) | (_) | | | (_| |_ 
   |____/|_|\___/  |_||_|  \__,_|_| |_|___/ .__/ \___/|_|  \__|\__|
                                          |_|                       
```

<div align="center">

**High-performance biotransport simulation library**

*C++ core | Python interface | Educational focus*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-293%20passing-brightgreen.svg)]()

</div>

---

## What is BioTransport?

BioTransport is a computational library for simulating transport phenomena in biological systems. It combines a high-performance C++ numerical core with an intuitive Python interface, making it ideal for both research applications and educational use in biomedical engineering courses.

```
                                    Transport Phenomena
                                           |
              +----------------------------+----------------------------+
              |                            |                            |
        Mass Transport              Fluid Dynamics               Heat Transfer
              |                            |                            |
    +---------+---------+        +---------+---------+                  |
    |         |         |        |         |         |                  |
 Diffusion  Advection  Reaction  Stokes  Navier-   Darcy           Bioheat
    |       Diffusion  Diffusion   |     Stokes    Flow            Equation
    |                              |         |
    +--- Membrane Transport        +--- Non-Newtonian Models
    |                                   (Blood Rheology)
    +--- Nernst-Planck (Ion Transport)
```

---

## Quick Start

```python
import biotransport as bt
import numpy as np

# Create a mesh
mesh = bt.mesh_1d(100, x_min=0, x_max=1)

# Define the problem
problem = (
    bt.Problem(mesh)
    .diffusivity(0.01)
    .initial_condition(bt.gaussian(mesh, center=0.5, width=0.05))
    .dirichlet(bt.Boundary.Left, 0.0)
    .dirichlet(bt.Boundary.Right, 0.0)
)

# Solve and visualize
result = bt.solve(problem, t_end=0.5)
bt.plot(mesh, result.solution(), title="Diffusion of a Gaussian Pulse")
```

---

## Installation

### Option 1: pip (Recommended)

```bash
git clone https://github.com/ZoOtMcNoOt/biotransport.git
cd biotransport
pip install -e .
```

### Option 2: Docker

```bash
docker build -t biotransport:latest .
docker run -it -v $(pwd):/biotransport biotransport:latest
./dev.sh build && ./dev.sh install
```

### Option 3: Conda

```bash
conda env create -f environment.yml
conda activate biotransport
./dev.sh build && ./dev.sh install
```

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | >= 3.9  |
| CMake       | >= 3.13 |
| C++ Compiler| C++17   |
| NumPy       | any     |
| Matplotlib  | any     |

> **Windows Users**: Install Visual Studio Build Tools with the C++ workload.

---

## Architecture

```
+===========================================================================+
|                              PYTHON LAYER                                  |
|   +-------+  +-------+  +-------+  +-------+  +-------+  +-------+        |
|   | solve |  | plot  |  |mesh_1d|  |gaussian|  |adaptive|  | RK4  |       |
|   +-------+  +-------+  +-------+  +-------+  +-------+  +-------+        |
|                              |                                             |
|                         pybind11                                           |
+===========================================================================+
                               |
+===========================================================================+
|                               C++ CORE                                     |
|                                                                            |
|   +-------------------+    +-------------------+    +-------------------+  |
|   |   StructuredMesh  |    |   ExplicitFD      |    |  TransportProblem |  |
|   |   CylindricalMesh |    |   CrankNicolson   |    |  (Fluent Builder) |  |
|   +-------------------+    |   ADI Solvers     |    +-------------------+  |
|                            |   Implicit        |                           |
|   +-------------------+    +-------------------+    +-------------------+  |
|   | Diffusion         |                             | Dimensionless Nos |  |
|   | Advection-Diff    |    +-------------------+    | Analytical Solns  |  |
|   | Reaction-Diff     |    | Stokes Flow       |    | Blood Rheology    |  |
|   | Nernst-Planck     |    | Navier-Stokes     |    +-------------------+  |
|   +-------------------+    | Darcy Flow        |                           |
|                            +-------------------+                           |
+===========================================================================+
```

---

## Module Reference

### Core Simulation

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `Problem` | Fluent problem builder | `.diffusivity()`, `.velocity()`, `.reaction()`, `.dirichlet()`, `.neumann()` |
| `solve` | One-line solver | `solve(problem, t_end)` |
| `run` | Full control solver | `run(problem, t_end, dt, callbacks)` |
| `mesh_1d`, `mesh_2d` | Mesh creation | `mesh_1d(nx, x_min, x_max)` |

### Time Integration

| Method | Order | Stability | Use Case |
|--------|-------|-----------|----------|
| `euler_step` | 1st | Conditional | Fast, simple problems |
| `heun_step` | 2nd | Conditional | Moderate accuracy |
| `rk4_step` | 4th | Conditional | High accuracy |
| `RK4Integrator` | 4th | Automatic dt | Production simulations |
| `AdaptiveTimeStepper` | Variable | Error-controlled | Stiff problems |

```python
# High-order time integration
result = bt.integrate(problem, t_end=1.0, method="rk4")

# Adaptive stepping with error control
result = bt.solve_adaptive(problem, t_end=1.0, tol=1e-6)
```

### Mass Transport Solvers

```
+------------------+     +----------------------+     +------------------+
|  DiffusionSolver |     | AdvectionDiffusion   |     | ReactionDiffusion|
|------------------|     |----------------------|     |------------------|
| - 1D, 2D, 3D     |     | - Upwind scheme      |     | - Linear         |
| - Uniform D      |     | - Central difference |     | - Logistic       |
| - Spatially-     |     | - Velocity fields    |     | - Michaelis-     |
|   varying D      |     |                      |     |   Menten         |
+------------------+     +----------------------+     +------------------+
         |                        |                          |
         +------------------------+---------------------------+
                                  |
                    +---------------------------+
                    | MembraneDiffusion1DSolver |
                    |---------------------------|
                    | - Partition coefficients  |
                    | - Hindered transport      |
                    | - Multi-layer membranes   |
                    | - Renkin equation         |
                    +---------------------------+
```

### Fluid Dynamics

| Solver | Regime | Features |
|--------|--------|----------|
| `StokesSolver` | Creeping flow (Re << 1) | Incompressible, pressure-velocity coupling |
| `NavierStokesSolver` | Inertial flow | Convection schemes, pressure projection |
| `DarcyFlowSolver` | Porous media | Permeability fields, pressure BC |

### Non-Newtonian Blood Rheology

```python
# Available viscosity models
models = [
    bt.NewtonianModel(mu=0.003),           # Constant viscosity
    bt.PowerLawModel(K=0.017, n=0.708),    # Shear-thinning
    bt.CarreauModel(mu_0=0.056, mu_inf=0.00345, lam=3.31, n=0.357),
    bt.CassonModel(tau_y=0.005, k=0.004),  # Yield stress
    bt.CrossModel(mu_0=0.056, mu_inf=0.00345, K=3.31, n=0.357),
]

# Built-in blood models
casson = bt.blood_casson_model()
carreau = bt.blood_carreau_model()
```

### Multi-Physics Applications

```
+----------------------------------+     +----------------------------------+
|     TumorDrugDeliverySolver      |     |     BioheatCryotherapySolver     |
|----------------------------------|     |----------------------------------|
|                                  |     |                                  |
|  Pressure Field (IFP)            |     |  Pennes Bioheat Equation         |
|         |                        |     |         |                        |
|         v                        |     |         v                        |
|  Velocity Field                  |     |  Phase Change (Freezing)         |
|         |                        |     |         |                        |
|         v                        |     |         v                        |
|  Drug Concentration              |     |  Arrhenius Tissue Damage         |
|                                  |     |                                  |
+----------------------------------+     +----------------------------------+
```

```python
from biotransport import TumorDrugDeliveryConfig, BioheatCryotherapyConfig

# Configure with documented parameters
config = TumorDrugDeliveryConfig(
    tumor_radius=0.003,      # 3mm tumor
    IFP_tumor=25.0,          # Elevated interstitial pressure (mmHg)
    D_drug_tumor=1e-11,      # Lower diffusivity in tumor
)

# View all parameters with units
print(config.describe())
```

### Electrochemical Transport (Nernst-Planck)

```python
# Ion transport with electric field coupling
from biotransport import IonSpecies, MultiIonSolver, ions, ghk

# Pre-defined ion species
na = ions.sodium()    # Na+
k = ions.potassium()  # K+
cl = ions.chloride()  # Cl-

# Goldman-Hodgkin-Katz equation
V_m = ghk.membrane_potential([na, k, cl], P_Na=1, P_K=1, P_Cl=0.45)
```

### Pattern Formation

```python
# Gray-Scott reaction-diffusion (Turing patterns)
solver = bt.GrayScottSolver(mesh, Du=0.16, Dv=0.08, F=0.035, k=0.065)
result = solver.run(t_end=10000, dt=1.0)
```

### Multi-Species Dynamics

```python
# Epidemic models
sir = bt.SIRReaction(beta=0.3, gamma=0.1)
seir = bt.SEIRReaction(beta=0.3, sigma=0.2, gamma=0.1)

# Ecological models  
predator_prey = bt.LotkaVolterraReaction(alpha=1.0, beta=0.1, gamma=0.1, delta=0.1)

# Biochemical reactions
brusselator = bt.BrusselatorReaction(a=1.0, b=3.0)
enzyme = bt.EnzymeCascadeReaction(k1=1.0, k2=0.5, k3=0.1)
```

---

## Educational Utilities (BMEN 341)

### Dimensionless Numbers

```python
from biotransport import dimensionless

Re = dimensionless.reynolds(rho=1000, v=0.1, L=0.01, mu=0.001)    # = 1000
Sc = dimensionless.schmidt(mu=0.001, rho=1000, D=1e-9)            # = 1000
Pe = dimensionless.peclet(v=0.001, L=0.01, D=1e-9)                # = 10000
Da = dimensionless.damkohler(k=0.1, L=0.01, D=1e-9)               # = 1e7
Bi = dimensionless.biot(h=100, L=0.01, k=0.5)                     # = 2.0
```

### Analytical Solutions

```python
from biotransport import analytical

# Semi-infinite diffusion
c = analytical.semi_infinite_diffusion(x=0.001, t=100, D=1e-9, c0=1.0)

# Poiseuille flow
v = analytical.poiseuille_velocity(r=0.001, R=0.005, dp_dx=1000, mu=0.001)

# Taylor-Couette flow  
v_theta = analytical.taylor_couette(r, R1=0.01, R2=0.02, omega1=10, omega2=0)
```

---

## Visualization

```python
# Simple plotting
bt.plot(mesh, solution, title="My Simulation")

# 1D with custom options
bt.plot_1d(mesh, solution, xlabel="Position (m)", ylabel="Concentration (mol/L)")

# 2D with colorbar
bt.plot_2d(mesh, solution, cmap="viridis", colorbar=True)

# 3D surface
bt.plot_2d_surface(mesh, solution, elevation=30, azimuth=45)
```

### VTK Export for ParaView

```python
# Single timestep
bt.write_vtk(mesh, {"concentration": c, "velocity": v}, "output.vtk")

# Time series
bt.write_vtk_series(mesh, snapshots, times, "simulation", "results/")
```

---

## Examples

### Basic

| Example | Description |
|---------|-------------|
| `1d_diffusion.py` | Simple diffusion with Gaussian IC |
| `heat_conduction.py` | 2D heat equation |

### Intermediate

| Example | Description |
|---------|-------------|
| `membrane_diffusion.py` | Transient membrane transport |
| `steady_membrane_diffusion.py` | Steady-state with partitioning |
| `oxygen_diffusion.py` | O2 consumption in tissue |
| `drug_diffusion_2d.py` | 2D drug release |
| `advection_diffusion.py` | Convection-diffusion |
| `darcy_flow.py` | Porous media flow |
| `stokes_flow.py` | Creeping viscous flow |
| `navier_stokes_flow.py` | Inertial viscous flow |
| `blood_rheology.py` | Non-Newtonian blood models |
| `cylindrical_coordinates.py` | Axisymmetric problems |
| `time_integration_methods.py` | Euler vs Heun vs RK4 |

### Advanced

| Example | Description |
|---------|-------------|
| `tumor_drug_delivery.py` | Multi-physics drug transport |
| `bioheat_cryotherapy.py` | Cryotherapy with tissue damage |
| `turing_patterns.py` | Gray-Scott reaction-diffusion |

### Verification

| Example | Description |
|---------|-------------|
| `verify_diffusion.py` | Analytical solution comparison |
| `verify_poiseuille.py` | Pipe flow validation |
| `verify_taylor_couette.py` | Rotating cylinders |
| `verify_viscoelastic.py` | Non-Newtonian validation |

```bash
# Run all examples
python run_examples.py
```

---

## Project Structure

```
biotransport/
|
+-- cpp/                          # C++ core library
|   +-- include/biotransport/
|   |   +-- core/                 # Mesh, numerics, analytical solutions
|   |   +-- physics/              # Fluid dynamics, mass transport, heat transfer
|   |   +-- solvers/              # Diffusion, advection-diffusion, explicit FD
|   +-- src/                      # Implementation files
|   +-- tests/                    # C++ unit tests (Google Test)
|   +-- benchmarks/               # Performance benchmarks
|
+-- python/
|   +-- bindings/                 # pybind11 bindings
|   +-- biotransport/             # Python package
|   |   +-- config/               # Configuration dataclasses
|   |   +-- adaptive.py           # Adaptive time-stepping
|   |   +-- convergence.py        # Grid convergence studies
|   |   +-- time_integrators.py   # RK4, Heun, Euler
|   |   +-- visualization.py      # Plotting utilities
|   |   +-- vtk_export.py         # VTK file output
|   +-- tests/                    # Python tests (pytest)
|
+-- examples/
|   +-- basic/                    # Introductory examples
|   +-- intermediate/             # Standard physics problems
|   +-- advanced/                 # Multi-physics simulations
|   +-- verification/             # Validation against analytical solutions
|
+-- docs/
|   +-- notes/                    # Development notes, gap analysis
|
+-- results/                      # Output directory for simulations
```

---

## API Cheatsheet

```python
import biotransport as bt
import numpy as np

# ---------------------------------------------------------------------------
#                              MESH CREATION
# ---------------------------------------------------------------------------
mesh = bt.mesh_1d(100, 0, 1)                    # 1D: 100 nodes, [0, 1]
mesh = bt.mesh_2d(50, 50, 0, 1, 0, 1)           # 2D: 50x50, [0,1] x [0,1]
mesh = bt.StructuredMesh(100, 0.0, 1.0)         # Alternative 1D
mesh = bt.StructuredMesh(50, 50, 0, 1, 0, 1)    # Alternative 2D
mesh = bt.CylindricalMesh(50, 50, 0, R, 0, L)   # Cylindrical (r, z)

# ---------------------------------------------------------------------------
#                           PROBLEM DEFINITION
# ---------------------------------------------------------------------------
problem = (
    bt.Problem(mesh)
    .diffusivity(D)                             # Scalar or array
    .velocity(vx, vy)                           # For advection
    .reaction(rate)                             # Reaction term
    .initial_condition(u0)                      # Array or helper
    .dirichlet(bt.Boundary.Left, value)         # Fixed value BC
    .neumann(bt.Boundary.Right, flux)           # Fixed flux BC
)

# ---------------------------------------------------------------------------
#                              SOLVING
# ---------------------------------------------------------------------------
result = bt.solve(problem, t_end=1.0)           # Simplest
result = bt.run(problem, t_end=1.0, dt=0.001)   # With dt control
result = bt.integrate(problem, 1.0, method="rk4")  # Higher-order
result = bt.solve_adaptive(problem, 1.0, tol=1e-6) # Adaptive

# ---------------------------------------------------------------------------
#                            VISUALIZATION
# ---------------------------------------------------------------------------
bt.plot(mesh, result.solution())                # Auto 1D/2D
bt.plot_1d(mesh, solution)                      # Force 1D
bt.plot_2d(mesh, solution, cmap="hot")          # Force 2D
bt.write_vtk(mesh, {"u": solution}, "out.vtk")  # ParaView export
```

---

## Testing

```bash
# Run all Python tests
python -m pytest python/tests/ -v

# Run with coverage
python -m pytest python/tests/ --cov=biotransport --cov-report=html

# Run C++ tests
cd build && ctest --output-on-failure
```

**Current Status**: 293 tests passing

---

## Troubleshooting

### Python Version Mismatch

```
ImportError: module was compiled for Python 3.12, but interpreter is 3.9
```

**Solution**: Rebuild with the correct Python version:
```bash
rm -rf build/
pip install -e .
```

### Windows Build Issues

- Ensure Visual Studio Build Tools with C++ workload is installed
- Verify CMake can find the MSVC compiler

### Docker Issues

```bash
# Rebuild without cache
docker build --no-cache -t biotransport:latest .

# Install GDB for debugging
docker run -it --name fix biotransport:latest bash -c "apt-get update && apt-get install -y gdb"
docker commit fix biotransport:latest
docker rm fix
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

See [docs/notes/GAP_ANALYSIS.md](docs/notes/GAP_ANALYSIS.md) for current development priorities and the project roadmap.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

```
    +-----------------------------------------------------------+
    |                                                           |
    |     "The purpose of computation is insight, not numbers"  |
    |                                                           |
    |                              - Richard Hamming            |
    |                                                           |
    +-----------------------------------------------------------+
```

**Built for Texas A&M University BMEN 341**

*Computational Methods in Biomedical Engineering*

</div>
