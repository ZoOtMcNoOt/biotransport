# BioTransport repository footprint

This document is a navigable map of the repository: directories/files, what they do, and the public API surface (C++ and Python).

## Scope and conventions

- This document focuses on the *source-of-truth* code and configuration.
- Generated/build artifacts are intentionally not exhaustively documented (they are machine outputs and can be deleted/recreated).

Generated (non-source) directories in this repo include:
- build/
- build-cpp-tests/
- cmake-build-debug/
- _deps/
- __pycache__/
- .pytest_cache/
- .ruff_cache/

## Precision policy (float vs double)

The library uses **`double`** precision for all scientific computation by default. This ensures:
- Sufficient precision for transport coefficients spanning many orders of magnitude
- Compatibility with standard scientific Python (NumPy defaults to float64)
- Accurate gradient calculations in fine meshes

**Exception:** `GrayScottSolver` uses **`float`** precision because:
- Pattern formation is qualitative, not quantitative
- Massive speedup on cache-limited 2D stencil operations
- Single precision is sufficient for visualization-focused output

When extending the library:
- Use `double` for new solvers unless performance profiling shows otherwise
- If adding GPU kernels, consider `float` for memory bandwidth
- Always document precision choice in class-level comments

## Top-level layout

- CMakeLists.txt: Top-level CMake project. Builds the C++ library and (optionally) the Python extension.
- cpp/: C++ library (headers, sources, tests).
- python/: Python package, plus pybind11 bindings for the compiled extension.
- examples/: End-user scripts (basic/intermediate/advanced).
- docs/: Documentation.
- results/: Example output location (plots, images, data dumps). Many examples write here by default.
- run_examples.py: Headless regression runner that executes every script in examples/.
- dev.sh: Convenience script for build/install/test/run in Unix-like environments.
- Dockerfile + docker-compose.yml: Containerized dev environment.
- environment.yml: Conda environment definition.
- pyproject.toml + setup.py: Python build metadata and CMake-backed extension build.
- pytest.ini: Pytest configuration.
- readme.md: User-facing introduction and quickstart.
- LICENSE: License.

## C++ library

### Directory map

- cpp/include/biotransport/: Public headers.
  - core/: Core primitives.
    - mesh/: Structured mesh types.
    - numerics/: Numerical utilities (stability helpers, etc.).
    - utils.hpp: Small CSV and vector comparison utilities.
  - physics/mass_transport/: Primary home for PDE/physics implementations.
  - solvers/: ExplicitFD façade API.
- cpp/src/: Implementations for headers in cpp/include/biotransport/.
- cpp/tests/: Standalone C++ test executables (simple assert-based).

### Public headers and API surface

#### cpp/include/biotransport/core/mesh/structured_mesh.hpp

- class StructuredMesh
  - StructuredMesh(int nx, double xmin, double xmax)
  - StructuredMesh(int nx, int ny, double xmin, double xmax, double ymin, double ymax)
  - int numNodes() const
  - int numCells() const
  - double dx() const
  - double dy() const
  - bool is1D() const
  - double x(int i) const
  - double y(int i, int j) const
  - int index(int i, int j = 0) const
  - int nx() const
  - int ny() const

Implementation: cpp/src/core/mesh/structured_mesh.cpp

#### cpp/include/biotransport/core/mesh/mesh.hpp

- using Mesh = StructuredMesh (backward compatibility alias)

#### cpp/include/biotransport/core/numerics/stability.hpp

Namespace: `biotransport::stability`

Time step helpers:
- `double suggest_diffusion_dt_1d(dx, D, safety=0.9)`
- `double suggest_diffusion_dt_2d(dx, dy, D, safety=0.9)`
- `double suggest_advection_dt_1d(dx, v, safety=0.9)`
- `double suggest_advection_dt_2d(dx, dy, vx, vy, safety=0.9)`
- `double suggest_advection_diffusion_dt_1d(dx, D, v, safety=0.9)`
- `double suggest_advection_diffusion_dt_2d(dx, dy, D, vx, vy, safety=0.9)`
- `double suggest_reaction_diffusion_dt_1d(dx, D, k, safety=0.9)`
- `double suggest_reaction_diffusion_dt_2d(dx, dy, D, k, safety=0.9)`
- `double suggest_michaelis_menten_dt_1d(dx, D, Vmax, Km, safety=0.9)`

Dimensionless number helpers:
- `double peclet_number(dx, v, D)` — cell Péclet number
- `double courant_number(dt, dx, v)` — CFL number
- `double fourier_number(dt, dx, D)` — diffusion number

#### cpp/include/biotransport/physics/mass_transport/diffusion.hpp

- enum class Boundary { Left, Right, Bottom, Top }
- enum class BoundaryType { DIRICHLET, NEUMANN }
- struct BoundaryCondition
  - BoundaryType type
  - double value
  - static BoundaryCondition Dirichlet(double value)
  - static BoundaryCondition Neumann(double flux)
- class DiffusionSolver
  - DiffusionSolver(const StructuredMesh& mesh, double diffusivity)
  - virtual void setInitialCondition(const std::vector<double>& values)
  - void setDirichletBoundary(int boundary_id, double value)
  - void setDirichletBoundary(Boundary boundary, double value)
  - void setNeumannBoundary(int boundary_id, double flux)
  - void setNeumannBoundary(Boundary boundary, double flux)
  - void setBoundaryCondition(int boundary_id, const BoundaryCondition& bc)
  - void setBoundaryCondition(Boundary boundary, const BoundaryCondition& bc)
  - virtual void solve(double dt, int num_steps)
  - const std::vector<double>& solution() const

Implementation: cpp/src/physics/mass_transport/diffusion.cpp

#### cpp/include/biotransport/physics/mass_transport/reaction_diffusion.hpp

- class ReactionDiffusionSolver : public DiffusionSolver
  - using ReactionFunction = std::function<double(double u, double x, double y, double t)>
  - ReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity, ReactionFunction reaction)
  - void solve(double dt, int num_steps) override

Implementation: cpp/src/physics/mass_transport/reaction_diffusion.cpp

#### Native reaction-diffusion solvers (no per-node callbacks)

These exist to avoid Python callback overhead and keep inner loops allocation-free.

- cpp/include/biotransport/physics/mass_transport/linear_reaction_diffusion.hpp
  - class LinearReactionDiffusionSolver : public DiffusionSolver
    - LinearReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity, double decay_rate)
    - void solve(double dt, int num_steps) override

- cpp/include/biotransport/physics/mass_transport/logistic_reaction_diffusion.hpp
  - class LogisticReactionDiffusionSolver : public DiffusionSolver
    - LogisticReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity, double growth_rate, double carrying_capacity)
    - void solve(double dt, int num_steps) override
    - double time() const
    - double growthRate() const
    - double carryingCapacity() const

- cpp/include/biotransport/physics/mass_transport/michaelis_menten_reaction_diffusion.hpp
  - class MichaelisMentenReactionDiffusionSolver : public DiffusionSolver
    - MichaelisMentenReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity, double vmax, double km)
    - void solve(double dt, int num_steps) override
    - double time() const
    - double vmax() const
    - double km() const

- cpp/include/biotransport/physics/mass_transport/constant_source_reaction_diffusion.hpp
  - class ConstantSourceReactionDiffusionSolver : public DiffusionSolver
    - ConstantSourceReactionDiffusionSolver(const StructuredMesh& mesh, double diffusivity, double source_rate)
    - void solve(double dt, int num_steps) override
    - double time() const
    - double sourceRate() const

Implementations:
- cpp/src/physics/mass_transport/linear_reaction_diffusion.cpp
- cpp/src/physics/mass_transport/logistic_reaction_diffusion.cpp
- cpp/src/physics/mass_transport/michaelis_menten_reaction_diffusion.cpp
- cpp/src/physics/mass_transport/constant_source_reaction_diffusion.cpp

#### Spatially varying diffusion

- cpp/include/biotransport/physics/mass_transport/variable_diffusion.hpp
  - class VariableDiffusionSolver
    - VariableDiffusionSolver(const StructuredMesh& mesh, std::vector<double> diffusivity)
    - void setInitialCondition(const std::vector<double>& values)
    - boundary setters: setDirichletBoundary / setNeumannBoundary / setBoundaryCondition (int or Boundary)
    - void solve(double dt, int num_steps)
    - const std::vector<double>& solution() const
    - double maxDiffusivity() const

Implementation: cpp/src/physics/mass_transport/variable_diffusion.cpp

#### Masked Michaelis–Menten reaction-diffusion

- cpp/include/biotransport/physics/mass_transport/masked_michaelis_menten_reaction_diffusion.hpp
  - class MaskedMichaelisMentenReactionDiffusionSolver : public DiffusionSolver
    - constructor takes mesh, diffusivity, vmax, km, mask (uint8), pinned_value
    - void solve(double dt, int num_steps) override

Implementation: cpp/src/physics/mass_transport/masked_michaelis_menten_reaction_diffusion.cpp

#### Gray–Scott (two-species, periodic)

- cpp/include/biotransport/physics/mass_transport/gray_scott.hpp
  - struct GrayScottRunResult
    - nx, ny, frames, steps_run
    - std::vector<int> frame_steps
    - std::vector<float> u_frames, v_frames packed as [frame][j][i]
  - class GrayScottSolver
    - GrayScottSolver(const StructuredMesh& mesh, double Du, double Dv, double f, double k)
    - GrayScottRunResult simulate(
        const std::vector<float>& u0,
        const std::vector<float>& v0,
        int total_steps,
        double dt,
        int steps_between_frames,
        int check_interval,
        double stable_tol,
        int min_frames_before_early_stop)

Implementation: cpp/src/physics/mass_transport/gray_scott.cpp

#### Tumor drug delivery

- cpp/include/biotransport/physics/mass_transport/tumor_drug_delivery.hpp
  - struct TumorDrugDeliverySaved
    - nx, ny, frames
    - std::vector<double> times_s
    - std::vector<double> free, bound, cellular, total packed as [frame][j][i]
  - class TumorDrugDeliverySolver
    - TumorDrugDeliverySolver(mesh, tumor_mask, hydraulic_conductivity, p_boundary, p_tumor)
    - std::vector<double> solvePressureSOR(int max_iter, double tol, double omega) const
    - TumorDrugDeliverySaved simulate(
        pressure,
        diffusivity,
        permeability,
        vessel_density,
        k_binding,
        k_uptake,
        c_plasma,
        dt,
        num_steps,
        times_to_save_s) const

Implementation: cpp/src/physics/mass_transport/tumor_drug_delivery.cpp

#### Bioheat cryotherapy

- cpp/include/biotransport/physics/mass_transport/bioheat_cryotherapy.hpp
  - struct BioheatSaved
    - nx, ny, frames
    - std::vector<double> times_s
    - std::vector<double> temperature_K, damage packed as [frame][j][i]
  - class BioheatCryotherapySolver
    - BioheatCryotherapySolver(mesh, probe_mask, perfusion_map, q_met_map, and many scalar physical parameters)
    - BioheatSaved simulate(double dt, int num_steps, const std::vector<double>& times_to_save_s) const

Implementation: cpp/src/physics/mass_transport/bioheat_cryotherapy.cpp

#### Facade API: problem objects + conservative run

- cpp/include/biotransport/solvers/explicit_fd.hpp
  - class DiffusionProblem
  - class LinearReactionDiffusionProblem
  - class ConstantSourceReactionDiffusionProblem
  - class MichaelisMentenReactionDiffusionProblem
  - class LogisticReactionDiffusionProblem
  - struct SolverStats
  - struct RunResult
  - class ExplicitFD
    - ExplicitFD& safetyFactor(double factor)
    - RunResult run(problem, double t_end) const  (overloads for each problem type)

Note: these problems are small configuration holders with a fluent API.

#### Compatibility bridge headers

The following headers exist mainly to preserve older include paths and forward to physics/mass_transport:

- cpp/include/biotransport/solvers/diffusion.hpp
- cpp/include/biotransport/solvers/reaction_diffusion.hpp
- cpp/include/biotransport/solvers/variable_diffusion.hpp
- cpp/include/biotransport/solvers/gray_scott.hpp
- cpp/include/biotransport/solvers/tumor_drug_delivery.hpp
- cpp/include/biotransport/solvers/bioheat_cryotherapy.hpp
- cpp/include/biotransport/solvers/linear_reaction_diffusion.hpp
- cpp/include/biotransport/solvers/logistic_reaction_diffusion.hpp
- cpp/include/biotransport/solvers/michaelis_menten_reaction_diffusion.hpp
- cpp/include/biotransport/solvers/constant_source_reaction_diffusion.hpp
- cpp/include/biotransport/solvers/masked_michaelis_menten_reaction_diffusion.hpp

### C++ utility header

- cpp/include/biotransport/core/utils.hpp
  - namespace biotransport::utils
    - bool writeCsv1D(filename, x, solution)
    - bool writeCsv2D(filename, x, y, solution, nx, ny)
    - double l2Norm(a, b)
    - double maxDifference(a, b)

Implementation: cpp/src/core/utils.cpp

### Dimensionless numbers (header-only)

- cpp/include/biotransport/core/dimensionless.hpp
  - namespace biotransport::dimensionless
    - double reynolds(velocity, length, kinematic_viscosity)
    - double peclet(velocity, length, diffusivity)
    - double schmidt(kinematic_viscosity, diffusivity)
    - double prandtl(kinematic_viscosity, thermal_diffusivity)
    - double lewis(thermal_diffusivity, mass_diffusivity)
    - double biot(h, L, k)
    - double fourier(alpha, t, L)
    - double nusselt(h, L, k)
    - double sherwood(k_c, L, D)
    - double damkohler_first(k, L, D)
    - double damkohler_second(k, C0, D, L)
    - double thiele(R, k, D)
    - double stanton(h, rho, Cp, U)

### Analytical solutions (header-only)

- cpp/include/biotransport/core/analytical.hpp
  - namespace biotransport::analytical
    - Diffusion:
      - double diffusion_1d_semi_infinite(x, t, D, C_surface, C_initial)
      - double diffusion_penetration_depth(D, t)
      - double lumped_exponential(C_0, C_inf, t, tau)
    - Poiseuille flow:
      - double poiseuille_velocity(r, R, dp_dz, viscosity)
      - double poiseuille_max_velocity(R, dp_dz, viscosity)
      - double poiseuille_flow_rate(R, dp_dz, viscosity)
      - double poiseuille_wall_shear(R, dp_dz)
    - Couette flow:
      - double couette_velocity(y, h, U_top)
      - double couette_max_velocity(U_top)
    - Bernoulli:
      - double bernoulli_velocity(p1, p2, rho, v1)
    - Taylor-Couette flow:
      - double taylor_couette_velocity(r, a, b, omega_a, omega_b)
      - double taylor_couette_torque(a, b, omega_a, omega_b, viscosity)
    - Viscoelastic models:
      - double maxwell_relaxation(E, eta, epsilon_0, t)
      - double maxwell_relaxation_time(E, eta)
      - double kelvin_voigt_creep(E, eta, sigma_0, t)
      - double sls_relaxation(E1, E2, eta, epsilon_0, t)
      - double sls_creep(E1, E2, eta, sigma_0, t)
      - double burgers_creep(E1, mu1, E2, mu2, sigma_0, t)
      - double burgers_compliance(E1, mu1, E2, mu2, t)
    - Complex modulus:
      - double complex_modulus_magnitude(G1, G2)
      - double loss_tangent(G1, G2)
      - double phase_angle(G1, G2)
    - Kinetics:
      - double first_order_decay(C_0, k, t)
      - double logistic_growth(C_0, carrying_capacity, growth_rate, t)

### Fluid Dynamics - Stokes Solver (header-only)

- cpp/include/biotransport/physics/fluid_dynamics/stokes.hpp
  - enum class VelocityBCType { DIRICHLET, NEUMANN, NOSLIP, INFLOW, OUTFLOW }
  - struct VelocityBC
    - VelocityBCType type
    - double u_value, v_value
    - static VelocityBC NoSlip()
    - static VelocityBC Inflow(u, v)
    - static VelocityBC Outflow()
    - static VelocityBC Dirichlet(u, v)
    - static VelocityBC StressFree()
  - struct StokesResult
    - bool converged
    - int iterations
    - double residual
    - std::vector<double> velocity_x, velocity_y, pressure
  - class StokesSolver
    - StokesSolver(mesh, viscosity)
    - void setVelocityBC(boundary, bc)
    - void setBodyForce(forces)
    - StokesResult solve(max_iter=10000, tol=1e-6, omega=1.5)

### Fluid Dynamics - Navier-Stokes Solver (header-only)

- cpp/include/biotransport/physics/fluid_dynamics/navier_stokes.hpp
  - enum class ConvectionScheme { UPWIND, CENTRAL, HYBRID, QUICK }
  - struct NavierStokesResult
    - bool converged
    - int iterations
    - double time
    - std::vector<double> velocity_x, velocity_y, pressure
  - class NavierStokesSolver
    - NavierStokesSolver(mesh, density, viscosity)
    - void setVelocityBC(boundary, bc)
    - void setConvectionScheme(scheme)
    - void setBodyForce(forces)
    - NavierStokesResult step(dt)
    - NavierStokesResult solveSteady(t_final, dt, tol, max_iter)

### Cylindrical Mesh (header-only)

- cpp/include/biotransport/core/mesh/cylindrical_mesh.hpp
  - enum class CylindricalMeshType { RADIAL_R, AXISYMMETRIC_RZ, FULL_3D }
  - class CylindricalMesh
    - CylindricalMesh(nr, r_min, r_max) // 1D radial
    - CylindricalMesh(nr, nz, r_max, z_max) // 2D axisymmetric
    - CylindricalMesh(nr, ntheta, nz, r_max, z_max) // 3D
    - CylindricalMeshType type() const
    - std::vector<double> r_coordinates() const
    - std::vector<double> z_coordinates() const
    - double dr() const, dz() const, dtheta() const
    - Differential operators: gradientR, gradientZ, laplacian, divergence

### Non-Newtonian Fluid Models (header-only)

- cpp/include/biotransport/physics/fluid_dynamics/non_newtonian.hpp
  - enum class FluidModel { NEWTONIAN, POWER_LAW, CARREAU, ... }
  - class ViscosityModel (abstract base)
    - virtual double viscosity(shear_rate) const = 0
    - virtual FluidModel type() const = 0
  - class NewtonianModel : public ViscosityModel
    - NewtonianModel(mu)
  - class PowerLawModel : public ViscosityModel
    - PowerLawModel(K, n)
  - class CarreauModel : public ViscosityModel
    - CarreauModel(mu_0, mu_inf, lambda, n)
  - class CarreauYasudaModel : public ViscosityModel
    - CarreauYasudaModel(mu_0, mu_inf, lambda, a, n)
  - class CrossModel : public ViscosityModel
    - CrossModel(mu_0, mu_inf, K, n)
  - class BinghamModel : public ViscosityModel
    - BinghamModel(tau_y, mu_p, epsilon=1e-6)
  - class HerschelBulkleyModel : public ViscosityModel
    - HerschelBulkleyModel(tau_y, K, n, epsilon=1e-6)
  - class CassonModel : public ViscosityModel
    - CassonModel(tau_y, mu_p, epsilon=1e-6)
    - double yield_stress() const
    - double plastic_viscosity() const
  - Blood rheology utilities:
    - CassonModel bloodCassonModel(hematocrit)
    - CarreauModel bloodCarreauModel(hematocrit)
    - double pipeWallShearRate(U_avg, R)

## Python package

### Directory map

- python/biotransport/: User-facing Python package.
  - __init__.py: Public re-exports for a minimal-import UX.
  - _core/: Internal wrapper around the compiled extension module.
  - config.py: Configuration dataclasses for multi-physics solvers.
  - mesh_utils.py: Coordinate helpers and reshape helpers.
  - run.py: Convenience run wrapper around the C++ ExplicitFD facade.
  - utils.py: Results directory helpers.
  - visualization.py: Plotting helpers and a single plot_field entry point.
- python/bindings/: pybind11 extension module build + binding definitions.
- python/tests/: Python unit tests.

Submodules exposed via bindings:
- biotransport.dimensionless: Dimensionless number calculations.
- biotransport.analytical: Analytical solution functions.

### Python public API

The intended beginner import surface is the top-level package.

From python/biotransport/__init__.py:

Core types (bound from C++):
- StructuredMesh
- Boundary, BoundaryType, BoundaryCondition
- DiffusionSolver
- ReactionDiffusionSolver
- LinearReactionDiffusionSolver
- LogisticReactionDiffusionSolver
- MichaelisMentenReactionDiffusionSolver
- MaskedMichaelisMentenReactionDiffusionSolver
- ConstantSourceReactionDiffusionSolver
- VariableDiffusionSolver
- GrayScottSolver, GrayScottRunResult
- TumorDrugDeliverySolver, TumorDrugDeliverySaved
- BioheatCryotherapySolver, BioheatSaved

Facade types (Problem + run):
- DiffusionProblem
- LinearReactionDiffusionProblem
- LogisticReactionDiffusionProblem
- MichaelisMentenReactionDiffusionProblem
- ConstantSourceReactionDiffusionProblem
- ExplicitFD
- SolverStats
- RunResult

Beginner helpers (pure Python):
- get_results_dir, get_result_path
- x_nodes, y_nodes, xy_grid
- as_1d, as_2d
- run
- plot_1d_solution, plot_2d_solution, plot_2d_surface
- plot_field

### Bindings map (pybind11)

Bindings live in python/bindings/biotransport_bindings.cpp and build via python/bindings/CMakeLists.txt.

Key naming conventions in Python bindings:
- C++ camelCase -> Python snake_case (examples)
  - StructuredMesh::numNodes -> mesh.num_nodes()
  - DiffusionSolver::setInitialCondition -> solver.set_initial_condition(...)
  - ExplicitFD::safetyFactor -> ExplicitFD().safety_factor(...)

NumPy interop conventions:
- solver.solution() returns a NumPy array view tied to the owning C++ object lifetime.
- For saved frame bundles (GrayScottRunResult, TumorDrugDeliverySaved, BioheatSaved), frame getters return shaped NumPy views:
  - u_frames(): (frames, ny, nx)
  - free()/bound()/cellular()/total(): (frames, ny, nx)
  - temperature_K()/damage(): (frames, ny, nx)

## Examples

- examples/basic/1d_diffusion.py: 1D diffusion demo with plotting.
- examples/basic/heat_conduction.py: Heat equation style diffusion demo.
- examples/intermediate/drug_diffusion_2d.py: 2D diffusion example.
- examples/intermediate/membrane_diffusion.py: Variable diffusion / membrane region demo.
- examples/intermediate/oxygen_diffusion.py: Masked Michaelis–Menten sink demo.
- examples/advanced/turing_patterns.py: Gray-Scott pattern formation demo.
- examples/advanced/tumor_drug_delivery.py: Pressure + convection-diffusion + binding/uptake demo.
- examples/advanced/bioheat_cryotherapy.py: Bioheat + freezing + Arrhenius damage demo.
- examples/intermediate/advection_diffusion.py: 1D/2D advection-diffusion demos.
- examples/intermediate/darcy_flow.py: Porous media flow demos.
- examples/intermediate/steady_membrane_diffusion.py: Steady-state membrane diffusion analysis.
- examples/verification/verify_poiseuille.py: Poiseuille flow analytical verification.
- examples/verification/verify_taylor_couette.py: Taylor-Couette flow analytical verification.
- examples/verification/verify_viscoelastic.py: Viscoelastic model (Maxwell, KV, SLS, Burgers) verification.
- examples/verification/verify_diffusion.py: Semi-infinite diffusion numerical vs analytical.

## Tests

### C++ tests (cpp/tests)

Each file builds a standalone executable with a main().

- cpp/tests/core/test_mesh.cpp
  - testStructuredMesh1D()
  - testStructuredMesh2D()
  - main()

- cpp/tests/core/test_analytical.cpp
  - Tests for analytical solutions: diffusion, Poiseuille, Couette, Taylor-Couette,
    Maxwell, Kelvin-Voigt, SLS, Burgers, complex modulus utilities.
  - main()

- cpp/tests/physics/test_boundary_corners.cpp
  - testNeumannSideDoesNotOverrideDirichletCorners()
  - main()

- cpp/tests/physics/test_diffusion.cpp
  - testDiffusion1D()
  - testReactionDiffusion1D()
  - main()

- cpp/tests/physics/test_diffusion_2d.cpp
  - computeMass2D()
  - testDiffusion2DNeumannMassConservation()
  - testDiffusion2DDirichletBoundaryPinned()
  - main()

- cpp/tests/physics/test_diffusion_1d_neumann.cpp
  - computeMass1D()
  - testDiffusion1DNeumannMassConservation()
  - main()

- cpp/tests/physics/test_linear_reaction_diffusion.cpp
  - testLinearReactionDiffusionMatchesExponentialDecayWhenNoDiffusion()
  - main()

- cpp/tests/physics/test_logistic_reaction_diffusion.cpp
  - logisticExact()
  - testLogisticReaction1DMatchesODEForUniformField()
  - main()

- cpp/tests/physics/test_michaelis_menten_reaction_diffusion.cpp
  - michaelisMentenExactU()
  - testMichaelisMentenSink1DMatchesODEForUniformField()
  - main()

- cpp/tests/physics/test_constant_source_reaction_diffusion.cpp
  - testConstantSource1DMatchesODEForUniformField()
  - main()

- cpp/tests/physics/test_explicit_fd_run.cpp
  - testExplicitFDRunUsesStableDtAndPinsDirichlet()
  - main()

- cpp/tests/physics/test_explicit_fd_run_constant_source.cpp
  - testExplicitFDConstantSource1DUniformGrowth()
  - main()

- cpp/tests/physics/test_explicit_fd_run_michaelis_menten.cpp
  - solveMichaelisMentenODE()
  - testExplicitFDMichaelisMenten1DUniformDecay()
  - main()

- cpp/tests/physics/test_explicit_fd_run_logistic.cpp
  - logisticSolution()
  - testExplicitFDLogistic1DUniformGrowth()
  - main()

### Python tests (python/tests)

- python/tests/test_diffusion.py
  - class TestDiffusion(unittest.TestCase)
    - test_1d_mesh
    - test_2d_mesh
    - test_diffusion_solver
    - test_reaction_diffusion_solver

- python/tests/test_explicit_fd_facade.py
  - class TestExplicitFDFacade(unittest.TestCase)
    - test_diffusion_problem_run_1d_dirichlet
    - test_diffusion_problem_run_2d_neumann_preserves_constant

## Build & packaging

### CMake

- CMakeLists.txt: Builds the library and optionally the Python extension.
- cpp/CMakeLists.txt: Defines the biotransport library and sources.
- python/bindings/CMakeLists.txt: Defines the pybind11 module named _core.

### Python packaging

- setup.py: setuptools build that delegates native compilation to CMake.
- pyproject.toml: declares build-system dependencies for PEP 517 builds.

### Tooling / CI-ish scripts

- run_examples.py: headless example runner used as a lightweight regression check.
- dev.sh: build/install/test/run convenience for Unix-like shells.
