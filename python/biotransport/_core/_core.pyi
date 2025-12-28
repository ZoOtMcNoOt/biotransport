"""Type stubs for biotransport._core._core extension module.

This file provides type hints for IDE autocompletion and static type checking.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, overload

import numpy as np
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[np.float64]
FloatArray = npt.NDArray[np.float64]

# =============================================================================
# Enumerations
# =============================================================================

class Boundary(Enum):
    """Boundary edge identifiers for 2D domains."""

    Left = 0
    Right = 1
    Top = 2
    Bottom = 3

# Convenience aliases
Left: Boundary
Right: Boundary
Top: Boundary
Bottom: Boundary

class BoundaryType(Enum):
    """Types of boundary conditions."""

    DIRICHLET = 0
    NEUMANN = 1

# Convenience aliases
DIRICHLET: BoundaryType
NEUMANN: BoundaryType

class VelocityBCType(Enum):
    """Types of velocity boundary conditions for fluid flow."""

    DIRICHLET = 0
    NEUMANN = 1
    NOSLIP = 2
    INFLOW = 3
    OUTFLOW = 4

# Convenience aliases
NOSLIP: VelocityBCType
INFLOW: VelocityBCType
OUTFLOW: VelocityBCType

class AdvectionScheme(Enum):
    """Advection discretization schemes."""

    UPWIND = 0
    CENTRAL = 1
    QUICK = 2
    HYBRID = 3

# Convenience aliases
UPWIND: AdvectionScheme
CENTRAL: AdvectionScheme
QUICK: AdvectionScheme
HYBRID: AdvectionScheme

class ConvectionScheme(Enum):
    """Convection discretization schemes for Navier-Stokes."""

    UPWIND = 0
    CENTRAL = 1
    QUICK = 2
    HYBRID = 3

class CylindricalMeshType(Enum):
    """Types of cylindrical coordinate meshes."""

    RADIAL_R = 0
    AXISYMMETRIC_RZ = 1
    FULL_3D = 2

# Convenience aliases
RADIAL_R: CylindricalMeshType
AXISYMMETRIC_RZ: CylindricalMeshType
FULL_3D: CylindricalMeshType

class ViscosityModel(Enum):
    """Base class for rheological viscosity models."""

    NEWTONIAN = 0
    POWER_LAW = 1
    CARREAU = 2
    CARREAU_YASUDA = 3
    CROSS = 4
    CASSON = 5
    BINGHAM = 6
    HERSCHEL_BULKLEY = 7

# Convenience aliases
NEWTONIAN: ViscosityModel
POWER_LAW: ViscosityModel
CARREAU: ViscosityModel
CARREAU_YASUDA: ViscosityModel
CROSS: ViscosityModel
CASSON: ViscosityModel
BINGHAM: ViscosityModel
HERSCHEL_BULKLEY: ViscosityModel

# =============================================================================
# Data Classes / Structs
# =============================================================================

class BoundaryCondition:
    """Represents a boundary condition for transport problems.

    Boundary conditions specify how the solution behaves at domain boundaries.
    Two types are supported:

    - **Dirichlet**: Fixed value at the boundary (e.g., constant concentration)
    - **Neumann**: Fixed flux at the boundary (e.g., insulated wall with zero flux)

    Examples:
        >>> # Fixed concentration at left boundary
        >>> bc_left = BoundaryCondition.dirichlet(1.0)
        >>> # Zero flux (insulated) at right boundary
        >>> bc_right = BoundaryCondition.neumann(0.0)

    Attributes:
        type: The type of boundary condition (DIRICHLET or NEUMANN).
        value: The boundary value (concentration for Dirichlet, flux for Neumann).
    """

    type: BoundaryType
    value: float

    @staticmethod
    def dirichlet(value: float) -> BoundaryCondition:
        """Create a Dirichlet (fixed value) boundary condition."""
        ...

    @staticmethod
    def neumann(flux: float) -> BoundaryCondition:
        """Create a Neumann (fixed flux) boundary condition."""
        ...

class VelocityBC:
    """Velocity boundary condition for fluid flow solvers.

    Specifies velocity constraints at domain boundaries for Stokes and
    Navier-Stokes solvers. Common types include:

    - **NoSlip**: Zero velocity at solid walls (u=v=0)
    - **Inflow**: Prescribed velocity at inlet
    - **Outflow**: Stress-free condition at outlet
    - **Dirichlet**: Arbitrary fixed velocity

    Examples:
        >>> # No-slip wall (solid boundary)
        >>> bc_wall = VelocityBC.no_slip()
        >>> # Inlet with horizontal velocity
        >>> bc_inlet = VelocityBC.inflow(u=0.1, v=0.0)
        >>> # Stress-free outlet
        >>> bc_outlet = VelocityBC.outflow()

    Attributes:
        type: The type of velocity boundary condition.
        u_value: x-component of velocity (if applicable).
        v_value: y-component of velocity (if applicable).
    """

    type: VelocityBCType
    u_value: float
    v_value: float

    @staticmethod
    def no_slip() -> VelocityBC:
        """Create a no-slip (zero velocity) boundary condition."""
        ...

    @staticmethod
    def inflow(u: float, v: float = 0.0) -> VelocityBC:
        """Create an inflow boundary condition with specified velocity."""
        ...

    @staticmethod
    def outflow() -> VelocityBC:
        """Create an outflow (stress-free) boundary condition."""
        ...

    @staticmethod
    def stress_free() -> VelocityBC:
        """Create a stress-free boundary condition."""
        ...

    @staticmethod
    def dirichlet(u: float, v: float) -> VelocityBC:
        """Create a Dirichlet boundary condition with specified velocities."""
        ...

class SolverStats:
    """Statistics from a time-stepping solver run.

    Contains diagnostic information about solver performance and solution
    quality, including timing, mass conservation, and solution bounds.

    Attributes:
        dt: Time step size used.
        steps: Total number of time steps taken.
        t_end: Final simulation time.
        wall_time_s: Wall-clock execution time in seconds.
        mass_initial: Initial total mass/concentration.
        mass_final: Final total mass/concentration.
        mass_abs_drift: Absolute mass drift (mass_final - mass_initial).
        mass_rel_drift: Relative mass drift.
        u_min_initial: Minimum solution value at t=0.
        u_max_initial: Maximum solution value at t=0.
        u_min_final: Minimum solution value at t=t_end.
        u_max_final: Maximum solution value at t=t_end.
    """

    dt: float
    steps: int
    t_end: float
    wall_time_s: float
    mass_initial: float
    mass_final: float
    mass_abs_drift: float
    mass_rel_drift: float
    u_min_initial: float
    u_max_initial: float
    u_min_final: float
    u_max_final: float

class RunResult:
    """Result from a reaction-diffusion solver run.

    Contains the final solution field and solver statistics.

    Examples:
        >>> result = solver.run(problem, t_end=1.0)
        >>> print(f"Completed in {result.stats.steps} steps")
        >>> print(f"Solution range: [{result.solution.min():.3f}, {result.solution.max():.3f}]")

    Attributes:
        solution: Final concentration/temperature field as a 1D NumPy array.
            For 2D problems, reshape to (ny+1, nx+1) for visualization.
        stats: Solver statistics including timing and mass conservation.
    """

    solution: FloatArray
    stats: SolverStats

class StokesResult:
    """Result from Stokes flow solver.

    Contains the steady-state velocity and pressure fields for creeping
    (low Reynolds number) flow, along with convergence information.

    The Stokes equations describe viscous-dominated flow where inertial
    effects are negligible (Re << 1), common in microfluidics and
    biological flows at the cellular scale.

    Attributes:
        u: x-component of velocity field.
        v: y-component of velocity field.
        pressure: Pressure field.
        divergence: Velocity divergence (should be ~0 for incompressible flow).
        converged: True if solver converged within tolerance.
        iterations: Number of iterations to convergence.
        residual: Final residual norm.
    """

    u: FloatArray
    v: FloatArray
    pressure: FloatArray
    divergence: FloatArray
    converged: bool
    iterations: int
    residual: float

class NavierStokesResult:
    """Result from Navier-Stokes solver.

    Contains the velocity and pressure fields at the final time step,
    along with flow characteristics and stability information.

    The Navier-Stokes equations describe incompressible viscous flow
    including both inertial and viscous effects, applicable to a wide
    range of Reynolds numbers.

    Attributes:
        u: x-component of velocity field.
        v: y-component of velocity field.
        pressure: Pressure field.
        time: Final simulation time.
        time_steps: Total number of time steps taken.
        reynolds: Reynolds number of the flow.
        max_velocity: Maximum velocity magnitude in the domain.
        stable: True if simulation remained numerically stable.
    """

    u: FloatArray
    v: FloatArray
    pressure: FloatArray
    time: float
    time_steps: int
    reynolds: float
    max_velocity: float
    stable: bool

class DarcyFlowResult:
    """Result from Darcy flow solver.

    Contains the pressure and velocity fields for flow through porous
    media governed by Darcy's law: v = -K/μ ∇p.

    Darcy flow is applicable to groundwater flow, tissue perfusion,
    and other porous media transport problems.

    Attributes:
        pressure: Pressure field.
        vx: x-component of Darcy velocity (superficial velocity).
        vy: y-component of Darcy velocity.
        converged: True if solver converged within tolerance.
        iterations: Number of iterations to convergence.
        residual: Final residual norm.
    """

    pressure: FloatArray
    vx: FloatArray
    vy: FloatArray
    converged: bool
    iterations: int
    residual: float

class MembraneDiffusionResult:
    """Result from membrane diffusion solver.

    Contains the steady-state concentration profile across a membrane
    and derived transport properties.

    Membrane diffusion is fundamental to drug delivery, dialysis,
    and cellular transport. The permeability P = D·K/L relates
    diffusivity D, partition coefficient K, and thickness L.

    Attributes:
        x: Position coordinates across the membrane.
        concentration: Steady-state concentration profile.
        flux: Mass flux through the membrane (mol/m²/s).
        permeability: Membrane permeability (m/s).
        effective_diffusivity: Effective diffusion coefficient accounting
            for hindered diffusion if applicable.
    """

    x: FloatArray
    concentration: FloatArray
    flux: float
    permeability: float
    effective_diffusivity: float

class GrayScottRunResult:
    """Result from Gray-Scott reaction-diffusion simulation.

    The Gray-Scott model describes pattern formation in a two-species
    autocatalytic reaction system. It produces a rich variety of
    spatiotemporal patterns including spots, stripes, and traveling waves.

    The equations are:
        ∂u/∂t = Du ∇²u - uv² + f(1-u)
        ∂v/∂t = Dv ∇²v + uv² - (f+k)v

    where f is the feed rate and k is the kill rate.

    Attributes:
        nx: Number of grid points in x-direction.
        ny: Number of grid points in y-direction.
        steps_run: Total simulation steps completed.
        frames: Number of saved frames.
        frame_steps: Steps between saved frames.
        u_frames: List of u-field snapshots.
        v_frames: List of v-field snapshots.
    """

    nx: int
    ny: int
    steps_run: int
    frames: int
    frame_steps: int
    u_frames: list[FloatArray]
    v_frames: list[FloatArray]

class TumorDrugDeliverySaved:
    """Saved frames from tumor drug delivery simulation.

    Contains time-series data from a coupled simulation of drug transport
    in tumor tissue, including convection (pressure-driven flow),
    diffusion, binding, and cellular uptake.

    The model tracks three drug compartments:
    - Free drug in interstitial space
    - Bound drug (reversibly bound to tissue)
    - Internalized drug (taken up by cells)

    Attributes:
        nx: Number of grid points in x-direction.
        ny: Number of grid points in y-direction.
        frames: Number of saved time frames.
        times_s: List of save times in seconds.
        free: Free drug concentration at each frame.
        bound: Bound drug concentration at each frame.
        cellular: Internalized drug concentration at each frame.
        total: Total drug concentration at each frame.
    """

    nx: int
    ny: int
    frames: int
    times_s: list[float]
    free: list[FloatArray]
    bound: list[FloatArray]
    cellular: list[FloatArray]
    total: list[FloatArray]

class BioheatSaved:
    """Saved frames from bioheat cryotherapy simulation.

    Contains time-series data from a coupled thermal-damage simulation
    modeling cryoablation therapy. The model includes:

    - Pennes bioheat equation with blood perfusion
    - Phase change (freezing/thawing) with latent heat
    - Arrhenius tissue damage accumulation

    Used for planning cryotherapy procedures in cancer treatment.

    Attributes:
        nx: Number of grid points in x-direction.
        ny: Number of grid points in y-direction.
        frames: Number of saved time frames.
        times_s: List of save times in seconds.
        temperature_K: Temperature field in Kelvin at each frame.
        damage: Cumulative damage parameter Ω at each frame.
            Ω > 1 indicates irreversible tissue damage.
    """

    nx: int
    ny: int
    frames: int
    times_s: list[float]
    temperature_K: list[FloatArray]
    damage: list[FloatArray]

# =============================================================================
# Meshes
# =============================================================================

class StructuredMesh:
    """Uniform structured mesh for 1D or 2D rectangular domains.

    A structured mesh divides the domain into a regular grid of cells.
    Nodes are located at cell corners, and the solution is typically
    stored at node locations.

    For 1D problems, the mesh has `nx` cells and `nx+1` nodes.
    For 2D problems, the mesh has `nx*ny` cells and `(nx+1)*(ny+1)` nodes.

    Examples:
        >>> # 1D mesh: 100 cells from x=0 to x=1
        >>> mesh_1d = StructuredMesh(100, 0.0, 1.0)
        >>> print(f"dx = {mesh_1d.dx()}")  # 0.01
        >>>
        >>> # 2D mesh: 50x50 cells on unit square
        >>> mesh_2d = StructuredMesh(50, 50, 0.0, 1.0, 0.0, 1.0)
        >>> print(f"Nodes: {mesh_2d.num_nodes()}")  # 2601

    Note:
        The mesh uses row-major (C-style) indexing. For a 2D mesh,
        node (i, j) maps to linear index `j * (nx+1) + i`.
    """

    @overload
    def __init__(self, nx: int, xmin: float, xmax: float) -> None:
        """Create a 1D structured mesh.

        Args:
            nx: Number of cells in x-direction.
            xmin: Minimum x-coordinate.
            xmax: Maximum x-coordinate.
        """
        ...

    @overload
    def __init__(
        self,
        nx: int,
        ny: int,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> None:
        """Create a 2D structured mesh.

        Args:
            nx: Number of cells in x-direction.
            ny: Number of cells in y-direction.
            xmin: Minimum x-coordinate.
            xmax: Maximum x-coordinate.
            ymin: Minimum y-coordinate.
            ymax: Maximum y-coordinate.
        """
        ...

    def __init__(self, *args: Any) -> None: ...
    def nx(self) -> int:
        """Number of cells in x-direction."""
        ...

    def ny(self) -> int:
        """Number of cells in y-direction."""
        ...

    def dx(self) -> float:
        """Cell spacing in x-direction."""
        ...

    def dy(self) -> float:
        """Cell spacing in y-direction."""
        ...

    def num_cells(self) -> int:
        """Total number of cells."""
        ...

    def num_nodes(self) -> int:
        """Total number of nodes."""
        ...

    def is_1d(self) -> bool:
        """True if mesh is 1D (ny == 1)."""
        ...

    def x(self) -> FloatArray:
        """Array of x-coordinates at cell centers."""
        ...

    def y(self) -> FloatArray:
        """Array of y-coordinates at cell centers."""
        ...

    def index(self, i: int, j: int) -> int:
        """Convert (i, j) indices to linear index."""
        ...

class CylindricalMesh:
    """Mesh in cylindrical coordinates (r, θ, z).

    Supports three configurations:

    - **Radial (1D)**: r-direction only, for radially symmetric problems
    - **Axisymmetric (2D)**: r-z plane, for problems with azimuthal symmetry
    - **Full 3D**: Complete (r, θ, z) discretization

    Cylindrical coordinates naturally handle the axis singularity at r=0
    and are ideal for pipe flows, vessel transport, and rotationally
    symmetric geometries.

    Examples:
        >>> # 1D radial mesh for pipe cross-section
        >>> mesh_r = CylindricalMesh(50, 0.0, 0.01)  # r: 0 to 10mm
        >>>
        >>> # 2D axisymmetric mesh for vessel segment
        >>> mesh_rz = CylindricalMesh(20, 100, 0.0, 0.005, 0.0, 0.1)

    Note:
        For meshes including r=0, special treatment is applied at the
        axis to handle the coordinate singularity.
    """

    @overload
    def __init__(self, nr: int, rmin: float, rmax: float) -> None:
        """Create a 1D radial mesh."""
        ...

    @overload
    def __init__(
        self,
        nr: int,
        nz: int,
        rmin: float,
        rmax: float,
        zmin: float,
        zmax: float,
    ) -> None:
        """Create a 2D axisymmetric (r, z) mesh."""
        ...

    @overload
    def __init__(
        self,
        nr: int,
        ntheta: int,
        nz: int,
        rmin: float,
        rmax: float,
        thetamin: float,
        thetamax: float,
        zmin: float,
        zmax: float,
    ) -> None:
        """Create a full 3D cylindrical mesh."""
        ...

    def __init__(self, *args: Any) -> None: ...
    def type(self) -> CylindricalMeshType:
        """Mesh coordinate type."""
        ...

    def nr(self) -> int:
        """Number of cells in r-direction."""
        ...

    def ntheta(self) -> int:
        """Number of cells in theta-direction."""
        ...

    def nz(self) -> int:
        """Number of cells in z-direction."""
        ...

    def dr(self) -> float:
        """Cell spacing in r-direction."""
        ...

    def dtheta(self) -> float:
        """Cell spacing in theta-direction."""
        ...

    def dz(self) -> float:
        """Cell spacing in z-direction."""
        ...

    def rmin(self) -> float:
        """Minimum radial coordinate."""
        ...

    def rmax(self) -> float:
        """Maximum radial coordinate."""
        ...

    def zmin(self) -> float:
        """Minimum axial coordinate."""
        ...

    def zmax(self) -> float:
        """Maximum axial coordinate."""
        ...

    def num_cells(self) -> int:
        """Total number of cells."""
        ...

    def num_nodes(self) -> int:
        """Total number of nodes."""
        ...

    def is_radial(self) -> bool:
        """True if 1D radial mesh."""
        ...

    def is_axisymmetric(self) -> bool:
        """True if 2D axisymmetric (r-z) mesh."""
        ...

    def is_3d(self) -> bool:
        """True if full 3D cylindrical mesh."""
        ...

    def has_axis_singularity(self) -> bool:
        """True if mesh includes r=0 axis."""
        ...

    def r(self) -> FloatArray:
        """Radial coordinates."""
        ...

    def theta(self) -> FloatArray:
        """Angular coordinates."""
        ...

    def z(self) -> FloatArray:
        """Axial coordinates."""
        ...

    def x(self) -> FloatArray:
        """Cartesian x coordinates."""
        ...

    def y(self) -> FloatArray:
        """Cartesian y coordinates."""
        ...

    def index(self, i: int, j: int, k: int = 0) -> int:
        """Convert indices to linear index."""
        ...

    def cell_area(self, i: int) -> float:
        """Area of cell at radial index i."""
        ...

    def cell_volume(self, i: int, j: int = 0) -> float:
        """Volume of cell at indices (i, j)."""
        ...

    def cross_section_area(self, i: int) -> float:
        """Cross-section area at radial index i."""
        ...

    def gradient_r(self, field: FloatArray) -> FloatArray:
        """Compute radial gradient of field."""
        ...

    def gradient_z(self, field: FloatArray) -> FloatArray:
        """Compute axial gradient of field."""
        ...

    def laplacian(self, field: FloatArray) -> FloatArray:
        """Compute Laplacian of field."""
        ...

    def divergence(self, vr: FloatArray, vz: FloatArray) -> FloatArray:
        """Compute divergence of velocity field."""
        ...

# =============================================================================
# Transport Problem Builder
# =============================================================================

class TransportProblem:
    """Fluent builder for configuring transport problems.

    TransportProblem provides a declarative interface for setting up
    advection-diffusion-reaction problems. Use method chaining to
    configure all aspects of the problem, then pass to a solver.

    The general transport equation solved is:
        ∂u/∂t + v·∇u = ∇·(D∇u) + R(u) + S

    where:
    - u is the concentration/temperature field
    - v is the velocity field (advection)
    - D is the diffusivity
    - R(u) is the reaction term (decay, growth, Michaelis-Menten)
    - S is a source term

    Examples:
        >>> mesh = StructuredMesh(100, 0.0, 1.0)
        >>> problem = TransportProblem(mesh)
        >>> problem.diffusivity(1e-9) \\
        ...        .velocity(0.01, 0.0) \\
        ...        .linear_decay(0.1) \\
        ...        .initial_condition(1.0) \\
        ...        .dirichlet(Boundary.Left, 0.0) \\
        ...        .neumann(Boundary.Right, 0.0)
        >>> result = ExplicitFD().run(problem, t_end=10.0)
    """

    def diffusivity(self, D: float) -> TransportProblem:
        """Set uniform diffusivity."""
        ...

    def diffusivity_field(self, D: FloatArray) -> TransportProblem:
        """Set spatially-varying diffusivity field."""
        ...

    def velocity(self, vx: float, vy: float) -> TransportProblem:
        """Set uniform velocity field."""
        ...

    def velocity_field(self, vx: FloatArray, vy: FloatArray) -> TransportProblem:
        """Set spatially-varying velocity field."""
        ...

    def advection_scheme(self, scheme: AdvectionScheme) -> TransportProblem:
        """Set advection discretization scheme."""
        ...

    def initial_condition(
        self, u0: float | FloatArray | Callable[[float, float], float]
    ) -> TransportProblem:
        """Set initial condition."""
        ...

    def dirichlet(self, boundary: Boundary, value: float) -> TransportProblem:
        """Set Dirichlet boundary condition."""
        ...

    def neumann(self, boundary: Boundary, flux: float) -> TransportProblem:
        """Set Neumann boundary condition."""
        ...

    def robin(self, boundary: Boundary, alpha: float, beta: float) -> TransportProblem:
        """Set Robin (mixed) boundary condition: alpha*u + beta*du/dn = 0."""
        ...

    def boundary(self, boundary: Boundary, bc: BoundaryCondition) -> TransportProblem:
        """Set boundary condition using BoundaryCondition object."""
        ...

    def constant_source(self, S: float) -> TransportProblem:
        """Add constant source term."""
        ...

    def linear_decay(self, k: float) -> TransportProblem:
        """Add linear decay: R = -k*u."""
        ...

    def logistic_growth(self, r: float, K: float) -> TransportProblem:
        """Add logistic growth: R = r*u*(1 - u/K)."""
        ...

    def michaelis_menten(self, V_max: float, K_m: float) -> TransportProblem:
        """Add Michaelis-Menten consumption: R = -V_max*u/(K_m + u)."""
        ...

# =============================================================================
# Diffusion and Advection-Diffusion Solvers
# =============================================================================

class DiffusionSolver:
    """Solver for the diffusion equation.

    Solves the transient diffusion (heat) equation:
        ∂u/∂t = D ∇²u

    using explicit finite differences with automatic stable time stepping.

    Diffusion governs passive molecular transport, heat conduction,
    and many other spreading phenomena in biological systems.

    Examples:
        >>> mesh = StructuredMesh(100, 0.0, 1.0)
        >>> solver = DiffusionSolver(mesh, diffusivity=1e-9)
        >>> solver.set_initial_condition(initial_concentration)
        >>> solver.set_dirichlet_boundary(Boundary.Left, 1.0)
        >>> solver.set_neumann_boundary(Boundary.Right, 0.0)
        >>> solver.solve(dt=1e-3, num_steps=1000)
        >>> solution = solver.solution()
    """

    def __init__(self, mesh: StructuredMesh, diffusivity: float) -> None:
        """Create diffusion solver.

        Args:
            mesh: Computational mesh.
            diffusivity: Diffusion coefficient.
        """
        ...

    def set_initial_condition(self, u0: FloatArray) -> None:
        """Set initial concentration field."""
        ...

    def set_boundary_condition(self, boundary: Boundary, bc: BoundaryCondition) -> None:
        """Set boundary condition."""
        ...

    def set_dirichlet_boundary(self, boundary: Boundary, value: float) -> None:
        """Set Dirichlet boundary condition."""
        ...

    def set_neumann_boundary(self, boundary: Boundary, flux: float) -> None:
        """Set Neumann boundary condition."""
        ...

    def solve(self, dt: float, num_steps: int) -> None:
        """Advance solution in time."""
        ...

    def solution(self) -> FloatArray:
        """Current solution field."""
        ...

class AdvectionDiffusionSolver:
    """Solver for the advection-diffusion equation.

    Solves the transient advection-diffusion equation:
        ∂u/∂t + v·∇u = D ∇²u

    Multiple advection schemes are available:
    - UPWIND: First-order upwind, stable but diffusive
    - CENTRAL: Second-order central, accurate but may oscillate
    - QUICK: Third-order QUICK scheme
    - HYBRID: Automatic switching based on cell Peclet number

    The cell Peclet number Pe = |v|Δx/D determines the relative
    importance of advection vs diffusion.

    Examples:
        >>> mesh = StructuredMesh(100, 0.0, 1.0)
        >>> solver = AdvectionDiffusionSolver(
        ...     mesh, diffusivity=1e-9, vx=0.01, vy=0.0,
        ...     scheme=AdvectionScheme.UPWIND
        ... )
    """

    @overload
    def __init__(
        self,
        mesh: StructuredMesh,
        diffusivity: float,
        vx: float,
        vy: float = 0.0,
        scheme: AdvectionScheme = ...,
    ) -> None:
        """Create advection-diffusion solver with uniform velocity."""
        ...

    @overload
    def __init__(
        self,
        mesh: StructuredMesh,
        diffusivity: float,
        vx_field: list[float],
        vy_field: list[float],
        scheme: AdvectionScheme = ...,
    ) -> None:
        """Create advection-diffusion solver with velocity field."""
        ...

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def scheme(self) -> AdvectionScheme:
        """Current advection scheme."""
        ...

    def cell_peclet(self) -> float:
        """Cell Peclet number."""
        ...

    def max_time_step(self) -> float:
        """Maximum stable time step."""
        ...

    def is_scheme_stable(self, dt: float) -> bool:
        """Check if scheme is stable for given time step."""
        ...

    def set_scheme(self, scheme: AdvectionScheme) -> None:
        """Set advection discretization scheme."""
        ...

    def set_initial_condition(self, u0: FloatArray) -> None:
        """Set initial concentration field."""
        ...

    def set_boundary(self, boundary: Boundary, bc: BoundaryCondition) -> None:
        """Set boundary condition."""
        ...

    def solve(self, dt: float, num_steps: int) -> None:
        """Advance solution in time."""
        ...

    def solution(self) -> FloatArray:
        """Current solution field."""
        ...

# =============================================================================
# Reaction-Diffusion Solvers
# =============================================================================

class ReactionDiffusionSolver:
    """Base solver for reaction-diffusion equations.

    Solves the general reaction-diffusion equation:
        ∂u/∂t = D ∇²u + R(u)

    where R(u) is a reaction term. Specialized subclasses implement
    specific reaction kinetics:

    - LinearReactionDiffusionSolver: R = -ku (first-order decay)
    - LogisticReactionDiffusionSolver: R = ru(1-u/K) (logistic growth)
    - MichaelisMentenReactionDiffusionSolver: R = -Vₘₐₓu/(Kₘ+u)
    - ConstantSourceReactionDiffusionSolver: R = S (constant source)

    These models describe oxygen consumption, drug metabolism,
    cell proliferation, and enzyme kinetics.
    """

    def set_initial_condition(self, u0: FloatArray) -> None:
        """Set initial concentration field."""
        ...

    @overload
    def set_dirichlet_boundary(self, boundary_id: int, value: float) -> None:
        """Set Dirichlet BC using boundary index (0=left, 1=right, 2=bottom, 3=top)."""
        ...

    @overload
    def set_dirichlet_boundary(self, boundary: Boundary, value: float) -> None:
        """Set Dirichlet BC using Boundary enum."""
        ...

    @overload
    def set_neumann_boundary(self, boundary_id: int, flux: float) -> None:
        """Set Neumann BC using boundary index."""
        ...

    @overload
    def set_neumann_boundary(self, boundary: Boundary, flux: float) -> None:
        """Set Neumann BC using Boundary enum."""
        ...

    def set_boundary(self, boundary: Boundary | int, bc: BoundaryCondition) -> None:
        """Set boundary condition."""
        ...

    def solve(self, dt: float, num_steps: int) -> None:
        """Solve for specified number of time steps."""
        ...

    def solution(self) -> FloatArray:
        """Current solution field."""
        ...

class ConstantSourceReactionDiffusionSolver(ReactionDiffusionSolver):
    """Reaction-diffusion with constant source term."""

    def __init__(
        self, mesh: StructuredMesh, diffusivity: float, source_rate: float
    ) -> None:
        """Create solver with constant source."""
        ...

class LinearReactionDiffusionSolver(ReactionDiffusionSolver):
    """Reaction-diffusion with linear decay: R = -k*u."""

    def __init__(
        self, mesh: StructuredMesh, diffusivity: float, decay_rate: float
    ) -> None:
        """Create solver with decay constant k."""
        ...

class LogisticReactionDiffusionSolver(ReactionDiffusionSolver):
    """Reaction-diffusion with logistic growth: R = r*u*(1-u/K)."""

    def __init__(
        self,
        mesh: StructuredMesh,
        diffusivity: float,
        growth_rate: float,
        carrying_capacity: float,
    ) -> None:
        """Create solver with growth rate r and carrying capacity K."""
        ...

class MichaelisMentenReactionDiffusionSolver(ReactionDiffusionSolver):
    """Reaction-diffusion with Michaelis-Menten kinetics: R = -V_max*u/(K_m+u)."""

    def __init__(
        self,
        mesh: StructuredMesh,
        diffusivity: float,
        vmax: float,
        km: float,
    ) -> None:
        """Create solver with max rate vmax and half-saturation km."""
        ...

class MaskedMichaelisMentenReactionDiffusionSolver(ReactionDiffusionSolver):
    """Michaelis-Menten reaction-diffusion with spatial masking."""

    def __init__(
        self,
        mesh: StructuredMesh,
        diffusivity: float,
        vmax: float,
        km: float,
        mask: list[int],
        pinned_value: float,
    ) -> None:
        """Create solver with mask (1 = active, 0 = inactive)."""
        ...

class GrayScottSolver:
    """Gray-Scott reaction-diffusion pattern formation solver.

    Simulates the Gray-Scott model, a classic system for studying
    pattern formation through reaction-diffusion instabilities.

    Equations:
        ∂u/∂t = Du ∇²u - uv² + f(1-u)
        ∂v/∂t = Dv ∇²v + uv² - (f+k)v

    Parameters:
    - Du, Dv: Diffusion coefficients (typically Du > Dv)
    - f: Feed rate (replenishes u)
    - k: Kill rate (removes v)

    Different (f, k) values produce different patterns:
    - Spots, stripes, labyrinths, solitons
    - Traveling waves and oscillations
    - Turing patterns

    Examples:
        >>> mesh = StructuredMesh(128, 128, 0.0, 2.5, 0.0, 2.5)
        >>> solver = GrayScottSolver(mesh, Du=0.16, Dv=0.08, f=0.035, k=0.065)
        >>> # Initialize with u=1, v=0 and seed perturbation
        >>> result = solver.simulate(dt=1.0, steps=10000, save_every=100)
    """

    def __init__(
        self,
        mesh: StructuredMesh,
        Du: float,
        Dv: float,
        f: float,
        k: float,
    ) -> None:
        """Create Gray-Scott solver.

        Args:
            mesh: Computational mesh.
            Du, Dv: Diffusion coefficients for u and v.
            f: Feed rate.
            k: Kill rate.
        """
        ...

    def simulate(
        self, dt: float, steps: int, save_every: int = 1
    ) -> GrayScottRunResult:
        """Run simulation and save frames."""
        ...

# =============================================================================
# Fluid Flow Solvers
# =============================================================================

class StokesSolver:
    """Solver for steady Stokes (creeping) flow.

    Solves the incompressible Stokes equations:
        -∇p + μ∇²v = f
        ∇·v = 0

    for steady-state velocity v and pressure p, where μ is viscosity
    and f is a body force (e.g., gravity).

    Stokes flow applies when the Reynolds number Re << 1, meaning
    viscous forces dominate inertia. This is common in:
    - Microfluidics and lab-on-chip devices
    - Blood flow in capillaries
    - Swimming of microorganisms
    - Flow in porous media

    Examples:
        >>> mesh = StructuredMesh(50, 50, 0.0, 0.001, 0.0, 0.001)
        >>> solver = StokesSolver(mesh, viscosity=0.001)
        >>> solver.set_velocity_bc(Boundary.Top, VelocityBC.dirichlet(0.1, 0.0))
        >>> solver.set_velocity_bc(Boundary.Bottom, VelocityBC.no_slip())
        >>> result = solver.solve()
    """

    def __init__(self, mesh: StructuredMesh, viscosity: float) -> None:
        """Create Stokes solver.

        Args:
            mesh: Computational mesh.
            viscosity: Dynamic viscosity.
        """
        ...

    def viscosity(self) -> float:
        """Dynamic viscosity."""
        ...

    def reynolds(self) -> float:
        """Reynolds number (always 0 for Stokes flow)."""
        ...

    def set_velocity_bc(self, boundary: Boundary, bc: VelocityBC) -> None:
        """Set velocity boundary condition."""
        ...

    def set_body_force(self, fx: float, fy: float) -> None:
        """Set body force (e.g., gravity)."""
        ...

    def set_tolerance(self, tol: float) -> None:
        """Set convergence tolerance."""
        ...

    def set_max_iterations(self, max_iter: int) -> None:
        """Set maximum iterations."""
        ...

    def set_velocity_relaxation(self, omega: float) -> None:
        """Set velocity under-relaxation factor."""
        ...

    def set_pressure_relaxation(self, omega: float) -> None:
        """Set pressure under-relaxation factor."""
        ...

    def solve(self) -> StokesResult:
        """Solve for steady state velocity and pressure."""
        ...

class NavierStokesSolver:
    """Solver for unsteady incompressible Navier-Stokes equations.

    Solves the incompressible Navier-Stokes equations:
        ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v + f
        ∇·v = 0

    using a projection method with explicit time stepping.

    The Reynolds number Re = ρUL/μ characterizes the flow regime:
    - Re << 1: Creeping flow (use StokesSolver instead)
    - Re ~ 1-1000: Laminar flow
    - Re >> 1000: Turbulent flow (this solver not recommended)

    Features:
    - Automatic CFL-based time stepping
    - Multiple convection schemes (UPWIND, CENTRAL, QUICK)
    - Pressure Poisson equation solved iteratively

    Examples:
        >>> mesh = StructuredMesh(100, 50, 0.0, 0.01, 0.0, 0.005)
        >>> solver = NavierStokesSolver(mesh, density=1000.0, viscosity=0.001)
        >>> solver.set_velocity_bc(Boundary.Left, VelocityBC.inflow(0.1, 0.0))
        >>> solver.set_velocity_bc(Boundary.Right, VelocityBC.outflow())
        >>> result = solver.solve(t_end=0.1)
    """

    def __init__(self, mesh: StructuredMesh, density: float, viscosity: float) -> None:
        """Create Navier-Stokes solver.

        Args:
            mesh: Computational mesh.
            density: Fluid density.
            viscosity: Dynamic viscosity.
        """
        ...

    def density(self) -> float:
        """Fluid density."""
        ...

    def viscosity(self) -> float:
        """Dynamic viscosity."""
        ...

    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity (nu = mu/rho)."""
        ...

    def reynolds(self) -> float:
        """Reynolds number."""
        ...

    def set_velocity_bc(self, boundary: Boundary, bc: VelocityBC) -> None:
        """Set velocity boundary condition."""
        ...

    def set_body_force(self, fx: float, fy: float) -> None:
        """Set body force."""
        ...

    def set_initial_velocity(self, u: FloatArray, v: FloatArray) -> None:
        """Set initial velocity field."""
        ...

    def set_time_step(self, dt: float) -> None:
        """Set time step."""
        ...

    def set_cfl(self, cfl: float) -> None:
        """Set CFL number for automatic time stepping."""
        ...

    def set_convection_scheme(self, scheme: ConvectionScheme) -> None:
        """Set convection discretization scheme."""
        ...

    def set_pressure_tolerance(self, tol: float) -> None:
        """Set pressure solver tolerance."""
        ...

    def set_max_pressure_iterations(self, max_iter: int) -> None:
        """Set maximum pressure solver iterations."""
        ...

    def solve(self, t_end: float) -> NavierStokesResult:
        """Solve to specified end time."""
        ...

    def solve_steps(self, n_steps: int) -> NavierStokesResult:
        """Solve for specified number of time steps."""
        ...

class DarcyFlowSolver:
    """Solver for Darcy flow in porous media.

    Solves Darcy's law for flow through porous media:
        v = -(K/μ) ∇p
        ∇·v = 0

    where K is permeability, μ is viscosity, p is pressure, and v is
    the superficial (Darcy) velocity.

    Applications include:
    - Groundwater flow and contaminant transport
    - Blood flow through tissue (interstitial flow)
    - Flow in biological scaffolds
    - Drug transport in tumors

    The solver supports spatially-varying permeability for
    heterogeneous media.

    Examples:
        >>> mesh = StructuredMesh(50, 50, 0.0, 0.01, 0.0, 0.01)
        >>> solver = DarcyFlowSolver(mesh, kappa=1e-12)  # m²
        >>> solver.set_dirichlet(Boundary.Left, pressure=1000.0)
        >>> solver.set_dirichlet(Boundary.Right, pressure=0.0)
        >>> result = solver.solve()
    """

    @overload
    def __init__(self, mesh: StructuredMesh, kappa: float) -> None:
        """Create Darcy flow solver with uniform permeability."""
        ...

    @overload
    def __init__(self, mesh: StructuredMesh, kappa: list[float]) -> None:
        """Create Darcy flow solver with permeability field."""
        ...

    def __init__(self, *args: Any) -> None: ...
    def kappa(self) -> float:
        """Permeability."""
        ...

    def set_dirichlet(self, boundary: Boundary, pressure: float) -> None:
        """Set pressure Dirichlet boundary condition."""
        ...

    def set_neumann(self, boundary: Boundary, flux: float) -> None:
        """Set flux Neumann boundary condition."""
        ...

    def set_internal_pressure(self, i: int, j: int, pressure: float) -> None:
        """Set internal pressure constraint."""
        ...

    def set_initial_guess(self, p0: FloatArray) -> None:
        """Set initial pressure guess."""
        ...

    def set_tolerance(self, tol: float) -> None:
        """Set convergence tolerance."""
        ...

    def set_max_iterations(self, max_iter: int) -> None:
        """Set maximum iterations."""
        ...

    def set_omega(self, omega: float) -> None:
        """Set SOR relaxation factor."""
        ...

    def solve(self) -> DarcyFlowResult:
        """Solve for pressure and velocity fields."""
        ...

# =============================================================================
# Membrane Diffusion Solvers
# =============================================================================

class MembraneDiffusion1DSolver:
    """1D steady-state membrane diffusion solver.

    Solves for the steady-state concentration profile across a membrane
    and computes transport properties (flux, permeability).

    The steady-state diffusion equation in a membrane:
        d/dx(D dC/dx) = 0

    with boundary conditions set by left/right concentrations and
    partition coefficients.

    Features:
    - Hindered diffusion for large solutes in small pores
    - Renkin-Faxen correction for steric effects
    - Partition coefficient for membrane-solution equilibrium

    Examples:
        >>> solver = MembraneDiffusion1DSolver()
        >>> solver.set_membrane_thickness(100e-6)  # 100 μm
        >>> solver.set_diffusivity(1e-10)  # m²/s
        >>> solver.set_left_concentration(1.0)  # mol/m³
        >>> solver.set_right_concentration(0.0)
        >>> result = solver.solve()
        >>> print(f"Flux: {result.flux:.2e} mol/m²/s")
    """

    def __init__(self) -> None:
        """Create membrane diffusion solver with default parameters."""
        ...

    def membrane_thickness(self) -> float:
        """Membrane thickness."""
        ...

    def diffusivity(self) -> float:
        """Diffusion coefficient."""
        ...

    def partition_coefficient(self) -> float:
        """Partition coefficient."""
        ...

    def left_concentration(self) -> float:
        """Left boundary concentration."""
        ...

    def right_concentration(self) -> float:
        """Right boundary concentration."""
        ...

    def lambda_ratio(self) -> float:
        """Solute-to-pore radius ratio for hindered diffusion."""
        ...

    def is_hindered_diffusion(self) -> bool:
        """True if hindered diffusion is enabled."""
        ...

    def set_membrane_thickness(self, thickness: float) -> None:
        """Set membrane thickness."""
        ...

    def set_diffusivity(self, D: float) -> None:
        """Set diffusion coefficient."""
        ...

    def set_partition_coefficient(self, K: float) -> None:
        """Set partition coefficient."""
        ...

    def set_left_concentration(self, C: float) -> None:
        """Set left boundary concentration."""
        ...

    def set_right_concentration(self, C: float) -> None:
        """Set right boundary concentration."""
        ...

    def set_num_nodes(self, n: int) -> None:
        """Set number of grid nodes."""
        ...

    def set_hindered_diffusion(self, lambda_ratio: float) -> None:
        """Enable hindered diffusion with given lambda ratio."""
        ...

    def disable_hindered_diffusion(self) -> None:
        """Disable hindered diffusion."""
        ...

    def solve(self) -> MembraneDiffusionResult:
        """Solve for steady-state concentration profile."""
        ...

    def compute_flux(self) -> float:
        """Compute steady-state flux."""
        ...

    def compute_permeability(self) -> float:
        """Compute membrane permeability."""
        ...

class MultiLayerMembraneSolver:
    """Multi-layer membrane diffusion solver.

    Solves steady-state diffusion through a composite membrane
    consisting of multiple layers with different properties.

    The total resistance is the sum of individual layer resistances:
        R_total = Σ(L_i / (D_i · K_i))

    Each layer can have different:
    - Thickness
    - Diffusivity
    - Partition coefficient

    Applications include:
    - Drug-eluting stent coatings
    - Skin permeation (stratum corneum + viable epidermis)
    - Controlled release devices

    Examples:
        >>> solver = MultiLayerMembraneSolver()
        >>> solver.add_layer(thickness=10e-6, diffusivity=1e-11, partition=0.5)
        >>> solver.add_layer(thickness=50e-6, diffusivity=1e-10, partition=1.0)
        >>> solver.set_left_concentration(1.0)
        >>> solver.set_right_concentration(0.0)
        >>> result = solver.solve()
    """

    def __init__(self) -> None:
        """Create multi-layer membrane solver."""
        ...

    def num_layers(self) -> int:
        """Number of membrane layers."""
        ...

    def total_thickness(self) -> float:
        """Total membrane thickness."""
        ...

    def add_layer(
        self, thickness: float, diffusivity: float, partition: float = 1.0
    ) -> None:
        """Add a membrane layer."""
        ...

    def clear_layers(self) -> None:
        """Remove all layers."""
        ...

    def set_left_concentration(self, C: float) -> None:
        """Set left boundary concentration."""
        ...

    def set_right_concentration(self, C: float) -> None:
        """Set right boundary concentration."""
        ...

    def solve(self) -> MembraneDiffusionResult:
        """Solve for steady-state concentration profile."""
        ...

# =============================================================================
# Application-Specific Solvers
# =============================================================================

class TumorDrugDeliverySolver:
    """Coupled solver for tumor drug delivery.

    Models drug transport in tumor tissue including:
    - Interstitial fluid flow (pressure-driven convection)
    - Drug diffusion in interstitial space
    - Binding to extracellular matrix
    - Cellular uptake

    The tumor microenvironment is characterized by:
    - Elevated interstitial fluid pressure (IFP)
    - Heterogeneous hydraulic conductivity
    - Tortuous diffusion paths

    This model helps predict drug penetration and distribution
    for optimizing cancer therapy.

    Examples:
        >>> mesh = StructuredMesh(50, 50, 0.0, 0.01, 0.0, 0.01)
        >>> tumor_mask = create_circular_tumor(mesh, center, radius)
        >>> solver = TumorDrugDeliverySolver(
        ...     mesh, tumor_mask, hydraulic_conductivity,
        ...     p_boundary=0.0, p_tumor=2000.0  # Pa
        ... )
        >>> result = solver.simulate(dt=1.0, t_end=3600.0)  # 1 hour
    """

    def __init__(
        self,
        mesh: StructuredMesh,
        tumor_mask: list[int],
        hydraulic_conductivity: list[float],
        p_boundary: float,
        p_tumor: float,
    ) -> None:
        """Create tumor drug delivery solver.

        Args:
            mesh: Computational mesh.
            tumor_mask: Binary mask indicating tumor cells.
            hydraulic_conductivity: Hydraulic conductivity field.
            p_boundary: Boundary pressure.
            p_tumor: Tumor pressure.
        """
        ...

    def simulate(
        self,
        dt: float,
        t_end: float,
        save_every: int = 1,
        C_initial: float = 0.0,
    ) -> TumorDrugDeliverySaved:
        """Run coupled simulation."""
        ...

    def solve_pressure_sor(
        self, tol: float = 1e-6, max_iter: int = 10000
    ) -> tuple[FloatArray, int]:
        """Solve pressure field with SOR."""
        ...

class BioheatCryotherapySolver:
    """Coupled bioheat-cryotherapy solver with tissue damage.

    Simulates cryoablation therapy using the Pennes bioheat equation
    with phase change and Arrhenius tissue damage kinetics.

    The model includes:
    - Heat conduction with temperature-dependent properties
    - Blood perfusion heat source (with perfusion shutdown in frozen tissue)
    - Metabolic heat generation
    - Phase change (freezing/thawing) with latent heat
    - Arrhenius damage integral: Ω = ∫A·exp(-Ea/RT)dt

    Tissue is considered destroyed when Ω > 1 (63% cell death).

    Applications:
    - Cryosurgery planning for liver, prostate, kidney tumors
    - Cryopreservation analysis
    - Thermal ablation optimization

    Examples:
        >>> # Set up mesh with probe region
        >>> mesh = StructuredMesh(100, 100, -0.02, 0.02, -0.02, 0.02)
        >>> probe_mask = create_probe_region(mesh)
        >>> solver = BioheatCryotherapySolver(
        ...     mesh, probe_mask, perfusion_map, q_met_map,
        ...     rho_tissue=1000, rho_blood=1000, c_blood=3600,
        ...     k_unfrozen=0.5, k_frozen=2.0, ...
        ... )
        >>> result = solver.simulate(dt=0.1, t_end=600.0)  # 10 min
    """

    def __init__(
        self,
        mesh: StructuredMesh,
        probe_mask: list[int],
        perfusion_map: list[float],
        q_met_map: list[float],
        rho_tissue: float,
        rho_blood: float,
        c_blood: float,
        k_unfrozen: float,
        k_frozen: float,
        c_unfrozen: float,
        c_frozen: float,
        T_body: float,
        T_probe: float,
        T_freeze: float,
        T_freeze_range: float,
        L_fusion: float,
        A: float,
        E_a: float,
        R_gas: float,
    ) -> None:
        """Create bioheat cryotherapy solver."""
        ...

    def simulate(
        self,
        dt: float,
        t_end: float,
        save_every: int = 1,
        T_initial: float = 310.15,
    ) -> BioheatSaved:
        """Run coupled thermal-damage simulation."""
        ...

# =============================================================================
# Viscosity Models (Rheology)
# =============================================================================

class NewtonianModel:
    """Newtonian (constant viscosity) fluid model.

    For a Newtonian fluid, the shear stress is linearly proportional
    to the shear rate:
        τ = μ · γ̇

    where μ is the constant dynamic viscosity.

    Most simple fluids (water, air, simple organic solvents) exhibit
    Newtonian behavior. This model serves as a baseline for comparison
    with non-Newtonian models.

    Examples:
        >>> model = NewtonianModel(mu=0.001)  # Water at 20°C
        >>> tau = model.shear_stress(100.0)  # Stress at γ̇ = 100 s⁻¹
        >>> print(f"Shear stress: {tau:.2f} Pa")
    """

    def __init__(self, mu: float) -> None:
        """Create Newtonian model with viscosity mu."""
        ...

    def mu0(self) -> float:
        """Viscosity."""
        ...

    def name(self) -> str:
        """Model name."""
        ...

    def type(self) -> ViscosityModel:
        """Model type enum."""
        ...

    def viscosity(self, gamma_dot: float) -> float:
        """Viscosity at given shear rate."""
        ...

    def shear_stress(self, gamma_dot: float) -> float:
        """Shear stress at given shear rate."""
        ...

class PowerLawModel:
    """Power-law (Ostwald-de Waele) fluid model.

    The power-law model relates shear stress to shear rate:
        τ = K · γ̇ⁿ

    and the apparent viscosity is:
        η(γ̇) = K · γ̇^(n-1)

    Parameters:
    - K: Consistency index (Pa·sⁿ)
    - n: Flow behavior index (dimensionless)

    Behavior:
    - n < 1: Shear-thinning (pseudoplastic) - e.g., blood, polymer solutions
    - n = 1: Newtonian
    - n > 1: Shear-thickening (dilatant) - e.g., cornstarch suspensions

    Limitation: Predicts infinite viscosity as γ̇ → 0 for n < 1.
    For more realistic behavior at low shear rates, use Carreau model.

    Examples:
        >>> # Blood at moderate shear rates
        >>> model = PowerLawModel(K=0.42, n=0.61)
        >>> print(f"Shear-thinning: {model.is_shear_thinning()}")
    """

    def __init__(self, K: float, n: float) -> None:
        """Create power-law model with consistency K and index n."""
        ...

    def K(self) -> float:
        """Consistency index."""
        ...

    def n(self) -> float:
        """Flow behavior index."""
        ...

    def is_shear_thinning(self) -> bool:
        """True if n < 1."""
        ...

    def is_shear_thickening(self) -> bool:
        """True if n > 1."""
        ...

    def name(self) -> str:
        """Model name."""
        ...

    def type(self) -> ViscosityModel:
        """Model type enum."""
        ...

    def viscosity(self, gamma_dot: float) -> float:
        """Viscosity at given shear rate."""
        ...

    def shear_stress(self, gamma_dot: float) -> float:
        """Shear stress at given shear rate."""
        ...

class CarreauModel:
    """Carreau viscosity model for shear-thinning fluids.

    The Carreau model provides a smooth transition between Newtonian
    plateaus at low and high shear rates:

        η(γ̇) = η∞ + (η₀ - η∞) · [1 + (λγ̇)²]^((n-1)/2)

    Parameters:
    - η₀ (mu0): Zero-shear viscosity (Pa·s)
    - η∞ (mu_inf): Infinite-shear viscosity (Pa·s)
    - λ (lambda_): Relaxation time (s)
    - n: Power-law index (dimensionless)

    This model is widely used for blood and polymer solutions because
    it captures realistic behavior across all shear rate ranges.

    For blood, typical parameters:
    - η₀ ≈ 0.056 Pa·s (at H=0.45)
    - η∞ ≈ 0.00345 Pa·s
    - λ ≈ 3.31 s
    - n ≈ 0.357

    Examples:
        >>> # Create Carreau model for blood
        >>> model = blood_carreau_model(hematocrit=0.45)
        >>> eta = model.viscosity(100.0)  # At γ̇ = 100 s⁻¹
    """

    def __init__(self, mu0: float, mu_inf: float, lambda_: float, n: float) -> None:
        """Create Carreau model.

        Args:
            mu0: Zero-shear viscosity.
            mu_inf: Infinite-shear viscosity.
            lambda_: Relaxation time.
            n: Power-law index.
        """
        ...

    def mu0(self) -> float:
        """Zero-shear viscosity."""
        ...

    def mu_inf(self) -> float:
        """Infinite-shear viscosity."""
        ...

    def lambda_(self) -> float:
        """Relaxation time."""
        ...

    def n(self) -> float:
        """Power-law index."""
        ...

    def name(self) -> str:
        """Model name."""
        ...

    def type(self) -> ViscosityModel:
        """Model type enum."""
        ...

    def viscosity(self, gamma_dot: float) -> float:
        """Viscosity at given shear rate."""
        ...

    def shear_stress(self, gamma_dot: float) -> float:
        """Shear stress at given shear rate."""
        ...

class CarreauYasudaModel:
    """Carreau-Yasuda viscosity model.

    An extension of the Carreau model with an additional parameter
    to control the transition region:

        η(γ̇) = η∞ + (η₀ - η∞) · [1 + (λγ̇)^a]^((n-1)/a)

    Parameters:
    - η₀ (mu0): Zero-shear viscosity (Pa·s)
    - η∞ (mu_inf): Infinite-shear viscosity (Pa·s)
    - λ (lambda_): Relaxation time (s)
    - a: Transition parameter (dimensionless)
    - n: Power-law index (dimensionless)

    When a = 2, this reduces to the standard Carreau model.
    The parameter 'a' controls the breadth of the transition region
    between Newtonian and power-law behavior.

    Examples:
        >>> model = CarreauYasudaModel(
        ...     mu0=0.056, mu_inf=0.0035, lambda_=3.31, a=1.25, n=0.357
        ... )
        >>> eta = model.viscosity(50.0)
    """

    def __init__(
        self, mu0: float, mu_inf: float, lambda_: float, a: float, n: float
    ) -> None:
        """Create Carreau-Yasuda model.

        Args:
            mu0: Zero-shear viscosity.
            mu_inf: Infinite-shear viscosity.
            lambda_: Relaxation time.
            a: Transition parameter.
            n: Power-law index.
        """
        ...

    def mu0(self) -> float:
        """Zero-shear viscosity."""
        ...

    def mu_inf(self) -> float:
        """Infinite-shear viscosity."""
        ...

    def lambda_(self) -> float:
        """Relaxation time."""
        ...

    def a(self) -> float:
        """Transition parameter."""
        ...

    def n(self) -> float:
        """Power-law index."""
        ...

    def name(self) -> str:
        """Model name."""
        ...

    @property
    def type(self) -> ViscosityModel:
        """Model type enum."""
        ...

    def viscosity(self, gamma_dot: float) -> float:
        """Viscosity at given shear rate."""
        ...

    def shear_stress(self, gamma_dot: float) -> float:
        """Shear stress at given shear rate."""
        ...

class CrossModel:
    """Cross viscosity model.

    An alternative to the Carreau model for shear-thinning fluids:

        η(γ̇) = η∞ + (η₀ - η∞) / [1 + (K·γ̇)^m]

    Parameters:
    - η₀ (mu0): Zero-shear viscosity (Pa·s)
    - η∞ (mu_inf): Infinite-shear viscosity (Pa·s)
    - K: Cross time constant (s)
    - m: Cross rate constant (dimensionless)

    The Cross model is mathematically simpler than Carreau and
    often provides adequate fits for polymer solutions and
    biological fluids.

    Examples:
        >>> model = CrossModel(mu0=0.1, mu_inf=0.003, K=2.0, m=0.8)
        >>> eta = model.viscosity(10.0)
    """

    def __init__(self, mu0: float, mu_inf: float, K: float, m: float) -> None:
        """Create Cross model.

        Args:
            mu0: Zero-shear viscosity.
            mu_inf: Infinite-shear viscosity.
            K: Cross time constant.
            m: Cross rate constant.
        """
        ...

    def mu0(self) -> float:
        """Zero-shear viscosity."""
        ...

    def mu_inf(self) -> float:
        """Infinite-shear viscosity."""
        ...

    def K(self) -> float:
        """Cross time constant."""
        ...

    def m(self) -> float:
        """Cross rate constant."""
        ...

    def name(self) -> str:
        """Model name."""
        ...

    def type(self) -> ViscosityModel:
        """Model type enum."""
        ...

    def viscosity(self, gamma_dot: float) -> float:
        """Viscosity at given shear rate."""
        ...

    def shear_stress(self, gamma_dot: float) -> float:
        """Shear stress at given shear rate."""
        ...

class CassonModel:
    """Casson viscosity model for blood.

    The Casson model captures yield stress behavior of blood:

        √τ = √τ_y + √(μ_p · γ̇)    for τ > τ_y
        γ̇ = 0                      for τ ≤ τ_y

    Parameters:
    - τ_y (tau_y): Yield stress (Pa)
    - μ_p (mu_p): Casson plastic viscosity (Pa·s)

    Blood exhibits a yield stress due to red blood cell aggregation
    at low shear rates. The Casson model is particularly good for
    describing blood flow in small vessels where shear rates are low.

    Typical values for blood at H=0.45:
    - τ_y ≈ 0.005-0.01 Pa
    - μ_p ≈ 0.003-0.004 Pa·s

    Examples:
        >>> model = blood_casson_model(hematocrit=0.45)
        >>> # Check if flow will occur under given stress
        >>> tau = 0.01  # Pa
        >>> if tau > model.yield_stress():
        ...     print("Flow will occur")
    """

    def __init__(self, tau_y: float, mu_p: float) -> None:
        """Create Casson model.

        Args:
            tau_y: Yield stress.
            mu_p: Plastic viscosity.
        """
        ...

    def yield_stress(self) -> float:
        """Yield stress."""
        ...

    def plastic_viscosity(self) -> float:
        """Plastic viscosity."""
        ...

    def name(self) -> str:
        """Model name."""
        ...

    def type(self) -> ViscosityModel:
        """Model type enum."""
        ...

    def viscosity(self, gamma_dot: float) -> float:
        """Apparent viscosity at given shear rate."""
        ...

    def shear_stress(self, gamma_dot: float) -> float:
        """Shear stress at given shear rate."""
        ...

class BinghamModel:
    """Bingham plastic model.

    The Bingham model describes a fluid with a yield stress:

        τ = τ_y + μ_p · γ̇    for τ > τ_y
        γ̇ = 0                for τ ≤ τ_y

    Parameters:
    - τ_y (tau_y): Yield stress (Pa)
    - μ_p (mu_p): Plastic viscosity (Pa·s)

    Unlike the Casson model, the Bingham model has a linear
    relationship between stress and shear rate above the yield point.

    Applications:
    - Toothpaste, mayonnaise, some gels
    - Drilling muds, cement slurries
    - Approximate blood behavior

    The Bingham number Bn = τ_y·L/(μ_p·U) characterizes the
    importance of yield stress relative to viscous effects.

    Examples:
        >>> model = BinghamModel(tau_y=5.0, mu_p=0.1)
        >>> Bn = model.bingham_number(L=0.01, U=0.1)
        >>> print(f"Bingham number: {Bn:.2f}")
    """

    def __init__(self, tau_y: float, mu_p: float) -> None:
        """Create Bingham model.

        Args:
            tau_y: Yield stress.
            mu_p: Plastic viscosity.
        """
        ...

    def yield_stress(self) -> float:
        """Yield stress."""
        ...

    def plastic_viscosity(self) -> float:
        """Plastic viscosity."""
        ...

    def name(self) -> str:
        """Model name."""
        ...

    def type(self) -> ViscosityModel:
        """Model type enum."""
        ...

    def bingham_number(self, L: float, U: float) -> float:
        """Compute Bingham number Bn = tau_y * L / (mu_p * U)."""
        ...

    def viscosity(self, gamma_dot: float) -> float:
        """Apparent viscosity at given shear rate."""
        ...

    def shear_stress(self, gamma_dot: float) -> float:
        """Shear stress at given shear rate."""
        ...

class HerschelBulkleyModel:
    """Herschel-Bulkley viscosity model.

    A generalized model combining yield stress with power-law behavior:

        τ = τ_y + K · γ̇ⁿ    for τ > τ_y
        γ̇ = 0                for τ ≤ τ_y

    Parameters:
    - τ_y (tau_y): Yield stress (Pa)
    - K: Consistency index (Pa·sⁿ)
    - n: Flow behavior index (dimensionless)

    This model reduces to:
    - Bingham model when n = 1
    - Power-law model when τ_y = 0
    - Newtonian when n = 1 and τ_y = 0

    The Herschel-Bulkley model is versatile and can describe
    many complex fluids including blood, food products, and
    drilling fluids.

    Examples:
        >>> # Shear-thinning fluid with yield stress
        >>> model = HerschelBulkleyModel(tau_y=2.0, K=0.5, n=0.7)
        >>> tau = model.shear_stress(100.0)
    """

    def __init__(self, tau_y: float, K: float, n: float) -> None:
        """Create Herschel-Bulkley model.

        Args:
            tau_y: Yield stress.
            K: Consistency index.
            n: Flow behavior index.
        """
        ...

    def yield_stress(self) -> float:
        """Yield stress."""
        ...

    def K(self) -> float:
        """Consistency index."""
        ...

    def n(self) -> float:
        """Flow behavior index."""
        ...

    def name(self) -> str:
        """Model name."""
        ...

    def type(self) -> ViscosityModel:
        """Model type enum."""
        ...

    def viscosity(self, gamma_dot: float) -> float:
        """Apparent viscosity at given shear rate."""
        ...

    def shear_stress(self, gamma_dot: float) -> float:
        """Shear stress at given shear rate."""
        ...

# =============================================================================
# Utility Functions
# =============================================================================

def blood_carreau_model(hematocrit: float) -> CarreauModel:
    """Create Carreau model for blood at given hematocrit (0-0.7).

    Uses empirical correlations for blood viscosity parameters.
    """
    ...

def blood_casson_model(hematocrit: float) -> CassonModel:
    """Create Casson model for blood at given hematocrit (0-0.7).

    Uses empirical correlations for blood viscosity parameters.
    """
    ...

def pipe_wall_shear_rate(Q: float, R: float) -> float:
    """Wall shear rate in pipe flow: gamma_w = 4Q/(pi*R^3).

    Args:
        Q: Volumetric flow rate.
        R: Pipe radius.

    Returns:
        Wall shear rate.
    """
    ...

def renkin_hindrance(lambda_ratio: float) -> float:
    """Compute Renkin hindrance factor H for spherical solute in cylindrical pore.

    H = (1-λ)² × (1 - 2.104λ + 2.09λ³ - 0.95λ⁵)

    where λ = solute_radius / pore_radius.

    Args:
        lambda_ratio: Ratio of solute radius to pore radius.

    Returns:
        Hindrance factor (0-1).
    """
    ...

# =============================================================================
# Submodules
# =============================================================================

class analytical:
    """Analytical solutions for validation and benchmarking.

    This module provides closed-form analytical solutions for
    fundamental transport and flow problems. These solutions are
    essential for:

    - Validating numerical solver implementations
    - Benchmarking solver accuracy and convergence
    - Quick estimates and order-of-magnitude calculations
    - Educational demonstrations

    Categories of solutions:

    **Diffusion/Heat Transfer:**
    - Semi-infinite diffusion (erfc solution)
    - First-order decay kinetics
    - Logistic growth dynamics
    - Lumped capacitance transients

    **Fluid Mechanics:**
    - Poiseuille (pipe) flow profiles
    - Couette (shear) flow
    - Taylor-Couette (rotating cylinders)
    - Bernoulli velocity

    **Viscoelasticity:**
    - Maxwell stress relaxation
    - Kelvin-Voigt creep
    - Standard linear solid (SLS) models
    - Burgers model creep
    - Complex modulus calculations

    Examples:
        >>> # Validate diffusion solver against analytical solution
        >>> C_analytical = analytical.diffusion_1d_semi_infinite(
        ...     x=0.001, t=100.0, D=1e-9, C0=1.0
        ... )
        >>> # Check Poiseuille flow centerline velocity
        >>> u_max = analytical.poiseuille_max_velocity(
        ...     R=0.002, dp_dx=-1000.0, mu=0.003
        ... )
    """

    @staticmethod
    def diffusion_1d_semi_infinite(x: float, t: float, D: float, C0: float) -> float:
        """Semi-infinite diffusion: C = C0 * erfc(x / (2*sqrt(D*t)))."""
        ...

    @staticmethod
    def diffusion_penetration_depth(D: float, t: float) -> float:
        """Penetration depth: delta = sqrt(4*D*t)."""
        ...

    @staticmethod
    def first_order_decay(C0: float, k: float, t: float) -> float:
        """First-order decay: C = C0 * exp(-k*t)."""
        ...

    @staticmethod
    def logistic_growth(u0: float, r: float, K: float, t: float) -> float:
        """Logistic growth: u = K / (1 + (K/u0 - 1)*exp(-r*t))."""
        ...

    @staticmethod
    def lumped_exponential(T0: float, T_inf: float, tau: float, t: float) -> float:
        """Lumped-capacitance: T = T_inf + (T0 - T_inf)*exp(-t/tau)."""
        ...

    @staticmethod
    def poiseuille_velocity(r: float, R: float, dp_dx: float, mu: float) -> float:
        """Poiseuille velocity profile: u(r) = (1/(4*mu))*(-dp/dx)*(R² - r²)."""
        ...

    @staticmethod
    def poiseuille_max_velocity(R: float, dp_dx: float, mu: float) -> float:
        """Maximum velocity in Poiseuille flow."""
        ...

    @staticmethod
    def poiseuille_flow_rate(R: float, dp_dx: float, mu: float) -> float:
        """Volumetric flow rate in Poiseuille flow: Q = pi*R⁴*(-dp/dx)/(8*mu)."""
        ...

    @staticmethod
    def poiseuille_wall_shear(R: float, dp_dx: float) -> float:
        """Wall shear stress in Poiseuille flow: tau_w = R*(-dp/dx)/2."""
        ...

    @staticmethod
    def couette_velocity(y: float, H: float, U: float) -> float:
        """Couette velocity profile: u(y) = U*y/H."""
        ...

    @staticmethod
    def couette_max_velocity(U: float) -> float:
        """Maximum velocity in Couette flow."""
        ...

    @staticmethod
    def taylor_couette_velocity(
        r: float, R1: float, R2: float, omega1: float, omega2: float
    ) -> float:
        """Taylor-Couette azimuthal velocity."""
        ...

    @staticmethod
    def taylor_couette_torque(
        R1: float, R2: float, omega1: float, omega2: float, mu: float, L: float
    ) -> float:
        """Torque in Taylor-Couette flow."""
        ...

    @staticmethod
    def bernoulli_velocity(p1: float, p2: float, rho: float) -> float:
        """Velocity from Bernoulli equation: v = sqrt(2*(p1-p2)/rho)."""
        ...

    @staticmethod
    def maxwell_relaxation(tau: float, t: float, G: float, epsilon0: float) -> float:
        """Maxwell stress relaxation: sigma(t) = G*epsilon0*exp(-t/tau)."""
        ...

    @staticmethod
    def maxwell_relaxation_time(eta: float, G: float) -> float:
        """Maxwell relaxation time: tau = eta/G."""
        ...

    @staticmethod
    def kelvin_voigt_creep(t: float, tau: float, G: float, sigma0: float) -> float:
        """Kelvin-Voigt creep: epsilon(t) = (sigma0/G)*(1 - exp(-t/tau))."""
        ...

    @staticmethod
    def sls_relaxation(
        t: float, tau: float, G0: float, G_inf: float, epsilon0: float
    ) -> float:
        """Standard linear solid stress relaxation."""
        ...

    @staticmethod
    def sls_creep(
        t: float, tau: float, J0: float, J_inf: float, sigma0: float
    ) -> float:
        """Standard linear solid creep."""
        ...

    @staticmethod
    def burgers_creep(
        t: float, G1: float, G2: float, eta1: float, eta2: float, sigma0: float
    ) -> float:
        """Burgers model creep response."""
        ...

    @staticmethod
    def burgers_compliance(
        t: float, G1: float, G2: float, eta1: float, eta2: float
    ) -> float:
        """Burgers model compliance."""
        ...

    @staticmethod
    def complex_modulus_magnitude(G_prime: float, G_double_prime: float) -> float:
        """Complex modulus magnitude: |G*| = sqrt(G'² + G''²)."""
        ...

    @staticmethod
    def phase_angle(G_prime: float, G_double_prime: float) -> float:
        """Phase angle: delta = atan(G''/G')."""
        ...

    @staticmethod
    def loss_tangent(G_prime: float, G_double_prime: float) -> float:
        """Loss tangent: tan(delta) = G''/G'."""
        ...

class dimensionless:
    """Dimensionless number calculations for transport analysis.

    Dimensionless numbers characterize the relative importance of
    different physical phenomena and are essential for:

    - Scaling analysis and similitude
    - Regime identification (laminar vs turbulent, etc.)
    - Validating solver stability and accuracy
    - Comparing different physical systems

    **Fluid Mechanics:**
    - Reynolds (Re): Inertia vs viscous forces

    **Mass Transfer:**
    - Peclet (Pe): Convection vs diffusion
    - Schmidt (Sc): Momentum vs mass diffusivity
    - Sherwood (Sh): Convective vs diffusive mass transfer

    **Heat Transfer:**
    - Biot (Bi): Surface vs internal thermal resistance
    - Fourier (Fo): Heat diffusion scaling

    These numbers guide solver selection:
    - Pe >> 1: Convection-dominated, may need upwinding
    - Pe << 1: Diffusion-dominated, central differences OK
    - Bi < 0.1: Lumped capacitance valid

    Examples:
        >>> # Check if flow is turbulent
        >>> Re = dimensionless.reynolds(rho=1000, U=1.0, L=0.01, mu=0.003)
        >>> print(f"Re = {Re:.0f}, {'turbulent' if Re > 2300 else 'laminar'}")
        >>>
        >>> # Check grid Peclet number for stability
        >>> Pe_grid = dimensionless.peclet(U=0.01, L=dx, D=1e-9)
        >>> if Pe_grid > 2:
        ...     print("Use upwind scheme for stability")
    """

    @staticmethod
    def reynolds(rho: float, U: float, L: float, mu: float) -> float:
        """Reynolds number: Re = rho*U*L/mu."""
        ...

    @staticmethod
    def reynolds_kinematic(U: float, L: float, nu: float) -> float:
        """Reynolds number using kinematic viscosity: Re = U*L/nu."""
        ...

    @staticmethod
    def peclet(U: float, L: float, D: float) -> float:
        """Peclet number: Pe = U*L/D."""
        ...

    @staticmethod
    def schmidt(nu: float, D: float) -> float:
        """Schmidt number: Sc = nu/D."""
        ...

    @staticmethod
    def schmidt_kinematic(mu: float, rho: float, D: float) -> float:
        """Schmidt number: Sc = mu/(rho*D)."""
        ...

    @staticmethod
    def sherwood(k_c: float, L: float, D: float) -> float:
        """Sherwood number: Sh = k_c*L/D."""
        ...

    @staticmethod
    def biot(h: float, L: float, k: float) -> float:
        """Biot number: Bi = h*L/k."""
        ...

    @staticmethod
    def fourier(alpha: float, t: float, L: float) -> float:
        """Fourier number: Fo = alpha*t/L²."""
        ...

    @staticmethod
    def is_lumped_valid(Bi: float) -> bool:
        """Check if lumped capacitance is valid: Bi < 0.1."""
        ...

    @staticmethod
    def is_convection_dominated(Pe: float) -> bool:
        """Check if convection dominates: Pe > 1."""
        ...
