"""Type stubs for biotransport._core._core extension module.

This file provides type hints for IDE autocompletion and static type checking.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Optional, Sequence, TypeAlias, Union, overload
from typing_extensions import Self

from biotransport.results import Result

import numpy as np
import numpy.typing as npt

# Type aliases
ArrayLike: TypeAlias = npt.NDArray[np.float64]
FloatArray: TypeAlias = npt.NDArray[np.float64]
Float32Array: TypeAlias = npt.NDArray[np.float32]

def native_build_info() -> dict[str, Any]: ...
def _high_order_laplacian_1d(field: ArrayLike, dx: float, order: int) -> FloatArray: ...
def _high_order_laplacian_2d(
    field: ArrayLike, nx: int, ny: int, dx: float, dy: float, order: int
) -> FloatArray: ...
def _high_order_gradient_1d(field: ArrayLike, dx: float, order: int) -> FloatArray: ...
def _high_order_stable_dt(
    diffusivity: float,
    dx: float,
    dy: float,
    order: int,
    safety_factor: float,
    is_2d: bool,
) -> float: ...
def _solve_high_order_diffusion(
    initial: ArrayLike,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    diffusivity: float,
    order: int,
    safety_factor: float,
    end_time: float,
    dt: Optional[float] = None,
    left: float = 0.0,
    right: float = 0.0,
    bottom: float = 0.0,
    top: float = 0.0,
    callback: Optional[Callable[[float, FloatArray], None]] = None,
) -> dict[str, Any]: ...
def _integrate_explicit_runge_kutta(
    initial: ArrayLike,
    rhs: Callable[..., ArrayLike],
    initial_time: float,
    end_time: float,
    dt: float,
    method: str = "rk4",
    autonomous: bool = False,
    maximum_steps: int = 10_000_000,
) -> dict[str, Any]: ...

# =============================================================================
# Enumerations
# =============================================================================

class Boundary(Enum):
    """Boundary edge identifiers for 2D domains."""

    Left = 0
    Right = 1
    Bottom = 2
    Top = 3

# Convenience aliases
Left: Boundary
Right: Boundary
Top: Boundary
Bottom: Boundary

class BalanceDimension(Enum):
    AMOUNT = 0
    ENERGY = 1
    VOLUME = 2

class BalanceUnit(Enum):
    MOLE = 0
    MILLIMOLE = 1
    MICROMOLE = 2
    JOULE = 3
    KILOJOULE = 4
    CUBIC_METER = 5
    LITER = 6
    MILLILITER = 7

class BalanceTransferDirection(Enum):
    INCOMING = 0
    OUTGOING = 1

class BalanceTerm:
    name: str
    magnitude: float
    unit: BalanceUnit

class BalanceTransfer:
    id: str
    counterparty: str
    magnitude: float
    unit: BalanceUnit
    direction: BalanceTransferDirection

class BalanceAudit:
    ledger_name: str
    dimension: BalanceDimension
    unit: BalanceUnit
    initial_inventory: float
    final_inventory: float
    observed_change: float
    boundary_in: float
    boundary_out: float
    generated: float
    consumed: float
    transfer_in: float
    transfer_out: float
    expected_change: float
    closure_residual: float
    def is_closed(self, absolute_tolerance: float) -> bool: ...

class BalanceLedger:
    def __init__(self, name: str, unit: BalanceUnit) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def unit(self) -> BalanceUnit: ...
    @property
    def dimension(self) -> BalanceDimension: ...
    @property
    def has_initial_inventory(self) -> bool: ...
    @property
    def has_final_inventory(self) -> bool: ...
    @property
    def initial_inventory(self) -> float: ...
    @property
    def final_inventory(self) -> float: ...
    @property
    def boundary_in_terms(self) -> list[BalanceTerm]: ...
    @property
    def boundary_out_terms(self) -> list[BalanceTerm]: ...
    @property
    def generated_terms(self) -> list[BalanceTerm]: ...
    @property
    def consumed_terms(self) -> list[BalanceTerm]: ...
    @property
    def transfers(self) -> list[BalanceTransfer]: ...
    def set_initial_inventory(self, magnitude: float) -> BalanceLedger: ...
    def set_final_inventory(self, magnitude: float) -> BalanceLedger: ...
    def add_boundary_in(self, name: str, magnitude: float) -> BalanceLedger: ...
    def add_boundary_out(self, name: str, magnitude: float) -> BalanceLedger: ...
    def add_generated(self, name: str, magnitude: float) -> BalanceLedger: ...
    def add_consumed(self, name: str, magnitude: float) -> BalanceLedger: ...
    def add_transfer_in(
        self, id: str, sender: str, magnitude: float, unit: Optional[BalanceUnit] = None
    ) -> BalanceLedger: ...
    def add_transfer_out(
        self,
        id: str,
        receiver: str,
        magnitude: float,
        unit: Optional[BalanceUnit] = None,
    ) -> BalanceLedger: ...
    def audit(self) -> BalanceAudit: ...

class MatchedBalanceTransfer:
    id: str
    sender: str
    receiver: str
    dimension: BalanceDimension
    base_unit: BalanceUnit
    magnitude_base: float

class DimensionBalanceAudit:
    dimension: BalanceDimension
    base_unit: BalanceUnit
    observed_change: float
    external_expected_change: float
    internal_transfer_net: float
    closure_residual: float
    representation_adjustment: float

class BalanceReconciliation:
    ledgers: list[BalanceAudit]
    matched_transfers: list[MatchedBalanceTransfer]
    dimensions: list[DimensionBalanceAudit]
    def is_closed(
        self,
        amount_absolute_tolerance: float = 0.0,
        energy_absolute_tolerance: float = 0.0,
        volume_absolute_tolerance: float = 0.0,
    ) -> bool: ...

def balance_dimension_name(dimension: BalanceDimension) -> str: ...
def balance_unit_symbol(unit: BalanceUnit) -> str: ...
def balance_unit_dimension(unit: BalanceUnit) -> BalanceDimension: ...
def balance_base_unit(dimension: BalanceDimension) -> BalanceUnit: ...
def convert_balance_value(
    value: float, from_unit: BalanceUnit, to_unit: BalanceUnit
) -> float: ...
def reconcile_balances(
    ledgers: Sequence[BalanceLedger],
    relative_transfer_tolerance: float = 1.0e-12,
    absolute_transfer_tolerance_base: float = 0.0,
) -> BalanceReconciliation: ...

class Boundary3D(Enum):
    """Boundary-face identifiers for Cartesian 3D domains."""

    XMin = 0
    XMax = 1
    YMin = 2
    YMax = 3
    ZMin = 4
    ZMax = 5

# Convenience aliases exported by pybind11
XMin: Boundary3D
XMax: Boundary3D
YMin: Boundary3D
YMax: Boundary3D
ZMin: Boundary3D
ZMax: Boundary3D

class BoundaryType(Enum):
    """Types of scalar boundary conditions.

    ``NEUMANN`` prescribes the outward-normal derivative of the field;
    ``OUTWARD_FLUX`` prescribes a physical flux (positive leaving the domain).
    They are deliberately distinct types so a derivative is never mistaken for
    a flux.
    """

    DIRICHLET = 0
    NEUMANN = 1
    ROBIN = 2
    OUTWARD_FLUX = 3

ROBIN: BoundaryType
OUTWARD_FLUX: BoundaryType

class VelocityBCType(Enum):
    """Types of velocity boundary conditions for fluid flow."""

    DIRICHLET = 0
    NEUMANN = 1
    NOSLIP = 2
    INFLOW = 3
    OUTFLOW = 4

# Convenience aliases
DIRICHLET: VelocityBCType
NEUMANN: VelocityBCType
NOSLIP: VelocityBCType
INFLOW: VelocityBCType
OUTFLOW: VelocityBCType

class AdvectionScheme(Enum):
    """Advection discretization schemes."""

    UPWIND = 0
    CENTRAL = 1
    HYBRID = 2
    QUICK = 3

class ConvectionScheme(Enum):
    """Convection discretization schemes for Navier-Stokes."""

    UPWIND = 0
    CENTRAL = 1
    QUICK = 2
    HYBRID = 3

# ``ConvectionScheme`` is registered later and owns the unqualified aliases.
UPWIND: ConvectionScheme
CENTRAL: ConvectionScheme
QUICK: ConvectionScheme
HYBRID: ConvectionScheme

class CylindricalMeshType(Enum):
    """Types of cylindrical coordinate meshes."""

    AXISYMMETRIC_RZ = 0
    RADIAL_R = 1
    FULL_3D = 2

# Convenience aliases
RADIAL_R: CylindricalMeshType
AXISYMMETRIC_RZ: CylindricalMeshType
FULL_3D: CylindricalMeshType

class FluidModel(Enum):
    """Rheological model identifiers returned by ``ViscosityModel.type()``."""

    NEWTONIAN = 0
    POWER_LAW = 1
    CARREAU = 2
    CARREAU_YASUDA = 3
    CROSS = 4
    BINGHAM = 5
    HERSCHEL_BULKLEY = 6
    CASSON = 7

# Convenience aliases
NEWTONIAN: FluidModel
POWER_LAW: FluidModel
CARREAU: FluidModel
CARREAU_YASUDA: FluidModel
CROSS: FluidModel
CASSON: FluidModel
BINGHAM: FluidModel
HERSCHEL_BULKLEY: FluidModel

# =============================================================================
# Data Classes / Structs
# =============================================================================

class BoundaryCondition:
    """Represents a boundary condition for transport problems.

    Boundary conditions specify how the solution behaves at domain boundaries.
    Three types are represented:

    - **Dirichlet**: Fixed value at the boundary (e.g., constant concentration)
    - **Neumann**: Fixed outward-normal derivative ``du/dn``
    - **Robin**: Mixed condition ``a*u + b*du/dn = c``

    Examples:
        >>> # Fixed concentration at left boundary
        >>> bc_left = BoundaryCondition.dirichlet(1.0)
        >>> # Zero flux (insulated) at right boundary
        >>> bc_right = BoundaryCondition.neumann(0.0)

    Attributes:
        type: The type of boundary condition (DIRICHLET or NEUMANN).
        value: The boundary value (concentration for Dirichlet, derivative for Neumann).
    """

    type: BoundaryType
    value: float
    a: float
    b: float
    c: float

    def __init__(self, type: BoundaryType, value: float) -> None:
        """Create scalar boundary metadata.

        ``value`` is the fixed value for Dirichlet data or the
        outward-normal derivative for Neumann data.
        """
        ...

    @staticmethod
    def dirichlet(value: float) -> BoundaryCondition:
        """Create a Dirichlet (fixed value) boundary condition."""
        ...

    @staticmethod
    def neumann(normal_derivative: float) -> BoundaryCondition:
        """Create a Neumann condition for the outward-normal derivative."""
        ...

    @staticmethod
    def robin(a: float, b: float, c: float) -> BoundaryCondition:
        """Create ``a*u + b*du/dn = c``."""
        ...
    @staticmethod
    def outward_flux(outward_flux: float) -> BoundaryCondition:
        """Create a prescribed physical-flux condition, positive leaving the domain.

        Distinct from :meth:`neumann`, which is a derivative. Solvers that only
        accept derivative data reject this condition instead of reinterpreting it.
        """
        ...

class VelocityBC:
    """Velocity boundary condition for fluid flow solvers.

    Specifies velocity constraints at domain boundaries for Stokes and
    Navier-Stokes solvers. Common types include:

    - **NoSlip**: Zero velocity at solid walls (u=v=0)
    - **Inflow**: Prescribed velocity at inlet
    - **Outflow**: Zero outward-normal velocity gradient
    - **Dirichlet**: Arbitrary fixed velocity

    Examples:
        >>> # No-slip wall (solid boundary)
        >>> bc_wall = VelocityBC.no_slip()
        >>> # Inlet with horizontal velocity
        >>> bc_inlet = VelocityBC.inflow(u=0.1, v=0.0)
        >>> # Zero-normal-gradient outlet
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
        """Create a zero-outward-normal-velocity-gradient condition."""
        ...

    @staticmethod
    def stress_free() -> VelocityBC:
        """Create traction-free metadata.

        Current Stokes/Navier-Stokes boundary applicators reject this
        unsupported traction condition instead of treating it as outflow.
        """
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

    def __init__(self) -> None: ...
    @property
    def dt(self) -> float: ...
    @property
    def steps(self) -> int: ...
    @property
    def t_end(self) -> float: ...
    @property
    def wall_time_s(self) -> float: ...
    @property
    def mass_initial(self) -> float: ...
    @property
    def mass_final(self) -> float: ...
    @property
    def mass_abs_drift(self) -> float: ...
    @property
    def mass_rel_drift(self) -> float: ...
    @property
    def u_min_initial(self) -> float: ...
    @property
    def u_max_initial(self) -> float: ...
    @property
    def u_min_final(self) -> float: ...
    @property
    def u_max_final(self) -> float: ...

class RunResult:
    """Result from a reaction-diffusion solver run.

    Contains the final solution field and solver statistics.

    Examples:
        >>> result = solver.run(problem, t_end=1.0)
        >>> print(f"Completed in {result.stats.steps} steps")
        >>> print(f"Solution range: [{result.solution.min():.3f}, {result.solution.max():.3f}]")

    ``stats`` is read-only. ``solution()`` returns an owned flat NumPy copy;
    for 2D problems reshape it to ``(ny+1, nx+1)``.
    """

    def __init__(self) -> None: ...
    @property
    def stats(self) -> SolverStats: ...
    def solution(self) -> FloatArray:
        """Return an owned copy of the final flat field."""
        ...

class StokesResult:
    """Result from Stokes flow solver.

    Contains the steady-state velocity and pressure fields for creeping
    (low Reynolds number) flow, along with convergence information.

    The Stokes equations describe viscous-dominated flow where inertial
    effects are negligible (Re << 1), common in microfluidics and
    biological flows at the cellular scale.

    The ``u()``, ``v()``, and ``pressure()`` methods each return an owned array
    copy. Scalar diagnostics are read-only.

    Attributes:
        divergence: Velocity divergence (should be ~0 for incompressible flow).
        converged: True if solver converged within tolerance.
        iterations: Number of iterations to convergence.
        residual: Final residual norm.
    """

    def __init__(self) -> None: ...
    def u(self) -> FloatArray: ...
    def v(self) -> FloatArray: ...
    def pressure(self) -> FloatArray: ...
    @property
    def divergence(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def residual(self) -> float: ...

class NavierStokesResult:
    """Result from Navier-Stokes solver.

    Contains the velocity and pressure fields at the final time step,
    along with flow characteristics and stability information.

    This result comes from the library's laminar bounded MAC-grid projection
    solver. It is not a turbulence model and should not be interpreted as
    validated across arbitrary Reynolds numbers.

    The ``u()``, ``v()``, and ``pressure()`` methods each return an owned array
    copy. Scalar diagnostics are read-only.

    Attributes:
        time: Final simulation time.
        time_steps: Total number of time steps taken.
        reynolds: Reynolds number of the flow.
        max_velocity: Maximum velocity magnitude in the domain.
        pressure_iterations: Iterations in the final pressure projection.
        pressure_residual: Relative final pressure-Poisson residual.
        divergence: Maximum final cell ``|div(u)|`` in inverse seconds.
        stable: True only for finite fields with a converged, divergence-controlled projection.
    """

    def __init__(self) -> None: ...
    def u(self) -> FloatArray: ...
    def v(self) -> FloatArray: ...
    def pressure(self) -> FloatArray: ...
    @property
    def time(self) -> float: ...
    @property
    def time_steps(self) -> int: ...
    @property
    def reynolds(self) -> float: ...
    @property
    def max_velocity(self) -> float: ...
    @property
    def pressure_iterations(self) -> int: ...
    @property
    def pressure_residual(self) -> float: ...
    @property
    def divergence(self) -> float: ...
    @property
    def stable(self) -> bool: ...

class DarcyFlowResult:
    """Result from Darcy flow solver.

    Contains the pressure and velocity fields for flow through porous
    media governed by Darcy's law: v = -K/μ ∇p.

    Darcy flow is applicable to groundwater flow, tissue perfusion,
    and other porous media transport problems.

    Attributes:
        pressure: Method returning an owned pressure-field copy.
        vx: Method returning an owned x-velocity copy.
        vy: Method returning an owned y-velocity copy.
        converged: True if solver converged within tolerance.
        iterations: Number of iterations to convergence.
        residual: Final residual norm.
    """

    def __init__(self) -> None: ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def residual(self) -> float: ...
    def pressure(self) -> FloatArray: ...
    def vx(self) -> FloatArray: ...
    def vy(self) -> FloatArray: ...

class MembraneDiffusionResult:
    """Result from membrane diffusion solver.

    Contains the steady-state concentration profile across a membrane
    and derived transport properties.

    Membrane diffusion is fundamental to drug delivery, dialysis,
    and cellular transport. The permeability P = D·K/L relates
    diffusivity D, partition coefficient K, and thickness L.

    Attributes:
        x: Method returning an owned position-coordinate copy.
        concentration: Method returning an owned concentration-profile copy.
        flux: Amount flux through the membrane in the caller's concentration
            unit times m/s (for mol/m³ input, mol/(m² s)).
        permeability: Membrane permeability (m/s).
        effective_diffusivity: Equivalent coefficient relative to the external
            concentration gradient, ``permeability * thickness``. It includes
            partition and hindrance effects and is not necessarily the membrane's
            intrinsic diffusion coefficient.
    """

    def __init__(self) -> None: ...
    @property
    def flux(self) -> float: ...
    @property
    def permeability(self) -> float: ...
    @property
    def effective_diffusivity(self) -> float: ...
    def x(self) -> FloatArray: ...
    def concentration(self) -> FloatArray: ...

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
        nx: Number of periodic cell-centred samples in x (``mesh.nx()``).
        ny: Number of periodic cell-centred samples in y (``mesh.ny()``).
        steps_run: Total simulation steps completed.
        frames: Number of saved frames.
        frame_steps: Steps between saved frames.
        u_frames: Method returning an owned ``(frames, ny, nx)`` float32 array.
        v_frames: Method returning an owned ``(frames, ny, nx)`` float32 array.
    """

    def __init__(self) -> None: ...
    @property
    def nx(self) -> int: ...
    @property
    def ny(self) -> int: ...
    @property
    def steps_run(self) -> int: ...
    @property
    def final_time(self) -> float: ...
    @property
    def frames(self) -> int: ...
    @property
    def frame_steps(self) -> list[int]: ...
    def u_frames(self) -> Float32Array: ...
    def v_frames(self) -> Float32Array: ...

class TumorDrugDeliverySaved:
    """Saved frames from tumor drug delivery simulation.

    Contains time-series data from a coupled simulation of drug transport
    in tumor tissue, including convection (pressure-driven flow),
    diffusion, binding, and cellular uptake.

    The model tracks three drug compartments:
    - Free drug in interstitial space
    - Bound drug (irreversibly sequestered by tissue in this reduced model)
    - Internalized drug (taken up by cells)

    Attributes:
        nx: Number of grid points in x-direction.
        ny: Number of grid points in y-direction.
        frames: Number of saved time frames.
        times_s: List of save times in seconds.
        free: Method returning an owned 3D free-drug array.
        bound: Method returning an owned 3D bound-drug array.
        cellular: Method returning an owned 3D internalized-drug array.
        total: Method returning an owned 3D total-drug array.
    """

    def __init__(self) -> None: ...
    @property
    def nx(self) -> int: ...
    @property
    def ny(self) -> int: ...
    @property
    def frames(self) -> int: ...
    @property
    def times_s(self) -> list[float]: ...
    @property
    def final_time_s(self) -> float: ...
    @property
    def stability_limit_s(self) -> float: ...
    @property
    def free_amount_per_depth(self) -> list[float]: ...
    @property
    def bound_amount_per_depth(self) -> list[float]: ...
    @property
    def cellular_amount_per_depth(self) -> list[float]: ...
    @property
    def total_amount_per_depth(self) -> list[float]: ...
    @property
    def cumulative_net_vascular_exchange_per_depth(self) -> list[float]: ...
    @property
    def cumulative_boundary_outflow_per_depth(self) -> list[float]: ...
    @property
    def mass_balance_error_per_depth(self) -> list[float]: ...
    def free(self) -> FloatArray: ...
    def bound(self) -> FloatArray: ...
    def cellular(self) -> FloatArray: ...
    def total(self) -> FloatArray: ...

class BioheatSaved:
    """Saved frames from bioheat cryotherapy simulation.

    Contains time-series data from a coupled thermal-damage simulation
    modeling cryoablation therapy. The model includes:

    - Pennes bioheat equation with blood perfusion
    - Phase change (freezing/thawing) with latent heat
    - Arrhenius tissue damage accumulation

    The Arrhenius field is a heat-injury diagnostic. It is not a validated
    cryogenic cell-death model.

    Attributes:
        nx: Number of grid points in x-direction.
        ny: Number of grid points in y-direction.
        frames: Number of saved time frames.
        times_s: List of save times in seconds.
        temperature_K: Method returning an owned Kelvin field array.
        damage: Method returning an owned Arrhenius-integral array.
        frozen_fraction: Method returning an owned apparent-frozen-fraction array.
    """

    def __init__(self) -> None: ...
    @property
    def nx(self) -> int: ...
    @property
    def ny(self) -> int: ...
    @property
    def frames(self) -> int: ...
    @property
    def times_s(self) -> list[float]: ...
    @property
    def minimum_temperature_K(self) -> list[float]: ...
    @property
    def maximum_temperature_K(self) -> list[float]: ...
    @property
    def maximum_stable_dt_s(self) -> float: ...
    def temperature_K(self) -> FloatArray: ...
    def damage(self) -> FloatArray: ...
    def frozen_fraction(self) -> FloatArray: ...

# =============================================================================
# Meshes
# =============================================================================

class NonuniformMesh1D:
    """Validated fitted 1D node mesh with node-centred control volumes."""

    def __init__(self, nodes: Sequence[float]) -> None: ...
    def num_nodes(self) -> int: ...
    def num_cells(self) -> int: ...
    def x(self, node: int) -> float: ...
    def nodes(self) -> FloatArray: ...
    def spacing(self, face: int) -> float: ...
    def face_coordinate(self, face: int) -> float: ...
    def control_volume(self, node: int) -> float: ...
    def control_volumes(self) -> FloatArray: ...
    def xmin(self) -> float: ...
    def xmax(self) -> float: ...
    def length(self) -> float: ...
    def minimum_spacing(self) -> float: ...

class NonuniformDiffusionDiagnostics:
    steps: int
    reference_time: float
    time: float
    stability_limit: float
    reference_mass: float
    total_mass: float
    cumulative_boundary_input: float
    mass_balance_error: float
    minimum_concentration: float
    maximum_concentration: float
    left_outward_flux: float
    right_outward_flux: float

class NonuniformDiffusion1D:
    @overload
    def __init__(self, mesh: NonuniformMesh1D, diffusivity: float) -> None: ...
    @overload
    def __init__(
        self, mesh: NonuniformMesh1D, nodal_diffusivity: Sequence[float]
    ) -> None: ...
    def set_initial_condition(
        self, concentration: Sequence[float]
    ) -> NonuniformDiffusion1D: ...
    def set_uniform_initial_condition(
        self, concentration: float
    ) -> NonuniformDiffusion1D: ...
    def set_boundary_condition(
        self, boundary: Boundary, condition: BoundaryCondition
    ) -> NonuniformDiffusion1D: ...
    def set_dirichlet_boundary(
        self, boundary: Boundary, concentration: float
    ) -> NonuniformDiffusion1D: ...
    def set_neumann_boundary(
        self, boundary: Boundary, outward_normal_derivative: float
    ) -> NonuniformDiffusion1D: ...
    def boundary_condition(self, boundary: Boundary) -> BoundaryCondition: ...
    def check_stability(self, dt: float) -> bool: ...
    def max_stable_time_step(self) -> float: ...
    def step(self, dt: float) -> None: ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> None: ...
    def solution(self) -> FloatArray: ...
    def diffusivity(self) -> FloatArray: ...
    def face_diffusivities(self) -> FloatArray: ...
    def face_fluxes(self) -> FloatArray: ...
    def mesh(self) -> NonuniformMesh1D: ...
    def time(self) -> float: ...
    def steps(self) -> int: ...
    def total_mass(self) -> float: ...
    def boundary_outward_flux(self, boundary: Boundary) -> float: ...
    def reset_balance_reference(self) -> None: ...
    def diagnostics(self) -> NonuniformDiffusionDiagnostics: ...

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
        """True for the 1D constructor (reported ``ny() == 0``)."""
        ...

    def x(self, i: int) -> float:
        """x-coordinate of node index ``i``."""
        ...

    def y(self, i: int, j: int = 0) -> float:
        """y-coordinate of node ``(i, j)`` (zero for a 1D mesh)."""
        ...

    def index(self, i: int, j: int = 0) -> int:
        """Convert (i, j) indices to linear index."""
        ...

class StructuredMesh3D:
    """Uniform Cartesian mesh with ``(nx+1)*(ny+1)*(nz+1)`` nodes."""

    @overload
    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        zmin: float,
        zmax: float,
    ) -> None: ...
    @overload
    def __init__(self, n: int, length: float) -> None:
        """Create an ``n`` by ``n`` by ``n`` cube on ``[0, length]^3``."""
        ...

    def num_nodes(self) -> int: ...
    def num_cells(self) -> int: ...
    def dx(self) -> float: ...
    def dy(self) -> float: ...
    def dz(self) -> float: ...
    def nx(self) -> int: ...
    def ny(self) -> int: ...
    def nz(self) -> int: ...
    def xmin(self) -> float: ...
    def xmax(self) -> float: ...
    def ymin(self) -> float: ...
    def ymax(self) -> float: ...
    def zmin(self) -> float: ...
    def zmax(self) -> float: ...
    def x(self, i: int) -> float: ...
    def y(self, j: int) -> float: ...
    def z(self, k: int) -> float: ...
    def index(self, i: int, j: int, k: int) -> int: ...
    def ijk(self, idx: int) -> list[int]:
        """Return the fixed-size ``[i, j, k]`` index list."""
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

    def thetamin(self) -> float:
        """Minimum azimuthal coordinate."""
        ...

    def thetamax(self) -> float:
        """Maximum azimuthal coordinate."""
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

    def theta_node_count(self) -> int:
        """Number of stored unique azimuthal nodes (one outside full 3D)."""
        ...

    def theta_periodic(self) -> bool:
        """Whether the azimuthal direction is periodic."""
        ...

    def r(self, i: int) -> float:
        """Radial coordinate at index ``i``."""
        ...

    def theta(self, j: int) -> float:
        """Angular coordinate at index ``j``."""
        ...

    def z(self, k: int) -> float:
        """Axial coordinate at index ``k``."""
        ...

    def x(self, i: int, j: int = 0) -> float:
        """Cartesian x coordinate for cylindrical indices ``(i, j)``."""
        ...

    def y(self, i: int, j: int = 0) -> float:
        """Cartesian y coordinate for cylindrical indices ``(i, j)``."""
        ...

    def index(self, i: int, j: int = 0, k: int = 0) -> int:
        """Convert indices to linear index."""
        ...

    def cell_area(self, i: int) -> float:
        """Area of cell at radial index i."""
        ...

    def cell_volume(self, i: int, j: int = 0, k: int = 0) -> float:
        """Nodal control volume at indices ``(i, j, k)``."""
        ...

    def cross_section_area(self) -> float:
        """Exact annular cross-section area represented by the mesh."""
        ...

    def gradient_r(self, phi: ArrayLike) -> FloatArray:
        """Compute radial gradient of field."""
        ...

    def gradient_theta(self, phi: ArrayLike) -> FloatArray:
        """Compute physical azimuthal gradient ``(1/r) d(field)/d(theta)``."""
        ...

    def gradient_z(self, phi: ArrayLike) -> FloatArray:
        """Compute axial gradient of field."""
        ...

    def laplacian(self, phi: ArrayLike) -> FloatArray:
        """Compute Laplacian of field."""
        ...

    @overload
    def divergence(self, vr: FloatArray, vz: FloatArray) -> FloatArray:
        """Compute axisymmetric divergence from radial and axial components."""
        ...

    @overload
    def divergence(
        self, vr: FloatArray, vtheta: FloatArray, vz: FloatArray
    ) -> FloatArray:
        """Compute full cylindrical divergence from all three components."""
        ...

class StencilOps:
    """Finite-difference stencil operations tied to a ``StructuredMesh``."""

    def __init__(self, mesh: StructuredMesh) -> None: ...
    def laplacian(self, u: ArrayLike, idx: int) -> float: ...
    def laplacian_4th_order(self, u: ArrayLike, idx: int) -> float: ...
    def laplacian_6th_order(self, u: ArrayLike, idx: int) -> float: ...
    def grad_x(self, u: ArrayLike, idx: int) -> float: ...
    def grad_y(self, u: ArrayLike, idx: int) -> float: ...
    def grad_x_4th_order(self, u: ArrayLike, idx: int) -> float: ...
    def grad_y_4th_order(self, u: ArrayLike, idx: int) -> float: ...
    def laplacian_4th_order_bulk_1d(self, u: ArrayLike) -> FloatArray: ...
    def laplacian_6th_order_bulk_1d(self, u: ArrayLike) -> FloatArray: ...
    def laplacian_4th_order_bulk_2d(self, u: ArrayLike) -> FloatArray: ...
    def gradient_4th_order_bulk_1d(self, u: ArrayLike) -> FloatArray: ...
    def inv_dx2(self) -> float: ...
    def inv_dy2(self) -> float: ...
    def inv_12_dx2(self) -> float: ...
    def inv_12_dy2(self) -> float: ...
    def stride(self) -> int: ...

# =============================================================================
# Optional Eigen-backed sparse and implicit solvers
# =============================================================================

def sparse_matrix_available() -> bool:
    """Whether this extension was built with Eigen sparse-solver support."""
    ...

class SparseSolverType(Enum):
    SparseLU = 0
    SimplicialLLT = 1
    SimplicialLDLT = 2
    ConjugateGradient = 3
    BiCGSTAB = 4

SparseLU: SparseSolverType
SimplicialLLT: SparseSolverType
SimplicialLDLT: SparseSolverType
ConjugateGradient: SparseSolverType
BiCGSTAB: SparseSolverType

class SparseSolveResult:
    """Sparse-solve diagnostics type retained by the native API."""

    def __init__(self) -> None: ...
    @property
    def success(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def residual(self) -> float: ...
    @property
    def error_message(self) -> str: ...

class Triplet:
    def __init__(self, row: int, col: int, value: float) -> None: ...
    @property
    def row(self) -> int: ...
    @property
    def col(self) -> int: ...
    @property
    def value(self) -> float: ...

class SparseMatrix:
    """Mutable triplet-assembly wrapper for an Eigen sparse matrix."""

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, rows: int, cols: int) -> None: ...
    @property
    def rows(self) -> int: ...
    @property
    def cols(self) -> int: ...
    @property
    def nnz(self) -> int: ...
    def reserve(self, nnz_estimate: int) -> None: ...
    def add_entry(self, row: int, col: int, value: float) -> None: ...
    def finalize(self) -> None: ...
    def is_finalized(self) -> bool: ...
    def solve(
        self,
        b: list[float],
        solver_type: SparseSolverType = SparseSolverType.SparseLU,
        tolerance: float = 1e-10,
        max_iterations: int = 1000,
    ) -> list[float]: ...
    def multiply(self, x: list[float]) -> list[float]: ...
    def clear(self) -> None: ...
    def resize(self, rows: int, cols: int) -> None: ...

def build_2d_laplacian(nx: int, ny: int, dx: float, dy: float) -> SparseMatrix: ...
def build_implicit_diffusion_2d(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    alpha: float,
    dt: float,
) -> SparseMatrix: ...
def build_implicit_diffusion_3d(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    alpha: float,
    dt: float,
) -> SparseMatrix: ...

class ImplicitSolveResult:
    """Diagnostics from a completed Backward Euler diffusion solve."""

    def __init__(self) -> None: ...
    @property
    def steps(self) -> int: ...
    @property
    def total_time(self) -> float: ...
    @property
    def residual(self) -> float: ...
    @property
    def success(self) -> bool: ...

class ImplicitDiffusion2D:
    """Conservative 2D Backward Euler ``div(D grad(u))`` solver.

    Diffusivity is scalar or nodal and harmonic face averaging preserves
    interface flux. Neumann values are outward derivatives, not Fickian fluxes.
    Returned solution and diffusivity arrays own their data.
    """

    @overload
    def __init__(self, mesh: StructuredMesh, diffusivity: float) -> None: ...
    @overload
    def __init__(self, mesh: StructuredMesh, diffusivity: ArrayLike) -> None: ...
    def set_initial_condition(self, values: ArrayLike) -> None: ...
    def set_dirichlet_boundary(self, boundary: Boundary, value: float) -> None: ...
    def set_neumann_boundary(
        self, boundary: Boundary, normal_derivative: float
    ) -> None: ...
    def set_source_term(
        self, source: Callable[[float, float, float], float]
    ) -> None: ...
    def clear_source_term(self) -> None: ...
    def set_solver_type(self, solver_type: SparseSolverType) -> None: ...
    def set_tolerance(self, tolerance: float) -> None: ...
    def set_max_iterations(self, max_iterations: int) -> None: ...
    def step(self, dt: float) -> ImplicitSolveResult: ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> ImplicitSolveResult: ...
    def solution(self) -> FloatArray: ...
    def diffusivity(self) -> FloatArray: ...
    def time(self) -> float: ...
    def mesh(self) -> StructuredMesh: ...

class ImplicitDiffusion3D:
    """Conservative 3D Backward Euler ``div(D grad(u))`` solver."""

    @overload
    def __init__(self, mesh: StructuredMesh3D, diffusivity: float) -> None: ...
    @overload
    def __init__(self, mesh: StructuredMesh3D, diffusivity: ArrayLike) -> None: ...
    def set_initial_condition(self, values: ArrayLike) -> None: ...
    def set_dirichlet_boundary(self, boundary: Boundary3D, value: float) -> None: ...
    def set_neumann_boundary(
        self, boundary: Boundary3D, normal_derivative: float
    ) -> None: ...
    def set_source_term(
        self, source: Callable[[float, float, float, float], float]
    ) -> None: ...
    def clear_source_term(self) -> None: ...
    def set_solver_type(self, solver_type: SparseSolverType) -> None: ...
    def set_tolerance(self, tolerance: float) -> None: ...
    def set_max_iterations(self, max_iterations: int) -> None: ...
    def step(self, dt: float) -> ImplicitSolveResult: ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> ImplicitSolveResult: ...
    def solution(self) -> FloatArray: ...
    def diffusivity(self) -> FloatArray: ...
    def time(self) -> float: ...
    def mesh(self) -> StructuredMesh3D: ...

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

    def __init__(self, mesh: StructuredMesh) -> None:
        """Create a problem that owns a copy of ``mesh``."""
        ...

    @overload
    def diffusivity(self, diffusivity: float) -> TransportProblem:
        """Set uniform diffusivity and return this problem."""
        ...

    @overload
    def diffusivity(self) -> float:
        """Return the configured uniform diffusivity value."""
        ...

    def diffusivity_field(self, D_field: ArrayLike) -> TransportProblem:
        """Set spatially-varying diffusivity field."""
        ...

    @overload
    def reaction(
        self, function: Callable[[float, float, float, float], float]
    ) -> TransportProblem:
        """Replace the reaction; automatic reaction stability is then uncertified."""
        ...

    @overload
    def reaction(
        self,
        function: Callable[[float, float, float, float], float],
        max_abs_dc: float,
    ) -> TransportProblem:
        """Replace the reaction and declare a global ``|dR/dc|`` bound."""
        ...

    @overload
    def add_reaction(
        self, function: Callable[[float, float, float, float], float]
    ) -> TransportProblem:
        """Compose an additional reaction; automatic reaction stability is uncertified."""
        ...

    @overload
    def add_reaction(
        self,
        function: Callable[[float, float, float, float], float],
        max_abs_dc: float,
    ) -> TransportProblem:
        """Compose an additional reaction and its ``|dR/dc|`` bound."""
        ...

    def velocity(self, vx: float, vy: float = 0.0) -> TransportProblem:
        """Set uniform velocity field."""
        ...

    @overload
    def velocity_field(self, vx: ArrayLike) -> TransportProblem:
        """Set a node-centred x velocity field for a 1D mesh."""
        ...

    @overload
    def velocity_field(self, vx: ArrayLike, vy: ArrayLike) -> TransportProblem:
        """Set node-centred x and y velocity fields."""
        ...

    def advection_scheme(self, scheme: AdvectionScheme) -> TransportProblem:
        """Set advection discretization scheme."""
        ...

    @overload
    def initial_condition(self, values: ArrayLike) -> TransportProblem:
        """Copy one initial value per node and return this problem."""
        ...

    @overload
    def initial_condition(self, value: float) -> TransportProblem:
        """Set a uniform initial value and return this problem."""
        ...

    def dirichlet(self, side: Boundary, value: float) -> TransportProblem:
        """Set Dirichlet boundary condition."""
        ...

    def neumann(self, side: Boundary, normal_derivative: float) -> TransportProblem:
        """Set the outward-normal derivative ``dc/dn`` on one side.

        This is a derivative, not a Fickian flux: the outward diffusive flux is
        ``-D * normal_derivative``.
        """
        ...

    def robin(self, side: Boundary, a: float, b: float, c: float) -> TransportProblem:
        """Set ``a*u + b*du/dn = c``."""
        ...

    def boundary(self, side: Boundary, bc: BoundaryCondition) -> TransportProblem:
        """Set boundary condition using BoundaryCondition object."""
        ...

    def constant_source(self, S: float) -> TransportProblem:
        """Replace the reaction with the constant source ``R=S``."""
        ...

    def add_constant_source(self, S: float) -> TransportProblem:
        """Compose the constant source ``R=S`` with existing reactions."""
        ...

    def linear_decay(self, k: float) -> TransportProblem:
        """Replace the reaction with linear decay ``R=-k*u``."""
        ...

    def add_linear_decay(self, k: float) -> TransportProblem:
        """Compose linear decay ``R=-k*u`` with existing reactions."""
        ...

    def logistic_growth(self, r: float, K: float) -> TransportProblem:
        """Replace the reaction with logistic growth ``R=r*u*(1-u/K)``."""
        ...

    def add_logistic_growth(self, r: float, K: float) -> TransportProblem:
        """Compose logistic growth with existing reactions."""
        ...

    def michaelis_menten(self, Vmax: float, Km: float) -> TransportProblem:
        """Replace the reaction with Michaelis-Menten consumption."""
        ...

    def add_michaelis_menten(self, Vmax: float, Km: float) -> TransportProblem:
        """Compose Michaelis-Menten consumption with existing reactions."""
        ...

    def clear_reaction(self) -> TransportProblem:
        """Remove every configured reaction term."""
        ...

    def has_uniform_diffusivity(self) -> bool: ...
    def has_advection(self) -> bool: ...
    def has_reaction(self) -> bool: ...
    def reaction_stability_bound_known(self) -> bool: ...
    def reaction_stability_rate_bound(self) -> float: ...
    def mesh(self) -> StructuredMesh:
        """Return the internally retained mesh object."""
        ...

    def initial(self) -> FloatArray:
        """Return an owned copy of the flat initial field."""
        ...

    def boundaries(self) -> list[BoundaryCondition]:
        """Return a copy of boundary metadata ordered left, right, bottom, top."""
        ...

class SolveOptions:
    """Controls the verified conservative explicit C++ transport solve."""

    final_time: float
    time_step: float
    safety_factor: float
    reaction_step_fraction: float
    max_steps: int
    check_finite: bool
    save_times: list[float]
    """Absolute times in ``[0, final_time]`` at which the field is recorded.

    Strictly increasing. Each save time partitions the step schedule so the
    field is captured exactly at that clock; the reaction term always receives
    the absolute time. Empty leaves the schedule unchanged.
    """

    def __init__(self) -> None: ...
    @staticmethod
    def until(final_time: float) -> SolveOptions: ...

class SolveDiagnostics:
    """Numerical and physical diagnostics from a transport solve."""

    @property
    def steps(self) -> int: ...
    @property
    def requested_final_time(self) -> float: ...
    @property
    def final_time(self) -> float: ...
    @property
    def requested_time_step(self) -> float: ...
    @property
    def minimum_time_step(self) -> float: ...
    @property
    def maximum_time_step(self) -> float: ...
    @property
    def transport_stable_time_step(self) -> float: ...
    @property
    def certified_stable_time_step(self) -> float: ...
    @property
    def maximum_transport_loss_rate(self) -> float: ...
    @property
    def reaction_rate_bound(self) -> float: ...
    @property
    def automatic_time_step(self) -> bool: ...
    @property
    def reaction_stability_bound_known(self) -> bool: ...
    @property
    def initial_mass(self) -> float: ...
    @property
    def final_mass(self) -> float: ...
    @property
    def mass_change(self) -> float: ...
    @property
    def initial_minimum(self) -> float: ...
    @property
    def initial_maximum(self) -> float: ...
    @property
    def final_minimum(self) -> float: ...
    @property
    def final_maximum(self) -> float: ...

class TransportResult:
    """Final scalar field, snapshots, exact physical time, and solve diagnostics."""

    @property
    def concentration(self) -> FloatArray:
        """Return an owned copy of the final concentration field."""
        ...
    @property
    def solution(self) -> FloatArray:
        """Deprecated alias of ``concentration`` (warns on access)."""
        ...
    @property
    def time(self) -> float: ...
    @property
    def diagnostics(self) -> SolveDiagnostics: ...
    @property
    def mesh(self) -> StructuredMesh:
        """Copy of the mesh the fields are defined on."""
        ...
    @property
    def snapshot_times(self) -> FloatArray:
        """Absolute times requested through ``SolveOptions.save_times``."""
        ...
    @property
    def snapshot_fields(self) -> list[FloatArray]:
        """Owned copies of the nodal field at each snapshot time."""
        ...

def solve_transport(
    problem: TransportProblem, options: SolveOptions
) -> TransportResult:
    """Solve every configured scalar-transport term in the C++ core."""
    ...

class ExplicitFD:
    """Legacy explicit facade over the unified ``TransportProblem`` surface."""

    def __init__(self) -> None: ...
    def safety_factor(self, factor: float) -> ExplicitFD: ...
    def run(self, problem: TransportProblem, t_end: float) -> RunResult: ...

# =============================================================================
# Diffusion and Advection-Diffusion Solvers
# =============================================================================

class CNSolveResult:
    """Algebraic diagnostics from one Crank-Nicolson step."""

    def __init__(self) -> None: ...
    @property
    def iterations(self) -> int: ...
    @property
    def residual(self) -> float: ...
    @property
    def converged(self) -> bool: ...

class CrankNicolsonDiffusion:
    """Constant-diffusivity Crank-Nicolson solver on a 1D/2D mesh.

    The linear diffusion method is A-stable and second-order in time, but is
    not L-stable; excessively large steps can remain bounded while oscillatory
    or inaccurate. Neumann data are outward-normal derivatives, not fluxes.
    """

    def __init__(self, mesh: StructuredMesh, diffusivity: float) -> None: ...
    def set_initial_condition(self, values: ArrayLike) -> None: ...
    def set_dirichlet_boundary(self, boundary: Boundary, value: float) -> None: ...
    def set_neumann_boundary(
        self, boundary: Boundary, normal_derivative: float
    ) -> None: ...
    def set_tolerance(self, tol: float) -> CrankNicolsonDiffusion: ...
    def set_max_iterations(self, max_iter: int) -> CrankNicolsonDiffusion: ...
    def step(self, dt: float) -> CNSolveResult: ...
    def mesh(self) -> StructuredMesh:
        """The mesh this solver was built on."""
        ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> None: ...
    def solution(self) -> FloatArray:
        """Return an owned copy of the current flat field."""
        ...
    def time(self) -> float: ...
    @property
    def diffusivity(self) -> float: ...

class ADISolveResult:
    """Time-step counts and time reached by a symmetric ADI solve."""

    def __init__(self) -> None: ...
    @property
    def steps(self) -> int: ...
    @property
    def substeps(self) -> int: ...
    @property
    def time(self) -> float: ...
    @property
    def total_time(self) -> float: ...
    @property
    def success(self) -> bool: ...

class ADIDiffusion2D:
    """Symmetric x/2-y-x/2 split solver for constant-D 2D diffusion.

    Directional linear subproblems are unconditionally stable. The advertised
    second-order convergence assumes smooth solutions and time-independent
    boundary data. Neumann values are outward derivatives, not physical fluxes.
    """

    def __init__(self, mesh: StructuredMesh, diffusivity: float) -> None: ...
    def set_initial_condition(self, values: ArrayLike) -> None: ...
    def set_dirichlet_boundary(self, boundary: Boundary, value: float) -> None: ...
    def set_neumann_boundary(
        self, boundary: Boundary, normal_derivative: float
    ) -> None: ...
    def step(self, dt: float) -> ADISolveResult: ...
    def mesh(self) -> StructuredMesh:
        """The mesh this solver was built on."""
        ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> ADISolveResult: ...
    def solution(self) -> FloatArray:
        """Return an owned copy of the current flat field."""
        ...
    def time(self) -> float: ...
    @property
    def diffusivity(self) -> float: ...

class ADIDiffusion3D:
    """Symmetric x/2-y/2-z-y/2-x/2 constant-D 3D diffusion solver."""

    def __init__(self, mesh: StructuredMesh3D, diffusivity: float) -> None: ...
    def set_initial_condition(self, values: ArrayLike) -> None: ...
    @overload
    def set_dirichlet_boundary(self, boundary: Boundary3D, value: float) -> None: ...
    @overload
    def set_dirichlet_boundary(self, boundary_id: int, value: float) -> None: ...
    @overload
    def set_neumann_boundary(
        self, boundary: Boundary3D, normal_derivative: float
    ) -> None: ...
    @overload
    def set_neumann_boundary(
        self, boundary_id: int, normal_derivative: float
    ) -> None: ...
    def step(self, dt: float) -> ADISolveResult: ...
    def mesh(self) -> StructuredMesh3D:
        """The mesh this solver was built on."""
        ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> ADISolveResult: ...
    def solution(self) -> FloatArray:
        """Return an owned copy of the current flat field."""
        ...
    def time(self) -> float: ...
    @property
    def diffusivity(self) -> float: ...

class DiffusionSolver:
    """Solver for the diffusion equation.

    Solves the transient diffusion (heat) equation:
        ∂u/∂t = D ∇²u

    using explicit finite differences. The caller supplies ``dt``; this legacy
    surface does not choose a stable step automatically.

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

    def set_initial_condition(self, values: ArrayLike) -> None:
        """Set initial concentration field."""
        ...

    @overload
    def set_boundary_condition(self, boundary_id: int, bc: BoundaryCondition) -> None:
        """Set boundary metadata by integer ID."""
        ...

    @overload
    def set_boundary_condition(self, boundary: Boundary, bc: BoundaryCondition) -> None:
        """Set boundary condition."""
        ...

    @overload
    def set_dirichlet_boundary(self, boundary_id: int, value: float) -> None: ...
    @overload
    def set_dirichlet_boundary(self, boundary: Boundary, value: float) -> None:
        """Set Dirichlet boundary condition."""
        ...

    @overload
    def set_neumann_boundary(
        self, boundary_id: int, normal_derivative: float
    ) -> None: ...
    @overload
    def set_neumann_boundary(
        self, boundary: Boundary, normal_derivative: float
    ) -> None:
        """Set the outward-normal derivative ``du/dn``.

        This is a derivative, not a Fickian flux; the outward diffusive flux is
        ``-D * normal_derivative``.
        """
        ...
    def max_stable_time_step(self) -> float:
        """Largest explicit step accepted by ``check_stability`` for pure diffusion.

        Returns ``inf`` when the diffusivity is zero.
        """
        ...

    def time(self) -> float:
        """Current simulation time advanced by ``solve``."""
        ...
    def check_stability(self, dt: float) -> bool:
        """Return whether ``dt`` satisfies the explicit stability condition."""
        ...
    def mesh(self) -> StructuredMesh:
        """The mesh this solver was built on."""
        ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> None:
        """Advance solution in time."""
        ...

    def solution(self) -> FloatArray:
        """Return an owned copy of the current flat field."""
        ...

class DiffusionSolver3D:
    """Conservative Forward Euler solver for constant-D Cartesian 3D diffusion."""

    def __init__(self, mesh: StructuredMesh3D, diffusivity: float) -> None: ...
    def set_initial_condition(self, values: ArrayLike) -> None: ...
    @overload
    def set_dirichlet_boundary(self, boundary_id: int, value: float) -> None: ...
    @overload
    def set_dirichlet_boundary(self, boundary: Boundary3D, value: float) -> None: ...
    @overload
    def set_neumann_boundary(
        self, boundary_id: int, normal_derivative: float
    ) -> None: ...
    @overload
    def set_neumann_boundary(
        self, boundary: Boundary3D, normal_derivative: float
    ) -> None: ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> None: ...
    def check_stability(self, dt: float) -> bool: ...
    def max_stable_time_step(self) -> float: ...
    def time(self) -> float: ...
    def solution(self) -> FloatArray:
        """Return an owned copy of the current flat field."""
        ...
    def mesh(self) -> StructuredMesh3D: ...

class LinearReactionDiffusionSolver3D:
    """3D Forward-Euler diffusion with Backward-Euler linear decay.

    The IMEX update is first order in time and remains restricted by the
    explicit diffusion CFL ceiling.
    """

    def __init__(
        self, mesh: StructuredMesh3D, diffusivity: float, decay_rate: float
    ) -> None: ...
    def set_initial_condition(self, values: ArrayLike) -> None: ...
    @overload
    def set_dirichlet_boundary(self, boundary_id: int, value: float) -> None: ...
    @overload
    def set_dirichlet_boundary(self, boundary: Boundary3D, value: float) -> None: ...
    @overload
    def set_neumann_boundary(
        self, boundary_id: int, normal_derivative: float
    ) -> None: ...
    @overload
    def set_neumann_boundary(
        self, boundary: Boundary3D, normal_derivative: float
    ) -> None: ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> None: ...
    def check_stability(self, dt: float) -> bool: ...
    def max_stable_time_step(self) -> float: ...
    def decay_rate(self) -> float: ...
    def time(self) -> float: ...
    def solution(self) -> FloatArray:
        """Return an owned copy of the current flat field."""
        ...
    def mesh(self) -> StructuredMesh3D: ...

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
        vx_field: ArrayLike,
        vy_field: ArrayLike,
        scheme: AdvectionScheme = ...,
    ) -> None:
        """Create advection-diffusion solver with velocity field."""
        ...

    def scheme(self) -> AdvectionScheme:
        """Current advection scheme."""
        ...

    def cell_peclet(self) -> float:
        """Cell Peclet number."""
        ...

    def max_time_step(self, safety: float = 0.4) -> float:
        """Maximum stable time step."""
        ...

    def is_scheme_stable(self) -> bool:
        """Whether the selected spatial scheme meets its cell-Peclet criterion."""
        ...

    def set_scheme(self, scheme: AdvectionScheme) -> None:
        """Set advection discretization scheme."""
        ...

    def set_initial_condition(self, values: ArrayLike) -> None:
        """Set initial concentration field."""
        ...

    @overload
    def set_boundary(self, boundary_id: int, bc: BoundaryCondition) -> None:
        """Set boundary metadata by integer ID."""
        ...

    @overload
    def set_boundary(self, boundary: Boundary, bc: BoundaryCondition) -> None:
        """Set boundary condition."""
        ...

    def time(self) -> float:
        """Current simulation time advanced by ``solve``."""
        ...
    def check_stability(self, dt: float) -> bool:
        """Return whether ``dt`` satisfies the explicit stability condition."""
        ...
    def mesh(self) -> StructuredMesh:
        """The mesh this solver was built on."""
        ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> None:
        """Advance solution in time."""
        ...

    def solution(self) -> FloatArray:
        """Return an owned copy of the current flat field."""
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

    def __init__(
        self,
        mesh: StructuredMesh,
        diffusivity: float,
        reaction: Callable[[float, float, float, float], float],
    ) -> None: ...
    def set_initial_condition(self, values: ArrayLike) -> None:
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
    def set_neumann_boundary(self, boundary_id: int, normal_derivative: float) -> None:
        """Set the outward-normal derivative ``du/dn`` by boundary ID."""
        ...

    @overload
    def set_neumann_boundary(
        self, boundary: Boundary, normal_derivative: float
    ) -> None:
        """Set the outward-normal derivative ``du/dn``; not a Fickian flux."""
        ...

    @overload
    def set_boundary(self, boundary_id: int, bc: BoundaryCondition) -> None: ...
    @overload
    def set_boundary(self, boundary: Boundary, bc: BoundaryCondition) -> None:
        """Set boundary condition metadata."""
        ...

    def time(self) -> float:
        """Current simulation time advanced by ``solve``."""
        ...
    def check_stability(self, dt: float) -> bool:
        """Return whether ``dt`` satisfies the explicit stability condition."""
        ...
    def mesh(self) -> StructuredMesh:
        """The mesh this solver was built on."""
        ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> None:
        """Solve for specified number of time steps."""
        ...

    def solution(self) -> FloatArray:
        """Return an owned copy of the current flat field."""
        ...

class ConstantSourceReactionDiffusionSolver:
    """Reaction-diffusion with constant source term."""

    def __init__(
        self, mesh: StructuredMesh, diffusivity: float, source_rate: float
    ) -> None:
        """Create solver with constant source."""
        ...

    def set_initial_condition(self, values: ArrayLike) -> None: ...
    @overload
    def set_boundary(self, boundary_id: int, bc: BoundaryCondition) -> None: ...
    @overload
    def set_boundary(self, boundary: Boundary, bc: BoundaryCondition) -> None: ...
    def time(self) -> float:
        """Current simulation time advanced by ``solve``."""
        ...
    def check_stability(self, dt: float) -> bool:
        """Return whether ``dt`` satisfies the explicit stability condition."""
        ...
    def mesh(self) -> StructuredMesh:
        """The mesh this solver was built on."""
        ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> None: ...
    def solution(self) -> FloatArray: ...

class LinearReactionDiffusionSolver:
    """Reaction-diffusion with linear decay: R = -k*u."""

    def __init__(
        self, mesh: StructuredMesh, diffusivity: float, decay_rate: float
    ) -> None:
        """Create solver with decay constant k."""
        ...

    def set_initial_condition(self, values: ArrayLike) -> None: ...
    @overload
    def set_boundary(self, boundary_id: int, bc: BoundaryCondition) -> None: ...
    @overload
    def set_boundary(self, boundary: Boundary, bc: BoundaryCondition) -> None: ...
    def time(self) -> float:
        """Current simulation time advanced by ``solve``."""
        ...
    def check_stability(self, dt: float) -> bool:
        """Return whether ``dt`` satisfies the explicit stability condition."""
        ...
    def mesh(self) -> StructuredMesh:
        """The mesh this solver was built on."""
        ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> None: ...
    def solution(self) -> FloatArray: ...

class LogisticReactionDiffusionSolver:
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

    def set_initial_condition(self, values: ArrayLike) -> None: ...
    @overload
    def set_boundary(self, boundary_id: int, bc: BoundaryCondition) -> None: ...
    @overload
    def set_boundary(self, boundary: Boundary, bc: BoundaryCondition) -> None: ...
    def time(self) -> float:
        """Current simulation time advanced by ``solve``."""
        ...
    def check_stability(self, dt: float) -> bool:
        """Return whether ``dt`` satisfies the explicit stability condition."""
        ...
    def mesh(self) -> StructuredMesh:
        """The mesh this solver was built on."""
        ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> None: ...
    def solution(self) -> FloatArray: ...

class MichaelisMentenReactionDiffusionSolver:
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

    def set_initial_condition(self, values: ArrayLike) -> None: ...
    @overload
    def set_boundary(self, boundary_id: int, bc: BoundaryCondition) -> None: ...
    @overload
    def set_boundary(self, boundary: Boundary, bc: BoundaryCondition) -> None: ...
    def time(self) -> float:
        """Current simulation time advanced by ``solve``."""
        ...
    def check_stability(self, dt: float) -> bool:
        """Return whether ``dt`` satisfies the explicit stability condition."""
        ...
    def mesh(self) -> StructuredMesh:
        """The mesh this solver was built on."""
        ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> None: ...
    def solution(self) -> FloatArray: ...

class MaskedMichaelisMentenReactionDiffusionSolver:
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

    def set_initial_condition(self, values: ArrayLike) -> None: ...
    @overload
    def set_boundary(self, boundary_id: int, bc: BoundaryCondition) -> None: ...
    @overload
    def set_boundary(self, boundary: Boundary, bc: BoundaryCondition) -> None: ...
    def time(self) -> float:
        """Current simulation time advanced by ``solve``."""
        ...
    def check_stability(self, dt: float) -> bool:
        """Return whether ``dt`` satisfies the explicit stability condition."""
        ...
    def mesh(self) -> StructuredMesh:
        """The mesh this solver was built on."""
        ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> None: ...
    def solution(self) -> FloatArray: ...

class MultiSpeciesSolver:
    """Explicit conservative N-species reaction-diffusion solver.

    ``max_stable_time_step`` and ``check_stability`` cover only the exact
    Forward Euler diffusion CFL condition. Reaction kinetics can impose a
    smaller admissible step. Neumann values are outward-normal concentration
    derivatives, not Fickian fluxes.
    """

    def __init__(
        self,
        mesh: StructuredMesh,
        diffusivities: list[float],
        num_species: int = 0,
    ) -> None: ...
    def set_reaction_function(
        self,
        reaction: Callable[
            [list[float], Sequence[float], float, float, float],
            Optional[Union[Sequence[float], FloatArray]],
        ],
    ) -> None:
        """Set ``reaction(rates, concentrations, x, y, t)``.

        ``rates`` starts as one zero per species and has concentration-per-time
        units. Mutate it and return ``None``, or return a one-dimensional
        sequence/NumPy array of rates; a returned value takes precedence.
        ``concentrations`` is read-only by contract. Both vectors use the
        solver's species order, and every returned rate must be finite.
        """
        ...
    @overload
    def set_reaction_model(self, model: LotkaVolterraReaction) -> None: ...
    @overload
    def set_reaction_model(self, model: SIRReaction) -> None: ...
    @overload
    def set_reaction_model(self, model: SEIRReaction) -> None: ...
    @overload
    def set_reaction_model(self, model: BrusselatorReaction) -> None: ...
    @overload
    def set_reaction_model(self, model: CompetitiveInhibitionReaction) -> None: ...
    @overload
    def set_reaction_model(self, model: EnzymeCascadeReaction) -> None: ...
    def set_initial_condition(self, species_idx: int, values: ArrayLike) -> None: ...
    def set_uniform_initial_condition(self, species_idx: int, value: float) -> None: ...
    @overload
    def set_dirichlet_boundary(
        self, species_idx: int, boundary: Boundary, value: float
    ) -> None: ...
    @overload
    def set_dirichlet_boundary(
        self, species_idx: int, boundary_id: int, value: float
    ) -> None: ...
    @overload
    def set_neumann_boundary(
        self,
        species_idx: int,
        boundary: Boundary,
        normal_derivative: float,
    ) -> None: ...
    @overload
    def set_neumann_boundary(
        self,
        species_idx: int,
        boundary_id: int,
        normal_derivative: float,
    ) -> None: ...
    def set_all_species_dirichlet(self, boundary: Boundary, value: float) -> None: ...
    def set_all_species_neumann(
        self, boundary: Boundary, normal_derivative: float
    ) -> None: ...
    def check_stability(self, dt: float) -> bool: ...
    def max_stable_time_step(self) -> float: ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> None: ...
    def solution(self, species_idx: int) -> FloatArray:
        """Return an owned flat solution copy for one species."""
        ...
    def all_solutions(self) -> list[FloatArray]:
        """Return owned flat copies for all species."""
        ...
    def mesh(self) -> StructuredMesh: ...
    def num_species(self) -> int: ...
    def diffusivity(self, species_idx: int) -> float: ...
    def time(self) -> float: ...
    def reset_time(self) -> None: ...
    def total_concentration(self, node_idx: int) -> float: ...
    def concentration(self, species_idx: int, node_idx: int) -> float: ...
    def solution_norm(self, species_idx: int) -> float: ...
    def total_mass(self, species_idx: int) -> float: ...

class LotkaVolterraReaction:
    """Two-species predator-prey kinetics with logistic prey growth."""

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        carrying_capacity: float = 100.0,
    ) -> None: ...
    @property
    def alpha(self) -> float: ...
    @property
    def beta(self) -> float: ...
    @property
    def gamma(self) -> float: ...
    @property
    def delta(self) -> float: ...
    @property
    def carrying_capacity(self) -> float: ...

class SIRReaction:
    """SIR kinetics normalized by a local reference population ``N``.

    For spatial density fields, ``N`` has the same local units as S/I/R and is
    not the domain-integrated population. ``R0`` is ``beta/gamma`` and its usual
    interpretation assumes the initial susceptible value is approximately N.
    """

    def __init__(self, beta: float, gamma: float, total_population: float) -> None: ...
    @property
    def beta(self) -> float: ...
    @property
    def gamma(self) -> float: ...
    @property
    def N(self) -> float: ...
    @property
    def R0(self) -> float: ...

class SEIRReaction:
    """SEIR kinetics normalized by a local reference population ``N``."""

    def __init__(
        self,
        beta: float,
        sigma: float,
        gamma: float,
        total_population: float,
    ) -> None: ...
    @property
    def beta(self) -> float: ...
    @property
    def sigma(self) -> float: ...
    @property
    def gamma(self) -> float: ...
    @property
    def N(self) -> float: ...

class BrusselatorReaction:
    """Conventional nondimensional two-species Brusselator kinetics."""

    def __init__(self, A: float, B: float) -> None: ...
    @property
    def A(self) -> float: ...
    @property
    def B(self) -> float: ...
    @property
    def is_oscillatory(self) -> bool: ...

class CompetitiveInhibitionReaction:
    """Three-species substrate/inhibitor/product kinetic model."""

    def __init__(
        self,
        vmax: float,
        km: float,
        ki: float,
        inhibitor_decay: float = 0.0,
    ) -> None: ...
    @property
    def vmax(self) -> float: ...
    @property
    def km(self) -> float: ...
    @property
    def ki(self) -> float: ...

class EnzymeCascadeReaction:
    """Sequential Michaelis-Menten activation/degradation kinetics."""

    def __init__(
        self,
        vmax_values: list[float],
        km_values: list[float],
        kdeg_values: list[float],
    ) -> None: ...
    @property
    def num_enzymes(self) -> int: ...

class GrayScottSolver:
    """Gray-Scott reaction-diffusion pattern formation solver.

    Simulates the nondimensional Gray-Scott model on the periodic cell-centred
    grid of a 2D ``StructuredMesh``. Input fields therefore have
    ``mesh.nx()*mesh.ny()`` values, rather than one value per mesh node.

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
    - Diffusion-driven and excitable patterns

    Examples:
        >>> mesh = StructuredMesh(128, 128, 0.0, 2.5, 0.0, 2.5)
        >>> solver = GrayScottSolver(mesh, Du=0.16, Dv=0.08, f=0.035, k=0.065)
        >>> # Initialize with u=1, v=0 and seed perturbation
        >>> result = solver.simulate(
        ...     u0, v0, total_steps=10000, dt=1.0, steps_between_frames=100
        ... )
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
        self,
        u0: ArrayLike,
        v0: ArrayLike,
        total_steps: int,
        dt: float,
        steps_between_frames: int = 1000,
        check_interval: int = 1000,
        stable_tol: float = 1e-4,
        min_frames_before_early_stop: int = 6,
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

    def reynolds(self, L: float, U: float, rho: float) -> float:
        """Return the diagnostic ``rho*U*L/viscosity``."""
        ...

    def set_velocity_bc(self, side: Boundary, bc: VelocityBC) -> StokesSolver:
        """Set velocity boundary condition."""
        ...

    def set_body_force(self, fx: float, fy: float) -> StokesSolver:
        """Set body force (e.g., gravity)."""
        ...

    def set_tolerance(self, tol: float) -> StokesSolver:
        """Set convergence tolerance."""
        ...

    def set_max_iterations(self, max_iter: int) -> StokesSolver:
        """Set maximum iterations."""
        ...

    def set_velocity_relaxation(self, omega_v: float) -> StokesSolver:
        """Set velocity under-relaxation factor."""
        ...

    def set_pressure_relaxation(self, omega_p: float) -> StokesSolver:
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

    The Reynolds number ``Re = rho*U*L/mu`` helps characterize a flow, but
    transition thresholds depend on geometry and disturbances. This solver has
    no turbulence closure and is intended for resolved laminar flows.

    This bounded MAC-grid implementation supports no-slip and flux-compatible
    Dirichlet velocity boundaries. Open/traction boundaries and the reserved
    QUICK/HYBRID schemes raise until compatible implementations are available.

    Examples:
        >>> mesh = StructuredMesh(40, 40, 0.0, 1.0, 0.0, 1.0)
        >>> solver = NavierStokesSolver(mesh, density=1000.0, viscosity=0.001)
        >>> solver.set_velocity_bc(Boundary.Top, VelocityBC.dirichlet(0.1, 0.0))
        >>> result = solver.solve(duration=0.1)
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

    def reynolds(self, L: float, U: float) -> float:
        """Return ``rho*U*L/mu``."""
        ...

    def set_velocity_bc(self, side: Boundary, bc: VelocityBC) -> NavierStokesSolver:
        """Set velocity boundary condition."""
        ...

    def set_body_force(self, fx: float, fy: float) -> NavierStokesSolver:
        """Set body force."""
        ...

    def set_initial_velocity(self, u0: ArrayLike, v0: ArrayLike) -> NavierStokesSolver:
        """Set initial velocity field."""
        ...

    def set_time_step(self, dt: float) -> NavierStokesSolver:
        """Set time step."""
        ...

    def set_cfl(self, cfl: float) -> NavierStokesSolver:
        """Set CFL number for automatic time stepping."""
        ...

    def set_convection_scheme(self, scheme: ConvectionScheme) -> NavierStokesSolver:
        """Set convection discretization scheme."""
        ...

    def set_pressure_tolerance(self, tol: float) -> NavierStokesSolver:
        """Set pressure solver tolerance."""
        ...

    def set_max_pressure_iterations(self, max_iter: int) -> NavierStokesSolver:
        """Set maximum pressure solver iterations."""
        ...

    def max_time_step(self, u: FloatArray, v: FloatArray) -> float:
        """Return the explicit stability ceiling for packed MAC fields."""
        ...

    def solve(
        self, duration: float, output_interval: float = 0.0
    ) -> NavierStokesResult:
        """Solve to specified end time."""
        ...

    def solve_steps(self, num_steps: int) -> NavierStokesResult:
        """Solve for specified number of time steps."""
        ...

class DarcyFlowSolver:
    """Solver for Darcy flow in porous media.

    Solves Darcy's law for flow through porous media:
        v = -κ ∇p
        ∇·v = 0

    where ``κ = K/μ`` is hydraulic conductivity [m²/(Pa s)], p is
    pressure, and v is the superficial (Darcy) velocity. The solver takes
    hydraulic conductivity directly; it does not take viscosity separately.

    Applications include:
    - Groundwater flow and contaminant transport
    - Blood flow through tissue (interstitial flow)
    - Flow in biological scaffolds
    - Drug transport in tumors

    The solver supports spatially-varying hydraulic conductivity for
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
        """Create a solver with uniform hydraulic conductivity ``kappa``."""
        ...

    @overload
    def __init__(self, mesh: StructuredMesh, kappa: ArrayLike) -> None:
        """Create a solver with a nodal hydraulic-conductivity field."""
        ...

    def kappa(self) -> FloatArray:
        """Return an owned copy of nodal hydraulic conductivity."""
        ...

    def set_dirichlet(self, side: Boundary, pressure: float) -> DarcyFlowSolver:
        """Set pressure Dirichlet boundary condition."""
        ...

    def set_outward_pressure_gradient(
        self, side: Boundary, outward_pressure_gradient_Pa_per_m: float
    ) -> DarcyFlowSolver:
        """Set outward ``dp/dn`` [Pa/m], not Darcy velocity or volumetric flux."""
        ...

    def set_neumann(self, side: Boundary, flux: float) -> DarcyFlowSolver:
        """Set outward ``dp/dn``.

        The runtime keyword ``flux`` is retained for compatibility but is a
        pressure derivative, not Darcy velocity or volumetric flux.
        """
        ...

    def set_internal_pressure(
        self, mask: list[int], pressure: float
    ) -> DarcyFlowSolver:
        """Pin pressure where the byte-valued node mask is nonzero."""
        ...

    def set_initial_guess(self, pressure: ArrayLike) -> DarcyFlowSolver:
        """Set initial pressure guess."""
        ...

    def set_tolerance(self, tol: float) -> DarcyFlowSolver:
        """Set convergence tolerance."""
        ...

    def set_max_iterations(self, max_iter: int) -> DarcyFlowSolver:
        """Set maximum iterations."""
        ...

    def set_omega(self, omega: float) -> DarcyFlowSolver:
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
        >>> print(f"Flux: {result.flux:.2e}")
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

    def set_membrane_thickness(self, L: float) -> MembraneDiffusion1DSolver:
        """Set membrane thickness."""
        ...

    def set_diffusivity(self, D: float) -> MembraneDiffusion1DSolver:
        """Set diffusion coefficient."""
        ...

    def set_partition_coefficient(self, Phi: float) -> MembraneDiffusion1DSolver:
        """Set partition coefficient."""
        ...

    def set_left_concentration(self, C: float) -> MembraneDiffusion1DSolver:
        """Set left boundary concentration."""
        ...

    def set_right_concentration(self, C: float) -> MembraneDiffusion1DSolver:
        """Set right boundary concentration."""
        ...

    def set_num_nodes(self, n: int) -> MembraneDiffusion1DSolver:
        """Set number of grid nodes."""
        ...

    def set_hindered_diffusion(
        self, solute_radius: float, pore_radius: float
    ) -> MembraneDiffusion1DSolver:
        """Enable Renkin hindrance using the two radii in consistent units."""
        ...

    def disable_hindered_diffusion(self) -> MembraneDiffusion1DSolver:
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
        >>> solver.add_layer(
        ...     thickness=10e-6, diffusivity=1e-11, partition_coefficient=0.5
        ... )
        >>> solver.add_layer(
        ...     thickness=50e-6, diffusivity=1e-10, partition_coefficient=1.0
        ... )
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
        self,
        thickness: float,
        diffusivity: float,
        partition_coefficient: float = 1.0,
    ) -> MultiLayerMembraneSolver:
        """Add a membrane layer."""
        ...

    def clear_layers(self) -> MultiLayerMembraneSolver:
        """Remove all layers."""
        ...

    def set_left_concentration(self, C: float) -> MultiLayerMembraneSolver:
        """Set left boundary concentration."""
        ...

    def set_right_concentration(self, C: float) -> MultiLayerMembraneSolver:
        """Set right boundary concentration."""
        ...

    def solve(self) -> MembraneDiffusionResult:
        """Solve for steady-state concentration profile."""
        ...

# =============================================================================
# Application-Specific Solvers
# =============================================================================

class TumorDrugDeliverySolver:
    """Reduced Darcy/vascular-exchange/drug-transport model.

    Models drug transport in tumor tissue including:
    - Interstitial fluid flow (pressure-driven convection)
    - Drug diffusion in interstitial space
    - Binding to extracellular matrix
    - Cellular uptake

    The tumor microenvironment is characterized by:
    - Elevated interstitial fluid pressure (IFP)
    - Heterogeneous hydraulic conductivity
    - Tortuous diffusion paths

    Plasma concentration is prescribed and constant. Binding and uptake are
    irreversible first-order compartments; saturation, pharmacokinetics, and
    patient-specific calibration are not represented.

    The pressure solve clamps tumor-mask nodes to ``p_tumor``; that represents
    an unresolved solute-free fluid source. It is not a Starling filtration
    model and there is no solvent-drag term in vascular solute exchange.

    Examples:
        >>> mesh = StructuredMesh(50, 50, 0.0, 0.01, 0.0, 0.01)
        >>> tumor_mask = create_circular_tumor(mesh, center, radius)
        >>> solver = TumorDrugDeliverySolver(
        ...     mesh, tumor_mask, hydraulic_conductivity,
        ...     p_boundary=0.0, p_tumor=2000.0  # Pa
        ... )
        >>> pressure = solver.solve_pressure_sor()
        >>> result = solver.simulate(
        ...     pressure, diffusivity, vessel_wall_solute_permeability,
        ...     vascular_surface_area_density, k_binding, k_uptake,
        ...     c_plasma, dt=1.0, num_steps=3600,
        ...     times_to_save_s=[0.0, 3600.0],
        ... )
    """

    def __init__(
        self,
        mesh: StructuredMesh,
        tumor_mask: list[int],
        hydraulic_conductivity: ArrayLike,
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
        pressure: ArrayLike,
        diffusivity: ArrayLike,
        vessel_wall_solute_permeability: ArrayLike,
        vascular_surface_area_density: ArrayLike,
        k_binding: float,
        k_uptake: float,
        c_plasma: float,
        dt: float,
        num_steps: int,
        times_to_save_s: list[float],
    ) -> TumorDrugDeliverySaved:
        """Run coupled simulation."""
        ...

    def solve_pressure_sor(
        self, max_iter: int = 20000, tol: float = 1e-10, omega: float = 1.8
    ) -> list[float]:
        """Solve pressure field with SOR."""
        ...

class BioheatCryotherapySolver:
    """Pennes bioheat phase-change solver with a heat-injury diagnostic.

    Solves the Pennes bioheat equation with an apparent-heat-capacity
    phase-change model. All temperature inputs are absolute kelvin.

    The model includes:
    - Heat conduction with temperature-dependent properties
    - Blood perfusion heat source (with perfusion shutdown in frozen tissue)
    - Metabolic heat generation
    - Phase change (freezing/thawing) with latent heat
    - Arrhenius damage integral: Ω = ∫A·exp(-Ea/RT)dt

    The Arrhenius integral describes heat injury for a parameterization supplied
    by the caller; it must not be interpreted as cryogenic cell death.

    Probe-mask nodes are embedded fixed-temperature nodes. Contact resistance,
    probe coolant flow, and a conjugate probe model are not represented.

    Examples:
        >>> # Set up mesh with probe region
        >>> mesh = StructuredMesh(100, 100, -0.02, 0.02, -0.02, 0.02)
        >>> probe_mask = create_probe_region(mesh)
        >>> solver = BioheatCryotherapySolver(
        ...     mesh, probe_mask, perfusion_map, q_met_map,
        ...     rho_tissue=1000, rho_blood=1000, c_blood=3600,
        ...     k_unfrozen=0.5, k_frozen=2.0, ...
        ... )
        >>> result = solver.simulate(
        ...     dt=0.05, num_steps=12000, times_to_save_s=[0.0, 600.0]
        ... )
    """

    def __init__(
        self,
        mesh: StructuredMesh,
        probe_mask: list[int],
        perfusion_map: ArrayLike,
        q_met_map: ArrayLike,
        rho_tissue: float,
        rho_blood: float,
        c_blood: float,
        k_unfrozen: float,
        k_frozen: float,
        c_unfrozen: float,
        c_frozen: float,
        T_body_K: float,
        T_probe_K: float,
        T_freeze_K: float,
        T_freeze_range_K: float,
        L_fusion: float,
        A: float,
        E_a: float,
        R_gas: float,
    ) -> None:
        """Create bioheat cryotherapy solver."""
        ...

    def set_initial_temperature_K(
        self, temperature_K: float
    ) -> BioheatCryotherapySolver: ...
    def set_initial_temperature_field_K(
        self, temperature_K: ArrayLike
    ) -> BioheatCryotherapySolver: ...
    def set_arterial_temperature_K(
        self, temperature_K: float
    ) -> BioheatCryotherapySolver: ...
    def set_boundary_temperature_K(
        self, temperature_K: float
    ) -> BioheatCryotherapySolver: ...
    def frozen_fraction(self, temperature_K: float) -> float: ...
    def thermal_conductivity(self, temperature_K: float) -> float: ...
    def effective_specific_heat(self, temperature_K: float) -> float: ...
    def arrhenius_heat_injury_rate(self, temperature_K: float) -> float: ...
    def maximum_stable_time_step_s(self) -> float: ...
    def simulate(
        self, dt: float, num_steps: int, times_to_save_s: list[float]
    ) -> BioheatSaved: ...

# =============================================================================
# Viscosity Models (Rheology)
# =============================================================================

class ViscosityModel:
    """Non-constructible base for scalar generalized-Newtonian laws."""

    def viscosity(self, gamma_dot: float) -> float:
        """Return apparent dynamic viscosity for a shear-rate magnitude."""
        ...
    def shear_stress(self, gamma_dot: float) -> float:
        """Return the model's nonnegative scalar shear-stress magnitude."""
        ...
    def name(self) -> str: ...
    def type(self) -> FluidModel: ...

class NewtonianModel(ViscosityModel):
    """Newtonian (constant viscosity) fluid model.

    For a Newtonian fluid, the shear stress is linearly proportional
    to the shear rate:
        τ = μ · γ̇

    where μ is the constant dynamic viscosity.

    Most simple fluids (water, air, simple organic solvents) exhibit
    Newtonian behavior. This model serves as a baseline for comparison
    with non-Newtonian models.

    Examples:
        >>> model = NewtonianModel(mu0=0.001)  # Water near room temperature
        >>> tau = model.shear_stress(100.0)  # Stress at γ̇ = 100 s⁻¹
        >>> print(f"Shear stress: {tau:.2f} Pa")
    """

    def __init__(self, mu0: float) -> None:
        """Create a Newtonian model with dynamic viscosity ``mu0`` [Pa s]."""
        ...

    def mu0(self) -> float:
        """Viscosity."""
        ...

    def name(self) -> str:
        """Model name."""
        ...

    def type(self) -> FluidModel:
        """Model type enum."""
        ...

    def viscosity(self, gamma_dot: float) -> float:
        """Viscosity at given shear rate."""
        ...

    def shear_stress(self, gamma_dot: float) -> float:
        """Shear stress at given shear rate."""
        ...

class PowerLawModel(ViscosityModel):
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

    def __init__(self, K: float, n: float, gamma_min: float = 1e-10) -> None:
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

    def type(self) -> FluidModel:
        """Model type enum."""
        ...

    def viscosity(self, gamma_dot: float) -> float:
        """Viscosity at given shear rate."""
        ...

    def shear_stress(self, gamma_dot: float) -> float:
        """Shear stress at given shear rate."""
        ...

class CarreauModel(ViscosityModel):
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

    def type(self) -> FluidModel:
        """Model type enum."""
        ...

    def viscosity(self, gamma_dot: float) -> float:
        """Viscosity at given shear rate."""
        ...

    def shear_stress(self, gamma_dot: float) -> float:
        """Shear stress at given shear rate."""
        ...

class CarreauYasudaModel(ViscosityModel):
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

    def type(self) -> FluidModel:
        """Model type enum."""
        ...

    def viscosity(self, gamma_dot: float) -> float:
        """Viscosity at given shear rate."""
        ...

    def shear_stress(self, gamma_dot: float) -> float:
        """Shear stress at given shear rate."""
        ...

class CrossModel(ViscosityModel):
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

    def type(self) -> FluidModel:
        """Model type enum."""
        ...

    def viscosity(self, gamma_dot: float) -> float:
        """Viscosity at given shear rate."""
        ...

    def shear_stress(self, gamma_dot: float) -> float:
        """Shear stress at given shear rate."""
        ...

class CassonModel(ViscosityModel):
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

    def __init__(self, tau_y: float, mu_p: float, epsilon: float = 1e-6) -> None:
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

    def type(self) -> FluidModel:
        """Model type enum."""
        ...

    def viscosity(self, gamma_dot: float) -> float:
        """Apparent viscosity at given shear rate."""
        ...

    def shear_stress(self, gamma_dot: float) -> float:
        """Shear stress at given shear rate."""
        ...

class BinghamModel(ViscosityModel):
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

    def __init__(self, tau_y: float, mu_p: float, epsilon: float = 1e-6) -> None:
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

    def type(self) -> FluidModel:
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

class HerschelBulkleyModel(ViscosityModel):
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

    def __init__(self, tau_y: float, K: float, n: float, epsilon: float = 1e-6) -> None:
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

    def type(self) -> FluidModel:
        """Model type enum."""
        ...

    def viscosity(self, gamma_dot: float) -> float:
        """Apparent viscosity at given shear rate."""
        ...

    def shear_stress(self, gamma_dot: float) -> float:
        """Shear stress at given shear rate."""
        ...

# =============================================================================
# Electrochemical transport
# =============================================================================

class IonSpecies:
    """Ion transport parameters with temperature-aware Einstein mobility."""

    def __init__(
        self,
        name: str,
        valence: int,
        diffusivity: float,
        temperature: float = 310.0,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def valence(self) -> int: ...
    @property
    def diffusivity(self) -> float: ...
    @property
    def mobility(self) -> float:
        """Electrical-mobility magnitude at ``mobility_temperature``."""
        ...
    @property
    def mobility_temperature(self) -> float: ...
    @staticmethod
    def thermal_voltage(temperature: float = 310.0) -> float: ...
    def mobility_at(self, temperature: float) -> float:
        """Evaluate mobility magnitude at an absolute temperature [K]."""
        ...

class NernstPlanckSolver:
    """Single-ion Nernst-Planck transport in a prescribed potential.

    Uses conservative fitted diffusion-drift fluxes. It does not solve
    Poisson's equation, membrane gating, or an action-potential model.
    """

    def __init__(
        self, mesh: StructuredMesh, ion: IonSpecies, temperature: float = 310.0
    ) -> None: ...
    def set_initial_condition(self, values: ArrayLike) -> None: ...
    def set_potential_field(self, phi: ArrayLike) -> None: ...
    def set_uniform_field(self, Ex: float, Ey: float = 0.0) -> None: ...
    @overload
    def set_dirichlet_boundary(self, boundary: Boundary, value: float) -> None: ...
    @overload
    def set_dirichlet_boundary(self, boundary_id: int, value: float) -> None: ...
    @overload
    def set_outward_flux_boundary(
        self, boundary: Boundary, outward_molar_flux: float
    ) -> None:
        """Prescribe the outward total molar flux ``N·n`` [mol/(m² s)].

        Positive values leave the domain. This is a physical flux, not the
        outward-normal derivative that ``set_neumann_boundary`` means on the
        scalar diffusion solvers.
        """
        ...
    @overload
    def set_outward_flux_boundary(
        self, boundary_id: int, outward_molar_flux: float
    ) -> None:
        """Prescribe the outward total molar flux on a boundary given by integer id."""
        ...
    @overload
    def set_neumann_boundary(self, boundary: Boundary, flux: float) -> None:
        """Deprecated spelling of :meth:`set_outward_flux_boundary`."""
        ...
    @overload
    def set_neumann_boundary(self, boundary_id: int, flux: float) -> None:
        """Deprecated spelling of :meth:`set_outward_flux_boundary`."""
        ...
    def check_stability(self, dt: float) -> bool: ...
    def maximum_stable_time_step(self) -> float: ...
    def recommended_time_step(self, safety: float = 0.9) -> float: ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> None: ...
    def solution(self) -> FloatArray:
        """Return an owned concentration copy."""
        ...
    def potential(self) -> FloatArray:
        """Return an owned potential copy."""
        ...
    def compute_current_density(self) -> FloatArray:
        """Return interleaved Cartesian ionic current density [A/m²]."""
        ...
    def time(self) -> float: ...
    def ion(self) -> IonSpecies:
        """Return the solver-owned species object; the solver keeps it alive."""
        ...
    def thermal_voltage(self) -> float: ...
    def electrical_mobility(self) -> float: ...
    def mesh(self) -> StructuredMesh:
        """Return the internally retained mesh object."""
        ...

class MultiIonSolver:
    """Multiple independent ions in one prescribed potential field.

    This solver does not solve Poisson's equation or enforce electroneutrality.
    """

    def __init__(
        self,
        mesh: StructuredMesh,
        ions: list[IonSpecies],
        temperature: float = 310.0,
    ) -> None: ...
    def set_initial_condition(self, species: int, values: ArrayLike) -> None: ...
    @overload
    def set_dirichlet_boundary(
        self, species: int, boundary: Boundary, value: float
    ) -> None: ...
    @overload
    def set_dirichlet_boundary(
        self, species: int, boundary_id: int, value: float
    ) -> None: ...
    def set_outward_flux_boundary(
        self, species: int, boundary: Boundary, outward_molar_flux: float
    ) -> None:
        """Prescribe the outward total molar flux of one species [mol/(m² s)].

        Positive values leave the domain. This is a physical flux, not a
        concentration derivative.
        """
        ...
    def set_neumann_boundary(
        self, species: int, boundary: Boundary, flux: float
    ) -> None:
        """Deprecated spelling of :meth:`set_outward_flux_boundary`."""
        ...
    def set_potential_field(self, phi: ArrayLike) -> None: ...
    def set_uniform_field(self, Ex: float, Ey: float = 0.0) -> None: ...
    def set_electroneutrality_mode(
        self, enable: bool, background_charge: float = 0.0
    ) -> None:
        """Compatibility method; enabling unsupported coupling raises."""
        ...
    def check_stability(self, dt: float) -> bool: ...
    def maximum_stable_time_step(self) -> float: ...
    def recommended_time_step(self, safety: float = 0.9) -> float: ...
    # Shared stepping vocabulary installed by biotransport.stepping.
    def solve_until(
        self,
        end_time: float,
        time_step: float | None = None,
        *,
        save_times: Sequence[float] | None = None,
        safety_factor: float = 0.8,
    ) -> Result:
        """Advance to ``end_time`` and return a :class:`biotransport.Result`."""
        ...
    def dirichlet(
        self, side: Any, value: float, *, species: int | None = None
    ) -> Self: ...
    def neumann(
        self, side: Any, normal_derivative: float, *, species: int | None = None
    ) -> Self: ...
    def robin(
        self, side: Any, a: float, b: float, rhs: float, *, species: int | None = None
    ) -> Self: ...
    def boundary(
        self, side: Any, condition: BoundaryCondition, *, species: int | None = None
    ) -> Self: ...
    def outward_flux(
        self, side: Any, molar_flux: float, *, species: int | None = None
    ) -> Self: ...
    def solve(self, dt: float, num_steps: int) -> None: ...
    def concentration(self, species: int) -> FloatArray:
        """Return an owned concentration copy."""
        ...
    def potential(self) -> FloatArray:
        """Return an owned potential copy."""
        ...
    def charge_density(self) -> FloatArray: ...
    def time(self) -> float: ...
    def num_species(self) -> int: ...
    def ion(self, index: int) -> IonSpecies:
        """Return a solver-owned species object; the solver keeps it alive."""
        ...
    def electrical_mobility(self, species: int) -> float: ...
    def mesh(self) -> StructuredMesh:
        """Return the internally retained mesh object."""
        ...

class constants:
    FARADAY: float
    GAS_CONSTANT: float
    BOLTZMANN: float
    ELEMENTARY_CHARGE: float
    VACUUM_PERMITTIVITY: float

class ions:
    """Representative aqueous infinite-dilution ion parameters."""

    @staticmethod
    def sodium() -> IonSpecies: ...
    @staticmethod
    def potassium() -> IonSpecies: ...
    @staticmethod
    def chloride() -> IonSpecies: ...
    @staticmethod
    def calcium() -> IonSpecies: ...
    @staticmethod
    def magnesium() -> IonSpecies: ...
    @staticmethod
    def hydrogen() -> IonSpecies: ...
    @staticmethod
    def hydroxide() -> IonSpecies: ...
    @staticmethod
    def bicarbonate() -> IonSpecies: ...

class ghk:
    @staticmethod
    def nernst_potential(
        z: int, c_in: float, c_out: float, temperature: float = 310.0
    ) -> float: ...
    @staticmethod
    def ghk_voltage(
        P_K: float,
        K_in: float,
        K_out: float,
        P_Na: float,
        Na_in: float,
        Na_out: float,
        P_Cl: float,
        Cl_in: float,
        Cl_out: float,
        temperature: float = 310.0,
    ) -> float: ...

# =============================================================================
# Utility Functions
# =============================================================================

def blood_carreau_model(hematocrit: float) -> CarreauModel:
    """Create an educational hematocrit-scaled Carreau surrogate on [0, 0.60].

    The model is anchored to a commonly reported 45% hematocrit fit; values
    away from that reference are a stated surrogate, not patient-specific data.
    """
    ...

def blood_casson_model(hematocrit: float) -> CassonModel:
    """Create the supported Casson blood correlation on [0, 0.60].

    Values outside the source study's 35%-55% range are extrapolations.
    """
    ...

def pipe_wall_shear_rate(Q: float, R: float) -> float:
    """Return ``4*abs(Q)/(pi*R**3)``, the Newtonian nominal magnitude.

    This is not a non-Newtonian wall correction.
    """
    ...

def apparent_viscosity_pipe(
    model: ViscosityModel,
    Q: float,
    R: float,
    pressure_gradient: float,
) -> float:
    """Infer apparent pipe viscosity with a Rabinowitsch-Mooney correction.

    ``Q`` and ``pressure_gradient`` must describe opposing flow and
    pressure-drop directions.
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
        ...     x=0.001, t=100.0, diffusivity=1e-9,
        ...     C_surface=1.0, C_initial=0.0
        ... )
        >>> # Check Poiseuille flow centerline velocity
        >>> u_max = analytical.poiseuille_max_velocity(
        ...     radius=0.002, dp_dz=-1000.0, viscosity=0.003
        ... )
    """

    @staticmethod
    def diffusion_1d_semi_infinite(
        x: float,
        t: float,
        diffusivity: float,
        C_surface: float,
        C_initial: float,
    ) -> float:
        """Semi-infinite constant-surface solution using ``erfc``."""
        ...

    @staticmethod
    def diffusion_penetration_depth(diffusivity: float, t: float) -> float:
        """Legacy name for the characteristic length sqrt(D*t)."""
        ...

    @staticmethod
    def diffusion_length(diffusivity: float, t: float) -> float:
        """Characteristic diffusion length sqrt(D*t)."""
        ...

    @staticmethod
    def first_order_decay(C_0: float, k: float, t: float) -> float:
        """First-order decay: C = C0 * exp(-k*t)."""
        ...

    @staticmethod
    def logistic_growth(
        C_0: float, carrying_capacity: float, growth_rate: float, t: float
    ) -> float:
        """Logistic growth from ``C_0`` toward ``carrying_capacity``."""
        ...

    @staticmethod
    def lumped_exponential(C_0: float, C_inf: float, t: float, tau: float) -> float:
        """``C_inf + (C_0-C_inf)*exp(-t/tau)``."""
        ...

    @staticmethod
    def poiseuille_velocity(
        r: float, radius: float, dp_dz: float, viscosity: float
    ) -> float:
        """Circular-pipe Poiseuille axial velocity."""
        ...

    @staticmethod
    def poiseuille_max_velocity(radius: float, dp_dz: float, viscosity: float) -> float:
        """Maximum velocity in Poiseuille flow."""
        ...

    @staticmethod
    def poiseuille_flow_rate(radius: float, dp_dz: float, viscosity: float) -> float:
        """Volumetric flow rate in Poiseuille flow: Q = pi*R⁴*(-dp/dx)/(8*mu)."""
        ...

    @staticmethod
    def poiseuille_wall_shear(radius: float, dp_dz: float) -> float:
        """Wall shear stress in Poiseuille flow: tau_w = R*(-dp/dx)/2."""
        ...

    @staticmethod
    def plane_poiseuille_velocity(
        y: float, half_height: float, dp_dx: float, viscosity: float
    ) -> float:
        """Pressure-driven velocity between two stationary parallel plates."""
        ...

    @staticmethod
    def plane_poiseuille_max_velocity(
        half_height: float, dp_dx: float, viscosity: float
    ) -> float:
        """Centerline plane-Poiseuille velocity."""
        ...

    @staticmethod
    def couette_velocity(
        y: float, gap_height: float, moving_wall_velocity: float
    ) -> float:
        """Couette velocity profile: u(y) = U*y/H."""
        ...

    @staticmethod
    def couette_max_velocity(moving_wall_velocity: float) -> float:
        """Maximum velocity in Couette flow."""
        ...

    @staticmethod
    def taylor_couette_velocity(
        r: float, a: float, b: float, omega_a: float, omega_b: float
    ) -> float:
        """Taylor-Couette azimuthal velocity."""
        ...

    @staticmethod
    def taylor_couette_torque(
        a: float,
        b: float,
        omega_a: float,
        omega_b: float,
        viscosity: float,
    ) -> float:
        """Torque per unit axial length on the inner cylinder."""
        ...

    @staticmethod
    def bernoulli_velocity(
        v1: float,
        p1: float,
        z1: float,
        p2: float,
        z2: float,
        density: float,
        g: float = 9.81,
    ) -> float:
        """Solve the steady inviscid Bernoulli relation for ``v2``."""
        ...

    @staticmethod
    def maxwell_relaxation(E: float, eta: float, epsilon_0: float, t: float) -> float:
        """Maxwell stress relaxation with ``tau=eta/E``."""
        ...

    @staticmethod
    def maxwell_relaxation_time(E: float, eta: float) -> float:
        """Maxwell relaxation time ``eta/E``."""
        ...

    @staticmethod
    def kelvin_voigt_creep(E: float, eta: float, sigma_0: float, t: float) -> float:
        """Kelvin-Voigt creep with ``tau=eta/E``."""
        ...

    @staticmethod
    def sls_relaxation(
        E1: float, E2: float, eta: float, epsilon_0: float, t: float
    ) -> float:
        """Standard linear solid stress relaxation."""
        ...

    @staticmethod
    def sls_creep(E1: float, E2: float, eta: float, sigma_0: float, t: float) -> float:
        """Standard linear solid creep."""
        ...

    @staticmethod
    def burgers_creep(
        E1: float,
        mu1: float,
        E2: float,
        mu2: float,
        sigma_0: float,
        t: float,
    ) -> float:
        """Burgers model creep response."""
        ...

    @staticmethod
    def burgers_compliance(
        E1: float, mu1: float, E2: float, mu2: float, t: float
    ) -> float:
        """Burgers model compliance."""
        ...

    @staticmethod
    def complex_modulus_magnitude(G1: float, G2: float) -> float:
        """Complex modulus magnitude: |G*| = sqrt(G'² + G''²)."""
        ...

    @staticmethod
    def phase_angle(G1: float, G2: float) -> float:
        """Phase angle: delta = atan(G''/G')."""
        ...

    @staticmethod
    def loss_tangent(G1: float, G2: float) -> float:
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

    **Mass-transfer resistance:**
    - Biot (Bi): Internal diffusive resistance divided by external resistance
    - Fourier (Fo): Heat diffusion scaling

    These numbers guide solver selection:
    - Pe >> 1: Convection-dominated, may need upwinding
    - Pe << 1: Diffusion-dominated, central differences OK
    - Bi < 0.1: Lumped capacitance valid

    Examples:
        >>> # Compare with a geometry-specific transition criterion (for example,
        >>> # the conventional fully developed circular-pipe heuristic)
        >>> Re = dimensionless.reynolds(
        ...     density=1000, velocity=1.0, length=0.01, viscosity=0.003
        ... )
        >>> print(f"Re = {Re:.0f}")
        >>>
        >>> # Check grid Peclet number for stability
        >>> Pe_grid = dimensionless.peclet(
        ...     velocity=0.01, length=dx, diffusivity=1e-9
        ... )
        >>> if Pe_grid > 2:
        ...     print("Use upwind scheme for stability")
    """

    @staticmethod
    def reynolds(
        density: float, velocity: float, length: float, viscosity: float
    ) -> float:
        """Reynolds number: Re = rho*U*L/mu."""
        ...

    @staticmethod
    def reynolds_kinematic(
        velocity: float, length: float, kinematic_viscosity: float
    ) -> float:
        """Reynolds number using kinematic viscosity: Re = U*L/nu."""
        ...

    @staticmethod
    def peclet(velocity: float, length: float, diffusivity: float) -> float:
        """Peclet number: Pe = U*L/D."""
        ...

    @staticmethod
    def schmidt(viscosity: float, density: float, diffusivity: float) -> float:
        """Schmidt number ``viscosity/(density*diffusivity)``."""
        ...

    @staticmethod
    def schmidt_kinematic(kinematic_viscosity: float, diffusivity: float) -> float:
        """Schmidt number ``kinematic_viscosity/diffusivity``."""
        ...

    @staticmethod
    def sherwood(h_m: float, length: float, diffusivity: float) -> float:
        """Sherwood number: Sh = k_c*L/D."""
        ...

    @staticmethod
    def biot(h_m: float, length: float, diffusivity: float) -> float:
        """Mass-transfer Biot number ``h_m*length/diffusivity``."""
        ...

    @staticmethod
    def fourier(diffusivity: float, time: float, length: float) -> float:
        """Diffusive Fourier number ``diffusivity*time/length**2``."""
        ...

    @staticmethod
    def is_lumped_valid(bi: float, threshold: float = 0.1) -> bool:
        """Whether ``bi < threshold``."""
        ...

    @staticmethod
    def is_convection_dominated(pe: float, threshold: float = 1.0) -> bool:
        """Whether ``pe > threshold``."""
        ...

# =============================================================================
# VTK output
# =============================================================================

def write_vtk(
    mesh: StructuredMesh,
    solution: ArrayLike,
    filename: str,
    field_name: str = "scalar",
) -> None:
    """Write one flat nodal field to a legacy VTK file."""
    ...

def write_vtk_series(
    mesh: StructuredMesh,
    solutions: list[ArrayLike],
    times: list[float],
    prefix: str,
    field_name: str = "scalar",
) -> None:
    """Write numbered VTK snapshots for the supplied times."""
    ...

def write_vtk_series_with_metadata(
    mesh: StructuredMesh,
    solutions: list[ArrayLike],
    times: list[float],
    prefix: str,
    field_name: str = "scalar",
) -> None:
    """Write numbered snapshots plus a ParaView ``.pvd`` time-series file."""
    ...
