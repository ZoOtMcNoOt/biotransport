"""
BioTransport - A library for modeling biotransport phenomena
"""

from ._core import (
    BalanceDimension,
    BalanceUnit,
    BalanceTransferDirection,
    BalanceTerm,
    BalanceTransfer,
    BalanceAudit,
    BalanceLedger,
    MatchedBalanceTransfer,
    DimensionBalanceAudit,
    BalanceReconciliation,
    balance_dimension_name,
    balance_unit_symbol,
    balance_unit_dimension,
    balance_base_unit,
    convert_balance_value,
    reconcile_balances,
    native_build_info,
    NonuniformMesh1D,
    NonuniformDiffusionDiagnostics,
    NonuniformDiffusion1D,
    StructuredMesh,
    StructuredMesh3D,
    Boundary3D,
    DiffusionSolver,
    DiffusionSolver3D,
    LinearReactionDiffusionSolver3D,
    CrankNicolsonDiffusion,
    CNSolveResult,
    # ADI solvers (Alternating Direction Implicit)
    ADIDiffusion2D,
    ADIDiffusion3D,
    ADISolveResult,
    # Sparse matrix and implicit solvers
    SparseSolverType,
    SparseSolveResult,
    Triplet,
    SparseMatrix,
    build_2d_laplacian,
    build_implicit_diffusion_2d,
    build_implicit_diffusion_3d,
    ImplicitSolveResult,
    ImplicitDiffusion2D,
    ImplicitDiffusion3D,
    sparse_matrix_available,
    ConstantSourceReactionDiffusionSolver,
    LinearReactionDiffusionSolver,
    LogisticReactionDiffusionSolver,
    MichaelisMentenReactionDiffusionSolver,
    ReactionDiffusionSolver,
    MaskedMichaelisMentenReactionDiffusionSolver,
    BoundaryType,
    Boundary,
    BoundaryCondition,
    GrayScottSolver,
    GrayScottRunResult,
    TumorDrugDeliverySolver,
    TumorDrugDeliverySaved,
    BioheatCryotherapySolver,
    BioheatSaved,
    TransportProblem,
    SolveOptions,
    SolveDiagnostics,
    TransportResult,
    solve_transport,
    # Advection-diffusion (Phase 2)
    AdvectionScheme,
    AdvectionDiffusionSolver,
    # Darcy flow (Phase 3)
    DarcyFlowResult,
    DarcyFlowSolver,
    # Membrane diffusion (Phase 4)
    MembraneDiffusionResult,
    MembraneDiffusion1DSolver,
    MultiLayerMembraneSolver,
    renkin_hindrance,
    # BMEN 341 utilities
    dimensionless,
    analytical,
    # Fluid dynamics (Stokes & Navier-Stokes)
    VelocityBCType,
    VelocityBC,
    StokesResult,
    StokesSolver,
    ConvectionScheme,
    NavierStokesResult,
    NavierStokesSolver,
    # Cylindrical mesh
    CylindricalMeshType,
    CylindricalMesh,
    # Non-Newtonian fluid models
    FluidModel,
    ViscosityModel,
    NewtonianModel,
    PowerLawModel,
    CarreauModel,
    CarreauYasudaModel,
    CrossModel,
    BinghamModel,
    HerschelBulkleyModel,
    CassonModel,
    # Blood rheology utilities
    blood_casson_model,
    blood_carreau_model,
    pipe_wall_shear_rate,
    apparent_viscosity_pipe,
    # Multi-species reaction-diffusion
    MultiSpeciesSolver,
    LotkaVolterraReaction,
    SIRReaction,
    SEIRReaction,
    BrusselatorReaction,
    CompetitiveInhibitionReaction,
    EnzymeCascadeReaction,
    # Nernst-Planck electrochemical transport
    IonSpecies,
    NernstPlanckSolver,
    MultiIonSolver,
    # Nernst-Planck submodules
    constants,
    ions,
    ghk,
)

# Expose utility functions
from .utils import get_results_dir, get_result_path

# VTK export (Python wrapper with dict-based API)
from .vtk_export import write_vtk, write_vtk_series

# Beginner-friendly convenience helpers
from .mesh_utils import (
    as_1d,
    as_2d,
    x_nodes,
    y_nodes,
    xy_grid,
    r_nodes,
    z_nodes,
    rz_grid,
    mesh_1d,
    mesh_2d,
    mesh_3d,
    sides,
)
from . import run as _run_module
from .results import Result, Snapshots
from .run import solve
from . import stepping
from .stepping import StepDiagnostics, solve_until
from ._deprecation import (
    ROOT_DEPRECATED,
    ROOT_LAZY,
    BioTransportDeprecationWarning,
    deprecated_callable,
    module_getattr,
)
from .visualization import plot

# Spatial field builders
from .fields import SpatialField, layered_1d

# Initial condition helpers
from .initial_conditions import gaussian, step, uniform, circle, sinusoidal

# Configuration dataclasses for multi-physics solvers
from .config import (
    TumorDrugDeliveryConfig,
    BioheatCryotherapyConfig,
    get_parameter_ranges,
)

# Grid convergence studies (verification)
from .convergence import (
    ConvergenceSolveResult,
    GridConvergenceStudy,
    ConvergenceResult,
    compute_order_of_accuracy,
    run_convergence_study,
    temporal_convergence_study,
    plot_convergence,
)

# Higher-order finite difference schemes
from .high_order import (
    laplacian_2nd_order,
    laplacian_4th_order,
    laplacian_6th_order,
    gradient_4th_order,
    d2dx2,
    ddx,
    HighOrderDiffusionSolver,
    HighOrderResult,
    RungeKuttaResult,
    integrate_explicit_runge_kutta,
    verify_order_of_accuracy,
)

# Discoverable, science-scoped API namespaces.
from . import (
    analysis,
    applications,
    balance,
    contracts,
    convergence,
    diffusion,
    electrochem,
    flow,
    high_order,
    provenance,
    reference,
    reproducibility,
    units,
)
from .analysis import (
    ParameterRange,
    latin_hypercube,
    local_sensitivity,
    parameter_sweep,
    propagate_uncertainty,
    standardized_regression_coefficients,
)
from .contracts import (
    PythonBackend,
    PythonNumericalContract,
    SolverContract,
    filter_contracts,
    get_contract,
    get_python_numerical_contract,
    list_contracts,
    list_python_numerical_contracts,
    list_python_numerical_symbols,
    python_registry_as_dict,
    registry_as_dict,
)
from .provenance import (
    ParameterProvenance,
    ParameterSetProvenance,
    illustrative_parameter_set,
)
from .reproducibility import (
    balance_residual,
    convergence_table,
    create_manifest,
    freeze_config,
    load_manifest,
    method_metadata,
    write_manifest,
)
from .units import Dimension, Quantity, Unit, convert, quantity

# ============================================================================
# User-friendly aliases
# ============================================================================

# Retired root-level spellings (see docs/notes/DEPRECATION_POLICY.md) resolve
# lazily through PEP 562 and warn on every access.
__getattr__ = module_getattr(__name__, ROOT_DEPRECATED, ROOT_LAZY)

# ``run`` shares its name with the submodule that defines ``solve``, so it is
# kept as an eager wrapper that warns on every call and forwards unchanged.
run = deprecated_callable(
    "bt.solve(problem, end_time=...)",
    reason=(
        "run() was a compatibility alias; solve() is the single canonical entry "
        "point and executes the same C++ solver"
    ),
    name="biotransport.run",
)(_run_module.run)

# "Problem" is the simplest, most intuitive name
Problem = TransportProblem

# The legacy ``DiffusionProblem`` / ``LinearReactionDiffusionProblem`` /
# ``AdvectionDiffusionProblem`` aliases are deprecated and resolve through
# ``__getattr__`` (see ``biotransport._deprecation.ROOT_DEPRECATED``).

__version__ = "0.1.0"

__all__ = [
    # ========== Tier 0: the canonical path ==========
    "Problem",  # The main problem builder (alias for TransportProblem)
    "TransportProblem",
    "solve",  # Simplest way to run a simulation
    "Result",  # What every solve returns
    "Snapshots",  # Fields recorded at save_times
    "solve_until",  # Same verb on every native stepping solver
    "StepDiagnostics",
    "SolveOptions",
    "SolveDiagnostics",
    "TransportResult",
    "solve_transport",
    # boundaries
    "Boundary",
    "Boundary3D",
    "BoundaryCondition",
    "BoundaryType",
    "sides",
    # meshes
    "StructuredMesh",
    "StructuredMesh3D",
    "CylindricalMesh",
    "CylindricalMeshType",
    "NonuniformMesh1D",
    "mesh_1d",
    "mesh_2d",
    "mesh_3d",
    "x_nodes",
    "y_nodes",
    "xy_grid",
    "r_nodes",
    "z_nodes",
    "rz_grid",
    "as_1d",
    "as_2d",
    # fields and initial conditions
    "gaussian",
    "step",
    "uniform",
    "circle",
    "sinusoidal",
    "SpatialField",
    "layered_1d",
    # output
    "plot",
    "write_vtk",
    "write_vtk_series",
    "get_result_path",
    "get_results_dir",
    # package
    "native_build_info",
    "BioTransportDeprecationWarning",
    "__version__",
    # ========== Tier 1: discoverable namespaces ==========
    "diffusion",
    "electrochem",
    "flow",
    "applications",
    "balance",
    "reference",
    "stepping",
    "analysis",
    "convergence",
    "contracts",
    "high_order",
    "provenance",
    "reproducibility",
    "units",
]

# Every specialized native solver, fluid model, balance object and workflow
# helper remains an attribute of this module (``bt.DiffusionSolver`` still
# works and tab-completes); the namespaces above are the documented way to
# find them.  Retired spellings resolve through ``__getattr__`` and warn.
