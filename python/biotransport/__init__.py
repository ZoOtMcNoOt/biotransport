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
    ExplicitFD,
    RunResult,
    SolverStats,
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
    # I/O and visualization (C++ version - single array API)
    write_vtk_series_with_metadata,
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
)
from . import run as _run_module
from .results import Result, Snapshots
from .run import CheckpointResult, run_checkpoints, solve
from ._deprecation import (
    ROOT_DEPRECATED,
    ROOT_LAZY,
    BioTransportDeprecationWarning,
    deprecated_callable,
    module_getattr,
)
from .visualization import (
    plot_1d_solution,
    plot_2d_solution,
    plot_2d_surface,
    plot_field,
    plot_1d,
    plot_2d,
    plot,
)

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

# Adaptive time-stepping
from .adaptive import (
    AdaptiveResult,
    AdaptiveTimeStepper,
    AdaptiveTimeStepperConfig,
    solve_adaptive,
)

# Higher-order time integration (RK4, Heun)
from .time_integrators import (
    RK4Integrator,
    HeunIntegrator,
    IntegrationResult,
    integrate,
    rk4_step,
    heun_step,
    euler_step,
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

# Pulsatile (time-varying) boundary conditions
from .pulsatile import (
    PulsatileBC,
    ConstantBC,
    SinusoidalBC,
    RampBC,
    StepBC,
    SquareWaveBC,
    CustomBC,
    ArterialPressureBC,
    VenousPressureBC,
    CardiacOutputBC,
    RespiratoryBC,
    DrugInfusionBC,
    CompositeBC,
    PulsatileResult,
    solve_pulsatile,
    heart_rate_to_period,
    period_to_heart_rate,
    sample_waveform,
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

# Newton-Raphson iteration for nonlinear steady-state problems
from .newton_raphson import (
    NewtonSolverError,
    NewtonEvaluationError,
    NewtonLinearSolveError,
    NewtonLineSearchError,
    NewtonRaphsonSolver,
    NonlinearDiffusionSolver,
    NewtonResult,
    ConvergenceCriterion,
    michaelis_menten,
    hill_kinetics,
    bistable,
    exponential_decay,
)

# Discoverable, science-scoped API namespaces.
from . import (
    analysis,
    applications,
    contracts,
    diffusion,
    electrochem,
    flow,
    provenance,
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
    # ========== Discoverable namespaces ==========
    "diffusion",
    "electrochem",
    "flow",
    "applications",
    "analysis",
    "contracts",
    "provenance",
    "reproducibility",
    "units",
    # ========== Scientific workflow helpers ==========
    "Dimension",
    "Quantity",
    "Unit",
    "quantity",
    "convert",
    "ParameterRange",
    "parameter_sweep",
    "local_sensitivity",
    "latin_hypercube",
    "propagate_uncertainty",
    "standardized_regression_coefficients",
    "ParameterProvenance",
    "ParameterSetProvenance",
    "illustrative_parameter_set",
    "SolverContract",
    "PythonBackend",
    "PythonNumericalContract",
    "get_contract",
    "get_python_numerical_contract",
    "list_contracts",
    "list_python_numerical_contracts",
    "list_python_numerical_symbols",
    "filter_contracts",
    "registry_as_dict",
    "python_registry_as_dict",
    "freeze_config",
    "method_metadata",
    "convergence_table",
    "balance_residual",
    "create_manifest",
    "write_manifest",
    "load_manifest",
    # ========== Scientific balance accounting ==========
    "BalanceDimension",
    "BalanceUnit",
    "BalanceTransferDirection",
    "BalanceTerm",
    "BalanceTransfer",
    "BalanceAudit",
    "BalanceLedger",
    "MatchedBalanceTransfer",
    "DimensionBalanceAudit",
    "BalanceReconciliation",
    "balance_dimension_name",
    "balance_unit_symbol",
    "balance_unit_dimension",
    "balance_base_unit",
    "convert_balance_value",
    "reconcile_balances",
    "native_build_info",
    "NonuniformMesh1D",
    "NonuniformDiffusionDiagnostics",
    "NonuniformDiffusion1D",
    # ========== Core (most commonly used) ==========
    "Problem",  # The main problem builder (alias for TransportProblem)
    "solve",  # Simplest way to run a simulation
    "Result",  # What every solve returns
    "Snapshots",  # Fields recorded at save_times
    "plot",  # Simplest way to visualize results
    "mesh_1d",  # Create 1D mesh
    "mesh_2d",  # Create 2D mesh
    "x_nodes",  # Get x coordinates from mesh
    "y_nodes",  # Get y coordinates from mesh
    "xy_grid",  # Get 2D meshgrid
    # ========== Initial condition helpers ==========
    "gaussian",
    "step",
    "uniform",
    "circle",
    "sinusoidal",
    # ========== Slightly more advanced ==========
    "StructuredMesh",
    "StructuredMesh3D",
    "Boundary3D",
    "DiffusionSolver3D",
    "LinearReactionDiffusionSolver3D",
    "TransportProblem",
    "ExplicitFD",
    "Boundary",
    "BoundaryCondition",
    "RunResult",
    "SolverStats",
    "SolveOptions",
    "SolveDiagnostics",
    "TransportResult",
    "solve_transport",
    "run_checkpoints",
    "CheckpointResult",
    "BioTransportDeprecationWarning",
    # ========== Adaptive time-stepping ==========
    "AdaptiveTimeStepper",
    "AdaptiveTimeStepperConfig",
    "AdaptiveResult",
    "solve_adaptive",
    # ========== Higher-order time integration ==========
    "RK4Integrator",
    "HeunIntegrator",
    "IntegrationResult",
    "integrate",
    "rk4_step",
    "heun_step",
    "euler_step",
    # ========== Grid convergence (verification) ==========
    "GridConvergenceStudy",
    "ConvergenceSolveResult",
    "ConvergenceResult",
    "compute_order_of_accuracy",
    "run_convergence_study",
    "temporal_convergence_study",
    "plot_convergence",
    # ========== Pulsatile boundary conditions ==========
    "PulsatileBC",
    "ConstantBC",
    "SinusoidalBC",
    "RampBC",
    "StepBC",
    "SquareWaveBC",
    "CustomBC",
    "ArterialPressureBC",
    "VenousPressureBC",
    "CardiacOutputBC",
    "RespiratoryBC",
    "DrugInfusionBC",
    "CompositeBC",
    "PulsatileResult",
    "solve_pulsatile",
    "heart_rate_to_period",
    "period_to_heart_rate",
    "sample_waveform",
    # ========== Higher-order finite difference schemes ==========
    "laplacian_2nd_order",
    "laplacian_4th_order",
    "laplacian_6th_order",
    "gradient_4th_order",
    "d2dx2",
    "ddx",
    "HighOrderDiffusionSolver",
    "HighOrderResult",
    "RungeKuttaResult",
    "integrate_explicit_runge_kutta",
    "verify_order_of_accuracy",
    # ========== Plotting variants ==========
    "plot_field",
    "plot_1d",
    "plot_2d",
    "plot_1d_solution",
    "plot_2d_solution",
    "plot_2d_surface",
    # ========== Mesh utilities ==========
    "r_nodes",
    "z_nodes",
    "rz_grid",
    "as_1d",
    "as_2d",
    # ========== Field builders ==========
    "SpatialField",
    "layered_1d",
    # ========== Specialized solvers ==========
    "DiffusionSolver",
    "CrankNicolsonDiffusion",
    "CNSolveResult",
    # ADI solvers
    "ADIDiffusion2D",
    "ADIDiffusion3D",
    "ADISolveResult",
    # Sparse matrix and implicit solvers
    "SparseSolverType",
    "SparseSolveResult",
    "Triplet",
    "SparseMatrix",
    "build_2d_laplacian",
    "build_implicit_diffusion_2d",
    "build_implicit_diffusion_3d",
    "ImplicitSolveResult",
    "ImplicitDiffusion2D",
    "ImplicitDiffusion3D",
    "sparse_matrix_available",
    "ConstantSourceReactionDiffusionSolver",
    "LinearReactionDiffusionSolver",
    "LogisticReactionDiffusionSolver",
    "MichaelisMentenReactionDiffusionSolver",
    "ReactionDiffusionSolver",
    "MaskedMichaelisMentenReactionDiffusionSolver",
    "BoundaryType",
    "AdvectionScheme",
    "AdvectionDiffusionSolver",
    "DarcyFlowResult",
    "DarcyFlowSolver",
    "MembraneDiffusionResult",
    "MembraneDiffusion1DSolver",
    "MultiLayerMembraneSolver",
    "renkin_hindrance",
    "GrayScottSolver",
    "GrayScottRunResult",
    "TumorDrugDeliverySolver",
    "TumorDrugDeliverySaved",
    "BioheatCryotherapySolver",
    "BioheatSaved",
    # ========== Fluid dynamics ==========
    "VelocityBCType",
    "VelocityBC",
    "StokesResult",
    "StokesSolver",
    "ConvectionScheme",
    "NavierStokesResult",
    "NavierStokesSolver",
    # ========== Cylindrical mesh ==========
    "CylindricalMeshType",
    "CylindricalMesh",
    # ========== Non-Newtonian fluid models ==========
    "FluidModel",
    "ViscosityModel",
    "NewtonianModel",
    "PowerLawModel",
    "CarreauModel",
    "CarreauYasudaModel",
    "CrossModel",
    "BinghamModel",
    "HerschelBulkleyModel",
    "CassonModel",
    "blood_casson_model",
    "blood_carreau_model",
    "pipe_wall_shear_rate",
    "apparent_viscosity_pipe",
    # ========== Utilities ==========
    "get_results_dir",
    "get_result_path",
    "write_vtk",
    "write_vtk_series",
    "write_vtk_series_with_metadata",
    "dimensionless",
    "analytical",
    # ========== Configuration ==========
    "TumorDrugDeliveryConfig",
    "BioheatCryotherapyConfig",
    "get_parameter_ranges",
    # ========== Multi-species reaction-diffusion ==========
    "MultiSpeciesSolver",
    "LotkaVolterraReaction",
    "SIRReaction",
    "SEIRReaction",
    "BrusselatorReaction",
    "CompetitiveInhibitionReaction",
    "EnzymeCascadeReaction",
    # ========== Nernst-Planck electrochemical transport ==========
    "IonSpecies",
    "NernstPlanckSolver",
    "MultiIonSolver",
    "constants",
    "ions",
    "ghk",
    # ========== Newton-Raphson nonlinear solvers ==========
    "NewtonSolverError",
    "NewtonEvaluationError",
    "NewtonLinearSolveError",
    "NewtonLineSearchError",
    "NewtonRaphsonSolver",
    "NonlinearDiffusionSolver",
    "NewtonResult",
    "ConvergenceCriterion",
    "michaelis_menten",
    "hill_kinetics",
    "bistable",
    "exponential_decay",
    "__version__",
]
