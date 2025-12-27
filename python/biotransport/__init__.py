"""
BioTransport - A library for modeling biotransport phenomena
"""

from ._core import (
    StructuredMesh,
    StructuredMesh3D,
    Boundary3D,
    DiffusionSolver,
    DiffusionSolver3D,
    LinearReactionDiffusionSolver3D,
    CrankNicolsonDiffusion,
    CNSolveResult,
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
    # I/O and visualization
    write_vtk,
    write_vtk_series,
    write_vtk_series_with_metadata,
)

# Expose utility functions
from .utils import get_results_dir, get_result_path

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
from .run import run, run_checkpoints, solve
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

# ============================================================================
# User-friendly aliases
# ============================================================================

# "Problem" is the simplest, most intuitive name
Problem = TransportProblem

# Backward-compatible aliases for legacy code
DiffusionProblem = TransportProblem
LinearReactionDiffusionProblem = TransportProblem
AdvectionDiffusionProblem = TransportProblem

__version__ = "0.1.0"

__all__ = [
    # ========== Core (most commonly used) ==========
    "Problem",  # The main problem builder (alias for TransportProblem)
    "solve",  # Simplest way to run a simulation
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
    "run",
    "run_checkpoints",
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
    # ========== Legacy aliases ==========
    "DiffusionProblem",
    "LinearReactionDiffusionProblem",
    "AdvectionDiffusionProblem",
    "__version__",
]
