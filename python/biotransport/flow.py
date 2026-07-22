"""Native porous-flow, incompressible-flow, and rheology APIs.

Darcy, Stokes, and Navier--Stokes are separate models and are not coupled to
scalar transport merely by importing them together.  The rheology classes
provide constitutive viscosity laws; users remain responsible for selecting
parameters and a solver whose assumptions fit the intended flow regime.

This module only re-exports existing compiled objects.
"""

from ._core import (
    Boundary,
    BinghamModel,
    CarreauModel,
    CarreauYasudaModel,
    CassonModel,
    ConvectionScheme,
    CrossModel,
    CylindricalMesh,
    CylindricalMeshType,
    DarcyFlowResult,
    DarcyFlowSolver,
    FluidModel,
    HerschelBulkleyModel,
    NavierStokesResult,
    NavierStokesSolver,
    NewtonianModel,
    PowerLawModel,
    StokesResult,
    StokesSolver,
    StructuredMesh,
    VelocityBC,
    VelocityBCType,
    ViscosityModel,
    apparent_viscosity_pipe,
    blood_carreau_model,
    blood_casson_model,
    pipe_wall_shear_rate,
)

__all__ = [
    # Mesh and boundary data
    "StructuredMesh",
    "CylindricalMesh",
    "CylindricalMeshType",
    "Boundary",
    "VelocityBC",
    "VelocityBCType",
    # Flow solvers
    "DarcyFlowSolver",
    "DarcyFlowResult",
    "StokesSolver",
    "StokesResult",
    "NavierStokesSolver",
    "NavierStokesResult",
    "ConvectionScheme",
    # Constitutive viscosity models
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
]
