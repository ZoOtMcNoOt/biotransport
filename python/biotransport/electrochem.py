"""Native electrodiffusion models and electrochemical utilities.

The Nernst--Planck solvers transport ions in a prescribed electric-potential
field.  They do not solve Poisson's equation, enforce electroneutrality, or
model membrane capacitance implicitly.  The ``ghk`` object exposes the
library's Goldman--Hodgkin--Katz utilities under their documented assumptions.

This module only re-exports compiled public objects.
"""

from ._core import (
    Boundary,
    IonSpecies,
    MultiIonSolver,
    NernstPlanckSolver,
    StructuredMesh,
    constants,
    ghk,
    ions,
)

__all__ = [
    "StructuredMesh",
    "Boundary",
    "IonSpecies",
    "NernstPlanckSolver",
    "MultiIonSolver",
    "constants",
    "ions",
    "ghk",
]
