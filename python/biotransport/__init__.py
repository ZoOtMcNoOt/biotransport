"""
BioTransport - A library for modeling biotransport phenomena
"""

from ._core import (
    StructuredMesh,
    DiffusionSolver,
    ReactionDiffusionSolver,
    BoundaryType
)

# Expose utility functions
from .utils import get_results_dir, get_result_path

__version__ = '0.1.0'