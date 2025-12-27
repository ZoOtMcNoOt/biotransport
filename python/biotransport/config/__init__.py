"""Configuration dataclasses for multi-physics solvers.

This package provides well-documented configuration objects for complex
multi-physics simulations like tumor drug delivery and bioheat cryotherapy.
Each parameter includes units, typical ranges, and physical meaning.

Usage:
    from biotransport.config import TumorDrugDeliveryConfig, BioheatCryotherapyConfig

    config = TumorDrugDeliveryConfig(
        domain_size=5e-3,
        tumor_radius=2e-3,
        D_drug_normal=5e-11,
    )
    # Use config.to_solver_kwargs() to pass to solver
"""

from .tumor_drug_delivery import TumorDrugDeliveryConfig
from .bioheat_cryotherapy import BioheatCryotherapyConfig
from .parameter_ranges import get_parameter_ranges

__all__ = [
    "TumorDrugDeliveryConfig",
    "BioheatCryotherapyConfig",
    "get_parameter_ranges",
]
