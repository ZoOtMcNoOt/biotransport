"""Configuration for tumor drug delivery simulations.

This module provides the TumorDrugDeliveryConfig dataclass for coupled
tumor interstitial pressure / drug transport simulations.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TumorDrugDeliveryConfig:
    """Configuration for tumor drug delivery simulations.

    This dataclass encapsulates all parameters needed for a coupled
    tumor interstitial pressure / drug transport simulation.

    The physics includes:
    - Darcy flow driven by interstitial fluid pressure (IFP) gradients
    - Drug diffusion with spatially varying diffusivity
    - Vascular drug source (transvascular exchange)
    - Drug binding and cellular uptake kinetics

    All units are SI unless otherwise noted.

    Attributes
    ----------
    domain_size : float
        Side length of the square simulation domain [m].
        Typical: 1e-3 to 10e-3 (1-10 mm tissue region).

    tumor_radius : float
        Radius of the spherical tumor region [m].
        Typical: 0.5e-3 to 5e-3 (0.5-5 mm).

    tumor_center : tuple of float
        (x, y) coordinates of tumor center [m].
        Default: centered in domain.

    D_drug_normal : float
        Drug diffusion coefficient in normal tissue [m²/s].
        Typical range: 1e-12 to 1e-10.
        Example: small molecules ~1e-10, antibodies ~1e-11.

    D_drug_tumor : float
        Drug diffusion coefficient in tumor tissue [m²/s].
        Often lower than normal due to dense extracellular matrix (ECM).
        Typical: 0.2-0.5 × D_drug_normal.

    k_binding : float
        Rate constant for drug binding to tissue components [1/s].
        Represents reversible binding to ECM, receptors, etc.
        Typical: 1e-4 to 1e-2.

    k_uptake : float
        Rate constant for irreversible cellular uptake [1/s].
        Represents internalization and metabolism.
        Typical: 1e-5 to 1e-3.

    MVD_normal : float
        Microvascular density in normal tissue [vessels/mm²].
        Typical: 50-200 vessels/mm².

    MVD_tumor_core : float
        Microvascular density in tumor core [vessels/mm²].
        Often lower due to hypoxia and poor vascularization.
        Typical: 10-50 vessels/mm².

    MVD_tumor_rim : float
        Microvascular density at tumor periphery [vessels/mm²].
        Often elevated due to active angiogenesis.
        Typical: 100-400 vessels/mm².

    P_vessel_normal : float
        Transvascular permeability in normal vessels [m/s].
        Typical: 1e-8 to 1e-6.

    P_vessel_tumor : float
        Transvascular permeability in tumor vessels [m/s].
        Enhanced due to "leaky" tumor vasculature.
        Typical: 5-20 × P_vessel_normal.

    C_plasma : float
        Normalized drug concentration in plasma [-].
        Usually set to 1.0 for normalization.

    IFP_normal : float
        Interstitial fluid pressure in normal tissue [mmHg].
        Typical: 0-3 mmHg.

    IFP_tumor : float
        Interstitial fluid pressure in tumor core [mmHg].
        Elevated due to lymphatic dysfunction.
        Typical: 10-40 mmHg.

    K_hydraulic_normal : float
        Hydraulic conductivity in normal tissue [m²/(Pa·s)].
        Governs Darcy flow velocity.
        Typical: 1e-13 to 1e-11.

    K_hydraulic_tumor : float
        Hydraulic conductivity in tumor [m²/(Pa·s)].
        Often reduced due to dense ECM.
        Typical: 0.3-0.7 × K_normal.

    rim_thickness : float
        Thickness of the tumor rim/periphery region [m].
        Transition zone between core and normal tissue.
        Typical: 0.2e-3 to 1e-3.

    nx : int
        Number of grid cells in x-direction.

    ny : int
        Number of grid cells in y-direction.

    Notes
    -----
    The tumor region is divided into:
    - Core: central hypoxic region with low MVD
    - Rim: peripheral angiogenic region with high MVD
    - Normal: surrounding healthy tissue

    References
    ----------
    .. [1] Jain, R.K. "Normalization of Tumor Vasculature." Science 307.5706 (2005).
    .. [2] Baxter & Jain. "Transport of fluid and macromolecules in tumors."
           Microvasc. Res. 37.1 (1989): 77-104.

    Examples
    --------
    >>> config = TumorDrugDeliveryConfig(
    ...     domain_size=5e-3,
    ...     tumor_radius=2e-3,
    ...     D_drug_normal=5e-11,
    ... )
    >>> print(f"Tumor occupies {100 * config.tumor_area_fraction:.1f}% of domain")
    """

    # Domain geometry
    domain_size: float = 5e-3  # [m]
    tumor_radius: float = 2e-3  # [m]
    tumor_center: Optional[Tuple[float, float]] = None  # [m], defaults to domain center
    rim_thickness: float = 0.5e-3  # [m]

    # Grid resolution
    nx: int = 100
    ny: int = 100

    # Drug diffusion coefficients [m²/s]
    D_drug_normal: float = 5e-11
    D_drug_tumor: float = 2e-11

    # Reaction kinetics [1/s]
    k_binding: float = 1e-3
    k_uptake: float = 5e-4

    # Vascular parameters
    MVD_normal: float = 100.0  # [vessels/mm²]
    MVD_tumor_core: float = 20.0
    MVD_tumor_rim: float = 200.0

    P_vessel_normal: float = 1e-7  # [m/s]
    P_vessel_tumor: float = 5e-7
    C_plasma: float = 1.0  # normalized

    # Interstitial pressure [mmHg]
    IFP_normal: float = 0.0
    IFP_tumor: float = 20.0

    # Hydraulic conductivity [m²/(Pa·s)]
    K_hydraulic_normal: float = 5e-12
    K_hydraulic_tumor: float = 2.5e-12

    def __post_init__(self) -> None:
        """Set default tumor center if not specified."""
        if self.tumor_center is None:
            self.tumor_center = (self.domain_size / 2, self.domain_size / 2)

    @property
    def IFP_normal_Pa(self) -> float:
        """Normal tissue IFP in Pascals."""
        return self.IFP_normal * 133.322

    @property
    def IFP_tumor_Pa(self) -> float:
        """Tumor IFP in Pascals."""
        return self.IFP_tumor * 133.322

    @property
    def tumor_area_fraction(self) -> float:
        """Fraction of domain area occupied by tumor."""
        tumor_area = np.pi * self.tumor_radius**2
        domain_area = self.domain_size**2
        return tumor_area / domain_area

    @property
    def grid_spacing(self) -> Tuple[float, float]:
        """(dx, dy) grid spacing in meters."""
        dx = self.domain_size / self.nx
        dy = self.domain_size / self.ny
        return (dx, dy)

    def describe(self) -> str:
        """Return a formatted description of the configuration."""
        lines = [
            "=== Tumor Drug Delivery Configuration ===",
            "",
            "Domain:",
            f"  Size: {self.domain_size * 1e3:.2f} mm × {self.domain_size * 1e3:.2f} mm",
            f"  Grid: {self.nx} × {self.ny} ({(self.nx + 1) * (self.ny + 1)} nodes)",
            f"  Spacing: {self.grid_spacing[0] * 1e6:.1f} µm",
            "",
            "Tumor:",
            f"  Radius: {self.tumor_radius * 1e3:.2f} mm",
            f"  Center: ({self.tumor_center[0] * 1e3:.2f}, {self.tumor_center[1] * 1e3:.2f}) mm",
            f"  Rim thickness: {self.rim_thickness * 1e3:.2f} mm",
            f"  Area fraction: {self.tumor_area_fraction * 100:.1f}%",
            "",
            "Drug Transport:",
            f"  D_normal: {self.D_drug_normal:.2e} m²/s",
            f"  D_tumor: {self.D_drug_tumor:.2e} m²/s",
            f"  k_binding: {self.k_binding:.2e} 1/s",
            f"  k_uptake: {self.k_uptake:.2e} 1/s",
            "",
            "Vasculature:",
            f"  MVD_normal: {self.MVD_normal:.0f} vessels/mm²",
            f"  MVD_tumor_core: {self.MVD_tumor_core:.0f} vessels/mm²",
            f"  MVD_tumor_rim: {self.MVD_tumor_rim:.0f} vessels/mm²",
            f"  P_vessel_tumor/normal: {self.P_vessel_tumor / self.P_vessel_normal:.1f}×",
            "",
            "Interstitial Pressure:",
            f"  IFP_normal: {self.IFP_normal:.1f} mmHg ({self.IFP_normal_Pa:.0f} Pa)",
            f"  IFP_tumor: {self.IFP_tumor:.1f} mmHg ({self.IFP_tumor_Pa:.0f} Pa)",
        ]
        return "\n".join(lines)
