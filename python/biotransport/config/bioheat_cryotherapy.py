"""Configuration for bioheat cryotherapy simulations.

This module provides the BioheatCryotherapyConfig dataclass for coupled
bioheat transfer with phase change and cell damage modeling.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class BioheatCryotherapyConfig:
    """Configuration for bioheat cryotherapy simulations.

    This dataclass encapsulates all parameters for coupled bioheat transfer
    with phase change and cell damage modeling.

    The physics includes:
    - Pennes bioheat equation (conduction + blood perfusion + metabolism)
    - Phase change (freezing) via effective heat capacity method
    - Arrhenius damage integral for cell death prediction

    All units are SI (K, m, s, W, J) unless otherwise noted.

    Attributes
    ----------
    domain_size_x : float
        Domain width [m]. Typical: 0.03-0.1 (3-10 cm).

    domain_size_y : float
        Domain height [m].

    rho_tissue : float
        Tissue density [kg/m³].
        Typical: 1000-1100 for soft tissue.

    c_tissue_unfrozen : float
        Specific heat of unfrozen tissue [J/(kg·K)].
        Typical: 3400-3800.

    c_tissue_frozen : float
        Specific heat of frozen tissue [J/(kg·K)].
        Typically about half of unfrozen value.

    k_tissue_unfrozen : float
        Thermal conductivity of unfrozen tissue [W/(m·K)].
        Typical: 0.4-0.6.

    k_tissue_frozen : float
        Thermal conductivity of frozen tissue [W/(m·K)].
        Typically 2-4× higher than unfrozen.

    rho_blood : float
        Blood density [kg/m³].
        Typical: 1050-1060.

    c_blood : float
        Specific heat of blood [J/(kg·K)].
        Typical: 3600-4000.

    w_b_normal : float
        Blood perfusion rate in normal tissue [1/s].
        Typical: 0.0003-0.001.

    w_b_tumor : float
        Blood perfusion rate in tumor [1/s].
        Often elevated due to angiogenesis.

    T_probe : float
        Cryoprobe temperature [K].
        Typical: 120-150 K (-150 to -120°C) for argon probes.

    probe_radius : float
        Cryoprobe radius [m].
        Typical: 1-3 mm.

    probe_position : tuple of float
        (x, y) position of probe center [m].

    q_met_normal : float
        Metabolic heat generation in normal tissue [W/m³].
        Typical: 400-500.

    q_met_tumor : float
        Metabolic heat generation in tumor [W/m³].
        Often elevated due to higher metabolic activity.

    T_freeze : float
        Freezing point of tissue [K].
        Slightly below 273.15 K due to solutes.

    T_freeze_range : float
        Temperature range for phase change [K].
        Models mushy zone in effective heat capacity.

    L_fusion : float
        Latent heat of fusion [J/kg].
        Water: 333,000 J/kg.

    E_activation : float
        Arrhenius activation energy for cell death [J/mol].
        Typical: 1.5e5 to 3e5.

    A_frequency : float
        Arrhenius frequency factor [1/s].
        Typical: 1e25 to 1e40.

    T_body : float
        Core body temperature [K].
        Typical: 310 K (37°C).

    tumor_radius : float
        Tumor radius [m].

    tumor_center : tuple of float
        (x, y) position of tumor center [m].

    nx : int
        Number of grid cells in x-direction.

    ny : int
        Number of grid cells in y-direction.

    dt : float
        Time step for simulation [s].

    Notes
    -----
    The Arrhenius damage integral Ω(t) = ∫ A·exp(-E_a/RT) dt
    represents cumulative thermal damage. Cell death probability
    is P_death = 1 - exp(-Ω).

    Common interpretation:
    - Ω = 1: 63% cell death
    - Ω = 4.6: 99% cell death

    References
    ----------
    .. [1] Pennes, H.H. "Analysis of tissue and arterial blood temperatures."
           J. Appl. Physiol. 1.2 (1948): 93-122.
    .. [2] Diller, K.R. "Modeling of bioheat transfer processes at high and
           low temperatures." Advances in Heat Transfer 22 (1992): 157-357.

    Examples
    --------
    >>> config = BioheatCryotherapyConfig(
    ...     domain_size_x=0.05,
    ...     domain_size_y=0.05,
    ...     T_probe=123.15,  # -150°C
    ... )
    >>> print(f"Probe at {config.T_probe - 273.15:.0f}°C")
    """

    # Domain geometry [m]
    domain_size_x: float = 0.05
    domain_size_y: float = 0.05

    # Grid resolution
    nx: int = 100
    ny: int = 100

    # Tissue thermal properties
    rho_tissue: float = 1050.0  # [kg/m³]
    c_tissue_unfrozen: float = 3600.0  # [J/(kg·K)]
    c_tissue_frozen: float = 1800.0  # [J/(kg·K)]
    k_tissue_unfrozen: float = 0.5  # [W/(m·K)]
    k_tissue_frozen: float = 2.0  # [W/(m·K)]

    # Blood properties
    rho_blood: float = 1060.0  # [kg/m³]
    c_blood: float = 3800.0  # [J/(kg·K)]
    w_b_normal: float = 0.0005  # [1/s]
    w_b_tumor: float = 0.002  # [1/s]

    # Cryoprobe parameters
    T_probe: float = 123.15  # [K] = -150°C
    probe_radius: float = 1.5e-3  # [m]
    probe_position: Optional[Tuple[float, float]] = None  # [m]

    # Metabolic heat generation [W/m³]
    q_met_normal: float = 420.0
    q_met_tumor: float = 840.0

    # Phase change parameters
    T_freeze: float = 272.15  # [K] = -1°C
    T_freeze_range: float = 2.0  # [K]
    L_fusion: float = 333000.0  # [J/kg]

    # Arrhenius damage parameters
    E_activation: float = 2.0e5  # [J/mol]
    A_frequency: float = 7.39e29  # [1/s]
    R_gas: float = 8.314  # [J/(mol·K)]

    # Boundary/initial conditions
    T_body: float = 310.15  # [K] = 37°C
    T_ambient: float = 293.15  # [K] = 20°C

    # Tumor geometry [m]
    tumor_radius: float = 0.01
    tumor_center: Optional[Tuple[float, float]] = None

    # Time stepping
    dt: float = 0.1  # [s]

    def __post_init__(self) -> None:
        """Set default positions if not specified."""
        if self.probe_position is None:
            self.probe_position = (self.domain_size_x / 2, self.domain_size_y / 2)
        if self.tumor_center is None:
            self.tumor_center = (self.domain_size_x / 2, self.domain_size_y / 2)

    @property
    def T_probe_celsius(self) -> float:
        """Probe temperature in Celsius."""
        return self.T_probe - 273.15

    @property
    def T_body_celsius(self) -> float:
        """Body temperature in Celsius."""
        return self.T_body - 273.15

    @property
    def effective_heat_capacity_frozen(self) -> float:
        """Effective heat capacity including latent heat [J/(kg·K)]."""
        # Approximate by distributing latent heat over freeze range
        return self.c_tissue_frozen + self.L_fusion / self.T_freeze_range

    @property
    def thermal_diffusivity_unfrozen(self) -> float:
        """Thermal diffusivity of unfrozen tissue [m²/s]."""
        return self.k_tissue_unfrozen / (self.rho_tissue * self.c_tissue_unfrozen)

    @property
    def thermal_diffusivity_frozen(self) -> float:
        """Thermal diffusivity of frozen tissue [m²/s]."""
        return self.k_tissue_frozen / (self.rho_tissue * self.c_tissue_frozen)

    @property
    def grid_spacing(self) -> Tuple[float, float]:
        """(dx, dy) grid spacing in meters."""
        dx = self.domain_size_x / self.nx
        dy = self.domain_size_y / self.ny
        return (dx, dy)

    def damage_rate(self, T: float) -> float:
        """Compute Arrhenius damage rate at temperature T [K].

        Parameters
        ----------
        T : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Damage rate dΩ/dt [1/s].
        """
        if T <= 0:
            return 0.0
        return self.A_frequency * np.exp(-self.E_activation / (self.R_gas * T))

    def death_probability(self, damage_integral: float) -> float:
        """Convert damage integral to cell death probability.

        Parameters
        ----------
        damage_integral : float
            Arrhenius damage integral Ω (dimensionless).

        Returns
        -------
        float
            Probability of cell death [0, 1].
        """
        return 1.0 - np.exp(-damage_integral)

    def describe(self) -> str:
        """Return a formatted description of the configuration."""
        lines = [
            "=== Bioheat Cryotherapy Configuration ===",
            "",
            "Domain:",
            f"  Size: {self.domain_size_x * 1e3:.1f} mm × {self.domain_size_y * 1e3:.1f} mm",
            f"  Grid: {self.nx} × {self.ny} ({(self.nx + 1) * (self.ny + 1)} nodes)",
            f"  Spacing: {self.grid_spacing[0] * 1e6:.0f} µm",
            "",
            "Cryoprobe:",
            f"  Temperature: {self.T_probe_celsius:.0f}°C ({self.T_probe:.1f} K)",
            f"  Radius: {self.probe_radius * 1e3:.1f} mm",
            f"  Position: ({self.probe_position[0] * 1e3:.1f}, {self.probe_position[1] * 1e3:.1f}) mm",
            "",
            "Tumor:",
            f"  Radius: {self.tumor_radius * 1e3:.1f} mm",
            f"  Center: ({self.tumor_center[0] * 1e3:.1f}, {self.tumor_center[1] * 1e3:.1f}) mm",
            "",
            "Tissue Properties:",
            f"  ρ: {self.rho_tissue:.0f} kg/m³",
            f"  c_unfrozen: {self.c_tissue_unfrozen:.0f} J/(kg·K)",
            f"  c_frozen: {self.c_tissue_frozen:.0f} J/(kg·K)",
            f"  k_unfrozen: {self.k_tissue_unfrozen:.2f} W/(m·K)",
            f"  k_frozen: {self.k_tissue_frozen:.2f} W/(m·K)",
            f"  α_unfrozen: {self.thermal_diffusivity_unfrozen:.2e} m²/s",
            "",
            "Blood Perfusion:",
            f"  w_b_normal: {self.w_b_normal:.4f} 1/s",
            f"  w_b_tumor: {self.w_b_tumor:.4f} 1/s",
            "",
            "Phase Change:",
            f"  T_freeze: {self.T_freeze - 273.15:.1f}°C",
            f"  L_fusion: {self.L_fusion / 1000:.0f} kJ/kg",
            "",
            "Arrhenius Damage:",
            f"  E_a: {self.E_activation / 1000:.0f} kJ/mol",
            f"  A: {self.A_frequency:.2e} 1/s",
            f"  Rate at 37°C: {self.damage_rate(310.15):.2e} 1/s",
            f"  Rate at -10°C: {self.damage_rate(263.15):.2e} 1/s",
        ]
        return "\n".join(lines)
