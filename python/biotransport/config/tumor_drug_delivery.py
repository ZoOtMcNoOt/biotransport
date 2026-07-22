"""Validated configuration for the prescribed-pressure tumor transport model.

The C++ solver uses a Darcy pressure surrogate plus conservative transport.  It
does **not** infer interstitial pressure from Starling filtration or lymphatic
drainage.  ``IFP_normal`` and ``IFP_tumor`` must therefore come from a stated
experimental assumption or a separate fluid-exchange model.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
import math
from typing import Dict, Optional, Tuple

from biotransport.provenance import (
    ParameterSetProvenance,
    ParameterValue,
    illustrative_parameter_set,
)


MMHG_TO_PA = 133.322387415
"""Exact conventional conversion used by this module [Pa/mmHg]."""


_TUMOR_PARAMETER_UNITS = {
    "domain_size": "m",
    "tumor_radius": "m",
    "tumor_center": "m (x, y)",
    "rim_thickness": "m",
    "nx": "cells",
    "ny": "cells",
    "D_drug_normal": "m^2/s",
    "D_drug_tumor": "m^2/s",
    "k_binding": "1/s",
    "k_uptake": "1/s",
    "MVD_normal": "vessels/mm^2",
    "MVD_tumor_core": "vessels/mm^2",
    "MVD_tumor_rim": "vessels/mm^2",
    "vessel_radius": "m",
    "P_vessel_normal": "m/s",
    "P_vessel_tumor": "m/s",
    "C_plasma": "model concentration units",
    "IFP_normal": "mmHg",
    "IFP_tumor": "mmHg",
    "K_hydraulic_normal": "m^2/(Pa s)",
    "K_hydraulic_tumor": "m^2/(Pa s)",
}


def _finite(name: str, value: float) -> float:
    """Return *value* as a float or reject non-numeric/non-finite input."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass
class TumorDrugDeliveryConfig:
    """Inputs for prescribed-pressure Darcy drug transport.

    All solver-facing quantities are SI.  Pressures remain user-facing in
    mmHg because that is conventional in tumor-IFP measurements; the
    ``*_Pa`` properties perform the conversion.

    ``MVD_*`` is microvessel profile density [vessels/mm²].  The C++ vascular
    source requires perfused vessel surface area per tissue volume ``S_v``
    [1/m], not a normalized vessel count.  The ``vascular_surface_area_*``
    properties make the explicit cylindrical-vessel closure

    ``S_v = 2*pi*vessel_radius*MVD*1e6``.

    This assumes the converted MVD represents vessel length per tissue volume
    [m/m³].  If that stereological assumption is unsuitable, construct the
    solver field from measured ``S_v`` directly.

    Binding and uptake are irreversible first-order transfers out of the free
    compartment.  The pressure-clamp surrogate implies an unresolved,
    solute-free fluid source at the clamp interface.  Saturable binding,
    unbinding, metabolism, systemic pharmacokinetics, Starling fluid and
    solvent-drag sources, and lymphatic sinks are outside this model.
    """

    # Square domain geometry [m]
    domain_size: float = 5e-3
    tumor_radius: float = 2e-3
    tumor_center: Optional[Tuple[float, float]] = None
    rim_thickness: float = 0.5e-3

    # Grid cell counts
    nx: int = 100
    ny: int = 100

    # Effective free-drug diffusion coefficients [m²/s]
    D_drug_normal: float = 5e-11
    D_drug_tumor: float = 2e-11

    # Irreversible first-order compartment-transfer rates [1/s]
    k_binding: float = 1e-3
    k_uptake: float = 5e-4

    # Microvessel profile density [vessels/mm²] and assumed vessel radius [m]
    MVD_normal: float = 100.0
    MVD_tumor_core: float = 20.0
    MVD_tumor_rim: float = 200.0
    vessel_radius: float = 5e-6

    # Vessel-wall permeability [m/s] and plasma concentration [model units]
    P_vessel_normal: float = 1e-7
    P_vessel_tumor: float = 5e-7
    C_plasma: float = 1.0

    # Prescribed interstitial gauge pressures [mmHg]
    IFP_normal: float = 0.0
    IFP_tumor: float = 20.0

    # Darcy mobility K [m²/(Pa*s)]
    K_hydraulic_normal: float = 5e-12
    K_hydraulic_tumor: float = 2.5e-12

    # Optional records supplied with the parameter factory. This InitVar keeps
    # the numerical dataclass representation and equality behavior compatible.
    parameter_provenance: InitVar[Optional[ParameterSetProvenance]] = None

    def __post_init__(
        self, parameter_provenance: Optional[ParameterSetProvenance]
    ) -> None:
        """Normalize numeric inputs and enforce the C++ model's physical domain."""

        scalar_names = (
            "domain_size",
            "tumor_radius",
            "rim_thickness",
            "D_drug_normal",
            "D_drug_tumor",
            "k_binding",
            "k_uptake",
            "MVD_normal",
            "MVD_tumor_core",
            "MVD_tumor_rim",
            "vessel_radius",
            "P_vessel_normal",
            "P_vessel_tumor",
            "C_plasma",
            "IFP_normal",
            "IFP_tumor",
            "K_hydraulic_normal",
            "K_hydraulic_tumor",
        )
        for name in scalar_names:
            setattr(self, name, _finite(name, getattr(self, name)))

        for name in ("nx", "ny"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer cell count")
            if value < 2:
                raise ValueError(f"{name} must be at least 2")

        if self.domain_size <= 0.0:
            raise ValueError("domain_size must be positive")
        if self.tumor_radius <= 0.0:
            raise ValueError("tumor_radius must be positive")
        if not 0.0 < self.rim_thickness <= self.tumor_radius:
            raise ValueError("rim_thickness must be in (0, tumor_radius]")

        if self.tumor_center is None:
            center = (self.domain_size / 2.0, self.domain_size / 2.0)
        else:
            if len(self.tumor_center) != 2:
                raise ValueError("tumor_center must contain exactly (x, y)")
            center = (
                _finite("tumor_center[0]", self.tumor_center[0]),
                _finite("tumor_center[1]", self.tumor_center[1]),
            )
        clearance = min(
            center[0],
            self.domain_size - center[0],
            center[1],
            self.domain_size - center[1],
        )
        if self.tumor_radius >= clearance:
            raise ValueError(
                "tumor must lie strictly inside the domain so its pressure mask does not "
                "conflict with the outer pressure boundary"
            )
        self.tumor_center = center

        for name in (
            "D_drug_normal",
            "D_drug_tumor",
            "k_binding",
            "k_uptake",
            "MVD_normal",
            "MVD_tumor_core",
            "MVD_tumor_rim",
            "P_vessel_normal",
            "P_vessel_tumor",
            "C_plasma",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.vessel_radius <= 0.0:
            raise ValueError("vessel_radius must be positive")
        if self.K_hydraulic_normal <= 0.0 or self.K_hydraulic_tumor <= 0.0:
            raise ValueError("hydraulic conductivities must be positive")
        if self.IFP_tumor < self.IFP_normal:
            raise ValueError(
                "IFP_tumor must be >= IFP_normal; lower tumor pressure would produce "
                "boundary inflow, for which no external drug concentration is defined"
            )
        if not math.isfinite(self.IFP_normal * MMHG_TO_PA) or not math.isfinite(
            self.IFP_tumor * MMHG_TO_PA
        ):
            raise ValueError("interstitial pressures are not representable in pascals")

        dx, dy = self.grid_spacing
        if dx <= 0.0 or dy <= 0.0:
            raise ValueError("domain_size/nx and domain_size/ny must be representable")

        surface_areas = (
            self.vascular_surface_area_normal,
            self.vascular_surface_area_tumor_core,
            self.vascular_surface_area_tumor_rim,
        )
        for permeability, surface_area in (
            (self.P_vessel_normal, surface_areas[0]),
            (self.P_vessel_tumor, surface_areas[1]),
            (self.P_vessel_tumor, surface_areas[2]),
        ):
            if not math.isfinite(permeability * surface_area):
                raise ValueError(
                    "vessel permeability times vascular surface area density must be "
                    "representable as a finite exchange rate"
                )

        if hasattr(self, "_parameter_provenance"):
            self.validate_parameter_provenance()
        else:
            self._uses_generated_provenance = parameter_provenance is None
            if parameter_provenance is None:
                self._parameter_provenance = self._build_illustrative_provenance()
            else:
                self._parameter_provenance = parameter_provenance
                self.validate_parameter_provenance()

    def _parameter_values(self) -> Dict[str, ParameterValue]:
        """Return the exact application and numerical values being used."""

        values: Dict[str, ParameterValue] = {}
        for name in _TUMOR_PARAMETER_UNITS:
            value = getattr(self, name)
            if value is None:
                raise ValueError(f"{name} must be resolved before recording provenance")
            values[name] = value
        return values

    def _build_illustrative_provenance(self) -> ParameterSetProvenance:
        return illustrative_parameter_set(
            "tumor_drug_delivery",
            self._parameter_values(),
            _TUMOR_PARAMETER_UNITS,
            population_or_material=(
                "Generic tumor/tissue model; population, tissue, and drug are "
                "unspecified."
            ),
            notes=(
                "Library defaults and unsourced user inputs are illustrative, not "
                "patient-specific recommendations."
            ),
        )

    @property
    def provenance(self) -> ParameterSetProvenance:
        """Traceable records for the exact current configuration values.

        Generated records follow mutations while remaining explicitly
        illustrative. Attached evidence records are immutable and must be
        replaced when a parameter value changes.
        """

        if self._uses_generated_provenance:
            self._parameter_provenance = self._build_illustrative_provenance()
        else:
            self.validate_parameter_provenance()
        return self._parameter_provenance

    def attach_parameter_provenance(self, provenance: ParameterSetProvenance) -> None:
        """Attach a complete traceable set matching all current values."""

        if provenance.model_identifier != "tumor_drug_delivery":
            raise ValueError(
                "tumor provenance model_identifier must be 'tumor_drug_delivery'"
            )
        provenance.validate_parameter_values(self._parameter_values())
        self._parameter_provenance = provenance
        self._uses_generated_provenance = False

    def reset_parameter_provenance_as_illustrative(self) -> None:
        """Discard attached claims and mark current values unprovenanced."""

        self._parameter_provenance = self._build_illustrative_provenance()
        self._uses_generated_provenance = True

    def validate_parameter_provenance(self) -> None:
        """Fail if attached records are missing, for another model, or stale."""

        if self._parameter_provenance.model_identifier != "tumor_drug_delivery":
            raise ValueError(
                "tumor provenance model_identifier must be 'tumor_drug_delivery'"
            )
        self._parameter_provenance.validate_parameter_values(self._parameter_values())

    @property
    def IFP_normal_Pa(self) -> float:
        """Prescribed outer-boundary pressure [Pa]."""

        return self.IFP_normal * MMHG_TO_PA

    @property
    def IFP_tumor_Pa(self) -> float:
        """Prescribed pressure on tumor-mask nodes [Pa]."""

        return self.IFP_tumor * MMHG_TO_PA

    def vascular_surface_area_density(self, mvd: float) -> float:
        """Convert MVD [vessels/mm²] to the assumed ``S_v`` [1/m]."""

        mvd_value = _finite("mvd", mvd)
        if mvd_value < 0.0:
            raise ValueError("mvd must be non-negative")
        surface_area_density = 2.0 * math.pi * self.vessel_radius * mvd_value * 1e6
        if not math.isfinite(surface_area_density):
            raise ValueError("vascular surface area density is not representable")
        return surface_area_density

    @property
    def vascular_surface_area_normal(self) -> float:
        """Normal-tissue perfused vascular surface area density [1/m]."""

        return self.vascular_surface_area_density(self.MVD_normal)

    @property
    def vascular_surface_area_tumor_core(self) -> float:
        """Tumor-core perfused vascular surface area density [1/m]."""

        return self.vascular_surface_area_density(self.MVD_tumor_core)

    @property
    def vascular_surface_area_tumor_rim(self) -> float:
        """Tumor-rim perfused vascular surface area density [1/m]."""

        return self.vascular_surface_area_density(self.MVD_tumor_rim)

    @property
    def vascular_exchange_rate_normal(self) -> float:
        """Normal-tissue solute exchange coefficient ``P*S_v`` [1/s]."""

        return self.P_vessel_normal * self.vascular_surface_area_normal

    @property
    def vascular_exchange_rate_tumor_core(self) -> float:
        """Tumor-core solute exchange coefficient ``P*S_v`` [1/s]."""

        return self.P_vessel_tumor * self.vascular_surface_area_tumor_core

    @property
    def vascular_exchange_rate_tumor_rim(self) -> float:
        """Tumor-rim solute exchange coefficient ``P*S_v`` [1/s]."""

        return self.P_vessel_tumor * self.vascular_surface_area_tumor_rim

    @property
    def tumor_area_fraction(self) -> float:
        """Geometric tumor area divided by square-domain area."""

        return math.pi * self.tumor_radius**2 / self.domain_size**2

    @property
    def grid_spacing(self) -> Tuple[float, float]:
        """Return ``(dx, dy)`` [m]."""

        return (self.domain_size / self.nx, self.domain_size / self.ny)

    def describe(self) -> str:
        """Return a concise, unit-explicit model summary."""

        permeability_ratio = (
            f"{self.P_vessel_tumor / self.P_vessel_normal:.1f}x"
            if self.P_vessel_normal > 0.0
            else "undefined (normal permeability is zero)"
        )
        assert self.tumor_center is not None
        return "\n".join(
            [
                "=== Tumor Drug Delivery Configuration ===",
                "Model: prescribed IFP + Darcy flow; no Starling/lymphatic pressure solve",
                "Pressure-clamp fluid source: unresolved and treated as solute-free",
                "",
                "Domain:",
                f"  Size: {self.domain_size * 1e3:.2f} mm x {self.domain_size * 1e3:.2f} mm",
                f"  Grid: {self.nx} x {self.ny} cells",
                f"  Spacing: ({self.grid_spacing[0] * 1e6:.1f}, "
                f"{self.grid_spacing[1] * 1e6:.1f}) um",
                "",
                "Tumor:",
                f"  Radius: {self.tumor_radius * 1e3:.2f} mm",
                f"  Center: ({self.tumor_center[0] * 1e3:.2f}, "
                f"{self.tumor_center[1] * 1e3:.2f}) mm",
                f"  Rim thickness: {self.rim_thickness * 1e3:.2f} mm",
                f"  Area fraction: {self.tumor_area_fraction * 100:.1f}%",
                "",
                "Free-drug transport:",
                f"  D_normal: {self.D_drug_normal:.2e} m^2/s",
                f"  D_tumor: {self.D_drug_tumor:.2e} m^2/s",
                f"  irreversible k_binding: {self.k_binding:.2e} 1/s",
                f"  irreversible k_uptake: {self.k_uptake:.2e} 1/s",
                "",
                "Vascular exchange:",
                f"  S_v normal/core/rim: {self.vascular_surface_area_normal:.2e} / "
                f"{self.vascular_surface_area_tumor_core:.2e} / "
                f"{self.vascular_surface_area_tumor_rim:.2e} 1/m",
                f"  P*S_v normal/core/rim: {self.vascular_exchange_rate_normal:.2e} / "
                f"{self.vascular_exchange_rate_tumor_core:.2e} / "
                f"{self.vascular_exchange_rate_tumor_rim:.2e} 1/s",
                f"  P_tumor/P_normal: {permeability_ratio}",
                "",
                "Prescribed interstitial pressure:",
                f"  Boundary: {self.IFP_normal:.1f} mmHg ({self.IFP_normal_Pa:.0f} Pa)",
                f"  Tumor mask: {self.IFP_tumor:.1f} mmHg ({self.IFP_tumor_Pa:.0f} Pa)",
            ]
        )
