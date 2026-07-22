"""Science-first configuration for bioheat cryotherapy simulations.

Every temperature-bearing public name states its unit. The configuration uses
kelvin internally and provides an explicit Celsius constructor for convenience.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import InitVar, dataclass
import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type, TypeVar

import numpy as np

from biotransport.provenance import (
    ParameterSetProvenance,
    ParameterValue,
    illustrative_parameter_set,
)

if TYPE_CHECKING:
    from biotransport._core._core import (
        BioheatCryotherapySolver,
        StructuredMesh,
    )


_ConfigT = TypeVar("_ConfigT", bound="BioheatCryotherapyConfig")


_BIOHEAT_PARAMETER_UNITS = {
    "domain_size_x": "m",
    "domain_size_y": "m",
    "nx": "cells",
    "ny": "cells",
    "rho_tissue": "kg/m^3",
    "c_tissue_unfrozen": "J/(kg K)",
    "c_tissue_frozen": "J/(kg K)",
    "k_tissue_unfrozen": "W/(m K)",
    "k_tissue_frozen": "W/(m K)",
    "rho_blood": "kg/m^3",
    "c_blood": "J/(kg K)",
    "w_b_normal": "1/s",
    "w_b_tumor": "1/s",
    "T_probe_K": "K",
    "probe_radius": "m",
    "probe_position": "m (x, y)",
    "q_met_normal": "W/m^3",
    "q_met_tumor": "W/m^3",
    "T_freeze_K": "K",
    "T_freeze_range_K": "K",
    "L_fusion": "J/kg",
    "E_activation": "J/mol",
    "A_frequency": "1/s",
    "R_gas": "J/(mol K)",
    "T_initial_K": "K",
    "T_arterial_K": "K",
    "T_boundary_K": "K",
    "tumor_radius": "m",
    "tumor_center": "m (x, y)",
    "dt": "s",
}


def kelvin_from_celsius(temperature_C: float) -> float:
    """Convert a finite Celsius temperature to kelvin.

    Raises
    ------
    ValueError
        If the input is non-finite or at/below absolute zero.
    """

    value = float(temperature_C)
    if not math.isfinite(value):
        raise ValueError("temperature_C must be finite")
    kelvin = value + 273.15
    if kelvin <= 0.0:
        raise ValueError("temperature_C must be above absolute zero")
    return kelvin


def celsius_from_kelvin(temperature_K: float) -> float:
    """Convert a positive finite absolute temperature to Celsius."""

    value = float(temperature_K)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("temperature_K must be finite and greater than zero")
    return value - 273.15


def _require_finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _require_positive(value: float, name: str) -> float:
    converted = _require_finite(value, name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return converted


def _require_nonnegative(value: float, name: str) -> float:
    converted = _require_finite(value, name)
    if converted < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return converted


@dataclass
class BioheatCryotherapyConfig:
    """Validated SI configuration for Pennes bioheat with phase change.

    The modeled energy balance is

    ``rho_t c_app(T) dT/dt = div(k(T) grad(T))``
    ``+ rho_b c_b w_b f_liquid(T) (T_arterial - T)``
    ``+ f_liquid(T) q_met``.

    ``w_b`` is volumetric blood perfusion in
    m³_blood/(m³_tissue s), numerically equivalent to 1/s. The apparent
    mass-specific heat capacity is ``c_sensible + L*(-df_frozen/dT)`` in
    J/(kg K); density is applied once, outside that quantity, by the PDE.

    ``T_freeze_range_K`` denotes a two-standard-deviation mushy-zone width.
    The Arrhenius parameters describe heat injury. They do not constitute a
    validated cryogenic cell-death model.

    Probe-mask nodes used by the C++ solver are fixed-temperature embedded
    nodes. The model does not include probe heat capacity, coolant dynamics,
    thermal contact resistance, or a calibrated cryoinjury response.

    Use :meth:`from_celsius` when Celsius inputs are more convenient. Passing
    ambiguous names such as ``T_probe`` is intentionally unsupported.
    """

    # Domain geometry [m] and cell counts
    domain_size_x: float = 0.05
    domain_size_y: float = 0.05
    nx: int = 100
    ny: int = 100

    # Tissue thermal properties
    rho_tissue: float = 1050.0  # [kg/m^3]
    c_tissue_unfrozen: float = 3600.0  # [J/(kg K)]
    c_tissue_frozen: float = 1800.0  # [J/(kg K)]
    k_tissue_unfrozen: float = 0.5  # [W/(m K)]
    k_tissue_frozen: float = 2.0  # [W/(m K)]

    # Blood properties and volumetric perfusion [1/s]
    rho_blood: float = 1060.0  # [kg/m^3]
    c_blood: float = 3800.0  # [J/(kg K)]
    w_b_normal: float = 0.0005
    w_b_tumor: float = 0.002

    # Cryoprobe geometry and absolute temperature
    T_probe_K: float = 123.15
    probe_radius: float = 1.5e-3
    probe_position: Optional[Tuple[float, float]] = None

    # Volumetric metabolic heat generation [W/m^3]
    q_met_normal: float = 420.0
    q_met_tumor: float = 840.0

    # Phase change
    T_freeze_K: float = 272.15
    T_freeze_range_K: float = 2.0
    L_fusion: float = 333000.0  # [J/kg]

    # Arrhenius heat-injury diagnostic (not cryogenic injury)
    E_activation: float = 2.0e5  # [J/mol]
    A_frequency: float = 7.39e29  # [1/s]
    R_gas: float = 8.31446261815324  # [J/(mol K)]

    # Initial, arterial, and fixed outer-boundary temperatures [K]
    T_initial_K: float = 310.15
    T_arterial_K: float = 310.15
    T_boundary_K: float = 310.15

    # Tumor geometry [m]
    tumor_radius: float = 0.01
    tumor_center: Optional[Tuple[float, float]] = None

    # Maximum explicit step [s]. Default satisfies the conservative bound.
    dt: float = 0.05

    # Optional traceable records for every configuration value. This is an
    # InitVar so equality and the numerical solver API remain unchanged.
    parameter_provenance: InitVar[Optional[ParameterSetProvenance]] = None

    def __post_init__(
        self, parameter_provenance: Optional[ParameterSetProvenance]
    ) -> None:
        """Fill centered defaults and validate all domains and units."""

        if self.probe_position is None:
            self.probe_position = (self.domain_size_x / 2.0, self.domain_size_y / 2.0)
        if self.tumor_center is None:
            self.tumor_center = (self.domain_size_x / 2.0, self.domain_size_y / 2.0)
        self.validate()
        self._uses_generated_provenance = parameter_provenance is None
        if parameter_provenance is None:
            self._parameter_provenance = self._build_illustrative_provenance()
        else:
            self._parameter_provenance = parameter_provenance
            self.validate_parameter_provenance()

    @classmethod
    def from_celsius(
        cls: Type[_ConfigT],
        *,
        probe_C: float = -150.0,
        freeze_C: float = -1.0,
        initial_C: float = 37.0,
        arterial_C: float = 37.0,
        boundary_C: float = 37.0,
        **kwargs: Any,
    ) -> _ConfigT:
        """Construct a configuration with explicitly named Celsius inputs."""

        conflicting = {
            "T_probe_K",
            "T_freeze_K",
            "T_initial_K",
            "T_arterial_K",
            "T_boundary_K",
        }.intersection(kwargs)
        if conflicting:
            names = ", ".join(sorted(conflicting))
            raise TypeError(
                f"do not combine Celsius inputs with Kelvin keyword(s): {names}"
            )
        return cls(
            T_probe_K=kelvin_from_celsius(probe_C),
            T_freeze_K=kelvin_from_celsius(freeze_C),
            T_initial_K=kelvin_from_celsius(initial_C),
            T_arterial_K=kelvin_from_celsius(arterial_C),
            T_boundary_K=kelvin_from_celsius(boundary_C),
            **kwargs,
        )

    def validate(self) -> None:
        """Validate the complete configuration after construction or mutation."""

        positive_names = (
            "domain_size_x",
            "domain_size_y",
            "rho_tissue",
            "c_tissue_unfrozen",
            "c_tissue_frozen",
            "k_tissue_unfrozen",
            "k_tissue_frozen",
            "rho_blood",
            "c_blood",
            "probe_radius",
            "T_probe_K",
            "T_freeze_K",
            "T_freeze_range_K",
            "R_gas",
            "T_initial_K",
            "T_arterial_K",
            "T_boundary_K",
            "tumor_radius",
            "dt",
        )
        for name in positive_names:
            _require_positive(getattr(self, name), name)

        nonnegative_names = (
            "w_b_normal",
            "w_b_tumor",
            "q_met_normal",
            "q_met_tumor",
            "L_fusion",
            "E_activation",
            "A_frequency",
        )
        for name in nonnegative_names:
            _require_nonnegative(getattr(self, name), name)

        for name in ("nx", "ny"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer with value at least 2")

        if not self.T_probe_K < self.T_freeze_K:
            raise ValueError("T_probe_K must be below T_freeze_K")
        if not self.T_freeze_K < self.T_arterial_K:
            raise ValueError("T_arterial_K must be above T_freeze_K")

        self._validate_position(self.probe_position, "probe_position")
        self._validate_position(self.tumor_center, "tumor_center")

        maximum_dt = self.maximum_stable_dt_s
        tolerance = 32.0 * np.finfo(float).eps * maximum_dt
        if self.dt > maximum_dt + tolerance:
            raise ValueError(
                f"dt={self.dt:g} s exceeds the conservative explicit stability "
                f"limit {maximum_dt:g} s; use recommended_dt_s or refine less"
            )

        if hasattr(self, "_parameter_provenance"):
            if self._uses_generated_provenance:
                self._parameter_provenance = self._build_illustrative_provenance()
            else:
                self.validate_parameter_provenance()

    def _parameter_values(self) -> Dict[str, ParameterValue]:
        """Return the exact application and numerical values being used."""

        values: Dict[str, ParameterValue] = {}
        for name in _BIOHEAT_PARAMETER_UNITS:
            value = getattr(self, name)
            if value is None:
                raise ValueError(f"{name} must be resolved before recording provenance")
            values[name] = value
        return values

    def _build_illustrative_provenance(self) -> ParameterSetProvenance:
        return illustrative_parameter_set(
            "bioheat_cryotherapy",
            self._parameter_values(),
            _BIOHEAT_PARAMETER_UNITS,
            population_or_material=(
                "Generic tissue model; population and material are unspecified."
            ),
            notes=(
                "Library defaults and unsourced user inputs are illustrative, not "
                "patient-specific recommendations."
            ),
        )

    @property
    def provenance(self) -> ParameterSetProvenance:
        """Traceable records for the exact current configuration values.

        Library-created records are regenerated after parameter mutation and
        remain explicitly illustrative/unprovenanced. User-attached records are
        never silently rewritten; stale records raise :class:`ValueError`.
        """

        if self._uses_generated_provenance:
            self._parameter_provenance = self._build_illustrative_provenance()
        else:
            self.validate_parameter_provenance()
        return self._parameter_provenance

    def attach_parameter_provenance(self, provenance: ParameterSetProvenance) -> None:
        """Attach a complete traceable set matching all current values."""

        if provenance.model_identifier != "bioheat_cryotherapy":
            raise ValueError(
                "bioheat provenance model_identifier must be 'bioheat_cryotherapy'"
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

        if self._parameter_provenance.model_identifier != "bioheat_cryotherapy":
            raise ValueError(
                "bioheat provenance model_identifier must be 'bioheat_cryotherapy'"
            )
        self._parameter_provenance.validate_parameter_values(self._parameter_values())

    def _validate_position(
        self, position: Optional[Tuple[float, float]], name: str
    ) -> None:
        if position is None or not isinstance(position, (tuple, list)):
            raise ValueError(f"{name} must contain exactly two coordinates")
        if len(position) != 2:
            raise ValueError(f"{name} must contain exactly two coordinates")
        x = _require_finite(position[0], f"{name}[0]")
        y = _require_finite(position[1], f"{name}[1]")
        if not 0.0 <= x <= self.domain_size_x:
            raise ValueError(f"{name}[0] must lie inside the x domain")
        if not 0.0 <= y <= self.domain_size_y:
            raise ValueError(f"{name}[1] must lie inside the y domain")

    def create_solver(
        self,
        mesh: StructuredMesh,
        *,
        probe_mask: Sequence[int],
        perfusion_map: Sequence[float],
        q_met_map: Sequence[float],
    ) -> BioheatCryotherapySolver:
        """Create and fully configure the compiled C++ solver.

        ``perfusion_map`` is volumetric blood perfusion [1/s] and
        ``q_met_map`` is metabolic heat generation [W/m^3]. The mesh must
        match this configuration's cell counts and spacing. All map and mask
        validation is repeated by the C++ constructor.
        """

        from biotransport._core._core import BioheatCryotherapySolver

        self.validate()
        if mesh.is_1d() or mesh.nx() != self.nx or mesh.ny() != self.ny:
            raise ValueError(
                "mesh must be two-dimensional and match the configured nx and ny"
            )
        expected_dx, expected_dy = self.grid_spacing
        if not math.isclose(mesh.dx(), expected_dx, rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError("mesh.dx() does not match domain_size_x / nx")
        if not math.isclose(mesh.dy(), expected_dy, rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError("mesh.dy() does not match domain_size_y / ny")
        bounds_match = (
            math.isclose(mesh.x(0), 0.0, rel_tol=0.0, abs_tol=1.0e-14)
            and math.isclose(
                mesh.x(self.nx),
                self.domain_size_x,
                rel_tol=1.0e-12,
                abs_tol=1.0e-14,
            )
            and math.isclose(mesh.y(0, 0), 0.0, rel_tol=0.0, abs_tol=1.0e-14)
            and math.isclose(
                mesh.y(0, self.ny),
                self.domain_size_y,
                rel_tol=1.0e-12,
                abs_tol=1.0e-14,
            )
        )
        if not bounds_match:
            raise ValueError(
                "mesh domain bounds must be [0, domain_size_x] x [0, domain_size_y]"
            )

        solver = BioheatCryotherapySolver(
            mesh=mesh,
            probe_mask=list(probe_mask),
            perfusion_map=list(perfusion_map),
            q_met_map=list(q_met_map),
            rho_tissue=self.rho_tissue,
            rho_blood=self.rho_blood,
            c_blood=self.c_blood,
            k_unfrozen=self.k_tissue_unfrozen,
            k_frozen=self.k_tissue_frozen,
            c_unfrozen=self.c_tissue_unfrozen,
            c_frozen=self.c_tissue_frozen,
            T_body_K=self.T_initial_K,
            T_probe_K=self.T_probe_K,
            T_freeze_K=self.T_freeze_K,
            T_freeze_range_K=self.T_freeze_range_K,
            L_fusion=self.L_fusion,
            A=self.A_frequency,
            E_a=self.E_activation,
            R_gas=self.R_gas,
        )
        solver.set_initial_temperature_K(self.T_initial_K)
        solver.set_arterial_temperature_K(self.T_arterial_K)
        solver.set_boundary_temperature_K(self.T_boundary_K)
        return solver

    @staticmethod
    def kelvin_from_celsius(temperature_C: float) -> float:
        """Convert an explicitly Celsius temperature to kelvin."""

        return kelvin_from_celsius(temperature_C)

    @staticmethod
    def celsius_from_kelvin(temperature_K: float) -> float:
        """Convert an explicitly Kelvin temperature to Celsius."""

        return celsius_from_kelvin(temperature_K)

    @property
    def T_probe_C(self) -> float:
        """Cryoprobe temperature [degC]."""

        return celsius_from_kelvin(self.T_probe_K)

    @property
    def T_freeze_C(self) -> float:
        """Center of the apparent freezing transition [degC]."""

        return celsius_from_kelvin(self.T_freeze_K)

    @property
    def T_initial_C(self) -> float:
        """Initial tissue temperature [degC]."""

        return celsius_from_kelvin(self.T_initial_K)

    @property
    def freeze_sigma_K(self) -> float:
        """Gaussian standard deviation used for the mushy zone [K]."""

        return 0.5 * self.T_freeze_range_K

    def frozen_fraction(self, temperature_K: float) -> float:
        """Return the apparent frozen fraction at a temperature in kelvin."""

        value = _require_positive(temperature_K, "temperature_K")
        standardized = (value - self.T_freeze_K) / (
            math.sqrt(2.0) * self.freeze_sigma_K
        )
        return 0.5 * math.erfc(standardized)

    def effective_specific_heat(self, temperature_K: float) -> float:
        """Return apparent mass-specific heat capacity [J/(kg K)]."""

        value = _require_positive(temperature_K, "temperature_K")
        frozen = self.frozen_fraction(value)
        sensible = (
            self.c_tissue_unfrozen * (1.0 - frozen) + self.c_tissue_frozen * frozen
        )
        z = (value - self.T_freeze_K) / self.freeze_sigma_K
        latent_density = math.exp(-0.5 * z * z) / (
            math.sqrt(2.0 * math.pi) * self.freeze_sigma_K
        )
        result = sensible + self.L_fusion * latent_density
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError("phase-change parameters produce invalid heat capacity")
        return result

    @property
    def apparent_specific_heat_at_freezing(self) -> float:
        """Peak apparent mass-specific heat capacity [J/(kg K)]."""

        return self.effective_specific_heat(self.T_freeze_K)

    @property
    def thermal_diffusivity_unfrozen(self) -> float:
        """Unfrozen thermal diffusivity [m^2/s]."""

        return self.k_tissue_unfrozen / (self.rho_tissue * self.c_tissue_unfrozen)

    @property
    def thermal_diffusivity_frozen(self) -> float:
        """Frozen thermal diffusivity [m^2/s]."""

        return self.k_tissue_frozen / (self.rho_tissue * self.c_tissue_frozen)

    @property
    def grid_spacing(self) -> Tuple[float, float]:
        """Return ``(dx, dy)`` [m] for the nodal grid."""

        return (self.domain_size_x / self.nx, self.domain_size_y / self.ny)

    @property
    def maximum_stable_dt_s(self) -> float:
        """Conservative positivity bound for the explicit C++ update [s]."""

        dx, dy = self.grid_spacing
        k_max = max(self.k_tissue_unfrozen, self.k_tissue_frozen)
        c_min = min(self.c_tissue_unfrozen, self.c_tissue_frozen)
        w_max = max(self.w_b_normal, self.w_b_tumor)
        diagonal = 2.0 * k_max * (1.0 / dx**2 + 1.0 / dy**2)
        diagonal += self.rho_blood * self.c_blood * w_max
        result = self.rho_tissue * c_min / diagonal
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(
                "configuration produces no positive finite stable time step"
            )
        return result

    @property
    def recommended_dt_s(self) -> float:
        """A 10% safety margin below the conservative stability bound [s]."""

        return 0.9 * self.maximum_stable_dt_s

    def arrhenius_heat_injury_rate(self, temperature_K: float) -> float:
        """Return the Arrhenius heat-injury rate [1/s].

        This rate generally decreases during cooling and must not be presented
        as a cryogenic cell-death model.
        """

        value = _require_positive(temperature_K, "temperature_K")
        result = self.A_frequency * math.exp(-self.E_activation / (self.R_gas * value))
        if not math.isfinite(result) or result < 0.0:
            raise ValueError("Arrhenius parameters produce a non-finite rate")
        return result

    @staticmethod
    def arrhenius_injury_probability(damage_integral: float) -> float:
        """Convert a nonnegative Arrhenius integral to ``1-exp(-Omega)``."""

        value = _require_nonnegative(damage_integral, "damage_integral")
        return -math.expm1(-value)

    def describe(self) -> str:
        """Return a compact, unit-explicit configuration summary."""

        assert self.probe_position is not None
        assert self.tumor_center is not None
        dx, dy = self.grid_spacing
        return "\n".join(
            [
                "=== Bioheat Cryotherapy Configuration ===",
                f"Domain: {self.domain_size_x * 1e3:.1f} x "
                f"{self.domain_size_y * 1e3:.1f} mm, {self.nx} x {self.ny} cells",
                f"Spacing: dx={dx * 1e6:.1f} um, dy={dy * 1e6:.1f} um",
                f"Probe: {self.T_probe_C:.1f} degC ({self.T_probe_K:.2f} K), "
                f"radius={self.probe_radius * 1e3:.2f} mm",
                f"Freeze transition: center={self.T_freeze_C:.2f} degC, "
                f"two-sigma width={self.T_freeze_range_K:.2f} K",
                f"Initial/arterial/boundary: {self.T_initial_K:.2f}/"
                f"{self.T_arterial_K:.2f}/{self.T_boundary_K:.2f} K",
                f"Conductivity (unfrozen/frozen): {self.k_tissue_unfrozen:g}/"
                f"{self.k_tissue_frozen:g} W/(m K)",
                f"Specific heat (unfrozen/frozen): {self.c_tissue_unfrozen:g}/"
                f"{self.c_tissue_frozen:g} J/(kg K)",
                f"Latent heat: {self.L_fusion / 1000.0:.1f} kJ/kg",
                f"Configured dt: {self.dt:.5g} s; conservative maximum: "
                f"{self.maximum_stable_dt_s:.5g} s",
                "Damage output: Arrhenius heat injury only (not cryogenic cell death).",
            ]
        )
