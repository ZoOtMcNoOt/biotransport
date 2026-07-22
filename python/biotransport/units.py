"""Small, dependency-free quantities for BioTransport API boundaries.

The numerical kernels use SI values.  This module makes conversion into those
values explicit while retaining a runtime dimension tag, so that (for example)
a pressure cannot accidentally be supplied where a diffusivity is expected.

The implementation is intentionally narrow.  It covers the quantities that
occur most often in BioTransport's public models without attempting symbolic
dimensional algebra or becoming a general-purpose units package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real as RealNumber
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Tuple, Union

Numeric = Union[int, float]


class UnitError(ValueError):
    """Base class for unit and quantity validation failures."""


class UnknownUnitError(UnitError):
    """Raised when a unit symbol is not in this module's explicit registry."""


class DimensionError(UnitError):
    """Raised when an operation mixes incompatible physical dimensions."""


class QuantityDomainError(UnitError):
    """Raised when a quantity is outside its physically meaningful domain."""


class ConversionContextError(UnitError):
    """Raised when a conversion needs physical context the caller omitted."""


class Dimension(Enum):
    """Semantic dimensions supported at BioTransport's Python boundary."""

    ABSOLUTE_TEMPERATURE = "absolute temperature"
    TEMPERATURE_DIFFERENCE = "temperature difference"
    PRESSURE = "pressure"
    MOLAR_CONCENTRATION = "molar concentration"
    DIFFUSIVITY = "diffusivity"
    SOLUTE_PERMEABILITY = "solute permeability"
    INTRINSIC_PERMEABILITY = "intrinsic permeability"
    DARCY_MOBILITY = "Darcy mobility"
    VOLUMETRIC_PERFUSION = "volumetric perfusion rate"
    MASS_SPECIFIC_PERFUSION = "mass-specific perfusion"
    LENGTH = "length"
    TIME = "time"
    AMOUNT = "amount of substance"
    ENERGY = "energy"


_SI_SYMBOLS = {
    Dimension.ABSOLUTE_TEMPERATURE: "K",
    Dimension.TEMPERATURE_DIFFERENCE: "delta_K",
    Dimension.PRESSURE: "Pa",
    Dimension.MOLAR_CONCENTRATION: "mol/m^3",
    Dimension.DIFFUSIVITY: "m^2/s",
    Dimension.SOLUTE_PERMEABILITY: "m/s",
    Dimension.INTRINSIC_PERMEABILITY: "m^2",
    Dimension.DARCY_MOBILITY: "m^2/(Pa*s)",
    Dimension.VOLUMETRIC_PERFUSION: "1/s",
    Dimension.MASS_SPECIFIC_PERFUSION: "m^3/(kg*s)",
    Dimension.LENGTH: "m",
    Dimension.TIME: "s",
    Dimension.AMOUNT: "mol",
    Dimension.ENERGY: "J",
}


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, RealNumber):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise QuantityDomainError(f"{name} must be finite, got {value!r}")
    return result


def _validate_si_domain(value: float, dimension: Dimension) -> None:
    if dimension is Dimension.ABSOLUTE_TEMPERATURE and value < 0.0:
        raise QuantityDomainError(
            f"absolute temperature cannot be below absolute zero, got {value:g} K"
        )

    nonnegative = {
        Dimension.MOLAR_CONCENTRATION,
        Dimension.DIFFUSIVITY,
        Dimension.SOLUTE_PERMEABILITY,
        Dimension.INTRINSIC_PERMEABILITY,
        Dimension.DARCY_MOBILITY,
        Dimension.VOLUMETRIC_PERFUSION,
        Dimension.MASS_SPECIFIC_PERFUSION,
        Dimension.LENGTH,
        Dimension.TIME,
        Dimension.AMOUNT,
    }
    if dimension in nonnegative and value < 0.0:
        raise QuantityDomainError(
            f"{dimension.value} must be non-negative, got {value:g} "
            f"{_SI_SYMBOLS[dimension]}"
        )


@dataclass(frozen=True)
class Unit:
    """An immutable affine conversion into one supported SI dimension.

    ``si = value * scale + offset``.  Offset units are used only for absolute
    temperature; all other registered units are linear.
    """

    symbol: str
    dimension: Dimension
    scale: float = 1.0
    offset: float = 0.0
    aliases: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError("unit symbol must be a non-empty string")
        if not isinstance(self.dimension, Dimension):
            raise TypeError("unit dimension must be a Dimension")
        scale = _finite_number(self.scale, "unit scale")
        offset = _finite_number(self.offset, "unit offset")
        if scale <= 0.0:
            raise ValueError("unit scale must be positive")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "aliases", tuple(self.aliases))

    def to_si(self, value: Numeric) -> float:
        """Convert a finite value in this unit to its SI representation."""

        source = _finite_number(value, "quantity value")
        result = source * self.scale + self.offset
        if not math.isfinite(result):
            raise QuantityDomainError("unit conversion produced a non-finite value")
        return result

    def from_si(self, value: Numeric) -> float:
        """Convert a finite SI value into this unit."""

        source = _finite_number(value, "SI value")
        result = (source - self.offset) / self.scale
        if not math.isfinite(result):
            raise QuantityDomainError("unit conversion produced a non-finite value")
        return result


def _unit(
    symbol: str,
    dimension: Dimension,
    scale: float = 1.0,
    offset: float = 0.0,
    aliases: Iterable[str] = (),
) -> Unit:
    return Unit(symbol, dimension, scale, offset, tuple(aliases))


_REGISTERED_UNITS = (
    # Absolute temperature.  Kelvin is exact; degree conversions follow the SI.
    _unit("K", Dimension.ABSOLUTE_TEMPERATURE, aliases=("kelvin",)),
    _unit(
        "degC",
        Dimension.ABSOLUTE_TEMPERATURE,
        offset=273.15,
        aliases=("°C", "Celsius"),
    ),
    _unit(
        "degF",
        Dimension.ABSOLUTE_TEMPERATURE,
        scale=5.0 / 9.0,
        offset=273.15 - 32.0 * 5.0 / 9.0,
        aliases=("°F", "Fahrenheit"),
    ),
    # Temperature intervals deliberately use distinct symbols and a dimension
    # distinct from absolute temperature.
    _unit(
        "delta_K",
        Dimension.TEMPERATURE_DIFFERENCE,
        aliases=("deltaK", "K_difference"),
    ),
    _unit(
        "delta_degC",
        Dimension.TEMPERATURE_DIFFERENCE,
        aliases=("delta_°C", "degC_difference"),
    ),
    _unit(
        "delta_degF",
        Dimension.TEMPERATURE_DIFFERENCE,
        scale=5.0 / 9.0,
        aliases=("delta_°F", "degF_difference"),
    ),
    # Pressure.  mmHg and cmH2O are conventional conversion factors.
    _unit("Pa", Dimension.PRESSURE, aliases=("pascal", "pascals")),
    _unit("kPa", Dimension.PRESSURE, 1e3),
    _unit("MPa", Dimension.PRESSURE, 1e6),
    _unit("bar", Dimension.PRESSURE, 1e5),
    _unit("mbar", Dimension.PRESSURE, 1e2),
    _unit("atm", Dimension.PRESSURE, 101325.0),
    _unit("torr", Dimension.PRESSURE, 101325.0 / 760.0),
    _unit("mmHg", Dimension.PRESSURE, 133.322387415),
    _unit("cmH2O", Dimension.PRESSURE, 98.0665, aliases=("cmH₂O",)),
    # Molar concentration.  Numerically, 1 mM == 1 mol/m^3.
    _unit(
        "mol/m^3",
        Dimension.MOLAR_CONCENTRATION,
        aliases=("mol/m3", "mol*m^-3", "mol·m^-3", "mol·m⁻³"),
    ),
    _unit("mol/L", Dimension.MOLAR_CONCENTRATION, 1e3, aliases=("M",)),
    _unit("mM", Dimension.MOLAR_CONCENTRATION, 1.0, aliases=("mmol/L",)),
    _unit(
        "uM",
        Dimension.MOLAR_CONCENTRATION,
        1e-3,
        aliases=("µM", "μM", "umol/L", "µmol/L"),
    ),
    _unit("nM", Dimension.MOLAR_CONCENTRATION, 1e-6, aliases=("nmol/L",)),
    _unit("pM", Dimension.MOLAR_CONCENTRATION, 1e-9, aliases=("pmol/L",)),
    # Molecular/effective diffusivity.
    _unit("m^2/s", Dimension.DIFFUSIVITY, aliases=("m2/s", "m²/s")),
    _unit("cm^2/s", Dimension.DIFFUSIVITY, 1e-4, aliases=("cm2/s", "cm²/s")),
    _unit("mm^2/s", Dimension.DIFFUSIVITY, 1e-6, aliases=("mm2/s", "mm²/s")),
    _unit(
        "um^2/s",
        Dimension.DIFFUSIVITY,
        1e-12,
        aliases=("um2/s", "µm^2/s", "µm²/s", "μm²/s"),
    ),
    # Solute permeability / mass-transfer velocity.
    _unit("m/s", Dimension.SOLUTE_PERMEABILITY),
    _unit("cm/s", Dimension.SOLUTE_PERMEABILITY, 1e-2),
    _unit("mm/s", Dimension.SOLUTE_PERMEABILITY, 1e-3),
    _unit("um/s", Dimension.SOLUTE_PERMEABILITY, 1e-6, aliases=("µm/s", "μm/s")),
    # Intrinsic permeability is L^2, not solute permeability.  The darcy is
    # defined here by its conventional exact conversion used in engineering.
    _unit("m^2", Dimension.INTRINSIC_PERMEABILITY, aliases=("m2", "m²")),
    _unit("darcy", Dimension.INTRINSIC_PERMEABILITY, 9.869233e-13, aliases=("D",)),
    _unit("mD", Dimension.INTRINSIC_PERMEABILITY, 9.869233e-16),
    # K/mu in Darcy flow.  It remains semantically distinct from intrinsic K.
    _unit(
        "m^2/(Pa*s)",
        Dimension.DARCY_MOBILITY,
        aliases=("m2/(Pa*s)", "m²/(Pa·s)"),
    ),
    _unit("cm^2/(Pa*s)", Dimension.DARCY_MOBILITY, 1e-4),
    # Volumetric perfusion: blood volume per tissue volume per time.
    _unit("1/s", Dimension.VOLUMETRIC_PERFUSION, aliases=("s^-1", "s⁻¹")),
    _unit("1/min", Dimension.VOLUMETRIC_PERFUSION, 1.0 / 60.0, aliases=("min^-1",)),
    _unit("1/h", Dimension.VOLUMETRIC_PERFUSION, 1.0 / 3600.0, aliases=("h^-1",)),
    _unit("%/min", Dimension.VOLUMETRIC_PERFUSION, 0.01 / 60.0),
    # Mass-specific perfusion: blood volume per tissue mass per time.  These
    # require a tissue bulk density before conversion to 1/s.
    _unit("m^3/(kg*s)", Dimension.MASS_SPECIFIC_PERFUSION),
    _unit(
        "mL/(kg*min)",
        Dimension.MASS_SPECIFIC_PERFUSION,
        1e-6 / 60.0,
        aliases=("mL/(min*kg)",),
    ),
    _unit(
        "mL/(100g*min)",
        Dimension.MASS_SPECIFIC_PERFUSION,
        1e-6 / (0.1 * 60.0),
        aliases=("mL/(min*100g)",),
    ),
    # Length and time magnitudes.
    _unit("m", Dimension.LENGTH, aliases=("metre", "meter")),
    _unit("cm", Dimension.LENGTH, 1e-2),
    _unit("mm", Dimension.LENGTH, 1e-3),
    _unit("um", Dimension.LENGTH, 1e-6, aliases=("µm", "μm")),
    _unit("nm", Dimension.LENGTH, 1e-9),
    _unit("s", Dimension.TIME, aliases=("second", "seconds")),
    _unit("ms", Dimension.TIME, 1e-3),
    _unit("us", Dimension.TIME, 1e-6, aliases=("µs", "μs")),
    _unit("min", Dimension.TIME, 60.0, aliases=("minute", "minutes")),
    _unit("h", Dimension.TIME, 3600.0, aliases=("hour", "hours")),
    _unit("day", Dimension.TIME, 86400.0, aliases=("days",)),
    # Amount of substance and energy.
    _unit("mol", Dimension.AMOUNT),
    _unit("mmol", Dimension.AMOUNT, 1e-3),
    _unit("umol", Dimension.AMOUNT, 1e-6, aliases=("µmol", "μmol")),
    _unit("nmol", Dimension.AMOUNT, 1e-9),
    _unit("pmol", Dimension.AMOUNT, 1e-12),
    _unit("J", Dimension.ENERGY, aliases=("joule", "joules")),
    _unit("kJ", Dimension.ENERGY, 1e3),
    _unit("mJ", Dimension.ENERGY, 1e-3),
    _unit("uJ", Dimension.ENERGY, 1e-6, aliases=("µJ", "μJ")),
    _unit("cal", Dimension.ENERGY, 4.184, aliases=("cal_th",)),
    _unit("kcal", Dimension.ENERGY, 4184.0, aliases=("kcal_th",)),
)


_TRANSLATION = str.maketrans({"−": "-", "–": "-", "⋅": "*", "·": "*"})


def _normalize_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        raise TypeError(f"unit must be a string or Unit, got {type(symbol).__name__}")
    return "".join(symbol.strip().translate(_TRANSLATION).split())


def _build_registry() -> Mapping[str, Unit]:
    registry: dict[str, Unit] = {}
    for unit_value in _REGISTERED_UNITS:
        for symbol in (unit_value.symbol,) + unit_value.aliases:
            key = _normalize_symbol(symbol)
            existing = registry.get(key)
            if existing is not None and existing != unit_value:
                raise RuntimeError(f"duplicate unit alias {symbol!r}")
            registry[key] = unit_value
    return MappingProxyType(registry)


UNITS: Mapping[str, Unit] = _build_registry()
"""Read-only mapping of accepted unit spellings to canonical :class:`Unit`s."""

UnitLike = Union[str, Unit]


def get_unit(unit: UnitLike) -> Unit:
    """Resolve a unit spelling, raising an actionable error if it is unknown."""

    if isinstance(unit, Unit):
        return unit
    key = _normalize_symbol(unit)
    try:
        return UNITS[key]
    except KeyError as exc:
        raise UnknownUnitError(
            f"unknown unit {unit!r}; call available_units() for supported symbols"
        ) from exc


def available_units(dimension: Optional[Dimension] = None) -> Tuple[str, ...]:
    """Return canonical symbols, optionally filtered by semantic dimension."""

    if dimension is not None and not isinstance(dimension, Dimension):
        raise TypeError("dimension must be a Dimension or None")
    return tuple(
        unit.symbol
        for unit in _REGISTERED_UNITS
        if dimension is None or unit.dimension is dimension
    )


def _density(value: Optional[Numeric]) -> float:
    if value is None:
        raise ConversionContextError(
            "converting mass-specific perfusion to or from volumetric perfusion "
            "requires tissue_density_kg_m3"
        )
    density = _finite_number(value, "tissue_density_kg_m3")
    if density <= 0.0:
        raise QuantityDomainError("tissue_density_kg_m3 must be positive")
    return density


@dataclass(frozen=True)
class Quantity:
    """Immutable SI value with a runtime semantic dimension.

    Construct quantities through :func:`quantity` or one of the dimension-
    named helpers.  ``si_value`` is always the value expected by a solver that
    documents SI inputs.  :meth:`to` returns a plain number in an explicitly
    requested unit.
    """

    si_value: float
    dimension: Dimension

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, Dimension):
            raise TypeError("quantity dimension must be a Dimension")
        value = _finite_number(self.si_value, "SI value")
        _validate_si_domain(value, self.dimension)
        object.__setattr__(self, "si_value", value)

    @classmethod
    def from_unit(cls, value: Numeric, unit: UnitLike) -> Quantity:
        """Construct a quantity from a finite value and explicit unit."""

        source = get_unit(unit)
        return cls(source.to_si(value), source.dimension)

    @property
    def si_unit(self) -> str:
        """Canonical SI symbol for this quantity's semantic dimension."""

        return _SI_SYMBOLS[self.dimension]

    def to(
        self,
        unit: UnitLike,
        *,
        tissue_density_kg_m3: Optional[Numeric] = None,
    ) -> float:
        """Return the value in ``unit`` after a dimension-safe conversion.

        Volumetric and mass-specific perfusion are the only distinct
        dimensions that may be bridged.  Their conversion requires an explicit
        tissue bulk density in kg/m^3.
        """

        target = get_unit(unit)
        if target.dimension is self.dimension:
            return target.from_si(self.si_value)

        if (
            self.dimension is Dimension.MASS_SPECIFIC_PERFUSION
            and target.dimension is Dimension.VOLUMETRIC_PERFUSION
        ):
            value_si = self.si_value * _density(tissue_density_kg_m3)
            return target.from_si(value_si)

        if (
            self.dimension is Dimension.VOLUMETRIC_PERFUSION
            and target.dimension is Dimension.MASS_SPECIFIC_PERFUSION
        ):
            value_si = self.si_value / _density(tissue_density_kg_m3)
            return target.from_si(value_si)

        raise DimensionError(
            f"cannot convert {self.dimension.value} to {target.dimension.value}"
        )

    def require(self, dimension: Dimension) -> float:
        """Return SI only if ``dimension`` matches, for typed solver handoff."""

        if not isinstance(dimension, Dimension):
            raise TypeError("dimension must be a Dimension")
        if self.dimension is not dimension:
            raise DimensionError(
                f"expected {dimension.value}, got {self.dimension.value}"
            )
        return self.si_value

    def format(self, unit: UnitLike, precision: int = 6) -> str:
        """Format an explicitly converted value with its canonical symbol."""

        if isinstance(precision, bool) or not isinstance(precision, int):
            raise TypeError("precision must be an integer")
        if precision < 0:
            raise ValueError("precision must be non-negative")
        target = get_unit(unit)
        return f"{self.to(target):.{precision}g} {target.symbol}"

    def _same_dimension(self, other: Quantity) -> None:
        if self.dimension is not other.dimension:
            raise DimensionError(
                f"cannot combine {self.dimension.value} with {other.dimension.value}"
            )

    def __add__(self, other: object) -> Quantity:
        if not isinstance(other, Quantity):
            return NotImplemented

        if (
            self.dimension is Dimension.ABSOLUTE_TEMPERATURE
            and other.dimension is Dimension.TEMPERATURE_DIFFERENCE
        ):
            return Quantity(
                self.si_value + other.si_value, Dimension.ABSOLUTE_TEMPERATURE
            )
        if (
            self.dimension is Dimension.TEMPERATURE_DIFFERENCE
            and other.dimension is Dimension.ABSOLUTE_TEMPERATURE
        ):
            return other + self
        self._same_dimension(other)
        if self.dimension is Dimension.ABSOLUTE_TEMPERATURE:
            raise DimensionError(
                "adding two absolute temperatures is undefined; add a "
                "temperature_difference instead"
            )
        return Quantity(self.si_value + other.si_value, self.dimension)

    def __sub__(self, other: object) -> Quantity:
        if not isinstance(other, Quantity):
            return NotImplemented

        if (
            self.dimension is Dimension.ABSOLUTE_TEMPERATURE
            and other.dimension is Dimension.ABSOLUTE_TEMPERATURE
        ):
            return Quantity(
                self.si_value - other.si_value,
                Dimension.TEMPERATURE_DIFFERENCE,
            )
        if (
            self.dimension is Dimension.ABSOLUTE_TEMPERATURE
            and other.dimension is Dimension.TEMPERATURE_DIFFERENCE
        ):
            return Quantity(
                self.si_value - other.si_value, Dimension.ABSOLUTE_TEMPERATURE
            )
        if (
            self.dimension is Dimension.TEMPERATURE_DIFFERENCE
            and other.dimension is Dimension.ABSOLUTE_TEMPERATURE
        ):
            raise DimensionError(
                "an absolute temperature cannot be subtracted from a "
                "temperature difference"
            )
        self._same_dimension(other)
        return Quantity(self.si_value - other.si_value, self.dimension)

    def __mul__(self, scalar: object) -> Quantity:
        if isinstance(scalar, Quantity):
            raise DimensionError(
                "quantity-by-quantity multiplication is outside the supported "
                "API; convert the derived quantity explicitly"
            )
        value = _finite_number(scalar, "scalar")
        return Quantity(self.si_value * value, self.dimension)

    def __rmul__(self, scalar: object) -> Quantity:
        return self * scalar

    def __truediv__(self, divisor: object) -> Union[Quantity, float]:
        if isinstance(divisor, Quantity):
            self._same_dimension(divisor)
            if divisor.si_value == 0.0:
                raise ZeroDivisionError("cannot divide by a zero quantity")
            return self.si_value / divisor.si_value
        value = _finite_number(divisor, "divisor")
        if value == 0.0:
            raise ZeroDivisionError("cannot divide a quantity by zero")
        return Quantity(self.si_value / value, self.dimension)

    def _ordered_quantity(self, other: object) -> Quantity:
        if not isinstance(other, Quantity):
            raise TypeError("quantities can only be ordered against another Quantity")
        return other

    def __lt__(self, other: object) -> bool:
        other = self._ordered_quantity(other)
        self._same_dimension(other)
        return self.si_value < other.si_value

    def __le__(self, other: object) -> bool:
        other = self._ordered_quantity(other)
        self._same_dimension(other)
        return self.si_value <= other.si_value

    def __gt__(self, other: object) -> bool:
        other = self._ordered_quantity(other)
        self._same_dimension(other)
        return self.si_value > other.si_value

    def __ge__(self, other: object) -> bool:
        other = self._ordered_quantity(other)
        self._same_dimension(other)
        return self.si_value >= other.si_value

    def __str__(self) -> str:
        return f"{self.si_value:g} {self.si_unit}"


def quantity(value: Numeric, unit: UnitLike) -> Quantity:
    """Create a validated quantity from an explicit unit."""

    return Quantity.from_unit(value, unit)


def convert(
    value: Numeric,
    from_unit: UnitLike,
    to_unit: UnitLike,
    *,
    tissue_density_kg_m3: Optional[Numeric] = None,
) -> float:
    """Convert one finite scalar, rejecting incompatible dimensions."""

    return quantity(value, from_unit).to(
        to_unit, tissue_density_kg_m3=tissue_density_kg_m3
    )


def _typed_quantity(
    value: Numeric,
    unit: UnitLike,
    expected: Union[Dimension, Tuple[Dimension, ...]],
    label: str,
) -> Quantity:
    result = quantity(value, unit)
    expected_dimensions = expected if isinstance(expected, tuple) else (expected,)
    if result.dimension not in expected_dimensions:
        names = " or ".join(item.value for item in expected_dimensions)
        raise DimensionError(f"{label} requires {names}, got {result.dimension.value}")
    return result


def temperature(value: Numeric, unit: UnitLike = "K") -> Quantity:
    """Create an absolute temperature; values below 0 K are rejected."""

    return _typed_quantity(value, unit, Dimension.ABSOLUTE_TEMPERATURE, "temperature")


def temperature_difference(value: Numeric, unit: UnitLike = "delta_K") -> Quantity:
    """Create a signed temperature interval, distinct from temperature."""

    return _typed_quantity(
        value, unit, Dimension.TEMPERATURE_DIFFERENCE, "temperature_difference"
    )


def pressure(value: Numeric, unit: UnitLike = "Pa") -> Quantity:
    """Create pressure; gauge-versus-absolute reference remains caller-owned."""

    return _typed_quantity(value, unit, Dimension.PRESSURE, "pressure")


def concentration(value: Numeric, unit: UnitLike = "mol/m^3") -> Quantity:
    """Create a non-negative molar concentration."""

    return _typed_quantity(value, unit, Dimension.MOLAR_CONCENTRATION, "concentration")


def diffusivity(value: Numeric, unit: UnitLike = "m^2/s") -> Quantity:
    """Create a non-negative molecular or effective diffusivity."""

    return _typed_quantity(value, unit, Dimension.DIFFUSIVITY, "diffusivity")


def solute_permeability(value: Numeric, unit: UnitLike = "m/s") -> Quantity:
    """Create non-negative membrane/solute permeability with units L/T."""

    return _typed_quantity(
        value, unit, Dimension.SOLUTE_PERMEABILITY, "solute_permeability"
    )


def intrinsic_permeability(value: Numeric, unit: UnitLike = "m^2") -> Quantity:
    """Create non-negative porous-medium intrinsic permeability with units L^2."""

    return _typed_quantity(
        value,
        unit,
        Dimension.INTRINSIC_PERMEABILITY,
        "intrinsic_permeability",
    )


def darcy_mobility(value: Numeric, unit: UnitLike = "m^2/(Pa*s)") -> Quantity:
    """Create non-negative Darcy mobility K/mu, distinct from intrinsic K."""

    return _typed_quantity(value, unit, Dimension.DARCY_MOBILITY, "darcy_mobility")


def permeability(value: Numeric, unit: UnitLike) -> Quantity:
    """Create one explicit permeability kind, inferred from the unit dimension.

    Prefer the more specific constructors in reusable code.  This convenience
    accepts solute permeability (L/T), intrinsic permeability (L^2), or Darcy
    mobility (L^2/(Pa s)); those dimensions never convert into one another.
    """

    return _typed_quantity(
        value,
        unit,
        (
            Dimension.SOLUTE_PERMEABILITY,
            Dimension.INTRINSIC_PERMEABILITY,
            Dimension.DARCY_MOBILITY,
        ),
        "permeability",
    )


def perfusion_rate(
    value: Numeric,
    unit: UnitLike = "1/s",
    *,
    tissue_density_kg_m3: Optional[Numeric] = None,
) -> Quantity:
    """Create volumetric perfusion, converting mass-specific data if explicit.

    A mass-specific unit such as ``mL/(min*100g)`` requires tissue bulk density
    in kg/m^3.  The returned quantity always has dimension
    :attr:`Dimension.VOLUMETRIC_PERFUSION` and SI unit ``1/s``.
    """

    result = _typed_quantity(
        value,
        unit,
        (
            Dimension.VOLUMETRIC_PERFUSION,
            Dimension.MASS_SPECIFIC_PERFUSION,
        ),
        "perfusion_rate",
    )
    if result.dimension is Dimension.VOLUMETRIC_PERFUSION:
        return result
    return Quantity(
        result.si_value * _density(tissue_density_kg_m3),
        Dimension.VOLUMETRIC_PERFUSION,
    )


def mass_specific_perfusion(value: Numeric, unit: UnitLike = "m^3/(kg*s)") -> Quantity:
    """Create mass-specific perfusion without assuming a tissue density."""

    return _typed_quantity(
        value,
        unit,
        Dimension.MASS_SPECIFIC_PERFUSION,
        "mass_specific_perfusion",
    )


def length(value: Numeric, unit: UnitLike = "m") -> Quantity:
    """Create a non-negative length magnitude."""

    return _typed_quantity(value, unit, Dimension.LENGTH, "length")


def time(value: Numeric, unit: UnitLike = "s") -> Quantity:
    """Create a non-negative elapsed time."""

    return _typed_quantity(value, unit, Dimension.TIME, "time")


def amount(value: Numeric, unit: UnitLike = "mol") -> Quantity:
    """Create a non-negative amount of substance."""

    return _typed_quantity(value, unit, Dimension.AMOUNT, "amount")


def energy(value: Numeric, unit: UnitLike = "J") -> Quantity:
    """Create signed energy in joules internally."""

    return _typed_quantity(value, unit, Dimension.ENERGY, "energy")


__all__ = [
    "ConversionContextError",
    "Dimension",
    "DimensionError",
    "Quantity",
    "QuantityDomainError",
    "UNITS",
    "Unit",
    "UnitError",
    "UnknownUnitError",
    "amount",
    "available_units",
    "concentration",
    "convert",
    "darcy_mobility",
    "diffusivity",
    "energy",
    "get_unit",
    "intrinsic_permeability",
    "length",
    "mass_specific_perfusion",
    "perfusion_rate",
    "permeability",
    "pressure",
    "quantity",
    "solute_permeability",
    "temperature",
    "temperature_difference",
    "time",
]
