# Runtime units at the Python boundary

BioTransport's C++ solvers operate on plain SI scalars for predictable native
performance. The `biotransport.units` module is a small, dependency-free Python
boundary layer: it accepts an explicitly named unit, stores one SI value plus a
semantic dimension, and refuses conversions or arithmetic that could silently
mix unrelated quantities.

This is deliberately not a symbolic units engine. It covers the quantities
most likely to be confused in the current APIs: absolute temperature,
temperature difference, pressure, molar concentration, diffusivity, three
different meanings of permeability, perfusion, length, time, amount of
substance, and energy.

The definitions and decimal-prefix conventions follow the
[BIPM SI Brochure, 9th edition](https://www.bipm.org/en/publications/si-brochure/)
([DOI 10.59161/AUEZ1291](https://doi.org/10.59161/AUEZ1291)). Non-SI units in
the registry have fixed conversions recorded below; accepting one says nothing
about whether it is an appropriate unit or parameter for a particular model.

## The basic pattern

```python
from biotransport import units

body_temperature = units.temperature(37.0, "degC")
sodium = units.concentration(140.0, "mM")
diffusion = units.diffusivity(1.33e-5, "cm^2/s")

assert body_temperature.to("K") == 310.15
assert sodium.to("mol/m^3") == 140.0
assert diffusion.to("m^2/s") == 1.33e-9
```

Every `Quantity` is immutable. Its `si_value` is always in the `si_unit`
reported by the object. Conversion is explicit: `q.to("unit")` returns a plain
number in the requested unit. `q.require(Dimension.DIFFUSIVITY)` both checks
the runtime dimension and returns the solver-ready SI scalar:

```python
D_m2_s = diffusion.require(units.Dimension.DIFFUSIVITY)
problem.diffusivity(D_m2_s)
```

Use `require` at reusable solver boundaries so that passing a length or
permeability instead of a diffusivity fails immediately. Direct access to
`si_value` is convenient when the dimension was just established by a named
constructor.

## Temperature is affine, not merely scaled

Absolute temperatures and temperature differences are separate dimensions.
This prevents two common mistakes: treating 20 degrees Celsius as 20 kelvin,
and applying the Celsius offset to a temperature interval.

```python
arterial = units.temperature(37.0, "degC")
drop = units.temperature_difference(5.0, "delta_degC")
cooled = arterial - drop

assert cooled.to("degC") == 32.0
assert (arterial - cooled).to("delta_K") == 5.0
```

Adding a temperature difference to an absolute temperature is supported.
Subtracting two absolute temperatures produces a difference. Adding two
absolute temperatures, converting an absolute temperature to `delta_degC`, or
constructing a value below 0 K raises an error.

Supported absolute units are `K`, `degC`, and `degF`. Supported interval units
are `delta_K`, `delta_degC`, and `delta_degF`.

## Pressure and concentration

Pressure units include `Pa`, `kPa`, `MPa`, `bar`, `mbar`, `atm`, `torr`,
`mmHg`, and `cmH2O`. The atmosphere is 101325 Pa, the torr is 1/760 atmosphere,
the conventional millimetre of mercury used here is 133.322387415 Pa, and the
conventional centimetre of water is 98.0665 Pa.

The dimension tag does not distinguish gauge pressure from absolute pressure.
That reference is part of the model contract and must still be stated by the
caller. Negative gauge pressure is therefore representable.

Concentration intentionally means **molar concentration**, not mass
concentration. Supported scales are `mol/m^3`, `mol/L` (alias `M`), `mM`, `uM`,
`nM`, and `pM`. A crucial identity is:

```text
1 mM = 1 mmol/L = 1 mol/m^3
```

Converting mass concentration to molar concentration requires a compound's
molar mass and is not guessed by this module. Non-negative molar concentration
is enforced.

## Permeability is not one dimension

Bioengineering uses “permeability” for several physically different
coefficients. The units layer keeps all three separate:

| Constructor | Meaning | SI unit |
|---|---|---|
| `solute_permeability` | membrane permeability or mass-transfer velocity | `m/s` |
| `intrinsic_permeability` | porous material intrinsic permeability | `m^2` |
| `darcy_mobility` | intrinsic permeability divided by dynamic viscosity | `m^2/(Pa*s)` |

`permeability(value, unit)` is a convenience that infers one of those semantic
dimensions from an unambiguous unit. No conversion between the three is
allowed. The more specific constructors are preferable in library code.

Diffusivity remains a fourth distinct dimension with SI unit `m^2/s`.
Supported convenience scales include `cm^2/s`, `mm^2/s`, and `um^2/s`.

## Perfusion requires a declared basis

The Pennes perfusion coefficient is commonly volumetric blood flow per tissue
volume and time, with dimension `1/s`. Clinical and experimental sources also
report blood volume per tissue **mass** and time. Those are not equivalent
without the tissue bulk density:

```text
volumetric perfusion [1/s]
    = mass-specific perfusion [m^3/(kg s)]
      * tissue bulk density [kg/m^3].
```

Accordingly, this ambiguous conversion fails unless density is explicit:

```python
from biotransport import units

# This raises ConversionContextError:
# units.perfusion_rate(60.0, "mL/(min*100g)")

w = units.perfusion_rate(
    60.0,
    "mL/(min*100g)",
    tissue_density_kg_m3=1000.0,
)
assert w.to("1/s") == 0.01
```

`mass_specific_perfusion` preserves the reported basis without assuming a
density. Calling its `to("1/s", tissue_density_kg_m3=...)` performs the same
explicit conversion. The reverse conversion also requires density. Density is
the wet bulk density of the tissue basis represented by the source data; it is
not automatically blood density.

Volumetric units are `1/s`, `1/min`, `1/h`, and `%/min`. Mass-specific units
are `m^3/(kg*s)`, `mL/(kg*min)`, and `mL/(100g*min)`, with equivalent reordered
aliases.

## Other supported quantities

| Dimension | Supported canonical units |
|---|---|
| length magnitude | `m`, `cm`, `mm`, `um`, `nm` |
| elapsed time | `s`, `ms`, `us`, `min`, `h`, `day` |
| amount of substance | `mol`, `mmol`, `umol`, `nmol`, `pmol` |
| energy | `J`, `kJ`, `mJ`, `uJ`, `cal`, `kcal` |

The calorie entries are thermochemical calories: 1 cal = 4.184 J exactly for
this registry. Energy may be signed because model energy changes can be signed;
length, elapsed time, amount, diffusivity, permeability, perfusion, and
concentration are non-negative magnitudes.

Unicode aliases such as `µm`, `µM`, and `m²/s` are accepted, while canonical
ASCII spellings make scripts and serialized configurations more portable.
`available_units()` lists canonical symbols, optionally filtered by a
`Dimension`.

## Failure behavior and scope

The layer rejects:

- non-finite values, including values that overflow during conversion;
- unknown unit spellings and a valid unit used with the wrong named factory;
- cross-dimension conversion, arithmetic, comparison, and ratios;
- quantity-by-quantity multiplication, because the resulting derived
  dimensions are outside this intentionally narrow API;
- physically negative magnitude quantities and temperatures below 0 K;
- mass-specific/volumetric perfusion conversion without a finite, positive
  `tissue_density_kg_m3`.

Multiplication or division by a finite scalar preserves the dimension. Dividing
two quantities of the same dimension returns their dimensionless ratio.

This layer checks units and basic physical domains; it does not establish
parameter provenance, uncertainty, model applicability, gauge reference,
activity corrections, or biological validity. Generic solvers that explicitly
allow any mutually consistent unit system remain caller-defined. For
quantitative BioTransport workflows, prefer SI at the solver boundary and
record the original measurement, conversion, material, temperature, and basis
alongside the model configuration.
