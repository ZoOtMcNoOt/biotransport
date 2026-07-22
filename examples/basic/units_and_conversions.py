"""Use explicit quantities at a BioTransport solver boundary.

The native solvers consume SI floats.  Quantities let an application accept
familiar laboratory or clinical units and make the SI handoff auditable.
"""

from biotransport import units


def main() -> None:
    body_temperature = units.temperature(37.0, "degC")
    sodium = units.concentration(140.0, "mM")
    sodium_diffusivity = units.diffusivity(1.33e-5, "cm^2/s")
    tumor_pressure = units.pressure(20.0, "mmHg")

    print("Explicit input conversions")
    print(f"  temperature: {body_temperature.format('K', 6)}")
    print(f"  sodium:      {sodium.format('mol/m^3', 6)}")
    print(f"  diffusivity: {sodium_diffusivity.format('m^2/s', 6)}")
    print(f"  pressure:    {tumor_pressure.format('Pa', 6)}")

    # ``require`` both checks the semantic dimension and returns the SI float
    # expected by a native solver constructor.
    temperature_k = body_temperature.require(units.Dimension.ABSOLUTE_TEMPERATURE)
    concentration_mol_m3 = sodium.require(units.Dimension.MOLAR_CONCENTRATION)
    diffusivity_m2_s = sodium_diffusivity.require(units.Dimension.DIFFUSIVITY)

    print("\nSolver-ready SI values")
    print(f"  T = {temperature_k:.6g} K")
    print(f"  c = {concentration_mol_m3:.6g} mol/m^3")
    print(f"  D = {diffusivity_m2_s:.6g} m^2/s")

    # Clinical perfusion data expressed per tissue mass cannot become the
    # volumetric 1/s coefficient in a Pennes term without tissue bulk density.
    perfusion = units.perfusion_rate(
        60.0,
        "mL/(min*100g)",
        tissue_density_kg_m3=1000.0,
    )
    print(f"  w = {perfusion.si_value:.6g} 1/s (density supplied explicitly)")


if __name__ == "__main__":
    main()
