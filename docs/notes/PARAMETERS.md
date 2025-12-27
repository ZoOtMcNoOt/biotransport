# Multi-Physics Solver Parameters Reference

This document provides a comprehensive reference for parameters used in the advanced multi-physics solvers. For programmatic access to these parameters with validation and documentation, use the configuration dataclasses in `biotransport.config`.

## Quick Start

```python
from biotransport import TumorDrugDeliveryConfig, BioheatCryotherapyConfig

# Create config with defaults
tumor_config = TumorDrugDeliveryConfig()
print(tumor_config.describe())

# Create config with custom values
cryo_config = BioheatCryotherapyConfig(T_probe=-180.0, blood_perfusion_rate=0.8e-3)
print(cryo_config.describe())
```

---

## Tumor Drug Delivery Parameters

These parameters model drug transport in vascularized tumor tissue, accounting for:
- Heterogeneous diffusion (normal vs tumor tissue)
- Drug binding and cellular uptake
- Vascular extravasation (drug leakage from blood vessels)
- Interstitial fluid pressure effects

### Domain Geometry

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| `domain_size` | L | 0.01 | m | Side length of square domain |
| `tumor_radius` | r_t | 0.003 | m | Radius of central tumor region |

### Diffusion Coefficients

| Parameter | Symbol | Default | Units | Typical Range | Description |
|-----------|--------|---------|-------|---------------|-------------|
| `D_drug_normal` | D_n | 5×10⁻¹¹ | m²/s | 10⁻¹² – 10⁻¹⁰ | Drug diffusivity in normal tissue |
| `D_drug_tumor` | D_t | 1×10⁻¹¹ | m²/s | 10⁻¹² – 10⁻¹⁰ | Drug diffusivity in tumor tissue |

**Physical interpretation:** Tumor tissue typically has lower diffusivity due to:
- Elevated interstitial fluid pressure
- Dense extracellular matrix
- Collapsed blood vessels

### Reaction Kinetics

| Parameter | Symbol | Default | Units | Typical Range | Description |
|-----------|--------|---------|-------|---------------|-------------|
| `k_binding` | k_b | 1×10⁻³ | 1/s | 10⁻⁴ – 10⁻² | Drug-receptor binding rate |
| `k_uptake` | k_u | 5×10⁻⁴ | 1/s | 10⁻⁴ – 10⁻³ | Cellular drug internalization rate |
| `k_clearance` | k_c | 1×10⁻⁴ | 1/s | 10⁻⁵ – 10⁻³ | Systemic drug elimination rate |

**Reaction model:**
```
∂C/∂t = ∇·(D∇C) - k_binding·C - k_uptake·C - k_clearance·C + S_vessel
```

### Vascular Parameters

| Parameter | Symbol | Default | Units | Typical Range | Description |
|-----------|--------|---------|-------|---------------|-------------|
| `MVD_normal` | MVD_n | 100 | 1/mm² | 50 – 200 | Microvessel density in normal tissue |
| `MVD_tumor` | MVD_t | 200 | 1/mm² | 100 – 400 | Microvessel density in tumor |
| `P_vessel_normal` | P_n | 1×10⁻⁷ | m/s | 10⁻⁸ – 10⁻⁶ | Vessel wall permeability (normal) |
| `P_vessel_tumor` | P_t | 5×10⁻⁷ | m/s | 10⁻⁷ – 10⁻⁵ | Vessel wall permeability (tumor) |
| `vessel_radius` | r_v | 5×10⁻⁶ | m | 3 – 10 μm | Typical capillary radius |

**Note:** Tumor vessels are often "leaky" due to poor structural integrity, resulting in higher permeability.

### Interstitial Fluid Pressure (IFP)

| Parameter | Symbol | Default | Units | Typical Range | Description |
|-----------|--------|---------|-------|---------------|-------------|
| `IFP_normal` | p_n | 0.0 | mmHg | 0 – 3 | IFP in normal tissue |
| `IFP_tumor` | p_t | 20.0 | mmHg | 10 – 60 | IFP in tumor center |
| `K_hydraulic_normal` | K_n | 1×10⁻¹³ | m²/(Pa·s) | 10⁻¹⁴ – 10⁻¹² | Hydraulic conductivity (normal) |
| `K_hydraulic_tumor` | K_t | 5×10⁻¹⁴ | m²/(Pa·s) | 10⁻¹⁵ – 10⁻¹³ | Hydraulic conductivity (tumor) |

**Physical interpretation:** Elevated tumor IFP creates outward pressure gradients that:
- Reduce drug convection into tumor
- Wash drug toward periphery
- Create transport barriers

**Unit conversion:** Use `config.IFP_normal_Pa` and `config.IFP_tumor_Pa` for values in Pascals.

### Concentration Parameters

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| `C_plasma` | C_p | 1.0 | arbitrary | Plasma drug concentration |
| `C_initial` | C_0 | 0.0 | arbitrary | Initial tissue concentration |
| `C_boundary` | C_b | 1.0 | arbitrary | Boundary concentration (Dirichlet) |

### Simulation Control

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| `nx` | N_x | 64 | — | Grid points in x-direction |
| `ny` | N_y | 64 | — | Grid points in y-direction |
| `t_end` | T | 3600.0 | s | Simulation end time (1 hour) |
| `dt` | Δt | 1.0 | s | Time step size |

---

## Bioheat Cryotherapy Parameters

These parameters model the Pennes bioheat equation with cryotherapy (freezing) effects:
- Tissue heat conduction
- Blood perfusion (heating)
- Metabolic heat generation
- Cryoprobe cooling
- Phase change (freezing)
- Thermal damage (Arrhenius model)

### Domain Geometry

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| `domain_length` | L_x | 0.05 | m | Domain length (x-direction) |
| `domain_width` | L_y | 0.05 | m | Domain width (y-direction) |
| `probe_x` | x_p | 0.025 | m | Cryoprobe x-position |
| `probe_y` | y_p | 0.025 | m | Cryoprobe y-position |
| `probe_radius` | r_p | 0.002 | m | Cryoprobe radius |

### Tissue Thermal Properties

| Parameter | Symbol | Default | Units | Typical Range | Description |
|-----------|--------|---------|-------|---------------|-------------|
| `rho_tissue` | ρ | 1050 | kg/m³ | 1000 – 1100 | Tissue density |
| `c_tissue` | c | 3600 | J/(kg·K) | 3200 – 4000 | Specific heat capacity |
| `k_tissue_unfrozen` | k_u | 0.5 | W/(m·K) | 0.4 – 0.6 | Thermal conductivity (unfrozen) |
| `k_tissue_frozen` | k_f | 2.0 | W/(m·K) | 1.5 – 2.5 | Thermal conductivity (frozen) |

**Physical interpretation:** Frozen tissue has ~4× higher thermal conductivity due to ice crystal formation.

**Derived quantity:** Use `config.thermal_diffusivity` for α = k/(ρc).

### Blood Perfusion

| Parameter | Symbol | Default | Units | Typical Range | Description |
|-----------|--------|---------|-------|---------------|-------------|
| `blood_perfusion_rate` | ω_b | 0.5×10⁻³ | 1/s | 0.1 – 2.0 ×10⁻³ | Blood perfusion rate |
| `rho_blood` | ρ_b | 1060 | kg/m³ | 1050 – 1070 | Blood density |
| `c_blood` | c_b | 3770 | J/(kg·K) | 3700 – 3900 | Blood specific heat |
| `T_arterial` | T_a | 37.0 | °C | 36 – 38 | Arterial blood temperature |

**Bioheat equation:**
```
ρc ∂T/∂t = ∇·(k∇T) + ω_b·ρ_b·c_b·(T_a - T) + Q_met
```

### Metabolic Heat

| Parameter | Symbol | Default | Units | Typical Range | Description |
|-----------|--------|---------|-------|---------------|-------------|
| `Q_metabolic` | Q_met | 420.0 | W/m³ | 200 – 1000 | Metabolic heat generation rate |

**Note:** Metabolic heat is typically small compared to perfusion and conduction in cryotherapy.

### Cryoprobe Parameters

| Parameter | Symbol | Default | Units | Typical Range | Description |
|-----------|--------|---------|-------|---------------|-------------|
| `T_probe` | T_p | -150.0 | °C | -196 to -40 | Cryoprobe temperature |
| `cooling_rate` | dT/dt | 50.0 | K/min | 10 – 100 | Probe cooling rate |

**Clinical context:**
- Argon-based cryoprobes: -140°C to -160°C
- Liquid nitrogen: -196°C
- Faster cooling → smaller ice crystals → more cell damage

**Unit access:** Use `config.T_probe_celsius` and `config.T_probe_kelvin`.

### Phase Change Parameters

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| `T_freeze_start` | T_fs | -0.5 | °C | Freezing onset temperature |
| `T_freeze_end` | T_fe | -8.0 | °C | Freezing completion temperature |
| `latent_heat` | L_f | 334000 | J/kg | Latent heat of fusion (water/ice) |

**Phase change model:** The mushy zone (T_fs to T_fe) uses an enthalpy method with apparent specific heat:
```
c_apparent = c_tissue + L_f / (T_fs - T_fe)  for T_fe < T < T_fs
```

### Thermal Damage (Arrhenius Model)

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|---------------|
| `A_damage` | A | 7.39×10³⁹ | 1/s | Arrhenius frequency factor |
| `E_a_damage` | E_a | 2.577×10⁵ | J/mol | Activation energy |

**Damage integral:**
```
Ω(t) = ∫₀ᵗ A·exp(-E_a/(R·T)) dt
```

**Interpretation:**
- Ω = 1 corresponds to 63% cell death probability
- Ω = 4.6 corresponds to 99% cell death

**Helper methods:**
```python
config.damage_rate(T_celsius)      # Returns damage rate at temperature T
config.death_probability(omega)    # Returns probability from damage integral
```

### Initial and Boundary Conditions

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| `T_initial` | T_0 | 37.0 | °C | Initial tissue temperature |
| `T_boundary` | T_b | 37.0 | °C | Boundary temperature (Dirichlet) |

### Simulation Control

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| `nx` | N_x | 64 | — | Grid points in x-direction |
| `ny` | N_y | 64 | — | Grid points in y-direction |
| `t_end` | T | 180.0 | s | Simulation end time (3 minutes) |
| `dt` | Δt | 0.1 | s | Time step size |

---

## Parameter Ranges Reference

The `get_parameter_ranges()` function returns typical literature values:

```python
from biotransport import get_parameter_ranges

ranges = get_parameter_ranges()

# Example: check diffusivity range
d_range = ranges['diffusion_coefficients']['drug_in_tissue']
print(f"Drug diffusivity: {d_range['min']:.2e} – {d_range['max']:.2e} {d_range['unit']}")
```

### Diffusion Coefficients

| Quantity | Min | Max | Unit | Reference |
|----------|-----|-----|------|-----------|
| Drug in tissue | 10⁻¹² | 10⁻¹⁰ | m²/s | Jain (1987) |
| Oxygen in tissue | 1.5×10⁻⁹ | 3×10⁻⁹ | m²/s | Krogh (1919) |
| Glucose in tissue | 2×10⁻¹⁰ | 5×10⁻¹⁰ | m²/s | Casciari (1988) |

### Thermal Properties

| Quantity | Min | Max | Unit | Reference |
|----------|-----|-----|------|-----------|
| Tissue conductivity | 0.3 | 0.6 | W/(m·K) | Duck (1990) |
| Tissue specific heat | 3200 | 4000 | J/(kg·K) | Duck (1990) |
| Blood perfusion | 10⁻⁴ | 2×10⁻³ | 1/s | Pennes (1948) |

### Vascular Parameters

| Quantity | Min | Max | Unit | Reference |
|----------|-----|-----|------|-----------|
| MVD normal | 50 | 200 | 1/mm² | Folkman (1992) |
| MVD tumor | 100 | 500 | 1/mm² | Folkman (1992) |
| Vessel permeability | 10⁻⁸ | 10⁻⁵ | m/s | Michel (1999) |
| IFP tumor | 5 | 60 | mmHg | Jain (1994) |

---

## See Also

- [examples/advanced/tumor_drug_delivery.py](../examples/advanced/tumor_drug_delivery.py) — Full tumor drug delivery simulation
- [examples/advanced/bioheat_cryotherapy.py](../examples/advanced/bioheat_cryotherapy.py) — Cryotherapy with phase change
- [BMEN341_BioTransport_Analysis.md](./BMEN341_BioTransport_Analysis.md) — Course context and learning objectives
