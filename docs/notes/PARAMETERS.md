# Scientific parameters and API conventions

This is the implementation-facing parameter reference for BioTransport's
specialized models. It records what each input means, the unit expected by the
current code, the mathematical domain enforced by validation, and the main
limits on interpretation.

It is **not** a table of universally valid physiological values. Defaults are
small, runnable examples. A value accepted by the API is only mathematically
admissible; it is not thereby calibrated for a tissue, solute, device, animal,
or patient. For governing-equation provenance and primary sources, see
[Model scope, provenance, and reference equations](MODEL_SCOPE_AND_REFERENCES.md).

## Rules that apply everywhere

- Solver-facing quantities use SI unless a name or this document states
  otherwise.
- A temperature passed into a C++ solver is an absolute temperature in kelvin.
- A mesh's `nx` and `ny` are cell counts. Node-centred fields therefore contain
  `(nx + 1) * (ny + 1)` values in 2D. The Gray--Scott solver is the explicit
  exception: it stores one periodic, cell-centred value per cell and expects
  `nx * ny` values.
- Concentration units may be chosen by the caller only when the whole model is
  linear in concentration. Nernst--Planck current and charge diagnostics assume
  molar concentration in `mol/m^3` because they multiply by Faraday's constant.
- `dt` must satisfy both the numerical operator's reported limit and any
  stricter state-dependent reaction or prescribed-outflow limit.
- Record the origin, uncertainty, temperature, material, preparation, and unit
  conversion for every parameter used in a scientific result. Always report
  grid/time convergence and the available balance residuals.

## Bioheat cryotherapy

### Implemented model

`BioheatCryotherapySolver` advances the Pennes-type balance

```text
rho_t c_app(T) dT/dt = div(k(T) grad(T))
                       + rho_b c_b w_b f_liquid(T) (T_arterial - T)
                       + f_liquid(T) q_met
```

The apparent mass-specific heat capacity is

```text
c_app = f_liquid c_unfrozen + f_frozen c_frozen
        + L_fusion (-d f_frozen/dT).
```

The phase fraction is a smooth Gaussian transition. `T_freeze_range_K` is its
**two-standard-deviation width**, so `freeze_sigma_K` is half that value. The
perfusion and metabolic terms are multiplied by the liquid fraction.

### Validated Python configuration

`BioheatCryotherapyConfig` is the preferred way to prepare this solver. Its
current defaults are:

| Field | Default | Unit | Meaning |
|---|---:|---|---|
| `domain_size_x`, `domain_size_y` | `0.05`, `0.05` | m | Rectangular domain lengths |
| `nx`, `ny` | `100`, `100` | cells | Mesh cell counts |
| `rho_tissue` | `1050` | kg/m^3 | Tissue density |
| `c_tissue_unfrozen` | `3600` | J/(kg K) | Unfrozen sensible specific heat |
| `c_tissue_frozen` | `1800` | J/(kg K) | Frozen sensible specific heat |
| `k_tissue_unfrozen` | `0.5` | W/(m K) | Unfrozen thermal conductivity |
| `k_tissue_frozen` | `2.0` | W/(m K) | Frozen thermal conductivity |
| `rho_blood` | `1060` | kg/m^3 | Blood density in the perfusion source |
| `c_blood` | `3800` | J/(kg K) | Blood specific heat |
| `w_b_normal` | `5e-4` | m^3_blood/(m^3_tissue s), numerically 1/s | Normal-tissue volumetric perfusion |
| `w_b_tumor` | `2e-3` | same | Tumor-region volumetric perfusion |
| `T_probe_K` | `123.15` | K | Fixed temperature of probe-mask nodes |
| `probe_radius` | `1.5e-3` | m | Radius used when constructing a probe mask |
| `probe_position` | domain centre | m | `(x, y)` centre used by examples |
| `q_met_normal` | `420` | W/m^3 | Normal-tissue metabolic source |
| `q_met_tumor` | `840` | W/m^3 | Tumor-region metabolic source |
| `T_freeze_K` | `272.15` | K | Centre of the phase transition |
| `T_freeze_range_K` | `2.0` | K | Two-sigma transition width |
| `L_fusion` | `333000` | J/kg | Latent heat in the apparent-capacity term |
| `E_activation` | `2e5` | J/mol | Arrhenius heat-injury activation energy |
| `A_frequency` | `7.39e29` | 1/s | Arrhenius heat-injury frequency factor |
| `R_gas` | `8.31446261815324` | J/(mol K) | Gas constant used by the diagnostic |
| `T_initial_K` | `310.15` | K | Initial tissue temperature |
| `T_arterial_K` | `310.15` | K | Arterial temperature in the Pennes term |
| `T_boundary_K` | `310.15` | K | Fixed outer-boundary temperature |
| `tumor_radius` | `0.01` | m | Radius used to construct example property maps |
| `tumor_center` | domain centre | m | `(x, y)` centre used by examples |
| `dt` | `0.05` | s | Maximum explicit step |

All density, heat-capacity, conductivity, geometry, absolute-temperature,
transition-width, gas-constant, and time-step fields must be finite and
positive. Perfusion, metabolic heat, latent heat, and Arrhenius coefficients
must be finite and non-negative. `nx` and `ny` must be integers at least two;
positions must lie inside the domain. The config also requires
`T_probe_K < T_freeze_K < T_arterial_K` and rejects `dt` above its conservative
explicit bound.

The canonical temperature names contain `_K`. For explicit Celsius input use
the conversion constructor; ambiguous legacy names such as `T_probe` are not
accepted:

```python
import biotransport as bt

cfg = bt.BioheatCryotherapyConfig.from_celsius(
    probe_C=-150.0,
    freeze_C=-1.0,
    initial_C=37.0,
    arterial_C=37.0,
    boundary_C=37.0,
)
print(cfg.T_probe_K, cfg.maximum_stable_dt_s)
```

`create_solver(mesh, probe_mask=..., perfusion_map=..., q_met_map=...)`
validates the mesh against the config and configures separate initial,
arterial, and boundary temperatures. The maps and mask must be node-centred,
row-major arrays with `mesh.num_nodes()` entries.

### Results and interpretation limits

`BioheatSaved` reports exact `times_s`, `temperature_K()`,
`frozen_fraction()`, `damage()`, frame-wise temperature extrema, and the
conservative `maximum_stable_dt_s`. `damage` is the integral of
`A exp(-E_a/(R T))`; it is an Arrhenius **heat-injury diagnostic**, not a
validated cryogenic cell-death law. The helper
`arrhenius_injury_probability(Omega)` only evaluates `1 - exp(-Omega)`; it does
not validate the coefficients or biological interpretation.

Probe-mask nodes are embedded Dirichlet nodes. The model has no probe heat
capacity, coolant dynamics, contact resistance, conjugate heat transfer, ice
mechanics, vascular geometry, or calibrated cryoinjury response. Extending
Pennes perfusion and metabolism into a freezing transition is phenomenological
and requires application-specific validation.

## Tumor drug delivery

### Implemented model

`TumorDrugDeliverySolver` first solves a prescribed-pressure Darcy surrogate

```text
div(K grad(p)) = 0
v = -K grad(p)
```

and then advances three compartments:

```text
dC_free/dt = -div(v C_free - D grad(C_free))
              + P S_v (C_plasma - C_free)
              - (k_binding + k_uptake) C_free
dC_bound/dt = k_binding C_free
dC_cellular/dt = k_uptake C_free.
```

`bound` is irreversible tissue sequestration, not reversible receptor binding.
The tissue starts drug-free. There is no `k_clearance` parameter in this model.

### Validated Python configuration

`TumorDrugDeliveryConfig` supplies geometry and map-building values:

| Field | Default | Unit | Meaning |
|---|---:|---|---|
| `domain_size` | `5e-3` | m | Side length of the square domain |
| `tumor_radius` | `2e-3` | m | Radius of the pressure-clamped tumor mask |
| `tumor_center` | domain centre | m | Mask centre |
| `rim_thickness` | `0.5e-3` | m | Width of the tumor-rim property region |
| `nx`, `ny` | `100`, `100` | cells | Mesh cell counts |
| `D_drug_normal` | `5e-11` | m^2/s | Effective free-drug diffusivity outside tumor |
| `D_drug_tumor` | `2e-11` | m^2/s | Effective free-drug diffusivity in tumor |
| `k_binding` | `1e-3` | 1/s | Irreversible free-to-bound transfer rate |
| `k_uptake` | `5e-4` | 1/s | Irreversible free-to-cellular transfer rate |
| `MVD_normal` | `100` | vessels/mm^2 | Normal-tissue microvessel profile density |
| `MVD_tumor_core` | `20` | vessels/mm^2 | Tumor-core profile density |
| `MVD_tumor_rim` | `200` | vessels/mm^2 | Tumor-rim profile density |
| `vessel_radius` | `5e-6` | m | Radius assumed by the MVD-to-area closure |
| `P_vessel_normal` | `1e-7` | m/s | Normal vessel-wall solute permeability |
| `P_vessel_tumor` | `5e-7` | m/s | Tumor vessel-wall solute permeability |
| `C_plasma` | `1.0` | chosen concentration unit | Constant prescribed plasma concentration |
| `IFP_normal` | `0.0` | mmHg | Prescribed outer-boundary gauge pressure |
| `IFP_tumor` | `20.0` | mmHg | Prescribed tumor-mask gauge pressure |
| `K_hydraulic_normal` | `5e-12` | m^2/(Pa s) | Normal-tissue Darcy mobility |
| `K_hydraulic_tumor` | `2.5e-12` | m^2/(Pa s) | Tumor Darcy mobility |

The config converts its user-facing pressures through `IFP_normal_Pa` and
`IFP_tumor_Pa`; the C++ solver accepts pascals. `IFP_tumor` must be at least
`IFP_normal`, because the transport API has no exterior concentration with
which to define boundary inflow. The tumor must lie strictly inside the outer
boundary. Diffusivities, rates, MVD values, permeabilities, and plasma
concentration may be zero but not negative; both Darcy mobilities and the
assumed vessel radius must be positive.

The vascular source requires **perfused vessel surface area per tissue volume**
`S_v` in `1/m`, not a normalized vessel-count map. The config's convenience
closure is

```text
S_v = 2 pi vessel_radius MVD 1e6.
```

It assumes that the profile density can be interpreted as vessel length per
tissue volume after conversion. Use measured `S_v` directly when that
stereological assumption is unsuitable. The Python call uses the explicit
names `vessel_wall_solute_permeability` and
`vascular_surface_area_density`.

`solve_pressure_sor(max_iter=20000, tol=1e-10, omega=1.8)` treats `tol` as an
absolute discrete pressure-defect tolerance in pascals and requires
`0 < omega < 2`. `simulate(...)` accepts a maximum explicit `dt`, a step count,
and exact `times_to_save_s`; it reports `stability_limit_s` and rejects a step
that would violate the monotonicity/positivity bound.

### Results and interpretation limits

`TumorDrugDeliverySaved` returns free, bound, cellular, and total fields. Its
integrals are amount per unit out-of-plane depth when concentration is
volumetric. It also reports cumulative net vascular exchange, cumulative
Darcy-boundary outflow, and

```text
mass_balance_error = total - initial - vascular_exchange + boundary_outflow.
```

The tumor mask clamps pressure; it is not merely a material label. The implied
fluid source at that clamp is unresolved and treated as solute-free. The model
does not solve Starling filtration, solvent drag, lymphatic drainage, unbinding,
saturable binding, metabolism, systemic pharmacokinetics, or a time-varying
plasma concentration. A prescribed IFP field must come from measurement, a
separate model, or an explicit stated assumption.

## Nernst--Planck electrochemical transport

### Units and sign convention

For each ideal dilute ion,

```text
N = -D [grad(c) + z c grad(phi) / V_T]
V_T = R T / F
dc/dt = -div(N).
```

Use `c` in `mol/m^3`, `D` in `m^2/s`, absolute `T` in K, potential `phi` in V,
electric field in `V/m`, and time in s. `set_uniform_field(Ex, Ey)` uses
`E = -grad(phi)`. `IonSpecies(name, valence, diffusivity, temperature=310.0)`
requires a nonempty name, a nonzero integer valence, positive diffusivity, and
positive temperature. Its stored `mobility` is a magnitude in `m^2/(V s)` at
`mobility_temperature`; a solver created at another temperature evaluates its
own mobility using that solver temperature.

`set_dirichlet_boundary` takes a non-negative molar concentration.
`set_neumann_boundary` takes the **outward total molar flux** `N dot n` in
`mol/(m^2 s)`; positive values leave the domain. The default on every boundary
is zero total molar flux. This differs deliberately from the generic
multispecies solver, where Neumann data mean `du/dn`.

The explicit Scharfetter--Gummel update exposes
`maximum_stable_time_step()` and `recommended_time_step(safety=0.9)`. The
former is the homogeneous fitted-operator positivity bound. A specified
outward flux can impose a smaller concentration-dependent step, which is
checked during `solve`.

`compute_current_density()` returns interleaved Cartesian components in
`A/m^2` (`Jx, Jy` for each node) when concentration is molar.

### Convenience ion data

The built-in ion constructors use these approximate aqueous,
infinite-dilution diffusivities:

| Helper | Valence | Diffusivity (m^2/s) |
|---|---:|---:|
| `ions.sodium()` | `+1` | `1.33e-9` |
| `ions.potassium()` | `+1` | `1.96e-9` |
| `ions.chloride()` | `-1` | `2.03e-9` |
| `ions.calcium()` | `+2` | `0.79e-9` |
| `ions.magnesium()` | `+2` | `0.71e-9` |
| `ions.hydrogen()` | `+1` | `9.31e-9` |
| `ions.hydroxide()` | `-1` | `5.27e-9` |
| `ions.bicarbonate()` | `-1` | `1.18e-9` |

These are convenience values, not physiological tissue coefficients. Supply a
coefficient measured or corrected for the medium and temperature in a
quantitative study.

`MultiIonSolver` advances species independently in the same prescribed
potential. `charge_density()` computes `F sum(z_i c_i)` in `C/m^3` when
concentrations are molar. It does not feed that charge back into the potential.
Enabling `set_electroneutrality_mode` raises because electroneutral coupling is
not implemented.

The `ghk.nernst_potential` and `ghk.ghk_voltage` helpers require positive
absolute temperature and valid positive logarithm arguments. GHK
permeabilities may use any common relative or physical unit because only their
weighted ratio enters the voltage.

This implementation does not solve Poisson's equation, electroneutrality,
activity corrections, finite ion size, reactions, fluid coupling, membrane
gating, or action potentials.

## Membrane diffusion

`MembraneDiffusion1DSolver` is a steady, homogeneous, one-dimensional model:

```text
j = D_eff Phi (C_left - C_right) / L
P = D_eff Phi / L.
```

The concentration unit may be molar, mass-based, or another amount density as
long as both boundaries use the same amount per cubic metre. Flux has the
corresponding amount per square metre per second.

| Input | Default | Domain | Unit |
|---|---:|---|---|
| membrane thickness `L` | `100e-6` | finite, `> 0` | m |
| membrane diffusivity `D` | `1e-10` | finite, `> 0` | m^2/s |
| partition coefficient `Phi` | `1.0` | finite, `> 0` | dimensionless |
| left concentration | `1.0` | finite, `>= 0` | amount/m^3 |
| right concentration | `0.0` | finite, `>= 0` | amount/m^3 |
| profile nodes | `101` | integer, `>= 2` | count |

`Phi = C_membrane/C_solution` is the equilibrium partition ratio at both
interfaces. `set_hindered_diffusion(solute_radius, pore_radius)` requires
`0 <= solute_radius < pore_radius`; both radii use the same length unit. It
multiplies `D` by the implemented Renkin factor for
`lambda = solute_radius/pore_radius`. Treat that correlation as a specific pore
model, not a universal biological-barrier correction.

In `MembraneDiffusionResult`, `effective_diffusivity` is the equivalent
external-gradient coefficient `permeability * L`. It includes both partition
and hindrance; it is not simply the diffusivity inside the membrane or pore.

For `MultiLayerMembraneSolver`, every layer has positive thickness,
diffusivity, and partition coefficient. The resistance is
`sum(L_i/(D_i Phi_i))`, where each `Phi_i` refers to one common solution-phase
concentration/activity coordinate. Different partition coefficients create
one-sided concentration jumps; the result intentionally includes both values
at an interface position.

These models omit transient storage, external films, reactions, swelling,
active transport, electrochemical migration, solvent drag, and porosity or
tortuosity beyond the supplied effective diffusivity and Renkin factor.

## Generic multispecies reaction--diffusion

`MultiSpeciesSolver` advances

```text
du_i/dt = D_i laplacian(u_i) + R_i(u, x, y, t)
```

on a node-centred 1D or 2D mesh. Units are caller-defined but must be mutually
consistent: mesh length `L`, model time `T`, `D_i` in `L^2/T`, `u_i` in one
declared concentration unit, and reaction rates in concentration/`T`.
Diffusivities and concentrations must be finite and non-negative.

The default boundary is `du/dn = 0`. A multispecies Neumann value is the
outward-normal derivative `du/dn`; the outward Fickian flux is `-D du/dn`.
Dirichlet concentrations must be non-negative. The forward-Euler
`max_stable_time_step()` is the exact **diffusion-only** CFL ceiling. A reaction
can impose a smaller state-dependent positivity limit; the solver rejects the
whole candidate step if it creates a material negative or non-finite value.
Arbitrary Python callbacks are evaluated serially; built-in immutable reaction
models may be parallelized.

For maximum diffusivity `D_max`, the ceiling is
`1 / (2 D_max sum(1/h_i^2))`. Python `solve_until(final_time, maximum_dt)`
(C++ `solveUntil`) equal-subdivides the remaining interval using the smaller
of the user ceiling and diffusion CFL ceiling and reaches the requested
absolute time; it does not silently adapt to reaction kinetics. Python
`total_mass()` (C++ `totalMass`) is the trapezoidal integral implied by the
half/quarter boundary control volumes.

Built-in reaction constructors enforce mathematical domains, not biological
validity:

| Model | Parameters and enforced domain | Unit convention |
|---|---|---|
| `LotkaVolterraReaction` | `alpha`, `beta`, `gamma`, `delta >= 0`; `carrying_capacity > 0` | `alpha`, `gamma`: 1/T; `beta`, `delta`: inverse concentration/time; capacity: prey concentration |
| `SIRReaction` | `beta`, `gamma >= 0`; `total_population > 0` | rates in 1/T; `N` uses the same local-density unit as S, I, R |
| `SEIRReaction` | `beta`, `sigma`, `gamma >= 0`; `total_population > 0` | rates in 1/T; same local-density convention |
| `EnzymeCascadeReaction` | `vmax >= 0`, `km > 0`, `kdeg >= 0`; vector sizes must match | `vmax`: target concentration/T; `km`: upstream concentration; `kdeg`: 1/T |
| `CompetitiveInhibitionReaction` | `vmax >= 0`, `km > 0`, `ki > 0`, inhibitor decay `>= 0` | concentrations and time must be consistent |
| `BrusselatorReaction` | `A > 0`, `B > 0` | conventional nondimensional model |

The SIR/SEIR classes are reaction equations on local fields; they are not a
validated epidemic model, mobility model, or population inference system.

### Gray--Scott specialization

`GrayScottSolver(mesh, Du, Dv, f, k)` is a two-dimensional, periodic,
single-precision pattern solver. Its `u` and `v` are dimensionless; `Du` and
`Dv` use mesh-length squared per model-time, while `f` and `k` use inverse
model-time. Inputs must be non-negative and representable in `float`. Initial
fields contain `mesh.nx() * mesh.ny()` values in row-major `[j][i]` order, with
no duplicated periodic endpoint.

`simulate(u0, v0, total_steps, dt, steps_between_frames=1000,
check_interval=1000, stable_tol=1e-4,
min_frames_before_early_stop=6)` checks a state-dependent diffusion/reaction
positivity limit on every step. `stable_tol` is the maximum absolute field
change since the prior `check_interval`; zero disables early termination.
Initial and terminal states are always saved. Result arrays are packed as
`[frame][ny][nx]`. `final_time` is
`steps_run * dt`, which can be earlier than `total_steps * dt` when the early
stop condition is met. Published-looking spots or stripes are not evidence of
a biological mechanism without an explicit dimensional mapping and validation.

## Generalized-Newtonian rheology

All rheology classes compute an instantaneous apparent viscosity from shear
rate. Shear rate is in `1/s`, viscosity in `Pa s`, and shear stress in `Pa`.
Viscosity uses the shear-rate magnitude; `shear_stress(gamma_dot)` preserves the
input sign.

| Model | Constructor parameters | Enforced mathematical domain |
|---|---|---|
| `NewtonianModel` | `mu0` [Pa s] | `mu0 > 0` |
| `PowerLawModel` | `K` [Pa s^n], `n` [-], `gamma_min=1e-10` [1/s] | all three `> 0`; viscosity is regularized below `gamma_min` |
| `CarreauModel` | `mu0`, `mu_inf` [Pa s], `lambda_` [s], `n` [-] | `mu0 > 0`, `0 <= mu_inf <= mu0`, `lambda_ > 0`, `0 < n <= 1` |
| `CarreauYasudaModel` | as above plus `a` [-] | Carreau limits plus `a > 0` |
| `CrossModel` | `mu0`, `mu_inf` [Pa s], `K` [s], `m` [-] | `mu0 > 0`, `0 <= mu_inf <= mu0`, `K,m > 0`; parameters must give monotone shear stress |
| `BinghamModel` | `tau_y` [Pa], `mu_p` [Pa s], `epsilon=1e-6` [1/s] | `tau_y >= 0`, `mu_p,epsilon > 0` |
| `HerschelBulkleyModel` | `tau_y` [Pa], `K` [Pa s^n], `n` [-], `epsilon=1e-6` [1/s] | `tau_y >= 0`, `K,n,epsilon > 0` |
| `CassonModel` | `tau_y` [Pa], `mu_p` [Pa s], `epsilon=1e-6` [1/s] | `tau_y >= 0`, `mu_p,epsilon > 0` |

The Bingham, Herschel--Bulkley, and Casson implementations are regularized
apparent-viscosity laws. Their finite viscosity near zero shear depends on
`epsilon`; they do not explicitly track an unyielded plug as a separate state.

`blood_casson_model(hematocrit)` accepts a red-cell volume fraction from 0 to
0.60. Its source correlation was studied over approximately 0.35--0.55, so
accepted values outside that interval are extrapolations. It is a
population-level correlation, not a patient-specific model.
`blood_carreau_model(hematocrit)` also accepts 0 to 0.60, but only the 0.45
hematocrit point is anchored to the stated Carreau fit (`mu0=0.056 Pa s`,
`mu_inf=0.00345 Pa s`, `lambda=3.313 s`, `n=0.3568`). Its hematocrit scaling
away from 0.45 is explicitly an educational surrogate.

`pipe_wall_shear_rate(Q, R)` returns the Newtonian nominal magnitude
`4 abs(Q)/(pi R^3)` for `Q` in `m^3/s` and positive `R` in m. It is not a
non-Newtonian wall correction. `apparent_viscosity_pipe(model, Q, R,
pressure_gradient)` uses a model-based Rabinowitsch--Mooney correction;
`Q != 0`, `pressure_gradient != 0`, and `R > 0` are required.

These are generalized-Newtonian constitutive laws. They do not represent
viscoelastic memory, thixotropy, red-cell migration, phase separation, vessel
compliance, temperature dependence, or sample-specific hematology unless the
caller adds and validates those effects elsewhere.

## Informational range helper

`get_parameter_ranges()` returns a small flat dictionary with the keys
`D_drug`, `D_oxygen`, `D_glucose`, `k_tissue`, `c_tissue`, `w_b`, `IFP_tumor`,
and `MVD`. Its entries are broad informational values only; they are not used
by solver validation and should not be cited as parameter provenance.

| Key | Minimum | Typical | Maximum | Unit |
|---|---:|---:|---:|---|
| `D_drug` | `1e-12` | `1e-11` | `1e-9` | m^2/s |
| `D_oxygen` | `1e-10` | `2e-9` | `5e-9` | m^2/s |
| `D_glucose` | `1e-10` | `6e-10` | `1e-9` | m^2/s |
| `k_tissue` | `0.2` | `0.5` | `0.8` | W/(m K) |
| `c_tissue` | `3000` | `3600` | `4000` | J/(kg K) |
| `w_b` | `1e-4` | `5e-4` | `1e-2` | 1/s |
| `IFP_tumor` | `5` | `20` | `60` | mmHg |
| `MVD` | `10` | `100` | `400` | vessels/mm^2 |

```python
import biotransport as bt

ranges = bt.get_parameter_ranges()
print(ranges["D_drug"]["typical"], ranges["D_drug"]["unit"])
```

Use the underlying experimental or review literature, not this convenience
dictionary, when selecting parameters for a scientific claim.

## Worked examples

- [Bioheat cryotherapy](../../examples/advanced/bioheat_cryotherapy.py)
- [Tumor drug delivery](../../examples/advanced/tumor_drug_delivery.py)
- [Nernst--Planck ion transport](../../examples/advanced/nernst_planck_ion_transport.py)
- [Steady membrane diffusion](../../examples/intermediate/steady_membrane_diffusion.py)
- [Multispecies reaction--diffusion](../../examples/intermediate/multi_species_reaction_diffusion.py)
- [Blood rheology](../../examples/intermediate/blood_rheology.py)

Examples demonstrate API usage and numerical behavior. They are not biological
validation studies.
