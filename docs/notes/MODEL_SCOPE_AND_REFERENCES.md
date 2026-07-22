# Model scope, provenance, and reference equations

This note records why a model exists and where its governing form comes from.
It does **not** certify a parameter set, tissue type, patient population, device,
or intended use.  Numerical verification answers whether code solves the stated
equations; biological validation asks whether those equations and parameters
adequately represent measured reality for a specific question.

## Scalar transport

The canonical scalar balance is

```text
dc/dt = div(D grad(c)) - div(v c) + R(c, x, t)
J = -D grad(c) + v c
```

`D` must be nonnegative, velocity and concentration must share a consistent
coordinate and unit system, and `R` has concentration-per-time units.  The
canonical discretization is conservative, but continuum-model assumptions such
as dilute transport, continuum scale, isotropic scalar diffusion, and prescribed
velocity remain the user's responsibility.

## Nernst--Planck transport

For an ideal dilute ion with valence `z`, concentration `c`, diffusivity `D`,
and electric potential `phi`, the flux convention is

```text
J = -D [grad(c) + z F c grad(phi)/(R T)]
dc/dt = -div(J)
```

The sign follows electric potential energy `z F phi`.  A prescribed-potential
Nernst--Planck model is not a Poisson--Nernst--Planck model: it does not solve
electrostatics, finite-ion-size effects, activity corrections, fluid coupling,
or electroneutral boundary layers unless those are explicitly added.

Historical basis: M. Planck, “Ueber die Erregung von Electricität und Wärme in
Electrolyten,” *Annalen der Physik* 275 (1890), 161--186,
[doi:10.1002/andp.18902750202](https://doi.org/10.1002/andp.18902750202).

## Membrane and hindered diffusion

The simple permeability relation `P = D K/L` assumes a homogeneous membrane,
constant diffusivity and partition coefficient, dilute solute, and steady 1D
transport without solvent drag.  The Renkin hindrance correlation is a pore
model with a limited radius-ratio domain; it is not a universal correction for
biological barriers.

Foundational sources:

- O. Kedem and A. Katchalsky, “Thermodynamic analysis of the permeability of
  biological membranes to non-electrolytes,” *Biochimica et Biophysica Acta* 27
  (1958), 229--246,
  [doi:10.1016/0006-3002(58)90330-5](https://doi.org/10.1016/0006-3002(58)90330-5).
- E. M. Renkin, “Filtration, diffusion, and molecular sieving through porous
  cellulose membranes,” *Journal of General Physiology* 38 (1954), 225--243,
  [PubMed 13211998](https://pubmed.ncbi.nlm.nih.gov/13211998/).  The published
  correction must be consulted when reproducing the original derivation:
  [PMCID PMC2147559](https://pmc.ncbi.nlm.nih.gov/articles/PMC2147559/).

## Gray--Scott reaction--diffusion

The common nondimensional pattern model is

```text
du/dt = Du laplacian(u) - u v^2 + f (1-u)
dv/dt = Dv laplacian(v) + u v^2 - (f+k) v
```

Its parameters are nondimensional model controls unless a separate
nondimensionalization maps them to a chemical system.  Pattern resemblance is
not evidence of a biological mechanism.

Primary reaction-system sources:

- P. Gray and S. K. Scott, “Autocatalytic reactions in the isothermal,
  continuous stirred tank reactor: Isolas and other forms of multistability,”
  *Chemical Engineering Science* 38 (1983), 29--43,
  [doi:10.1016/0009-2509(83)80132-8](https://doi.org/10.1016/0009-2509(83)80132-8).
- P. Gray and S. K. Scott, “Autocatalytic reactions in the isothermal,
  continuous stirred tank reactor: Oscillations and instabilities in the system
  A + 2B -> 3B; B -> C,” *Chemical Engineering Science* 39 (1984), 1087--1097,
  [doi:10.1016/0009-2509(84)87017-7](https://doi.org/10.1016/0009-2509(84)87017-7).

## Pennes bioheat and thermal damage

The Pennes source form is

```text
rho_t c_t dT/dt = div(k grad(T))
                  + omega_b rho_b c_b (T_a - T) + Q_met + Q_external
```

All temperatures used in differences may be Celsius or Kelvin, but absolute
temperature in an Arrhenius exponential must be Kelvin.  Perfusion conventions
vary between volumetric blood perfusion and mass perfusion; the units determine
whether `rho_b` belongs in the term.  Phase change introduces an enthalpy or
apparent-heat-capacity model and its own calibration assumptions.

The classic Pennes equation came from resting forearm measurements, not frozen
tumor tissue: H. H. Pennes, “Analysis of Tissue and Arterial Blood Temperatures
in the Resting Human Forearm,” *Journal of Applied Physiology* 1 (1948), 93--122,
[doi:10.1152/jappl.1948.1.2.93](https://doi.org/10.1152/jappl.1948.1.2.93).

The historical thermal-injury experiments are likewise tissue- and
temperature-regime specific: F. C. Henriques and A. R. Moritz, “Studies of
Thermal Injury: I,” *American Journal of Pathology* 23 (1947), 530--549,
[PubMed 19970945](https://pubmed.ncbi.nlm.nih.gov/19970945/).  Heat-damage
Arrhenius coefficients must not be reused for cryogenic injury without an
independent validated model for that regime.

## Tumor interstitial transport

A Darcy/Starling/advection--diffusion tumor model combines several closures:
porous-media flow, vascular exchange, lymphatic drainage, interstitial
diffusion/convection, binding, uptake, and clearance.  Every exchange term must
have compatible volume or area density factors, and total drug accounting must
include all modeled compartments and boundary fluxes.

R. K. Jain's reviews describe why vascular permeability, exchange area,
interstitial pressure, matrix structure, and lymphatic function cannot be
collapsed into a universal tumor default:

- “Transport of molecules in the tumor interstitium,” *Cancer Research* 47
  (1987), 3039--3051,
  [PubMed 3555767](https://pubmed.ncbi.nlm.nih.gov/3555767/).
- “Transport of molecules across tumor vasculature,” *Cancer Metastasis
  Reviews* 6 (1987), 559--593,
  [doi:10.1007/BF00047468](https://doi.org/10.1007/BF00047468).

## Minimum application-model record

For every parameter used in a scientific result, record:

1. symbol, definition, and SI unit;
2. source and exact table/figure/fit when available;
3. species, tissue/material, temperature, preparation, and measurement method;
4. central estimate, range or distribution, and correlations;
5. transformations or unit conversions;
6. calibration data distinct from evaluation data;
7. sensitivity of reported conclusions to the parameter;
8. numerical grid/time convergence and balance residuals;
9. software version, compiler, configuration, and random seed; and
10. limitations on the inference the model can support.

Defaults in examples are demonstrations, not recommended physiological values.
