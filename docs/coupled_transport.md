# Multiple domains, multiple species

`CoupledModel` joins well-mixed compartments, spatial transport domains and
prescribed external baths.
Name the species, add the domains, connect their boundaries, and describe the
chemistry. The engine assembles one system and integrates every domain together.
This API is independent of Studio and does not require a browser.

## A complete model

This illustrative model transfers a solute from a reservoir into a tissue slab,
where it converts into a second species. Values are illustrative, not measured
biological parameters.

```python
import biotransport as bt

model = bt.CoupledModel(["drug", "metabolite"])
model.compartment("reservoir", volume=5e-6, initial={"drug": 1.0})
model.domain(
    "tissue", bt.mesh_1d(60, 0.0, 1e-3),
    cross_section=1e-4,
    diffusivity={"drug": 1e-9, "metabolite": 5e-10},
)
model.membrane(
    "wall", "reservoir", ("tissue", "left"),
    area=1e-4, permeability={"drug": 2e-6}, partition={"drug": 2.0},
)
model.mass_action(
    "tissue", reactants={"drug": 1}, products={"metabolite": 1},
    rate_constant=0.01,
)
model.conserve("drug equivalents", {"drug": 1, "metabolite": 1})

solution = model.solve(end_time=600, frames=40)
concentration = solution.field("tissue", "drug")
history = solution.history("tissue", "metabolite")
amount = solution.amount("drug")
transfer_rate = solution.interface_rate("wall", "drug")
print(solution.balance("drug equivalents"))
```

`field` returns the final nodal values. `history` has shape `(times, nodes)`;
a compartment has one node. `amount` returns moles at each saved time across all
domains, or within `domain="tissue"`. Membrane rates are signed from source to
target in mol/s. Returned arrays are owned copies.

All domains contain the named species. Omitted initial concentrations and
diffusivities are zero. An omitted membrane permeability is zero: the example
allows only the drug to cross. Scalar permeability and partition values apply
to all species. Unconnected boundaries are sealed.

## Units and geometry

Plain numbers use SI: coordinates in m, time in s, concentrations in mol/m^3,
diffusivity in m^2/s, permeability in m/s, volume in m^3, area in m^2.
Existing `bt.quantity` values are also accepted for concentration, diffusivity,
permeability, time and cylinder length, with dimensional checks. Area and volume
currently use plain SI numbers; the engine does not infer them from a drawing.

| Domain | Required geometry | Physical volume |
|---|---|---|
| Compartment | `volume` | Supplied m^3 |
| Cartesian 1D | `cross_section` | Area times length |
| Cylindrical 1D | `axial_length` | Full annular cylinder |
| Spherical 1D | Neither extra parameter | Full spherical shell |

Native mesh control volumes are multiplied by the physical cross section,
`2*pi*axial_length`, or `4*pi`. A node at a domain boundary owns a half cell.
The radial origin has zero face area and cannot carry a membrane connection.
Multiple membranes can share a boundary if their combined area fits that face.

Connect two spatial domains with endpoints such as `("skin", "right")` and
`("tissue", "left")`. Their boundary concentrations remain distinct. A membrane
is a finite-resistance connection with negligible storage; model a layer with
its own storage as another spatial domain.

## Equations and conservation

For species s in a control volume i, the concentration balance is

$$
V_i \frac{dc_{s,i}}{dt} = \sum_j F_{s,j\to i}
    + V_i \sum_r \nu_{s,r}\,v_r(c_i,t).
$$

Interior diffusion transfers amount from left to right at rate

$$
F_s = \frac{A_f D_{s,f}}{\Delta x}(c_{s,L}-c_{s,R}),
\qquad D_{s,f}=\frac{2D_{s,L}D_{s,R}}{D_{s,L}+D_{s,R}}.
$$

If either diffusivity is zero, the face is impermeable. The harmonic mean
represents the two half-cell resistances. This mesh family is uniform in its
coordinate; the physical radial face areas and volumes are not uniform.

A membrane transfers

$$
F_s=P_s A\left(c_{s,\mathrm{source}}-
              \frac{c_{s,\mathrm{target}}}{K_s}\right).
$$

**The partition convention is `K = target/source` at equilibrium.** Positive F
removes amount from the source and adds exactly that amount to the target.
Different domain volumes therefore produce different concentration changes.
The opposite contributions are assembled together, without sequential exchange
updates or operator splitting. Interface P and K are positive physical inputs
(P may be zero); interface area is positive.

Mass-action reactions use

$$
v_r = k_r \prod_s c_s^{\alpha_{s,r}},
\qquad \nu_{s,r}=\beta_{s,r}-\alpha_{s,r}.
$$

Reactant stoichiometry is also the kinetic order; reactant coefficients must be
positive integers. The rate constant has units `(mol/m^3)^(1-order)/s`.
No combinatorial factorial is included. For `2 A -> B`, consumption is `2*k*A^2`
and production is `k*A^2`; the conserved combination is `A + 2 B`.
Two reaction declarations represent a reversible reaction.

`conserve(name, weights)` verifies `weights dot stoichiometry = 0` for every
reaction, including future edits. It checks a chemical invariant, not whether
the biological model is appropriate. The balance report measures drift of
`sum(weights[s] * V[i] * c[s,i])` against its initial amount plus integrated
external exchange. Relative drift uses the largest absolute expected amount
over saved states, and is undefined when that scale is zero. Absolute drift
always remains available. Closed networks have zero external exchange.

## Reversible binding and custom kinetics

Binding uses three named species and two elementary reactions:

```python
binding = bt.CoupledModel(["drug", "site", "bound"])
binding.compartment("tissue", volume=1e-6, initial={"drug": 1, "site": 2})
binding.mass_action("tissue", reactants={"drug": 1, "site": 1},
                    products={"bound": 1}, rate_constant=3.0)
binding.mass_action("tissue", reactants={"bound": 1},
                    products={"drug": 1, "site": 1}, rate_constant=0.4)
binding.conserve("drug", {"drug": 1, "bound": 1})
binding.conserve("sites", {"site": 1, "bound": 1})
bound = binding.solve(20).field("tissue", "bound")
```

For a different rate law, supply its local rate and analytic partial derivatives:

```python
custom = bt.CoupledModel(["drug", "metabolite"])
custom.compartment("tissue", volume=1e-6, initial={"drug": 1})
vmax, km = 0.02, 0.3
custom.reaction(
    "tissue", stoichiometry={"drug": -1, "metabolite": 1},
    rate=lambda t, c: vmax * c["drug"] / (km + c["drug"]),
    derivative=lambda t, c: {"drug": vmax * km / (km + c["drug"])**2},
)
custom.conserve("drug", {"drug": 1, "metabolite": 1})
custom_solution = custom.solve(20)
```

Callbacks receive time and a mapping of species to owned read-only arrays.
Return a scalar or one rate per local node. Derivative entries are partial
derivatives of that same rate; omitted species derivatives mean zero. Spatially
varying coefficients can be captured as arrays in a closure. Callbacks must
act locally at each node: nonlocal reactions require a different operator.
Keep callbacks pure because their captured external state cannot be frozen.
Do not clip Newton trial concentrations: the supplied rate and derivative must
agree in the neighborhood the nonlinear solver explores.

## External baths, dosing and washout

A bath is maintained at a prescribed concentration. Its supply and depletion
are outside the modeled system. Use a compartment when the reservoir has a
finite volume whose concentration should change through exchange.

```python
pulse = bt.ConcentrationSchedule([0, 60, 120], [0, 1, 0])
dosing = bt.CoupledModel(["drug", "metabolite"])
dosing.compartment("tissue", volume=1e-9)
dosing.bath("dose", concentration={"drug": pulse})
dosing.membrane("wall", "dose", "tissue", area=1e-6,
                 permeability={"drug": 1e-5})
dosing.mass_action("tissue", reactants={"drug": 1},
                    products={"metabolite": 1}, rate_constant=0.02)
dosing.conserve("drug equivalents", {"drug": 1, "metabolite": 1})

dose_result = dosing.solve(240, save_at=[30, 90, 240])
assert dose_result.times.tolist() == [0, 30, 60, 90, 120, 240]
stored_drug = dose_result.amount("drug")
net_supplied = dose_result.external_amount("drug", membrane="wall")
prescribed = dose_result.bath_history("dose", "drug")
report = dose_result.balance("drug equivalents")
print(report.expected_final, report.final, report.maximum_absolute_drift)
```

Each bath species accepts a nonnegative scalar, a concentration quantity, or a
`ConcentrationSchedule`. Missing species have zero concentration. A bath can
connect to a compartment or a spatial boundary through the same membrane law
and partition convention. Two baths cannot connect directly without a modeled
domain. Baths do not add concentration states, and `amount` counts only modeled
domains. `bath_history` shows the prescribed input.

Schedules start at zero and have strictly increasing times. They own their
input data. The default `interpolation="step"` holds each concentration until
the next change; `interpolation="linear"` provides ramps between values. Both
hold the last value indefinitely. Times accept seconds or time quantities;
values accept mol/m^3 or concentration quantities.

**The solver lands on every connected schedule knot**, including knots between
the requested output frames. Those times are always saved. A finishing interval
uses the boundary value immediately before a jump; the next interval uses the
new value. Domain concentrations and cumulative transferred amounts remain
continuous. `bath_history` and `interface_rate` show the new instantaneous value
at a jump. A change exactly at `end_time` therefore changes the reported bath
concentration and instantaneous rate, but has not yet delivered material.
Scalar constants and inactive, impermeable bath species add no restart times.

The engine integrates a signed amount ledger for each permeable bath connection
alongside the concentrations:

$$
\frac{dQ_{m,s}}{dt} = F_{m,s}^{\mathrm{into\ model}},\qquad Q_{m,s}(0)=0.
$$

`external_amount` is cumulative net moles entering the modeled domains, with
negative increments during removal. Its positive direction always means into
the model. `interface_rate` follows the membrane's declared source-to-target
direction, so the signs are opposite when a bath is the membrane target.
Selecting one membrane preserves separate entry and exit records when multiple
baths exchange material with a stationary domain.

The balance expectation is

$$
M_{\mathrm{expected}}(t)=M(0)+\sum_{m,s} w_s Q_{m,s}(t).
$$

Accounting uses the same integration stages as transport. It does not estimate
amounts from sparsely saved fluxes. Internally each ledger variable is divided
by its connected physical control volume, giving concentration units and the
same absolute tolerance as that domain/species state. This keeps tiny-volume
amounts within the integrator's error controls. Chemical invariants continue
to validate reaction stoichiometry; external entry and exit are handled by the
ledger, including when initial concentrations are zero.

## Research access and integration controls

```python
compiled = model.compile()
print(compiled.summary())
state = compiled.initial_state
drug_indices = compiled.state_slice("tissue", "drug")
x = compiled.coordinates("tissue")
rate = compiled.rhs(0.0, state)
jacobian = compiled.jacobian(0.0, state)
transport = compiled.transport_matrix
volumes = compiled.state_volumes

precise = compiled.solve(
    600, save_at=[10, 60, 300], method="Radau", rtol=1e-9,
    atol={"drug": 1e-12, "metabolite": 1e-12},
)
```

The public RHS and sparse Jacobian can be passed to SciPy or other integrators.
For scheduled baths, external integrator users must split at
`compiled.breakpoints`. At the finishing endpoint, `compiled.rhs(t, c,
bath_side="left")` evaluates the pre-jump bath without shifting reaction time;
the default evaluates the post-jump bath. `external_rates` accepts the same
control and returns mol/s into the model, keyed by `(membrane, species)`, for
independent amount integration. `ConcentrationSchedule.at` and
`bath_concentration` expose the time limits with `side="left"` or `"right"`.
The public state remains concentrations only; built-in `solve` handles its
internal amount ledger automatically.

State ordering is domains in insertion order, then species, then
nodes. Prefer `state_slice` to hand-written offsets. Configuration arrays are
owned and compiled models are independent of later builder edits.

The default method is SciPy BDF; Radau is also supported. Both use sparse
Jacobians. Diffusion, membranes and first-order reactions are assembled once;
nonlinear rates and their derivatives are vectorized over each domain.
Output times do not restart integration. `frames=40` requests 41 states including
the initial state; any additional bath schedule knots are also saved. `save_at`
sorts and deduplicates times and includes the final time. Scheduled bath changes
restart integration with the appropriate boundary law. `max_step` bounds
internal steps when a custom reaction callback has rapid time dependence.

`rtol` and `atol` control local error estimates, not global error. Use
species-specific absolute tolerances when concentration scales differ. Implicit
integration does not guarantee nonnegativity; the diagnostics report the
minimum at saved times without clipping it. Check tighter tolerances, spatial
refinement, conserved amounts and accepted concentration ranges for a study.
The integration controls follow the [SciPy solve_ivp contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

## Verified scope and next boundaries

Tests cover unequal-volume exchange with partitioning, exact first- and
second-order reaction kinetics, reversible binding, separate spatial domains,
second-order slab diffusion convergence, native radial diffusion agreement,
physical shell volumes, conservation, sparse Jacobians and owned state.
Protocol checks include exact open-system kinetics, independent quadrature and
matrix-exponential references, one-microsecond pulses, ramps, interleaved baths,
one-sided endpoint behavior and volume-scaled transfer accounting down to 1e-18 m^3.
`examples/verification/benchmark_coupled.py` measures assembly and integration
with balance and tighter-tolerance checks; it does not claim biological accuracy.
`examples/verification/benchmark_protocols.py` measures a prescribed bath pulse
with sparse implicit transport, reversible binding and integrated amount checks.

This API covers open and closed networks of compartments and 1D Cartesian,
cylindrical and spherical domains with local reactions. Bath concentration
protocols support steps and linear ramps; permeability and partition coefficients
remain constant. Flow/advection, 2D/3D interface mappings, nonuniform meshes, coupled steady
solves, parameter fitting and model serialization are not implemented in this
API. Existing single-domain solvers remain available for their documented scope.
The UI is deferred; future interfaces can build on this engine.
