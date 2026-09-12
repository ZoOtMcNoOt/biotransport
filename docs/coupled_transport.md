# Multiple domains, multiple species

`CoupledModel` joins well-mixed compartments and spatial transport domains.
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
`sum(weights[s] * V[i] * c[s,i])` over saved states. Its relative drift is
undefined when the initial amount is zero; absolute drift remains available.

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

The public RHS and sparse Jacobian can be passed directly to SciPy or other
integrators. State ordering is domains in insertion order, then species, then
nodes. Prefer `state_slice` to hand-written offsets. Configuration arrays are
owned and compiled models are independent of later builder edits.

The default method is SciPy BDF; Radau is also supported. Both use sparse
Jacobians. Diffusion, membranes and first-order reactions are assembled once;
nonlinear rates and their derivatives are vectorized over each domain.
Output times do not restart integration. `frames=40` saves 41 states including
the initial state. `save_at` sorts and deduplicates times and includes the final
time. `max_step` bounds internal steps when a callback has rapid time dependence.

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
`examples/verification/benchmark_coupled.py` measures assembly and integration
with balance and tighter-tolerance checks; it does not claim biological accuracy.

This increment covers closed networks of compartments and 1D Cartesian,
cylindrical and spherical domains with local reactions. Prescribed external
baths, flow/advection, 2D/3D interface mappings, nonuniform meshes, coupled steady
solves, parameter fitting and model serialization are not implemented in this
API. Existing single-domain solvers remain available for their documented scope.
The UI is deferred; future interfaces can build on this engine.
