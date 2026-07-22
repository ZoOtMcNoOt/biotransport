# Darcy-flow verification

The Darcy solver is checked against independent steady, incompressible
solutions of

\[
\nabla\cdot\left(\kappa\nabla p\right)=0,
\qquad
\mathbf{v}=-\kappa\nabla p,
\]

where pressure \(p\) is in Pa, hydraulic mobility \(\kappa=K/\mu\) is in
m²/(Pa·s), and Darcy velocity \(\mathbf{v}\) is in m/s. The solver accepts
mobility directly; it does not accept intrinsic permeability and viscosity as
separate inputs.

## Boundary convention

`setDirichlet` / `set_dirichlet` prescribes pressure [Pa]. `setNeumann` /
`set_neumann` prescribes the **outward-normal pressure derivative**
\(\partial p/\partial n\) [Pa/m]. It is not a Darcy flux, despite the legacy
Python argument name `flux`. The corresponding outward velocity is

\[
\mathbf{v}\cdot\mathbf{n}=-\kappa\frac{\partial p}{\partial n}.
\]

The sign check prescribes +25,000 Pa/m independently on the left and right
boundaries. In both cases the pressure rises by 500 Pa over a 0.02 m slab and
the outward Darcy velocity is -1.0e-5 m/s for
\(\kappa=4.0\times10^{-10}\) m²/(Pa·s).

## Analytical cases

The uniform-mobility case uses left/right pressures of 3200 Pa and 800 Pa over
0.04 m, zero outward pressure derivative on the top and bottom, and
\(\kappa=2.5\times10^{-10}\) m²/(Pa·s). Its exact solution is linear, with
\(dp/dx=-60{,}000\) Pa/m and \(v_x=1.5\times10^{-5}\) m/s. The measured native
errors were:

- pressure L-infinity error: 6.906702765e-9 Pa;
- x-velocity L-infinity error: 1.610362099e-16 m/s;
- maximum transverse speed: 2.370370566e-17 m/s;
- maximum fixed-point defect: 9.822542779e-11 Pa after 605 iterations.

The two-material case places a face-aligned interface at 0.5 m, with
\(\kappa_1=2.0\times10^{-10}\) and
\(\kappa_2=8.0\times10^{-10}\) m²/(Pa·s). For pressure limits of 6000 Pa and
1000 Pa, the series-resistance solution is

\[
q=\frac{6000-1000}{0.5/\kappa_1+0.5/\kappa_2}
  =1.6\times10^{-6}\ \mathrm{m/s},
\]

with an interface pressure of 2000 Pa. Reconstructing the normal face flux
from the public nodal pressure and mobility fields gave a relative flux spread
of 2.117582368e-14 and a pressure L-infinity error of 2.728484105e-12 Pa. Face
flux is reconstructed because the current result object exposes nodal
velocities, not a face-flux diagnostic.

## Honest interface refinement

An aligned discontinuity can be represented in two distinct ways:

- An odd cell count places the interface on a face. Harmonic face mobility then
  reproduces the two resistances in series to solver tolerance.
- An even cell count places the physical interface on a node. Assigning that
  node to the left material locates the represented jump half a cell to the
  right, so the dominant error is first order rather than second order.

The measured node-aligned sequence was:

| x cells | spacing [m] | pressure L-infinity error [Pa] | observed order |
|---:|---:|---:|---:|
| 16 | 0.0625 | 144.5783132 | — |
| 32 | 0.03125 | 73.61963192 | 0.9736887225 |
| 64 | 0.015625 | 37.15170282 | 0.9866621993 |

These results document the actual discontinuous-interface behavior; they do
not claim second-order convergence for a node-labeled jump.

## Failure contracts

An all-Neumann problem is singular because its pressure gauge is missing and
must raise an invalid-argument / `ValueError`. A deliberately insufficient
iteration limit must raise a runtime / `RuntimeError` instead of returning an
unconverged field.

## Reproduce

The evidence lives in:

- `cpp/tests/physics/test_darcy_science.cpp`;
- `python/tests/test_darcy_science.py`;
- `examples/verification/verify_darcy.py`.

Run the Python evidence with:

```text
python -m pytest python/tests/test_darcy_science.py -q
python examples/verification/verify_darcy.py
```

The verification is deterministic, bounded, and headless. It validates this
steady isotropic Darcy model only; it does not establish accuracy for
anisotropic permeability, deformable porous media, multiphase flow, or coupled
solute transport.
