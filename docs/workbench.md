# Build once, run from Python or Studio

BioTransport now has a shared experiment format and a local visual workbench.
An experiment describes a domain, physics components, boundary conditions and a
run. It compiles into the existing `Problem` API; transient calculations still
run through the native conservative transport solver. The browser contains no
numerical solver.

## Try Studio

After installing the package, run:

```bash
python -m biotransport.studio
# or: biotransport-studio
```

Open `http://127.0.0.1:8766`. Choose an example, drag a component onto the model
or use its add button, edit its properties, and run. Diffusion, initial fields,
and boundary components replace their matching role; reactions add together.
Undo restores model edits. Save JSON produces an experiment that Python can
open without Studio. Open JSON validates the entire document before replacing
the current model. Unsaved edits live in the page; save before closing it.

The workbench shows saved concentration profiles, a space–time map, frame
playback, dimensionless explanations, and the solver report. The profile scale
stays fixed across all saved frames. An exact-reference overlay appears only
when its assumptions match the experiment and its series has converged.
Inventory changes are separate from conservation checks: a boundary or reaction
can legitimately change the total. A data table and CSV export retain every
saved node; there is no hidden display downsampling in the exported values.

All distances are metres and times are seconds. Concentration uses one
consistent scale chosen by the author. Bundled examples are illustrative
teaching configurations, not calibrated biological parameters.

## Use the same model in Python

```python
import json
import biotransport as bt

with open("experiment.json") as stream:
    experiment = bt.Experiment.from_dict(json.load(stream))

solution = experiment.run()
solution.plot()

# Or use the compiled problem with the existing scientific workflow API.
problem = experiment.build()
solution = bt.solve(problem, end_time=1000, frames=20)
```

`Experiment.from_dict` validates the schema version, unknown fields, finite
numbers, exclusive component roles, boundaries, geometry and solver support.
It takes an owned snapshot of both the document and selected definitions.
`to_dict()` returns a fresh JSON-compatible document. Validation errors include
`issues`, a list of `{path, message}` objects suitable for any editor.

`experiment.plan()` reports the exact scheduled explicit-step count, selected
step-size cap and saved-field storage before integration. It includes both
reaction accuracy guards and the shorter steps at saved frames. Call
`experiment.plan(max_steps=...)` to inspect `within_step_budget`; a plan can
exceed the budget, while `run()` rejects it before taking a step. Steady plans
report one field and do not predict Newton convergence. Direct native users
can call `bt.plan_transport(problem, bt.SolveOptions.until(duration))`.

Studio shows this preview after each edit, explains invalid models and disables
running until the settings fit its interactive limits. Larger valid Python
documents can still be opened, edited and saved. Every boundary remains
selectable even when an edit leaves duplicate sides, so the model can be repaired.

## Add a component

Register the component once. Its parameter definitions produce both validation
and the Studio property controls. Registries belong to the caller; registering
a component does not change another application’s registry.

```python
import biotransport as bt
from biotransport.studio.server import StudioServer

registry = bt.builtin_registry()
registry.register(bt.ComponentDefinition(
    type="course.uptake",
    label="Tissue uptake",
    description="First-order uptake with a course-specific rate.",
    category="reaction",
    parameters=(bt.Parameter(
        name="rate", label="Uptake rate", type="number",
        default=0.0005, min=0, unit="1/s",
    ),),
    apply=lambda problem, values: problem.add_linear_decay(values["rate"]),
    supports_steady=True,
))

# Appears in the palette and property inspector without frontend changes.
with StudioServer(8766, registry=registry) as server:
    server.serve_forever()
```

A custom initial field declares `slot="initial"`; a custom boundary declares
`slot="boundary"` and a `side` choice parameter. These replace matching roles
regardless of their type name. Leave `slot=None` for additive components.
`geometries` and `supports_steady` declare the supported scope. Callbacks are
trusted Python code: they must preserve additive semantics, supply reaction
derivative bounds where needed, and have their own numerical evidence. A JSON
file only chooses explicitly registered types and cannot execute code or load
a module. Replacing a registered type raises an error.

## Architecture and present scope

```text
Python / notebooks                 Studio (HTML + ES modules)
       │                                  │ JSON / component catalog
       └──────────── Experiment ──────────┘
                          │ validate + compile
                       Problem
                          │
                solve / solve_steady
                          │
                       Solution
                          │
                plots / tables / JSON
```

The experiment adapter currently covers **one scalar field in 1D**: Cartesian,
cylindrical or spherical transient and steady diffusion/reaction transport.
The palette includes saturable Michaelis–Menten uptake, with maximum rate
`Vmax` and half-saturation concentration `Km`, and two tissue-sphere examples.
Geometry and solver restrictions come from component metadata. The radial origin must
have zero diffusive flux. Flow uses the canonical first-order upwind method;
`boundary.sealed` stops diffusion but does not stop advective transport.

The local HTTP adapter binds only to loopback, rejects cross-origin requests,
and runs one solve at a time. It limits documents to 256 KB, interactive meshes
to 1,000 cells, saved intervals to 120 and explicit steps to 200,000. A transient
run includes its initial field, so 40 intervals produce 41 saved states. The
Python experiment adapter allows 2,000 cells and 200 intervals; the underlying
`Problem` API supports larger meshes. Studio needs no frontend build, CDN or
additional Python server dependency, and ships inside the wheel. It is a local
workbench, not an authenticated multiuser hosting service.

The next significant engine extension is explicit coupling between domains,
species and membranes, with units, interface fluxes and conservation evidence.
A connected-node editor should build on that coupling contract. The current
component composition does not claim to implement arbitrary network coupling,
2D editing, mesh adaptation or the library’s specialized multiphysics solvers.

## Verify changes

```bash
python -m pytest -q
node --test tests/frontend/studio-model.test.mjs
python -m ruff check setup.py python
python examples/verification/benchmark_steady.py --help
```

The frontend tests exercise custom component roles and conservation labeling.
HTTP tests compare the workbench fields against the public engine, check exact
reference applicability, malformed input, resource limits and origin checks.
Numerical tests independently cover unit scaling, analytic profiles, sparse
singularity, extreme analytical limits and fine grids. See
[`notes/ENGINE_WORKBENCH.md`](notes/ENGINE_WORKBENCH.md) for the revision,
measurement conditions and verification record for this increment.
