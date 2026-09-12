import test from "node:test";
import assert from "node:assert/strict";
import {
  addModelComponent,
  categoryOf,
  isClosedDiffusion,
  slotOf,
  componentAvailability,
} from "../../python/biotransport/studio/static/model.mjs";

test("component support follows registry geometry and solver metadata", () => {
  const radial = { domain: { geometry: "spherical" }, run: { mode: "steady" } };
  assert.match(componentAvailability({ geometries: ["cartesian"] }, radial), /cartesian/);
  assert.match(componentAvailability({ supports_steady: false }, radial), /evolution/);
  assert.equal(componentAvailability({ geometries: ["spherical"], supports_steady: true }, radial), null);
  assert.equal(componentAvailability({ supports_steady: false }, { ...radial, run: { mode: "transient" } }), null);
});

const catalog = [
  {
    type: "initial.uniform",
    category: "initial",
    slot: "initial",
    parameters: [{ name: "value", default: 0 }],
  },
  {
    type: "lab.band",
    category: "initial",
    slot: "initial",
    parameters: [{ name: "value", default: 2 }],
  },
  {
    type: "boundary.fixed",
    category: "boundary",
    slot: "boundary",
    parameters: [
      { name: "side", default: "left" },
      { name: "value", default: 1 },
    ],
  },
  {
    type: "lab.face",
    category: "boundary",
    slot: "boundary",
    parameters: [{ name: "side", default: "left" }],
  },
  {
    type: "reaction.decay",
    category: "reaction",
    slot: null,
    parameters: [{ name: "rate", default: 1 }],
  },
  {
    type: "lab.uptake",
    category: "reaction",
    slot: null,
    parameters: [{ name: "rate", default: 2 }],
  },
  {
    type: "advection",
    category: "transport",
    slot: "advection",
    parameters: [{ name: "velocity", default: 1 }],
  },
  {
    type: "lab.flow",
    category: "transport",
    slot: "advection",
    parameters: [{ name: "velocity", default: 2 }],
  },
];
const model = () => ({
  components: [
    { id: "initial", type: "initial.uniform", parameters: { value: 0 } },
    {
      id: "left",
      type: "boundary.fixed",
      parameters: { side: "left", value: 1 },
    },
    {
      id: "right",
      type: "boundary.fixed",
      parameters: { side: "right", value: 1 },
    },
  ],
});

test("custom initial role replaces the old field without mutating the source", () => {
  const source = model();
  const next = addModelComponent(source, catalog, "lab.band", "@domain", "new");
  assert.equal(next.model.components.length, 3);
  assert.equal(next.selected, "initial");
  assert.equal(next.model.components[0].type, "lab.band");
  assert.equal(source.components[0].type, "initial.uniform");
  assert.equal(categoryOf(catalog[1]), "initial");
});

test("custom boundary replaces only the selected side", () => {
  const next = addModelComponent(model(), catalog, "lab.face", "right", "new");
  assert.equal(next.selected, "right");
  assert.equal(next.model.components[1].type, "boundary.fixed");
  assert.deepEqual(next.model.components[2].parameters, { side: "right" });
  assert.equal(slotOf(catalog, next.model.components[2]), "boundary");
});

test("adding a missing right face keeps the existing left face", () => {
  const source = model();
  source.components.pop();
  const next = addModelComponent(
    source,
    catalog,
    "boundary.fixed",
    "@domain",
    "right-again",
    "right",
  );
  assert.equal(next.model.components.length, 3);
  assert.equal(next.model.components[1].id, "left");
  assert.equal(next.model.components[2].parameters.side, "right");
});

test("additive reactions accumulate and custom flow replaces its role", () => {
  let next = addModelComponent(model(), catalog, "lab.uptake", "@domain", "r1");
  next = addModelComponent(
    next.model,
    catalog,
    "lab.uptake",
    next.selected,
    "r2",
  );
  assert.equal(next.model.components.length, 5);
  next = addModelComponent(
    next.model,
    catalog,
    "advection",
    next.selected,
    "v",
  );
  next = addModelComponent(
    next.model,
    catalog,
    "lab.flow",
    next.selected,
    "v2",
  );
  assert.equal(next.model.components.length, 6);
  assert.equal(next.selected, "v");
});

test("a custom reaction cannot inherit a pure-diffusion conservation label", () => {
  const closed = {
    components: [
      { type: "diffusion" },
      { type: "initial.gaussian" },
      { type: "boundary.sealed" },
      { type: "boundary.sealed" },
    ],
  };
  assert.equal(isClosedDiffusion(closed), true);
  closed.components.push({ type: "lab.uptake" });
  assert.equal(isClosedDiffusion(closed), false);
});

test("unknown palette entries fail without changing a model", () => {
  assert.throws(
    () => addModelComponent(model(), catalog, "unknown", "@domain", "new"),
    /Unknown component/,
  );
});
