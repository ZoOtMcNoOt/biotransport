import {
  addModelComponent,
  categoryOf,
  slotOf,
  isClosedDiffusion,
  componentAvailability,
} from "/model.mjs";
import { connectPaletteDrag } from "/drag.mjs";

const $ = (selector) => document.querySelector(selector);
const svgNS = "http://www.w3.org/2000/svg";
const state = {
  catalog: [],
  examples: [],
  model: null,
  selected: "@domain",
  undo: [],
  result: null,
  revision: 0,
  resultRevision: -1,
  frame: 0,
  view: "profile",
  busy: false,
  timer: null,
  plan: null,
  planRevision: -1,
  planTimer: null,
};
const clone = (value) => JSON.parse(JSON.stringify(value));
const fmt = (value, digits = 3) =>
  Number.isFinite(value) ? Number(value.toPrecision(digits)).toString() : "—";
const categoryNames = {
  transport: "Transport",
  diffusion: "Transport",
  initial: "Starting field",
  boundary: "Boundary conditions",
  reaction: "Reactions",
  advection: "Transport",
};
const definitionOf = (type) =>
  state.catalog.find((definition) => definition.type === type);
const selectedPart = () =>
  state.model.components.find((part) => part.id === state.selected);
const say = (message) => {
  $("#announcement").textContent = message;
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function svgElement(tag, attributes = {}) {
  const node = document.createElementNS(svgNS, tag);
  for (const [name, value] of Object.entries(attributes))
    node.setAttribute(name, String(value));
  return node;
}

function icon(type) {
  const container = element("span", "component-icon");
  const svg = svgElement("svg", {
    viewBox: "0 0 24 24",
    "aria-hidden": "true",
  });
  const paths = {
    diffusion:
      "M3 8h6m6 0h6M6 5 3 8l3 3m12-6 3 3-3 3M3 16h6m6 0h6M6 13l-3 3 3 3m12-6 3 3-3 3",
    initial: "M3 19h18M4 16c4 0 4-11 8-11s4 11 8 11",
    boundary: "M8 3v18m8-18v18M4 6h4m-4 6h4m-4 6h4m8-12h4m-4 6h4m-4 6h4",
    reaction: "M4 6h6l4 12h6M4 18h6L14 6h6",
    advection: "M3 12h17m-6-6 6 6-6 6M3 6h5M3 18h5",
    domain: "M3 5h18v14H3zM9 5v14m6-14v14",
  };
  const key = type.split(".")[0];
  svg.append(svgElement("path", { d: paths[key] || paths.reaction }));
  container.append(svg);
  return container;
}

function markChanged() {
  state.revision++;
  $("#edit-state").textContent = "Edited";
  $("#undo-button").disabled = state.undo.length === 0;
  if (state.result)
    $("#result-caption").textContent =
      "Previous run · Your model has changed. Run again to update these results.";
  $("#run-status").textContent = "Changes ready to simulate";
  $("#run-error").hidden = true;
  state.plan = null;
  state.planRevision = -1;
  $("#run-button").disabled = true;
  $("#run-plan").replaceChildren(element("p", "", "Checking this model…"));
  clearTimeout(state.planTimer);
  state.planTimer = setTimeout(refreshPlan, 250);
}

function mutate(change, redraw = true) {
  state.undo.push({ model: clone(state.model), selected: state.selected });
  if (state.undo.length > 40) state.undo.shift();
  change();
  markChanged();
  if (redraw) renderModel();
}

function select(id) {
  state.selected = id;
  renderModel();
  renderProperties();
}

function renderLibrary() {
  const root = $("#component-library");
  root.replaceChildren();
  const groups = new Map();
  for (const definition of state.catalog) {
    const category = categoryOf(definition);
    if (!groups.has(category)) {
      const group = element("section", "library-group");
      group.append(element("h3", "", categoryNames[category] || category));
      root.append(group);
      groups.set(category, group);
    }
    const row = element("div", "library-item");
    const unavailable = componentAvailability(definition, state.model);
    if (!unavailable) row.dataset.type = definition.type;
    row.classList.toggle("unavailable", Boolean(unavailable));
    row.title = definition.description;
    row.append(icon(definition.type));
    const copy = element("span", "item-copy");
    copy.append(element("strong", "", definition.label));
    if (unavailable) copy.append(element("small", "library-reason", unavailable));
    const add = element("button", "", "+");
    add.disabled = Boolean(unavailable);
    add.setAttribute("aria-label", `Add ${definition.label}`);
    add.addEventListener("click", () => addComponent(definition.type));
    row.append(copy, add);
    groups.get(category).append(row);
  }
  $("#component-count").textContent = state.catalog.length;
}

function addComponent(type, side) {
  const definition = definitionOf(type);
  if (!definition) return;
  const unavailable = componentAvailability(definition, state.model);
  if (unavailable) { say(unavailable); return; }
  mutate(() => {
    Object.assign(
      state,
      addModelComponent(
        state.model,
        state.catalog,
        type,
        state.selected,
        `component-${crypto.randomUUID().slice(0, 8)}`,
        side,
      ),
    );
  });
  renderProperties();
  say(`${definition.label} added. Edit its properties or run the simulation.`);
}

function partSummary(part) {
  const definition = definitionOf(part.type);
  if (!definition) return part.type;
  return definition.parameters
    .map((parameter) => {
      const value = part.parameters[parameter.name];
      return `${typeof value === "number" ? fmt(value) : (value ?? "unset")}${parameter.unit ? ` ${parameter.unit}` : ""}`;
    })
    .join(" · ");
}

function renderModel() {
  if (!state.model) return;
  renderLibrary();
  const domain = state.model.domain;
  const geometry = {
    cartesian: "Slab",
    cylindrical: "Cylinder · radial section",
    spherical: "Sphere · radial section",
  }[domain.geometry];
  $("#experiment-name").value = state.model.name;
  $("#domain-title").textContent = geometry;
  $("#domain-caption").textContent =
    `1D ${domain.geometry === "cartesian" ? "Cartesian" : "radial"} domain · ${domain.cells} cells`;
  $("#domain-size").textContent = `${fmt(domain.length * 1000)} mm`;
  $("#domain-button").classList.toggle(
    "selected",
    state.selected === "@domain",
  );
  const art = $("#domain-art");
  art.replaceChildren();
  art.append(
    svgElement("rect", {
      x: 1,
      y: 4,
      width: 518,
      height: 56,
      rx: 3,
      fill: "#d2e5d4",
      stroke: "#76a280",
      "stroke-width": 1,
    }),
  );
  // A schematic of the mesh, never a substitute for result data.
  const cells = Math.min(28, Math.max(2, domain.cells || 2));
  for (let i = 1; i < cells; i++)
    art.append(
      svgElement("line", {
        x1: 1 + (i * 518) / cells,
        x2: 1 + (i * 518) / cells,
        y1: 4,
        y2: 60,
        stroke: "#8eb697",
        "stroke-width": 0.6,
      }),
    );
  art.append(
    svgElement("line", {
      x1: 0,
      x2: 520,
      y1: 32,
      y2: 32,
      stroke: "#729d7d",
      "stroke-width": 0.6,
      "stroke-dasharray": "3 4",
    }),
  );
  for (const side of ["left", "right"]) {
    const part = state.model.components.find(
      (item) =>
        slotOf(state.catalog, item) === "boundary" &&
        item.parameters.side === side,
    );
    const button = $(`#boundary-${side}`);
    $(`#${side}-caption`).textContent = part
      ? part.type === "boundary.sealed"
        ? state.model.components.some(
            (item) => slotOf(state.catalog, item) === "advection",
          )
          ? "Zero diffusive flux"
          : "Sealed"
        : part.type === "boundary.fixed"
          ? `c = ${fmt(part.parameters.value)}`
          : definitionOf(part.type).label
      : "Not set";
    button.classList.toggle("selected", part?.id === state.selected);
    button.onclick = () =>
      part ? select(part.id) : addComponent("boundary.fixed", side);
  }
  const root = $("#model-components");
  root.replaceChildren();
  // Keep every component reachable, including conflicting boundary edits.
  for (const part of state.model.components) {
    const definition = definitionOf(part.type);
    const button = element("button", "model-component");
    button.dataset.componentId = part.id;
    const side = slotOf(state.catalog, part) === "boundary" ? ` (${part.parameters.side})` : "";
    button.setAttribute("aria-label", `Edit ${definition?.label || part.type}${side}`);
    button.classList.toggle("selected", state.selected === part.id);
    button.append(icon(part.type));
    const text = element("span");
    text.append(
      element("strong", "", definition?.label || part.type),
      element("small", "", partSummary(part)),
    );
    button.append(text);
    button.onclick = () => select(part.id);
    root.append(button);
  }
  const terms = ["∇·(D∇c)"];
  if (
    state.model.components.some(
      (part) => slotOf(state.catalog, part) === "advection",
    )
  )
    terms.push("− ∇·(vc)");
  if (
    state.model.components.some(
      (part) =>
        categoryOf(definitionOf(part.type) || { type: "" }) === "reaction",
    )
  )
    terms.push("+ R(c)");
  $("#model-equation").textContent =
    `${state.model.run.mode === "steady" ? "0" : "∂c/∂t"} = ${terms.join(" ")}`;
}

function propertyInput(root, parameter, value, change) {
  const id = `parameter-${parameter.name}`;
  const label = element("label", "", parameter.label);
  label.htmlFor = id;
  if (parameter.unit) label.append(element("span", "", parameter.unit));
  const input = element(parameter.type === "choice" ? "select" : "input");
  input.id = id;
  if (parameter.type === "choice") {
    for (const choice of parameter.choices) {
      const option = element("option", "", choice);
      option.value = choice;
      input.append(option);
    }
  } else {
    input.type = "number";
    input.step = parameter.type === "integer" ? "1" : "any";
    if (parameter.min !== undefined) input.min = parameter.min;
    if (parameter.max !== undefined) input.max = parameter.max;
    input.required = true;
  }
  input.value = value ?? "";
  input.addEventListener("input", () =>
    change(
      parameter.type === "choice"
        ? input.value
        : Number.isFinite(input.valueAsNumber)
          ? input.valueAsNumber
          : null,
    ),
  );
  root.append(label, input);
}

function renderProperties() {
  const root = $("#properties");
  root.replaceChildren();
  const part = selectedPart();
  if (state.selected === "@domain" || !part) {
    state.selected = "@domain";
    const heading = element("div", "property-title");
    heading.append(icon("domain"), element("h3", "", "Transport domain"));
    root.append(
      heading,
      element(
        "p",
        "property-description",
        "Choose the geometry and resolution of the space where transport happens.",
      ),
    );
    const parameters = [
      {
        name: "geometry",
        label: "Geometry",
        type: "choice",
        choices: ["cartesian", "cylindrical", "spherical"],
      },
      {
        name: "length",
        label: "Length / outer radius",
        unit: "m",
        type: "number",
        min: 0,
      },
      {
        name: "cells",
        label: "Mesh cells",
        type: "integer",
        min: 2,
        max: 1000,
      },
    ];
    for (const parameter of parameters)
      propertyInput(
        root,
        parameter,
        state.model.domain[parameter.name],
        (value) =>
          mutate(() => {
            state.model.domain[parameter.name] = value;
          }),
      );
  } else {
    const definition = definitionOf(part.type);
    const heading = element("div", "property-title");
    heading.append(icon(part.type), element("h3", "", definition.label));
    root.append(
      heading,
      element("p", "property-description", definition.description),
    );
    for (const parameter of definition.parameters)
      propertyInput(root, parameter, part.parameters[parameter.name], (value) =>
        mutate(() => {
          part.parameters[parameter.name] = value;
        }),
      );
    const remove = element("button", "remove-button", "Remove component");
    remove.onclick = () => {
      mutate(() => {
        state.model.components = state.model.components.filter(
          (item) => item.id !== part.id,
        );
        state.selected = "@domain";
      });
      renderProperties();
      say(`${definition.label} removed.`);
    };
    root.append(remove);
  }
}

function renderSettings() {
  $("#duration").value = state.model.run.duration;
  $("#frames").value = state.model.run.frames;
  $("#run-mode").value = state.model.run.mode;
  $("#time-settings").hidden = state.model.run.mode === "steady";
  $("#method-note").textContent =
    state.model.run.mode === "steady"
      ? "Solve the final balance directly. Steady state has no physical time."
      : "The engine chooses a stable time step for your model.";
}

function showError(error) {
  const node = $("#run-error");
  node.replaceChildren(
    element(
      "strong",
      "",
      error.issues?.length
        ? "Check these model settings before running."
        : error.error || error.message || "The experiment could not run.",
    ),
  );
  if (error.issues?.length) {
    const list = element("ul");
    for (const issue of error.issues)
      list.append(element("li", "", `${issue.path}: ${issue.message}`));
    node.append(list);
  }
  node.hidden = false;
  $("#run-status").textContent = "Check the highlighted message and try again";
}

async function api(path, payload) {
  const response = await fetch(
    path,
    payload
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      : undefined,
  );
  const body = await response.json();
  if (!response.ok) throw body;
  return body;
}

function issueLabel(path) {
  const names = {
    "domain.cells": "Number of cells",
    "domain.length": "Domain length",
    "domain.geometry": "Geometry",
    "run.mode": "Solution method",
    "run.frames": "Saved frames",
    "run.duration": "Duration",
  };
  if (names[path]) return names[path];
  const match = path.match(/^components\[(\d+)\](?:\.parameters\.(.+))?/);
  if (match) {
    const part = state.model.components[Number(match[1])];
    const definition = definitionOf(part?.type);
    const parameter = definition?.parameters.find((p) => p.name === match[2]);
    return [definition?.label || part?.id || "Component", parameter?.label]
      .filter(Boolean).join(" · ");
  }
  return path === "components" ? "Model components" : "Model";
}

async function refreshPlan() {
  clearTimeout(state.planTimer);
  if (!state.model) return;
  const revision = state.revision;
  try {
    const response = await api("/api/plan", clone(state.model));
    if (revision !== state.revision) return;
    state.plan = response;
    state.planRevision = revision;
    const { plan } = response;
    const panel = $("#run-plan");
    panel.replaceChildren();
    const schedule = plan.mode === "steady"
      ? "Direct steady solve"
      : `${plan.planned_steps.toLocaleString()} time steps`;
    panel.append(element("strong", "plan-schedule", schedule));
    panel.append(element("p", "", `${plan.nodes.toLocaleString()} positions · ${plan.saved_states} saved ${plan.saved_states === 1 ? "state" : "states"}`));
    if (plan.selected_time_step !== null)
      panel.append(element("p", "", `Step size up to ${fmt(plan.selected_time_step)} s`));
    if (response.issues.length) {
      const issues = element("ul", "plan-issues");
      for (const issue of response.issues)
        issues.append(element("li", "", `${issueLabel(issue.path)}: ${issue.message}`));
      panel.append(issues);
    }
    $("#run-button").disabled = state.busy || !response.runnable;
    if (!state.busy)
      $("#run-status").textContent = response.runnable
        ? "Ready to simulate" : "Adjust the model before running";
  } catch (error) {
    if (revision !== state.revision) return;
    state.plan = null;
    state.planRevision = revision;
    const panel = $("#run-plan");
    panel.replaceChildren(element("strong", "plan-schedule", "Model needs attention"));
    const issues = element("ul", "plan-issues");
    for (const issue of error.issues?.length ? error.issues : [{ path: "", message: error.error || error.message || "Could not check this model. Reload to reconnect." }])
      issues.append(element("li", "", `${issueLabel(issue.path)}: ${issue.message}`));
    panel.append(issues);
    $("#run-button").disabled = true;
    if (!state.busy) $("#run-status").textContent = "Adjust the model before running";
  }
}

async function runSimulation() {
  if (state.busy || !state.model) return;
  if (state.planRevision !== state.revision) await refreshPlan();
  if (state.busy || !state.plan?.runnable || state.planRevision !== state.revision) return;
  stopPlayback();
  state.busy = true;
  $("#run-button").disabled = true;
  $("#run-button span").textContent = "Simulating…";
  $("#run-status").textContent = "Computing with the transport engine…";
  $("#run-error").hidden = true;
  document.body.classList.add("running");
  const revision = state.revision;
  try {
    const result = await api("/api/run", clone(state.model));
    state.result = result;
    state.resultRevision = revision;
    state.frame = result.solution.times.length - 1;
    $("#results-empty").hidden = true;
    $("#results-content").hidden = false;
    $("#csv-button").disabled = false;
    $("#result-caption").textContent =
      revision === state.revision
        ? `${result.experiment.name} · ${result.solution.steady ? "Steady solution" : `${result.solution.times.length} saved states`}`
        : "Previous run · Your model changed while it was running. Run again to update.";
    $("#run-status").textContent =
      `Solved in ${fmt(result.solution.elapsed_seconds * 1000)} ms`;
    $("#edit-state").textContent =
      revision === state.revision ? "Simulated" : "Edited";
    $("#time-slider").max = result.solution.times.length - 1;
    $("#time-slider").disabled = result.solution.steady;
    $("#play-button").disabled =
      result.solution.steady || result.solution.times.length < 2;
    $("#history-tab").disabled = result.solution.steady;
    if (result.solution.steady) setView("profile");
    renderChecks();
    renderFrame();
    say("Simulation complete. Results are available.");
  } catch (error) {
    showError(error);
  } finally {
    state.busy = false;
    $("#run-button").disabled = !state.plan?.runnable || state.planRevision !== state.revision;
    $("#run-button span").textContent = "Run simulation";
    document.body.classList.remove("running");
  }
}

function renderChecks() {
  const { solution, reference, experiment } = state.result;
  const checks = $("#result-checks");
  checks.replaceChildren();
  const check = (label, value, note) => {
    const node = element("div", "check-item");
    node.append(
      element("small", "", label),
      element("strong", "", value),
      element("p", "", note),
    );
    checks.append(node);
  };
  if (reference)
    check(
      "Reference comparison",
      `${fmt(reference.max_abs_error)} max error`,
      "Absolute error at the final frame",
    );
  else
    check(
      "Reference comparison",
      "No exact overlay",
      "Check model-specific evidence and refinement",
    );
  const closed = isClosedDiffusion(experiment);
  const delta = solution.totals.at(-1) - solution.totals[0];
  if (solution.steady)
    check(
      "Integrated concentration",
      fmt(solution.totals.at(-1)),
      "Volume-weighted steady inventory",
    );
  else
    check(
      closed ? "Sealed-domain conservation" : "Inventory change",
      `${delta > 0 ? "+" : ""}${fmt(delta)}`,
      closed
        ? "Final minus initial integrated concentration"
        : "Boundaries and reactions can change the total",
    );
  check(
    "Solver work",
    solution.steady
      ? "Steady balance"
      : `${solution.steps.toLocaleString()} steps`,
    `${fmt(solution.elapsed_seconds * 1000)} ms measured solve time`,
  );
  const groups = $("#dimensionless");
  groups.replaceChildren();
  for (const group of solution.dimensionless)
    groups.append(
      element(
        "dt",
        "",
        `${group.name} = ${group.value === null ? "∞" : fmt(group.value)}`,
      ),
      element("dd", "", group.description),
    );
  if (!solution.dimensionless.length)
    groups.append(
      element(
        "dd",
        "",
        "No dimensionless groups are available for this configuration.",
      ),
    );
  $("#solver-report").textContent = solution.summary;
}

function plotProfile() {
  const { solution, reference } = state.result;
  const values = solution.fields[state.frame];
  const plot = $("#profile-plot");
  plot.replaceChildren();
  const area = { left: 61, right: 778, top: 24, bottom: 246 };
  const span = solution.maximum - solution.minimum;
  const padding =
    span > 0 ? span * 0.07 : Math.max(Math.abs(solution.maximum) * 0.1, 1e-12);
  const low = solution.minimum - padding,
    high = solution.maximum + padding;
  const xpos = (x) =>
    area.left +
    ((x - solution.x[0]) / (solution.x.at(-1) - solution.x[0])) *
      (area.right - area.left);
  const ypos = (c) =>
    area.bottom - ((c - low) / (high - low)) * (area.bottom - area.top);
  const title = svgElement("title");
  title.textContent = `${solution.steady ? "Steady" : `Time ${fmt(solution.times[state.frame])} s`}: concentration from ${fmt(Math.min(...values))} to ${fmt(Math.max(...values))}`;
  plot.append(title);
  const text = (x, y, value, anchor = "middle") => {
    const node = svgElement("text", { x, y, "text-anchor": anchor });
    node.textContent = value;
    plot.append(node);
  };
  for (let i = 0; i <= 4; i++) {
    const y = area.top + (i * (area.bottom - area.top)) / 4;
    plot.append(
      svgElement("line", {
        x1: area.left,
        x2: area.right,
        y1: y,
        y2: y,
        class: "plot-grid",
      }),
    );
    text(area.left - 12, y + 4, fmt(high - (i * (high - low)) / 4), "end");
  }
  for (let i = 0; i <= 5; i++) {
    const x = area.left + (i * (area.right - area.left)) / 5;
    text(
      x,
      area.bottom + 20,
      fmt(
        (solution.x[0] + (i * (solution.x.at(-1) - solution.x[0])) / 5) * 1000,
      ),
    );
  }
  text((area.left + area.right) / 2, 283, "Position (mm)");
  text(area.left, 12, "Concentration (model units)", "start");
  const path = (field) =>
    field
      .map(
        (value, index) =>
          `${index ? "L" : "M"}${xpos(solution.x[index]).toFixed(2)},${ypos(value).toFixed(2)}`,
      )
      .join(" ");
  plot.append(
    svgElement("path", { d: path(values), class: "simulation-curve" }),
  );
  if (reference)
    plot.append(
      svgElement("path", {
        d: path(reference.fields[state.frame]),
        class: "reference-curve",
      }),
    );
}

function plotHistory() {
  const { solution } = state.result;
  const canvas = $("#history-plot");
  const ctx = canvas.getContext("2d");
  const width = solution.x.length,
    height = solution.fields.length;
  canvas.width = width;
  canvas.height = height;
  const pixels = ctx.createImageData(width, height);
  const range = solution.maximum - solution.minimum || 1;
  for (let row = 0; row < height; row++)
    for (let col = 0; col < width; col++) {
      const v = (solution.fields[row][col] - solution.minimum) / range;
      const offset = ((height - 1 - row) * width + col) * 4;
      const light = [235, 245, 224],
        dark = [18, 91, 76];
      for (let channel = 0; channel < 3; channel++)
        pixels.data[offset + channel] =
          light[channel] + v * (dark[channel] - light[channel]);
      pixels.data[offset + 3] = 255;
    }
  ctx.putImageData(pixels, 0, 0);
  $("#heatmap-time").textContent = `Time ↑ 0–${fmt(solution.times.at(-1))} s`;
  $("#heatmap-scale").textContent =
    `Light ${fmt(solution.minimum)} → dark ${fmt(solution.maximum)}`;
}

function renderFrame() {
  if (!state.result) return;
  const { solution, reference } = state.result;
  $("#time-slider").value = state.frame;
  $("#time-readout").textContent = solution.steady
    ? "Steady"
    : `${fmt(solution.times[state.frame])} s`;
  $("#plot-time").textContent = solution.steady
    ? "Steady state"
    : state.view === "history"
      ? `0–${fmt(solution.times.at(-1))} s`
      : `t = ${fmt(solution.times[state.frame])} s`;
  $("#reference-legend").hidden = !reference || state.view === "history";
  if (state.view === "profile") plotProfile();
  else plotHistory();
  const rows = document.createDocumentFragment();
  const values = solution.fields[state.frame];
  for (let index = 0; index < solution.x.length; index++) {
    const row = element("tr");
    row.append(
      element("td", "", fmt(solution.x[index], 8)),
      element("td", "", fmt(values[index], 8)),
    );
    rows.append(row);
  }
  $("#result-table tbody").replaceChildren(rows);
}

function setView(view) {
  state.view = view;
  for (const name of ["profile", "history"]) {
    $(`#${name}-tab`).setAttribute("aria-selected", String(name === view));
    $(`#${name}-tab`).tabIndex = name === view ? 0 : -1;
  }
  $("#plot-panel").setAttribute("aria-labelledby", `${view}-tab`);
  $("#profile-plot").hidden = view !== "profile";
  $("#heatmap-wrap").hidden = view !== "history";
  $(".playback").hidden = view === "history";
  if (view === "history") stopPlayback();
  renderFrame();
}

function stopPlayback() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  $("#play-button").textContent = "Play";
  $("#play-button").setAttribute("aria-label", "Play saved frames");
}

function download(contents, filename, mime) {
  const url = URL.createObjectURL(new Blob([contents], { type: mime }));
  const anchor = element("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function wireEvents() {
  $("#help-button").onclick = () => {
    const open = $("#help").hidden;
    $("#help").hidden = !open;
    $("#help-button").setAttribute("aria-expanded", String(open));
  };
  $("#domain-button").onclick = () => select("@domain");
  $("#experiment-name").oninput = (event) =>
    mutate(() => {
      state.model.name = event.target.value;
    });
  $("#run-mode").onchange = (event) => {
    mutate(() => {
      state.model.run.mode = event.target.value;
    });
    renderSettings();
  };
  for (const [selector, key] of [
    ["#duration", "duration"],
    ["#frames", "frames"],
  ])
    $(selector).oninput = (event) =>
      mutate(() => {
        state.model.run[key] = Number.isFinite(event.target.valueAsNumber)
          ? event.target.valueAsNumber
          : null;
      }, false);
  $("#undo-button").onclick = () => {
    const old = state.undo.pop();
    if (old) {
      state.model = old.model;
      state.selected = old.selected;
      markChanged();
      renderModel();
      renderProperties();
      renderSettings();
      say("Last model change undone.");
    }
  };
  $("#example-select").onchange = (event) => {
    mutate(() => {
      state.model = clone(state.examples[Number(event.target.value)]);
      state.selected = "@domain";
    });
    renderProperties();
    renderSettings();
  };
  $("#run-button").onclick = runSimulation;
  $("#profile-tab").onclick = () => setView("profile");
  $("#history-tab").onclick = () => setView("history");
  $(".result-tabs").onkeydown = (event) => {
    if (
      ["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key) &&
      !$("#history-tab").disabled
    ) {
      event.preventDefault();
      setView(state.view === "profile" ? "history" : "profile");
      $(`#${state.view}-tab`).focus();
    }
  };
  $("#time-slider").oninput = (event) => {
    stopPlayback();
    state.frame = Number(event.target.value);
    renderFrame();
  };
  $("#play-button").onclick = () => {
    if (state.timer) {
      stopPlayback();
      return;
    }
    if (state.frame === state.result.solution.times.length - 1) state.frame = 0;
    $("#play-button").textContent = "Pause";
    $("#play-button").setAttribute("aria-label", "Pause saved frames");
    state.timer = setInterval(() => {
      renderFrame();
      if (state.frame >= state.result.solution.times.length - 1) stopPlayback();
      else state.frame++;
    }, 160);
  };
  $("#export-button").onclick = async () => {
    try {
      const validated = await api("/api/validate", state.model);
      download(
        JSON.stringify(validated.experiment, null, 2) + "\n",
        "experiment.json",
        "application/json",
      );
      say("Experiment JSON saved.");
    } catch (error) {
      showError(error);
    }
  };
  $("#import-button").onclick = () => $("#import-file").click();
  $("#import-file").onchange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      if (file.size > 262144)
        throw new Error("Experiment files must be smaller than 256 KB.");
      const data = await api("/api/validate", JSON.parse(await file.text()));
      mutate(() => {
        state.model = data.experiment;
        state.selected = "@domain";
      });
      renderProperties();
      renderSettings();
      say("Experiment opened.");
    } catch (error) {
      showError(error);
    }
    event.target.value = "";
  };
  $("#csv-button").onclick = () => {
    const { solution } = state.result;
    const rows = ["time_s,position_m,concentration\n"];
    solution.fields.forEach((field, i) =>
      field.forEach((value, j) =>
        rows.push(
          `${solution.steady ? "" : solution.times[i]},${solution.x[j]},${value}\n`,
        ),
      ),
    );
    download(rows.join(""), "biotransport-results.csv", "text/csv");
  };
  connectPaletteDrag($("#component-library"), $("#model-canvas"), addComponent);
}

async function init() {
  wireEvents();
  try {
    const data = await api("/api/catalog");
    state.catalog = data.components;
    state.examples = data.examples;
    state.model = clone(data.examples[0]);
    const choices = $("#example-select");
    choices.replaceChildren();
    data.examples.forEach((example, i) => {
      const option = element("option", "", example.name);
      option.value = i;
      choices.append(option);
    });
    renderLibrary();
    renderModel();
    renderProperties();
    renderSettings();
    await refreshPlan();
    await runSimulation();
  } catch (error) {
    showError(error);
    $("#run-status").textContent =
      "Could not connect. Restart the local workbench and reload.";
  }
}

init();
