// Editor operations consume engine metadata. Custom types need no UI switch.
export const categoryOf = (definition) => definition.category || "other";

export function componentAvailability(definition, model) {
  if (definition.geometries && !definition.geometries.includes(model.domain.geometry))
    return `Available for ${definition.geometries.join(", ")} geometry`;
  if (model.run.mode === "steady" && definition.supports_steady === false)
    return "Available in evolution over time";
  return null;
}

export function slotOf(catalog, part) {
  return catalog.find((definition) => definition.type === part?.type)?.slot;
}

export function addModelComponent(model, catalog, type, selected, newId, side) {
  const definition = catalog.find((item) => item.type === type);
  if (!definition) throw new Error(`Unknown component: ${type}`);
  const next = structuredClone(model);
  const parameters = Object.fromEntries(
    definition.parameters.map((parameter) => [
      parameter.name,
      structuredClone(parameter.default),
    ]),
  );
  if (definition.slot === "boundary") {
    const current = next.components.find((part) => part.id === selected);
    if (side) parameters.side = side;
    else if (slotOf(catalog, current) === "boundary")
      parameters.side = current.parameters.side;
  }
  const exclusive =
    definition.slot &&
    next.components.find(
      (part) =>
        slotOf(catalog, part) === definition.slot &&
        (definition.slot !== "boundary" ||
          part.parameters.side === parameters.side),
    );
  const part = { id: exclusive?.id || newId, type, parameters };
  if (exclusive)
    next.components.splice(next.components.indexOf(exclusive), 1, part);
  else next.components.push(part);
  return { model: next, selected: part.id };
}

export function isClosedDiffusion(document) {
  // Unknown additive extensions can consume or produce material. Only the
  // known pure diffusion configuration earns a conservation label here.
  const types = document.components.map((part) => part.type);
  return (
    types.filter((type) => type === "boundary.sealed").length === 2 &&
    types.every((type) =>
      [
        "diffusion",
        "initial.uniform",
        "initial.gaussian",
        "boundary.sealed",
      ].includes(type),
    )
  );
}
