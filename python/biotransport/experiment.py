"""Declarative experiments shared by Python extensions and visual editors.

An experiment is a small JSON document, not another numerical solver. It builds
the same :class:`~biotransport.Problem` used in Python and delegates execution to
``solve`` or ``solve_steady``. Values use metres, seconds, and one consistent
concentration unit chosen by the author. Parameter defaults describe the editor
palette; saved experiments must explicitly supply every parameter.

Example::

    from biotransport.experiment import Experiment

    experiment = Experiment.from_dict({
        "schema_version": 1,
        "name": "Oxygen in tissue",
        "domain": {"geometry": "cartesian", "length": 0.01, "cells": 120},
        "components": [
            {"id": "diffusion", "type": "diffusion",
             "parameters": {"coefficient": 1e-9}},
            {"id": "initial", "type": "initial.uniform",
             "parameters": {"value": 0}},
            {"id": "left", "type": "boundary.fixed",
             "parameters": {"side": "left", "value": 1}},
            {"id": "right", "type": "boundary.sealed",
             "parameters": {"side": "right"}},
        ],
        "run": {"mode": "transient", "duration": 2000, "frames": 40},
    })
    solution = experiment.run()
    solution.plot()

Extensions are ordinary, explicitly registered Python functions. JSON can only
select a registered type; it cannot contain code, module names, or expressions::

    registry = builtin_registry()
    registry.register(ComponentDefinition(
        type="lab.uptake", label="Uptake", description="First-order uptake.",
        category="reaction",
        parameters=(Parameter("rate", "Rate", "number", 0.001, min=0),),
        apply=lambda problem, values: problem.add_linear_decay(values["rate"]),
        supports_steady=True,
    ))
    experiment = Experiment.from_dict(document, registry=registry)

Registries are local to their owner. Definitions are immutable, and an
experiment snapshots its document and selected definitions at validation time.
An extension callback is trusted Python code and must preserve additive terms,
declare any reaction derivative bound, and accurately state its supported scope.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from .initial_conditions import gaussian
from .mesh_utils import mesh_1d
from .problem import Problem
from .run import _saved_times, solve
from .solution import Solution
from .steady import solve_steady

__all__ = [
    "Parameter",
    "ComponentDefinition",
    "ComponentRegistry",
    "ExperimentValidationError",
    "Experiment",
    "ExperimentPlan",
    "builtin_registry",
]

_GEOMETRIES = ("cartesian", "cylindrical", "spherical")
_SLOTS = ("diffusion", "initial", "boundary", "advection")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


@dataclass(frozen=True)
class ExperimentPlan:
    """Cost and storage for the selected run, without advancing the solution.

    Transient step counts use the native planner for every saved interval,
    including its reaction accuracy guard and floating-point endpoint rule.
    ``selected_time_step`` is the largest nominal cap across these intervals;
    the last step of an interval can be shorter. Steady plans have no time step
    or explicit-step count and do not predict Newton convergence.

    ``field_storage_bytes`` counts saved float64 concentration arrays only,
    excluding working memory, coordinates, Python objects and JSON encoding.
    """

    mode: str
    geometry: str
    cells: int
    nodes: int
    saved_states: int
    planned_steps: int | None
    selected_time_step: float | None
    max_steps: int
    within_step_budget: bool
    field_storage_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return independent, finite JSON-compatible planning data."""
        return asdict(self)


class ExperimentValidationError(ValueError):
    """Invalid experiment, with ``[{"path": ..., "message": ...}]`` for an editor."""

    def __init__(self, issues: Iterable[Mapping[str, str]]) -> None:
        self.issues = [dict(issue) for issue in issues]
        super().__init__(
            "; ".join(f"{issue['path']}: {issue['message']}" for issue in self.issues)
        )


def _finite_number(value: Any) -> bool:
    # JSON booleans are Python ints, and coercion accepts strings and arrays.
    # Keep the on-disk/API format strictly JSON numeric.
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class Parameter:
    """One editor field and its validation rule.

    ``type`` is ``"number"`` or ``"choice"``. Bounds are inclusive unless
    ``exclusive_min`` is true. Defaults are palette suggestions, never silently
    inserted into an experiment. Choice values and numeric defaults are scalar.
    """

    name: str
    label: str
    type: str
    default: Any
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] = ()
    unit: str | None = None
    exclusive_min: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _IDENTIFIER.fullmatch(self.name):
            raise ValueError("parameter name must be a short identifier")
        if not _text(self.label):
            raise ValueError("parameter label must be nonempty text")
        if self.type not in ("number", "choice"):
            raise ValueError("parameter type must be 'number' or 'choice'")
        if self.unit is not None and not _text(self.unit):
            raise ValueError("parameter unit must be nonempty text when present")
        if type(self.exclusive_min) is not bool:
            raise ValueError("exclusive_min must be a boolean")
        if isinstance(self.choices, str):
            raise ValueError("choices must be a sequence of strings")
        object.__setattr__(self, "choices", tuple(self.choices))
        if self.type == "number":
            if self.choices:
                raise ValueError("numeric parameters cannot have choices")
            for bound in (self.min, self.max):
                if bound is not None and not _finite_number(bound):
                    raise ValueError("parameter bounds must be finite numbers")
            if self.min is not None and self.max is not None and self.min > self.max:
                raise ValueError("parameter minimum must not exceed its maximum")
            if self.exclusive_min and self.min is None:
                raise ValueError("exclusive_min requires a minimum")
        else:
            if not self.choices or not all(_text(choice) for choice in self.choices):
                raise ValueError("choice parameters need nonempty string choices")
            if len(set(self.choices)) != len(self.choices):
                raise ValueError("parameter choices must be unique")
            if self.min is not None or self.max is not None or self.exclusive_min:
                raise ValueError("choice parameters cannot have numeric bounds")
        error = self._error(self.default)
        if error:
            raise ValueError(f"invalid default for {self.name}: {error}")

    def _error(self, value: Any) -> str | None:
        if self.type == "choice":
            if not isinstance(value, str) or value not in self.choices:
                return f"choose one of: {', '.join(self.choices)}"
            return None
        if not _finite_number(value):
            return "must be a finite number (booleans and numeric strings are invalid)"
        if self.min is not None:
            if self.exclusive_min and value <= self.min:
                return f"must be greater than {self.min:g}"
            if value < self.min:
                return f"must be at least {self.min:g}"
        if self.max is not None and value > self.max:
            return f"must be at most {self.max:g}"
        return None

    def _catalog(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "default": self.default,
        }
        for key in ("min", "max", "unit"):
            value = getattr(self, key)
            if value is not None:
                entry[key] = value
        if self.choices:
            entry["choices"] = list(self.choices)
        if self.exclusive_min:
            entry["exclusive_min"] = True
        return entry


@dataclass(frozen=True)
class ComponentDefinition:
    """An explicit physics extension, its editor fields, and its compiler step.

    ``apply(problem, parameters)`` configures the existing ``Problem``; its return
    value is ignored. ``slot`` identifies an exclusive role: ``diffusion``,
    ``initial``, ``boundary`` (one per ``side``), or ``advection``. Leave it unset
    for additive components such as reactions. Steady support is opt-in because
    the current steady path cannot differentiate arbitrary Python reactions.
    """

    type: str
    label: str
    description: str
    category: str
    parameters: tuple[Parameter, ...]
    apply: Callable[[Problem, Mapping[str, Any]], Any]
    slot: str | None = None
    supports_steady: bool = False
    geometries: tuple[str, ...] = _GEOMETRIES

    def __post_init__(self) -> None:
        if not isinstance(self.type, str) or not _IDENTIFIER.fullmatch(self.type):
            raise ValueError("component type must be a short identifier")
        if not all(
            _text(value) for value in (self.label, self.description, self.category)
        ):
            raise ValueError(
                "component label, description and category must be nonempty"
            )
        if not callable(self.apply):
            raise TypeError("component apply must be callable")
        if self.slot is not None and self.slot not in _SLOTS:
            raise ValueError(f"component slot must be one of {_SLOTS}, or None")
        if type(self.supports_steady) is not bool:
            raise ValueError("supports_steady must be a boolean")
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if not all(isinstance(parameter, Parameter) for parameter in self.parameters):
            raise TypeError("component parameters must be Parameter definitions")
        names = [parameter.name for parameter in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("component parameter names must be unique")
        if self.slot == "boundary":
            side = next((p for p in self.parameters if p.name == "side"), None)
            if (
                side is None
                or side.type != "choice"
                or set(side.choices) != {"left", "right"}
            ):
                raise ValueError(
                    "boundary components need a side choice of left or right"
                )
        if isinstance(self.geometries, str):
            raise ValueError("geometries must be a sequence of supported geometries")
        object.__setattr__(self, "geometries", tuple(self.geometries))
        if not self.geometries or any(g not in _GEOMETRIES for g in self.geometries):
            raise ValueError(
                f"component geometries must be selected from {_GEOMETRIES}"
            )


class ComponentRegistry:
    """A local catalog of trusted definitions; registration never replaces a type."""

    def __init__(self, definitions: Iterable[ComponentDefinition] = ()) -> None:
        self._definitions: dict[str, ComponentDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: ComponentDefinition) -> ComponentRegistry:
        """Register an immutable definition on this registry, returning ``self``."""
        if not isinstance(definition, ComponentDefinition):
            raise TypeError("register expects a ComponentDefinition")
        if definition.type in self._definitions:
            raise ValueError(
                f"component type {definition.type!r} is already registered"
            )
        self._definitions[definition.type] = definition
        return self

    def catalog(self) -> list[dict[str, Any]]:
        """Return independent JSON data suitable for rendering a component palette."""
        return [
            {
                "type": definition.type,
                "label": definition.label,
                "description": definition.description,
                "category": definition.category,
                "parameters": [
                    parameter._catalog() for parameter in definition.parameters
                ],
                "slot": definition.slot,
                "supports_steady": definition.supports_steady,
                "geometries": list(definition.geometries),
            }
            for definition in self._definitions.values()
        ]


def builtin_registry() -> ComponentRegistry:
    """Create an independent registry of the verified one-dimensional building blocks."""
    side = Parameter("side", "Side", "choice", "left", choices=("left", "right"))

    def concentration(name: str, label: str, default: float) -> Parameter:
        return Parameter(name, label, "number", default, min=0, unit="concentration")

    return ComponentRegistry(
        (
            ComponentDefinition(
                "diffusion",
                "Diffusion",
                "Transport down concentration gradients.",
                "transport",
                (
                    Parameter(
                        "coefficient", "Diffusivity", "number", 1e-9, min=0, unit="m²/s"
                    ),
                ),
                lambda problem, values: problem.diffusivity(values["coefficient"]),
                slot="diffusion",
                supports_steady=True,
            ),
            ComponentDefinition(
                "initial.uniform",
                "Uniform start",
                "Start with the same concentration everywhere.",
                "initial",
                (concentration("value", "Initial concentration", 0),),
                lambda problem, values: problem.initial(values["value"]),
                slot="initial",
                supports_steady=True,
            ),
            ComponentDefinition(
                "initial.gaussian",
                "Gaussian pulse",
                "Start with a localized concentration pulse.",
                "initial",
                (
                    Parameter(
                        "center", "Pulse center", "number", 0.005, min=0, unit="m"
                    ),
                    Parameter(
                        "width",
                        "Standard deviation",
                        "number",
                        0.001,
                        min=0,
                        exclusive_min=True,
                        unit="m",
                    ),
                    concentration("amplitude", "Peak concentration", 1),
                ),
                lambda problem, values: problem.initial(
                    gaussian(problem.mesh(), **values)
                ),
                slot="initial",
                supports_steady=True,
            ),
            ComponentDefinition(
                "boundary.fixed",
                "Fixed concentration",
                "Hold one face at a prescribed concentration.",
                "boundary",
                (side, concentration("value", "Concentration", 1)),
                lambda problem, values: problem.dirichlet(
                    values["side"], values["value"]
                ),
                slot="boundary",
                supports_steady=True,
            ),
            ComponentDefinition(
                "boundary.sealed",
                "Zero diffusive flux",
                "No diffusive flux through this face; advection can still carry material through.",
                "boundary",
                (side,),
                lambda problem, values: problem.sealed(values["side"]),
                slot="boundary",
                supports_steady=True,
            ),
            ComponentDefinition(
                "reaction.decay",
                "First-order decay",
                "Consume concentration at rate R = −k c; additive with other reactions.",
                "reaction",
                (Parameter("rate", "Decay rate", "number", 0.001, min=0, unit="1/s"),),
                lambda problem, values: problem.add_linear_decay(values["rate"]),
                supports_steady=True,
            ),
            ComponentDefinition(
                "reaction.uptake",
                "Saturable uptake",
                "Michaelis–Menten consumption: uptake approaches a maximum as concentration rises.",
                "reaction",
                (
                    Parameter(
                        "Vmax",
                        "Maximum uptake",
                        "number",
                        0.0001,
                        min=0,
                        unit="concentration/s",
                    ),
                    Parameter(
                        "Km",
                        "Half-saturation concentration",
                        "number",
                        0.25,
                        min=0,
                        exclusive_min=True,
                        unit="concentration",
                    ),
                ),
                lambda problem, values: problem.add_michaelis_menten(**values),
                supports_steady=True,
            ),
            ComponentDefinition(
                "reaction.source",
                "Uniform source",
                "Produce concentration at a constant rate; additive with other reactions.",
                "reaction",
                (
                    Parameter(
                        "rate",
                        "Production rate",
                        "number",
                        0.0001,
                        min=0,
                        unit="concentration/s",
                    ),
                ),
                lambda problem, values: problem.add_constant_source(values["rate"]),
                supports_steady=True,
            ),
            ComponentDefinition(
                "advection",
                "Uniform flow",
                "Carry concentration with a signed velocity using the verified upwind scheme.",
                "transport",
                (Parameter("velocity", "Velocity", "number", 1e-6, unit="m/s"),),
                lambda problem, values: problem.velocity(values["velocity"]),
                slot="advection",
                geometries=("cartesian",),
            ),
        )
    )


def _object(
    value: Any, path: str, required: tuple[str, ...], issues: list[dict[str, str]]
) -> bool:
    if not isinstance(value, dict):
        issues.append({"path": path, "message": "must be a JSON object"})
        return False
    for name in required:
        if name not in value:
            issues.append({"path": f"{path}.{name}", "message": "is required"})
    for name in value:
        if name not in required:
            issues.append({"path": f"{path}.{name}", "message": "unknown field"})
    return True


def _validate(
    payload: Any, registry: ComponentRegistry
) -> tuple[ComponentDefinition, ...]:
    issues: list[dict[str, str]] = []

    def issue(path: str, message: str) -> None:
        issues.append({"path": path, "message": message})

    if not _object(
        payload, "$", ("schema_version", "name", "domain", "components", "run"), issues
    ):
        raise ExperimentValidationError(issues)
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        issue("schema_version", "must be the integer 1")
    if not _text(payload.get("name")) or len(payload["name"]) > 120:
        issue("name", "must be nonempty text of at most 120 characters")

    domain = payload.get("domain")
    if _object(domain, "domain", ("geometry", "length", "cells"), issues):
        if (
            not isinstance(domain.get("geometry"), str)
            or domain["geometry"] not in _GEOMETRIES
        ):
            issue("domain.geometry", "choose cartesian, cylindrical or spherical")
        length = domain.get("length")
        if not _finite_number(length) or length <= 0:
            issue("domain.length", "must be a finite positive length in metres")
        cells = domain.get("cells")
        if type(cells) is not int or not 2 <= cells <= 2000:
            issue("domain.cells", "must be an integer between 2 and 2000")

    run = payload.get("run")
    if _object(run, "run", ("mode", "duration", "frames"), issues):
        if not isinstance(run.get("mode"), str) or run["mode"] not in (
            "transient",
            "steady",
        ):
            issue("run.mode", "choose transient or steady")
        if not _finite_number(run.get("duration")) or run["duration"] <= 0:
            issue("run.duration", "must be a finite positive duration in seconds")
        if type(run.get("frames")) is not int or not 1 <= run["frames"] <= 200:
            issue("run.frames", "must be an integer between 1 and 200")

    components = payload.get("components")
    definitions: list[ComponentDefinition] = []
    ids: set[str] = set()
    if not isinstance(components, list) or not 4 <= len(components) <= 32:
        issue("components", "must be a list containing between 4 and 32 components")
    else:
        for index, component in enumerate(components):
            path = f"components[{index}]"
            if not _object(component, path, ("id", "type", "parameters"), issues):
                continue
            identifier = component.get("id")
            if not isinstance(identifier, str) or not _IDENTIFIER.fullmatch(identifier):
                issue(
                    f"{path}.id",
                    "use 1–80 letters, digits, dots, dashes or underscores; begin with a letter or digit",
                )
            elif identifier in ids:
                issue(f"{path}.id", "component IDs must be unique")
            else:
                ids.add(identifier)
            kind = component.get("type")
            definition = (
                registry._definitions.get(kind) if isinstance(kind, str) else None
            )
            if definition is None:
                issue(
                    f"{path}.type",
                    "unknown component type; register a definition explicitly",
                )
                continue
            definitions.append(definition)
            parameters = component.get("parameters")
            names = tuple(parameter.name for parameter in definition.parameters)
            if not _object(parameters, f"{path}.parameters", names, issues):
                continue
            for parameter in definition.parameters:
                if parameter.name in parameters:
                    error = parameter._error(parameters[parameter.name])
                    if error:
                        issue(f"{path}.parameters.{parameter.name}", error)

    if issues:
        raise ExperimentValidationError(issues)

    slots: dict[str, list[int]] = {}
    for index, (component, definition) in enumerate(zip(components, definitions)):
        slot = definition.slot
        if slot == "boundary":
            slot = f"boundary.{component['parameters']['side']}"
        if slot:
            slots.setdefault(slot, []).append(index)
        if domain["geometry"] not in definition.geometries:
            issue(
                f"components[{index}].type",
                f"{definition.label} does not support {domain['geometry']} geometry",
            )
        if run["mode"] == "steady" and not definition.supports_steady:
            issue(
                f"components[{index}].type",
                f"{definition.label} does not support steady solving; use transient mode",
            )

    for slot in ("diffusion", "initial", "boundary.left", "boundary.right"):
        count = len(slots.get(slot, []))
        if count != 1:
            issue(
                "components", f"exactly one {slot} component is required; found {count}"
            )
    if len(slots.get("advection", [])) > 1:
        issue("components", "at most one advection component is allowed")

    if domain["geometry"] != "cartesian":
        for index in slots.get("boundary.left", []):
            if components[index]["type"] != "boundary.sealed":
                issue(
                    f"components[{index}].type",
                    "the radial origin requires boundary.sealed for symmetry",
                )

    for index, component in enumerate(components):
        path = f"components[{index}].parameters"
        if component["type"] == "initial.gaussian":
            if component["parameters"]["center"] > domain["length"]:
                issue(f"{path}.center", "must lie within the domain")
        if run["mode"] == "steady" and component["type"] == "diffusion":
            if component["parameters"]["coefficient"] <= 0:
                issue(
                    f"{path}.coefficient",
                    "steady solving requires positive diffusivity",
                )

    # Two zero-gradient walls leave the pure diffusion steady field undetermined.
    # A positive uptake term removes that nullspace; extension reactions
    # explicitly supporting steady solving are checked by the underlying solver.
    boundary_indices = slots.get("boundary.left", []) + slots.get("boundary.right", [])
    all_sealed = len(boundary_indices) == 2 and all(
        components[index]["type"] == "boundary.sealed" for index in boundary_indices
    )
    positive_decay = any(
        component["type"] == "reaction.decay" and component["parameters"]["rate"] > 0
        for component in components
    )
    uptake_capacity = sum(
        component["parameters"]["Vmax"]
        for component in components
        if component["type"] == "reaction.uptake"
    )
    extension_reaction = any(
        definition.slot is None
        and component["type"]
        not in ("reaction.decay", "reaction.source", "reaction.uptake")
        for component, definition in zip(components, definitions)
    )
    if (
        run["mode"] == "steady"
        and all_sealed
        and not positive_decay
        and not uptake_capacity
        and not extension_reaction
    ):
        issue(
            "run.mode",
            "two sealed boundaries need positive decay or uptake for a unique steady state; use transient mode or fix a boundary",
        )

    if (
        run["mode"] == "steady"
        and all_sealed
        and uptake_capacity
        and not positive_decay
        and not extension_reaction
    ):
        production = sum(
            component["parameters"]["rate"]
            for component in components
            if component["type"] == "reaction.source"
        )
        if production >= uptake_capacity:
            issue(
                "run.mode",
                "a closed model needs production below its maximum uptake capacity for a finite steady state; reduce the source or use transient mode",
            )

    spacing = domain["length"] / domain["cells"]
    spacing_squared = spacing * spacing
    dimension = _GEOMETRIES.index(domain["geometry"]) + 1
    try:
        # The smallest radial cell and the full radial measure must remain
        # representable. This checks numerical scale, not an arbitrary physical
        # limit on how small or large the modeled domain can be.
        origin_volume = (spacing / 2) ** dimension / dimension
        domain_volume = domain["length"] ** dimension / dimension
        representable = (
            spacing_squared > 0
            and math.isfinite(spacing_squared)
            and math.isfinite(1 / spacing_squared)
            and origin_volume > 0
            and math.isfinite(domain_volume)
        )
    except OverflowError:
        representable = False
    if not representable:
        issue(
            "domain.length",
            "length and cell count are outside the representable mesh scale",
        )

    if issues:
        raise ExperimentValidationError(issues)
    return tuple(definitions)


class Experiment:
    """An owned, validated experiment that compiles into the existing engine.

    Schema version 1 supports a uniform 1D Cartesian, cylindrical, or spherical
    domain. It requires explicit diffusion, initial state, and both boundaries.
    ``run.duration`` and ``run.frames`` remain in steady documents so switching
    back to transient mode preserves editor settings; a steady run returns one
    field. Resource limits are 2–2000 cells, 1–200 saved frames, and 32 components.
    """

    __slots__ = ("_document", "_definitions")

    def __init__(
        self, payload: Mapping[str, Any], registry: ComponentRegistry | None = None
    ) -> None:
        selected = builtin_registry() if registry is None else registry
        if not isinstance(selected, ComponentRegistry):
            raise TypeError("registry must be a ComponentRegistry")
        self._definitions = _validate(payload, selected)
        self._document = json.dumps(payload, allow_nan=False, ensure_ascii=False)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], registry: ComponentRegistry | None = None
    ) -> Experiment:
        """Validate explicit JSON fields and take an independent snapshot."""
        return cls(payload, registry=registry)

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON data; editing it never mutates this experiment."""
        return json.loads(self._document)

    def build(self) -> Problem:
        """Compile a fresh native-backed ``Problem`` with no global registry state."""
        document = self.to_dict()
        domain = document["domain"]
        try:
            mesh = mesh_1d(domain["cells"], 0, domain["length"], domain["geometry"])
        except (TypeError, ValueError, OverflowError) as error:
            raise ExperimentValidationError(
                [{"path": "domain", "message": str(error)}]
            ) from error
        problem = Problem(mesh)
        for index, (component, definition) in enumerate(
            zip(document["components"], self._definitions)
        ):
            try:
                definition.apply(problem, MappingProxyType(component["parameters"]))
            except (TypeError, ValueError, OverflowError) as error:
                raise ExperimentValidationError(
                    [{"path": f"components[{index}]", "message": str(error)}]
                ) from error
        return problem

    def run(self, *, max_steps: int = 200_000) -> Solution:
        """Execute the compiled problem with the existing verified solver.

        ``max_steps`` is the cumulative transient-step budget, including saved
        frames. Steady runs use ``solve_steady``'s own Newton iteration budget.
        Native numerical diagnostics and errors are preserved without a fallback
        to another method or changed physics.
        """
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        problem = self.build()
        settings = self.to_dict()["run"]
        if settings["mode"] == "steady":
            return solve_steady(problem)
        plan = self._plan(problem, max_steps)
        if not plan.within_step_budget:
            raise RuntimeError(
                f"this run requires {plan.planned_steps:,} explicit steps, exceeding "
                f"max_steps={max_steps}; shorten the duration or reduce the cell count. "
                "Use steady mode if you only need the final balance."
            )
        return solve(
            problem,
            end_time=settings["duration"],
            frames=settings["frames"],
            max_steps=max_steps,
        )

    def plan(self, *, max_steps: int = 200_000) -> ExperimentPlan:
        """Compile and plan the run without taking any time steps.

        A plan above ``max_steps`` is returned with ``within_step_budget=False``
        so an editor can explain the cost before execution. Invalid scientific
        settings still raise an error. Component callbacks run once to build
        the model; reaction functions are not evaluated during planning.
        """
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        return self._plan(self.build(), max_steps)

    def _plan(self, problem: Problem, max_steps: int) -> ExperimentPlan:
        from ._core import SolveOptions, plan_transport

        document = self.to_dict()
        domain, settings = document["domain"], document["run"]
        planned_steps, selected_step = None, None
        saved_states = 1
        if settings["mode"] == "transient":
            checkpoints = _saved_times(
                settings["duration"], None, None, settings["frames"]
            )
            planned_steps, selected_step = 0, 0.0
            elapsed = 0.0
            for target in checkpoints:
                options = SolveOptions()
                options.final_time = target - elapsed
                options.max_steps = max_steps
                plan = plan_transport(problem, options)
                planned_steps += plan.planned_steps
                selected_step = max(selected_step, plan.selected_time_step)
                elapsed = target
            saved_states = len(checkpoints) + 1
        nodes = int(problem.mesh().num_nodes())
        return ExperimentPlan(
            mode=settings["mode"],
            geometry=domain["geometry"],
            cells=domain["cells"],
            nodes=nodes,
            saved_states=saved_states,
            planned_steps=planned_steps,
            selected_time_step=selected_step,
            max_steps=max_steps,
            within_step_budget=planned_steps is None or planned_steps <= max_steps,
            field_storage_bytes=nodes * saved_states * 8,
        )
