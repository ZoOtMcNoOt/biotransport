"""Experiments preserve the native science path and validate the editor boundary."""

from __future__ import annotations

import copy
import dataclasses
import json
import math

import numpy as np
import pytest

import biotransport as bt
from biotransport.experiment import (
    ComponentDefinition,
    ComponentRegistry,
    Experiment,
    ExperimentValidationError,
    Parameter,
    builtin_registry,
)


@pytest.fixture
def document():
    return {
        "schema_version": 1,
        "name": "Diffusion through a slab",
        "domain": {"geometry": "cartesian", "length": 1.0, "cells": 40},
        "components": [
            {
                "id": "diffusion",
                "type": "diffusion",
                "parameters": {"coefficient": 0.1},
            },
            {"id": "initial", "type": "initial.uniform", "parameters": {"value": 0.0}},
            {
                "id": "left",
                "type": "boundary.fixed",
                "parameters": {"side": "left", "value": 1.0},
            },
            {"id": "right", "type": "boundary.sealed", "parameters": {"side": "right"}},
        ],
        "run": {"mode": "transient", "duration": 0.2, "frames": 8},
    }


def _closed(document, initial=1.0):
    document["components"][1]["parameters"]["value"] = initial
    document["components"][2] = {
        "id": "left",
        "type": "boundary.sealed",
        "parameters": {"side": "left"},
    }


def _reaction(document, kind, rate, identifier=None):
    document["components"].append(
        {"id": identifier or kind, "type": kind, "parameters": {"rate": rate}}
    )


def test_json_roundtrip_owns_all_mutable_input_and_output(document):
    expected = copy.deepcopy(document)
    experiment = Experiment.from_dict(json.loads(json.dumps(document)))
    document["components"][0]["parameters"]["coefficient"] = 99
    exported = experiment.to_dict()
    exported["domain"]["length"] = 99
    exported["components"].clear()
    assert experiment.to_dict() == expected
    assert json.loads(json.dumps(experiment.to_dict(), allow_nan=False)) == expected


def test_transient_matches_direct_native_workflow_and_saved_times(document):
    mesh = bt.mesh_1d(40, 0, 1)
    problem = (
        bt.Problem(mesh)
        .diffusivity(0.1)
        .initial(0)
        .dirichlet("left", 1)
        .sealed("right")
    )
    reference = bt.solve(problem, end_time=0.2, frames=8)
    actual = Experiment.from_dict(document).run()
    np.testing.assert_array_equal(actual.history, reference.history)
    assert actual.times == reference.times
    assert len(actual.times) == 9  # initial state plus requested saved frames
    assert actual.steps == reference.steps


def test_build_and_run_are_repeatable_independent_problem_instances(document):
    experiment = Experiment.from_dict(document)
    first, second = experiment.build(), experiment.build()
    first.diffusivity(5).initial(9)
    assert second.diffusivity() == 0.1
    np.testing.assert_array_equal(second.initial(), 0)
    np.testing.assert_array_equal(experiment.run().history, experiment.run().history)


def test_components_reorder_without_changing_additive_physics(document):
    _closed(document, initial=2)
    _reaction(document, "reaction.decay", 0.2, "uptake")
    _reaction(document, "reaction.source", 0.3, "production")
    _reaction(document, "reaction.decay", 0.3, "clearance")
    _reaction(document, "reaction.source", 0.2, "infusion")
    experiment = Experiment.from_dict(document)
    document["components"].reverse()
    reordered = Experiment.from_dict(document).run()
    actual = experiment.run()
    np.testing.assert_allclose(actual.history, reordered.history, rtol=0, atol=2e-15)
    # Uniform solution of dc/dt = 0.5 - 0.5*c, independent of discretization.
    np.testing.assert_allclose(actual.c, 1 + math.exp(-0.1), rtol=1e-4)


def test_steady_agrees_with_analytic_first_order_profile(document):
    document["run"]["mode"] = "steady"
    document["domain"]["cells"] = 120
    _reaction(document, "reaction.decay", 0.2)
    actual = Experiment.from_dict(document).run()
    reference = np.cosh(math.sqrt(2) * (1 - actual.x)) / np.cosh(math.sqrt(2))
    assert actual.steady
    assert len(actual.times) == 1
    assert actual.newton.converged
    np.testing.assert_allclose(actual.c, reference, rtol=2e-5)


def test_steady_sealed_domain_with_source_and_decay_has_unique_equilibrium(document):
    _closed(document, initial=0)
    _reaction(document, "reaction.source", 0.2)
    _reaction(document, "reaction.decay", 0.1)
    document["run"]["mode"] = "steady"
    np.testing.assert_allclose(Experiment.from_dict(document).run().c, 2, atol=1e-10)


@pytest.mark.parametrize("geometry", ["cartesian", "cylindrical", "spherical"])
def test_geometry_is_preserved_against_direct_solve(document, geometry):
    _closed(document)
    document["domain"]["geometry"] = geometry
    document["components"][3] = {
        "id": "right",
        "type": "boundary.fixed",
        "parameters": {"side": "right", "value": 0},
    }
    direct = (
        bt.Problem(bt.mesh_1d(40, 0, 1, geometry))
        .diffusivity(0.1)
        .initial(1)
        .sealed("left")
        .dirichlet("right", 0)
    )
    reference = bt.solve(direct, end_time=0.2, frames=8)
    actual = Experiment.from_dict(document).run()
    np.testing.assert_array_equal(actual.history, reference.history)
    assert actual.mesh.geometry() == direct.mesh().geometry()


def test_gaussian_and_advection_use_existing_helpers(document):
    document["components"][1] = {
        "id": "pulse",
        "type": "initial.gaussian",
        "parameters": {"center": 0.4, "width": 0.08, "amplitude": 2},
    }
    document["components"].append(
        {"id": "flow", "type": "advection", "parameters": {"velocity": 0.3}}
    )
    problem = Experiment.from_dict(document).build()
    reference = (
        bt.Problem(bt.mesh_1d(40, 0, 1))
        .diffusivity(0.1)
        .initial(bt.gaussian(problem.mesh(), center=0.4, width=0.08, amplitude=2))
        .dirichlet("left", 1)
        .sealed("right")
        .velocity(0.3)
    )
    np.testing.assert_array_equal(problem.initial(), reference.initial())
    np.testing.assert_array_equal(
        Experiment.from_dict(document).run().history,
        bt.solve(reference, end_time=0.2, frames=8).history,
    )


def test_explicit_local_extension_preserves_solver_and_registry_isolation(document):
    _closed(document)
    custom = ComponentDefinition(
        type="lab.uptake",
        label="Lab uptake",
        description="Validated first-order uptake.",
        category="reaction",
        parameters=(Parameter("rate", "Rate", "number", 0.2, min=0),),
        apply=lambda problem, values: problem.add_linear_decay(values["rate"]),
        supports_steady=True,
    )
    registry = builtin_registry().register(custom)
    _reaction(document, "lab.uptake", 0.2)
    experiment = Experiment.from_dict(document, registry=registry)
    with pytest.raises(ExperimentValidationError, match="unknown component"):
        Experiment.from_dict(document)
    assert "lab.uptake" not in [entry["type"] for entry in builtin_registry().catalog()]
    with pytest.raises(ValueError, match="already registered"):
        registry.register(custom)
    # A registry mutation after validation does not replace the experiment's definitions.
    registry._definitions.clear()
    native = (
        bt.Problem(bt.mesh_1d(40, 0, 1))
        .diffusivity(0.1)
        .initial(1)
        .sealed("left")
        .sealed("right")
        .add_linear_decay(0.2)
    )
    np.testing.assert_array_equal(
        experiment.run().history, bt.solve(native, end_time=0.2, frames=8).history
    )


def test_extension_can_supply_an_exclusive_initial_role(document):
    registry = builtin_registry().register(
        ComponentDefinition(
            type="lab.band",
            label="Central band",
            description="A normalized central concentration band.",
            category="initial",
            parameters=(),
            slot="initial",
            supports_steady=True,
            apply=lambda problem, _: problem.initial(
                np.where(
                    (bt.x_nodes(problem.mesh()) >= 0.4)
                    & (bt.x_nodes(problem.mesh()) <= 0.6),
                    1.0,
                    0.0,
                )
            ),
        )
    )
    document["components"][1] = {"id": "initial", "type": "lab.band", "parameters": {}}
    problem = Experiment.from_dict(document, registry=registry).build()
    assert set(problem.initial()) == {0, 1}


def test_custom_reaction_requires_explicit_steady_support(document):
    registry = builtin_registry().register(
        ComponentDefinition(
            type="lab.custom",
            label="Custom",
            description="Custom constant production.",
            category="reaction",
            parameters=(),
            apply=lambda problem, _: problem.add_reaction(
                lambda c, x, y, t: 0.1, max_abs_dc=0
            ),
        )
    )
    document["components"].append(
        {"id": "custom", "type": "lab.custom", "parameters": {}}
    )
    Experiment.from_dict(document, registry=registry).run()
    document["run"]["mode"] = "steady"
    with pytest.raises(ExperimentValidationError, match="does not support steady"):
        Experiment.from_dict(document, registry=registry)


def test_catalog_is_json_serializable_and_does_not_mutate_registry():
    registry = builtin_registry()
    expected = registry.catalog()
    exported = registry.catalog()
    exported[0]["parameters"][0]["default"] = 5
    exported[0]["geometries"].clear()
    assert registry.catalog() == expected
    assert json.loads(json.dumps(expected, allow_nan=False)) == expected
    assert {item["type"] for item in expected} == {
        "diffusion",
        "initial.uniform",
        "initial.gaussian",
        "boundary.fixed",
        "boundary.sealed",
        "reaction.decay",
        "reaction.source",
        "reaction.uptake",
        "advection",
    }
    sealed = next(item for item in expected if item["type"] == "boundary.sealed")
    assert "advection" in sealed["description"]


def test_definitions_freeze_lists_and_reject_invalid_schemas():
    choices = ["left", "right"]
    parameter = Parameter("side", "Side", "choice", "left", choices=choices)
    choices.clear()
    assert parameter.choices == ("left", "right")
    parameters = [parameter]
    definition = ComponentDefinition(
        "lab.boundary",
        "Boundary",
        "Example.",
        "boundary",
        parameters,
        lambda p, v: None,
        slot="boundary",
    )
    parameters.clear()
    assert definition.parameters == (parameter,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        parameter.default = "right"
    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.type = "replaced"
    with pytest.raises(ValueError, match="default"):
        Parameter("rate", "Rate", "number", float("nan"))
    with pytest.raises(ValueError, match="unique"):
        ComponentDefinition(
            "duplicate",
            "Duplicate",
            "Example.",
            "reaction",
            (parameter, parameter),
            lambda p, v: None,
        )
    with pytest.raises(ValueError, match="side choice"):
        ComponentDefinition(
            "bad", "Bad", "Example.", "boundary", (), lambda p, v: None, slot="boundary"
        )
    with pytest.raises(ValueError, match="choice"):
        Parameter("side", "Side", "choice", "left", choices="left")


@pytest.mark.parametrize("version", [True, 1.0, "1", 0, 2, None])
def test_schema_version_is_an_exact_integer(document, version):
    document["schema_version"] = version
    with pytest.raises(ExperimentValidationError) as caught:
        Experiment.from_dict(document)
    assert any(issue["path"] == "schema_version" for issue in caught.value.issues)


@pytest.mark.parametrize(
    "bad",
    [
        True,
        False,
        "0.1",
        None,
        float("nan"),
        float("inf"),
        -float("inf"),
        10**1000,
        [],
        {},
    ],
)
def test_numeric_fields_reject_coercion_and_nonfinite_values(document, bad):
    document["components"][0]["parameters"]["coefficient"] = bad
    with pytest.raises(ExperimentValidationError) as caught:
        Experiment.from_dict(document)
    assert caught.value.issues == [
        {
            "path": "components[0].parameters.coefficient",
            "message": "must be a finite number (booleans and numeric strings are invalid)",
        }
    ]


@pytest.mark.parametrize(
    "field,bad",
    [
        ("cells", True),
        ("cells", 2.0),
        ("cells", 1),
        ("cells", 2001),
        ("length", 0),
        ("length", -1),
        ("length", 1e-300),
        ("length", 1e300),
    ],
)
def test_invalid_domain_and_unsafe_mesh_scales(document, field, bad):
    document["domain"][field] = bad
    with pytest.raises(ExperimentValidationError):
        Experiment.from_dict(document)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("frames", True),
        ("frames", 2.0),
        ("frames", 0),
        ("frames", 201),
        ("duration", 0),
        ("duration", -1),
        ("duration", float("nan")),
        ("mode", "implicit"),
    ],
)
def test_invalid_run_settings(document, field, bad):
    document["run"][field] = bad
    with pytest.raises(ExperimentValidationError):
        Experiment.from_dict(document)


@pytest.mark.parametrize(
    "container,key", [(None, "metadata"), ("domain", "dimensions"), ("run", "method")]
)
def test_unknown_fields_are_never_silently_ignored(document, container, key):
    target = document if container is None else document[container]
    target[key] = 1
    with pytest.raises(ExperimentValidationError, match="unknown field"):
        Experiment.from_dict(document)


def test_missing_parameters_and_components_have_structured_errors(document):
    del document["components"][0]["parameters"]["coefficient"]
    with pytest.raises(ExperimentValidationError) as caught:
        Experiment.from_dict(document)
    assert caught.value.issues == [
        {"path": "components[0].parameters.coefficient", "message": "is required"}
    ]
    document["components"].clear()
    with pytest.raises(ExperimentValidationError, match="between 4 and 32"):
        Experiment.from_dict(document)


@pytest.mark.parametrize(
    "kind", ["id", "diffusion", "initial", "boundary", "advection"]
)
def test_duplicate_identity_or_exclusive_roles_are_rejected(document, kind):
    if kind == "id":
        document["components"][1]["id"] = "diffusion"
    elif kind == "advection":
        for identifier in ("flow1", "flow2"):
            document["components"].append(
                {"id": identifier, "type": "advection", "parameters": {"velocity": 0.1}}
            )
    else:
        index = {"diffusion": 0, "initial": 1, "boundary": 2}[kind]
        duplicate = copy.deepcopy(document["components"][index])
        duplicate["id"] = "duplicate"
        document["components"].append(duplicate)
    with pytest.raises(ExperimentValidationError):
        Experiment.from_dict(document)


@pytest.mark.parametrize("kind", ["os.system", "__import__('os')", "missing", [], None])
def test_json_cannot_choose_unregistered_code(document, kind):
    document["components"][0]["type"] = kind
    with pytest.raises(ExperimentValidationError, match="unknown component"):
        Experiment.from_dict(document)


def test_parameter_bounds_and_gaussian_position_are_validated(document):
    document["components"][1] = {
        "id": "initial",
        "type": "initial.gaussian",
        "parameters": {"center": 2, "width": 0.1, "amplitude": 1},
    }
    with pytest.raises(ExperimentValidationError, match="within the domain"):
        Experiment.from_dict(document)
    document["components"][1]["parameters"].update(center=0.5, width=0)
    with pytest.raises(ExperimentValidationError, match="greater than 0"):
        Experiment.from_dict(document)
    document["components"][1]["parameters"].update(width=0.1, amplitude=-1)
    with pytest.raises(ExperimentValidationError, match="at least 0"):
        Experiment.from_dict(document)


@pytest.mark.parametrize("geometry", ["cylindrical", "spherical"])
def test_radial_origin_flow_and_steady_scope_are_explicit(document, geometry):
    document["domain"]["geometry"] = geometry
    with pytest.raises(
        ExperimentValidationError, match="origin requires boundary.sealed"
    ):
        Experiment.from_dict(document)
    _closed(document)
    document["run"]["mode"] = "steady"
    with pytest.raises(ExperimentValidationError, match="unique steady state"):
        Experiment.from_dict(document)
    document["run"]["mode"] = "transient"
    document["components"].append(
        {"id": "flow", "type": "advection", "parameters": {"velocity": 0.1}}
    )
    with pytest.raises(ExperimentValidationError, match="does not support"):
        Experiment.from_dict(document)


def test_steady_rejects_advection_zero_diffusion_and_nullspace(document):
    document["run"]["mode"] = "steady"
    document["components"].append(
        {"id": "flow", "type": "advection", "parameters": {"velocity": 0.1}}
    )
    with pytest.raises(ExperimentValidationError, match="does not support steady"):
        Experiment.from_dict(document)
    document["components"].pop()
    document["components"][0]["parameters"]["coefficient"] = 0
    with pytest.raises(ExperimentValidationError, match="positive diffusivity"):
        Experiment.from_dict(document)
    document["components"][0]["parameters"]["coefficient"] = 0.1
    _closed(document)
    with pytest.raises(ExperimentValidationError, match="unique steady state"):
        Experiment.from_dict(document)
    _reaction(document, "reaction.source", 0.2)
    with pytest.raises(ExperimentValidationError, match="unique steady state"):
        Experiment.from_dict(document)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_transient_budget_is_strict(document, value):
    with pytest.raises(ValueError, match="positive integer"):
        Experiment.from_dict(document).run(max_steps=value)


def test_solver_step_budget_is_enforced_across_frames(document):
    with pytest.raises(RuntimeError, match="max_steps"):
        Experiment.from_dict(document).run(max_steps=1)


def test_malformed_containers_remain_validation_errors(document):
    for payload in (None, [], "bad", {"schema_version": 1}):
        with pytest.raises(ExperimentValidationError):
            Experiment.from_dict(payload)
    for field in ("domain", "components", "run"):
        broken = copy.deepcopy(document)
        broken[field] = None
        with pytest.raises(ExperimentValidationError):
            Experiment.from_dict(broken)
    document["components"][1] = None
    with pytest.raises(ExperimentValidationError):
        Experiment.from_dict(document)


def test_empty_registry_does_not_use_builtins_as_an_implicit_fallback(document):
    with pytest.raises(ExperimentValidationError, match="unknown component"):
        Experiment.from_dict(document, registry=ComponentRegistry())
