"""Illustrative, reproducible starting points for the workbench."""

from copy import deepcopy


def examples() -> list[dict]:
    """Return fresh experiment documents; values are teaching examples."""
    slab = {
        "schema_version": 1,
        "name": "Diffusion into a slab",
        "domain": {"geometry": "cartesian", "length": 0.01, "cells": 120},
        "components": [
            {
                "id": "diffusion",
                "type": "diffusion",
                "parameters": {"coefficient": 1e-9},
            },
            {"id": "initial", "type": "initial.uniform", "parameters": {"value": 0.0}},
            {
                "id": "left",
                "type": "boundary.fixed",
                "parameters": {"side": "left", "value": 1.0},
            },
            {
                "id": "right",
                "type": "boundary.fixed",
                "parameters": {"side": "right", "value": 1.0},
            },
        ],
        "run": {"mode": "transient", "duration": 2000.0, "frames": 40},
    }
    pulse = deepcopy(slab)
    pulse["name"] = "A pulse in a sealed channel"
    pulse["components"][1] = {
        "id": "initial",
        "type": "initial.gaussian",
        "parameters": {"center": 0.005, "width": 0.0005, "amplitude": 1.0},
    }
    for index, side in ((2, "left"), (3, "right")):
        pulse["components"][index] = {
            "id": side,
            "type": "boundary.sealed",
            "parameters": {"side": side},
        }
    pulse["run"]["duration"] = 600.0
    tissue = deepcopy(slab)
    tissue["name"] = "Diffusion with tissue uptake"
    tissue["components"].append(
        {
            "id": "uptake",
            "type": "reaction.decay",
            "parameters": {"rate": 0.0005},
        }
    )
    tissue["run"]["mode"] = "steady"
    sphere = deepcopy(tissue)
    sphere["name"] = "Uptake in a tissue sphere"
    sphere["domain"].update(geometry="spherical", length=0.0005)
    sphere["components"][0]["parameters"]["coefficient"] = 2e-9
    sphere["components"][1]["parameters"]["value"] = 0.05
    sphere["components"][2] = {
        "id": "left", "type": "boundary.sealed", "parameters": {"side": "left"},
    }
    sphere["components"][3]["parameters"]["value"] = 0.05
    sphere["components"][-1]["parameters"]["rate"] = 0.03
    sphere["run"]["duration"] = 120
    saturable = deepcopy(sphere)
    saturable["name"] = "Saturable nutrient uptake"
    saturable["components"][-1] = {
        "id": "uptake", "type": "reaction.uptake",
        "parameters": {"Vmax": 0.0005, "Km": 0.01},
    }
    return [slab, pulse, tissue, sphere, saturable]
