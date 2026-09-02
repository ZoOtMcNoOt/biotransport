"""Import-contract tests for the organized public API namespaces."""

import warnings
from importlib import import_module
from types import ModuleType

import pytest

import biotransport as bt
from biotransport.config.bioheat_cryotherapy import (
    celsius_from_kelvin,
    kelvin_from_celsius,
)
from biotransport.config.tumor_drug_delivery import MMHG_TO_PA


APPLICATION_HELPERS = {
    "kelvin_from_celsius": kelvin_from_celsius,
    "celsius_from_kelvin": celsius_from_kelvin,
    "MMHG_TO_PA": MMHG_TO_PA,
}


NAMESPACE_ANCHORS = {
    "diffusion": (
        "TransportProblem",
        "solve",
        "CrankNicolsonDiffusion",
        "NonuniformDiffusion1D",
        "MultiSpeciesSolver",
        "MembraneDiffusion1DSolver",
    ),
    "electrochem": (
        "IonSpecies",
        "NernstPlanckSolver",
        "MultiIonSolver",
        "ghk",
    ),
    "flow": (
        "DarcyFlowSolver",
        "StokesSolver",
        "NavierStokesSolver",
        "CarreauYasudaModel",
    ),
    "applications": (
        "BioheatCryotherapyConfig",
        "BioheatCryotherapySolver",
        "TumorDrugDeliveryConfig",
        "TumorDrugDeliverySolver",
    ),
    "balance": (
        "BalanceLedger",
        "reconcile_balances",
        "balance_residual",
    ),
    "reference": (
        "AdaptiveTimeStepper",
        "integrate",
        "solve_pulsatile",
        "NewtonRaphsonSolver",
    ),
}


@pytest.mark.parametrize("namespace_name", NAMESPACE_ANCHORS)
def test_namespace_exports_are_explicit_and_discoverable(namespace_name: str) -> None:
    namespace = import_module(f"biotransport.{namespace_name}")

    assert isinstance(namespace, ModuleType)
    assert namespace.__doc__
    assert namespace.__all__
    assert len(namespace.__all__) == len(set(namespace.__all__))
    assert all(not name.startswith("_") for name in namespace.__all__)
    assert all(hasattr(namespace, name) for name in namespace.__all__)
    assert set(NAMESPACE_ANCHORS[namespace_name]).issubset(namespace.__all__)

    for symbol_name in namespace.__all__:
        expected = APPLICATION_HELPERS.get(symbol_name)
        if expected is None:
            # Retired root spellings still resolve (with a warning) to the same
            # object the namespace exports.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", bt.BioTransportDeprecationWarning)
                expected = getattr(bt, symbol_name, None)
        assert expected is not None
        assert getattr(namespace, symbol_name) is expected


def test_root_all_is_the_canonical_tier_only() -> None:
    """The root advertises the canonical path and the namespaces, nothing else."""
    for namespace_name in ("reference", "balance"):
        namespace = import_module(f"biotransport.{namespace_name}")
        leaked = sorted(name for name in namespace.__all__ if name in bt.__all__)
        assert not leaked, f"{namespace_name} symbols leaked into bt.__all__: {leaked}"
    specialized = (
        "DiffusionSolver",
        "CrankNicolsonDiffusion",
        "NavierStokesSolver",
        "NernstPlanckSolver",
        "BalanceLedger",
        "HighOrderDiffusionSolver",
        "GridConvergenceStudy",
    )
    assert not set(specialized) & set(bt.__all__)
    # Specialized native classes stay reachable as attributes without warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert bt.DiffusionSolver is bt.diffusion.DiffusionSolver
        assert bt.NavierStokesSolver is bt.flow.NavierStokesSolver


@pytest.mark.parametrize(
    ("namespace_name", "symbol_name"),
    [
        (namespace_name, symbol_name)
        for namespace_name, symbol_names in NAMESPACE_ANCHORS.items()
        for symbol_name in symbol_names
    ],
)
def test_namespace_symbols_are_reexports(namespace_name: str, symbol_name: str) -> None:
    namespace = import_module(f"biotransport.{namespace_name}")

    with warnings.catch_warnings():
        # Reference symbols are deprecated at the root but must be the same objects.
        warnings.simplefilter("ignore", bt.BioTransportDeprecationWarning)
        root_object = getattr(bt, symbol_name)
    assert getattr(namespace, symbol_name) is root_object
