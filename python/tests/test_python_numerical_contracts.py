"""Integrity gates for separately disclosed Python numerical surfaces."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib
import json
from pathlib import Path

import pytest

import biotransport as bt
from biotransport.contracts import (
    PYTHON_NUMERICAL_CONTRACTS,
    PythonBackend,
    PythonNumericalContract,
    get_python_numerical_contract,
    list_python_numerical_contracts,
    list_python_numerical_symbols,
    python_registry_as_dict,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_registry_exactly_owns_public_symbols_from_governed_python_modules() -> None:
    contracts = list_python_numerical_contracts()
    runtime_symbols: set[str] = set()
    for contract in contracts:
        module = importlib.import_module(contract.module)
        if hasattr(module, "__all__"):
            module_symbols = set(module.__all__)
        else:
            module_symbols = {
                name
                for name, value in vars(module).items()
                if not name.startswith("_")
                and getattr(value, "__module__", None) == module.__name__
            }

        assert set(contract.public_symbols) == module_symbols
        runtime_symbols.update(module_symbols)
        for symbol in contract.public_symbols:
            assert hasattr(module, symbol)
            if symbol in bt.__all__:
                assert getattr(bt, symbol) is getattr(module, symbol)

    assert set(list_python_numerical_symbols()) == runtime_symbols


def test_python_contract_ids_modules_and_symbols_have_one_owner() -> None:
    contracts = list_python_numerical_contracts()
    ids = [contract.contract_id for contract in contracts]
    modules = [contract.module for contract in contracts]
    symbols = [symbol for contract in contracts for symbol in contract.public_symbols]

    assert len(ids) == len(set(ids))
    assert len(modules) == len(set(modules))
    assert len(symbols) == len(set(symbols))
    assert set(PYTHON_NUMERICAL_CONTRACTS) == set(ids)
    for symbol in symbols:
        assert get_python_numerical_contract(symbol).contract_id in ids


def test_python_contract_schema_is_immutable_and_json_ready() -> None:
    contracts = list_python_numerical_contracts()
    payload = python_registry_as_dict()

    assert json.dumps(payload, sort_keys=True)
    assert set(payload) == set(PYTHON_NUMERICAL_CONTRACTS)
    for contract in contracts:
        assert isinstance(contract, PythonNumericalContract)
        assert isinstance(contract.backend, PythonBackend)
        assert contract.public_symbols
        assert contract.mathematical_scope
        assert contract.numerical_method
        assert contract.failure_policy
        assert contract.evidence
        assert contract.disposition
        assert contract.to_dict()["evidence_level"] == contract.evidence_level.value

    with pytest.raises(FrozenInstanceError):
        contracts[0].title = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        PYTHON_NUMERICAL_CONTRACTS["changed"] = contracts[0]  # type: ignore[index]


def test_python_contract_evidence_references_are_current() -> None:
    for contract in list_python_numerical_contracts():
        for evidence in contract.evidence:
            for reference in evidence.references:
                relative_path, selector = reference.split("::", maxsplit=1)
                evidence_path = REPOSITORY_ROOT / relative_path

                assert evidence_path.is_file(), (
                    f"{contract.contract_id}: stale evidence path {relative_path}"
                )
                assert evidence_path.suffix in {".cpp", ".py"}
                assert evidence_path != Path(__file__).resolve()
                assert selector in evidence_path.read_text(encoding="utf-8"), (
                    f"{contract.contract_id}: stale selector {selector!r} "
                    f"in {relative_path}"
                )


def test_queries_separate_python_surfaces_from_native_solver_contracts() -> None:
    canonical = get_python_numerical_contract("python.canonical.adapters")

    assert get_python_numerical_contract("solve") is canonical
    assert canonical.backend is PythonBackend.NATIVE_ADAPTER
    assert get_python_numerical_contract("NewtonRaphsonSolver").backend is (
        PythonBackend.PYTHON_REFERENCE
    )
    assert get_python_numerical_contract("parameter_sweep").backend is (
        PythonBackend.WORKFLOW
    )
    with pytest.raises(KeyError, match="unknown Python numerical contract"):
        get_python_numerical_contract("solve_transport")


def test_legacy_python_solver_dispositions_do_not_claim_native_performance() -> None:
    for contract_id in (
        "python.legacy.adaptive_diffusion",
        "python.legacy.time_integrators",
        "python.reference.pulsatile_diffusion",
    ):
        contract = get_python_numerical_contract(contract_id)
        combined = " ".join(
            (contract.disposition, contract.numerical_method, *contract.warnings)
        ).casefold()

        assert "python" in combined
        assert "native" in combined or "deprecat" in combined
