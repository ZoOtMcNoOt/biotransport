"""Mechanical integrity tests for the native scientific-contract registry."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import inspect
import json
from pathlib import Path

import pytest

from biotransport._core import _core as native
from biotransport.contracts import (
    EVIDENCE_DISCLAIMER,
    SOLVER_CONTRACTS,
    EvidenceLevel,
    EvidenceRecord,
    SolverContract,
    filter_contracts,
    get_contract,
    list_contracts,
    list_native_solver_symbols,
    registry_as_dict,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_SOURCE = REPOSITORY_ROOT / "python/biotransport/contracts.py"


def _runtime_native_solver_symbols() -> set[str]:
    """Discover compiled simulation entry points without using the registry."""

    symbols = {
        name
        for name in dir(native)
        if not name.startswith("_")
        and inspect.isclass(getattr(native, name))
        and any(
            callable(getattr(getattr(native, name), method_name, None))
            for method_name in ("solve", "simulate", "run")
        )
    }
    # SparseMatrix.solve is a linear-algebra primitive, not a physical solver
    # contract. Physical sparse diffusion is represented by ImplicitDiffusion*.
    symbols.remove("SparseMatrix")
    assert inspect.isbuiltin(native.solve_transport)
    symbols.add("solve_transport")
    return symbols


def test_registry_exactly_covers_runtime_native_solver_entry_points() -> None:
    runtime_symbols = _runtime_native_solver_symbols()
    registered_symbols = set(list_native_solver_symbols())

    assert registered_symbols == runtime_symbols, (
        f"missing={sorted(runtime_symbols - registered_symbols)}, "
        f"stale={sorted(registered_symbols - runtime_symbols)}"
    )


def test_contract_ids_and_native_symbols_have_one_owner() -> None:
    contracts = list_contracts()
    ids = [contract.contract_id for contract in contracts]
    symbols = [symbol for contract in contracts for symbol in contract.native_symbols]

    assert len(ids) == len(set(ids))
    assert len(symbols) == len(set(symbols))
    assert set(SOLVER_CONTRACTS) == set(ids)
    for symbol in symbols:
        assert get_contract(symbol).contract_id in SOLVER_CONTRACTS
        assert hasattr(native, symbol)


def test_contract_schema_is_complete_and_machine_readable() -> None:
    payload = registry_as_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert serialized
    assert set(payload) == set(SOLVER_CONTRACTS)
    for contract in list_contracts():
        assert isinstance(contract, SolverContract)
        assert contract.equation
        assert contract.unknowns
        assert contract.locations
        assert contract.input_units
        assert contract.output_units
        assert contract.supported_dimensions
        assert contract.supported_terms
        assert contract.supported_boundary_conditions
        assert contract.numerical_method
        assert contract.stability_policy
        assert contract.convergence_policy
        assert contract.evidence
        assert contract.exclusions
        assert contract.warnings
        assert contract.to_dict()["evidence_level"] == contract.evidence_level.value


def test_contract_and_evidence_types_are_immutable() -> None:
    contract = get_contract("CrankNicolsonDiffusion")
    record = contract.evidence[0]

    with pytest.raises(FrozenInstanceError):
        contract.title = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        record.claim = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        SOLVER_CONTRACTS["changed"] = contract  # type: ignore[index]

    coerced = EvidenceRecord(
        EvidenceLevel.API,
        "A scoped test claim.",
        [
            "python/tests/test_solver_contracts.py::test_contract_and_evidence_types_are_immutable"
        ],  # type: ignore[arg-type]
    )
    assert isinstance(coerced.references, tuple)


def test_evidence_references_resolve_to_current_test_selectors() -> None:
    for contract in list_contracts():
        for record in contract.evidence:
            if record.level is EvidenceLevel.UNTESTED:
                assert not record.references
                continue
            assert record.references
            for reference in record.references:
                relative_path, selector = reference.split("::", maxsplit=1)
                evidence_path = REPOSITORY_ROOT / relative_path
                assert evidence_path.is_file(), (
                    f"{contract.contract_id}: stale evidence path {relative_path}"
                )
                assert evidence_path.suffix in {".cpp", ".py"}
                assert evidence_path != Path(__file__).resolve(), (
                    f"{contract.contract_id}: registry integrity tests cannot serve as "
                    "solver evidence"
                )
                source = evidence_path.read_text(encoding="utf-8")
                assert selector and selector in source, (
                    f"{contract.contract_id}: stale selector {selector!r} in {relative_path}"
                )


def test_queries_support_ids_symbols_units_and_filters() -> None:
    by_id = get_contract("diffusion.crank_nicolson")
    by_symbol = get_contract("CrankNicolsonDiffusion")

    assert by_id is by_symbol
    assert by_symbol.unit_for("diffusivity") == "length^2/time"
    with pytest.raises(KeyError, match="documented output quantity"):
        by_symbol.unit_for("diffusivity", output=True)
    with pytest.raises(KeyError, match="unknown solver contract"):
        get_contract("NotASolver")

    three_dimensional = filter_contracts(dimension="3d")
    assert three_dimensional
    assert all("3D" in contract.supported_dimensions for contract in three_dimensional)

    convergence = filter_contracts(minimum_evidence="convergence")
    assert convergence
    assert all(
        contract.evidence_level is EvidenceLevel.CONVERGENCE for contract in convergence
    )

    constant_diffusion = filter_contracts(term="constant isotropic diffusion")
    assert get_contract("DiffusionSolver") in constant_diffusion
    assert get_contract("CrankNicolsonDiffusion") in constant_diffusion

    with pytest.raises(ValueError, match="unknown evidence level"):
        filter_contracts(minimum_evidence="validated")


def test_evidence_language_disclaims_validation_and_asme_compliance() -> None:
    normalized = EVIDENCE_DISCLAIMER.casefold()
    assert "not experimental or biological validation" in normalized
    assert "not" in normalized and "asme v&v 20 compliance" in normalized


def test_contract_module_syntax_is_python_39_compatible() -> None:
    source = CONTRACTS_SOURCE.read_text(encoding="utf-8")
    ast.parse(source, filename=str(CONTRACTS_SOURCE), feature_version=(3, 9))
