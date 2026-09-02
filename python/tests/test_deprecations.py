"""Every retired public spelling must warn, name its replacement, and forward.

This test is driven by the tables in :mod:`biotransport._deprecation`, so a
deprecation cannot be added without being covered here.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

import biotransport as bt
from biotransport import _deprecation
from biotransport._deprecation import (
    BioTransportDeprecationWarning,
    DeprecatedName,
    ROOT_DEPRECATED,
    deprecated_callable,
    deprecated_keyword,
    deprecation_message,
    resolve,
)


def _problem():
    mesh = bt.mesh_1d(20)
    return (
        bt.Problem(mesh)
        .diffusivity(0.05)
        .initial_condition(bt.gaussian(mesh, center=0.5, width=0.1))
        .dirichlet(bt.Boundary.Left, 0.0)
        .dirichlet(bt.Boundary.Right, 0.0)
    )


def test_warning_category_is_a_deprecation_warning() -> None:
    assert issubclass(BioTransportDeprecationWarning, DeprecationWarning)
    assert bt.BioTransportDeprecationWarning is BioTransportDeprecationWarning


def test_importing_the_package_emits_no_deprecation_warning() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-c",
            "import biotransport",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_message_format_names_versions_replacement_and_reason() -> None:
    message = deprecation_message(
        "old", "new()", reason="it was ambiguous", since="0.2.0", removal="0.4.0"
    )
    assert message == (
        "old is deprecated since 0.2.0 and will be removed in 0.4.0; use new(). "
        "it was ambiguous."
    )


def test_deprecated_name_requires_a_real_window_and_a_module_target() -> None:
    with pytest.raises(ValueError, match="later than"):
        DeprecatedName("x", "m:x", "y", since="0.2.0", removal="0.2.0")
    with pytest.raises(ValueError, match="module:attribute"):
        DeprecatedName("x", "not-a-target", "y")


@pytest.mark.parametrize("name", sorted(ROOT_DEPRECATED))
def test_every_root_deprecation_warns_and_resolves_to_its_target(name: str) -> None:
    entry = ROOT_DEPRECATED[name]
    assert name not in bt.__all__
    assert name not in dir(bt)
    with pytest.warns(BioTransportDeprecationWarning) as record:
        value = getattr(bt, name)
    assert value is resolve(entry.target)
    message = str(record[0].message)
    assert f"biotransport.{name} is deprecated since {entry.since}" in message
    assert f"removed in {entry.removal}" in message
    assert entry.replacement in message
    if entry.reason:
        assert entry.reason.rstrip(".") in message


def test_unknown_root_attribute_still_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'definitely_missing'"):
        bt.definitely_missing  # noqa: B018


def test_run_alias_forwards_to_solve_with_identical_results() -> None:
    problem = _problem()
    expected = bt.solve(problem, end_time=0.01)
    assert "run" not in bt.__all__
    assert bt.run.__wrapped__ is importlib.import_module("biotransport.run").run
    with pytest.warns(BioTransportDeprecationWarning, match="bt.solve") as record:
        legacy = bt.run(problem, 0.01)
    assert "biotransport.run is deprecated since" in str(record[0].message)
    assert legacy.time == expected.time
    assert legacy.diagnostics.steps == expected.diagnostics.steps
    np.testing.assert_array_equal(legacy.concentration, expected.concentration)


def test_deprecated_keyword_folds_warns_and_rejects_double_spelling() -> None:
    kwargs = {"dt": 0.5}
    with pytest.warns(BioTransportDeprecationWarning, match=r"f\(time_step=\.\.\.\)"):
        value = deprecated_keyword(kwargs, "dt", "time_step", None, function="f")
    assert value == 0.5
    assert kwargs == {}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert deprecated_keyword({}, "dt", "time_step", 1.0, function="f") == 1.0

    with pytest.raises(TypeError, match="either time_step or dt"):
        deprecated_keyword({"dt": 0.5}, "dt", "time_step", 1.0, function="f")


def test_deprecated_callable_warns_once_per_call_and_forwards() -> None:
    @deprecated_callable("new_function()", reason="renamed for clarity")
    def old_function(x: int) -> int:
        """Docstring survives."""
        return x + 1

    assert old_function.__doc__ == "Docstring survives."
    with pytest.warns(BioTransportDeprecationWarning, match="new_function") as record:
        assert old_function(1) == 2
    assert len(record) == 1


def test_problem_aliases_resolve_to_the_problem_builder() -> None:
    for name in (
        "DiffusionProblem",
        "LinearReactionDiffusionProblem",
        "AdvectionDiffusionProblem",
    ):
        with pytest.warns(BioTransportDeprecationWarning, match="bt.Problem"):
            assert getattr(bt, name) is bt.Problem


def test_transport_result_solution_alias_warns_and_matches_concentration() -> None:
    native = bt.solve(_problem(), end_time=0.01).native
    with pytest.warns(BioTransportDeprecationWarning, match="concentration"):
        legacy = native.solution
    np.testing.assert_array_equal(legacy, native.concentration)


def test_nernst_planck_neumann_spelling_warns_and_installs_the_same_flux() -> None:
    mesh = bt.mesh_1d(10)
    ion = bt.IonSpecies("Na+", 1, 1.33e-9)

    def make() -> bt.NernstPlanckSolver:
        solver = bt.NernstPlanckSolver(mesh, ion)
        solver.set_initial_condition(np.full(mesh.num_nodes(), 1.0))
        solver.set_uniform_field(50.0)
        solver.set_dirichlet_boundary(bt.Boundary.Left, 1.0)
        return solver

    preferred = make()
    preferred.set_outward_flux_boundary(bt.Boundary.Right, 2.0e-9)
    deprecated = make()
    with pytest.warns(BioTransportDeprecationWarning) as record:
        deprecated.set_neumann_boundary(bt.Boundary.Right, 2.0e-9)
    message = str(record[0].message)
    assert "set_outward_flux_boundary" in message
    assert "physical molar flux" in message

    dt = 0.5 * preferred.maximum_stable_time_step()
    preferred.solve(dt, 5)
    deprecated.solve(dt, 5)
    np.testing.assert_array_equal(preferred.solution(), deprecated.solution())


def test_tables_are_read_only() -> None:
    with pytest.raises(TypeError):
        ROOT_DEPRECATED["x"] = ROOT_DEPRECATED["DiffusionProblem"]  # type: ignore[index]
    with pytest.raises(TypeError):
        _deprecation.ROOT_LAZY["x"] = "m:x"  # type: ignore[index]
