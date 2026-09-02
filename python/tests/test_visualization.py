"""Headless behavior for the beginner-friendly plotting API."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import types

import pytest

import biotransport as bt


def _canonical_result(mesh):
    problem = bt.Problem(mesh).diffusivity(0.0).initial_condition(2.0)
    return bt.solve(problem, end_time=0.0)


def test_plot_accepts_mesh_and_transport_result() -> None:
    mesh = bt.mesh_1d(4, 0.0, 1.0)
    result = _canonical_result(mesh)

    figure = bt.plot(mesh, result, show=False)
    try:
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), 2.0)
    finally:
        plt.close(figure)


def test_plot_accepts_result_with_callable_concentration() -> None:
    solver = (
        bt.MembraneDiffusion1DSolver()
        .set_membrane_thickness(1.0e-4)
        .set_diffusivity(1.0e-9)
        .set_partition_coefficient(0.5)
        .set_left_concentration(100.0)
        .set_right_concentration(20.0)
        .set_num_nodes(101)
    )
    result = solver.solve()
    mesh = bt.mesh_1d(100, 0.0, 1.0e-4)

    figure = bt.plot(mesh, result, show=False)
    try:
        np.testing.assert_allclose(
            figure.axes[0].lines[0].get_ydata(), result.concentration()
        )
    finally:
        plt.close(figure)


def test_plot_accepts_a_result_that_carries_its_mesh() -> None:
    mesh = bt.mesh_1d(4, 0.0, 1.0)
    result = _canonical_result(mesh)

    figure = bt.plot(result, show=False)
    assert figure.axes[0].lines[0].get_ydata().tolist() == result.concentration.tolist()
    assert result.plot(show=False) is not None


def test_plot_result_without_mesh_error_explicitly_names_the_fix() -> None:
    meshless = types.SimpleNamespace(concentration=np.zeros(5))

    with pytest.raises(ValueError, match=r"bt\.plot\(mesh, result\)"):
        bt.plot(meshless, show=False)


@pytest.mark.parametrize(
    ("mesh", "kind", "message"),
    [
        (bt.mesh_1d(4, 0.0, 1.0), "surface", "1D mesh"),
        (bt.mesh_2d(4, 3, 0.0, 1.0, 0.0, 1.0), "line", "2D mesh"),
        (bt.mesh_2d(4, 3, 0.0, 1.0, 0.0, 1.0), "banana", "2D mesh"),
    ],
)
def test_plot_rejects_kind_incompatible_with_mesh_dimension(
    mesh, kind: str, message: str
) -> None:
    values = np.zeros(mesh.num_nodes())

    with pytest.raises(ValueError, match=message):
        bt.plot(mesh, values, kind=kind, show=False)


def test_plot_forwards_1d_labels_and_does_not_show(monkeypatch) -> None:
    mesh = bt.mesh_1d(4, 0.0, 1.0)
    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append("show"))

    figure = bt.plot(
        mesh,
        np.arange(mesh.num_nodes()),
        kind="line",
        xlabel="Distance [m]",
        ylabel="Concentration [mol/m^3]",
        show=False,
    )
    try:
        axis = figure.axes[0]
        assert axis.get_xlabel() == "Distance [m]"
        assert axis.get_ylabel() == "Concentration [mol/m^3]"
        assert calls == []
    finally:
        plt.close(figure)


def test_plot_show_true_calls_matplotlib(monkeypatch) -> None:
    mesh = bt.mesh_1d(4, 0.0, 1.0)
    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append("show"))

    figure = bt.plot(mesh, np.zeros(mesh.num_nodes()), show=True)
    try:
        assert calls == ["show"]
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("kind", None, TypeError),
        ("show", "no", TypeError),
    ],
)
def test_plot_rejects_ambiguous_control_types(keyword, value, error) -> None:
    mesh = bt.mesh_1d(4, 0.0, 1.0)

    with pytest.raises(error):
        bt.plot(mesh, np.zeros(mesh.num_nodes()), **{keyword: value})


def test_plot_save_to_writes_the_figure(tmp_path) -> None:
    mesh = bt.mesh_2d(4, 3)
    values = bt.gaussian(mesh, center=0.5, width=0.2)
    target = tmp_path / "field.png"

    figure = bt.plot(mesh, values, kind="surface", zlabel="c", save_to=target)
    try:
        assert target.exists() and target.stat().st_size > 0
        assert figure.axes[0].get_zlabel() == "c"
    finally:
        plt.close(figure)


def test_plot_accepts_arrays_from_the_initial_condition_helpers() -> None:
    mesh = bt.mesh_1d(8)
    figure = bt.plot(mesh, bt.step(mesh, position=0.5), xlabel="x", ylabel="c")
    try:
        assert figure.axes[0].get_xlabel() == "x"
        assert figure.axes[0].lines[0].get_ydata().tolist() == bt.step(mesh).tolist()
    finally:
        plt.close(figure)


def test_plot_module_does_not_import_pyplot_eagerly() -> None:
    import subprocess
    import sys

    code = (
        "import sys, biotransport; "
        "sys.exit(1 if 'matplotlib.pyplot' in sys.modules else 0)"
    )
    completed = subprocess.run([sys.executable, "-c", code], check=False)
    assert completed.returncode == 0, (
        "importing biotransport imported matplotlib.pyplot"
    )


@pytest.mark.parametrize(
    "name, args, kwargs",
    [
        ("plot_1d_solution", ("1d",), {}),
        ("plot_1d", ("1d",), {"show_grid": False}),
        ("plot_2d_solution", ("2d",), {}),
        ("plot_2d", ("2d",), {}),
        ("plot_2d_surface", ("2d",), {}),
        ("plot_field", ("2d",), {"kind": "surface", "xlabel": "u"}),
    ],
)
def test_legacy_plot_spellings_warn_and_forward(name, args, kwargs) -> None:
    mesh = bt.mesh_1d(4) if args[0] == "1d" else bt.mesh_2d(4, 4)
    values = bt.uniform(mesh, 1.0)
    with pytest.warns(bt.BioTransportDeprecationWarning, match=name):
        figure = getattr(bt, name)(mesh, values, **kwargs)
    try:
        assert figure.axes
    finally:
        plt.close(figure)


def test_plot_solution_keyword_is_deprecated() -> None:
    mesh = bt.mesh_1d(4)
    with pytest.warns(bt.BioTransportDeprecationWarning, match="values"):
        figure = bt.plot(mesh, solution=bt.uniform(mesh, 2.0))
    try:
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), 2.0)
    finally:
        plt.close(figure)
    with (
        pytest.raises(TypeError, match="once"),
        pytest.warns(bt.BioTransportDeprecationWarning),
    ):
        bt.plot(mesh, bt.uniform(mesh, 1.0), solution=bt.uniform(mesh, 2.0))
