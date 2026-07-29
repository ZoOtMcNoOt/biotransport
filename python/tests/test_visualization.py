"""Headless behavior for the beginner-friendly plotting API."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import biotransport as bt
import biotransport.visualization as visualization


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


def test_plot_result_only_error_explicitly_names_missing_mesh() -> None:
    mesh = bt.mesh_1d(4, 0.0, 1.0)

    with pytest.raises(ValueError, match=r"bt\.plot\(mesh, result\)"):
        bt.plot(_canonical_result(mesh), show=False)


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
    monkeypatch.setattr(visualization.plt, "show", lambda: calls.append("show"))

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
    monkeypatch.setattr(visualization.plt, "show", lambda: calls.append("show"))

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
