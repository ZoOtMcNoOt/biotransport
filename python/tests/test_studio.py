"""Exercise the real HTTP adapter and scientific result format."""

from copy import deepcopy
import http.client
import json
import socket
import threading

import numpy as np
import pytest

import biotransport as bt
from biotransport.studio.examples import examples
from biotransport.studio.results import result_payload
from biotransport.studio.server import MAX_BODY, StudioServer


@pytest.fixture
def studio():
    server = StudioServer(0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield server
    server.shutdown()
    server.server_close()
    worker.join(timeout=5)


def request(server, path, payload=None, *, headers=None, raw=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    body = (
        raw if raw is not None else json.dumps(payload) if payload is not None else None
    )
    method = "POST" if body is not None else "GET"
    content_headers = {"Content-Type": "application/json"} if body is not None else {}
    content_headers.update(headers or {})
    connection.request(method, path, body=body, headers=content_headers)
    response = connection.getresponse()
    status, response_headers, body = (
        response.status,
        dict(response.getheaders()),
        response.read(),
    )
    connection.close()
    return status, response_headers, body


def test_public_experiment_api():
    model = bt.Experiment.from_dict(examples()[0])
    assert isinstance(model.build(), bt.Problem)
    assert isinstance(bt.builtin_registry(), bt.ComponentRegistry)


def test_custom_registry_publishes_editor_metadata_and_runs():
    registry = bt.builtin_registry().register(
        bt.ComponentDefinition(
            type="course.uptake",
            label="Course uptake",
            description="Test uptake.",
            category="reaction",
            parameters=(bt.Parameter("rate", "Rate", "number", 0.001, min=0),),
            apply=lambda problem, values: problem.add_linear_decay(values["rate"]),
            supports_steady=True,
        )
    )
    with StudioServer(0, registry=registry) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            _, _, body = request(server, "/api/catalog")
            catalog = json.loads(body)["components"]
            custom = next(d for d in catalog if d["type"] == "course.uptake")
            assert custom["category"] == "reaction" and custom["slot"] is None
            document = examples()[1]
            document["components"].append(
                {"id": "uptake", "type": "course.uptake", "parameters": {"rate": 0.001}}
            )
            status, _, body = request(server, "/api/run", document)
            assert status == 200, body
            totals = json.loads(body)["solution"]["totals"]
            assert totals[-1] < totals[0]
        finally:
            server.shutdown()
            worker.join(timeout=5)


def test_catalog_and_packaged_assets(studio):
    status, headers, body = request(studio, "/api/catalog")
    assert status == 200
    catalog = json.loads(body)
    assert len(catalog["components"]) == 9
    assert len(catalog["examples"]) == 5
    assert headers["Cache-Control"] == "no-store"
    for path, content_type in (
        ("/", "text/html"),
        ("/style.css", "text/css"),
        ("/app.js", "text/javascript"),
        ("/model.mjs", "text/javascript"),
        ("/drag.mjs", "text/javascript"),
    ):
        status, headers, body = request(studio, path)
        assert status == 200 and len(body) > 100
        assert headers["Content-Type"].startswith(content_type)
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        assert headers["X-Content-Type-Options"] == "nosniff"
    assert request(studio, "/../pyproject.toml")[0] == 404


@pytest.mark.parametrize("index", range(5))
def test_example_run_uses_public_engine(studio, index):
    document = examples()[index]
    status, _, body = request(studio, "/api/run", document)
    assert status == 200, body
    data = json.loads(body)
    expected = bt.Experiment.from_dict(document).run()
    actual = data["solution"]
    np.testing.assert_array_equal(actual["fields"], expected.history)
    np.testing.assert_array_equal(actual["times"], expected.times)
    np.testing.assert_allclose(actual["totals"], expected.history @ expected.weights)
    assert actual["elapsed_seconds"] >= 0
    assert data["experiment"] == document
    if index == 0:
        assert data["reference"]["max_abs_error"] < 2e-4
    if index == 1:
        assert data["reference"] is None
        assert abs(actual["totals"][-1] - actual["totals"][0]) < 1e-15
    if index == 2:
        assert actual["steady"] and len(actual["fields"]) == 1
        assert data["reference"]["max_abs_error"] < 1e-3
        assert "dimensionless" in actual["summary"]
        assert all(
            g["name"] not in {"Fourier", "sqrt(D t) / L"}
            for g in actual["dimensionless"]
        )


def test_extreme_steady_reference_does_not_discard_valid_solution():
    document = examples()[2]
    document["domain"].update(cells=2, length=1e-150)
    document["components"][0]["parameters"]["coefficient"] = 1e-308
    document["components"][-1]["parameters"]["rate"] = 10
    solution = bt.Experiment.from_dict(document).run()
    payload = result_payload(document, solution, 0)
    assert np.all(np.isfinite(payload["solution"]["fields"]))
    assert payload["reference"]["fields"][0][1] == 0
    json.dumps(payload, allow_nan=False)


def test_validate_never_solves(studio, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("validation must not run a solver")

    monkeypatch.setattr(bt.Experiment, "run", forbidden)
    status, _, body = request(studio, "/api/validate", examples()[0])
    assert status == 200
    assert json.loads(body)["experiment"] == examples()[0]


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://unrelated.example"},
        {"Host": "unrelated.example"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Origin": "null"},
    ],
)
def test_reject_cross_origin_and_rebinding(studio, headers):
    assert request(studio, "/api/run", examples()[0], headers=headers)[0] == 403


def test_allow_same_origin(studio):
    assert (
        request(
            studio,
            "/api/validate",
            examples()[0],
            headers={"Origin": f"http://127.0.0.1:{studio.server_port}"},
        )[0]
        == 200
    )


@pytest.mark.parametrize(
    "path,extra_headers,status",
    [
        ("/api/run", "Origin: https://unrelated.example\r\n", 403),
        ("/api/run", "Sec-Fetch-Site: cross-site\r\n", 403),
        ("/missing", "", 404),
        ("/api/run", "Content-Type: text/plain\r\n", 415),
    ],
)
def test_rejected_upload_can_finish_receiving_its_error(
    studio, path, extra_headers, status
):
    # Send headers first, then continue the upload after rejection has started.
    # An immediate full close can reset TCP and discard the response on Windows.
    body = b"x" * MAX_BODY
    headers = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{studio.server_port}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"{extra_headers}\r\n"
    ).encode("ascii")
    with socket.create_connection(
        ("127.0.0.1", studio.server_port), timeout=10
    ) as stream:
        stream.sendall(headers)
        with http.client.HTTPResponse(stream) as response:
            response.begin()
            stream.sendall(body)
            assert response.status == status
            assert json.loads(response.read())["error"]


@pytest.mark.parametrize("raw", ['{"bad":', "[]", "null", '{"schema_version":NaN}'])
def test_bad_json_is_actionable(studio, raw):
    status, _, body = request(studio, "/api/run", raw=raw)
    assert status == 422
    assert json.loads(body)["error"]


def test_scientific_error_has_field_path(studio):
    payload = examples()[0]
    payload["domain"]["length"] = -1
    status, _, body = request(studio, "/api/run", payload)
    assert status == 422
    assert any("length" in item["path"] for item in json.loads(body)["issues"])


@pytest.mark.parametrize("field,value", [("cells", 1001), ("frames", 121)])
def test_interactive_budget(studio, field, value):
    payload = examples()[0]
    payload["domain" if field == "cells" else "run"][field] = value
    status, _, body = request(studio, "/api/run", payload)
    assert status == 422
    assert "interactive workbench" in json.loads(body)["error"]


def test_request_and_concurrency_budgets(studio):
    assert request(studio, "/api/run", raw="x" * (MAX_BODY + 1))[0] == 413
    assert (
        request(studio, "/api/run", raw="{}", headers={"Content-Type": "text/plain"})[0]
        == 415
    )
    with studio.solve_lock:
        status, _, body = request(studio, "/api/run", examples()[0])
    assert status == 409
    assert "already running" in json.loads(body)["error"]


def test_reference_assumptions_are_not_borrowed():
    base = examples()[0]
    for change in ("reaction", "unequal"):
        document = deepcopy(base)
        if change == "reaction":
            document["components"].append(
                {
                    "id": "source",
                    "type": "reaction.source",
                    "parameters": {"rate": 0.0001},
                }
            )
        elif change == "unequal":
            document["components"][2]["parameters"]["value"] = 2
        solution = bt.Experiment.from_dict(document).run()
        assert result_payload(document, solution, 0)["reference"] is None


@pytest.mark.parametrize("geometry", ["cylindrical", "spherical"])
def test_radial_reference_uses_its_own_geometry(geometry):
    document = examples()[3]
    document["domain"]["geometry"] = geometry
    solution = bt.Experiment.from_dict(document).run()
    reference = result_payload(document, solution, 0)["reference"]
    expected = bt.analytical.steady_radial_first_order(
        solution.x, D=2e-9, k=0.03, R=0.0005, c_surface=0.05, geometry=geometry
    )
    np.testing.assert_array_equal(reference["fields"][0], expected)
    assert reference["max_abs_error"] < 1e-5
    document["components"][-1] = examples()[4]["components"][-1]
    nonlinear = bt.Experiment.from_dict(document).run()
    assert result_payload(document, nonlinear, 0)["reference"] is None


def test_plan_and_import_remain_available_during_a_solve(studio, monkeypatch):
    monkeypatch.setattr(bt.Experiment, "run", lambda *a, **k: pytest.fail("must not solve"))
    document = examples()[0]
    document["domain"]["cells"] = 2000
    document["run"]["frames"] = 200
    with studio.solve_lock:
        assert request(studio, "/api/validate", document)[0] == 200
        status, _, body = request(studio, "/api/plan", document)
    assert status == 200
    data = json.loads(body)
    assert not data["runnable"]
    assert {"domain.cells", "run.frames"} <= {issue["path"] for issue in data["issues"]}


def test_plan_explains_step_budget_before_execution(studio):
    document = examples()[3]
    document["run"].update(mode="transient", duration=20_000, frames=10)
    status, _, body = request(studio, "/api/plan", document)
    assert status == 200
    data = json.loads(body)
    assert not data["runnable"] and data["plan"]["planned_steps"] > 200_000
    assert any("shorten" in issue["message"] for issue in data["issues"])
    status, _, body = request(studio, "/api/run", document)
    assert status == 422 and "max_steps" in json.loads(body)["error"]


def test_unconverged_reference_is_not_shown():
    document = examples()[0]
    document["run"]["duration"] = 1e-10
    solution = bt.Experiment.from_dict(document).run()
    assert result_payload(document, solution, 0)["reference"] is None


def test_pure_advection_group_serializes_without_nonfinite_json():
    document = examples()[0]
    document["components"][0]["parameters"]["coefficient"] = 0
    document["components"].append(
        {"id": "flow", "type": "advection", "parameters": {"velocity": 1e-6}}
    )
    solution = bt.Experiment.from_dict(document).run()
    payload = result_payload(document, solution, 0)
    json.dumps(payload, allow_nan=False)
    assert (
        next(
            g["value"]
            for g in payload["solution"]["dimensionless"]
            if g["name"] == "Peclet"
        )
        is None
    )
