"""Bounded loopback-only adapter; all model logic lives in experiment.py."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import socket
import threading
import time
from urllib.parse import urlsplit

from ..experiment import Experiment, ExperimentValidationError, builtin_registry
from .examples import examples
from .results import result_payload

MAX_BODY = 262144
MAX_CELLS = 1000
MAX_FRAMES = 120
MAX_STEPS = 200000
ASSETS = {
    "/": ("index.html", "text/html"),
    "/app.js": ("app.js", "text/javascript"),
    "/model.mjs": ("model.mjs", "text/javascript"),
    "/drag.mjs": ("drag.mjs", "text/javascript"),
    "/style.css": ("style.css", "text/css"),
}


def _interactive_issues(document):
    issues = []
    for path, value, maximum in (
        ("domain.cells", document["domain"]["cells"], MAX_CELLS),
        ("run.frames", document["run"]["frames"], MAX_FRAMES),
    ):
        if value > maximum:
            issues.append(
                {
                    "path": path,
                    "message": f"the interactive workbench supports up to {maximum}; reduce this setting or run from Python",
                }
            )
    return issues


class StudioServer(ThreadingHTTPServer):
    """Local teaching server. One native solve at a time bounds resource use."""

    daemon_threads = True

    def __init__(self, port: int = 8766, *, registry=None):
        self.registry = registry if registry is not None else builtin_registry()
        self.solve_lock = threading.Lock()
        super().__init__(("127.0.0.1", port), StudioHandler)

    def shutdown_request(self, request):
        # Finish sending the response before closing an early-rejected upload.
        # A full close with unread input can reset TCP and erase the response.
        # RFC 9112 section 9.6 recommends this staged close; time and byte limits
        # keep an abandoned or excessive upload from retaining a worker.
        try:
            request.shutdown(socket.SHUT_WR)
            deadline = time.monotonic() + 1.0
            remaining = 2 * MAX_BODY
            while remaining > 0:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                request.settimeout(timeout)
                chunk = request.recv(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
        except OSError:
            pass
        finally:
            self.close_request(request)


class StudioHandler(BaseHTTPRequestHandler):
    server: StudioServer

    def log_message(self, format, *args):
        pass

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self._send(status, body, "application/json")

    def _error(self, status: int, message: str, issues=None):
        self._json(status, {"error": message, "issues": issues or []})

    def _local_request(self) -> bool:
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host not in hosts or (origin is not None and origin != f"http://{host}"):
            self._error(
                HTTPStatus.FORBIDDEN, "Open the workbench from its local address."
            )
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._error(HTTPStatus.FORBIDDEN, "Cross-site requests are not supported.")
            return False
        return True

    def do_GET(self):
        if not self._local_request():
            return
        path = urlsplit(self.path).path
        if path == "/api/catalog":
            self._json(
                200,
                {
                    "components": self.server.registry.catalog(),
                    "examples": examples(),
                    "limits": {
                        "cells": MAX_CELLS,
                        "frames": MAX_FRAMES,
                        "steps": MAX_STEPS,
                    },
                },
            )
        elif path in ASSETS:
            name, mime = ASSETS[path]
            self._send(
                200,
                files("biotransport.studio").joinpath("static", name).read_bytes(),
                mime,
            )
        else:
            self._error(404, "This workbench page does not exist.")

    def do_POST(self):
        if not self._local_request():
            return
        path = urlsplit(self.path).path
        if path not in {"/api/run", "/api/validate", "/api/plan"}:
            self._error(404, "This workbench action does not exist.")
            return
        if self.headers.get_content_type() != "application/json":
            self._error(415, "Send the experiment as application/json.")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if self.headers.get("Transfer-Encoding") or not 0 < length <= MAX_BODY:
            self.close_connection = True
            self._error(413, "Experiment files must be smaller than 256 KB.")
            return
        locked = False
        try:
            self.connection.settimeout(10)
            payload = json.loads(self.rfile.read(length))
            experiment = Experiment.from_dict(payload, registry=self.server.registry)
            document = experiment.to_dict()
            if path == "/api/validate":
                # Import/export uses the portable schema, not the interactive
                # run budget. A larger valid model can be opened and reduced.
                self._json(200, {"experiment": document})
                return
            issues = _interactive_issues(document)
            if path == "/api/plan":
                plan = experiment.plan(max_steps=MAX_STEPS)
                if not plan.within_step_budget:
                    issues.append(
                        {
                            "path": "run.duration",
                            "message": f"this run needs {plan.planned_steps:,} steps, above the {MAX_STEPS:,} interactive limit; shorten the duration, reduce cells, or choose steady state for the final balance",
                        }
                    )
                self._json(
                    200,
                    {"plan": plan.to_dict(), "runnable": not issues, "issues": issues},
                )
                return
            if issues:
                raise ExperimentValidationError(issues)
            locked = self.server.solve_lock.acquire(blocking=False)
            if not locked:
                self._error(
                    409, "A simulation is already running. Try again when it finishes."
                )
                return
            start = time.perf_counter()
            solution = experiment.run(max_steps=MAX_STEPS)
            self._json(
                200, result_payload(document, solution, time.perf_counter() - start)
            )
        except ExperimentValidationError as error:
            self._error(422, str(error), error.issues)
        except (ValueError, TypeError, RuntimeError, OverflowError) as error:
            self._error(422, str(error))
        except (OSError, UnicodeError):
            self.close_connection = True
            self._error(
                400, "The experiment could not be read. Check the file and try again."
            )
        finally:
            if locked:
                self.server.solve_lock.release()
