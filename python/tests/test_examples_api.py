"""Fast guards that keep runnable examples aligned with the public API."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import biotransport as bt
import pytest
from typing_extensions import TypeGuard


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _example_sources() -> Iterator[tuple[Path, str, ast.Module]]:
    for path in sorted(EXAMPLES.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        yield path, source, ast.parse(source, filename=str(path))


def _is_bt_call(node: ast.AST, name: str) -> TypeGuard[ast.Call]:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "bt"
        and node.func.attr == name
    )


def _scope_nodes(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.AST]:
    """Walk one lexical scope without attributing nested-function names to it."""
    stack: list[ast.AST] = list(reversed(scope.body))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _assigned_names(target: ast.AST) -> Iterator[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _assigned_names(element)


@pytest.mark.parametrize(
    ("path", "source", "tree"),
    list(_example_sources()),
    ids=lambda value: value.name if isinstance(value, Path) else None,
)
def test_examples_compile_and_reference_exported_root_api(
    path: Path, source: str, tree: ast.Module
) -> None:
    compile(source, str(path), "exec")
    referenced = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "bt"
    }
    missing = sorted(name for name in referenced if not hasattr(bt, name))
    assert not missing, f"{path} references missing biotransport exports: {missing}"


def test_canonical_solve_examples_use_the_current_contract() -> None:
    failures: list[str] = []

    for path, _, tree in _example_sources():
        relative = path.relative_to(EXAMPLES)
        for node in ast.walk(tree):
            if _is_bt_call(node, "solve"):
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                aliases = sorted({"t", "dt"}.intersection(keywords))
                if aliases:
                    failures.append(
                        f"{relative}:{node.lineno} uses compatibility alias(es) "
                        f"{', '.join(aliases)} instead of end_time/time_step"
                    )
                method = keywords.get("method")
                if method is not None:
                    if not isinstance(method, ast.Constant) or not isinstance(
                        method.value, str
                    ):
                        failures.append(
                            f"{relative}:{node.lineno} uses a dynamic canonical method"
                        )
                    elif method.value.lower().replace("-", "_") not in {
                        "conservative",
                        "explicit",
                        "explicit_euler",
                    }:
                        failures.append(
                            f"{relative}:{node.lineno} requests unsupported canonical "
                            f"method {method.value!r}"
                        )

            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "advection_scheme"
                and node.args
                and isinstance(node.args[0], ast.Attribute)
                and isinstance(node.args[0].value, ast.Attribute)
                and isinstance(node.args[0].value.value, ast.Name)
                and node.args[0].value.value.id == "bt"
                and node.args[0].value.attr == "AdvectionScheme"
                and node.args[0].attr != "UPWIND"
            ):
                failures.append(
                    f"{relative}:{node.lineno} configures unsupported canonical "
                    f"advection scheme {node.args[0].attr}"
                )

        scopes: list[ast.Module | ast.FunctionDef | ast.AsyncFunctionDef] = [tree]
        scopes.extend(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        for scope in scopes:
            nodes = list(_scope_nodes(scope))
            canonical_names: set[str] = set()
            for node in nodes:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    value = node.value
                    if value is not None and _is_bt_call(value, "solve"):
                        targets = (
                            node.targets
                            if isinstance(node, ast.Assign)
                            else [node.target]
                        )
                        for target in targets:
                            canonical_names.update(_assigned_names(target))

            for node in nodes:
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "solution"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in canonical_names
                ):
                    failures.append(
                        f"{relative}:{node.lineno} calls canonical result.solution; "
                        "use result.concentration"
                    )
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "stats"
                    and isinstance(node.value, ast.Name)
                    and node.value.id in canonical_names
                ):
                    failures.append(
                        f"{relative}:{node.lineno} reads canonical result.stats; "
                        "use result.diagnostics"
                    )

    assert not failures, "\n" + "\n".join(failures)


def test_vtk_examples_use_mapping_based_signatures() -> None:
    failures: list[str] = []
    for path, _, tree in _example_sources():
        relative = path.relative_to(EXAMPLES)
        for node in ast.walk(tree):
            if not (
                _is_bt_call(node, "write_vtk") or _is_bt_call(node, "write_vtk_series")
            ):
                continue
            if any(keyword.arg == "field_name" for keyword in node.keywords):
                failures.append(
                    f"{relative}:{node.lineno} uses the removed field_name keyword"
                )
            if len(node.args) != 3:
                failures.append(
                    f"{relative}:{node.lineno} must pass mesh, mapped fields/snapshots, "
                    "and a filename"
                )

    assert not failures, "\n" + "\n".join(failures)


def test_plot_examples_only_forward_supported_keywords() -> None:
    supported = {
        "title",
        "ax",
        "kind",
        "xlabel",
        "ylabel",
        "colorbar_label",
        "zlabel",
        "save_to",
    }
    failures: list[str] = []
    for path, _, tree in _example_sources():
        relative = path.relative_to(EXAMPLES)
        for node in ast.walk(tree):
            if not _is_bt_call(node, "plot"):
                continue
            unsupported = sorted(
                keyword.arg
                for keyword in node.keywords
                if keyword.arg is not None and keyword.arg not in supported
            )
            if unsupported:
                failures.append(
                    f"{relative}:{node.lineno} forwards unsupported bt.plot "
                    f"keyword(s): {', '.join(unsupported)}"
                )

    assert not failures, "\n" + "\n".join(failures)


def test_print_literals_are_safe_for_default_windows_consoles() -> None:
    """Examples should not crash merely because stdout still uses cp1252."""
    failures: list[str] = []
    for path, _, tree in _example_sources():
        relative = path.relative_to(EXAMPLES)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                continue

            literals: list[str] = []
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(
                    argument.value, str
                ):
                    literals.append(argument.value)
                elif isinstance(argument, ast.JoinedStr):
                    literals.extend(
                        part.value
                        for part in argument.values
                        if isinstance(part, ast.Constant)
                        and isinstance(part.value, str)
                    )
            try:
                "".join(literals).encode("cp1252")
            except UnicodeEncodeError as error:
                escaped = str(error.object[error.start : error.end]).encode(
                    "ascii", "backslashreplace"
                )
                failures.append(
                    f"{relative}:{node.lineno} prints a non-cp1252 literal "
                    f"{escaped.decode('ascii')!r}"
                )

    assert not failures, "\n" + "\n".join(failures)


def test_minimal_canonical_example_runs_headlessly(tmp_path: Path) -> None:
    """Exercise construction, native solve, result access, and plotting."""
    script = EXAMPLES / "basic" / "1d_diffusion.py"
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["BIOTRANSPORT_RESULTS_DIR"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=EXAMPLES.parent,
        env=environment,
        capture_output=True,
        timeout=30,
        check=False,
    )
    output = (completed.stdout + completed.stderr).decode(errors="backslashreplace")
    assert completed.returncode == 0, output


def test_reproducible_artifact_example_is_idempotent(tmp_path: Path) -> None:
    script = EXAMPLES / "verification" / "reproducible_artifact.py"
    output_path = tmp_path / "manifest.json"
    command = [sys.executable, str(script), "--output", str(output_path)]
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["BIOTRANSPORT_RESULTS_DIR"] = str(tmp_path)

    for _ in range(2):
        completed = subprocess.run(
            command,
            cwd=EXAMPLES.parent,
            env=environment,
            capture_output=True,
            timeout=30,
            check=False,
        )
        output = (completed.stdout + completed.stderr).decode(errors="backslashreplace")
        assert completed.returncode == 0, output
