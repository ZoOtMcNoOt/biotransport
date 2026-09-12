"""Execute the code blocks in the readme and the tutorial.

These two documents are what a newcomer copies from, so a broken snippet in
either is a worse bug than a broken function nobody calls. Nothing else in the
suite reads them, which is how several stale examples survived for a while.

Blocks run in order, sharing a namespace per document, because the documents are
written as a narrative where later snippets build on earlier ones. A block that
the surrounding prose presents as *failing* -- the stability-limit demonstration,
for instance -- is expected to raise, and that is detected from the error output
shown immediately after it rather than from a hand-maintained list.
"""

from __future__ import annotations

import os
from pathlib import Path
import re

import pytest


matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

REPO = Path(__file__).resolve().parents[2]
DOCUMENTS = [REPO / "readme.md", REPO / "docs" / "tutorial.md"]

_BLOCK = re.compile(r"```(python|text)\n(.*?)```", re.DOTALL)

# Errors the documents deliberately demonstrate.
_EXPECTED_ERROR_NAMES = ("ValueError", "TypeError", "RuntimeError")


def _blocks(path: Path) -> list[tuple[str, str, bool]]:
    """Return ``(language, source, expected_to_raise)`` for each fenced block."""

    found = _BLOCK.findall(path.read_text(encoding="utf-8"))
    result: list[tuple[str, str, bool]] = []
    for index, (language, body) in enumerate(found):
        follows = found[index + 1] if index + 1 < len(found) else None
        shows_error = bool(
            follows
            and follows[0] == "text"
            and follows[1].lstrip().startswith(_EXPECTED_ERROR_NAMES)
        )
        result.append((language, body, shows_error))
    return result


def _python_blocks(path: Path) -> list[tuple[int, str, bool]]:
    return [
        (number, body, raises)
        for number, (language, body, raises) in enumerate(_blocks(path))
        if language == "python"
    ]


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: p.name)
def test_documented_code_blocks_run(document: Path, tmp_path: Path) -> None:
    """Every Python block in the document runs, or fails exactly as advertised."""

    assert document.exists(), f"{document} is missing"
    blocks = _python_blocks(document)
    assert blocks, f"no Python blocks found in {document.name}"

    namespace: dict[str, object] = {"__name__": "__doc_snippet__"}
    original = Path.cwd()
    os.chdir(tmp_path)  # snippets that save figures should not litter the repo
    try:
        for number, source, expected_to_raise in blocks:
            try:
                exec(
                    compile(source, f"{document.name}#block{number}", "exec"), namespace
                )
            except Exception as error:  # noqa: BLE001 - reported below with context
                if expected_to_raise:
                    continue
                pytest.fail(
                    f"{document.name} block {number} raised "
                    f"{type(error).__name__}: {error}\n\n{source}"
                )
            else:
                if expected_to_raise:
                    pytest.fail(
                        f"{document.name} block {number} shows an error in the "
                        f"document but ran without raising:\n\n{source}"
                    )
    finally:
        os.chdir(original)
        matplotlib.pyplot.close("all")


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: p.name)
def test_documented_snippets_stay_console_safe(document: Path) -> None:
    """Code blocks must survive a default Windows console encoding.

    Prose is free to use typographic characters -- it is rendered, not executed.
    A snippet is different: someone will paste it into a terminal that is still
    on cp1252, and a stray Greek letter in a print will crash it there.
    """

    offenders: list[str] = []
    for number, (language, body, _raises) in enumerate(_blocks(document)):
        if language != "python":
            continue
        for line in body.splitlines():
            try:
                line.encode("cp1252")
            except UnicodeEncodeError as error:
                offenders.append(
                    f"{document.name} block {number}: {line[error.start : error.end]!r} "
                    f"in {line.strip()!r}"
                )
    assert not offenders, "\n".join(offenders)
