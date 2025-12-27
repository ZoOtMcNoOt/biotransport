"""Utility functions for BioTransport simulations."""

from __future__ import annotations

import os
import time
from pathlib import Path


def _find_repo_root_from_cwd() -> Path | None:
    """Best-effort detection of the repo root when running examples.

    This helps beginners get stable output paths even if they run scripts from
    inside subfolders (e.g. examples/) or from VS Code where CWD can vary.
    """

    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        if (parent / "pyproject.toml").is_file() and (
            parent / "python" / "biotransport"
        ).is_dir():
            return parent
    return None


def get_results_dir(
    subfolder=None,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    env_var: str = "BIOTRANSPORT_RESULTS_DIR",
):
    """Get the path to the results directory, creating it if it doesn't exist.

    Args:
        subfolder: Optional subfolder name within results directory. Use
            "timestamp" to create a timestamped subfolder.
        base_dir: Optional override for where the top-level `results/` folder
            should live.
        env_var: Environment variable name that, if set, overrides `base_dir`.

    Returns:
        str: Path to the results directory.
    """

    env_base = os.environ.get(env_var)
    if env_base:
        root = Path(env_base).expanduser()
    elif base_dir is not None:
        root = Path(base_dir)
    else:
        root = _find_repo_root_from_cwd() or Path.cwd()

    results_dir = root / "results"

    if subfolder == "timestamp":
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_dir = results_dir / timestamp
    elif subfolder:
        results_dir = results_dir / str(subfolder)

    results_dir.mkdir(parents=True, exist_ok=True)
    return str(results_dir)


def get_result_path(
    filename,
    subfolder=None,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    env_var: str = "BIOTRANSPORT_RESULTS_DIR",
):
    """Get the full path for a result file in the results directory."""

    return str(
        Path(get_results_dir(subfolder, base_dir=base_dir, env_var=env_var)) / filename
    )
