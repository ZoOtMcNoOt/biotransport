"""Guard: the hand-written ``_core.pyi`` stub must track the compiled module.

The stub is compared against the runtime ``biotransport._core._core`` extension
purely by name (via :mod:`ast`), in both directions:

* every public runtime class / builtin function is declared in the stub;
* every stub top-level class / function exists at runtime;
* per class, every stub member exists on the runtime class and every public
  runtime member defined in the class ``__dict__`` is declared in the stub.

The pybind11 submodules (``constants``, ``ions``, ``ghk``, ``analytical``,
``dimensionless``) are declared as classes in the stub; they are compared
against ``dir()`` of the runtime submodule.
"""

from __future__ import annotations

import ast
import inspect
import types
from pathlib import Path

import pytest

import biotransport._core._core as _core

STUB_PATH = Path(_core.__file__).with_name("_core.pyi")

# pybind11 submodules that the stub spells as ``class <name>:`` blocks.
SUBMODULES = ("constants", "ions", "ghk", "analytical", "dimensionless")

# Pre-existing stub/runtime drift that is tolerated. Keys are
# ``"ClassName"`` (or ``"<module>"`` for top-level names); values are the set
# of member names to ignore in *both* directions. Every entry must carry a
# comment explaining why it exists. Currently the stub and the runtime agree
# exactly, so this stays empty; add entries here (never silently widen the
# comparison logic) if a future binding change is intentionally unstubbed.
ALLOWLIST: dict[str, frozenset[str]] = {}

# Members that pybind11 places in every enum's ``__dict__`` but that the stub
# expresses through ``class X(Enum)`` inheritance instead.
_PYBIND_ENUM_IMPLICIT = frozenset({"__init__", "name", "value"})


# --------------------------------------------------------------------------
# Stub side
# --------------------------------------------------------------------------


def _members_of(class_node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(item.name)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            names.add(item.target.id)
        elif isinstance(item, ast.Assign):
            names.update(t.id for t in item.targets if isinstance(t, ast.Name))
        elif isinstance(item, ast.ClassDef):
            names.add(item.name)
    return names


def _is_enum_stub(class_node: ast.ClassDef) -> bool:
    return any(isinstance(b, ast.Name) and b.id == "Enum" for b in class_node.bases)


def _parse_stub() -> tuple[dict[str, ast.ClassDef], set[str], set[str]]:
    tree = ast.parse(STUB_PATH.read_text(encoding="utf-8"), filename=str(STUB_PATH))
    classes: dict[str, ast.ClassDef] = {}
    functions: set[str] = set()
    aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes[node.name] = node
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is None
        ):
            # ``Left: Boundary`` style module-level enum aliases. Annotated
            # assignments *with* a value are ``TypeAlias`` helpers, not runtime
            # objects, and are skipped.
            aliases.add(node.target.id)
    return classes, functions, aliases


STUB_CLASSES, STUB_FUNCTIONS, STUB_ALIASES = _parse_stub()


# --------------------------------------------------------------------------
# Runtime side
# --------------------------------------------------------------------------


def _runtime_public_names() -> tuple[set[str], set[str]]:
    classes: set[str] = set()
    functions: set[str] = set()
    for name in dir(_core):
        if name.startswith("_"):
            continue
        obj = getattr(_core, name)
        if inspect.isclass(obj):
            classes.add(name)
        elif inspect.isbuiltin(obj) or inspect.isroutine(obj):
            functions.add(name)
    return classes, functions


RUNTIME_CLASSES, RUNTIME_FUNCTIONS = _runtime_public_names()


def _runtime_declared_members(name: str) -> tuple[set[str], set[str]]:
    """Return ``(all_names, public_names)`` for a runtime class or submodule.

    ``all_names`` is used to validate stub members (inherited attributes count,
    so ``hasattr`` semantics apply); ``public_names`` is the set the stub is
    required to declare (own ``__dict__`` only, dunders excluded except a real
    pybind11 ``__init__``).
    """
    obj = getattr(_core, name)
    if isinstance(obj, types.ModuleType):
        public = {n for n in dir(obj) if not n.startswith("_")}
        return set(dir(obj)), public

    own = dict(vars(obj))
    public: set[str] = set()
    for member, value in own.items():
        if member == "__init__":
            # pybind11 gives non-constructible classes a slot-wrapper
            # ``__init__`` that only raises; there is nothing to stub.
            if type(value).__name__ == "instancemethod":
                public.add(member)
        elif not member.startswith("_"):
            public.add(member)
    return set(dir(obj)), public


def _allowed(name: str) -> frozenset[str]:
    return ALLOWLIST.get(name, frozenset())


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_stub_file_exists() -> None:
    assert STUB_PATH.is_file(), f"missing stub {STUB_PATH}"


def test_runtime_public_names_are_declared_in_stub() -> None:
    top_allow = _allowed("<module>")
    missing_classes = RUNTIME_CLASSES - set(STUB_CLASSES) - top_allow
    missing_functions = RUNTIME_FUNCTIONS - STUB_FUNCTIONS - top_allow
    assert not missing_classes, (
        f"runtime classes missing from stub: {sorted(missing_classes)}"
    )
    assert not missing_functions, (
        f"runtime functions missing from stub: {sorted(missing_functions)}"
    )


def test_stub_top_level_names_exist_at_runtime() -> None:
    top_allow = _allowed("<module>")
    runtime_names = set(dir(_core))
    stub_classes = set(STUB_CLASSES) - top_allow
    phantom_classes = {
        n
        for n in stub_classes
        if n not in RUNTIME_CLASSES
        and not (
            n in SUBMODULES and isinstance(getattr(_core, n, None), types.ModuleType)
        )
    }
    phantom_functions = {
        n
        for n in STUB_FUNCTIONS - top_allow
        if not (n in runtime_names and inspect.isroutine(getattr(_core, n)))
    }
    phantom_aliases = STUB_ALIASES - runtime_names - top_allow
    assert not phantom_classes, (
        f"stub classes absent at runtime: {sorted(phantom_classes)}"
    )
    assert not phantom_functions, (
        f"stub functions absent at runtime: {sorted(phantom_functions)}"
    )
    assert not phantom_aliases, (
        f"stub aliases absent at runtime: {sorted(phantom_aliases)}"
    )


def test_submodules_are_stubbed_as_classes() -> None:
    for name in SUBMODULES:
        assert isinstance(getattr(_core, name), types.ModuleType), name
        assert name in STUB_CLASSES, f"submodule {name!r} has no stub class block"


@pytest.mark.parametrize("class_name", sorted(STUB_CLASSES))
def test_class_members_match_runtime(class_name: str) -> None:
    node = STUB_CLASSES[class_name]
    if not hasattr(_core, class_name):
        pytest.skip("reported by test_stub_top_level_names_exist_at_runtime")

    allowed = _allowed(class_name)
    stub_members = _members_of(node) - allowed
    runtime_all, runtime_public = _runtime_declared_members(class_name)
    runtime_public -= allowed
    if _is_enum_stub(node):
        runtime_public -= _PYBIND_ENUM_IMPLICIT

    stub_only = {m for m in stub_members if m not in runtime_all}
    runtime_only = runtime_public - stub_members
    assert not stub_only, (
        f"{class_name}: stub declares members absent at runtime: {sorted(stub_only)}"
    )
    assert not runtime_only, (
        f"{class_name}: runtime members missing from stub: {sorted(runtime_only)}"
    )


def test_allowlist_entries_are_still_needed() -> None:
    """An allowlist entry that no longer masks any drift must be deleted."""
    stale: list[str] = []
    for name, members in ALLOWLIST.items():
        if name == "<module>":
            live = (RUNTIME_CLASSES | RUNTIME_FUNCTIONS) ^ (
                set(STUB_CLASSES) | STUB_FUNCTIONS
            )
        elif name in STUB_CLASSES and hasattr(_core, name):
            runtime_all, runtime_public = _runtime_declared_members(name)
            stub_members = _members_of(STUB_CLASSES[name])
            live = (stub_members - runtime_all) | (runtime_public - stub_members)
        else:
            live = set()
        stale.extend(f"{name}.{m}" for m in members if m not in live)
    assert not stale, f"ALLOWLIST entries no longer mask drift: {stale}"


# --------------------------------------------------------------------------
# Policy: only Nernst-Planck Neumann setters take a *flux*
# --------------------------------------------------------------------------

# Classes whose Neumann-type data genuinely are molar fluxes (Nernst-Planck).
NERNST_PLANCK_CLASSES = frozenset({"NernstPlanckSolver", "MultiIonSolver"})

# ``DarcyFlowSolver.set_neumann(side, flux)`` sets the outward pressure
# derivative dp/dn under a compatibility keyword that the stub docstring
# already flags; the canonical spelling is ``set_outward_pressure_gradient``.
# It is not part of the three-binding ``flux`` -> ``normal_derivative`` rename,
# so it is exempted here explicitly rather than hidden inside the xfail below.
FLUX_KWARG_EXEMPTIONS = frozenset({("DarcyFlowSolver", "set_neumann")})


def _neumann_setters_with_flux_kwarg() -> list[str]:
    offenders: list[str] = []
    for class_name, node in STUB_CLASSES.items():
        if class_name in NERNST_PLANCK_CLASSES:
            continue
        for item in node.body:
            if (
                not isinstance(item, ast.FunctionDef)
                or "neumann" not in item.name.lower()
            ):
                continue
            if (class_name, item.name) in FLUX_KWARG_EXEMPTIONS:
                continue
            params = {a.arg for a in item.args.args + item.args.kwonlyargs}
            if "flux" in params:
                offenders.append(f"{class_name}.{item.name}")
    return sorted(set(offenders))


@pytest.mark.xfail(
    strict=True,
    reason="pybind kwarg 'flux' is renamed to 'normal_derivative' in a later step",
)
def test_only_nernst_planck_neumann_setters_take_flux() -> None:
    offenders = _neumann_setters_with_flux_kwarg()
    assert not offenders, (
        "Neumann setters outside Nernst-Planck name their derivative 'flux': "
        f"{offenders}"
    )


def test_nernst_planck_neumann_setters_still_take_flux() -> None:
    """The Nernst-Planck setters really do take a flux; keep them spelled so."""
    for class_name in sorted(NERNST_PLANCK_CLASSES):
        node = STUB_CLASSES[class_name]
        signatures = [
            item
            for item in node.body
            if isinstance(item, ast.FunctionDef) and item.name == "set_neumann_boundary"
        ]
        assert signatures, f"{class_name}.set_neumann_boundary missing from stub"
        for sig in signatures:
            params = {a.arg for a in sig.args.args + sig.args.kwonlyargs}
            assert "flux" in params, (
                f"{class_name}.set_neumann_boundary must take 'flux'"
            )
