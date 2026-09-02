"""Single source of truth for every deprecated public spelling in BioTransport.

Policy (see ``docs/notes/DEPRECATION_POLICY.md``): a public name or keyword that is
retired keeps working for at least two minor releases and emits
:class:`BioTransportDeprecationWarning` on every use.  The warning always names the
version that deprecated the spelling, the version that removes it, the replacement,
and the reason.  Nothing in this module changes numerical behaviour: a deprecated
spelling resolves to exactly the same object or code path as its replacement.

This module deliberately imports nothing from the rest of the package at import
time so the compiled extension can use it to emit the same warning category.
"""

from __future__ import annotations

import functools
import importlib
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, TypeVar

__all__ = [
    "BioTransportDeprecationWarning",
    "DeprecatedName",
    "ROOT_DEPRECATED",
    "ROOT_LAZY",
    "deprecated_callable",
    "deprecated_keyword",
    "module_getattr",
    "resolve",
    "warn_deprecated",
]

DEFAULT_SINCE = "0.2.0"
DEFAULT_REMOVAL = "0.4.0"

F = TypeVar("F", bound=Callable[..., Any])


class BioTransportDeprecationWarning(DeprecationWarning):
    """Category for every deprecation this package emits.

    Filter it with ``warnings.filterwarnings("error", category=BioTransportDeprecationWarning)``
    to make retired spellings fail loudly in your own test suite.
    """


@dataclass(frozen=True)
class DeprecatedName:
    """One retired public spelling and the object it must still resolve to."""

    old: str
    target: str
    replacement: str
    reason: str = ""
    since: str = DEFAULT_SINCE
    removal: str = DEFAULT_REMOVAL

    def __post_init__(self) -> None:
        if ":" not in self.target:
            raise ValueError(
                f"target for {self.old!r} must be written as 'module:attribute'"
            )
        if _version_key(self.removal) <= _version_key(self.since):
            raise ValueError(
                f"{self.old!r}: removal version {self.removal} must be later than "
                f"deprecation version {self.since}"
            )


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def resolve(target: str) -> Any:
    """Return the object named by a ``'module:attribute'`` target string."""

    module_name, _, attribute = target.partition(":")
    return getattr(importlib.import_module(module_name), attribute)


def deprecation_message(
    old: str,
    replacement: str,
    *,
    reason: str = "",
    since: str = DEFAULT_SINCE,
    removal: str = DEFAULT_REMOVAL,
) -> str:
    """Build the one message format every deprecation in this package uses."""

    message = (
        f"{old} is deprecated since {since} and will be removed in {removal}; "
        f"use {replacement}."
    )
    if reason:
        message += f" {reason.rstrip('.')}."
    return message


def warn_deprecated(
    old: str,
    replacement: str,
    *,
    reason: str = "",
    since: str = DEFAULT_SINCE,
    removal: str = DEFAULT_REMOVAL,
    stacklevel: int = 3,
) -> None:
    """Emit the package's deprecation warning with the standard message."""

    warnings.warn(
        deprecation_message(
            old, replacement, reason=reason, since=since, removal=removal
        ),
        BioTransportDeprecationWarning,
        stacklevel=stacklevel,
    )


def module_getattr(
    module_name: str,
    deprecated: Mapping[str, DeprecatedName],
    lazy: Mapping[str, str] | None = None,
) -> Callable[[str], Any]:
    """Return a PEP 562 ``__getattr__`` for ``module_name``.

    ``lazy`` maps names to ``'module:attribute'`` targets that resolve silently
    (they are simply not eagerly imported); ``deprecated`` maps retired names to
    :class:`DeprecatedName` records that warn on every access.
    """

    lazy_table = dict(lazy or {})

    def __getattr__(name: str) -> Any:
        target = lazy_table.get(name)
        if target is not None:
            return resolve(target)
        entry = deprecated.get(name)
        if entry is not None:
            warn_deprecated(
                f"{module_name}.{entry.old}",
                entry.replacement,
                reason=entry.reason,
                since=entry.since,
                removal=entry.removal,
                stacklevel=3,
            )
            return resolve(entry.target)
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return __getattr__


def deprecated_keyword(
    kwargs: dict[str, Any],
    old: str,
    new: str,
    current: Any,
    *,
    function: str,
    reason: str = "",
    since: str = DEFAULT_SINCE,
    removal: str = DEFAULT_REMOVAL,
) -> Any:
    """Fold a retired keyword argument into its replacement.

    Returns the value to use for ``new``.  Raises :class:`TypeError` when both
    spellings are supplied, exactly as the previous alias handling did.
    """

    if old not in kwargs:
        return current
    value = kwargs.pop(old)
    if current is not None:
        raise TypeError(f"Pass either {new} or {old}, not both")
    warn_deprecated(
        f"{function}({old}=...)",
        f"{function}({new}=...)",
        reason=reason,
        since=since,
        removal=removal,
        stacklevel=4,
    )
    return value


def deprecated_callable(
    replacement: str,
    *,
    reason: str = "",
    since: str = DEFAULT_SINCE,
    removal: str = DEFAULT_REMOVAL,
    name: str | None = None,
) -> Callable[[F], F]:
    """Decorate a function so every call warns and then forwards unchanged."""

    def decorate(function: F) -> F:
        old = name or function.__qualname__

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warn_deprecated(
                old, replacement, reason=reason, since=since, removal=removal
            )
            return function(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


# ---------------------------------------------------------------------------
# Root-namespace tables (biotransport.__init__ wires these through PEP 562).
# ---------------------------------------------------------------------------

_PROBLEM_ALIAS_REASON = (
    "the alias named one physics configuration of the same declarative builder; "
    "Problem describes diffusion, advection and reaction terms alike"
)

_ROOT_DEPRECATED: dict[str, DeprecatedName] = {
    "DiffusionProblem": DeprecatedName(
        old="DiffusionProblem",
        target="biotransport._core:TransportProblem",
        replacement="bt.Problem",
        reason=_PROBLEM_ALIAS_REASON,
    ),
    "LinearReactionDiffusionProblem": DeprecatedName(
        old="LinearReactionDiffusionProblem",
        target="biotransport._core:TransportProblem",
        replacement="bt.Problem",
        reason=_PROBLEM_ALIAS_REASON,
    ),
    "AdvectionDiffusionProblem": DeprecatedName(
        old="AdvectionDiffusionProblem",
        target="biotransport._core:TransportProblem",
        replacement="bt.Problem",
        reason=_PROBLEM_ALIAS_REASON,
    ),
}
# ``biotransport.run`` cannot live in this table: ``run`` is also the name of the
# submodule that defines ``solve``, so the package keeps an eager, warning
# wrapper for the retired function instead (see ``biotransport/__init__.py``).

#: Retired root-level names. Access through ``biotransport.<name>`` warns and
#: resolves to the target object.
ROOT_DEPRECATED: Mapping[str, DeprecatedName] = MappingProxyType(_ROOT_DEPRECATED)

_ROOT_LAZY: dict[str, str] = {}

#: Root-level names that resolve silently on demand without being part of
#: ``biotransport.__all__``.  Populated when the namespace is tiered.
ROOT_LAZY: Mapping[str, str] = MappingProxyType(_ROOT_LAZY)
