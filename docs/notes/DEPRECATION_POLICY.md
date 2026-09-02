# Deprecation policy

BioTransport is alpha software, but a scientific user's script should never
change meaning silently. Retiring a public spelling therefore follows one rule
set, implemented in `python/biotransport/_deprecation.py` and enforced by
`python/tests/test_deprecations.py`.

## Rules

1. **Window.** A retired name, keyword, or attribute keeps working for at least
   two minor releases after the release that deprecates it. The default window
   is "deprecated in 0.2.0, removed in 0.4.0".
2. **Same behaviour.** A deprecated spelling resolves to exactly the same
   object or code path as its replacement. Deprecation never changes numerics,
   validation, or exceptions.
3. **One warning category.** Every deprecation emits
   `biotransport.BioTransportDeprecationWarning`, a subclass of
   `DeprecationWarning`, so it can be filtered or promoted to an error by class.
4. **One message format.** `<old> is deprecated since <since> and will be
   removed in <removal>; use <replacement>. <reason>.` The replacement is the
   exact spelling to write instead; the reason says what was ambiguous or
   duplicated.
5. **One table.** Root-level retirements live in `_deprecation.ROOT_DEPRECATED`;
   keyword retirements use `deprecated_keyword`; function retirements use
   `deprecated_callable`. A parametrized test asserts that every entry warns
   and resolves to its documented target.
6. **Examples and documentation never use a deprecated spelling.** The examples
   API guard fails if one appears.
7. **Removal is a separate, announced change** recorded in `CHANGELOG.md`.

## Silencing or enforcing

```python
import warnings
from biotransport import BioTransportDeprecationWarning

warnings.filterwarnings("ignore", category=BioTransportDeprecationWarning)  # silence
warnings.filterwarnings("error", category=BioTransportDeprecationWarning)   # enforce
```

The package's own test suite runs with the warning promoted to an error so that
no internal code path uses a retired spelling.
