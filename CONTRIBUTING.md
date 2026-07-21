# Contributing to NautiPy

NautiPy values small changes that make coordinate handling or navigation work easier, safer, and more understandable.

Read [AGENTS.md](AGENTS.md), [docs/PRODUCT.md](docs/PRODUCT.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [ROADMAP.md](ROADMAP.md) before substantial work.

## Clean-slate policy

The old repository code is not a supported API. Contributions to the rewrite should not add:

- compatibility wrappers for experimental names;
- deprecated aliases or migration shims;
- a `legacy` package;
- tests that preserve incorrect formulas; or
- abstractions whose only purpose is supporting the old layout.

Reuse an old idea only after expressing it through the new API and independently verifying its behavior.

## Standard development setup

NautiPy must be developable with ordinary Python tooling:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

To build distributions:

```bash
python -m pip install build
python -m build
```

Activate the virtual environment using the normal command for your shell, or call its Python executable directly.

The project must not require Poetry, uv, Conda, Make, Docker, pre-commit, a Unix shell, or an editor plugin. Contributors may use any of them locally. The documented `python -m ...` commands remain the portable reference.

## Choosing work

Prefer the first incomplete milestone in [ROADMAP.md](ROADMAP.md), a clearly scoped issue, or a bug with a reproducible example.

A feature should strengthen one of these workflows:

1. parse and normalize coordinates;
2. inspect or convert coordinate formats;
3. calculate WGS84 navigation values;
4. exchange positions through lightweight formats; or
5. optionally estimate a position from bearings and ranges.

Features outside this scope need an explicit product-direction decision before implementation.

## Pull requests

Keep each pull request a working vertical slice. Packaging, coordinate formats, geodesics, release automation, and the optional fix engine should normally remain separate.

A useful pull request description states:

- the user problem;
- the observable behavior;
- examples or reference data;
- tests added;
- runtime and optional dependency effects;
- numerical assumptions and tolerances; and
- known limitations.

Do not mix broad cleanup into a behavioral change unless the cleanup is required. During Milestone 0, deleting the experimental implementation is required cleanup.

## Tests

Use standard-library `unittest` unless a concrete testing requirement justifies another tool. Tests should exercise public behavior rather than private implementation details.

### Coordinate changes

Include relevant cases for:

- ordinary valid input;
- harmless syntax variation;
- malformed input;
- legal range boundaries;
- NaN and infinity;
- sign and hemisphere conflicts;
- ambiguous coordinate order;
- decimal-comma ambiguity;
- formatting carry and negative zero; and
- parse/format round trips.

[docs/COORDINATES.md](docs/COORDINATES.md) is the coordinate behavior contract.

### Numerical changes

Include:

- at least one independent reference case;
- normal operating cases;
- difficult geometry;
- impossible or degenerate geometry;
- justified tolerances; and
- explicit failure or convergence behavior.

Synthetic cases with a known generating position are useful but do not replace independent reference results.

Tests must not depend on network services, changing online datasets, wall-clock dates, locale settings, or unseeded randomness.

## Public API

- Keep top-level exports small and intentional.
- Type-annotate public functions, dataclasses, methods, and result fields.
- Use descriptive exceptions, not assertions, for caller errors.
- State units in names, signatures, or documentation.
- Do not silently change coordinate order, Earth model, or units.
- Update public examples with the implementation.
- Before 1.0, prefer a clean correction over a deprecation layer for unreleased behavior.
- At and after 1.0, follow semantic versioning for documented APIs.

Internal names beginning with an underscore are not public unless documentation states otherwise.

## Dependency policy

Runtime dependencies carry long-term installation and support cost.

The target architecture is:

- coordinate parsing, formatting, GeoJSON, CLI, and models: standard library;
- normal navigation: at most one focused pure-Python WGS84 dependency;
- advanced fixing: NumPy and SciPy only in an optional `fix` extra.

Before adding a dependency, document:

1. the shipped feature that needs it;
2. the correctness or maintenance risk it removes;
3. why the standard library or an existing dependency is insufficient;
4. whether it can be isolated in an optional extra;
5. its Python and platform support; and
6. its import impact on coordinate-only use.

Do not add runtime dependencies for validation, unit enums, command-line parsing, logging, JSON, formatting, HTTP, testing convenience, or documentation generation.

Avoid exact runtime pins unless a verified incompatibility requires one. Use bounds only when they describe tested compatibility, and explain temporary exclusions in an issue or comment.

## Coordinate implementation

The parser should be a pipeline:

1. preserve the original input;
2. normalize harmless syntax;
3. extract axis and separator evidence;
4. run independent format candidates;
5. validate components and ranges;
6. resolve equivalent candidates; and
7. raise an actionable error when distinct candidates remain.

Do not grow one regular expression that accepts every format. Do not use likely geography to infer coordinate order.

Coordinate modules must remain usable without importing geodesic or scientific packages.

## Numerical implementation

- Default to WGS84 ellipsoidal calculations.
- Use the selected mature geodesic implementation rather than handwritten approximations.
- Keep third-party objects behind NautiPy result types.
- Normalize generated bearings and wrapped bearing residuals correctly.
- Treat non-convergence and weak geometry as outcomes to report.
- Never round intermediate values for display.

## Documentation

Examples should be copyable and tested against the built wheel. Clearly distinguish:

- accepted input from guessed input;
- true bearing from magnetic bearing;
- metres from display units;
- WGS84 geodesics from approximations; and
- solver convergence from good observation geometry.

Do not claim suitability for certified or safety-critical navigation.

## Build and release checks

Before review, run the canonical tests and build commands. Packaging and import changes must also be tested from the built wheel outside the source checkout.

Pull requests never publish packages. Maintainers create intentional semantic-version tags after merging release preparation. See [docs/RELEASING.md](docs/RELEASING.md).

## Security-sensitive reports

Do not post credentials, private position data, or a reproducible exploit in a public issue. Use GitHub private security reporting when enabled or contact the maintainer through the repository's published contact information.