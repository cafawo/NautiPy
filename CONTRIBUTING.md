# Contributing to NautiPy

Thank you for improving NautiPy. The project values small, understandable changes that make coordinate handling or navigation fixes easier and more trustworthy.

Read [AGENTS.md](AGENTS.md), [docs/PRODUCT.md](docs/PRODUCT.md), and [ROADMAP.md](ROADMAP.md) before proposing a substantial feature.

## Development baseline

NautiPy must be developable with standard Python tooling. Alternative tools are welcome, but they are not prerequisites.

After Milestone 0 in the roadmap is complete, the canonical setup is:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest
python -m build
```

Activate the virtual environment using the normal command for your shell, or call its Python executable directly.

The project must not require Poetry, uv, Conda, Make, Docker, pre-commit, a Unix shell, or an editor plugin. Contributors may use any of them locally. The repository's documented `python -m ...` commands remain the portable reference.

Until the packaging-baseline milestone lands, contributors and agents should implement it before relying on the commands above.

## Choosing work

Prefer the first incomplete item in [ROADMAP.md](ROADMAP.md), a clearly scoped issue, or a bug with a reproducible example.

Before adding a feature, confirm that it strengthens one of the core workflows:

- parse and normalize coordinates;
- convert coordinate formats;
- calculate navigation primitives;
- estimate a position from bearings and/or ranges; or
- inspect and exchange the resulting data.

Features outside that scope need an explicit product-direction discussion before code is added.

## Pull requests

Keep pull requests focused. Packaging, parsing, numerical solving, and release automation should normally be separate changes.

A useful pull request description includes:

- the user problem;
- the chosen behavior;
- examples or reference data;
- tests added;
- dependency or compatibility effects;
- numerical assumptions and tolerances; and
- known limitations.

Do not mix broad cleanup into a behavioral change unless the cleanup is required to implement or test it.

## Tests

Every public behavior needs tests. A test should fail for the relevant reason before the fix and pass after it.

### Coordinate changes

Include:

- ordinary valid inputs;
- harmless syntax variants;
- malformed input;
- range boundaries;
- sign and hemisphere conflicts;
- ambiguous coordinate order;
- decimal-comma ambiguity where relevant;
- formatting carry and negative-zero behavior; and
- parse/format round trips.

Use [docs/COORDINATES.md](docs/COORDINATES.md) as the behavioral contract.

### Numerical changes

Include:

- an independent reference case;
- the normal operating case;
- edge geometry;
- impossible or degenerate geometry;
- tolerance justification; and
- failure/convergence behavior.

Synthetic cases with a known generating position are useful, but they are not a substitute for at least one independent reference result.

Avoid tests that depend on network services, changing online datasets, wall-clock dates, locale settings, or random values without a fixed seed.

## Public API and compatibility

- Keep the top-level API small and intentional.
- Type-annotate public functions, methods, dataclasses, and result fields.
- Use descriptive exceptions instead of assertions for caller errors.
- State units in names, types, or documentation. NautiPy's internal defaults are metres and true degrees.
- Do not silently change coordinate order, Earth model, or units.
- Add deprecation warnings and migration guidance when a useful historical call can be supported safely.
- Do not preserve numerically incorrect behavior merely for compatibility.
- Update examples and API documentation in the same pull request as a public behavior change.

Internal modules and names beginning with an underscore are not public API unless documentation says otherwise.

## Dependency policy

Runtime dependencies carry long-term support cost. Add one only when it provides substantial correctness or numerical functionality that would be risky to reproduce.

Before adding a runtime dependency, document:

1. the hard problem it solves;
2. why the standard library or an existing dependency is insufficient;
3. its import and installation impact;
4. whether it is required for the core workflow or can be isolated; and
5. how its supported Python/platform range affects NautiPy.

Expected examples of justified dependencies are a mature WGS84 geodesic implementation and NumPy/SciPy for weighted nonlinear solving.

Do not add runtime dependencies for:

- argument validation;
- unit enums or simple conversion;
- command-line parsing;
- logging;
- JSON or GeoJSON point serialization;
- formatting;
- HTTP access; or
- development convenience.

Avoid exact runtime pins unless a known incompatibility requires one. Use lower or upper bounds only when they describe tested correctness, and add a comment or issue explaining temporary exclusions.

Development tools belong in an optional extra and should have one canonical configuration. Do not require overlapping tools that enforce the same rule.

## Coordinate implementation guidance

The coordinate parser should be a pipeline of normalization, candidate parsing, validation, and ambiguity resolution. Do not grow a single regular expression that accepts all formats.

Keep coordinate parsing and formatting in standard-library-only modules. Importing them must not load scientific solver dependencies.

When input is ambiguous, improve the error and explicit controls rather than adding another heuristic.

## Numerical implementation guidance

- Default to WGS84 ellipsoidal calculations.
- Use a mature geodesic implementation instead of another approximate formula.
- Keep optimizer details behind NautiPy result types.
- Scale mixed residuals using explicit uncertainty or documented defaults.
- Normalize bearing residuals across the `0°/360°` boundary.
- Treat non-convergence and weak geometry as results to report, not conditions to hide.
- Never round intermediate values for presentation.

## Documentation

Examples should be copyable and should use the current public API. Prefer a small complete example over long prose describing untested code.

Documentation must distinguish:

- accepted input from guessed input;
- true bearing from magnetic bearing;
- metres from display units;
- WGS84 geodesics from spherical approximations; and
- a converged result from a well-conditioned result.

Do not claim suitability for certified or safety-critical navigation.

## Build and release checks

Before requesting review, run the canonical tests and build commands. Changes to packaging or public imports must also be tested from the built wheel, not only from the source checkout.

Contributors do not publish releases from pull requests. Maintainers create intentional semantic-version tags after merging a release-preparation change. See [docs/RELEASING.md](docs/RELEASING.md).

## Reporting security-sensitive problems

Avoid posting credentials, private position data, or a reproducible exploit in a public issue. Use GitHub's private security-reporting mechanism when it is enabled for the repository, or contact the maintainer privately using the repository's published contact information.
