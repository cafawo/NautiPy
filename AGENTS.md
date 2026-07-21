# NautiPy coding-agent guide

This file is the repository-level operating contract for coding agents and contributors.

## Read first

1. [Product direction](docs/PRODUCT.md)
2. [Architecture and dependency policy](docs/ARCHITECTURE.md)
3. [Coordinate behavior](docs/COORDINATES.md)
4. [Implementation roadmap](ROADMAP.md)
5. [Release and distribution plan](docs/RELEASING.md)
6. [Contributor workflow](CONTRIBUTING.md)

When documents disagree, prefer the more specific document and update the conflicting documentation in the same pull request.

## Clean-slate decision

The repository contains an early experiment, but it has not established a supported public package or API. Treat the existing implementation as reference material only.

For the rewrite:

- do not preserve the old module layout;
- do not add compatibility wrappers, aliases, deprecation layers, or a migration guide;
- do not retain incorrect behavior because an old example used it;
- keep only the project name, MIT license, useful domain ideas, and independently verified reference cases; and
- begin semantic compatibility promises with the first public release, with a stable contract starting at version 1.0.

It is acceptable—and expected—to delete `setup.py`, the old `nautipy/nautipy.py` implementation, and placeholder tests when the clean package foundation replaces them.

## Mission

NautiPy should make coordinate and small-scale navigation work unusually easy while remaining explicit about ambiguity and numerical limits.

Its value is, in order:

1. accept coordinates in the forms people actually paste, type, or receive from devices;
2. detect, validate, inspect, and convert those formats safely;
3. provide a small WGS84 navigation API for distance, bearing, destination, and interpolation; and
4. later provide position fixes from bearings and ranges without forcing scientific dependencies on every user.

NautiPy is not a general GIS framework or a live-navigation system.

## Core principles

### Ease of use is part of correctness

The common path should be one obvious function call:

```python
from nautipy import parse_position

position = parse_position("50° 7' 19.2\" N; 8° 39' 56.5\" E")
```

Public functions that consume positions should accept a validated `Position` and, where it stays unambiguous, the same convenient position-like inputs accepted by `parse_position`.

### Be permissive about presentation, strict about meaning

Normalize whitespace, Unicode symbols, hemisphere placement, and decimal separators only when meaning is preserved. Never silently choose between different valid positions. Raise an actionable ambiguity error instead.

### Keep the package small

Prefer a coherent top-level API over feature count. Do not add plugin systems, backend abstractions, broad GIS types, dataframe integrations, or framework-style configuration without a demonstrated user need.

### Dependency budget

- Coordinate parsing, formatting, validation, data models, GeoJSON, and CLI plumbing use the Python standard library.
- The initial package foundation has zero runtime dependencies.
- The navigation layer may add one mature, pure-Python WGS84 geodesic dependency when that layer is implemented.
- NumPy and SciPy may be added only in a later optional `fix` extra for overdetermined and mixed-observation solvers.
- Optional dependencies must not be imported during ordinary coordinate use.
- Do not add dependencies for validation, units, logging, argument parsing, JSON, formatting, HTTP, or development convenience.

A dependency is admitted only when it removes substantial correctness or numerical risk and is used directly by a shipped feature.

### Build useful vertical slices

Do not spend a pull request creating an empty package skeleton. Each milestone should leave a small working path that can be installed and demonstrated.

## Engineering rules

- Use `pyproject.toml` as the package metadata and dependency source of truth.
- Use a `src/nautipy/` layout and intentionally define the top-level API.
- Keep the import name `nautipy`.
- Use ordinary Python dataclasses, enums, protocols, and exceptions where they clarify behavior.
- Store latitude/longitude as decimal degrees, distances as metres, and bearings as true degrees internally.
- Default navigation calculations to WGS84.
- Do not use `assert` for public input validation.
- Do not silently swap latitude and longitude, wrap invalid input, or change units.
- Do not perform network access during import or calculations.
- Do not expose raw third-party result or optimizer objects as public API.
- Keep optional functionality behind clear module and import boundaries.
- Use standard-library `unittest` unless a concrete testing need justifies another dependency.
- Avoid generated boilerplate and abstractions that have only one implementation.

## Agent workflow

1. Read the documents above and select the first incomplete roadmap milestone.
2. Define observable behavior in tests before or with implementation.
3. Make one coherent vertical slice; avoid unrelated cleanup.
4. Update examples and public documentation with the code.
5. Run the canonical tests, build the wheel and sdist, and test the built wheel.
6. Report commands run, dependency changes, numerical assumptions, and remaining limitations.

Before version 1.0, breaking changes are allowed when they simplify or correct the emerging public API. Do not build deprecation machinery for unreleased behavior.

## Definition of done

A change is complete when:

- public behavior and error cases have meaningful tests;
- coordinate changes cover malformed and ambiguous input;
- numerical changes include independent reference cases and difficult geometry;
- the wheel and source distribution build successfully;
- the built wheel installs and imports in a clean environment;
- documentation matches the implemented API;
- no unnecessary dependency or compatibility layer was introduced; and
- CI passes on the supported Python range.

## First assignment

Implement **Milestone 0: clean package and decimal-degree vertical slice** from [ROADMAP.md](ROADMAP.md).

The first implementation pull request should:

- remove the legacy packaging, implementation, and placeholder test rather than wrapping them;
- add `pyproject.toml` and a `src/nautipy/` package;
- implement the immutable `Position` model, core coordinate exceptions, decimal-degree parsing for text and structured pairs, and canonical decimal formatting;
- define a small intentional top-level API;
- use standard-library tests;
- build and smoke-test the wheel and sdist; and
- add pull-request CI.

Do not implement legacy aliases, geodesics, DDM/DMS parsing, or the solver in that pull request.