# NautiPy coding-agent guide

This file is the repository-level operating contract for coding agents and human contributors. Read it before changing code.

## Read first

1. [Product direction](docs/PRODUCT.md)
2. [Coordinate input and conversion specification](docs/COORDINATES.md)
3. [Implementation roadmap](ROADMAP.md)
4. [Release and distribution plan](docs/RELEASING.md)
5. [Contributor workflow](CONTRIBUTING.md)

When these documents disagree, prefer the more specific document. Record intentional changes to product direction in the same pull request as the code that needs them.

## Mission

NautiPy should make common nautical-position work easy without hiding uncertainty or silently guessing. Its main value is:

- accepting the coordinate formats people actually paste or type;
- normalizing them into one safe internal representation;
- converting them into clear output formats;
- calculating dependable WGS84 navigation primitives; and
- estimating a position from bearing, range, or mixed observations while returning useful diagnostics.

NautiPy is not intended to become a general GIS framework.

## Core principles

### 1. Ease of use is a feature

The simple path must be obvious and require little configuration. Common inputs should work directly:

```python
from nautipy import parse_position

position = parse_position("50° 7' 19.2\" N, 8° 39' 56.5\" E")
```

Advanced users must still have explicit controls for coordinate order, format, units, Earth model, tolerances, and uncertainty.

### 2. Be permissive at the input boundary and strict internally

Normalize harmless differences such as whitespace, Unicode degree/minute/second symbols, hemisphere placement, and decimal separators where they are unambiguous. Once parsed, store validated numeric values in a single canonical representation.

Never silently choose between genuinely different interpretations. Return an ambiguity error that explains the candidates and the argument that resolves them.

### 3. Prefer a small coherent API over broad feature count

Add functionality when it strengthens the coordinate-to-position or observation-to-fix workflows. Do not add unrelated GIS, charting, live-data, or application-framework features.

### 4. Use established algorithms for hard numerical work

Do not maintain hand-written approximations when a mature, well-tested library already provides the geodesic or numerical primitive. Standard-library code is preferred for parsing, validation, formatting, data models, and file handling. A runtime dependency is justified only when it provides substantial correctness or numerical value.

The expected dependency shape is:

- coordinate parsing and formatting: Python standard library;
- ellipsoidal geodesics: a mature WGS84 implementation such as GeographicLib;
- weighted nonlinear fixing: NumPy/SciPy when the solver requires them.

Do not add a runtime dependency for logging, validation, units, command-line parsing, formatting, or simple serialization.

### 5. Avoid stale or fragile requirements

Do not bundle online data, magnetic-declination tables, chart databases, tide/current data, or assumptions tied to a particular external service. Do not require a particular environment manager, editor extension, shell, operating system, or hosted service for local development.

Use current Python packaging standards, but do not pin user dependencies more tightly than correctness requires. Keep support policy in `pyproject.toml`, not duplicated across prose files.

### 6. Make correctness inspectable

A successful fix must be more than a latitude/longitude pair. Return convergence state, residuals, uncertainty or geometry information when available, and warnings for weak or ambiguous geometry. Invalid inputs must raise descriptive exceptions rather than fail through assertions.

## Product boundaries

### In scope

- automatic detection and conversion of common coordinate formats;
- validated latitude/longitude positions on WGS84;
- distance, initial bearing, destination, interpolation, and related navigation primitives;
- bearing-only, range-only, and mixed-observation position fixes;
- weighted observations, residuals, uncertainty estimates, and geometry diagnostics;
- nearest-position queries;
- lightweight GeoJSON interchange;
- a small command-line interface when it directly improves coordinate conversion or fix inspection;
- compatibility wrappers for useful parts of the historical API.

### Out of scope unless product direction is deliberately changed

- arbitrary CRS transformation and general GIS analysis;
- nautical charts, routing, collision avoidance, AIS, vessel control, or live navigation systems;
- weather, tides, currents, magnetic models, or other time-varying datasets;
- a complete NMEA sentence-processing stack;
- GUI, web service, database, or plotting frameworks;
- privacy/encryption schemes;
- reimplementing mature geodesic libraries.

## Engineering rules

- Use `pyproject.toml` as the packaging and dependency source of truth.
- Preserve the import name `nautipy`.
- Keep public values typed and documented. Prefer dataclasses and enums from the standard library where they clarify the API.
- Store positions as decimal degrees, distances as metres, and bearings as degrees clockwise from true north internally. Convert only at API boundaries.
- Default to WGS84. Any spherical approximation must be explicitly named and requested.
- Do not perform network access at import time or during ordinary calculations.
- Do not use `assert` for public input validation.
- Do not silently swap latitude and longitude.
- Do not round intermediate calculations for display convenience.
- Do not expose raw optimizer objects as the public result type.
- Keep optional integrations outside the core import path.
- Preserve backwards compatibility through documented wrappers and deprecations where practical; do not preserve incorrect numerical behavior.

## Agent workflow

1. Inspect the repository and read the documents listed above.
2. Select the first incomplete roadmap milestone or a clearly scoped issue.
3. State the intended behavior in tests before or with the implementation.
4. Make one coherent change. Avoid opportunistic rewrites unrelated to the selected milestone.
5. Update public documentation and examples whenever behavior changes.
6. Run the repository's canonical test and package-build commands.
7. Report remaining limitations, numerical assumptions, and compatibility effects in the pull request.

Do not invent requirements that are absent from the product documents. When a decision is necessary, choose the simplest standard approach that preserves correctness and note the decision in the pull request.

## Definition of done

A change is complete when:

- public behavior is covered by meaningful tests, including error cases;
- coordinate changes include ambiguous and malformed inputs, not only happy paths;
- numerical changes include independent reference cases and edge geometry;
- the wheel and source distribution build successfully;
- the built wheel can be installed and imported in a clean environment;
- documentation matches the public API;
- no unnecessary runtime dependency has been introduced; and
- compatibility or deprecation behavior is explicit.

## First assignment

Start with **Milestone 0: packaging and trustworthy baseline** in [ROADMAP.md](ROADMAP.md). Do not begin the new solver before the package builds cleanly, the placeholder failing test is replaced, and continuous integration verifies the installable artifact.

After Milestone 0, implement **Milestone 1: coordinate intake and conversion** before expanding navigation algorithms. Coordinate usability is the first product differentiator and should establish the public `Position` model used by later milestones.
