# Architecture and dependency policy

## Goal

NautiPy should be easy to install, easy to understand, and useful without pulling a GIS or scientific-computing stack into every environment.

The architecture follows the user journey:

```text
messy coordinate input
        ↓
validated Position
        ↓
format conversion or WGS84 navigation
        ↓
optional bearing/range fix
```

The repository is a clean rewrite. The old implementation does not define module boundaries, public names, or compatibility requirements.

## Package layers

### 1. Coordinate core — standard library only

This layer provides:

- immutable `Position` values;
- coordinate normalization and detection;
- parsing and validation;
- formatting and conversion;
- parser inspection metadata;
- coordinate-specific exceptions; and
- GeoJSON Point and FeatureCollection interchange.

It must not import GeographicLib, NumPy, SciPy, pandas, pyproj, Shapely, or network clients.

Typical public functions:

```python
parse_position(...)
inspect_position(...)
format_position(...)
convert_position(...)
```

GeoJSON interchange remains grouped in `nautipy.geojson` through explicit
Point and FeatureCollection functions. The `nautipy` console script delegates
to the same coordinate functions and does not add another parsing layer.

### 2. Navigation core — one focused dependency

This layer provides:

- inverse geodesics: distance and initial/final bearings;
- direct geodesics: destination from position, bearing, and distance;
- interpolation along a geodesic; and
- nearest-position lookup for ordinary iterables.

Use one mature, pure-Python implementation of WGS84 geodesics rather than
maintaining approximate formulas. GeographicLib 2.1 or newer is the selected
implementation. It is loaded lazily so coordinate-only use remains independent
of geodesic code.

The dependency is part of the normal `nautipy` installation only when the navigation API ships. Do not add a broad CRS/GIS framework merely for direct and inverse geodesics.

### 3. Fix engine — optional extra

The fix engine is valuable but materially heavier. It is isolated from the
stable coordinate and navigation layers behind an optional extra.

Installation:

```bash
python -m pip install "nautipy[fix]"
```

The extra contains only NumPy and SciPy for:

- overdetermined bearing-only fixes;
- range-only fixes;
- mixed bearing/range fixes;
- weighted residuals;
- covariance or confidence estimates; and
- geometry diagnostics.

Coordinate and navigation imports must continue to work when the optional extra is absent. Calling optional functionality without its dependencies should raise one short error with the exact installation command.

`nautipy.fix` itself contains standard-library observation and result models.
It loads the private numerical implementation, NumPy, and SciPy only when a
candidate or solver function is called. The numerical layer uses bounded local
east/north metre coordinates and exact WGS84 predictions. See [FIXES.md](FIXES.md).

## Dependency budget

### Normal installation

Target at first public release:

- Python standard library;
- at most one runtime dependency, used for WGS84 geodesics.

### Optional fix installation

- NumPy;
- SciPy;
- no additional optimization framework unless a demonstrated requirement cannot be met by SciPy.

### Development and release tooling

Prefer:

- `unittest` for tests;
- `build` for wheel/sdist creation;
- GitHub Actions for CI and release automation; and
- PyPI Trusted Publishing for deployment.

Do not make Poetry, uv, Conda, Make, Docker, pre-commit, a Unix shell, or an editor extension mandatory. Contributors may use them locally.

## Dependency admission checklist

Before adding a runtime dependency, answer all of the following in the pull request:

1. Which shipped user-facing feature needs it?
2. What correctness or maintenance risk does it remove?
3. Why is the standard library or an existing dependency insufficient?
4. Can it be isolated in an optional extra?
5. Does it support the Python versions and platforms declared by NautiPy?
6. Does importing coordinate-only functionality avoid importing it?

If the answers are weak, do not add the dependency.

## Initial module layout

Start small and split only when a module has a clear independent responsibility:

```text
src/
└── nautipy/
    ├── __init__.py
    ├── position.py
    ├── coordinates.py
    ├── errors.py
    ├── geodesic.py       # added with the navigation milestone
    ├── geojson.py        # added when interchange ships
    ├── cli.py            # added only if the CLI ships
    ├── fix.py            # dependency-free public optional models/API
    └── _fix_solver.py    # lazily loaded numerical implementation
```

Parser internals may later move into a private `nautipy._coordinates` package when separate normalization, candidate parsing, and formatting modules make the code clearer. Do not create that hierarchy before it is needed.

Tests mirror public behavior rather than internal file structure:

```text
tests/
├── test_position.py
├── test_coordinates.py
├── test_geodesic.py
├── test_geojson.py
└── test_fix.py
```

## Public API

The top-level namespace should contain the small set most users need. The exact list is reviewed before each pre-1.0 release.

Target 0.1 surface:

```python
Position
ParseResult
InverseResult

parse_position
inspect_position
format_position
convert_position

inverse
distance
initial_bearing
destination
interpolate
nearest_position

NautiPyError
NavigationError
CoordinateError
CoordinateParseError
CoordinateRangeError
AmbiguousCoordinateError
FixError
FixDependencyError
```

The less frequently used `to_geojson_point`, `from_geojson_point`,
`to_geojson_feature_collection`, and `from_geojson_feature_collection`
functions are public from `nautipy.geojson` rather than expanding the
top-level namespace.

Observation models, candidate geometry, and solver functions are public from
`nautipy.fix`, not the top-level namespace. Only the common fix exception types
join the top-level exception hierarchy.

Detailed parser helpers, token types, backend objects, and third-party result values remain private.

All functions accepting a location should support `Position`. Functions may also accept documented position-like values by passing them through one shared coercion function. Do not duplicate parsing logic in geodesic or GeoJSON modules.

## Data conventions

- Latitude and longitude: finite decimal-degree `float` values.
- Optional position identifier: a string or finite JSON-style number.
- Optional position description: a string; metadata does not affect equality.
- Latitude range: `[-90, 90]`.
- Longitude input range: `[-180, 180]`; invalid user input is not silently wrapped.
- Distance: metres internally.
- Bearing: degrees clockwise from true north, normalized to `[0, 360)` for generated results.
- Earth model: WGS84 for public navigation defaults.
- Display rounding: only at formatting boundaries.

## Error design

Caller errors use a small documented exception hierarchy. Messages should explain how to correct the input.

Recoverable parser details belong in `ParseResult`, not in global logging or routine Python warnings. Numerical result objects should distinguish successful convergence from good geometry; the two are not equivalent.

## Deliberate omissions

Do not add:

- backwards-compatibility wrappers for the experimental repository code;
- a plugin architecture or selectable geodesic backends;
- a general units framework;
- generic CRS support;
- dataframe-specific return types;
- a network service or remote data lookup;
- global configuration; or
- import-time environment detection.

These omissions are part of the lightweight design, not missing scaffolding.

## Packaging shape

- PEP 517/PEP 621 metadata in `pyproject.toml`.
- Standard setuptools build backend.
- `src/` layout.
- Pure-Python wheel (`py3-none-any`) while the project contains no compiled code.
- Static package version in project metadata until a more complex mechanism proves necessary.
- Wheel and sdist built once per release and tested before publishing.
- A `nautipy` console entry point backed by the standard-library CLI module.
- Tag-only publication through tested artifacts and short-lived OIDC identity.

The package should begin at version `0.1.0` for its first public release. Repository experiments that were never distributed do not require a higher starting version or migration path.
