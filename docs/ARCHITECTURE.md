# Architecture and dependency policy

## Goal

NautiPy supports two connected workflows:

```text
messy coordinate input
        ↓
validated Position
        ├──→ format conversion or GeoJSON interchange
        └──→ WGS84 navigation

reference Positions + bearing/range observations
        ↓
diagnosed position fix
```

One ordinary installation provides the complete package. Internal layers keep
simple coordinate work independent from geodesic and scientific imports.

## Runtime layers

### Coordinate layer

The coordinate layer uses only the Python standard library. It owns:

- immutable `Position` values;
- coordinate normalization, detection, parsing, and validation;
- formatting, conversion, and inspection metadata;
- coordinate exceptions;
- two-dimensional GeoJSON interchange; and
- coordinate conversion and inspection CLI plumbing.

It does not import GeographicLib, NumPy, SciPy, GIS frameworks, dataframe
libraries, or network clients. GeoJSON code works with ordinary mappings;
callers use the standard-library `json` module for text or files.

See [COORDINATES.md](COORDINATES.md) and [GEOJSON.md](GEOJSON.md).

### Navigation layer

The navigation layer uses GeographicLib for WGS84 inverse, direct, and
geodesic-line calculations. It owns:

- distance and initial/final bearings;
- destinations;
- interpolation; and
- nearest-position lookup for ordinary iterables.

NautiPy does not maintain a second spherical backend or depend on a broad GIS
framework for these operations. GeographicLib is imported only when a
navigation or fix calculation needs it.

See [NAVIGATION.md](NAVIGATION.md).

### Fix layer

Public observation models, statuses, result models, and lightweight validation
are NautiPy-owned types. The numerical solver uses:

- NumPy for arrays and linear algebra;
- SciPy for bounded nonlinear least squares; and
- GeographicLib for exact WGS84 bearing and range predictions.

The solver works in bounded local east/north metre coordinates while evaluating
returned positions and residuals on WGS84. NumPy, SciPy, GeographicLib, and the
private solver module are loaded only when a candidate or fix calculation
requires them.

See [FIXES.md](FIXES.md).

## Import boundaries

Import boundaries are a tested design property:

- importing `nautipy`, parsing or formatting coordinates, using coordinate
  models, or importing the CLI does not load GeographicLib, NumPy, SciPy, or
  the private numerical solver;
- requesting a navigation calculation loads GeographicLib but not NumPy or
  SciPy; and
- requesting candidate geometry or `solve_fix` loads the numerical solver and
  its scientific dependencies.

The dependencies are installed in every environment; lazy imports preserve
separation of concerns and startup cost, not separate product variants.

## Dependency policy

`pyproject.toml` is the dependency source of truth. The current direct runtime
dependencies are:

- GeographicLib for WGS84 geodesics;
- NumPy for numerical arrays and linear algebra; and
- SciPy for nonlinear least-squares optimization.

Do not add dependencies for validation, units, logging, argument parsing, JSON,
formatting, HTTP, development convenience, or an abstraction with only one
implementation.

Before adding a runtime dependency, establish:

1. which shipped user-facing feature requires it;
2. what correctness or maintenance risk it removes;
3. why the standard library and existing dependencies are insufficient;
4. whether its imports remain outside coordinate-only use;
5. whether it supports every Python version and platform declared by NautiPy;
   and
6. how built-artifact tests exercise it.

If those answers are weak, do not add the dependency.

## Public API ownership

The package and public-module `__all__` values define the intentional import
surface:

- `nautipy` exposes the common coordinate, navigation, and fixing API;
- `nautipy.geojson` exposes specialized GeoJSON helpers without expanding the
  top-level namespace; and
- public submodules may expose documented typing aliases and related names for
  advanced users.

Modules and names beginning with an underscore are private. Third-party result
objects, parser internals, optimizer state, and backend dictionaries never
become public results.

Functions consuming locations share `Position` and the documented
`PositionInput` forms where accepting those forms is unambiguous. Parsing logic
is centralized rather than duplicated in navigation, fixing, or interchange
code. `format_position` deliberately accepts a validated `Position`;
`convert_position` is the parse-and-format convenience API.

## Data conventions

- Latitude and longitude are finite decimal-degree `float` values.
- User latitude is in `[-90, 90]`; user longitude is in `[-180, 180]`.
- Invalid input is rejected rather than wrapped or silently reordered.
- Distances are metres internally.
- Bearings are true degrees clockwise from north and generated bearings are
  normalized to `[0, 360)`.
- Navigation and fixing use WGS84.
- Display rounding occurs only at formatting boundaries.
- Optional position metadata does not affect coordinate equality or hashing.

## Error and result design

Caller errors use the documented NautiPy exception hierarchy and explain how
to correct invalid input where practical. Public validation never relies on
`assert`.

Recoverable parser information belongs in `ParseResult`, not global logging or
routine Python warnings. Numerical result objects distinguish convergence,
uniqueness, geometry quality, and fit quality; a successful optimizer step is
not by itself a trustworthy fix.

## Packaging

NautiPy uses PEP 517/PEP 621 metadata, setuptools, and a `src/` package layout.
The NautiPy wheel itself is platform-independent; NumPy and SciPy may install
platform-specific dependency wheels. The `nautipy` console entry point uses the
standard-library CLI implementation.

Build and release procedure belongs in [RELEASING.md](RELEASING.md). Supported
API and versioning policy belongs in [SUPPORT.md](SUPPORT.md).
