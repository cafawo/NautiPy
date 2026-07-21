# NautiPy product direction

## One-sentence promise

**NautiPy turns coordinates and imperfect navigation observations into validated positions with a small, understandable Python API.**

The package should feel useful within the first minute, remain explicit when an input is ambiguous, and expose enough diagnostics for users to understand whether a calculated fix is trustworthy.

## Why this package should exist

Python already has excellent libraries for geodesics, projections, vector navigation, and GIS. NautiPy should not compete by duplicating all of them. Its value is the workflow around those primitives:

1. Accept coordinates in the forms people actually receive them.
2. Detect and convert those forms safely.
3. Represent positions consistently.
4. Combine bearings and ranges into a fix.
5. Explain the result, its residuals, and its uncertainty.

This is a narrower and more approachable problem than general geospatial computing, and it is especially useful for scripts, field data cleanup, education, sensor prototypes, and small navigation tools.

## Primary users

- developers receiving coordinates from humans, spreadsheets, logs, devices, or copied text;
- navigators and engineers calculating positions from known landmarks, bearings, or ranges;
- educators demonstrating geodesic and position-fixing concepts;
- data analysts who need a lightweight bridge from messy coordinate text to GeoJSON or numerical analysis;
- library authors who want a clear position/fix API without adopting a broad GIS framework.

NautiPy is a calculation library, not certified navigation equipment. Documentation must not imply that it replaces regulated or safety-critical systems.

## Product principles

### Ease of use first

The common path should be one function call:

```python
from nautipy import parse_position

position = parse_position("N 50° 7' 19.2\"; E 8° 39' 56.5\"")
```

Users should not need to identify DMS versus decimal degrees before parsing. They should not need NumPy arrays for single positions. They should not need to understand the optimizer to solve a standard fix.

### Safe automation, not silent guessing

NautiPy should automatically handle harmless syntax variation. It must stop and explain when two interpretations would produce different positions.

Examples:

- `50.12, 8.66` can follow the documented default order `latlon`.
- `8.66, 50.12` with `order="auto"` is ambiguous and must not be silently swapped.
- `120.0, 50.0` with `order="auto"` can be identified as `lonlat` because `120.0` cannot be a latitude.
- `-50 N` is contradictory and should raise an error rather than applying one sign arbitrarily.

### Trustworthy results

Public calculations should default to WGS84 ellipsoidal geodesics. A position fix should return diagnostics, not only coordinates. Weak geometry, competing solutions, high residuals, or non-convergence should be represented explicitly.

### Minimal, durable foundations

Use the Python standard library for parsing, formatting, data models, JSON, CLI plumbing, and validation. Use mature scientific dependencies for numerical work that should not be reinvented. Avoid online services and bundled datasets that age independently of the package.

### Progressive disclosure

Simple functions should cover simple tasks. Detailed result objects and explicit options should be available without forcing every user to configure them.

## Core workflows

### 1. Parse and normalize a position

```python
from nautipy import parse_position

p1 = parse_position("50.12257, 8.66570")
p2 = parse_position("50° 7.3542' N; 8° 39.942' E")
p3 = parse_position("+50.12257+008.66570/")

assert p1 == p2 == p3
```

The exact examples and supported variations are specified in [COORDINATES.md](COORDINATES.md).

### 2. Convert coordinate formats

```python
from nautipy import convert_position

text = convert_position(
    "50.12257, 8.66570",
    to="dms",
    hemisphere=True,
)
```

Conversion should support practical output controls without exposing internal parser details.

### 3. Calculate navigation primitives

```python
from nautipy import destination, distance, initial_bearing, parse_position

start = parse_position("50.12257, 8.66570")
end = destination(start, bearing=90, distance=12_000)

metres = distance(start, end)
degrees_true = initial_bearing(start, end)
```

Distances are metres and bearings are true degrees by default. Alternative display units belong at the API boundary.

### 4. Solve a position fix

```python
from nautipy import BearingObservation, Position, RangeObservation, solve_fix

result = solve_fix(
    [
        BearingObservation(
            station=Position(50.116135, 8.670277),
            bearing=164.71,
            sigma=1.0,
        ),
        BearingObservation(
            station=Position(50.110347, 8.659873),
            bearing=192.22,
            sigma=1.0,
        ),
        RangeObservation(
            station=Position(50.112836, 8.666753),
            distance=1_599.2,
            sigma=15.0,
        ),
    ]
)

print(result.position)
print(result.residuals)
print(result.rms_error)
print(result.warnings)
```

The final names may evolve during implementation, but the workflow should remain this direct.

## Capabilities to ship

### Coordinate intake and conversion

- decimal degrees, signed or hemisphere-qualified;
- degrees and decimal minutes (`ddm`; accept `dmm` as an alias);
- degrees, minutes, and seconds;
- common Unicode and ASCII symbol variants;
- ISO 6709 forms that can be parsed unambiguously;
- NMEA latitude/longitude fields and pairs, without becoming a full NMEA sentence library;
- decimal-comma input where separators make the interpretation unambiguous;
- strings, numeric pairs, mappings with named latitude/longitude fields, and recognized GeoJSON points;
- explicit coordinate-order controls and informative ambiguity errors;
- formatting to canonical decimal degrees, DDM, DMS, ISO 6709, and NMEA coordinate fields;
- an inspection function that reports the detected format and normalization decisions.

### Position and geodesic calculations

- immutable validated `Position` values;
- inverse calculation: distance and initial/final bearing;
- direct calculation: destination from position, bearing, and distance;
- interpolation along a geodesic;
- nearest-position lookup;
- clearly named spherical approximations only where they provide a real use case.

### Position fixing

- two-bearing intersection;
- range-circle intersection with zero, one, or two candidate solutions;
- overdetermined bearing-only fixes;
- overdetermined range-only fixes;
- mixed bearing/range fixes;
- optional observation uncertainty and weighting;
- residuals in the observation's natural unit;
- convergence state, iteration summary, and objective value;
- geometry warnings and competing-solution handling;
- covariance or confidence-region estimates where the geometry and solver support them.

### Interchange and usability

- GeoJSON import/export for points and feature collections;
- compatibility adapters for the useful historical `Pos`, `haversine`, `bearing`, `triangulate`, and `multilaterate` entry points;
- concise examples and an optional small CLI for conversion and fix inspection;
- installable wheel and source distribution on PyPI;
- a conda-forge package maintained through the normal feedstock process.

## Deliberate non-goals

The following should not be added merely because they are nautical or geospatial:

- arbitrary coordinate reference system transformations;
- chart display, route planning, bathymetry, AIS, autopilot, or collision-avoidance systems;
- live GPS/device connections;
- complete NMEA sentence decoding and streaming;
- magnetic declination calculation or bundled magnetic models;
- tide, current, weather, ephemeris, or other time-varying data;
- map tiles, hosted APIs, remote lookup services, or data downloads at runtime;
- plotting, GUI, web-server, database, or dataframe frameworks;
- encrypted proximity matching;
- generic computational geometry unrelated to navigation fixes.

Users needing CRS transformation should combine NautiPy with `pyproj`. Users needing broad geodesic variants may use GeographicLib, PyGeodesy, or nvector directly. NautiPy should interoperate with those tools rather than absorb their entire scope.

## Public API character

The public API should be:

- small enough to learn from the top-level package namespace;
- type-annotated;
- based on ordinary Python values and dataclasses;
- explicit about units and coordinate order;
- free of required global configuration;
- deterministic and offline;
- stable once version 1.0 is reached.

Detailed parser and solver machinery should remain in submodules. Internal implementation classes are not automatically public API.

## Compatibility policy

The existing project is an early proof of concept. The modernization may change internals and correct numerical behavior. Preserve useful call patterns through compatibility wrappers where doing so does not compromise correctness.

Before version 1.0:

- breaking changes are allowed when documented in release notes;
- deprecation warnings are preferred when a practical migration path exists;
- incorrect or ambiguous behavior may be removed without reproducing it in the new core.

At and after version 1.0:

- follow semantic versioning;
- keep documented APIs compatible across minor and patch releases;
- reserve major releases for intentional breaking changes.

## Success criteria for version 1.0

Version 1.0 is ready when:

- common coordinate inputs work without a format argument;
- ambiguous inputs fail with actionable guidance;
- parsing, formatting, and geodesic calculations have independent reference tests;
- bearing, range, and mixed fixes return diagnostics and handle poor geometry explicitly;
- the package has a coherent typed API and migration path from the historical API;
- wheel and source distributions are reproducibly built and published through GitHub Actions;
- the package is accepted into conda-forge or has an active staged-recipes submission; and
- documentation contains copyable end-to-end examples for the four core workflows.
