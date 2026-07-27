# NautiPy product direction

## Promise

NautiPy turns real-world coordinate input into validated positions and makes
common WGS84 navigation calculations easy through one small Python package.

It should be useful within a minute of installation, explicit when input or
geometry is ambiguous, deterministic, and fully offline during ordinary use.

## Product priorities

NautiPy's value is, in order:

1. accepting coordinates in the forms people and devices actually use;
2. detecting, validating, inspecting, and converting those forms safely;
3. representing positions consistently;
4. calculating common WGS84 navigation quantities; and
5. estimating positions from bearings and ranges with useful diagnostics.

Coordinate usability is the first differentiator. Diagnosed position fixing is
the advanced differentiator. General GIS is not the goal.

## Principles

### Make the common path obvious

Users should not need to identify DD, DDM, DMS, ISO 6709, or NMEA notation
before parsing a position:

```python
from nautipy import parse_position

position = parse_position("N 50° 7' 19.2\"; E 8° 39' 56.5\"")
```

The common coordinate, navigation, and fixing API is available from the
top-level `nautipy` namespace. GeoJSON helpers and specialized typing aliases
remain grouped in their public submodules.

### Normalize presentation, never meaning

NautiPy accepts harmless variations in whitespace, symbols, decimal
separators, and hemisphere placement. It never chooses silently between
different valid locations. An ambiguity error explains the competing
interpretations and how to select one.

### Provide one complete installation

A published NautiPy release uses one command, `python -m pip install nautipy`,
and provides every shipped feature. A repository checkout provides the same
feature set through `python -m pip install .`. GeographicLib supplies WGS84
geodesics; NumPy and SciPy supply the numerical foundation for bearing and
range fixes. There is no feature extra or reduced installation variant.

The implementation remains layered: coordinate-only use does not load
GeographicLib, NumPy, SciPy, or the numerical fix solver.

### Keep the public surface coherent

The top-level API contains the position, coordinate, navigation, and fixing
types and functions users normally need. Parser tokens, optimizer objects,
third-party results, and other implementation details remain private.

### Make correctness inspectable

Navigation defaults to WGS84 and true bearings. A position fix reports
convergence, residuals, geometry, ambiguity, and local uncertainty where
meaningful instead of returning only a plausible-looking coordinate.

## Current capability

The package implements:

- immutable, validated `Position` values;
- detection and conversion of DD, DDM, DMS, two-dimensional ISO 6709, and
  NMEA coordinate fields;
- explicit coordinate-order controls and scalar and batch inspection
  diagnostics;
- WGS84 distance, endpoint bearings, destination, interpolation, and
  nearest-position lookup;
- GeoJSON Point and FeatureCollection interchange;
- a coordinate conversion and inspection CLI;
- two-bearing and two-range candidate geometry;
- weighted bearing-only, range-only, and mixed-observation fixes; and
- residual, convergence, geometry, ambiguity, and local uncertainty
  diagnostics.

The behavioral references are:

- [coordinate input and conversion](COORDINATES.md);
- [WGS84 navigation](NAVIGATION.md);
- [GeoJSON interchange](GEOJSON.md); and
- [bearing and range fixes](FIXES.md).

Release work and future features belong in the
[roadmap](../ROADMAP.md) and [release plan](RELEASING.md), not in this product
contract.

## Scope

NautiPy supports small, explicit coordinate-to-position,
position-to-navigation, and observation-to-fix workflows. It favors ordinary
Python values and focused result models over framework abstractions.

The package is not certified navigation equipment. Results remain subject to
the accuracy of the supplied coordinates, observations, uncertainties, and
documented numerical model.

## Non-goals

Do not add these without an intentional change to product direction:

- arbitrary CRS transformation or general GIS analysis;
- chart display, routing, collision avoidance, AIS, vessel control, or live
  navigation;
- complete NMEA sentence decoding or streaming;
- live GPS or other device connections;
- magnetic models, tides, currents, weather, or ephemerides;
- map tiles, hosted APIs, or runtime downloads;
- plotting, GUI, web-server, database, or dataframe frameworks;
- plugin systems or selectable calculation backends;
- a general units package; or
- generic computational geometry unrelated to positions and fixes.

Users needing broader GIS, CRS, visualization, or live-data capabilities
should combine NautiPy with specialist packages.

## Compatibility direction

Published compatibility is defined in [SUPPORT.md](SUPPORT.md). Undocumented
historical names and behavior are not part of that contract.
