# NautiPy

**Effortless coordinate handling, trustworthy WGS84 navigation, and diagnosed
position fixes in one Python package.**

> **Status:** clean pre-release rewrite. The experimental code previously in this repository has been replaced and is not a supported API. The first public package will start at version `0.1.0`.

## Current development slice

The installable development package provides complete coordinate intake,
inspection, formatting, and conversion for DD, DDM, DMS, ISO 6709, and NMEA
field pairs. It also provides WGS84 distance, bearings, destinations,
interpolation, nearest-position lookup, GeoJSON interchange, and a small
command-line interface. Diagnosed bearing and range position fixes are
included in the same package and exposed through the same top-level API:

```bash
python -m pip install nautipy
```

```python
from nautipy import format_position, parse_position

position = parse_position("50° 7' 21.252\" N; 8° 39' 56.52\" E")
print(format_position(position))  # 50.122570, 8.665700
```

Use `order="lonlat"` for longitude-first input. `order="auto"` accepts cases
where numeric range proves the order or both orders produce the same position;
materially different candidates raise an actionable
`AmbiguousCoordinateError`. Detection is automatic, or `format="dd"`,
`"ddm"` (`"dmm"` is accepted as an input alias), `"dms"`, `"iso6709"`, or
`"nmea"` can select a format explicitly. Direction words and decimal commas
with an unambiguous pair separator are also accepted.

## What NautiPy is for

NautiPy focuses on the path from messy coordinate input to a validated, useful position:

```text
paste or receive coordinates
          ↓
automatically detect and validate them
          ↓
convert, inspect, exchange, or calculate navigation values
```

The main value is:

- automatic detection of common coordinate formats;
- clear errors when latitude/longitude order or syntax is genuinely ambiguous;
- conversion among decimal degrees, DDM, DMS, ISO 6709, and NMEA fields;
- a small immutable `Position` model;
- WGS84 distance, bearing, destination, and interpolation;
- diagnosed bearing, range, and mixed-observation position fixes;
- GeoJSON Point and FeatureCollection interchange;
- `nautipy convert` and `nautipy inspect` command-line workflows; and
- one installation, with the complete fixing API available at top level.

GeographicLib, NumPy, and SciPy are normal runtime dependencies. The
coordinate implementation itself remains standard-library-only and
coordinate-only module use does not import the geodesic or scientific layers.

## Coordinate API

These examples are covered by the installed-artifact smoke test.

### Parse whatever notation you have

```python
from nautipy import parse_position

p1 = parse_position("50.12257, 8.66570")
p2 = parse_position("50° 7.3542' N; 8° 39.942' E")
p3 = parse_position("+50.12257+008.66570/")

assert p1 == p2 == p3
```

### Inspect what was detected

```python
from nautipy import inspect_position

result = inspect_position("5007.3542,N,00839.9420,E")
print(result.position)
print(result.format)
print(result.evidence)
print(result.normalizations)
```

### Convert formats

```python
from nautipy import convert_position

text = convert_position(
    "50.12257, 8.66570",
    to="dms",
)

assert text == "50° 7′ 21.25″ N; 8° 39′ 56.52″ E"
```

`format_position` uses documented fixed defaults: six degree decimals for DD
and ISO 6709, four minute decimals for DDM and NMEA, and two second decimals
for DMS. Pass `precision=` to choose the decimal places in the least-significant
displayed unit. DDM and DMS default to Unicode symbols and hemispheres; use
`notation="signed"` or `symbols="ascii"` when needed.

### Calculate navigation values

```python
from nautipy import destination, distance, inverse

start = "50.12257, 8.66570"
end = destination(start, bearing=90, distance=12_000)

assert abs(distance(start, end) - 12_000) < 1e-6

result = inverse(start, end)
print(result.initial_bearing)
print(result.final_bearing)
```

Distances use metres and bearings use true degrees. Navigation follows WGS84
ellipsoidal geodesics through GeographicLib; it does not use a spherical
approximation. See the [navigation specification](docs/NAVIGATION.md) for
coincident positions, interpolation, and nearest-position behavior.

### Estimate a bearing/range fix

```python
from nautipy import Position, RangeObservation, solve_fix

references = (
    Position(50.116135, 8.670277),
    Position(50.112836, 8.666753),
    Position(50.110347, 8.659873),
)
observations = tuple(
    RangeObservation(reference, measured, uncertainty=2.0)
    for reference, measured in zip(
        references,
        (1_275.251, 1_599.237, 1_917.145),
    )
)

result = solve_fix(ranges=observations)
if result.success:
    print(result.position)
else:
    print(result.status, result.competing_positions)
```

Bearings are measured at the unknown position toward known references. Every
observation requires a one-standard-deviation uncertainty, and every result
reports residuals plus convergence and geometry diagnostics. See the
[bearing/range fix specification](docs/FIXES.md) for direction, units,
ambiguity, regional search bounds, and covariance behavior.

### Exchange GeoJSON

```python
from nautipy import Position
from nautipy.geojson import (
    from_geojson_feature_collection,
    to_geojson_feature_collection,
)

stations = [
    Position(
        50.12257,
        8.66570,
        identifier="station-1",
        description="Reference station",
    ),
]
collection = to_geojson_feature_collection(stations)
assert from_geojson_feature_collection(collection) == tuple(stations)
```

GeoJSON coordinates are always longitude/latitude. Collection identifiers and
descriptions round trip through standard Feature members; unsupported
geometry is rejected. See [GeoJSON interchange](docs/GEOJSON.md).

### Convert and inspect from the command line

```console
$ nautipy convert "50° 7.3542' N; 8° 39.942' E" --to dd
50.122570, 8.665700

$ nautipy inspect "+50.12257+008.66570/"
```

`inspect` writes deterministic JSON containing the selected position, detected
format, normalization evidence, source resolution, and candidate diagnostics.
Both commands are offline and use the same public parser and formatter as the
Python API. They write UTF-8 output consistently, including when output is
redirected. `python -m nautipy` provides the same interface.

## Input philosophy

NautiPy is permissive about presentation and strict about meaning.

It should handle harmless variation in whitespace, Unicode symbols, decimal separators, and hemisphere placement. It must not silently choose between two valid locations.

```python
parse_position("120, 50", order="auto")  # longitude/latitude is provable
parse_position("8, 50", order="auto")    # raises AmbiguousCoordinateError
```

## Dependency architecture

The package is deliberately layered:

- **coordinates:** Python standard library only;
- **navigation:** GeographicLib for mature WGS84 geodesics; and
- **advanced fixes:** NumPy and SciPy for numerical arrays, linear algebra,
  and nonlinear least squares.

All three libraries install with NautiPy so every feature works immediately.
Lazy module boundaries keep them out of coordinate-only imports. NautiPy does
not require pyproj, Shapely, pandas, or a general GIS framework.

## Scope boundaries

NautiPy is not intended to provide general CRS transformation, chart display, route planning, AIS, live GPS connections, magnetic models, tides, weather, plotting, a GUI, or a web service.

It is a calculation library, not certified navigation equipment.

## Development

The repository-local plan is designed for both human contributors and coding agents:

- [Coding-agent guide](AGENTS.md)
- [Product direction](docs/PRODUCT.md)
- [Architecture and dependency policy](docs/ARCHITECTURE.md)
- [Coordinate detection and conversion specification](docs/COORDINATES.md)
- [WGS84 navigation specification](docs/NAVIGATION.md)
- [Bearing and range fix specification](docs/FIXES.md)
- [GeoJSON interchange](docs/GEOJSON.md)
- [Implementation roadmap](ROADMAP.md)
- [Release and distribution plan](docs/RELEASING.md)
- [Support and API stability policy](docs/SUPPORT.md)
- [Contribution guide](CONTRIBUTING.md)

The clean rewrite provides an immutable `Position`, complete coordinate
conversion, WGS84 navigation, standard-library tests, package builds, and CI.
It also provides GeoJSON interchange and a coordinate conversion/inspection
CLI. The integrated fix engine adds weighted regional position estimates with
diagnostics through the top-level package, and the rewrite intentionally
provides no compatibility layer for the old code.

## Distribution plan

After the 0.1 feature set passes artifact tests:

1. an intentional semantic-version tag triggers GitHub Actions;
2. the workflow validates, builds, and tests the exact wheel and sdist;
3. those artifacts are published to PyPI through Trusted Publishing;
4. the same artifacts are attached to a GitHub Release; and
5. the stable PyPI sdist is submitted to conda-forge through staged-recipes.

Merging a pull request never publishes a package.
