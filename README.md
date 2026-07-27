# NautiPy

**Easy coordinate handling, trustworthy WGS84 navigation, and diagnosed
position fixes in one Python package.**

NautiPy accepts the coordinate formats people commonly paste or type, converts
them into a validated `Position`, and provides navigation and position-fixing
tools without silently guessing when an input is ambiguous.

## Install

Use a Python version accepted by the
[`requires-python` setting](https://github.com/cafawo/NautiPy/blob/master/pyproject.toml),
then install the complete package from PyPI:

```console
python -m pip install nautipy
```

The installation includes GeographicLib, NumPy, and SciPy, so every feature is
available immediately.

## Quick start

### Parse and convert coordinates

```python
from nautipy import convert_position, parse_position

p1 = parse_position("50.12257, 8.66570")
p2 = parse_position("50° 7.3542' N; 8° 39.942' E")
p3 = parse_position("+50.12257+008.66570/")

assert p1 == p2 == p3
assert convert_position(
    "50.12257, 8.66570",
    to="dms",
) == "50° 7′ 21.25″ N; 8° 39′ 56.52″ E"
```

Detection covers decimal degrees (DD), degrees and decimal minutes (DDM),
degrees/minutes/seconds (DMS), two-dimensional ISO 6709, and NMEA coordinate
field pairs. Use `order="lonlat"` for unmarked longitude-first input.
`order="auto"` accepts hard axis evidence or equivalent source orders;
otherwise it raises `AmbiguousCoordinateError`.

### Calculate WGS84 navigation values

```python
from nautipy import destination, distance, inverse

start = "50.12257, 8.66570"
end = destination(start, bearing=90, distance=12_000)

assert abs(distance(start, end) - 12_000) < 1e-6

result = inverse(start, end)
print(result.initial_bearing)
print(result.final_bearing)
```

Distances are in metres and bearings are true degrees clockwise from north.
Calculations use WGS84 ellipsoidal geodesics through GeographicLib.

### Estimate a position from observations

```python
from nautipy import Position, RangeObservation, solve_fix

references = (
    Position(50.116135, 8.670277),
    Position(50.112836, 8.666753),
    Position(50.110347, 8.659873),
)
ranges = tuple(
    RangeObservation(reference, measured, uncertainty=2.0)
    for reference, measured in zip(
        references,
        (1_275.251, 1_599.237, 1_917.145),
    )
)

result = solve_fix(ranges=ranges)
if result.success:
    print(result.position)
else:
    print(result.status, result.competing_positions)
```

Bearing, range, and mixed-observation fixes report residuals, convergence, and
geometry diagnostics. A successful optimization does not by itself guarantee
good observation geometry, so callers should inspect the complete result.

### Use the command line

```console
$ nautipy convert "50° 7.3542' N; 8° 39.942' E" --to dd
50.122570, 8.665700

$ nautipy inspect "+50.12257+008.66570/"
```

`inspect` writes deterministic JSON describing the detected format,
normalizations, resolution, and candidate interpretations. `python -m nautipy`
provides the same interface.

## What is included

- coordinate parsing, inspection, formatting, and conversion;
- an immutable, validated `Position` model;
- WGS84 distance, bearing, destination, interpolation, and nearest-position
  calculations;
- diagnosed bearing, range, and mixed-observation fixes;
- GeoJSON Point and FeatureCollection interchange; and
- offline `convert` and `inspect` commands.

NautiPy is deliberately not a general GIS framework, charting application,
route planner, live-data client, or certified navigation system.

## Documentation

For using NautiPy:

- [Coordinate input and conversion](https://github.com/cafawo/NautiPy/blob/master/docs/COORDINATES.md)
- [WGS84 navigation](https://github.com/cafawo/NautiPy/blob/master/docs/NAVIGATION.md)
- [Bearing and range position fixes](https://github.com/cafawo/NautiPy/blob/master/docs/FIXES.md)
- [GeoJSON interchange](https://github.com/cafawo/NautiPy/blob/master/docs/GEOJSON.md)
- [Support and API stability](https://github.com/cafawo/NautiPy/blob/master/docs/SUPPORT.md)

For contributing and maintaining the project:

- [Contribution guide](https://github.com/cafawo/NautiPy/blob/master/CONTRIBUTING.md)
- [Product direction](https://github.com/cafawo/NautiPy/blob/master/docs/PRODUCT.md)
- [Architecture and dependency policy](https://github.com/cafawo/NautiPy/blob/master/docs/ARCHITECTURE.md)
- [Implementation roadmap](https://github.com/cafawo/NautiPy/blob/master/ROADMAP.md)
- [Release procedure](https://github.com/cafawo/NautiPy/blob/master/docs/RELEASING.md)
- [Changelog](https://github.com/cafawo/NautiPy/blob/master/CHANGELOG.md)

Please report ordinary bugs through the
[issue tracker](https://github.com/cafawo/NautiPy/issues). Report suspected
security problems through the
[security policy](https://github.com/cafawo/NautiPy/security/policy).
Participation is governed by the
[Code of Conduct](https://github.com/cafawo/NautiPy/blob/master/CODE_OF_CONDUCT.md).

NautiPy is available under the
[MIT License](https://github.com/cafawo/NautiPy/blob/master/LICENSE.txt).
