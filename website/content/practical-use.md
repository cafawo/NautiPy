# Practical Use

These recipes use the supported public API. Distances are metres, positions
store decimal degrees, and generated bearings are true degrees clockwise from
north.

## Install

```console
python -m pip install nautipy
```

The one installation includes every shipped feature and its GeographicLib,
NumPy, and SciPy dependencies.

## Parse coordinate text safely

```python
from nautipy import parse_position

position = parse_position("N 50° 7' 21.252\"; E 8° 39' 56.52\"")
print(position.latitude, position.longitude)
```

For an unmarked longitude-first pair, say so:

```python
position = parse_position((8.66570, 50.12257), order="lonlat")
```

Use `order="auto"` only when the input contains hard axis evidence. It refuses
to guess when both orders are valid places.

## Inspect unfamiliar input

```python
from nautipy import inspect_position

result = inspect_position("5007.3542,N,00839.9420,E")

print(result.position)
print(result.format)
print(result.source_order)
print(result.evidence)
print(result.normalizations)
print(result.latitude_resolution)
```

The inspection result records how the position was selected. Ambiguity remains
an exception because there is no safe selected position to return.

## Convert without losing the intended order

```python
from nautipy import convert_position

converted = convert_position(
    "50.12257, 8.66570",
    to="dms",
)

assert converted == "50° 7′ 21.25″ N; 8° 39′ 56.52″ E"
```

For a machine-oriented representation:

```python
iso_text = convert_position(
    "50° 7.3542' N; 8° 39.942' E",
    to="iso6709",
)
print(iso_text)
```

ISO 6709 output is always latitude/longitude. `output_order` independently
controls human-format output.

## Calculate a journey

```python
from nautipy import destination, inverse, interpolate

start = "50.12257, 8.66570"
end = destination(start, bearing=90, distance=12_000)
route = inverse(start, end)
midpoint = interpolate(start, end)

print(route.distance)
print(route.initial_bearing)
print(route.final_bearing)
print(midpoint)
```

Every location argument accepts the same unambiguous position-like forms as
`parse_position`. Parse specialized order explicitly before handing a value to
navigation functions.

## Exchange GeoJSON points

GeoJSON helpers live in `nautipy.geojson`. They work with ordinary Python
mappings; use the standard-library `json` module for text or files.

```python
import json

from nautipy import Position
from nautipy.geojson import from_geojson_point, to_geojson_point

position = Position(50.12257, 8.66570)
point = to_geojson_point(position)

assert point == {
    "type": "Point",
    "coordinates": [8.6657, 50.12257],
}

text = json.dumps(point)
restored = from_geojson_point(json.loads(text))
assert restored == position
```

GeoJSON’s coordinate order is longitude, latitude. NautiPy supports
two-dimensional Points and Point FeatureCollections, not arbitrary geometry.

Use a FeatureCollection to preserve identifiers and descriptions:

```python
from nautipy.geojson import (
    from_geojson_feature_collection,
    to_geojson_feature_collection,
)

stations = [
    Position(
        50.116135,
        8.670277,
        identifier="station-1",
        description="Reference station",
    ),
]

collection = to_geojson_feature_collection(stations)
restored = from_geojson_feature_collection(collection)
assert restored == tuple(stations)
```

## Estimate a position from ranges

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
    print(result.residuals)
    print(result.rank, result.condition_number)
    print(result.uncertainty)
else:
    print(result.status, result.message)
    print(result.competing_positions)
```

Never discard status and diagnostics just to obtain a coordinate. See
[Can You Trust the Fix?](learn/trusting-a-fix.md) for a reading checklist.

## Convert or inspect at the command line

```console
nautipy convert "50° 7.3542' N; 8° 39.942' E" --to dd
```

Output:

```text
50.122570, 8.665700
```

Request deterministic JSON diagnostics:

```console
nautipy inspect "+50.12257+008.66570/"
```

The module entry point is equivalent:

```console
python -m nautipy convert "50.12257, 8.66570" --to nmea
```

Run `nautipy convert --help` or `nautipy inspect --help` for choices and
defaults. Invalid input exits with status 2 and a concise message rather than a
traceback.

## Handle caller errors

```python
from nautipy import (
    AmbiguousCoordinateError,
    CoordinateError,
    FixError,
    NavigationError,
    parse_position,
)

try:
    parse_position("50, 8", order="auto")
except AmbiguousCoordinateError as error:
    print(error)
    print(error.candidates)
except CoordinateError as error:
    print("Other coordinate problem:", error)
```

`NavigationError` covers invalid navigation scalars and undefined navigation
results. `FixError` covers invalid observations and solver configuration.
Invalid reference coordinates retain their applicable `CoordinateError`
subtype.

## Before relying on a result

- Confirm latitude/longitude order and datum.
- Confirm distances are metres and bearings are true, not magnetic.
- Keep source precision separate from measurement accuracy.
- For a fix, inspect status, warnings, residuals, rank, condition number,
  competing positions, search domain, and uncertainty.
- Account separately for altitude, motion, current, refraction, timing,
  correlated errors, and common bias where they matter.

> NautiPy is not certified navigation equipment. Use independent safeguards
> appropriate to the consequences of an error.

Exact behavior is defined by the
[coordinate](https://github.com/cafawo/NautiPy/blob/master/docs/COORDINATES.md),
[navigation](https://github.com/cafawo/NautiPy/blob/master/docs/NAVIGATION.md),
[GeoJSON](https://github.com/cafawo/NautiPy/blob/master/docs/GEOJSON.md), and
[position-fix](https://github.com/cafawo/NautiPy/blob/master/docs/FIXES.md)
specifications.

