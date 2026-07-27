# GeoJSON interchange

## Overview

NautiPy exchanges two-dimensional WGS84 points through ordinary Python
mappings. It uses the standard library and performs no file or network access.
Use `json.load`, `json.loads`, `json.dump`, or `json.dumps` when JSON text or a
file is needed.

GeoJSON helpers are public from `nautipy.geojson`:

```text
to_geojson_point(value, *, order="latlon", format=None) -> dict
from_geojson_point(value) -> Position
to_geojson_feature_collection(
    values,
    *,
    order="latlon",
    format=None,
) -> dict
from_geojson_feature_collection(value) -> tuple[Position, ...]
```

Export functions accept the same position-like values as `parse_position`.
`order` and `format` control those inputs; they never change GeoJSON's required
longitude/latitude coordinate order.

## Points

```python
from nautipy import Position
from nautipy.geojson import from_geojson_point, to_geojson_point

position = Position(50.12257, 8.66570)
point = to_geojson_point(position)

assert point == {
    "type": "Point",
    "coordinates": [8.6657, 50.12257],
}
assert from_geojson_point(point) == position
```

Point coordinates contain exactly two numeric values. Altitude and every
non-Point geometry are rejected.

A bare Point has no standard Feature metadata location. Exporting a `Position`
with an identifier or description as a bare Point raises
`CoordinateParseError`; use a FeatureCollection to preserve metadata.

## FeatureCollections

A `Position` can carry an optional GeoJSON-compatible identifier and
description. These fields do not affect position equality or hashing.

```python
from nautipy import Position
from nautipy.geojson import (
    from_geojson_feature_collection,
    to_geojson_feature_collection,
)

positions = [
    Position(
        50.12257,
        8.66570,
        identifier="station-1",
        description="Reference station",
    ),
]

collection = to_geojson_feature_collection(positions)
restored = from_geojson_feature_collection(collection)

assert restored == tuple(positions)
assert restored[0].identifier == "station-1"
assert restored[0].description == "Reference station"
```

Identifiers use the top-level Feature `id` member. Descriptions use
`properties.description`. String and finite numeric identifiers are accepted;
a missing identifier becomes `None`. A non-null description is a string.

Every imported Feature contains:

- a non-null, two-dimensional Point `geometry`; and
- a `properties` member that is a mapping or `null`.

`properties.description` may be a string or `null`. A missing or null
description becomes `None`. Collection order and duplicate positions are
preserved. Empty FeatureCollections are accepted.

The importer rejects line, polygon, multi-point, null, and mixed geometries
instead of skipping them. Legacy `crs` members and members reserved for a
different GeoJSON object type are rejected. Ordinary foreign members and
unknown Feature properties are accepted but not preserved.

Prefer a string identifier when an integer may exceed a receiving JSON
system's numeric limits.

## Errors and scope

Invalid structure and unsupported geometry raise `CoordinateParseError`.
Non-finite or out-of-range coordinates raise `CoordinateRangeError`.
Ambiguous position-like values supplied to exporters retain
`AmbiguousCoordinateError`.

NautiPy intentionally supports Point and Point FeatureCollection interchange,
not general GeoJSON geometry processing.
