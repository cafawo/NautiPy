# GeoJSON interchange

NautiPy exchanges two-dimensional WGS84 points through ordinary Python
mappings. The implementation uses only the standard library and performs no
file or network access. Use `json.load` and `json.dump` when a file or JSON
text is needed.

## Points

`to_geojson_point` exports one position as a GeoJSON Point. GeoJSON always
places longitude before latitude, independently of NautiPy's normal input
`order` option:

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

Point coordinates must contain exactly two numbers. Altitude and every
non-Point geometry are outside NautiPy's coordinate-to-position scope.

## Feature collections

A `Position` can carry an optional GeoJSON-compatible identifier and a
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

Identifiers use the standard top-level Feature `id` member. Descriptions use
`properties.description`. String and finite numeric identifiers are accepted;
descriptions must be strings. Collection order and duplicate positions are
preserved. Prefer a string identifier when its integer size may exceed a
receiving JSON system's numeric limits.

The collection importer requires every feature to contain a non-null,
two-dimensional Point geometry. It rejects line, polygon, multi-point, null,
and mixed geometries instead of skipping them. Unknown Feature properties and
ordinary GeoJSON foreign members are accepted but are not preserved; legacy
`crs` members and members reserved for another GeoJSON object type are
rejected.

A bare Point has no standard Feature metadata location. Exporting a Position
with an identifier or description as a bare Point therefore raises an error;
use a FeatureCollection to preserve the metadata.
