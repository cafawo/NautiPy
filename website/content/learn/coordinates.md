# Coordinates on Earth

A geographic position answers two questions: how far north or south of the
equator, and how far east or west around Earth? The answers are
[latitude](https://en.wikipedia.org/wiki/Latitude) and
[longitude](https://en.wikipedia.org/wiki/Longitude).

NautiPy stores both as decimal degrees:

```python
from nautipy import Position

position = Position(latitude=50.12257, longitude=8.66570)
```

Latitude must be between −90° and 90°. Longitude must be between −180° and
180°. NautiPy rejects values outside those ranges instead of wrapping them.

## One place, several notations

Degrees may be subdivided in two familiar sexagesimal ways. One degree contains
60 minutes, and one minute contains 60 seconds.

<div class="educational-figure" tabindex="0" role="group" aria-label="Scrollable diagram" markdown>

![The same latitude and longitude represented as decimal degrees, degrees and
decimal minutes, degrees-minutes-seconds, ISO 6709, and NMEA coordinate
fields.](../assets/images/coordinate-notation.svg)

</div>

| Name | Example | What is fractional? |
| --- | --- | --- |
| Decimal degrees (DD) | `50.122570, 8.665700` | degrees |
| Degrees and decimal minutes (DDM) | `50° 7.3542′ N; 8° 39.9420′ E` | minutes |
| Degrees, minutes, seconds (DMS) | `50° 7′ 21.25″ N; 8° 39′ 56.52″ E` | seconds |
| ISO 6709 subset | `+50.122570+008.665700/` | least-significant unit |
| NMEA coordinate fields | `5007.3542,N,00839.9420,E` | minutes |

DD, DDM, and DMS describe the same angles. ISO 6709 defines representations
for exchanging point coordinates; NautiPy deliberately supports an
unambiguous, two-dimensional signed subset. NMEA fields are the coordinate and
direction fields used in marine electronics, not complete NMEA sentences.

You do not need to identify the notation first:

```python
from nautipy import format_position, parse_position

position = parse_position("50° 7.3542' N; 8° 39.942' E")

assert format_position(position, to="dd") == "50.122570, 8.665700"
assert (
    format_position(position, to="dms")
    == "50° 7′ 21.25″ N; 8° 39′ 56.52″ E"
)
```

The parser accepts harmless differences such as common prime symbols,
direction words, whitespace, and unambiguous decimal commas. It still validates
every component: minutes and seconds must be below 60, and 90° latitude or
180° longitude cannot have a non-zero remainder.

## Coordinate order changes the place

The pair `(50, 8)` and the pair `(8, 50)` are both valid, but they are far
apart. A parser cannot safely infer which one a person intended.

<div class="educational-figure" tabindex="0" role="group" aria-label="Scrollable diagram" markdown>

![Two valid latitude/longitude orderings of the same unmarked numbers,
showing why swapping axes changes the location.](../assets/images/coordinate-order.svg)

</div>

NautiPy follows explicit evidence:

- The default `order="latlon"` treats the first unmarked component as
  latitude.
- `order="lonlat"` treats the first unmarked component as longitude.
- `order="auto"` requires hard evidence such as hemisphere markers, named
  fields, NMEA widths, GeoJSON structure, or a value that cannot be latitude.
- If both unmarked values fit either axis, `order="auto"` raises
  `AmbiguousCoordinateError` rather than using geography or likelihood to
  guess.

```python
from nautipy import parse_position

latlon = parse_position((50.12257, 8.66570))
lonlat = parse_position((8.66570, 50.12257), order="lonlat")

assert latlon == lonlat
```

Named data is clearer still:

```python
position = parse_position({
    "latitude": "50° 7.3542' N",
    "longitude": "8° 39.942' E",
})
```

### The GeoJSON exception

[GeoJSON](https://en.wikipedia.org/wiki/GeoJSON) specifies coordinates in
**longitude, latitude** order. NautiPy honors that rule for a GeoJSON Point:

```python
point = {
    "type": "Point",
    "coordinates": [8.66570, 50.12257],
}
position = parse_position(point)
```

That order belongs to GeoJSON structure. It does not silently carry over to an
ordinary Python list or tuple.

## Inspect before converting

`inspect_position` explains how an input was understood. This is useful when
teaching, importing unfamiliar data, or diagnosing a rejected value.

```python
from nautipy import inspect_position

inspection = inspect_position("5007.3542,N,00839.9420,E")

print(inspection.format)               # nmea
print(inspection.position)
print(inspection.source_order)
print(inspection.normalizations)
print(inspection.latitude_resolution)
```

Inspection records the detected component formats, order evidence,
normalizations, source text, inferred angular resolution, and candidate
interpretations. If materially different interpretations remain, it raises an
ambiguity error whose `candidates` describe the competition.

## Precision is not accuracy

Writing more digits makes a *representation* more precise, but it does not make
the measurement more accurate. `50.122570` has a finer displayed increment
than `50.12`; that says nothing by itself about the sensor, survey, datum, or
age of the source.

NautiPy can infer the lexical angular resolution of many text inputs. It uses
that resolution to avoid needlessly coarse conversion when `precision=None`.
It does not turn source resolution into measurement uncertainty.

This distinction also appears in [RFC 7946’s discussion of coordinate
precision](https://www.rfc-editor.org/rfc/rfc7946.html#section-11.2).

## Boundaries and limitations

- Positions are two-dimensional. Altitude, depth, and extra numeric fields are
  rejected.
- The coordinate layer does not transform datums or arbitrary coordinate
  reference systems.
- NautiPy accepts NMEA coordinate fields, not `$GPGGA`, `$GPRMC`, or other
  complete messages.
- It does not repair corrupted input, infer operating-system locale, or guess
  a location from plausibility.
- A valid coordinate is not necessarily an accurate or current observation.

For exact accepted forms and failure behavior, see the
[coordinate behavior specification](https://github.com/cafawo/NautiPy/blob/master/docs/COORDINATES.md)
and the
[GeoJSON behavior specification](https://github.com/cafawo/NautiPy/blob/master/docs/GEOJSON.md).

## Learn more

- [Geographic coordinate system](https://en.wikipedia.org/wiki/Geographic_coordinate_system)
  gives a friendly overview of latitude and longitude.
- [Sexagesimal notation](https://en.wikipedia.org/wiki/Sexagesimal) explains
  the base-60 division behind minutes and seconds.
- [ISO 6709:2022](https://www.iso.org/standard/75147.html) is the primary
  coordinate-representation standard.
- [RFC 7946](https://www.rfc-editor.org/info/rfc7946) defines GeoJSON,
  including its coordinate order.
- The [NMEA 0183 standard page](https://www.nmea.org/nmea-0183.html) describes
  the marine data standard from which NMEA coordinate fields come.

Next: [Navigation on an Ellipsoid](navigation.md).
