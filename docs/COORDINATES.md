# Coordinate detection and conversion specification

## Purpose

Coordinate input is NautiPy's first product differentiator. Users should be able to paste common human-readable, machine-readable, and device-style coordinates without first identifying the notation.

The parser must be forgiving about presentation and strict about meaning:

- normalize harmless syntax variation;
- validate latitude and longitude rigorously;
- infer only what the input proves;
- never silently choose between materially different positions; and
- provide a simple conversion API built on the same parser.

This document defines observable behavior. Internal parser structure may change as long as these guarantees remain true.

## Proposed public API

The common path returns an ordinary validated `Position`:

```python
from nautipy import parse_position

position = parse_position("50° 7.3542' N; 8° 39.942' E")
```

The intended public surface is:

```python
parse_coordinate(value, *, axis=None, format=None) -> float
parse_position(value, *, order="latlon", format=None) -> Position
inspect_position(value, *, order="latlon", format=None) -> ParseResult
format_coordinate(value, *, axis, to="dd", precision=None, **options) -> str
format_position(position, *, to="dd", order="latlon", precision=None, **options) -> str
convert_position(value, *, to="dd", order="latlon", precision=None, **options) -> str
```

Names may be refined during implementation, but the package must retain both:

1. a low-friction function returning `Position`; and
2. an inspection path returning detection metadata, normalized tokens, warnings, and rejected alternatives.

Passing `format=` means "interpret this input as this format" and bypasses automatic format selection after common normalization. It must still validate the data.

## Canonical internal representation

A parsed position is represented as:

- latitude: finite Python `float` in `[-90, 90]` decimal degrees;
- longitude: finite Python `float` in `[-180, 180]` decimal degrees;
- WGS84 geographic coordinates;
- no display formatting or unit suffix stored in the value itself.

The `Position` model may store user metadata such as a description or identifier, but formatting and parser diagnostics do not participate in position equality.

Do not silently wrap out-of-range user input. `181° E` is an error, not `179° W`. Geodesic calculations may normalize generated destinations separately.

## Accepted input families

### Decimal degrees (`dd`)

Accept numeric values and strings with optional sign or hemisphere:

```text
50.12257
+50.12257
-8.66570
50.12257 N
N 50.12257
8.66570° E
```

A hemisphere may appear before or after the number. A sign and hemisphere may coexist only when they agree. `-50 S` is positive latitude only if the API explicitly documents double-negation; the preferred and safer behavior is to reject mixed sign-plus-hemisphere input unless the sign is non-negative. `-50 N` and `+50 S` are always contradictory.

### Degrees and decimal minutes (`ddm`)

Use `ddm` as the canonical name. Accept `dmm` as a compatibility alias.

```text
50° 7.3542' N
N 50 7.3542
50 deg 7.3542 min N
8:39.942 E
```

A direction marker or explicit `axis=` is required when a standalone unsigned value could be either latitude or longitude.

### Degrees, minutes, and seconds (`dms`)

```text
50° 7' 21.252" N
N 50 7 21.252
50 deg 7 min 21.252 sec N
8:39:56.52 E
```

Accept common ASCII and Unicode degree, prime, and double-prime forms after normalization.

### ISO 6709

Support separated and compact forms that can be interpreted without guessing, including a trailing slash where used:

```text
+50.12257+008.66570/
+50.12257 -008.66570
+50.12257,+008.66570
```

Compact ISO 6709 parsing must determine component boundaries from valid latitude/longitude widths and syntax. Do not use loose splitting that can reinterpret malformed data.

Altitude or CRS suffixes are outside the first implementation unless they can be safely ignored with an explicit warning. Never treat altitude as longitude.

### NMEA coordinate fields

Support latitude/longitude coordinate fields and direction fields without attempting to become a full NMEA sentence decoder:

```text
5007.3542,N,00839.9420,E
5007.3542 N; 00839.9420 E
```

NMEA latitude uses `ddmm.mmmm`; longitude uses `dddmm.mmmm`. Direction fields are required for automatic NMEA detection. Full `$GPGGA`, `$GPRMC`, or other sentence parsing remains out of scope for the core coordinate parser.

### Structured Python values

Accept:

```python
parse_position((50.12257, 8.66570))
parse_position([50.12257, 8.66570])
parse_position({"lat": 50.12257, "lon": 8.66570})
parse_position({"latitude": "50° 7.3542' N", "longitude": "8° 39.942' E"})
```

Recognize GeoJSON Point objects as longitude/latitude by the GeoJSON specification:

```python
parse_position({
    "type": "Point",
    "coordinates": [8.66570, 50.12257],
})
```

Do not apply the GeoJSON order to an ordinary two-element list unless `order="lonlat"` is provided.

## Text normalization

Before format-specific parsing, normalize presentation without changing meaning:

- trim outer and repeated internal whitespace where separators allow it;
- normalize Unicode minus signs to ASCII `-`;
- recognize common degree symbols such as `°` and `º`;
- recognize straight and typographic minute/second marks;
- compare hemisphere letters case-insensitively;
- accept full words such as `north`, `south`, `east`, and `west` when unambiguous;
- normalize `deg`, `degree`, `degrees`, `min`, `minute`, `sec`, and equivalent casing;
- preserve the original input in diagnostics.

Normalization must not remove punctuation before deciding whether it is a decimal separator, component separator, or pair separator.

## Decimal-comma behavior

Decimal commas are common and should be supported when syntax makes them unambiguous:

```text
50,12257 N
50,12257 N; 8,66570 E
50° 7,3542' N; 8° 39,942' E
```

A semicolon, slash, hemisphere markers, named fields, or structured input can disambiguate the coordinate pair.

The following is ambiguous and must fail with guidance:

```text
50,12257, 8,66570
```

The error should suggest `50,12257; 8,66570`, explicit `order=`, or dot decimals.

## Axis and coordinate-order rules

### Single coordinates

A standalone unsigned coordinate requires one of:

- `axis="lat"` or `axis="lon"`;
- a latitude/longitude hemisphere marker; or
- a format whose structure includes the axis.

Signed decimal degrees can be parsed without an axis, but range validation is limited to the selected or inferred axis. Prefer requiring `axis` for values between `-90` and `90` when validation matters.

### Position pairs

`parse_position` defaults to `order="latlon"`. This is a documented contract, not an inference.

Supported order values:

- `latlon`: first value is latitude, second is longitude;
- `lonlat`: first value is longitude, second is latitude;
- `auto`: use only hard evidence, otherwise raise `AmbiguousCoordinateError`.

Hard evidence includes:

- hemisphere or named-field axis markers;
- a recognized GeoJSON object;
- NMEA field widths plus directions;
- one numeric component outside the latitude range but inside the longitude range.

When both values are within `[-90, 90]` and no axis marker exists, `order="auto"` is ambiguous. Do not use geography, likely inhabited regions, sign patterns, or "most common" assumptions to choose.

Hemisphere markers override textual order. A pair such as `8 E, 50 N` may be returned correctly even under `order="auto"`, and inspection metadata should note the source order.

## Detection strategy

Implement detection as a staged candidate parser, not one monolithic regular expression.

Recommended flow:

1. Classify Python structure versus text.
2. Normalize Unicode and vocabulary while preserving separator information.
3. Extract explicit axis markers and named fields.
4. Generate plausible format candidates based on syntax.
5. Parse each candidate independently.
6. Validate components and ranges.
7. If exactly one interpretation remains, return it.
8. If multiple interpretations produce the same normalized position, return it and record equivalent candidates.
9. If multiple interpretations produce different positions, raise `AmbiguousCoordinateError` with candidates.
10. If none remain, raise `CoordinateParseError` containing the failing token or rule where possible.

Candidate selection may rank strong syntax evidence, but it must not choose a lower-confidence interpretation merely to avoid an error.

The inspection result should expose enough information to answer:

- Which format was selected?
- Which axis/order evidence was used?
- What normalization occurred?
- Was precision inferred?
- Were equivalent candidates found?
- Why were alternatives rejected?

## Component validation

For DDM and DMS:

- degrees must be integral unless the selected format explicitly permits decimal degrees;
- minutes must satisfy `0 <= minutes < 60`;
- seconds must satisfy `0 <= seconds < 60`;
- latitude degrees must satisfy `0 <= degrees <= 90`;
- longitude degrees must satisfy `0 <= degrees <= 180`;
- latitude at exactly 90 degrees requires zero minutes and seconds;
- longitude at exactly 180 degrees requires zero minutes and seconds.

For all formats:

- reject NaN and infinity;
- reject missing components;
- reject repeated or conflicting hemisphere markers;
- reject sign/hemisphere conflicts;
- reject unexpected trailing tokens rather than ignoring them;
- reject an extra numeric field that could be altitude unless altitude support was explicitly requested.

Errors must be exceptions with useful messages, not assertions.

## Formatting and conversion

Canonical output names:

- `dd`: decimal degrees;
- `ddm`: degrees and decimal minutes;
- `dms`: degrees, minutes, and seconds;
- `iso6709`: ISO 6709 coordinate pair;
- `nmea`: NMEA coordinate fields plus directions.

Accept `dmm` as an alias for `ddm`, but emit canonical names in metadata and documentation.

Formatting options should cover:

- signed versus hemisphere notation where applicable;
- latitude/longitude output order;
- ASCII versus Unicode symbols;
- explicit precision;
- compact versus separated ISO 6709;
- pair separator.

Defaults should be readable and round-trip safely. Do not emit negative zero. Carry rounding into minutes/degrees correctly: `59.999...` seconds must not become an invalid `60.000` output.

### Precision policy

Do not imply more source accuracy than is known.

- `inspect_position` should report lexical precision or estimated angular resolution when it can be inferred from text.
- `convert_position` should preserve at least the source resolution by default where practical.
- `format_position(Position(...))`, where source precision is unavailable, should use a documented sensible default and accept explicit `precision=`.
- Internal calculations remain full precision; display rounding happens only during formatting.

## Errors and warnings

Provide a small exception hierarchy:

```text
NautiPyError
└── CoordinateError
    ├── CoordinateParseError
    ├── CoordinateRangeError
    └── AmbiguousCoordinateError
```

An ambiguity error should include candidate interpretations and a concrete resolution, for example:

```text
Could not determine coordinate order for "8.66570, 50.12257".
Both lat/lon and lon/lat are valid. Pass order="latlon" or order="lonlat".
```

Warnings belong in the inspection result for recoverable normalization, such as a deprecated alias. Routine whitespace or Unicode normalization should not emit Python warnings.

## Required examples for acceptance tests

The test suite must include equivalent representations of a shared reference position:

```text
50.12257, 8.66570
+50.12257 +008.66570
50.12257 N; 8.66570 E
50° 7.3542' N; 8° 39.942' E
50° 7' 21.252" N; 8° 39' 56.52" E
+50.12257+008.66570/
5007.3542,N,00839.9420,E
50,12257 N; 8,66570 E
```

Also test:

- hemisphere prefix and suffix;
- lowercase and full-word directions;
- ASCII and Unicode symbols;
- arbitrary harmless whitespace;
- latitude/longitude at zero and legal extrema;
- minute/second carry during formatting;
- GeoJSON order;
- explicit `latlon`, `lonlat`, and `auto` behavior;
- contradictory sign and hemisphere;
- out-of-range degrees, minutes, and seconds;
- NaN and infinity;
- missing and extra fields;
- decimal-comma ambiguity;
- numeric pairs where both orders are plausible;
- numeric pairs where range proves the order;
- parse-format-parse round trips for every output format.

Generated round-trip tests may use standard-library loops and deterministic random seeds. A property-testing dependency is optional, not required.

## Performance and dependency constraints

Coordinate parsing and formatting must:

- use only the Python standard library;
- perform no network or filesystem access;
- import without loading NumPy, SciPy, or geodesic solver modules;
- handle ordinary single inputs without array conversion; and
- remain deterministic across supported platforms.

## Non-goals for the coordinate module

- arbitrary CRS or datum conversion;
- UTM, MGRS, geohash, plus codes, or proprietary grids in the first modern release;
- altitude, depth, speed, time, or full sensor-message parsing;
- repairing arbitrary corrupted text;
- inferring a real-world location to resolve coordinate order;
- locale detection from operating-system settings.

Additional coordinate systems should be added only when there is a concrete navigation use case, a stable specification, and a clear ambiguity policy.
