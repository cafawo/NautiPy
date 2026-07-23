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

Implementation status: the current development slice covers the full
Milestone 1 coordinate workflow: DD, DDM, DMS, two-dimensional ISO 6709 and
NMEA position pairs, structured inputs, inspection metadata, canonical output,
and resolution-aware conversion.

## Public API

The common path returns an ordinary validated `Position`:

```python
from nautipy import parse_position

position = parse_position("50° 7.3542' N; 8° 39.942' E")
```

The public surface is:

```python
parse_position(value, *, order="latlon", format=None) -> Position
inspect_position(value, *, order="latlon", format=None) -> ParseResult
format_position(position, *, to="dd", order="latlon", precision=None, **options) -> str
convert_position(
    value,
    *,
    to="dd",
    order="latlon",
    output_order="latlon",
    format=None,
    precision=None,
    **options,
) -> str
```

The package retains both:

1. a low-friction function returning `Position`; and
2. an inspection path returning detection metadata, normalized tokens, warnings, and rejected alternatives.

`PositionInput` is the public typing alias for accepted position-like values.
`CandidateDiagnostic` is the immutable public record used for each entry in
`ParseResult.candidates`.

Each `CandidateDiagnostic` contains the candidate `format`, detected
`source_order`, parsed `position` when one exists, selection `outcome`,
supporting `evidence`, and a rejection or competition `reason` when relevant.
The collections are immutable tuples, and rejected candidates may have no
position.

The `nautipy.coordinates` module also exports the typing aliases used by these
signatures and records:

- `CoordinateOrder`: `"latlon"`, `"lonlat"`, or evidence-only `"auto"`;
- `OutputOrder`: `"latlon"` or `"lonlat"`;
- `CoordinateFormat`: `"dd"`, `"ddm"`, `"dms"`, `"iso6709"`, or `"nmea"`;
- `DetectedFormat`: a `CoordinateFormat` or `"mixed"`; and
- `CandidateOutcome`: `"selected"`, `"equivalent"`, `"rejected"`, or
  `"competing"`.

Passing `format=` means "interpret this input as this format" and bypasses automatic format selection after common normalization. It must still validate the data.

`ParseResult` contains the normalized `position`, canonical detected `format`,
axis-aligned `component_formats`, source-order evidence, the original text and
normalized tokens for textual input, normalization labels, warnings, angular
resolution per axis where it can be inferred, and selected, equivalent, or
rejected candidate diagnostics. `source_order` is `None` when equal values make
both source orders equivalent. Structured numeric input has no lexical tokens
or inferred source resolution.

## Canonical internal representation

A parsed position is represented as:

- latitude: finite Python `float` in `[-90, 90]` decimal degrees;
- longitude: finite Python `float` in `[-180, 180]` decimal degrees;
- WGS84 geographic coordinates;
- no display formatting or unit suffix stored in the value itself.

The `Position` model stores optional keyword-only `identifier` and
`description` metadata. Identifiers are strings or finite JSON-style numbers;
descriptions are strings. Metadata, formatting, and parser diagnostics do not
participate in position equality or hashing.

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

A hemisphere may appear before or after the number. A negative sign combined
with any hemisphere is rejected rather than interpreted as double negation. An
explicit `+` may accompany `N` or `E`, but contradicts `S` or `W`; use an
unsigned magnitude with `S` or `W`.

### Degrees and decimal minutes (`ddm`)

Use `ddm` as the canonical name. Accept `dmm` as a common input alias.

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

NautiPy's supported two-dimensional subset requires signs and zero-padded
latitude/longitude fields. It accepts decimal-degree widths (`DD`/`DDD`), DDM
widths (`DDMM`/`DDDMM`), and DMS widths (`DDMMSS`/`DDDMMSS`), with a fraction
on the least-significant unit. Both components must use the same form. A space
or comma may separate them; compact input needs no separator.

Altitude or CRS suffixes are outside the first implementation unless they can be safely ignored with an explicit warning. Never treat altitude as longitude.

### NMEA coordinate fields

Support latitude/longitude coordinate fields and direction fields without attempting to become a full NMEA sentence decoder:

```text
5007.3542,N,00839.9420,E
5007.3542 N; 00839.9420 E
```

NMEA latitude uses `ddmm.mmmm`; longitude uses `dddmm.mmmm`. Direction fields are required for automatic NMEA detection. Full `$GPGGA`, `$GPRMC`, or other sentence parsing remains out of scope for the core coordinate parser.

Fields require a decimal point and at least one fractional-minute digit. The
four- and five-digit widths before that point are exact. NautiPy accepts the
documented four-field comma form and two-component semicolon form only. As with
other axis-marked input, directions may prove the axes when source order is
reversed.

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

Named mappings require exactly one latitude key (`lat` or `latitude`) and one
longitude key (`lon` or `longitude`); extra fields are rejected. GeoJSON Point
coordinates must contain exactly two numeric values. Three-dimensional
positions are rejected while altitude support remains out of scope.

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

The error should suggest `50,12257; 8,66570` or dot decimals. Coordinate order
cannot resolve punctuation ambiguity.

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

Formatting options cover:

- signed versus hemisphere notation where applicable;
- latitude/longitude output order;
- ASCII versus Unicode symbols;
- explicit precision;
- compact versus separated ISO 6709;
- pair separator.

Defaults should be readable and round-trip safely. Do not emit negative zero. Carry rounding into minutes/degrees correctly: `59.999...` seconds must not become an invalid `60.000` output.

The canonical defaults for `Position(50.12257, 8.66570)` are:

| Format | Precision | Output |
| --- | ---: | --- |
| DD | 6 degree decimals | `50.122570, 8.665700` |
| DDM | 4 minute decimals | `50° 7.3542′ N; 8° 39.9420′ E` |
| DMS | 2 second decimals | `50° 7′ 21.25″ N; 8° 39′ 56.52″ E` |
| ISO 6709 | 6 degree decimals | `+50.122570+008.665700/` |
| NMEA | 4 minute decimals | `5007.3542,N,00839.9420,E` |

Explicit precision is limited to 0 through 15 decimal places in the
least-significant displayed unit. NMEA requires at least one fractional-minute
digit, so its minimum is 1. Formatting uses round-half-even, performs carry
before rendering, and retains trailing zeros. Values that round to zero use
unsigned zero, `N`/`E`, or an ISO `+` sign.

Precision is a display choice, not extra storage precision. At deliberately
over-fine settings near binary64 spacing, parsing can select an adjacent
representable float and a later formatting pass can change the final digit.
The documented defaults avoid that regime and round-trip canonically.

DD defaults to signed notation. DDM and DMS default to hemispheres and Unicode
symbols. `notation="signed"` or `"hemisphere"` and `symbols="unicode"` or
`"ascii"` apply to those human-readable formats. Human separators are limited
to `", "`, `"; "`, and `" / "`, so every emitted form remains parseable.

ISO output is the signed, zero-padded decimal-degree form with a terminal
slash. It is necessarily latitude/longitude; requesting longitude/latitude is
an error. `compact=False` separates fields with a space by default, and a comma
separator is also available. NMEA supports its four-field comma form and the
two-component `"; "` form. Human and NMEA output support either coordinate
order.

### Precision policy

Do not imply more source accuracy than is known.

- `inspect_position` should report lexical precision or estimated angular resolution when it can be inferred from text.
- `convert_position` should preserve at least the source resolution by default where practical.
- `format_position(Position(...))`, where source precision is unavailable, should use a documented sensible default and accept explicit `precision=`.
- Internal calculations remain full precision; display rounding happens only during formatting.

Inspection reports resolution as a positive, exact standard-library
`Decimal` or `Fraction` angular quantum in decimal degrees, not as measurement
accuracy or uncertainty. DD resolution is
exponent-aware; DDM and NMEA resolution is the fractional-minute quantum
divided by 60, and DMS resolution is the fractional-second quantum divided by
3600. `convert_position(..., precision=None)` chooses the smallest target
precision no coarser than the finest known axis resolution. It uses the table
above when the source has no lexical resolution. If preserving the source
would require more than the supported 15 display places, conversion raises
unless the caller explicitly chooses a precision and accepts rounding.
`order` controls input interpretation and `output_order` independently controls
conversion output.

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
