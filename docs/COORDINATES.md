# Coordinate input and conversion

## Purpose

NautiPy accepts common human-readable, machine-readable, and device-style
position pairs without requiring callers to identify the notation first. It is
forgiving about presentation and strict about meaning:

- harmless syntax differences are normalized;
- latitude, longitude, and subcomponents are validated;
- order is inferred only from hard evidence; and
- materially different interpretations raise an actionable ambiguity error.

This document defines observable coordinate behavior. Parser internals may
change without changing these guarantees.

## Public API

```text
parse_position(value, *, order="latlon", format=None) -> Position
inspect_position(value, *, order="latlon", format=None) -> ParseResult
inspect_positions(
    values,
    *,
    order="latlon",
    format=None,
    errors="collect",
) -> BatchInspectionResult
format_position(
    position,
    *,
    to="dd",
    order="latlon",
    precision=None,
    notation=None,
    symbols=None,
    compact=None,
    separator=None,
) -> str
convert_position(
    value,
    *,
    to="dd",
    order="latlon",
    output_order="latlon",
    format=None,
    precision=None,
    notation=None,
    symbols=None,
    compact=None,
    separator=None,
) -> str
```

`parse_position` is the low-friction path:

```python
from nautipy import parse_position

position = parse_position("50° 7.3542' N; 8° 39.942' E")
```

`inspect_positions` applies the scalar inspection contract to each item from
an ordinary iterable while preserving successful diagnostics and coordinate
failures.
`format_position` accepts an already validated `Position`.
`convert_position` combines parsing and formatting for other position-like
inputs.

With `format=None`, NautiPy detects the input format. Passing `format="dd"`,
`"ddm"`, `"dms"`, `"iso6709"`, or `"nmea"` selects that interpretation while
retaining common normalization and full validation.

`PositionInput` is the public typing alias for accepted position text,
two-value sequences, named mappings, GeoJSON Points, and `Position` itself.
The `nautipy.coordinates` module also exposes typing aliases for coordinate
format and order values, `BatchErrorMode`, and `BatchInspectionItem`.

## Position and inspection results

A `Position` stores:

- latitude as a finite `float` in `[-90, 90]`;
- longitude as a finite `float` in `[-180, 180]`;
- optional keyword-only `identifier` and `description` metadata; and
- no display notation or source precision.

Identifiers are strings or finite JSON-style numbers. Descriptions are
strings. Metadata does not affect equality or hashing.

`inspect_position` returns an immutable `ParseResult` containing:

- the parsed `position`;
- the detected whole-position `format` and each axis's component format;
- source-order evidence;
- original and normalized text information;
- normalization labels;
- inferred angular resolution where available; and
- selected, equivalent, and rejected `CandidateDiagnostic` records.

If interpretations compete, inspection raises `AmbiguousCoordinateError`
instead of returning a selected `ParseResult`. The exception's `candidates`
attribute contains the competing diagnostics.

With `order="auto"`, `source_order` is `None` when equal unmarked values make
both source orders equivalent. Structured numeric input has no lexical tokens
or inferred source resolution.
The `warnings` field is reserved for recoverable concerns; current harmless
normalizations, including the `dmm` alias, are reported in `normalizations`
rather than as warnings.

### Batch inspection results

`inspect_positions` returns an immutable `BatchInspectionResult`. Its `items`
tuple contains one immutable record for each yielded input, in source order:

- `BatchInspectionSuccess(index, result)` holds the zero-based source `index`
  and the scalar `ParseResult`;
- `BatchInspectionFailure(index, error_type, message, candidates)` holds the
  zero-based source `index`, the public `CoordinateError` subclass, its
  message, and any competing `CandidateDiagnostic` records; and
- `BatchInspectionItem` is the
  `BatchInspectionSuccess | BatchInspectionFailure` typing alias.

`BatchInspectionSuccess`, `BatchInspectionFailure`, and
`BatchInspectionResult` are available from both `nautipy` and
`nautipy.coordinates`. `BatchInspectionItem` and the `BatchErrorMode` alias
for `"collect"` and `"raise"` are available from `nautipy.coordinates`.

The result provides derived, read-only `total_count`, `parsed_count`,
`ambiguous_count`, and `invalid_count` properties. Ambiguous failures are
those whose `error_type` is an `AmbiguousCoordinateError` subclass;
`invalid_count` covers every other coordinate failure. Therefore:

```text
total_count == parsed_count + ambiguous_count + invalid_count
```

With `errors="collect"`, the default, every yielded record is inspected.
Successful and failed records remain in their original positions; nothing is
silently dropped. With `errors="raise"`, inspection stops at the first
coordinate failure and raises the same public exception subclass. Its message
identifies the zero-based `positions[index]`, and an
`AmbiguousCoordinateError` retains its competing candidates. The original
scalar exception is retained as the raised exception's `__cause__`. Scalar
`inspect_position` behavior is unchanged. If every record succeeds, both modes
return the same success-only batch result.

The `order` and `format` options apply uniformly to every record. All options
are validated before iteration, including for an empty iterable. The default
remains `order="latlon"`; callers should select `order="auto"` only when each
otherwise ambiguous record contains hard axis evidence. An empty iterable
returns an empty result whose four counts are zero.

The outer argument must be an iterable of position inputs. A `str`, `bytes`,
`bytearray`, `Position`, or mapping is a scalar or structured value rather
than a batch and is rejected as the outer argument. Every other sequence is
treated as the batch itself, including a sequence of length two. Numeric
position pairs must therefore be nested:

```python
from nautipy import inspect_positions

batch = inspect_positions([
    (50.12257, 8.66570),
    (51.0, 9.0),
])
```

The iterable is consumed at most once. A non-iterable outer input, including a
`TypeError` while obtaining its iterator, produces `CoordinateParseError`.
Once the iterator has been obtained, an exception raised while advancing it
is not a record-level coordinate failure and propagates unchanged in both
modes. For a yielded record, only `CoordinateError` follows the selected
collect-or-raise policy; another exception propagates unchanged.

Direct construction of the public models uses the same coordinate-error
validation style. Indices must be non-negative integers; success records
require a `ParseResult`; failure records require a `CoordinateError` subclass
and string message. Failure candidates are normalized to a tuple and may be
non-empty only for an `AmbiguousCoordinateError` subclass.
`BatchInspectionResult` normalizes its item iterable to a tuple and requires
contiguous indices matching item order. Invalid model values raise
`CoordinateParseError`.

## Accepted position forms

### Decimal degrees (`dd`)

Decimal-degree components may use a sign, a hemisphere, or both where they do
not conflict:

```text
50.12257, 8.66570
50.12257 N; 8.66570 E
N 50.12257; E 8.66570
```

A negative sign combined with any hemisphere is rejected rather than treated
as double negation. An explicit `+` may accompany `N` or `E`, but contradicts
`S` or `W`.

### Degrees and decimal minutes (`ddm`)

`ddm` is the canonical name. `dmm` is accepted as an input and output option
alias and is canonicalized to `ddm` in inspection metadata.

```text
50° 7.3542' N; 8° 39.942' E
N 50 7.3542; E 8 39.942
50 deg 7.3542 min N; 8 deg 39.942 min E
50:7.3542 N; 8:39.942 E
```

### Degrees, minutes, and seconds (`dms`)

```text
50° 7' 21.252" N; 8° 39' 56.52" E
N 50 7 21.252; E 8 39 56.52
50 deg 7 min 21.252 sec N; 8 deg 39 min 56.52 sec E
50:7:21.252 N; 8:39:56.52 E
```

DD, DDM, and DMS component forms are accepted within a complete position pair
or a named latitude/longitude mapping. NautiPy does not expose a standalone
single-coordinate parser or an `axis=` argument. Latitude and longitude may
use different human-readable formats; inspection then reports the
whole-position format as `mixed`.

### ISO 6709 (`iso6709`)

NautiPy supports an unambiguous two-dimensional signed subset:

```text
+50.12257+008.66570/
+50.12257 -008.66570
+50.12257,+008.66570
```

Latitude and longitude fields require signs and the widths defined by their
axes. Decimal-degree (`DD`/`DDD`), DDM (`DDMM`/`DDDMM`), and DMS
(`DDMMSS`/`DDDMMSS`) fields are accepted, with a fraction on the
least-significant unit. Both components use the same form. Compact input has
no separator; a space or comma may separate fields.

Altitude, extra signed fields, and CRS suffixes are rejected. NautiPy never
treats them as longitude or silently discards them.

### NMEA coordinate fields (`nmea`)

NautiPy accepts latitude/longitude fields and direction fields without decoding
complete NMEA sentences:

```text
5007.3542,N,00839.9420,E
5007.3542 N; 00839.9420 E
```

Latitude uses `ddmm.mmmm`; longitude uses `dddmm.mmmm`. The decimal point and
at least one fractional-minute digit are required. The widths before the point
are exact. Supported pairs use either the four-field comma form or the
two-component semicolon form.

Directions are required and may prove the axes when input order is reversed.
Complete `$GPGGA`, `$GPRMC`, and other NMEA sentences are rejected.

### Structured Python values

Accepted forms include:

```python
parse_position((50.12257, 8.66570))
parse_position([50.12257, 8.66570])
parse_position({"lat": 50.12257, "lon": 8.66570})
parse_position({
    "latitude": "50° 7.3542' N",
    "longitude": "8° 39.942' E",
})
```

Named mappings contain exactly one latitude key (`lat` or `latitude`) and one
longitude key (`lon` or `longitude`). Unknown and extra fields are rejected.

A GeoJSON Point has specification-defined longitude/latitude order:

```python
parse_position({
    "type": "Point",
    "coordinates": [8.66570, 50.12257],
})
```

Point coordinates contain exactly two numeric values. Three-dimensional
coordinates, legacy `crs` members, conflicting named coordinate fields, and
members belonging to another GeoJSON object type are rejected. Ordinary
GeoJSON foreign members are accepted but not preserved.

The GeoJSON order does not apply to an ordinary two-value sequence; pass
`order="lonlat"` for such input.

## Normalization

NautiPy normalizes presentation only when meaning is preserved. Supported
normalizations include:

- outer and harmless repeated whitespace;
- Unicode minus signs;
- common degree, prime, and double-prime characters;
- case-insensitive hemisphere letters and full direction words;
- common degree/minute/second words and abbreviations; and
- decimal commas when pair syntax proves their role.

The original text remains available through `ParseResult`. Punctuation is not
discarded before the parser determines whether it is a decimal, component, or
pair separator.

### Decimal commas

These forms are unambiguous:

```text
50,12257 N; 8,66570 E
50° 7,3542' N; 8° 39,942' E
```

This form is not:

```text
50,12257, 8,66570
```

It raises `AmbiguousCoordinateError` with guidance to use a semicolon or dot
decimals. Coordinate order cannot resolve separator ambiguity.

## Coordinate order

`parse_position` and `inspect_position` default to `order="latlon"`.

- `latlon`: the first unmarked component is latitude.
- `lonlat`: the first unmarked component is longitude.
- `auto`: hard evidence is required; otherwise parsing is ambiguous.

Hard evidence includes axis markers, named fields, GeoJSON structure, NMEA
widths and directions, or one numeric value outside the latitude range but
inside the longitude range.

When both unmarked values are within `[-90, 90]`, `order="auto"` raises
`AmbiguousCoordinateError` unless both orders produce the same position.
NautiPy does not use geography, inhabited regions, sign patterns, or
statistical likelihood to guess. Explicit hemisphere and named-axis evidence
overrides textual order.

`output_order` independently controls `convert_position` output. ISO 6709
output is always latitude/longitude.

## Validation

For DDM and DMS:

- degrees are integral;
- minutes and seconds are in `[0, 60)`;
- latitude degrees are at most 90;
- longitude degrees are at most 180; and
- exactly 90 or 180 degrees requires zero subordinate components.

Every format rejects:

- NaN and infinity;
- missing components;
- repeated or conflicting hemisphere markers;
- sign/hemisphere conflicts;
- unexpected trailing tokens;
- out-of-range values; and
- altitude or other extra numeric fields.

User longitude outside `[-180, 180]` is rejected rather than wrapped.

## Formatting and conversion

Output formats are `dd`, `ddm`, `dms`, `iso6709`, and `nmea`. Defaults for
`Position(50.12257, 8.66570)` are:

| Format | Precision | Output |
| --- | ---: | --- |
| DD | 6 degree decimals | `50.122570, 8.665700` |
| DDM | 4 minute decimals | `50° 7.3542′ N; 8° 39.9420′ E` |
| DMS | 2 second decimals | `50° 7′ 21.25″ N; 8° 39′ 56.52″ E` |
| ISO 6709 | 6 degree decimals | `+50.122570+008.665700/` |
| NMEA | 4 minute decimals | `5007.3542,N,00839.9420,E` |

`precision` is the number of decimal places in the least-significant displayed
unit. It is an integer from 0 through 15; NMEA requires at least 1.
Formatting uses round-half-even, retains trailing zeros, carries rounding into
higher units, and never emits negative zero.

DD defaults to signed notation. DDM and DMS default to hemispheres and Unicode
symbols. The applicable options are:

- `notation="signed"` or `"hemisphere"` for DD, DDM, and DMS;
- `symbols="unicode"` or `"ascii"` for DD, DDM, and DMS;
- `compact=False` for separated ISO 6709 output;
- `separator=", "`, `"; "`, or `" / "` for human output;
- `separator=" "`, `","`, or `", "` for separated ISO 6709 output; and
- `separator=","` or `"; "` for NMEA output.

ISO 6709 output is signed, zero-padded, latitude/longitude, and ends with `/`.
NMEA output supports the four-field comma and two-component semicolon forms.

`inspect_position` reports lexical angular resolution as a positive exact
`Decimal` or `Fraction` quantum in decimal degrees where it can be inferred.
This is source resolution, not measurement accuracy.

With `precision=None`, `convert_position` selects the smallest supported output
precision that is no coarser than the finest known axis resolution. It uses the
documented defaults when source resolution is unavailable. If preservation
would require more than 15 places, conversion raises unless the caller chooses
an explicit precision and accepts rounding.

## Command line

The installed `nautipy` command exposes coordinate conversion and inspection:

```text
nautipy convert VALUE [--order ORDER] [--format FORMAT] [--to FORMAT]
    [--output-order ORDER] [--precision N] [--notation NOTATION]
    [--symbols SYMBOLS] [--compact | --no-compact] [--separator TEXT]

nautipy inspect VALUE [--order ORDER] [--format FORMAT]
```

Input and output choices match the Python API described above. `convert`
writes one formatted position. `inspect` writes deterministic JSON containing
the `ParseResult` fields; exact resolution values are encoded as strings.
Invalid input produces a concise command-line error and exit status 2 without
a traceback.

Run `nautipy COMMAND --help` for option choices and defaults.
`python -m nautipy` provides the same interface.

## Errors

```text
NautiPyError
└── CoordinateError
    ├── CoordinateParseError
    ├── CoordinateRangeError
    └── AmbiguousCoordinateError
```

`CoordinateParseError` covers invalid syntax, shape, and formatting options.
`CoordinateRangeError` covers non-finite and out-of-range numeric values.
`AmbiguousCoordinateError` exposes competing interpretations where available
and explains which order or syntax resolves the ambiguity. Batch collect mode
records these public exception types, messages, and ambiguity candidates
without changing the scalar hierarchy.

## Constraints and non-goals

Coordinate parsing, scalar and batch inspection, and formatting use only the
Python standard library, perform no file or network access, and do not load
geodesic or scientific modules.

The coordinate layer does not support arbitrary CRS or datum conversion, UTM,
MGRS, geohash, plus codes, altitude, depth, speed, time, full sensor messages,
corrupted-text repair, or operating-system locale inference.
