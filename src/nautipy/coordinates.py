"""Coordinate parsing and decimal-degree formatting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Context, Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from math import isfinite
from numbers import Rational, Real
import re
from typing import Literal, TypeAlias, cast

from ._machine import parse_iso6709, parse_nmea
from .errors import (
    AmbiguousCoordinateError,
    CoordinateError,
    CoordinateParseError,
    CoordinateRangeError,
)
from .position import Position

CoordinateOrder: TypeAlias = Literal["latlon", "lonlat", "auto"]
OutputOrder: TypeAlias = Literal["latlon", "lonlat"]
HumanCoordinateFormat: TypeAlias = Literal["dd", "ddm", "dms"]
CoordinateFormat: TypeAlias = Literal[
    "dd",
    "ddm",
    "dms",
    "iso6709",
    "nmea",
]
DetectedFormat: TypeAlias = CoordinateFormat | Literal["mixed"]
Axis: TypeAlias = Literal["lat", "lon"]
CandidateOutcome: TypeAlias = Literal[
    "selected",
    "equivalent",
    "rejected",
    "competing",
]
PositionInput: TypeAlias = (
    Position | str | Mapping[object, object] | Sequence[object]
)
_ExactNumber: TypeAlias = Decimal | Fraction

__all__ = [
    "CandidateDiagnostic",
    "CandidateOutcome",
    "CoordinateFormat",
    "CoordinateOrder",
    "DetectedFormat",
    "OutputOrder",
    "ParseResult",
    "PositionInput",
    "convert_position",
    "format_position",
    "inspect_position",
    "parse_position",
]

_SIGNED_NUMBER = (
    r"[+-]?(?:(?:\d+(?:[.,]\d+)?|[.,]\d+)"
    r"(?:[eE][+-]?\d+)?|nan|inf(?:inity)?)"
)
_SIGNED_COMPONENT_NUMBER = r"[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)"
_SIGNED_INTEGER = r"[+-]?\d+"
_MAX_TEXT_LENGTH = 4096
_MAX_PAIR_SPLITS = 256
_FORMAT_DECIMAL_CONTEXT = Context(
    prec=64,
    rounding=ROUND_HALF_EVEN,
    Emin=-999999,
    Emax=999999,
)


@dataclass(frozen=True, slots=True)
class CandidateDiagnostic:
    """One accepted or rejected interpretation of coordinate input."""

    format: str
    source_order: OutputOrder | None
    position: Position | None
    outcome: CandidateOutcome
    evidence: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ParseResult:
    """A parsed position together with detection and normalization metadata."""

    position: Position
    format: DetectedFormat
    component_formats: tuple[str, str]
    source_order: OutputOrder | None
    evidence: tuple[str, ...]
    original_text: str | None
    normalized_tokens: tuple[str, ...]
    normalizations: tuple[str, ...]
    warnings: tuple[str, ...]
    latitude_resolution: Decimal | Fraction | None
    longitude_resolution: Decimal | Fraction | None
    candidates: tuple[CandidateDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class _ComponentCandidate:
    value: _ExactNumber
    axis: Axis | None
    format: HumanCoordinateFormat
    precision: int | None
    resolution_degrees: _ExactNumber | None
    normalizations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ResolvedCandidate:
    position: Position
    exact_latitude: _ExactNumber
    exact_longitude: _ExactNumber
    format: DetectedFormat
    component_formats: tuple[str, str]
    source_order: OutputOrder | None
    order_evidence: str
    precision: tuple[int | None, int | None]
    resolution_degrees: tuple[
        _ExactNumber | None,
        _ExactNumber | None,
    ]
    normalizations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _CandidateSelection:
    selected: _ResolvedCandidate
    equivalents: tuple[_ResolvedCandidate, ...] = ()
    normalized_text: str | None = None


def _validate_parse_order(order: str) -> CoordinateOrder:
    if not isinstance(order, str) or order not in {
        "latlon",
        "lonlat",
        "auto",
    }:
        raise CoordinateParseError(
            'order must be "latlon", "lonlat", or "auto"'
        )
    return cast(CoordinateOrder, order)


def _validate_output_order(order: str) -> OutputOrder:
    if not isinstance(order, str) or order not in {"latlon", "lonlat"}:
        raise CoordinateParseError('output order must be "latlon" or "lonlat"')
    return cast(OutputOrder, order)


def _validate_input_format(value: str | None) -> CoordinateFormat | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CoordinateParseError(
            'format must be "dd", "ddm", "dms", "iso6709", or "nmea"'
        )
    normalized = value.casefold()
    if normalized == "dmm":
        normalized = "ddm"
    if normalized not in {"dd", "ddm", "dms", "iso6709", "nmea"}:
        raise CoordinateParseError(
            'format must be "dd", "ddm", "dms", "iso6709", or "nmea"'
        )
    return cast(CoordinateFormat, normalized)


def _normalize_text(value: str) -> tuple[str, tuple[str, ...]]:
    if len(value) > _MAX_TEXT_LENGTH:
        raise CoordinateParseError(
            f"position text must not exceed {_MAX_TEXT_LENGTH} characters"
        )
    text = value.strip()
    if not text:
        raise CoordinateParseError("position text cannot be empty")

    normalizations: list[str] = []
    if text != value:
        normalizations.append("trimmed outer whitespace")
    translated = text.translate(
        str.maketrans(
            {
                "−": "-",
                "º": "°",
                "′": "'",
                "’": "'",
                "‘": "'",
                "″": '"',
                "”": '"',
                "“": '"',
            }
        )
    )
    if translated != text:
        normalizations.append("normalized Unicode coordinate symbols")
        text = translated

    replacements = (
        (r"\bnorth\b", "N", "normalized direction words"),
        (r"\bsouth\b", "S", "normalized direction words"),
        (r"\beast\b", "E", "normalized direction words"),
        (r"\bwest\b", "W", "normalized direction words"),
        (r"\b(?:degrees?|deg)\b", "°", "normalized degree units"),
        (r"\b(?:minutes?|mins?|min)\b", "'", "normalized minute units"),
        (r"\b(?:seconds?|secs?|sec)\b", '"', "normalized second units"),
    )
    for pattern, replacement, label in replacements:
        replaced = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE | re.ASCII,
        )
        if replaced != text and label not in normalizations:
            normalizations.append(label)
        text = replaced

    direction_patterns = (
        r"(^|[\s,;/])([nsew])(?=$|[\s,;/])",
        r"(^|[\s,;/])([nsew])(?=[+\-\d.,])",
        r"(?<=[0-9°'\"])([nsew])(?=$|[\s,;/])",
    )
    for pattern in direction_patterns:
        replaced = re.sub(
            pattern,
            lambda match: "".join(
                part.upper() if part is not None else ""
                for part in match.groups()
            ),
            text,
            flags=re.ASCII,
        )
        if replaced != text:
            if "normalized hemisphere letter case" not in normalizations:
                normalizations.append("normalized hemisphere letter case")
            text = replaced

    collapsed = re.sub(r"\s+", " ", text).strip()
    if collapsed != text:
        normalizations.append("collapsed whitespace")
    return collapsed, tuple(normalizations)


def _extract_direction(text: str) -> tuple[str, str | None, bool]:
    body = text.strip()
    prefix_match = re.match(
        r"^([NSEW])(?=\s|[+\-\d.,])\s*",
        body,
        flags=re.IGNORECASE | re.ASCII,
    )
    prefix_text = prefix_match.group(1) if prefix_match else None
    prefix = prefix_text.upper() if prefix_text else None
    if prefix_match:
        body = body[prefix_match.end() :].strip()

    suffix_match = re.search(
        r"\s*(?<=[0-9°'\".,\s])([NSEW])\s*$",
        body,
        flags=re.IGNORECASE | re.ASCII,
    )
    suffix_text = suffix_match.group(1) if suffix_match else None
    suffix = suffix_text.upper() if suffix_text else None
    if suffix_match:
        body = body[: suffix_match.start()].strip()

    if prefix is not None and suffix is not None:
        raise CoordinateParseError(
            "a coordinate component may contain only one hemisphere marker"
        )
    direction_text = prefix_text or suffix_text
    case_normalized = (
        direction_text is not None and direction_text != direction_text.upper()
    )
    return body, prefix or suffix, case_normalized


def _component_normalizations(
    normalizations: tuple[str, ...],
    *,
    direction_case_normalized: bool,
    decimal_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    values = list(normalizations)
    if direction_case_normalized:
        values.append("normalized hemisphere letter case")
    if any("," in token for token in decimal_tokens):
        values.append("normalized decimal comma")
    return tuple(dict.fromkeys(values))


def _axis_for_direction(direction: str | None) -> Axis | None:
    if direction in {"N", "S"}:
        return "lat"
    if direction in {"E", "W"}:
        return "lon"
    return None


def _parse_decimal_token(
    token: str,
    *,
    component: str,
) -> tuple[Decimal, int, Decimal]:
    normalized = token.strip()
    if normalized.count(",") > 1 or (
        "," in normalized and "." in normalized
    ):
        raise CoordinateParseError(f"invalid {component}: {token!r}")
    normalized = normalized.replace(",", ".")
    try:
        value = Decimal(normalized)
    except InvalidOperation as error:
        raise CoordinateParseError(f"invalid {component}: {token!r}") from error
    if not value.is_finite():
        raise CoordinateRangeError(f"{component} must be finite")

    exponent = value.as_tuple().exponent
    if exponent < -323 or exponent > 308:
        raise CoordinateParseError(
            f"{component} exponent or precision is outside the supported "
            "float range"
        )

    mantissa = normalized.lower().split("e", 1)[0]
    precision = len(mantissa.partition(".")[2]) if "." in mantissa else 0
    resolution = Decimal((0, (1,), exponent))
    return value, precision, resolution


def _exact_abs(value: _ExactNumber) -> _ExactNumber:
    return value.copy_abs() if isinstance(value, Decimal) else abs(value)


def _exact_negate(value: _ExactNumber) -> _ExactNumber:
    return value.copy_negate() if isinstance(value, Decimal) else -value


def _signed_value(
    degree_token: str,
    magnitude: _ExactNumber,
    direction: str | None,
) -> _ExactNumber:
    token = degree_token.strip()
    explicit_sign = token[0] if token and token[0] in "+-" else None

    if direction is not None:
        if explicit_sign == "-":
            raise CoordinateParseError(
                "do not combine a negative sign with a hemisphere marker"
            )
        if explicit_sign == "+" and direction in {"S", "W"}:
            raise CoordinateParseError(
                "positive sign contradicts south/west hemisphere marker"
            )
        return _exact_negate(magnitude) if direction in {"S", "W"} else magnitude
    return _exact_negate(magnitude) if explicit_sign == "-" else magnitude


def _parse_dd_component(
    text: str,
    normalizations: tuple[str, ...],
) -> _ComponentCandidate | None:
    body, direction, direction_case_normalized = _extract_direction(text)
    match = re.fullmatch(
        rf"\s*(?P<degrees>{_SIGNED_NUMBER})\s*°?\s*",
        body,
        flags=re.IGNORECASE | re.ASCII,
    )
    if match is None:
        return None

    degrees, precision, resolution = _parse_decimal_token(
        match.group("degrees"),
        component="decimal degrees",
    )
    value = _signed_value(
        match.group("degrees"),
        _exact_abs(degrees),
        direction,
    )
    return _ComponentCandidate(
        value=value,
        axis=_axis_for_direction(direction),
        format="dd",
        precision=precision,
        resolution_degrees=resolution,
        normalizations=_component_normalizations(
            normalizations,
            direction_case_normalized=direction_case_normalized,
            decimal_tokens=(match.group("degrees"),),
        ),
    )


def _match_ddm(body: str) -> re.Match[str] | None:
    patterns = (
        rf"(?P<degrees>{_SIGNED_INTEGER})\s*°\s*"
        rf"(?P<minutes>{_SIGNED_COMPONENT_NUMBER})\s*'?",
        rf"(?P<degrees>{_SIGNED_INTEGER})\s*:\s*"
        rf"(?P<minutes>{_SIGNED_COMPONENT_NUMBER})\s*'?",
        rf"(?P<degrees>{_SIGNED_INTEGER})\s+"
        rf"(?P<minutes>{_SIGNED_COMPONENT_NUMBER})\s*'?",
    )
    for pattern in patterns:
        match = re.fullmatch(
            pattern,
            body,
            flags=re.IGNORECASE | re.ASCII,
        )
        if match is not None:
            return match
    return None


def _parse_ddm_component(
    text: str,
    normalizations: tuple[str, ...],
) -> _ComponentCandidate | None:
    body, direction, direction_case_normalized = _extract_direction(text)
    match = _match_ddm(body)
    if match is None:
        return None

    degrees, _, _ = _parse_decimal_token(
        match.group("degrees"),
        component="degrees",
    )
    minutes, precision, minute_resolution = _parse_decimal_token(
        match.group("minutes"),
        component="minutes",
    )
    if not Decimal(0) <= minutes < Decimal(60):
        raise CoordinateRangeError("minutes must be in the range [0, 60)")

    magnitude = Fraction(_exact_abs(degrees)) + Fraction(minutes) / 60
    value = _signed_value(match.group("degrees"), magnitude, direction)
    return _ComponentCandidate(
        value=value,
        axis=_axis_for_direction(direction),
        format="ddm",
        precision=precision,
        resolution_degrees=Fraction(minute_resolution) / 60,
        normalizations=_component_normalizations(
            normalizations,
            direction_case_normalized=direction_case_normalized,
            decimal_tokens=(match.group("degrees"), match.group("minutes")),
        ),
    )


def _match_dms(body: str) -> re.Match[str] | None:
    patterns = (
        rf"(?P<degrees>{_SIGNED_INTEGER})\s*°\s*"
        rf"(?P<minutes>{_SIGNED_COMPONENT_NUMBER})\s*'\s*"
        rf'(?P<seconds>{_SIGNED_COMPONENT_NUMBER})\s*"?',
        rf"(?P<degrees>{_SIGNED_INTEGER})\s*:\s*"
        rf"(?P<minutes>{_SIGNED_COMPONENT_NUMBER})\s*:\s*"
        rf'(?P<seconds>{_SIGNED_COMPONENT_NUMBER})\s*"?',
        rf"(?P<degrees>{_SIGNED_INTEGER})\s+"
        rf"(?P<minutes>{_SIGNED_COMPONENT_NUMBER})\s+"
        rf'(?P<seconds>{_SIGNED_COMPONENT_NUMBER})\s*"?',
    )
    for pattern in patterns:
        match = re.fullmatch(
            pattern,
            body,
            flags=re.IGNORECASE | re.ASCII,
        )
        if match is not None:
            return match
    return None


def _parse_dms_component(
    text: str,
    normalizations: tuple[str, ...],
) -> _ComponentCandidate | None:
    body, direction, direction_case_normalized = _extract_direction(text)
    match = _match_dms(body)
    if match is None:
        return None

    degrees, _, _ = _parse_decimal_token(
        match.group("degrees"),
        component="degrees",
    )
    minutes, _, _ = _parse_decimal_token(
        match.group("minutes"),
        component="minutes",
    )
    seconds, precision, second_resolution = _parse_decimal_token(
        match.group("seconds"),
        component="seconds",
    )
    if not Decimal(0) <= minutes < Decimal(60):
        raise CoordinateRangeError("minutes must be in the range [0, 60)")
    if not Decimal(0) <= seconds < Decimal(60):
        raise CoordinateRangeError("seconds must be in the range [0, 60)")

    magnitude = (
        Fraction(_exact_abs(degrees))
        + Fraction(minutes) / 60
        + Fraction(seconds) / 3600
    )
    value = _signed_value(match.group("degrees"), magnitude, direction)
    return _ComponentCandidate(
        value=value,
        axis=_axis_for_direction(direction),
        format="dms",
        precision=precision,
        resolution_degrees=Fraction(second_resolution) / 3600,
        normalizations=_component_normalizations(
            normalizations,
            direction_case_normalized=direction_case_normalized,
            decimal_tokens=(
                match.group("degrees"),
                match.group("minutes"),
                match.group("seconds"),
            ),
        ),
    )


_COMPONENT_PARSERS = {
    "dd": _parse_dd_component,
    "ddm": _parse_ddm_component,
    "dms": _parse_dms_component,
}


def _valid_for_axis(value: _ExactNumber, axis: Axis) -> bool:
    limit = Decimal(90 if axis == "lat" else 180)
    return limit.copy_negate() <= value <= limit


def _resolved_candidate(
    latitude: _ExactNumber,
    longitude: _ExactNumber,
    *,
    format: DetectedFormat,
    source_order: OutputOrder | None,
    order_evidence: str,
    component_formats: tuple[str, str] | None = None,
    precision: tuple[int | None, int | None] = (None, None),
    resolution_degrees: tuple[
        _ExactNumber | None,
        _ExactNumber | None,
    ] = (None, None),
    normalizations: tuple[str, ...] = (),
) -> _ResolvedCandidate:
    if not _valid_for_axis(latitude, "lat"):
        raise CoordinateRangeError(
            "latitude must be between -90 and 90 degrees"
        )
    if not _valid_for_axis(longitude, "lon"):
        raise CoordinateRangeError(
            "longitude must be between -180 and 180 degrees"
        )
    position = Position(latitude, longitude)
    return _ResolvedCandidate(
        position=position,
        exact_latitude=latitude,
        exact_longitude=longitude,
        format=format,
        component_formats=(
            (format, format) if component_formats is None else component_formats
        ),
        source_order=source_order,
        order_evidence=order_evidence,
        precision=precision,
        resolution_degrees=resolution_degrees,
        normalizations=normalizations,
    )


def _resolve_components(
    first: _ComponentCandidate,
    second: _ComponentCandidate,
    order: CoordinateOrder,
) -> _ResolvedCandidate:
    normalizations = tuple(
        dict.fromkeys(first.normalizations + second.normalizations)
    )
    detected_format: DetectedFormat = (
        first.format if first.format == second.format else "mixed"
    )

    def resolve_in_order(
        latitude: _ComponentCandidate,
        longitude: _ComponentCandidate,
        *,
        source_order: OutputOrder | None,
        order_evidence: str,
    ) -> _ResolvedCandidate:
        return _resolved_candidate(
            latitude.value,
            longitude.value,
            format=detected_format,
            component_formats=(latitude.format, longitude.format),
            source_order=source_order,
            order_evidence=order_evidence,
            precision=(latitude.precision, longitude.precision),
            resolution_degrees=(
                latitude.resolution_degrees,
                longitude.resolution_degrees,
            ),
            normalizations=normalizations,
        )

    if first.axis is not None or second.axis is not None:
        if first.axis is not None and first.axis == second.axis:
            raise CoordinateParseError(
                "both coordinate components describe the same axis"
            )
        if first.axis == "lat" or second.axis == "lon":
            latitude, longitude = first, second
            source_order: OutputOrder = "latlon"
        elif first.axis == "lon" or second.axis == "lat":
            latitude, longitude = second, first
            source_order = "lonlat"
        else:
            raise CoordinateParseError(
                "could not resolve coordinate axes from hemisphere markers"
            )
        return resolve_in_order(
            latitude,
            longitude,
            source_order=source_order,
            order_evidence="hemisphere markers",
        )

    if order == "latlon":
        return resolve_in_order(
            first,
            second,
            source_order="latlon",
            order_evidence="order argument or default",
        )
    if order == "lonlat":
        return resolve_in_order(
            second,
            first,
            source_order="lonlat",
            order_evidence="order argument or default",
        )

    latlon_valid = _valid_for_axis(first.value, "lat") and _valid_for_axis(
        second.value,
        "lon",
    )
    lonlat_valid = _valid_for_axis(first.value, "lon") and _valid_for_axis(
        second.value,
        "lat",
    )
    if latlon_valid and lonlat_valid:
        if first.value == second.value:
            known_resolutions = tuple(
                resolution
                for resolution in (
                    first.resolution_degrees,
                    second.resolution_degrees,
                )
                if resolution is not None
            )
            conservative_resolution = (
                min(known_resolutions) if known_resolutions else None
            )
            known_precision = tuple(
                precision
                for precision in (first.precision, second.precision)
                if precision is not None
            )
            conservative_precision = (
                max(known_precision) if known_precision else None
            )
            return _resolved_candidate(
                first.value,
                second.value,
                format=detected_format,
                source_order=None,
                order_evidence="equivalent coordinate orders",
                component_formats=(detected_format, detected_format),
                precision=(conservative_precision, conservative_precision),
                resolution_degrees=(
                    conservative_resolution,
                    conservative_resolution,
                ),
                normalizations=normalizations,
            )
        latlon_candidate = resolve_in_order(
            first,
            second,
            source_order="latlon",
            order_evidence="valid candidate coordinate order",
        )
        lonlat_candidate = resolve_in_order(
            second,
            first,
            source_order="lonlat",
            order_evidence="valid candidate coordinate order",
        )
        raise AmbiguousCoordinateError(
            "could not determine coordinate order; "
            f"latlon -> ({latlon_candidate.exact_latitude}, "
            f"{latlon_candidate.exact_longitude}), "
            f"lonlat -> ({lonlat_candidate.exact_latitude}, "
            f"{lonlat_candidate.exact_longitude}); "
            'pass order="latlon" or order="lonlat"',
            candidates=(
                _candidate_diagnostic(
                    latlon_candidate,
                    outcome="competing",
                ),
                _candidate_diagnostic(
                    lonlat_candidate,
                    outcome="competing",
                ),
            ),
        )
    if latlon_valid:
        return resolve_in_order(
            first,
            second,
            source_order="latlon",
            order_evidence="numeric range",
        )
    if lonlat_valid:
        return resolve_in_order(
            second,
            first,
            source_order="lonlat",
            order_evidence="numeric range",
        )
    raise CoordinateRangeError(
        f"no valid latitude/longitude order for {first.value}, {second.value}"
    )


def _contains_axis_marker(text: str) -> bool:
    prefix = re.search(
        r"(?:^|[\s,;/])[NSEW](?=\s*[+\-\d.,])",
        text,
        flags=re.IGNORECASE | re.ASCII,
    )
    suffix = re.search(
        r"(?<=[0-9°'\".,\s])[NSEW](?=$|[\s,;/])",
        text,
        flags=re.IGNORECASE | re.ASCII,
    )
    return prefix is not None or suffix is not None


def _pair_splits(text: str) -> tuple[tuple[str, str], ...]:
    for separator in (";", "/"):
        if separator in text:
            if text.count(separator) != 1:
                raise CoordinateParseError(
                    f"position must contain exactly one {separator!r} separator"
                )
            first, second = (part.strip() for part in text.split(separator))
            if not first or not second:
                raise CoordinateParseError(
                    "position must contain exactly two coordinate components"
                )
            return ((first, second),)

    candidates: list[tuple[str, str]] = []
    for index, character in enumerate(text):
        if character == ",":
            first, second = text[:index].strip(), text[index + 1 :].strip()
            if first and second:
                candidates.append((first, second))
                if len(candidates) > _MAX_PAIR_SPLITS:
                    raise CoordinateParseError(
                        "position text contains too many possible separators"
                    )
    for match in re.finditer(r"\s+", text):
        first, second = text[: match.start()].strip(), text[match.end() :].strip()
        if first and second:
            candidates.append((first, second))
            if len(candidates) > _MAX_PAIR_SPLITS:
                raise CoordinateParseError(
                    "position text contains too many possible separators"
                )

    return tuple(dict.fromkeys(candidates))


def _has_unmarked_decimal_comma(text: str) -> bool:
    return (
        text.count(",") > 1
        and re.search(
            r"(?<![0-9.])[+-]?(?:[0-9]+,[0-9]+|,[0-9]+)(?![0-9.])",
            text,
            flags=re.ASCII,
        )
        is not None
        and not _contains_axis_marker(text)
        and ";" not in text
        and "/" not in text
    )


def _decimal_comma_ambiguity() -> AmbiguousCoordinateError:
    return AmbiguousCoordinateError(
        "decimal commas and the coordinate separator are ambiguous; "
        "separate the coordinates with a semicolon or use dot decimals"
    )


def _has_bare_numeric_grouping(text: str) -> bool:
    tokens = text.split()
    return len(tokens) > 2 and all(
        re.fullmatch(
            _SIGNED_NUMBER,
            token,
            flags=re.IGNORECASE | re.ASCII,
        )
        is not None
        for token in tokens
    )


def _bare_numeric_ambiguity() -> AmbiguousCoordinateError:
    return AmbiguousCoordinateError(
        "three or more bare numeric fields could include an altitude or use "
        "different coordinate groupings; insert ';' between the two "
        "coordinates, add axis/unit markers, or pass an explicit format"
    )


def _raise_best_error(
    errors: list[CoordinateError],
    *,
    selected_format: CoordinateFormat | None,
) -> None:
    for error_type in (
        AmbiguousCoordinateError,
        CoordinateRangeError,
        CoordinateParseError,
    ):
        for error in errors:
            if isinstance(error, error_type):
                raise error
    if selected_format is None:
        raise CoordinateParseError(
            "could not parse position as DD, DDM, DMS, ISO 6709, or NMEA"
        )
    raise CoordinateParseError(
        f"input does not match the requested {selected_format} format"
    )


def _deduplicate_candidates(
    candidates: list[_ResolvedCandidate],
    *,
    normalized_text: str | None = None,
) -> _CandidateSelection:
    unique: dict[
        tuple[_ExactNumber, _ExactNumber],
        _ResolvedCandidate,
    ] = {}
    for candidate in candidates:
        key = (candidate.exact_latitude, candidate.exact_longitude)
        unique.setdefault(key, candidate)
    if len(unique) > 1:
        details = ", ".join(
            f"{candidate.format}: "
            f"({candidate.exact_latitude}, {candidate.exact_longitude})"
            for candidate in unique.values()
        )
        raise AmbiguousCoordinateError(
            f"coordinate text has multiple valid interpretations: {details}; "
            "insert ';' between coordinate components or pass an explicit "
            "format",
            candidates=tuple(
                _candidate_diagnostic(candidate, outcome="competing")
                for candidate in unique.values()
            ),
        )
    selected = next(iter(unique.values()))
    equivalent_candidates: list[_ResolvedCandidate] = []
    seen = {
        (
            selected.format,
            selected.source_order,
            selected.precision,
            selected.resolution_degrees,
        )
    }
    for candidate in candidates:
        if candidate is selected:
            continue
        key = (
            candidate.format,
            candidate.source_order,
            candidate.precision,
            candidate.resolution_degrees,
        )
        if key not in seen:
            equivalent_candidates.append(candidate)
            seen.add(key)
    return _CandidateSelection(
        selected,
        tuple(equivalent_candidates),
        normalized_text,
    )


def _candidate_formats(
    selected_format: CoordinateFormat | None,
) -> tuple[HumanCoordinateFormat, ...]:
    if selected_format in {"dd", "ddm", "dms"}:
        return (selected_format,)
    if selected_format is not None:
        return ()
    return ("dd", "ddm", "dms")


def _text_component_candidates(
    text: str,
    formats: tuple[HumanCoordinateFormat, ...],
    normalizations: tuple[str, ...],
) -> tuple[list[_ComponentCandidate], list[CoordinateError], bool]:
    candidates: list[_ComponentCandidate] = []
    errors: list[CoordinateError] = []
    saw_non_dd_syntax = False
    for format_name in formats:
        try:
            candidate = _COMPONENT_PARSERS[format_name](text, normalizations)
            if candidate is not None:
                candidates.append(candidate)
                if format_name != "dd":
                    saw_non_dd_syntax = True
        except CoordinateError as error:
            errors.append(error)
            if format_name != "dd":
                saw_non_dd_syntax = True
    return candidates, errors, saw_non_dd_syntax


def _human_position_candidates(
    text: str,
    *,
    order: CoordinateOrder,
    formats: tuple[HumanCoordinateFormat, ...],
    normalizations: tuple[str, ...],
) -> tuple[list[_ResolvedCandidate], list[CoordinateError], bool]:
    splits = _pair_splits(text)
    if not splits:
        return [], [], False

    candidates: list[_ResolvedCandidate] = []
    errors: list[CoordinateError] = []
    saw_non_dd_syntax = False
    for first_text, second_text in splits:
        (
            first_candidates,
            first_errors,
            first_non_dd,
        ) = _text_component_candidates(first_text, formats, normalizations)
        (
            second_candidates,
            second_errors,
            second_non_dd,
        ) = _text_component_candidates(
            second_text,
            formats,
            normalizations,
        )
        errors.extend(first_errors)
        errors.extend(second_errors)
        saw_non_dd_syntax = saw_non_dd_syntax or first_non_dd or second_non_dd
        for first in first_candidates:
            for second in second_candidates:
                try:
                    candidates.append(_resolve_components(first, second, order))
                except CoordinateError as error:
                    errors.append(error)
    return candidates, errors, saw_non_dd_syntax


def _equivalent_human_candidates(
    text: str,
    *,
    selected: _ResolvedCandidate,
    order: CoordinateOrder,
    normalizations: tuple[str, ...],
) -> tuple[_ResolvedCandidate, ...]:
    if text.endswith("/") or not any(character in text for character in " ,"):
        return ()
    try:
        candidates, _, _ = _human_position_candidates(
            text,
            order=order,
            formats=("dd", "ddm", "dms"),
            normalizations=normalizations,
        )
    except CoordinateError:
        return ()
    equivalents: list[_ResolvedCandidate] = []
    seen: set[tuple[object, ...]] = set()
    for candidate in candidates:
        if (
            candidate.exact_latitude != selected.exact_latitude
            or candidate.exact_longitude != selected.exact_longitude
        ):
            continue
        key = (
            candidate.format,
            candidate.source_order,
            candidate.precision,
            candidate.resolution_degrees,
        )
        if key not in seen:
            equivalents.append(candidate)
            seen.add(key)
    return tuple(equivalents)


def _parse_text_position(
    value: str,
    *,
    order: CoordinateOrder,
    selected_format: CoordinateFormat | None,
) -> _CandidateSelection:
    text, normalizations = _normalize_text(value)

    if selected_format in {None, "iso6709"}:
        machine = parse_iso6709(
            text,
            required=selected_format == "iso6709",
        )
        if machine is not None:
            resolved = _resolved_candidate(
                machine.latitude,
                machine.longitude,
                format=machine.format,
                component_formats=(machine.format, machine.format),
                source_order=machine.source_order,
                order_evidence="ISO 6709 field widths and signs",
                precision=machine.precision,
                resolution_degrees=machine.resolution_degrees,
                normalizations=normalizations,
            )
            return _CandidateSelection(
                resolved,
                _equivalent_human_candidates(
                    text,
                    selected=resolved,
                    order=order,
                    normalizations=normalizations,
                ),
                normalized_text=text,
            )
    if selected_format in {None, "nmea"}:
        machine = parse_nmea(
            text,
            required=selected_format == "nmea",
        )
        if machine is not None:
            return _CandidateSelection(
                _resolved_candidate(
                    machine.latitude,
                    machine.longitude,
                    format=machine.format,
                    component_formats=(machine.format, machine.format),
                    source_order=machine.source_order,
                    order_evidence="NMEA directions and field widths",
                    precision=machine.precision,
                    resolution_degrees=machine.resolution_degrees,
                    normalizations=normalizations,
                ),
                normalized_text=text,
            )

    formats = _candidate_formats(selected_format)
    candidates, errors, saw_non_dd_syntax = _human_position_candidates(
        text,
        order=order,
        formats=formats,
        normalizations=normalizations,
    )
    if not candidates and not errors and not _pair_splits(text):
        raise CoordinateParseError(
            "position text must contain exactly two coordinate components"
        )

    if (
        candidates
        and selected_format is None
        and _has_bare_numeric_grouping(text)
    ):
        raise _bare_numeric_ambiguity()
    if not candidates:
        if _has_unmarked_decimal_comma(text) and not saw_non_dd_syntax:
            raise _decimal_comma_ambiguity()
        _raise_best_error(errors, selected_format=selected_format)
    if _has_unmarked_decimal_comma(text) and all(
        candidate.format == "dd" for candidate in candidates
    ):
        raise _decimal_comma_ambiguity()
    return _deduplicate_candidates(candidates, normalized_text=text)


def _sequence_pair(value: object) -> tuple[object, object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise CoordinateParseError(
            "position must be coordinate text or a two-value sequence"
        )
    if len(value) != 2:
        raise CoordinateParseError("position sequence must contain exactly two values")
    return value[0], value[1]


def _exact_from_real(value: object) -> _ExactNumber:
    if isinstance(value, bool):
        raise CoordinateParseError("coordinate values must be real numbers")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise CoordinateRangeError("coordinate values must be finite")
        return value
    if not isinstance(value, Real):
        raise CoordinateParseError("coordinate values must be real numbers")
    if isinstance(value, Rational):
        try:
            return Fraction(int(value.numerator), int(value.denominator))
        except (AttributeError, TypeError, ValueError, ZeroDivisionError) as error:
            raise CoordinateParseError(
                "coordinate values must be real numbers"
            ) from error

    index_method = getattr(value, "__index__", None)
    if callable(index_method):
        try:
            return Fraction(int(index_method()), 1)
        except (OverflowError, TypeError, ValueError) as error:
            raise CoordinateParseError(
                "coordinate values must be real numbers"
            ) from error

    ratio_method = getattr(value, "as_integer_ratio", None)
    if callable(ratio_method):
        try:
            numerator, denominator = ratio_method()
            return Fraction(int(numerator), int(denominator))
        except (OverflowError, TypeError, ValueError, ZeroDivisionError):
            pass

    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise CoordinateParseError(
            "coordinate values must be real numbers"
        ) from error
    if not isfinite(number):
        raise CoordinateRangeError("coordinate values must be finite")
    return Fraction.from_float(number)


def _component_from_structured_value(
    value: object,
    *,
    format_name: HumanCoordinateFormat,
) -> _ComponentCandidate | None:
    if isinstance(value, str):
        normalized, normalizations = _normalize_text(value)
        return _COMPONENT_PARSERS[format_name](normalized, normalizations)
    if format_name != "dd":
        return None
    return _ComponentCandidate(
        value=_exact_from_real(value),
        axis=None,
        format="dd",
        precision=None,
        resolution_degrees=None,
    )


def _structured_component_candidates(
    value: object,
    formats: tuple[HumanCoordinateFormat, ...],
) -> tuple[list[_ComponentCandidate], list[CoordinateError]]:
    candidates: list[_ComponentCandidate] = []
    errors: list[CoordinateError] = []
    for format_name in formats:
        try:
            candidate = _component_from_structured_value(
                value,
                format_name=format_name,
            )
            if candidate is not None:
                candidates.append(candidate)
        except CoordinateError as error:
            errors.append(error)
    return candidates, errors


def _resolve_named_components(
    latitude: _ComponentCandidate,
    longitude: _ComponentCandidate,
) -> _ResolvedCandidate:
    if latitude.axis not in {None, "lat"}:
        raise CoordinateParseError(
            "latitude mapping value contradicts its named axis"
        )
    if longitude.axis not in {None, "lon"}:
        raise CoordinateParseError(
            "longitude mapping value contradicts its named axis"
        )
    detected_format: DetectedFormat = (
        latitude.format
        if latitude.format == longitude.format
        else "mixed"
    )
    normalizations = tuple(
        dict.fromkeys(latitude.normalizations + longitude.normalizations)
    )
    return _resolved_candidate(
        latitude.value,
        longitude.value,
        format=detected_format,
        component_formats=(latitude.format, longitude.format),
        source_order="latlon",
        order_evidence="named fields",
        precision=(latitude.precision, longitude.precision),
        resolution_degrees=(
            latitude.resolution_degrees,
            longitude.resolution_degrees,
        ),
        normalizations=normalizations,
    )


def _parse_named_mapping(
    value: Mapping[object, object],
    *,
    selected_format: CoordinateFormat | None,
) -> _CandidateSelection:
    latitude_keys = [key for key in ("lat", "latitude") if key in value]
    longitude_keys = [key for key in ("lon", "longitude") if key in value]
    if len(latitude_keys) != 1 or len(longitude_keys) != 1:
        raise CoordinateParseError(
            "position mapping must contain exactly one latitude key "
            '("lat" or "latitude") and one longitude key '
            '("lon" or "longitude")'
        )
    expected_keys = {latitude_keys[0], longitude_keys[0]}
    if set(value) != expected_keys:
        raise CoordinateParseError(
            "named position mappings cannot contain unknown or extra fields"
        )

    formats = _candidate_formats(selected_format)
    latitude_candidates, latitude_errors = _structured_component_candidates(
        value[latitude_keys[0]],
        formats,
    )
    longitude_candidates, longitude_errors = _structured_component_candidates(
        value[longitude_keys[0]],
        formats,
    )
    candidates: list[_ResolvedCandidate] = []
    errors = latitude_errors + longitude_errors
    for latitude in latitude_candidates:
        for longitude in longitude_candidates:
            try:
                candidates.append(
                    _resolve_named_components(latitude, longitude)
                )
            except CoordinateError as error:
                errors.append(error)
    if not candidates:
        _raise_best_error(errors, selected_format=selected_format)
    return _deduplicate_candidates(candidates)


def _parse_geojson_point(
    value: Mapping[object, object],
    *,
    selected_format: CoordinateFormat | None,
) -> _CandidateSelection:
    if value.get("type") != "Point":
        raise CoordinateParseError(
            'GeoJSON geometry type must be exactly "Point"'
        )
    if "coordinates" not in value:
        raise CoordinateParseError("GeoJSON Point is missing coordinates")
    if "crs" in value:
        raise CoordinateParseError("legacy GeoJSON CRS members are unsupported")
    cross_type_members = sorted(
        member
        for member in ("features", "geometries", "geometry", "properties")
        if member in value
    )
    if cross_type_members:
        rendered = ", ".join(repr(member) for member in cross_type_members)
        raise CoordinateParseError(
            "GeoJSON Point contains member(s) belonging to another "
            f"GeoJSON type: {rendered}"
        )
    if any(
        key in value for key in ("lat", "latitude", "lon", "longitude")
    ):
        raise CoordinateParseError(
            "GeoJSON Point cannot contain competing named coordinate fields"
        )
    if selected_format not in {None, "dd"}:
        raise CoordinateParseError(
            f"GeoJSON Point input does not match the requested "
            f"{selected_format} format"
        )

    coordinates = value["coordinates"]
    if (
        isinstance(coordinates, (str, bytes, bytearray))
        or not isinstance(coordinates, Sequence)
        or len(coordinates) != 2
    ):
        raise CoordinateParseError(
            "GeoJSON Point coordinates must contain exactly two numeric "
            "values in longitude/latitude order"
        )
    longitude = _exact_from_real(coordinates[0])
    latitude = _exact_from_real(coordinates[1])
    return _CandidateSelection(
        _resolved_candidate(
            latitude,
            longitude,
            format="dd",
            component_formats=("dd", "dd"),
            source_order="lonlat",
            order_evidence="GeoJSON longitude/latitude order",
            normalizations=(),
        )
    )


def _parse_mapping_position(
    value: Mapping[object, object],
    *,
    selected_format: CoordinateFormat | None,
) -> _CandidateSelection:
    if "type" in value or "coordinates" in value:
        return _parse_geojson_point(value, selected_format=selected_format)
    return _parse_named_mapping(value, selected_format=selected_format)


def _parse_sequence_position(
    value: Sequence[object],
    *,
    order: CoordinateOrder,
    selected_format: CoordinateFormat | None,
) -> _CandidateSelection:
    first_raw, second_raw = _sequence_pair(value)
    formats = _candidate_formats(selected_format)
    candidates: list[_ResolvedCandidate] = []
    first_candidates, first_errors = _structured_component_candidates(
        first_raw,
        formats,
    )
    second_candidates, second_errors = _structured_component_candidates(
        second_raw,
        formats,
    )
    errors = first_errors + second_errors
    for first in first_candidates:
        for second in second_candidates:
            try:
                candidates.append(_resolve_components(first, second, order))
            except CoordinateError as error:
                errors.append(error)

    if not candidates:
        _raise_best_error(errors, selected_format=selected_format)
    return _deduplicate_candidates(candidates)


def _position_selection(
    value: Position | str | Mapping[object, object] | Sequence[object],
    *,
    order: CoordinateOrder,
    selected_format: CoordinateFormat | None,
) -> _CandidateSelection:
    if isinstance(value, Position):
        if selected_format not in {None, "dd"}:
            raise CoordinateParseError(
                f"Position input does not match the requested "
                f"{selected_format} format"
            )
        exact_latitude = _exact_from_real(value.latitude)
        exact_longitude = _exact_from_real(value.longitude)
        return _CandidateSelection(
            _ResolvedCandidate(
                position=value,
                exact_latitude=exact_latitude,
                exact_longitude=exact_longitude,
                format="dd",
                component_formats=("dd", "dd"),
                source_order="latlon",
                order_evidence="Position fields",
                precision=(None, None),
                resolution_degrees=(None, None),
            )
        )
    if isinstance(value, str):
        return _parse_text_position(
            value,
            order=order,
            selected_format=selected_format,
        )
    if isinstance(value, Mapping):
        return _parse_mapping_position(
            value,
            selected_format=selected_format,
        )
    return _parse_sequence_position(
        value,
        order=order,
        selected_format=selected_format,
    )


def _diagnostic_tokens(
    text: str | None,
    normalizations: tuple[str, ...],
) -> tuple[str, ...]:
    if text is None:
        return ()
    tokens = re.findall(
        r"[+-]?[0-9]+(?:[.,][0-9]+)?(?:[eE][+-]?[0-9]+)?|"
        r"(?<![0-9A-Za-z])[+-]?[.,][0-9]+(?:[eE][+-]?[0-9]+)?|"
        r"[NSEW]|[°'\"]|[;,/:]|[^\s;,/:°'\"]+",
        text,
        flags=re.IGNORECASE | re.ASCII,
    )
    if "normalized decimal comma" in normalizations:
        tokens = [
            token.replace(",", ".")
            if token != "," and any(character.isdigit() for character in token)
            else token
            for token in tokens
        ]
    if "normalized hemisphere letter case" in normalizations:
        tokens = [
            token.upper() if token in "nNsSeEwW" else token
            for token in tokens
        ]
    return tuple(tokens)


def _candidate_diagnostic(
    candidate: _ResolvedCandidate,
    *,
    outcome: CandidateOutcome,
) -> CandidateDiagnostic:
    return CandidateDiagnostic(
        format=candidate.format,
        source_order=candidate.source_order,
        position=candidate.position,
        outcome=outcome,
        evidence=(
            candidate.order_evidence,
            "latitude component: " + candidate.component_formats[0],
            "longitude component: " + candidate.component_formats[1],
        ),
    )


def _parse_result(
    value: Position | str | Mapping[object, object] | Sequence[object],
    *,
    order: CoordinateOrder,
    selected_format: CoordinateFormat | None,
    format_alias_used: bool,
) -> ParseResult:
    selection = _position_selection(
        value,
        order=order,
        selected_format=selected_format,
    )
    selected = selection.selected
    diagnostics = [_candidate_diagnostic(selected, outcome="selected")]
    diagnostics.extend(
        _candidate_diagnostic(candidate, outcome="equivalent")
        for candidate in selection.equivalents
    )
    if (
        selected.order_evidence == "numeric range"
        and selected.source_order is not None
    ):
        rejected_order: OutputOrder = (
            "lonlat" if selected.source_order == "latlon" else "latlon"
        )
        invalid_latitude = selected.position.longitude
        diagnostics.append(
            CandidateDiagnostic(
                format=selected.format,
                source_order=rejected_order,
                position=None,
                outcome="rejected",
                evidence=("candidate coordinate order",),
                reason=(
                    f"latitude {invalid_latitude:g} is outside "
                    "[-90, 90]"
                ),
            )
        )

    if (
        isinstance(value, str)
        and selected_format is None
    ):
        accepted_formats = {
            candidate.format
            for candidate in (selected, *selection.equivalents)
        }
        for format_name in ("dd", "ddm", "dms", "iso6709", "nmea"):
            if format_name not in accepted_formats:
                if selected.format in {"iso6709", "nmea"}:
                    reason = (
                        f"{selected.format} field syntax defines the "
                        "whole-position interpretation"
                    )
                else:
                    reason = (
                        "did not yield a valid whole-position interpretation"
                    )
                diagnostics.append(
                    CandidateDiagnostic(
                        format=format_name,
                        source_order=None,
                        position=None,
                        outcome="rejected",
                        reason=reason,
                    )
                )

    normalizations = list(selected.normalizations)
    if format_alias_used:
        normalizations.append("canonicalized format alias 'dmm' to 'ddm'")
    latitude_resolution, longitude_resolution = selected.resolution_degrees
    evidence = (
        f"selected {selected.format} interpretation",
        selected.order_evidence,
    )
    return ParseResult(
        position=selected.position,
        format=selected.format,
        component_formats=selected.component_formats,
        source_order=selected.source_order,
        evidence=evidence,
        original_text=value if isinstance(value, str) else None,
        normalized_tokens=_diagnostic_tokens(
            selection.normalized_text,
            tuple(dict.fromkeys(normalizations)),
        ),
        normalizations=tuple(dict.fromkeys(normalizations)),
        warnings=(),
        latitude_resolution=latitude_resolution,
        longitude_resolution=longitude_resolution,
        candidates=tuple(diagnostics),
    )


def inspect_position(
    value: PositionInput,
    *,
    order: CoordinateOrder = "latlon",
    format: str | None = None,
) -> ParseResult:
    """Parse a position and return detection and normalization diagnostics."""

    selected_order = _validate_parse_order(order)
    selected_format = _validate_input_format(format)
    alias_used = isinstance(format, str) and format.casefold() == "dmm"
    return _parse_result(
        value,
        order=selected_order,
        selected_format=selected_format,
        format_alias_used=alias_used,
    )


def parse_position(
    value: PositionInput,
    *,
    order: CoordinateOrder = "latlon",
    format: str | None = None,
) -> Position:
    """Parse supported text, sequences, mappings, and GeoJSON Points.

    Text detection covers DD, DDM, DMS, two-dimensional ISO 6709, and NMEA
    coordinate fields. Axis markers and format-defined axes override textual
    order. Without axis evidence, auto order uses exact range evidence or
    accepts interpretations that produce the same position; materially
    different candidates raise AmbiguousCoordinateError.
    """

    selected_order = _validate_parse_order(order)
    selected_format = _validate_input_format(format)
    return _position_selection(
        value,
        order=selected_order,
        selected_format=selected_format,
    ).selected.position


_DEFAULT_OUTPUT_PRECISION = {
    "dd": 6,
    "ddm": 4,
    "dms": 2,
    "iso6709": 6,
    "nmea": 4,
}


def _validate_output_format(value: str) -> CoordinateFormat:
    if not isinstance(value, str):
        raise CoordinateParseError(
            'to must be "dd", "ddm", "dms", "iso6709", or "nmea"'
        )
    normalized = value.casefold()
    if normalized == "dmm":
        normalized = "ddm"
    if normalized not in _DEFAULT_OUTPUT_PRECISION:
        raise CoordinateParseError(
            'to must be "dd", "ddm", "dms", "iso6709", or "nmea"'
        )
    return cast(CoordinateFormat, normalized)


def _validate_precision(
    precision: int | None,
    *,
    output_format: CoordinateFormat,
) -> int:
    if precision is None:
        return _DEFAULT_OUTPUT_PRECISION[output_format]
    if (
        isinstance(precision, bool)
        or not isinstance(precision, int)
        or not 0 <= precision <= 15
    ):
        raise CoordinateParseError(
            "precision must be an integer from 0 through 15"
        )
    if output_format == "nmea" and precision == 0:
        raise CoordinateParseError(
            "NMEA precision must be an integer from 1 through 15"
        )
    return precision


def _validate_notation(
    notation: str | None,
    *,
    output_format: CoordinateFormat,
) -> Literal["signed", "hemisphere"] | None:
    if output_format not in {"dd", "ddm", "dms"}:
        if notation is not None:
            raise CoordinateParseError(
                "notation applies only to DD, DDM, and DMS output"
            )
        return None
    if notation is None:
        return "signed" if output_format == "dd" else "hemisphere"
    if not isinstance(notation, str) or notation not in {
        "signed",
        "hemisphere",
    }:
        raise CoordinateParseError(
            'notation must be "signed" or "hemisphere"'
        )
    return cast(Literal["signed", "hemisphere"], notation)


def _validate_symbols(
    symbols: str | None,
    *,
    output_format: CoordinateFormat,
) -> Literal["unicode", "ascii"]:
    if output_format not in {"dd", "ddm", "dms"}:
        if symbols is not None:
            raise CoordinateParseError(
                "symbols applies only to DD, DDM, and DMS output"
            )
        return "unicode"
    if symbols is None:
        return "unicode"
    if not isinstance(symbols, str) or symbols not in {"unicode", "ascii"}:
        raise CoordinateParseError(
            'symbols must be "unicode" or "ascii"'
        )
    return cast(Literal["unicode", "ascii"], symbols)


def _validate_compact(
    compact: bool | None,
    *,
    output_format: CoordinateFormat,
) -> bool | None:
    if output_format != "iso6709":
        if compact is not None:
            raise CoordinateParseError(
                "compact applies only to ISO 6709 output"
            )
        return None
    if compact is None:
        return True
    if not isinstance(compact, bool):
        raise CoordinateParseError("compact must be a boolean")
    return compact


def _validate_separator(
    separator: str | None,
    *,
    output_format: CoordinateFormat,
    notation: Literal["signed", "hemisphere"] | None,
    compact: bool | None,
) -> str:
    if separator is not None and not isinstance(separator, str):
        raise CoordinateParseError("separator must be a string")
    if output_format == "iso6709":
        if compact:
            if separator is not None:
                raise CoordinateParseError(
                    "compact ISO 6709 output cannot use a separator"
                )
            return ""
        if separator is None:
            return " "
        if separator not in {" ", ",", ", "}:
            raise CoordinateParseError(
                'separated ISO 6709 output requires " ", ",", or ", "'
            )
        return separator

    if output_format == "nmea":
        if separator is None:
            return ","
        if separator not in {",", "; "}:
            raise CoordinateParseError(
                'NMEA separator must be "," or "; "'
            )
        return separator

    default = "; " if notation == "hemisphere" else ", "
    if separator is None:
        return default
    if separator not in {", ", "; ", " / "}:
        raise CoordinateParseError(
            'human-format separator must be ", ", "; ", or " / "'
        )
    return separator


def _rounded_magnitude(
    value: float,
    *,
    factor: int,
    precision: int,
) -> tuple[Decimal, bool]:
    with localcontext(_FORMAT_DECIMAL_CONTEXT):
        quantum = Decimal((0, (1,), -precision))
        rounded = (Decimal(str(abs(value))) * factor).quantize(
            quantum,
            rounding=ROUND_HALF_EVEN,
        )
    negative = value < 0.0 and not rounded.is_zero()
    return rounded, negative


def _fixed(value: Decimal, precision: int) -> str:
    return f"{value:.{precision}f}"


def _direction(axis: Axis, negative: bool) -> str:
    if axis == "lat":
        return "S" if negative else "N"
    return "W" if negative else "E"


def _ordered_components(
    position: Position,
    order: OutputOrder,
) -> tuple[tuple[float, Axis], tuple[float, Axis]]:
    latitude = (position.latitude, cast(Axis, "lat"))
    longitude = (position.longitude, cast(Axis, "lon"))
    return (latitude, longitude) if order == "latlon" else (longitude, latitude)


def _format_dd_component(
    value: float,
    *,
    axis: Axis,
    precision: int,
    notation: Literal["signed", "hemisphere"],
    symbols: Literal["unicode", "ascii"],
) -> str:
    magnitude, negative = _rounded_magnitude(
        value,
        factor=1,
        precision=precision,
    )
    number = _fixed(magnitude, precision)
    if notation == "signed":
        return ("-" if negative else "") + number
    suffix = "°" if symbols == "unicode" else " deg"
    return f"{number}{suffix} {_direction(axis, negative)}"


def _degrees_minutes(
    value: float,
    *,
    precision: int,
) -> tuple[int, Decimal, bool]:
    with localcontext(_FORMAT_DECIMAL_CONTEXT):
        total_minutes, negative = _rounded_magnitude(
            value,
            factor=60,
            precision=precision,
        )
        degrees = int(total_minutes // 60)
        minutes = total_minutes - Decimal(degrees * 60)
    return degrees, minutes, negative


def _format_ddm_component(
    value: float,
    *,
    axis: Axis,
    precision: int,
    notation: Literal["signed", "hemisphere"],
    symbols: Literal["unicode", "ascii"],
) -> str:
    degrees, minutes, negative = _degrees_minutes(
        value,
        precision=precision,
    )
    sign = "-" if notation == "signed" and negative else ""
    minute_text = _fixed(minutes, precision)
    if symbols == "unicode":
        text = f"{sign}{degrees}° {minute_text}′"
    else:
        text = f"{sign}{degrees} deg {minute_text} min"
    if notation == "hemisphere":
        text += f" {_direction(axis, negative)}"
    return text


def _degrees_minutes_seconds(
    value: float,
    *,
    precision: int,
) -> tuple[int, int, Decimal, bool]:
    with localcontext(_FORMAT_DECIMAL_CONTEXT):
        total_seconds, negative = _rounded_magnitude(
            value,
            factor=3600,
            precision=precision,
        )
        degrees = int(total_seconds // 3600)
        remainder = total_seconds - Decimal(degrees * 3600)
        minutes = int(remainder // 60)
        seconds = remainder - Decimal(minutes * 60)
    return degrees, minutes, seconds, negative


def _format_dms_component(
    value: float,
    *,
    axis: Axis,
    precision: int,
    notation: Literal["signed", "hemisphere"],
    symbols: Literal["unicode", "ascii"],
) -> str:
    degrees, minutes, seconds, negative = _degrees_minutes_seconds(
        value,
        precision=precision,
    )
    sign = "-" if notation == "signed" and negative else ""
    second_text = _fixed(seconds, precision)
    if symbols == "unicode":
        text = f"{sign}{degrees}° {minutes}′ {second_text}″"
    else:
        text = (
            f"{sign}{degrees} deg {minutes} min {second_text} sec"
        )
    if notation == "hemisphere":
        text += f" {_direction(axis, negative)}"
    return text


def _format_iso_component(
    value: float,
    *,
    axis: Axis,
    precision: int,
) -> str:
    magnitude, negative = _rounded_magnitude(
        value,
        factor=1,
        precision=precision,
    )
    degree_width = 2 if axis == "lat" else 3
    total_width = degree_width + (precision + 1 if precision else 0)
    number = f"{magnitude:0{total_width}.{precision}f}"
    return ("-" if negative else "+") + number


def _format_nmea_component(
    value: float,
    *,
    axis: Axis,
    precision: int,
) -> tuple[str, str]:
    degrees, minutes, negative = _degrees_minutes(
        value,
        precision=precision,
    )
    degree_width = 2 if axis == "lat" else 3
    minute_width = 2 + 1 + precision
    field = (
        f"{degrees:0{degree_width}d}"
        f"{minutes:0{minute_width}.{precision}f}"
    )
    return field, _direction(axis, negative)


def format_position(
    position: Position,
    *,
    to: str = "dd",
    order: OutputOrder = "latlon",
    precision: int | None = None,
    notation: str | None = None,
    symbols: str | None = None,
    compact: bool | None = None,
    separator: str | None = None,
) -> str:
    """Format a Position as canonical DD, DDM, DMS, ISO, or NMEA text."""

    if not isinstance(position, Position):
        raise CoordinateParseError("position must be a Position instance")
    output_format = _validate_output_format(to)
    selected_order = _validate_output_order(order)
    if output_format == "iso6709" and selected_order != "latlon":
        raise CoordinateParseError(
            "ISO 6709 output requires latitude/longitude order"
        )
    selected_precision = _validate_precision(
        precision,
        output_format=output_format,
    )
    selected_notation = _validate_notation(
        notation,
        output_format=output_format,
    )
    selected_symbols = _validate_symbols(
        symbols,
        output_format=output_format,
    )
    selected_compact = _validate_compact(
        compact,
        output_format=output_format,
    )
    selected_separator = _validate_separator(
        separator,
        output_format=output_format,
        notation=selected_notation,
        compact=selected_compact,
    )
    components = _ordered_components(position, selected_order)

    if output_format in {"dd", "ddm", "dms"}:
        if selected_notation is None:
            raise CoordinateParseError("human output requires a notation")
        formatter = {
            "dd": _format_dd_component,
            "ddm": _format_ddm_component,
            "dms": _format_dms_component,
        }[output_format]
        return selected_separator.join(
            formatter(
                value,
                axis=axis,
                precision=selected_precision,
                notation=selected_notation,
                symbols=selected_symbols,
            )
            for value, axis in components
        )

    if output_format == "iso6709":
        latitude = _format_iso_component(
            position.latitude,
            axis="lat",
            precision=selected_precision,
        )
        longitude = _format_iso_component(
            position.longitude,
            axis="lon",
            precision=selected_precision,
        )
        return latitude + selected_separator + longitude + "/"

    nmea_components = [
        _format_nmea_component(
            value,
            axis=axis,
            precision=selected_precision,
        )
        for value, axis in components
    ]
    if selected_separator == ",":
        return ",".join(
            item
            for component in nmea_components
            for item in component
        )
    return selected_separator.join(
        f"{field} {direction}" for field, direction in nmea_components
    )


def _inferred_output_precision(
    resolution_degrees: tuple[
        _ExactNumber | None,
        _ExactNumber | None,
    ],
    *,
    output_format: CoordinateFormat,
) -> int:
    resolutions = tuple(
        Fraction(resolution)
        for resolution in resolution_degrees
        if resolution is not None and resolution > 0
    )
    if not resolutions:
        return _DEFAULT_OUTPUT_PRECISION[output_format]
    factor = {
        "dd": 1,
        "ddm": 60,
        "dms": 3600,
        "iso6709": 1,
        "nmea": 60,
    }[output_format]
    source_quantum = min(resolutions) * factor
    minimum = 1 if output_format == "nmea" else 0
    for candidate in range(minimum, 16):
        if Fraction(1, 10**candidate) <= source_quantum:
            return candidate
    raise CoordinateParseError(
        "source resolution requires more than the supported 15 output "
        "decimal places; pass an explicit precision to accept display "
        "rounding"
    )


def convert_position(
    value: PositionInput,
    *,
    to: str = "dd",
    order: CoordinateOrder = "latlon",
    output_order: OutputOrder = "latlon",
    format: str | None = None,
    precision: int | None = None,
    notation: str | None = None,
    symbols: str | None = None,
    compact: bool | None = None,
    separator: str | None = None,
) -> str:
    """Parse and format a position, preserving known source resolution."""

    output_format = _validate_output_format(to)
    selected_order = _validate_parse_order(order)
    selected_format = _validate_input_format(format)
    selection = _position_selection(
        value,
        order=selected_order,
        selected_format=selected_format,
    )
    selected_precision = (
        _inferred_output_precision(
            selection.selected.resolution_degrees,
            output_format=output_format,
        )
        if precision is None
        else precision
    )
    return format_position(
        selection.selected.position,
        to=output_format,
        order=output_order,
        precision=selected_precision,
        notation=notation,
        symbols=symbols,
        compact=compact,
        separator=separator,
    )
