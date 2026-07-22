"""Strict parsers for machine-readable coordinate-pair formats."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from typing import Literal, TypeAlias

from .errors import CoordinateParseError, CoordinateRangeError

MachineFormat: TypeAlias = Literal["iso6709", "nmea"]
MachineSourceOrder: TypeAlias = Literal["latlon", "lonlat"]
ExactNumber: TypeAlias = Decimal | Fraction


@dataclass(frozen=True, slots=True)
class MachinePosition:
    """An exactly parsed, axis-resolved machine-format position."""

    latitude: ExactNumber
    longitude: ExactNumber
    format: MachineFormat
    source_order: MachineSourceOrder
    precision: tuple[int, int]
    resolution_degrees: tuple[ExactNumber, ExactNumber]


_ISO_FORMS = (
    ("dd", 2, 3),
    ("ddm", 4, 5),
    ("dms", 6, 7),
)
_NMEA_DECIMAL = re.compile(r"[0-9]+\.[0-9]+")


def _decimal(value: str, *, component: str) -> Decimal:
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise CoordinateParseError(f"invalid {component}: {value!r}") from error
    if not number.is_finite():
        raise CoordinateRangeError(f"{component} must be finite")
    return number


def _validate_pair(
    latitude: ExactNumber,
    longitude: ExactNumber,
    *,
    format_name: MachineFormat,
    source_order: MachineSourceOrder,
    precision: tuple[int, int],
    resolution_degrees: tuple[ExactNumber, ExactNumber],
) -> MachinePosition:
    latitude_limit = Decimal(90)
    longitude_limit = Decimal(180)
    if not latitude_limit.copy_negate() <= latitude <= latitude_limit:
        raise CoordinateRangeError(
            "latitude must be between -90 and 90 degrees"
        )
    if not longitude_limit.copy_negate() <= longitude <= longitude_limit:
        raise CoordinateRangeError(
            "longitude must be between -180 and 180 degrees"
        )
    return MachinePosition(
        latitude,
        longitude,
        format_name,
        source_order,
        precision,
        resolution_degrees,
    )


def _lexical_resolution(
    fraction: str,
    *,
    unit_divisor: int,
) -> ExactNumber:
    unit = Decimal((0, (1,), -len(fraction)))
    return Fraction(unit) / unit_divisor


def _iso_component(
    token: str,
    *,
    degree_width: int,
    form: str,
    axis: str,
) -> tuple[ExactNumber, int, ExactNumber]:
    sign = -1 if token[0] == "-" else 1
    body = token[1:]
    integer, dot, fraction = body.partition(".")
    if dot and len(fraction) > 323:
        raise CoordinateParseError(
            f"ISO {axis} precision is outside the supported float range"
        )
    if form == "dd":
        magnitude: ExactNumber = _decimal(body, component=f"ISO {axis}")
    elif form == "ddm":
        degrees = Decimal(integer[:degree_width])
        minute_text = integer[degree_width:] + (
            f".{fraction}" if dot else ""
        )
        minutes = _decimal(minute_text, component=f"ISO {axis} minutes")
        if not Decimal(0) <= minutes < Decimal(60):
            raise CoordinateRangeError(
                f"ISO {axis} minutes must be in the range [0, 60)"
            )
        magnitude = Fraction(degrees) + Fraction(minutes) / 60
    else:
        degrees = Decimal(integer[:degree_width])
        minutes = Decimal(integer[degree_width : degree_width + 2])
        second_text = integer[degree_width + 2 :] + (
            f".{fraction}" if dot else ""
        )
        seconds = _decimal(second_text, component=f"ISO {axis} seconds")
        if not Decimal(0) <= minutes < Decimal(60):
            raise CoordinateRangeError(
                f"ISO {axis} minutes must be in the range [0, 60)"
            )
        if not Decimal(0) <= seconds < Decimal(60):
            raise CoordinateRangeError(
                f"ISO {axis} seconds must be in the range [0, 60)"
            )
        magnitude = (
            Fraction(degrees)
            + Fraction(minutes) / 60
            + Fraction(seconds) / 3600
        )
    precision = len(fraction) if dot else 0
    unit_divisor = 1 if form == "dd" else 60 if form == "ddm" else 3600
    resolution = _lexical_resolution(
        fraction if dot else "",
        unit_divisor=unit_divisor,
    )
    value = (
        magnitude.copy_negate()
        if sign < 0 and isinstance(magnitude, Decimal)
        else -magnitude
        if sign < 0
        else magnitude
    )
    return value, precision, resolution


def _looks_like_iso_altitude(text: str) -> bool:
    signed_numbers = re.findall(
        r"(?<![eE])[+-][0-9]+(?:\.[0-9]+)?",
        text,
    )
    return text.startswith(("+", "-")) and len(signed_numbers) >= 3


def _has_compact_iso_intent(text: str) -> bool:
    return re.match(r"^[+-][0-9]+(?:\.[0-9]+)?[+-][0-9]", text) is not None


def parse_iso6709(
    text: str,
    *,
    required: bool = False,
) -> MachinePosition | None:
    """Parse NautiPy's strict two-dimensional ISO 6709 subset.

    The supported DD, DDM, and DMS forms use fixed-width latitude and
    longitude fields, mandatory signs, an optional space/comma separator, and
    an optional terminal slash.
    """

    for form, latitude_digits, longitude_digits in _ISO_FORMS:
        pattern = re.compile(
            rf"(?P<latitude>[+-][0-9]{{{latitude_digits}}}"
            rf"(?:\.[0-9]+)?)"
            rf"(?:(?: +)|(?: *, *))?"
            rf"(?P<longitude>[+-][0-9]{{{longitude_digits}}}"
            rf"(?:\.[0-9]+)?)/?"
        )
        match = pattern.fullmatch(text)
        if match is None:
            continue
        latitude, latitude_precision, latitude_resolution = _iso_component(
            match.group("latitude"),
            degree_width=2,
            form=form,
            axis="latitude",
        )
        longitude, longitude_precision, longitude_resolution = _iso_component(
            match.group("longitude"),
            degree_width=3,
            form=form,
            axis="longitude",
        )
        return _validate_pair(
            latitude,
            longitude,
            format_name="iso6709",
            source_order="latlon",
            precision=(latitude_precision, longitude_precision),
            resolution_degrees=(latitude_resolution, longitude_resolution),
        )

    if _looks_like_iso_altitude(text):
        raise CoordinateParseError(
            "ISO 6709 altitude or an extra signed field is not supported"
        )
    if required or text.endswith("/") or _has_compact_iso_intent(text):
        raise CoordinateParseError(
            "input does not match the supported two-dimensional ISO 6709 "
            "DD, DDM, or DMS syntax"
        )
    return None


def _nmea_component(
    field: str,
    direction: str,
    *,
    axis: Literal["latitude", "longitude"],
) -> tuple[ExactNumber, int, ExactNumber]:
    degree_width = 2 if axis == "latitude" else 3
    expected_digits = degree_width + 2
    if _NMEA_DECIMAL.fullmatch(field) is None:
        raise CoordinateParseError(
            f"NMEA {axis} must contain exactly {expected_digits} digits "
            "before a decimal point followed by at least one digit"
        )
    integer, fraction = field.split(".", 1)
    if len(fraction) > 323:
        raise CoordinateParseError(
            f"NMEA {axis} precision is outside the supported float range"
        )
    if len(integer) != expected_digits:
        raise CoordinateParseError(
            f"NMEA {axis} must have exactly {expected_digits} digits before "
            "the decimal point"
        )
    degrees = Decimal(integer[:degree_width])
    minutes = _decimal(
        f"{integer[degree_width:]}.{fraction}",
        component=f"NMEA {axis} minutes",
    )
    if not Decimal(0) <= minutes < Decimal(60):
        raise CoordinateRangeError(
            f"NMEA {axis} minutes must be in the range [0, 60)"
        )
    magnitude: ExactNumber = Fraction(degrees) + Fraction(minutes) / 60
    value = -magnitude if direction in {"S", "W"} else magnitude
    precision = len(fraction)
    resolution = _lexical_resolution(fraction, unit_divisor=60)
    return value, precision, resolution


def _nmea_position(
    first_field: str,
    first_direction: str,
    second_field: str,
    second_direction: str,
) -> MachinePosition:
    if not _is_ascii_direction(first_direction) or not _is_ascii_direction(
        second_direction
    ):
        raise CoordinateParseError(
            "NMEA directions must use ASCII N, S, E, or W letters"
        )
    first_direction = first_direction.upper()
    second_direction = second_direction.upper()
    first_axis = "latitude" if first_direction in {"N", "S"} else "longitude"
    second_axis = (
        "latitude" if second_direction in {"N", "S"} else "longitude"
    )
    if first_axis == second_axis:
        raise CoordinateParseError(
            "NMEA fields must contain one N/S latitude and one E/W longitude"
        )
    first_value, first_precision, first_resolution = _nmea_component(
        first_field,
        first_direction,
        axis=first_axis,
    )
    second_value, second_precision, second_resolution = _nmea_component(
        second_field,
        second_direction,
        axis=second_axis,
    )
    latitude, longitude, precision, resolution = (
        (
            first_value,
            second_value,
            (first_precision, second_precision),
            (first_resolution, second_resolution),
        )
        if first_axis == "latitude"
        else (
            second_value,
            first_value,
            (second_precision, first_precision),
            (second_resolution, first_resolution),
        )
    )
    return _validate_pair(
        latitude,
        longitude,
        format_name="nmea",
        source_order=("latlon" if first_axis == "latitude" else "lonlat"),
        precision=precision,
        resolution_degrees=resolution,
    )


def _is_ascii_direction(value: str) -> bool:
    return len(value) == 1 and value in "NnSsEeWw"


def _attached_nmea_component(text: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"(?P<field>\S+) +(?P<direction>[NnSsEeWw])",
        text,
    )
    if match is None:
        return None
    return match.group("field"), match.group("direction")


def _is_exact_nmea_field(field: str, direction: str) -> bool:
    if not _is_ascii_direction(direction):
        return False
    digits = 4 if direction.upper() in {"N", "S"} else 5
    match = _NMEA_DECIMAL.fullmatch(field)
    return match is not None and len(field.split(".", 1)[0]) == digits


def parse_nmea(
    text: str,
    *,
    required: bool = False,
) -> MachinePosition | None:
    """Parse a strict NMEA latitude/direction and longitude/direction pair."""

    if text.startswith("$"):
        raise CoordinateParseError(
            "full NMEA sentences are not supported; pass only the coordinate "
            "and direction fields"
        )

    comma_fields = [field.strip() for field in text.split(",")]
    comma_signature = (
        len(comma_fields) >= 2
        and _is_ascii_direction(comma_fields[1])
    )
    if len(comma_fields) == 4 and (
        _is_ascii_direction(comma_fields[1])
        and _is_ascii_direction(comma_fields[3])
    ):
        return _nmea_position(
            comma_fields[0],
            comma_fields[1],
            comma_fields[2],
            comma_fields[3],
        )
    if comma_signature:
        raise CoordinateParseError(
            "NMEA comma input must contain exactly four fields: "
            "coordinate, direction, coordinate, direction, with one N/S "
            "latitude and one E/W longitude"
        )

    if text.count(";") == 1:
        first_text, second_text = (part.strip() for part in text.split(";"))
        first = _attached_nmea_component(first_text)
        second = _attached_nmea_component(second_text)
        if first is not None and second is not None:
            exact_signature = (
                _is_exact_nmea_field(first[0], first[1])
                and _is_exact_nmea_field(second[0], second[1])
                and (
                    (first[1].upper() in {"N", "S"})
                    != (second[1].upper() in {"N", "S"})
                )
            )
            if required or exact_signature:
                return _nmea_position(first[0], first[1], second[0], second[1])

    if required:
        raise CoordinateParseError(
            "input does not match NMEA latitude/direction and "
            "longitude/direction field syntax"
        )
    return None
