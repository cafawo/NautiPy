"""Decimal-degree parsing and formatting."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from typing import Literal, TypeAlias, cast

from .errors import (
    AmbiguousCoordinateError,
    CoordinateParseError,
    CoordinateRangeError,
)
from .position import Position

CoordinateOrder: TypeAlias = Literal["latlon", "lonlat", "auto"]
OutputOrder: TypeAlias = Literal["latlon", "lonlat"]


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


def _text_pair(value: str) -> tuple[str, str]:
    text = value.strip()
    if not text:
        raise CoordinateParseError("position text cannot be empty")

    if text.count(",") == 1:
        parts = [part.strip() for part in text.split(",")]
    elif "," in text:
        raise CoordinateParseError(
            "decimal-degree position text must contain exactly two values"
        )
    else:
        parts = text.split()

    if len(parts) != 2 or any(not part for part in parts):
        raise CoordinateParseError(
            "decimal-degree position text must contain exactly two values"
        )
    return parts[0], parts[1]


def _sequence_pair(value: object) -> tuple[object, object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise CoordinateParseError(
            "position must be decimal-degree text or a two-value sequence"
        )
    if len(value) != 2:
        raise CoordinateParseError("position sequence must contain exactly two values")
    return value[0], value[1]


def _decimal_value(value: object) -> float:
    if isinstance(value, bool):
        raise CoordinateParseError("coordinate values must be real numbers")

    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError as error:
            raise CoordinateParseError(
                f"invalid decimal-degree value: {value!r}"
            ) from error
    elif isinstance(value, Real):
        try:
            number = float(value)
        except (OverflowError, ValueError) as error:
            raise CoordinateRangeError(
                "coordinate values must be finite"
            ) from error
    else:
        raise CoordinateParseError("coordinate values must be real numbers")

    if not isfinite(number):
        raise CoordinateRangeError("coordinate values must be finite")
    return number


def _position_from_auto(first: float, second: float) -> Position:
    latlon_valid = -90.0 <= first <= 90.0 and -180.0 <= second <= 180.0
    lonlat_valid = -180.0 <= first <= 180.0 and -90.0 <= second <= 90.0

    if latlon_valid and lonlat_valid:
        if first == second:
            return Position(first, second)
        raise AmbiguousCoordinateError(
            f"could not determine coordinate order for {first!r}, {second!r}; "
            'pass order="latlon" or order="lonlat"'
        )
    if latlon_valid:
        return Position(first, second)
    if lonlat_valid:
        return Position(second, first)
    raise CoordinateRangeError(
        f"no valid latitude/longitude order for {first!r}, {second!r}"
    )


def parse_position(
    value: Position | str | Sequence[object],
    *,
    order: CoordinateOrder = "latlon",
) -> Position:
    """Parse a decimal-degree position without guessing coordinate order.

    ``order="auto"`` succeeds when range evidence proves one order or when
    both orders produce the same position. Materially different valid
    interpretations raise ``AmbiguousCoordinateError``.
    """

    selected_order = _validate_parse_order(order)
    if isinstance(value, Position):
        return value

    if isinstance(value, str):
        first_raw, second_raw = _text_pair(value)
    else:
        first_raw, second_raw = _sequence_pair(value)

    first = _decimal_value(first_raw)
    second = _decimal_value(second_raw)

    if selected_order == "latlon":
        return Position(first, second)
    if selected_order == "lonlat":
        return Position(second, first)
    return _position_from_auto(first, second)


def _format_component(value: float, precision: int | None) -> str:
    if value == 0.0:
        value = 0.0
    if precision is None:
        return repr(value)

    rounded = round(value, precision)
    if rounded == 0.0:
        rounded = 0.0
    return f"{rounded:.{precision}f}"


def format_position(
    position: Position,
    *,
    order: OutputOrder = "latlon",
    precision: int | None = None,
) -> str:
    """Format a ``Position`` as a canonical decimal-degree pair.

    Explicit precision is limited to the 15 meaningful decimal digits carried
    by the internal binary ``float`` representation.
    """

    if not isinstance(position, Position):
        raise CoordinateParseError("position must be a Position instance")
    selected_order = _validate_output_order(order)
    if precision is not None and (
        isinstance(precision, bool)
        or not isinstance(precision, int)
        or not 0 <= precision <= 15
    ):
        raise CoordinateParseError("precision must be an integer from 0 through 15")

    components = (position.latitude, position.longitude)
    if selected_order == "lonlat":
        components = (position.longitude, position.latitude)
    return ", ".join(_format_component(value, precision) for value in components)
