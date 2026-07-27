"""WGS84 navigation primitives backed by GeographicLib."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from math import isfinite
from numbers import Rational, Real
import typing as _typing

from .coordinates import PositionInput, parse_position
from .errors import NavigationError
from .position import Position

if _typing.TYPE_CHECKING:
    from geographiclib.geodesic import Geodesic

_MIN_OUTPUT_DISTANCE_METRES = 1e-7
_MIN_OUTPUT_DISTANCE_EXACT = Fraction(1, 10_000_000)

__all__ = [
    "InverseResult",
    "destination",
    "distance",
    "initial_bearing",
    "interpolate",
    "inverse",
    "nearest_position",
]


@dataclass(frozen=True, slots=True)
class InverseResult:
    """A shortest-path WGS84 inverse result.

    Distance is in metres. Bearings are true degrees clockwise from north and
    normalized to ``[0, 360)``. Bearings are ``None`` for coincident physical
    positions because no direction is defined there.
    """

    distance: float
    initial_bearing: float | None
    final_bearing: float | None


@lru_cache(maxsize=1)
def _wgs84() -> Geodesic:
    # Keep the coordinate layer importable without loading GeographicLib.
    from geographiclib.geodesic import Geodesic

    return Geodesic.WGS84


def _coerce_position(value: PositionInput) -> Position:
    return value if isinstance(value, Position) else parse_position(value)


def _navigation_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise NavigationError(f"{name} must be a real number")
    if isinstance(value, Decimal) and not value.is_finite():
        raise NavigationError(f"{name} must be finite")
    if isinstance(value, float) and not isfinite(value):
        raise NavigationError(f"{name} must be finite")

    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise NavigationError(
            f"{name} must be representable as a finite float"
        ) from error
    if not isfinite(number):
        raise NavigationError(
            f"{name} must be representable as a finite float"
        )
    if value != 0 and number == 0.0:
        raise NavigationError(
            f"{name} magnitude is too small for the internal float "
            "representation"
        )
    return number


def _bounded_navigation_number(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    number = _navigation_number(value, name=name)
    try:
        outside = value < minimum or (
            maximum is not None and value > maximum
        )
    except TypeError as error:
        raise NavigationError(f"{name} must be a real number") from error
    if outside:
        if maximum is None:
            raise NavigationError(f"{name} must be at least {minimum:g}")
        raise NavigationError(
            f"{name} must be between {minimum:g} and {maximum:g}"
        )
    if value != minimum and number == minimum:
        raise NavigationError(
            f"{name} is too close to {minimum:g} for the internal float "
            "representation"
        )
    if maximum is not None and value != maximum and number == maximum:
        raise NavigationError(
            f"{name} is too close to {maximum:g} for the internal float "
            "representation"
        )
    return number


def _normalized_bearing(value: float) -> float:
    normalized = value % 360.0
    return 0.0 if normalized in {0.0, 360.0} else normalized


def _digits_modulo_360(digits: tuple[int, ...]) -> int:
    remainder = 0
    for digit in digits:
        remainder = (remainder * 10 + digit) % 360
    return remainder


def _decimal_bearing(value: Decimal) -> float:
    if not value.is_finite():
        raise NavigationError("bearing must be finite")
    if value.is_zero():
        return 0.0

    sign, digits, exponent = value.as_tuple()
    if exponent >= 0:
        remainder = (
            _digits_modulo_360(digits) * pow(10, exponent, 360)
        ) % 360
        number = float(remainder)
    else:
        decimal_point = len(digits) + exponent
        if decimal_point <= 0:
            positive_remainder = value.copy_abs()
        else:
            whole_remainder = _digits_modulo_360(
                digits[:decimal_point]
            )
            whole_digits = tuple(int(character) for character in str(
                whole_remainder
            ))
            positive_remainder = Decimal(
                (0, whole_digits + digits[decimal_point:], exponent)
            )
        number = float(positive_remainder)
        if not positive_remainder.is_zero() and number == 0.0:
            return 0.0
        if sign and not positive_remainder.is_zero():
            number = float(Fraction(360) - Fraction(positive_remainder))
    return _normalized_bearing(-number if sign and exponent >= 0 else number)


def _exact_ratio_bearing(value: object) -> float | None:
    if isinstance(value, Rational):
        remainder = value % 360
    else:
        ratio_method = getattr(value, "as_integer_ratio", None)
        if not callable(ratio_method):
            return None
        try:
            numerator, denominator = ratio_method()
            remainder = Fraction(int(numerator), int(denominator)) % 360
        except (OverflowError, TypeError, ValueError, ZeroDivisionError):
            return None
    number = float(remainder)
    if remainder != 0 and number == 0.0:
        return 0.0
    return _normalized_bearing(number)


def _normalized_input_bearing(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise NavigationError("bearing must be a real number")
    if isinstance(value, Decimal):
        return _decimal_bearing(value)
    if isinstance(value, float) and not isfinite(value):
        raise NavigationError("bearing must be finite")
    exact = _exact_ratio_bearing(value)
    if exact is not None:
        return exact
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise NavigationError("bearing must be finite") from error
    if not isfinite(number):
        raise NavigationError("bearing must be finite")
    return _normalized_bearing(number)


def _physically_coincident(start: Position, end: Position) -> bool:
    if start.latitude != end.latitude:
        return False
    if abs(start.latitude) == 90.0:
        return True
    longitude_difference = abs(start.longitude - end.longitude)
    return longitude_difference in {0.0, 360.0}


def _below_output_distance_resolution(
    original: object,
    number: float,
) -> bool:
    if isinstance(original, (Decimal, Rational)):
        return original < _MIN_OUTPUT_DISTANCE_EXACT
    ratio_method = getattr(original, "as_integer_ratio", None)
    if callable(ratio_method) and not isinstance(original, float):
        try:
            numerator, denominator = ratio_method()
            return (
                Fraction(int(numerator), int(denominator))
                < _MIN_OUTPUT_DISTANCE_EXACT
            )
        except (OverflowError, TypeError, ValueError, ZeroDivisionError):
            pass
    return number < _MIN_OUTPUT_DISTANCE_METRES


def _backend_number(result: Mapping[str, object], key: str) -> float:
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise NavigationError(
            f"WGS84 calculation did not return a valid {key} value"
        )
    number = float(value)
    if not isfinite(number):
        raise NavigationError(
            f"WGS84 calculation did not return a finite {key} value"
        )
    return number


def _inverse_positions(start: Position, end: Position) -> InverseResult:
    if _physically_coincident(start, end):
        return InverseResult(0.0, None, None)
    raw = _wgs84().Inverse(
        start.latitude,
        start.longitude,
        end.latitude,
        end.longitude,
    )
    distance_metres = _backend_number(raw, "s12")
    if distance_metres < 0:
        raise NavigationError("WGS84 calculation returned a negative distance")
    if distance_metres == 0.0:
        raise NavigationError(
            "positions are distinct but below WGS84 numerical resolution"
        )
    return InverseResult(
        distance=distance_metres,
        initial_bearing=_normalized_bearing(
            _backend_number(raw, "azi1")
        ),
        final_bearing=_normalized_bearing(_backend_number(raw, "azi2")),
    )


def inverse(start: PositionInput, end: PositionInput) -> InverseResult:
    """Return the shortest-path WGS84 distance and endpoint bearings."""

    return _inverse_positions(
        _coerce_position(start),
        _coerce_position(end),
    )


def distance(start: PositionInput, end: PositionInput) -> float:
    """Return the shortest WGS84 geodesic distance in metres."""

    return inverse(start, end).distance


def initial_bearing(start: PositionInput, end: PositionInput) -> float:
    """Return the initial true bearing in degrees clockwise from north."""

    bearing = inverse(start, end).initial_bearing
    if bearing is None:
        raise NavigationError(
            "initial bearing is undefined for coincident positions"
        )
    return bearing


def destination(
    start: PositionInput,
    *,
    bearing: Real | Decimal,
    distance: Real | Decimal,
) -> Position:
    """Return the WGS84 destination for a true bearing and distance."""

    position = _coerce_position(start)
    bearing_degrees = _normalized_input_bearing(bearing)
    distance_metres = _bounded_navigation_number(
        distance,
        name="distance",
        minimum=0,
    )
    if distance_metres == 0.0:
        return position
    if _below_output_distance_resolution(distance, distance_metres):
        raise NavigationError(
            "distance must be zero or at least 1e-7 metres for a "
            "resolved destination"
        )

    raw = _wgs84().Direct(
        position.latitude,
        position.longitude,
        bearing_degrees,
        distance_metres,
    )
    return Position(
        _backend_number(raw, "lat2"),
        _backend_number(raw, "lon2"),
    )


def interpolate(
    start: PositionInput,
    end: PositionInput,
    *,
    fraction: Real | Decimal = 0.5,
) -> Position:
    """Return a point at ``fraction`` along the shortest WGS84 geodesic."""

    start_position = _coerce_position(start)
    end_position = _coerce_position(end)
    selected_fraction = _bounded_navigation_number(
        fraction,
        name="fraction",
        minimum=0,
        maximum=1,
    )
    if selected_fraction == 0.0:
        return start_position
    if selected_fraction == 1.0:
        return end_position
    if _physically_coincident(start_position, end_position):
        return start_position

    line = _wgs84().InverseLine(
        start_position.latitude,
        start_position.longitude,
        end_position.latitude,
        end_position.longitude,
    )
    line_distance = float(line.s13)
    if not isfinite(line_distance) or line_distance < 0:
        raise NavigationError(
            "WGS84 interpolation did not return a valid line distance"
        )
    if line_distance == 0.0:
        raise NavigationError(
            "positions are distinct but below WGS84 numerical resolution"
        )
    offset = line_distance * selected_fraction
    remaining = line_distance - offset
    if min(offset, remaining) < _MIN_OUTPUT_DISTANCE_METRES:
        raise NavigationError(
            "interpolation must remain at least 1e-7 metres from each "
            "endpoint"
        )
    raw = line.Position(offset)
    return Position(
        _backend_number(raw, "lat2"),
        _backend_number(raw, "lon2"),
    )


def nearest_position(
    origin: PositionInput,
    candidates: Iterable[PositionInput],
) -> Position:
    """Return the first candidate nearest to ``origin`` on WGS84."""

    if isinstance(candidates, (str, bytes, bytearray)):
        raise NavigationError("candidates must be an iterable of positions")
    try:
        iterator = iter(candidates)
    except TypeError as error:
        raise NavigationError(
            "candidates must be an iterable of positions"
        ) from error

    origin_position = _coerce_position(origin)
    nearest: Position | None = None
    nearest_distance = float("inf")
    for candidate in iterator:
        position = _coerce_position(candidate)
        candidate_distance = _inverse_positions(
            origin_position,
            position,
        ).distance
        if candidate_distance < nearest_distance:
            nearest = position
            nearest_distance = candidate_distance

    if nearest is None:
        raise NavigationError("candidates must contain at least one position")
    return nearest
