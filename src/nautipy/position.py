"""Validated position values."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from .errors import CoordinateParseError, CoordinateRangeError


def _validated_component(value: object, *, axis: str, limit: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CoordinateParseError(f"{axis} must be a real number")

    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise CoordinateRangeError(f"{axis} must be finite") from error
    if not isfinite(number):
        raise CoordinateRangeError(f"{axis} must be finite")
    if not -limit <= number <= limit:
        raise CoordinateRangeError(
            f"{axis} must be between {-limit:g} and {limit:g} degrees"
        )
    return number


@dataclass(frozen=True, slots=True)
class Position:
    """An immutable WGS84 latitude/longitude position in decimal degrees."""

    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        latitude = _validated_component(
            self.latitude,
            axis="latitude",
            limit=90.0,
        )
        longitude = _validated_component(
            self.longitude,
            axis="longitude",
            limit=180.0,
        )
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)
