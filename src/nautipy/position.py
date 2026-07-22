"""Validated position values."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from math import isfinite
from numbers import Real

from .errors import CoordinateParseError, CoordinateRangeError


def _validated_component(value: object, *, axis: str, limit: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise CoordinateParseError(f"{axis} must be a real number")

    if isinstance(value, Decimal) and not value.is_finite():
        raise CoordinateRangeError(f"{axis} must be finite")
    if isinstance(value, float) and not isfinite(value):
        raise CoordinateRangeError(f"{axis} must be finite")

    exact_limit: float | Decimal
    exact_limit = Decimal(str(limit)) if isinstance(value, Decimal) else limit
    lower_limit = (
        exact_limit.copy_negate()
        if isinstance(exact_limit, Decimal)
        else -exact_limit
    )
    if value < lower_limit or value > exact_limit:
        raise CoordinateRangeError(
            f"{axis} must be between {-limit:g} and {limit:g} degrees"
        )

    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise CoordinateRangeError(f"{axis} must be finite") from error
    if not isfinite(number):
        raise CoordinateRangeError(f"{axis} must be finite")
    if value != 0 and number == 0.0:
        raise CoordinateRangeError(
            f"{axis} magnitude is too small for the internal float "
            "representation"
        )
    return number


@dataclass(frozen=True, slots=True)
class Position:
    """An immutable WGS84 position with optional non-coordinate metadata."""

    latitude: float
    longitude: float
    identifier: str | int | float | None = field(
        default=None,
        compare=False,
        kw_only=True,
    )
    description: str | None = field(
        default=None,
        compare=False,
        kw_only=True,
    )

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
        if isinstance(self.identifier, bool) or not isinstance(
            self.identifier,
            (str, int, float, type(None)),
        ):
            raise CoordinateParseError(
                "identifier must be a string, number, or None"
            )
        if isinstance(self.identifier, float) and not isfinite(self.identifier):
            raise CoordinateRangeError("identifier must be finite")
        if self.description is not None and not isinstance(self.description, str):
            raise CoordinateParseError("description must be a string or None")
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)
