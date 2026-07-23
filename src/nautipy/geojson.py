"""Lightweight two-dimensional GeoJSON interchange."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from typing import TypeAlias

from .coordinates import (
    CoordinateOrder,
    PositionInput,
    _validate_input_format,
    _validate_parse_order,
    parse_position,
)
from .errors import (
    AmbiguousCoordinateError,
    CoordinateError,
    CoordinateParseError,
    CoordinateRangeError,
)
from .position import Position

_GeoJSONMapping: TypeAlias = dict[str, object]
_TEXT_TYPES = (str, bytes, bytearray)


def _reject_crs(value: Mapping[object, object], *, path: str) -> None:
    if "crs" in value:
        raise CoordinateParseError(
            f"{path} cannot contain a legacy GeoJSON CRS member"
        )


def _reject_cross_type_members(
    value: Mapping[object, object],
    *,
    forbidden: frozenset[str],
    path: str,
) -> None:
    present = sorted(member for member in forbidden if member in value)
    if present:
        rendered = ", ".join(repr(member) for member in present)
        raise CoordinateParseError(
            f"{path} contains member(s) belonging to another GeoJSON type: "
            f"{rendered}"
        )


def _point_position(
    value: Mapping[object, object],
    *,
    path: str,
) -> Position:
    _reject_crs(value, path=path)
    geometry_type = value.get("type")
    if geometry_type != "Point":
        if geometry_type is None:
            raise CoordinateParseError(f"{path} is missing its GeoJSON type")
        raise CoordinateParseError(
            f"{path} has unsupported geometry type {geometry_type!r}; "
            'only "Point" is supported'
        )
    _reject_cross_type_members(
        value,
        forbidden=frozenset({"features", "geometries", "geometry", "properties"}),
        path=path,
    )
    try:
        return parse_position(value)
    except CoordinateRangeError as error:
        raise CoordinateRangeError(f"{path}: {error}") from error
    except CoordinateError as error:
        raise CoordinateParseError(f"{path}: {error}") from error


def _coordinate(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _point_mapping(position: Position) -> _GeoJSONMapping:
    return {
        "type": "Point",
        "coordinates": [
            _coordinate(position.longitude),
            _coordinate(position.latitude),
        ],
    }


def to_geojson_point(
    value: PositionInput,
    *,
    order: CoordinateOrder = "latlon",
    format: str | None = None,
) -> _GeoJSONMapping:
    """Return a GeoJSON Point mapping for one position-like value.

    A bare Point has no standard place for feature metadata. Metadata-bearing
    positions must therefore be exported through a FeatureCollection.
    """

    position = parse_position(value, order=order, format=format)
    if position.identifier is not None or position.description is not None:
        raise CoordinateParseError(
            "a bare GeoJSON Point cannot preserve identifier or description; "
            "use to_geojson_feature_collection instead"
        )
    return _point_mapping(position)


def from_geojson_point(value: Mapping[object, object]) -> Position:
    """Parse one two-dimensional GeoJSON Point mapping."""

    if not isinstance(value, Mapping):
        raise CoordinateParseError("GeoJSON Point input must be a mapping")
    return _point_position(value, path="GeoJSON Point")


def _feature_position(
    value: object,
    *,
    index: int,
) -> Position:
    path = f"features[{index}]"
    if not isinstance(value, Mapping):
        raise CoordinateParseError(f"{path} must be a GeoJSON Feature mapping")
    _reject_crs(value, path=path)
    _reject_cross_type_members(
        value,
        forbidden=frozenset({"coordinates", "features", "geometries"}),
        path=path,
    )
    if value.get("type") != "Feature":
        raise CoordinateParseError(
            f'{path} must have GeoJSON type "Feature"'
        )
    if "geometry" not in value:
        raise CoordinateParseError(f"{path} is missing geometry")
    geometry = value["geometry"]
    if geometry is None:
        raise CoordinateParseError(
            f"{path}.geometry is null; unlocated Features are unsupported"
        )
    if not isinstance(geometry, Mapping):
        raise CoordinateParseError(f"{path}.geometry must be a mapping")
    position = _point_position(geometry, path=f"{path}.geometry")

    if "properties" not in value:
        raise CoordinateParseError(f"{path} is missing properties")
    properties = value["properties"]
    if properties is not None and not isinstance(properties, Mapping):
        raise CoordinateParseError(
            f"{path}.properties must be a mapping or null"
        )
    description: str | None = None
    if properties is not None and "description" in properties:
        raw_description = properties["description"]
        if raw_description is not None and not isinstance(raw_description, str):
            raise CoordinateParseError(
                f"{path}.properties.description must be a string or null"
            )
        description = raw_description

    identifier: str | int | float | None = None
    if "id" in value:
        raw_identifier = value["id"]
        if isinstance(raw_identifier, bool) or not isinstance(
            raw_identifier,
            (str, int, float),
        ):
            raise CoordinateParseError(
                f"{path}.id must be a JSON string or number"
            )
        if isinstance(raw_identifier, float) and not isfinite(raw_identifier):
            raise CoordinateRangeError(f"{path}.id must be finite")
        identifier = raw_identifier

    return Position(
        position.latitude,
        position.longitude,
        identifier=identifier,
        description=description,
    )


def from_geojson_feature_collection(
    value: Mapping[object, object],
) -> tuple[Position, ...]:
    """Parse an ordered collection of two-dimensional Point Features."""

    if not isinstance(value, Mapping):
        raise CoordinateParseError(
            "GeoJSON FeatureCollection input must be a mapping"
        )
    path = "GeoJSON FeatureCollection"
    _reject_crs(value, path=path)
    _reject_cross_type_members(
        value,
        forbidden=frozenset(
            {"coordinates", "geometries", "geometry", "properties"}
        ),
        path=path,
    )
    if value.get("type") != "FeatureCollection":
        raise CoordinateParseError(
            'GeoJSON collection must have type "FeatureCollection"'
        )
    if "features" not in value:
        raise CoordinateParseError(
            "GeoJSON FeatureCollection is missing features"
        )
    features = value["features"]
    if isinstance(features, _TEXT_TYPES) or not isinstance(features, Sequence):
        raise CoordinateParseError(
            "GeoJSON FeatureCollection features must be an array"
        )
    return tuple(
        _feature_position(feature, index=index)
        for index, feature in enumerate(features)
    )


def to_geojson_feature_collection(
    values: Iterable[PositionInput],
    *,
    order: CoordinateOrder = "latlon",
    format: str | None = None,
) -> _GeoJSONMapping:
    """Return a GeoJSON FeatureCollection mapping for an iterable of positions."""

    _validate_parse_order(order)
    _validate_input_format(format)
    if isinstance(values, (Position, Mapping, *_TEXT_TYPES)):
        raise CoordinateParseError(
            "FeatureCollection export requires an iterable of position values"
        )
    try:
        iterator = iter(values)
    except TypeError as error:
        raise CoordinateParseError(
            "FeatureCollection export requires an iterable of position values"
        ) from error

    features: list[_GeoJSONMapping] = []
    for index, value in enumerate(iterator):
        try:
            position = parse_position(value, order=order, format=format)
        except AmbiguousCoordinateError as error:
            raise AmbiguousCoordinateError(
                f"positions[{index}]: {error}",
                candidates=error.candidates,
            ) from error
        except CoordinateError as error:
            raise type(error)(f"positions[{index}]: {error}") from error
        properties: dict[str, object] = {}
        if position.description is not None:
            properties["description"] = position.description
        feature: _GeoJSONMapping = {
            "type": "Feature",
            "geometry": _point_mapping(position),
            "properties": properties,
        }
        if position.identifier is not None:
            feature["id"] = position.identifier
        features.append(feature)
    return {"type": "FeatureCollection", "features": features}


__all__ = [
    "to_geojson_point",
    "from_geojson_point",
    "to_geojson_feature_collection",
    "from_geojson_feature_collection",
]
