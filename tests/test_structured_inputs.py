from decimal import Decimal
from fractions import Fraction
import math
from numbers import Real
import unittest

from nautipy import (
    CoordinateParseError,
    CoordinateRangeError,
    Position,
    parse_position,
)


@Real.register
class RatioReal:
    def __init__(self, numerator: int, denominator: int) -> None:
        self._ratio = (numerator, denominator)

    def as_integer_ratio(self) -> tuple[int, int]:
        return self._ratio

    def __float__(self) -> float:
        numerator, denominator = self._ratio
        return numerator / denominator


@Real.register
class IndexReal:
    def __init__(self, value: int) -> None:
        self._value = value

    def __index__(self) -> int:
        return self._value

    def __float__(self) -> float:
        return float(self._value)


class NamedMappingTests(unittest.TestCase):
    reference = Position(50.12257, 8.66570)

    def assertReference(self, value: object, **options: object) -> None:
        position = parse_position(value, **options)  # type: ignore[arg-type]
        self.assertAlmostEqual(position.latitude, self.reference.latitude, places=12)
        self.assertAlmostEqual(
            position.longitude,
            self.reference.longitude,
            places=12,
        )

    def test_accepts_short_full_and_mixed_axis_names(self) -> None:
        values = (
            {"lat": 50.12257, "lon": 8.66570},
            {"latitude": 50.12257, "longitude": 8.66570},
            {"lat": 50.12257, "longitude": 8.66570},
            {"longitude": 8.66570, "latitude": 50.12257},
        )
        for value in values:
            with self.subTest(value=value):
                self.assertReference(value)

    def test_named_axes_override_order_and_parse_human_components(self) -> None:
        value = {
            "latitude": "50° 7.3542'",
            "longitude": "8° 39.942'",
        }
        for order in ("latlon", "lonlat", "auto"):
            with self.subTest(order=order):
                self.assertReference(value, order=order)
        self.assertReference(value, format="ddm")
        self.assertReference({"lat": "50,12257", "lon": "8,66570"})

    def test_named_axis_markers_must_agree_with_keys(self) -> None:
        with self.assertRaisesRegex(CoordinateParseError, "latitude.*axis"):
            parse_position({"lat": "8 E", "lon": "50 N"})
        with self.assertRaisesRegex(CoordinateParseError, "longitude.*axis"):
            parse_position({"lat": "50 N", "lon": "8 N"})

    def test_rejects_missing_duplicate_unknown_and_extra_keys(self) -> None:
        invalid = (
            {},
            {"lat": 50},
            {"lon": 8},
            {"lat": 50, "latitude": 50, "lon": 8},
            {"lat": 50, "lon": 8, "alt": 100},
            {"lat": 50, "lng": 8},
            {"LAT": 50, "LON": 8},
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(CoordinateParseError):
                    parse_position(value)

    def test_named_values_use_exact_range_validation(self) -> None:
        self.assertEqual(
            parse_position({"lat": Decimal("90"), "lon": Fraction(180)}),
            Position(90, 180),
        )
        with self.assertRaisesRegex(CoordinateRangeError, "latitude"):
            parse_position(
                {"lat": Decimal("90.000000000000001"), "lon": 0}
            )
        with self.assertRaises(CoordinateParseError):
            parse_position({"lat": 50, "lon": 8}, format="iso6709")

    def test_accepts_real_scalars_with_standard_exact_conversion_hooks(self) -> None:
        latitude = RatioReal(5_012_257, 100_000)
        longitude = RatioReal(866_570, 100_000)
        expected = Position(50.12257, 8.66570)

        self.assertEqual(parse_position((latitude, longitude)), expected)
        self.assertEqual(
            parse_position({"lat": latitude, "lon": IndexReal(8)}),
            Position(50.12257, 8),
        )
        self.assertEqual(
            parse_position(
                {"type": "Point", "coordinates": [longitude, latitude]}
            ),
            expected,
        )
        with self.assertRaises(CoordinateRangeError):
            parse_position((RatioReal(9_000_000_000_000_001, 10**14), 0))


class GeoJsonPointTests(unittest.TestCase):
    reference = Position(50.12257, 8.66570)

    def test_geojson_uses_fixed_longitude_latitude_order(self) -> None:
        value = {"type": "Point", "coordinates": [8.66570, 50.12257]}
        for order in ("latlon", "lonlat", "auto"):
            with self.subTest(order=order):
                self.assertEqual(parse_position(value, order=order), self.reference)

        self.assertEqual(parse_position(value, format="dd"), self.reference)
        self.assertEqual(
            parse_position([8.66570, 50.12257]),
            Position(8.66570, 50.12257),
        )

    def test_accepts_tuple_coordinates_and_foreign_members(self) -> None:
        value = {
            "type": "Point",
            "coordinates": (8.66570, 50.12257),
            "bbox": [8, 50, 9, 51],
            "description": "reference",
        }
        self.assertEqual(parse_position(value), self.reference)

    def test_geojson_extrema_are_validated_exactly(self) -> None:
        self.assertEqual(
            parse_position(
                {
                    "type": "Point",
                    "coordinates": [Decimal("180"), Fraction(90)],
                }
            ),
            Position(90, 180),
        )
        with self.assertRaisesRegex(CoordinateRangeError, "longitude"):
            parse_position(
                {
                    "type": "Point",
                    "coordinates": [Decimal("180.000000000000001"), 0],
                }
            )
        with self.assertRaisesRegex(CoordinateRangeError, "latitude"):
            parse_position(
                {"type": "Point", "coordinates": [0, Decimal("90.0001")]}
            )

    def test_rejects_invalid_geojson_shapes_and_types(self) -> None:
        invalid = (
            {"type": "point", "coordinates": [8, 50]},
            {"type": "LineString", "coordinates": [[8, 50], [9, 51]]},
            {"type": "Feature", "geometry": None},
            {"type": "Point"},
            {"coordinates": [8, 50]},
            {"type": "Point", "coordinates": 8},
            {"type": "Point", "coordinates": "8, 50"},
            {"type": "Point", "coordinates": [8]},
            {"type": "Point", "coordinates": [8, 50, 100]},
            {"type": "Point", "coordinates": [[8], 50]},
            {"type": "Point", "coordinates": [True, 50]},
            {"type": "Point", "coordinates": ["8", "50"]},
            {"type": "Point", "coordinates": [8, 50], "crs": {}},
            {"type": "Point", "coordinates": [8, 50], "lat": 50},
            {"type": "Point", "coordinates": [8, 50], "features": []},
            {"type": "Point", "coordinates": [8, 50], "geometries": []},
            {"type": "Point", "coordinates": [8, 50], "geometry": None},
            {"type": "Point", "coordinates": [8, 50], "properties": {}},
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(CoordinateParseError):
                    parse_position(value)

        with self.assertRaises(CoordinateParseError):
            parse_position(
                {"type": "Point", "coordinates": [8, 50]},
                format="dms",
            )

    def test_rejects_non_finite_geojson_values(self) -> None:
        for coordinates in ([math.inf, 50], [8, math.nan]):
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(CoordinateRangeError):
                    parse_position({"type": "Point", "coordinates": coordinates})


if __name__ == "__main__":
    unittest.main()
