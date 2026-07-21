from dataclasses import FrozenInstanceError
import math
import unittest

import nautipy
from nautipy import (
    AmbiguousCoordinateError,
    CoordinateParseError,
    CoordinateRangeError,
    Position,
    format_position,
    parse_position,
)


class PositionTests(unittest.TestCase):
    def test_position_is_immutable_and_normalizes_real_values_to_float(self) -> None:
        position = Position(50, 8)

        self.assertEqual(position, Position(50.0, 8.0))
        self.assertIsInstance(position.latitude, float)
        self.assertIsInstance(position.longitude, float)
        with self.assertRaises(FrozenInstanceError):
            position.latitude = 51.0  # type: ignore[misc]

    def test_position_accepts_legal_extrema(self) -> None:
        self.assertEqual(Position(-90, -180), Position(-90.0, -180.0))
        self.assertEqual(Position(90, 180), Position(90.0, 180.0))

    def test_position_rejects_non_finite_values(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            for coordinates in ((value, 0), (0, value)):
                with self.subTest(coordinates=coordinates):
                    with self.assertRaisesRegex(CoordinateRangeError, "finite"):
                        Position(*coordinates)

        with self.assertRaisesRegex(CoordinateRangeError, "finite"):
            Position(10**400, 0)

    def test_position_rejects_out_of_range_values(self) -> None:
        with self.assertRaisesRegex(CoordinateRangeError, "latitude"):
            Position(90.0001, 0)
        with self.assertRaisesRegex(CoordinateRangeError, "longitude"):
            Position(0, -180.0001)

    def test_position_rejects_non_numeric_values(self) -> None:
        with self.assertRaisesRegex(CoordinateParseError, "latitude"):
            Position("50", 8)  # type: ignore[arg-type]
        with self.assertRaises(CoordinateParseError):
            Position(True, 8)


class ParsePositionTests(unittest.TestCase):
    def test_parses_decimal_degree_text_and_sequences(self) -> None:
        expected = Position(50.12257, 8.66570)

        self.assertEqual(parse_position("50.12257, 8.66570"), expected)
        self.assertEqual(parse_position("50.12257 8.66570"), expected)
        self.assertEqual(parse_position((50.12257, 8.66570)), expected)
        self.assertIs(parse_position(expected), expected)

    def test_explicit_lonlat_order_swaps_components(self) -> None:
        expected = Position(50.12257, 8.66570)

        self.assertEqual(
            parse_position([50.12257, 8.66570], order="latlon"),
            expected,
        )
        self.assertEqual(
            parse_position([8.66570, 50.12257], order="lonlat"),
            expected,
        )
        self.assertEqual(
            parse_position([80, 40], order="latlon"),
            Position(80, 40),
        )
        with self.assertRaises(CoordinateRangeError):
            parse_position((120, 50), order="latlon")
        with self.assertRaises(CoordinateRangeError):
            parse_position((50, 120), order="lonlat")

    def test_parses_legal_extrema_in_both_explicit_orders(self) -> None:
        self.assertEqual(parse_position("-90, -180"), Position(-90, -180))
        self.assertEqual(
            parse_position("180, 90", order="lonlat"),
            Position(90, 180),
        )

    def test_auto_order_uses_range_evidence_in_either_direction(self) -> None:
        expected = Position(50.0, 120.0)

        self.assertEqual(parse_position("50, 120", order="auto"), expected)
        self.assertEqual(parse_position("120, 50", order="auto"), expected)

    def test_auto_order_rejects_ambiguous_values_with_resolution(self) -> None:
        with self.assertRaises(AmbiguousCoordinateError) as raised:
            parse_position("8, 50", order="auto")

        self.assertIn('order="latlon"', str(raised.exception))
        self.assertIn('order="lonlat"', str(raised.exception))

    def test_auto_order_accepts_equivalent_candidates(self) -> None:
        self.assertEqual(parse_position("8, 8", order="auto"), Position(8, 8))
        self.assertEqual(
            parse_position((-0.0, 0.0), order="auto"),
            Position(0.0, 0.0),
        )

    def test_auto_order_rejects_values_valid_in_neither_order(self) -> None:
        with self.assertRaisesRegex(CoordinateRangeError, "no valid"):
            parse_position("120, 95", order="auto")

    def test_rejects_malformed_pairs(self) -> None:
        malformed = ("", "50", "50, 8, 1", "north, east")
        for value in malformed:
            with self.subTest(value=value):
                with self.assertRaises(CoordinateParseError):
                    parse_position(value)

        for value in ([], [50], [50, 8, 1], 50):
            with self.subTest(value=value):
                with self.assertRaises(CoordinateParseError):
                    parse_position(value)  # type: ignore[arg-type]

    def test_rejects_non_finite_text_and_invalid_order(self) -> None:
        for value in ("nan, 8", "50, inf", (10**400, 0)):
            with self.subTest(value=value):
                with self.assertRaises(CoordinateRangeError):
                    parse_position(value)
        with self.assertRaisesRegex(CoordinateParseError, "order"):
            parse_position("50, 8", order="guess")  # type: ignore[arg-type]
        with self.assertRaisesRegex(CoordinateParseError, "order"):
            parse_position("50, 8", order=[])  # type: ignore[arg-type]


class FormatPositionTests(unittest.TestCase):
    def test_formats_canonical_decimal_degrees(self) -> None:
        position = Position(50.12257, 8.66570)

        self.assertEqual(format_position(position), "50.12257, 8.6657")
        self.assertEqual(
            format_position(position, precision=3),
            "50.123, 8.666",
        )
        self.assertEqual(
            format_position(position, order="lonlat"),
            "8.6657, 50.12257",
        )

    def test_formatting_does_not_emit_negative_zero(self) -> None:
        self.assertEqual(
            format_position(Position(-0.0, -0.0001), precision=3),
            "0.000, 0.000",
        )

    def test_rejects_invalid_format_options_and_values(self) -> None:
        with self.assertRaises(CoordinateParseError):
            format_position((50, 8))  # type: ignore[arg-type]
        with self.assertRaisesRegex(CoordinateParseError, "output order"):
            format_position(Position(50, 8), order="auto")  # type: ignore[arg-type]
        for precision in (-1, 16, 1.5, True):
            with self.subTest(precision=precision):
                with self.assertRaises(CoordinateParseError):
                    format_position(
                        Position(50, 8),
                        precision=precision,  # type: ignore[arg-type]
                    )


class PublicApiTests(unittest.TestCase):
    def test_top_level_api_is_intentional_and_has_no_legacy_aliases(self) -> None:
        self.assertEqual(
            nautipy.__all__,
            [
                "Position",
                "parse_position",
                "format_position",
                "NautiPyError",
                "CoordinateError",
                "CoordinateParseError",
                "CoordinateRangeError",
                "AmbiguousCoordinateError",
            ],
        )
        for legacy_name in ("Pos", "haversine", "triangulate", "multilaterate"):
            with self.subTest(name=legacy_name):
                self.assertFalse(hasattr(nautipy, legacy_name))


if __name__ == "__main__":
    unittest.main()
