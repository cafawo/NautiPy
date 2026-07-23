from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction
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

        with self.assertRaisesRegex(CoordinateRangeError, "latitude"):
            Position(10**400, 0)

    def test_position_rejects_out_of_range_values(self) -> None:
        with self.assertRaisesRegex(CoordinateRangeError, "latitude"):
            Position(90.0001, 0)
        with self.assertRaisesRegex(CoordinateRangeError, "longitude"):
            Position(0, -180.0001)
        with self.assertRaisesRegex(CoordinateRangeError, "latitude"):
            Position(Fraction(90000000000000001, 10**15), 0)
        with self.assertRaisesRegex(CoordinateRangeError, "longitude"):
            Position(0, Decimal("180.000000000000001"))

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
        self.assertEqual(parse_position("+5e+1, +8e+0"), Position(50, 8))

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
        malformed = (
            "",
            "50",
            "50, 8, 1",
            "+50.1,+008.1,100",
            "north, east",
        )
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

    def test_exact_range_validation_happens_before_float_conversion(self) -> None:
        just_over_latitude = Fraction(90000000000000001, 10**15)
        with self.assertRaisesRegex(CoordinateRangeError, "latitude"):
            parse_position("90.000000000000001, 0")
        with self.assertRaisesRegex(CoordinateRangeError, "longitude"):
            parse_position("0, 180.000000000000001")
        with self.assertRaisesRegex(CoordinateRangeError, "latitude"):
            parse_position((just_over_latitude, 0))

        self.assertEqual(
            parse_position("90.000000000000001, 0", order="auto"),
            Position(0, 90),
        )
        self.assertEqual(
            parse_position((just_over_latitude, 0), order="auto"),
            Position(0, 90),
        )


class HumanReadableFormatTests(unittest.TestCase):
    reference = Position(50.12257, 8.66570)

    def assertReference(self, value: object, **options: object) -> None:
        position = parse_position(value, **options)  # type: ignore[arg-type]
        self.assertAlmostEqual(position.latitude, self.reference.latitude, places=10)
        self.assertAlmostEqual(position.longitude, self.reference.longitude, places=10)

    def test_parses_decimal_degrees_with_hemisphere_variants(self) -> None:
        values = (
            "50.12257 N; 8.66570 E",
            "N 50.12257; E 8.66570",
            "50.12257 north; 8.66570 east",
            "50.12257n, 8.66570e",
            "  north   50.12257 ; east   8.66570  ",
            "+50.12257 N; +8.66570 E",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertReference(value)

        self.assertEqual(
            parse_position("50.12257 S; 8.66570 W"),
            Position(-50.12257, -8.66570),
        )
        self.assertEqual(
            parse_position("S 50.12257; W 8.66570"),
            Position(-50.12257, -8.66570),
        )

    def test_parses_ddm_symbol_colon_word_and_space_forms(self) -> None:
        values = (
            "50° 7.3542' N; 8° 39.942' E",
            "N 50 7.3542; E 8 39.942",
            "50 deg 7.3542 min N; 8 degrees 39.942 minutes E",
            "50:7.3542 N; 8:39.942 E",
            ("50° 7.3542' N", "8° 39.942' E"),
        )
        for value in values:
            with self.subTest(value=value):
                self.assertReference(value)

    def test_parses_dms_ascii_unicode_word_and_space_forms(self) -> None:
        values = (
            "50° 7' 21.252\" N; 8° 39' 56.52\" E",
            "N 50 7 21.252; E 8 39 56.52",
            "50 degrees 7 minutes 21.252 seconds N; "
            "8 degrees 39 minutes 56.52 seconds E",
            "50:7:21.252 N; 8:39:56.52 E",
            "50º 7′ 21.252″ N; 8º 39′ 56.52″ E",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertReference(value)

    def test_normalizes_unicode_minus(self) -> None:
        self.assertEqual(
            parse_position("−50.5, −8.25"),
            Position(-50.5, -8.25),
        )

    def test_axis_evidence_overrides_source_and_requested_order(self) -> None:
        reversed_text = "8.66570 E, 50.12257 N"
        for order in ("latlon", "lonlat", "auto"):
            with self.subTest(order=order):
                self.assertReference(reversed_text, order=order)

        self.assertReference("50.12257 N; 8.66570", order="auto")
        self.assertReference("8.66570; 50.12257 N", order="auto")

    def test_explicit_format_selection_and_alias(self) -> None:
        self.assertReference("50.12257, 8.66570", format="dd")
        self.assertReference(
            "50° 7.3542' N; 8° 39.942' E",
            format="ddm",
        )
        self.assertReference(
            "50° 7.3542' N; 8° 39.942' E",
            format="dmm",
        )
        self.assertReference(
            "50° 7' 21.252\" N; 8° 39' 56.52\" E",
            format="dms",
        )

        with self.assertRaisesRegex(CoordinateParseError, "requested dd"):
            parse_position(
                "50° 7.3542' N; 8° 39.942' E",
                format="dd",
            )
        with self.assertRaisesRegex(CoordinateParseError, "requested ddm"):
            parse_position((50.12257, 8.66570), format="ddm")
        with self.assertRaisesRegex(CoordinateParseError, "format"):
            parse_position("50, 8", format="utm")

    def test_detects_each_component_format_independently(self) -> None:
        mixed_values = (
            "50.12257 N; 8° 39.942' E",
            "N 50° 7.3542'; 8.66570 E",
            ("50.12257 N", "8° 39.942' E"),
            (50.12257, "8° 39.942' E"),
        )
        for value in mixed_values:
            with self.subTest(value=value):
                self.assertReference(value)

        with self.assertRaisesRegex(CoordinateParseError, "requested dd"):
            parse_position(
                "50.12257 N; 8° 39.942' E",
                format="dd",
            )

    def test_supports_unambiguous_decimal_comma(self) -> None:
        self.assertReference("50,12257; 8,66570")
        self.assertReference("50,12257 / 8,66570")
        self.assertReference("50,12257 N; 8,66570 E")
        self.assertReference("50° 7,3542' N; 8° 39,942' E")
        self.assertReference("50,12257 N, 8,66570 E")
        self.assertReference("50° 7,3542', 8° 39,942'")
        self.assertReference("50° 7' 21,252\", 8° 39' 56,52\"")

        with self.assertRaisesRegex(
            AmbiguousCoordinateError,
            "semicolon|dot decimals",
        ):
            parse_position("50,12257, 8,66570")
        with self.assertRaises(AmbiguousCoordinateError):
            parse_position("50,12257, 8.66570")
        with self.assertRaises(AmbiguousCoordinateError):
            parse_position("50,12257 8,66570")
        with self.assertRaises(AmbiguousCoordinateError):
            parse_position("50, 8,200")
        with self.assertRaises(AmbiguousCoordinateError):
            parse_position("50,12257°, 8,66570°")
        with self.assertRaises(AmbiguousCoordinateError):
            parse_position(",5, ,8")

    def test_rejects_sign_and_hemisphere_conflicts(self) -> None:
        invalid = (
            "-50 N; 8 E",
            "-50 S; 8 E",
            "+50 S; 8 E",
            "50 N N; 8 E",
            "50 N; 8 E E",
            "50 N; 8 N",
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(CoordinateParseError):
                    parse_position(value)

    def test_rejects_invalid_subcomponents_and_extrema(self) -> None:
        range_errors = (
            "50° 60' N; 8° 0' E",
            "50° -1' N; 8° 0' E",
            "50° 0' 60\" N; 8° 0' 0\" E",
            "50° 0' -1\" N; 8° 0' 0\" E",
            "90° 0.1' N; 8° 0' E",
            "90° 0.000000000000000000000000000001' N; 8° 0' E",
            "50° 0' N; 180° 0.1' E",
            "50° 0' 0\" N; "
            "180° 0' 0.000000000000000000000000000001\" E",
            "50° 60,1', 8° 39,9'",
            "50° 7' 60,1\", 8° 39' 56,5\"",
        )
        for value in range_errors:
            with self.subTest(value=value):
                with self.assertRaises(CoordinateRangeError):
                    parse_position(value)

        with self.assertRaises(CoordinateParseError):
            parse_position("50.5° 7' N; 8° 0' E")
        with self.assertRaises(CoordinateParseError):
            parse_position("50° 7' N extra; 8° 0' E")

    def test_rejects_pathologically_long_or_split_heavy_text(self) -> None:
        with self.assertRaisesRegex(CoordinateParseError, "must not exceed"):
            parse_position("1 " * 3000)
        with self.assertRaisesRegex(CoordinateParseError, "too many"):
            parse_position(" ".join("1" for _ in range(300)))

    def test_rejects_bare_numeric_grouping_that_could_hide_altitude(self) -> None:
        invalid = (
            "50 8 100",
            "50 8 10 100",
            "50 8 10 100 20",
            "50 7 8 39",
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    AmbiguousCoordinateError,
                    "altitude|insert ';'|explicit format",
                ):
                    parse_position(value)

        self.assertReference("50 7.3542; 8 39.942", format="ddm")
        for value in ("999 999 999", "nan 8 100", "inf inf inf"):
            with self.subTest(value=value):
                with self.assertRaises(CoordinateRangeError):
                    parse_position(value)

    def test_accepts_legal_ddm_and_dms_extrema(self) -> None:
        self.assertEqual(
            parse_position("90° 0' N; 180° 0' E"),
            Position(90, 180),
        )
        self.assertEqual(
            parse_position("90° 0' 0\" S; 180° 0' 0\" W"),
            Position(-90, -180),
        )

    def test_rejects_non_ascii_hemisphere_lookalikes(self) -> None:
        with self.assertRaises(CoordinateParseError):
            parse_position("50 ſ; 8 E")

    def test_human_formats_round_trip_through_canonical_dd(self) -> None:
        inputs = (
            "50.12257 N; 8.66570 E",
            "50° 7.3542' N; 8° 39.942' E",
            "50° 7' 21.252\" N; 8° 39' 56.52\" E",
        )
        for value in inputs:
            with self.subTest(value=value):
                parsed = parse_position(value)
                self.assertEqual(parse_position(format_position(parsed)), parsed)


class FormatPositionTests(unittest.TestCase):
    def test_formats_canonical_decimal_degrees(self) -> None:
        position = Position(50.12257, 8.66570)

        self.assertEqual(format_position(position), "50.122570, 8.665700")
        self.assertEqual(
            format_position(position, precision=3),
            "50.123, 8.666",
        )
        self.assertEqual(
            format_position(position, order="lonlat"),
            "8.665700, 50.122570",
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
                "PositionInput",
                "CandidateDiagnostic",
                "ParseResult",
                "parse_position",
                "inspect_position",
                "format_position",
                "convert_position",
                "InverseResult",
                "inverse",
                "distance",
                "initial_bearing",
                "destination",
                "interpolate",
                "nearest_position",
                "BearingObservation",
                "RangeObservation",
                "ObservationResidual",
                "FixUncertainty",
                "FixStatus",
                "CandidateStatus",
                "CandidateResult",
                "FixResult",
                "two_bearing_candidates",
                "two_range_candidates",
                "solve_fix",
                "NautiPyError",
                "NavigationError",
                "FixError",
                "CoordinateError",
                "CoordinateParseError",
                "CoordinateRangeError",
                "AmbiguousCoordinateError",
            ],
        )
        for legacy_name in ("Pos", "haversine", "triangulate", "multilaterate"):
            with self.subTest(name=legacy_name):
                self.assertFalse(hasattr(nautipy, legacy_name))

        for implementation_name in ("import_module", "TYPE_CHECKING"):
            with self.subTest(name=implementation_name):
                self.assertFalse(hasattr(nautipy, implementation_name))

    def test_public_modules_define_explicit_star_import_surfaces(self) -> None:
        import nautipy.coordinates as coordinates
        import nautipy.errors as errors
        import nautipy.fix as fix
        import nautipy.geodesic as geodesic
        import nautipy.geojson as geojson
        import nautipy.position as position

        self.assertEqual(
            coordinates.__all__,
            [
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
            ],
        )
        self.assertEqual(
            geodesic.__all__,
            [
                "InverseResult",
                "destination",
                "distance",
                "initial_bearing",
                "interpolate",
                "inverse",
                "nearest_position",
            ],
        )
        self.assertEqual(position.__all__, ["Position"])
        self.assertEqual(
            geojson.__all__,
            [
                "to_geojson_point",
                "from_geojson_point",
                "to_geojson_feature_collection",
                "from_geojson_feature_collection",
            ],
        )
        self.assertEqual(
            fix.__all__,
            [
                "BearingObservation",
                "RangeObservation",
                "ObservationResidual",
                "FixUncertainty",
                "FixStatus",
                "CandidateStatus",
                "CandidateResult",
                "FixResult",
                "FixError",
                "two_bearing_candidates",
                "two_range_candidates",
                "solve_fix",
            ],
        )
        self.assertEqual(
            errors.__all__,
            [
                "AmbiguousCoordinateError",
                "CoordinateError",
                "CoordinateParseError",
                "CoordinateRangeError",
                "FixError",
                "NautiPyError",
                "NavigationError",
            ],
        )

        for module in (
            nautipy,
            coordinates,
            errors,
            fix,
            geodesic,
            geojson,
            position,
        ):
            with self.subTest(module=module.__name__):
                self.assertEqual(len(module.__all__), len(set(module.__all__)))
                for name in module.__all__:
                    self.assertFalse(name.startswith("_"), name)
                    self.assertTrue(hasattr(module, name), name)


if __name__ == "__main__":
    unittest.main()
