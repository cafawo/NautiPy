from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
import random
import unittest

from nautipy import (
    AmbiguousCoordinateError,
    CoordinateParseError,
    CoordinateRangeError,
    ParseResult,
    Position,
    convert_position,
    format_position,
    inspect_position,
    parse_position,
)


class InspectPositionTests(unittest.TestCase):
    reference = Position(50.12257, 8.66570)

    def assertResolution(
        self,
        result: ParseResult,
        latitude: float,
        longitude: float,
    ) -> None:
        self.assertIsNotNone(result.latitude_resolution)
        self.assertIsNotNone(result.longitude_resolution)
        self.assertAlmostEqual(
            float(result.latitude_resolution or 0.0),
            latitude,
        )
        self.assertAlmostEqual(
            float(result.longitude_resolution or 0.0),
            longitude,
        )

    def test_reports_formats_source_order_and_lexical_resolution(self) -> None:
        cases = (
            ("50.12257, 8.66570", "dd", 1e-5, 1e-5),
            (
                "50° 7.3542′ N; 8° 39.942′ E",
                "ddm",
                1e-4 / 60,
                1e-3 / 60,
            ),
            (
                "50° 7′ 21.252″ N; 8° 39′ 56.52″ E",
                "dms",
                1e-3 / 3600,
                1e-2 / 3600,
            ),
            ("+50.12257+008.66570/", "iso6709", 1e-5, 1e-5),
            (
                "+5007.3542+00839.9420/",
                "iso6709",
                1e-4 / 60,
                1e-4 / 60,
            ),
            (
                "+500721.252+0083956.52/",
                "iso6709",
                1e-3 / 3600,
                1e-2 / 3600,
            ),
            (
                "5007.3542,N,00839.9420,E",
                "nmea",
                1e-4 / 60,
                1e-4 / 60,
            ),
        )
        for value, format_name, latitude, longitude in cases:
            with self.subTest(value=value):
                result = inspect_position(value)
                self.assertEqual(result.position, self.reference)
                self.assertEqual(result.format, format_name)
                self.assertEqual(result.source_order, "latlon")
                self.assertEqual(result.warnings, ())
                self.assertResolution(result, latitude, longitude)
                self.assertEqual(result.candidates[0].outcome, "selected")

    def test_preserves_axis_aligned_metadata_for_reversed_input(self) -> None:
        result = inspect_position("00839.9420,E,5007.3542,N")

        self.assertEqual(result.position, self.reference)
        self.assertEqual(result.source_order, "lonlat")
        self.assertResolution(result, 1e-4 / 60, 1e-4 / 60)

        result = inspect_position("8.66570 E, 50.12257 N")
        self.assertEqual(result.source_order, "lonlat")
        self.assertResolution(result, 1e-5, 1e-5)

    def test_records_normalized_tokens_without_warnings(self) -> None:
        original = "  50,12257 n; 8,66570 e  "
        result = inspect_position(original)

        self.assertEqual(result.original_text, original)
        self.assertEqual(
            result.normalized_tokens,
            ("50.12257", "N", ";", "8.66570", "E"),
        )
        self.assertEqual(
            result.normalizations,
            (
                "trimmed outer whitespace",
                "normalized hemisphere letter case",
                "normalized decimal comma",
            ),
        )
        self.assertEqual(result.warnings, ())

        nmea = inspect_position("5007.3542,n,00839.9420,e")
        self.assertEqual(
            nmea.normalized_tokens,
            ("5007.3542", ",", "N", ",", "00839.9420", ",", "E"),
        )

    def test_reports_equivalent_and_rejected_candidates(self) -> None:
        equivalent = inspect_position("+50.12257 +008.66570")
        self.assertEqual(
            [
                (item.format, item.outcome)
                for item in equivalent.candidates
                if item.outcome != "rejected"
            ],
            [("iso6709", "selected"), ("dd", "equivalent")],
        )
        self.assertTrue(
            any(item.outcome == "rejected" for item in equivalent.candidates)
        )

        ranged = inspect_position("120, 50", order="auto")
        rejected = [
            item
            for item in ranged.candidates
            if item.format == "dd" and item.outcome == "rejected"
        ]
        self.assertEqual(ranged.source_order, "lonlat")
        self.assertEqual(len(rejected), 1)
        self.assertIn("latitude 120", rejected[0].reason or "")

        with self.assertRaises(AmbiguousCoordinateError) as raised:
            inspect_position("8.66570, 50.12257", order="auto")
        competing = raised.exception.candidates
        self.assertEqual(len(competing), 2)
        self.assertEqual(
            [candidate.outcome for candidate in competing],
            ["competing", "competing"],
        )
        self.assertEqual(
            [candidate.position for candidate in competing],
            [
                Position(8.66570, 50.12257),
                Position(50.12257, 8.66570),
            ],
        )

    def test_indeterminate_equal_order_uses_conservative_resolution(self) -> None:
        result = inspect_position("8.0, 8.00", order="auto")

        self.assertIsNone(result.source_order)
        self.assertIn("equivalent coordinate orders", result.evidence)
        self.assertResolution(result, 0.01, 0.01)

    def test_reports_mixed_components_and_structured_input(self) -> None:
        mixed = inspect_position("50.12257 N; 8° 39.942′ E")
        self.assertEqual(mixed.format, "mixed")
        self.assertEqual(mixed.component_formats, ("dd", "ddm"))

        structured = inspect_position((50.12257, 8.66570))
        self.assertIsNone(structured.original_text)
        self.assertEqual(structured.normalized_tokens, ())
        self.assertIsNone(structured.latitude_resolution)
        self.assertIsNone(structured.longitude_resolution)
        with self.assertRaisesRegex(CoordinateParseError, "Position input"):
            inspect_position(self.reference, format="nmea")

    def test_scientific_notation_resolution_is_exponent_aware(self) -> None:
        precise = inspect_position("5.012257e1, 8.66570")
        coarse = inspect_position("5e1, 8")

        self.assertResolution(precise, 1e-5, 1e-5)
        self.assertResolution(coarse, 10.0, 1.0)
        with self.assertRaises(CoordinateParseError):
            parse_position("50° 1e-100000′ N; 8° 0′ E")
        unsupported_fraction = "0" * 324
        with self.assertRaisesRegex(CoordinateParseError, "precision"):
            inspect_position(
                f"+00.{unsupported_fraction}+000.{unsupported_fraction}/"
            )
        with self.assertRaisesRegex(CoordinateParseError, "precision"):
            inspect_position(
                f"0000.{unsupported_fraction},N,"
                f"00000.{unsupported_fraction},E"
            )
        supported_fraction = "0" * 322 + "1"
        tiny = inspect_position(
            f"1° 0.{supported_fraction}′ N; "
            f"1° 0.{supported_fraction}′ E"
        )
        self.assertGreater(tiny.latitude_resolution or 0, 0)
        self.assertGreater(tiny.longitude_resolution or 0, 0)

        with self.assertRaisesRegex(CoordinateRangeError, "too small"):
            inspect_position(
                f"0° 0.{supported_fraction}′ N; "
                f"0° 0.{supported_fraction}′ E"
            )
        with self.assertRaisesRegex(CoordinateRangeError, "too small"):
            Position(Decimal("1e-1000"), 0)

    def test_dmm_alias_is_canonicalized_in_metadata(self) -> None:
        result = inspect_position(
            "50° 7.3542′ N; 8° 39.942′ E",
            format="dmm",
        )

        self.assertEqual(result.format, "ddm")
        self.assertIn("canonicalized format alias", result.normalizations[-1])
        self.assertEqual(result.warnings, ())

    def test_parse_result_is_immutable(self) -> None:
        result = inspect_position("50, 8")
        with self.assertRaises(FrozenInstanceError):
            result.format = "dms"  # type: ignore[misc]


class FormatPositionAllFormatsTests(unittest.TestCase):
    reference = Position(50.12257, 8.66570)

    def test_canonical_defaults(self) -> None:
        expected = {
            "dd": "50.122570, 8.665700",
            "ddm": "50° 7.3542′ N; 8° 39.9420′ E",
            "dms": "50° 7′ 21.25″ N; 8° 39′ 56.52″ E",
            "iso6709": "+50.122570+008.665700/",
            "nmea": "5007.3542,N,00839.9420,E",
        }
        for format_name, text in expected.items():
            with self.subTest(format=format_name):
                self.assertEqual(
                    format_position(self.reference, to=format_name),
                    text,
                )

    def test_negative_zero_and_extrema_outputs(self) -> None:
        negative = Position(-50.12257, -8.66570)
        zero = Position(-0.0, -0.0)
        extrema = Position(90, 180)
        expected_negative = {
            "dd": "-50.122570, -8.665700",
            "ddm": "50° 7.3542′ S; 8° 39.9420′ W",
            "dms": "50° 7′ 21.25″ S; 8° 39′ 56.52″ W",
            "iso6709": "-50.122570-008.665700/",
            "nmea": "5007.3542,S,00839.9420,W",
        }
        expected_zero = {
            "dd": "0.000000, 0.000000",
            "ddm": "0° 0.0000′ N; 0° 0.0000′ E",
            "dms": "0° 0′ 0.00″ N; 0° 0′ 0.00″ E",
            "iso6709": "+00.000000+000.000000/",
            "nmea": "0000.0000,N,00000.0000,E",
        }
        expected_extrema = {
            "dd": "90.000000, 180.000000",
            "ddm": "90° 0.0000′ N; 180° 0.0000′ E",
            "dms": "90° 0′ 0.00″ N; 180° 0′ 0.00″ E",
            "iso6709": "+90.000000+180.000000/",
            "nmea": "9000.0000,N,18000.0000,E",
        }
        for format_name in expected_negative:
            with self.subTest(format=format_name):
                self.assertEqual(
                    format_position(negative, to=format_name),
                    expected_negative[format_name],
                )
                self.assertEqual(
                    format_position(zero, to=format_name),
                    expected_zero[format_name],
                )
                self.assertEqual(
                    format_position(extrema, to=format_name),
                    expected_extrema[format_name],
                )

    def test_output_order_and_machine_format_constraints(self) -> None:
        expected = {
            "dd": "8.665700, 50.122570",
            "ddm": "8° 39.9420′ E; 50° 7.3542′ N",
            "dms": "8° 39′ 56.52″ E; 50° 7′ 21.25″ N",
            "nmea": "00839.9420,E,5007.3542,N",
        }
        for format_name, text in expected.items():
            with self.subTest(format=format_name):
                self.assertEqual(
                    format_position(
                        self.reference,
                        to=format_name,
                        order="lonlat",
                    ),
                    text,
                )
        with self.assertRaisesRegex(CoordinateParseError, "latitude/longitude"):
            format_position(self.reference, to="iso6709", order="lonlat")

    def test_notation_symbols_and_separators(self) -> None:
        self.assertEqual(
            format_position(
                self.reference,
                notation="hemisphere",
                symbols="ascii",
            ),
            "50.122570 deg N; 8.665700 deg E",
        )
        self.assertEqual(
            format_position(
                Position(-50.12257, -8.66570),
                to="ddm",
                notation="signed",
                symbols="ascii",
            ),
            "-50 deg 7.3542 min, -8 deg 39.9420 min",
        )
        self.assertEqual(
            format_position(self.reference, to="iso6709", compact=False),
            "+50.122570 +008.665700/",
        )
        self.assertEqual(
            format_position(
                self.reference,
                to="iso6709",
                compact=False,
                separator=",",
            ),
            "+50.122570,+008.665700/",
        )
        self.assertEqual(
            format_position(self.reference, to="nmea", separator="; "),
            "5007.3542 N; 00839.9420 E",
        )
        self.assertEqual(
            format_position(self.reference, to="dmm"),
            format_position(self.reference, to="ddm"),
        )

    def test_rounding_carries_without_invalid_sixty_fields(self) -> None:
        ddm = parse_position("12° 59.9996′ N; 8° 59.9996′ W")
        self.assertEqual(
            format_position(ddm, to="ddm", precision=3),
            "13° 0.000′ N; 9° 0.000′ W",
        )
        self.assertEqual(
            format_position(ddm, to="nmea", precision=3),
            "1300.000,N,00900.000,W",
        )

        dms = parse_position(
            "12° 34′ 59.9996″ N; 8° 59′ 59.9996″ W"
        )
        self.assertEqual(
            format_position(dms, to="dms", precision=3),
            "12° 35′ 0.000″ N; 9° 0′ 0.000″ W",
        )
        boundary = parse_position(
            "89° 59′ 59.9996″ N; 179° 59′ 59.9996″ E"
        )
        self.assertEqual(
            format_position(boundary, to="dms", precision=3),
            "90° 0′ 0.000″ N; 180° 0′ 0.000″ E",
        )

    def test_round_half_even_and_small_negative_zero(self) -> None:
        self.assertEqual(
            format_position(Position(1.2345645, 1.2345655), precision=6),
            "1.234564, 1.234566",
        )
        small = Position(-0.0000001, -0.0000001)
        for format_name in ("dd", "ddm", "dms", "iso6709", "nmea"):
            with self.subTest(format=format_name):
                text = format_position(small, to=format_name)
                self.assertNotIn("-", text)
                self.assertNotIn(",S", text)
                self.assertNotIn(",W", text)

    def test_rejects_invalid_or_inapplicable_options(self) -> None:
        invalid_calls = (
            {"to": "utm"},
            {"notation": "degrees"},
            {"symbols": "emoji"},
            {"to": "nmea", "notation": "signed"},
            {"to": "nmea", "symbols": "ascii"},
            {"to": "dd", "compact": True},
            {"separator": "|"},
            {"separator": []},
            {"to": "iso6709", "separator": " "},
            {"to": "nmea", "precision": 0},
        )
        for options in invalid_calls:
            with self.subTest(options=options):
                with self.assertRaises(CoordinateParseError):
                    format_position(self.reference, **options)  # type: ignore[arg-type]


class ConversionAndRoundTripTests(unittest.TestCase):
    reference = Position(50.12257, 8.66570)

    def test_conversion_infers_minimal_precision_from_dd_source(self) -> None:
        source = "50.12257, 8.66570"
        expected = {
            "dd": "50.12257, 8.66570",
            "ddm": "50° 7.3542′ N; 8° 39.9420′ E",
            "dms": "50° 7′ 21.25″ N; 8° 39′ 56.52″ E",
            "iso6709": "+50.12257+008.66570/",
            "nmea": "5007.3542,N,00839.9420,E",
        }
        for format_name, text in expected.items():
            with self.subTest(format=format_name):
                self.assertEqual(
                    convert_position(source, to=format_name),
                    text,
                )

    def test_conversion_uses_finer_axis_resolution_and_explicit_override(self) -> None:
        source = "50° 7′ 21.252″ N; 8° 39′ 56.52″ E"
        expected = {
            "dd": "50.1225700, 8.6657000",
            "ddm": "50° 7.35420′ N; 8° 39.94200′ E",
            "dms": "50° 7′ 21.252″ N; 8° 39′ 56.520″ E",
            "iso6709": "+50.1225700+008.6657000/",
            "nmea": "5007.35420,N,00839.94200,E",
        }
        for format_name, text in expected.items():
            with self.subTest(format=format_name):
                self.assertEqual(
                    convert_position(source, to=format_name),
                    text,
                )
        self.assertEqual(
            convert_position(source, to="dd", precision=3),
            "50.123, 8.666",
        )

    def test_input_and_output_orders_are_independent(self) -> None:
        self.assertEqual(
            convert_position(
                "8.66570, 50.12257",
                order="lonlat",
                output_order="latlon",
                precision=5,
            ),
            "50.12257, 8.66570",
        )
        self.assertEqual(
            convert_position(
                "8.66570, 50.12257",
                order="lonlat",
                output_order="lonlat",
                precision=5,
            ),
            "8.66570, 50.12257",
        )

    def test_conversion_does_not_silently_coarsen_beyond_precision_limit(self) -> None:
        source = "1.1234567890123456, 2.1234567890123456"
        with self.assertRaisesRegex(CoordinateParseError, "more than.*15"):
            convert_position(source)
        self.assertEqual(
            convert_position(source, precision=15),
            "1.123456789012346, 2.123456789012346",
        )

    def test_each_format_round_trips_with_documented_precision(self) -> None:
        settings = (
            ("dd", 5),
            ("ddm", 4),
            ("dms", 3),
            ("iso6709", 5),
            ("nmea", 4),
        )
        for format_name, precision in settings:
            with self.subTest(format=format_name):
                text = format_position(
                    self.reference,
                    to=format_name,
                    precision=precision,
                )
                self.assertEqual(parse_position(text), self.reference)
                self.assertEqual(
                    format_position(
                        parse_position(text),
                        to=format_name,
                        precision=precision,
                    ),
                    text,
                )

    def test_deterministic_round_trip_error_stays_within_output_quantum(self) -> None:
        generator = random.Random(20260722)
        settings = (
            ("dd", 6, 0.5e-6),
            ("ddm", 4, 0.5e-4 / 60),
            ("dms", 2, 0.5e-2 / 3600),
            ("iso6709", 6, 0.5e-6),
            ("nmea", 4, 0.5e-4 / 60),
        )
        for _ in range(50):
            position = Position(
                generator.uniform(-90, 90),
                generator.uniform(-180, 180),
            )
            for format_name, precision, tolerance in settings:
                with self.subTest(format=format_name, position=position):
                    text = format_position(
                        position,
                        to=format_name,
                        precision=precision,
                    )
                    parsed = parse_position(text)
                    self.assertLessEqual(
                        abs(parsed.latitude - position.latitude),
                        tolerance + 1e-14,
                    )
                    self.assertLessEqual(
                        abs(parsed.longitude - position.longitude),
                        tolerance + 1e-14,
                    )

    def test_decimal_context_does_not_change_results(self) -> None:
        with localcontext() as context:
            context.prec = 1
            context.Emax = 0
            context.Emin = 0
            for signal in context.traps:
                context.traps[signal] = True
            sources = (
                "50.12257, 8.66570",
                "50° 7.3542′ N; 8° 39.942′ E",
                "50° 7′ 21.252″ N; 8° 39′ 56.52″ E",
                "+50.12257+008.66570/",
                "5007.3542,N,00839.9420,E",
            )
            for source in sources:
                with self.subTest(source=source):
                    self.assertEqual(parse_position(source), self.reference)
            self.assertEqual(
                format_position(self.reference, precision=6),
                "50.122570, 8.665700",
            )
            for format_name in ("dd", "ddm", "dms", "iso6709", "nmea"):
                with self.subTest(format=format_name):
                    text = format_position(
                        self.reference,
                        to=format_name,
                        precision=15,
                    )
                    self.assertEqual(parse_position(text), self.reference)
            with self.assertRaisesRegex(CoordinateRangeError, "longitude"):
                parse_position("0, -180.1")
            with self.assertRaisesRegex(CoordinateRangeError, "longitude"):
                Position(0, Decimal("-180.1"))
            with self.assertRaisesRegex(CoordinateParseError, "exponent"):
                parse_position("0, 1e-100000000")


if __name__ == "__main__":
    unittest.main()
