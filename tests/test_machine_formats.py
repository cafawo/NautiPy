import unittest

from nautipy import (
    CoordinateParseError,
    CoordinateRangeError,
    Position,
    parse_position,
)


class Iso6709ParsingTests(unittest.TestCase):
    reference = Position(50.12257, 8.66570)

    def assertReference(self, value: str, **options: object) -> None:
        position = parse_position(value, **options)  # type: ignore[arg-type]
        self.assertAlmostEqual(position.latitude, self.reference.latitude, places=12)
        self.assertAlmostEqual(
            position.longitude,
            self.reference.longitude,
            places=12,
        )

    def test_parses_compact_and_separated_decimal_degree_forms(self) -> None:
        values = (
            "+50.12257+008.66570/",
            "+50.12257+008.66570",
            "+50.12257 +008.66570",
            "+50.12257,+008.66570",
            "+50.12257, +008.66570/",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertReference(value)

    def test_parses_fixed_width_ddm_and_dms_forms(self) -> None:
        values = (
            "+5007.3542+00839.9420/",
            "+500721.252+0083956.52/",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertReference(value)

        self.assertEqual(
            parse_position("−5007.3542−00839.9420/"),
            Position(-50.12257, -8.66570),
        )
        self.assertEqual(
            parse_position("+50.12257-008.66570/"),
            Position(50.12257, -8.66570),
        )
        self.assertEqual(
            parse_position("-50.12257+008.66570/"),
            Position(-50.12257, 8.66570),
        )

    def test_iso_order_is_defined_by_the_format(self) -> None:
        for order in ("latlon", "lonlat", "auto"):
            with self.subTest(order=order):
                self.assertReference(
                    "+50.12257+008.66570/",
                    order=order,
                )
        self.assertReference(
            "+50.12257+008.66570/",
            format="iso6709",
        )
        with self.assertRaises(CoordinateParseError):
            parse_position(
                "+50.12257+008.66570/",
                format="dd",
            )
        self.assertEqual(
            parse_position(
                "+50.12257 +008.66570",
                format="dd",
                order="lonlat",
            ),
            Position(8.66570, 50.12257),
        )

    def test_accepts_exact_extrema_in_each_iso_representation(self) -> None:
        values = (
            "+90+180/",
            "-90-180/",
            "+9000+18000/",
            "-9000-18000/",
            "+900000+1800000/",
            "-900000-1800000/",
        )
        for value in values:
            with self.subTest(value=value):
                expected = (
                    Position(-90, -180)
                    if value.startswith("-")
                    else Position(90, 180)
                )
                self.assertEqual(parse_position(value), expected)

    def test_rejects_iso_ranges_before_float_conversion(self) -> None:
        invalid = (
            "+90.000000000000001+000/",
            "+00+180.000000000000001/",
            "+9000.0001+18000/",
            "+8960+00800/",
            "+900000.0001+1800000/",
            "+895960+0080000/",
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(CoordinateRangeError):
                    parse_position(value)

    def test_rejects_malformed_or_extended_iso_input(self) -> None:
        invalid = (
            "+5.0+008.0/",
            "+05.0+08.0/",
            "+050.0+008.0/",
            "+50.+008.1/",
            "50.1+008.1/",
            "+50.1 008.1/",
            "+50e1+008.1/",
            "+50.1N+008.1E/",
            "+50.1+008.1//",
            "+50.1+008.1/CRS84",
            "+50.1+008.1+100/",
            "+50.1+008.1-100/",
            "+50.1,+008.1,100/",
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(CoordinateParseError):
                    parse_position(value, format="iso6709")

        with self.assertRaisesRegex(CoordinateParseError, "altitude|extra"):
            parse_position("+50.1+008.1+100/")


class NmeaParsingTests(unittest.TestCase):
    reference = Position(50.12257, 8.66570)

    def assertReference(self, value: str, **options: object) -> None:
        position = parse_position(value, **options)  # type: ignore[arg-type]
        self.assertAlmostEqual(position.latitude, self.reference.latitude, places=12)
        self.assertAlmostEqual(
            position.longitude,
            self.reference.longitude,
            places=12,
        )

    def test_parses_documented_nmea_pair_forms(self) -> None:
        values = (
            "5007.3542,N,00839.9420,E",
            "5007.3542 N; 00839.9420 E",
            "  5007.3542   n ; 00839.9420   e  ",
            "5007.3542 north; 00839.9420 east",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertReference(value)

    def test_nmea_axes_override_requested_order(self) -> None:
        for order in ("latlon", "lonlat", "auto"):
            with self.subTest(order=order):
                self.assertReference(
                    "5007.3542,N,00839.9420,E",
                    order=order,
                )
        self.assertReference(
            "5007.3542 N; 00839.9420 E",
            format="nmea",
        )
        self.assertReference(
            "00839.9420,E,5007.3542,N",
            order="auto",
        )
        self.assertReference(
            "00839.9420 E; 5007.3542 N",
            order="auto",
        )

    def test_nmea_specificity_prevents_decimal_degree_fallback(self) -> None:
        position = parse_position("0059.9999 N; 00059.9999 E")
        expected = 59.9999 / 60
        self.assertAlmostEqual(position.latitude, expected, places=12)
        self.assertAlmostEqual(position.longitude, expected, places=12)

    def test_parses_negative_directions_and_legal_extrema(self) -> None:
        self.assertEqual(
            parse_position("5007.3542,S,00839.9420,W"),
            Position(-50.12257, -8.66570),
        )
        self.assertEqual(
            parse_position("5007.3542,S,00839.9420,E"),
            Position(-50.12257, 8.66570),
        )
        self.assertEqual(
            parse_position("5007.3542,N,00839.9420,W"),
            Position(50.12257, -8.66570),
        )
        legal = (
            ("0000.0,N,00000.0,E", Position(0, 0)),
            ("9000.0000,N,18000.0000,E", Position(90, 180)),
            ("9000.0000,S,18000.0000,W", Position(-90, -180)),
        )
        for value, expected in legal:
            with self.subTest(value=value):
                self.assertEqual(parse_position(value), expected)

    def test_rejects_nmea_range_errors_without_fallback(self) -> None:
        invalid = (
            "5060.0000,N,00839.0000,E",
            "5007.0000,N,00860.0000,E",
            "9100.0000,N,00839.0000,E",
            "5007.0000,N,18100.0000,E",
            "9000.0000000001,N,00839.0000,E",
            "5007.0000,N,18000.0000000001,E",
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(CoordinateRangeError):
                    parse_position(value)

    def test_rejects_malformed_nmea_fields_and_sentences(self) -> None:
        invalid = (
            "507.3542,N,00839.9420,E",
            "50007.3542,N,00839.9420,E",
            "5007.3542,N,0839.9420,E",
            "5007.3542,N,000839.9420,E",
            "5007,N,00839.9420,E",
            "+5007.3542,N,00839.9420,E",
            "5007.3542,E,00839.9420,N",
            "5007.3542,ſ,00839.9420,E",
            "５００７.3542,N,00839.9420,E",
            "5007.3542,N,00839.9420,E,100",
            "$GPGGA,123519,5007.3542,N,00839.9420,E,1,08",
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(CoordinateParseError):
                    parse_position(value, format="nmea")

        with self.assertRaisesRegex(CoordinateParseError, "full NMEA"):
            parse_position("$GPRMC,5007.3542,N,00839.9420,E")
        with self.assertRaises(CoordinateParseError):
            parse_position(("5007.3542", "00839.9420"), format="nmea")


if __name__ == "__main__":
    unittest.main()
