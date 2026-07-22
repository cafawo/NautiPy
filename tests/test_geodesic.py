from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from nautipy import (
    CoordinateRangeError,
    InverseResult,
    NavigationError,
    Position,
    destination,
    distance,
    initial_bearing,
    interpolate,
    inverse,
    nearest_position,
)


class InverseGeodesicTests(unittest.TestCase):
    # Selected rows from GeographicLib's independently generated GeodTest
    # corpus. The source values are computed with 100-digit arithmetic:
    # https://geographiclib.sourceforge.io/C++/doc/geodesic.html#testgeod
    reference_cases = (
        (
            Position(36.530042355041, 0),
            Position(-48.16427077909777, 5.76234469467651),
            9_398_502.0434687,
            176.125875162171,
            175.3343083162854,
        ),
        (
            Position(8.226828747671, 0),
            Position(-8.516119211674269, 178.68897958262922),
            19_886_305.6710041,
            111.1269645725,
            68.98279854495524,
        ),
        (
            Position(19.707097385334, 0),
            Position(19.707739582810257, 0.000260256428101053),
            76.1478894,
            20.996796804557,
            20.996884567489088,
        ),
        (
            Position(89.997817345644, 0),
            Position(39.575446357285395, 177.375133897726),
            5_619_818.6599569,
            2.624783190362,
            179.99987006158308,
        ),
    )

    def test_inverse_matches_independent_wgs84_reference_cases(self) -> None:
        for start, end, expected_distance, expected_initial, expected_final in (
            self.reference_cases
        ):
            with self.subTest(start=start, end=end):
                result = inverse(start, end)

                self.assertAlmostEqual(
                    result.distance,
                    expected_distance,
                    delta=1e-6,
                )
                self.assertIsNotNone(result.initial_bearing)
                self.assertIsNotNone(result.final_bearing)
                self.assertAlmostEqual(
                    result.initial_bearing or 0,
                    expected_initial,
                    delta=5e-10,
                )
                self.assertAlmostEqual(
                    result.final_bearing or 0,
                    expected_final,
                    delta=5e-10,
                )
                self.assertGreaterEqual(result.initial_bearing or 0, 0)
                self.assertLess(result.initial_bearing or 0, 360)
                self.assertGreaterEqual(result.final_bearing or 0, 0)
                self.assertLess(result.final_bearing or 0, 360)
                direct = destination(
                    start,
                    bearing=result.initial_bearing,
                    distance=result.distance,
                )
                self.assertAlmostEqual(
                    direct.latitude,
                    end.latitude,
                    delta=1e-12,
                )
                self.assertAlmostEqual(
                    direct.longitude,
                    end.longitude,
                    delta=1e-12,
                )

    def test_antimeridian_reference_case(self) -> None:
        result = inverse(Position(0, 179), Position(0, -179))

        self.assertAlmostEqual(result.distance, 222_638.9815865471)
        self.assertEqual(result.initial_bearing, 90)
        self.assertEqual(result.final_bearing, 90)
        self.assertAlmostEqual(
            destination(
                Position(0, 179),
                bearing=90,
                distance=222_638.9815865471,
            ).longitude,
            -179,
            delta=1e-12,
        )

    def test_exact_antipode_uses_documented_canonical_geodesic(self) -> None:
        start = Position(0, 0)
        end = Position(0, 180)
        result = inverse(start, end)

        self.assertAlmostEqual(result.distance, 20_003_931.458625447)
        self.assertEqual(result.initial_bearing, 0.0)
        self.assertEqual(result.final_bearing, 180.0)
        midpoint = interpolate(start, end)
        self.assertGreater(midpoint.latitude, 89.999999999)
        self.assertEqual(midpoint.longitude, 180.0)

    def test_result_model_is_immutable_and_wrappers_share_it(self) -> None:
        start, end, expected_distance, expected_initial, _ = (
            self.reference_cases[0]
        )
        result = inverse(start, end)

        self.assertIsInstance(result, InverseResult)
        self.assertAlmostEqual(distance(start, end), expected_distance)
        self.assertAlmostEqual(
            initial_bearing(start, end),
            expected_initial,
        )
        with self.assertRaises(FrozenInstanceError):
            result.distance = 0  # type: ignore[misc]

    def test_coincident_physical_positions_have_no_bearing(self) -> None:
        result = inverse("0, 180", "0, -180")

        self.assertEqual(result.distance, 0.0)
        self.assertIsNone(result.initial_bearing)
        self.assertIsNone(result.final_bearing)
        self.assertEqual(distance("0, 180", "0, -180"), 0.0)
        with self.assertRaisesRegex(NavigationError, "coincident"):
            initial_bearing("0, 180", "0, -180")

        for latitude in (-90, 90):
            with self.subTest(latitude=latitude):
                polar = inverse(
                    Position(latitude, -123),
                    Position(latitude, 45),
                )
                self.assertEqual(polar, InverseResult(0.0, None, None))

    def test_distinct_sub_resolution_positions_raise(self) -> None:
        with self.assertRaisesRegex(NavigationError, "numerical resolution"):
            inverse(Position(0, 0), Position(1e-20, 0))

    def test_tiny_negative_azimuth_normalizes_to_zero_not_360(self) -> None:
        result = inverse(Position(0, 0), Position(1, -1e-17))

        self.assertEqual(result.initial_bearing, 0.0)
        self.assertGreaterEqual(result.final_bearing or 0, 0)
        self.assertLess(result.final_bearing or 0, 360)

    def test_all_position_inputs_use_coordinate_coercion(self) -> None:
        expected = inverse(
            Position(50.12257, 8.66570),
            Position(51.0, 9.0),
        )
        actual = inverse(
            "50° 7.3542' N; 8° 39.942' E",
            {"type": "Point", "coordinates": [9.0, 51.0]},
        )

        self.assertEqual(actual, expected)
        with self.assertRaises(CoordinateRangeError):
            distance((91, 0), (0, 0))


class DirectGeodesicTests(unittest.TestCase):
    def test_destination_matches_independent_reference(self) -> None:
        result = destination(
            Position(36.530042355041, 0),
            bearing=176.125875162171,
            distance=9_398_502.0434687,
        )

        self.assertAlmostEqual(
            result.latitude,
            -48.16427077909777,
            delta=1e-12,
        )
        self.assertAlmostEqual(
            result.longitude,
            5.76234469467651,
            delta=1e-12,
        )

    def test_bearing_wraps_and_zero_distance_preserves_start(self) -> None:
        start = Position(36.530042355041, 0)

        self.assertEqual(
            destination(start, bearing=-183.874124837829, distance=10_000),
            destination(start, bearing=176.125875162171, distance=10_000),
        )
        self.assertIs(
            destination(start, bearing=12_345, distance=0),
            start,
        )
        self.assertEqual(
            destination("0, 180", bearing=90, distance=0),
            Position(0, 180),
        )

    def test_large_exact_bearings_are_reduced_before_float_conversion(self) -> None:
        start = Position(0, 0)
        expected = destination(start, bearing=33, distance=1_000_000)
        values = (
            2**53 + 1,
            Decimal(2**53 + 1),
            Fraction(2**53 + 1),
            360 * 10**100 + 33,
            Decimal(360 * 10**100 + 33),
            Fraction(360 * 10**100 + 33),
            -(360 * 10**100) + 33,
            Decimal(-(360 * 10**100) + 33),
            Fraction(-(360 * 10**100) + 33),
        )
        for bearing in values:
            with self.subTest(bearing=bearing):
                self.assertEqual(
                    destination(
                        start,
                        bearing=bearing,
                        distance=1_000_000,
                    ),
                    expected,
                )
        self.assertEqual(
            destination(
                start,
                bearing=Decimal("1e1000"),
                distance=1_000_000,
            ),
            destination(start, bearing=280, distance=1_000_000),
        )
        for bearing in (
            Decimal("1e-1000"),
            Decimal("-1e-1000"),
            Fraction(1, 10**1000),
            Fraction(-1, 10**1000),
        ):
            with self.subTest(tiny_bearing=bearing):
                self.assertEqual(
                    destination(start, bearing=bearing, distance=1_000_000),
                    destination(start, bearing=0, distance=1_000_000),
                )

    def test_destination_accepts_real_scalars(self) -> None:
        result = destination(
            Position(0, 0),
            bearing=Decimal("90"),
            distance=Fraction(1_000),
        )

        self.assertAlmostEqual(distance(Position(0, 0), result), 1_000)
        self.assertAlmostEqual(initial_bearing(Position(0, 0), result), 90)
        for minimum in (1e-7, Decimal("1e-7"), Fraction(1, 10_000_000)):
            with self.subTest(minimum=minimum):
                resolved_minimum = destination(
                    Position(0, 0),
                    bearing=0,
                    distance=minimum,
                )
                self.assertGreater(resolved_minimum.latitude, 0)

    def test_destination_rejects_invalid_navigation_scalars(self) -> None:
        invalid_bearings = (True, "90", math.nan, math.inf)
        for bearing in invalid_bearings:
            with self.subTest(bearing=bearing):
                with self.assertRaises(NavigationError):
                    destination(Position(0, 0), bearing=bearing, distance=1)

        invalid_distances = (
            True,
            "1",
            -1,
            1e-20,
            Decimal("0.0000000999999999999999999999999999"),
            Fraction(999_999_999_999_999, 10**22),
            math.nan,
            math.inf,
            Decimal("1e-1000"),
        )
        for value in invalid_distances:
            with self.subTest(distance=value):
                with self.assertRaises(NavigationError):
                    destination(Position(0, 0), bearing=0, distance=value)


class InterpolationTests(unittest.TestCase):
    start = Position(36.530042355041, 0)
    end = Position(-48.16427077909777, 5.76234469467651)

    def test_default_is_geodesic_midpoint(self) -> None:
        midpoint = interpolate(self.start, self.end)
        total = distance(self.start, self.end)

        self.assertAlmostEqual(midpoint.latitude, -5.850859318228765)
        self.assertAlmostEqual(midpoint.longitude, 2.614146476229401)
        self.assertAlmostEqual(distance(self.start, midpoint), total / 2)
        self.assertAlmostEqual(
            distance(midpoint, self.end),
            total / 2,
        )

    def test_interpolation_matches_official_dateline_waypoint(self) -> None:
        # The official waypoint example reports the point 5000 km along this
        # geodesic as 58.43499 N, 183.03167 E (unrolled longitude).
        beijing = Position(40.1, 116.6)
        san_francisco = Position(37.6, -122.4)
        total = distance(beijing, san_francisco)
        waypoint = interpolate(
            beijing,
            san_francisco,
            fraction=5_000_000 / total,
        )

        self.assertAlmostEqual(waypoint.latitude, 58.43499, delta=5e-6)
        self.assertAlmostEqual(waypoint.longitude, -176.96833, delta=5e-6)

    def test_endpoints_are_preserved_exactly(self) -> None:
        self.assertIs(
            interpolate(self.start, self.end, fraction=0),
            self.start,
        )
        self.assertIs(
            interpolate(self.start, self.end, fraction=1),
            self.end,
        )
        self.assertEqual(
            interpolate("0, 180", "0, -180", fraction=Decimal("0.5")),
            Position(0, 180),
        )

    def test_fraction_must_be_finite_and_inside_segment(self) -> None:
        invalid = (
            True,
            "0.5",
            -0.1,
            1.1,
            math.nan,
            math.inf,
            Decimal("1.0000000000000000001"),
            Decimal("0.99999999999999999999"),
            Decimal("1e-1000"),
        )
        for fraction in invalid:
            with self.subTest(fraction=fraction):
                with self.assertRaises(NavigationError):
                    interpolate(
                        self.start,
                        self.end,
                        fraction=fraction,
                    )

    def test_distinct_sub_resolution_segment_raises(self) -> None:
        with self.assertRaisesRegex(NavigationError, "numerical resolution"):
            interpolate(
                Position(0, 0),
                Position(1e-20, 0),
                fraction=0.5,
            )
        with self.assertRaisesRegex(NavigationError, "1e-7 metres"):
            interpolate(
                Position(0, 0),
                Position(0, 1),
                fraction=1e-15,
            )


class NearestPositionTests(unittest.TestCase):
    def test_returns_first_nearest_position_from_one_pass_iterable(self) -> None:
        origin = Position(0, 0)
        first_tie = {"lat": 0, "lon": 1}
        candidates = (
            candidate
            for candidate in (
                "0, 10",
                first_tie,
                {"type": "Point", "coordinates": [-1, 0]},
            )
        )

        self.assertEqual(
            nearest_position(origin, candidates),
            Position(0, 1),
        )

    def test_empty_and_non_collection_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(NavigationError, "at least one"):
            nearest_position(Position(0, 0), iter(()))
        with self.assertRaisesRegex(NavigationError, "iterable"):
            nearest_position(Position(0, 0), "0, 1")

    def test_invalid_candidate_errors_are_not_hidden(self) -> None:
        with self.assertRaises(CoordinateRangeError):
            nearest_position(Position(0, 0), [(0, 1), (91, 0)])


class ImportBoundaryTests(unittest.TestCase):
    def test_top_level_coordinate_use_does_not_load_geographiclib(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        source = str(repository / "src")
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source if not existing else source + os.pathsep + existing
        )
        code = (
            "import sys; import nautipy; "
            "assert 'nautipy.geodesic' not in sys.modules; "
            "assert not any(name == 'geographiclib' or "
            "name.startswith('geographiclib.') for name in sys.modules); "
            "nautipy.parse_position('50.12257, 8.66570'); "
            "assert 'nautipy.geodesic' not in sys.modules; "
            "assert not any(name == 'geographiclib' or "
            "name.startswith('geographiclib.') for name in sys.modules)"
        )
        with TemporaryDirectory(prefix="nautipy-import-") as directory:
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
