from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

from nautipy import Position, destination, distance, initial_bearing
from nautipy.errors import FixError, NavigationError
from nautipy.fix import (
    BearingObservation,
    CandidateStatus,
    FixStatus,
    RangeObservation,
    solve_fix,
    two_bearing_candidates,
    two_range_candidates,
)


SCIENTIFIC_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("geographiclib", "numpy", "scipy")
)


@unittest.skipUnless(
    SCIENTIFIC_AVAILABLE,
    'solver tests require python -m pip install "nautipy[fix]"',
)
class FixSolverTests(unittest.TestCase):
    target = Position(0.01, 0.01)
    stations = (
        Position(0.0, 0.0),
        Position(0.0, 0.02),
        Position(0.02, 0.0),
    )

    def bearings(
        self,
        target: Position | None = None,
        stations: tuple[Position, ...] | None = None,
        uncertainty: float = 0.1,
    ) -> tuple[BearingObservation, ...]:
        selected_target = target or self.target
        selected_stations = stations or self.stations
        return tuple(
            BearingObservation(
                station,
                initial_bearing(selected_target, station),
                uncertainty,
            )
            for station in selected_stations
        )

    def ranges(
        self,
        target: Position | None = None,
        stations: tuple[Position, ...] | None = None,
        uncertainty: float = 1.0,
    ) -> tuple[RangeObservation, ...]:
        selected_target = target or self.target
        selected_stations = stations or self.stations
        return tuple(
            RangeObservation(
                station,
                distance(selected_target, station),
                uncertainty,
            )
            for station in selected_stations
        )

    def assertNear(
        self,
        actual: Position,
        expected: Position,
        tolerance_metres: float,
    ) -> None:
        try:
            separation = distance(actual, expected)
        except NavigationError as error:
            if str(error) != (
                "positions are distinct but below WGS84 numerical resolution"
            ):
                raise
            separation = 0.0
        self.assertLessEqual(separation, tolerance_metres)

    def test_wrapped_bearing_residual_has_the_short_signed_difference(self) -> None:
        from nautipy._fix_solver import _wrap_bearing

        self.assertAlmostEqual(_wrap_bearing(0.1 - 359.9), 0.2)
        self.assertAlmostEqual(_wrap_bearing(359.9 - 0.1), -0.2)
        self.assertEqual(_wrap_bearing(180.0), -180.0)

    def test_exact_bearing_range_and_mixed_fixes(self) -> None:
        cases = (
            {"bearings": self.bearings()},
            {"ranges": self.ranges()},
            {"bearings": self.bearings(), "ranges": self.ranges()},
        )
        for arguments in cases:
            with self.subTest(arguments=tuple(arguments)):
                result = solve_fix(**arguments, search_radius=10_000)
                self.assertTrue(result.success, result)
                self.assertIs(result.status, FixStatus.CONVERGED)
                self.assertNear(result.position, self.target, 1e-5)
                self.assertEqual(result.rank, 2)
                self.assertGreaterEqual(result.condition_number, 1.0)
                self.assertIsNotNone(result.uncertainty)
                self.assertGreater(result.iterations, 0)
                self.assertGreater(result.function_evaluations, 0)
                self.assertIsInstance(result.objective, float)
                self.assertTrue(
                    all(
                        isinstance(value, float)
                        for row in result.uncertainty.covariance
                        for value in row
                    )
                )

    def test_exact_bearing_range_pair_retains_both_distant_roots(self) -> None:
        target = Position(45, 20)
        alternative = Position(43.131187048495285, 22.00466817261492)
        bearing = BearingObservation(
            Position(37.57332997765144, 28.25721503487552),
            137.21510448566713,
            0.01,
        )
        range_observation = RangeObservation(
            Position(46.37348707094291, 25.481434622327143),
            453_349.02189037704,
            1,
        )
        result = solve_fix(
            bearings=(bearing,),
            ranges=(range_observation,),
            search_radius=1_900_000,
        )
        self.assertIs(result.status, FixStatus.AMBIGUOUS)
        self.assertEqual(len(result.competing_positions), 2)
        for expected in (target, alternative):
            self.assertTrue(
                any(
                    distance(expected, position) <= 1
                    for position in result.competing_positions
                )
            )

    def test_historical_frankfurt_unknown_to_reference_fixture(self) -> None:
        # Frozen independently from the historical README at commit c55d9a3.
        # Its spherical/rounded observations intentionally differ slightly
        # from current GeographicLib predictions at the documented target.
        target = Position(50.127198, 8.665562)
        references = (
            Position(50.116135, 8.670277),
            Position(50.112836, 8.666753),
            Position(50.110347, 8.659873),
        )
        bearing_candidates = two_bearing_candidates(
            BearingObservation(references[0], 164.71, 0.1),
            BearingObservation(references[2], 192.22, 0.1),
            search_center=target,
            search_radius=10_000,
        )
        self.assertIs(bearing_candidates.status, CandidateStatus.UNIQUE)
        self.assertNear(bearing_candidates.positions[0], target, 5.0)

        ranges = tuple(
            RangeObservation(reference, measured, 1.0)
            for reference, measured in zip(
                references,
                (1275.251, 1599.237, 1917.145),
            )
        )
        range_result = solve_fix(ranges=ranges, search_radius=10_000)
        self.assertTrue(range_result.success, range_result)
        self.assertNear(range_result.position, target, 2.0)

    def test_reciprocal_bearings_are_not_silently_accepted(self) -> None:
        target = Position(50.127198, 8.665562)
        first = Position(50.116135, 8.670277)
        second = Position(50.110347, 8.659873)
        result = two_bearing_candidates(
            BearingObservation(first, initial_bearing(first, target), 0.1),
            BearingObservation(second, initial_bearing(second, target), 0.1),
            search_center=target,
            search_radius=10_000,
        )
        self.assertIs(result.status, CandidateStatus.NO_SOLUTION)
        self.assertEqual(result.positions, ())

    def test_two_bearings_use_exact_wgs84_geometry_beyond_planar_scale(self) -> None:
        target = Position(45, 20)
        references = (
            Position(43.53977174920387, 36.32641169732056),
            Position(40.47305757839476, 7.949152459594256),
        )
        observations = tuple(
            BearingObservation(
                reference,
                initial_bearing(target, reference),
                0.1,
            )
            for reference in references
        )
        result = two_bearing_candidates(
            *observations,
            search_center=target,
            search_radius=1_900_000,
        )
        self.assertIs(result.status, CandidateStatus.UNIQUE)
        self.assertNear(result.positions[0], target, 0.01)

    def test_candidate_geometry_is_invariant_to_observation_uncertainty(self) -> None:
        target = Position(0.01, 0.01)
        references = (Position(0, 0), Position(0, 0.02))

        def bearing_candidates(uncertainties: tuple[float, float]):
            observations = tuple(
                BearingObservation(
                    reference,
                    initial_bearing(target, reference),
                    uncertainty,
                )
                for reference, uncertainty in zip(references, uncertainties)
            )
            return two_bearing_candidates(
                *observations,
                search_center=target,
                search_radius=10_000,
            )

        equal_bearings = bearing_candidates((1, 1))
        skewed_bearings = bearing_candidates((1e-3, 1e3))
        self.assertIs(equal_bearings.status, CandidateStatus.UNIQUE)
        self.assertEqual(skewed_bearings.status, equal_bearings.status)
        self.assertNear(
            skewed_bearings.positions[0],
            equal_bearings.positions[0],
            0.001,
        )

        def range_candidates(uncertainties: tuple[float, float]):
            observations = tuple(
                RangeObservation(
                    reference,
                    distance(target, reference),
                    uncertainty,
                )
                for reference, uncertainty in zip(references, uncertainties)
            )
            return two_range_candidates(*observations)

        equal_ranges = range_candidates((1, 1))
        skewed_ranges = range_candidates((1e-3, 1e3))
        self.assertIs(equal_ranges.status, CandidateStatus.AMBIGUOUS)
        self.assertEqual(skewed_ranges.status, equal_ranges.status)
        self.assertEqual(len(skewed_ranges.positions), len(equal_ranges.positions))
        for position in equal_ranges.positions:
            self.assertTrue(
                any(
                    distance(position, other) <= 0.001
                    for other in skewed_ranges.positions
                )
            )

    def test_parallel_and_same_reference_bearing_geometry(self) -> None:
        parallel = two_bearing_candidates(
            BearingObservation(Position(0, 0), 0, 0.1),
            BearingObservation(Position(0, 0.02), 0, 0.1),
            search_center=Position(0, 0.01),
            search_radius=10_000,
        )
        self.assertIs(parallel.status, CandidateStatus.DEGENERATE)

        same = Position(0, 0)
        equivalent = two_bearing_candidates(
            BearingObservation(same, 12, 0.1),
            BearingObservation(same, 12, 0.1),
            search_radius=10_000,
        )
        inconsistent = two_bearing_candidates(
            BearingObservation(same, 12, 0.1),
            BearingObservation(same, 13, 0.1),
            search_radius=10_000,
        )
        self.assertIs(equivalent.status, CandidateStatus.DEGENERATE)
        self.assertIs(inconsistent.status, CandidateStatus.NO_SOLUTION)

    def test_exact_effectively_parallel_bearings_are_degenerate(self) -> None:
        target = Position(0, 0)
        observations = (
            BearingObservation(
                destination(target, bearing=0, distance=1_000),
                0,
                0.1,
            ),
            BearingObservation(
                destination(target, bearing=0.0001, distance=2_000),
                0.0001,
                0.1,
            ),
        )
        result = two_bearing_candidates(
            *observations,
            search_center=target,
            search_radius=5_000,
        )
        self.assertIs(result.status, CandidateStatus.DEGENERATE)
        self.assertEqual(result.positions, ())
        self.assertIn("unstable", " ".join(result.warnings))

    def test_two_range_zero_one_two_and_concentric_geometry(self) -> None:
        first = Position(0, -0.01)
        second = Position(0, 0.01)
        positive = Position(0.01, 0)
        symmetric = two_range_candidates(
            RangeObservation(first, distance(positive, first), 1),
            RangeObservation(second, distance(positive, second), 1),
        )
        self.assertIs(symmetric.status, CandidateStatus.AMBIGUOUS)
        self.assertEqual(len(symmetric.positions), 2)
        self.assertTrue(
            any(position.latitude > 0 for position in symmetric.positions)
        )
        self.assertTrue(
            any(position.latitude < 0 for position in symmetric.positions)
        )

        separated = distance(first, second)
        no_solution = two_range_candidates(
            RangeObservation(first, 100, 1),
            RangeObservation(second, 100, 1),
        )
        tangent = two_range_candidates(
            RangeObservation(first, separated / 2, 1),
            RangeObservation(second, separated / 2, 1),
        )
        zero_radius = two_range_candidates(
            RangeObservation(first, 0, 1),
            RangeObservation(second, separated, 1),
        )
        concentric = two_range_candidates(
            RangeObservation(first, 100, 1),
            RangeObservation(first, 100, 1),
        )
        concentric_mismatch = two_range_candidates(
            RangeObservation(first, 100, 1),
            RangeObservation(first, 200, 1),
        )
        coincident_zero_observations = (
            RangeObservation(first, 0, 1),
            RangeObservation(first, 0, 1),
        )
        coincident_zero = two_range_candidates(*coincident_zero_observations)
        coincident_zero_fix = solve_fix(
            ranges=coincident_zero_observations,
            search_center=first,
            search_radius=1_000,
        )
        self.assertIs(no_solution.status, CandidateStatus.NO_SOLUTION)
        self.assertIs(tangent.status, CandidateStatus.UNIQUE)
        self.assertEqual(len(tangent.positions), 1)
        self.assertTrue(tangent.warnings)
        self.assertIs(zero_radius.status, CandidateStatus.UNIQUE)
        self.assertNear(zero_radius.positions[0], first, 1e-6)
        self.assertIs(concentric.status, CandidateStatus.DEGENERATE)
        self.assertIs(
            concentric_mismatch.status,
            CandidateStatus.NO_SOLUTION,
        )
        self.assertIs(coincident_zero.status, CandidateStatus.UNIQUE)
        self.assertEqual(coincident_zero.positions, (first,))
        self.assertIs(coincident_zero_fix.status, FixStatus.DEGENERATE)
        self.assertFalse(coincident_zero_fix.success)
        self.assertIsNone(coincident_zero_fix.position)

    def test_two_range_candidate_scope_is_rejected_as_unsupported_input(self) -> None:
        reference = Position(0, 0)
        with self.assertRaisesRegex(FixError, "2000000"):
            two_range_candidates(
                RangeObservation(reference, 2_000_001, 1),
                RangeObservation(reference, 0, 1),
            )

    def test_solver_accepts_ranges_beyond_candidate_construction_scope(self) -> None:
        target = Position(0, 0)
        references = tuple(
            destination(target, bearing=bearing, distance=2_100_000)
            for bearing in (240, 120, 0)
        )
        observations = tuple(
            RangeObservation(reference, distance(target, reference), 1)
            for reference in references
        )
        for count in (2, 3):
            with self.subTest(count=count):
                result = solve_fix(
                    ranges=observations[:count],
                    search_center=target,
                    search_radius=100_000,
                )
                self.assertTrue(result.success, result)
                self.assertIs(result.status, FixStatus.CONVERGED)
                self.assertNear(result.position, target, 0.001)

    def test_submetre_range_alternatives_are_not_clustered_together(self) -> None:
        center = Position(0, 0)
        references = (
            destination(center, bearing=270, distance=1_000),
            destination(center, bearing=90, distance=1_000),
        )
        target = destination(center, bearing=0, distance=0.4)
        observations = tuple(
            RangeObservation(reference, distance(target, reference), 0.01)
            for reference in references
        )
        candidates = two_range_candidates(*observations)
        result = solve_fix(
            ranges=observations,
            search_center=center,
            search_radius=100,
        )
        self.assertIs(candidates.status, CandidateStatus.AMBIGUOUS)
        self.assertEqual(len(candidates.positions), 2)
        self.assertGreater(distance(*candidates.positions), 0.79)
        self.assertIs(result.status, FixStatus.AMBIGUOUS)
        self.assertEqual(len(result.competing_positions), 2)

    def test_large_range_network_preserves_small_circle_intersections(self) -> None:
        center = Position(0, 0)
        references = (
            destination(center, bearing=270, distance=1_000_000),
            destination(center, bearing=90, distance=1_000_000),
        )
        north = destination(center, bearing=0, distance=0.05)
        south = destination(center, bearing=180, distance=0.05)
        observations = tuple(
            RangeObservation(reference, distance(north, reference), 0.01)
            for reference in references
        )
        candidates = two_range_candidates(*observations)
        result = solve_fix(
            ranges=observations,
            search_center=center,
            search_radius=10,
        )
        self.assertIs(candidates.status, CandidateStatus.AMBIGUOUS)
        self.assertEqual(len(candidates.positions), 2)
        self.assertGreater(distance(*candidates.positions), 0.09)
        for expected in (north, south):
            self.assertTrue(
                any(
                    distance(expected, position) <= 0.01
                    for position in candidates.positions
                )
            )
        self.assertIs(result.status, FixStatus.AMBIGUOUS)
        self.assertEqual(len(result.competing_positions), 2)

    def test_two_range_solve_preserves_ambiguity_and_search_disk(self) -> None:
        first = Position(0, -0.01)
        second = Position(0, 0.01)
        positive = Position(0.01, 0)
        observations = (
            RangeObservation(first, distance(positive, first), 1),
            RangeObservation(second, distance(positive, second), 1),
        )
        ambiguous = solve_fix(ranges=observations, search_radius=5_000)
        selected = solve_fix(
            ranges=observations,
            initial=positive,
            search_center=positive,
            search_radius=500,
        )
        excluded = solve_fix(
            ranges=observations,
            search_center=Position(0.05, 0),
            search_radius=100,
        )
        self.assertIs(ambiguous.status, FixStatus.AMBIGUOUS)
        self.assertFalse(ambiguous.success)
        self.assertIsNone(ambiguous.position)
        self.assertEqual(len(ambiguous.competing_positions), 2)
        self.assertTrue(selected.success, selected)
        self.assertNear(selected.position, positive, 1e-5)
        self.assertIs(excluded.status, FixStatus.NO_SOLUTION)

    def test_tangent_range_fix_is_explicitly_degenerate(self) -> None:
        first = Position(0, 0)
        second = Position(0, 0.02)
        separation = distance(first, second)
        result = solve_fix(
            ranges=(
                RangeObservation(first, separation / 2, 1),
                RangeObservation(second, separation / 2, 1),
            ),
            search_center=Position(0, 0.01),
            search_radius=5_000,
        )
        self.assertIs(result.status, FixStatus.DEGENERATE)
        self.assertFalse(result.success)
        self.assertIsNone(result.position)
        self.assertEqual(result.rank, 1)
        self.assertIsNone(result.uncertainty)

    def test_uncertainty_weight_moves_an_inconsistent_fix(self) -> None:
        stations = self.stations + (Position(0.02, 0.02),)
        exact = tuple(distance(self.target, station) for station in stations)

        def solve(biased_uncertainty: float):
            observations = (
                RangeObservation(
                    stations[0],
                    exact[0] + 200,
                    biased_uncertainty,
                ),
                *(
                    RangeObservation(station, measured, 10)
                    for station, measured in zip(stations[1:], exact[1:])
                ),
            )
            return solve_fix(ranges=observations, search_radius=10_000)

        high_weight = solve(1)
        low_weight = solve(1_000)
        self.assertTrue(high_weight.success, high_weight)
        self.assertTrue(low_weight.success, low_weight)
        self.assertGreater(
            distance(high_weight.position, self.target),
            distance(low_weight.position, self.target) * 100,
        )
        self.assertIn("residuals are large", " ".join(high_weight.warnings))

    def test_common_uncertainty_scale_preserves_fix_and_scales_covariance(self) -> None:
        target = Position(0.011, 0.013)
        stations = (
            Position(0, 0),
            Position(0, 0.03),
            Position(0.03, 0),
            Position(0.025, 0.028),
        )
        factor = 8.5

        def solve(scale: float):
            return solve_fix(
                bearings=self.bearings(target, stations[:2], 0.2 * scale),
                ranges=self.ranges(target, stations[2:], 3.0 * scale),
                search_radius=10_000,
            )

        baseline = solve(1.0)
        scaled = solve(factor)
        self.assertTrue(baseline.success, baseline)
        self.assertTrue(scaled.success, scaled)
        self.assertNear(scaled.position, baseline.position, 1e-5)

        for baseline_row, scaled_row in zip(
            baseline.uncertainty.covariance,
            scaled.uncertainty.covariance,
        ):
            for baseline_value, scaled_value in zip(
                baseline_row,
                scaled_row,
            ):
                expected = baseline_value * factor**2
                self.assertAlmostEqual(
                    scaled_value,
                    expected,
                    delta=max(1e-9, abs(expected) * 1e-8),
                )

        for attribute in (
            "east_standard_deviation",
            "north_standard_deviation",
            "semi_major_95",
            "semi_minor_95",
        ):
            expected = getattr(baseline.uncertainty, attribute) * factor
            actual = getattr(scaled.uncertainty, attribute)
            self.assertAlmostEqual(
                actual,
                expected,
                delta=max(1e-9, abs(expected) * 1e-8),
            )
        self.assertAlmostEqual(
            scaled.uncertainty.correlation,
            baseline.uncertainty.correlation,
            places=9,
        )
        self.assertAlmostEqual(
            scaled.uncertainty.major_axis_bearing,
            baseline.uncertainty.major_axis_bearing,
            places=7,
        )

    def test_nonconvergence_is_returned_without_a_position(self) -> None:
        target = Position(0.011, 0.013)
        stations = (
            Position(0, 0),
            Position(0, 0.03),
            Position(0.03, 0),
        )
        observations = tuple(
            BearingObservation(
                station,
                round(initial_bearing(target, station), 1),
                0.1,
            )
            for station in stations
        )
        result = solve_fix(
            bearings=observations,
            initial=Position(0.02, 0.02),
            search_radius=10_000,
            max_iterations=1,
        )
        self.assertIs(result.status, FixStatus.NON_CONVERGED)
        self.assertFalse(result.success)
        self.assertIsNone(result.position)
        self.assertIsNone(result.uncertainty)
        self.assertLessEqual(result.iterations, 1)

    def test_exact_fixes_on_cardinal_and_diagonal_search_edges_converge(self) -> None:
        center = Position(0, 0)
        for heading in range(0, 360, 45):
            with self.subTest(heading=heading):
                target = destination(
                    center,
                    bearing=heading,
                    distance=10_000,
                )
                references = tuple(
                    destination(target, bearing=bearing, distance=1_500)
                    for bearing in (20, 140, 260)
                )
                result = solve_fix(
                    ranges=self.ranges(target, references),
                    search_center=center,
                    search_radius=10_000,
                )
                self.assertTrue(result.success, result)
                self.assertNear(result.position, target, 0.001)
                self.assertIn("near the edge", " ".join(result.warnings))

    def test_out_of_disk_optima_are_not_projected_to_the_boundary(self) -> None:
        center = Position(0, 0)
        for heading in (0, 45):
            with self.subTest(heading=heading):
                target = destination(
                    center,
                    bearing=heading,
                    distance=11_000,
                )
                references = tuple(
                    destination(target, bearing=bearing, distance=1_500)
                    for bearing in (20, 140, 260)
                )
                result = solve_fix(
                    ranges=self.ranges(target, references),
                    search_center=center,
                    search_radius=10_000,
                )
                self.assertIs(result.status, FixStatus.NO_SOLUTION)
                self.assertFalse(result.success)
                self.assertIsNone(result.position)
                self.assertTrue(result.residuals)
                self.assertIn("not projected", " ".join(result.warnings))

    def test_explicit_initial_seed_is_always_run_before_generated_seeds(self) -> None:
        from nautipy import _fix_solver

        target = Position(0, 0)
        initial = destination(target, bearing=45, distance=9_000)
        references = tuple(
            destination(target, bearing=index * 18, distance=1_000)
            for index in range(20)
        )
        starts: list[tuple[float, float]] = []

        def record_start(specs, chart, start, **kwargs):
            starts.append(start)
            return None

        with patch.object(_fix_solver, "_optimize", side_effect=record_start):
            result = solve_fix(
                ranges=self.ranges(target, references),
                initial=initial,
                search_center=target,
                search_radius=10_000,
            )

        expected = _fix_solver._Chart(target).to_local(initial)
        self.assertIs(result.status, FixStatus.NON_CONVERGED)
        self.assertEqual(len(starts), 32)
        self.assertLessEqual(
            distance(
                _fix_solver._Chart(target).to_position(*starts[0]),
                _fix_solver._Chart(target).to_position(*expected),
            ),
            0.001,
        )

    def test_one_shot_observation_iterables_are_consumed_once(self) -> None:
        result = solve_fix(
            bearings=(observation for observation in self.bearings()),
            ranges=(observation for observation in self.ranges()),
            search_radius=10_000,
        )
        self.assertTrue(result.success, result)
        self.assertNear(result.position, self.target, 1e-5)

    def test_antimeridian_and_high_latitude_mixed_fixes(self) -> None:
        cases = (
            (
                Position(10, 179.99),
                (
                    Position(9.99, 179.98),
                    Position(10.01, -179.99),
                    Position(10.02, 179.97),
                ),
            ),
            (
                Position(85, 40),
                (
                    Position(84.99, 39.9),
                    Position(85.01, 40.1),
                    Position(85.02, 39.95),
                ),
            ),
        )
        for target, stations in cases:
            with self.subTest(target=target):
                result = solve_fix(
                    bearings=self.bearings(target, stations),
                    ranges=self.ranges(target, stations),
                    search_radius=20_000,
                )
                self.assertTrue(result.success, result)
                self.assertNear(result.position, target, 1e-5)

    def test_invalid_regional_solver_limits_fail_before_optimization(self) -> None:
        with self.assertRaisesRegex(FixError, "2000000"):
            solve_fix(
                ranges=self.ranges(),
                search_radius=2_000_001,
            )
        with self.assertRaisesRegex(FixError, "positive integer"):
            solve_fix(
                ranges=self.ranges(),
                max_iterations=0,
            )
        with self.assertRaisesRegex(FixError, "initial.*search disk"):
            solve_fix(
                ranges=self.ranges(),
                initial=destination(self.target, bearing=0, distance=2_000),
                search_center=self.target,
                search_radius=1_000,
            )


if __name__ == "__main__":
    unittest.main()
