from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from fractions import Fraction
import inspect
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import nautipy
from nautipy import CoordinateRangeError, Position
from nautipy.fix import (
    BearingObservation,
    CandidateResult,
    CandidateStatus,
    FixError,
    FixResult,
    FixStatus,
    FixUncertainty,
    ObservationResidual,
    RangeObservation,
    solve_fix,
    two_bearing_candidates,
    two_range_candidates,
)
import nautipy.fix as fix_module


class ObservationModelTests(unittest.TestCase):
    def test_observations_coerce_positions_units_and_bearings(self) -> None:
        existing = Position(50, 8, identifier="station")
        bearing = BearingObservation(existing, -10, Decimal("0.25"))
        range_observation = RangeObservation(
            "51° N; 9° E",
            Fraction(250, 2),
            5,
        )

        self.assertIs(bearing.reference, existing)
        self.assertEqual(bearing.bearing, 350.0)
        self.assertEqual(bearing.uncertainty, 0.25)
        self.assertEqual(range_observation.reference, Position(51, 9))
        self.assertEqual(range_observation.distance, 125.0)
        self.assertEqual(range_observation.uncertainty, 5.0)
        self.assertIs(type(bearing.bearing), float)
        self.assertIs(type(range_observation.distance), float)

        huge_bearing = 360 * 10**1000 + 33
        self.assertEqual(
            BearingObservation((0, 0), huge_bearing, 1).bearing,
            33.0,
        )
        self.assertEqual(
            BearingObservation((0, 0), Decimal("1e-1000"), 1).bearing,
            0.0,
        )

    def test_uncertainty_is_required_and_models_are_frozen_slots(self) -> None:
        with self.assertRaises(TypeError):
            BearingObservation((0, 0), 90)  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            RangeObservation((0, 0), 100)  # type: ignore[call-arg]

        observation = BearingObservation((0, 0), 90, 1)
        self.assertFalse(hasattr(observation, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            observation.bearing = 0  # type: ignore[misc]

    def test_constructor_signatures_accept_position_like_references(self) -> None:
        bearing_signature = inspect.signature(BearingObservation)
        range_signature = inspect.signature(RangeObservation)

        self.assertEqual(
            tuple(bearing_signature.parameters),
            ("reference", "bearing", "uncertainty"),
        )
        self.assertEqual(
            tuple(range_signature.parameters),
            ("reference", "distance", "uncertainty"),
        )
        self.assertIn(
            "PositionInput",
            str(bearing_signature.parameters["reference"].annotation),
        )
        self.assertIn(
            "PositionInput",
            str(range_signature.parameters["reference"].annotation),
        )

        original = BearingObservation((0, 0), 10, 1)
        updated = replace(original, reference="1, 2", bearing=-10)
        self.assertEqual(updated.reference, Position(1, 2))
        self.assertEqual(updated.bearing, 350.0)

    def test_bearing_rejects_invalid_numeric_boundaries(self) -> None:
        invalid = (
            True,
            "90",
            None,
            1 + 0j,
            math.nan,
            math.inf,
            -math.inf,
            Decimal("NaN"),
            Decimal("Infinity"),
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(FixError):
                    BearingObservation((0, 0), value, 1)  # type: ignore[arg-type]

    def test_ranges_reject_invalid_numeric_boundaries(self) -> None:
        invalid = (
            True,
            "100",
            None,
            1 + 0j,
            -1,
            math.nan,
            math.inf,
            -math.inf,
            Decimal("NaN"),
            Decimal("1e-10000"),
            10**400,
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(FixError):
                    RangeObservation((0, 0), value, 1)  # type: ignore[arg-type]

        self.assertEqual(RangeObservation((0, 0), -0.0, 1).distance, -0.0)
        self.assertEqual(RangeObservation((0, 0), 0, 1).distance, 0.0)

    def test_uncertainty_must_be_strictly_positive_and_representable(self) -> None:
        invalid = (
            True,
            "1",
            None,
            0,
            -0.0,
            -1,
            math.nan,
            math.inf,
            Decimal("1e-10000"),
            10**400,
        )
        for value in invalid:
            with self.subTest(value=value, observation="bearing"):
                with self.assertRaises(FixError):
                    BearingObservation((0, 0), 90, value)  # type: ignore[arg-type]
            with self.subTest(value=value, observation="range"):
                with self.assertRaises(FixError):
                    RangeObservation((0, 0), 100, value)  # type: ignore[arg-type]

    def test_reference_uses_the_ordinary_coordinate_contract(self) -> None:
        self.assertEqual(
            BearingObservation(
                {"latitude": 50, "longitude": 8},
                90,
                1,
            ).reference,
            Position(50, 8),
        )
        with self.assertRaises(CoordinateRangeError):
            RangeObservation((91, 0), 100, 1)


class ResidualAndUncertaintyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bearing = BearingObservation((0, 0), 359, 2)
        self.range = RangeObservation((1, 1), 100, 5)

    def test_residuals_preserve_natural_and_standardized_units(self) -> None:
        bearing = ObservationResidual(self.bearing, 1, 2, 1)
        range_residual = ObservationResidual(self.range, 110, 10, 2)

        self.assertEqual(bearing.residual, 2.0)
        self.assertEqual(bearing.standardized_residual, 1.0)
        self.assertEqual(range_residual.residual, 10.0)
        self.assertEqual(range_residual.standardized_residual, 2.0)
        self.assertFalse(hasattr(bearing, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            bearing.residual = 0  # type: ignore[misc]

    def test_residuals_reject_inconsistent_or_noncanonical_values(self) -> None:
        invalid_arguments = (
            (self.bearing, 360, 1, 0.5),
            (self.bearing, 1, 180, 90),
            (self.bearing, 1, 1, 0.5),
            (self.bearing, 1, 2, 2),
            (self.range, -1, -101, -20.2),
            (self.range, 110, 5, 1),
            (self.range, 110, 10, math.inf),
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(FixError):
                    ObservationResidual(*arguments)

        with self.assertRaises(FixError):
            ObservationResidual(object(), 1, 1, 1)  # type: ignore[arg-type]

        long_range = RangeObservation((0, 0), 0, 1)
        with self.assertRaisesRegex(FixError, "predicted minus observed"):
            ObservationResidual(
                long_range,
                2_000_000,
                2_000_000.001,
                2_000_000.001,
            )
        with self.assertRaisesRegex(FixError, "divided by"):
            ObservationResidual(
                long_range,
                2_000_000,
                2_000_000,
                2_000_000.001,
            )

        adjacent = math.nextafter(2_000_000.0, math.inf)
        accepted = ObservationResidual(
            long_range,
            2_000_000,
            adjacent,
            adjacent,
        )
        self.assertEqual(accepted.residual, adjacent)

        smallest_positive = math.ulp(0.0)
        overflow_scale = RangeObservation(
            (0, 0),
            0,
            smallest_positive,
        )
        with self.assertRaisesRegex(FixError, "divided by"):
            ObservationResidual(
                overflow_scale,
                sys.float_info.max,
                sys.float_info.max,
                0,
            )

    def test_uncertainty_validates_covariance_and_95_percent_ellipse(self) -> None:
        scale = math.sqrt(-2 * math.log(0.05))
        uncertainty = FixUncertainty(
            covariance=((4, 0), (0, 1)),
            east_standard_deviation=2,
            north_standard_deviation=1,
            correlation=0,
            semi_major_95=2 * scale,
            semi_minor_95=scale,
            major_axis_bearing=270,
        )

        self.assertEqual(uncertainty.covariance, ((4.0, 0.0), (0.0, 1.0)))
        self.assertEqual(uncertainty.major_axis_bearing, 90.0)
        self.assertIs(type(uncertainty.covariance[0][0]), float)
        self.assertFalse(hasattr(uncertainty, "__dict__"))

        isotropic = FixUncertainty(
            ((1, 0), (0, 1)),
            1,
            1,
            0,
            scale,
            scale,
            None,
        )
        self.assertIsNone(isotropic.major_axis_bearing)

    def test_uncertainty_rejects_invalid_matrix_diagnostics(self) -> None:
        scale = math.sqrt(-2 * math.log(0.05))
        invalid = (
            (((1, 0),), 1, 1, 0, scale, scale, None),
            (((1, 1), (0, 1)), 1, 1, 0, scale, scale, None),
            (((-1, 0), (0, 1)), 1, 1, 0, scale, scale, None),
            (((1, 2), (2, 1)), 1, 1, 2, scale, scale, None),
            (((4, 0), (0, 1)), 1, 1, 0, 2 * scale, scale, 90),
            (((4, 0), (0, 1)), 2, 1, 0.5, 2 * scale, scale, 90),
            (((4, 0), (0, 1)), 2, 1, 0, scale, scale, 90),
            (((4, 0), (0, 1)), 2, 1, 0, 2 * scale, scale, None),
            (((4, 0), (0, 1)), 2, 1, 0, 2 * scale, scale, 0),
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(FixError):
                    FixUncertainty(*arguments)

    def test_uncertainty_rejects_scale_amplified_inconsistencies(self) -> None:
        scale = math.sqrt(-2 * math.log(0.05))
        expected_major = math.sqrt(6_000_000_000_000.0) * scale
        expected_minor = math.sqrt(2_000_000_000_000.0) * scale
        baseline: dict[str, object] = {
            "covariance": (
                (4_000_000_000_000.0, 2_000_000_000_000.0),
                (2_000_000_000_000.0, 4_000_000_000_000.0),
            ),
            "east_standard_deviation": 2_000_000.0,
            "north_standard_deviation": 2_000_000.0,
            "correlation": 0.5,
            "semi_major_95": expected_major,
            "semi_minor_95": expected_minor,
            "major_axis_bearing": 45.0,
        }
        invalid = (
            (
                "symmetric",
                {
                    "covariance": (
                        (4_000_000_000_000.0, 2_000_000_000_000.0),
                        (2_000_000_001_000.0, 4_000_000_000_000.0),
                    )
                },
            ),
            (
                "east standard deviation",
                {"east_standard_deviation": 2_000_000.001},
            ),
            ("correlation", {"correlation": 0.5000000004}),
            ("ellipse axes", {"semi_major_95": expected_major + 0.001}),
        )

        for message, overrides in invalid:
            with self.subTest(message=message):
                arguments = baseline | overrides
                with self.assertRaisesRegex(FixError, message):
                    FixUncertainty(**arguments)  # type: ignore[arg-type]

    def test_uncertainty_accepts_and_canonicalizes_roundoff_noise(self) -> None:
        scale = math.sqrt(-2 * math.log(0.05))
        covariance = 2_000_000_000_000.0
        standard_deviation = 2_000_000.0
        expected_major = math.sqrt(6_000_000_000_000.0) * scale
        expected_minor = math.sqrt(2_000_000_000_000.0) * scale

        uncertainty = FixUncertainty(
            covariance=(
                (4_000_000_000_000.0, covariance),
                (
                    math.nextafter(covariance, math.inf),
                    4_000_000_000_000.0,
                ),
            ),
            east_standard_deviation=math.nextafter(
                standard_deviation,
                math.inf,
            ),
            north_standard_deviation=math.nextafter(
                standard_deviation,
                math.inf,
            ),
            correlation=math.nextafter(0.5, math.inf),
            semi_major_95=math.nextafter(expected_major, math.inf),
            semi_minor_95=math.nextafter(expected_minor, math.inf),
            major_axis_bearing=45,
        )

        self.assertEqual(
            uncertainty.covariance[0][1],
            uncertainty.covariance[1][0],
        )
        self.assertEqual(
            uncertainty.east_standard_deviation,
            math.sqrt(uncertainty.covariance[0][0]),
        )
        self.assertEqual(
            uncertainty.north_standard_deviation,
            math.sqrt(uncertainty.covariance[1][1]),
        )
        self.assertEqual(
            uncertainty.correlation,
            uncertainty.covariance[0][1]
            / (
                uncertainty.east_standard_deviation
                * uncertainty.north_standard_deviation
            ),
        )
        self.assertEqual(uncertainty.semi_major_95, expected_major)
        self.assertEqual(uncertainty.semi_minor_95, expected_minor)

    def test_uncertainty_uses_one_isotropy_rule_for_models_and_solver(self) -> None:
        scale = math.sqrt(-2 * math.log(0.05))

        effectively_isotropic = FixUncertainty(
            ((1.0 + 5e-10, 0.0), (0.0, 1.0)),
            math.sqrt(1.0 + 5e-10),
            1.0,
            0.0,
            math.sqrt(1.0 + 5e-10) * scale,
            scale,
            None,
        )
        self.assertIsNone(effectively_isotropic.major_axis_bearing)

        anisotropic_arguments = (
            ((1.0 + 2e-9, 0.0), (0.0, 1.0)),
            math.sqrt(1.0 + 2e-9),
            1.0,
            0.0,
            math.sqrt(1.0 + 2e-9) * scale,
            scale,
        )
        with self.assertRaisesRegex(FixError, "required for anisotropic"):
            FixUncertainty(*anisotropic_arguments, None)
        anisotropic = FixUncertainty(*anisotropic_arguments, 90)
        self.assertEqual(anisotropic.major_axis_bearing, 90.0)

        below_absolute_floor = FixUncertainty(
            ((5e-13, 0.0), (0.0, 0.0)),
            math.sqrt(5e-13),
            0.0,
            0.0,
            math.sqrt(5e-13) * scale,
            0.0,
            None,
        )
        self.assertIsNone(below_absolute_floor.major_axis_bearing)

    def test_uncertainty_rejects_indefinite_zero_variance_matrix(self) -> None:
        scale = math.sqrt(-2 * math.log(0.05))
        with self.assertRaisesRegex(FixError, "positive semidefinite"):
            FixUncertainty(
                ((0.0, 5e-7), (5e-7, 0.0)),
                0.0,
                0.0,
                0.0,
                math.sqrt(5e-7) * scale,
                0.0,
                45.0,
            )

    def test_uncertainty_axes_are_stable_across_large_scales(self) -> None:
        scale = math.sqrt(-2 * math.log(0.05))
        high_dynamic_range = FixUncertainty(
            ((1e20, 0.0), (0.0, 1.0)),
            1e10,
            1.0,
            0.0,
            1e10 * scale,
            scale,
            90.0,
        )
        self.assertEqual(high_dynamic_range.semi_minor_95, scale)

        rotated_east = 500_000_000_000.5
        rotated_cross = 499_999_999_999.5
        rotated = FixUncertainty(
            (
                (rotated_east, rotated_cross),
                (rotated_cross, rotated_east),
            ),
            math.sqrt(rotated_east),
            math.sqrt(rotated_east),
            rotated_cross / rotated_east,
            1_000_000.0 * scale,
            scale,
            45.0,
        )
        self.assertAlmostEqual(rotated.semi_minor_95, scale, places=5)

        largest = sys.float_info.max
        largest_axis = math.sqrt(largest) * scale
        finite_extreme = FixUncertainty(
            ((largest, 0.0), (0.0, largest)),
            math.sqrt(largest),
            math.sqrt(largest),
            0.0,
            largest_axis,
            largest_axis,
            None,
        )
        self.assertTrue(math.isfinite(finite_extreme.semi_major_95))
        with self.assertRaisesRegex(FixError, "ellipse axes"):
            FixUncertainty(
                ((largest, 0.0), (0.0, largest)),
                math.sqrt(largest),
                math.sqrt(largest),
                0.0,
                largest,
                largest,
                None,
            )

        # Principal deviations below are independently rounded from
        # high-precision Decimal eigenvalue calculations.
        rank_one_axis = 1.414213562373095e154 * scale
        rank_one_extreme = FixUncertainty(
            ((1e308, 1e308), (1e308, 1e308)),
            math.sqrt(1e308),
            math.sqrt(1e308),
            1.0,
            rank_one_axis,
            0.0,
            45.0,
        )
        self.assertEqual(rank_one_extreme.semi_major_95, rank_one_axis)
        self.assertTrue(math.isfinite(rank_one_extreme.semi_major_95))

        largest_rank_one_axis = 1.8961503816218352e154 * scale
        largest_rank_one = FixUncertainty(
            (
                (sys.float_info.max, sys.float_info.max),
                (sys.float_info.max, sys.float_info.max),
            ),
            math.sqrt(sys.float_info.max),
            math.sqrt(sys.float_info.max),
            1.0,
            largest_rank_one_axis,
            0.0,
            45.0,
        )
        self.assertEqual(largest_rank_one.semi_major_95, largest_rank_one_axis)
        self.assertEqual(largest_rank_one.correlation, 1.0)

        separated_extreme = FixUncertainty(
            ((1e308, 1e-100), (1e-100, 1.0)),
            1e154,
            1.0,
            1e-254,
            1e154 * scale,
            scale,
            90.0,
        )
        self.assertEqual(separated_extreme.semi_major_95, 1e154 * scale)
        self.assertEqual(separated_extreme.semi_minor_95, scale)

        rotated_high_condition = FixUncertainty(
            (
                (8.922352282417819e215, -1.04103733313374e216),
                (-1.04103733313374e216, 1.2146558381401727e216),
            ),
            math.sqrt(8.922352282417819e215),
            math.sqrt(1.2146558381401727e216),
            -1.04103733313374e216
            / (
                math.sqrt(8.922352282417819e215)
                * math.sqrt(1.2146558381401727e216)
            ),
            1.4515133710655078e108 * scale,
            1.3574092814144726e100 * scale,
            139.40132251370582,
        )
        self.assertGreater(rotated_high_condition.semi_minor_95, 0.0)

        underflow_rotation = FixUncertainty(
            (
                (math.ulp(0.0), 1.1113793747425388e-262),
                (1.1113793747425388e-262, 1e-200),
            ),
            math.sqrt(math.ulp(0.0)),
            1e-100,
            0.5,
            1e-100 * scale,
            1.924965543538208e-162 * scale,
            None,
        )
        self.assertGreater(
            underflow_rotation.semi_major_95,
            underflow_rotation.semi_minor_95,
        )

        subnormal = FixUncertainty(
            (
                (math.ulp(0.0), 1.1e-322),
                (1.1e-322, 1e-320),
            ),
            math.sqrt(math.ulp(0.0)),
            math.sqrt(1e-320),
            1.1e-322
            / (math.sqrt(math.ulp(0.0)) * math.sqrt(1e-320)),
            1.0000535274377892e-160 * scale,
            1.9387498233632617e-162 * scale,
            None,
        )
        self.assertEqual(
            subnormal.semi_major_95,
            1.0000535274377892e-160 * scale,
        )
        self.assertEqual(
            subnormal.semi_minor_95,
            1.9387498233632617e-162 * scale,
        )

    def test_uncertainty_is_independent_of_decimal_context(self) -> None:
        scale = math.sqrt(-2 * math.log(0.05))
        expected = FixUncertainty(
            ((1e20, 0.0), (0.0, 1.0)),
            1e10,
            1.0,
            0.0,
            1e10 * scale,
            scale,
            90.0,
        )

        with localcontext() as context:
            context.prec = 1
            context.Emax = 9
            context.Emin = -9
            for signal in context.traps:
                context.traps[signal] = True
            actual = FixUncertainty(
                ((1e20, 0.0), (0.0, 1.0)),
                1e10,
                1.0,
                0.0,
                1e10 * scale,
                scale,
                90.0,
            )

        self.assertEqual(actual, expected)

    def test_uncertainty_snaps_only_roundoff_beyond_psd_boundary(self) -> None:
        scale = math.sqrt(-2 * math.log(0.05))
        within = 1.0
        for _ in range(8):
            within = math.nextafter(within, math.inf)
        accepted = FixUncertainty(
            ((1.0, within), (within, 1.0)),
            1.0,
            1.0,
            1.0,
            math.sqrt(2.0) * scale,
            0.0,
            math.nextafter(45.0, math.inf),
        )
        self.assertEqual(accepted.covariance, ((1.0, 1.0), (1.0, 1.0)))
        self.assertEqual(accepted.major_axis_bearing, 45.0)

        outside = math.nextafter(within, math.inf)
        with self.assertRaisesRegex(FixError, "positive semidefinite"):
            FixUncertainty(
                ((1.0, outside), (outside, 1.0)),
                1.0,
                1.0,
                1.0,
                math.sqrt(2.0) * scale,
                0.0,
                45.0,
            )


class ResultModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bearing_observation = BearingObservation((0, 0), 10, 2)
        self.range_observation = RangeObservation((1, 1), 100, 5)
        self.bearing_residual = ObservationResidual(
            self.bearing_observation,
            12,
            2,
            1,
        )
        self.range_residual = ObservationResidual(
            self.range_observation,
            110,
            10,
            2,
        )

    def result(self, **overrides: object) -> FixResult:
        values: dict[str, object] = {
            "position": Position(0.5, 0.5),
            "success": True,
            "status": FixStatus.CONVERGED,
            "residuals": [self.bearing_residual, self.range_residual],
            "objective": 5,
            "rms": math.sqrt(2.5),
            "bearing_rms": 2,
            "range_rms": 10,
            "iterations": 4,
            "function_evaluations": 9,
            "warnings": ["example"],
            "uncertainty": None,
            "rank": 2,
            "condition_number": 3,
            "degrees_of_freedom": 0,
            "reduced_chi_square": None,
            "competing_positions": [],
            "message": "converged",
        }
        values.update(overrides)
        return FixResult(**values)  # type: ignore[arg-type]

    def test_fix_result_normalizes_immutable_collections_and_metrics(self) -> None:
        result = self.result(status="converged")

        self.assertEqual(result.status, FixStatus.CONVERGED)
        self.assertIs(type(result.objective), float)
        self.assertIsInstance(result.residuals, tuple)
        self.assertIsInstance(result.warnings, tuple)
        self.assertFalse(hasattr(result, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            result.success = False  # type: ignore[misc]

    def test_fix_result_computes_large_natural_rms_without_overflow(self) -> None:
        observation = RangeObservation((0, 0), 0, sys.float_info.max)
        residual = ObservationResidual(
            observation,
            sys.float_info.max,
            sys.float_info.max,
            1,
        )

        result = self.result(
            residuals=(residual, residual),
            objective=2,
            rms=1,
            bearing_rms=None,
            range_rms=sys.float_info.max,
        )

        self.assertEqual(result.range_rms, sys.float_info.max)

    def test_solver_and_result_share_stable_natural_rms_arithmetic(self) -> None:
        from nautipy import _fix_solver

        values = (sys.float_info.max, sys.float_info.max)
        metrics = _fix_solver._metrics(
            tuple(SimpleNamespace(kind="range") for _ in values),
            SimpleNamespace(objective=2.0, natural=values),
        )
        stable = fix_module._root_mean_square(values)
        naive = math.sqrt(
            sum(value * value for value in values) / len(values)
        )

        self.assertEqual(metrics[3], stable)
        self.assertEqual(stable, sys.float_info.max)
        self.assertTrue(math.isinf(naive))

    def test_fix_result_rejects_inconsistent_success_and_diagnostics(self) -> None:
        invalid_overrides = (
            {"success": False},
            {"position": None},
            {"objective": 4},
            {"objective": 5.000000004},
            {"rms": 1},
            {"rms": math.sqrt(2.5) + 1e-9},
            {"bearing_rms": 1},
            {"bearing_rms": 2.000000001},
            {"range_rms": None},
            {"range_rms": 10.000000009},
            {"iterations": True},
            {"function_evaluations": -1},
            {"rank": 1, "condition_number": None},
            {"condition_number": math.inf},
            {"degrees_of_freedom": 1},
            {"reduced_chi_square": 5},
            {
                "residuals": (
                    self.bearing_residual,
                    self.range_residual,
                    self.range_residual,
                ),
                "objective": 9,
                "rms": math.sqrt(3),
                "degrees_of_freedom": 1,
                "reduced_chi_square": 9.000000008,
            },
            {"warnings": [1]},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(FixError):
                    self.result(**overrides)

    def test_failure_result_diagnostic_groups_are_complete(self) -> None:
        no_fit = self.result(
            position=None,
            success=False,
            status=FixStatus.NON_CONVERGED,
            residuals=(),
            objective=None,
            rms=None,
            bearing_rms=None,
            range_rms=None,
            rank=None,
            condition_number=None,
            degrees_of_freedom=None,
            reduced_chi_square=None,
            iterations=0,
            function_evaluations=0,
            message="no evaluated fit",
        )
        rank_deficient = replace(
            no_fit,
            status=FixStatus.DEGENERATE,
            rank=1,
        )
        self.assertIsNone(rank_deficient.condition_number)

        evaluated = self.result(
            position=None,
            success=False,
            status=FixStatus.NON_CONVERGED,
            message="evaluated fit",
        )
        evaluated_three = replace(
            evaluated,
            residuals=(
                self.bearing_residual,
                self.range_residual,
                self.range_residual,
            ),
            objective=9,
            rms=math.sqrt(3),
            degrees_of_freedom=1,
            reduced_chi_square=9,
        )
        self.assertEqual(evaluated_three.degrees_of_freedom, 1)
        rank_deficient_fit = replace(
            evaluated,
            rank=1,
            condition_number=None,
        )
        self.assertEqual(rank_deficient_fit.rank, 1)
        ill_conditioned = replace(
            evaluated,
            status=FixStatus.DEGENERATE,
            condition_number=1_000_001,
        )
        self.assertIs(ill_conditioned.status, FixStatus.DEGENERATE)

        invalid = (
            {"condition_number": 2},
            {"rank": 2},
            {"rank": 1},
            {"degrees_of_freedom": 0},
            {"residuals": (self.bearing_residual, self.range_residual)},
            {"iterations": 1},
            {"function_evaluations": 1},
        )
        for overrides in invalid:
            with self.subTest(no_fit_overrides=overrides):
                with self.assertRaises(FixError):
                    replace(no_fit, **overrides)

        with self.assertRaises(FixError):
            replace(evaluated, degrees_of_freedom=None)
        with self.assertRaises(FixError):
            replace(evaluated, condition_number=None)
        with self.assertRaises(FixError):
            replace(evaluated_three, reduced_chi_square=None)
        with self.assertRaises(FixError):
            replace(no_fit, rank=2, condition_number=2)
        with self.assertRaises(FixError):
            replace(evaluated, rank=None, condition_number=None)
        with self.assertRaises(FixError):
            replace(evaluated, iterations=0)
        with self.assertRaises(FixError):
            replace(evaluated, function_evaluations=0)
        with self.assertRaises(FixError):
            replace(
                evaluated,
                residuals=(self.range_residual, self.bearing_residual),
            )
        with self.assertRaises(FixError):
            replace(evaluated, status=FixStatus.DEGENERATE)
        with self.assertRaises(FixError):
            self.result(condition_number=1_000_001)
        with self.assertRaises(FixError):
            replace(
                no_fit,
                status=FixStatus.AMBIGUOUS,
                competing_positions=(Position(1, 1), Position(-1, -1)),
                rank=1,
            )
        with self.assertRaises(FixError):
            replace(
                evaluated,
                status=FixStatus.AMBIGUOUS,
                competing_positions=(Position(1, 1), Position(-1, -1)),
            )
        with self.assertRaises(FixError):
            replace(
                evaluated,
                residuals=(self.bearing_residual,),
                objective=1,
                rms=1,
                bearing_rms=2,
                range_rms=None,
                degrees_of_freedom=None,
            )

    def test_ambiguous_and_other_failure_cardinalities_are_explicit(self) -> None:
        first = Position(1, 1)
        second = Position(-1, -1)
        ambiguous = self.result(
            position=None,
            success=False,
            status=FixStatus.AMBIGUOUS,
            residuals=(),
            objective=None,
            rms=None,
            bearing_rms=None,
            range_rms=None,
            rank=None,
            condition_number=None,
            degrees_of_freedom=None,
            iterations=0,
            function_evaluations=0,
            competing_positions=(first, second),
            message="two equally plausible fixes",
        )
        self.assertEqual(ambiguous.competing_positions, (first, second))

        with self.assertRaises(FixError):
            replace(ambiguous, competing_positions=(first,))
        with self.assertRaises(FixError):
            replace(ambiguous, competing_positions=(first, first))
        with self.assertRaises(FixError):
            replace(
                ambiguous,
                status=FixStatus.DEGENERATE,
                competing_positions=(first, second),
            )
        with self.assertRaises(FixError):
            replace(ambiguous, position=first)

    def test_candidate_status_controls_position_cardinality(self) -> None:
        first = Position(1, 1)
        second = Position(-1, -1)
        unique = CandidateResult("unique", [first], [], "one")
        ambiguous = CandidateResult(
            CandidateStatus.AMBIGUOUS,
            [first, second],
            ["weak geometry"],
            "two",
        )

        self.assertEqual(unique.status, CandidateStatus.UNIQUE)
        self.assertEqual(unique.positions, (first,))
        self.assertEqual(ambiguous.positions, (first, second))
        with self.assertRaises(FixError):
            CandidateResult(CandidateStatus.UNIQUE, (), (), "none")
        with self.assertRaises(FixError):
            CandidateResult(CandidateStatus.AMBIGUOUS, (first,), (), "one")
        with self.assertRaises(FixError):
            CandidateResult(
                CandidateStatus.AMBIGUOUS,
                (first, first),
                (),
                "duplicate",
            )
        with self.assertRaises(FixError):
            CandidateResult(CandidateStatus.DEGENERATE, (first,), (), "bad")
        with self.assertRaises(FixError):
            CandidateResult(
                CandidateStatus.NO_SOLUTION,
                (),
                (1,),  # type: ignore[arg-type]
                "bad",
            )


class PublicFixApiTests(unittest.TestCase):
    def test_public_wrappers_delegate_to_private_solver(self) -> None:
        bearing = BearingObservation((0, 0), 10, 1)
        range_observation = RangeObservation((0, 0), 100, 2)
        expected = object()
        solver = SimpleNamespace(
            two_bearing_candidates=Mock(return_value=expected),
            two_range_candidates=Mock(return_value=expected),
            solve_fix=Mock(return_value=expected),
        )

        with patch.dict(
            sys.modules,
            {"nautipy._fix_solver": solver},
        ), patch.object(
            nautipy,
            "_fix_solver",
            solver,
            create=True,
        ):
            self.assertIs(
                two_bearing_candidates(
                    bearing,
                    bearing,
                    search_center="1, 2",
                    search_radius=123,
                ),
                expected,
            )
            self.assertIs(
                two_range_candidates(range_observation, range_observation),
                expected,
            )
            self.assertIs(
                solve_fix(
                    bearings=[bearing],
                    ranges=[range_observation],
                    initial="1, 2",
                    search_center="3, 4",
                    search_radius=456,
                    max_iterations=7,
                ),
                expected,
            )

        solver.two_bearing_candidates.assert_called_once_with(
            bearing,
            bearing,
            search_center="1, 2",
            search_radius=123,
        )
        solver.two_range_candidates.assert_called_once_with(
            range_observation,
            range_observation,
        )
        solver.solve_fix.assert_called_once_with(
            bearings=[bearing],
            ranges=[range_observation],
            initial="1, 2",
            search_center="3, 4",
            search_radius=456,
            max_iterations=7,
        )

    def test_complete_fix_api_is_exported_at_package_top_level(self) -> None:
        names = (
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
        )
        for name in names:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(nautipy, name),
                    getattr(fix_module, name),
                )
        self.assertFalse(hasattr(nautipy, "FixDependencyError"))


if __name__ == "__main__":
    unittest.main()
