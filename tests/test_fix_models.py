from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from fractions import Fraction
import builtins
import inspect
import math
import os
from pathlib import Path
import subprocess
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
    FixDependencyError,
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
            "_PositionInput",
            str(bearing_signature.parameters["reference"].annotation),
        )
        self.assertIn(
            "_PositionInput",
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

    def test_fix_result_rejects_inconsistent_success_and_diagnostics(self) -> None:
        invalid_overrides = (
            {"success": False},
            {"position": None},
            {"objective": 4},
            {"rms": 1},
            {"bearing_rms": 1},
            {"range_rms": None},
            {"iterations": True},
            {"function_evaluations": -1},
            {"rank": 1, "condition_number": None},
            {"condition_number": math.inf},
            {"degrees_of_freedom": 1},
            {"reduced_chi_square": 5},
            {"warnings": [1]},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(FixError):
                    self.result(**overrides)

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
            competing_positions=(first, second),
            message="two equally plausible fixes",
        )
        self.assertEqual(ambiguous.competing_positions, (first, second))

        with self.assertRaises(FixError):
            replace(ambiguous, competing_positions=(first,))
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
            CandidateResult(CandidateStatus.DEGENERATE, (first,), (), "bad")
        with self.assertRaises(FixError):
            CandidateResult(
                CandidateStatus.NO_SOLUTION,
                (),
                (1,),  # type: ignore[arg-type]
                "bad",
            )


class DependencyBoundaryTests(unittest.TestCase):
    def test_fix_import_and_model_construction_need_no_optional_modules(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        script = r'''
import builtins
import sys

real_import = builtins.__import__
blocked = {"numpy", "scipy", "geographiclib"}

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.partition(".")[0]
    if root in blocked:
        raise ModuleNotFoundError(f"blocked {root}", name=root)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import nautipy.fix as fix

observation = fix.BearingObservation("50, 8", 370, 1)
assert observation.bearing == 10.0
assert not (blocked & set(sys.modules))
try:
    fix._scientific()
except fix.FixDependencyError as error:
    assert str(error) == ('optional fix calculations require NumPy and SciPy; '
                          'install them with: python -m pip install "nautipy[fix]"')
else:
    raise AssertionError("missing dependency error was not raised")
'''
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_root)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=source_root.parent,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_missing_scipy_has_exact_error_and_nested_import_errors_propagate(
        self,
    ) -> None:
        real_import = builtins.__import__

        def scipy_missing(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: object = (),
            level: int = 0,
        ) -> object:
            if name == "numpy":
                return object()
            if name == "scipy.optimize":
                raise ModuleNotFoundError("missing scipy", name="scipy")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=scipy_missing):
            with self.assertRaises(FixDependencyError) as raised:
                fix_module._scientific()
        self.assertEqual(
            str(raised.exception),
            "optional fix calculations require NumPy and SciPy; install them "
            'with: python -m pip install "nautipy[fix]"',
        )
        self.assertIsNone(raised.exception.__cause__)

        nested = ModuleNotFoundError("missing backend", name="array_backend")

        def nested_missing(*args: object, **kwargs: object) -> object:
            raise nested

        with patch("builtins.__import__", side_effect=nested_missing):
            with self.assertRaises(ModuleNotFoundError) as propagated:
                fix_module._scientific()
        self.assertIs(propagated.exception, nested)

    def test_public_wrappers_inject_dependencies_into_private_solver(self) -> None:
        bearing = BearingObservation((0, 0), 10, 1)
        range_observation = RangeObservation((0, 0), 100, 2)
        expected = object()
        solver = SimpleNamespace(
            two_bearing_candidates=Mock(return_value=expected),
            two_range_candidates=Mock(return_value=expected),
            solve_fix=Mock(return_value=expected),
        )
        scientific = (object(), object())

        with patch.object(
            fix_module,
            "_scientific",
            return_value=scientific,
        ), patch.dict(
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

        injected = {
            "numpy": scientific[0],
            "least_squares": scientific[1],
        }
        solver.two_bearing_candidates.assert_called_once_with(
            bearing,
            bearing,
            search_center="1, 2",
            search_radius=123,
            **injected,
        )
        solver.two_range_candidates.assert_called_once_with(
            range_observation,
            range_observation,
            **injected,
        )
        solver.solve_fix.assert_called_once_with(
            bearings=[bearing],
            ranges=[range_observation],
            initial="1, 2",
            search_center="3, 4",
            search_radius=456,
            max_iterations=7,
            **injected,
        )

    def test_only_fix_errors_are_exported_at_package_top_level(self) -> None:
        self.assertIs(nautipy.FixError, FixError)
        self.assertIs(nautipy.FixDependencyError, FixDependencyError)
        self.assertTrue(issubclass(FixDependencyError, FixError))
        self.assertTrue(issubclass(FixDependencyError, ImportError))
        self.assertFalse(hasattr(nautipy, "BearingObservation"))
        self.assertFalse(hasattr(nautipy, "solve_fix"))


if __name__ == "__main__":
    unittest.main()
