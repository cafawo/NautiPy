from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError
import unittest

from nautipy import (
    AmbiguousCoordinateError,
    BatchInspectionFailure,
    BatchInspectionResult,
    BatchInspectionSuccess,
    CoordinateError,
    CoordinateParseError,
    CoordinateRangeError,
    Position,
    inspect_position,
    inspect_positions,
)


class OnePassIterable:
    def __init__(self, values: tuple[object, ...]) -> None:
        self.values = values
        self.iterations = 0

    def __iter__(self) -> Iterator[object]:
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("batch input was iterated more than once")
        return iter(self.values)


class BombIterable:
    def __init__(self) -> None:
        self.iterations = 0

    def __iter__(self) -> Iterator[object]:
        self.iterations += 1
        raise AssertionError("invalid options must be rejected before iteration")


class FailingIterator:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.next_calls = 0

    def __iter__(self) -> FailingIterator:
        return self

    def __next__(self) -> object:
        self.next_calls += 1
        if self.next_calls == 1:
            return "50, 8"
        raise self.error


class ExplodingMapping(Mapping[object, object]):
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __getitem__(self, key: object) -> object:
        raise self.error

    def __iter__(self) -> Iterator[object]:
        return iter(())

    def __len__(self) -> int:
        return 0


def capture_coordinate_error(
    value: object,
    *,
    order: str = "latlon",
    format: str | None = None,
) -> CoordinateError:
    try:
        inspect_position(value, order=order, format=format)  # type: ignore[arg-type]
    except CoordinateError as error:
        return error
    raise AssertionError("input unexpectedly produced a successful inspection")


class BatchInspectionTests(unittest.TestCase):
    def test_collect_preserves_order_indices_and_scalar_diagnostics(self) -> None:
        metadata_position = Position(
            5,
            6,
            identifier=0,
            description="",
        )
        values = (
            "50.12257, 8.66570",
            (40.0, -3.0),
            {"lat": 51.5, "lon": -0.1},
            {"type": "Point", "coordinates": [2.35, 48.86]},
            "5007.3542,N,00839.9420,E",
            metadata_position,
        )
        source = OnePassIterable(values)

        result = inspect_positions(source)

        self.assertEqual(source.iterations, 1)
        self.assertIsInstance(result, BatchInspectionResult)
        self.assertIsInstance(result.items, tuple)
        self.assertEqual([item.index for item in result.items], list(range(6)))
        self.assertTrue(
            all(isinstance(item, BatchInspectionSuccess) for item in result.items)
        )
        for item, value in zip(result.items, values):
            self.assertIsInstance(item, BatchInspectionSuccess)
            expected = inspect_position(value)  # type: ignore[arg-type]
            self.assertEqual(item.result, expected)

        metadata_result = result.items[-1]
        self.assertIsInstance(metadata_result, BatchInspectionSuccess)
        self.assertIs(metadata_result.result.position, metadata_position)
        self.assertEqual(metadata_result.result.position.identifier, 0)
        self.assertEqual(metadata_result.result.position.description, "")
        self.assertEqual(result.total_count, 6)
        self.assertEqual(result.parsed_count, 6)
        self.assertEqual(result.ambiguous_count, 0)
        self.assertEqual(result.invalid_count, 0)

    def test_collect_continues_and_classifies_every_failure(self) -> None:
        values = (
            "120, 50",
            "",
            "8, 50",
            (91, 181),
            {"lat": 2, "lon": 3},
        )
        result = inspect_positions(
            (value for value in values),
            order="auto",
        )

        self.assertEqual(
            [type(item) for item in result.items],
            [
                BatchInspectionSuccess,
                BatchInspectionFailure,
                BatchInspectionFailure,
                BatchInspectionFailure,
                BatchInspectionSuccess,
            ],
        )
        expected_errors = {
            1: capture_coordinate_error(values[1], order="auto"),
            2: capture_coordinate_error(values[2], order="auto"),
            3: capture_coordinate_error(values[3], order="auto"),
        }
        for index, expected in expected_errors.items():
            item = result.items[index]
            self.assertIsInstance(item, BatchInspectionFailure)
            self.assertIs(item.error_type, type(expected))
            self.assertEqual(item.message, str(expected))
            self.assertEqual(
                item.candidates,
                getattr(expected, "candidates", ()),
            )

        ambiguous = result.items[2]
        self.assertIsInstance(ambiguous, BatchInspectionFailure)
        self.assertIs(ambiguous.error_type, AmbiguousCoordinateError)
        self.assertEqual(
            [candidate.position for candidate in ambiguous.candidates],
            [Position(8, 50), Position(50, 8)],
        )
        self.assertEqual(
            [candidate.outcome for candidate in ambiguous.candidates],
            ["competing", "competing"],
        )

        self.assertEqual(result.total_count, len(result.items))
        self.assertEqual(
            result.total_count,
            result.parsed_count
            + result.ambiguous_count
            + result.invalid_count,
        )
        self.assertEqual(result.total_count, 5)
        self.assertEqual(result.parsed_count, 2)
        self.assertEqual(result.ambiguous_count, 1)
        self.assertEqual(result.invalid_count, 2)
        self.assertEqual([item.index for item in result.items], [0, 1, 2, 3, 4])

    def test_empty_iterables_return_zero_counts_in_both_modes(self) -> None:
        for errors in ("collect", "raise"):
            with self.subTest(errors=errors):
                result = inspect_positions(iter(()), errors=errors)
                self.assertEqual(result.items, ())
                self.assertEqual(result.total_count, 0)
                self.assertEqual(result.parsed_count, 0)
                self.assertEqual(result.ambiguous_count, 0)
                self.assertEqual(result.invalid_count, 0)

    def test_raise_stops_at_first_failure_and_preserves_exception_details(
        self,
    ) -> None:
        cases = (
            ("", CoordinateParseError),
            ((91, 181), CoordinateRangeError),
            ("8, 50", AmbiguousCoordinateError),
        )
        for invalid, error_type in cases:
            consumed: list[int] = []

            def values() -> Iterator[object]:
                consumed.append(0)
                yield "120, 50"
                consumed.append(1)
                yield invalid
                consumed.append(2)
                raise AssertionError("raise mode consumed past the first failure")

            expected = capture_coordinate_error(invalid, order="auto")
            with self.subTest(error_type=error_type.__name__):
                with self.assertRaises(error_type) as raised:
                    inspect_positions(
                        values(),
                        order="auto",
                        errors="raise",
                    )

                self.assertIs(type(raised.exception), error_type)
                self.assertIn("positions[1]:", str(raised.exception))
                self.assertEqual(consumed, [0, 1])
                self.assertIsInstance(raised.exception.__cause__, error_type)
                self.assertEqual(str(raised.exception.__cause__), str(expected))
                if isinstance(raised.exception, AmbiguousCoordinateError):
                    self.assertEqual(
                        raised.exception.candidates,
                        getattr(expected, "candidates", ()),
                    )

    def test_decimal_comma_rows_keep_scalar_semantics(self) -> None:
        result = inspect_positions(
            (
                "50,12257 N; 8,66570 E",
                "50,12257, 8,66570",
            ),
            order="lonlat",
        )

        success, failure = result.items
        self.assertIsInstance(success, BatchInspectionSuccess)
        self.assertEqual(success.result.position, Position(50.12257, 8.66570))
        self.assertIn(
            "normalized decimal comma",
            success.result.normalizations,
        )
        self.assertIsInstance(failure, BatchInspectionFailure)
        self.assertIs(failure.error_type, AmbiguousCoordinateError)
        self.assertIn("decimal commas", failure.message)
        self.assertEqual(result.total_count, 2)
        self.assertEqual(result.parsed_count, 1)
        self.assertEqual(result.ambiguous_count, 1)
        self.assertEqual(result.invalid_count, 0)

    def test_default_order_and_format_alias_match_scalar_inspection(self) -> None:
        default = inspect_positions(["8, 50"])
        default_item = default.items[0]
        self.assertIsInstance(default_item, BatchInspectionSuccess)
        self.assertEqual(default_item.result.position, Position(8, 50))

        automatic = inspect_positions(["8, 50"], order="auto")
        automatic_item = automatic.items[0]
        self.assertIsInstance(automatic_item, BatchInspectionFailure)
        self.assertIs(automatic_item.error_type, AmbiguousCoordinateError)

        alias = inspect_positions(
            ["50° 7.3542′ N; 8° 39.942′ E"],
            format="DMM",
        )
        alias_item = alias.items[0]
        self.assertIsInstance(alias_item, BatchInspectionSuccess)
        self.assertEqual(alias_item.result.format, "ddm")
        self.assertIn(
            "canonicalized format alias 'dmm' to 'ddm'",
            alias_item.result.normalizations,
        )

        selected = inspect_positions(
            (
                "5007.3542,N,00839.9420,E",
                "50.12257, 8.66570",
            ),
            format="nmea",
        )
        self.assertIsInstance(selected.items[0], BatchInspectionSuccess)
        self.assertEqual(selected.items[0].result.format, "nmea")
        self.assertIsInstance(selected.items[1], BatchInspectionFailure)
        self.assertIs(
            selected.items[1].error_type,
            CoordinateParseError,
        )

    def test_raise_mode_returns_the_same_nonempty_success_result(self) -> None:
        values = ("50, 8", (51, 9))

        self.assertEqual(
            inspect_positions(values),
            inspect_positions(iter(values), errors="raise"),
        )

    def test_invalid_options_are_rejected_before_iteration(self) -> None:
        cases = (
            ("errors", {"errors": "ignore"}),
            ("errors", {"errors": "COLLECT"}),
            ("errors", {"errors": None}),
            ("errors", {"errors": True}),
            ("order", {"order": "northfirst"}),
            ("order", {"order": None}),
            ("format", {"format": "utm"}),
            ("format", {"format": 1}),
        )
        for message, options in cases:
            source = BombIterable()
            with self.subTest(options=options):
                with self.assertRaisesRegex(CoordinateParseError, message):
                    inspect_positions(source, **options)  # type: ignore[arg-type]
                self.assertEqual(source.iterations, 0)

    def test_rejects_accidental_single_position_inputs(self) -> None:
        single_values = (
            Position(50, 8),
            "50, 8",
            b"50, 8",
            bytearray(b"50, 8"),
            {"lat": 50, "lon": 8},
            {"type": "Point", "coordinates": [8, 50]},
            42,
        )
        for value in single_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    CoordinateParseError,
                    "iterable of position values",
                ):
                    inspect_positions(value)  # type: ignore[arg-type]

    def test_sequences_are_outer_batches_not_single_positions(self) -> None:
        for values in ([50, 8], (50, 8)):
            with self.subTest(type=type(values).__name__):
                result = inspect_positions(values)
                self.assertEqual(result.total_count, 2)
                self.assertEqual(result.parsed_count, 0)
                self.assertEqual(result.ambiguous_count, 0)
                self.assertEqual(result.invalid_count, 2)
                self.assertEqual(
                    [item.index for item in result.items],
                    [0, 1],
                )
                for item in result.items:
                    self.assertIsInstance(item, BatchInspectionFailure)
                    self.assertIs(item.error_type, CoordinateParseError)

        parsed = inspect_positions(((50, 8), (51, 9)))
        self.assertEqual(parsed.parsed_count, 2)
        self.assertEqual(
            [
                item.result.position
                for item in parsed.items
                if isinstance(item, BatchInspectionSuccess)
            ],
            [Position(50, 8), Position(51, 9)],
        )

    def test_iterator_exceptions_propagate_without_becoming_row_failures(
        self,
    ) -> None:
        for errors in ("collect", "raise"):
            for error in (
                RuntimeError("source failed"),
                CoordinateParseError("source coordinate error"),
            ):
                source = FailingIterator(error)
                with self.subTest(errors=errors, error_type=type(error).__name__):
                    with self.assertRaises(type(error)) as raised:
                        inspect_positions(source, errors=errors)
                    self.assertIs(raised.exception, error)
                    self.assertEqual(source.next_calls, 2)

        iteration_error = RuntimeError("could not start source")

        class BrokenIterable:
            def __iter__(self) -> Iterator[object]:
                raise iteration_error

        with self.assertRaises(RuntimeError) as raised:
            inspect_positions(BrokenIterable())
        self.assertIs(raised.exception, iteration_error)

    def test_unexpected_row_protocol_exceptions_are_not_collected(self) -> None:
        error = RuntimeError("mapping protocol failed")
        with self.assertRaises(RuntimeError) as raised:
            inspect_positions([ExplodingMapping(error)])
        self.assertIs(raised.exception, error)

    def test_duplicate_rows_remain_distinct_and_results_are_deterministic(
        self,
    ) -> None:
        values = ("50, 8", "50, 8", "", "")
        first = inspect_positions(values)
        second = inspect_positions(iter(values))

        self.assertEqual(first, second)
        self.assertEqual([item.index for item in first.items], [0, 1, 2, 3])
        self.assertEqual(first.total_count, 4)
        self.assertEqual(first.parsed_count, 2)
        self.assertEqual(first.invalid_count, 2)


class BatchInspectionModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parse_result = inspect_position("50, 8")

    def test_models_are_immutable_and_normalize_public_collections(self) -> None:
        success = BatchInspectionSuccess(0, self.parse_result)
        candidate = object()
        failure = BatchInspectionFailure(
            1,
            AmbiguousCoordinateError,
            "ambiguous",
            [candidate],  # type: ignore[arg-type]
        )
        result = BatchInspectionResult([success, failure])  # type: ignore[arg-type]

        self.assertIsInstance(failure.candidates, tuple)
        self.assertEqual(failure.candidates, (candidate,))
        self.assertIsInstance(result.items, tuple)
        self.assertEqual(result.items, (success, failure))
        self.assertEqual(result.total_count, 2)
        self.assertEqual(result.parsed_count, 1)
        self.assertEqual(result.ambiguous_count, 1)
        self.assertEqual(result.invalid_count, 0)

        with self.assertRaises(FrozenInstanceError):
            success.index = 2  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            failure.message = "changed"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.total_count = 99  # type: ignore[misc]

    def test_success_rejects_invalid_index_or_result(self) -> None:
        for index in (-1, True, 1.5, "0"):
            with self.subTest(index=index):
                with self.assertRaises(CoordinateParseError):
                    BatchInspectionSuccess(
                        index,  # type: ignore[arg-type]
                        self.parse_result,
                    )
        with self.assertRaises(CoordinateParseError):
            BatchInspectionSuccess(0, object())  # type: ignore[arg-type]

    def test_failure_validates_fields_and_candidate_semantics(self) -> None:
        for index in (-1, True, 1.5, "0"):
            with self.subTest(index=index):
                with self.assertRaises(CoordinateParseError):
                    BatchInspectionFailure(
                        index,  # type: ignore[arg-type]
                        CoordinateParseError,
                        "invalid",
                    )
        for error_type in (Exception, CoordinateParseError("bad"), "parse"):
            with self.subTest(error_type=error_type):
                with self.assertRaises(CoordinateParseError):
                    BatchInspectionFailure(
                        0,
                        error_type,  # type: ignore[arg-type]
                        "invalid",
                    )
        with self.assertRaises(CoordinateParseError):
            BatchInspectionFailure(
                0,
                CoordinateParseError,
                1,  # type: ignore[arg-type]
            )
        with self.assertRaises(CoordinateParseError):
            BatchInspectionFailure(
                0,
                CoordinateParseError,
                "invalid",
                (object(),),
            )
        for candidates in ("candidate", 1):
            with self.subTest(candidates=candidates):
                with self.assertRaises(CoordinateParseError):
                    BatchInspectionFailure(
                        0,
                        AmbiguousCoordinateError,
                        "ambiguous",
                        candidates,  # type: ignore[arg-type]
                    )

    def test_result_rejects_invalid_items_and_noncontiguous_indices(self) -> None:
        success = BatchInspectionSuccess(0, self.parse_result)
        for items in ([object()], "item", 1):
            with self.subTest(items=items):
                with self.assertRaises(CoordinateParseError):
                    BatchInspectionResult(items)  # type: ignore[arg-type]
        with self.assertRaises(CoordinateParseError):
            BatchInspectionResult(
                [BatchInspectionSuccess(1, self.parse_result)]
            )  # type: ignore[arg-type]
        with self.assertRaises(CoordinateParseError):
            BatchInspectionResult(
                [
                    success,
                    BatchInspectionFailure(
                        2,
                        CoordinateParseError,
                        "invalid",
                    ),
                ]
            )  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
