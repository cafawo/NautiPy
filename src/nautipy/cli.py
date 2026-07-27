"""Command-line interface for coordinate conversion and inspection."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from io import TextIOWrapper
import sys

from .coordinates import ParseResult, convert_position, inspect_position
from .errors import CoordinateError
from .position import Position


_FORMAT_CHOICES = ("dd", "ddm", "dmm", "dms", "iso6709", "nmea")
_ORDER_CHOICES = ("latlon", "lonlat", "auto")
_OUTPUT_ORDER_CHOICES = ("latlon", "lonlat")


def _configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="strict")


def _position_payload(position: Position | None) -> dict[str, float] | None:
    if position is None:
        return None
    return {
        "latitude": position.latitude,
        "longitude": position.longitude,
    }


def _resolution_payload(value: object | None) -> str | None:
    return None if value is None else str(value)


def _inspection_payload(result: ParseResult) -> dict[str, object]:
    return {
        "position": _position_payload(result.position),
        "format": result.format,
        "component_formats": list(result.component_formats),
        "source_order": result.source_order,
        "evidence": list(result.evidence),
        "original_text": result.original_text,
        "normalized_tokens": list(result.normalized_tokens),
        "normalizations": list(result.normalizations),
        "warnings": list(result.warnings),
        "latitude_resolution": _resolution_payload(
            result.latitude_resolution
        ),
        "longitude_resolution": _resolution_payload(
            result.longitude_resolution
        ),
        "candidates": [
            {
                "format": candidate.format,
                "source_order": candidate.source_order,
                "position": _position_payload(candidate.position),
                "outcome": candidate.outcome,
                "evidence": list(candidate.evidence),
                "reason": candidate.reason,
            }
            for candidate in result.candidates
        ],
    }


def _add_input_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--order",
        choices=_ORDER_CHOICES,
        default="latlon",
        help="interpret unmarked components in this order (default: latlon)",
    )
    parser.add_argument(
        "--format",
        dest="input_format",
        type=str.casefold,
        choices=_FORMAT_CHOICES,
        help="require this input coordinate format",
    )


def _convert(arguments: argparse.Namespace) -> int:
    converted = convert_position(
        arguments.value,
        to=arguments.to,
        order=arguments.order,
        output_order=arguments.output_order,
        format=arguments.input_format,
        precision=arguments.precision,
        notation=arguments.notation,
        symbols=arguments.symbols,
        compact=arguments.compact,
        separator=arguments.separator,
    )
    print(converted)
    return 0


def _inspect(arguments: argparse.Namespace) -> int:
    result = inspect_position(
        arguments.value,
        order=arguments.order,
        format=arguments.input_format,
    )
    print(
        json.dumps(
            _inspection_payload(result),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nautipy",
        description="Convert and inspect nautical position coordinates.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    convert_parser = subparsers.add_parser(
        "convert",
        help="parse and convert one position",
    )
    convert_parser.add_argument("value", metavar="VALUE")
    _add_input_options(convert_parser)
    convert_parser.add_argument(
        "--to",
        type=str.casefold,
        choices=_FORMAT_CHOICES,
        default="dd",
        help="output coordinate format (default: dd)",
    )
    convert_parser.add_argument(
        "--output-order",
        choices=_OUTPUT_ORDER_CHOICES,
        default="latlon",
        help="output component order (default: latlon)",
    )
    convert_parser.add_argument(
        "--precision",
        type=int,
        help="decimal places in the least-significant displayed unit",
    )
    convert_parser.add_argument(
        "--notation",
        choices=("signed", "hemisphere"),
        help="notation for DD, DDM, or DMS output",
    )
    convert_parser.add_argument(
        "--symbols",
        choices=("unicode", "ascii"),
        help="symbols for DD, DDM, or DMS output",
    )
    convert_parser.add_argument(
        "--compact",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use compact ISO 6709 output",
    )
    convert_parser.add_argument(
        "--separator",
        help="output component separator",
    )
    convert_parser.set_defaults(handler=_convert)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="parse one position and show detection diagnostics",
    )
    inspect_parser.add_argument("value", metavar="VALUE")
    _add_input_options(inspect_parser)
    inspect_parser.set_defaults(handler=_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the NautiPy command-line interface."""

    _configure_utf8_output()
    parser = _parser()
    arguments = parser.parse_args(None if argv is None else list(argv))
    handler: Callable[[argparse.Namespace], int] = arguments.handler
    try:
        return handler(arguments)
    except CoordinateError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
