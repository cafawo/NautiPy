"""Small public API for NautiPy coordinates and WGS84 navigation."""

import importlib as _importlib
import typing as _typing

from .coordinates import (
    CandidateDiagnostic,
    ParseResult,
    PositionInput,
    convert_position,
    format_position,
    inspect_position,
    parse_position,
)
from .errors import (
    AmbiguousCoordinateError,
    CoordinateError,
    CoordinateParseError,
    CoordinateRangeError,
    FixDependencyError,
    FixError,
    NavigationError,
    NautiPyError,
)
from .position import Position

if _typing.TYPE_CHECKING:
    from .geodesic import (
        InverseResult,
        destination,
        distance,
        initial_bearing,
        interpolate,
        inverse,
        nearest_position,
    )


_NAVIGATION_EXPORTS = frozenset(
    {
        "InverseResult",
        "inverse",
        "distance",
        "initial_bearing",
        "destination",
        "interpolate",
        "nearest_position",
    }
)

__all__ = [
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
    "NautiPyError",
    "NavigationError",
    "FixError",
    "FixDependencyError",
    "CoordinateError",
    "CoordinateParseError",
    "CoordinateRangeError",
    "AmbiguousCoordinateError",
]


def __getattr__(name: str) -> object:
    if name in _NAVIGATION_EXPORTS:
        module = _importlib.import_module(".geodesic", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
