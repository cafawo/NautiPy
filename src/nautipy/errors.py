"""Public exception hierarchy for NautiPy."""

__all__ = [
    "AmbiguousCoordinateError",
    "CoordinateError",
    "CoordinateParseError",
    "CoordinateRangeError",
    "FixError",
    "NautiPyError",
    "NavigationError",
]


class NautiPyError(Exception):
    """Base class for package-specific errors."""


class NavigationError(NautiPyError):
    """Raised for invalid or undefined navigation calculations."""


class FixError(NautiPyError):
    """Raised for invalid position-fix inputs or calculations."""


class CoordinateError(NautiPyError):
    """Base class for coordinate input and formatting errors."""


class CoordinateParseError(CoordinateError):
    """Raised when coordinate input cannot be parsed."""


class CoordinateRangeError(CoordinateError):
    """Raised when a coordinate is non-finite or outside its legal range."""


class AmbiguousCoordinateError(CoordinateError):
    """Raised when multiple materially different positions are valid."""

    def __init__(
        self,
        message: str,
        *,
        candidates: tuple[object, ...] = (),
    ) -> None:
        super().__init__(message)
        self.candidates = candidates
