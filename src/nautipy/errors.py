"""Public exception hierarchy for NautiPy."""


class NautiPyError(Exception):
    """Base class for package-specific errors."""


class CoordinateError(NautiPyError):
    """Base class for coordinate input and formatting errors."""


class CoordinateParseError(CoordinateError):
    """Raised when coordinate input cannot be parsed."""


class CoordinateRangeError(CoordinateError):
    """Raised when a coordinate is non-finite or outside its legal range."""


class AmbiguousCoordinateError(CoordinateError):
    """Raised when multiple materially different positions are valid."""
