"""Optional bearing/range position-fix models and entry points.

The public models in this module use only the Python standard library.  NumPy
and SciPy are loaded lazily when a calculation is requested, so importing the
module and constructing observations does not require the ``fix`` extra.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from math import atan2, degrees, hypot, isclose, isfinite, log, sqrt
from numbers import Rational, Real
from typing import TypeAlias

from .coordinates import parse_position
from .errors import FixDependencyError, FixError
from .position import Position


_PositionInput: TypeAlias = (
    Position | str | Mapping[object, object] | Sequence[object]
)
_FIX_INSTALL_MESSAGE = (
    "optional fix calculations require NumPy and SciPy; install them with: "
    'python -m pip install "nautipy[fix]"'
)
_CHI_SQUARE_2D_95_SCALE = sqrt(-2.0 * log(0.05))

__all__ = [
    "BearingObservation",
    "RangeObservation",
    "ObservationResidual",
    "FixUncertainty",
    "FixStatus",
    "CandidateStatus",
    "CandidateResult",
    "FixResult",
    "FixError",
    "FixDependencyError",
    "two_bearing_candidates",
    "two_range_candidates",
    "solve_fix",
]


def _finite_number(
    value: object,
    *,
    name: str,
    reject_underflow: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise FixError(f"{name} must be a real number")
    if isinstance(value, Decimal) and not value.is_finite():
        raise FixError(f"{name} must be finite")
    if isinstance(value, float) and not isfinite(value):
        raise FixError(f"{name} must be finite")

    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise FixError(
            f"{name} must be representable as a finite float"
        ) from error
    if not isfinite(number):
        raise FixError(f"{name} must be representable as a finite float")
    if reject_underflow and value != 0 and number == 0.0:
        raise FixError(
            f"{name} magnitude is too small for the internal float "
            "representation"
        )
    return number


def _nonnegative_number(
    value: object,
    *,
    name: str,
    strictly_positive: bool = False,
) -> float:
    number = _finite_number(value, name=name)
    if strictly_positive:
        if number <= 0.0:
            raise FixError(f"{name} must be greater than zero")
    elif number < 0.0:
        raise FixError(f"{name} must be at least zero")
    return number


def _digits_modulo(digits: tuple[int, ...], modulus: int) -> int:
    remainder = 0
    for digit in digits:
        remainder = (remainder * 10 + digit) % modulus
    return remainder


def _normalized_float(value: float, *, period: float) -> float:
    normalized = value % period
    return 0.0 if normalized in {0.0, period} else normalized


def _decimal_modulo(value: Decimal, *, period: int, name: str) -> float:
    if not value.is_finite():
        raise FixError(f"{name} must be finite")
    if value.is_zero():
        return 0.0

    sign, digits, exponent = value.as_tuple()
    if exponent >= 0:
        remainder = (
            _digits_modulo(digits, period) * pow(10, exponent, period)
        ) % period
        number = float(remainder)
    else:
        decimal_point = len(digits) + exponent
        if decimal_point <= 0:
            positive_remainder = value.copy_abs()
        else:
            whole_remainder = _digits_modulo(digits[:decimal_point], period)
            whole_digits = tuple(
                int(character) for character in str(whole_remainder)
            )
            positive_remainder = Decimal(
                (0, whole_digits + digits[decimal_point:], exponent)
            )
        number = float(positive_remainder)
        if not positive_remainder.is_zero() and number == 0.0:
            return 0.0
        if sign and not positive_remainder.is_zero():
            number = float(Fraction(period) - Fraction(positive_remainder))
    signed = -number if sign and exponent >= 0 else number
    return _normalized_float(signed, period=float(period))


def _exact_ratio_modulo(value: object, *, period: int) -> float | None:
    if isinstance(value, Rational):
        remainder = value % period
    else:
        ratio_method = getattr(value, "as_integer_ratio", None)
        if not callable(ratio_method):
            return None
        try:
            numerator, denominator = ratio_method()
            remainder = Fraction(int(numerator), int(denominator)) % period
        except (OverflowError, TypeError, ValueError, ZeroDivisionError):
            return None
    number = float(remainder)
    if remainder != 0 and number == 0.0:
        return 0.0
    return _normalized_float(number, period=float(period))


def _normalized_angle(value: object, *, period: int, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise FixError(f"{name} must be a real number")
    if isinstance(value, Decimal):
        return _decimal_modulo(value, period=period, name=name)
    if isinstance(value, float) and not isfinite(value):
        raise FixError(f"{name} must be finite")
    exact = _exact_ratio_modulo(value, period=period)
    if exact is not None:
        return exact
    number = _finite_number(value, name=name, reject_underflow=False)
    return _normalized_float(number, period=float(period))


def _as_tuple(value: object, *, name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise FixError(f"{name} must be an iterable, not text")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise FixError(f"{name} must be an iterable") from error


def _warning_tuple(value: object) -> tuple[str, ...]:
    warnings = _as_tuple(value, name="warnings")
    if any(not isinstance(warning, str) for warning in warnings):
        raise FixError("warnings must contain only strings")
    return warnings  # type: ignore[return-value]


def _position_tuple(value: object, *, name: str) -> tuple[Position, ...]:
    positions = _as_tuple(value, name=name)
    if any(not isinstance(position, Position) for position in positions):
        raise FixError(f"{name} must contain only Position values")
    return positions  # type: ignore[return-value]


def _residual_tuple(value: object) -> tuple[ObservationResidual, ...]:
    residuals = _as_tuple(value, name="residuals")
    if any(not isinstance(item, ObservationResidual) for item in residuals):
        raise FixError(
            "residuals must contain only ObservationResidual values"
        )
    return residuals  # type: ignore[return-value]


def _optional_nonnegative(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _nonnegative_number(value, name=name)


def _close(first: float, second: float) -> bool:
    return isclose(first, second, rel_tol=1e-9, abs_tol=1e-12)


@dataclass(frozen=True, slots=True, init=False)
class BearingObservation:
    """A true bearing measured at the unknown fix toward ``reference``.

    ``bearing`` and its required one-standard-deviation ``uncertainty`` are in
    degrees.  Bearings are normalized to ``[0, 360)``.
    """

    reference: Position
    bearing: float
    uncertainty: float

    def __init__(
        self,
        reference: _PositionInput,
        bearing: Real | Decimal,
        uncertainty: Real | Decimal,
    ) -> None:
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "bearing", bearing)
        object.__setattr__(self, "uncertainty", uncertainty)
        self.__post_init__()

    def __post_init__(self) -> None:
        reference = parse_position(self.reference)
        bearing = _normalized_angle(
            self.bearing,
            period=360,
            name="bearing",
        )
        uncertainty = _nonnegative_number(
            self.uncertainty,
            name="bearing uncertainty",
            strictly_positive=True,
        )
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "bearing", bearing)
        object.__setattr__(self, "uncertainty", uncertainty)


@dataclass(frozen=True, slots=True, init=False)
class RangeObservation:
    """A WGS84 distance from the unknown fix to ``reference``.

    ``distance`` and its required one-standard-deviation ``uncertainty`` are
    in metres.
    """

    reference: Position
    distance: float
    uncertainty: float

    def __init__(
        self,
        reference: _PositionInput,
        distance: Real | Decimal,
        uncertainty: Real | Decimal,
    ) -> None:
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "uncertainty", uncertainty)
        self.__post_init__()

    def __post_init__(self) -> None:
        reference = parse_position(self.reference)
        distance = _nonnegative_number(self.distance, name="distance")
        uncertainty = _nonnegative_number(
            self.uncertainty,
            name="range uncertainty",
            strictly_positive=True,
        )
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "uncertainty", uncertainty)


@dataclass(frozen=True, slots=True)
class ObservationResidual:
    """Model-minus-observed diagnostics in natural and standardized units.

    Residuals use ``predicted - observed``.  Bearing residuals are degrees in
    ``[-180, 180)``; range residuals are metres.  ``standardized_residual`` is
    the natural residual divided by the observation uncertainty.
    """

    observation: BearingObservation | RangeObservation
    predicted: float
    residual: float
    standardized_residual: float

    def __post_init__(self) -> None:
        if not isinstance(
            self.observation,
            (BearingObservation, RangeObservation),
        ):
            raise FixError(
                "observation must be a BearingObservation or RangeObservation"
            )
        predicted = _finite_number(self.predicted, name="predicted")
        residual = _finite_number(self.residual, name="residual")
        standardized = _finite_number(
            self.standardized_residual,
            name="standardized residual",
        )

        if isinstance(self.observation, BearingObservation):
            if not 0.0 <= predicted < 360.0:
                raise FixError(
                    "predicted bearing must be between 0 and 360 degrees"
                )
            if not -180.0 <= residual < 180.0:
                raise FixError(
                    "bearing residual must be in the interval [-180, 180)"
                )
            expected_residual = (
                predicted - self.observation.bearing + 180.0
            ) % 360.0 - 180.0
            if expected_residual == 0.0:
                expected_residual = 0.0
        else:
            if predicted < 0.0:
                raise FixError("predicted range must be at least zero")
            expected_residual = predicted - self.observation.distance

        if not _close(residual, expected_residual):
            raise FixError(
                "residual must equal predicted minus observed in its natural "
                "unit"
            )

        expected_standardized = residual / self.observation.uncertainty
        if not _close(standardized, expected_standardized):
            raise FixError(
                "standardized residual must equal residual divided by "
                "observation uncertainty"
            )
        object.__setattr__(self, "predicted", predicted)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "standardized_residual", standardized)


@dataclass(frozen=True, slots=True)
class FixUncertainty:
    """Local east/north covariance and a 95% confidence ellipse.

    Covariance entries use square metres.  Standard deviations and ellipse
    semiaxes use metres.  ``major_axis_bearing`` is a true axial bearing in
    ``[0, 180)`` and is ``None`` for an isotropic covariance.
    """

    covariance: tuple[tuple[float, float], tuple[float, float]]
    east_standard_deviation: float
    north_standard_deviation: float
    correlation: float
    semi_major_95: float
    semi_minor_95: float
    major_axis_bearing: float | None

    def __post_init__(self) -> None:
        rows = _as_tuple(self.covariance, name="covariance")
        if len(rows) != 2:
            raise FixError("covariance must be a 2 by 2 matrix")
        first_row = _as_tuple(rows[0], name="covariance row")
        second_row = _as_tuple(rows[1], name="covariance row")
        if len(first_row) != 2 or len(second_row) != 2:
            raise FixError("covariance must be a 2 by 2 matrix")
        east_variance = _finite_number(
            first_row[0],
            name="east covariance variance",
        )
        east_north = _finite_number(
            first_row[1],
            name="east/north covariance",
        )
        north_east = _finite_number(
            second_row[0],
            name="north/east covariance",
        )
        north_variance = _finite_number(
            second_row[1],
            name="north covariance variance",
        )
        if not _close(east_north, north_east):
            raise FixError("covariance must be symmetric")
        if east_variance < 0.0 or north_variance < 0.0:
            raise FixError("covariance variances must be non-negative")
        determinant = east_variance * north_variance - east_north**2
        determinant_scale = max(
            1.0,
            abs(east_variance * north_variance),
            east_north**2,
        )
        if determinant < -1e-12 * determinant_scale:
            raise FixError("covariance must be positive semidefinite")

        east_standard_deviation = _nonnegative_number(
            self.east_standard_deviation,
            name="east standard deviation",
        )
        north_standard_deviation = _nonnegative_number(
            self.north_standard_deviation,
            name="north standard deviation",
        )
        correlation = _finite_number(self.correlation, name="correlation")
        if not -1.0 <= correlation <= 1.0:
            raise FixError("correlation must be between -1 and 1")
        if not _close(east_standard_deviation, sqrt(east_variance)):
            raise FixError(
                "east standard deviation must match covariance"
            )
        if not _close(north_standard_deviation, sqrt(north_variance)):
            raise FixError(
                "north standard deviation must match covariance"
            )
        correlation_denominator = (
            east_standard_deviation * north_standard_deviation
        )
        expected_correlation = (
            east_north / correlation_denominator
            if correlation_denominator > 0.0
            else 0.0
        )
        if not _close(correlation, expected_correlation):
            raise FixError("correlation must match covariance")

        semi_major = _nonnegative_number(
            self.semi_major_95,
            name="95% semi-major axis",
        )
        semi_minor = _nonnegative_number(
            self.semi_minor_95,
            name="95% semi-minor axis",
        )
        if semi_major < semi_minor:
            raise FixError(
                "95% semi-major axis must not be shorter than semi-minor axis"
            )
        trace = east_variance + north_variance
        discriminant = hypot(
            east_variance - north_variance,
            2.0 * east_north,
        )
        major_variance = max(0.0, (trace + discriminant) / 2.0)
        minor_variance = max(0.0, (trace - discriminant) / 2.0)
        expected_major = sqrt(major_variance) * _CHI_SQUARE_2D_95_SCALE
        expected_minor = sqrt(minor_variance) * _CHI_SQUARE_2D_95_SCALE
        if not _close(semi_major, expected_major) or not _close(
            semi_minor,
            expected_minor,
        ):
            raise FixError("95% ellipse axes must match covariance")

        isotropic = _close(major_variance, minor_variance)
        if isotropic:
            if self.major_axis_bearing is not None:
                raise FixError(
                    "major axis bearing must be None for isotropic uncertainty"
                )
            major_axis_bearing = None
        else:
            if self.major_axis_bearing is None:
                raise FixError(
                    "major axis bearing is required for anisotropic uncertainty"
                )
            major_axis_bearing = _normalized_angle(
                self.major_axis_bearing,
                period=180,
                name="major axis bearing",
            )
            axis_from_east = 0.5 * atan2(
                2.0 * east_north,
                east_variance - north_variance,
            )
            expected_bearing = (90.0 - degrees(axis_from_east)) % 180.0
            bearing_difference = abs(major_axis_bearing - expected_bearing)
            bearing_difference = min(
                bearing_difference,
                180.0 - bearing_difference,
            )
            if bearing_difference > 1e-7:
                raise FixError("major axis bearing must match covariance")

        covariance = (
            (east_variance, east_north),
            (north_east, north_variance),
        )
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(
            self,
            "east_standard_deviation",
            east_standard_deviation,
        )
        object.__setattr__(
            self,
            "north_standard_deviation",
            north_standard_deviation,
        )
        object.__setattr__(self, "correlation", correlation)
        object.__setattr__(self, "semi_major_95", semi_major)
        object.__setattr__(self, "semi_minor_95", semi_minor)
        object.__setattr__(self, "major_axis_bearing", major_axis_bearing)


class FixStatus(str, Enum):
    """Outcome of an attempted position fix."""

    CONVERGED = "converged"
    NON_CONVERGED = "non_converged"
    AMBIGUOUS = "ambiguous"
    DEGENERATE = "degenerate"
    NO_SOLUTION = "no_solution"


class CandidateStatus(str, Enum):
    """Outcome of an exact two-observation candidate calculation."""

    UNIQUE = "unique"
    AMBIGUOUS = "ambiguous"
    NO_SOLUTION = "no_solution"
    DEGENERATE = "degenerate"


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Candidate positions and an explicit geometric outcome."""

    status: CandidateStatus
    positions: tuple[Position, ...]
    warnings: tuple[str, ...]
    message: str

    def __post_init__(self) -> None:
        try:
            status = CandidateStatus(self.status)
        except (TypeError, ValueError) as error:
            raise FixError("status must be a CandidateStatus") from error
        positions = _position_tuple(self.positions, name="positions")
        warnings = _warning_tuple(self.warnings)
        if not isinstance(self.message, str):
            raise FixError("message must be a string")
        if status is CandidateStatus.UNIQUE and len(positions) != 1:
            raise FixError("unique candidate results require one position")
        if status is CandidateStatus.AMBIGUOUS and len(positions) < 2:
            raise FixError(
                "ambiguous candidate results require at least two positions"
            )
        if status in {
            CandidateStatus.NO_SOLUTION,
            CandidateStatus.DEGENERATE,
        } and positions:
            raise FixError(
                "no-solution and degenerate candidate results have no positions"
            )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "warnings", warnings)


@dataclass(frozen=True, slots=True)
class FixResult:
    """A NautiPy-owned fix result with convergence and geometry diagnostics."""

    position: Position | None
    success: bool
    status: FixStatus
    residuals: tuple[ObservationResidual, ...]
    objective: float | None
    rms: float | None
    bearing_rms: float | None
    range_rms: float | None
    iterations: int
    function_evaluations: int
    warnings: tuple[str, ...]
    uncertainty: FixUncertainty | None
    rank: int | None
    condition_number: float | None
    degrees_of_freedom: int | None
    reduced_chi_square: float | None
    competing_positions: tuple[Position, ...]
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise FixError("success must be a boolean")
        try:
            status = FixStatus(self.status)
        except (TypeError, ValueError) as error:
            raise FixError("status must be a FixStatus") from error
        if self.success is not (status is FixStatus.CONVERGED):
            raise FixError("success must be true exactly when status converged")
        if self.position is not None and not isinstance(self.position, Position):
            raise FixError("position must be a Position or None")
        if status is FixStatus.CONVERGED:
            if self.position is None:
                raise FixError("a converged result requires a position")
        elif self.position is not None:
            raise FixError("a non-successful result cannot select a position")

        residuals = _residual_tuple(self.residuals)
        warnings = _warning_tuple(self.warnings)
        competing_positions = _position_tuple(
            self.competing_positions,
            name="competing_positions",
        )
        if status is FixStatus.AMBIGUOUS:
            if len(competing_positions) < 2:
                raise FixError(
                    "an ambiguous result requires at least two competing positions"
                )
        elif competing_positions:
            raise FixError(
                "only an ambiguous result can contain competing positions"
            )
        if not isinstance(self.message, str):
            raise FixError("message must be a string")

        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations < 0
        ):
            raise FixError("iterations must be a non-negative integer")
        if (
            isinstance(self.function_evaluations, bool)
            or not isinstance(self.function_evaluations, int)
            or self.function_evaluations < 0
        ):
            raise FixError(
                "function evaluations must be a non-negative integer"
            )

        objective = _optional_nonnegative(self.objective, name="objective")
        rms = _optional_nonnegative(self.rms, name="RMS")
        bearing_rms = _optional_nonnegative(
            self.bearing_rms,
            name="bearing RMS",
        )
        range_rms = _optional_nonnegative(
            self.range_rms,
            name="range RMS",
        )
        if objective is None:
            if any(
                value is not None for value in (rms, bearing_rms, range_rms)
            ):
                raise FixError("RMS values require an objective")
        else:
            if not residuals:
                raise FixError("an objective requires residuals")
            expected_objective = sum(
                item.standardized_residual**2 for item in residuals
            )
            if not _close(objective, expected_objective):
                raise FixError(
                    "objective must equal the standardized residual sum of squares"
                )
            expected_rms = sqrt(expected_objective / len(residuals))
            if rms is None or not _close(rms, expected_rms):
                raise FixError("RMS must match the objective and residual count")

            bearing_values = [
                item.residual
                for item in residuals
                if isinstance(item.observation, BearingObservation)
            ]
            range_values = [
                item.residual
                for item in residuals
                if isinstance(item.observation, RangeObservation)
            ]
            expected_bearing_rms = (
                sqrt(sum(value**2 for value in bearing_values) / len(bearing_values))
                if bearing_values
                else None
            )
            expected_range_rms = (
                sqrt(sum(value**2 for value in range_values) / len(range_values))
                if range_values
                else None
            )
            if expected_bearing_rms is None:
                if bearing_rms is not None:
                    raise FixError("bearing RMS requires bearing residuals")
            elif bearing_rms is None or not _close(
                bearing_rms,
                expected_bearing_rms,
            ):
                raise FixError("bearing RMS must match bearing residuals")
            if expected_range_rms is None:
                if range_rms is not None:
                    raise FixError("range RMS requires range residuals")
            elif range_rms is None or not _close(
                range_rms,
                expected_range_rms,
            ):
                raise FixError("range RMS must match range residuals")

        if self.rank is None:
            rank = None
        elif (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or not 0 <= self.rank <= 2
        ):
            raise FixError("rank must be 0, 1, 2, or None")
        else:
            rank = self.rank
        condition_number = _optional_nonnegative(
            self.condition_number,
            name="condition number",
        )
        if condition_number is not None and condition_number < 1.0:
            raise FixError("condition number must be at least one")
        if rank is not None and rank < 2 and condition_number is not None:
            raise FixError(
                "condition number must be None for rank-deficient geometry"
            )

        if self.degrees_of_freedom is None:
            degrees_of_freedom = None
        elif (
            isinstance(self.degrees_of_freedom, bool)
            or not isinstance(self.degrees_of_freedom, int)
            or self.degrees_of_freedom < 0
        ):
            raise FixError(
                "degrees of freedom must be a non-negative integer or None"
            )
        else:
            degrees_of_freedom = self.degrees_of_freedom
        if residuals and degrees_of_freedom is not None:
            expected_degrees = len(residuals) - 2
            if expected_degrees < 0 or degrees_of_freedom != expected_degrees:
                raise FixError(
                    "degrees of freedom must equal residual count minus two"
                )
        reduced_chi_square = _optional_nonnegative(
            self.reduced_chi_square,
            name="reduced chi-square",
        )
        if reduced_chi_square is not None:
            if (
                objective is None
                or degrees_of_freedom is None
                or degrees_of_freedom == 0
            ):
                raise FixError(
                    "reduced chi-square requires an objective and positive "
                    "degrees of freedom"
                )
            if not _close(
                reduced_chi_square,
                objective / degrees_of_freedom,
            ):
                raise FixError(
                    "reduced chi-square must equal objective divided by "
                    "degrees of freedom"
                )

        if self.uncertainty is not None and not isinstance(
            self.uncertainty,
            FixUncertainty,
        ):
            raise FixError("uncertainty must be a FixUncertainty or None")
        if status is not FixStatus.CONVERGED and self.uncertainty is not None:
            raise FixError("uncertainty is available only for converged results")
        if status is FixStatus.CONVERGED:
            if len(residuals) < 2 or objective is None:
                raise FixError(
                    "a converged result requires at least two residuals and metrics"
                )
            if rank != 2 or condition_number is None:
                raise FixError(
                    "a converged result requires full-rank geometry diagnostics"
                )
            expected_degrees = len(residuals) - 2
            if degrees_of_freedom != expected_degrees:
                raise FixError(
                    "a converged result requires degrees-of-freedom diagnostics"
                )
            if expected_degrees > 0 and reduced_chi_square is None:
                raise FixError(
                    "positive degrees of freedom require reduced chi-square"
                )
            if expected_degrees == 0 and reduced_chi_square is not None:
                raise FixError(
                    "reduced chi-square is undefined with zero degrees of freedom"
                )

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "rms", rms)
        object.__setattr__(self, "bearing_rms", bearing_rms)
        object.__setattr__(self, "range_rms", range_rms)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "condition_number", condition_number)
        object.__setattr__(self, "degrees_of_freedom", degrees_of_freedom)
        object.__setattr__(
            self,
            "reduced_chi_square",
            reduced_chi_square,
        )
        object.__setattr__(self, "competing_positions", competing_positions)


def _scientific() -> tuple[object, object]:
    """Return optional numerical dependencies without importing them eagerly."""

    try:
        import numpy
    except ModuleNotFoundError as error:
        if error.name != "numpy":
            raise
        raise FixDependencyError(_FIX_INSTALL_MESSAGE) from None

    try:
        from scipy.optimize import least_squares
    except ModuleNotFoundError as error:
        if error.name not in {"numpy", "scipy"}:
            raise
        raise FixDependencyError(_FIX_INSTALL_MESSAGE) from None
    return numpy, least_squares


def two_bearing_candidates(
    first: BearingObservation,
    second: BearingObservation,
    *,
    search_center: _PositionInput | None = None,
    search_radius: float = 500_000.0,
) -> CandidateResult:
    """Return all two-bearing candidates in the bounded WGS84 search disk."""

    numpy, least_squares = _scientific()
    from . import _fix_solver

    return _fix_solver.two_bearing_candidates(
        first,
        second,
        search_center=search_center,
        search_radius=search_radius,
        numpy=numpy,
        least_squares=least_squares,
    )


def two_range_candidates(
    first: RangeObservation,
    second: RangeObservation,
) -> CandidateResult:
    """Return all supported regional WGS84 candidates for two ranges."""

    numpy, least_squares = _scientific()
    from . import _fix_solver

    return _fix_solver.two_range_candidates(
        first,
        second,
        numpy=numpy,
        least_squares=least_squares,
    )


def solve_fix(
    *,
    bearings: Iterable[BearingObservation] = (),
    ranges: Iterable[RangeObservation] = (),
    initial: _PositionInput | None = None,
    search_center: _PositionInput | None = None,
    search_radius: float = 500_000.0,
    max_iterations: int = 200,
) -> FixResult:
    """Estimate a position from bearing and range observations."""

    numpy, least_squares = _scientific()
    from . import _fix_solver

    return _fix_solver.solve_fix(
        bearings=bearings,
        ranges=ranges,
        initial=initial,
        search_center=search_center,
        search_radius=search_radius,
        max_iterations=max_iterations,
        numpy=numpy,
        least_squares=least_squares,
    )
