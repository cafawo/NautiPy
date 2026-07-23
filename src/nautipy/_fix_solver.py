"""Private WGS84 bearing/range candidate geometry and least-squares engine.

The public models and dependency boundary live in :mod:`nautipy.fix`.  This
module deliberately receives NumPy and SciPy's ``least_squares`` callable from
that boundary so importing ordinary NautiPy functionality remains light.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import (
    atan2,
    cos,
    degrees,
    hypot,
    isfinite,
    log,
    radians,
    sin,
    sqrt,
    ulp,
)
from numbers import Real
from typing import Iterable, Sequence

from .coordinates import parse_position
from .errors import FixError
from .fix import _DEGENERATE_CONDITION
from .geodesic import _wgs84
from .position import Position


_MAX_REGIONAL_RADIUS = 2_000_000.0
# Refined roots closer than one millimetre are numerical duplicates.  A
# metre-scale tolerance would erase real, closely spaced alternatives.
_CLUSTER_METRES = 1e-3
_DIFFERENCE_STEP_METRES = 0.5
_MAX_STARTS = 32
_COMPETING_DELTA_CHI_SQUARE = 5.99146454710798
_CONFIDENCE_SCALE_95 = sqrt(-2.0 * log(0.05))
_WEAK_CONDITION = 1_000.0
_COINCIDENT_METRES = 1e-7
_EQUIVALENT_BEARING_DEGREES = 1e-10
_DEGENERATE_CROSSING_SINE = 1e-6
_WEAK_CROSSING_SINE = 1e-3
_EXACT_BEARING_RESIDUAL_DEGREES = 1e-5
_EXACT_RANGE_RESIDUAL_METRES = 1e-3
_LARGE_STANDARDIZED_RMS = 2.0
_DOMAIN_EDGE_FRACTION = 0.9
_LARGE_UNCERTAINTY_FRACTION = 0.25
_RANGE_CIRCLE_SAMPLES = 1_440


def _domain_tolerance(radius: float) -> float:
    return max(1e-3, radius * 1e-10)


def _linear_geometry_tolerance(*values: float) -> float:
    scale = max(1.0, *(abs(value) for value in values))
    return max(1e-6, 64.0 * ulp(scale))


def _squared_geometry_tolerance(*values: float) -> float:
    scale_squared = max(1.0, *(abs(value) ** 2 for value in values))
    return max(1e-12, 8.0 * ulp(scale_squared))


@dataclass(frozen=True, slots=True)
class _Spec:
    observation: object
    kind: str
    scale_override: float | None = None


def _spec_scale(spec: _Spec) -> float:
    if spec.scale_override is not None:
        return spec.scale_override
    return float(spec.observation.uncertainty)  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class _Run:
    position: Position
    converged: bool
    objective: float
    predicted: tuple[float, ...]
    natural: tuple[float, ...]
    standardized: tuple[float, ...]
    jacobian: object
    rank: int
    condition: float | None
    iterations: int
    function_evaluations: int
    at_boundary: bool


class _Chart:
    """Geodesic normal coordinates, east/north metres about one anchor."""

    def __init__(self, anchor: Position) -> None:
        self.anchor = anchor

    def to_position(self, east: float, north: float) -> Position:
        distance = hypot(east, north)
        if distance == 0.0:
            return self.anchor
        azimuth = degrees(atan2(east, north))
        raw = _wgs84().Direct(
            self.anchor.latitude,
            self.anchor.longitude,
            azimuth,
            distance,
        )
        return Position(float(raw["lat2"]), float(raw["lon2"]))

    def to_local(self, position: Position) -> tuple[float, float]:
        distance, bearing = _inverse(self.anchor, position)
        if distance <= _COINCIDENT_METRES or bearing is None:
            return (0.0, 0.0)
        angle = radians(bearing)
        return (distance * sin(angle), distance * cos(angle))


def _inverse(start: Position, end: Position) -> tuple[float, float | None]:
    raw = _wgs84().Inverse(
        start.latitude,
        start.longitude,
        end.latitude,
        end.longitude,
    )
    distance = float(raw["s12"])
    if not isfinite(distance) or distance < 0.0:
        raise FixError("WGS84 prediction returned an invalid distance")
    if distance <= _COINCIDENT_METRES:
        return (0.0, None)
    bearing = float(raw["azi1"]) % 360.0
    if not isfinite(bearing):
        raise FixError("WGS84 prediction returned an invalid bearing")
    return (distance, 0.0 if bearing in {0.0, 360.0} else bearing)


def _wrap_bearing(value: float) -> float:
    wrapped = (value + 180.0) % 360.0 - 180.0
    return 0.0 if wrapped == 0.0 else wrapped


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise FixError(f"{name} must be a real number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise FixError(f"{name} must be representable as a finite float") from error
    if not isfinite(number):
        raise FixError(f"{name} must be finite")
    return number


def _search_radius(value: object) -> float:
    radius = _finite_float(value, name="search radius")
    if not 0.0 < radius <= _MAX_REGIONAL_RADIUS:
        raise FixError(
            "search radius must be greater than zero and no more than "
            "2000000 metres"
        )
    return radius


def _iteration_limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FixError("max_iterations must be a positive integer")
    return value


def _coerce_position(value: object, *, name: str) -> Position:
    try:
        return value if isinstance(value, Position) else parse_position(value)
    except Exception as error:
        if isinstance(error, FixError):
            raise
        raise FixError(f"{name} must be a valid position") from error


def _validated_specs(
    bearings: Iterable[object],
    ranges: Iterable[object],
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[_Spec, ...]]:
    from .fix import BearingObservation, RangeObservation

    if isinstance(bearings, (str, bytes, bytearray)):
        raise FixError("bearings must be an iterable of BearingObservation values")
    if isinstance(ranges, (str, bytes, bytearray)):
        raise FixError("ranges must be an iterable of RangeObservation values")
    try:
        bearing_values = tuple(bearings)
        range_values = tuple(ranges)
    except TypeError as error:
        raise FixError("bearings and ranges must be iterables") from error
    if any(not isinstance(item, BearingObservation) for item in bearing_values):
        raise FixError("bearings must contain only BearingObservation values")
    if any(not isinstance(item, RangeObservation) for item in range_values):
        raise FixError("ranges must contain only RangeObservation values")
    specs = tuple(_Spec(item, "bearing") for item in bearing_values) + tuple(
        _Spec(item, "range") for item in range_values
    )
    if len(specs) < 2:
        raise FixError("at least two bearing/range observations are required")
    return bearing_values, range_values, specs


def _prediction(spec: _Spec, position: Position) -> tuple[float, float]:
    observation = spec.observation
    distance, bearing = _inverse(position, observation.reference)  # type: ignore[attr-defined]
    if spec.kind == "bearing":
        observed = float(observation.bearing)  # type: ignore[attr-defined]
        if bearing is None:
            predicted = (observed + 180.0) % 360.0
        else:
            predicted = bearing
        return predicted, _wrap_bearing(predicted - observed)
    observed_distance = float(observation.distance)  # type: ignore[attr-defined]
    return distance, distance - observed_distance


def _values(
    specs: Sequence[_Spec],
    position: Position,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    predicted: list[float] = []
    natural: list[float] = []
    standardized: list[float] = []
    for spec in specs:
        model, residual = _prediction(spec, position)
        predicted.append(model)
        natural.append(residual)
        standardized.append(residual / _spec_scale(spec))
    return tuple(predicted), tuple(natural), tuple(standardized)


class _Problem:
    def __init__(self, specs: Sequence[_Spec], chart: _Chart, numpy: object) -> None:
        self.specs = specs
        self.chart = chart
        self.numpy = numpy
        self.iterations = 0

    def residuals(self, coordinates: object) -> object:
        east = float(coordinates[0])  # type: ignore[index]
        north = float(coordinates[1])  # type: ignore[index]
        position = self.chart.to_position(east, north)
        standardized = _values(self.specs, position)[2]
        return self.numpy.asarray(standardized, dtype=float)  # type: ignore[attr-defined]

    def jacobian(self, coordinates: object) -> object:
        self.iterations += 1
        east = float(coordinates[0])  # type: ignore[index]
        north = float(coordinates[1])  # type: ignore[index]
        step = _DIFFERENCE_STEP_METRES
        columns: list[list[float]] = []
        for delta_east, delta_north in ((step, 0.0), (0.0, step)):
            plus = self.chart.to_position(east + delta_east, north + delta_north)
            minus = self.chart.to_position(east - delta_east, north - delta_north)
            column: list[float] = []
            for spec in self.specs:
                plus_prediction, _ = _prediction(spec, plus)
                minus_prediction, _ = _prediction(spec, minus)
                difference = (
                    _wrap_bearing(plus_prediction - minus_prediction)
                    if spec.kind == "bearing"
                    else plus_prediction - minus_prediction
                )
                column.append(difference / (2.0 * step * _spec_scale(spec)))
            columns.append(column)
        return self.numpy.asarray(columns, dtype=float).T  # type: ignore[attr-defined]


def _rank_condition(jacobian: object, numpy: object) -> tuple[int, float | None]:
    try:
        singular = numpy.linalg.svd(  # type: ignore[attr-defined]
            jacobian,
            compute_uv=False,
            full_matrices=False,
        )
    except Exception:
        return (0, None)
    values = tuple(float(value) for value in singular)
    if not values or not isfinite(values[0]) or values[0] <= 0.0:
        return (0, None)
    epsilon = float(numpy.finfo(float).eps)  # type: ignore[attr-defined]
    shape = getattr(jacobian, "shape", (len(values), 2))
    threshold = max(int(shape[0]), int(shape[1])) * epsilon * values[0]
    rank = sum(value > threshold for value in values)
    if rank < 2 or len(values) < 2 or values[1] <= 0.0:
        return (rank, None)
    condition = values[0] / values[1]
    if not isfinite(condition):
        return (rank, None)
    return (rank, max(1.0, condition))


def _optimize(
    specs: Sequence[_Spec],
    chart: _Chart,
    start: tuple[float, float],
    *,
    bound: float,
    max_iterations: int,
    numpy: object,
    least_squares: object,
) -> _Run | None:
    margin = max(1e-6, bound * 1e-12)
    clipped = (
        max(-bound + margin, min(bound - margin, float(start[0]))),
        max(-bound + margin, min(bound - margin, float(start[1]))),
    )
    problem = _Problem(specs, chart, numpy)
    scale = max(1.0, min(100_000.0, bound / 10.0))
    try:
        result = least_squares(  # type: ignore[operator]
            problem.residuals,
            numpy.asarray(clipped, dtype=float),  # type: ignore[attr-defined]
            jac=problem.jacobian,
            bounds=(
                numpy.asarray((-bound, -bound), dtype=float),  # type: ignore[attr-defined]
                numpy.asarray((bound, bound), dtype=float),  # type: ignore[attr-defined]
            ),
            method="trf",
            loss="linear",
            x_scale=numpy.asarray((scale, scale), dtype=float),  # type: ignore[attr-defined]
            max_nfev=max_iterations,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        coordinates = (float(result.x[0]), float(result.x[1]))
        if not all(isfinite(value) for value in coordinates):
            return None
        position = chart.to_position(*coordinates)
        predicted, natural, standardized = _values(specs, position)
        if not all(isfinite(value) for value in standardized):
            return None
        optimizer_iterations = problem.iterations
        jacobian = problem.jacobian(result.x)
        rank, condition = _rank_condition(jacobian, numpy)
        objective = sum(value * value for value in standardized)
        if not isfinite(objective):
            return None
        at_boundary = any(int(value) != 0 for value in result.active_mask)
        return _Run(
            position=position,
            converged=bool(result.success),
            objective=float(objective),
            predicted=predicted,
            natural=natural,
            standardized=standardized,
            jacobian=jacobian,
            rank=rank,
            condition=condition,
            iterations=int(optimizer_iterations),
            function_evaluations=int(getattr(result, "nfev", 0)),
            at_boundary=at_boundary,
        )
    except (ArithmeticError, TypeError, ValueError, RuntimeError):
        return None


def _distance(first: Position, second: Position) -> float:
    return _inverse(first, second)[0]


def _regional_center(references: Sequence[Position]) -> Position:
    if not references:
        raise FixError("at least one observation reference is required")
    medoid = min(
        references,
        key=lambda candidate: sum(_distance(candidate, other) for other in references),
    )
    chart = _Chart(medoid)
    local = [chart.to_local(reference) for reference in references]
    east = sum(point[0] for point in local) / len(local)
    north = sum(point[1] for point in local) / len(local)
    if hypot(east, north) > _MAX_REGIONAL_RADIUS:
        return medoid
    return chart.to_position(east, north)


def _line_for_bearing(
    observation: object,
    chart: _Chart,
    step: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    reference = observation.reference  # type: ignore[attr-defined]
    origin = chart.to_local(reference)
    raw = _wgs84().Direct(
        reference.latitude,
        reference.longitude,
        (float(observation.bearing) + 180.0) % 360.0,  # type: ignore[attr-defined]
        step,
    )
    forward = chart.to_local(Position(float(raw["lat2"]), float(raw["lon2"])))
    direction = (forward[0] - origin[0], forward[1] - origin[1])
    norm = hypot(*direction)
    if norm <= _COINCIDENT_METRES:
        angle = radians((float(observation.bearing) + 180.0) % 360.0)  # type: ignore[attr-defined]
        return origin, (sin(angle), cos(angle))
    return origin, (direction[0] / norm, direction[1] / norm)


def _cross(first: tuple[float, float], second: tuple[float, float]) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _bearing_pair_seed(
    first: object,
    second: object,
    chart: _Chart,
    scale: float,
) -> tuple[tuple[float, float] | None, str, float]:
    if (
        _distance(first.reference, second.reference)  # type: ignore[attr-defined]
        <= _COINCIDENT_METRES
    ):
        return None, "degenerate", 0.0
    step = max(100.0, min(1_000.0, scale * 1e-3))
    first_origin, first_direction = _line_for_bearing(first, chart, step)
    second_origin, second_direction = _line_for_bearing(second, chart, step)
    determinant = _cross(first_direction, second_direction)
    sine_crossing = abs(determinant)
    if sine_crossing < _DEGENERATE_CROSSING_SINE:
        return None, "degenerate", sine_crossing
    separation = (
        second_origin[0] - first_origin[0],
        second_origin[1] - first_origin[1],
    )
    first_along = _cross(separation, second_direction) / determinant
    second_along = _cross(separation, first_direction) / determinant
    if first_along < -1.0 or second_along < -1.0:
        return None, "reverse", sine_crossing
    seed = (
        first_origin[0] + first_along * first_direction[0],
        first_origin[1] + first_along * first_direction[1],
    )
    return seed, "ok", sine_crossing


def _circle_pair_seeds(
    first_center: tuple[float, float],
    first_radius: float,
    second_center: tuple[float, float],
    second_radius: float,
) -> tuple[str, tuple[tuple[float, float], ...]]:
    delta = (
        second_center[0] - first_center[0],
        second_center[1] - first_center[1],
    )
    separation = hypot(*delta)
    if first_radius == 0.0 and second_radius == 0.0:
        if separation <= _COINCIDENT_METRES:
            return "point", (first_center,)
        return "none", ()
    tolerance = _linear_geometry_tolerance(
        first_radius,
        second_radius,
        separation,
    )
    if separation <= tolerance:
        if abs(first_radius - second_radius) <= tolerance:
            return "concentric", ()
        return "none", ()
    if separation > first_radius + second_radius + tolerance:
        return "none", ()
    if separation < abs(first_radius - second_radius) - tolerance:
        return "none", ()
    along = (
        first_radius * first_radius
        - second_radius * second_radius
        + separation * separation
    ) / (2.0 * separation)
    height_squared = (first_radius - along) * (first_radius + along)
    height_tolerance = _squared_geometry_tolerance(first_radius, along)
    unit = (delta[0] / separation, delta[1] / separation)
    base = (
        first_center[0] + along * unit[0],
        first_center[1] + along * unit[1],
    )
    if height_squared <= 0.0:
        if height_squared < -height_tolerance:
            return "none", ()
        return "tangent", (base,)
    height = sqrt(height_squared)
    perpendicular = (-unit[1], unit[0])
    return (
        "two",
        (
            (
                base[0] + height * perpendicular[0],
                base[1] + height * perpendicular[1],
            ),
            (
                base[0] - height * perpendicular[0],
                base[1] - height * perpendicular[1],
            ),
        ),
    )


def _ray_circle_seeds(
    bearing: object,
    range_observation: object,
    chart: _Chart,
    scale: float,
) -> tuple[tuple[float, float], ...]:
    origin, direction = _line_for_bearing(bearing, chart, max(100.0, min(1_000.0, scale * 1e-3)))
    center = chart.to_local(range_observation.reference)  # type: ignore[attr-defined]
    relative = (origin[0] - center[0], origin[1] - center[1])
    projection = direction[0] * relative[0] + direction[1] * relative[1]
    constant = (
        relative[0] * relative[0]
        + relative[1] * relative[1]
        - float(range_observation.distance) ** 2  # type: ignore[attr-defined]
    )
    discriminant = projection * projection - constant
    tolerance = max(1e-6, scale * scale * 1e-12)
    if discriminant < -tolerance:
        return ()
    root = sqrt(max(0.0, discriminant))
    seeds: list[tuple[float, float]] = []
    for along in (-projection - root, -projection + root):
        if along >= -1.0:
            seeds.append(
                (
                    origin[0] + max(0.0, along) * direction[0],
                    origin[1] + max(0.0, along) * direction[1],
                )
            )
    return tuple(seeds)


def _range_circle_position(observation: object, azimuth: float) -> Position:
    reference = observation.reference  # type: ignore[attr-defined]
    raw = _wgs84().Direct(
        reference.latitude,
        reference.longitude,
        azimuth,
        float(observation.distance),  # type: ignore[attr-defined]
    )
    return Position(float(raw["lat2"]), float(raw["lon2"]))


def _bearing_circle_signal(
    bearing: object,
    range_observation: object,
    azimuth: float,
) -> tuple[float, float]:
    position = _range_circle_position(range_observation, azimuth)
    _, error = _prediction(_Spec(bearing, "bearing", 1.0), position)
    angle = radians(error)
    return sin(angle), cos(angle)


def _exact_bearing_range_seeds(
    bearing: object,
    range_observation: object,
    chart: _Chart,
) -> tuple[tuple[float, float], ...]:
    """Bracket exact bearing roots around a WGS84 range circle."""

    if float(range_observation.distance) == 0.0:  # type: ignore[attr-defined]
        return (chart.to_local(range_observation.reference),)  # type: ignore[attr-defined]
    step = 360.0 / _RANGE_CIRCLE_SAMPLES
    samples = tuple(
        _bearing_circle_signal(bearing, range_observation, index * step)
        for index in range(_RANGE_CIRCLE_SAMPLES)
    )
    angles: list[float] = []
    for index, (low_signal, low_alignment) in enumerate(samples):
        low_angle = index * step
        high_angle = (index + 1) * step
        high_signal, _ = samples[(index + 1) % _RANGE_CIRCLE_SAMPLES]
        if abs(low_signal) <= 1e-13 and low_alignment > 0.0:
            angles.append(low_angle)
        if low_signal * high_signal >= 0.0:
            continue
        low = low_angle
        high = high_angle
        for _ in range(60):
            middle = (low + high) / 2.0
            middle_signal, _ = _bearing_circle_signal(
                bearing,
                range_observation,
                middle % 360.0,
            )
            if low_signal * middle_signal <= 0.0:
                high = middle
            else:
                low = middle
                low_signal = middle_signal
        root = (low + high) / 2.0
        _, alignment = _bearing_circle_signal(
            bearing,
            range_observation,
            root % 360.0,
        )
        if alignment > 0.0:
            angles.append(root % 360.0)

    # A tangential root need not change sign.  Retaining local minima supplies
    # deterministic refinement starts while the exact residual filter remains
    # responsible for accepting or rejecting them.
    for index, (signal, alignment) in enumerate(samples):
        previous = abs(samples[index - 1][0])
        following = abs(samples[(index + 1) % _RANGE_CIRCLE_SAMPLES][0])
        if alignment > 0.0 and abs(signal) <= previous and abs(signal) <= following:
            angles.append(index * step)

    return tuple(
        chart.to_local(_range_circle_position(range_observation, angle))
        for angle in angles
    )


def _deduplicate_points(
    points: Iterable[tuple[float, float]],
    radius: float,
) -> list[tuple[float, float]]:
    unique: list[tuple[float, float]] = []
    for point in points:
        if not all(isfinite(value) for value in point):
            continue
        if hypot(*point) > radius + _domain_tolerance(radius):
            continue
        if all(
            hypot(point[0] - other[0], point[1] - other[1])
            > _CLUSTER_METRES
            for other in unique
        ):
            unique.append((float(point[0]), float(point[1])))
    return unique


def _cluster_positions(positions: Iterable[Position]) -> tuple[Position, ...]:
    unique: list[Position] = []
    for position in positions:
        if all(_distance(position, other) > _CLUSTER_METRES for other in unique):
            unique.append(position)
    return tuple(unique)


def _candidate_result(
    status: object,
    positions: Sequence[Position],
    warnings: Sequence[str],
    message: str,
) -> object:
    from .fix import CandidateResult

    return CandidateResult(
        status=status,
        positions=tuple(positions),
        warnings=tuple(warnings),
        message=message,
    )


def two_bearing_candidates(
    first: object,
    second: object,
    *,
    search_center: object,
    search_radius: object,
    numpy: object,
    least_squares: object,
) -> object:
    """Return the bounded regional candidate for two unknown-to-reference bearings."""

    from .fix import BearingObservation, CandidateStatus

    if not isinstance(first, BearingObservation) or not isinstance(second, BearingObservation):
        raise FixError("two-bearing candidates require BearingObservation values")
    radius = _search_radius(search_radius)
    center = (
        _regional_center((first.reference, second.reference))
        if search_center is None
        else _coerce_position(search_center, name="search_center")
    )
    if abs(center.latitude) == 90.0:
        raise FixError("an exact pole cannot be used as the local search center")
    if _distance(first.reference, second.reference) <= _COINCIDENT_METRES:
        if (
            abs(_wrap_bearing(first.bearing - second.bearing))
            <= _EQUIVALENT_BEARING_DEGREES
        ):
            return _candidate_result(
                CandidateStatus.DEGENERATE,
                (),
                ("equivalent bearings from one reference define a ray, not a point",),
                "two-bearing geometry is degenerate",
            )
        return _candidate_result(
            CandidateStatus.NO_SOLUTION,
            (),
            ("distinct bearings toward one reference are inconsistent",),
            "no two-bearing candidate exists",
        )
    chart = _Chart(center)
    seed, heuristic_reason, crossing = _bearing_pair_seed(
        first,
        second,
        chart,
        radius,
    )
    starts: list[tuple[float, float]] = [(0.0, 0.0)]
    if seed is not None:
        starts.append(seed)
    for fraction in (0.2, 0.45, 0.7, 0.9):
        ring = radius * fraction
        for index in range(16):
            angle = radians(index * 22.5)
            starts.append((ring * sin(angle), ring * cos(angle)))
    starts = _deduplicate_points(starts, radius)
    specs = (_Spec(first, "bearing", 1.0), _Spec(second, "bearing", 1.0))
    runs: list[_Run] = []
    for start in starts:
        run = _optimize(
            specs,
            chart,
            start,
            bound=radius,
            max_iterations=200,
            numpy=numpy,
            least_squares=least_squares,
        )
        if (
            run is not None
            and run.converged
            and _distance(center, run.position)
            <= radius + _domain_tolerance(radius)
            and max(abs(value) for value in run.natural)
            <= _EXACT_BEARING_RESIDUAL_DEGREES
        ):
            runs.append(run)
    candidates = _cluster_runs(runs)
    if not candidates:
        if heuristic_reason == "degenerate":
            return _candidate_result(
                CandidateStatus.DEGENERATE,
                (),
                ("bearing loci are parallel or have no stable intersection",),
                "two-bearing geometry is degenerate",
            )
        return _candidate_result(
            CandidateStatus.NO_SOLUTION,
            (),
            ("no exact bearing-locus intersection exists in the search region",),
            "no bounded WGS84 bearing candidate was found",
        )
    stable = tuple(
        run
        for run in candidates
        if run.rank == 2
        and run.condition is not None
        and run.condition <= _DEGENERATE_CONDITION
    )
    if not stable:
        return _candidate_result(
            CandidateStatus.DEGENERATE,
            (),
            ("exact bearing-locus intersections are rank deficient or unstable",),
            "two-bearing geometry is degenerate",
        )
    if len(stable) > 1:
        return _candidate_result(
            CandidateStatus.AMBIGUOUS,
            tuple(run.position for run in stable),
            (
                "multiple exact bearing-locus intersections exist in the "
                "search region",
            ),
            "multiple bounded WGS84 bearing candidates were found",
        )
    run = stable[0]
    warnings: list[str] = []
    if crossing < _WEAK_CROSSING_SINE or run.condition > _WEAK_CONDITION:
        warnings.append("bearing loci intersect at a weak angle")
    return _candidate_result(
        CandidateStatus.UNIQUE,
        (run.position,),
        warnings,
        "one bounded WGS84 bearing candidate was found",
    )


def two_range_candidates(
    first: object,
    second: object,
    *,
    numpy: object,
    least_squares: object,
) -> object:
    """Return all regional WGS84 intersections of two range circles."""

    from .fix import CandidateStatus, RangeObservation

    if not isinstance(first, RangeObservation) or not isinstance(second, RangeObservation):
        raise FixError("two-range candidates require RangeObservation values")
    if max(first.distance, second.distance) > _MAX_REGIONAL_RADIUS:
        raise FixError(
            "two-range candidate distances must not exceed 2000000 metres"
        )
    chart = _Chart(first.reference)
    second_center = chart.to_local(second.reference)
    geometry, seeds = _circle_pair_seeds(
        (0.0, 0.0),
        float(first.distance),
        second_center,
        float(second.distance),
    )
    if geometry == "concentric":
        return _candidate_result(
            CandidateStatus.DEGENERATE,
            (),
            ("coincident equal range circles have infinitely many solutions",),
            "two-range geometry is degenerate",
        )
    if geometry == "none":
        return _candidate_result(
            CandidateStatus.NO_SOLUTION,
            (),
            ("the range circles do not intersect",),
            "no two-range candidate exists",
        )
    if geometry == "point":
        return _candidate_result(
            CandidateStatus.UNIQUE,
            (first.reference,),
            ("coincident zero-radius circles identify their shared reference",),
            "one zero-radius WGS84 range candidate was found",
        )
    specs = (_Spec(first, "range", 1.0), _Spec(second, "range", 1.0))
    refined: list[Position] = []
    scale = max(1_000.0, first.distance, second.distance, hypot(*second_center))
    bound = min(_MAX_REGIONAL_RADIUS, max(10_000.0, scale * 0.25))
    for seed in seeds:
        position = chart.to_position(*seed)
        local_chart = _Chart(position)
        run = _optimize(
            specs,
            local_chart,
            (0.0, 0.0),
            bound=bound,
            max_iterations=200,
            numpy=numpy,
            least_squares=least_squares,
        )
        if (
            run is not None
            and run.converged
            and not run.at_boundary
            and max(abs(value) for value in run.natural)
            <= _EXACT_RANGE_RESIDUAL_METRES
        ):
            refined.append(run.position)
    positions = _cluster_positions(refined)
    if geometry == "tangent" and len(positions) == 1:
        return _candidate_result(
            CandidateStatus.UNIQUE,
            positions,
            ("tangent range circles have a unique but rank-deficient candidate",),
            "one tangent WGS84 range candidate was found",
        )
    if len(positions) == 1:
        return _candidate_result(
            CandidateStatus.UNIQUE,
            positions,
            ("only one exact WGS84 range intersection was retained",),
            "one WGS84 range candidate was found",
        )
    if len(positions) < 2:
        return _candidate_result(
            CandidateStatus.NO_SOLUTION,
            (),
            ("exact WGS84 refinement did not preserve both range intersections",),
            "no reliable two-range candidate set was found",
        )
    return _candidate_result(
        CandidateStatus.AMBIGUOUS,
        positions,
        ("two range observations produce two valid positions",),
        "two WGS84 range candidates were found",
    )


def _linear_bearing_seed(
    bearings: Sequence[object],
    chart: _Chart,
    scale: float,
    numpy: object,
) -> tuple[float, float] | None:
    if len(bearings) < 2:
        return None
    rows: list[tuple[float, float]] = []
    values: list[float] = []
    for observation in bearings:
        origin, direction = _line_for_bearing(
            observation,
            chart,
            max(100.0, min(1_000.0, scale * 1e-3)),
        )
        normal = (-direction[1], direction[0])
        weight = 1.0 / float(observation.uncertainty)
        rows.append((normal[0] * weight, normal[1] * weight))
        values.append((normal[0] * origin[0] + normal[1] * origin[1]) * weight)
    try:
        solution = numpy.linalg.lstsq(  # type: ignore[attr-defined]
            numpy.asarray(rows, dtype=float),  # type: ignore[attr-defined]
            numpy.asarray(values, dtype=float),  # type: ignore[attr-defined]
            rcond=None,
        )[0]
        return (float(solution[0]), float(solution[1]))
    except Exception:
        return None


def _linear_range_seed(
    ranges: Sequence[object],
    chart: _Chart,
    numpy: object,
) -> tuple[float, float] | None:
    if len(ranges) < 3:
        return None
    centers = [chart.to_local(observation.reference) for observation in ranges]
    first_center = centers[0]
    first_radius = float(ranges[0].distance)
    rows: list[tuple[float, float]] = []
    values: list[float] = []
    for observation, center in zip(ranges[1:], centers[1:]):
        rows.append(
            (
                2.0 * (center[0] - first_center[0]),
                2.0 * (center[1] - first_center[1]),
            )
        )
        values.append(
            center[0] ** 2
            + center[1] ** 2
            - first_center[0] ** 2
            - first_center[1] ** 2
            - float(observation.distance) ** 2
            + first_radius**2
        )
    try:
        solution = numpy.linalg.lstsq(  # type: ignore[attr-defined]
            numpy.asarray(rows, dtype=float),  # type: ignore[attr-defined]
            numpy.asarray(values, dtype=float),  # type: ignore[attr-defined]
            rcond=None,
        )[0]
        return (float(solution[0]), float(solution[1]))
    except Exception:
        return None


def _seed_points(
    bearings: Sequence[object],
    ranges: Sequence[object],
    chart: _Chart,
    radius: float,
    numpy: object,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for observation in ranges:
        points.append(chart.to_local(observation.reference))

    for first_index, first in enumerate(bearings):
        for second in bearings[first_index + 1 :]:
            seed, reason, _ = _bearing_pair_seed(first, second, chart, radius)
            if seed is not None and reason == "ok":
                points.append(seed)
    for first_index, first in enumerate(ranges):
        first_center = chart.to_local(first.reference)
        for second in ranges[first_index + 1 :]:
            _, seeds = _circle_pair_seeds(
                first_center,
                float(first.distance),
                chart.to_local(second.reference),
                float(second.distance),
            )
            points.extend(seeds)
    for bearing in bearings:
        for range_observation in ranges:
            points.extend(_ray_circle_seeds(bearing, range_observation, chart, radius))
    if len(bearings) == 1 and len(ranges) == 1:
        points.extend(
            _exact_bearing_range_seeds(bearings[0], ranges[0], chart)
        )

    bearing_seed = _linear_bearing_seed(bearings, chart, radius, numpy)
    if bearing_seed is not None:
        points.append(bearing_seed)
    range_seed = _linear_range_seed(ranges, chart, numpy)
    if range_seed is not None:
        points.append(range_seed)

    characteristic = [
        float(observation.distance)
        for observation in ranges
        if 0.0 < float(observation.distance) <= radius
    ]
    ring = min(radius * 0.5, max(characteristic, default=radius * 0.25))
    if ring > 0.0:
        for index in range(8):
            angle = radians(index * 45.0)
            points.append((ring * sin(angle), ring * cos(angle)))
    return _deduplicate_points(points, radius)


def _score_seed(specs: Sequence[_Spec], chart: _Chart, seed: tuple[float, float]) -> float:
    try:
        standardized = _values(specs, chart.to_position(*seed))[2]
        return sum(value * value for value in standardized)
    except (ArithmeticError, FixError, ValueError):
        return float("inf")


def _cluster_runs(runs: Iterable[_Run]) -> list[_Run]:
    clustered: list[_Run] = []
    for run in sorted(runs, key=lambda item: item.objective):
        existing = next(
            (
                index
                for index, other in enumerate(clustered)
                if _distance(run.position, other.position) <= _CLUSTER_METRES
            ),
            None,
        )
        if existing is None:
            clustered.append(run)
        elif run.objective < clustered[existing].objective:
            clustered[existing] = run
    return sorted(clustered, key=lambda item: item.objective)


def _observation_residuals(specs: Sequence[_Spec], run: _Run) -> tuple[object, ...]:
    from .fix import ObservationResidual

    return tuple(
        ObservationResidual(
            observation=spec.observation,
            predicted=float(predicted),
            residual=float(natural),
            standardized_residual=float(standardized),
        )
        for spec, predicted, natural, standardized in zip(
            specs,
            run.predicted,
            run.natural,
            run.standardized,
        )
    )


def _metrics(
    specs: Sequence[_Spec],
    run: _Run,
) -> tuple[float, float, float | None, float | None, int, float | None]:
    from .fix import _root_mean_square

    objective = float(run.objective)
    rms = sqrt(objective / len(specs))
    bearing_values = [
        residual for spec, residual in zip(specs, run.natural) if spec.kind == "bearing"
    ]
    range_values = [
        residual for spec, residual in zip(specs, run.natural) if spec.kind == "range"
    ]
    bearing_rms = (
        _root_mean_square(bearing_values)
        if bearing_values
        else None
    )
    range_rms = (
        _root_mean_square(range_values)
        if range_values
        else None
    )
    degrees_of_freedom = len(specs) - 2
    reduced = objective / degrees_of_freedom if degrees_of_freedom > 0 else None
    return objective, rms, bearing_rms, range_rms, degrees_of_freedom, reduced


def _centered_diagnostics(
    specs: Sequence[_Spec],
    position: Position,
    numpy: object,
) -> tuple[object, int, float | None]:
    problem = _Problem(specs, _Chart(position), numpy)
    coordinates = numpy.asarray((0.0, 0.0), dtype=float)  # type: ignore[attr-defined]
    jacobian = problem.jacobian(coordinates)
    rank, condition = _rank_condition(jacobian, numpy)
    return jacobian, rank, condition


def _uncertainty(jacobian: object, numpy: object) -> object | None:
    from .fix import (
        FixUncertainty,
        _covariance_axis_from_east,
        _covariance_principal_standard_deviations,
    )

    try:
        information = jacobian.T @ jacobian  # type: ignore[operator]
        covariance = numpy.linalg.inv(information)  # type: ignore[attr-defined]
        east_variance = float(covariance[0, 0])
        east_north = float((covariance[0, 1] + covariance[1, 0]) / 2.0)
        north_variance = float(covariance[1, 1])
        if min(east_variance, north_variance) < 0.0:
            return None
        east_sd = sqrt(max(0.0, east_variance))
        north_sd = sqrt(max(0.0, north_variance))
        denominator = east_sd * north_sd
        correlation = east_north / denominator if denominator > 0.0 else 0.0
        correlation = max(-1.0, min(1.0, correlation))
        major_deviation, minor_deviation, isotropic = (
            _covariance_principal_standard_deviations(
                east_variance,
                east_north,
                north_variance,
            )
        )
        if isotropic:
            bearing = None
        else:
            axis_from_east = _covariance_axis_from_east(
                east_variance,
                east_north,
                north_variance,
            )
            bearing = (90.0 - degrees(axis_from_east)) % 180.0
        return FixUncertainty(
            covariance=(
                (east_variance, east_north),
                (east_north, north_variance),
            ),
            east_standard_deviation=east_sd,
            north_standard_deviation=north_sd,
            correlation=correlation,
            semi_major_95=major_deviation * _CONFIDENCE_SCALE_95,
            semi_minor_95=minor_deviation * _CONFIDENCE_SCALE_95,
            major_axis_bearing=bearing,
        )
    except (ArithmeticError, FixError, ValueError):
        return None


def _failed_result(
    status: object,
    *,
    message: str,
    warnings: Sequence[str] = (),
    competing: Sequence[Position] = (),
    specs: Sequence[_Spec] = (),
    run: _Run | None = None,
    rank: int | None = None,
    condition: float | None = None,
) -> object:
    from .fix import FixResult

    if run is None:
        residuals: tuple[object, ...] = ()
        objective = rms = bearing_rms = range_rms = reduced = None
        degrees_of_freedom = None
        iterations = function_evaluations = 0
    else:
        residuals = _observation_residuals(specs, run)
        objective, rms, bearing_rms, range_rms, degrees_of_freedom, reduced = _metrics(specs, run)
        iterations = run.iterations
        function_evaluations = run.function_evaluations
    return FixResult(
        position=None,
        success=False,
        status=status,
        residuals=residuals,
        objective=objective,
        rms=rms,
        bearing_rms=bearing_rms,
        range_rms=range_rms,
        iterations=int(iterations),
        function_evaluations=int(function_evaluations),
        warnings=tuple(warnings),
        uncertainty=None,
        rank=rank,
        condition_number=condition,
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi_square=reduced,
        competing_positions=tuple(competing),
        message=message,
    )


def solve_fix(
    *,
    bearings: Iterable[object],
    ranges: Iterable[object],
    initial: object,
    search_center: object,
    search_radius: object,
    max_iterations: object,
    numpy: object,
    least_squares: object,
) -> object:
    """Solve a bounded regional WGS84 fix with deterministic multistart.

    The declared circular region is a closed post-fit domain.  Converged
    optima outside it are rejected rather than projected onto its boundary.
    """

    from .fix import CandidateStatus, FixResult, FixStatus

    bearing_values, range_values, specs = _validated_specs(bearings, ranges)
    radius = _search_radius(search_radius)
    iteration_limit = _iteration_limit(max_iterations)
    initial_position = None if initial is None else _coerce_position(initial, name="initial")
    references = tuple(
        observation.reference for observation in (*bearing_values, *range_values)
    )
    center = (
        _regional_center(references)
        if search_center is None
        else _coerce_position(search_center, name="search_center")
    )
    if abs(center.latitude) == 90.0:
        raise FixError("an exact pole cannot be used as the local search center")
    if (
        initial_position is not None
        and _distance(center, initial_position)
        > radius + _domain_tolerance(radius)
    ):
        raise FixError("initial must lie within the declared search disk")

    forced_seed: Position | None = None
    if (
        len(specs) == 2
        and len(range_values) == 2
        and max(observation.distance for observation in range_values)
        <= _MAX_REGIONAL_RADIUS
    ):
        candidates = two_range_candidates(
            range_values[0],
            range_values[1],
            numpy=numpy,
            least_squares=least_squares,
        )
        in_domain = tuple(
            position
            for position in candidates.positions
            if _distance(center, position)
            <= radius + _domain_tolerance(radius)
        )
        if len(in_domain) >= 2:
            return _failed_result(
                FixStatus.AMBIGUOUS,
                message="two ranges leave two competing positions",
                warnings=candidates.warnings,
                competing=in_domain,
            )
        if len(in_domain) == 1:
            forced_seed = in_domain[0]
        elif candidates.positions:
            return _failed_result(
                FixStatus.NO_SOLUTION,
                message="all two-range candidates are outside the search region",
                warnings=("the declared search disk excludes every range intersection",),
            )
        if candidates.status is CandidateStatus.DEGENERATE:
            return _failed_result(
                FixStatus.DEGENERATE,
                message=candidates.message,
                warnings=candidates.warnings,
            )
        if candidates.status is CandidateStatus.NO_SOLUTION:
            return _failed_result(
                FixStatus.NO_SOLUTION,
                message=candidates.message,
                warnings=candidates.warnings,
            )
        if any("rank-deficient" in warning for warning in candidates.warnings):
            return _failed_result(
                FixStatus.DEGENERATE,
                message="tangent ranges do not determine two stable position axes",
                warnings=candidates.warnings,
                rank=1,
            )
        if forced_seed is None:
            forced_seed = candidates.positions[0]
    elif len(specs) == 2 and len(bearing_values) == 2:
        candidates = two_bearing_candidates(
            bearing_values[0],
            bearing_values[1],
            search_center=center,
            search_radius=radius,
            numpy=numpy,
            least_squares=least_squares,
        )
        if candidates.status is CandidateStatus.DEGENERATE:
            return _failed_result(
                FixStatus.DEGENERATE,
                message=candidates.message,
                warnings=candidates.warnings,
            )
        if candidates.status is CandidateStatus.NO_SOLUTION:
            return _failed_result(
                FixStatus.NO_SOLUTION,
                message=candidates.message,
                warnings=candidates.warnings,
            )
        if candidates.status is CandidateStatus.AMBIGUOUS:
            return _failed_result(
                FixStatus.AMBIGUOUS,
                message=candidates.message,
                warnings=candidates.warnings,
                competing=candidates.positions,
            )
        forced_seed = candidates.positions[0]

    chart = _Chart(center)
    priority_positions = tuple(
        position
        for position in (initial_position, forced_seed)
        if position is not None
    )
    priority_seeds = _deduplicate_points(
        (chart.to_local(position) for position in priority_positions),
        radius,
    )
    generated_seeds = _seed_points(
        bearing_values,
        range_values,
        chart,
        radius,
        numpy,
    )
    generated_seeds = [
        seed
        for seed in generated_seeds
        if all(
            hypot(seed[0] - priority[0], seed[1] - priority[1])
            > _CLUSTER_METRES
            for priority in priority_seeds
        )
    ]
    generated_seeds.sort(key=lambda point: _score_seed(specs, chart, point))
    remaining = max(0, _MAX_STARTS - len(priority_seeds))
    seeds = priority_seeds + generated_seeds[:remaining]
    # The square is only a numerical enclosure for the declared WGS84 disk.
    # Pad it so a valid cardinal-edge optimum is not also an optimizer bound.
    optimizer_bound = radius + max(1.0, 10.0 * _domain_tolerance(radius))

    completed: list[_Run] = []
    singular: list[_Run] = []
    domain_excluded: list[_Run] = []
    incomplete: list[_Run] = []
    for seed in seeds:
        run = _optimize(
            specs,
            chart,
            seed,
            bound=optimizer_bound,
            max_iterations=iteration_limit,
            numpy=numpy,
            least_squares=least_squares,
        )
        if run is None:
            continue
        inside = (
            _distance(center, run.position)
            <= radius + _domain_tolerance(radius)
        )
        if not run.converged:
            incomplete.append(run)
        elif not inside:
            domain_excluded.append(run)
        elif (
            run.rank < 2
            or run.condition is None
            or run.condition > _DEGENERATE_CONDITION
        ):
            singular.append(run)
        else:
            completed.append(run)

    basins = _cluster_runs(completed)
    if not basins:
        best_singular = min(singular, key=lambda item: item.objective, default=None)
        if best_singular is not None:
            return _failed_result(
                FixStatus.DEGENERATE,
                message=(
                    "the converged observations do not determine two stable "
                    "position axes"
                ),
                warnings=("only rank-deficient or effectively singular fits converged",),
                specs=specs,
                run=best_singular,
                rank=best_singular.rank,
                condition=best_singular.condition,
            )
        best_excluded = min(
            domain_excluded,
            key=lambda item: item.objective,
            default=None,
        )
        if best_excluded is not None:
            return _failed_result(
                FixStatus.NO_SOLUTION,
                message="the converged optimum lies outside the search region",
                warnings=(
                    "the declared search disk excludes the converged optimum; "
                    "out-of-domain fits are not projected onto its boundary",
                ),
                specs=specs,
                run=best_excluded,
                rank=best_excluded.rank,
                condition=best_excluded.condition,
            )
        best_incomplete = min(incomplete, key=lambda item: item.objective, default=None)
        return _failed_result(
            FixStatus.NON_CONVERGED,
            message="least-squares fixing did not converge inside the search region",
            warnings=("no deterministic start produced a bounded converged fix",),
            specs=specs,
            run=best_incomplete,
            rank=best_incomplete.rank if best_incomplete is not None else None,
            condition=best_incomplete.condition if best_incomplete is not None else None,
        )

    best = basins[0]
    competing_runs = [
        run
        for run in basins
        if run.objective <= best.objective + _COMPETING_DELTA_CHI_SQUARE
    ]
    if len(competing_runs) > 1:
        return _failed_result(
            FixStatus.AMBIGUOUS,
            message="multiple statistically comparable fixes exist in the search region",
            warnings=("no single position was selected from competing local minima",),
            competing=tuple(run.position for run in competing_runs),
        )

    centered_jacobian, rank, condition = _centered_diagnostics(
        specs,
        best.position,
        numpy,
    )
    if rank < 2 or condition is None or condition > _DEGENERATE_CONDITION:
        return _failed_result(
            FixStatus.DEGENERATE,
            message="the converged observations do not determine two stable position axes",
            warnings=("fix geometry is rank deficient or effectively singular",),
            specs=specs,
            run=best,
            rank=rank,
            condition=condition,
        )

    residuals = _observation_residuals(specs, best)
    objective, rms, bearing_rms, range_rms, degrees_of_freedom, reduced = _metrics(specs, best)
    warnings: list[str] = []
    if condition > _WEAK_CONDITION:
        warnings.append("fix geometry is weak and strongly anisotropic")
    if rms > _LARGE_STANDARDIZED_RMS:
        warnings.append("residuals are large relative to stated uncertainties")
    if _distance(center, best.position) > radius * _DOMAIN_EDGE_FRACTION:
        warnings.append("fix lies near the edge of the circular search region")
    uncertainty = _uncertainty(centered_jacobian, numpy)
    if uncertainty is None:
        warnings.append("linearized covariance could not be computed")
    elif (
        uncertainty.semi_major_95
        > radius * _LARGE_UNCERTAINTY_FRACTION
    ):
        warnings.append("uncertainty is large relative to the search region")

    return FixResult(
        position=best.position,
        success=True,
        status=FixStatus.CONVERGED,
        residuals=residuals,
        objective=objective,
        rms=rms,
        bearing_rms=bearing_rms,
        range_rms=range_rms,
        iterations=int(best.iterations),
        function_evaluations=int(best.function_evaluations),
        warnings=tuple(warnings),
        uncertainty=uncertainty,
        rank=rank,
        condition_number=condition,
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi_square=reduced,
        competing_positions=(),
        message="bounded WGS84 least-squares fix converged",
    )
