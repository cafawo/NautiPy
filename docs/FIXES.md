# Bearing and range position fixes

## Overview

NautiPy estimates a two-dimensional WGS84 position from true bearings, surface
ranges, or both. A result reports convergence, residuals, geometry, competing
solutions, and local uncertainty rather than returning an unqualified
latitude/longitude pair.

The fixing API is available directly from `nautipy` after an ordinary
installation.

## Public API

```text
BearingObservation(reference, bearing, uncertainty)
RangeObservation(reference, distance, uncertainty)

two_bearing_candidates(
    first,
    second,
    *,
    search_center=None,
    search_radius=500_000.0,
) -> CandidateResult

two_range_candidates(first, second) -> CandidateResult

solve_fix(
    *,
    bearings=(),
    ranges=(),
    initial=None,
    search_center=None,
    search_radius=500_000.0,
    max_iterations=200,
) -> FixResult
```

Observation references, `initial`, and `search_center` accept the documented
position-like inputs and are stored or used as validated `Position` values.

## Observations

Each observation relates the unknown fix to a known `reference`.

- `BearingObservation` is the true initial bearing measured **at the unknown
  position toward the reference**, clockwise from true north in degrees.
- `RangeObservation` is the shortest WGS84 surface distance between the
  unknown position and the reference, in metres.
- `uncertainty` is a required, finite, strictly positive one-standard-deviation
  value in the observation's natural unit.

This bearing direction matches taking a bearing from a vessel to a known
landmark. It is not the bearing from the landmark toward the vessel. Do not
reverse it by simply adding 180 degrees: reciprocal initial bearings are not
generally exact opposites on an ellipsoid.

```python
from nautipy import BearingObservation, Position, RangeObservation

landmark = Position(50.116135, 8.670277)

bearing = BearingObservation(
    landmark,
    164.71,
    uncertainty=0.05,
)
observed_range = RangeObservation(
    landmark,
    1_275.251,
    uncertainty=2.0,
)
```

Bearings are normalized to `[0, 360)`. Ranges may be zero but not negative.
Booleans, non-numeric values, non-finite values, and zero or negative
uncertainties are rejected with `FixError`.

Uncertainties are treated as independent, absolute Gaussian standard
deviations. They are required because a mixed metre/degree objective has no
sound implicit weighting. Correlated observations, shared bias estimation,
magnetic correction, and robust-loss models are not supported.

## Solving a fix

```python
from nautipy import Position, RangeObservation, solve_fix

references = (
    Position(50.116135, 8.670277),
    Position(50.112836, 8.666753),
    Position(50.110347, 8.659873),
)
ranges = (
    RangeObservation(references[0], 1_275.251, uncertainty=2.0),
    RangeObservation(references[1], 1_599.237, uncertainty=2.0),
    RangeObservation(references[2], 1_917.145, uncertainty=2.0),
)

result = solve_fix(ranges=ranges)
if result.success:
    print(result.position)
else:
    print(result.status, result.message, result.competing_positions)
```

`solve_fix` accepts bearings, ranges, or both. It consumes each supplied
iterable once. At least two scalar observations are required. Two ranges
commonly have two intersections, so additional evidence is normally required
for a unique fix.

Residuals retain deterministic input order: bearings first, followed by
ranges. Invalid observation numeric fields, observation collection entries,
and solver configuration raise `FixError`. Invalid references passed to an
observation constructor retain the applicable `CoordinateError` subtype.
Numerical and geometry outcomes return a `FixResult`.

`max_iterations` is a positive integer controlling the per-start numerical
search effort. `initial`, when supplied, is the first retained solver start. It
must lie inside the declared search disk and does not enlarge that disk or
authorize the solver to ignore another solution.

## Regional search domain

The fix engine solves regional surface-navigation problems, not unrestricted
global geodesic intersections. Its solution domain is a closed WGS84-distance
disk:

- `search_center` defaults to a deterministic center derived from the
  references;
- `search_radius` defaults to 500,000 metres; and
- radius is finite, strictly positive, and at most 2,000,000 metres.

An exact North or South Pole cannot be the local search center.

The result's uniqueness is a claim only about this domain and NautiPy's
deterministic multistart search. Extended geodesics may intersect elsewhere on
Earth.

Domain membership uses WGS84 surface distance, not rectangular local
coordinates. A fix on the circular boundary is valid. An optimum outside the
disk is never projected onto the boundary. An otherwise valid optimum found
only outside the disk produces `NO_SOLUTION`; rank-deficient or unfinished
searches retain their `DEGENERATE` or `NON_CONVERGED` status.

The optimizer uses bounded local east/north metre coordinates around a WGS84
anchor while evaluating every predicted bearing and range with GeographicLib.
Returned positions and residuals remain WGS84 quantities.

## Residuals and fit metrics

Each `ObservationResidual` contains the original observation, predicted value,
natural residual, and standardized residual.

- Bearing residual is wrapped `predicted - observed` in `[-180, 180)` degrees.
- Range residual is `predicted - observed` in metres.
- Standardized residual is natural residual divided by uncertainty.

A positive bearing residual is clockwise of the observation. A positive range
residual means the prediction is too long.

`FixResult` reports:

- `position`, `success`, `status`, and `message`;
- ordered `residuals`;
- standardized sum-of-squares `objective` and dimensionless `rms`;
- natural-unit `bearing_rms` and `range_rms` where applicable;
- `iterations` and `function_evaluations`;
- local Jacobian `rank` and full-rank `condition_number`;
- `degrees_of_freedom` and `reduced_chi_square`;
- `warnings`;
- selected `uncertainty` where meaningful; and
- `competing_positions` for an ambiguous result.

Degrees of freedom is the residual count minus two. Reduced chi-square is
`objective / degrees_of_freedom` only when that count is positive.

When no fit was evaluated, residuals and derived fit metrics are absent and
solver counts are zero. Every retained evaluated fit has a complete metric
group and positive solver counts. Condition number is present only for
rank-two geometry.

## Fix status

`FixStatus.CONVERGED` is the only successful status. It means the solver found
one in-domain position, converged, and measured a rank-two local Jacobian.

- `AMBIGUOUS`: multiple materially distinct comparable positions fit.
- `DEGENERATE`: observations do not stably constrain both local dimensions.
- `NO_SOLUTION`: no valid solution exists in the declared domain.
- `NON_CONVERGED`: numerical search ended without a valid converged basin.

Every non-success result has `position=None` and `uncertainty=None`.
Ambiguity is represented by deterministic `competing_positions`; NautiPy does
not choose a candidate merely because its optimizer cost is microscopically
lower.

A poor statistical fit can still converge and is reported as a warning. Weak
but full-rank geometry, proximity to the search boundary, and large local
uncertainty also produce warnings.

## Two-observation candidate geometry

`two_bearing_candidates` and `two_range_candidates` expose exact
two-observation geometry through `CandidateResult`.

`CandidateStatus` distinguishes:

- `UNIQUE`: one mathematical candidate;
- `AMBIGUOUS`: multiple candidates;
- `NO_SOLUTION`: no candidate satisfies the observations; and
- `DEGENERATE`: geometry cannot define isolated candidates.

Candidate positions and cardinality depend on measurements and reference
geometry, not observation uncertainty. Uncertainty still controls weighting in
`solve_fix`.

`two_bearing_candidates` searches its declared regional disk.
`two_range_candidates` accepts observed ranges no greater than 2,000,000
metres. A larger range raises `FixError` because it is outside that helper's
regional scope; `solve_fix` does not impose the same observation-distance cap.

Tangent range circles produce one mathematical candidate with
`CandidateStatus.UNIQUE` and a rank-deficiency warning. The same two
observations passed to `solve_fix` produce `FixStatus.DEGENERATE`, because a
successful fix requires two stable local axes. Nearly parallel bearing
geometry may similarly return a candidate with a weak-geometry warning.

## Classification thresholds

These thresholds affect public status or warnings:

- candidate roots or solver basins at most 1 millimetre apart are the same
  position;
- search-disk membership includes 1 millimetre of numerical tolerance;
- distinct converged basins within `5.99146454710798` objective units of the
  best fit are statistically comparable and produce `AMBIGUOUS`;
- full-rank condition number above `1,000` warns about weak geometry, while
  missing rank or condition number above `1,000,000` produces `DEGENERATE`;
- standardized RMS above `2` warns that residuals are large relative to the
  declared uncertainties;
- a fix beyond 90% of the search radius warns about the domain edge; and
- a 95% semi-major uncertainty axis above 25% of the search radius warns that
  uncertainty is large relative to the domain.

Smaller floating-point guards used to implement these classifications are
private regression details rather than measurement tolerances.

## Local uncertainty

For a unique, converged, full-rank fix, `FixUncertainty.covariance` is the
linearized `(JᵀJ)⁻¹` covariance in local east/north square metres. It is not
multiplied by fitted residual variance because observation uncertainties are
absolute standard deviations.

The result also reports:

- east and north standard deviations;
- their correlation;
- semi-major and semi-minor axes of the local 95% confidence ellipse; and
- the major-axis true bearing in `[0, 180)`.

The major-axis bearing is `None` for isotropic covariance. This is a local
linearized estimate, not a safety bound. It is withheld for ambiguous,
non-converged, rank-deficient, or numerically invalid geometry.

## Limitations

The fix engine assumes stationary two-dimensional positions, shortest WGS84
surface ranges, true initial bearings, independent errors, and no common
measurement bias.

It does not account for altitude, refraction, magnetic variation, platform
motion, time correlation, chart datum differences, or near-antipodal/global
networks. It is not certified navigation equipment.

Dependency and import boundaries are defined in
[ARCHITECTURE.md](ARCHITECTURE.md).
