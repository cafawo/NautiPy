# Bearing and range position fixes

The optional `nautipy.fix` module estimates a two-dimensional WGS84 position
from true bearings, surface ranges, or both. It reports convergence, residuals,
geometry, competing solutions, and local uncertainty instead of returning an
unqualified latitude/longitude pair.

Install the numerical dependencies explicitly:

```bash
python -m pip install "nautipy[fix]"
```

The normal installation does not include NumPy or SciPy. Importing `nautipy`,
using coordinates or navigation, importing `nautipy.fix`, and constructing
observation models do not load them. Calling a candidate or solver function
without the extra raises `FixDependencyError` and includes the installation
command above.

## Observation meaning and units

Each observation points to a known WGS84 `reference` position.

- A `BearingObservation` is the true initial bearing measured **at the unknown
  position toward the reference**, clockwise from true north in degrees.
- A `RangeObservation` is the shortest WGS84 surface distance between the
  unknown position and the reference, in metres.
- `uncertainty` is a required, finite, strictly positive one-standard-deviation
  value in the observation's natural unit.

This bearing direction matches the nautical workflow of taking bearings from
a vessel to known landmarks. It is not a bearing from a shore station toward
the vessel. Do not reverse a bearing by adding 180 degrees: reciprocal initial
geodesic bearings are not generally exact opposites on an ellipsoid.

```python
from nautipy import Position
from nautipy.fix import BearingObservation, RangeObservation

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

References accept the same documented position-like values as the coordinate
and navigation APIs and are stored as validated `Position` objects. Bearings
are normalized to `[0, 360)`. Negative ranges, zero or negative uncertainty,
booleans, non-numeric values, and non-finite values are rejected.

The uncertainties are treated as independent, absolute Gaussian standard
deviations. NautiPy requires them because a mixed metre/degree objective has no
sound implicit weighting. Correlated observations, shared bias estimation,
magnetic corrections, and robust-loss models are outside this first API.

## Solving a fix

```python
from nautipy import Position
from nautipy.fix import RangeObservation, solve_fix

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

`solve_fix` accepts `bearings=`, `ranges=`, or both. It consumes each iterable
once and reports residuals in the same deterministic order: supplied bearings
first, followed by supplied ranges. At least two scalar observations are
required. Two ranges commonly have two valid intersections, so additional
evidence is normally required for a unique fix.

When numerical multistart search is required, the optional `initial=` position
is guaranteed to be the first retained solver start, even when the generated
start set reaches its limit. It must lie inside the declared search disk and
does not enlarge that disk or authorize the solver to discard another
solution. Invalid observations and configuration raise `FixError`; numerical
and geometry outcomes are returned as a `FixResult`.

## Regional search domain

The first fix engine deliberately solves regional surface-navigation
problems, not global geodesic intersection problems. The declared solution
domain is a closed WGS84-distance disk, so its boundary is included:

- `search_center` defaults to a deterministic center of the references;
- `search_radius` defaults to 500,000 metres; and
- the supported maximum is 2,000,000 metres.

Pass both values when the default domain does not describe the operating area.
The result's uniqueness is only a claim about that declared domain and the
documented deterministic search. This bound avoids pretending that a pair of
geodesic bearings has one global intersection: extended geodesics can meet
again elsewhere on Earth.

Domain membership is determined by WGS84 surface distance, not by the local
chart's rectangular numerical bounds. A numerically exact fix on the circular
boundary is valid. A converged optimum outside the disk is never projected
onto the boundary; when no valid in-domain basin exists, the result is
`NO_SOLUTION`.

The optimizer uses east/north metre coordinates around a WGS84 anchor, with
exact GeographicLib inverse calculations for every predicted bearing and
range. It does not optimize raw latitude/longitude degrees. The local chart is
only a numerical parameterization; returned positions and residuals remain
ellipsoidal WGS84 quantities.

## Residuals and objective

For every observation, `ObservationResidual` contains the original
observation, its predicted value, the natural residual, and the standardized
residual.

- bearing residual: wrapped `predicted - observed` in `[-180, 180)` degrees;
- range residual: `predicted - observed` in metres; and
- standardized residual: natural residual divided by observation uncertainty.

A positive bearing residual is clockwise of the observation. A positive range
residual means the predicted range is too long. The combined `objective` is
the sum of squared standardized residuals, and `rms` is its dimensionless root
mean square. `bearing_rms` and `range_rms` separately report natural-unit RMS
values when that observation kind is present.

`iterations` counts solver linearizations performed through NautiPy's explicit
Jacobian callback. `function_evaluations` is reported separately; neither is a
raw SciPy result object.

## Status and geometry

`FixStatus.CONVERGED` is the only successful status. A successful result has a
unique position in the search domain, a converged optimizer, and a rank-two
weighted local Jacobian. Other statuses are explicit:

- `AMBIGUOUS`: two or more materially distinct competing positions fit;
- `DEGENERATE`: the observations do not constrain both local dimensions;
- `NO_SOLUTION`: no valid solution exists in the declared domain, including
  when the converged optimum is outside it; and
- `NON_CONVERGED`: the numerical search ended without a valid converged basin.

For every non-success status, `position` and `uncertainty` are `None`.
Ambiguity is exposed through deterministic `competing_positions`; NautiPy does
not select whichever candidate happened to have a microscopically lower
optimizer cost. Warnings distinguish a poor statistical fit, weak but
full-rank geometry, and a solution close to the search boundary.

`two_bearing_candidates` and `two_range_candidates` expose the corresponding
two-observation geometry directly through `CandidateResult`. Candidate status
distinguishes one solution, multiple solutions, no solution, and degenerate
geometry. Candidate positions and cardinality depend only on the measurements
and reference geometry; changing otherwise valid observation uncertainties
does not merge, create, or remove candidates. Uncertainty still controls
weighting in `solve_fix`.

`two_range_candidates` supports observed ranges no greater than 2,000,000
metres. A larger range is outside this regional algorithm's configured scope
and raises `FixError`; it is not misclassified as a mathematical
`NO_SOLUTION`. This helper limit does not impose an observation-distance cap
on `solve_fix`; the solver's declared regional bound is its search disk.

Tangent range circles therefore produce one mathematical candidate with
`CandidateStatus.UNIQUE` and a rank-deficiency warning. The same two
observations passed to `solve_fix` produce `FixStatus.DEGENERATE`, because a
successful fix requires two stable local position axes. Nearly parallel
bearing geometry likewise carries a warning even when it has one candidate.

### Deterministic classification thresholds

The current numerical classifications use these explicit thresholds:

- candidate roots and solver basins at most 1 millimetre apart are treated as
  the same position;
- distinct converged basins whose objective is no more than
  `5.99146454710798`
  above the best objective are statistically comparable and produce
  `AMBIGUOUS`;
- a rank-two Jacobian condition number above `1,000` warns about weak geometry,
  while a missing second rank or a condition number above `1,000,000` produces
  `DEGENERATE`;
- a two-bearing candidate also warns about weak crossing geometry when the
  absolute sine of its ray-crossing angle is below `0.001`;
- dimensionless standardized RMS above `2` warns that residuals are large
  relative to the stated uncertainties;
- a fix beyond 90% of the search radius warns that it is near the domain edge;
  and
- a 95% semi-major uncertainty axis above 25% of the search radius warns that
  uncertainty is large relative to the domain.

## Local uncertainty

For a unique, converged, full-rank fix, `FixUncertainty.covariance` is the
linearized `(JᵀJ)⁻¹` covariance in local east/north square metres. It is
not multiplied by the fitted residual variance because observation
uncertainties are declared as absolute standard deviations.

The result also reports east and north standard deviations, their correlation,
and the semi-major and semi-minor axes of the local 95% confidence ellipse.
The major-axis bearing is clockwise from true north in `[0, 180)` and is `None`
for effectively isotropic uncertainty. This is a local linearized estimate,
not a safety bound. It is withheld for ambiguous, non-converged, rank-deficient,
or numerically invalid geometry.

## Limitations

The fix engine assumes stationary two-dimensional positions, shortest WGS84
surface ranges, true initial bearings, independent errors, and no common
measurement bias. It does not account for altitude, refraction, magnetic
variation, platform motion, time correlation, chart datum differences, or
near-antipodal/global networks. It is not certified navigation equipment.
