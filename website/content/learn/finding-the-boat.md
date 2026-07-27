# Finding the Boat

A **position fix** estimates an unknown position from observations tied to
known reference positions. NautiPy works with:

- a true bearing measured at the unknown boat toward a reference; and
- a shortest WGS84 surface range between the boat and a reference.

The observations can be all bearings, all ranges, or a mixture. Their geometry
determines whether the boat is uniquely located.

## Bearings: directions from the boat

Imagine looking from the boat toward a known lighthouse and measuring its true
bearing. A second landmark supplies another directional constraint. Their
intersection is a form of
[resection](https://en.wikipedia.org/wiki/Resection_(orientation)).

<div class="educational-figure" tabindex="0" role="group" aria-label="Scrollable diagram" markdown>

![A boat taking true bearings toward known references, including examples of
strong crossing geometry and weak nearly parallel
geometry.](../assets/images/bearing-geometry.svg)

</div>

The direction convention is important:

```text
unknown boat ── measured bearing ──→ known reference
```

It is **not** the bearing from the reference toward the boat. On an ellipsoid,
you should not reverse an initial bearing by blindly adding 180°.

```python
from nautipy import BearingObservation, Position

lighthouse = Position(50.116135, 8.670277)
observation = BearingObservation(
    lighthouse,
    bearing=164.71,     # true degrees at the boat toward the lighthouse
    uncertainty=0.05,  # one standard deviation in degrees
)
```

When bearing lines cross at a healthy angle, small angular errors tend to move
their intersection modestly. Nearly parallel directions can move it a long
way: that is weak geometry.

## Ranges: circles around references

A measured range says that the boat lies a given surface distance from a known
reference. In a local schematic this looks like a circle.

<div class="educational-figure" tabindex="0" role="group" aria-label="Scrollable diagram" markdown>

![Two range constraints around known references, showing two intersections, a
tangent single candidate, and separated no-solution
geometry.](../assets/images/range-geometry.svg)

</div>

Two range circles may:

- cross twice, leaving two possible positions;
- touch once, producing a tangent mathematical candidate;
- remain apart or one inside the other, producing no candidate; or
- coincide, failing to isolate any position.

This idea is often called
[trilateration](https://en.wikipedia.org/wiki/Trilateration). A third
independent observation usually selects one of the two intersections.

You can expose the exact two-observation geometry:

```python
from nautipy import (
    Position,
    RangeObservation,
    distance,
    two_range_candidates,
)

boat_for_example = Position(50.12257, 8.66570)
first_reference = Position(50.116135, 8.670277)
second_reference = Position(50.112836, 8.666753)

first = RangeObservation(
    first_reference,
    distance(boat_for_example, first_reference),
    uncertainty=2.0,
)
second = RangeObservation(
    second_reference,
    distance(boat_for_example, second_reference),
    uncertainty=2.0,
)

candidates = two_range_candidates(first, second)
print(candidates.status)
print(candidates.positions)
```

The diagram is flat and schematic. NautiPy calculates candidate positions with
WGS84 surface distances.

## Why uncertainty is required

A bearing residual is measured in degrees; a range residual is measured in
metres. Adding their raw squares would give arbitrary weight to the choice of
units.

Each observation therefore requires a finite, positive one-standard-deviation
uncertainty in its natural unit. NautiPy divides each residual by that
uncertainty before fitting. A 2 m range miss and a 0.2° bearing miss both count
as one standardized unit if their declared uncertainties are 2 m and 0.2°.

The model treats uncertainties as independent, absolute Gaussian standard
deviations. It does not estimate shared biases or correlations.

## Solving a mixed fix

This example creates self-consistent observations from a teaching position,
then asks NautiPy to recover a fix:

```python
from nautipy import (
    BearingObservation,
    Position,
    RangeObservation,
    distance,
    initial_bearing,
    solve_fix,
)

teaching_position = Position(50.12257, 8.66570)
references = (
    Position(50.116135, 8.670277),
    Position(50.112836, 8.666753),
    Position(50.110347, 8.659873),
)

bearings = tuple(
    BearingObservation(
        reference,
        initial_bearing(teaching_position, reference),
        uncertainty=0.2,
    )
    for reference in references[:2]
)
ranges = (
    RangeObservation(
        references[2],
        distance(teaching_position, references[2]),
        uncertainty=3.0,
    ),
)

result = solve_fix(bearings=bearings, ranges=ranges)
print(result.status)
print(result.position)
print(result.residuals)
print(result.warnings)
```

`solve_fix` searches a regional disk. By default, its center is derived
deterministically from the references and its radius is 500 km. You can supply
`search_center` and `search_radius` when your problem has a justified domain.
The domain is part of the result’s meaning: uniqueness is claimed only inside
that disk and within NautiPy’s deterministic multistart search.

## A candidate is not yet a trustworthy fix

Exact two-observation helpers classify mathematical candidates as unique,
ambiguous, absent, or degenerate. The general solver goes further:

- it will not select one of several comparable positions;
- it requires the local geometry to constrain two dimensions stably;
- it reports residuals in natural and standardized units; and
- it attaches local uncertainty only to a unique, converged, full-rank fix.

A tangent range intersection, for example, is one mathematical candidate but
does not constrain two local axes stably. It can be `UNIQUE` as candidate
geometry and `DEGENERATE` as a solved fix.

[Learn how to judge the complete result →](trusting-a-fix.md)

## Boundaries and limitations

- Bearings are true initial bearings at the boat; NautiPy does not apply
  magnetic variation.
- Ranges are shortest WGS84 surface distances, not slant ranges through
  three-dimensional space.
- The model assumes a stationary two-dimensional problem and independent
  errors.
- It does not account for refraction, current, platform motion, time
  correlation, or common sensor bias.
- The regional fixer is not intended for global or near-antipodal networks.

For the exact model, thresholds, and failure states, see the
[position-fix behavior specification](https://github.com/cafawo/NautiPy/blob/master/docs/FIXES.md).

> **Navigation safety**
>
> A plausible coordinate can still come from ambiguous, weak, biased, or
> inconsistent observations. Inspect the complete result and use appropriate
> independent navigation safeguards.

## Learn more

- [Position fixing](https://en.wikipedia.org/wiki/Position_fixing) surveys the
  general navigation idea.
- [Resection](https://en.wikipedia.org/wiki/Resection_(orientation)) and
  [trilateration](https://en.wikipedia.org/wiki/Trilateration) introduce
  bearing- and range-based geometry.
- The NGA’s
  [*American Practical Navigator*](https://msi.nga.mil/Publications/APN)
  is a primary practical reference for navigation concepts.
- Karney’s
  [*Algorithms for geodesics*](https://doi.org/10.1007/s00190-012-0578-z)
  describes the WGS84 geodesic calculations behind predicted observations.

Next: [Can You Trust the Fix?](trusting-a-fix.md).
