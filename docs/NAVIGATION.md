# WGS84 navigation

## Overview

NautiPy provides a small set of offline navigation primitives on the WGS84
ellipsoid. Distances are metres. Bearings are true degrees clockwise from
north. These functions use GeographicLib rather than a spherical
approximation.

```text
inverse(start, end) -> InverseResult
distance(start, end) -> float
initial_bearing(start, end) -> float
destination(start, *, bearing, distance) -> Position
interpolate(start, end, *, fraction=0.5) -> Position
nearest_position(origin, candidates) -> Position
```

Every location argument accepts a `Position` or a position-like string,
two-value sequence, named mapping, or GeoJSON Point accepted by
`parse_position`. Navigation functions use the parser's default
latitude/longitude order. Parse longitude-first or otherwise specialized input
explicitly before passing it to navigation functions.

Invalid location inputs retain the applicable `CoordinateError` subtype.
Invalid navigation scalars and undefined or unresolved calculations raise
`NavigationError`.

## Example

```python
from nautipy import destination, distance, inverse

start = "50.12257, 8.66570"
end = destination(start, bearing=90, distance=12_000)

assert abs(distance(start, end) - 12_000) < 1e-6

result = inverse(start, end)
print(result.initial_bearing)
print(result.final_bearing)
```

## Inverse calculations

`inverse(start, end)` returns an immutable `InverseResult`:

```python
result.distance         # metres
result.initial_bearing  # degrees in [0, 360), or None
result.final_bearing    # degrees in [0, 360), or None
```

The final bearing is the forward azimuth on arrival while continuing along the
geodesic. It is not the reciprocal bearing from the endpoint back to the
start. `distance` and `initial_bearing` are concise wrappers around the same
inverse calculation.

Coincident physical positions have distance `0.0` and no unique direction.
Their inverse bearings are `None`, and `initial_bearing` raises
`NavigationError`. Distinct inputs too close for the WGS84 calculation to
resolve also raise `NavigationError` rather than being reported as coincident.

Where distinct endpoints admit more than one equally short geodesic, NautiPy
returns GeographicLib's deterministic canonical solution.

## Destination

`destination(start, bearing=..., distance=...)` returns the endpoint of a WGS84
geodesic.

- Distance is a finite, non-negative real number in metres.
- Bearing is any finite real number and is normalized modulo 360 degrees.
- Zero distance returns the validated start unchanged.
- Positive distances below `1e-7` metres are rejected because they cannot
  produce a dependable output coordinate.

Exact integer, `Decimal`, and `Fraction` bearings are reduced before float
conversion, so very large values retain their direction where representable.
Generated longitudes use `[-180, 180]`; parser input is never silently wrapped.

## Interpolation

`interpolate(start, end, fraction=0.5)` follows the selected shortest WGS84
geodesic.

- Fraction is finite and in `[0, 1]`.
- Zero and one return the validated endpoints exactly.
- The default returns the geodesic midpoint.
- Interpolation never extrapolates beyond the segment.
- An interior result must remain at least `1e-7` metres from each endpoint.

A non-boundary fraction that collapses numerically to an endpoint raises
`NavigationError`.

## Nearest position

`nearest_position(origin, candidates)` performs a single-pass linear search
over an ordinary iterable. It returns the parsed candidate with the shortest
WGS84 distance.

An empty iterable raises `NavigationError`. Exact-distance ties select the
first candidate. Invalid candidates are not silently skipped.

## Numerical meaning

Navigation results are deterministic calculations from the supplied
coordinates. Numerical regression tolerances are not measurement uncertainty
and do not imply that source coordinates are accurate to the same scale.

GeographicLib is imported only when a navigation calculation is requested.
Navigation does not load NumPy or SciPy. Dependency and import policy is
defined in [ARCHITECTURE.md](ARCHITECTURE.md).

NautiPy is a calculation library, not certified navigation equipment.
