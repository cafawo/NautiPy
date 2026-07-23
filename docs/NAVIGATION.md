# WGS84 navigation specification

## Purpose

NautiPy provides a small set of offline navigation primitives on the WGS84
ellipsoid. Distances are metres. Bearings are true degrees clockwise from
north. The package does not use a spherical approximation for these functions.

The public API is:

```python
from nautipy import (
    InverseResult,
    destination,
    distance,
    initial_bearing,
    interpolate,
    inverse,
    nearest_position,
)
```

Every location argument accepts a `Position` or any string, two-value
sequence, named mapping, or GeoJSON Point accepted by `parse_position`.
Navigation functions use the coordinate parser's default latitude/longitude
order. Parse longitude-first or otherwise specialized input explicitly before
passing it to navigation functions.

## Inverse calculations

`inverse(start, end)` returns an immutable `InverseResult`:

```python
result.distance         # metres
result.initial_bearing  # true degrees in [0, 360), or None
result.final_bearing    # true degrees in [0, 360), or None
```

The final bearing is the forward azimuth on arrival, continuing along the
geodesic. It is not the reciprocal bearing from the endpoint back to the
start. `distance(start, end)` and `initial_bearing(start, end)` are concise
wrappers around the same calculation.

Coincident physical positions have distance `0.0` and no unique direction.
Their inverse bearings are therefore `None`, and `initial_bearing` raises
`NavigationError`. Where distinct endpoints admit multiple equally short
geodesics, NautiPy returns GeographicLib's deterministic canonical solution.
Distinct inputs below the backend's numerical distance resolution raise
`NavigationError` rather than being reported as coincident.

## Direct calculations

`destination(start, bearing=..., distance=...)` returns the endpoint of a
WGS84 geodesic. Distance must be a finite, non-negative real number. Bearing
may be any finite real number and is normalized modulo 360 degrees. A zero
distance returns the validated start unchanged. Distances must be representable
as binary64 values, and positive distances below `1e-7` metres are rejected
because they cannot produce a dependable output coordinate. Exact integer,
`Decimal`, and `Fraction` bearings are reduced before float conversion, so even
very large bearings retain their correct direction. Residual angles smaller
than binary64 can represent normalize to zero degrees.

Generated longitudes use the normalized `[-180, 180]` range. NautiPy does not
wrap out-of-range user coordinates during parsing.

## Interpolation

`interpolate(start, end, fraction=0.5)` follows the selected shortest WGS84
geodesic. Fraction must be finite and in `[0, 1]`. Zero and one return the
validated endpoints exactly; the default returns the geodesic midpoint.
Interpolation does not extrapolate beyond the segment. Non-boundary fractions
that collapse to zero or one in binary64 are rejected. Interior results must
remain at least `1e-7` metres from both endpoints so their coordinates are
numerically resolved.

## Nearest position

`nearest_position(origin, candidates)` performs a single-pass linear search
over an ordinary iterable and returns the parsed `Position` with the shortest
WGS84 distance. An empty iterable raises `NavigationError`. Exact-distance ties
select the first candidate, and invalid candidates are not silently skipped.

## Dependency decision

Navigation uses
[GeographicLib 2.1 or newer](https://pypi.org/project/geographiclib/) as the
normal installation's sole runtime dependency. GeographicLib is a maintained,
MIT-licensed, pure-Python implementation of robust ellipsoidal direct, inverse,
and geodesic-line algorithms. Version 2.1 declares Python 3.7 and newer and
ships a platform-independent wheel with no transitive dependencies.

The backend is imported only when a navigation calculation is requested.
Importing `nautipy`, parsing coordinates, and formatting coordinates do not
load GeographicLib. Third-party dictionaries and line objects never appear in
the public API.

Reference tests use GeographicLib's independently generated WGS84
[geodesic test data](https://geographiclib.sourceforge.io/C++/doc/geodesic.html#testgeod),
including short, near-antipodal, high-latitude, and antimeridian cases.

## Reference tolerances

For the frozen high-precision GeodTest cases, inverse distances must agree
within 1 micrometre and endpoint bearings within `5e-10` degrees. Replaying an
inverse result through `destination` must recover each endpoint within
`1e-12` degrees per axis. Published examples rounded to five decimal places,
such as the dateline waypoint, use the corresponding half-unit tolerance of
`5e-6` degrees.

These are regression thresholds for the current reference corpus, not
uncertainty estimates or guarantees that user inputs are accurate to those
scales. Display rounding and input precision remain separate concerns.

NautiPy is a calculation library, not certified navigation equipment.
