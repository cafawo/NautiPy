# Navigation on an Ellipsoid

On a flat sheet, the shortest route between two points is a straight line.
Earth is curved and slightly flattened at the poles, so dependable global
navigation needs a different model.

[WGS 84](https://en.wikipedia.org/wiki/World_Geodetic_System) represents the
reference surface as an oblate ellipsoid. A
[geodesic](https://en.wikipedia.org/wiki/Geodesic) is the locally straight
path on that surface. NautiPy uses GeographicLib to solve WGS84 geodesic
problems rather than substituting a sphere.

<div class="educational-figure" tabindex="0" role="group" aria-label="Scrollable diagram" markdown>

![A shortest geodesic joining two positions on an ellipsoid, with different
initial and final forward bearings.](../assets/images/ellipsoid-geodesic.svg)

</div>

## Two complementary questions

Geodesic navigation is commonly divided into two problems:

- **Inverse:** given the start and end, find the shortest surface distance and
  endpoint bearings.
- **Direct:** given a start, initial bearing, and distance, find the
  destination.

NautiPy uses metres for distance and true degrees clockwise from north for
bearings. Zero degrees is north, 90° east, 180° south, and 270° west.

## The inverse problem

```python
from nautipy import Position, inverse

start = Position(50.12257, 8.66570)
end = Position(53.55110, 9.99370)

result = inverse(start, end)

print(result.distance)
print(result.initial_bearing)
print(result.final_bearing)
```

The initial bearing is the forward direction at the start. The final bearing
is the forward direction on arrival **while continuing along the same
geodesic**. It is not the bearing from the endpoint back to the start.

Why can the two forward bearings differ? Lines of longitude converge toward
the poles, so the geodesic’s direction relative to local north changes along
most routes.

For concise calculations, `distance(start, end)` and
`initial_bearing(start, end)` wrap the same inverse solution.

## The direct problem

```python
from nautipy import destination, distance

start = "50.12257, 8.66570"
end = destination(start, bearing=90, distance=12_000)

assert abs(distance(start, end) - 12_000) < 1e-6
```

The bearing may be any finite real number; NautiPy normalizes it modulo 360°.
Distance must be finite and non-negative. A zero-distance destination is the
validated starting position.

## Positions along a route

`interpolate` follows the selected shortest WGS84 geodesic. Its fraction is a
share of geodesic distance, not a linear average of latitude and longitude.

```python
from nautipy import interpolate

quarter = interpolate(start, end, fraction=0.25)
midpoint = interpolate(start, end)  # fraction=0.5
```

Fractions stay in `[0, 1]`; the function does not extrapolate. The boundary
values return the validated endpoints exactly.

To choose the closest member of an ordinary iterable:

```python
from nautipy import nearest_position

closest = nearest_position(
    start,
    [
        "50.20, 8.70",
        "50.13, 8.67",
        "50.30, 8.90",
    ],
)
```

Exact-distance ties keep the first candidate, and invalid candidates are not
silently skipped.

## Difficult geography

### Antimeridian

The [antimeridian](https://en.wikipedia.org/wiki/180th_meridian) is where
longitude changes between +180° and −180°. A short geodesic can cross that
numbering seam without making a nearly complete trip around Earth. NautiPy’s
generated longitudes remain in `[-180, 180]`.

### Poles

Longitude lines meet at a pole. Bearings are local directions, so polar and
near-polar geometry deserves care even though the ellipsoidal solver handles
the WGS84 surface.

### Coincident positions

The distance from a position to itself is zero, but there is no unique
direction to travel. `inverse` therefore reports both bearings as `None`, and
`initial_bearing` raises `NavigationError`.

### Multiple shortest geodesics

Some widely separated endpoints admit more than one equally short geodesic.
NautiPy returns GeographicLib’s deterministic canonical solution. Near
antipodal geometry is also outside the intended regional use of the position
fixer.

## What the numbers mean

The output is a deterministic calculation from the coordinates you supplied.
A sub-millimetre numerical agreement does not imply that the source
coordinates are known that accurately. WGS84 is a reference ellipsoid; it does
not model tides, currents, terrain, magnetic variation, or the vessel’s
motion.

For exact boundaries and errors, see the
[navigation behavior specification](https://github.com/cafawo/NautiPy/blob/master/docs/NAVIGATION.md).

> **Navigation safety**
>
> These calculations are not certified navigation equipment and should not be
> the only basis for a safety-critical decision.

## Learn more

- [Geodesics on an ellipsoid](https://en.wikipedia.org/wiki/Geodesics_on_an_ellipsoid)
  gives an approachable mathematical overview.
- [Azimuth](https://en.wikipedia.org/wiki/Azimuth) explains clockwise angular
  direction from a reference north.
- The US National Geospatial-Intelligence Agency maintains the
  [WGS 84 reference material](https://earth-info.nga.mil/?action=wgs84&dir=wgs84).
- Charles F. F. Karney’s paper,
  [*Algorithms for geodesics*](https://doi.org/10.1007/s00190-012-0578-z),
  describes the numerical algorithms used by GeographicLib.
- The open [GeographicLib geodesic documentation](https://geographiclib.sourceforge.io/html/python/code.html)
  documents the underlying implementation interface.

Next: [Finding the Boat](finding-the-boat.md).
