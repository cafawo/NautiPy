# Learn positions and navigation with NautiPy

NautiPy turns coordinate text into validated positions, performs common
[WGS 84](https://en.wikipedia.org/wiki/World_Geodetic_System) navigation
calculations, and estimates a position from bearings and ranges. This site
explains the ideas as well as the Python.

There are two connected journeys:

1. **Coordinates to navigation:** read a position safely, then calculate a
   distance, bearing, destination, or point along a route.
2. **Observations to a fix:** combine bearings and ranges to known references,
   then examine whether the resulting position is unique and trustworthy.

<div class="educational-figure" tabindex="0" role="group" aria-label="Scrollable diagram" markdown>

![Two NautiPy workflows: coordinate input becomes a Position for navigation or
interchange, while observations become a diagnosed position
fix.](assets/images/package-flow.svg)

</div>

## Install the complete package

```console
python -m pip install nautipy
```

One installation provides coordinate handling, navigation, GeoJSON, the
command line, and position fixing. Ordinary calculations are offline.

## Your first position

People write the same location in many ways. NautiPy recognizes decimal
degrees (DD), degrees and decimal minutes (DDM), degrees/minutes/seconds (DMS),
a two-dimensional subset of ISO 6709, and NMEA coordinate fields.

```python
from nautipy import convert_position, parse_position

position = parse_position("N 50° 7' 21.252\"; E 8° 39' 56.52\"")

print(position.latitude, position.longitude)
print(convert_position(position, to="ddm"))
```

The result is an immutable `Position` in decimal degrees. NautiPy normalizes
harmless presentation differences, but it does not guess between two
different places.

[Learn how coordinates work →](learn/coordinates.md)

## Your first WGS84 calculation

An initial bearing of 90° points due east at the start. A destination 12 km
away is:

```python
from nautipy import destination, inverse

start = "50.12257, 8.66570"
end = destination(start, bearing=90, distance=12_000)
journey = inverse(start, end)

print(end)
print(journey.distance)          # metres
print(journey.initial_bearing)   # true degrees
print(journey.final_bearing)     # true forward bearing on arrival
```

The path is a shortest geodesic on the WGS84 ellipsoid, not a straight line on
a flat map or a great circle on a perfect sphere.

[Explore ellipsoidal navigation →](learn/navigation.md)

## Your first position fix

Suppose three known stations report surface ranges to a boat. Each observation
also carries a one-standard-deviation uncertainty, because metre and degree
errors need meaningful weights.

```python
from nautipy import Position, RangeObservation, solve_fix

references = (
    Position(50.116135, 8.670277),
    Position(50.112836, 8.666753),
    Position(50.110347, 8.659873),
)
ranges = tuple(
    RangeObservation(reference, measured, uncertainty=2.0)
    for reference, measured in zip(
        references,
        (1_275.251, 1_599.237, 1_917.145),
    )
)

result = solve_fix(ranges=ranges)
if result.success:
    print(result.position)
    print(result.warnings)
else:
    print(result.status, result.message)
    print(result.competing_positions)
```

The complete `FixResult` matters. It reports convergence, residuals, geometry,
ambiguity, and a local uncertainty estimate where meaningful.

[See how a boat is found →](learn/finding-the-boat.md)

## Choose a path

- [Coordinates on Earth](learn/coordinates.md) explains notation, order, and
  why printed precision is not accuracy.
- [Navigation on an Ellipsoid](learn/navigation.md) introduces inverse and
  direct geodesic problems.
- [Finding the Boat](learn/finding-the-boat.md) builds bearing and range fixes
  from geometry.
- [Can You Trust the Fix?](learn/trusting-a-fix.md) teaches residuals,
  conditioning, statuses, and uncertainty.
- [Fix Lab](learn/fix-lab.md) lets you compare precomputed observation
  scenarios.
- [Practical Use](practical-use.md) collects small recipes for Python, GeoJSON,
  and the command line.
- [How NautiPy Works](how-nautipy-works.md) follows both workflows through the
  package.
- [Glossary](reference/glossary.md) and
  [Further Reading](reference/further-reading.md) connect the concepts to
  approachable introductions and primary sources.

> **Navigation safety**
>
> NautiPy is an educational calculation library, not certified navigation
> equipment. Its results are only as good as the supplied coordinates,
> observations, uncertainty assumptions, datum, and model. Do not use it as
> the sole source for safety-critical navigation.
