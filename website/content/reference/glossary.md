# Glossary

This glossary uses NautiPy’s conventions. Follow the links for longer
explanations and scientific context.

## Accuracy

How closely a measurement agrees with the value of interest. Accuracy is not
created by printing more digits. Compare [precision](#precision).

## Ambiguity

A situation in which materially different positions satisfy the available
syntax or observations. NautiPy reports ambiguity instead of silently choosing
one interpretation. See [coordinate order](../learn/coordinates.md#coordinate-order-changes-the-place)
and [two-range geometry](../learn/finding-the-boat.md#ranges-circles-around-references).

## Antimeridian

The longitude-numbering seam at 180° east/west. A short geodesic may cross it
normally. [Wikipedia overview](https://en.wikipedia.org/wiki/180th_meridian).

## Azimuth / bearing

An angular direction. In NautiPy, bearings are true degrees clockwise from
north and generated values are normalized to `[0, 360)`.
[Azimuth background](https://en.wikipedia.org/wiki/Azimuth).

## Bearing observation

A true initial bearing measured **at the unknown position toward a known
reference**, in degrees. Its required uncertainty is one standard deviation
in degrees.

## Candidate

A mathematical position satisfying an exact two-bearing or two-range
construction. A candidate may still have weak or rank-deficient geometry and
is not automatically a successful general fix.

## Condition number

A local measure of how unevenly the observations constrain different
directions. Large values indicate sensitivity and weak geometry.
[Background](https://en.wikipedia.org/wiki/Condition_number).

## Confidence ellipse

NautiPy’s local, linearized 95% position-uncertainty region, summarized by
semi-major and semi-minor axes and a major-axis true bearing. It is not a
safety boundary. See [the uncertainty guide](../learn/trusting-a-fix.md#the-local-95-uncertainty-ellipse).

## Coordinate order

Whether a pair is interpreted as latitude/longitude or longitude/latitude.
NautiPy defaults ordinary unmarked input to `latlon`; GeoJSON Points use their
specified `lonlat` order.

## Covariance

A matrix describing local east/north variances and their co-variation.
NautiPy reports it in square metres for a suitable converged fix.
[Background](https://en.wikipedia.org/wiki/Covariance_matrix).

## DD

Decimal degrees, such as `50.122570, 8.665700`.

## DDM

Degrees and decimal minutes, such as
`50° 7.3542′ N; 8° 39.9420′ E`. `dmm` is accepted as an alias.

## Degrees of freedom

For a fix with *n* scalar observations and two fitted local coordinates,
NautiPy reports *n* − 2. Reduced chi-square is available only when this is
positive.

## DMS

Degrees, minutes, and seconds, such as
`50° 7′ 21.25″ N; 8° 39′ 56.52″ E`.

## Ellipsoid

A mathematically smooth, slightly flattened surface used to model Earth.
NautiPy navigation uses the WGS84 reference ellipsoid rather than a sphere.

## Final bearing

The forward azimuth on arrival while continuing along the geodesic. It is not
the reciprocal direction from end back to start.

## Fix

An estimated unknown position from known references and observed bearings,
ranges, or both. NautiPy couples a fix with status and diagnostics rather than
returning only coordinates.

## GeoJSON

A JSON-based geospatial interchange format defined by
[RFC 7946](https://www.rfc-editor.org/info/rfc7946). Its coordinate arrays use
longitude, latitude order. NautiPy supports two-dimensional Points and Point
FeatureCollections.

## Geodesic

A locally straight surface path. NautiPy uses shortest WGS84 ellipsoidal
geodesics for navigation and observation predictions.
[Background](https://en.wikipedia.org/wiki/Geodesics_on_an_ellipsoid).

## Initial bearing

The forward azimuth at the start of a geodesic.

## ISO 6709

An international standard for representing geographic point coordinates.
NautiPy accepts and emits an unambiguous signed two-dimensional subset.
[Standard page](https://www.iso.org/standard/75147.html).

## Latitude

Angular position north or south of the equator. NautiPy accepts values in
`[-90, 90]` degrees.

## Longitude

Angular position east or west around Earth. NautiPy accepts values in
`[-180, 180]` degrees and never wraps invalid parser input.

## NMEA coordinate fields

Fixed-width latitude and longitude fields accompanied by required direction
fields. NautiPy reads coordinate-field pairs, not complete NMEA sentences.
[NMEA 0183 standard page](https://www.nmea.org/nmea-0183.html).

## Objective

The sum of squared standardized residuals in a fitted position fix.

## Position

NautiPy’s immutable validated value containing finite latitude and longitude
in decimal degrees, plus optional identifier and description metadata.

## Precision

The fineness of a representation or repeatability of measurements. NautiPy
can infer lexical angular resolution, but it does not treat displayed digits
as measurement accuracy.

## Range observation

A shortest WGS84 surface distance between the unknown position and a known
reference, in metres. Its required uncertainty is one standard deviation in
metres.

## Rank

The number of independent local dimensions constrained by the observation
Jacobian. A successful two-dimensional fix requires rank 2.

## Residual

Prediction minus observation. Bearings use a wrapped degree difference;
ranges use metres.

## RMS

Root mean square. `FixResult.rms` summarizes standardized residuals and is
dimensionless; bearing and range RMS fields retain their natural units.

## Standard deviation

A scale for uncertainty or spread. Fix observations require a positive
one-standard-deviation uncertainty in degrees or metres.
[Background](https://en.wikipedia.org/wiki/Standard_deviation).

## Standardized residual

A natural residual divided by its observation uncertainty. Standardization
lets degree and metre observations contribute on a meaningful common scale.

## True north

Direction along the local meridian toward the geographic North Pole, as
distinct from magnetic north. NautiPy does not apply magnetic variation.

## WGS84

The World Geodetic System 1984 reference system used by NautiPy navigation,
GeoJSON point interchange, and position fixing.
[NGA reference material](https://earth-info.nga.mil/?action=wgs84&dir=wgs84).

## Exact NautiPy contracts

- [Coordinate input and conversion](https://github.com/cafawo/NautiPy/blob/master/docs/COORDINATES.md)
- [WGS84 navigation](https://github.com/cafawo/NautiPy/blob/master/docs/NAVIGATION.md)
- [GeoJSON interchange](https://github.com/cafawo/NautiPy/blob/master/docs/GEOJSON.md)
- [Bearing and range position fixes](https://github.com/cafawo/NautiPy/blob/master/docs/FIXES.md)
- [Product direction](https://github.com/cafawo/NautiPy/blob/master/docs/PRODUCT.md)
- [Architecture and dependency policy](https://github.com/cafawo/NautiPy/blob/master/docs/ARCHITECTURE.md)
