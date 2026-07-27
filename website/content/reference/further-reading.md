# Further Reading

The first group offers friendly introductions. The second group contains
standards, scientific work, and implementation references behind NautiPy’s
documented choices. Links are provided instead of bundling copied figures or
documents.

## Friendly introductions

### Coordinates and Earth

- [Geographic coordinate system](https://en.wikipedia.org/wiki/Geographic_coordinate_system)
  introduces latitude, longitude, and reference surfaces.
- [Decimal degrees](https://en.wikipedia.org/wiki/Decimal_degrees) compares
  common degree notations.
- [Sexagesimal](https://en.wikipedia.org/wiki/Sexagesimal) explains the
  base-60 origin of angular minutes and seconds.
- [World Geodetic System](https://en.wikipedia.org/wiki/World_Geodetic_System)
  introduces WGS84.
- [GeoJSON](https://en.wikipedia.org/wiki/GeoJSON) introduces Point, Feature,
  and FeatureCollection objects.

### Navigation geometry

- [Geodesics on an ellipsoid](https://en.wikipedia.org/wiki/Geodesics_on_an_ellipsoid)
  develops shortest paths on an ellipsoidal Earth model.
- [Azimuth](https://en.wikipedia.org/wiki/Azimuth) introduces angular
  direction from north.
- [Position fixing](https://en.wikipedia.org/wiki/Position_fixing) surveys
  how observations locate a vessel.
- [Resection](https://en.wikipedia.org/wiki/Resection_(orientation)) covers
  directional constraints to known references.
- [Trilateration](https://en.wikipedia.org/wiki/Trilateration) covers
  distance-intersection geometry.

### Fitting and uncertainty

- [Least squares](https://en.wikipedia.org/wiki/Least_squares) introduces
  residual minimization.
- [Condition number](https://en.wikipedia.org/wiki/Condition_number) explains
  sensitivity to input error.
- [Covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix)
  explains multidimensional variation.
- [Confidence region](https://en.wikipedia.org/wiki/Confidence_region)
  introduces regions such as a confidence ellipse.
- [Reduced chi-squared statistic](https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic)
  provides context for comparing weighted mismatch and degrees of freedom.

Wikipedia is useful orientation, not NautiPy’s behavioral authority. The
project specifications and primary sources below settle the package’s exact
conventions.

## Standards and primary sources

### Coordinate representation and interchange

- **ISO 6709:2022 — Standard representation of geographic point location by
  coordinates.** The
  [ISO catalogue page](https://www.iso.org/standard/75147.html) identifies the
  current standard. NautiPy implements a deliberately smaller, unambiguous
  signed two-dimensional subset; consult the
  [coordinate specification](https://github.com/cafawo/NautiPy/blob/master/docs/COORDINATES.md)
  for that boundary.
- **NMEA 0183.** The
  [official NMEA standard page](https://www.nmea.org/nmea-0183.html) describes
  the marine-electronics data standard. NautiPy supports coordinate and
  direction field pairs, not complete sentences or streams.
- **RFC 7946 — The GeoJSON Format.**
  [The RFC Editor copy](https://www.rfc-editor.org/info/rfc7946) defines
  GeoJSON’s WGS84 decimal-degree position and longitude/latitude array order.
  NautiPy intentionally limits interchange to two-dimensional Points and
  Point FeatureCollections.

### WGS84 and geodesics

- **National Geospatial-Intelligence Agency WGS 84 resources.** The
  [official NGA WGS 84 page](https://earth-info.nga.mil/?action=wgs84&dir=wgs84)
  collects reference-system definitions and supporting material.
- **Karney, C. F. F. (2013), “Algorithms for geodesics.”**
  [DOI 10.1007/s00190-012-0578-z](https://doi.org/10.1007/s00190-012-0578-z).
  This paper develops accurate direct and inverse algorithms for geodesics on
  an ellipsoid.
- **GeographicLib.** The
  [Python geodesic documentation](https://geographiclib.sourceforge.io/html/python/code.html)
  documents the WGS84 implementation used by NautiPy. NautiPy returns its own
  small result models rather than exposing backend dictionaries.
- **Bowditch, *The American Practical Navigator*.** The NGA provides the
  [official publication page](https://msi.nga.mil/Publications/APN), a broad
  primary reference for practical navigation concepts.

### Least squares and measurement uncertainty

- **SciPy `least_squares`.** The
  [SciPy 1.14.1 reference](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.optimize.least_squares.html)
  documents the bounded nonlinear least-squares optimizer underlying
  NautiPy’s private fix solver.
- **NIST measurement uncertainty.** NIST’s
  [Uncertainty of Measurement portal](https://physics.nist.gov/cuu/Uncertainty/index.html)
  links its policy, guidelines, and Technical Note 1297. These materials
  explain careful uncertainty statements; NautiPy’s exact statistical model
  remains the one defined in its position-fix specification.

## NautiPy behavior specifications

These repository documents are the authoritative contracts for what the
package accepts, calculates, and rejects:

- [Coordinate input and conversion](https://github.com/cafawo/NautiPy/blob/master/docs/COORDINATES.md)
- [WGS84 navigation](https://github.com/cafawo/NautiPy/blob/master/docs/NAVIGATION.md)
- [GeoJSON interchange](https://github.com/cafawo/NautiPy/blob/master/docs/GEOJSON.md)
- [Bearing and range position fixes](https://github.com/cafawo/NautiPy/blob/master/docs/FIXES.md)
- [Product direction](https://github.com/cafawo/NautiPy/blob/master/docs/PRODUCT.md)
- [Architecture and dependency policy](https://github.com/cafawo/NautiPy/blob/master/docs/ARCHITECTURE.md)
- [Support and public API](https://github.com/cafawo/NautiPy/blob/master/docs/SUPPORT.md)

The [Glossary](glossary.md) gives shorter definitions, while the
[learning path](../index.md) develops the concepts through original visuals
and examples.
