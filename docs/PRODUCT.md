# NautiPy product direction

## Promise

**NautiPy turns real-world coordinate input into validated positions and makes common WGS84 navigation calculations easy through a small Python API.**

The package should be useful within a minute of installation, clear when input
is ambiguous, deterministic, offline, and complete after one ordinary install.

## Why this package should exist

Python already has strong geodesic, projection, and GIS libraries. NautiPy should not duplicate their breadth. Its value is the simple workflow around a position:

1. accept coordinates as people and devices actually express them;
2. detect, validate, inspect, and convert those forms;
3. represent positions consistently;
4. calculate the navigation quantities users commonly need; and
5. estimate a position from bearings and ranges with diagnostics.

Coordinate usability is the first differentiator. Diagnosed position fixing is
the advanced differentiator. General GIS is not the goal.

## Clean start

The old repository code was experimental and did not establish a supported public package. The new package begins with a clean API and version `0.1.0`.

There will be:

- no compatibility wrappers for `Pos`, `haversine`, `triangulate`, or other experimental names;
- no deprecation cycle for code that was never released as a supported API;
- no migration guide from the experimental layout; and
- no obligation to preserve incorrect formulas or awkward call patterns.

Before version 1.0, minor releases may change the API when doing so materially
improves simplicity or correctness; patch releases should remain compatible.
Starting at 1.0, documented APIs follow semantic versioning.

## Product principles

### Ease of use first

The common path is one call and no format declaration:

```python
from nautipy import parse_position

position = parse_position("N 50° 7' 19.2\"; E 8° 39' 56.5\"")
```

Users should not need to know whether input is DD, DDM, DMS, ISO 6709, or NMEA before parsing it. They should not need arrays, a configuration object, or a GIS data model for one position.

### Safe automation

NautiPy normalizes harmless syntax differences but never silently chooses between different valid locations.

```python
parse_position("120, 50", order="auto")  # order is provable: lon/lat
parse_position("8, 50", order="auto")    # raises an ambiguity error
```

Errors should name the ambiguity and show the argument or syntax that resolves it.

### One complete installation

`python -m pip install nautipy` installs the complete package. GeographicLib
provides correct WGS84 geodesics, while NumPy and SciPy provide the mature
numerical foundation for bearing/range fixes. These are direct runtime
dependencies rather than user-selected extras.

The coordinate implementation still uses only the Python standard library.
Coordinate-only module use must not import GeographicLib, NumPy, SciPy, or the
fix solver, so implementation boundaries remain clear without creating a
second installation mode.

### Small public surface

A user should be able to learn the main package from the top-level namespace. Internal parser tokens, backend objects, and optimizer details remain private.

### Trustworthy results

Navigation defaults to WGS84 and true bearings. A position-fix result includes
residuals and geometry/convergence information rather than returning only
plausible-looking coordinates.

## Target workflows

### Parse and normalize

```python
from nautipy import parse_position

p1 = parse_position("50.12257, 8.66570")
p2 = parse_position("50° 7.3542' N; 8° 39.942' E")
p3 = parse_position("+50.12257+008.66570/")

assert p1 == p2 == p3
```

### Inspect detection

```python
from nautipy import inspect_position

result = inspect_position("5007.3542,N,00839.9420,E")
print(result.position)
print(result.format)
print(result.normalizations)
```

### Convert formats

```python
from nautipy import convert_position

text = convert_position(
    "50.12257, 8.66570",
    to="dms",
    precision=2,
)
```

### Calculate navigation values

Position-taking functions should accept `Position` and documented position-like input:

```python
from nautipy import destination, distance, initial_bearing

start = "50.12257, 8.66570"
end = destination(start, bearing=90, distance=12_000)

metres = distance(start, end)
degrees_true = initial_bearing(start, end)
```

Distances are metres and bearings are true degrees by default. Display units are converted at the API boundary.

### Solve a position fix

```python
from nautipy import (
    BearingObservation,
    Position,
    RangeObservation,
    solve_fix,
)

references = (
    Position(50.116135, 8.670277),
    Position(50.112836, 8.666753),
    Position(50.110347, 8.659873),
)
result = solve_fix(
    bearings=[
        BearingObservation(references[0], 164.71, uncertainty=0.05),
        BearingObservation(references[2], 192.22, uncertainty=0.05),
    ],
    ranges=[
        RangeObservation(references[1], 1_599.237, uncertainty=2.0),
    ],
)

print(result.position)
print(result.residuals)
print(result.warnings)
```

Bearings are true initial bearings measured at the unknown position toward a
known reference. Uncertainties are required one-standard-deviation values in
degrees or metres. Ambiguous and degenerate cases do not carry a selected
position.

## First public release: 0.1.0

The initial useful release should ship:

- immutable validated `Position` values;
- automatic detection of common coordinate formats;
- explicit coordinate-order controls and useful ambiguity errors;
- formatting and conversion among DD, DDM, DMS, ISO 6709, and NMEA fields;
- parser inspection metadata;
- WGS84 distance, initial/final bearing, destination, and interpolation;
- lightweight GeoJSON Point/FeatureCollection interchange;
- a small `argparse` CLI for conversion and inspection;
- an integrated bearing/range fix engine satisfying its numerical acceptance
  criteria;
- wheel and source distributions on PyPI;
- automated tested releases from semantic-version tags; and
- a conda-forge staged-recipes submission after the PyPI release.

The 0.1.0 installation includes coordinates, navigation, interchange, CLI,
and the nonlinear fix solver. The complete coordinate, navigation, and fixing
APIs are available from top-level `nautipy` after one ordinary installation;
specialized GeoJSON functions remain grouped in `nautipy.geojson`.

## Position-fix capability

The fix engine provides:

- two-bearing and two-range candidate geometry;
- overdetermined bearing-only fixes;
- range-only fixes;
- mixed bearing/range fixes;
- observation uncertainty and weighting;
- residuals in natural units;
- convergence and geometry diagnostics; and
- covariance or confidence information where valid.

NumPy and SciPy are normal package dependencies used directly by this
capability. Coordinate parsing, formatting, GeoJSON, and CLI plumbing remain
standard-library implementations and must not import the scientific stack
during coordinate-only module use.

The detailed contract is [FIXES.md](FIXES.md). It fixes bearing direction,
units, regional search bounds, result status, residual signs, ambiguity, and
the conditions under which local covariance is meaningful.

## Coordinate capability

The detailed contract is [COORDINATES.md](COORDINATES.md). It includes:

- signed and hemisphere-qualified decimal degrees;
- DDM and DMS;
- common ASCII and Unicode symbol variants;
- unambiguous ISO 6709 forms;
- NMEA latitude/longitude fields without a full sentence stack;
- decimal-comma input when separators prove the meaning;
- strings, numeric pairs, named mappings, and GeoJSON Points;
- explicit `latlon`, `lonlat`, and evidence-only `auto` ordering;
- canonical formatting and round trips; and
- an inspection result explaining what was detected.

## Deliberate non-goals

Do not add these merely because they are nautical or geospatial:

- arbitrary CRS transformation;
- chart display, route planning, bathymetry, AIS, autopilot, or collision avoidance;
- live GPS/device connections;
- complete NMEA sentence decoding or streaming;
- magnetic models, tides, currents, weather, or ephemerides;
- map tiles, hosted APIs, or runtime downloads;
- plotting, GUI, web-server, database, or dataframe frameworks;
- a plugin system or selectable geodesic backends;
- a general units package; or
- generic computational geometry unrelated to positions and fixes.

Users needing broader GIS or CRS work should combine NautiPy with specialist packages rather than expanding NautiPy into their replacement.

## Success criteria

NautiPy is succeeding when:

- ordinary coordinate input works without users identifying its notation;
- ambiguous input fails with a concrete resolution;
- the core API is small enough to understand from examples;
- one ordinary installation provides every documented feature;
- coordinate-only module use remains isolated from geodesic and scientific
  imports;
- navigation results match independent WGS84 references;
- built artifacts install and work outside the repository checkout;
- releases are automated but intentional; and
- new features make the main workflows simpler rather than merely increasing feature count.
