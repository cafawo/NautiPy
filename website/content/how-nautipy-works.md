# How NautiPy Works

NautiPy has two public workflows and three internal layers. The separation is
partly conceptual—each layer has a different job—and partly practical:
coordinate-only use does not load geodesic or scientific implementation code.

<div class="educational-figure" tabindex="0" role="group" aria-label="Scrollable diagram" markdown>

![NautiPy package flow from messy coordinate input through a validated
Position to navigation or GeoJSON, and from observations through WGS84
predictions and a bounded fit to diagnostics.](assets/images/package-flow.svg)

</div>

## Workflow 1: text becomes a position

### 1. Preserve meaning while normalizing presentation

The coordinate parser can normalize whitespace, common symbols, direction
words, and unambiguous decimal separators. It keeps enough source information
to determine whether punctuation is a decimal mark, a component separator, or
a pair separator.

### 2. Generate and test interpretations

An input may have evidence for a notation and axis order. NautiPy validates
candidate interpretations against:

- DD, DDM, DMS, ISO 6709 subset, and NMEA-field grammar;
- latitude, longitude, minute, and second ranges;
- signs and hemisphere markers;
- named fields or GeoJSON structure; and
- the caller’s explicit `format` and `order` choices.

One valid interpretation becomes a `ParseResult`. Materially different valid
positions produce `AmbiguousCoordinateError`, not a best guess.

```python
from nautipy import inspect_position

result = inspect_position("50° 7.3542' N; 8° 39.942' E")
print(result.position)
print(result.format)
print(result.evidence)
```

### 3. Store one simple representation

`Position` stores finite latitude and longitude floats in decimal degrees,
plus optional identifier and description metadata. It does not remember a
display notation: formatting is a separate boundary.

### 4. Send the position onward

The validated position can go to:

- coordinate formatting or inspection;
- Point and Point FeatureCollection GeoJSON interchange; or
- WGS84 distance, bearing, destination, interpolation, and nearest-position
  calculations.

Coordinate parsing, formatting, models, GeoJSON, and command-line plumbing use
only the Python standard library. GeographicLib is loaded only when a
navigation or fix calculation needs it.

## Workflow 2: observations become a diagnosed fix

### 1. Validate the observations

`BearingObservation` and `RangeObservation` convert their references to
validated `Position` values. Bearings are true degrees at the unknown
position toward a reference; ranges are metres. Every observation includes a
positive one-standard-deviation uncertainty in its natural unit.

### 2. Declare a regional search

The solution domain is a closed WGS84-distance disk. A caller may provide its
center and radius, or use a deterministic center derived from the references
and the 500 km default radius.

This is an intentional regional model. Search-domain membership is measured on
WGS84 rather than by checking a rectangular latitude/longitude box.

### 3. Search in local coordinates, predict on WGS84

The private solver works in bounded local east/north metre coordinates around
an anchor. It uses deterministic multiple starting points to look for
different basins.

At every trial position it predicts:

- shortest WGS84 surface distance for each range; and
- true WGS84 initial bearing from the trial position toward each bearing
  reference.

The residuals are divided by their observation uncertainties, then fitted by
bounded nonlinear least squares. NumPy supplies arrays and linear algebra;
SciPy supplies the optimizer; GeographicLib supplies the geodesic predictions.

### 4. Classify, do not merely return

NautiPy compares retained basins and inspects the local Jacobian. The public
`FixResult` reports:

- status, position, message, and warnings;
- natural and standardized residuals;
- objective and RMS fit summaries;
- rank, condition number, and degrees of freedom;
- competing positions when ambiguous; and
- local uncertainty when it is meaningful.

Raw GeographicLib dictionaries, NumPy arrays, SciPy optimizer results, and
private solver state never become the public result.

## Why lazy layers matter

One ordinary installation includes GeographicLib, NumPy, and SciPy, so there
is no reduced edition or optional fix extra. Lazy imports still keep the
conceptual boundary clean:

```text
parse or format coordinates  → standard library only
request navigation           → add GeographicLib
request a position fix       → add NumPy and SciPy
```

This is about startup and separation of concerns, not about hiding features
behind another installation command.

## Deterministic and offline by design

Ordinary parsing and calculations:

- perform no network access;
- do not download maps, models, or data;
- do not depend on locale, wall-clock time, or random starts; and
- use explicit WGS84, metre, and true-bearing conventions.

That makes examples reproducible. It does not make source observations more
accurate or turn the package into certified navigation equipment.

## What NautiPy deliberately does not do

NautiPy is not a general GIS, chart display, route planner, live GPS reader,
AIS client, weather service, arbitrary coordinate-reference-system engine, or
magnetic model. Those are separate domains with separate data and safety
requirements.

The exact design contracts are the
[product direction](https://github.com/cafawo/NautiPy/blob/master/docs/PRODUCT.md),
[architecture and dependency policy](https://github.com/cafawo/NautiPy/blob/master/docs/ARCHITECTURE.md),
[coordinate specification](https://github.com/cafawo/NautiPy/blob/master/docs/COORDINATES.md),
[navigation specification](https://github.com/cafawo/NautiPy/blob/master/docs/NAVIGATION.md),
[GeoJSON specification](https://github.com/cafawo/NautiPy/blob/master/docs/GEOJSON.md),
and
[position-fix specification](https://github.com/cafawo/NautiPy/blob/master/docs/FIXES.md).

Continue with [Practical Use](practical-use.md) or return to the
[learning path](index.md).
