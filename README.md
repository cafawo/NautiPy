# NautiPy

**A small Python package for effortless coordinate handling and trustworthy WGS84 navigation calculations.**

> **Status:** clean pre-release rewrite. The experimental code previously in this repository is being replaced and is not a supported API. The first public package will start at version `0.1.0`.

## What NautiPy is for

NautiPy focuses on the path from messy coordinate input to a validated, useful position:

```text
paste or receive coordinates
          ↓
automatically detect and validate them
          ↓
convert, inspect, exchange, or calculate navigation values
```

The main value is:

- automatic detection of common coordinate formats;
- clear errors when latitude/longitude order or syntax is genuinely ambiguous;
- conversion among decimal degrees, DDM, DMS, ISO 6709, and NMEA fields;
- a small immutable `Position` model;
- WGS84 distance, bearing, destination, and interpolation; and
- a lightweight installation without a general GIS or scientific stack.

A bearing/range position-fix engine is planned later as an optional extra so ordinary users do not install NumPy and SciPy unnecessarily.

## Target 0.1 API

The examples below describe the intended first public release and will be converted into tested examples as milestones land.

### Parse whatever notation you have

```python
from nautipy import parse_position

p1 = parse_position("50.12257, 8.66570")
p2 = parse_position("50° 7.3542' N; 8° 39.942' E")
p3 = parse_position("+50.12257+008.66570/")

assert p1 == p2 == p3
```

### Inspect what was detected

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

```python
from nautipy import destination, distance, initial_bearing

start = "50.12257, 8.66570"
end = destination(start, bearing=90, distance=12_000)

print(distance(start, end))
print(initial_bearing(start, end))
```

Distances use metres and bearings use true degrees by default.

## Input philosophy

NautiPy is permissive about presentation and strict about meaning.

It should handle harmless variation in whitespace, Unicode symbols, decimal separators, and hemisphere placement. It must not silently choose between two valid locations.

```python
parse_position("120, 50", order="auto")  # longitude/latitude is provable
parse_position("8, 50", order="auto")    # raises AmbiguousCoordinateError
```

## Lightweight architecture

The package is deliberately layered:

- **coordinates:** Python standard library only;
- **normal navigation:** at most one focused pure-Python WGS84 dependency;
- **advanced fixes:** optional `nautipy[fix]` extra in a later release.

NautiPy will not require pyproj, Shapely, pandas, NumPy, or SciPy for ordinary coordinate and navigation use.

## Scope boundaries

NautiPy is not intended to provide general CRS transformation, chart display, route planning, AIS, live GPS connections, magnetic models, tides, weather, plotting, a GUI, or a web service.

It is a calculation library, not certified navigation equipment.

## Development

The repository-local plan is designed for both human contributors and coding agents:

- [Coding-agent guide](AGENTS.md)
- [Product direction](docs/PRODUCT.md)
- [Architecture and dependency policy](docs/ARCHITECTURE.md)
- [Coordinate detection and conversion specification](docs/COORDINATES.md)
- [Implementation roadmap](ROADMAP.md)
- [Release and distribution plan](docs/RELEASING.md)
- [Contribution guide](CONTRIBUTING.md)

The first implementation milestone replaces the experimental project with a clean `src/` package, immutable `Position`, decimal-degree parsing and formatting, standard-library tests, package builds, and CI. It intentionally provides no compatibility layer for the old code.

## Distribution plan

After the 0.1 feature set passes artifact tests:

1. an intentional semantic-version tag triggers GitHub Actions;
2. the workflow validates, builds, and tests the exact wheel and sdist;
3. those artifacts are published to PyPI through Trusted Publishing;
4. the same artifacts are attached to a GitHub Release; and
5. the stable PyPI sdist is submitted to conda-forge through staged-recipes.

Merging a pull request never publishes a package.