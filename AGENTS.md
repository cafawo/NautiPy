# NautiPy coding-agent guide

This file is the repository contract for coding agents. Human contributors
should start with [CONTRIBUTING.md](CONTRIBUTING.md).

## Read before changing code

Read:

1. [Product direction](docs/PRODUCT.md)
2. [Architecture and dependency policy](docs/ARCHITECTURE.md)
3. [Contributor workflow](CONTRIBUTING.md)
4. The behavior specification relevant to the change:
   [coordinates](docs/COORDINATES.md),
   [navigation](docs/NAVIGATION.md),
   [GeoJSON](docs/GEOJSON.md), or
   [position fixes](docs/FIXES.md)

Read [ROADMAP.md](ROADMAP.md) when selecting new work and
[docs/RELEASING.md](docs/RELEASING.md) for release changes.

When documents disagree, prefer the most specific behavior specification and
update every conflicting document in the same change.

## Current project state

The repository contains the complete planned `0.1.0` feature baseline. One
ordinary installation includes GeographicLib, NumPy, and SciPy and provides
coordinates, navigation, GeoJSON, CLI, and position-fix functionality.

Do not reintroduce a separate fix extra, a missing-extra compatibility path, or
a reduced installation variant. Internal modules may stay lazy so
coordinate-only use does not load geodesic or scientific implementation code.

## Product boundaries

NautiPy exists to:

- accept and safely normalize common coordinate formats;
- represent positions consistently;
- perform dependable WGS84 navigation calculations; and
- estimate positions from bearing and range observations with useful
  diagnostics.

NautiPy is not a general GIS framework or a live-navigation system. Do not add
charts, routing, AIS, device control, weather, tides, magnetic models, plotting,
GUI/web frameworks, databases, or arbitrary CRS analysis without an explicit
product-direction change.

## Engineering invariants

- `pyproject.toml` is the source of truth for version, Python support, package
  metadata, and dependencies.
- Preserve the import name `nautipy`.
- Keep the common API available from the top-level package.
- Store positions as decimal degrees, distances as metres, and bearings as
  true degrees clockwise from north.
- Default navigation and fixing calculations to WGS84.
- Never silently swap latitude and longitude, guess between materially
  different interpretations, wrap invalid input, or change units.
- Use descriptive exceptions for public input errors; do not use assertions
  for validation.
- Do not round intermediate calculations for display convenience.
- Do not expose raw GeographicLib, NumPy, SciPy, or optimizer result objects.
- Coordinate parsing, formatting, models, GeoJSON, and CLI plumbing remain
  standard-library implementations.
- Importing and using coordinate-only functionality must not load
  GeographicLib, NumPy, SciPy, or the private fix solver.
- Use mature numerical libraries for geodesics and nonlinear optimization;
  do not replace them with unvalidated approximations.
- Do not perform network access during import or ordinary calculations.
- Prefer standard-library `unittest` unless a concrete need justifies another
  test dependency.
- Keep the educational GitHub Pages source and toolchain under `website/`.
  Website content, generated HTML, figures, CSS, JavaScript, and documentation
  dependencies must remain absent from both package distributions and runtime
  dependency metadata.
- PyPI is the only maintained package index. Do not create, submit, or
  maintain conda or conda-forge recipes or feedstocks unless the product's
  distribution policy is explicitly changed.

## Working method

1. Inspect the current implementation, tests, and relevant specification
   before deciding what to change.
2. Select a scoped roadmap item, issue, or reproducible bug. Do not redo
   completed foundation work.
3. Define observable behavior in tests before or alongside implementation.
4. Make one coherent change and avoid unrelated rewrites.
5. Update public examples and specifications whenever behavior changes.
6. Run the smallest relevant tests while iterating, then the complete suite.
7. For packaging, imports, or public API changes, also build and smoke-test
   both the wheel and source distribution.
8. For public-site changes, verify Fix Lab fixtures through the public API and
   run a strict MkDocs build with the pinned `website/requirements.txt`.
9. Report commands run, dependency effects, numerical assumptions,
   compatibility effects, and remaining limitations.

## Definition of done

A change is complete when:

- public behavior and failures have meaningful tests;
- coordinate changes cover malformed and ambiguous input;
- numerical changes include independent references and difficult geometry;
- documentation matches the implemented API;
- the complete test suite passes on a supported Python;
- packaging-sensitive changes pass wheel and sdist checks in clean
  environments;
- no unnecessary dependency or compatibility layer was added; and
- the repository remains ready for CI across the supported Python range.

Before `1.0`, prefer a clear correction over compatibility machinery for
behavior that has never been publicly released. Once a release exists, follow
[the support policy](docs/SUPPORT.md).
