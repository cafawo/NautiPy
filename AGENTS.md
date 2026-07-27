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
5. The educational website page or pages that teach the affected behavior
   under [website/content](website/content)

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
   and educational page before deciding what to change.
2. Select a scoped roadmap item, issue, or reproducible bug. Do not redo
   completed foundation work.
3. Record the documentation impact before implementation: name the exact
   behavior specification and educational page or explain why the change has
   no externally observable effect.
4. Define observable behavior in tests before or alongside implementation.
5. Make one coherent change and update its specifications, learning material,
   examples, and diagnostics in the same change.
6. Run the smallest relevant tests while iterating, then the complete suite.
7. For packaging, imports, or public API changes, also build and smoke-test
   both the wheel and source distribution.
8. For public functionality or public-site changes, verify Fix Lab fixtures
   through the public API, run the website tests, and run a strict MkDocs
   build with the pinned `website/requirements.txt`.
9. Report commands run, dependency effects, numerical assumptions,
   compatibility effects, documentation impact, and remaining limitations.

## Documentation impact

Treat documentation as part of the implementation, not as follow-up work.
A change has public documentation impact when it alters an accepted input,
result, error, unit, default, diagnostic, CLI or GeoJSON behavior, public name,
dependency expectation, or documented limitation.

Update both layers in the same change:

- the exact contract in `docs/`; and
- the explanation, example, and any affected visual or Fix Lab scenario under
  `website/content/`.

Use this routing:

- coordinate parsing, inspection, conversion, or `Position` behavior:
  `docs/COORDINATES.md`, `website/content/learn/coordinates.md`, and relevant
  recipes or glossary entries;
- distance, bearings, destinations, interpolation, or nearest-position
  behavior: `docs/NAVIGATION.md`,
  `website/content/learn/navigation.md`, and relevant recipes;
- observation conventions and candidate geometry: `docs/FIXES.md` and the
  `website/content/learn/finding-the-boat.md` page;
- solver status, residual, conditioning, weighting, or uncertainty behavior:
  `docs/FIXES.md` and `website/content/learn/trusting-a-fix.md`;
- scenario semantics or numerical teaching fixtures: `docs/FIXES.md`, the Fix
  Lab page at `website/content/learn/fix-lab.md`,
  `website/tools/generate_fix_lab.py`, and the committed fixture;
- GeoJSON structure or ordering: `docs/GEOJSON.md`, the coordinate explanation
  at `website/content/learn/coordinates.md` when order concepts change, and
  `website/content/practical-use.md` when interchange recipes change;
- CLI behavior: the relevant behavior specification, README example, and
  `website/content/practical-use.md`; and
- installation, dependencies, public surface, architecture, or product scope:
  `pyproject.toml` when metadata changes; the applicable
  `docs/ARCHITECTURE.md`, `docs/SUPPORT.md`, `docs/PRODUCT.md`, and behavior
  specification; plus README and the affected `website/content/index.md`,
  `website/content/how-nautipy-works.md`, or
  `website/content/practical-use.md` material.

Update `CHANGELOG.md` for user-visible release changes. If a change is wholly
private—for example, a refactor with identical observable behavior—do not make
a meaningless documentation edit. Instead, state `Documentation impact: none`
with the concrete reason in the pull request or final handoff.

An agent must not report a public functionality change as complete when only
tests or a behavior specification changed while the educational site remains
stale.

## Definition of done

A change is complete when:

- public behavior and failures have meaningful tests;
- coordinate changes cover malformed and ambiguous input;
- numerical changes include independent references and difficult geometry;
- public functionality changes update both the authoritative behavior
  specification and the affected educational website material;
- changes with no documentation edits include a concrete no-impact
  explanation;
- website fixtures, tests, and the strict build pass whenever public
  functionality or public-site content changes;
- the complete test suite passes on a supported Python;
- packaging-sensitive changes pass wheel and sdist checks in clean
  environments;
- no unnecessary dependency or compatibility layer was added; and
- the repository remains ready for CI across the supported Python range.

Before `1.0`, prefer a clear correction over compatibility machinery for
behavior that has never been publicly released. Once a release exists, follow
[the support policy](docs/SUPPORT.md).
