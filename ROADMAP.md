# NautiPy roadmap

This roadmap describes what remains to be done. Completed implementation detail
belongs in [CHANGELOG.md](CHANGELOG.md) and the Git history, not in a second
historical checklist.

NautiPy is still preparing its first public release. Until that release is
successfully published, the repository may correct or simplify the emerging
API without compatibility wrappers.

## Implemented 0.1 baseline

The repository already contains the intended first-release feature set:

- an installable `src/` package with an intentional top-level API;
- safe coordinate parsing, inspection, formatting, and conversion;
- immutable validated positions and lightweight GeoJSON interchange;
- WGS84 distance, bearing, destination, interpolation, and nearest-position
  calculations;
- coordinate conversion and inspection commands;
- bearing-only, range-only, and mixed-observation position fixes with
  diagnostics;
- one normal installation containing GeographicLib, NumPy, and SciPy while
  preserving lazy coordinate-only import boundaries;
- tests across the supported Python range, exact-minimum dependency coverage,
  cross-platform smoke checks, and built-artifact tests; and
- tag-triggered PyPI and GitHub Release automation.

Treat this baseline as implemented. Remaining work begins below; position
fixing is part of the ordinary package rather than a separate installation.

## First public release: 0.1.0

The immediate priority is to release the implemented baseline, not to add
another feature area.

### Repository work

- [x] Keep user, contributor, architecture, support, and release documentation
  consistent with the implemented API.
- [ ] Merge the release candidate through a pull request with the required
  `CI success` check passing.
- [ ] Prepare a focused release pull request that confirms the version and
  moves reviewed notes into the exact dated changelog section required by the
  release validator.

### Maintainer release work

- [ ] Confirm the PyPI project or pending Trusted Publisher, protected `pypi`
  environment, and default-branch protection described in
  [docs/RELEASING.md](docs/RELEASING.md).
- [ ] Create and push an annotated `v0.1.0` tag only from the merged
  default-branch release commit.
- [ ] Verify the tested wheel and source distribution on PyPI and the matching
  GitHub Release.

The release is complete only when a clean environment can install
`nautipy==0.1.0`, import the top-level coordinate, navigation, and fixing API,
and pass dependency consistency checks.

## Distribution policy

PyPI is NautiPy's only maintained package index. Matching tested artifacts are
also attached to each GitHub Release. Other package indexes and downstream
redistributions are outside the current roadmap and must not be added without
an explicit product-direction change.

## Version 0.1.x

Patch releases should concentrate on evidence-backed maintenance:

- correct parsing, formatting, navigation, or fixing defects;
- add independently verified reference and difficult-geometry cases;
- improve actionable errors and numerical diagnostics;
- resolve installation or supported-platform problems; and
- clarify documentation from real user questions.

Patch releases must not intentionally break documented behavior or broaden the
project into general GIS or live navigation.

## Version 0.2.x

Minor pre-1.0 releases may refine the public API when real use demonstrates a
clear benefit. Priorities are:

- simplify common coordinate and fixing workflows;
- resolve documented ambiguities without silent guessing;
- improve tolerances and diagnostics using reproducible evidence; and
- remove accidental public surface before it becomes a 1.0 commitment.

New dependencies or feature areas require an explicit product decision.

## Version 1.0

Version 1.0 is ready when maintainers can:

- freeze a small, intentional top-level API;
- document all supported input, output, error, and diagnostic behavior;
- justify numerical tolerances against the reference corpus;
- confirm the supported Python and platform policy in CI;
- keep PyPI metadata, release artifacts, and documentation aligned; and
- maintain the package without compatibility layers for accidental internals.

## Continuing non-goals

The roadmap does not include general CRS transformation, charts, routing, AIS,
live GPS connections, magnetic models, tides, weather, plotting, a GUI, a web
service, or a plugin framework. A proposal in one of these areas must first
change the product direction explicitly.

Contributors should select a clearly scoped issue or reproducible bug. The
first-release checklist above is maintained and executed by project
maintainers; completed baseline work is not an open milestone.
