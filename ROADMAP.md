# NautiPy implementation roadmap

This roadmap turns the product direction into an ordered coding plan. The sequence is deliberate: establish an installable and testable baseline, make coordinate input excellent, then build navigation and fix capabilities on that stable model.

Checkboxes describe repository state, not aspiration. Mark them complete only when their acceptance criteria pass on the default branch.

## Current state

The repository is an early proof of concept:

- packaging uses a legacy `setup.py`;
- implementation is concentrated in `nautipy/nautipy.py`;
- public exports are not deliberately defined;
- the dependency list includes the standard-library module `math`;
- calculations mix spherical formulas and planar assumptions;
- public validation uses assertions in places;
- multilateration returns little diagnostic information;
- the existing test contains an unconditional failure; and
- there is no release automation.

Treat existing examples as useful product clues, not as a correctness specification.

## Milestone 0 — Packaging and trustworthy baseline

**Goal:** create a conventional Python project that an agent can install, test, build, and change safely.

### Work

- [ ] Add `pyproject.toml` using PEP 517/PEP 621 metadata and the standard setuptools backend.
- [ ] Move implementation to a `src/nautipy/` layout, or document a compelling reason to retain the current layout.
- [ ] Preserve the package import name `nautipy` and define intentional top-level exports.
- [ ] Replace the invalid dependency metadata. Never list standard-library modules as dependencies.
- [ ] Declare the supported Python range once in `pyproject.toml`; keep classifiers and CI aligned with it.
- [ ] Add a `test` or `dev` extra containing only tools needed to test and build the package.
- [ ] Replace the placeholder failing test with meaningful baseline tests for the historical public behavior that remains supported.
- [ ] Add a build test that produces both an sdist and wheel.
- [ ] Add an install smoke test that imports the built wheel in a clean environment.
- [ ] Add GitHub Actions CI for pull requests and default-branch pushes.
- [ ] Keep the MIT license and include it in built distributions.
- [ ] Establish a changelog with an `Unreleased` section.

### Packaging decisions

- Use `pyproject.toml` as the single source of package metadata and dependency declarations.
- Use a plain version value in project metadata for the first modernization. Do not add dynamic version machinery unless it removes more complexity than it creates.
- Do not require Poetry, uv, Conda, Make, pre-commit, or a particular shell. Contributors may use them, but standard `venv` plus `pip` must work.
- Do not add a linter, formatter, type checker, or documentation generator merely to complete this milestone. A later pull request may add one when there is a concrete benefit and one canonical configuration.
- Coordinate parsing must remain standard-library-only even if the complete package later depends on scientific libraries.

### Acceptance criteria

- `python -m pip install -e ".[test]"` succeeds in a fresh virtual environment.
- The canonical test command exits successfully.
- `python -m build` creates an sdist and wheel.
- Installing the wheel into a fresh environment allows `import nautipy`.
- CI runs the lower bound and newest stable Python covered by `requires-python`, plus cross-platform smoke coverage where practical.
- No test is a placeholder and no public input validation relies on `assert`.
- The README's current examples either work or are explicitly marked as legacy pending a later milestone.

## Milestone 1 — Coordinate intake and conversion

**Goal:** make common coordinate input work without a format argument while refusing unsafe guesses.

The observable specification is [docs/COORDINATES.md](docs/COORDINATES.md).

### Work

- [ ] Add an immutable validated `Position` data model with decimal-degree latitude and longitude.
- [ ] Implement a normalization/tokenization stage that preserves separator meaning.
- [ ] Implement independent parsers for decimal degrees, DDM, DMS, ISO 6709, and NMEA coordinate fields.
- [ ] Support structured pairs, named mappings, and GeoJSON Points.
- [ ] Implement explicit `latlon`, `lonlat`, and evidence-only `auto` ordering.
- [ ] Implement decimal-comma input where pair separators make it unambiguous.
- [ ] Add a candidate-based detection pipeline.
- [ ] Add `parse_coordinate`, `parse_position`, and an inspection API.
- [ ] Add `format_coordinate`, `format_position`, and `convert_position`.
- [ ] Add a coordinate-specific exception hierarchy with actionable messages.
- [ ] Preserve source precision or estimated resolution in inspection metadata where practical.
- [ ] Add examples to the README before calling the milestone complete.

### Implementation guidance

- Do not implement the parser as one giant regular expression.
- Harmless normalization may be permissive; semantic validation must be strict.
- Never infer coordinate order from a likely real-world location.
- Never silently wrap out-of-range longitude input.
- Ensure the coordinate module imports without NumPy, SciPy, or geodesic modules.
- Canonicalize format names in metadata; accept documented aliases such as `dmm` for `ddm`.

### Acceptance criteria

- All reference inputs in `docs/COORDINATES.md` resolve to the same `Position` within their stated precision.
- Auto-detection works for every supported input family without a format argument.
- Ambiguous numeric order and decimal-comma cases raise `AmbiguousCoordinateError` with a concrete resolution.
- Formatting never emits invalid 60-minute or 60-second components after rounding.
- Parse-format-parse round trips pass for every output format.
- Boundary and malformed inputs are covered, including NaN, infinity, legal extrema, conflicting signs, and extra fields.
- Typical parser use has no network, filesystem, NumPy, or SciPy dependency.

### Suggested release checkpoint

The first modern PyPI release can be made after Milestones 0 and 1 if the API is clearly marked pre-1.0. A reasonable next version from the historical `0.1` is `0.2.0`, but the release manager makes the final choice.

## Milestone 2 — Position model and geodesic primitives

**Goal:** replace ad hoc spherical calculations with a small, dependable WGS84 navigation API.

### Work

- [ ] Introduce a geodesic adapter backed by a mature ellipsoidal implementation such as GeographicLib.
- [ ] Implement inverse calculation returning distance, initial bearing, and final bearing.
- [ ] Implement destination calculation from position, true bearing, and distance.
- [ ] Implement geodesic interpolation by fraction or distance.
- [ ] Implement nearest-position lookup over ordinary iterables.
- [ ] Normalize generated bearings to `[0, 360)` and document longitude behavior at the antimeridian.
- [ ] Add explicit unit conversion helpers at API boundaries; store distances in metres internally.
- [ ] Add compatibility wrappers for useful historical `haversine`, `bearing`, and `Pos.displace` calls.
- [ ] Deprecate misleading legacy names or spherical behavior where needed.

### Dependency policy

A mature pure-Python WGS84 dependency is preferred to maintaining another geodesic implementation. Add only the dependency used by the public implementation and record why it was selected. Do not depend on a broad GIS framework solely for direct/inverse geodesics.

### Acceptance criteria

- Results match independent published or library reference cases within documented tolerances.
- Tests cover ordinary routes, short distances, antimeridian crossing, high latitudes, and near-antipodal inputs.
- Public defaults are WGS84 and true bearing.
- A spherical approximation, if retained, is explicitly requested and named.
- Coordinate-only imports remain lightweight.
- Compatibility wrappers are tested and emit useful deprecation guidance where behavior changed.

## Milestone 3 — Observation and fix engine

**Goal:** deliver the package's second main differentiator: easy bearing, range, and mixed-observation fixes with diagnostics.

### Public models

- [ ] `BearingObservation`: known station, true bearing, optional standard uncertainty, optional identifier.
- [ ] `RangeObservation`: known station, distance in metres, optional standard uncertainty, optional identifier.
- [ ] `FixResult`: position, success state, residuals, objective/RMS information, iterations, warnings, and uncertainty information when valid.
- [ ] Structured exceptions or result states for insufficient data, impossible geometry, ambiguity, and non-convergence.

The final class names may change, but ordinary users must be able to construct observations without arrays or optimizer-specific concepts.

### Exact and candidate geometry

- [ ] Two-bearing intersection with explicit handling of parallel, nearly parallel, coincident, and backward-ray geometry.
- [ ] Two-range circle intersection returning zero, one, or two candidates rather than choosing silently.
- [ ] Candidate filtering and initial-guess generation for later least-squares solving.

### Weighted solver

- [ ] Bearing-only fixes with more observations than the mathematical minimum.
- [ ] Range-only fixes with three or more observations.
- [ ] Mixed bearing/range fixes.
- [ ] Wrapped angular residuals for bearing observations.
- [ ] Residual scaling from provided uncertainty.
- [ ] A stable local coordinate representation for optimization; do not optimize raw latitude/longitude degrees without a documented numerical justification.
- [ ] Bounds, convergence checks, and finite-result validation.
- [ ] Multiple starting candidates where geometry can produce local alternatives.
- [ ] Covariance or confidence ellipse when the local linearization is meaningful.
- [ ] Geometry diagnostics for weak dilution, near-collinearity, parallel bearings, and unresolved competing solutions.

### Dependency policy

NumPy and SciPy are acceptable core dependencies when used for the weighted nonlinear solver and diagnostics. They must not be imported by coordinate-only modules. Do not expose SciPy result objects as the NautiPy API, and do not require optional optimization packages for the standard fix workflow.

### Acceptance criteria

- Exact synthetic cases recover their known positions within documented tolerance.
- Noisy overdetermined cases improve or remain stable when valid observations are added.
- Observation weighting changes the solution in the expected direction.
- Bearing residuals behave correctly across the `0°/360°` boundary.
- Degenerate and impossible geometry never returns an unexplained plausible-looking position.
- Competing range or fix solutions are returned or reported explicitly.
- `FixResult` remains serializable to ordinary Python/JSON-compatible data after converting `Position` values.
- Historical `triangulate` and `multilaterate` calls have tested compatibility adapters or documented migrations.

## Milestone 4 — Interchange, CLI, and migration experience

**Goal:** make the core workflows practical without turning NautiPy into an application framework.

### Work

- [ ] Replace ad hoc GeoJSON handling with validated Point and FeatureCollection import/export.
- [ ] Preserve descriptions and identifiers when round-tripping positions.
- [ ] Reject unsupported geometry types with clear messages.
- [ ] Add a small standard-library CLI if it reduces friction for coordinate conversion or fix inspection.
- [ ] Add copyable end-to-end examples for parsing, conversion, geodesics, and fixes.
- [ ] Add a migration guide from the historical API.
- [ ] Remove obsolete compatibility code only after its documented deprecation period.

### CLI boundaries

A CLI may provide commands such as:

```text
nautipy convert "50° 7.3542' N; 8° 39.942' E" --to dd
nautipy inspect "+50.12257+008.66570/"
```

Do not add an interactive UI, network service, map renderer, or configuration framework.

### Acceptance criteria

- GeoJSON round trips preserve position and supported metadata.
- The CLI, if shipped, uses the same public parser/formatter functions as Python callers.
- Examples execute in CI or are covered by equivalent tests.
- Migration guidance names replacements and unit differences explicitly.

## Milestone 5 — Release automation and distribution

**Goal:** publish tested artifacts with minimal manual credential handling and make NautiPy available to pip and conda-forge users.

The complete process is defined in [docs/RELEASING.md](docs/RELEASING.md).

### Work

- [ ] Add a tag-triggered GitHub Actions release workflow.
- [ ] Validate that the tag, package metadata version, and changelog agree.
- [ ] Build the sdist and wheel once, test those exact artifacts, and pass them to the publish job.
- [ ] Publish to PyPI through Trusted Publishing/OIDC with a protected GitHub environment.
- [ ] Create a GitHub Release from the same tag and artifacts.
- [ ] Pin third-party actions immutably and configure automated update PRs for those pins.
- [ ] Document one-time PyPI project and trusted-publisher setup.
- [ ] After a stable PyPI release, submit a conda-forge staged-recipes recipe.
- [ ] Maintain subsequent conda releases through the generated feedstock and conda-forge bot PRs.

### Acceptance criteria

- Pull requests cannot publish packages.
- A release requires an intentional semantic-version tag.
- The publish job has only the permissions it needs and uses no long-lived PyPI token.
- The artifact published to PyPI is byte-for-byte the artifact tested by the release workflow.
- Duplicate or mismatched versions fail loudly.
- `pip install nautipy` works from PyPI for every supported Python version with available dependencies.
- `conda install -c conda-forge nautipy` works once the feedstock is accepted.

## Milestone 6 — Version 1.0 stabilization

**Goal:** declare a small stable API after real use, not merely after feature completion.

### Work

- [ ] Review top-level exports and remove accidental public internals.
- [ ] Resolve all known input ambiguities with documented behavior.
- [ ] Review numerical tolerances and warning thresholds using a reference-case corpus.
- [ ] Complete compatibility deprecations planned for 1.0.
- [ ] Freeze the documented coordinate, geodesic, observation, and result APIs.
- [ ] Publish API stability and support policies.
- [ ] Ensure conda-forge metadata matches PyPI runtime requirements.

### Acceptance criteria

- The four core workflows in `docs/PRODUCT.md` are stable, tested, and documented.
- No known case returns a plausible-looking fix after solver failure or ambiguous geometry.
- Public exceptions and result fields are documented.
- The release process has successfully published at least one pre-1.0 release.
- Backwards compatibility follows semantic versioning from this point onward.

## Recommended pull-request sequence for an agent

Keep early changes reviewable. A practical sequence is:

1. **Packaging baseline:** `pyproject.toml`, package layout, exports, real tests, build smoke test.
2. **CI baseline:** pull-request tests and artifact build, without publishing.
3. **Coordinate core:** `Position`, exceptions, normalization, decimal degrees, explicit order.
4. **Coordinate formats:** DDM, DMS, ISO 6709, NMEA fields, decimal comma.
5. **Formatting and diagnostics:** conversion, inspection results, round-trip and ambiguity tests.
6. **Geodesics:** WGS84 adapter and public direct/inverse functions.
7. **Fix models and exact intersections.**
8. **Weighted mixed solver and diagnostics.**
9. **GeoJSON, migration guide, and optional CLI.**
10. **Release workflow and first PyPI release.**
11. **conda-forge staged-recipes submission.**

Do not combine the packaging rewrite, parser, solver, and release workflow into one pull request.

## Feature admission test

Before adding a new feature, answer all of the following:

1. Does it directly improve coordinate-to-position or observation-to-fix work?
2. Can it be implemented against a stable public specification or mature dependency?
3. Can ambiguity and failure be represented honestly?
4. Can it remain deterministic and offline?
5. Does its user value exceed its API and maintenance cost?
6. Can it be tested without fragile external services or time-varying datasets?

A "no" does not permanently reject a feature, but it means the product documents must be changed deliberately before implementation.
