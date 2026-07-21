# NautiPy implementation roadmap

This roadmap builds the smallest useful package first, then adds capability without making the default installation heavy.

Checkboxes describe repository state. Mark an item complete only when its acceptance criteria pass on the default branch.

## Starting decision

The existing repository is an experiment, not a supported package. The rewrite is clean-slate:

- remove the legacy packaging and implementation;
- do not preserve experimental imports or behavior;
- do not add compatibility wrappers or deprecation machinery;
- begin the public package at version `0.1.0`; and
- optimize for a small, clear API and a low dependency budget.

The old code may supply domain examples, but every numerical result must be independently verified before reuse.

## Release shape

### Version 0.1.0

A polished coordinate and WGS84 navigation package:

- coordinate detection, parsing, inspection, formatting, and conversion;
- immutable `Position` values;
- geodesic distance, bearings, destination, and interpolation;
- lightweight GeoJSON and CLI support;
- pip/PyPI release automation; and
- conda-forge submission.

### Later 0.x release

An optional `fix` extra for bearing/range position estimation using scientific dependencies.

This sequencing protects the main value and keeps `pip install nautipy` light.

---

## Milestone 0 — Clean package and decimal-degree vertical slice

**Goal:** replace the experiment with a real installable package that already performs one useful task.

### Remove

- [ ] Delete legacy `setup.py` after `pyproject.toml` replaces it.
- [ ] Delete the old `nautipy/nautipy.py` implementation and accidental package exports.
- [ ] Delete placeholder and unconditional-failure tests.
- [ ] Remove old README examples that imply unsupported behavior.

Do not move these elements into a `legacy` module and do not wrap them.

### Package foundation

- [ ] Add PEP 517/PEP 621 metadata in `pyproject.toml` using setuptools.
- [ ] Use a `src/nautipy/` layout.
- [ ] Set the initial development version to `0.1.0`.
- [ ] Declare a concrete supported Python range in `pyproject.toml` and align CI with it.
- [ ] Keep runtime dependencies empty for this milestone.
- [ ] Include the MIT license and project documentation in built distributions.
- [ ] Add `CHANGELOG.md` with an `Unreleased` section.

### Useful vertical slice

- [ ] Add an immutable validated `Position(latitude, longitude)` dataclass.
- [ ] Reject non-finite and out-of-range values with descriptive exceptions.
- [ ] Add `parse_position` for decimal-degree strings and two-value Python sequences.
- [ ] Support explicit `order="latlon"` and `order="lonlat"`.
- [ ] Add evidence-only `order="auto"` for range-proven or equivalent orders.
- [ ] Raise `AmbiguousCoordinateError` when valid orders produce different positions.
- [ ] Add canonical decimal-degree `format_position` output.
- [ ] Define intentional top-level exports in `nautipy.__init__`.

The first slice should not attempt DDM, DMS, ISO 6709, NMEA, geodesics, GeoJSON, or fixing.

### Tests and CI

- [ ] Use standard-library `unittest` for public behavior and errors.
- [ ] Test legal extrema, NaN, infinity, malformed pairs, explicit order, and ambiguous auto order.
- [ ] Add GitHub Actions CI for pull requests and default-branch pushes.
- [ ] Test the oldest and newest stable Python versions declared by the package.
- [ ] Build both wheel and sdist.
- [ ] Install the built wheel in a clean environment and run an import plus parse/format smoke test.

### Acceptance criteria

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m build
```

All commands succeed in a fresh virtual environment after installing the non-runtime build tool required by the last command.

The built wheel:

- installs without pulling runtime dependencies;
- imports outside the source checkout;
- parses and formats a decimal-degree position; and
- exposes no legacy aliases.

---

## Milestone 1 — Complete coordinate detection and conversion

**Goal:** make coordinate intake the package's first standout feature.

The observable behavior is defined in [docs/COORDINATES.md](docs/COORDINATES.md).

### Parsing

- [ ] Add a normalization/tokenization stage that preserves separator meaning.
- [ ] Add decimal degrees with sign and hemisphere markers.
- [ ] Add degrees and decimal minutes (`ddm`, with common `dmm` input alias).
- [ ] Add degrees, minutes, and seconds.
- [ ] Add unambiguous ISO 6709 forms.
- [ ] Add NMEA latitude/longitude field pairs without full sentence decoding.
- [ ] Add decimal-comma input where pair syntax makes it unambiguous.
- [ ] Add named mappings and recognized GeoJSON Point objects.
- [ ] Keep explicit `latlon`, `lonlat`, and evidence-only `auto` order.
- [ ] Use a candidate parser; do not build one giant regular expression.

### Inspection and formatting

- [ ] Add `inspect_position` returning position, detected format, normalizations, warnings, and candidate information.
- [ ] Add canonical DD, DDM, DMS, ISO 6709, and NMEA formatting.
- [ ] Add `convert_position` as parse plus format.
- [ ] Handle precision, carry, and negative zero correctly.
- [ ] Add a small documented coordinate exception hierarchy.

### Dependency rule

Coordinate modules remain standard-library-only. Importing or using them must not load a geodesic or scientific dependency.

### Acceptance criteria

- Every reference input in `docs/COORDINATES.md` resolves correctly without a format argument.
- Material ambiguity raises an error showing how to resolve it.
- Parse-format-parse round trips pass for every output format.
- Boundary, malformed, conflicting-sign, extra-field, decimal-comma, and Unicode cases are tested.
- The wheel remains pure Python and coordinate use has no runtime dependency.

---

## Milestone 2 — WGS84 navigation core

**Goal:** make the normal NautiPy installation useful for navigation while adding only one focused dependency.

### Dependency decision

- [ ] Evaluate the current maintained GeographicLib Python package against the required Python range.
- [ ] Record the choice in the pull request.
- [ ] Add one runtime dependency only after tests demonstrate its use for the public API.
- [ ] Do not add pyproj, Shapely, NumPy, or SciPy to the normal installation.

### Public API

- [ ] Add an inverse result model containing distance, initial bearing, and final bearing.
- [ ] Add `distance(start, end)` returning metres.
- [ ] Add `initial_bearing(start, end)` returning true degrees.
- [ ] Add `destination(start, bearing, distance)`.
- [ ] Add `interpolate(start, end, fraction=...)` or an equally small documented API.
- [ ] Add nearest-position lookup over ordinary iterables if it remains simple and demonstrably useful.
- [ ] Accept `Position` and documented position-like values through one shared coercion path.

### Correctness

- [ ] Match independent WGS84 reference cases.
- [ ] Cover short distances, antimeridian crossing, high latitudes, and near-antipodal inputs.
- [ ] Normalize generated bearings to `[0, 360)`.
- [ ] Keep distance metres and bearings true degrees internally.
- [ ] Do not expose third-party result objects.

### Acceptance criteria

- Navigation results match documented references within justified tolerances.
- Coordinate-only modules still import without loading the geodesic implementation.
- The normal installation has at most one runtime dependency.
- No spherical approximation is presented as the WGS84 default.

---

## Milestone 3 — Interchange, CLI, and first public release

**Goal:** finish the practical 0.1.0 experience and publish it reproducibly.

### GeoJSON

- [ ] Add standard-library GeoJSON Point and FeatureCollection import/export.
- [ ] Preserve supported identifiers and descriptions.
- [ ] Follow GeoJSON longitude/latitude order explicitly.
- [ ] Reject unsupported geometry instead of ignoring it.

### CLI

- [ ] Add a small `argparse` CLI only for shipped library workflows.
- [ ] Provide `nautipy convert` and `nautipy inspect`.
- [ ] Reuse public parsing and formatting functions; do not duplicate logic.
- [ ] Keep the CLI deterministic and offline.

Example target:

```text
nautipy convert "50° 7.3542' N; 8° 39.942' E" --to dd
nautipy inspect "+50.12257+008.66570/"
```

### Documentation

- [ ] Replace planned examples with examples verified against the installed wheel.
- [ ] Add concise API examples for parsing, conversion, navigation, GeoJSON, and CLI use.
- [ ] State that NautiPy is not certified navigation equipment.

### Release automation

- [ ] Add a tag-triggered GitHub Actions release workflow.
- [ ] Validate that tag, project version, and changelog agree.
- [ ] Build the wheel and sdist once.
- [ ] Test those exact artifacts before publication.
- [ ] Publish to PyPI through Trusted Publishing/OIDC.
- [ ] Create a GitHub Release from the same artifacts.
- [ ] Keep pull-request workflows unable to publish.

### Distribution

- [ ] Release `0.1.0` on PyPI after name ownership and trusted-publisher setup are confirmed.
- [ ] Verify `pip install nautipy` in a clean environment.
- [ ] Submit a conda-forge staged-recipes recipe built from the PyPI sdist.
- [ ] After acceptance, maintain conda releases through the generated feedstock and bot PRs.

### Acceptance criteria

- A tagged release publishes only artifacts already tested by CI.
- `pip install nautipy` provides the complete coordinate and navigation API.
- `conda install -c conda-forge nautipy` works after feedstock acceptance.
- The default installation contains no scientific or GIS framework stack.

---

## Milestone 4 — Optional bearing/range fix engine

**Goal:** add the advanced differentiator without changing the lightweight normal installation.

### Packaging

- [ ] Add a `fix` optional extra containing only the required scientific dependencies.
- [ ] Ensure `pip install nautipy` remains usable without NumPy or SciPy.
- [ ] Provide one clear missing-extra error containing `pip install "nautipy[fix]"`.
- [ ] Keep optional imports out of coordinate and geodesic module import paths.

### Models

- [ ] Add simple bearing and range observation dataclasses.
- [ ] Add a NautiPy-owned `FixResult` with position, success, residuals, objective/RMS values, iterations, warnings, and uncertainty information when valid.
- [ ] Do not expose arrays or optimizer result objects in the ordinary API.

### Geometry and solver

- [ ] Add two-bearing and two-range candidate geometry with explicit ambiguous/degenerate outcomes.
- [ ] Add overdetermined bearing-only fixes.
- [ ] Add range-only fixes.
- [ ] Add mixed bearing/range fixes.
- [ ] Wrap angular residuals across `0°/360°`.
- [ ] Scale residuals using observation uncertainty.
- [ ] Optimize in a stable local coordinate representation, not raw latitude/longitude degrees without justification.
- [ ] Report non-convergence, weak geometry, and competing solutions explicitly.
- [ ] Add covariance or confidence information only where mathematically meaningful.

### Acceptance criteria

- Exact synthetic cases recover their known positions.
- Independent reference cases match within documented tolerances.
- Weighting affects solutions in the expected direction.
- Degenerate geometry never returns an unexplained plausible-looking fix.
- Normal coordinate/navigation installations and tests pass without the optional dependencies installed.

---

## Milestone 5 — Version 1.0 stabilization

**Goal:** stabilize a deliberately small API after real pre-1.0 use.

- [ ] Review and freeze top-level exports.
- [ ] Remove accidental public internals.
- [ ] Resolve known parser ambiguities with documented behavior.
- [ ] Review numerical tolerances against the reference corpus.
- [ ] Decide whether the optional fix API is mature enough for the 1.0 contract.
- [ ] Publish API stability and supported-Python policies.
- [ ] Confirm PyPI and conda-forge metadata agree.

Version 1.0 is ready when the documented API is coherent, tested from built artifacts, used in real examples, and small enough to maintain.

---

## Recommended coding-agent pull-request sequence

1. **Clean vertical slice:** replace the experiment with packaging, `Position`, decimal parsing/formatting, tests, and CI.
2. **Coordinate syntax:** normalization, DD/DDM/DMS, signs, hemispheres, and ambiguity.
3. **Machine formats:** ISO 6709, NMEA fields, structured values, and GeoJSON Point parsing.
4. **Formatting and inspection:** all output formats, precision behavior, and diagnostics.
5. **Geodesic dependency and API:** WGS84 inverse/direct calculations and references.
6. **Interchange and CLI:** GeoJSON collections plus `convert` and `inspect` commands.
7. **Release automation:** artifact testing, PyPI Trusted Publishing, and GitHub Release.
8. **Conda-forge submission:** staged-recipes after the stable PyPI release.
9. **Optional fix extra:** observation models, candidate geometry, solver, and diagnostics.

Each pull request should leave the package installable and demonstrably more useful. None should add historical compatibility code.
