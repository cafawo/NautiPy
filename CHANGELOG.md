# Changelog

All notable changes to NautiPy will be documented in this file.

## Unreleased

### Added

- Added an immutable, validated `Position` model.
- Added decimal-degree `parse_position` and `format_position` functions with
  explicit coordinate-order controls and actionable ambiguity errors.
- Added automatic DD, DDM, and DMS parsing with hemisphere prefixes/suffixes,
  direction words, Unicode symbols, decimal commas, colon and word-unit forms,
  explicit format selection, and the common `dmm` input alias.
- Added strict two-dimensional ISO 6709 and NMEA field-pair parsing.
- Added named latitude/longitude mappings and GeoJSON Point input.
- Added exact range validation before conversion to the internal float model.
- Added immutable `ParseResult` inspection metadata with normalized tokens,
  source-order evidence, lexical angular resolution, and candidate diagnostics.
- Added canonical DD, DDM, DMS, ISO 6709, and NMEA formatting with explicit
  precision, notation, symbols, output order, and supported separator options.
- Added resolution-aware `convert_position`, carry-safe round-half-even
  formatting, and negative-zero handling across every output format.
- Added immutable WGS84 inverse results, distance and bearing helpers,
  destination calculation, geodesic interpolation, and nearest-position
  lookup backed by GeographicLib.
- Added independently generated reference cases for short, antimeridian,
  high-latitude, and near-antipodal geodesics.
- Added keyword-only position identifiers and descriptions that do not affect
  coordinate equality or hashing.
- Added strict two-dimensional GeoJSON Point and FeatureCollection interchange
  with identifier and description preservation.
- Added deterministic `nautipy convert` and `nautipy inspect` commands backed
  by the public coordinate parser and formatter.
- Added tag-only release automation that validates versions and changelog
  notes, builds and tests artifacts once, and uses PyPI Trusted Publishing.
- Added optional bearing, range, and mixed-observation WGS84 position fixes
  with explicit candidate ambiguity, weighted residuals, convergence and
  geometry diagnostics, and conditional local uncertainty.
- Added dependency-free observation/result models and a clear missing-extra
  error for `python -m pip install "nautipy[fix]"`.
- Added standard-library tests, distribution build checks, clean-wheel smoke
  testing, and cross-platform GitHub Actions CI.

### Changed

- Replaced the experimental package with a clean PEP 517/PEP 621 `src` layout.
- Set the first public package version to `0.1.0`.
- Added GeographicLib 2.1 or newer as the sole runtime dependency while
  preserving lazy, standard-library-only coordinate use.
- Added NumPy and SciPy only to the isolated `fix` optional extra; normal
  coordinate and navigation installations remain free of the scientific stack.

### Removed

- Removed the unsupported experimental API, legacy packaging, checked-in
  distribution artifact, and placeholder test.
