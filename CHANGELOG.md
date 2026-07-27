# Changelog

Notable user-facing changes to NautiPy are recorded here.

## Unreleased

### Added

- Added an education-first GitHub Pages site with original diagrams, a
  verified-data Fix Lab, scientific references, and automated deployment,
  while keeping every website asset out of PyPI distributions.

## 0.1.0 - 2026-07-27

### Added

- Added an immutable, validated `Position` model and actionable coordinate
  exception hierarchy.
- Added automatic DD, DDM, DMS, ISO 6709, and NMEA field-pair parsing,
  including common hemisphere, Unicode, decimal-comma, mapping, and GeoJSON
  Point inputs.
- Added explicit coordinate-order and format controls so ambiguous inputs fail
  instead of being guessed.
- Added coordinate inspection metadata and canonical DD, DDM, DMS, ISO 6709,
  and NMEA formatting and conversion.
- Added WGS84 distance, bearing, destination, interpolation, and
  nearest-position calculations backed by GeographicLib.
- Added strict GeoJSON Point and FeatureCollection interchange with supported
  identifier and description preservation.
- Added deterministic `nautipy convert` and `nautipy inspect` commands.
- Added bearing-only, range-only, and mixed-observation WGS84 position fixes
  with candidate ambiguity, weighting, residuals, convergence and geometry
  diagnostics, and conditional local uncertainty.
- Added the common coordinate, navigation, and fixing API to the top-level
  `nautipy` namespace.
- Added supported-Python CI, exact-minimum dependency tests, cross-platform
  smoke checks, clean wheel and source-distribution tests, and tag-only release
  automation.

### Changed

- Established a typed PEP 517/PEP 621 package using a `src` layout.
- Made GeographicLib, NumPy, and SciPy normal runtime dependencies so one
  installation provides every shipped feature.
- Kept coordinate parsing, formatting, GeoJSON, and CLI plumbing in the
  standard library, with lazy boundaries that avoid importing geodesic or
  scientific implementations during coordinate-only use.
- Reorganized project documentation around the implemented 0.1 baseline, a
  concise contributor path, and the actual maintainer release workflow.
- Defined PyPI as the only maintained package index and documented a direct
  sole-maintainer release path.

### Fixed

- Preserved valid noisy fixes on a circular search boundary without projecting
  out-of-domain optima into the search region.
- Hardened fix residual, result, covariance, and confidence-ellipse validation
  against overflow and floating-point cancellation.
