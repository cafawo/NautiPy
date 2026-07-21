# Changelog

All notable changes to NautiPy will be documented in this file.

## Unreleased

### Added

- Added an immutable, validated `Position` model.
- Added decimal-degree `parse_position` and `format_position` functions with
  explicit coordinate-order controls and actionable ambiguity errors.
- Added standard-library tests, distribution build checks, clean-wheel smoke
  testing, and cross-platform GitHub Actions CI.

### Changed

- Replaced the experimental package with a clean PEP 517/PEP 621 `src` layout.
- Set the first public package version to `0.1.0` with no runtime dependencies.

### Removed

- Removed the unsupported experimental API, legacy packaging, checked-in
  distribution artifact, and placeholder test.
