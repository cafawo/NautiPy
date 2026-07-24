# Support and API stability

This document defines the compatibility contract for published NautiPy
releases. Support begins with the first public release. Repository snapshots
and development branches are not supported releases, even when their static
metadata contains an upcoming version number.

## Public surface

The supported API consists of:

- names listed in the package and public-module `__all__` values;
- documented constructors, functions, dataclass fields, enums, exceptions,
  command-line options, and interchange formats; and
- behavior explicitly specified in the user documentation.

Names beginning with an underscore are private. Imported implementation names,
optimizer objects, third-party backend details, exact exception wording, and
undocumented serialization details are not compatibility promises. Public
exception types and documented diagnostic fields are part of the API even
when their full message text is not.

The common coordinate, navigation, and fixing API is available from
`nautipy`. Public submodules organize related implementations and remain
supported where they define documented `__all__` exports, but users do not
need a special module path to access fixing. `nautipy.cli` and
`nautipy.__main__` implement the documented command-line interface, but their
Python-level names are not public API. Modules whose final component begins
with an underscore are private.

## Versioning

After the first public release and before `1.0.0`, minor releases may make
breaking corrections when they
materially improve correctness or simplify the public API. Such changes must
be called out in the changelog and release notes. Patch releases should not
intentionally break documented behavior.

Starting with `1.0.0`, documented APIs follow semantic versioning. Compatible
features are added in minor releases, compatible fixes are made in patch
releases, and intentional removals or incompatible API changes require a major
release. When practical, a planned removal is deprecated for at least one
minor release first. Urgent security or correctness fixes may bypass a normal
deprecation period, but must be explicit in the release notes.

## Supported releases

Once releases exist, NautiPy maintains only its newest published release.
Older releases remain available but do not receive routine fixes or backports;
users may need to upgrade before a problem can be addressed. Security
advisories identify affected versions and any exceptional backport explicitly.

## Python versions and platforms

The authoritative Python range is the `requires-python` value in
`pyproject.toml`. Classifiers, dependency constraints, CI, built metadata, and
conda-forge metadata must agree with it. Prose documentation deliberately does
not duplicate the range.

Every supported Python version receives the complete test suite with normally
resolved dependencies. A separate oldest-Python job tests the exact declared
minimum GeographicLib, NumPy, and SciPy versions. Built-artifact checks run on
the newest supported Python. Linux, macOS, and Windows receive representative
plain-install smoke coverage of the top-level coordinate, navigation, and
fixing API.

Adding or dropping a Python version is a minor-release change and is never
hidden in a patch release. A planned removal should be announced one minor
release in advance when practical and must appear in the changelog. NautiPy
does not claim support for a Python version or platform that its release CI
cannot exercise.

## Reporting problems

Compatibility, installation, parsing, and numerical issues belong in the
project issue tracker. Numerical reports should include the inputs, units,
expected reference, observed result, NautiPy version, Python version, and
dependency versions needed to reproduce the result.
