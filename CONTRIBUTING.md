# Contributing to NautiPy

Thank you for helping make coordinate and small-scale navigation work easier,
safer, and more understandable.

This is the human contributor guide. Coding agents should also follow
[AGENTS.md](AGENTS.md).

## Set up a development checkout

Use Git and a Python version accepted by the
[`requires-python` setting](pyproject.toml).

If you do not have write access to the main repository, fork it on GitHub
first and use your fork's URL in the `git clone` command below.

On macOS or Linux:

```console
git clone https://github.com/cafawo/NautiPy.git
cd NautiPy
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
python -m unittest discover -s tests -v
```

On Windows PowerShell:

```console
git clone https://github.com/cafawo/NautiPy.git
cd NautiPy
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
python -m unittest discover -s tests -v
```

The `python3.12` and `py -3.12` commands are examples; another supported
interpreter works as well. After activation, `python` refers to the virtual
environment. The editable install includes GeographicLib, NumPy, and SciPy and
enables the complete test suite.

No particular environment manager, editor, shell, or operating system is
required. The `python -m ...` commands above are the portable reference once
the environment is active.

Create a focused branch, make and verify the change, push that branch, and open
a pull request against NautiPy. These steps work through VS Code's Git
interface or ordinary Git commands; the GitHub CLI is not required. The
[issue tracker](https://github.com/cafawo/NautiPy/issues) is the best place to
find or discuss scoped work.

## Choose a focused change

Start with a clearly scoped issue or a reproducible bug. The
[roadmap](ROADMAP.md) provides direction, but its first-release checklist
contains maintainer operations rather than general contribution tasks. A
contribution should strengthen one of these workflows:

1. parse, inspect, or convert coordinates;
2. calculate WGS84 navigation values;
3. exchange positions through GeoJSON or the command line; or
4. estimate a position from bearings and ranges.

Discuss features outside the documented
[product direction](docs/PRODUCT.md) before implementing them. Keep each pull
request focused on one working behavior rather than combining unrelated
cleanup.

Read the specification related to your change:

- [Coordinates](docs/COORDINATES.md)
- [Navigation](docs/NAVIGATION.md)
- [Position fixes](docs/FIXES.md)
- [GeoJSON](docs/GEOJSON.md)
- [Architecture and dependencies](docs/ARCHITECTURE.md)
- [Support and public API](docs/SUPPORT.md)

## Add meaningful tests

Use standard-library `unittest` unless a concrete need justifies another test
dependency. Exercise public behavior and error cases, not only private
implementation details.

Coordinate changes should cover relevant malformed, ambiguous, boundary, and
round-trip cases. Numerical changes should include an independent reference,
difficult or degenerate geometry, justified tolerances, and explicit failure
behavior. Tests must not depend on network services, changing datasets,
wall-clock dates, locale settings, or unseeded randomness.

Run the quick checks whenever you change code:

```console
python -m pip check
python -m unittest discover -s tests -v
```

## Check distributions before review

Before requesting review, build and test the installable artifacts. These
checks are especially important for packaging, dependency, import-boundary,
and release-preparation changes:

```console
python -m pip install build twine
python -m build
python -m twine check dist/*
python scripts/smoke_test_artifact.py dist/nautipy-VERSION-py3-none-any.whl
python scripts/smoke_test_artifact.py dist/nautipy-VERSION.tar.gz
```

Replace `VERSION` with the version shown in the filenames produced by
`python -m build`. Each smoke command creates a clean temporary environment
and downloads the artifact's declared dependencies, so it requires network
access.

Use the quick checks during development and the full artifact checks before
review.

## Prepare the pull request

A useful pull request explains:

- the user problem and observable behavior;
- examples or independent reference data;
- tests and commands run;
- public API or runtime dependency effects;
- numerical assumptions, units, and tolerances; and
- known limitations.

Update public examples and documentation whenever behavior changes. Keep
third-party objects behind NautiPy result types, use descriptive exceptions
for caller errors, and never silently change coordinate order, Earth model, or
units. Before version 1.0, prefer a clean correction over a compatibility layer
for behavior that has never been publicly released.

Pull requests never publish packages. Maintainers create intentional release
tags according to the [release procedure](docs/RELEASING.md).

## Community and sensitive reports

Follow the [Code of Conduct](CODE_OF_CONDUCT.md) in all project spaces.

Do not put credentials, private position data, or a suspected vulnerability in
a public issue. Use the private process in [SECURITY.md](SECURITY.md) instead.
