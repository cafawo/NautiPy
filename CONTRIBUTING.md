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

## Plan the documentation impact

Public functionality and its documentation are one change. Before
implementation, identify both the authoritative specification and the
educational page that will need to change. Public functionality includes
accepted inputs, outputs, exceptions, units, defaults, diagnostics, public
names, CLI and GeoJSON behavior, dependency expectations, and documented
limitations.

The main routes are:

- coordinates and positions: `docs/COORDINATES.md` and
  `website/content/learn/coordinates.md`;
- WGS84 navigation: `docs/NAVIGATION.md` and
  `website/content/learn/navigation.md`;
- observation conventions and candidate geometry: `docs/FIXES.md` and
  `website/content/learn/finding-the-boat.md`;
- solver diagnostics, weighting, and uncertainty: `docs/FIXES.md` and
  `website/content/learn/trusting-a-fix.md`;
- scenario semantics or teaching fixtures: `docs/FIXES.md` and the Fix Lab
  page, `website/tools/generate_fix_lab.py`, and committed data;
- GeoJSON: `docs/GEOJSON.md`, plus
  `website/content/learn/coordinates.md` for order concepts or
  `website/content/practical-use.md` for interchange recipes;
- CLI behavior: the relevant specification, README example, and Practical
  Use; and
- installation, dependencies, API support, architecture, or scope: the
  applicable project metadata, `docs/ARCHITECTURE.md`, `docs/SUPPORT.md`,
  `docs/PRODUCT.md`, behavior specification, README, and affected home, How
  NautiPy Works, or Practical Use page.

Update examples, diagrams, glossary entries, and generated Fix Lab fixtures
when their lesson changes. Record user-visible release changes in
`CHANGELOG.md`.

A private refactor with identical observable behavior does not need a
meaningless documentation edit. Its pull request must instead say
`Documentation impact: none` and explain why users cannot observe the change.

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

## Work on the educational website

The public learning site is maintained under `website/` and deployed through
GitHub Pages. Start from an editable NautiPy installation, then install its
separate, pinned documentation toolchain:

```console
python -m pip install --requirement website/requirements.txt
python website/tools/generate_fix_lab.py --check
python -m unittest discover -s website/tests -v
python -m mkdocs build --clean --strict --config-file website/mkdocs.yml
```

The fixture check recomputes the Fix Lab scenarios through NautiPy's public API
and fails if the committed data is stale. The site tests check examples,
internal links, figures, and generated scenarios. The strict build checks
navigation and site configuration. For a local preview, run:

```console
python -m mkdocs serve --config-file website/mkdocs.yml
```

Generated site output is written to the ignored `website/site/` directory.
Everything below `website/` is excluded from the wheel and source distribution,
so site figures and interactive assets do not increase a `pip` installation.
Documentation dependencies belong only in `website/requirements.txt`; do not
add them to the project's runtime dependencies.

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
- the exact behavior specification and educational pages updated, or a
  concrete explanation of why documentation is unaffected;
- numerical assumptions, units, and tolerances; and
- known limitations.

Do not merge a public functionality change with only tests or a specification
updated while the educational website remains stale. Keep third-party objects
behind NautiPy result types, use descriptive exceptions for caller errors, and
never silently change coordinate order, Earth model, or units. Before version
1.0, prefer a clean correction over a compatibility layer for behavior that
has never been publicly released.

Pull requests never publish packages. Maintainers create intentional release
tags according to the [release procedure](docs/RELEASING.md).

## Community and sensitive reports

Follow the [Code of Conduct](CODE_OF_CONDUCT.md) in all project spaces.

Do not put credentials, private position data, or a suspected vulnerability in
a public issue. Use the private process in [SECURITY.md](SECURITY.md) instead.
