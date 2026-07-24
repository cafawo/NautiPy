# Releasing NautiPy

This is the maintainer runbook for publishing NautiPy. Pull requests and
default-branch pushes validate the package but never publish it. An intentional
release tag starts `.github/workflows/release.yml`.

Maintainers need ordinary Python and Git locally. The GitHub CLI is not needed
for local preparation or tagging. The hosted release workflow uses its own
preinstalled `gh` command only to create the GitHub Release after PyPI
publication succeeds.

## Release contract

NautiPy uses a deliberately small version syntax:

- stable: `X.Y.Z`, tagged as `vX.Y.Z`;
- alpha: `X.Y.ZaN`, tagged as `vX.Y.ZaN`;
- beta: `X.Y.ZbN`, tagged as `vX.Y.ZbN`; and
- release candidate: `X.Y.ZrcN`, tagged as `vX.Y.ZrcN`.

Each numeric component and prerelease number is a non-negative integer without
unnecessary leading zeroes. Forms such as `v1.2`, `v1.2.3-rc.1`,
`v1.2.3.post1`, development versions, and local versions are not accepted by
the release validator.

For every release:

- `[project].version` in `pyproject.toml` is the version without `v`;
- the annotated Git tag is `v` plus that exact version;
- `CHANGELOG.md` contains exactly one nonempty
  `## X.Y.Z - YYYY-MM-DD` section for it;
- the tagged commit is the checked-out commit and is reachable from the
  repository's default branch;
- that version does not already exist on PyPI; and
- the wheel and source distribution are built once, tested, and then passed
  unchanged to PyPI and the GitHub Release.

The static project version, dated changelog section, and release tag are the
sources of truth. Do not replace artifacts, move a pushed release tag, or reuse
a version for different bytes.

Before `1.0.0`, documented breaking corrections belong in a minor release;
patch releases should remain compatible. Starting with `1.0.0`, documented
public APIs follow semantic versioning. The full compatibility policy is in
[SUPPORT.md](SUPPORT.md).

## One-time external setup

These settings live on PyPI and GitHub and cannot be established by a repository
commit:

1. Confirm that the normalized PyPI project name `nautipy` is controlled by the
   maintainers, or configure a pending Trusted Publisher for the first release.
2. Configure this exact Trusted Publisher identity:

   ```text
   PyPI project: nautipy
   GitHub owner: cafawo
   GitHub repository: NautiPy
   Workflow filename: release.yml
   Environment: pypi
   ```

3. Create a protected GitHub environment named `pypi`. Restrict deployment to
   release tags and require maintainer approval when appropriate.
4. Protect the default branch and require the always-running `CI success` check.
   Requiring only a dependent job is insufficient because it may be skipped
   after an upstream failure.
5. Allow the release workflow's narrowly scoped OIDC and GitHub Release
   permissions. Do not add a stored PyPI token.
6. Update the identity above if the repository owner, repository name, workflow
   filename, or environment ever changes.

TestPyPI is optional. Use it only for a specific check not already provided by
the clean artifact tests.

## Prepare the release pull request

Make release preparation a focused pull request:

1. Confirm that the target in [ROADMAP.md](../ROADMAP.md) is complete and the
   default branch is green.
2. Choose a version accepted by the syntax above and set it in
   `pyproject.toml`.
3. Move the reviewed entries from `## Unreleased` into one exact dated heading:

   ```markdown
   ## Unreleased

   ## 0.1.0 - YYYY-MM-DD

   ### Added

   - Release note.
   ```

   Replace `YYYY-MM-DD` with the release date. Always leave a new
   `## Unreleased` section at the top for subsequent work. Do not use brackets
   around the version heading.
4. Make README installation text publication-ready. The README is embedded in
   the wheel and source distribution and becomes the PyPI project description,
   so it cannot be corrected inside already-published artifacts. For the first
   release, replace the pre-release banner and checkout-only default with
   ordinary `pip install nautipy` instructions in the release commit. Do not
   defer that transition until after publication.
5. Run the normal source and package checks:

   ```bash
   python -m pip install -e .
   python -m pip check
   python -m unittest discover -s tests -v
   python -m pip install build twine
   python -m build
   python -m twine check dist/*
   ```

6. Smoke-test both locally built artifacts:

   ```bash
   python scripts/smoke_test_artifact.py dist/nautipy-X.Y.Z-py3-none-any.whl
   python scripts/smoke_test_artifact.py dist/nautipy-X.Y.Z.tar.gz
   ```

   Replace `X.Y.Z` with the project version. These local files are a rehearsal;
   the release workflow rebuilds once from the eventual tagged commit and
   publishes only its own tested artifacts.
7. Validate the version, changelog, and read-only PyPI availability check:

   ```bash
   python scripts/release.py validate vX.Y.Z --check-pypi
   ```

   Replace `vX.Y.Z` with the intended tag. `--check-git` is intentionally
   omitted until the annotated tag exists.
8. Merge only after review and the required `CI success` check passes.

## Create the release tag

Tag the merged default-branch commit, not a pull-request branch:

```bash
git switch master
git pull --ff-only origin master
git status --short
python scripts/release.py validate vX.Y.Z --check-pypi
git tag -a vX.Y.Z -m "NautiPy X.Y.Z"
python scripts/release.py validate vX.Y.Z --check-git --default-branch origin/master --check-pypi
git push origin vX.Y.Z
```

Replace `X.Y.Z` consistently. The second validation requires the annotated tag
to point at `HEAD` and confirms that the commit is reachable from
`origin/master`. `git status --short` must produce no output; do not create or
push the tag from a dirty working tree or if any check fails.

Pushing the tag starts and authorizes the publication workflow. Publication
occurs only when its gated PyPI job succeeds. A normal commit, pull request,
merge, or branch push cannot publish a package.

## What the release workflow does

The tag-triggered workflow performs these jobs in order:

1. **Validate release intent:** check the tag, static project version, dated
   changelog entry, annotated-tag ancestry, and absence of that version on
   PyPI.
2. **Test release source:** run the complete suite on the oldest and newest
   supported Python versions. Normal CI has already tested every supported
   Python version, exact minimum dependencies, and cross-platform smoke cases.
3. **Build release artifacts once:** create the wheel and source distribution,
   run Twine checks, record SHA-256 hashes, and extract the matching changelog
   section as release notes.
4. **Test exact artifacts:** download the build output, verify its hashes,
   install the wheel and source distribution separately in clean environments,
   exercise representative coordinate, navigation, GeoJSON, CLI, and fixing
   workflows and the intended top-level exports, and run `pip check`.
5. **Publish tested artifacts:** use PyPI Trusted Publishing and short-lived
   OpenID Connect identity. This job does not rebuild or use `skip-existing`.
6. **Create the GitHub Release:** attach the same wheel, source distribution,
   and checksum file and use the reviewed changelog section as release notes.

Publishing jobs cannot run unless artifact tests succeed. Third-party actions
are pinned to immutable commit SHAs, with Dependabot responsible for proposing
updates.

## Verify a successful release

After the workflow completes:

1. Confirm that the PyPI version, project metadata, dependency declarations,
   README, wheel, and source distribution are present.
2. Install `nautipy==X.Y.Z` in a new environment using ordinary `pip`, run
   `pip check`, and exercise coordinate, navigation, and fixing imports.
3. Confirm that the GitHub Release uses the same tag and contains the wheel,
   source distribution, and `SHA256SUMS`.
4. Compare the published artifact hashes with the workflow's checksum file.
5. Confirm that PyPI displays the publication-ready README embedded in the
   tagged artifacts. A later documentation commit does not change that
   published description.
6. Update the roadmap release status while retaining the new `Unreleased`
   changelog section.

## Failed releases

Before pushing a tag, fix the release pull request and repeat the checks.

After a tag has been pushed but before PyPI publication:

- rerun an unchanged tagged workflow after correcting a transient service or
  external-configuration problem; or
- if source, metadata, or workflow code must change, prepare a new release
  commit and version rather than moving or reusing the pushed tag.

After PyPI publication:

- never replace files under the same version;
- rerun only the failed downstream GitHub Release work when the published
  artifacts are sound;
- yank a broken PyPI release when appropriate;
- publish a corrected patch release; and
- follow conda-forge's process for any already-published downstream build.

The workflow must fail on mismatched versions, duplicate uploads, unexpected
artifacts, checksum mismatches, or failed artifact tests.

## Conda-forge follows PyPI

Conda-forge distribution is maintained outside this repository. NautiPy's
release workflow must not upload directly to it.

After a stable release is available on PyPI:

1. Fork `conda-forge/staged-recipes`.
2. Add a recipe using the exact published PyPI source distribution and
   checksum.
3. Declare the package's Python constraint and GeographicLib, NumPy, and SciPy
   runtime dependencies.
4. Use `noarch: python` only when the current conda-forge rules permit it.
5. Add import and small coordinate, navigation, and fixing tests through the
   top-level API.
6. Submit the recipe for review.

After acceptance, conda-forge creates `nautipy-feedstock`; its bot normally
proposes later version updates. Feedstock maintainers review dependency
changes, migrations, CI, and bot failures before merging.

Official references:

- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [Adding a PyPI publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
- [Using a PyPI publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
- [Adding packages to conda-forge](https://conda-forge.org/docs/maintainer/adding_pkgs/)
