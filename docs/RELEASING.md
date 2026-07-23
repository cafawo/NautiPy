# Release and distribution plan

## Objectives

NautiPy should publish through a small, intentional, repeatable process:

- pull requests prove that source and built artifacts work;
- a maintainer chooses a version and creates a semantic-version tag;
- GitHub Actions builds the wheel and sdist once;
- the exact tested artifacts are published to PyPI;
- authentication uses short-lived OpenID Connect identity rather than a stored PyPI token; and
- conda-forge distribution follows staged-recipes and the generated feedstock.

Automation performs a release; it does not decide that a release is ready.

## Initial version

The first supported public package is `0.1.0`.

The older repository experiment was not a supported distribution and does not
require version continuity, compatibility wrappers, or migration notes. From
`0.1.0` until `1.0.0`, clearly documented breaking API improvements belong in
minor releases; patch releases should remain compatible. At and after `1.0.0`,
documented APIs follow semantic versioning.

The compatibility and supported-Python rules are defined in
[SUPPORT.md](SUPPORT.md). The Python range itself remains authoritative only
in `pyproject.toml`.

## Publish every intentional release tag

Do not deploy only major versions. Patch and minor releases carry fixes and features.

Accepted examples:

```text
v0.1.0
v0.1.1
v0.2.0
v1.0.0
v2.0.0rc1
```

A workflow may trigger on `v*`, but an early validation job must reject invalid or mismatched versions. A merge to the default branch never publishes by itself.

## Sources of truth

- Package version and metadata: `pyproject.toml`.
- Human-facing release notes: `CHANGELOG.md`.
- Release tag: `v` plus the exact project version.
- Distributions: generated from the tagged commit.

The release fails if these disagree. Keep a static version in `pyproject.toml` until dynamic versioning demonstrably removes more complexity than it adds.
Each release uses one exact, dated, nonempty changelog heading such as
`## 0.1.0 - 2026-07-22`; the text below that heading becomes the GitHub
Release notes.

## Continuous integration

Create `.github/workflows/ci.yml` for:

- pull requests;
- pushes to the default branch; and
- optional manual diagnostic runs.

CI responsibilities:

1. Install the package with ordinary `pip` commands.
2. Run `python -m unittest discover -s tests -v`.
3. Test the oldest and newest stable Python versions declared in `requires-python`.
4. Add small Linux, macOS, and Windows smoke coverage where practical.
5. Build wheel and sdist.
6. Inspect package metadata with an appropriate PyPA tool.
7. Install the wheel in a clean environment outside the checkout.
8. Run import and core-function smoke tests against that wheel.
9. Upload artifacts for inspection without publishing them.

Keep the matrix aligned with the package's actual Python support. Testing unsupported or pre-release Python versions may be informative but must not block releases unless project policy says so.

## Release workflow

Create `.github/workflows/release.yml` triggered by tags:

```yaml
on:
  push:
    tags:
      - "v*"
```

Recommended jobs:

### 1. Validate

- validate the version as an accepted PEP 440 release;
- ensure the tag and `pyproject.toml` version match;
- ensure `CHANGELOG.md` contains the version;
- ensure the tagged commit is eligible for release; and
- fail on a version that already exists.

### 2. Test

Run the release-critical test suite on the tagged source.

### 3. Build

- build wheel and sdist exactly once;
- record their hashes; and
- upload them as a workflow artifact.

### 4. Test artifacts

- download the build artifact;
- install the wheel in a fresh environment;
- run import plus coordinate/navigation smoke tests;
- optionally build and install from the sdist; and
- fail before publication on any error.

### 5. Publish to PyPI

- download the already tested artifact;
- publish only those files;
- use PyPI Trusted Publishing/OIDC;
- do not rebuild; and
- do not use `skip-existing` to hide duplicate-version mistakes.

### 6. Create GitHub Release

- create the release for the same tag;
- attach the same wheel and sdist; and
- use the matching changelog section as reviewed release notes.

Publishing jobs depend on successful artifact tests.

## PyPI authentication

Use a protected GitHub environment named `pypi` and limit identity permission to the publish job:

```yaml
environment:
  name: pypi
permissions:
  id-token: write
```

Other jobs should use read-only permissions unless a narrowly scoped write is required. Never run untrusted pull-request code in a job with publishing identity.

Use the official PyPA publishing action in a dedicated job. Pin third-party actions to immutable commit SHAs and configure Dependabot to propose SHA updates.

Official references:

- https://docs.pypi.org/trusted-publishers/
- https://docs.pypi.org/trusted-publishers/adding-a-publisher/
- https://docs.pypi.org/trusted-publishers/using-a-publisher/
- https://docs.github.com/en/actions/how-tos/use-cases-and-examples/building-and-testing/building-and-testing-python

## One-time PyPI setup

Before `v0.1.0`:

1. Confirm that the normalized project name `nautipy` is available or controlled by the maintainers. Search results alone do not prove ownership.
2. Configure the PyPI project and pending trusted publisher using the current PyPI process.
3. Match the exact GitHub owner, repository, workflow filename, and environment.
4. Create the protected GitHub `pypi` environment.
5. Restrict deployment to release tags and optionally require maintainer approval.
6. Protect the default branch and require the always-running `CI success`
   aggregate check before merge. Do not require only a dependent job that can
   be skipped when an upstream check fails.
7. Verify package links, license metadata, wheel contents, and sdist contents before tagging.

The pending PyPI trusted-publisher identity must use these exact values:

```text
PyPI project: nautipy
GitHub owner: cafawo
GitHub repository: NautiPy
Workflow filename: release.yml
Environment: pypi
```

The identity contains no token or secret. PyPI and GitHub still require the
maintainer to create and protect their respective project/environment state.

TestPyPI is optional and should be used only when it validates something the production artifact tests do not.

## Human release procedure

1. Confirm the target roadmap milestone and CI are complete.
2. Move changes from `Unreleased` into a dated `CHANGELOG.md` section.
3. Set the version in `pyproject.toml`.
4. Open and merge a focused release-preparation pull request.
5. Create and push an annotated `vX.Y.Z` tag on that commit.
6. Approve the protected `pypi` deployment if required.
7. Verify the PyPI page and installation in a clean environment.
8. Verify the GitHub Release and artifact hashes.
9. Review the conda-forge update PR when a feedstock exists.

Do not reuse a version or tag for different bytes.

## Failed releases

Before publication, cancel and fix the workflow.

After publication:

- never replace files under the same version;
- yank a broken PyPI release when appropriate;
- publish a corrected patch release; and
- follow conda-forge's current process for broken builds after conda publication.

The workflow must fail on version mismatch, duplicate upload, missing artifacts, or failed artifact tests.

## Conda-forge

Conda-forge is maintained outside this repository through its community feedstock process. Do not add a direct conda-forge upload job to NautiPy's release workflow.

### Initial submission

After stable `0.1.0` is on PyPI:

1. Fork `conda-forge/staged-recipes`.
2. Add a recipe using the currently accepted staged-recipes format.
3. Build from the exact PyPI sdist.
4. Include the sdist checksum, license file, runtime dependencies, Python constraint, homepage, and source repository.
5. Use `noarch: python` when the distribution is pure Python and current conda-forge rules permit it.
6. Add import and small coordinate/navigation functional tests.
7. Open a staged-recipes pull request and address review feedback.

Official references:

- https://conda-forge.org/docs/maintainer/adding_pkgs/
- https://conda-forge.org/docs/maintainer/understanding_conda_forge/staged_recipes/
- https://conda-forge.org/docs/maintainer/updating_pkgs/

### Subsequent releases

After acceptance, conda-forge creates `nautipy-feedstock`. Its update bot normally proposes new versions after upstream releases.

Feedstock maintainers still review dependency changes, CI, migrations, and bot failures before merging. This review is intentional and should not be bypassed by an upstream upload workflow.

## What remains manual

Do not automate:

- deciding that the package is ready;
- choosing version impact without review;
- approving breaking API changes;
- merging release preparation;
- accepting generated release notes without review;
- blindly merging feedstock updates; or
- yanking releases based only on an automated downstream report.

The durable division is:

- humans choose and approve releases;
- GitHub Actions validates, builds, tests, and publishes exact artifacts;
- conda-forge automation proposes and builds its distribution; and
- feedstock maintainers review and merge it.
