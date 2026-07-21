# Release and distribution plan

## Objectives

NautiPy should publish installable artifacts through a small, secure, repeatable process:

- pull requests prove that the package tests and builds;
- an intentional semantic-version tag starts a release;
- the exact tested artifacts are published to PyPI;
- GitHub supplies short-lived publishing identity instead of a stored PyPI token; and
- conda-forge packages follow the normal staged-recipes and feedstock workflow.

Release automation must reduce repetitive work without making version decisions or breaking changes automatic.

## Release cadence and versioning

Use semantic versioning:

- **patch** (`X.Y.Z`): compatible fixes and documentation corrections;
- **minor** (`X.Y.0`): compatible features;
- **major** (`X.0.0`): intentional breaking changes after version 1.0.

Before 1.0, document breaking changes in release notes and use minor releases for them.

### Publish every intentional release tag

Do not restrict deployment to new major versions. Patch and minor releases are how users receive fixes and features. The release workflow should accept tags shaped like:

```text
v0.2.0
v0.2.1
v1.0.0
v2.0.0rc1
```

GitHub tag filters are not a semantic-version validator. A workflow may trigger on `v*`, but an early validation step must reject tags that are not valid, supported version strings.

A merge to the default branch must never publish a package by itself.

## Sources of truth

- Package metadata and the release version live in `pyproject.toml`.
- Human-facing changes live in `CHANGELOG.md`.
- The Git tag is `v` plus the exact package version.
- Built distributions are generated from the tagged commit.

A release fails if these disagree. Avoid adding a dynamic version plugin until there is a demonstrated maintenance benefit; a small explicit version check is easier to understand and recover.

## GitHub Actions design

Implement two workflows rather than one workflow with many unrelated triggers.

### Continuous integration: `.github/workflows/ci.yml`

Triggers:

- `pull_request`;
- pushes to the default branch;
- optional manual dispatch for diagnostics.

Responsibilities:

1. Install the package and test dependencies using ordinary `pip` commands.
2. Run the canonical test suite against the Python versions declared by project policy.
3. Include at least lower-bound and newest-supported Python coverage.
4. Include lightweight smoke coverage on Linux, macOS, and Windows where practical.
5. Build both sdist and wheel.
6. Inspect package metadata using a standard PyPA tool.
7. Install the built wheel into a clean environment and run import/core smoke tests.
8. Upload build artifacts for inspection, but never publish them.

The workflow should test supported behavior rather than a large matrix assembled for appearance. Keep the matrix aligned with `requires-python` and dependency wheel availability.

### Release: `.github/workflows/release.yml`

Trigger:

```yaml
on:
  push:
    tags:
      - "v*"
```

An alternative `release: published` trigger is valid, but tag-triggering keeps the source commit and package version relationship direct. Choose one model and document it; do not support both unless idempotency is designed and tested.

Recommended job flow:

1. **validate**
   - validate semantic-version syntax;
   - ensure tag `vX.Y.Z` matches project version `X.Y.Z`;
   - ensure the changelog contains the version;
   - ensure the tag points to an allowed release commit.
2. **test**
   - run the release-critical test suite on the tagged commit.
3. **build**
   - build sdist and wheel exactly once;
   - record hashes;
   - upload them as a workflow artifact.
4. **smoke-test-artifacts**
   - download the artifacts;
   - install the wheel in a fresh environment;
   - run import and core API smoke tests;
   - optionally build/install from the sdist as a second check.
5. **publish-pypi**
   - download the already tested artifacts;
   - publish only those files through PyPI Trusted Publishing;
   - do not rebuild in the publish job.
6. **github-release**
   - create the GitHub Release for the tag;
   - attach the same sdist and wheel;
   - use the matching changelog section as release notes.

The publish and GitHub Release jobs should depend on successful artifact testing.

## PyPI authentication

Use PyPI Trusted Publishing with GitHub Actions and OpenID Connect. Do not store a long-lived PyPI API token in repository secrets.

The publish job should have:

```yaml
environment:
  name: pypi
permissions:
  id-token: write
```

Keep `id-token: write` at the publish job, not the entire workflow. Other jobs should use read-only permissions unless they need a narrowly scoped write permission.

Use the PyPA publishing action in a dedicated publish job. Do not invoke it from a local composite action or rebuild distributions in that job.

Official references:

- PyPI Trusted Publishing: https://docs.pypi.org/trusted-publishers/
- Adding a GitHub publisher: https://docs.pypi.org/trusted-publishers/adding-a-publisher/
- Publishing with a trusted publisher: https://docs.pypi.org/trusted-publishers/using-a-publisher/
- GitHub Python build/publish guidance: https://docs.github.com/en/actions/how-tos/use-cases-and-examples/building-and-testing/building-and-testing-python

## One-time PyPI setup

Before the first public release:

1. Confirm that the normalized project name `nautipy` is available or already controlled by the maintainers. Search results alone are not ownership confirmation.
2. Create or reserve the PyPI project using PyPI's current trusted-publisher process.
3. Configure the GitHub owner, repository, exact workflow filename, and environment name in PyPI.
4. Create a GitHub environment named `pypi`.
5. Restrict that environment to release tags and, if desired, require maintainer approval.
6. Protect the default branch and require CI before merge.
7. Test package building locally and in CI before creating a real tag.

Use TestPyPI only if it provides a concrete validation benefit. The production workflow must not rely on `skip-existing`; duplicate versions should fail loudly.

## Action and workflow maintenance

- Pin third-party GitHub Actions to immutable commit SHAs for reproducibility and supply-chain safety.
- Configure Dependabot or an equivalent GitHub-native updater to propose action-SHA updates.
- Prefer official GitHub and PyPA actions with narrow purposes.
- Keep build and publish jobs separate.
- Set explicit minimal permissions at workflow or job level.
- Never run untrusted pull-request code in a job with publishing identity or write credentials.
- Do not use self-hosted runners for public pull requests unless they are isolated for that purpose.

The workflow files should contain concise comments explaining security-sensitive permissions, not copies of this entire document.

## Human release procedure

Automation begins after a maintainer decides to release.

1. Ensure the milestone is complete and CI passes on the default branch.
2. Update `CHANGELOG.md`: move relevant entries from `Unreleased` into a dated version section.
3. Update the version in `pyproject.toml`.
4. Open and merge a release-preparation pull request.
5. Create an annotated tag `vX.Y.Z` on the release commit and push it.
6. Approve the protected `pypi` environment deployment if approval is enabled.
7. Verify the PyPI page, installation command, project links, wheel, and sdist.
8. Verify the GitHub Release and artifact hashes.
9. Review the resulting conda-forge bot PR when a feedstock exists.

A major release additionally requires a migration guide and explicit maintainer review of breaking changes. The release workflow itself does not need a separate major-version path.

## Failed releases and immutability

PyPI and conda-forge artifacts are effectively immutable. Never attempt to replace a file under an existing version.

When a release is broken:

- stop or cancel the workflow before publishing when possible;
- after PyPI publication, yank the affected release when appropriate and publish a corrected patch release;
- never reuse the tag or version for different bytes;
- after conda-forge publication, follow conda-forge's current process for marking builds broken and submit a corrected build or version.

Release workflows must fail on version mismatch, duplicate upload, missing artifacts, or failed smoke tests.

## Pre-releases

PEP 440 pre-releases such as `1.0.0rc1` may be published through the same tag workflow.

- Clearly mark them as pre-releases in the changelog and GitHub Release.
- Do not submit a conda-forge stable recipe for a release candidate unless there is a specific testing need and the feedstock policy supports it.
- Do not make pre-release tags a separate code path unless behavior truly differs.

## Conda-forge distribution

Conda-forge is not normally published directly from this repository. Its community workflow creates and maintains a separate feedstock.

### Initial submission

After a stable PyPI release exists:

1. Fork `conda-forge/staged-recipes`.
2. Add a recipe using the format and template currently accepted by staged-recipes.
3. Build from the PyPI sdist for the exact released version.
4. Include the sdist checksum, license file, runtime dependencies, Python constraint, homepage, and source repository.
5. Mark the package `noarch: python` when the NautiPy distribution contains no compiled platform-specific code and the current conda-forge rules permit it.
6. Add import and small functional tests, including coordinate parsing.
7. Open a staged-recipes pull request and address review/CI feedback.

Do not copy conda-forge's generated CI machinery into the NautiPy repository. Keep any initial recipe draft in a temporary branch or issue if useful, then maintain the accepted recipe in the feedstock.

Official references:

- Adding packages: https://conda-forge.org/docs/maintainer/adding_pkgs/
- Staged-recipes lifecycle: https://conda-forge.org/docs/maintainer/understanding_conda_forge/staged_recipes/
- Maintaining packages: https://conda-forge.org/docs/maintainer/updating_pkgs/

### Subsequent releases

Once the recipe is accepted, conda-forge creates `nautipy-feedstock` and its build infrastructure. The conda-forge version-update bot normally detects new PyPI or GitHub releases and opens feedstock pull requests.

Feedstock maintainers must still:

- review dependency and metadata changes;
- verify CI results;
- merge the update PR;
- handle bot failures or upstream packaging changes; and
- respond to migrations for new Python or dependency versions.

This is intentionally not an automatic upload from the NautiPy release workflow. Conda-forge review and feedstock automation are part of its trust and maintenance model.

## What should remain manual

Do not automate:

- deciding that a version is ready;
- choosing major/minor/patch impact without review;
- approving breaking API changes;
- generating release notes without maintainer review;
- merging conda-forge feedstock updates blindly; or
- deleting/yanking a release based only on a failing downstream report.

The durable division of responsibility is:

- **humans choose and approve a release;**
- **GitHub Actions validates, builds, and publishes exact artifacts;**
- **conda-forge automation proposes and builds its distribution;**
- **feedstock maintainers review and merge it.**
