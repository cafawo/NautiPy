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

1. In the PyPI account settings, verify the maintainer email address, enable
   two-factor authentication, and securely store recovery codes.
2. In the GitHub repository, open **Settings → Environments**, create an
   environment named exactly `pypi`, and select **Selected branches and tags**
   under its deployment rules. Add a tag rule for `v*`. A required maintainer
   reviewer is recommended when the repository and account plan support it.
   A sole maintainer must not enable **Prevent self-review** unless another
   eligible reviewer is available.
   Do not add a PyPI password, API token, environment secret, or variable.
3. If the `nautipy` project does not exist on PyPI yet, open **Publishing**
   from the PyPI account sidebar and add a pending **GitHub Actions** publisher
   with this exact identity:

   ```text
   PyPI project: nautipy
   GitHub owner: cafawo
   GitHub repository: NautiPy
   Workflow filename: release.yml
   Environment: pypi
   ```

   Enter only the workflow filename, not `.github/workflows/release.yml`.
   Names are case-sensitive and must match the repository and workflow. A
   pending publisher creates the PyPI project on its first successful upload;
   it does not reserve the project name beforehand. If the project already
   exists and is controlled by the maintainers, configure the same identity
   from that project's **Manage → Publishing** page instead.
4. Keep CI enabled for pull requests and default-branch pushes. If branch
   protection is used, require the always-running `CI success` check. Branch
   protection is optional for a sole-maintainer repository.
5. Under **Settings → Actions → General**, ensure Actions are enabled and, if
   an action allowlist is used, permit the pinned `actions/*` actions and
   `pypa/gh-action-pypi-publish`. Keep the repository's default workflow
   permissions read-only; the workflow grants only job-scoped permissions:
   `id-token: write` for PyPI and `contents: write` for the GitHub Release. Do
   not add a stored PyPI token.
6. Update the identity above if the repository owner, repository name, workflow
   filename, or environment ever changes.

TestPyPI is optional. Use it only for a specific check not already provided by
the clean artifact tests.

## One-time documentation hosting setup

GitHub Pages is deployed by the normal CI workflow, independently of package
releases. In the repository's **Settings → Pages → Build and deployment**,
select **GitHub Actions** as the source. The workflow's restricted
`GITHUB_TOKEN` deliberately cannot make this repository-level setting.

The `wbk.ing` custom domain belongs to the `cafawo.github.io` user site, so this
project inherits `https://wbk.ing/NautiPy/`. Do not add a project-specific
`CNAME` file. After the first successful `master` deployment, confirm that both
`https://wbk.ing/NautiPy/` and `https://cafawo.github.io/NautiPy/` resolve; the
CI deployment job follows redirects and requires a successful response from
both URLs.

## Prepare the release commit

Prepare the release on an up-to-date default branch. A sole maintainer may
commit it directly; a team may prefer a focused pull request. Neither path
publishes to PyPI.

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
8. Commit and push the preparation to the default branch, then wait for its
   `CI success` check. Do not create the release tag if that check fails.

## Create the release tag

Tag the tested default-branch commit, not a pull-request branch:

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
- publish a corrected patch release.

The workflow must fail on mismatched versions, duplicate uploads, unexpected
artifacts, checksum mismatches, or failed artifact tests.

## Distribution scope

PyPI is NautiPy's only maintained package index. The same tested wheel and
source distribution are attached to the GitHub Release for each version.
Other package indexes and downstream redistributions are outside this
project's release and support scope.

Official references:

- [PyPI account and two-factor help](https://pypi.org/help/)
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [Creating a PyPI project with a pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
- [Adding a PyPI publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
- [Using a PyPI publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [Managing GitHub environments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments)
- [GitHub Actions for Python](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
