from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO
from io import StringIO
from pathlib import Path
import re
import subprocess
import tarfile
from tempfile import TemporaryDirectory
import unittest
from urllib.error import HTTPError, URLError
import zipfile

from scripts import release, smoke_test_artifact


class ReleaseVersionTests(unittest.TestCase):
    def test_accepts_canonical_release_tags(self) -> None:
        expected = {
            "v0.1.0": "0.1.0",
            "v1.2.3a1": "1.2.3a1",
            "v2.0.0b0": "2.0.0b0",
            "v10.20.30rc12": "10.20.30rc12",
        }
        for tag, version in expected.items():
            with self.subTest(tag=tag):
                self.assertEqual(release.version_from_tag(tag), version)

    def test_rejects_noncanonical_or_unsupported_tags(self) -> None:
        tags = (
            "0.1.0",
            "v0.1",
            "v01.2.3",
            "v1.02.3",
            "v1.2.03",
            "v1.2.3-rc.1",
            "v1.2.3RC1",
            "v1.2.3rc01",
            "v1.2.3.dev1",
            "v1.2.3.post1",
            "v1!1.2.3",
            "v1.2.3+local",
        )
        for tag in tags:
            with self.subTest(tag=tag):
                with self.assertRaises(release.ReleaseValidationError):
                    release.version_from_tag(tag)


class ReleaseMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.pyproject = self.directory / "pyproject.toml"
        self.changelog = self.directory / "CHANGELOG.md"

    def write_pyproject(self, project_body: str) -> None:
        self.pyproject.write_text(
            "[build-system]\n"
            "requires = [\"setuptools\"]\n\n"
            "[project]\n"
            f"{project_body}\n\n"
            "[project.urls]\n"
            'Homepage = "https://example.invalid"\n',
            encoding="utf-8",
        )

    def write_changelog(self, release_section: str) -> None:
        self.changelog.write_text(
            "# Changelog\n\n"
            "## Unreleased\n\n"
            "- Future work.\n\n"
            f"{release_section}\n\n"
            "## 0.0.1 - 2025-01-02\n\n"
            "- Earlier release.\n",
            encoding="utf-8",
        )

    def test_reads_one_static_project_version(self) -> None:
        self.write_pyproject('name = "nautipy"\nversion = "0.1.0rc1"')
        self.assertEqual(release.read_project_version(self.pyproject), "0.1.0rc1")

    def test_rejects_missing_malformed_duplicate_or_dynamic_version(self) -> None:
        project_bodies = (
            'name = "nautipy"',
            'name = "nautipy"\nversion = 0.1',
            'version = "0.1.0"\nversion = "0.1.1"',
            'version = "0.1.0"\ndynamic = ["version"]',
            'version = "0.1.0"\ndynamic = [\n  "version",\n]',
        )
        for project_body in project_bodies:
            with self.subTest(project_body=project_body):
                self.write_pyproject(project_body)
                with self.assertRaises(release.ReleaseValidationError):
                    release.read_project_version(self.pyproject)

    def test_rejects_duplicate_project_tables(self) -> None:
        self.pyproject.write_text(
            '[project]\nversion = "0.1.0"\n\n'
            '[tool.example]\nvalue = true\n\n'
            '[project]\nversion = "0.1.0"\n',
            encoding="utf-8",
        )
        with self.assertRaises(release.ReleaseValidationError):
            release.read_project_version(self.pyproject)

    def test_extracts_exact_dated_nonempty_changelog_section(self) -> None:
        self.write_changelog(
            "## 0.1.0 - 2026-07-22\n\n"
            "### Added\n\n"
            "- Release automation."
        )
        entry = release.read_changelog_release(self.changelog, "0.1.0")
        self.assertEqual(entry.release_date.isoformat(), "2026-07-22")
        self.assertEqual(entry.notes, "### Added\n\n- Release automation.")

    def test_rejects_missing_invalid_duplicate_or_empty_changelog_section(self) -> None:
        sections = (
            "## [0.1.0] - 2026-07-22\n\n- Bracketed.",
            "## 0.1.0 - 2026-02-29\n\n- Invalid date.",
            "## 0.1.0 - 2026-07-22\n\n### Added",
            (
                "## 0.1.0 - 2026-07-22\n\n- First.\n\n"
                "## 0.1.0 - 2026-07-23\n\n- Duplicate."
            ),
        )
        for section in sections:
            with self.subTest(section=section):
                self.write_changelog(section)
                with self.assertRaises(release.ReleaseValidationError):
                    release.read_changelog_release(self.changelog, "0.1.0")

    def test_validates_matching_metadata_and_reports_prerelease(self) -> None:
        self.write_pyproject('name = "nautipy"\nversion = "0.1.0rc1"')
        self.write_changelog(
            "## 0.1.0rc1 - 2026-07-22\n\n- Release candidate."
        )
        info = release.validate_release(
            "v0.1.0rc1",
            pyproject=self.pyproject,
            changelog=self.changelog,
        )
        self.assertEqual(info.version, "0.1.0rc1")
        self.assertTrue(info.prerelease)

    def test_rejects_tag_and_project_version_mismatch(self) -> None:
        self.write_pyproject('name = "nautipy"\nversion = "0.1.1"')
        self.write_changelog("## 0.1.0 - 2026-07-22\n\n- Release.")
        with self.assertRaisesRegex(
            release.ReleaseValidationError,
            "does not match project version",
        ):
            release.validate_release(
                "v0.1.0",
                pyproject=self.pyproject,
                changelog=self.changelog,
            )


class GitTagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository = Path(self.temporary_directory.name)
        self.git("init", "--quiet")
        self.git("branch", "-M", "master")
        self.git("config", "user.name", "NautiPy Tests")
        self.git("config", "user.email", "tests@example.invalid")
        (self.repository / "tracked.txt").write_text("base\n", encoding="utf-8")
        self.git("add", "tracked.txt")
        self.git("commit", "--quiet", "-m", "base")

    def git(self, *arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=self.repository,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_accepts_annotated_tag_at_reachable_head(self) -> None:
        self.git("tag", "-a", "v0.1.0", "-m", "release")
        release.validate_git_tag(self.repository, "v0.1.0", "master")

    def test_rejects_lightweight_tag(self) -> None:
        self.git("tag", "v0.1.0")
        with self.assertRaisesRegex(
            release.ReleaseValidationError,
            "annotated tag",
        ):
            release.validate_git_tag(self.repository, "v0.1.0", "master")

    def test_rejects_tag_not_reachable_from_default_branch(self) -> None:
        self.git("checkout", "--quiet", "-b", "release-work")
        (self.repository / "tracked.txt").write_text("release\n", encoding="utf-8")
        self.git("add", "tracked.txt")
        self.git("commit", "--quiet", "-m", "release")
        self.git("tag", "-a", "v0.1.0", "-m", "release")
        with self.assertRaisesRegex(
            release.ReleaseValidationError,
            "not reachable",
        ):
            release.validate_git_tag(self.repository, "v0.1.0", "master")

    def test_rejects_tag_that_is_not_checked_out(self) -> None:
        self.git("tag", "-a", "v0.1.0", "-m", "release")
        (self.repository / "tracked.txt").write_text("later\n", encoding="utf-8")
        self.git("add", "tracked.txt")
        self.git("commit", "--quiet", "-m", "later")
        with self.assertRaisesRegex(
            release.ReleaseValidationError,
            "does not match",
        ):
            release.validate_git_tag(self.repository, "v0.1.0", "master")


class PyPIDuplicateTests(unittest.TestCase):
    class Response:
        def __init__(self, status: int) -> None:
            self.status = status

        def __enter__(self) -> "PyPIDuplicateTests.Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def test_accepts_explicit_not_found_response(self) -> None:
        def not_found(request: object, *, timeout: float) -> object:
            raise HTTPError(
                "https://pypi.org/",
                404,
                "not found",
                hdrs=None,
                fp=None,
            )

        release.ensure_pypi_version_absent(
            "nautipy",
            "0.1.0",
            opener=not_found,
        )

    def test_rejects_existing_version(self) -> None:
        def existing(request: object, *, timeout: float) -> object:
            return self.Response(200)

        with self.assertRaisesRegex(
            release.ReleaseValidationError,
            "already exists",
        ):
            release.ensure_pypi_version_absent(
                "nautipy",
                "0.1.0",
                opener=existing,
            )

    def test_fails_closed_on_http_network_or_unexpected_status(self) -> None:
        def server_error(request: object, *, timeout: float) -> object:
            raise HTTPError(
                "https://pypi.org/",
                503,
                "unavailable",
                hdrs=None,
                fp=None,
            )

        def network_error(request: object, *, timeout: float) -> object:
            raise URLError("offline")

        def unexpected(request: object, *, timeout: float) -> object:
            return self.Response(204)

        for opener in (server_error, network_error, unexpected):
            with self.subTest(opener=opener.__name__):
                with self.assertRaises(release.ReleaseValidationError):
                    release.ensure_pypi_version_absent(
                        "nautipy",
                        "0.1.0",
                        opener=opener,
                    )


class ReleaseArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.bundle = Path(self.temporary_directory.name) / "release-bundle"
        self.packages = self.bundle / "packages"
        self.packages.mkdir(parents=True)
        self.wheel = self.packages / "nautipy-0.1.0-py3-none-any.whl"
        self.sdist = self.packages / "nautipy-0.1.0.tar.gz"
        self.wheel.write_bytes(b"wheel")
        self.sdist.write_bytes(b"sdist")
        self.checksums = self.bundle / "SHA256SUMS"

    def test_writes_and_verifies_exact_artifact_checksums(self) -> None:
        artifacts = release.release_artifacts(
            self.packages,
            "nautipy",
            "0.1.0",
        )
        release.write_checksums(artifacts, self.checksums)
        release.verify_checksums(artifacts, self.checksums)
        manifest = self.checksums.read_text(encoding="utf-8")
        self.assertIn("packages/nautipy-0.1.0-py3-none-any.whl", manifest)
        self.assertIn("packages/nautipy-0.1.0.tar.gz", manifest)

    def test_rejects_missing_or_extra_artifacts(self) -> None:
        self.wheel.unlink()
        with self.assertRaises(release.ReleaseValidationError):
            release.release_artifacts(self.packages, "nautipy", "0.1.0")

        self.wheel.write_bytes(b"wheel")
        (self.packages / "unexpected.txt").write_text("extra", encoding="utf-8")
        with self.assertRaises(release.ReleaseValidationError):
            release.release_artifacts(self.packages, "nautipy", "0.1.0")

    def test_rejects_tampered_artifact(self) -> None:
        artifacts = release.release_artifacts(
            self.packages,
            "nautipy",
            "0.1.0",
        )
        release.write_checksums(artifacts, self.checksums)
        self.wheel.write_bytes(b"tampered")
        with self.assertRaisesRegex(
            release.ReleaseValidationError,
            "checksum mismatch",
        ):
            release.verify_checksums(artifacts, self.checksums)

    def test_rejects_checksum_manifest_with_unexpected_entry(self) -> None:
        artifacts = release.release_artifacts(
            self.packages,
            "nautipy",
            "0.1.0",
        )
        release.write_checksums(artifacts, self.checksums)
        with self.checksums.open("a", encoding="utf-8") as stream:
            stream.write(f"{'0' * 64}  packages/unexpected.txt\n")
        with self.assertRaises(release.ReleaseValidationError):
            release.verify_checksums(artifacts, self.checksums)


class DistributionContentTests(unittest.TestCase):
    def test_accepts_distributions_without_website_content(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            wheel = directory / "nautipy-0.1.0-py3-none-any.whl"
            sdist = directory / "nautipy-0.1.0.tar.gz"
            with zipfile.ZipFile(wheel, mode="w") as archive:
                archive.writestr("nautipy/__init__.py", "")
            with tarfile.open(sdist, mode="w:gz") as archive:
                self._add_tar_text(
                    archive,
                    "nautipy-0.1.0/src/nautipy/__init__.py",
                )

            smoke_test_artifact.validate_distribution_contents(
                wheel,
                is_wheel=True,
            )
            smoke_test_artifact.validate_distribution_contents(
                sdist,
                is_wheel=False,
            )

    def test_rejects_website_content_in_wheel_or_sdist(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            wheel = directory / "nautipy-0.1.0-py3-none-any.whl"
            sdist = directory / "nautipy-0.1.0.tar.gz"
            with zipfile.ZipFile(wheel, mode="w") as archive:
                archive.writestr("website/assets/boat.svg", "<svg/>")
            with tarfile.open(sdist, mode="w:gz") as archive:
                self._add_tar_text(
                    archive,
                    "nautipy-0.1.0/website/content/index.md",
                )

            for artifact, is_wheel in ((wheel, True), (sdist, False)):
                with self.subTest(artifact=artifact.name):
                    with self.assertRaisesRegex(
                        SystemExit,
                        "excluded website content",
                    ):
                        smoke_test_artifact.validate_distribution_contents(
                            artifact,
                            is_wheel=is_wheel,
                        )

    @staticmethod
    def _add_tar_text(archive: tarfile.TarFile, name: str) -> None:
        data = b"content"
        member = tarfile.TarInfo(name)
        member.size = len(data)
        archive.addfile(member, BytesIO(data))


class ReleaseCommandTests(unittest.TestCase):
    def test_validate_writes_workflow_outputs_and_release_notes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            pyproject = directory / "pyproject.toml"
            changelog = directory / "CHANGELOG.md"
            outputs = directory / "github-output"
            notes = directory / "release-notes.md"
            pyproject.write_text(
                '[project]\nname = "nautipy"\nversion = "0.1.0rc1"\n',
                encoding="utf-8",
            )
            changelog.write_text(
                "# Changelog\n\n"
                "## 0.1.0rc1 - 2026-07-22\n\n"
                "- Candidate notes.\n",
                encoding="utf-8",
            )
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = release.main(
                    [
                        "validate",
                        "v0.1.0rc1",
                        "--pyproject",
                        str(pyproject),
                        "--changelog",
                        str(changelog),
                        "--notes-output",
                        str(notes),
                        "--github-output",
                        str(outputs),
                    ]
                )
            self.assertEqual(result, 0, stderr.getvalue())
            self.assertEqual(
                outputs.read_text(encoding="utf-8"),
                "version=0.1.0rc1\nprerelease=true\n",
            )
            self.assertEqual(
                notes.read_text(encoding="utf-8"),
                "- Candidate notes.\n",
            )


class ReleaseWorkflowTests(unittest.TestCase):
    def test_ci_and_release_use_the_same_current_action_pins(self) -> None:
        root = Path(__file__).resolve().parents[1]
        ci_workflow = root / ".github/workflows/ci.yml"
        release_workflow = root / ".github/workflows/release.yml"
        if not ci_workflow.is_file() or not release_workflow.is_file():
            self.skipTest(
                "repository workflows are not shipped in the source archive"
            )

        ci_text = ci_workflow.read_text(encoding="utf-8")
        release_text = release_workflow.read_text(encoding="utf-8")

        def pins(text: str, action: str) -> set[str]:
            marker = f"uses: {action}@"
            return {
                line.split(marker, 1)[1].split()[0]
                for line in text.splitlines()
                if marker in line
            }

        for action in (
            "actions/checkout",
            "actions/setup-python",
            "actions/upload-artifact",
        ):
            with self.subTest(action=action):
                ci_pins = pins(ci_text, action)
                self.assertEqual(len(ci_pins), 1)
                self.assertEqual(ci_pins, pins(release_text, action))

        for workflow_text in (ci_text, release_text):
            for line in workflow_text.splitlines():
                stripped = line.strip()
                if not stripped.startswith(("- uses:", "uses:")):
                    continue
                action = stripped.split("uses:", 1)[1].strip()
                self.assertIn("@", action)
                reference = action.rsplit("@", 1)[1].split()[0]
                self.assertRegex(reference, r"^[0-9a-f]{40}$")

    def test_ci_has_an_always_running_aggregate_gate(self) -> None:
        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github/workflows/ci.yml"
        )
        if not workflow.is_file():
            self.skipTest(
                "repository workflow is not shipped in the source archive"
            )

        text = workflow.read_text(encoding="utf-8")
        aggregate = text.split("  ci-success:\n", 1)[1]
        self.assertIn("name: CI success", aggregate)
        self.assertIn("if: ${{ always() }}", aggregate)
        for job in (
            "test",
            "minimum-dependencies",
            "cross-platform-smoke",
            "documentation",
            "build",
        ):
            with self.subTest(job=job):
                self.assertIn(f"${{{{ needs.{job}.result }}}}", aggregate)

    def test_ci_builds_docs_and_deploys_only_gated_master_pushes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = root / ".github/workflows/ci.yml"
        requirements = root / "website/requirements.txt"
        if not workflow.is_file() or not requirements.is_file():
            self.skipTest(
                "website source and repository workflows are excluded from "
                "package distributions"
            )

        text = workflow.read_text(encoding="utf-8")
        documentation = text.split("  documentation:\n", 1)[1]
        documentation = documentation.split("\n  build:", 1)[0]
        self.assertIn(
            "python -m pip install --requirement website/requirements.txt",
            documentation,
        )
        self.assertIn(
            "python website/tools/generate_fix_lab.py --check",
            documentation,
        )
        self.assertIn(
            "python -m unittest discover -s website/tests -v",
            documentation,
        )
        self.assertIn(
            "node --check website/content/assets/javascripts/fix-lab.js",
            documentation,
        )
        self.assertIn("actions/setup-node@", documentation)
        self.assertIn('node-version: "24"', documentation)
        self.assertIn("python -m mkdocs build --clean --strict", documentation)
        self.assertIn("--config-file website/mkdocs.yml", documentation)
        self.assertIn("pages: read", documentation)
        self.assertIn("actions/configure-pages@", documentation)
        self.assertIn("actions/upload-pages-artifact@", documentation)
        self.assertLess(
            documentation.index("actions/configure-pages@"),
            documentation.index("actions/upload-pages-artifact@"),
        )
        self.assertIn("path: website/site", documentation)
        self.assertEqual(
            requirements.read_text(encoding="utf-8").strip(),
            "mkdocs-material==9.7.7",
        )

        deployment = text.split("  deploy-pages:\n", 1)[1]
        self.assertIn(
            "github.event_name == 'push' && "
            "github.ref == 'refs/heads/master'",
            deployment,
        )
        self.assertIn("- documentation", deployment)
        self.assertIn("- ci-success", deployment)
        self.assertIn("pages: write", deployment)
        self.assertIn("id-token: write", deployment)
        self.assertNotIn("contents: write", deployment)
        self.assertNotIn("actions/configure-pages@", deployment)
        self.assertIn("actions/deploy-pages@", deployment)
        self.assertIn("https://wbk.ing/NautiPy/", deployment)
        self.assertIn("https://cafawo.github.io/NautiPy/", deployment)
        self.assertIn("--fail", deployment)
        self.assertIn("--location", deployment)
        self.assertIn('--write-out "%{http_code}"', deployment)
        self.assertIn('[[ "$status" != "200" ]]', deployment)

    def test_website_is_linked_but_excluded_from_distributions(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = root / "MANIFEST.in"
        pyproject = root / "pyproject.toml"
        if not manifest.is_file() or not pyproject.is_file():
            self.skipTest("packaging sources are unavailable")

        self.assertRegex(
            manifest.read_text(encoding="utf-8"),
            r"(?m)^prune website$",
        )
        text = pyproject.read_text(encoding="utf-8")
        self.assertIn(
            'Documentation = "https://wbk.ing/NautiPy/"',
            text,
        )
        self.assertNotIn("mkdocs", text.lower())

    def test_ci_pins_every_declared_minimum_dependency_exactly(self) -> None:
        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github/workflows/ci.yml"
        )
        if not workflow.is_file():
            self.skipTest(
                "repository workflow is not shipped in the source archive"
            )

        text = workflow.read_text(encoding="utf-8")
        minimum_job = text.split("  minimum-dependencies:\n", 1)[1]
        minimum_job = minimum_job.split("\n  cross-platform-smoke:", 1)[0]
        for requirement in (
            "geographiclib==2.1",
            "numpy==1.23.5",
            "scipy==1.14.1",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(f'"{requirement}"', minimum_job)

    def test_one_package_workflows_have_no_optional_fix_path(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = (
            root / ".github/workflows/ci.yml",
            root / ".github/workflows/release.yml",
            root / "scripts/smoke_test_artifact.py",
        )
        if any(not path.is_file() for path in paths):
            self.skipTest(
                "repository workflows and scripts are not shipped in the "
                "source archive"
            )

        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertNotIn(".[fix]", combined)
        self.assertNotIn("--fix", combined)
        self.assertNotIn("fix-test:", combined)
        self.assertNotIn("cross-platform-fix-smoke:", combined)
        self.assertIn("solve_fix", combined)

    def test_project_declares_one_complete_runtime_dependency_set(self) -> None:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        if not pyproject.is_file():
            self.skipTest("project metadata is not shipped in the source archive")

        text = pyproject.read_text(encoding="utf-8")
        dependency_match = re.search(
            r"(?ms)^dependencies\s*=\s*\[(.*?)^\]",
            text,
        )
        if dependency_match is None:
            self.fail("pyproject.toml has no project dependency list")
        requirements = sorted(
            re.findall(r'"([^"]+)"', dependency_match.group(1))
        )
        self.assertEqual(
            requirements,
            [
                "geographiclib>=2.1",
                "numpy>=1.23.5",
                "scipy>=1.14.1",
            ],
        )
        self.assertNotIn("[project.optional-dependencies]", text)

    def test_github_release_has_explicit_repository_context(self) -> None:
        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github/workflows/release.yml"
        )
        if not workflow.is_file():
            self.skipTest(
                "repository workflow is not shipped in the source archive"
            )

        text = workflow.read_text(encoding="utf-8")
        github_release_job = text.split("  github-release:\n", 1)[1]
        self.assertIn("GH_REPO: ${{ github.repository }}", github_release_job)
        self.assertIn('gh release create "${GITHUB_REF_NAME}"', github_release_job)


if __name__ == "__main__":
    unittest.main()
