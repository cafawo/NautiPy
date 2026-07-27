#!/usr/bin/env python3
"""Validate releases and protect the build-once artifact handoff."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import hashlib
import hmac
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


_VERSION_TEXT = (
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
    r"(?:(?:a|b|rc)(?:0|[1-9][0-9]*))?"
)
_VERSION_RE = re.compile(rf"^{_VERSION_TEXT}$")
_TAG_RE = re.compile(rf"^v(?P<version>{_VERSION_TEXT})$")
_PRERELEASE_RE = re.compile(r"(?:a|b|rc)[0-9]+$")
_SECTION_RE = re.compile(r"^\s*\[(?P<name>[^]]+)]\s*(?:#.*)?$")
_VERSION_ASSIGNMENT_RE = re.compile(
    r"^\s*version\s*=\s*(?P<quote>['\"])(?P<version>[^'\"]+)"
    r"(?P=quote)\s*(?:#.*)?$"
)
_DYNAMIC_RE = re.compile(
    r"^\s*dynamic\s*=\s*\[(?P<body>.*?)]",
    flags=re.MULTILINE | re.DOTALL,
)
_DYNAMIC_VERSION_RE = re.compile(r"['\"]version['\"]")
_LEVEL_TWO_HEADING_RE = re.compile(r"^##(?:\s|$)", flags=re.MULTILINE)
_CHECKSUM_LINE_RE = re.compile(
    r"^(?P<digest>[0-9a-f]{64})  (?P<path>[^\r\n]+)$"
)


class ReleaseValidationError(ValueError):
    """A release input or artifact violates the repository release contract."""


@dataclass(frozen=True)
class ChangelogRelease:
    """The dated changelog entry selected for a release."""

    release_date: date
    notes: str


@dataclass(frozen=True)
class ReleaseInfo:
    """Validated release metadata used by the workflow."""

    tag: str
    version: str
    prerelease: bool
    changelog: ChangelogRelease


def version_from_tag(tag: str) -> str:
    """Return a canonical release version from a ``v``-prefixed tag."""

    match = _TAG_RE.fullmatch(tag)
    if match is None:
        raise ReleaseValidationError(
            "release tag must be vX.Y.Z with an optional aN, bN, or rcN suffix"
        )
    return match.group("version")


def validate_version(version: str) -> str:
    """Validate the repository's canonical PEP 440 semantic-release subset."""

    if _VERSION_RE.fullmatch(version) is None:
        raise ReleaseValidationError(
            "project version must be X.Y.Z with an optional aN, bN, or rcN suffix"
        )
    return version


def _project_section(text: str) -> str:
    lines = text.splitlines()
    sections: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = _SECTION_RE.fullmatch(line)
        if match is not None:
            sections.append((index, match.group("name").strip()))

    project_sections = [index for index, name in sections if name == "project"]
    if not project_sections:
        raise ReleaseValidationError("pyproject.toml has no [project] table")
    if len(project_sections) != 1:
        raise ReleaseValidationError(
            "pyproject.toml contains more than one [project] table"
        )
    project_header = project_sections[0]
    start = project_header + 1
    end = next(
        (index for index, _ in sections if index > project_header),
        len(lines),
    )
    return "\n".join(lines[start:end])


def read_project_version(path: Path) -> str:
    """Read the required one-line static version from ``[project]``."""

    section = _project_section(path.read_text(encoding="utf-8"))
    versions: list[str] = []
    for line in section.splitlines():
        match = _VERSION_ASSIGNMENT_RE.fullmatch(line)
        if match is not None:
            versions.append(match.group("version"))
        elif re.match(r"^\s*version\s*=", line):
            raise ReleaseValidationError(
                "[project].version must be a one-line quoted static value"
            )

    if not versions:
        raise ReleaseValidationError("pyproject.toml has no static [project].version")
    if len(versions) != 1:
        raise ReleaseValidationError(
            "pyproject.toml contains more than one [project].version"
        )

    dynamic = _DYNAMIC_RE.search(section)
    if dynamic is not None and _DYNAMIC_VERSION_RE.search(dynamic.group("body")):
        raise ReleaseValidationError(
            "[project].version cannot also be listed as dynamic"
        )
    return validate_version(versions[0])


def read_changelog_release(path: Path, version: str) -> ChangelogRelease:
    """Extract one exact, dated, nonempty level-two changelog section."""

    validate_version(version)
    text = path.read_text(encoding="utf-8")
    heading = re.compile(
        rf"^## {re.escape(version)} - "
        r"(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})[ \t]*$",
        flags=re.MULTILINE,
    )
    matches = list(heading.finditer(text))
    if not matches:
        raise ReleaseValidationError(
            f"CHANGELOG.md needs an exact '## {version} - YYYY-MM-DD' section"
        )
    if len(matches) != 1:
        raise ReleaseValidationError(
            f"CHANGELOG.md contains more than one section for {version}"
        )

    match = matches[0]
    try:
        release_date = date.fromisoformat(match.group("date"))
    except ValueError as error:
        raise ReleaseValidationError(
            f"CHANGELOG.md has an invalid date for {version}"
        ) from error

    remainder = text[match.end() :]
    next_heading = _LEVEL_TWO_HEADING_RE.search(remainder)
    if next_heading is not None:
        remainder = remainder[: next_heading.start()]
    notes = remainder.strip()
    meaningful_lines = [
        line
        for line in notes.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not meaningful_lines:
        raise ReleaseValidationError(
            f"CHANGELOG.md section for {version} has no release notes"
        )
    return ChangelogRelease(release_date=release_date, notes=notes)


def _git(
    repository: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=check,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise ReleaseValidationError("git is required for tag validation") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "git command failed"
        raise ReleaseValidationError(detail) from error


def validate_git_tag(repository: Path, tag: str, default_branch: str) -> None:
    """Require an annotated tag at HEAD whose commit is on the default branch."""

    version_from_tag(tag)
    tag_reference = f"refs/tags/{tag}"
    tag_type = _git(repository, "cat-file", "-t", tag_reference).stdout.strip()
    if tag_type != "tag":
        raise ReleaseValidationError(f"{tag} must be an annotated tag")

    tag_commit = _git(
        repository,
        "rev-parse",
        "--verify",
        f"{tag_reference}^{{commit}}",
    ).stdout.strip()
    head_commit = _git(
        repository,
        "rev-parse",
        "--verify",
        "HEAD^{commit}",
    ).stdout.strip()
    if tag_commit != head_commit:
        raise ReleaseValidationError(
            f"checked-out commit {head_commit} does not match {tag} ({tag_commit})"
        )

    branch_commit = _git(
        repository,
        "rev-parse",
        "--verify",
        f"{default_branch}^{{commit}}",
    ).stdout.strip()
    ancestry = _git(
        repository,
        "merge-base",
        "--is-ancestor",
        tag_commit,
        branch_commit,
        check=False,
    )
    if ancestry.returncode == 1:
        raise ReleaseValidationError(
            f"tagged commit is not reachable from {default_branch}"
        )
    if ancestry.returncode != 0:
        detail = ancestry.stderr.strip() or "could not check default-branch ancestry"
        raise ReleaseValidationError(detail)


def ensure_pypi_version_absent(
    project_name: str,
    version: str,
    *,
    opener: Callable[..., Any] | None = None,
    timeout: float = 10.0,
) -> None:
    """Fail closed unless PyPI explicitly reports that a version is absent."""

    validate_version(version)
    if timeout <= 0:
        raise ReleaseValidationError("PyPI lookup timeout must be positive")
    request = Request(
        "https://pypi.org/pypi/"
        f"{quote(project_name, safe='')}/{quote(version, safe='')}/json",
        headers={"User-Agent": "NautiPy-release-validator/0.1"},
    )
    open_url = opener or urlopen
    try:
        with open_url(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
    except HTTPError as error:
        if error.code == 404:
            return
        raise ReleaseValidationError(
            f"PyPI duplicate check failed with HTTP {error.code}"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise ReleaseValidationError(f"PyPI duplicate check failed: {error}") from error

    if status == 404:
        return
    if status == 200:
        raise ReleaseValidationError(
            f"{project_name} {version} already exists on PyPI"
        )
    raise ReleaseValidationError(
        f"PyPI duplicate check returned unexpected HTTP {status}"
    )


def validate_release(
    tag: str,
    *,
    pyproject: Path,
    changelog: Path,
    check_git: bool = False,
    repository: Path = Path("."),
    default_branch: str = "origin/master",
    check_pypi: bool = False,
    project_name: str = "nautipy",
    pypi_timeout: float = 10.0,
    opener: Callable[..., Any] | None = None,
) -> ReleaseInfo:
    """Validate all requested release gates and return reusable metadata."""

    version = version_from_tag(tag)
    project_version = read_project_version(pyproject)
    if project_version != version:
        raise ReleaseValidationError(
            f"tag version {version} does not match project version {project_version}"
        )
    changelog_release = read_changelog_release(changelog, version)
    if check_git:
        validate_git_tag(repository, tag, default_branch)
    if check_pypi:
        ensure_pypi_version_absent(
            project_name,
            version,
            opener=opener,
            timeout=pypi_timeout,
        )
    return ReleaseInfo(
        tag=tag,
        version=version,
        prerelease=_PRERELEASE_RE.search(version) is not None,
        changelog=changelog_release,
    )


def _distribution_names(project_name: str, version: str) -> tuple[str, str]:
    validate_version(version)
    wheel_name = re.sub(r"[-_.]+", "_", project_name).lower()
    sdist_name = re.sub(r"[-_.]+", "-", project_name).lower()
    return (
        f"{wheel_name}-{version}-py3-none-any.whl",
        f"{sdist_name}-{version}.tar.gz",
    )


def release_artifacts(
    packages_directory: Path,
    project_name: str,
    version: str,
) -> tuple[Path, Path]:
    """Return the exact expected pure-Python wheel and sdist."""

    if not packages_directory.is_dir():
        raise ReleaseValidationError(
            f"artifact directory does not exist: {packages_directory}"
        )
    expected_names = _distribution_names(project_name, version)
    actual_names = {path.name for path in packages_directory.iterdir()}
    if actual_names != set(expected_names):
        expected = ", ".join(expected_names)
        actual = ", ".join(sorted(actual_names)) or "none"
        raise ReleaseValidationError(
            f"expected exactly {expected}; artifact directory contains {actual}"
        )
    return (
        packages_directory / expected_names[0],
        packages_directory / expected_names[1],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(artifacts: Sequence[Path], output: Path) -> None:
    """Write deterministic ``sha256sum``-compatible entries."""

    output.parent.mkdir(parents=True, exist_ok=True)
    base = output.parent.resolve()
    lines: list[str] = []
    for artifact in sorted(artifacts, key=lambda item: item.name):
        try:
            relative = artifact.resolve().relative_to(base)
        except ValueError as error:
            raise ReleaseValidationError(
                "release artifacts must be below the checksum-file directory"
            ) from error
        lines.append(f"{_sha256(artifact)}  {relative.as_posix()}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_checksums(artifacts: Sequence[Path], checksums: Path) -> None:
    """Verify a strict checksum manifest for exactly the expected artifacts."""

    base = checksums.parent.resolve()
    expected_paths: dict[str, Path] = {}
    for artifact in artifacts:
        try:
            relative = artifact.resolve().relative_to(base).as_posix()
        except ValueError as error:
            raise ReleaseValidationError(
                "release artifacts must be below the checksum-file directory"
            ) from error
        expected_paths[relative] = artifact

    entries: dict[str, str] = {}
    for line in checksums.read_text(encoding="utf-8").splitlines():
        match = _CHECKSUM_LINE_RE.fullmatch(line)
        if match is None:
            raise ReleaseValidationError("SHA256SUMS contains an invalid line")
        relative = match.group("path")
        if relative in entries:
            raise ReleaseValidationError(
                f"SHA256SUMS contains a duplicate entry for {relative}"
            )
        entries[relative] = match.group("digest")

    if entries.keys() != expected_paths.keys():
        raise ReleaseValidationError(
            "SHA256SUMS must contain exactly the wheel and source distribution"
        )
    for relative, artifact in expected_paths.items():
        if not hmac.compare_digest(entries[relative], _sha256(artifact)):
            raise ReleaseValidationError(f"checksum mismatch for {relative}")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_github_output(path: Path, info: ReleaseInfo) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"version={info.version}\n")
        stream.write(f"prerelease={str(info.prerelease).lower()}\n")


def _add_metadata_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tag", help="v-prefixed release tag")
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
    )
    parser.add_argument(
        "--changelog",
        type=Path,
        default=Path("CHANGELOG.md"),
    )
    parser.add_argument("--project-name", default="nautipy")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate release metadata")
    _add_metadata_arguments(validate)
    validate.add_argument("--check-git", action="store_true")
    validate.add_argument("--repository", type=Path, default=Path("."))
    validate.add_argument("--default-branch", default="origin/master")
    validate.add_argument("--check-pypi", action="store_true")
    validate.add_argument("--pypi-timeout", type=float, default=10.0)
    validate.add_argument("--notes-output", type=Path)
    validate.add_argument("--github-output", type=Path)

    prepare = commands.add_parser(
        "prepare-artifacts",
        help="validate and describe a newly built wheel and sdist",
    )
    _add_metadata_arguments(prepare)
    prepare.add_argument("packages_directory", type=Path)
    prepare.add_argument("--checksums-output", type=Path, required=True)
    prepare.add_argument("--notes-output", type=Path, required=True)

    verify = commands.add_parser(
        "verify-artifacts",
        help="verify the exact wheel, sdist, and checksum manifest",
    )
    verify.add_argument("tag", help="v-prefixed release tag")
    verify.add_argument("packages_directory", type=Path)
    verify.add_argument("--project-name", default="nautipy")
    verify.add_argument("--checksums", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        if args.command == "validate":
            info = validate_release(
                args.tag,
                pyproject=args.pyproject,
                changelog=args.changelog,
                check_git=args.check_git,
                repository=args.repository,
                default_branch=args.default_branch,
                check_pypi=args.check_pypi,
                project_name=args.project_name,
                pypi_timeout=args.pypi_timeout,
            )
            if args.notes_output is not None:
                _write_text(args.notes_output, info.changelog.notes)
            if args.github_output is not None:
                _write_github_output(args.github_output, info)
            print(f"validated release {info.tag}")
            return 0

        if args.command == "prepare-artifacts":
            info = validate_release(
                args.tag,
                pyproject=args.pyproject,
                changelog=args.changelog,
                project_name=args.project_name,
            )
            artifacts = release_artifacts(
                args.packages_directory,
                args.project_name,
                info.version,
            )
            write_checksums(artifacts, args.checksums_output)
            _write_text(args.notes_output, info.changelog.notes)
            print(f"prepared artifacts for {info.tag}")
            return 0

        version = version_from_tag(args.tag)
        artifacts = release_artifacts(
            args.packages_directory,
            args.project_name,
            version,
        )
        verify_checksums(artifacts, args.checksums)
        print(f"verified artifacts for {args.tag}")
        return 0
    except (OSError, ReleaseValidationError) as error:
        print(f"release validation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
