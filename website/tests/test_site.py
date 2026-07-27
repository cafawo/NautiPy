from __future__ import annotations

import ast
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ElementTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WEBSITE_ROOT = REPOSITORY_ROOT / "website"
CONTENT_ROOT = WEBSITE_ROOT / "content"
IMAGE_ROOT = CONTENT_ROOT / "assets" / "images"
FIXTURE_PATH = CONTENT_ROOT / "assets" / "data" / "fix-lab.json"
GENERATOR_PATH = WEBSITE_ROOT / "tools" / "generate_fix_lab.py"
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"


class _HtmlInventory(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.identifiers: set[str] = set()
        self.references: list[
            tuple[str, str, str, dict[str, str | None]]
        ] = []

    def handle_starttag(
        self,
        tag: str,
        attributes: list[tuple[str, str | None]],
    ) -> None:
        self._record(tag, attributes)

    def handle_startendtag(
        self,
        tag: str,
        attributes: list[tuple[str, str | None]],
    ) -> None:
        self._record(tag, attributes)

    def _record(
        self,
        tag: str,
        attributes: list[tuple[str, str | None]],
    ) -> None:
        values = dict(attributes)
        identifier = values.get("id")
        if identifier:
            self.identifiers.add(identifier)
        for name in ("href", "src", "data-fixture-url"):
            value = values.get(name)
            if value:
                self.references.append((tag, name, value, values))


def load_generator():
    specification = importlib.util.spec_from_file_location(
        "nautipy_fix_lab_generator",
        GENERATOR_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load the Fix Lab generator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class BuiltSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        temporary_directory = TemporaryDirectory()
        cls.addClassCleanup(temporary_directory.cleanup)
        cls.site_root = Path(temporary_directory.name).resolve()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mkdocs",
                "build",
                "--clean",
                "--strict",
                "--config-file",
                str(WEBSITE_ROOT / "mkdocs.yml"),
                "--site-dir",
                str(cls.site_root),
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise AssertionError(
                "strict MkDocs build failed during site tests:\n"
                f"{result.stdout}\n{result.stderr}"
            )

        cls.documents: dict[Path, _HtmlInventory] = {}
        for path in sorted(cls.site_root.rglob("*.html")):
            inventory = _HtmlInventory()
            inventory.feed(path.read_text(encoding="utf-8"))
            cls.documents[path.resolve()] = inventory

    @classmethod
    def _local_target(
        cls,
        source: Path,
        url: str,
    ) -> tuple[Path, str] | None:
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc or url.startswith("//"):
            return None

        relative_path = unquote(parsed.path)
        site_prefix = "/NautiPy/"
        if relative_path.startswith(site_prefix):
            target = cls.site_root / relative_path[len(site_prefix) :]
        elif relative_path.startswith("/"):
            target = cls.site_root / relative_path.lstrip("/")
        elif relative_path:
            target = source.parent / relative_path
        else:
            target = source
        target = target.resolve()
        try:
            target.relative_to(cls.site_root)
        except ValueError as error:
            raise AssertionError(
                f"{source.relative_to(cls.site_root)}: link escapes the site: {url}"
            ) from error
        if target.is_dir():
            target /= "index.html"
        return target, unquote(parsed.fragment)

    def test_every_internal_link_asset_and_anchor_resolves(self) -> None:
        failures: list[str] = []
        for source, inventory in self.documents.items():
            for _, _, url, _ in inventory.references:
                local = self._local_target(source, url)
                if local is None:
                    continue
                target, fragment = local
                description = f"{source.relative_to(self.site_root)} -> {url}"
                if not target.is_file():
                    failures.append(f"{description} (missing target)")
                    continue
                if fragment and target.suffix == ".html":
                    target_inventory = self.documents.get(target.resolve())
                    if (
                        target_inventory is None
                        or fragment not in target_inventory.identifiers
                    ):
                        failures.append(f"{description} (missing anchor)")
        self.assertEqual(failures, [])

    def test_site_loads_no_automatic_remote_resources(self) -> None:
        failures: list[str] = []
        resource_attributes = {
            ("iframe", "src"),
            ("img", "src"),
            ("script", "src"),
            ("source", "src"),
        }
        fetched_link_relations = {
            "icon",
            "manifest",
            "modulepreload",
            "preload",
            "stylesheet",
        }
        for source, inventory in self.documents.items():
            source_text = source.read_text(encoding="utf-8")
            if 'data-md-component="source"' in source_text:
                failures.append(
                    f"{source.relative_to(self.site_root)} enables repository API data"
                )
            for tag, name, url, attributes in inventory.references:
                is_resource = (tag, name) in resource_attributes
                if tag == "link" and name == "href":
                    relations = set((attributes.get("rel") or "").split())
                    is_resource = bool(relations & fetched_link_relations)
                if not is_resource:
                    continue
                parsed = urlsplit(url)
                if parsed.scheme in {"http", "https"} or parsed.netloc:
                    failures.append(
                        f"{source.relative_to(self.site_root)} loads {url}"
                    )

        for path in sorted(self.site_root.rglob("*.css")):
            if re.search(
                r"(?:url\(\s*|@import\s+(?:url\(\s*)?)[\"']?(?:https?:)?//",
                path.read_text(encoding="utf-8"),
                re.IGNORECASE,
            ):
                failures.append(
                    f"{path.relative_to(self.site_root)} loads a remote resource"
                )
        self.assertEqual(failures, [])


class MarkdownExampleTests(unittest.TestCase):
    def test_every_fenced_python_example_compiles(self) -> None:
        pattern = re.compile(
            r"^```(?:python|py)(?:[^\n]*)\n(.*?)^```\s*$",
            re.MULTILINE | re.DOTALL,
        )
        example_count = 0
        for path in sorted(CONTENT_ROOT.rglob("*.md")):
            source = path.read_text(encoding="utf-8")
            for index, match in enumerate(pattern.finditer(source), start=1):
                example_count += 1
                with self.subTest(path=path.relative_to(REPOSITORY_ROOT), index=index):
                    compile(
                        match.group(1),
                        f"{path}:python-example-{index}",
                        "exec",
                    )
        self.assertGreaterEqual(example_count, 8)

    def test_site_has_no_unrendered_tex_delimiters(self) -> None:
        offenders: list[str] = []
        for path in sorted(CONTENT_ROOT.rglob("*.md")):
            source = path.read_text(encoding="utf-8")
            if "\\(" in source or "\\[" in source or "$$" in source:
                offenders.append(str(path.relative_to(REPOSITORY_ROOT)))
        self.assertEqual(offenders, [])


class SvgAssetTests(unittest.TestCase):
    expected_names = {
        "bearing-geometry.svg",
        "coordinate-notation.svg",
        "coordinate-order.svg",
        "ellipsoid-geodesic.svg",
        "package-flow.svg",
        "range-geometry.svg",
        "uncertainty-ellipse.svg",
    }

    def test_required_original_svg_assets_are_present_and_accessible(self) -> None:
        paths = sorted(IMAGE_ROOT.glob("*.svg"))
        self.assertEqual({path.name for path in paths}, self.expected_names)
        for path in paths:
            with self.subTest(path=path.name):
                tree = ElementTree.parse(path)
                root = tree.getroot()
                self.assertEqual(root.tag, f"{SVG_NAMESPACE}svg")
                self.assertEqual(root.get("role"), "img")
                title = root.find(f"{SVG_NAMESPACE}title")
                description = root.find(f"{SVG_NAMESPACE}desc")
                self.assertIsNotNone(title)
                self.assertIsNotNone(description)
                self.assertTrue((title.text or "").strip())
                self.assertTrue((description.text or "").strip())
                labelled_by = set((root.get("aria-labelledby") or "").split())
                self.assertIn(title.get("id"), labelled_by)
                self.assertIn(description.get("id"), labelled_by)

    def test_svgs_embed_no_remote_resources(self) -> None:
        remote_reference = re.compile(
            r"(?:\bhref\s*=|url\()\s*[\"']?(?:https?:)?//",
            re.IGNORECASE,
        )
        for path in sorted(IMAGE_ROOT.glob("*.svg")):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIsNone(remote_reference.search(source))
                root = ElementTree.fromstring(source)
                for element in root.iter():
                    for name, value in element.attrib.items():
                        if name.endswith("href"):
                            self.assertFalse(
                                value.startswith(("http:", "https:", "//")),
                                (path, value),
                            )

    def test_static_bearing_arrows_explicitly_start_at_the_boat(self) -> None:
        root = ElementTree.parse(IMAGE_ROOT / "bearing-geometry.svg").getroot()
        arrows = [
            element
            for element in root.iter(f"{SVG_NAMESPACE}line")
            if element.get("data-bearing-origin") == "boat"
        ]
        self.assertEqual(len(arrows), 4)
        starts = [(arrow.get("x1"), arrow.get("y1")) for arrow in arrows]
        self.assertEqual(starts.count(("175", "242")), 2)
        self.assertEqual(starts.count(("170", "252")), 2)


class FixLabFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = load_generator()
        cls.generated = cls.generator.build_document()
        cls.committed = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        cls.scenarios = {
            scenario["id"]: scenario for scenario in cls.generated["scenarios"]
        }

    def test_committed_fixture_matches_public_api_generator(self) -> None:
        self.assertEqual(self.committed, self.generated)
        self.assertLess(FIXTURE_PATH.stat().st_size, 100_000)
        self.assertEqual(self.generated["earth_model"], "WGS84")
        self.assertIn("not certified", self.generated["safety_note"])
        self.assertIn("does not solve", self.generated["browser_model"])

    def test_generator_imports_only_the_public_nautipy_namespace(self) -> None:
        tree = ast.parse(GENERATOR_PATH.read_text(encoding="utf-8"))
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        self.assertIn("nautipy", imported_modules)
        self.assertFalse(
            any(module.startswith("nautipy.") for module in imported_modules)
        )
        self.assertTrue(
            {"numpy", "scipy", "geographiclib"}.isdisjoint(imported_modules)
        )

    def test_frankfurt_range_scenarios_preserve_ambiguity_then_converge(self) -> None:
        ambiguous = self.scenarios["two-ranges"]
        resolved = self.scenarios["third-range"]
        self.assertEqual(ambiguous["diagnostics"]["status"], "ambiguous")
        self.assertEqual(ambiguous["diagnostics"]["candidate_count"], 2)
        self.assertEqual(resolved["diagnostics"]["status"], "converged")
        fix = next(
            position
            for position in resolved["positions"]
            if position["id"] == "fix"
        )
        self.assertAlmostEqual(fix["latitude"], 50.127198, places=7)
        self.assertAlmostEqual(fix["longitude"], 8.665562, places=7)

    def test_tangent_and_bearing_geometry_are_explicit(self) -> None:
        tangent = self.scenarios["tangent-ranges"]["diagnostics"]
        strong = self.scenarios["strong-bearings"]["diagnostics"]
        weak = self.scenarios["weak-bearings"]["diagnostics"]
        self.assertEqual(tangent["candidate_status"], "unique")
        self.assertEqual(tangent["status"], "degenerate")
        self.assertLess(strong["condition_number"], 10)
        self.assertGreater(weak["condition_number"], 1_000)
        self.assertTrue(any("weak" in warning for warning in weak["warnings"]))

    def test_weighting_and_uncertainty_scenarios_retain_numerical_lessons(self) -> None:
        weighting = self.scenarios["uncertainty-weighting"]["diagnostics"]
        uncertainty = self.scenarios["uncertainty-ellipse"]["diagnostics"]
        self.assertGreater(
            weighting["high_weight_shift_m"],
            weighting["low_weight_shift_m"] * 100,
        )
        self.assertAlmostEqual(
            uncertainty["scaled_semi_major_95_m"]
            / uncertainty["baseline_semi_major_95_m"],
            uncertainty["uncertainty_scale"],
            delta=0.002,
        )
        for ellipse in self.scenarios["uncertainty-ellipse"]["ellipses"]:
            self.assertIn("not a safety bound", ellipse["note"])

    def test_noisy_mixed_scenario_contains_both_natural_units(self) -> None:
        scenario = self.scenarios["noisy-mixed"]
        self.assertEqual(scenario["diagnostics"]["status"], "converged")
        self.assertEqual(
            {observation["kind"] for observation in scenario["observations"]},
            {"bearing", "range"},
        )
        self.assertGreater(scenario["diagnostics"]["rms"], 0)
        residuals = scenario["diagnostics"]["residuals"]
        self.assertEqual(len(residuals), 4)
        self.assertEqual(
            {residual["natural_unit"] for residual in residuals},
            {"degrees", "metres"},
        )
        self.assertTrue(
            all(
                isinstance(residual["standardized_residual"], float)
                for residual in residuals
            )
        )

    def test_every_bearing_arrow_origin_is_a_boat_position(self) -> None:
        bearing_count = 0
        for scenario in self.generated["scenarios"]:
            point_ids = {position["id"] for position in scenario["positions"]}
            for observation in scenario["observations"]:
                if observation["kind"] != "bearing":
                    continue
                bearing_count += 1
                self.assertEqual(observation["origin_id"], "truth")
                self.assertIn(observation["origin_id"], point_ids)
        self.assertGreaterEqual(bearing_count, 6)


if __name__ == "__main__":
    unittest.main()
