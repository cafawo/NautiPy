from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
import subprocess
import sys
import unittest

from nautipy.cli import main


class CommandLineTests(unittest.TestCase):
    def invoke(self, *arguments: str) -> tuple[int, str, str]:
        stdout = StringIO()
        stderr = StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                return_code = main(arguments)
        except SystemExit as error:
            return_code = int(error.code or 0)
        return return_code, stdout.getvalue(), stderr.getvalue()

    def test_convert_documented_example(self) -> None:
        return_code, stdout, stderr = self.invoke(
            "convert",
            "50° 7.3542' N; 8° 39.942' E",
            "--to",
            "dd",
        )

        self.assertEqual(return_code, 0)
        self.assertEqual(stdout, "50.122570, 8.665700\n")
        self.assertEqual(stderr, "")

    def test_convert_exposes_public_formatting_controls(self) -> None:
        return_code, stdout, stderr = self.invoke(
            "convert",
            "50.12257, 8.66570",
            "--to",
            "DMS",
            "--output-order",
            "lonlat",
            "--precision",
            "1",
            "--notation",
            "signed",
            "--symbols",
            "ascii",
            "--separator",
            " / ",
        )

        self.assertEqual(return_code, 0)
        self.assertEqual(
            stdout,
            "8 deg 39 min 56.5 sec / 50 deg 7 min 21.3 sec\n",
        )
        self.assertEqual(stderr, "")

        return_code, stdout, stderr = self.invoke(
            "convert",
            "50.12257, 8.66570",
            "--to",
            "iso6709",
            "--precision",
            "5",
            "--no-compact",
        )
        self.assertEqual(return_code, 0)
        self.assertEqual(stdout, "+50.12257 +008.66570/\n")
        self.assertEqual(stderr, "")

    def test_convert_accepts_explicit_input_format_and_leading_sign(self) -> None:
        return_code, stdout, stderr = self.invoke(
            "convert",
            "--format",
            "DD",
            "--precision",
            "1",
            "--",
            "-50, -8",
        )

        self.assertEqual(return_code, 0)
        self.assertEqual(stdout, "-50.0, -8.0\n")
        self.assertEqual(stderr, "")

    def test_inspect_emits_complete_deterministic_json(self) -> None:
        arguments = ("inspect", "+50.12257+008.66570/")
        first = self.invoke(*arguments)
        second = self.invoke(*arguments)

        self.assertEqual(first, second)
        return_code, stdout, stderr = first
        self.assertEqual(return_code, 0)
        self.assertEqual(stderr, "")
        self.assertTrue(stdout.endswith("\n"))
        payload = json.loads(stdout)
        self.assertEqual(
            set(payload),
            {
                "candidates",
                "component_formats",
                "evidence",
                "format",
                "latitude_resolution",
                "longitude_resolution",
                "normalizations",
                "normalized_tokens",
                "original_text",
                "position",
                "source_order",
                "warnings",
            },
        )
        self.assertEqual(
            payload["position"],
            {"latitude": 50.12257, "longitude": 8.6657},
        )
        self.assertEqual(payload["format"], "iso6709")
        self.assertEqual(payload["component_formats"], ["iso6709"] * 2)
        self.assertEqual(payload["source_order"], "latlon")
        self.assertEqual(payload["latitude_resolution"], "1/100000")
        self.assertEqual(payload["longitude_resolution"], "1/100000")
        self.assertEqual(
            set(payload["candidates"][0]),
            {
                "evidence",
                "format",
                "outcome",
                "position",
                "reason",
                "source_order",
            },
        )
        self.assertEqual(payload["candidates"][0]["outcome"], "selected")
        self.assertTrue(
            any(
                candidate["outcome"] == "rejected"
                for candidate in payload["candidates"]
            )
        )

    def test_inspect_preserves_unicode_and_normalization_details(self) -> None:
        value = "50,12257 n; 8,66570 e"
        return_code, stdout, stderr = self.invoke("inspect", value)

        self.assertEqual(return_code, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["original_text"], value)
        self.assertEqual(
            payload["normalized_tokens"],
            ["50.12257", "N", ";", "8.66570", "E"],
        )
        self.assertIn(
            "normalized decimal comma",
            payload["normalizations"],
        )

    def test_coordinate_errors_use_argparse_exit_semantics(self) -> None:
        return_code, stdout, stderr = self.invoke(
            "convert",
            "8, 50",
            "--order",
            "auto",
        )

        self.assertEqual(return_code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("could not determine coordinate order", stderr)
        self.assertIn('order="latlon"', stderr)
        self.assertNotIn("Traceback", stderr)

        return_code, stdout, stderr = self.invoke(
            "convert",
            "50, 8",
            "--to",
            "nmea",
            "--precision",
            "0",
        )
        self.assertEqual(return_code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("NMEA precision", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_argparse_help_and_usage_errors(self) -> None:
        return_code, stdout, stderr = self.invoke("--help")
        self.assertEqual(return_code, 0)
        self.assertIn("{convert,inspect}", stdout)
        self.assertEqual(stderr, "")

        return_code, stdout, stderr = self.invoke()
        self.assertEqual(return_code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("the following arguments are required: command", stderr)

        return_code, stdout, stderr = self.invoke(
            "convert",
            "50, 8",
            "--to",
            "unknown",
        )
        self.assertEqual(return_code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("invalid choice", stderr)

    def test_module_entry_point_and_import_boundary(self) -> None:
        import_check = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import nautipy.cli; "
                    "assert 'nautipy.geodesic' not in sys.modules; "
                    "assert 'geographiclib' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(import_check.returncode, 0, import_check.stderr)

        module_help = subprocess.run(
            [sys.executable, "-m", "nautipy", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(module_help.returncode, 0, module_help.stderr)
        self.assertIn("{convert,inspect}", module_help.stdout)
        self.assertEqual(module_help.stderr, "")

    def test_unicode_output_is_utf8_under_a_legacy_stream_encoding(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "cp1252"
        converted = subprocess.run(
            [
                sys.executable,
                "-m",
                "nautipy",
                "convert",
                "50, 8",
                "--to",
                "dms",
            ],
            check=False,
            capture_output=True,
            env=environment,
        )

        self.assertEqual(converted.returncode, 0, converted.stderr)
        self.assertEqual(
            converted.stdout.decode("utf-8"),
            "50° 0′ 0″ N; 8° 0′ 0″ E\n",
        )
        self.assertEqual(converted.stderr, b"")


if __name__ == "__main__":
    unittest.main()
