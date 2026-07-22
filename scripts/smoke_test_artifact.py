"""Install a NautiPy distribution artifact and exercise its public API."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import venv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install and smoke-test one NautiPy distribution artifact."
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="install the 'fix' extra and exercise the optional solver",
    )
    parser.add_argument("artifact", type=Path, metavar="DIST.whl|DIST.tar.gz")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact = args.artifact.resolve()
    is_wheel = artifact.suffix == ".whl"
    is_sdist = artifact.name.endswith(".tar.gz")
    if not artifact.is_file() or not (is_wheel or is_sdist):
        raise SystemExit(f"unsupported distribution artifact: {artifact}")

    with TemporaryDirectory(prefix="nautipy-artifact-") as directory:
        environment = Path(directory)
        venv.EnvBuilder(with_pip=True).create(environment)
        executable = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        install_command = [str(executable), "-m", "pip", "install"]
        install_target = f"{artifact}[fix]" if args.fix else str(artifact)
        install_command.append(install_target)
        subprocess.run(install_command, check=True)
        subprocess.run(
            [str(executable), "-m", "pip", "check"],
            check=True,
            cwd=environment,
        )
        subprocess.run(
            [
                str(executable),
                "-c",
                (
                    "from importlib.metadata import distributions, requires, version; "
                    "import sys; import nautipy; "
                    f"fix_mode = {args.fix!r}; "
                    "scientific = {'numpy', 'scipy'}; "
                    "installed = {distribution.metadata.get('Name', '').lower() "
                    "for distribution in distributions()}; "
                    "assert (scientific <= installed if fix_mode else "
                    "scientific.isdisjoint(installed)); "
                    "assert scientific.isdisjoint(sys.modules); "
                    "from nautipy.fix import (BearingObservation, "
                    "RangeObservation, solve_fix); "
                    "bearing_model = BearingObservation((0, 0), 90, 1); "
                    "range_model = RangeObservation((0, 0), 100, 1); "
                    "assert bearing_model.reference == nautipy.Position(0, 0); "
                    "assert range_model.reference == nautipy.Position(0, 0); "
                    "assert callable(solve_fix); "
                    "assert scientific.isdisjoint(sys.modules); "
                    "from nautipy.geojson import ("
                    "from_geojson_feature_collection, "
                    "to_geojson_feature_collection, to_geojson_point); "
                    "assert version('nautipy'); "
                    "assert 'nautipy.geodesic' not in sys.modules; "
                    "assert 'geographiclib' not in sys.modules; "
                    "position = nautipy.parse_position('50.12257, 8.66570'); "
                    "assert nautipy.format_position(position) == "
                    "'50.122570, 8.665700'; "
                    "human = nautipy.parse_position("
                    "\"50° 7' 21.252\\\" N; 8° 39' 56.52\\\" E\"); "
                    "assert human == position; "
                    "iso = nautipy.parse_position("
                    "'+50.12257+008.66570/'); "
                    "nmea = nautipy.parse_position("
                    "'5007.3542,N,00839.9420,E'); "
                    "geojson = nautipy.parse_position("
                    "{'type': 'Point', 'coordinates': [8.66570, 50.12257]}); "
                    "assert iso == nmea == geojson == position; "
                    "inspection = nautipy.inspect_position("
                    "'5007.3542,N,00839.9420,E'); "
                    "assert inspection.format == 'nmea'; "
                    "assert inspection.position == position; "
                    "converted = nautipy.convert_position("
                    "'50.12257, 8.66570', to='dms'); "
                    "assert converted == "
                    "'50° 7′ 21.25″ N; 8° 39′ 56.52″ E'; "
                    "assert nautipy.format_position("
                    "position, to='iso6709') == "
                    "'+50.122570+008.665700/'; "
                    "assert to_geojson_point(position) == {"
                    "'type': 'Point', 'coordinates': [8.6657, 50.12257]}; "
                    "labeled = nautipy.Position("
                    "50.12257, 8.66570, identifier='station-1', "
                    "description='Reference station'); "
                    "collection = to_geojson_feature_collection([labeled]); "
                    "restored = from_geojson_feature_collection(collection); "
                    "assert restored == (labeled,); "
                    "assert restored[0].identifier == 'station-1'; "
                    "assert restored[0].description == 'Reference station'; "
                    "assert 'nautipy.geodesic' not in sys.modules; "
                    "assert 'geographiclib' not in sys.modules; "
                    "end = nautipy.destination("
                    "position, bearing=90, distance=12000); "
                    "result = nautipy.inverse(position, end); "
                    "assert abs(result.distance - 12000) < 1e-6; "
                    "assert abs(nautipy.initial_bearing(position, end) "
                    "- 90) < 1e-10; "
                    "assert nautipy.interpolate(position, end); "
                    "assert nautipy.nearest_position("
                    "position, [end, position]) == position; "
                    "assert 'nautipy.geodesic' in sys.modules; "
                    "assert 'geographiclib.geodesic' in sys.modules; "
                    "assert fix_mode or scientific.isdisjoint(sys.modules); "
                    "assert not hasattr(nautipy, 'Pos'); "
                    "requirements = requires('nautipy') or []; "
                    "normalized = sorted(requirement.replace(' ', '').replace("
                    "\"'\", '\"') for requirement in requirements); "
                    "base = [requirement for requirement in normalized "
                    "if ';extra==' not in requirement]; "
                    "fix = [requirement for requirement in normalized "
                    "if ';extra==' in requirement]; "
                    "assert base == ['geographiclib>=2.1']; "
                    "assert fix == ["
                    "'numpy>=1.23.5;extra==\"fix\"', "
                    "'scipy>=1.14.1;extra==\"fix\"']"
                ),
            ],
            check=True,
            cwd=environment,
        )

        if args.fix:
            subprocess.run(
                [
                    str(executable),
                    "-c",
                    (
                        "from nautipy import (Position, distance, "
                        "initial_bearing); "
                        "from nautipy.fix import (BearingObservation, "
                        "RangeObservation, solve_fix); "
                        "target = Position(0.01, 0.01); "
                        "stations = (Position(0, 0), Position(0, 0.02), "
                        "Position(0.02, 0)); "
                        "bearings = tuple(BearingObservation("
                        "station, initial_bearing(target, station), "
                        "uncertainty=1.0) for station in stations); "
                        "ranges = tuple(RangeObservation("
                        "station, distance(station, target), "
                        "uncertainty=1.0) for station in stations); "
                        "result = solve_fix(bearings=bearings, ranges=ranges); "
                        "assert result.success; "
                        "assert abs(result.position.latitude - "
                        "target.latitude) < 1e-8; "
                        "assert abs(result.position.longitude - "
                        "target.longitude) < 1e-8; "
                        "import sys; "
                        "assert {'numpy', 'scipy'} <= set(sys.modules)"
                    ),
                ],
                check=True,
                cwd=environment,
            )

        command = environment / (
            "Scripts/nautipy.exe" if sys.platform == "win32" else "bin/nautipy"
        )
        converted = subprocess.run(
            [
                str(command),
                "convert",
                "50° 7.3542' N; 8° 39.942' E",
                "--to",
                "dd",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=environment,
        )
        if converted.returncode != 0:
            raise RuntimeError(converted.stderr)
        if converted.stdout != "50.122570, 8.665700\n":
            raise RuntimeError(f"unexpected CLI conversion: {converted.stdout!r}")
        if converted.stderr:
            raise RuntimeError(f"unexpected CLI stderr: {converted.stderr!r}")

        inspected = subprocess.run(
            [str(command), "inspect", "+50.12257+008.66570/"],
            check=False,
            capture_output=True,
            text=True,
            cwd=environment,
        )
        if inspected.returncode != 0:
            raise RuntimeError(inspected.stderr)
        inspection = json.loads(inspected.stdout)
        if inspection["format"] != "iso6709":
            raise RuntimeError(f"unexpected CLI inspection: {inspection!r}")
        if inspection["latitude_resolution"] != "1/100000":
            raise RuntimeError(f"unexpected CLI resolution: {inspection!r}")

        invalid = subprocess.run(
            [str(command), "convert", "8, 50", "--order", "auto"],
            check=False,
            capture_output=True,
            text=True,
            cwd=environment,
        )
        if invalid.returncode != 2 or invalid.stdout:
            raise RuntimeError("malformed CLI input did not fail cleanly")
        if "coordinate order" not in invalid.stderr or "Traceback" in invalid.stderr:
            raise RuntimeError(f"unexpected CLI diagnostic: {invalid.stderr!r}")


if __name__ == "__main__":
    main()
