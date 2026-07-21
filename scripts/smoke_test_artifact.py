"""Install a NautiPy distribution artifact and exercise its public API."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import venv


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python scripts/smoke_test_artifact.py DIST.whl|DIST.tar.gz"
        )

    artifact = Path(sys.argv[1]).resolve()
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
        if is_wheel:
            install_command.append("--no-index")
        install_command.append(str(artifact))
        subprocess.run(install_command, check=True)
        subprocess.run(
            [
                str(executable),
                "-c",
                (
                    "from importlib.metadata import requires; import nautipy; "
                    "position = nautipy.parse_position('50.12257, 8.66570'); "
                    "assert nautipy.format_position(position) == "
                    "'50.12257, 8.6657'; "
                    "assert not hasattr(nautipy, 'Pos'); "
                    "assert not requires('nautipy')"
                ),
            ],
            check=True,
            cwd=environment,
        )


if __name__ == "__main__":
    main()
