"""Capture mlx-lm's `--help` output once per pinned release and save it as JSON.

Run on an Apple silicon Mac; mlx-lm only declares its `mlx` dependency there.
The script installs the wheel into a throwaway venv, runs `--help` for every
console script, and writes the output
with the versions and platform it ran on. Index builds read that file, so they
never need mlx-lm installed.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mlx_mcp.mlx_lm_sources import LM_SOURCE
from mlx_mcp.packages import Package

COMMAND_TIMEOUT_SECONDS = 60
# lm-eval and tqdm are imported unconditionally by evaluate/share/quant modules,
# even though the base wheel only requires them for the "evaluate" extra.
EXTRAS = ["evaluate"]


@dataclass(frozen=True)
class HelpRun:
    """One `--help` run: its argv, exit code and output.

    The package's mlx_lm_docs.CapturedCommand is the checked, citable record the
    index reads back from the saved JSON; this is the raw run before saving.
    """

    name: str
    argv: list[str]
    exit_code: int
    help: str
    stderr: str


def console_script_names(wheel: Path) -> list[str]:
    """Read the console_scripts entry-point names the wheel declares."""
    with zipfile.ZipFile(wheel) as archive:
        entry_points_path = next(
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/entry_points.txt")
        )
        text = archive.read(entry_points_path).decode()
    names: list[str] = []
    in_console_scripts = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_console_scripts = stripped == "[console_scripts]"
            continue
        if in_console_scripts and "=" in stripped:
            names.append(stripped.split("=", 1)[0].strip())
    if not names:
        raise ValueError(f"No [console_scripts] entries found in {wheel}")
    return names


def run_venv(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )


def create_environment(scratch: Path, wheel: Path) -> Path:
    """Install the wheel into a throwaway venv and return the venv directory."""
    venv_dir = scratch / "venv"
    created = subprocess.run(
        ["uv", "venv", "--python", "3.12", str(venv_dir)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if created.returncode != 0:
        raise RuntimeError(f"uv venv failed:\n{created.stdout}\n{created.stderr}")
    python = venv_dir / "bin" / "python"
    # uv requires the canonical `name-version-...whl` filename to resolve extras;
    # the local copy may have any name.
    canonical_wheel = scratch / LM_SOURCE.wheel_filename
    canonical_wheel.write_bytes(wheel.read_bytes())
    extras = f"[{','.join(EXTRAS)}]" if EXTRAS else ""
    installed = subprocess.run(
        ["uv", "pip", "install", "--python", str(python), f"{canonical_wheel}{extras}"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if installed.returncode != 0:
        raise RuntimeError(
            f"uv pip install failed:\n{installed.stdout}\n{installed.stderr}"
        )
    return venv_dir


def installed_mlx_version(venv_dir: Path) -> str:
    result = run_venv(
        [
            str(venv_dir / "bin" / "python"),
            "-c",
            "import mlx.core as mx; print(mx.__version__)",
        ]
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"Could not read the installed mlx version:\n{result.stderr}"
        )
    return result.stdout.strip()


def capture_help(venv_dir: Path, argv: list[str]) -> HelpRun:
    """Run one installed command offline at 100 columns and keep its output."""
    executable = venv_dir / "bin" / argv[0]
    if not executable.exists():
        raise RuntimeError(
            f"Console script not installed in the throwaway venv: {argv[0]}"
        )
    full_argv = [str(executable), *argv[1:]]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "COLUMNS": "100"}
    try:
        result = subprocess.run(
            full_argv,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"`{' '.join(full_argv)}` timed out") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(full_argv)}` exited {result.returncode}:\n{result.stderr}"
        )
    if not result.stdout.strip():
        raise RuntimeError(f"`{' '.join(full_argv)}` printed no --help text on stdout")
    # Record the portable invocation (console-script name), not the throwaway
    # venv's absolute executable path.
    return HelpRun(argv[0], argv, result.returncode, result.stdout, result.stderr)


def capture_all(wheel: Path, output: Path) -> None:
    """Capture --help for every console script, including the `mlx_lm` dispatcher.

    `mlx_lm <subcommand>` forwards to these same scripts, so subcommand help is
    not captured a second time through the dispatcher. The index reads only
    "commands".
    """
    scripts = console_script_names(wheel)
    with tempfile.TemporaryDirectory(prefix="mlx-lm-help-capture-") as scratch:
        venv_dir = create_environment(Path(scratch), wheel)
        mlx_version = installed_mlx_version(venv_dir)
        commands = [capture_help(venv_dir, [name, "--help"]) for name in scripts]
    warnings = {c.name: c.stderr for c in commands if c.stderr.strip()}
    capture = {
        "schema_version": 1,
        "package": Package.LM.value,
        "version": LM_SOURCE.version,
        "wheel_filename": LM_SOURCE.wheel_filename,
        "captured_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "mlx_version": mlx_version,
        "commands": [
            {"name": c.name, "argv": c.argv, "exit_code": c.exit_code, "help": c.help}
            for c in commands
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(capture, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Captured {len(commands)} console scripts -> {output}",
        file=sys.stderr,
    )
    if warnings:
        print(f"Commands with non-empty stderr: {sorted(warnings)}", file=sys.stderr)


class CaptureOptions(argparse.Namespace):
    wheel: Path
    output: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        type=Path,
        required=True,
        help="Path to the pinned mlx-lm wheel",
    )
    parser.add_argument("--output", type=Path, default=LM_SOURCE.capture_file)
    args = CaptureOptions()
    parser.parse_args(namespace=args)
    capture_all(args.wheel, args.output)


if __name__ == "__main__":
    main()
