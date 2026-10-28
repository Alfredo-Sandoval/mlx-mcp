"""The mlx-mcp command: serve the docs to an agent and manage the search index."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from mlx_mcp import __version__
from mlx_mcp.configuration import cache_root, refresh_command


class _CommandOptions(argparse.Namespace):
    command: Literal["serve", "prepare", "info"]
    cache: Path | None = None
    refresh: bool = False


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="mlx-mcp",
        description="Serve the MLX, MLX Data and MLX LM docs to coding agents.",
    )
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True, metavar="COMMAND")
    commands.add_parser(
        "serve",
        help="Run the server for one agent over stdin and stdout. Agents start "
        "this themselves",
    )
    prepare = commands.add_parser(
        "prepare", help="Download the search model and check that search works"
    )
    prepare.add_argument(
        "--refresh",
        action="store_true",
        help="Download the latest MLX and MLX Data docs first",
    )
    info = commands.add_parser("info", help="Show which docs versions mlx-mcp has")
    for command in (prepare, info):
        command.add_argument(
            "--cache",
            type=Path,
            help="Folder for the search model and index (default: MLX_MCP_CACHE "
            "or ~/.cache/mlx-mcp)",
        )
    return result


def _run_command(args: _CommandOptions) -> None:
    if args.command == "serve":
        import asyncio

        from mlx_mcp.server import create_server

        asyncio.run(create_server().run_stdio_async())
        return
    from mlx_mcp.indexing import select_index_directory

    root = (args.cache or cache_root()).expanduser().resolve()
    if args.command == "info":
        from mlx_mcp.service import DocumentationService

        print(
            json.dumps(
                DocumentationService(select_index_directory(root)).documentation_info(),
                indent=2,
                sort_keys=True,
            )
        )
        return
    from mlx_mcp.service_setup import prepare

    result = prepare(root, refresh=args.refresh, progress=_progress)
    print(f"Ready. Search and fetch work on {result.passage_count} sections of docs.")
    print(f"Files are in {result.cache}")


def _failure_hints(command: str, cache: Path | None) -> list[str]:
    """Return next steps for a failed command; cache is its --cache value, if any."""
    if command == "serve":
        return []
    hints = [
        (
            "If the index is from another mlx-mcp version, rebuild it with "
            f"`{refresh_command(cache or cache_root())}`."
        ),
        "If your files are somewhere else, pass --cache or set MLX_MCP_CACHE.",
    ]
    # A failed prepare would only point at itself.
    if command == "info":
        hints.insert(0, "Run `mlx-mcp prepare` to download the search model.")
    return hints


def failure_message(error: BaseException) -> str:
    """Expose nested MCP transport failures at the CLI error-reporting boundary."""
    if isinstance(error, BaseExceptionGroup):
        group = cast(BaseExceptionGroup[BaseException], error)
        return "; ".join(failure_message(item) for item in group.exceptions)
    return str(error)


def main(argv: Sequence[str] | None = None) -> None:
    """Run one mlx-mcp command, reporting failures as short messages on stderr."""
    args = _CommandOptions()
    _parser().parse_args(argv, namespace=args)
    try:
        _run_command(args)
    except Exception as exc:  # CLI boundary: concise, actionable error on stderr.
        message = failure_message(exc)
        hints = _failure_hints(args.command, args.cache)
        _progress(f"Error: {message}")
        for hint in hints:
            _progress(hint)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
