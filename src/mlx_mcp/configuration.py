"""Defaults and the environment variables that override them."""

import os
import shlex
from pathlib import Path

DEFAULT_SEARCH_LIMIT = 3
# search_mlx_docs returns at most this many sections.
MAX_SEARCH_LIMIT = 6

CACHE_VARIABLE = "MLX_MCP_CACHE"


def cache_root() -> Path:
    """Return MLX_MCP_CACHE, defaulting to ~/.cache/mlx-mcp."""
    return Path(os.environ.get(CACHE_VARIABLE, "~/.cache/mlx-mcp")).expanduser()


def refresh_command(index_directory: Path) -> str:
    """Return the shell command that rebuilds the index stored in index_directory.

    `mlx-mcp prepare --refresh` writes to MLX_MCP_CACHE (default
    ~/.cache/mlx-mcp) unless given --cache. An agent can start mlx-mcp with a different MLX_MCP_CACHE
    than the user's shell, so any cache directory is named in full. The index
    shipped inside this package is read-only; a rebuilt cache index takes
    precedence over it, so the plain command is right for that one.
    """
    directory = index_directory.expanduser().resolve()
    if directory.is_relative_to(Path(__file__).resolve().parent):
        return "mlx-mcp prepare --refresh"
    return f"mlx-mcp prepare --refresh --cache {shlex.quote(str(directory))}"
