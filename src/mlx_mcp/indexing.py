"""Select the index to serve, and build or refresh it."""

import logging
import sys
from pathlib import Path

from mlx_mcp import __version__
from mlx_mcp.configuration import cache_root, refresh_command
from mlx_mcp.index_file import (
    INDEX_VERSION,
    read_index_identity,
    write_index,
)
from mlx_mcp.model_runtime import load_rerank_tokenizer
from mlx_mcp.packages import Package
from mlx_mcp.passages import Passage, validate_passages

# Release wheels ship a prebuilt index here.
BUNDLED_ROOT = Path(__file__).parent / "data"

logger = logging.getLogger(__name__)
# Local indexes already reported as stale; select_index_directory runs more than once per command.
_reported_stale: set[Path] = set()


def select_index_directory(local: Path | None = None) -> Path:
    """Prefer an index refreshed by this release, then the bundled index.

    A local index with another format or builder release is ignored with one
    warning per process. Raise FileNotFoundError when neither index is current.
    """
    local = local if local is not None else cache_root()
    identity = None
    if (local / "index.npz").is_file():
        identity = read_index_identity(local)
        if identity == (INDEX_VERSION, __version__):
            return local
    bundled = (BUNDLED_ROOT / "index.npz").is_file() and read_index_identity(
        BUNDLED_ROOT
    ) == (INDEX_VERSION, __version__)
    if identity is not None:
        version, built_by = identity
        if local not in _reported_stale:
            _reported_stale.add(local)
            logger.warning(
                "Skipping %s/index.npz. mlx-mcp %s built it in format %s, and "
                "this is mlx-mcp %s, which reads format %s.%s "
                "Run `%s` to rebuild it, or delete it.",
                local,
                built_by or "unknown",
                version,
                __version__,
                INDEX_VERSION,
                " Using the bundled index instead." if bundled else "",
                refresh_command(local),
            )
    if bundled:
        return BUNDLED_ROOT
    raise FileNotFoundError(
        f"No usable MLX docs index. Rebuild it with `{refresh_command(local)}`."
    )


def build_index(
    root: Path, passages: list[Passage], *, model_cache: Path | None = None
) -> None:
    """Split passages into ranking windows, then atomically write root/index.npz.

    Windows are measured with the reranker's tokenizer, loaded from model_cache,
    which defaults to cache_root().
    """
    validate_passages(passages)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = load_rerank_tokenizer(
        model_cache if model_cache is not None else cache_root()
    )
    write_index(root, tokenizer.prepare_index_windows(passages))


def collect_passages() -> list[Passage]:
    """Download every package's sources."""
    # The server imports this module to select an index; keep download and
    # parsing dependencies out of its startup.
    from mlx_mcp.downloads import download_package
    from mlx_mcp.mlx_lm_docs import download_lm

    return [
        passage
        for package in Package
        if package.sphinx
        for passage in download_package(package)
    ] + download_lm()


def refresh(root: Path, *, model_cache: Path | None = None) -> None:
    """Rebuild root/index.npz from the published docs and pinned MLX LM sources.

    The reranker's tokenizer loads from model_cache, which defaults to
    cache_root().
    """
    passages = collect_passages()
    print(f"Indexing {len(passages)} passages...", file=sys.stderr)
    build_index(root, passages, model_cache=model_cache)
    print(f"Wrote {root / 'index.npz'}", file=sys.stderr)
