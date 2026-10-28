"""Download what search needs and check that search and fetch work.

`prepare` builds an index when asked or when none is usable, downloads the
reranker if it is missing, then runs one search and one fetch.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mlx_mcp import indexing
from mlx_mcp.configuration import refresh_command
from mlx_mcp.indexing import select_index_directory
from mlx_mcp.responses import source_blocks
from mlx_mcp.service import DocumentationService


@dataclass(frozen=True)
class Readiness:
    """Evidence produced only after the reranker, search, and fetch succeed."""

    cache: Path
    passage_count: int


def check_readiness(root: Path, *, progress: Callable[[str], None]) -> Readiness:
    """Check the reranker, search, and fetch; release the reranker even on failure."""
    selected = select_index_directory(root)
    progress(f"Checking the index in {selected}")
    index = DocumentationService(selected, model_cache=root)
    try:
        progress("Loading the search model...")
        query = "How do I evaluate a lazy array?"
        index.score_candidates(query, index.index.passages[:1])
        progress("Trying a search and a fetch...")
        result = asyncio.run(index.search(query))
        blocks = source_blocks(result)
        if not blocks or not any(block.text.strip() for block in blocks):
            # Name root, not selected: when the bundled index was selected, the
            # rebuilt index belongs in this cache, where it then takes precedence.
            raise ValueError(
                "A test search found no docs. Rebuild the index with "
                f"`{refresh_command(root)}`."
            )
        url = next(block.url for block in blocks if block.text.strip())
        if not index.fetch(url).strip():
            raise ValueError("A test fetch returned no text.")
        return Readiness(root, len(index.index.passages))
    finally:
        index.close()


def prepare(root: Path, *, refresh: bool, progress: Callable[[str], None]) -> Readiness:
    """Build an index when refresh is set or none is usable, then check readiness.

    A missing reranker is downloaded.
    """
    selected = None
    if not refresh:
        try:
            selected = select_index_directory(root)
        except FileNotFoundError:
            refresh = True
    if refresh:
        progress(
            "Downloading the latest MLX and MLX Data docs and the bundled MLX LM "
            "version, then building the index..."
        )
        indexing.refresh(root, model_cache=root)
    else:
        kind = "bundled" if selected == indexing.BUNDLED_ROOT else "refreshed"
        progress(
            f"Using the {kind} index. Add --refresh to download the latest MLX and "
            "MLX Data docs."
        )
    return check_readiness(root, progress=progress)
