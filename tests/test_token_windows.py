"""Token-bounded windows over canonical passages."""

from dataclasses import replace
from pathlib import Path

from helpers import fake_cross_encoder

from mlx_mcp.passages import source_text
from mlx_mcp.reranking import WindowedReranker
from mlx_mcp.sphinx_pages import parse_page


def test_windows_keep_complete_link_lines_and_rewindowing_repairs_old_boundaries(
    tmp_path: Path,
) -> None:

    model = WindowedReranker(fake_cross_encoder(tmp_path))
    html = (
        '<article class="bd-article"><ul>'
        + "".join(
            f'<li><a href="https://docs/long_link_{i}">word word word {i}</a></li>'
            for i in range(1000)
        )
        + "</ul></article>"
    )
    original = parse_page(html, "https://docs/catalog")
    windows = model.prepare_index_windows(original)
    assert len(windows) > 1
    assert all(
        p.text.startswith("- [") and p.text.rstrip().endswith(")") for p in windows
    )
    assert source_text(windows) == source_text(original)
    # A previous index may have cut inside a link; canonical segments restore it.
    first = original[0]
    midpoint = len(first.text) // 2
    old = [
        replace(first, text=first.text[:midpoint], chunk_end=midpoint),
        replace(first, text=first.text[midpoint:], chunk_start=midpoint),
    ]
    assert model.prepare_index_windows(old) == windows
