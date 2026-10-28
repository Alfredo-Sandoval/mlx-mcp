"""Passage validation, chunking, segments, and surrounding context."""

from dataclasses import replace

import pytest
from helpers import (
    REPEATED_PAGE,
    data_passage,
)

from mlx_mcp.packages import Package
from mlx_mcp.passages import (
    Passage,
    expand_passage,
    source_text,
    split_text,
    validate_passages,
)
from mlx_mcp.responses import render_sections
from mlx_mcp.sphinx_pages import parse_page


def test_code_block_is_never_cut() -> None:
    code = "```python\n" + "mx.eval(x)\n" * 250 + "```"
    chunks = split_text("Intro\n\n" + code + "\n\nTail")
    assert code in chunks
    assert chunks[-1] == "Tail"
    assert all(split_text(code + "\n\n" + code))


def test_package_identity_cannot_disagree_with_official_source() -> None:
    with pytest.raises(ValueError, match="package must match"):
        replace(data_passage(), package=Package.MLX)


def test_expand_passage_adds_bounded_source_context_after_selection() -> None:
    source = (
        "# mlx.core.demo\n\n"
        "demo(a, flag=False)\n:   Keep the final axis.\n\n"
        "The signature controls the output.\n\n"
        "The selected paragraph answers the question.\n\n"
        "A nearby example shows the same call."
    )
    passage = Passage(
        "mlx.core.demo",
        "https://docs/python/demo",
        "The selected paragraph answers the question.",
        source,
    )

    rendered = expand_passage(passage)

    assert passage.text in rendered
    assert "demo(a, flag=False)" in rendered
    assert "Keep the final axis." in rendered
    assert "A nearby example shows the same call." in rendered
    assert len(rendered) <= 6000


def test_chunking_keeps_tail_and_bounds_excerpts() -> None:
    text = "\n\n".join(f"line {i}: " + "x" * 80 for i in range(100))
    chunks = split_text(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 1800 for chunk in chunks)
    assert "line 99:" in chunks[-1]
    assert all(f"line {i}:" in "\n".join(chunks) for i in range(100))


def test_source_ranges_reject_corruption_and_preserve_repeated_identical_chunks() -> (
    None
):
    passage = parse_page(
        '<article class="bd-article"><p>Same.</p><p>Same.</p></article>', REPEATED_PAGE
    )[0]
    source = passage.segment_text
    first = replace(passage, text="Same.", chunk_start=0, chunk_end=5)
    second = replace(passage, text="Same.", chunk_start=7, chunk_end=12)
    validate_passages([first, second])
    assert source_text([second, first, first]) == source
    assert render_sections([second, first, first], 1).count("Same.") == 2
    for bad in [
        replace(second, chunk_start=6),
        replace(second, section_urls=("https://other",)),
        replace(second, section_path=("wrong",)),
    ]:
        with pytest.raises(ValueError):
            validate_passages([first, bad])
    with pytest.raises(ValueError, match="uncovered tail"):
        validate_passages([first])
