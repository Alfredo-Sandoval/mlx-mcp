"""DocumentationService construction, documentation info, and lifecycle."""

import asyncio
from pathlib import Path

import pytest

from mlx_mcp.index_file import load_index, write_index
from mlx_mcp.passages import Passage
from mlx_mcp.service import DocumentationService, parse_search_limit
from mlx_mcp.sphinx_pages import parse_page


def test_info_reports_each_packages_own_version_and_counts(
    mixed_index: DocumentationService,
) -> None:
    info = mixed_index.documentation_info()
    assert info["packages"] == {
        "mlx": {
            "documentation_versions": ["0.32.2"],
            "page_count": 1,
            "passage_count": 1,
            "evidence_kinds": ["published documentation"],
            "source_revisions": [],
        },
        "mlx-data": {
            "documentation_versions": ["0.0.2"],
            "page_count": 1,
            "passage_count": 1,
            "evidence_kinds": ["published documentation"],
            "source_revisions": [],
        },
    }


def test_page_provenance_survives_the_snapshot(tmp_path: Path) -> None:
    html = (
        "<title>Guide — MLX 0.32.2 documentation</title>"
        '<article class="bd-article"><h1>Guide</h1><p>Source.</p></article>'
    )
    timestamp = "2026-09-14T01:00:00+00:00"
    passages = parse_page(html, "https://docs/guide", retrieved_at=timestamp)
    assert passages[0].documentation_version == "0.32.2"
    assert passages[0].retrieved_at == timestamp
    write_index(tmp_path, passages)
    assert load_index(tmp_path).passages == passages
    info = DocumentationService(tmp_path).documentation_info()
    packages = info["packages"]
    assert isinstance(packages, dict)
    assert packages["mlx"]["documentation_versions"] == ["0.32.2"]
    assert info["retrieved_at_first"] == info["retrieved_at_last"] == timestamp
    assert info["pages_without_version"] == 0


def test_mixed_and_unknown_provenance_remain_visible(tmp_path: Path) -> None:
    passages = [
        Passage("A", "https://docs/a#one", "A", documentation_version="0.32.1"),
        Passage("A", "https://docs/a#two", "B", documentation_version="0.32.1"),
        Passage("B", "https://docs/b", "B", documentation_version="0.32.2"),
        Passage("C", "https://docs/c", "C"),
    ]
    write_index(tmp_path, passages)
    info = DocumentationService(tmp_path).documentation_info()
    packages = info["packages"]
    assert isinstance(packages, dict)
    assert packages["mlx"]["documentation_versions"] == ["0.32.1", "0.32.2"]
    assert info["page_count"] == 3 and info["passage_count"] == 4
    assert info["pages_without_version"] == 1
    assert info["retrieved_at_first"] is None and info["retrieved_at_last"] is None
    unknown = parse_page(
        '<article class="bd-article"><p>Source.</p></article>', "https://docs/a"
    )
    assert unknown[0].documentation_version == unknown[0].retrieved_at == ""


@pytest.mark.parametrize("limit", [0, True])
def test_direct_search_calls_validate_limit(root: Path, limit: float) -> None:
    with pytest.raises(ValueError, match="limit"):
        parse_search_limit(limit)
    if isinstance(limit, int):
        with pytest.raises(ValueError, match="limit"):
            asyncio.run(DocumentationService(root).search("question", limit))
