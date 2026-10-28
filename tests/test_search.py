"""DocumentationService.search: returned sources, ranking outcomes, and notices."""

import asyncio
from pathlib import Path

import numpy as np
import pytest
from helpers import (
    CORE,
    FakeReranker,
    fixed_order_service,
    service_in_memory,
)

from mlx_mcp.packages import (
    Package,
)
from mlx_mcp.passages import Passage
from mlx_mcp.responses import (
    source_blocks,
)
from mlx_mcp.service import DocumentationService


def test_search_fills_unique_section_slots_without_losing_selected_chunks() -> None:
    passages = [
        Passage("A", "https://docs/a#one", "First fact."),
        Passage("A", "https://docs/a#one", "Second fact."),
        Passage("B", "https://docs/b#two", "Other fact."),
    ]
    index = service_in_memory(passages)
    index.retrieve = lambda query, language="all", package=None: passages
    index.score_candidates = lambda query, passages: np.array([3.0, 2.0, 1.0])

    result = asyncio.run(index.search("facts", 2))
    assert result.count("Source:") == 2
    assert result.count("https://docs/a#one") == 1
    assert all(p.text in result for p in passages)


@pytest.mark.parametrize("member", ["strides", "Strides"])
def test_missing_member_is_distinguished_from_related_search_results(
    member: str,
) -> None:
    passages = [
        Passage(
            "mlx.core.array.astype", "https://docs/python/astype", "Cast an array."
        ),
        Passage(
            "mlx.core.as_strided", "https://docs/python/as_strided", "Strided views."
        ),
    ]
    index = fixed_order_service(passages)
    result = asyncio.run(index.search(f"array.{member} Python", limit=1))
    assert f"The indexed docs don't define `array.{member}`" in result
    assert "never mention" not in result
    assert "may still exist in the version you have installed" in result
    assert passages[0].text in result
    assert index.index.symbol_index.missing_definitions(
        "MLX array strides property contiguous transpose strides",
    ) == ["array.strides"]
    assert not index.index.symbol_index.missing_definitions(
        "mx.array.astype np.asarray https://example.org"
    )


@pytest.mark.parametrize(
    "query",
    ["mlx_lm.generate with prompt caching and sampling", "MLX-LM prompt caching"],
)
def test_external_package_requests_explain_the_index_boundary(query: str) -> None:
    passage = Passage(
        "Llama", "https://docs/examples/llama", "Core generation example."
    )
    index = service_in_memory([passage])
    index.retrieve = lambda query, language="all", package=None: [passage]
    result = asyncio.run(index.search(query))
    assert "mlx-mcp has no docs for `mlx_lm`" in result
    assert "It covers MLX." in result
    assert passage.text in result
    assert "external package" not in asyncio.run(
        index.search("prompt caching with MLX")
    )


def test_ranking_preserves_full_question_evidence() -> None:
    passages = [
        Passage("mlx.core.eval", "https://docs/eval", "Evaluate arrays."),
        Passage(
            "Indexing", "https://docs/indexing", "Slicing indexed updates add values."
        ),
        Passage("Dtypes", "https://docs/types", "Casting values changes types."),
        Passage(
            "Lazy evaluation", "https://docs/lazy", "Saving arrays evaluates them."
        ),
    ]
    query = "MLX slicing indexed updates casting lazy evaluation mx.eval saving"
    index = service_in_memory(passages)
    index.retrieve = lambda query, language="all", package=None: passages
    index.models.reranker = FakeReranker(
        lambda question, documents: (
            [1.0, 3.0, 2.0, 5.0] if query in question else [0.0, 6.0, 4.0, 1.0]
        )
    )
    result = asyncio.run(index.search(query, limit=3))
    assert all(p.text in result for p in [passages[0], passages[1], passages[3]])
    assert passages[2].text not in result


def test_search_flags_names_that_no_indexed_source_mentions() -> None:
    index = fixed_order_service(
        [Passage("Streams", "https://docs/streams", "Pass np.asarray inputs.")]
    )
    result = asyncio.run(
        index.search("Does cudaMemcpyAsync overlap work like torch.compile.")
    )
    assert (
        "The indexed docs never mention `cudaMemcpyAsync`, `torch.compile`." in result
    )
    assert "don't document those names" in result
    assert "Streams" in result
    for query in (
        "Is np.asarray zero-copy?",
        "Load train.jsonl and adapters.safetensors",
        "mx.unknown_function and mlx_lm.unknown",
        "Explain streams.",
    ):
        assert "never mention" not in asyncio.run(index.search(query))


def test_search_distinguishes_unsupported_query_from_empty_filters() -> None:
    passage = Passage("Evaluation", "https://docs/python/eval", "Evaluate arrays.")
    index = service_in_memory([passage])
    try:
        assert asyncio.run(index.search("watershed")) == (
            "Nothing in the docs matches this query for the package and language you chose."
        )
        assert asyncio.run(index.search("Evaluate", language="cpp")) == (
            "No docs match the package and language you chose."
        )
        assert asyncio.run(index.search("Evaluate", package=Package.LM)) == (
            "No docs match the package and language you chose."
        )
        assert passage.text in asyncio.run(index.search("Evaluate"))
    finally:
        index.close()


def test_search_returns_selected_passage_with_bounded_section_context() -> None:
    passage = Passage(
        "mlx.core.demo",
        "https://docs/python/demo#demo",
        "The selected answer.",
        "# mlx.core.demo\n\n demo(a, flag=False)\n:   Keep the axis.\n\nThe selected answer.",
    )
    other = Passage("Other", "https://docs/other", "Other answer")
    index = service_in_memory([passage, other])
    index.retrieve = lambda query, language="all", package=None: [passage, other]
    seen_documents: list[str] = []
    index.models.reranker = FakeReranker(
        lambda query, documents: seen_documents.extend(documents) or [1.0, 0.0],
    )

    result = asyncio.run(index.search("demo"))

    assert "demo(a, flag=False)" in result
    assert "The selected answer." in result
    assert result.count("Source:") == 2
    assert all("demo(a, flag=False)" not in document for document in seen_documents)


def test_limit_changes_only_selected_source_count() -> None:
    passages = [Passage(str(i), f"https://docs/{i}", "Fact") for i in range(7)]
    index = fixed_order_service(passages)
    index.score_candidates = lambda query, passages: np.array(
        [2.0, 4.0, 6.0, 1.0, 0.0, -1.0, 5.0]
    )
    results = [
        asyncio.run(index.search("ordinary request", limit=n)) for n in (1, 2, 6)
    ]

    def first(result: str) -> str:
        return next(line for line in result.splitlines() if line.startswith("Source:"))

    assert {first(result) for result in results} == {"Source: https://docs/2"}


def test_over_long_query_fails_before_reranking_whatever_the_candidate_count(
    mixed_index: DocumentationService,
) -> None:
    reranked: list[str] = []

    def rerank(query: str, texts: list[str]) -> range:
        reranked.append(query)
        return range(len(texts))

    mixed_index.models.reranker = FakeReranker(rerank)
    long_query = "word " * 300
    # Two candidates, one candidate (never reranked before), and none: the same
    # query used to succeed or fail with the number of results it found.
    for package in (None, Package.DATA, Package.LM):
        with pytest.raises(
            ValueError, match="Query too long.*the limit is 256. Shorten the query"
        ):
            asyncio.run(mixed_index.search(long_query, package=package))
    assert reranked == []
    assert "prefetch" in asyncio.run(
        mixed_index.search("prefetch samples", package=Package.DATA)
    )


def test_reranker_promotes_relevant_results_and_keeps_stable_score_ties(
    root: Path,
) -> None:
    index = DocumentationService(root)
    index.retrieve = lambda query, language="all", package=None: index.index.passages
    index.models.reranker = FakeReranker(lambda query, texts: iter([0.0, 1.0, 1.0]))
    result = asyncio.run(index.search("question", limit=3))
    assert result.count("Source:") == 3
    assert result.index("api#1") < result.index("api#2") < result.index("api#0")


def _search_names(
    index: DocumentationService,
    query: str,
    limit: int = 3,
    package: Package | None = None,
) -> str:
    index.models.reranker = FakeReranker(lambda query, texts: range(len(texts)))
    return asyncio.run(index.search(query, limit, package=package))


def test_one_api_name_returns_its_definition_first(
    navigation_index: DocumentationService,
) -> None:
    first = source_blocks(_search_names(navigation_index, "mx.astype"))[0]
    assert first.url == CORE + "python/astype.html#astype"
    assert "astype(a, dtype)" in first.text
    assert "Documentation version: 0.32.2" in _search_names(
        navigation_index, "mx.astype"
    )
    data = source_blocks(_search_names(navigation_index, "dx.Buffer.prefetch"))
    assert "prefetch(size)" in data[0].text
    # MLX LM's Python API leads its command-line help of the same name.
    lm = source_blocks(_search_names(navigation_index, "mlx_lm.generate"))
    assert "API ONLY" in lm[0].text
    other = _search_names(navigation_index, "mx.astype", package=Package.DATA)
    assert "astype(a, dtype)" not in other


def test_an_ambiguous_api_name_says_how_many_definitions_match(
    navigation_index: DocumentationService,
) -> None:
    one = _search_names(navigation_index, "astype", limit=1)
    assert "2 definitions match `astype`, and this shows 1." in one
    both = _search_names(navigation_index, "astype")
    assert "astype(a, dtype)" in both and "array.astype(dtype)" in both
    assert "definitions match `" not in both
