"""Pinned model loading, reranker output checks, and model unloading."""

import asyncio
import weakref
from pathlib import Path

import huggingface_hub
import pytest
from helpers import FakeReranker, fake_cross_encoder
from huggingface_hub.errors import LocalEntryNotFoundError

from mlx_mcp import model_runtime
from mlx_mcp.model_runtime import load_rerank_tokenizer, resolve_model_snapshot
from mlx_mcp.passages import Passage
from mlx_mcp.pinned_models import RERANK_SOURCE, ModelSource
from mlx_mcp.reranking import MAX_TOKENS
from mlx_mcp.service import DocumentationService


def test_query_length_check_reads_only_the_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # conftest replaces model_runtime.load_rerank_tokenizer for every test; this
    # one exercises the real function, imported before that patch applied.
    snapshot = tmp_path / "snapshot"
    fake_cross_encoder(snapshot)
    requested: list[tuple[ModelSource, Path]] = []

    def resolve(source: ModelSource, cache: Path, **options: object) -> Path:
        requested.append((source, cache))
        return snapshot

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("The length check must not load the reranker")

    monkeypatch.setattr(model_runtime, "resolve_model_snapshot", resolve)
    monkeypatch.setattr(model_runtime, "load_reranker", forbidden)
    tokenizer = load_rerank_tokenizer(tmp_path)
    assert requested == [(RERANK_SOURCE, tmp_path / "models")]
    assert tokenizer.max_tokens == MAX_TOKENS
    tokenizer.check_query("word " * (MAX_TOKENS // 2))
    with pytest.raises(ValueError, match="Query too long: 257 tokens"):
        tokenizer.check_query("word " * (MAX_TOKENS // 2 + 1))


@pytest.mark.parametrize(
    "scores", [[], [1.0] * 4, [0.0, float("nan"), 1.0], ["1", "2", "3"]]
)
def test_invalid_reranker_scores_are_not_silently_truncated(
    root: Path, scores: list[float] | list[str]
) -> None:
    index = DocumentationService(root)
    index.retrieve = lambda query, language="all", package=None: index.index.passages

    class BrokenReranker:
        def score(self, query: str, passages: list[Passage]) -> list[float] | list[str]:
            return scores

    index.models.reranker = BrokenReranker()
    with pytest.raises(ValueError, match="one finite score per candidate"):
        asyncio.run(index.search("question"))


def test_idle_models_are_released_and_search_reloads_them(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    references: list[weakref.ReferenceType[object]] = []

    def reranker(root: Path, **kwargs: object):
        model = FakeReranker(lambda query, texts: range(len(texts)))
        references.append(weakref.ref(model))
        return model

    monkeypatch.setattr(model_runtime, "load_reranker", reranker)
    index = DocumentationService(root)

    async def check() -> None:
        expected = await index.search("Body")
        await index.models.release_idle_models(60)
        assert all(reference() is not None for reference in references)
        await index.models.release_idle_models(0)
        assert all(reference() is None for reference in references)
        assert await index.search("Body") == expected
        assert len(references) == 2

    try:
        asyncio.run(check())
    finally:
        index.close()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("cached", [False, True])
def test_cache_lookup_and_download_use_the_same_pinned_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached: bool
) -> None:
    source = RERANK_SOURCE
    calls: list[bool] = []
    destination = tmp_path / "snapshots" / source.revision

    def download(
        *,
        repo_id: str,
        revision: str,
        allow_patterns: list[str],
        cache_dir: str,
        local_files_only: bool = False,
    ) -> str:
        assert repo_id == source.repository
        assert revision == source.revision
        assert allow_patterns == list(source.files)
        assert cache_dir == str(tmp_path)
        calls.append(local_files_only)
        if local_files_only and not cached:
            raise LocalEntryNotFoundError("not cached")
        for name in source.files:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"cached")
        return str(destination)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert resolve_model_snapshot(source, tmp_path) == destination
    assert calls == ([True] if cached else [True, False])
