"""Preparation over real indexes with a model boundary fake."""

import weakref
from pathlib import Path

import pytest
from helpers import FakeReranker

from mlx_mcp import indexing, model_runtime, service_setup
from mlx_mcp.index_file import load_index, write_index
from mlx_mcp.model_runtime import ModelAccess
from mlx_mcp.packages import Package
from mlx_mcp.passages import Passage
from mlx_mcp.service import DocumentationService


def source(name: str) -> Passage:
    return Passage(
        name,
        Package.MLX.base_url + f"usage/{name}.html",
        "Evaluate a lazy array with mx.eval.",
    )


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "bundle"
    write_index(root, [source("bundled")])
    monkeypatch.setattr(indexing, "BUNDLED_ROOT", root)
    return root


@pytest.fixture
def models(
    monkeypatch: pytest.MonkeyPatch,
) -> list[weakref.ReferenceType[FakeReranker]]:
    loaded: list[weakref.ReferenceType[FakeReranker]] = []

    def load(root: Path, *, access: ModelAccess = ModelAccess.DOWNLOAD) -> FakeReranker:
        marker = root / "cached-model"
        if not marker.is_file():
            root.mkdir(parents=True, exist_ok=True)
            marker.write_text("downloaded model")
        model = FakeReranker(lambda query, documents: [1.0] * len(documents))
        loaded.append(weakref.ref(model))
        return model

    monkeypatch.setattr(model_runtime, "load_reranker", load)
    return loaded


@pytest.mark.parametrize("local", [False, True])
def test_prepare_uses_a_usable_index_and_downloads_into_the_requested_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle: Path,
    models: list[weakref.ReferenceType[FakeReranker]],
    local: bool,
) -> None:
    cache = tmp_path / "requested cache"
    unrelated = tmp_path / "environment cache"
    monkeypatch.setenv("MLX_MCP_CACHE", str(unrelated))
    expected = source("local" if local else "bundled")
    if local:
        write_index(cache, [expected])

    def unexpected_download() -> list[Passage]:
        pytest.fail("A usable index must not trigger a documentation download")

    monkeypatch.setattr(indexing, "collect_passages", unexpected_download)
    result = service_setup.prepare(cache, refresh=False, progress=lambda message: None)
    assert result == service_setup.Readiness(cache, 1)
    assert (cache / "cached-model").is_file()
    assert (cache / "index.npz").exists() == local
    assert not unrelated.exists()
    assert load_index(bundle).passages == [source("bundled")]
    assert models and all(model() is None for model in models)


@pytest.mark.parametrize("refresh", [False, True])
def test_prepare_builds_missing_or_explicitly_refreshed_documentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    models: list[weakref.ReferenceType[FakeReranker]],
    refresh: bool,
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setattr(indexing, "BUNDLED_ROOT", tmp_path / "no-bundle")
    if refresh:
        write_index(cache, [source("old")])
    updated = source("updated")
    monkeypatch.setattr(indexing, "collect_passages", lambda: [updated])
    result = service_setup.prepare(
        cache, refresh=refresh, progress=lambda message: None
    )
    assert result == service_setup.Readiness(cache, 1)
    assert load_index(cache).passages == [updated]
    assert models and all(model() is None for model in models)


def test_failed_refresh_preserves_the_index_and_does_not_download_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = source("original")
    write_index(tmp_path, [original])

    def unavailable() -> list[Passage]:
        raise OSError("Documentation download failed")

    monkeypatch.setattr(indexing, "collect_passages", unavailable)
    with pytest.raises(OSError, match="Documentation download failed"):
        service_setup.prepare(tmp_path, refresh=True, progress=lambda message: None)
    assert load_index(tmp_path).passages == [original]
    assert not (tmp_path / "cached-model").exists()


def test_readiness_releases_the_model_when_fetch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle: Path,
    models: list[weakref.ReferenceType[FakeReranker]],
) -> None:
    def unavailable(self: DocumentationService, url: str, offset: int = 0) -> str:
        raise OSError("Source read failed")

    monkeypatch.setattr(DocumentationService, "fetch", unavailable)
    with pytest.raises(OSError, match="Source read failed"):
        service_setup.prepare(
            tmp_path / "cache", refresh=False, progress=lambda _: None
        )
    assert load_index(bundle).passages == [source("bundled")]
    assert models and all(model() is None for model in models)
