"""Pytest fixtures shared across test files: model stand-ins and small indexes."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from helpers import (
    CORE,
    SYNTHETIC_TOKENIZER,
    FakeReranker,
    data_passage,
)
from tokenizers import Tokenizer

from mlx_mcp import indexing, model_runtime
from mlx_mcp.index_file import write_index
from mlx_mcp.model_runtime import ModelAccess
from mlx_mcp.packages import (
    EvidenceKind,
    Package,
)
from mlx_mcp.passages import Passage
from mlx_mcp.service import DocumentationService
from mlx_mcp.sphinx_pages import parse_page
from mlx_mcp.token_windows import TokenWindows

# An MLX LM API source URL pinned to a full commit.
LM = Package.LM.base_url + "blob/" + "a" * 40 + "/mlx_lm/generate.py#L1"

# Avoid native tokenizer worker-pool teardown during live model tests.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(autouse=True)
def synthetic_rerank_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measure query length with the synthetic word tokenizer, never MiniLM's.

    Every search checks its query length before reranking, which would otherwise
    resolve the real model snapshot. indexing imports the loader by name, so it is
    patched there too. test_model_runtime.py tests the real loader directly,
    importing it before this patch applies.
    """

    def load(root: Path, *, access: ModelAccess = ModelAccess.DOWNLOAD) -> TokenWindows:
        return TokenWindows(Tokenizer.from_file(str(SYNTHETIC_TOKENIZER)), 512)

    monkeypatch.setattr(model_runtime, "load_rerank_tokenizer", load)
    monkeypatch.setattr(indexing, "load_rerank_tokenizer", load)


@pytest.fixture
def mixed_index(tmp_path: Path) -> Iterator[DocumentationService]:
    passages = [
        Passage(
            "mlx.core.eval",
            Package.MLX.base_url + "python/eval.html",
            "Evaluate lazy arrays.",
            documentation_version="0.32.2",
        ),
        data_passage(),
    ]
    write_index(tmp_path, passages)
    index = DocumentationService(tmp_path)
    index.models.reranker = FakeReranker(lambda query, texts: range(len(texts)))
    yield index
    index.models.worker.close()


@pytest.fixture
def navigation_index(tmp_path: Path) -> DocumentationService:
    passages = [
        Passage(
            "mlx.core.astype",
            CORE + "python/astype.html#astype",
            "astype(a, dtype)",
            documentation_version="0.32.2",
        ),
        Passage(
            "mlx.core.array.astype",
            CORE + "python/array.html#astype",
            "array.astype(dtype)",
        ),
        Passage("mlx.nn.RMSNorm", CORE + "python/rms.html", "class RMSNorm(dims)"),
        Passage("mlx.core.linalg.qr", CORE + "python/qr.html", "qr(a)"),
        Passage("Guide", CORE + "usage/example.html", "Call mlx.core.missing() here."),
        Passage(
            "mlx_lm.generate command-line options",
            "https://docs/cli#generate",
            "CLI ONLY --model",
            package=Package.LM,
            evidence_kind=EvidenceKind.CLI,
        ),
        Passage(
            "mlx_lm.generate",
            LM,
            "```python\ndef generate(model, tokenizer, prompt):\n```\n\nAPI ONLY",
            package=Package.LM,
            documentation_version="0.31.3",
            evidence_kind=EvidenceKind.API,
            source_revision="a" * 40,
        ),
        Passage(
            "mlx.data.Buffer.prefetch",
            Package.DATA.base_url + "python/prefetch.html",
            "prefetch(size)",
            package=Package.DATA,
        ),
    ]
    passages += parse_page(
        '<article class="bd-article"><h1>Guide</h1><p>Intro.</p><section id="one"><h2>First</h2><p>One.</p>'
        '<section id="nested"><h3>Nested</h3><p>Nested body.</p></section></section>'
        '<section id="two"><h2>Second</h2><p>Two.</p></section></article>',
        CORE + "usage/guide.html",
    )
    write_index(tmp_path, passages)
    # Any accidental model use fails against this deliberately empty cache.
    return DocumentationService(
        tmp_path, model_cache=tmp_path / "empty", access=ModelAccess.CACHED
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    passages = [
        Passage(f"API {i}", f"https://docs/python/api#{i}", f"Body {i}", f"Section {i}")
        for i in range(3)
    ]
    write_index(tmp_path, passages)
    return tmp_path
