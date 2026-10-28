"""Shared test builders: model fakes, passages, services and tool-output readers."""

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NoReturn, TypedDict

import numpy as np
from mcp.types import CallToolResult, TextContent, Tool
from pydantic import TypeAdapter
from tokenizers import Tokenizer

from mlx_mcp.index_file import write_index
from mlx_mcp.model_runtime import ModelAccess, ModelRuntime
from mlx_mcp.packages import Package
from mlx_mcp.passages import Passage
from mlx_mcp.responses import RESPONSE_CHAR_BUDGET
from mlx_mcp.retrieval import RetrievalIndex
from mlx_mcp.search_text import retrieval_input
from mlx_mcp.service import DocumentationService

# A small WordLevel tokenizer with the reranker's 512-token truncation.
SYNTHETIC_TOKENIZER = Path(__file__).parent / "fixtures/synthetic_tokenizer.json"


class FakeReranker:
    """Score passages from the same title-plus-text input the real reranker reads."""

    def __init__(self, rerank: Callable[[str, list[str]], Iterable[float]]) -> None:
        self._rerank = rerank

    def score(self, query: str, passages: list[Passage]) -> list[float]:
        return list(
            self._rerank(query, [retrieval_input(p.title, p.text) for p in passages])
        )


def _unconfigured_pair_scores(query: str, documents: list[str]) -> NoReturn:
    raise AssertionError("The test must declare its ranking scores")


class FakeCrossEncoder:
    """A real tokenizer with test-supplied pair scores, for WindowedReranker."""

    def __init__(self, root: Path) -> None:
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.max_tokens = 512
        self.pair_scores: Callable[[str, list[str]], Iterable[float]] = (
            _unconfigured_pair_scores
        )


def fake_cross_encoder(root: Path) -> FakeCrossEncoder:
    root.mkdir(parents=True, exist_ok=True)
    # Native WordLevel serialization includes the original 512-token truncation.
    tokenizer = Tokenizer.from_file(str(SYNTHETIC_TOKENIZER))
    (root / "tokenizer.json").write_text(tokenizer.to_str())
    return FakeCrossEncoder(root)


CORE = Package.MLX.base_url
OPS_PAGE = Package.MLX.base_url + "cpp/ops.html"
FIXTURE_PAGE = "https://ml-explore.github.io/mlx/build/html/usage/review_fixture.html"
REPEATED_PAGE = "https://docs/repeated"


def definition(name: str, length: int = 1) -> Passage:
    """An MLX API definition passage; ``length`` repeats its parameter line."""
    return Passage(
        name,
        Package.MLX.base_url + f"python/{name}.html",
        f"{name}(input)\n" + "Parameter detail.\n" * length,
    )


def data_passage() -> Passage:
    return Passage(
        "mlx.data.Stream.prefetch",
        Package.DATA.base_url + "python/_autosummary/mlx.data.Stream.prefetch.html",
        "Prefetch samples in background threads with prefetch_size and num_threads.",
        documentation_version="0.0.2",
        package=Package.DATA,
    )


def service_on_disk(root: Path, passages: list[Passage]) -> DocumentationService:
    """Write a real index file and load a service over it."""
    write_index(root, passages)
    return DocumentationService(root)


def service_in_memory(passages: list[Passage]) -> DocumentationService:
    """Build a service over passages without an index file or model loading.

    Callers supply model fakes when they exercise inference. Tests of stored
    metadata use service_on_disk, which loads a real validated index file.
    """
    service = DocumentationService.__new__(DocumentationService)
    service.index = RetrievalIndex(passages)
    service.models = ModelRuntime(
        Path(".unused-test-model-cache"), access=ModelAccess.CACHED
    )
    return service


def fixed_order_service(passages: list[Passage]) -> DocumentationService:
    """An in-memory service whose search returns ``passages`` in order.

    Retrieval always returns every passage and scores descend with position,
    so the test controls ranking and exercises only what happens after it. The
    query length check is skipped too, since it belongs to the ranking stage.
    """
    service = service_in_memory(passages)
    service.check_query = lambda query: None
    service.retrieve = lambda query, language="all", package=None: passages
    service.score_candidates = lambda query, passages: np.arange(len(passages), 0, -1)
    return service


def tool_text(result: CallToolResult) -> str:
    """Parse the string-returning tools' SDK envelope without propagating Any."""
    if not result.content or not isinstance(result.content[0], TextContent):
        raise ValueError("Expected text in the tool result")
    text = result.content[0].text
    if result.structured_content != {"result": text}:
        raise ValueError("Tool text and structured result disagree")
    return text


class ToolSchema(TypedDict):
    required: list[str]
    properties: dict[str, dict[str, object]]


def tool_schema(tool: Tool) -> ToolSchema:
    return TypeAdapter(ToolSchema).validate_python(tool.input_schema, strict=True)


# The readers below parse fetch output independently of score_search.parse_fetch,
# so a bug shared by the renderer and its parser cannot hide in these tests.


def source_page(result: str) -> tuple[int, int, int, str]:
    """Read the public range and fenced source without discarding whitespace.

    Returns ``(start, end, total, body)`` for one fetch page.
    """
    header, fenced = result.split("\n\n", 1)
    match = re.search(r"Characters: (\d+):(\d+) of (\d+)", header)
    assert match is not None
    start, end, total = map(int, match.groups())
    opening, rest = fenced.split("\n", 1)
    fence = opening.removesuffix("text")
    assert rest.endswith("\n" + fence)
    body = rest[: -len(fence) - 1]
    assert len(body) == end - start
    return start, end, total, body


def complete_fetch(service: DocumentationService, url: str) -> str:
    """Follow the public Next: offsets and join every page's source text."""
    result = service.fetch(url)
    if "\nCharacters: " not in result:
        return result.split("\n\n", 1)[1]
    parts: list[str] = []
    while True:
        assert len(result) <= RESPONSE_CHAR_BUDGET
        parts.append(source_page(result)[3])
        header = result.split("\n\n", 1)[0]
        next_call = re.search(
            r'Next: fetch_mlx_docs\(url=(".*"), offset=(\d+)\)', header
        )
        if next_call is None:
            assert "End of source." in header
            return "".join(parts)
        result = service.fetch(json.loads(next_call[1]), int(next_call[2]))
