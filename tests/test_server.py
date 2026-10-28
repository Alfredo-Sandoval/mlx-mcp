"""MCP tool schemas, argument validation, error results, and server instructions."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import NoReturn

import pytest
from helpers import (
    FakeReranker,
    fixed_order_service,
    tool_schema,
    tool_text,
)
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent
from pydantic import TypeAdapter

from mlx_mcp import (
    server,
)
from mlx_mcp.index_file import write_index
from mlx_mcp.model_runtime import ModelAccess
from mlx_mcp.packages import Package
from mlx_mcp.passages import Passage
from mlx_mcp.service import DocumentationInfo, DocumentationService
from mlx_mcp.workers import LazyResource


def error_text(result: CallToolResult) -> str:
    """Return the text of a tool error, failing if the call succeeded."""
    assert result.is_error
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


def test_search_protocol_rejects_coerced_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    index = fixed_order_service(
        [Passage(f"Guide {i}", f"https://docs/{i}", "Answer.") for i in range(6)]
    )

    def fake_get_service():
        return index

    monkeypatch.setattr(server, "get_service", fake_get_service)

    async def check() -> None:
        async with Client(server.create_server()) as client:
            # bool/str/float inputs fail pydantic's strict int check before
            # the bound check ever runs.
            for limit in [True, "2", 2.5]:
                result = await client.call_tool(
                    "search_mlx_docs", {"query": "arrays", "limit": limit}
                )
                assert "valid integer" in error_text(result), limit
            # 0 and 7 are each one step outside the documented 1..6 bound, on
            # opposite sides, so the specific error names the bound it hit.
            result = await client.call_tool(
                "search_mlx_docs", {"query": "arrays", "limit": 0}
            )
            assert "greater than or equal to 1" in error_text(result)
            result = await client.call_tool(
                "search_mlx_docs", {"query": "arrays", "limit": 7}
            )
            assert "less than or equal to 6" in error_text(result)
            result = await client.call_tool(
                "search_mlx_docs", {"query": "arrays", "limit": 2}
            )
            assert tool_text(result).count("Source:") == 2

    asyncio.run(check())


def test_protocol_catalog_input_and_missing_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    monkeypatch.setenv("MLX_MCP_CACHE", str(tmp_path))
    monkeypatch.setattr(
        server, "_service", LazyResource(lambda: DocumentationService(tmp_path))
    )

    async def check() -> None:
        async with Client(server.create_server()) as client:
            catalog = await client.list_tools()
            tools_by_name = {tool.name: tool for tool in catalog.tools}
            assert tools_by_name.keys() == {
                "search_mlx_docs",
                "fetch_mlx_docs",
                "get_mlx_docs_info",
            }
            schema = tool_schema(tools_by_name["search_mlx_docs"])
            assert schema["required"] == ["query"]
            assert set(schema["properties"]) == {
                "query",
                "limit",
                "language",
                "package",
            }
            missing = await client.call_tool(
                "search_mlx_docs", {"query": "lazy evaluation"}
            )
            assert missing.is_error
            assert missing.content[0].type == "text"
            # The service's cache is MLX_MCP_CACHE; the advice must name it.
            assert (
                f"Run `mlx-mcp prepare --refresh --cache {tmp_path}`, then start a new session."
                in missing.content[0].text
            )
            for query in [" ", "x" * 2001]:
                invalid = await client.call_tool("search_mlx_docs", {"query": query})
                assert invalid.is_error

    asyncio.run(check())


def test_metadata_tool_reports_snapshot_versions_without_loading_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_index(
        tmp_path,
        [
            Passage(
                "mlx.core.eval",
                Package.MLX.base_url + "python/eval.html",
                "Evaluate arrays.",
                documentation_version="0.32.4",
            )
        ],
    )
    model_cache = tmp_path / "no-model-downloads"
    index = DocumentationService(
        tmp_path, model_cache=model_cache, access=ModelAccess.CACHED
    )

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        pytest.fail("Metadata must not load the search model or tokenizer")

    monkeypatch.setattr(index.models, "check_query", forbidden)
    monkeypatch.setattr(index.models, "score", forbidden)
    monkeypatch.setattr(server, "get_service", lambda: index)

    async def check() -> None:
        async with Client(server.create_server()) as client:
            result = await client.call_tool("get_mlx_docs_info", {})
            assert not result.is_error
            info = TypeAdapter(DocumentationInfo).validate_json(
                tool_text(result), strict=True
            )
            assert info["packages"]["mlx"]["documentation_versions"] == ["0.32.4"]
            assert info["page_count"] == info["passage_count"] == 1
            assert info["index"] == str(tmp_path)
            assert not model_cache.exists()

    try:
        asyncio.run(check())
    finally:
        index.close()


def test_missing_model_reports_prepare_without_misidentifying_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    write_index(
        tmp_path,
        [
            Passage("Guide", "https://docs/guide", "Evaluation source."),
            Passage("Other", "https://docs/other", "More evaluation source."),
        ],
    )
    monkeypatch.setattr(
        server,
        "_service",
        LazyResource(
            lambda: DocumentationService(
                tmp_path, model_cache=tmp_path, access=ModelAccess.CACHED
            )
        ),
    )

    async def check() -> None:
        async with Client(server.create_server()) as client:
            result = await client.call_tool("search_mlx_docs", {"query": "evaluation"})
            assert result.is_error
            assert result.content[0].type == "text"
            assert "mlx-mcp prepare" in result.content[0].text

    asyncio.run(check())


TOOL_PAGE = Package.MLX.base_url + "usage/release-fixture.html"


@pytest.fixture
def tool_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DocumentationService:
    write_index(
        tmp_path,
        [
            Passage("Guide", TOOL_PAGE, "Retained evidence."),
            Passage(
                "Other guide", TOOL_PAGE + "/other", "Additional retained evidence."
            ),
        ],
    )
    index = DocumentationService(tmp_path)
    index.models.reranker = FakeReranker(
        lambda query, documents: [1.0] * len(documents)
    )
    monkeypatch.setattr(server, "get_service", lambda: index)
    return index


async def assert_tool_failure(action: Callable[[], Awaitable[str]]) -> None:
    with pytest.raises(ToolError) as failure:
        await action()
    assert type(failure.value.__cause__) is ValueError
    assert str(failure.value) == str(failure.value.__cause__)


def test_backend_value_error_is_reported_as_a_tool_error(
    tool_index: DocumentationService,
) -> None:
    tool_index.models.reranker = FakeReranker(
        lambda query, documents: [float("nan")] * len(documents)
    )
    try:
        asyncio.run(
            assert_tool_failure(
                lambda: server.search_mlx_docs("retained evidence"),
            )
        )
    finally:
        tool_index.close()
