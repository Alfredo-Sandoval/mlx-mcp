"""MLX Data through the MCP tools: the package filter, its schema, and fetch."""

import asyncio
import json

import pytest
from helpers import (
    data_passage,
    tool_schema,
    tool_text,
)
from mcp import Client

from mlx_mcp import server
from mlx_mcp.service import DocumentationService
from mlx_mcp.workers import LazyResource


def test_mcp_package_schema_filter_fetch_and_invalid_value(
    mixed_index: DocumentationService, monkeypatch: pytest.MonkeyPatch
) -> None:

    monkeypatch.setattr(server, "_service", LazyResource(lambda: mixed_index))

    async def check() -> None:
        async with Client(server.create_server()) as client:
            catalog = await client.list_tools()
            search = next(t for t in catalog.tools if t.name == "search_mlx_docs")
            assert "mlx-data" in json.dumps(tool_schema(search))
            assert tool_schema(search)["properties"]["limit"]["default"] == 3
            result = await client.call_tool(
                "search_mlx_docs", {"query": "prefetch", "package": "mlx-data"}
            )
            assert not result.is_error and result.structured_content is not None
            text = tool_text(result)
            assert "Package: MLX Data" in text and "mlx.core.eval" not in text
            assert len(text) <= 8000
            fetched = await client.call_tool(
                "fetch_mlx_docs", {"url": data_passage().url}
            )
            assert not fetched.is_error and "Package: MLX Data" in tool_text(fetched)
            invalid = await client.call_tool(
                "search_mlx_docs", {"query": "prefetch", "package": "mlx-whisper"}
            )
            assert invalid.is_error

    asyncio.run(check())
