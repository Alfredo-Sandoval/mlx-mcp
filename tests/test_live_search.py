"""Live retrieval checks over the real bundled index (MLX_MCP_LIVE_TESTS=1)."""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlsplit

import pytest
from helpers import tool_text
from mcp import Client, StdioServerParameters
from pydantic import TypeAdapter

from mlx_mcp.indexing import select_index_directory
from mlx_mcp.model_runtime import ModelAccess
from mlx_mcp.packages import Package
from mlx_mcp.responses import RESPONSE_CHAR_BUDGET
from mlx_mcp.service import DocumentationInfo, DocumentationService


class DevelopmentCase(TypedDict):
    """A question whose results must include one of expected_sources.

    When a known_source_gap case misses, the test marks it as an expected failure.
    """

    id: str
    query: str
    limit: int
    expected_sources: list[str]
    known_source_gap: bool


class DevelopmentFile(TypedDict):
    cases: list[DevelopmentCase]


DEVELOPMENT_CASES = TypeAdapter(DevelopmentFile).validate_json(
    (Path(__file__).parent / "fixtures/development_queries.json").read_text(),
    strict=True,
)["cases"]


def serve_over_stdio() -> Client:
    """Start `mlx-mcp serve` from this environment, the way an agent does."""
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mlx_mcp.cli", "serve"],
        # The SDK passes only a few variables by default; keep the cache settings.
        env=dict(os.environ),
    )
    return Client(server, mode="legacy", read_timeout_seconds=180)


@pytest.fixture(scope="module")
def live_index() -> DocumentationService:
    return DocumentationService(select_index_directory())


@pytest.mark.skipif(
    os.environ.get("MLX_MCP_LIVE_TESTS") != "1", reason="requires downloaded MLX index"
)
@pytest.mark.parametrize("case", DEVELOPMENT_CASES, ids=lambda case: case["id"])
def test_development_query_source_coverage(
    live_index: DocumentationService, case: DevelopmentCase
) -> None:
    text = asyncio.run(
        live_index.search(case["query"], limit=case["limit"], language="python")
    )
    assert len(text) <= 8000
    sources = [
        line.removeprefix("Source: ")
        for line in text.splitlines()
        if line.startswith("Source: ")
    ]
    assert sources
    assert "external package(s)" not in text
    matched = any(
        (source if "#" in expected else urlsplit(source).path).endswith("/" + expected)
        for source in sources
        for expected in case["expected_sources"]
    )
    print(json.dumps({"id": case["id"], "sources": sources, "matched": matched}))
    if not matched and case["known_source_gap"]:
        pytest.xfail(
            "Known development source-coverage gap; retained without changing the question"
        )
    assert matched, {"query": case["query"], "sources": sources}
    assert sources[0] in live_index.fetch(sources[0])


@pytest.mark.skipif(
    os.environ.get("MLX_MCP_LIVE_TESTS") != "1", reason="requires downloaded MLX index"
)
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "3D convolution input layout channels last NDHWC",
            ("mlx.core.conv3d.html", "mlx.nn.Conv3d.html"),
        ),
        (
            "2-D transposed convolution input layout",
            ("mlx.core.conv_transpose2d.html", "mlx.nn.ConvTranspose2d.html"),
        ),
        (
            "1D convolution input layout channels last",
            ("mlx.core.conv1d.html", "mlx.nn.Conv1d.html"),
        ),
        (
            "Convolution performance channels-last transposed arrays contiguous",
            tuple(
                f"mlx.{owner}.Conv{dimension}d.html"
                if owner == "nn"
                else f"mlx.core.conv{dimension}d.html"
                for owner in ("core", "nn")
                for dimension in (1, 2, 3)
            ),
        ),
    ],
)
def test_live_convolution_layout_queries(
    live_index: DocumentationService, query: str, expected: tuple[str, ...]
) -> None:
    """Prefer the requested convolution contract over incidental layout vocabulary."""
    result = asyncio.run(
        live_index.search(query, limit=3, language="python", package=Package.MLX)
    )
    sources = [
        line.removeprefix("Source: ")
        for line in result.splitlines()
        if line.startswith("Source: ")
    ]
    assert sources and urlsplit(sources[0]).path.endswith(expected), result
    assert sources[0] in live_index.fetch(sources[0])


@pytest.mark.skipif(
    os.environ.get("MLX_MCP_LIVE_TESTS") != "1", reason="requires downloaded MLX index"
)
@pytest.mark.parametrize(
    "query",
    [
        "Why does changing an array captured by mx.compile not change the result?",
        "Compilation constant folding closed-over arrays and dynamic inputs",
        "How do I pass mutable state captured by a compiled function?",
    ],
)
def test_live_compiled_state_reaches_the_explanation(
    live_index: DocumentationService, query: str
) -> None:
    """A short result list must retain the capture contract and its runnable examples."""
    result = asyncio.run(
        live_index.search(query, limit=2, language="python", package=Package.MLX)
    )
    url = Package.MLX.base_url + "usage/compile.html#pure-functions"
    assert f"Source: {url}" in result, result
    fetched = live_index.fetch(url)
    assert "treat any inputs not in the parameter list as" in fetched
    assert "inputs=state" in fetched
    assert "Documentation version:" in fetched


@pytest.mark.skipif(
    os.environ.get("MLX_MCP_LIVE_TESTS") != "1", reason="requires downloaded MLX index"
)
@pytest.mark.parametrize(
    "query",
    [
        "I indexed past the end of an array on the GPU and got garbage instead of an exception. Why is there no error?",
        "Does MLX validate array indices on the GPU before reading an element?",
        "Why do invalid GPU array indices not raise a bounds error in MLX?",
    ],
)
def test_live_gpu_indexing_errors_reach_the_bounds_contract(
    live_index: DocumentationService, query: str
) -> None:
    result = asyncio.run(
        live_index.search(query, limit=3, language="python", package=Package.MLX)
    )
    url = Package.MLX.base_url + "usage/indexing.html#differences-from-numpy"
    assert f"Source: {url}" in result, result
    fetched = live_index.fetch(url)
    assert "does not perform bounds checking" in fetched
    assert "exceptions cannot propagate" in fetched


@pytest.mark.skipif(
    os.environ.get("MLX_MCP_LIVE_TESTS") != "1", reason="requires downloaded MLX index"
)
@pytest.mark.parametrize(
    ("query", "source_suffix", "evidence"),
    [
        (
            query,
            "/README.md#streaming",
            'print(response.text, end="", flush=True)',
        )
        for query in (
            "How can I print the model's output token by token as it's generated, instead of waiting for the whole response?",
            "Show generated text immediately as each token arrives",
            "How do I display a language model response incrementally in Python?",
        )
    ]
    + [
        (
            query,
            "/mlx.nn.ALiBi.html#mlx-nn-alibi",
            "static, non-learnable bias matrix",
        )
        for query in (
            "I want attention scores biased by how far apart two tokens are so the model extrapolates to longer sequences. Is there a built-in layer?",
            "Is there a layer that adds linear distance biases to attention?",
            "How do I use a fixed relative token distance bias for length extrapolation?",
        )
    ],
)
def test_live_concept_questions_without_package_filters(
    live_index: DocumentationService, query: str, source_suffix: str, evidence: str
) -> None:
    """Concept wording should reach the correct package without a package hint."""
    result = asyncio.run(live_index.search(query, limit=3, language="python"))
    sources = [
        line.removeprefix("Source: ")
        for line in result.splitlines()
        if line.startswith("Source: ")
    ]
    matching = [source for source in sources if source.endswith(source_suffix)]
    assert matching, result
    assert evidence in live_index.fetch(matching[0])


@pytest.mark.skipif(
    os.environ.get("MLX_MCP_LIVE_TESTS") != "1", reason="requires downloaded MLX index"
)
@pytest.mark.parametrize(
    "query",
    [
        "How can I efficiently generate text for a whole list of prompts at once instead of looping over them one at a time?",
        "How do I generate responses for several tokenized prompts together?",
        "Show the API for batched language model inference over a list of prompts",
    ],
)
def test_live_batch_generation_keeps_the_batch_api(
    live_index: DocumentationService, query: str
) -> None:
    """Streaming vocabulary must not displace the distinct batch-generation API."""
    result = asyncio.run(live_index.search(query, limit=3, language="python"))
    assert "### mlx_lm.batch_generate\nSource: " in result, result
    assert "Generate responses for the given batch of prompts." in result


@pytest.mark.skipif(
    os.environ.get("MLX_MCP_LIVE_TESTS") != "1",
    reason="requires the downloaded MLX index",
)
def test_live_bounded_evidence_and_class_anchor() -> None:

    async def check() -> None:
        async with serve_over_stdio() as client:
            cases = (
                [
                    (
                        {
                            "query": "mlx::core::conv2d",
                            "language": "cpp",
                            "limit": limit,
                        },
                        ("array conv2d(", "int groups = 1"),
                    )
                    for limit in (1, 2, 6)
                ]
                + [
                    (
                        {
                            "query": "How do I write a custom Metal kernel that handles non-contiguous input strides?",
                            "limit": limit,
                        },
                        ("elem_to_loc", "ensure_row_contiguous=False", "a[::2]"),
                    )
                    for limit in (2, 6)
                ]
                + [
                    (
                        {"query": query, "limit": 6},
                        (
                            "arr[2:8:2]",
                            "array.astype",
                            "Cast the array",
                            "mlx.core.eval",
                            "Each argument can be a single array",
                        ),
                    )
                    for query in (
                        "MLX slicing indexed updates casting lazy evaluation mx.eval",
                        "MLX indexing and casting with mx.eval",
                    )
                ]
                + [
                    (
                        {
                            "query": "Slice an MLX array, update indexed entries, convert dtype, then force computation",
                            "limit": 6,
                        },
                        (
                            "arr[2:8:2]",
                            "a[2] = 0",
                            "Cast the array",
                            "Each argument can be a single array",
                        ),
                    ),
                    (
                        {
                            "query": "How do I slice every other element between indices 2 and 8?",
                            "limit": 1,
                        },
                        ("arr[2:8:2]", "array([2, 4, 6], dtype=int32)"),
                    ),
                    (
                        {"query": "mlx.nn.Module", "limit": 2},
                        (
                            "class Module",
                            "Base class for building neural networks with MLX",
                            "mx.eval(model.parameters())",
                        ),
                    ),
                    (
                        {
                            "query": "How do I vectorize a function but leave one input unbatched?",
                            "limit": 2,
                        },
                        ("mlx.core.vmap", "None"),
                    ),
                ]
            )
            for arguments, facts in cases:
                result = await client.call_tool("search_mlx_docs", arguments)
                assert not result.is_error
                text = tool_text(result)
                assert result.content[0].type == "text"
                assert result.content[0].text == text
                assert len(text) <= RESPONSE_CHAR_BUDGET
                assert all(fact in text for fact in facts), (arguments, facts, text)
            # nn.Module is mentioned widely; its definition must still lead.
            lookup = await client.call_tool("search_mlx_docs", {"query": "nn.Module"})
            assert not lookup.is_error
            canonical = next(
                line.removeprefix("Source: ")
                for line in tool_text(lookup).splitlines()
                if line.startswith("Source: ")
            )
            url = canonical.split("#")[0] + "#mlx.nn.Module"
            result = await client.call_tool("fetch_mlx_docs", {"url": url})
            assert not result.is_error
            text = tool_text(result)
            assert len(text) <= RESPONSE_CHAR_BUDGET
            assert "Source: " + canonical in text
            assert f"Requested API anchor: {url}" in text
            assert "class Module" in text

    asyncio.run(check())


@pytest.mark.skipif(
    os.environ.get("MLX_MCP_LIVE_TESTS") != "1",
    reason="live models and combined index required",
)
@pytest.mark.parametrize(
    "query, expected, fact",
    [
        (
            "How does mlx.data.Stream.prefetch use prefetch_size and num_threads?",
            "mlx.data.Stream.prefetch.html",
            "Fetch samples in background threads.",
        ),
        (
            "How does dx.buffer_from_vector work?",
            "mlx.data.buffer_from_vector.html",
            "Make a buffer from a list of dictionaries.",
        ),
        (
            "What is the difference between buffers and streams in MLX Data?",
            "buffers_streams_samples.html",
            "potentially",
        ),
    ],
)
def test_live_data_search_and_full_source(query: str, expected: str, fact: str) -> None:

    index = DocumentationService(select_index_directory(), access=ModelAccess.CACHED)
    try:
        result = asyncio.run(index.search(query, package=Package.DATA))
        urls = re.findall(r"^Source: (.+)$", result, re.MULTILINE)
        assert urls and all(url.startswith(Package.DATA.base_url) for url in urls)
        matching = [url for url in urls if expected in url]
        assert matching and len(result) <= 8000
        assert fact in index.fetch(matching[0])
        assert "external package" not in result
    finally:
        index.models.worker.close()


LIVE_CASES = [
    (
        "How do I force evaluation of lazy MLX arrays with mx.eval?",
        ("lazy_evaluation.html", "mlx.core.eval.html"),
    ),
    ("mlx.core.conv3d grouped convolution groups parameter", ("mlx.core.conv3d.html",)),
    (
        "Train a linear regression model with gradient descent in MLX",
        ("examples/linear_regression.html",),
    ),
    (
        "How can I vectorize a function over a batch using mlx.core.vmap?",
        ("mlx.core.vmap.html",),
    ),
    (
        "How do I save arrays to a safetensors file with MLX?",
        ("mlx.core.save_safetensors.html",),
    ),
    (
        "Compute a function value and its gradient with mlx.core.value_and_grad",
        ("mlx.core.value_and_grad.html",),
    ),
    (
        "How do I select a variable number of elements using a boolean mask inside a compiled function?",
        ("usage/indexing.html#differences-from-numpy",),
    ),
    (
        "How do I disable dropout for inference, and what is the difference between model.eval() and mx.eval()?",
        ("mlx.nn.Module.eval.html",),
    ),
    ("How do I wait for a GPU stream to finish?", ("mlx.core.synchronize.html",)),
    (
        "Why do my GPU computations appear to take no time when I benchmark them?",
        ("usage/lazy_evaluation.html#why-lazy-evaluation",),
    ),
    (
        "How do I convert an array from float32 to float16?",
        ("mlx.core.array.astype.html",),
    ),
    (
        "What is the default dtype in MLX, and does the GPU support float64?",
        ("python/data_types.html#data-types",),
    ),
    (
        "AdamW optimizer with cosine learning rate",
        ("mlx.optimizers.cosine_decay.html",),
    ),
    (
        "How can I check current and peak GPU memory usage and clear the cache?",
        ("python/memory_management.html#memory-management",),
    ),
    (
        "What are the restrictions of shapeless=True?",
        (
            "usage/compile.html#shapeless-compilation",
            "usage/export.html#shapeless-exports",
        ),
    ),
    ("How do I use quantizied matmul?", ("mlx.core.quantized_matmul.html",)),
    (
        "How do I perform an all-reduce sum across distributed processes?",
        ("mlx.core.distributed.all_sum.html",),
    ),
    (
        "Is there an mx.no_grad() context manager?",
        ("usage/function_transforms.html#automatic-differentiation",),
    ),
    (
        "Does MLX support indexing an array with a boolean mask to select a variable number of elements?",
        ("usage/indexing.html#differences-from-numpy",),
    ),
    (
        "Can I read a[mask] using a boolean mask?",
        ("usage/indexing.html#differences-from-numpy",),
    ),
    (
        "Gather values from an array using a boolean mask",
        ("usage/indexing.html#differences-from-numpy",),
    ),
    ("mlx.nn.Conv3d input shape and weight shape", ("mlx.nn.Conv3d.html",)),
    ("mlx.nn.Conv2d input shape and weight shape", ("mlx.nn.Conv2d.html",)),
    ("What input layout does mlx.nn.Conv1d require?", ("mlx.nn.Conv1d.html",)),
]


REQUIRED_FACTS = {
    "python/memory_management.html#memory-management": (
        "Get the actively used memory in bytes",
        "Get the peak amount of used memory in bytes",
        "Clear the memory cache",
    ),
    "usage/indexing.html#differences-from-numpy": ("supported for assignment only",),
    "mlx.nn.Conv3d.html": ("NDHWC", "(C_out, KD, KH, KW, C_in)"),
    "mlx.nn.Conv2d.html": ("NHWC", "(C_out, KH, KW, C_in)"),
    "mlx.nn.Conv1d.html": ("NLC",),
}


@pytest.mark.skipif(
    os.environ.get("MLX_MCP_LIVE_TESTS") != "1",
    reason="requires the downloaded MLX index",
)
def test_live_stdio_protocol_and_source_smoke() -> None:
    """Check real model search and local fetch through `mlx-mcp serve`."""

    async def check() -> None:

        async with serve_over_stdio() as client:
            print(f"Negotiated MCP protocol: {client.protocol_version}")
            catalog = await client.list_tools()
            assert {tool.name for tool in catalog.tools} == {
                "search_mlx_docs",
                "fetch_mlx_docs",
                "get_mlx_docs_info",
            }
            metadata = await client.call_tool("get_mlx_docs_info", {})
            assert not metadata.is_error
            info = TypeAdapter(DocumentationInfo).validate_json(
                tool_text(metadata), strict=True
            )
            assert info["passage_count"] > 0 and "mlx" in info["packages"]
            source_failures: list[dict[str, object]] = []
            for query, expected_sources in LIVE_CASES:
                start = time.perf_counter()
                result = await client.call_tool("search_mlx_docs", {"query": query})
                print(f"end-to-end seconds: {time.perf_counter() - start:.3f}")
                if result.is_error:
                    source_failures.append(
                        {"query": query, "error": str(result.content)}
                    )
                    continue
                text = tool_text(result)
                print(f"Query: {query}")
                print(
                    "\n".join(
                        line
                        for line in text.splitlines()
                        if line.startswith("Source: ")
                    )
                )
                sources = [
                    line for line in text.splitlines() if line.startswith("Source: ")
                ]
                for expected in expected_sources:
                    for fact in REQUIRED_FACTS.get(expected, ()):
                        assert fact in text
                if not any(
                    (
                        source
                        if "#" in expected
                        else urlsplit(source.removeprefix("Source: ")).path
                    ).endswith("/" + expected)
                    for source in sources
                    for expected in expected_sources
                ):
                    source_failures.append({"query": query, "sources": sources})
                assert text.count("Source: ") == 3
                url = sources[0].removeprefix("Source: ")
                fetched = await client.call_tool("fetch_mlx_docs", {"url": url})
                assert not fetched.is_error
                assert url in tool_text(fetched)
                assert "Documentation version:" in tool_text(fetched)
            assert not source_failures, source_failures
            lookup = await client.call_tool(
                "search_mlx_docs", {"query": "mlx.core.compile"}
            )
            assert not lookup.is_error
            canonical = next(
                line.removeprefix("Source: ")
                for line in tool_text(lookup).splitlines()
                if line.startswith("Source: ")
            )
            page = canonical.split("#")[0]
            linked = await client.call_tool(
                "fetch_mlx_docs", {"url": page + "#mlx.core.compile"}
            )
            assert not linked.is_error
            assert f"Source: {canonical}" in tool_text(linked)
            assert f"Requested API anchor: {page}#mlx.core.compile" in tool_text(linked)

    asyncio.run(check())
