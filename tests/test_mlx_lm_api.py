"""MLX LM API reference passages read from the pinned wheel's source."""

import ast
import asyncio
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from helpers import (
    FakeReranker,
    complete_fetch,
    service_in_memory,
    tool_schema,
    tool_text,
)
from hypothesis import given
from hypothesis import strategies as st
from mcp import Client

from mlx_mcp import server
from mlx_mcp.index_file import load_index, write_index
from mlx_mcp.mlx_lm_api import api_passages
from mlx_mcp.mlx_lm_docs import help_passages, parse_markdown
from mlx_mcp.mlx_lm_sources import LM_SOURCE
from mlx_mcp.packages import EvidenceKind, Package
from mlx_mcp.passages import Passage, validate_passages
from mlx_mcp.service import DocumentationService
from mlx_mcp.symbols import build_symbol_index
from mlx_mcp.workers import LazyResource

_VERSION = LM_SOURCE.version


_RETRIEVED_AT = "2026-01-01T00:00:00+00:00"


_INIT_PY = """
from .generate import generate, stream_generate
from .convert import convert
from .utils import load

__all__ = ["convert", "generate", "stream_generate", "load"]
"""


_GENERATE_PY = '''
"""Generation helpers."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class GenerationResponse:
    """Container for one generation step.

    Args:
        text (str): Decoded text chunk.
        token (int): The next token.
    """

    text: str
    token: int
    finish_reason: Optional[str] = None


def generate(
    model,
    tokenizer,
    prompt: str,
    *args: int,
    max_tokens: int = 256,
    verbose: bool = False,
    **kwargs: Any,
) -> str:
    """Generate a complete response from the model.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        prompt (str): The input prompt.
        max_tokens (int): The maximum number of tokens. Default: ``256``.
        verbose (bool): Whether to print progress. Default: ``False``.

    Returns:
        str: The generated text.
    """
    return prompt


def stream_generate(model, tokenizer, prompt, /, max_tokens: int = 256):
    """Stream a response from the model."""
    yield prompt


def _internal_helper(x):
    """Should never become a passage."""
    return x
'''


_CONVERT_PY = '''
def convert(repo: str, quantize: bool = False) -> None:
    """Convert and optionally quantize a model."""
'''


_UTILS_PY = '''
def load(path_or_repo: str):
    """Load a model and tokenizer."""
'''


_SAMPLE_UTILS_PY = '''
def make_sampler(temp: float = 0.0, top_p: float = 0.0):
    """Build a sampler callable."""


def _private_sampler_helper():
    """Should never become a passage."""
'''


_TOKENIZER_UTILS_PY = '''
class TokenizerWrapper:
    """Wraps an HF tokenizer with streaming detokenization."""

    def __init__(
        self,
        tokenizer,
        detokenizer_class=None,
        eos_token_ids: set[int] | None = None,
        *,
        chat_template: str | None = None,
    ):
        """Initialize the wrapper around the required tokenizer.

        Args:
            tokenizer: Tokenizer used for encoding and decoding.
            detokenizer_class: Streaming detokenizer implementation.
            eos_token_ids: Override the tokenizer's end-of-sequence tokens.
            chat_template: Template used to format conversations.
        """
        raise RuntimeError("Source extraction must never construct the wrapper")

    def decode(self, tokens: list, skip_special_tokens: bool = True) -> str:
        """Decode tokens back to text."""
        return ""

    def _private_helper(self):
        """Should never become a passage."""
        return None
'''


_CACHE_PY = '''
def make_prompt_cache(model, max_kv_size=None):
    """Construct the model's cache for use in generation."""
'''


_TUNER_UTILS_PY = '''
def build_schedule(schedule_config: dict):
    """Build a learning-rate schedule from a config dict."""
'''


_MODULE_SOURCES = {
    "__init__.py": _INIT_PY,
    "generate.py": _GENERATE_PY,
    "convert.py": _CONVERT_PY,
    "utils.py": _UTILS_PY,
    "sample_utils.py": _SAMPLE_UTILS_PY,
    "tokenizer_utils.py": _TOKENIZER_UTILS_PY,
    "models/cache.py": _CACHE_PY,
    "tuner/utils.py": _TUNER_UTILS_PY,
}


def _build_wheel(tmp_path: Path, sources: dict[str, str] | None = None) -> Path:
    """Assemble a minimal mlx_lm wheel zip covering every module MODULES reads."""
    merged = {**_MODULE_SOURCES, **(sources or {})}
    wheel_path = tmp_path / LM_SOURCE.wheel_filename
    with zipfile.ZipFile(wheel_path, "w") as archive:
        for name, source in merged.items():
            archive.writestr(f"mlx_lm/{name}", source)
    return wheel_path


@pytest.fixture
def passages(tmp_path: Path) -> list[Passage]:
    wheel = _build_wheel(tmp_path)
    return api_passages(
        wheel,
        retrieved_at=_RETRIEVED_AT,
    )


@pytest.fixture
def by_title(passages: list[Passage]) -> dict[str, list[Passage]]:
    grouped: dict[str, list[Passage]] = {}
    for passage in passages:
        grouped.setdefault(passage.title, []).append(passage)
    return grouped


def test_produces_only_valid_passages(passages: list[Passage]) -> None:
    validate_passages(passages)  # raises on any contract violation
    assert passages
    assert all(p.documentation_version == _VERSION for p in passages)
    assert all(p.retrieved_at == _RETRIEVED_AT for p in passages)


def test_reexported_names_use_the_init_import_path(
    by_title: dict[str, list[Passage]],
) -> None:
    """Re-exported names are titled by their mlx_lm.__init__ path.

    convert, generate, stream_generate, and load are not titled by the submodule
    that happens to define them.
    """
    assert "mlx_lm.convert" in by_title
    assert "mlx_lm.generate" in by_title
    assert "mlx_lm.stream_generate" in by_title
    assert "mlx_lm.load" in by_title
    assert "mlx_lm.convert.convert" not in by_title


def test_non_reexported_names_use_the_module_path(
    by_title: dict[str, list[Passage]],
) -> None:
    assert "mlx_lm.sample_utils.make_sampler" in by_title
    assert "mlx_lm.models.cache.make_prompt_cache" in by_title
    assert "mlx_lm.tuner.utils.build_schedule" in by_title


def test_private_names_are_excluded(by_title: dict[str, list[Passage]]) -> None:
    sample_utils_titles = {
        title for title in by_title if title.startswith("mlx_lm.sample_utils.")
    }
    assert sample_utils_titles == {"mlx_lm.sample_utils.make_sampler"}
    wrapper_method_titles = {
        title
        for title in by_title
        if title.startswith("mlx_lm.tokenizer_utils.TokenizerWrapper.")
    }
    assert wrapper_method_titles == {"mlx_lm.tokenizer_utils.TokenizerWrapper.decode"}


def test_constructor_contract_survives_class_search_and_full_fetch(
    passages: list[Passage],
) -> None:
    title = "mlx_lm.tokenizer_utils.TokenizerWrapper"
    index = service_in_memory(passages)
    index.models.reranker = FakeReranker(lambda query, texts: range(len(texts)))
    try:
        searched = asyncio.run(index.search(title, limit=1, package=Package.LM))
        url = next(p.url for p in passages if p.title == title)
        fetched = complete_fetch(index, url)
        for result in (searched, fetched):
            assert "def __init__(" in result
            assert "tokenizer," in result
            assert "detokenizer_class=None" in result
            assert "eos_token_ids: set[int] | None = None" in result
            assert "Initialize the wrapper around the required tokenizer." in result
        assert "def decode(" in fetched
        assert "_private_helper" not in fetched
    finally:
        index.close()


def test_dataclass_fields_remain_in_the_class_reference(
    by_title: dict[str, list[Passage]],
) -> None:
    text = by_title["mlx_lm.generate.GenerationResponse"][0].text
    assert "@dataclass\nclass GenerationResponse:" in text
    assert "    text: str\n    token: int" in text
    assert "    finish_reason: Optional[str] = None" in text
    assert "def __init__(" not in text


def _constructor_source(parameters: str) -> str:
    """Build source whose undefined dependencies prohibit runtime introspection."""
    return f'''
@decorator
class Adapter(Base, metaclass=Meta):
    """Class contract."""

    revision: int = 2

    def __init__({parameters}) -> None:
        """Initialize tokenizer_state without importing its implementation.

        Args:
            tokenizer: Required tokenizer_state.
        """
        raise RuntimeError("Do not execute source")

    def decode(self):
        """Public method contract."""

    def _private(self):
        """Private implementation contract."""

    def __repr__(self):
        """Private special-method contract."""
'''


@given(
    parameter_form=st.sampled_from(
        (
            "self, tokenizer{annotation}, option{annotation} = {default}",
            "self, tokenizer{annotation}, /, option{annotation} = {default}",
            "self, tokenizer{annotation}, *, option{annotation} = {default}",
            "self, tokenizer{annotation}, *args: int, option{annotation} = {default}, **kwargs: object",
            "self, tokenizer{annotation}, /, option{annotation} = {default}, *, required: str, **kwargs: object",
        )
    ),
    annotation=st.sampled_from(("", ": object", ": list[int] | None", ": 'Tokenizer'")),
    default=st.sampled_from(
        (
            "None",
            "False",
            "-3",
            "'tokenizer_state'",
            "(1, 2)",
            "[1, 2]",
            "{'eos': 3}",
            "factory(2)",
            "Defaults.STANDARD",
            "lambda token: token",
        )
    ),
)
def test_constructor_signatures_round_trip_without_execution(
    parameter_form: str, annotation: str, default: str
) -> None:
    parameters = parameter_form.format(annotation=annotation, default=default)
    source = _constructor_source(parameters)
    original = ast.parse(source).body[0]
    assert isinstance(original, ast.ClassDef)
    with TemporaryDirectory() as temporary:
        wheel = _build_wheel(Path(temporary), {"sample_utils.py": source})
        passages = api_passages(wheel, retrieved_at=_RETRIEVED_AT)
    validate_passages(passages)
    title = "mlx_lm.sample_utils.Adapter"
    adapter = next(p for p in passages if p.title == title)
    code = adapter.segment_text.split("```python\n", 1)[1].split("\n```", 1)[0]
    assert "def __init__(" in code
    rendered = ast.parse(code + "\n        pass\n").body[0]
    assert isinstance(rendered, ast.ClassDef)
    constructors = [n for n in rendered.body if isinstance(n, ast.FunctionDef)]
    assert [n.name for n in constructors] == ["__init__"]
    original_init = next(n for n in original.body if isinstance(n, ast.FunctionDef))
    assert ast.dump(constructors[0].args) == ast.dump(original_init.args)
    assert constructors[0].returns is not None and original_init.returns is not None
    assert ast.dump(constructors[0].returns) == ast.dump(original_init.returns)
    assert ast.dump(rendered.decorator_list[0]) == ast.dump(original.decorator_list[0])
    assert "class Adapter(Base, metaclass=Meta):" in code
    assert "    revision: int = 2" in code
    assert "Required tokenizer_state." in adapter.segment_text
    assert "Class contract." in adapter.segment_text
    assert "Private" not in adapter.segment_text
    assert "raise RuntimeError" not in adapter.segment_text
    assert (
        adapter.url
        == LM_SOURCE.blob_base + f"mlx_lm/sample_utils.py#L{original.lineno}"
    )
    assert adapter.source_revision == LM_SOURCE.commit
    assert adapter.section_urls == (adapter.url,)
    assert adapter.package is Package.LM
    assert adapter.evidence_kind is EvidenceKind.API
    assert {p.title for p in passages if p.title.startswith(title)} == {
        title,
        title + ".decode",
    }


def test_all_lm_evidence_survives_index_and_mcp_package_boundary(
    passages: list[Passage], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    guides = parse_markdown(
        "# Models\n\nUse the guide.",
        LM_SOURCE.blob_base + "README.md",
        retrieved_at=_RETRIEVED_AT,
    )
    lm = passages + guides + help_passages(LM_SOURCE.capture_file)
    passages = lm + [
        Passage("Core", Package.MLX.base_url + "index.html", "Core"),
        Passage(
            "Data", Package.DATA.base_url + "index.html", "Data", package=Package.DATA
        ),
    ]

    write_index(tmp_path, passages)
    assert load_index(tmp_path).passages == passages
    index = DocumentationService(tmp_path)
    index.models.reranker = FakeReranker(lambda query, texts: range(len(texts)))
    assert {p.package for p in index.retrieve("generate", package=Package.LM)} == {
        Package.LM
    }
    for p in lm:
        fetched = index.fetch(p.url)
        assert (
            "Package: MLX LM" in fetched
            and f"Evidence: {p.evidence_kind.value}" in fetched
        )
    packages = index.documentation_info()["packages"]
    assert isinstance(packages, dict)
    info = packages["mlx-lm"]
    assert info["documentation_versions"] == [LM_SOURCE.version]
    assert set(info["evidence_kinds"]) == {
        kind.value for kind in (EvidenceKind.GUIDE, EvidenceKind.API, EvidenceKind.CLI)
    }
    assert build_symbol_index(guides).missing_packages("mlx-lm guides") == []
    monkeypatch.setattr(server, "_service", LazyResource(lambda: index))

    async def check() -> None:
        async with Client(server.create_server()) as client:
            catalog = await client.list_tools()
            assert "mlx-lm" in json.dumps(
                next(
                    tool_schema(t) for t in catalog.tools if t.name == "search_mlx_docs"
                )
            )
            result = await client.call_tool(
                "search_mlx_docs", {"query": "mlx_lm.generate", "package": "mlx-lm"}
            )
            assert not result.is_error and "Package: MLX LM" in tool_text(result)

    try:
        asyncio.run(check())
    finally:
        index.models.worker.close()
