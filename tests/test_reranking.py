"""The MLX MiniLM cross-encoder and windowed passage scoring."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
import pytest
from helpers import SYNTHETIC_TOKENIZER, fake_cross_encoder
from hypothesis import given, settings
from hypothesis import strategies as st

from mlx_mcp.passages import Passage, source_text, validate_passages
from mlx_mcp.reranking import BATCH_SIZE, MLXCrossEncoder, WindowedReranker
from mlx_mcp.search_text import retrieval_input
from mlx_mcp.sphinx_pages import parse_page
from mlx_mcp.token_windows import DOCUMENT_TOKENS


class _Saver(Protocol):
    """mx.save_safetensors as called here; its stub has no return type."""

    def save_safetensors(self, file: str, arrays: dict[str, mx.array]) -> None: ...


def tiny_encoder(root: Path) -> MLXCrossEncoder:
    """A one-layer MiniLM-shaped model with seeded random weights."""
    width, hidden = 8, 16
    (root / "tokenizer.json").write_bytes(SYNTHETIC_TOKENIZER.read_bytes())
    (root / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 512, "pad_token": "[UNK]"})
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "hidden_act": "gelu",
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "max_position_embeddings": 512,
                "layer_norm_eps": 1e-12,
            }
        )
    )
    shapes = {
        "bert.embeddings.word_embeddings.weight": (2, width),
        "bert.embeddings.position_embeddings.weight": (512, width),
        "bert.embeddings.token_type_embeddings.weight": (2, width),
        "bert.pooler.dense.weight": (width, width),
        "classifier.weight": (1, width),
    }
    layer = "bert.encoder.layer.0."
    for name in ("query", "key", "value"):
        shapes[f"{layer}attention.self.{name}.weight"] = (width, width)
    shapes[f"{layer}attention.output.dense.weight"] = (width, width)
    shapes[f"{layer}intermediate.dense.weight"] = (hidden, width)
    shapes[f"{layer}output.dense.weight"] = (width, hidden)
    mx.random.seed(0)
    weights: dict[str, mx.array] = {}
    for name, shape in shapes.items():
        weights[name] = mx.random.normal(shape)
        weights[name.removesuffix(".weight") + ".bias"] = mx.random.normal(shape[:1])
    for norm in (
        "bert.embeddings.LayerNorm",
        f"{layer}attention.output.LayerNorm",
        f"{layer}output.LayerNorm",
    ):
        weights[f"{norm}.weight"] = mx.ones((width,))
        weights[f"{norm}.bias"] = mx.zeros((width,))
    cast(_Saver, mx).save_safetensors(str(root / "model.safetensors"), weights)
    return MLXCrossEncoder(root)


def test_batched_scores_match_each_pair_scored_alone(tmp_path: Path) -> None:
    encoder = tiny_encoder(tmp_path)

    # Up to 40 documents crosses a batch boundary; 510 words plus the query
    # word stays within 512 tokens.
    @settings(max_examples=25, deadline=None)
    @given(st.lists(st.integers(min_value=0, max_value=510), max_size=40))
    def check(lengths: list[int]) -> None:
        documents = ["word " * length for length in lengths]
        batched = list(encoder.pair_scores("word", documents))
        alone = [next(iter(encoder.pair_scores("word", [text]))) for text in documents]
        # Half precision rounds by batch shape, up to 0.004 here. Unmasked
        # padding moves these scores by whole units.
        assert batched == pytest.approx(alone, abs=0.02)

    assert BATCH_SIZE < 40
    check()


def test_oversized_pair_fails(tmp_path: Path) -> None:
    encoder = tiny_encoder(tmp_path)
    with pytest.raises(ValueError, match="exceeds the model's token limit"):
        list(encoder.pair_scores("word", ["word", "word " * 512]))


def test_ranking_windows_cover_code_tail_without_changing_canonical_source(
    tmp_path: Path,
) -> None:
    model = WindowedReranker(fake_cross_encoder(tmp_path))
    html = (
        '<article class="bd-article"><pre>'
        + "word " * 2000
        + "ESSENTIAL_TAIL</pre></article>"
    )
    original = parse_page(html, "https://docs/code")
    windows = model.prepare_index_windows(original)
    assert len(windows) > 1 and "ESSENTIAL_TAIL" in windows[-1].text
    assert all(
        model.token_count(retrieval_input(p.title, p.text)) <= DOCUMENT_TOKENS
        for p in windows
    )
    assert source_text(windows) == source_text(original)
    validate_passages(windows)
    assert model.prepare_index_windows(windows) == windows
    query = "word " * 250
    pairs = [model.windows(p, query) for p in windows]
    assert all(
        model.token_count(query, retrieval_input(p.title, p.text)) <= 512
        for group in pairs
        for p in group
    )
    model.backend.pair_scores = lambda query, docs: [
        float("ESSENTIAL_TAIL" in text) for text in docs
    ]
    scores = model.score(query, windows)
    assert scores[-1] == 1 and sum(scores) == 1
    with pytest.raises(ValueError, match="Shorten the query"):
        model.score("word " * 600, windows)


def test_named_field_ranking_keeps_its_definition_and_source_range(
    tmp_path: Path,
) -> None:
    field, chunk_start = "timeout", 0
    model = WindowedReranker(fake_cross_encoder(tmp_path))
    definition = (
        f"- `{field}`: Wait for the configured duration.\n  Zero disables waiting.\n"
    )
    example = f"- `{field}`: This is a code example, not a field definition.\n"
    source = (
        "# Fields\n\n"
        + definition
        + "- `capacity`: Maximum number of queued jobs.\n\n```text\n"
        + example
        + "\nOther example code.\n```"
    )
    passage = Passage(
        "Reference / Fields",
        "https://docs/reference#fields",
        source[chunk_start:],
        source,
        section_path=("reference", "fields"),
        section_urls=("https://docs/reference", "https://docs/reference#fields"),
        segment_index=0,
        chunk_start=chunk_start,
        chunk_end=len(source),
    )
    model.backend.pair_scores = lambda query, docs: [
        2.0
        if text
        == f"Reference / Fields\n{field}: Wait for the configured duration.\nZero disables waiting."
        else 3.0
        if text
        == f"Reference / Fields\n{field}: This is a code example, not a field definition."
        else 1.0
        for text in docs
    ]
    assert model.score(f"How do I set {field}?", [passage]).tolist() == [2.0]
    # A prefix is not the documented field; the complete original still gets scored.
    assert model.score(f"How do I set {field}_extra?", [passage]).tolist() == [1.0]
    # The same field-looking line remains code when a chunk omits its opening fence.
    example_start = source.index(example)
    inside_fence = replace(
        passage,
        text=source[example_start:-3],
        chunk_start=example_start,
        chunk_end=len(source) - 3,
    )
    assert model.score(f"How do I set {field}?", [inside_fence]).tolist() == [1.0]
    continuation = source.index("  Zero")
    without_definition = replace(
        passage,
        text=source[continuation:],
        chunk_start=continuation,
    )
    assert model.score(f"How do I set {field}?", [without_definition]).tolist() == [1.0]
    assert source_text([passage]) == source
