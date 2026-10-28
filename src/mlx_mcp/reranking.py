"""Rerank passages with pinned MiniLM on MLX, scoring token-bounded windows.

MLXCrossEncoder scores query/window pairs in padded batches; WindowedReranker
scores each passage by its best window.
"""

import math
import re
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast

import mlx.core as mx
import numpy as np
from markdown_it import MarkdownIt
from numpy.typing import NDArray
from pydantic import TypeAdapter
from tokenizers import Tokenizer

from mlx_mcp.passages import Passage
from mlx_mcp.search_text import retrieval_input
from mlx_mcp.token_windows import TokenWindows

# MiniLM's position embeddings cover 512 tokens, query and window together.
MAX_TOKENS = 512
# Pairs per forward pass; candidate passages can contribute several windows.
BATCH_SIZE = 32
# Narrowest head width MLX 0.32's fused attention takes; 32 to 56 fall back to
# building the full attention matrix.
_FUSED_HEAD_SIZE = 64


def _gelu(x: mx.array) -> mx.array:
    """BERT's GELU, the exact erf form rather than the tanh approximation."""
    return x * (1 + mx.erf(x / math.sqrt(2))) / 2


# Compile GELU without specializing on batch and sequence lengths.
_fused_gelu = mx.compile(_gelu, shapeless=True)


class _Evaluator(Protocol):
    """mx.eval as called here; its stub leaves the arguments untyped."""

    def eval(self, x: mx.array, /) -> None: ...


class _BertConfig(TypedDict):
    hidden_act: Literal["gelu"]
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    layer_norm_eps: float


class _TokenizerConfig(TypedDict):
    model_max_length: int
    pad_token: str


class MLXCrossEncoder:
    """Pinned MiniLM pair logits, computed by MLX in half precision.

    This is BertForSequenceClassification: embeddings, encoder layers, the
    pooler on the first token, then one relevance logit. Each batch is right
    padded to its longest pair, and padding is masked out of attention.
    """

    def __init__(self, directory: Path) -> None:
        config = TypeAdapter(_BertConfig).validate_json(
            (directory / "config.json").read_text()
        )
        tokenizer_config = TypeAdapter(_TokenizerConfig).validate_json(
            (directory / "tokenizer_config.json").read_text(), strict=True
        )
        if (
            config["max_position_embeddings"] != MAX_TOKENS
            or tokenizer_config["model_max_length"] != MAX_TOKENS
        ):
            raise ValueError("MiniLM token limit does not match the reranker's")
        self.max_tokens = MAX_TOKENS
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        pad_id = self.tokenizer.token_to_id(tokenizer_config["pad_token"])
        if pad_id is None:
            raise ValueError("MiniLM padding token is missing from its tokenizer")
        self.pad_id = pad_id
        self.layers = config["num_hidden_layers"]
        self.heads = config["num_attention_heads"]
        self.eps = config["layer_norm_eps"]
        # A .safetensors file always loads as a name-to-array mapping.
        weights = cast(
            dict[str, mx.array], mx.load(str(directory / "model.safetensors"))
        )
        # Keep inference weights in half precision; the reference-logit test
        # checks the resulting numerical error on fixed upstream-model inputs.
        self.weights = {
            name: array.astype(mx.float16) for name, array in weights.items()
        }

    def _linear(self, x: mx.array, name: str) -> mx.array:
        return mx.addmm(
            self.weights[f"{name}.bias"], x, self.weights[f"{name}.weight"].T
        )

    def _norm(self, x: mx.array, name: str) -> mx.array:
        return mx.fast.layer_norm(
            x, self.weights[f"{name}.weight"], self.weights[f"{name}.bias"], self.eps
        )

    def _logits(self, ids: mx.array, types: mx.array, keep: mx.array) -> mx.array:
        """Return one logit per row; keep marks real tokens, not padding."""
        w = self.weights
        batch, length = ids.shape
        x = self._norm(
            w["bert.embeddings.word_embeddings.weight"][ids]
            + w["bert.embeddings.position_embeddings.weight"][:length]
            + w["bert.embeddings.token_type_embeddings.weight"][types],
            "bert.embeddings.LayerNorm",
        )
        width = x.shape[-1]
        size = width // self.heads
        # Pad MiniLM's 32-wide heads for MLX's fused attention kernel. Preserve
        # the original scale and discard padded output channels: zero padding
        # preserves the real-valued operation, though kernel rounding differs.
        pad = max(0, _FUSED_HEAD_SIZE - size)

        def split(t: mx.array) -> mx.array:
            heads = mx.reshape(t, (batch, length, self.heads, size))
            heads = mx.pad(heads, [(0, 0), (0, 0), (0, 0), (0, pad)])
            return mx.transpose(heads, (0, 2, 1, 3))

        for layer in range(self.layers):
            p = f"bert.encoder.layer.{layer}."
            attention = mx.fast.scaled_dot_product_attention(
                split(self._linear(x, p + "attention.self.query")),
                split(self._linear(x, p + "attention.self.key")),
                split(self._linear(x, p + "attention.self.value")),
                scale=1 / math.sqrt(size),
                mask=keep[:, None, None, :],
            )
            attention = mx.reshape(
                mx.transpose(attention[..., :size], (0, 2, 1, 3)),
                (batch, length, width),
            )
            x = self._norm(
                x + self._linear(attention, p + "attention.output.dense"),
                p + "attention.output.LayerNorm",
            )
            hidden = _fused_gelu(self._linear(x, p + "intermediate.dense"))
            x = self._norm(
                x + self._linear(hidden, p + "output.dense"), p + "output.LayerNorm"
            )
            # Materialize this output so later layers do not retain its lazy graph.
            cast(_Evaluator, mx).eval(x)
        pooled = mx.tanh(self._linear(x[:, 0], "bert.pooler.dense"))
        return self._linear(pooled, "classifier")[:, 0]

    def pair_scores(self, query: str, documents: list[str]) -> Iterable[float]:
        """Return one logit per document, in input order.

        Query/document pairs include special tokens in max_tokens. An oversized
        pair raises ValueError before any inference; an empty list returns
        nothing.
        """
        encoded = self.tokenizer.encode_batch([(query, text) for text in documents])
        if any(len(item.ids) > self.max_tokens for item in encoded):
            raise ValueError("Ranking window exceeds the model's token limit")
        scores: list[float] = []
        for start in range(0, len(encoded), BATCH_SIZE):
            batch = encoded[start : start + BATCH_SIZE]
            shape = (len(batch), max(len(item.ids) for item in batch))
            ids = np.full(shape, self.pad_id, dtype=np.int32)
            types = np.zeros(shape, dtype=np.int32)
            for row, item in enumerate(batch):
                ids[row, : len(item.ids)] = item.ids
                types[row, : len(item.ids)] = item.type_ids
            lengths = mx.array([len(item.ids) for item in batch])
            keep = mx.arange(shape[1])[None, :] < lengths[:, None]
            logits = self._logits(mx.array(ids), mx.array(types), keep)
            scores.extend(np.asarray(logits, dtype=np.float64).tolist())
        # Return unused allocator buffers after each request to limit retained
        # cache memory between searches. MLX reuses cached allocations by size.
        mx.clear_cache()
        return scores


class CrossEncoder(Protocol):
    """A loaded pair scorer and its exact tokenizer/window limit."""

    @property
    def tokenizer(self) -> Tokenizer: ...

    @property
    def max_tokens(self) -> int: ...

    def pair_scores(self, query: str, documents: list[str], /) -> Iterable[float]: ...


class WindowedReranker(TokenWindows):
    """Score each canonical passage by its best exact token-bounded window."""

    def __init__(self, backend: CrossEncoder) -> None:
        self.backend = backend
        super().__init__(backend.tokenizer, backend.max_tokens)

    def _ranking_windows(self, passage: Passage, query: str) -> list[Passage]:
        """Keep full windows and separately score list fields named in query.

        A long list can dilute one requested field's definition. Parse canonical
        source so chunks beginning inside fences remain code. Definitions must
        start within the passage; continuation text keeps its existing bounds.
        """
        windows = self.windows(passage, query)
        source = passage.segment_text if passage.section_path else passage.text
        lower = passage.chunk_start if passage.section_path else 0
        upper = lower + len(passage.text)
        offsets = [0]
        for line in source.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        for item in MarkdownIt("commonmark").parse(source):
            if item.type != "list_item_open" or item.map is None:
                continue
            start, end = (offsets[line] for line in item.map)
            if not lower <= start < upper:
                continue
            end = min(end, upper)
            text = source[start:end]
            field = re.match(r"\s*(?:[-+*]|\d+[.)])\s+`([A-Za-z_]\w*)`\s*:", text)
            if field is None or not re.search(
                rf"(?<!\w){re.escape(field.group(1))}(?!\w)", query, re.IGNORECASE
            ):
                continue
            definition = replace(
                passage,
                text=text,
                chunk_start=start if passage.section_path else passage.chunk_start,
                chunk_end=end if passage.section_path else passage.chunk_end,
            )
            windows.extend(
                window
                for window in self.windows(definition, query)
                if window not in windows
            )
        return windows

    def score(self, query: str, passages: list[Passage]) -> NDArray[np.float64]:
        """Score each passage by its best full or named-field window.

        Raises ValueError for a query that check_query rejects.
        """
        self.check_query(query)
        groups = [self._ranking_windows(passage, query) for passage in passages]
        documents = [
            retrieval_input(p.title, p.text) for group in groups for p in group
        ]
        scores = np.asarray(list(self.backend.pair_scores(query, documents)))
        if (
            scores.shape != (len(documents),)
            or scores.dtype.kind not in "fiu"
            or not np.isfinite(scores).all()
        ):
            raise ValueError("Reranker must return one finite score per ranking window")
        ends = np.cumsum([len(group) for group in groups])
        return np.array(
            [max(scores[start:end]) for start, end in zip([0, *ends[:-1]], ends)],
            dtype=np.float64,
        )
