"""Independent upstream logits for the production six-layer, padded-head model."""

import os
from collections import defaultdict

import mlx.core as mx
import pytest
from pydantic import TypeAdapter

from mlx_mcp.configuration import cache_root
from mlx_mcp.model_runtime import ModelAccess, resolve_model_snapshot
from mlx_mcp.pinned_models import RERANK_SOURCE
from mlx_mcp.reranking import BATCH_SIZE, MLXCrossEncoder
from scripts.generate_reranker_reference import FIXTURE, ReferenceFile, ReferencePair

# Absolute logit tolerance for this fixed fixture, not an all-input error bound.
LOGIT_TOLERANCE = 0.03


def reference() -> ReferenceFile:
    return TypeAdapter(ReferenceFile).validate_json(FIXTURE.read_text(), strict=True)


@pytest.mark.skipif(
    os.environ.get("MLX_MCP_LIVE_TESTS") != "1", reason="requires cached pinned MiniLM"
)
def test_production_logits_match_independent_cpu_reference() -> None:
    directory = resolve_model_snapshot(
        RERANK_SOURCE, cache_root() / "models", access=ModelAccess.CACHED
    )
    groups: dict[str, list[ReferencePair]] = defaultdict(list)
    for pair in reference()["pairs"]:
        groups[pair["query"]].append(pair)
    errors: list[float] = []
    with mx.stream(mx.gpu):
        encoder = MLXCrossEncoder(directory)
        assert encoder.layers == 6 and encoder.heads == 12
        for query, pairs in groups.items():
            for pair in pairs:
                encoded = encoder.tokenizer.encode(query, pair["document"])
                assert encoded.ids == pair["input_ids"]
                assert encoded.type_ids == pair["token_type_ids"]
            # Repeat fixed inputs across the production batch boundary. Mixed
            # lengths exercise masks; singleton runs independently vary padding.
            batch = (pairs * (BATCH_SIZE // len(pairs) + 1))[: BATCH_SIZE + 1]
            observed = list(encoder.pair_scores(query, [p["document"] for p in batch]))
            expected = [p["logit"] for p in batch]
            errors.extend(abs(a - b) for a, b in zip(observed, expected, strict=True))
            assert observed == pytest.approx(expected, abs=LOGIT_TOLERANCE, rel=0)
            singles = [
                next(iter(encoder.pair_scores(query, [p["document"]]))) for p in pairs
            ]
            assert singles == pytest.approx(
                [p["logit"] for p in pairs], abs=LOGIT_TOLERANCE, rel=0
            )
            for i, left in enumerate(pairs):
                for j, right in enumerate(pairs):
                    if left["logit"] - right["logit"] > 2 * LOGIT_TOLERANCE:
                        assert singles[i] > singles[j] and observed[i] > observed[j]
    print(f"maximum batched absolute logit error: {max(errors):.8f}")
