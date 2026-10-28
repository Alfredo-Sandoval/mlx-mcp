"""Pinned model sources reject floating or malformed revisions."""

from dataclasses import replace

import pytest

from mlx_mcp.pinned_models import RERANK_SOURCE


@pytest.mark.parametrize("revision", ["main", "a" * 39])
def test_floating_or_invalid_model_revision_cannot_be_constructed(
    revision: str,
) -> None:
    with pytest.raises(ValueError, match="immutable"):
        replace(RERANK_SOURCE, revision=revision)
