"""Canonical identity of the immutable reranker snapshot used by search."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSource:
    """One immutable Hugging Face snapshot and its required runtime files."""

    name: str
    repository: str
    revision: str
    files: tuple[str, ...]

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ValueError(
                "Model sources require an immutable 40-character commit revision"
            )


RERANK_SOURCE = ModelSource(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"),
)
