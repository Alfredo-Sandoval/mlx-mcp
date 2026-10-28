"""Regenerate independent CPU/float32 MiniLM logits in a separate reference env.

Requires torch and transformers in that environment, not in mlx-mcp. Read the
queries/documents from --inputs, load the pinned local snapshot, and write a new
fixture. This script never imports the native MLX model implementation.
"""

import argparse
import json
from contextlib import AbstractContextManager
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast

from pydantic import TypeAdapter
from tokenizers import Tokenizer

from mlx_mcp.pinned_models import RERANK_SOURCE

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/reranker_reference.json"


class ReferencePair(TypedDict):
    id: str
    query: str
    document: str
    input_ids: list[int]
    token_type_ids: list[int]
    logit: float


class ReferenceFile(TypedDict):
    """Fixed inputs and independently computed, unnormalized relevance logits."""

    schema_version: Literal[1]
    repository: str
    revision: str
    versions: dict[str, str]
    device: Literal["cpu"]
    dtype: Literal["float32"]
    attention: Literal["eager"]
    pairs: list[ReferencePair]


class _Tensor(Protocol):
    def tolist(self) -> list[list[float]]: ...


class _Output(Protocol):
    logits: _Tensor


class _Model(Protocol):
    def eval(self) -> object: ...
    def to(self, device: str) -> object: ...
    def __call__(
        self, *, input_ids: _Tensor, token_type_ids: _Tensor, attention_mask: _Tensor
    ) -> _Output: ...


class _Loader(Protocol):
    def from_pretrained(
        self,
        path: str,
        *,
        local_files_only: bool,
        dtype: object,
        attn_implementation: str,
        use_safetensors: bool,
    ) -> _Model: ...


class _Torch(Protocol):
    """Only the installed PyTorch interfaces used by this optional generator."""

    float32: object
    long: object

    def tensor(
        self, data: list[list[int]], *, dtype: object, device: str
    ) -> _Tensor: ...
    def inference_mode(self) -> AbstractContextManager[object]: ...


def reference_pairs(directory: Path, pairs: list[ReferencePair]) -> list[ReferencePair]:
    """Use upstream BERT, independently of native MLX attention and batching."""
    torch = cast(_Torch, import_module("torch"))
    loader = cast(_Loader, import_module("transformers").BertForSequenceClassification)
    model = loader.from_pretrained(
        str(directory),
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
        use_safetensors=True,
    )
    model.to("cpu")
    model.eval()
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    results: list[ReferencePair] = []
    with torch.inference_mode():
        for pair in pairs:
            encoded = tokenizer.encode(pair["query"], pair["document"])
            if len(encoded.ids) > 512:
                raise ValueError(f"Reference pair {pair['id']} exceeds 512 tokens")
            tensors = [
                torch.tensor([values], dtype=torch.long, device="cpu")
                for values in (encoded.ids, encoded.type_ids, [1] * len(encoded.ids))
            ]
            logits = model(
                input_ids=tensors[0],
                token_type_ids=tensors[1],
                attention_mask=tensors[2],
            ).logits.tolist()
            results.append(
                {
                    **pair,
                    "input_ids": encoded.ids,
                    "token_type_ids": encoded.type_ids,
                    "logit": logits[0][0],
                }
            )
    return results


class Options(argparse.Namespace):
    snapshot: Path
    inputs: Path
    output: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(namespace=Options())
    if args.output.exists():
        raise ValueError(
            "Output must be a new file; review it before replacing the fixture"
        )
    if args.snapshot.name != RERANK_SOURCE.revision:
        raise ValueError("Use the pinned Hugging Face snapshot directory")
    for name in RERANK_SOURCE.files:
        if not (args.snapshot / name).is_file():
            raise FileNotFoundError(args.snapshot / name)
    inputs = TypeAdapter(ReferenceFile).validate_json(
        args.inputs.read_text(), strict=True
    )
    result: ReferenceFile = {
        "schema_version": 1,
        "repository": RERANK_SOURCE.repository,
        "revision": RERANK_SOURCE.revision,
        "versions": {
            name: version(name) for name in ("torch", "transformers", "tokenizers")
        },
        "device": "cpu",
        "dtype": "float32",
        "attention": "eager",
        "pairs": reference_pairs(args.snapshot, inputs["pairs"]),
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
