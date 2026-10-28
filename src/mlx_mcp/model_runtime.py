"""Load the pinned reranker and serialize its inference on one worker."""

from __future__ import annotations

import gc
import logging
import time
from enum import Enum
from functools import cached_property, partial
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mlx_mcp.index_file import FloatScalar
from mlx_mcp.passages import Passage
from mlx_mcp.pinned_models import RERANK_SOURCE, ModelSource
from mlx_mcp.workers import BlockingWorker, LazyResource

if TYPE_CHECKING:
    from mlx_mcp.reranking import WindowedReranker
    from mlx_mcp.token_windows import TokenWindows

type ScoreArray = NDArray[
    FloatScalar
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
]


class _Hub(Protocol):
    """snapshot_download as called here; huggingface_hub's overloads are partly untyped."""

    def snapshot_download(
        self,
        repo_id: str,
        *,
        revision: str,
        allow_patterns: list[str],
        cache_dir: str,
        local_files_only: bool,
    ) -> str: ...


class ModelAccess(Enum):
    """Whether a model loader may download missing files."""

    CACHED = "cached"
    DOWNLOAD = "download"


class Reranker(Protocol):
    """Cross-encoder scoring: one relevance score per passage."""

    def score(self, query: str, passages: list[Passage]) -> ArrayLike: ...


def resolve_model_snapshot(
    source: ModelSource, cache: Path, *, access: ModelAccess = ModelAccess.DOWNLOAD
) -> Path:
    """Return the pinned snapshot, downloading it only when it is not cached.

    With ModelAccess.CACHED, raise FileNotFoundError instead of downloading.
    """
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    hub = cast(_Hub, huggingface_hub)

    def download(local_files_only: bool) -> Path:
        # The Hub tells anonymous downloads to set HF_TOKEN. The pinned model is
        # public, so that advice would only send users to make an account.
        hub_http = logging.getLogger("huggingface_hub.utils._http")
        level = hub_http.level
        hub_http.setLevel(logging.ERROR)
        try:
            return Path(
                hub.snapshot_download(
                    repo_id=source.repository,
                    revision=source.revision,
                    allow_patterns=list(source.files),
                    cache_dir=str(cache),
                    local_files_only=local_files_only,
                )
            )
        finally:
            hub_http.setLevel(level)

    try:
        directory = download(local_files_only=True)
        if all((directory / name).is_file() for name in source.files):
            return directory
    except LocalEntryNotFoundError:
        pass
    if access is ModelAccess.CACHED:
        raise FileNotFoundError(
            f"The search model {source.name} isn't fully downloaded. Run `mlx-mcp prepare`."
        )
    directory = download(local_files_only=False)
    missing = [name for name in source.files if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"The download of {source.name} is missing {', '.join(missing)}"
        )
    return directory


def load_reranker(
    root: Path, *, access: ModelAccess = ModelAccess.DOWNLOAD
) -> WindowedReranker:
    """Load pinned MiniLM on MLX."""
    from mlx_mcp.reranking import MLXCrossEncoder, WindowedReranker

    directory = resolve_model_snapshot(RERANK_SOURCE, root / "models", access=access)
    return WindowedReranker(MLXCrossEncoder(directory))


def load_rerank_tokenizer(
    root: Path, *, access: ModelAccess = ModelAccess.DOWNLOAD
) -> TokenWindows:
    """Load only MiniLM's tokenizer, to measure queries without loading the model.

    It reads the same snapshot and uses the same token budget as load_reranker,
    so its counts match the length check inside the reranker's score.
    """
    from tokenizers import Tokenizer

    from mlx_mcp.reranking import MAX_TOKENS
    from mlx_mcp.token_windows import TokenWindows

    directory = resolve_model_snapshot(RERANK_SOURCE, root / "models", access=access)
    return TokenWindows(
        Tokenizer.from_file(str(directory / "tokenizer.json")), MAX_TOKENS
    )


class ModelRuntime:
    """One serialized inference worker and its lazily loaded reranker.

    The service submits inference to worker, and cleanup is queued on that
    same worker after use.
    """

    def __init__(
        self, root: Path, *, access: ModelAccess = ModelAccess.DOWNLOAD
    ) -> None:
        self._reranker = LazyResource(lambda: load_reranker(root, access=access))
        self._rerank_tokenizer = LazyResource(
            lambda: load_rerank_tokenizer(root, access=access)
        )
        self.lock = Lock()
        self._last_model_use = time.monotonic()

    @cached_property
    def reranker(self) -> Reranker:
        return self._reranker.get()

    @cached_property
    def rerank_tokenizer(self) -> TokenWindows:
        """The reranker's tokenizer alone; small enough to stay loaded when idle."""
        return self._rerank_tokenizer.get()

    @cached_property
    def worker(self) -> BlockingWorker:
        return BlockingWorker("inference", capacity=64)

    def release_models(self, idle_seconds: float = 0) -> None:
        """Unload the reranker once the idle interval has elapsed.

        Run cleanup on the calling thread; zero idle_seconds requests immediate
        cleanup. Once the inference worker serves requests, callers must schedule
        cleanup there. release_idle_models handles that scheduling for idle checks.
        """
        with self.lock:
            if time.monotonic() - self._last_model_use < idle_seconds:
                return
            if "reranker" not in self.__dict__ and self._reranker.peek() is None:
                return
            self.__dict__.pop("reranker", None)
            self._reranker.clear()
            gc.collect()
            # Loading the reranker imported MLX already. Freed weights stay in
            # MLX's buffer cache until it is cleared.
            import mlx.core as mx

            mx.clear_cache()

    async def release_idle_models(self, idle_seconds: float) -> None:
        """Check the idle reranker on its worker, without creating an unused worker."""
        worker = self.__dict__.get("worker")
        if isinstance(worker, BlockingWorker):
            await worker.maintain(partial(self.release_models, idle_seconds))

    def close(self) -> None:
        """Drain inference, then release the reranker on the owning thread."""
        worker = self.__dict__.get("worker")
        if isinstance(worker, BlockingWorker):
            worker.close(self.release_models)
        else:
            self.release_models()

    def check_query(self, query: str) -> None:
        """Raise ValueError if an expanded query is too long for the reranker.

        Measures with the reranker's tokenizer alone, so a search that scores
        nothing (zero or one candidate) still never loads the reranker.
        """
        self.rerank_tokenizer.check_query(query)

    def score(self, query: str, passages: list[Passage]) -> ScoreArray:
        """Score an already expanded query, requiring one finite logit per passage."""
        with self.lock:
            try:
                scores = self.reranker.score(query, passages)
            finally:
                self._last_model_use = time.monotonic()
        scores = np.asarray(scores)
        if (
            scores.shape != (len(passages),)
            or scores.dtype.kind not in "fiu"
            or not np.isfinite(scores).all()
        ):
            raise ValueError(
                "Reranker must return one finite score per candidate passage"
            )
        return cast(ScoreArray, scores)
