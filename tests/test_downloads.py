"""Documentation downloads: allowed origins, redirects, size limits, and retries."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError

import pytest

from mlx_mcp import downloads
from mlx_mcp.mlx_lm_sources import LM_SOURCE
from mlx_mcp.packages import Package


@pytest.mark.parametrize(
    "url",
    [
        LM_SOURCE.raw_base.replace(LM_SOURCE.commit, "main") + "README.md",
        LM_SOURCE.raw_base + "../secret",
    ],
)
def test_only_exact_manifest_inputs_are_downloadable(url: str) -> None:
    with pytest.raises(ValueError, match="exact pinned LM sources"):
        downloads.download_bytes(url)


def test_empty_package_cannot_silently_disappear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download_text(url: str):
        return 'Search.setIndex({"docnames": []})'

    monkeypatch.setattr(downloads, "download_text", fake_download_text)
    with pytest.raises(ValueError, match="No documentation pages listed for MLX Data"):
        downloads.download_package(Package.DATA)


@dataclass
class Upstream:
    hits: list[str] = field(default_factory=list[str])
    statuses: list[int] = field(default_factory=lambda: [200])
    body: bytes = b"documentation"
    location: str = ""


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> Iterator[Upstream]:
    state = Upstream()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            state.hits.append(self.path)
            status = state.statuses[min(len(state.hits) - 1, len(state.statuses) - 1)]
            self.send_response(status)
            if state.location:
                self.send_header("Location", state.location)
            self.end_headers()
            self.wfile.write(state.body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    monkeypatch.setattr(
        Package,
        "base_url",
        property(lambda self: f"http://127.0.0.1:{server.server_port}/docs/"),
    )
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("status", [302, 308])
@pytest.mark.parametrize("external", [False, True])
def test_redirect_is_rejected_before_a_second_request(
    upstream: Upstream, status: int, external: bool
) -> None:
    upstream.statuses = [status]
    upstream.location = (
        "http://127.0.0.1:1/private" if external else Package.MLX.base_url + "target"
    )
    with pytest.raises(ValueError, match="redirects are not allowed"):
        downloads.download_text(Package.MLX.base_url + "start")
    assert upstream.hits == ["/docs/start"]


def test_download_size_bound(
    upstream: Upstream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(downloads, "MAX_DOWNLOAD_BYTES", 32)
    upstream.body = b"x" * 33
    with pytest.raises(ValueError, match="download exceeds"):
        downloads.download_text(Package.MLX.base_url + "large")
    assert len(upstream.hits) == 1


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "https://ml-explore.github.io.evil.invalid/mlx/build/html/index.html",
        "http://ml-explore.github.io/mlx/build/html/index.html",
        "https://ml-explore.github.io/mlx-data/build/html/../../secret",
    ],
)
def test_unapproved_initial_urls_fail_before_network(url: str) -> None:
    with pytest.raises(ValueError, match="official MLX"):
        downloads.download_text(url)


@pytest.mark.parametrize(
    "statuses, success, count",
    [
        ([503, 200], True, 2),
        ([404], False, 1),
    ],
)
def test_retry_is_bounded_and_only_transient(
    upstream: Upstream,
    monkeypatch: pytest.MonkeyPatch,
    statuses: list[int],
    success: bool,
    count: int,
) -> None:
    delays: list[float] = []
    monkeypatch.setattr(downloads.time, "sleep", delays.append)
    upstream.statuses = statuses
    if success:
        assert (
            downloads.download_text(Package.MLX.base_url + "retry") == "documentation"
        )
    else:
        with pytest.raises(HTTPError):
            downloads.download_text(Package.MLX.base_url + "retry")
    assert len(upstream.hits) == count
    assert delays == ([1] if count == 2 else [])
