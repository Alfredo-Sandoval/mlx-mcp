"""Allowlisted downloads of the published docs and pinned MLX LM build inputs."""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from email.message import Message
from itertools import batched
from socket import EAI_AGAIN, gaierror
from typing import IO, NoReturn
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from mlx_mcp import __version__
from mlx_mcp.packages import Package
from mlx_mcp.passages import Passage
from mlx_mcp.sphinx_pages import parse_page, sphinx_docnames

MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024


class _RejectRedirects(HTTPRedirectHandler):
    """Fail on any redirect so downloads stay on the official docs URLs."""

    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> NoReturn:
        fp.close()
        raise ValueError(
            "Documentation redirects are not allowed; check the source URL"
        )


def _transient_download_error(exc: URLError | TimeoutError | ConnectionError) -> bool:
    """Return whether a download error is worth one retry."""
    if isinstance(exc, HTTPError):
        return exc.code in {429, 500, 502, 503, 504}
    reason = exc.reason if isinstance(exc, URLError) else exc
    return isinstance(reason, (TimeoutError, ConnectionError)) or (
        isinstance(reason, gaierror) and reason.errno == EAI_AGAIN
    )


def download_text(url: str) -> str:
    """Download an official docs URL as UTF-8 text, retrying one transient failure.

    Redirects and responses over MAX_DOWNLOAD_BYTES are rejected.
    """
    if not any(package.accepts_url(url) for package in Package):
        raise ValueError(
            "Downloads must use an official MLX or MLX Data documentation URL"
        )
    return download_bytes(url).decode("utf-8")


def download_bytes(url: str) -> bytes:
    """Download an allowlisted docs page or an exact pinned LM build input."""
    from mlx_mcp.mlx_lm_api import MODULES
    from mlx_mcp.mlx_lm_sources import LM_SOURCE

    lm_urls = {
        LM_SOURCE.wheel_url,
        *(LM_SOURCE.raw_base + p for p in LM_SOURCE.guides),
        *(LM_SOURCE.raw_base + "mlx_lm/" + m.path for m in MODULES),
        LM_SOURCE.raw_base + "mlx_lm/__init__.py",
    }
    if url not in lm_urls and not any(p.accepts_url(url) for p in Package):
        raise ValueError("Downloads must use official docs or exact pinned LM sources")
    request = Request(
        url, headers={"User-Agent": f"mlx-mcp/{__version__} docs indexer"}
    )
    opener = build_opener(_RejectRedirects())
    for attempt in range(2):
        try:
            with opener.open(request, timeout=30) as response:
                data = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(data) > MAX_DOWNLOAD_BYTES:
                raise ValueError("Documentation download exceeds the 8 MiB limit")
            return data
        except (URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, HTTPError):
                exc.close()
            if attempt or not _transient_download_error(exc):
                raise
            time.sleep(1)
    raise AssertionError("unreachable")


def download_package(package: Package) -> list[Passage]:
    """Download up to eight Sphinx pages at a time and parse them in source order."""
    if not package.sphinx:
        raise ValueError(f"{package.label} is not a Sphinx documentation site")
    names = sphinx_docnames(download_text(f"{package.base_url}searchindex.js"), package)
    print(
        f"Downloading {len(names)} {package.label} documentation pages...",
        file=sys.stderr,
    )
    passages: list[Passage] = []
    urls = (f"{package.base_url}{name}.html" for name in names)
    count = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        # mdurl publishes partially initialized URL caches, so only downloads
        # run in workers; the caller owns all parsing for this package.
        for batch in batched(urls, 8):
            for url, html in zip(batch, pool.map(download_text, batch), strict=True):
                passages.extend(
                    parse_page(
                        html,
                        url,
                        retrieved_at=datetime.now(UTC).isoformat(),
                        package=package,
                    )
                )
                count += 1
                if count % 50 == 0:
                    print(f"Downloaded {count}/{len(names)} pages", file=sys.stderr)
    if not passages:
        raise ValueError(f"No documentation passages collected for {package.label}")
    return passages
