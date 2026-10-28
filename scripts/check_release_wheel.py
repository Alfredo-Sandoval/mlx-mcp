"""Install a release wheel fresh and check it works online and with networking blocked.

Needs uv and macOS sandbox-exec on an Apple silicon Mac. The wheel is installed
outside the checkout from its declared dependency ranges, not the lockfile.
"""

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, TypedDict

from mcp import Client, StdioServerParameters
from pydantic import TypeAdapter

from mlx_mcp.service import DocumentationInfo


class EvidenceResult(TypedDict):
    query: str
    url: str
    kind: str
    search: str
    fetch: str


class ServerCheck(TypedDict):
    schema_version: Literal[1]
    offline: bool
    network_control_passed: bool
    info: DocumentationInfo
    results: list[EvidenceResult]
    installed: dict[str, str]


class ReleaseOptions(argparse.Namespace):
    wheel: Path | None
    output: Path
    check_refresh: bool
    check_server: bool
    offline: bool
    port: int | None


def network_isolation() -> list[str]:
    """Block all networking; the server talks to its agent over stdin and stdout."""
    if sys.platform != "darwin":
        raise RuntimeError("The release wheel check requires macOS sandbox-exec")
    return ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)"]


@asynccontextmanager
async def connect_installed_server(cache: Path) -> AsyncGenerator[Client]:
    """Start the installed `mlx-mcp serve` the way an agent does, and connect to it."""
    environment = {
        **os.environ,
        "MLX_MCP_CACHE": str(cache),
        "HF_HOME": str(cache.parent / "huggingface"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    parameters = StdioServerParameters(
        command=str(Path(sys.executable).parent / "mlx-mcp"),
        args=["serve"],
        env=environment,
    )
    async with Client(parameters, mode="legacy", read_timeout_seconds=600) as client:
        yield client


async def check_installed_server(offline: bool, port: int) -> ServerCheck:
    """Search and fetch one known fact from MLX, MLX Data and each MLX LM evidence kind.

    The online run must start with an empty cache and reach the parent's
    listener. The offline run must fail to reach it and answer from the cached
    reranker and the bundled index.
    """
    from importlib.metadata import distributions

    import mlx_mcp
    from mlx_mcp.index_file import load_index
    from mlx_mcp.indexing import BUNDLED_ROOT
    from mlx_mcp.mlx_lm_sources import LM_SOURCE
    from mlx_mcp.packages import EvidenceKind, Package
    from mlx_mcp.responses import RESPONSE_CHAR_BUDGET, source_blocks

    if not Path(mlx_mcp.__file__).is_relative_to(Path(sys.prefix)):
        raise RuntimeError("mlx_mcp was not imported from the isolated installation")
    # The parent keeps this listener open: online must connect, offline must fail.
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=3):
            connected = True
    except OSError:
        connected = False
    if connected == offline:
        raise RuntimeError("Network isolation control failed")
    cache = Path.cwd() / "cache"
    if not offline and cache.exists():
        raise RuntimeError("The first server must start with an empty MLX_MCP_CACHE")
    index = load_index(BUNDLED_ROOT)
    async with connect_installed_server(cache) as client:

        async def call(name: str, arguments: dict[str, object]) -> str:
            result = await client.call_tool(name, arguments)
            if (
                result.is_error
                or not result.content
                or result.content[0].type != "text"
            ):
                raise RuntimeError(f"{name} failed: {result}")
            text = result.content[0].text
            if (
                result.structured_content != {"result": text}
                or len(text) > RESPONSE_CHAR_BUDGET
            ):
                raise RuntimeError(f"{name} returned an invalid response envelope")
            return text

        results: list[EvidenceResult] = []
        for package, symbol, fact, kind, source_prefix in (
            (
                Package.MLX,
                "mlx.core.eval",
                "Each argument can be a single array",
                EvidenceKind.DOCUMENTATION,
                Package.MLX.base_url,
            ),
            (
                Package.DATA,
                "mlx.data.Stream.prefetch",
                "Fetch samples in background threads.",
                EvidenceKind.DOCUMENTATION,
                Package.DATA.base_url,
            ),
            (
                Package.LM,
                "MLX LM DWQ lower precision 2-bit 4-bit",
                "Use `mlx_lm.dwq` to run DWQ",
                EvidenceKind.GUIDE,
                LM_SOURCE.blob_base + "mlx_lm/LEARNED_QUANTS.md#dwq",
            ),
            (
                Package.LM,
                "mlx_lm.models.cache.trim_prompt_cache",
                "Trim the model's cache by the given number of tokens.",
                EvidenceKind.API,
                LM_SOURCE.blob_base + "mlx_lm/models/cache.py#L",
            ),
            (
                Package.LM,
                "mlx_lm.generate --xtc-probability --xtc-threshold command-line options",
                "Probability of XTC sampling to happen each next token",
                EvidenceKind.CLI,
                LM_SOURCE.capture_url + "#L",
            ),
        ):
            search = await call(
                "search_mlx_docs",
                {"query": symbol, "package": package.value, "limit": 3},
            )
            matching = [
                block
                for block in source_blocks(search)
                if block.url.startswith(source_prefix)
                and any(
                    p.url == block.url
                    and p.package is package
                    and p.evidence_kind is kind
                    for p in index.passages
                )
            ]
            if not matching:
                raise RuntimeError(
                    f"Search did not retrieve {package.label} {kind.value}: {search}"
                )
            url = matching[0].url
            # The fact must appear under this source, not under another result.
            if fact not in matching[0].text:
                raise RuntimeError(f"Search omitted the fact under {url}: {search}")
            fetched = await call("fetch_mlx_docs", {"url": url})
            if not fetched.startswith(f"Source: {url}\n") or fact not in fetched:
                raise RuntimeError(
                    f"Fetch did not recover the cited {package.label} evidence"
                )
            if package is Package.LM and f"Evidence: {kind.value}" not in fetched:
                raise RuntimeError("Fetch lost LM evidence identity")
            results.append(
                {
                    "query": symbol,
                    "url": url,
                    "kind": kind.value,
                    "search": search,
                    "fetch": fetched,
                }
            )
    reported = await asyncio.to_thread(
        subprocess.run,
        [str(Path(sys.executable).parent / "mlx-mcp"), "info", "--cache", str(cache)],
        check=True,
        capture_output=True,
        text=True,
    )
    info = TypeAdapter(DocumentationInfo).validate_json(reported.stdout, strict=True)
    if info["index"] != "bundled" or (cache / "index.npz").exists():
        raise RuntimeError("The installed package did not select its bundled index")
    return {
        "schema_version": 1,
        "offline": offline,
        "network_control_passed": True,
        "info": info,
        "results": results,
        "installed": {d.metadata["Name"]: d.version for d in distributions()},
    }


def check_wheel(wheel: Path, output: Path) -> None:
    """Install the wheel in a fresh venv, check it online and offline, then rebuild.

    Both server checks must return identical evidence. The combined result
    goes to `output`, which must not exist yet.
    """
    if output.exists():
        raise FileExistsError(f"Result file already exists: {output}")
    with TemporaryDirectory(prefix="mlx-mcp-release-") as directory:
        work = Path(directory).resolve()
        env = {
            k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "UV_"))
        }
        python = work / "venv/bin/python"
        subprocess.run(
            ["uv", "--no-config", "venv", "--python", "3.13", str(work / "venv")],
            cwd=work,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "--no-config",
                "pip",
                "install",
                "--python",
                str(python),
                str(wheel),
            ],
            cwd=work,
            env=env,
            check=True,
        )
        script = work / "check_release_wheel.py"
        shutil.copyfile(__file__, script)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = str(listener.getsockname()[1])
            isolation = network_isolation()
            command = [str(python), "-I", str(script), "--check-server", "--port", port]
            subprocess.run(command, cwd=work, env=env, check=True, timeout=900)
            subprocess.run(
                isolation + command + ["--offline"],
                cwd=work,
                env=env,
                check=True,
                timeout=180,
            )
        online = TypeAdapter(ServerCheck).validate_json(
            (work / "online.json").read_text(), strict=True
        )
        offline = TypeAdapter(ServerCheck).validate_json(
            (work / "offline.json").read_text(), strict=True
        )
        if online["results"] != offline["results"] or online["info"] != offline["info"]:
            raise RuntimeError("Offline restart changed the bundled evidence")
        subprocess.run(
            [str(python), "-I", str(script), "--check-refresh"],
            cwd=work,
            env={**env, "MLX_MCP_CACHE": str(work / "cache")},
            check=True,
            timeout=1800,
        )
        refreshed = TypeAdapter(DocumentationInfo).validate_json(
            (work / "refresh.json").read_text(), strict=True
        )
        report = {
            "schema_version": 1,
            "installed_refresh": refreshed,
            "wheel": wheel.name,
            "platform": sys.platform,
            "network_isolation": isolation,
            "online": online,
            "offline": offline,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    print(f"Installed wheel accepted over stdio with networking blocked: {output}")


def check_installed_refresh() -> None:
    """Rebuild the index from the installed package and check nothing went missing."""
    from mlx_mcp.indexing import refresh
    from mlx_mcp.packages import EvidenceKind, Package
    from mlx_mcp.service import DocumentationService

    root = Path.cwd() / "refreshed"
    refresh(root)
    index = DocumentationService(root)
    if {p.package for p in index.index.passages} != set(Package):
        raise RuntimeError("Installed refresh lost a package")
    kinds = {p.evidence_kind for p in index.index.passages if p.package is Package.LM}
    if kinds != {EvidenceKind.GUIDE, EvidenceKind.API, EvidenceKind.CLI}:
        raise RuntimeError("Installed refresh lost an LM evidence type")
    Path("refresh.json").write_text(
        json.dumps(index.documentation_info(), indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "wheel", nargs="?", type=Path, help="Candidate wheel to install and test"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("release-wheel-check.json"),
        help="New result file; an existing file is never overwritten",
    )
    parser.add_argument("--check-refresh", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--check-server", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--offline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = ReleaseOptions()
    parser.parse_args(namespace=args)
    if args.check_refresh:
        check_installed_refresh()
    elif args.check_server:
        if args.port is None:
            parser.error("--check-server requires --port")
        result = asyncio.run(check_installed_server(args.offline, args.port))
        Path("offline.json" if args.offline else "online.json").write_text(
            json.dumps(result)
        )
    elif args.wheel is None or not args.wheel.is_file():
        parser.error("Specify the built candidate wheel")
    else:
        check_wheel(args.wheel.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
