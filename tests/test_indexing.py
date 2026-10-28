"""Index builds and refreshes, index selection, and keeping the previous index on failure."""

from pathlib import Path

import numpy as np
import pytest

from mlx_mcp import (
    downloads,
    indexing,
    mlx_lm_docs,
)
from mlx_mcp.configuration import refresh_command
from mlx_mcp.index_file import load_index, write_index
from mlx_mcp.indexing import build_index
from mlx_mcp.packages import Package
from mlx_mcp.passages import Passage


def test_fresh_download_collects_every_documentation_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def package_source(package: Package) -> list[Passage]:
        return [
            Passage(
                package.label, package.base_url, "Official source.", package=package
            )
        ]

    monkeypatch.setattr(downloads, "download_package", package_source)
    monkeypatch.setattr(mlx_lm_docs, "download_lm", list)
    passages = indexing.collect_passages()
    assert [p.package for p in passages] == [Package.MLX, Package.DATA]


def test_failed_refresh_or_empty_build_keeps_the_previous_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "index.npz"
    target.write_bytes(b"previous complete index")

    def download(package: Package) -> list[Passage]:
        if package is Package.DATA:
            raise TimeoutError("MLX Data unavailable")
        return [Passage("Core", Package.MLX.base_url + "index.html", "Core")]

    monkeypatch.setattr(downloads, "download_package", download)
    with pytest.raises(TimeoutError, match="MLX Data unavailable"):
        indexing.refresh(tmp_path)
    with pytest.raises(ValueError, match="No MLX documentation"):
        build_index(tmp_path, [])
    assert target.read_bytes() == b"previous complete index"


def test_corrupt_local_snapshot_fails_closed_even_with_a_valid_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundled"
    write_index(bundle, [Passage("Guide", "https://docs/guide", "Source.")])
    monkeypatch.setattr(indexing, "BUNDLED_ROOT", bundle)
    (tmp_path / "index.npz").write_bytes(b"corrupt local snapshot")
    with pytest.raises(ValueError, match="Rebuild it"):
        indexing.select_index_directory(tmp_path)


def test_local_index_from_this_release_takes_precedence(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundled = tmp_path / "bundled"
    write_index(bundled, load_index(root).passages)
    monkeypatch.setattr(indexing, "BUNDLED_ROOT", bundled)
    monkeypatch.setenv("MLX_MCP_CACHE", str(root))
    assert indexing.select_index_directory() == root


def test_stale_local_index_is_reported_and_bundled_index_used(
    root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bundled = tmp_path / "bundled"
    write_index(bundled, load_index(root).passages)
    with np.load(root / "index.npz", allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files}
    fields["built_by"] = np.array("0.0.1")
    np.savez(root / "index.npz", **fields)
    monkeypatch.setattr(indexing, "BUNDLED_ROOT", bundled)
    monkeypatch.setenv("MLX_MCP_CACHE", str(root))
    assert indexing.select_index_directory() == bundled
    assert indexing.select_index_directory() == bundled
    assert caplog.text.count("mlx-mcp 0.0.1 built it") == 1
    assert (
        f"Run `mlx-mcp prepare --refresh --cache {root}` to rebuild it" in caplog.text
    )


def test_rebuild_command_names_the_cache_unless_it_is_the_bundled_index(
    tmp_path: Path,
) -> None:
    command = "mlx-mcp prepare --refresh"
    assert refresh_command(tmp_path) == f"{command} --cache {tmp_path}"
    spaced = tmp_path / "my cache"
    assert refresh_command(spaced) == f"{command} --cache '{spaced}'"
    assert refresh_command(indexing.BUNDLED_ROOT) == command
