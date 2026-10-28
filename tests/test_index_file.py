"""Writing and loading index.npz: format checks and passage fields."""

import json
import os
import re
from dataclasses import asdict, replace
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO, NoReturn

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import ArrayLike

from mlx_mcp import (
    __version__,
    index_file,
)
from mlx_mcp.index_file import (
    INDEX_VERSION,
    load_index,
    read_index_identity,
    write_index,
)
from mlx_mcp.packages import Package
from mlx_mcp.passages import Passage, source_text
from mlx_mcp.service import DocumentationService
from mlx_mcp.sphinx_pages import parse_page


def rebuild_advice(root: Path) -> str:
    """Match the advice an unusable index in root gives: rebuild that directory."""
    return re.escape(
        f"Rebuild it with `mlx-mcp prepare --refresh --cache {root.resolve()}`"
    )


@given(
    st.sampled_from([p for p in Package if p.sphinx]),
    st.text(alphabet="abc XYZ012 λ🌱\n", min_size=1, max_size=200),
)
def test_parser_and_snapshot_roundtrip_package_and_source(
    package: Package, text: str
) -> None:
    # The sentinel keeps whitespace-only generated input meaningful to the parser.
    html = (
        f"<title>Guide — {package.label} 0.1.2 documentation</title>"
        '<article class="bd-article"><section id="guide"><h1>Guide</h1>'
        f"<pre>sentinel:{escape(text)}</pre></section></article>"
    )
    passages = parse_page(html, package.base_url + "guide.html", package=package)
    assert passages and all(p.package is package for p in passages)
    version = "0.1.2" if package.titled_version else ""
    assert all(p.documentation_version == version for p in passages)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        write_index(root, passages)
        restored = load_index(root).passages
        assert restored == passages
        assert source_text(restored) == source_text(passages)


@pytest.fixture
def passages() -> list[Passage]:
    return [
        Passage(
            "Guide", "https://docs/python/guide#eval", "mx.eval(x)", "Complete section"
        )
    ]


def archive_fields(passages: list[Passage], version: int = INDEX_VERSION):
    return {
        "built_by": __version__,
        "version": version,
        "passages": json.dumps([asdict(p) for p in passages]),
    }


def test_roundtrip_keeps_passages_anchors_and_release(
    tmp_path: Path, passages: list[Passage]
) -> None:
    anchored = replace(passages[0], anchors=("explicit-label", "mlx.Example.__init__"))
    write_index(tmp_path, [anchored])
    loaded = load_index(tmp_path)
    assert loaded.passages == [anchored]
    assert loaded.built_by == __version__


def test_invalid_provenance_does_not_publish(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_index(tmp_path, [replace(Passage("G", "url", "text"), retrieved_at="x")])
    assert not (tmp_path / "index.npz").exists()


def test_older_index_formats_are_rejected_with_rebuild_instructions(root: Path) -> None:
    with np.load(root / "index.npz", allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files}
    fields["version"] = np.array(INDEX_VERSION - 1)
    np.savez(root / "index.npz", **fields)
    with pytest.raises(
        ValueError, match=f"Index format changed.*{rebuild_advice(root)}"
    ):
        DocumentationService(root)


@pytest.mark.parametrize(
    "bad",
    [
        {"version": 1.5},
        {"passages": "not JSON"},
        {"passages": "[null]"},
        {"passages": '[{"title": 1, "url": "url", "text": "text"}]'},
    ],
)
def test_malformed_archives_give_rebuild_error(
    tmp_path: Path, passages: list[Passage], bad: dict[str, ArrayLike]
) -> None:
    np.savez(
        tmp_path / "index.npz", allow_pickle=False, **(archive_fields(passages) | bad)
    )
    with pytest.raises(ValueError, match=rebuild_advice(tmp_path)):
        load_index(tmp_path)


def test_missing_index_stays_distinct_from_corruption(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path)
    (tmp_path / "index.npz").write_bytes(b"not an index")
    with pytest.raises(ValueError, match="Invalid MLX index"):
        load_index(tmp_path)
    with pytest.raises(ValueError, match=rebuild_advice(tmp_path)):
        read_index_identity(tmp_path)


def test_failed_publication_preserves_index_and_cleans_temporary(
    tmp_path: Path, passages: list[Passage], monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "index.npz"
    target.write_bytes(b"previous snapshot")

    def fail_write(file: BinaryIO, *args: object, **kwargs: object) -> NoReturn:
        file.write(b"partial archive")
        raise OSError("injected publication failure")

    monkeypatch.setattr(index_file.np, "savez_compressed", fail_write)
    with pytest.raises(OSError, match="injected"):
        write_index(tmp_path, passages)
    assert target.read_bytes() == b"previous snapshot"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("anchors", ["label", [""], [3]])
def test_malformed_source_anchors_fail_closed(
    tmp_path: Path, passages: list[Passage], anchors: object
) -> None:
    record = asdict(passages[0]) | {"anchors": anchors}
    np.savez(
        tmp_path / "index.npz",
        allow_pickle=False,
        **(archive_fields(passages) | {"passages": json.dumps([record])}),
    )
    with pytest.raises(ValueError, match=rebuild_advice(tmp_path)):
        load_index(tmp_path)


def test_loaded_file_metadata_detects_replaced_text(tmp_path: Path) -> None:
    original = Passage(
        "Guide",
        Package.MLX.base_url + "usage/honest.html",
        "ORIGINAL",
        retrieved_at="2026-10-21T00:00:00+00:00",
    )
    path = tmp_path / "index.npz"
    write_index(tmp_path, [original])
    stamp = 1_790_000_000_000_000_000
    os.utime(path, ns=(stamp, stamp))
    old = DocumentationService(tmp_path)
    write_index(tmp_path, [replace(original, text="MODIFIED")])
    os.utime(path, ns=(stamp + 1_000_000, stamp + 1_000_000))
    current = DocumentationService(tmp_path)
    old_info, current_info = old.documentation_info(), current.documentation_info()
    assert old_info["index_file"]["modified_at_ns"] == stamp
    assert old_info["index_file"] != current_info["index_file"]
