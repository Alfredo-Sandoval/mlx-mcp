"""The mlx-mcp command line: argument parsing, dispatch, and failure messages."""

from pathlib import Path

import pytest

from mlx_mcp import indexing
from mlx_mcp.cli import main


def test_info_without_an_index_names_the_rebuild_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(indexing, "BUNDLED_ROOT", tmp_path / "no-bundled-index")
    cache = tmp_path / "cache with spaces"
    with pytest.raises(SystemExit) as failure:
        main(["info", "--cache", str(cache)])
    assert failure.value.code == 1
    output = capsys.readouterr()
    assert "Error:" in output.err and "mlx-mcp prepare" in output.err
    assert f"`mlx-mcp prepare --refresh --cache '{cache.resolve()}'`" in output.err
    assert not cache.exists()
