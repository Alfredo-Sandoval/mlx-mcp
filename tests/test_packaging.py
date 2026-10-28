"""Release archives retain runtime inputs and exclude unrelated local work."""

import shutil
import tarfile
import tomllib
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import NotRequired, TypedDict

import pytest
from hatchling.builders.sdist import SdistBuilder
from hatchling.builders.wheel import WheelBuilder
from pydantic import TypeAdapter

from mlx_mcp.index_file import load_index, write_index
from mlx_mcp.passages import Passage

# uv.lock tables use hyphenated keys, so these use the functional form.
LockDependency = TypedDict(
    "LockDependency", {"name": str, "extra": NotRequired[list[str]]}
)


LockPackage = TypedDict(
    "LockPackage",
    {
        "name": str,
        "dependencies": NotRequired[list[LockDependency]],
        "optional-dependencies": NotRequired[dict[str, list[LockDependency]]],
    },
)


class Lockfile(TypedDict):
    package: list[LockPackage]


@pytest.mark.parametrize("with_index", [False, True])
def test_sdist_excludes_local_work_and_rebuilds_runtime_wheel(
    tmp_path: Path, with_index: bool
) -> None:
    """Exercise source and prepared builds without borrowing a developer's index."""
    repository = Path(__file__).parents[1]
    project = tmp_path / "project"
    project.mkdir()
    for name in ("src", "tests", "scripts"):
        shutil.copytree(
            repository / name,
            project / name,
            ignore=shutil.ignore_patterns("__pycache__", "index.npz"),
        )
    for name in (
        ".gitignore",
        ".python-version",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "SECURITY.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
    ):
        shutil.copy2(repository / name, project / name)
    private_files = (
        "draft.html",
        "research_notes/private.txt",
        "paper/private.txt",
        "reports/private.json",
        ".claude/worktrees/private.txt",
        "docs/private.html",
        "AGENTS.md",
        ".github/private.yml",
    )
    for name in private_files:
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("PRIVATE_LOCAL_WORK")
    passages = [Passage("Packaging fixture", "https://docs/fixture", "Fixture text")]
    if with_index:
        write_index(project / "src/mlx_mcp/data", passages)

    sdist = Path(
        next(SdistBuilder(str(project)).build(directory=str(tmp_path / "dist")))
    )
    with tarfile.open(sdist) as archive:
        names = {
            Path(name).relative_to(sdist.name.removesuffix(".tar.gz")).as_posix()
            for name in archive.getnames()
        }
        assert not set(private_files) & names
        assert ("src/mlx_mcp/data/index.npz" in names) is with_index
        assert {name.split("/")[0] for name in names} == {
            "src",
            ".gitignore",
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "PKG-INFO",
        }
        archive.extractall(tmp_path / "unpacked", filter="data")
    unpacked = tmp_path / "unpacked" / sdist.name.removesuffix(".tar.gz")
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        assert (unpacked / name).read_bytes() == (repository / name).read_bytes()
    wheel = Path(
        next(WheelBuilder(str(unpacked)).build(directory=str(tmp_path / "wheel")))
    )
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert {
            "mlx_mcp/server.py",
            "mlx_mcp/source_data/mlx_lm.json",
            "mlx_mcp/source_data/mlx_lm_cli_help.json",
        } <= names
        assert ("mlx_mcp/data/index.npz" in archive.namelist()) is with_index
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            assert (
                archive.read(f"{unpacked.name}.dist-info/licenses/{name}")
                == (repository / name).read_bytes()
            )
        if with_index:
            archive.extract("mlx_mcp/data/index.npz", tmp_path / "installed")
            index = load_index(tmp_path / "installed/mlx_mcp/data")
            assert index.passages == passages


def test_no_install_path_requires_pytorch() -> None:
    # pyproject.toml names only direct dependencies; PyTorch can still arrive
    # transitively or through an extra such as transformers[torch]. uv.lock
    # holds the resolved graph, so walk it from mlx-mcp and each of its extras,
    # following every dependency's requested extras too.
    lockfile = TypeAdapter(Lockfile).validate_python(
        tomllib.loads((Path(__file__).parents[1] / "uv.lock").read_text()),
        strict=True,
    )
    edges: defaultdict[tuple[str, str | None], list[LockDependency]] = defaultdict(list)
    for package in lockfile["package"]:
        edges[package["name"], None] += package.get("dependencies", [])
        for extra, dependencies in package.get("optional-dependencies", {}).items():
            edges[package["name"], extra] += dependencies
    extras = [extra for name, extra in edges if name == "mlx-mcp" and extra]
    assert extras, "uv.lock lists no mlx-mcp extras; this test no longer covers dev"

    pending: list[tuple[str, str | None]] = [("mlx-mcp", None)]
    pending += [("mlx-mcp", extra) for extra in extras]
    reachable: set[tuple[str, str | None]] = set()
    while pending:
        node = pending.pop()
        if node in reachable:
            continue
        reachable.add(node)
        for dependency in edges.get(node, []):
            pending.append((dependency["name"], None))
            pending += [(dependency["name"], e) for e in dependency.get("extra", [])]
    offenders = {name for name, _ in reachable} & {"torch", "triton"}
    assert not offenders, f"uv.lock pulls {sorted(offenders)} into an mlx-mcp install"
