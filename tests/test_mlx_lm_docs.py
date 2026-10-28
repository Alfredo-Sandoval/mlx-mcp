"""MLX LM guides and captured CLI help turned into passages."""

import json
from pathlib import Path

import pytest
from helpers import service_in_memory

from mlx_mcp.mlx_lm_docs import (
    download_guides,
    github_anchor,
    help_passages,
    parse_markdown,
)
from mlx_mcp.mlx_lm_sources import LM_SOURCE
from mlx_mcp.packages import EvidenceKind, Package
from mlx_mcp.passages import Passage, validate_passages

VERSION = LM_SOURCE.version


REAL_CAPTURE = LM_SOURCE.capture_file


CONSOLE_SCRIPTS = {
    "mlx_lm",
    "mlx_lm.awq",
    "mlx_lm.benchmark",
    "mlx_lm.cache_prompt",
    "mlx_lm.chat",
    "mlx_lm.convert",
    "mlx_lm.dwq",
    "mlx_lm.dynamic_quant",
    "mlx_lm.evaluate",
    "mlx_lm.fuse",
    "mlx_lm.generate",
    "mlx_lm.gptq",
    "mlx_lm.lora",
    "mlx_lm.manage",
    "mlx_lm.perplexity",
    "mlx_lm.server",
    "mlx_lm.share",
    "mlx_lm.upload",
}


def capture(
    commands: list[dict[str, object]], **overrides: object
) -> dict[str, object]:
    base: dict[str, object] = {
        "package": "mlx-lm",
        "version": VERSION,
        "wheel_filename": LM_SOURCE.wheel_filename,
        "captured_at": "2026-09-26T12:00:00+00:00",
        "python_version": "3.12.12",
        "platform": "macOS-27.0-arm64-arm-64bit",
        "mlx_version": "0.32.2",
        "commands": commands,
    }
    base.update(overrides)
    return base


def command(
    name: str = "mlx_lm.generate",
    *,
    help_text: str = "usage: mlx_lm.generate [-h]\n\noptions:\n  -h, --help  show help and exit\n",
    exit_code: int = 0,
    argv: list[str] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "argv": argv or [name, "--help"],
        "exit_code": exit_code,
        "help": help_text,
    }


def write_capture(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "cli_help.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def build(
    tmp_path: Path, commands: list[dict[str, object]], **overrides: object
) -> list[Passage]:
    path = write_capture(tmp_path, capture(commands, **overrides))
    return help_passages(path)


def test_passages_have_one_section_per_command_with_exact_help_and_anchors(
    tmp_path: Path,
) -> None:
    help_text = (
        "usage: mlx_lm.generate [-h] [--model MODEL]\n\n"
        "options:\n  -h, --help     show this help message and exit\n"
        "  --model MODEL  path to the model\n"
    )
    commands = [
        command("mlx_lm.generate", help_text=help_text),
        command(
            "mlx_lm.server",
            help_text="usage: mlx_lm.server [-h]\n\noptions:\n  -h, --help  show help\n",
        ),
    ]
    passages = build(tmp_path, commands)
    validate_passages(passages)

    generate_url = next(
        p.url for p in passages if p.title == "mlx_lm.generate command-line options"
    )
    generate_passages = [p for p in passages if p.url == generate_url]
    assert generate_passages, "expected at least one passage for mlx_lm.generate"
    assert generate_passages[0].section_path == ("mlx_lm.generate",)
    assert generate_passages[0].section_urls == (generate_url,)
    assert generate_passages[0].anchors == ("mlx_lm.generate",)
    assert generate_passages[0].documentation_version == VERSION
    assert generate_passages[0].retrieved_at == "2026-09-26T12:00:00+00:00"
    assert generate_passages[0].title == "mlx_lm.generate command-line options"

    segment_text = generate_passages[0].segment_text
    # The fenced block must reproduce the captured help text exactly, including
    # its internal blank line and indentation, not a reformatted version.
    fenced = segment_text.split("```text\n", 1)[1].rsplit("\n```", 1)[0]
    assert fenced == help_text

    server_url = next(
        p.url for p in passages if p.title == "mlx_lm.server command-line options"
    )
    # Two commands sharing one page must not collide in the same segment group.
    assert {p.segment_index for p in generate_passages} != {
        p.segment_index for p in passages if p.url == server_url
    }


def test_fetch_by_command_anchor_returns_that_sections_help_text(
    tmp_path: Path,
) -> None:
    help_text = "usage: mlx_lm.chat [-h]\n\noptions:\n  -h, --help  show help\n"
    passages = build(
        tmp_path,
        [command("mlx_lm.chat", help_text=help_text), command("mlx_lm.fuse")],
    )
    index = service_in_memory(passages)
    result = index.fetch(passages[0].url)
    assert help_text in result
    assert "mlx_lm.fuse" not in result


def test_rejects_wrong_version(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="version"):
        build(tmp_path, [command()], version="0.31.2")


def test_rejects_nonzero_exit_code(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exited"):
        build(tmp_path, [command(exit_code=1)])


def test_real_capture_covers_all_eighteen_console_scripts() -> None:
    passages = help_passages(REAL_CAPTURE)
    validate_passages(passages)
    assert {p.section_path[0] for p in passages} == CONSOLE_SCRIPTS
    lines = REAL_CAPTURE.read_text().splitlines()
    for passage in passages:
        assert (
            passage.package is Package.LM and passage.evidence_kind is EvidenceKind.CLI
        )
        assert passage.url.startswith(LM_SOURCE.capture_url + "#L")
        assert passage.source_revision == ""  # Release tags are not commit IDs.
        first, last = (int(n) for n in passage.url.split("#L")[1].split("-L"))
        cited = "\n".join(lines[first - 1 : last])
        assert f'"name": "{passage.section_path[0]}"' in cited
        assert '"help":' in cited
        assert LM_SOURCE.package_url in passage.segment_text


RETRIEVED_AT = "2026-09-26T00:00:00+00:00"


URL = "https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/GUIDE.md"


def _parse(text: str) -> list[Passage]:
    return parse_markdown(text, URL, retrieved_at=RETRIEVED_AT)


def test_github_anchor_slugifies_and_hyphenates() -> None:
    seen: dict[str, int] = {}
    assert github_anchor("Fine-Tuning with LoRA or QLoRA", seen) == (
        "fine-tuning-with-lora-or-qlora"
    )
    assert github_anchor("MLX LM", {}) == "mlx-lm"
    # Punctuation is dropped; hyphens and spaces (turned into hyphens) survive.
    assert github_anchor("Request Fields!", {}) == "request-fields"
    # Each whitespace character becomes its own hyphen (not collapsed), matching
    # GitHub's slugger: removing "&" here leaves two adjacent spaces -> "--".
    assert github_anchor("A, B & C", {}) == "a-b--c"


def test_parse_markdown_nests_sections_and_titles() -> None:
    text = (
        "# Guide Title\n\n"
        "Intro paragraph.\n\n"
        "## First Section\n\n"
        "First body.\n\n"
        "### Nested Subsection\n\n"
        "Nested body.\n\n"
        "## Second Section\n\n"
        "Second body.\n"
    )
    passages = _parse(text)
    titles = {p.section_urls[-1].split("#", 1)[1]: p.title for p in passages}
    assert titles == {
        "guide-title": "Guide Title",
        "first-section": "Guide Title / First Section",
        "nested-subsection": "Guide Title / First Section / Nested Subsection",
        "second-section": "Guide Title / Second Section",
    }
    # parse_page's ancestry always includes a leading root-article entry, so a
    # heading N levels deep (H1=1) has an len(section_path) of N + 1.
    nested = next(p for p in passages if "Nested body" in p.text)
    assert len(nested.section_path) == 4
    assert nested.section_urls[-1] == f"{URL}#nested-subsection"


def test_code_fence_and_table_survive_intact() -> None:
    text = (
        "# Data\n\n"
        "```python\n"
        "x = 1\n"
        "y = 2\n"
        "```\n\n"
        "Col A | Col B\n"
        "----- | -----\n"
        "1     | 2\n"
        "3     | 4\n"
    )
    passages = _parse(text)
    combined = "\n".join(p.text for p in passages)
    assert "x = 1\ny = 2" in combined
    assert "```" in combined
    assert "| Col A | Col B |" in combined
    assert "| 1 | 2 |" in combined
    assert "| 3 | 4 |" in combined


def test_relative_links_resolve_to_absolute_github_urls_keeping_fragment() -> None:
    text = (
        "# T\n\n"
        "See [sibling](../README.md) and [same-page](#t) and "
        "[with fragment](OTHER.md#section).\n"
    )
    passages = _parse(text)
    text_out = passages[0].text
    assert "(https://github.com/ml-explore/mlx-lm/blob/v0.31.3/README.md)" in text_out
    assert f"({URL}#t)" in text_out
    assert (
        "(https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/OTHER.md#section)"
        in text_out
    )


def test_duplicate_headings_get_suffixed_in_document_order() -> None:
    text = "# T\n\n## Evaluate\n\nFirst.\n\n## Evaluate\n\nSecond.\n\n## Evaluate\n\nThird.\n"
    passages = _parse(text)
    anchors = [p.section_urls[-1].split("#", 1)[1] for p in passages if p.section_path]
    assert anchors == ["t", "evaluate", "evaluate-1", "evaluate-2"]


def test_download_guides_builds_pinned_raw_urls_and_blob_urls() -> None:
    requested: list[str] = []

    def fake_fetch(url: str) -> str:
        requested.append(url)
        return "# Stub\n\nStub body.\n"

    passages = download_guides(fake_fetch)

    assert requested == [
        f"https://raw.githubusercontent.com/ml-explore/mlx-lm/{LM_SOURCE.commit}/{path}"
        for path in LM_SOURCE.guides
    ]
    expected_blob_urls = {
        f"https://github.com/ml-explore/mlx-lm/blob/{LM_SOURCE.commit}/{path}"
        for path in LM_SOURCE.guides
    }
    assert {p.section_urls[0].split("#")[0] for p in passages} == expected_blob_urls
    assert all(p.documentation_version == LM_SOURCE.version for p in passages)
    validate_passages(passages)
