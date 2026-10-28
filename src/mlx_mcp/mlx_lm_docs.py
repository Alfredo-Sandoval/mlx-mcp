"""Reconstruct MLX LM evidence from shipped captures and immutable upstream inputs.

LM passages come from three sources: API references extracted from the pinned
wheel, the pinned markdown guides, and captured CLI help.

Guides are rendered to flat HTML with markdown-it-py, the flat h1..h6 siblings
are regrouped into a Sphinx-shaped `<section id=anchor>` tree, and that HTML is
handed to `sphinx_pages.parse_page`, so chunking and section ancestry follow exactly
the same rules as the Sphinx docs pipeline. CLI help passages cite exact lines
of their immutable JSON capture.
"""

import json
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from bs4 import BeautifulSoup, Tag
from markdown_it import MarkdownIt

from mlx_mcp.downloads import download_bytes
from mlx_mcp.mlx_lm_api import MODULES, api_passages
from mlx_mcp.mlx_lm_sources import LM_SOURCE
from mlx_mcp.packages import EvidenceKind, Package
from mlx_mcp.passages import Passage, chunk_spans
from mlx_mcp.sphinx_pages import parse_page


def download_lm() -> list[Passage]:
    """Collect MLX LM guides, wheel API references, and captured CLI help.

    Each module extracted from the wheel must match the cited commit byte for byte.
    """
    passages = help_passages(LM_SOURCE.capture_file)
    passages.extend(download_guides(lambda url: download_bytes(url).decode("utf-8")))
    data = download_bytes(LM_SOURCE.wheel_url)
    with TemporaryDirectory(prefix="mlx-lm-source-") as temporary:
        wheel = Path(temporary) / LM_SOURCE.wheel_filename
        wheel.write_bytes(data)
        # Line citations are valid only when the cited commit has the exact code
        # extracted from the wheel, including the re-export namespace mapping.
        with zipfile.ZipFile(wheel) as archive:
            for path in ("__init__.py", *(module.path for module in MODULES)):
                relative = "mlx_lm/" + path
                if archive.read(relative) != download_bytes(
                    LM_SOURCE.raw_base + relative
                ):
                    raise ValueError(
                        f"Wheel source differs from cited commit: {relative}"
                    )
        passages.extend(
            api_passages(
                wheel,
                retrieved_at=datetime.now(UTC).isoformat(),
            )
        )
    return passages


_HEADING_RE = re.compile(r"^h([1-6])$")
# Matches GitHub's rendered-anchor punctuation removal: keep letters, digits,
# underscore (all \w), spaces, and hyphens; drop everything else.
_NON_SLUG_RE = re.compile(r"[^\w\s-]", re.UNICODE)


def github_anchor(heading_text: str, seen: dict[str, int]) -> str:
    """Reproduce the id GitHub renders for a heading's already-visible text.

    `heading_text` must be the rendered/visible heading text (no markdown
    emphasis, code backticks, or link syntax) -- e.g. a heading tag's
    `get_text()` after markdown rendering, not the raw markdown source line.
    `seen` accumulates counts per page; pass the same dict for every heading
    on one page so repeats get GitHub's -1, -2, ... suffixes in document order.
    """
    slug = _NON_SLUG_RE.sub("", heading_text.strip().lower()).replace(" ", "-")
    base = slug
    while slug in seen:
        seen[base] += 1
        slug = f"{base}-{seen[base]}"
    seen[slug] = 0
    return slug


def _drop_images(soup: BeautifulSoup) -> None:
    """Images carry no searchable text here; keep only their alt text."""
    for img in soup.find_all("img"):
        alt = img.get("alt", "")
        # BeautifulSoup's HTML parser treats alt as a scalar attribute.
        assert isinstance(alt, str)
        img.replace_with(alt)


def _wrap_sections(article: Tag, soup: BeautifulSoup) -> None:
    """Nest markdown-it's flat h1..h6 siblings into `<section id=anchor>` trees.

    This is the layout sphinx_pages.parse_page's `collect()` expects: each section owns
    one heading plus the content up to the next heading of equal or shallower depth.
    """
    stack: list[tuple[int, Tag]] = []
    seen: dict[str, int] = {}
    for child in list(article.children):
        match = _HEADING_RE.match(child.name) if isinstance(child, Tag) else None
        if match is None:
            child.extract()
            (stack[-1][1] if stack else article).append(child)
            continue
        level = int(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        anchor = github_anchor(child.get_text().strip(), seen)
        section = soup.new_tag("section", id=anchor)
        child.extract()
        section.append(child)
        (stack[-1][1] if stack else article).append(section)
        stack.append((level, section))


def parse_markdown(text: str, url: str, *, retrieved_at: str) -> list[Passage]:
    """Parse one markdown guide's `text` into passages addressable at `url`.

    `url` is the guide's canonical GitHub blob URL (e.g.
    ".../blob/<commit>/mlx_lm/LORA.md"); relative markdown links resolve
    against it, so they land on sibling guides' blob URLs at the same commit.
    """
    # GFM tables aren't in markdown-it-py's default "commonmark" rule set;
    # enable just that rule (no linkify-it-py dependency needed for it).
    html = MarkdownIt("commonmark").enable("table").render(text)
    soup = BeautifulSoup(html, "html.parser")
    _drop_images(soup)
    article = soup.new_tag("article")
    article["class"] = "bd-article"
    for node in list(soup.contents):
        node.extract()
        article.append(node)
    soup.append(article)
    _wrap_sections(article, soup)
    passages = parse_page(str(soup), url, retrieved_at=retrieved_at, package=Package.LM)
    # parse_page derives documentation_version from a Sphinx <title> pattern
    # that these guides don't have; stamp the pinned release explicitly.
    return [
        replace(
            passage,
            documentation_version=LM_SOURCE.version,
            evidence_kind=EvidenceKind.GUIDE,
            source_revision=LM_SOURCE.commit,
        )
        for passage in passages
    ]


def download_guides(fetch: Callable[[str], str]) -> list[Passage]:
    """Download the pinned mlx-lm guides and parse each into passages.

    fetch receives each pinned raw-content URL and returns its Markdown. Guide
    paths and revisions come from LM_SOURCE; the caller owns download policy.
    """
    retrieved_at = datetime.now(UTC).isoformat()
    passages: list[Passage] = []
    for path in LM_SOURCE.guides:
        text = fetch(f"{LM_SOURCE.raw_base}{path}")
        url = f"{LM_SOURCE.blob_base}{path}"
        passages.extend(parse_markdown(text, url, retrieved_at=retrieved_at))
    return passages


@dataclass(frozen=True)
class CapturedCommand:
    """One captured console command: its argv, --help text, and citation URL."""

    name: str
    argv: tuple[str, ...]
    help: str
    url: str


@dataclass(frozen=True)
class HelpCapture:
    """A verified CLI capture: when it was taken and the commands it holds."""

    captured_at: str
    commands: tuple[CapturedCommand, ...]


def _command(entry: object, text: str) -> CapturedCommand:
    if not isinstance(entry, dict):
        raise ValueError("Each captured command must be a JSON object")  # noqa: TRY004
    fields = cast(dict[str, object], entry)
    name, argv, help_text = (fields.get(key) for key in ("name", "argv", "help"))
    if not isinstance(name, str) or not name:
        raise ValueError("Command name must be a nonempty string")
    items = cast(list[object], argv) if isinstance(argv, list) else []
    arguments = tuple(item for item in items if isinstance(item, str))
    if not arguments or len(arguments) != len(items):
        raise ValueError("Captured argv must be a nonempty list of strings")
    exit_code = fields.get("exit_code")
    if type(exit_code) is not int or exit_code != 0:
        raise ValueError(
            f"Captured command {name!r} exited with a nonzero or invalid code"
        )
    if not isinstance(help_text, str) or not help_text.strip():
        raise ValueError(f"Captured command {name!r} has empty --help text")
    # The capture is the evidence. Point to its actual JSON lines, without
    # manufacturing command fragments on the upstream package's PyPI page.
    marker = f'"name": {json.dumps(name, ensure_ascii=False)}'
    if text.count(marker) != 1:
        raise ValueError("Capture must have one identifiable JSON section per command")
    start = text.index(marker)
    help_marker = f'"help": {json.dumps(help_text, ensure_ascii=False)}'
    end = text.index(help_marker, start) + len(help_marker)
    first, last = text.count("\n", 0, start) + 1, text.count("\n", 0, end) + 1
    return CapturedCommand(
        name, arguments, help_text, f"{LM_SOURCE.capture_url}#L{first}-L{last}"
    )


def read_capture(path: Path) -> HelpCapture:
    """Load the CLI capture at path and verify it against the manifest.

    Each command's URL cites its exact JSON line range in the capture file.
    """
    text = path.read_text(encoding="utf-8")
    decoded: object = json.loads(text)
    if not isinstance(decoded, dict):
        raise ValueError("CLI help capture must be a JSON object")  # noqa: TRY004
    raw = cast(dict[str, object], decoded)
    if (
        raw.get("package") != Package.LM.value
        or raw.get("version") != LM_SOURCE.version
    ):
        raise ValueError("CLI help capture package/version does not match the manifest")
    stamp = raw.get("captured_at")
    if not isinstance(stamp, str) or datetime.fromisoformat(stamp).utcoffset() is None:
        raise ValueError("CLI capture timestamp must include a timezone")
    entries = raw.get("commands")
    if not isinstance(entries, list) or not entries:
        raise ValueError("CLI help capture has no commands")
    commands = tuple(_command(entry, text) for entry in cast(list[object], entries))
    if len({c.name for c in commands}) != len(commands):
        raise ValueError("Captured command names must be unique")
    if len({c.url for c in commands}) != len(commands):
        raise ValueError("Captured commands must occupy distinct JSON line ranges")
    return HelpCapture(stamp, commands)


def _section_text(command: CapturedCommand, capture: HelpCapture) -> str:
    longest = max((len(run) for run in re.findall(r"`+", command.help)), default=0)
    fence = "`" * max(3, longest + 1)
    return (
        f"## {command.name} command-line options\n\n"
        f"Captured CLI output: MLX LM {LM_SOURCE.version}; {capture.captured_at}.\n\n"
        f"Upstream package: {LM_SOURCE.package_url}\n\n"
        f"Usage: {' '.join(command.argv)}\n\n{fence}text\n{command.help}\n{fence}"
    )


def help_passages(capture: Path) -> list[Passage]:
    """Verify the shipped capture and preserve command identity and provenance."""
    captured = read_capture(capture)
    passages: list[Passage] = []
    for segment_index, command in enumerate(captured.commands):
        segment_text = _section_text(command, captured)
        for position, (start, end, chunk) in enumerate(chunk_spans(segment_text)):
            passages.append(
                Passage(
                    f"{command.name} command-line options",
                    command.url,
                    chunk,
                    segment_text,
                    (command.name,) if position == 0 else (),
                    documentation_version=LM_SOURCE.version,
                    retrieved_at=captured.captured_at,
                    section_path=(command.name,),
                    section_urls=(command.url,),
                    segment_index=segment_index,
                    chunk_start=start,
                    chunk_end=end,
                    package=Package.LM,
                    evidence_kind=EvidenceKind.CLI,
                )
            )
    return passages
