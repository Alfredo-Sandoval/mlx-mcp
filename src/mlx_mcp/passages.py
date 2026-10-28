"""Canonical source segments, exact passage ranges, and Markdown projections.

Offsets are Python character offsets into segment_text, never token offsets.
These operations need neither an index file nor a model or network client.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from itertools import pairwise
from urllib.parse import urldefrag

from markdown_it import MarkdownIt

from mlx_mcp.packages import EvidenceKind, Package

type SegmentKey = tuple[str, int | str]

EVIDENCE_CHAR_BUDGET = 6000
_IDENTIFIER = r"[A-Za-z_]\w*"
_HEADING = re.compile(r"^#{1,6}\s+")


@dataclass(frozen=True)
class Passage:
    """One indexed chunk of documentation text and where it came from.

    segment_text is the canonical segment the chunk was cut from: content its
    section owns directly, between nested sections. chunk_start and chunk_end
    locate text within it. Passages built without HTML structure leave the
    structural fields at their defaults. package must match the URL's site.
    """

    title: str
    url: str
    text: str
    segment_text: str = ""
    anchors: tuple[str, ...] = ()
    documentation_version: str = ""
    retrieved_at: str = ""
    # Paths identify HTML containers; segment_text is one directly owned segment.
    section_path: tuple[str, ...] = ()
    section_urls: tuple[str, ...] = ()
    segment_index: int = -1
    chunk_start: int = -1
    chunk_end: int = -1
    package: Package = Package.MLX
    evidence_kind: EvidenceKind = EvidenceKind.DOCUMENTATION
    source_revision: str = ""

    def __post_init__(self) -> None:
        if self.source_revision and not re.fullmatch(
            r"[a-f0-9]{40}", self.source_revision
        ):
            raise ValueError("Source revision must be an immutable Git commit")
        for package in Package:
            if self.url.startswith(package.base_url) and package is not self.package:
                raise ValueError(
                    "Passage package must match its official documentation URL"
                )


def section_key(passage: Passage) -> tuple[str, str]:
    """Identify the section that owns a passage: its page and container ID."""
    return (
        (urldefrag(passage.url)[0], passage.section_path[-1])
        if passage.section_path
        else (
            passage.url,
            "",
        )
    )


def segment_key(passage: Passage) -> SegmentKey:
    """Identify the canonical segment a passage was cut from."""
    if passage.section_path:
        return urldefrag(passage.url)[0], passage.segment_index
    # Old archives retain direct strings but cannot recover HTML ancestry.
    return passage.url, passage.segment_text or passage.text


def resolve_chunk_start(passage: Passage) -> int:
    """Return where passage.text starts in segment_text, or -1 if not found."""
    return (
        passage.chunk_start
        if passage.section_path
        else passage.segment_text.find(passage.text)
    )


def group_passages_by_segment(passages: list[Passage]) -> list[list[Passage]]:
    """Group passages by canonical segment, in page order."""
    grouped: dict[SegmentKey, list[Passage]] = {}
    for passage in passages:
        grouped.setdefault(segment_key(passage), []).append(passage)
    return sorted(grouped.values(), key=lambda group: group[0].segment_index)


def source_text(passages: list[Passage]) -> str:
    """Join the full text of the passages' segments, in page order."""
    return "\n\n".join(
        group[0].segment_text or group[0].text
        for group in group_passages_by_segment(passages)
    )


def validate_structure(passages: list[Passage]) -> None:
    """Raise ValueError unless ancestry and ranges agree and cover each segment."""
    nodes: dict[tuple[str, tuple[str, ...]], str] = {}
    node_paths: dict[tuple[str, str], tuple[str, ...]] = {}
    page_modes: dict[str, set[bool]] = defaultdict(set)
    grouped: dict[SegmentKey, list[Passage]] = defaultdict(list)
    for p in passages:
        page = urldefrag(p.url)[0]
        page_modes[page].add(bool(p.section_path))
        if (
            len(p.section_path) != len(p.section_urls)
            or not all(p.section_path + p.section_urls)
            or len(set(p.section_path)) != len(p.section_path)
        ):
            raise ValueError("Invalid section ancestry")
        if not p.section_path:
            if (p.segment_index, p.chunk_start, p.chunk_end) != (-1, -1, -1):
                raise ValueError("Source ranges require section ancestry")
            continue
        if p.section_urls[-1] != p.url or any(
            urldefrag(url)[0] != page for url in p.section_urls
        ):
            raise ValueError("Section ancestry must remain on its source page")
        if (
            any(
                type(v) is not int
                for v in (p.segment_index, p.chunk_start, p.chunk_end)
            )
            or p.segment_index < 0
            or not 0 <= p.chunk_start < p.chunk_end <= len(p.segment_text)
            or p.segment_text[p.chunk_start : p.chunk_end] != p.text
        ):
            raise ValueError("Chunk range must match its canonical source segment")
        for depth, url in enumerate(p.section_urls, 1):
            key = page, p.section_path[:depth]
            if nodes.setdefault(key, url) != url:
                raise ValueError("Conflicting section ancestry")
            identity = page, p.section_path[depth - 1]
            if (
                node_paths.setdefault(identity, p.section_path[:depth])
                != p.section_path[:depth]
            ):
                raise ValueError("Section IDs must be unique within each page")
        grouped[segment_key(p)].append(p)
    if any(len(modes) != 1 for modes in page_modes.values()):
        raise ValueError("A page cannot mix known and unknown source structure")
    for group in grouped.values():
        first = group[0]
        fields = (
            "section_path",
            "section_urls",
            "segment_text",
            "title",
            "url",
            "documentation_version",
            "retrieved_at",
            "package",
            "evidence_kind",
            "source_revision",
        )
        if any(
            any(getattr(p, field) != getattr(first, field) for field in fields)
            for p in group
        ):
            raise ValueError("Conflicting canonical source segment")
        cursor = 0
        for p in sorted(group, key=lambda p: p.chunk_start):
            if (
                p.chunk_start > cursor
                and first.segment_text[cursor : p.chunk_start].strip()
            ):
                raise ValueError("Source segment has an uncovered range")
            cursor = max(cursor, p.chunk_end)
        if first.segment_text[cursor:].strip():
            raise ValueError("Source segment has an uncovered tail")


def validate_passages(passages: list[Passage]) -> None:
    """Raise ValueError unless passages are nonempty and consistent.

    Field types are checked where stored records are decoded; see _decode_passage.
    """
    if not passages:
        raise ValueError("No MLX documentation passages were collected")
    for passage in passages:
        if not all(
            value.strip() for value in (passage.title, passage.url, passage.text)
        ):
            raise ValueError(
                "Passages must have nonempty string titles, URLs, and text"
            )
        if not all(anchor.strip() for anchor in passage.anchors):
            raise ValueError("Passage anchors must be nonempty strings")
        if (
            passage.retrieved_at
            and datetime.fromisoformat(passage.retrieved_at).utcoffset() is None
        ):
            raise ValueError("Source retrieval timestamp must include a timezone")
    validate_structure(passages)


def split_text(text: str, size: int = 1800) -> list[str]:
    """Pack whole top-level Markdown blocks with size as a soft character limit.

    A block longer than size becomes its own chunk rather than being cut, so code
    and tables stay intact.
    """
    if size < 1:
        raise ValueError("Chunk size must be positive")
    lines = text.splitlines(keepends=True)
    boundaries = {0, len(lines)}
    for token in MarkdownIt().parse(text):
        if token.level == 0 and token.map:
            boundaries.update(token.map)
    points = sorted(boundaries)
    chunks: list[str] = []
    current = ""
    for start, end in pairwise(points):
        block = "".join(lines[start:end])
        if current.strip() and len(current) + len(block) > size:
            chunks.append(current.strip())
            current = ""
        current += block
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_spans(segment_text: str, size: int = 1800) -> list[tuple[int, int, str]]:
    """Split segment_text with split_text; return each chunk with its range.

    Ranges are (start, end) character offsets into segment_text, so passages
    cut from the segment can record exact chunk_start and chunk_end values.
    """
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for chunk in split_text(segment_text, size):
        start = segment_text.index(chunk, cursor)
        cursor = start + len(chunk)
        spans.append((start, cursor, chunk))
    return spans


def has_explanatory_content(markdown: str) -> bool:
    """Return whether markdown has code or prose beyond headings and link text."""
    heading = False
    for token in MarkdownIt("commonmark").parse(markdown):
        if token.type == "heading_open":
            heading = True
        elif token.type == "heading_close":
            heading = False
        elif token.type in {"fence", "code_block"} and token.content.strip():
            return True
        elif token.type == "inline" and not heading:
            # MarkdownIt's inline pass always initializes children.
            assert token.children is not None
            linked = False
            for child in token.children:
                if child.type == "link_open":
                    linked = True
                elif child.type == "link_close":
                    linked = False
                elif not linked and re.search(r"\w", child.content):
                    return True
    return False


@lru_cache(maxsize=2048)
def ranking_text(markdown: str) -> str:
    """Return markdown's readable text and code without link destinations."""
    blocks: list[str] = []
    for token in MarkdownIt("commonmark").parse(markdown):
        if token.type == "inline":
            assert token.children is not None
            blocks.append(
                "".join(
                    "\n" if child.type in {"softbreak", "hardbreak"} else child.content
                    for child in token.children
                )
            )
        elif token.type == "code_block" and "Parameters:\n:   - **" in token.content:
            # Sphinx field lists are indented prose, not literal source code.
            blocks.append(ranking_text(token.content))
        elif token.type in {"fence", "code_block"}:
            blocks.append(token.content)
    return "\n".join(blocks)


def selected_evidence(passage: Passage, text: str | None = None) -> str:
    """Return text, prefixed by its section's class declaration if it lacks one.

    A later chunk of a class section would otherwise lose the class signature.
    """
    text = passage.text if text is None else text
    source = passage.segment_text
    start = resolve_chunk_start(passage) if source else -1
    name = passage.title.rsplit(" / ", 1)[-1].rsplit(".", 1)[-1]
    if start > 0 and re.fullmatch(_IDENTIFIER, name):
        declaration = re.search(
            rf"^`class {re.escape(name)}[(`][^\n]*(?:\n:   [^\n]+)?",
            source[:start],
            re.MULTILINE,
        )
        if declaration and declaration.group() not in text:
            return declaration.group() + "\n\n" + text
    return text


@lru_cache(maxsize=1024)
def _section_blocks(source: str) -> tuple[tuple[int, int, str], ...]:
    """Split source into top-level Markdown blocks with their (start, end) offsets."""
    # A size of one makes every top-level block its own chunk.
    return tuple(chunk_spans(source, size=1))


def explanatory_passage(passage: Passage) -> bool:
    """Return whether the source blocks a passage overlaps have explanatory content."""
    if not passage.section_path:
        return has_explanatory_content(passage.text)
    # Ranking windows can begin inside a Markdown link. Classify their complete
    # original blocks so a sliced catalog does not become explanatory prose.
    return any(
        has_explanatory_content(block)
        for start, end, block in _section_blocks(passage.segment_text)
        if start < passage.chunk_end and end > passage.chunk_start
    )


def partial_source_block(passage: Passage) -> bool:
    """Return whether a passage starts or ends inside a Markdown block."""
    if not passage.section_path:
        return False
    blocks = _section_blocks(passage.segment_text)
    return passage.chunk_start not in {
        start for start, _, _ in blocks
    } or passage.chunk_end not in {end for _, end, _ in blocks}


def _context_spans(
    passage: Passage, neighbor_blocks: int
) -> list[tuple[int, int, str]]:
    """Return the selected blocks plus heading, opening, and neighbor context."""
    source = passage.segment_text
    if not source or not passage.text:
        return []
    selected_start = resolve_chunk_start(passage)
    if selected_start < 0:
        return []
    selected_end = selected_start + len(passage.text)
    blocks = _section_blocks(source)
    selected = [
        index
        for index, (start, end, _) in enumerate(blocks)
        if start < selected_end and end > selected_start
    ]
    if not selected:
        return []
    wanted = set(selected)
    heading_indices = [
        index for index, (_, _, block) in enumerate(blocks) if _HEADING.match(block)
    ]
    if heading_indices and heading_indices[0] < selected[0]:
        wanted.add(heading_indices[0])
    opening = [
        index
        for index, (_, _, block) in enumerate(blocks)
        if index > (heading_indices[0] if heading_indices else -1)
        and index < selected[0]
        and not _HEADING.match(block)
    ]
    wanted.update(opening[:2])
    wanted.update(
        range(max(0, selected[0] - neighbor_blocks), selected[-1] + neighbor_blocks + 1)
    )
    return [
        (start, end, block)
        for index, (start, end, block) in enumerate(blocks)
        if index in wanted
    ]


def expand_passage(
    passage: Passage,
    *,
    char_budget: int = EVIDENCE_CHAR_BUDGET,
    neighbor_blocks: int = 1,
) -> str:
    """Return selected text plus bounded heading, signature, and nearby context.

    The budget includes the selected passage and added context. An original
    passage exceeding it is returned intact without extras. Context is added
    only after ranking; oversized context blocks are omitted rather than cut.
    """
    if type(char_budget) is not int or char_budget < 1:
        raise ValueError("char_budget must be positive")
    if type(neighbor_blocks) is not int or neighbor_blocks < 0:
        raise ValueError("neighbor_blocks must be nonnegative")
    return "\n\n".join(
        text for _, _, text in _expanded_spans(passage, char_budget, neighbor_blocks)
    )


def _expanded_spans(
    passage: Passage, char_budget: int = EVIDENCE_CHAR_BUDGET, neighbor_blocks: int = 1
) -> list[tuple[int, int, str]]:
    selected_start = resolve_chunk_start(passage)
    selected_end = selected_start + len(passage.text)
    chosen = [(selected_start, selected_end, passage.text)]
    context = _context_spans(passage, neighbor_blocks)
    remaining = char_budget - len(passage.text)
    extras = [
        (start, end, block)
        for start, end, block in context
        if not (start < selected_end and end > selected_start)
    ]
    for start, end, block in sorted(
        extras, key=lambda span: abs(span[0] - selected_start)
    ):
        addition = len(block) + 2
        if addition <= remaining:
            chosen.append((start, end, block))
            remaining -= addition
    return sorted(chosen)


def expand_section(passages: list[Passage], *, include_context: bool = True) -> str:
    """Merge the passages' spans per segment and render them in source order.

    Each selected excerpt stays intact; include_context adds its nearby context.
    """
    return "\n\n".join(
        _expand_segment(group, include_context)
        for group in group_passages_by_segment(passages)
    )


def _expand_segment(passages: list[Passage], include_context: bool) -> str:
    source = passages[0].segment_text
    if not source:
        return "\n\n".join(dict.fromkeys(p.text for p in passages))
    spans: list[tuple[int, int]] = []
    for passage in passages:
        start = resolve_chunk_start(passage)
        if passage.segment_text != source or start < 0:
            raise ValueError("Section chunks must share their canonical source")
        spans.append((start, start + len(passage.text)))
        if include_context:
            spans.extend((start, end) for start, end, _ in _expanded_spans(passage))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return "\n\n".join(source[start:end] for start, end in merged)
