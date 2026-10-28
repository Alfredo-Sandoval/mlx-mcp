"""Bounded source responses and documentation catalogs for the MCP tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from markdown_it import MarkdownIt

from mlx_mcp.packages import EvidenceKind, Package
from mlx_mcp.passages import (
    expand_passage,
    expand_section,
    partial_source_block,
    section_key,
    selected_evidence,
    source_text,
)
from mlx_mcp.search_text import VersionRequest

if TYPE_CHECKING:
    from mlx_mcp.passages import Passage

RESPONSE_CHAR_BUDGET = 8000
SOURCE_ID = r"https?://\S+"

_LINK = re.compile(
    r"(?<!!)\[((?:[^\[\]\n]|\[[^\[\]\n]*\])*)\]\((https?://[^\s()]+)(?:\s+\"[^\"\n]*\")?\)"
)
_FENCE = re.compile(r"^(```|~~~).*?^\1", re.DOTALL | re.MULTILINE)
# Sphinx cross-references label a link with one word, a code span, or emphasis.
_CROSS_REFERENCE = re.compile(r"\S+|(`+)[^`]+\1|\*+[^*]+\*+")
_INDEXED_SITES = ("https://ml-explore.github.io/", "https://github.com/ml-explore/")


def _compact_link(match: re.Match[str]) -> str:
    label, url = match.group(1), match.group(2)
    if _CROSS_REFERENCE.fullmatch(label) or not url.startswith(_INDEXED_SITES):
        return label
    return f"[{label}]({url})"


def compact_links(text: str) -> str:
    """Drop link titles, cross-reference URLs, and external URLs outside fences.

    Sphinx renders every type and API mention as a titled link, which can double
    a section's length. Labels stay; multi-word prose links to MLX sources keep
    their URL.
    """
    parts: list[str] = []
    position = 0
    for fence in _FENCE.finditer(text):
        parts += [_LINK.sub(_compact_link, text[position : fence.start()]), fence[0]]
        position = fence.end()
    parts.append(_LINK.sub(_compact_link, text[position:]))
    return "".join(parts)


def _source_fence(text: str) -> str:
    """Choose a fence that source text cannot close."""
    # A source fragment can contain its own fences or begin inside a code block.
    fences = [
        char
        * max(
            3,
            1
            + max(
                (
                    len(m.group())
                    for m in re.finditer(
                        re.escape(char) + "+",
                        text,
                    )
                ),
                default=0,
            ),
        )
        for char in ("`", "~")
    ]
    return min(fences, key=len)


def fence_source(text: str) -> str:
    """Quote source text inside a fence that it cannot close or impersonate."""
    fence = _source_fence(text)
    return f"{fence}text\n{text}\n{fence}"


@dataclass(frozen=True)
class SourceBlock:
    """Source identity from a rendered header, bound to its quoted evidence."""

    url: str
    text: str


def source_blocks(response: str) -> list[SourceBlock]:
    """Read only fenced evidence under top-level source headers.

    The renderer quotes every source body. Headings, links, and instructions in
    that body cannot introduce another source. Read the original lines to retain
    whitespace that Markdown's fence tokens would otherwise normalize.
    """
    tokens = MarkdownIt().parse(response)
    lines = re.split(r"(?<=\n)|(?<=\r)(?!\n)", response)
    blocks: list[SourceBlock] = []
    url: str | None = None
    for i, token in enumerate(tokens):
        if token.type == "heading_open" and token.level == 0:
            url = None
        if (
            i >= 3
            and token.type == "paragraph_open"
            and token.level == 0
            and tokens[i - 3].type == "heading_open"
            and tokens[i - 3].tag == "h3"
        ):
            match = re.match(rf"Source: ({SOURCE_ID})(?:\n|$)", tokens[i + 1].content)
            url = match[1] if match else None
        if url is not None and token.type == "fence" and token.level == 0:
            if token.map is None:
                raise ValueError("Source fence lacks its line range")
            start, end = token.map
            if not re.fullmatch(
                r"[ \t]*" + re.escape(token.markup) + r"[ \t]*",
                lines[end - 1].rstrip("\n"),
            ):
                raise ValueError("Source fence is not closed")
            text = "".join(lines[start + 1 : end - 1]).removesuffix("\n")
            blocks.append(SourceBlock(url, text))
            url = None
    return blocks


def bounded_notice(text: str, char_budget: int) -> str:
    """Bound advisory text so it cannot consume the source evidence budget."""
    if len(text) <= char_budget:
        return text
    notice = "\nAdditional notices omitted; narrow the request or fetch the source.\n\n"
    if char_budget < len(notice):
        raise ValueError("Notice budget is too small for its omission message")
    return text[: char_budget - len(notice)] + notice


def _source_slice(text: str, offset: int, char_budget: int) -> tuple[int, str]:
    """Return (end, fenced slice) for source text from offset within char_budget.

    The fence is longer than any backtick or tilde run in the slice. The slice
    ends at a line break when one falls in the second half of the available space.
    """
    fence = _source_fence(text[offset : offset + char_budget])
    opening, closing = f"{fence}text\n", f"\n{fence}"
    available = char_budget - len(opening) - len(closing)
    if available < 1:
        raise ValueError(
            "Response metadata or source fences exceed the response size limit"
        )
    end = min(len(text), offset + available)
    if end < len(text):
        newline = text.rfind("\n", offset + available // 2, end)
        if newline >= 0:
            end = newline + 1
    return end, opening + text[offset:end] + closing


def bounded_excerpt(text: str, char_budget: int) -> str:
    """Fence text whole if it fits char_budget; otherwise fence a marked prefix.

    A shortened prefix is fenced once, after the shortening notice.
    """
    whole = fence_source(text)
    if len(whole) <= char_budget:
        return whole
    notice = (
        "Excerpt shortened, so code or tables may be cut off. "
        "Use fetch_mlx_docs on the URL above for the full text.\n\n"
    )
    _, body = _source_slice(text, 0, char_budget - len(notice))
    return notice + body


def render_fetch(
    text: str, url: str, requested_url: str, offset: int, *, notice: str = ""
) -> str:
    """Render a fetch response: the whole source if it fits, otherwise one page.

    A page is the fenced slice starting at offset, headed by its character range
    and the exact call that fetches the next page.
    """
    header = f"Source: {url}\n"
    if url != requested_url:
        header += f"Requested API anchor: {requested_url}\n"
    if notice:
        header += notice + "\n"
    if offset == 0 and len(header) + 1 + len(text) <= RESPONSE_CHAR_BUDGET:
        return header + "\n" + text
    if offset >= len(text):
        raise ValueError(
            f"offset must be less than the source length ({len(text)} characters)"
        )

    def page_header(end: int, more: bool) -> str:
        continuation = (
            f"Next: fetch_mlx_docs(url={json.dumps(url)}, offset={end})\n"
            if more
            else "End of source.\n"
        )
        return (
            header
            + f"Characters: {offset}:{end} of {len(text)} (0-based, end exclusive).\n"
            + continuation
            + "This is one page of the source, so code or tables may continue "
            "on the next.\n\n"
        )

    # Budget for the longest header (largest end offset, with a Next: line)
    # before choosing the slice, so the header cannot outgrow its reservation.
    budget = RESPONSE_CHAR_BUDGET - len(page_header(len(text), True))
    end, body = _source_slice(text, offset, budget)
    return page_header(end, end < len(text)) + body


def version_warning(
    requests: tuple[VersionRequest, ...], passages: list[Passage]
) -> str:
    """Warn for each requested version that the returned passages do not match.

    Each request is compared only with passages from its own package. Passages
    without a version, such as a docs page lacking one, trigger the warning.
    """
    warnings: list[str] = []
    for request in requests:
        own = [p for p in passages if p.package is request.package]
        versions = {p.documentation_version for p in own}
        if (
            versions
            and "" not in versions
            and len(versions) == 1
            and all(
                v == request.version or v.startswith(request.version + ".")
                for v in versions
            )
        ):
            continue
        label = request.package.label
        found = (
            "The results below are from version "
            + ", ".join(sorted(v or "unknown" for v in versions))
            if versions
            else f"None of the results below are from {label}"
        )
        warnings.append(
            f"You asked for {label} {request.version}. {found}, and mlx-mcp has "
            f"only one version of these docs. Check the {label} {request.version} "
            "docs for differences.\n\n"
        )
    return "".join(warnings)


def bounded_ranked_section(passages: list[Passage], char_budget: int) -> str:
    """Keep the highest-ranked excerpt when a merged section cannot fit.

    Select whole additional excerpts in relevance order before merging them in
    source order. A large, lower-ranked opening must not push the leading hit out
    of the response. Omission is explicit; fetch returns the section's complete
    indexed text, though not byte for byte: like search, it runs compact_links,
    which drops link titles and reduces cross-reference links and links to other
    sites to their labels.
    """
    notice = (
        "Excerpt shortened. Less relevant parts of this section were left out to "
        "make room. Use fetch_mlx_docs on the URL above for the full text.\n\n"
    )
    available = char_budget - len(notice)
    chosen = passages[:1]

    def evidence(group: list[Passage]) -> str:
        text = (
            selected_evidence(group[0])
            if len(group) == 1
            else expand_section(group, include_context=False)
        )
        return compact_links(text)

    body = evidence(chosen)
    for passage in passages[1:]:
        candidate = [*chosen, passage]
        text = evidence(candidate)
        rendered = fence_source(text)
        if len(rendered) <= available:
            chosen, body = candidate, text
    return notice + bounded_excerpt(body, available)


def section_header(
    passages: list[Passage],
    *,
    include_versions: bool = False,
) -> str:
    """Render a section's title, source URL, package and evidence metadata."""
    first = passages[0]
    if any(character.isspace() for character in first.url):
        raise ValueError("Source URLs must not contain whitespace")

    def line(label: str, value: str) -> str:
        return f"{label}{' '.join(value.splitlines())}\n"

    header = line("### ", first.title) + line("Source: ", first.url)
    parents = tuple(parent for parent in first.section_urls[:-1] if parent != first.url)
    if parents:
        header += line("Context source: ", parents[-1])
    header += line("Package: ", first.package.label)
    if first.evidence_kind is not EvidenceKind.DOCUMENTATION:
        header += line("Evidence: ", first.evidence_kind.value)
    if first.source_revision:
        header += line("Source revision: ", first.source_revision)
    if include_versions or first.package is Package.LM:
        versions = sorted({p.documentation_version or "unknown" for p in passages})
        header += line("Documentation version: ", ", ".join(versions))
    if any(partial_source_block(p) for p in passages):
        header += (
            "Some code or tables in this excerpt are cut off. Fetch the full section "
            "before using them.\n"
        )
    return header + "\n"


def render_sections(
    ranked: list[Passage],
    limit: int,
    char_budget: int = RESPONSE_CHAR_BUDGET,
    requested_versions: tuple[VersionRequest, ...] = (),
    *,
    include_versions: bool = False,
) -> str:
    """Render the first limit distinct sections of ranked within char_budget.

    Sections keep rank order, and each gets a short preview. Short sources, and
    explanations competing with captured CLI help, are then completed in full; the
    remaining budget fills other selected evidence in rank order. Oversized merged
    sections retain their highest-ranked excerpt before lower-ranked source spans.
    Surrounding context replaces an excerpt only when it fits whole. Authored
    guidance keeps the selected excerpts; fetch supplies its complete record.
    """
    groups: dict[tuple[str, str], list[Passage]] = {}
    for passage in ranked:
        key = section_key(passage)
        if key in groups or len(groups) < limit:
            groups.setdefault(key, []).append(passage)
    headers: list[str] = []
    bodies: list[str] = []
    sources: list[str] = []
    contexts: list[str] = []
    for passages in groups.values():
        if len(passages) == 1:
            body = selected_evidence(passages[0])
            context = selected_evidence(passages[0], expand_passage(passages[0]))
        else:
            body = expand_section(passages, include_context=False)
            context = expand_section(passages)
        body, context = compact_links(body), compact_links(context)
        sources.append(body)
        headers.append(
            section_header(
                passages, include_versions=include_versions or bool(requested_versions)
            )
        )
        bodies.append(fence_source(body))
        contexts.append(fence_source(context))
    separator = "\n\n---\n\n"
    available = (
        char_budget - sum(map(len, headers)) - len(separator) * max(0, len(bodies) - 1)
    )
    if available < 256 * len(bodies):
        # An evidence condition is indivisible; return links instead of clipping it.
        links = ""
        for passages in groups.values():
            link = f"Source: {passages[0].url}\nFetch this URL for the full text.\n"
            if len(links + link) <= char_budget:
                links += link
        return links
    # Reserve short previews, then preserve selected evidence in rank order.
    # Surrounding context must not displace a selected signature or code block.
    preview = min(512, available // max(1, len(bodies)))
    budgets = [min(len(body), preview) for body in bodies]
    remaining = available - sum(budgets)
    kinds = [passages[0].evidence_kind for passages in groups.values()]
    # Preserve the rank priority of long code examples. When captured CLI help
    # competes with an explanation, finish that explanation before expanding help.
    for i, body in enumerate(bodies):
        extra = len(body) - budgets[i]
        before_cli = EvidenceKind.CLI in kinds and kinds[i] is not EvidenceKind.CLI
        if (len(body) <= 1024 or before_cli) and extra <= remaining:
            budgets[i] += extra
            remaining -= extra
    for i, body in enumerate(bodies):
        extra = min(len(body) - budgets[i], remaining)
        budgets[i] += extra
        remaining -= extra
        # Shorten the unfenced source so a partial block is fenced only once.
        passages = list(groups.values())[i]
        bodies[i] = (
            bounded_ranked_section(passages, budgets[i])
            if len(passages) > 1 and len(body) > budgets[i]
            else bounded_excerpt(sources[i], budgets[i])
        )
    remaining = available - sum(map(len, bodies))
    for i, context in enumerate(contexts):
        extra = len(context) - len(bodies[i])
        if extra <= remaining:
            bodies[i] = context
            remaining -= extra
    return separator.join(
        header + body for header, body in zip(headers, bodies, strict=True)
    )


def fetch_text(passages: list[Passage]) -> str:
    """Assemble indexed source text with links in compact form."""
    return compact_links(source_text(passages))
