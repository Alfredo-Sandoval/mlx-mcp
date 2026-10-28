"""Score development and independently authored documentation questions.

The default fixtures are tests/fixtures/questions.json and unseen_questions.json.
Use --group to report regression, review or validation questions separately.

Each case runs through the production search. The script then reads every
returned source through fetch, following its declared continuation pages up to
--fetch-pages, and checks for the case's expected source text. Elapsed times
are rough diagnostics, not a latency benchmark. Counts measure source-text
retrieval rather than generated-answer accuracy and retain each evaluation group.
"""

import argparse
import asyncio
import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NotRequired, TypedDict
from urllib.parse import SplitResult, unquote, urlsplit

from markdown_it import MarkdownIt
from pydantic import TypeAdapter

from mlx_mcp.indexing import select_index_directory
from mlx_mcp.model_runtime import ModelAccess
from mlx_mcp.packages import Language, Package
from mlx_mcp.passages import ranking_text, source_text
from mlx_mcp.responses import (
    RESPONSE_CHAR_BUDGET,
    SOURCE_ID,
    SourceBlock,
    compact_links,
    source_blocks,
)
from mlx_mcp.service import DocumentationService, parse_search_limit


@dataclass(frozen=True)
class FetchPage:
    """One parsed fetch page and the offset of the next page, if any."""

    source: SourceBlock
    next_offset: int | None


def parse_fetch(response: str, offset: int = 0) -> FetchPage:
    """Recover the exact source text of one render_fetch page.

    offset is the offset the page was requested at. Raise ValueError for a
    malformed header or fence, or a range that does not continue from offset.
    """
    header, body = response.split("\n\n", 1)
    source = re.match(rf"Source: ({SOURCE_ID})\n", header + "\n")
    if source is None:
        raise ValueError("Fetch response lacks a source URL")
    match = re.search(r"^Characters: (\d+):(\d+) of (\d+)", header, re.MULTILINE)
    if match is None:
        if offset or "\nCharacters:" in header:
            raise ValueError("Continuation lacks a character range")
        return FetchPage(SourceBlock(source[1], body), None)
    start, end, total = map(int, match.groups())
    opening, rest = body.split("\n", 1)
    fence = opening.removesuffix("text")
    if not re.fullmatch(r"`{3,}|~{3,}", fence) or not rest.endswith("\n" + fence):
        raise ValueError("Fetch page has an invalid source fence")
    text = rest[: -len(fence) - 1]
    if start != offset or not start < end <= total or len(text) != end - start:
        raise ValueError("Fetch page has a discontinuous character range")
    if end == total and "End of source." not in header:
        raise ValueError("Final fetch page lacks its end marker")
    return FetchPage(SourceBlock(source[1], text), end if end < total else None)


QUESTIONS = Path(__file__).resolve().parents[1] / "tests/fixtures/questions.json"
UNSEEN_QUESTIONS = QUESTIONS.with_name("unseen_questions.json")

type Scope = Literal["scoped", "unscoped"]
type CaseGroup = Literal["regression", "review", "validation"]


class Case(TypedDict):
    """One frozen question and the source text a good answer must cite.

    A case with no support is a control: it passes only by showing warning and
    keeping forbidden sources out of the results. An absent group belongs to
    regression; group labels preserve evaluation history, not answer accuracy.
    """

    id: str
    query: str
    limit: int
    support: list[tuple[str, str]]
    group: NotRequired[CaseGroup]
    language: NotRequired[Language]
    package: NotRequired[Package]
    warning: NotRequired[str]
    forbidden_sources: NotRequired[list[str]]


class QuestionFile(TypedDict):
    schema_version: Literal[1]
    cases: list[Case]


@dataclass(frozen=True)
class Support:
    """A required evidence span and the source it must come from."""

    source: str
    span: str

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.span.strip():
            raise ValueError("Support requires a source identity and evidence span")


def normalized(text: str) -> str:
    """Read Markdown text/code as exact word and punctuation tokens.

    HTML plaintext quotes and indexed Markdown differ in presentation and spaces
    around signatures. Preserve lexical content and pad token boundaries so a
    support identifier cannot match only part of a longer identifier.
    """
    lines = text.splitlines(keepends=True)
    for token in MarkdownIt("commonmark").parse(text):
        if token.type != "inline" or token.map is None:
            continue
        for position in range(*token.map):
            if re.match(r"^ {0,3}: {3}", lines[position]):
                # Sphinx definition prose uses code-like indentation in CommonMark.
                # A list marker admits that prose without changing fenced code.
                lines[position] = lines[position].replace(":   ", "-   ", 1)
    plain = ranking_text("".join(lines))
    return " " + " ".join(re.findall(r"\w+|[^\w\s]", plain)) + " "


def contains_support(text: str, span: str) -> bool:
    """Match a literal Markdown fragment or rendered source-text token sequence.

    Keep the original literal representation for partial quotes and unframed
    code, which need not form standalone Markdown documents. The rendered
    representation also accepts HTML plaintext quotes of the same content.
    """
    literal_span = " ".join(compact_links(span).split())
    literal_text = " ".join(compact_links(text).split())
    pattern = re.escape(literal_span)
    if re.match(r"\w", literal_span):
        pattern = r"(?<!\w)" + pattern
    if re.search(r"\w$", literal_span):
        pattern += r"(?!\w)"
    if literal_span and re.search(pattern, literal_text) is not None:
        return True
    rendered_span = normalized(span)
    return bool(rendered_span.strip()) and rendered_span in normalized(text)


def canonical_source(url: str) -> SplitResult:
    """Map exact HTTPS raw GitHub file URLs to the same repository's blob URL."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "raw.githubusercontent.com":
        return parsed
    match = re.fullmatch(r"/([^/]+)/([^/]+)/([^/]+)/(.+)", parsed.path)
    if match is None:
        return parsed
    owner, repository, revision, path = match.groups()
    return parsed._replace(
        netloc="github.com", path=f"/{owner}/{repository}/blob/{revision}/{path}"
    )


def source_matches(url: str, expected: str) -> bool:
    """Match an indexed source identity and its requested section fragment.

    A relative expected source resolves against the MLX docs site. Without a
    fragment, any section of the page matches. Raw and blob GitHub URLs match
    only when owner, repository, revision and file path are the same.
    """
    target = canonical_source(
        expected if "://" in expected else Package.MLX.base_url + expected
    )
    try:
        actual = canonical_source(url)
    except ValueError:
        return False
    return (actual.scheme, actual.netloc, actual.path) == (
        target.scheme,
        target.netloc,
        target.path,
    ) and (not target.fragment or unquote(actual.fragment) == unquote(target.fragment))


def missing_from_blocks(
    blocks: list[SourceBlock], support: list[Support]
) -> list[Support]:
    """Return support not found in its own source; text elsewhere never counts."""
    return [
        item
        for item in support
        if not any(
            source_matches(block.url, item.source)
            and contains_support(block.text, item.span)
            for block in blocks
        )
    ]


def validate_support(index: DocumentationService, cases: list[Case]) -> None:
    """Reject cases whose spans left the index or that check nothing at all."""
    for case in cases:
        support = [Support(*pair) for pair in case["support"]]
        if not support and not (case.get("warning") or case.get("forbidden_sources")):
            raise ValueError(
                f"Case {case['id']} has neither source support nor a negative constraint"
            )
        for item in support:
            source = source_text(
                [p for p in index.index.passages if source_matches(p.url, item.source)]
            )
            if not contains_support(source, item.span):
                raise ValueError(
                    f"Case {case['id']} support span changed in {item.source}"
                )


@dataclass(frozen=True)
class EvidenceScore:
    """Which support a response missed and which case constraints it broke."""

    missing: tuple[Support, ...]
    warning_ok: bool
    forbidden_sources: tuple[str, ...]
    language_ok: bool

    @property
    def complete(self) -> bool:
        return (
            not self.missing
            and self.warning_ok
            and not self.forbidden_sources
            and self.language_ok
        )


def score_evidence(
    case: Case, response: str, *, fetched: Sequence[SourceBlock] = ()
) -> EvidenceScore:
    """Score support in the response plus any fetched sources.

    The warning, forbidden-source and language checks apply to the search
    response only, so fetching can add support but cannot repair them.
    """
    initial = source_blocks(response)
    urls = [block.url for block in initial]
    language = case.get("language", "python")
    language_ok = (
        all("/cpp/" not in url for url in urls)
        if language == "python"
        else language == "all" or all("/python/" not in url for url in urls)
    )
    return EvidenceScore(
        tuple(
            missing_from_blocks(
                initial + list(fetched), [Support(*p) for p in case["support"]]
            )
        ),
        case.get("warning", "") in response,
        tuple(
            suffix
            for suffix in case.get("forbidden_sources", [])
            if any(suffix in url for url in urls)
        ),
        language_ok,
    )


@dataclass(frozen=True)
class FetchCall:
    """One complete service response and the exact continuation it declared."""

    requested_url: str
    offset: int
    response: str
    source: SourceBlock
    next_offset: int | None


@dataclass(frozen=True)
class FetchRecord:
    """Reading of one search-selected source, including any budget exhaustion."""

    source: SourceBlock
    pages: tuple[FetchCall, ...]

    @property
    def complete_source(self) -> bool:
        return self.pages[-1].next_offset is None


@dataclass(frozen=True)
class CaseMeasurement:
    """One case's search response, the sources it fetched, and their scores."""

    id: str
    query: str
    limit: int
    language: Language
    package: Package | None
    candidate_sources: tuple[str, ...]
    missing_candidate_sources: tuple[str, ...]
    response: str
    fetched: tuple[FetchRecord, ...]
    search: EvidenceScore
    workflow: EvidenceScore
    ms: float


def fetch_record(index: DocumentationService, url: str, pages: int) -> FetchRecord:
    """Follow only declared continuation offsets and retain every raw response."""
    if not 1 <= pages <= 4:
        raise ValueError("Fetch budget must be between 1 and 4 pages per source")
    records: list[FetchCall] = []
    offset = 0
    for _ in range(pages):
        response = index.fetch(url, offset)
        if len(response) > RESPONSE_CHAR_BUDGET:
            raise ValueError(f"Fetch exceeded the response budget for {url}")
        page = parse_fetch(response, offset)
        if records and page.source.url != records[0].source.url:
            raise ValueError(f"Fetch changed source identity for {url}")
        if page.next_offset is not None and page.next_offset <= offset:
            raise ValueError(f"Fetch did not advance the offset for {url}")
        records.append(FetchCall(url, offset, response, page.source, page.next_offset))
        if page.next_offset is None:
            break
        offset = page.next_offset
    return FetchRecord(
        SourceBlock(records[0].source.url, "".join(r.source.text for r in records)),
        tuple(records),
    )


async def measure_case(
    index: DocumentationService, case: Case, pages: int, scope: Scope
) -> CaseMeasurement:
    """Run the public service action, then score its search-selected reading."""
    package = case.get("package") if scope == "scoped" else None
    language = case.get("language", "python")
    # Measure the recall boundary separately: reranking cannot select a source
    # excluded here. Source presence alone says nothing about answer correctness.
    candidate_sources = tuple(
        p.url for p in index.retrieve(case["query"], language, package)
    )
    missing_sources = tuple(
        source
        for source in dict.fromkeys(source for source, _ in case["support"])
        if not any(source_matches(url, source) for url in candidate_sources)
    )
    started = time.perf_counter()
    response = await index.search(case["query"], case["limit"], language, package)
    elapsed = (time.perf_counter() - started) * 1000
    if len(response) > RESPONSE_CHAR_BUDGET:
        raise ValueError(f"Search exceeded the response budget for {case['id']}")
    urls = list(dict.fromkeys(b.url for b in source_blocks(response)))
    fetched = tuple(fetch_record(index, url, pages) for url in urls) if pages else ()
    return CaseMeasurement(
        case["id"],
        case["query"],
        case["limit"],
        language,
        package,
        candidate_sources,
        missing_sources,
        response,
        fetched,
        score_evidence(case, response),
        score_evidence(case, response, fetched=[f.source for f in fetched]),
        elapsed,
    )


def validate_cases(cases: list[Case]) -> None:
    """Reject ambiguous identities and queries outside the public tool contract."""
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Scoring needs nonempty cases with unique IDs")
    for case in cases:
        query = case["query"]
        if (
            not case["id"].strip()
            or query != query.strip()
            or not 1 <= len(query) <= 2000
        ):
            raise ValueError(f"Invalid case identity or query: {case['id']!r}")
        parse_search_limit(case["limit"])


def select_cases(cases: list[Case], group: CaseGroup | None) -> list[Case]:
    """Select one evaluation group or all cases, rejecting an empty selection."""
    selected = [
        case
        for case in cases
        if group is None or case.get("group", "regression") == group
    ]
    if not selected:
        raise ValueError(f"No cases selected for group {group!r}")
    return selected


async def score_cases(
    index: DocumentationService,
    cases: list[Case],
    pages: int,
    scope: Scope = "scoped",
) -> list[CaseMeasurement]:
    """Measure each case in order."""
    validate_cases(cases)
    if not 0 <= pages <= 4 or scope not in ("scoped", "unscoped"):
        raise ValueError("Provide a 0-4 fetch budget and a valid scope")
    return [await measure_case(index, case, pages, scope) for case in cases]


def measurement_record(row: CaseMeasurement) -> dict[str, object]:
    """Serialize exact observations with derived counts convenient for inspection."""
    return {
        **asdict(row),
        "search_complete": row.search.complete,
        "search_fetch_complete": row.workflow.complete,
        "response_characters": len(row.response),
        "fetch_calls": sum(len(f.pages) for f in row.fetched),
        "fetched_source_characters": sum(len(f.source.text) for f in row.fetched),
        "fetch_response_characters": sum(
            len(p.response) for f in row.fetched for p in f.pages
        ),
        "fetched_sources_complete": all(f.complete_source for f in row.fetched),
    }


def retrieval_counts(rows: list[CaseMeasurement], cases: list[Case]) -> dict[str, int]:
    """Count source-text retrieval separately from cases with no expected text."""
    support_ids = {c["id"] for c in cases if c["support"]}
    support = [r for r in rows if r.id in support_ids]
    controls = [r for r in rows if r.id not in support_ids]
    return {
        "cases": len(support),
        "candidate_sources_complete": sum(
            not r.missing_candidate_sources for r in support
        ),
        "search_complete": sum(r.search.complete for r in support),
        "search_fetch_complete": sum(r.workflow.complete for r in support),
        "warning_controls": len(controls),
        "warning_controls_passed": sum(r.search.complete for r in controls),
        "fetch_calls": sum(len(f.pages) for r in support for f in r.fetched),
    }


def summary(rows: list[CaseMeasurement], cases: list[Case]) -> dict[str, object]:
    """Keep overall counts and raw observations once, with counts for each group."""
    groups: dict[CaseGroup, None] = dict.fromkeys(
        case.get("group", "regression") for case in cases
    )
    grouped: dict[CaseGroup, dict[str, int]] = {}
    for group in groups:
        selected = select_cases(cases, group)
        ids = {case["id"] for case in selected}
        grouped[group] = retrieval_counts(
            [row for row in rows if row.id in ids], selected
        )
    return {
        **retrieval_counts(rows, cases),
        "source_text_retrieval_by_group": grouped,
        "case_results": [measurement_record(row) for row in rows],
    }


class ScoreOptions(argparse.Namespace):
    fixture: list[Path] | None
    output: Path
    fetch_pages: int
    scope: Scope
    index: Path | None
    model_cache: Path | None
    group: CaseGroup | None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        action="append",
        help="Question file; repeat to score several (default: both bundled question files)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--scope", choices=("scoped", "unscoped"), default="scoped")
    parser.add_argument(
        "--group",
        choices=("regression", "review", "validation"),
        help="Score one evaluation group (default: all; absent case group: regression)",
    )
    parser.add_argument("--fetch-pages", type=int, choices=range(5), default=4)
    args = parser.parse_args(namespace=ScoreOptions())
    fixtures = args.fixture or [QUESTIONS, UNSEEN_QUESTIONS]
    if args.output.exists() or args.output.resolve() in {p.resolve() for p in fixtures}:
        raise ValueError("Output must be a new file distinct from its inputs")
    cases = [
        case
        for path in fixtures
        for case in TypeAdapter(QuestionFile).validate_json(
            path.read_text(), strict=True
        )["cases"]
    ]
    validate_cases(cases)
    cases = select_cases(cases, args.group)
    root = args.index.resolve() if args.index else select_index_directory()
    index = DocumentationService(
        root, model_cache=args.model_cache, access=ModelAccess.CACHED
    )
    try:
        validate_support(index, cases)
        for case in cases:
            index.check_query(case["query"])
        started_at = datetime.now(UTC).isoformat()
        rows = asyncio.run(score_cases(index, cases, args.fetch_pages, args.scope))
        result = summary(rows, cases)
        report = {
            "schema_version": 1,
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "fixtures": [str(p) for p in fixtures],
            "snapshot": index.documentation_info(),
            "scope": args.scope,
            "fetch_pages_per_source": args.fetch_pages,
            "selected_group": args.group,
            "results": result,
        }
    finally:
        index.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k != "case_results"}, indent=2)
    )


if __name__ == "__main__":
    main()
