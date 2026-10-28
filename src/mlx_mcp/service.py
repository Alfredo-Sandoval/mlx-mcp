"""Headless documentation actions shared by MCP and evaluation callers."""

from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import TypedDict
from urllib.parse import unquote, urldefrag, urlsplit

from mlx_mcp import __version__
from mlx_mcp.configuration import DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT, cache_root
from mlx_mcp.index_file import IndexFileInfo, load_index
from mlx_mcp.indexing import BUNDLED_ROOT
from mlx_mcp.model_runtime import (
    ModelAccess,
    ModelRuntime,
    ScoreArray,
)
from mlx_mcp.packages import EvidenceKind, Language, Package
from mlx_mcp.passages import (
    Passage,
    section_key,
)
from mlx_mcp.responses import (
    RESPONSE_CHAR_BUDGET,
    bounded_notice,
    fetch_text,
    render_fetch,
    render_sections,
    version_warning,
)
from mlx_mcp.retrieval import (
    RetrievalIndex,
    apply_ranking_policy,
    matches_language,
)
from mlx_mcp.search_text import expand_query, plan_query


class PackageCoverage(TypedDict):
    """Versions, sizes, and sources indexed for one package."""

    documentation_versions: list[str]
    page_count: int
    passage_count: int
    evidence_kinds: list[str]
    source_revisions: list[str]


class DocumentationInfo(TypedDict):
    """The `mlx-mcp info` report: index provenance and coverage."""

    build_version: str
    index: str
    index_built_by: str
    index_file: IndexFileInfo
    packages: dict[str, PackageCoverage]
    page_count: int
    passage_count: int
    pages_without_version: int
    retrieved_at_first: str | None
    retrieved_at_last: str | None


def parse_search_limit(value: object) -> int:
    """Parse an untrusted limit before admitting an inference job."""
    if type(value) is not int or not 1 <= value <= MAX_SEARCH_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_SEARCH_LIMIT}")
    return value


def parse_source_url(url: str) -> tuple[str, str]:
    """Split a requested source URL and strictly decode its section fragment."""
    try:
        page, fragment = urldefrag(url)
    except ValueError as exc:
        raise ValueError("Invalid source URL") from exc
    if re.search(r"%(?![0-9a-fA-F]{2})", fragment):
        raise ValueError("Invalid percent-encoded URL fragment")
    try:
        return page, unquote(fragment, errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("Invalid UTF-8 in URL fragment") from exc


def suggest_urls(url: str, passages: list[Passage]) -> list[str]:
    """Return up to three similar indexed URLs to suggest; never fetch them."""
    requested = urlsplit(url)
    name = requested.path.rsplit("/", 1)[-1].removesuffix(".html")
    scores: dict[str, float] = {}
    for passage in passages:
        candidate = urlsplit(passage.url)
        if candidate.netloc != requested.netloc:
            continue
        leaf = candidate.path.rsplit("/", 1)[-1].removesuffix(".html")
        score = SequenceMatcher(None, name, leaf.split(".")[-1]).ratio()
        if requested.path == candidate.path:
            score = (
                1
                + SequenceMatcher(None, requested.fragment, candidate.fragment).ratio()
            )
        if score >= 0.8:
            scores[passage.url] = score
    return sorted(scores, key=lambda candidate: (-scores[candidate], candidate))[:3]


class DocumentationService:
    """Search, fetch, and metadata over one validated index."""

    def __init__(
        self,
        root: Path,
        *,
        model_cache: Path | None = None,
        access: ModelAccess = ModelAccess.DOWNLOAD,
    ) -> None:
        self.root = root
        self.loaded_index = load_index(root)
        self.index = RetrievalIndex(self.loaded_index.passages)
        # Bundled indexes remain read-only; model caches always belong to the user.
        self.models = ModelRuntime(
            model_cache if model_cache is not None else cache_root(), access=access
        )

    def close(self) -> None:
        """Drain requests and release the service's inference resources."""
        self.models.close()

    def retrieve(
        self, query: str, language: Language = "all", package: Package | None = None
    ) -> list[Passage]:
        """Recall candidates within the requested filters, without loading models."""
        eligible = self.index.eligible(query, language, package)
        if not eligible:
            return []
        return self.index.candidates(plan_query(query), eligible)

    def check_query(self, query: str) -> None:
        """Raise ValueError if the reranker could not take this query once expanded.

        The reranker checks length only while scoring, and a search that recalls
        a single candidate never scores, so the same query used to pass or fail
        with the number of results. rank calls this first, before recall, so the
        outcome depends only on the query. It reads only the reranker's
        tokenizer and does not load the model.
        """
        self.models.check_query(expand_query(query))

    def score_candidates(self, query: str, passages: list[Passage]) -> ScoreArray:
        """Score candidates against the expanded query without changing their order."""
        return self.models.score(expand_query(query), passages)

    def documentation_info(self) -> DocumentationInfo:
        """Report provenance recorded in the loaded index, never the live docs."""
        pages: dict[str, set[str]] = {}
        for passage in self.index.passages:
            pages.setdefault(urldefrag(passage.url)[0], set()).add(
                passage.documentation_version
            )
        timestamps = sorted(
            {p.retrieved_at for p in self.index.passages if p.retrieved_at},
            key=datetime.fromisoformat,
        )
        return {
            "build_version": __version__,
            "index": "bundled" if self.root == BUNDLED_ROOT else str(self.root),
            "index_built_by": self.loaded_index.built_by,
            "index_file": self.loaded_index.file_info,
            "packages": {
                package.value: {
                    "documentation_versions": sorted(
                        {
                            p.documentation_version
                            for p in self.index.passages
                            if p.package is package and p.documentation_version
                        }
                    ),
                    "page_count": len(
                        {
                            urldefrag(p.url)[0]
                            for p in self.index.passages
                            if p.package is package
                        }
                    ),
                    "passage_count": sum(
                        p.package is package for p in self.index.passages
                    ),
                    "evidence_kinds": sorted(
                        {
                            p.evidence_kind.value
                            for p in self.index.passages
                            if p.package is package
                        }
                    ),
                    "source_revisions": sorted(
                        {
                            p.source_revision
                            for p in self.index.passages
                            if p.package is package and p.source_revision
                        }
                    ),
                }
                for package in Package
                if any(p.package is package for p in self.index.passages)
            },
            "page_count": len(pages),
            "passage_count": len(self.index.passages),
            "pages_without_version": sum("" in versions for versions in pages.values()),
            "retrieved_at_first": timestamps[0] if timestamps else None,
            "retrieved_at_last": timestamps[-1] if timestamps else None,
        }

    def fetch(self, url: str, offset: int = 0) -> str:
        """Render the indexed page, or the section a fragment names, from offset.

        A fragment matches a section URL or an HTML anchor, and the section's
        nested subsections are included. Unknown or ambiguous URLs raise
        ValueError with similar indexed URLs.
        """
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        requested_url = url
        page, fragment = parse_source_url(url)
        url = page + (f"#{fragment}" if fragment else "")
        page_passages = [p for p in self.index.passages if urldefrag(p.url)[0] == page]
        targets: dict[tuple[str, ...], str] = {}
        for p in page_passages:
            if p.section_path:
                for depth, target in enumerate(p.section_urls, 1):
                    if unquote(urldefrag(target)[1], errors="strict") == fragment:
                        targets[p.section_path[:depth]] = target
                if fragment in p.anchors:
                    targets[p.section_path] = p.url
            elif (
                unquote(urldefrag(p.url)[1], errors="strict") == fragment
                or fragment in p.anchors
            ):
                targets[(p.url,)] = p.url
        sections = []
        if not fragment:
            sections = page_passages
        elif len(targets) == 1:
            path, url = next(iter(targets.items()))
            sections = [
                p
                for p in page_passages
                if (
                    p.section_path[: len(path)] == path
                    if p.section_path
                    else p.url == url
                )
            ]
        if not sections:
            suggestions = suggest_urls(url, self.index.passages)
            message = (
                "This URL isn't in the indexed docs. Use a Source: URL from a search."
            )
            if len(targets) > 1:
                message += " Its #anchor matches more than one section."
            if suggestions:
                message += "\nDid you mean one of these? (not fetched)\n" + "\n".join(
                    suggestions
                )
            raise ValueError(message)
        notice = (
            # Only an index built without heading paths, such as an older local
            # build, lacks section_path; the shipped index has it on every passage.
            "This index does not record which headings sit under this one, so only "
            "this section's own text is shown. Rebuild the index to get its subsections."
            if fragment and not sections[0].section_path
            else ""
        )
        notice = f"Package: {sections[0].package.label}" + (
            "\n" + notice if notice else ""
        )
        if sections[0].evidence_kind is not EvidenceKind.DOCUMENTATION:
            notice += f"\nEvidence: {sections[0].evidence_kind.value}"
        # A fetched page can contain several recorded versions, or sections
        # whose version was unavailable. Do not silently label all with the
        # first section's version; keep this metadata on continuation pages too.
        versions = sorted(
            {
                " ".join((section.documentation_version or "unknown").splitlines())
                for section in sections
            }
        )
        notice += f"\nDocumentation version: {', '.join(versions)}"
        if sections[0].source_revision:
            notice += f"\nSource revision: {sections[0].source_revision}"
        return render_fetch(
            fetch_text(sections),
            url,
            requested_url,
            offset,
            notice=notice,
        )

    def exact_definitions(
        self, query: str, language: Language = "all", package: Package | None = None
    ) -> list[Passage]:
        """Return the definitions of the one API name a query spells, within filters.

        Empty unless the whole query is one API name, such as mx.eval,
        `mlx.nn.Linear` or mlx::core::conv2d(), matched by full name or import
        alias. Exact case wins; a case-insensitive match is used only when that
        spelling is absent. Mentions, guides and CLI help never count.
        """
        name = query.strip().strip("`").removesuffix("()")
        try:
            indices = self.index.symbol_index.lookup(name)
        except ValueError:  # Not a single API name.
            return []
        return [
            self.index.passages[i]
            for i in indices
            if matches_language(self.index.passages[i], language)
            and (package is None or self.index.passages[i].package is package)
        ]

    def rank(
        self, query: str, language: Language = "all", package: Package | None = None
    ) -> list[Passage]:
        """Return the filtered passages in search order, before the section limit.

        Reject invalid filters, then a query too long for the reranker; recall
        candidates, score them with the cross-encoder unless only one remains,
        apply the ranking policy, then promote a linked code example. A query
        that is one API name then puts that API's definitions first, even ones
        recall missed. Empty when filters exclude every passage or recall finds
        no lexical or API-name support for the query.
        """
        # Filter errors need no model files; the length check loads the
        # reranker's tokenizer, which may mean a download.
        self.index.eligible(query, language, package)
        self.check_query(query)
        passages = self.retrieve(query, language, package)
        if not passages:
            return []
        if len(passages) == 1:
            ranked = passages
        else:
            scores = self.score_candidates(query, passages)
            ranked = apply_ranking_policy(query, passages, scores)
        definitions = self.exact_definitions(query, language, package)
        if not definitions:
            return ranked
        keys = {section_key(p) for p in definitions}
        # Recalled definitions keep the reranker's order; missed ones follow.
        leading = [p for p in ranked if section_key(p) in keys]
        recalled = {section_key(p) for p in leading}
        leading += [p for p in definitions if section_key(p) not in recalled]
        return leading + [p for p in ranked if section_key(p) not in keys]

    async def search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        language: Language = "all",
        package: Package | None = None,
    ) -> str:
        """Search on the inference worker and render up to limit sections."""
        limit = parse_search_limit(limit)
        return await self.models.worker.run(
            self.search_on_worker, query, limit, language, package
        )

    def search_on_worker(
        self,
        query: str,
        limit: int,
        language: Language,
        package: Package | None = None,
    ) -> str:
        """Assemble the production response on the owning inference worker.

        Callers must validate limit with parse_search_limit before submitting
        work.
        """
        selected = self.rank(query, language, package)
        if not selected:
            if self.index.eligible(query, language, package):
                return "Nothing in the docs matches this query for the package and language you chose."
            return "No docs match the package and language you chose."
        warning = ""
        missing = self.index.symbol_index.missing_definitions(query)
        if missing:
            names = ", ".join(f"`{name}`" for name in missing)
            warning = (
                f"The indexed docs don't define {names}. The results below are related. "
                "The name may still exist in the version you have installed.\n\n"
            )
        packages = self.index.symbol_index.missing_packages(query)
        if packages:
            names = ", ".join(f"`{name}`" for name in packages)
            warning = (
                f"mlx-mcp has no docs for {names}. "
                f"It covers {', '.join(p.label for p in sorted(self.index.symbol_index.packages))}. "
                f"The results below are related, but they don't describe {names}.\n\n"
                + warning
            )
        possible = self.index.symbol_index.possible_packages(query)
        if possible:
            names = ", ".join(f"`{name}`" for name in possible)
            one = len(possible) == 1
            warning = (
                f"{names} may be {'another package' if one else 'other packages'}. "
                f"If so, mlx-mcp has no docs for {'it' if one else 'them'}. It covers "
                f"{', '.join(p.label for p in sorted(self.index.symbol_index.packages))}.\n\n"
                + warning
            )
        # A name already reported as a missing MLX definition gets one notice.
        missing_spellings = {name.lower() for name in missing}
        unindexed = [
            name
            for name in self.index.unindexed_names(query)
            if name.lower() not in missing_spellings
        ]
        if unindexed:
            names = ", ".join(f"`{name}`" for name in unindexed)
            warning += (
                f"The indexed docs never mention {names}. mlx-mcp covers "
                f"{', '.join(p.label for p in sorted(self.index.symbol_index.packages))}. "
                "The results below are related, but they don't document "
                f"{'that name' if len(unindexed) == 1 else 'those names'}.\n\n"
            )
        plan = plan_query(query)
        keys = list(dict.fromkeys(section_key(p) for p in selected))[:limit]
        returned = [p for p in selected if section_key(p) in keys]
        warning += version_warning(plan.requested_versions, returned)
        missing_configuration = self.index.symbol_index.missing_configuration(query)
        if missing_configuration:
            warning += (
                f"The indexed docs don't define {', '.join(missing_configuration)}. "
                "A passing mention below doesn't say what it does.\n\n"
            )
        definitions = {
            section_key(p) for p in self.exact_definitions(query, language, package)
        }
        if len(definitions) > limit:
            warning += (
                f"{len(definitions)} definitions match `{query}`, and this shows "
                f"{limit}. Use the full name, pick a package or language, or raise limit.\n\n"
            )
        # Advisory text gets a separate, fixed share; arbitrary version/API lists
        # must leave enough room for cited evidence within the public total cap.
        warning = bounded_notice(warning, 1600)
        return warning + render_sections(
            selected,
            limit,
            RESPONSE_CHAR_BUDGET - len(warning),
            # An API-name search shows each section's docs version.
            include_versions=bool(definitions),
            requested_versions=plan.requested_versions,
        )
