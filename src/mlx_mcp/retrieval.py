"""Keyword and API-name candidate recall and score-based ordering over an in-memory index."""

from __future__ import annotations

import re
from collections import Counter
from difflib import get_close_matches
from functools import cached_property
from urllib.parse import urlsplit

import numpy as np

from mlx_mcp.model_runtime import (
    ScoreArray,
)
from mlx_mcp.packages import Language, Package
from mlx_mcp.passages import (
    Passage,
    explanatory_passage,
)
from mlx_mcp.search_text import QueryPlan, plan_query, retrieval_input
from mlx_mcp.symbols import (
    FILE_SUFFIXES,
    build_symbol_index,
    definition_identifiers,
)

# Top results from each recall signal that get priority for rerank candidacy.
RESERVED_INPUT_LIMIT = 6
# Bound the cross-encoder's work after keyword and API-name recall. The review
# questions exposed relevant lexical matches below the former 16-row cutoff.
RERANK_CANDIDATE_LIMIT = 64
# Reranker score added to API definitions inferred from the query's wording.
INFERRED_SYMBOL_BONUS = 5.0
# camelCase and dotted names in a query, such as cudaMemcpyAsync or torch.compile.
# A trailing sentence period is not part of the name.
_CAMEL_NAME = re.compile(r"(?<![\w.])[a-z][a-z0-9]*[A-Z]\w*(?!\w|\.\w)")
_DOTTED_NAME = re.compile(r"(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+(?!\w|\.\w)")

# A word that opens with digits and then a letter, such as 4bit, 2d or 8bit_adam.
# A word character or dot just before it means it is the tail of a longer word or
# a decimal such as 1.5b, which get no whole-word token.
_DIGIT_WORD = re.compile(r"(?<![\w.])[0-9]+[a-z]\w*")


def tokenize(text: str) -> list[str]:
    """Lowercase identifiers, keeping qualified names alongside their parts.

    A name that starts with a letter or underscore is kept whole (``float16``)
    and, when it is dotted, ``::``-qualified or has underscores, also as its
    parts: ``mlx.core.array`` gives ``mlx.core.array``, ``mlx``, ``core`` and
    ``array``.

    A word that starts with digits and then a letter (``4bit``, ``2d``, ``3x3``)
    keeps the tokens it has always produced from its letters (``bit``, ``d``,
    ``x3``) and also gives the whole word, so a search for ``4bit`` can match
    ``4bit`` exactly without losing matches on ``bit``. For ``8bit_adam`` the
    extra tokens are ``8bit_adam`` and ``8bit``; the later parts already come
    from the letters. Numbers with no letter (``16``, ``2026``) and decimals
    such as ``1.5b`` give no extra token.

    The keyword counts for each passage are built from this when the index
    loads and are not stored in the index file, so a change here needs no
    index rebuild.
    """
    lowered = text.lower()
    symbols = re.findall(r"[a-zA-Z_][\w]*(?:(?:\.|::)[a-zA-Z_][\w]*)*", lowered)
    tokens = [
        part
        for symbol in symbols
        for part in (
            [symbol] + re.split(r"\.|::|_", symbol)
            if "." in symbol or "::" in symbol or "_" in symbol
            else [symbol]
        )
    ]
    for word in _DIGIT_WORD.findall(lowered):
        tokens.append(word)
        if "_" in word:
            tokens.append(word.split("_", 1)[0])
    return tokens


def lexical_scores(
    query: str, lexical: list[Counter[str]], eligible: list[int]
) -> dict[int, float]:
    """Score eligible rows with BM25 (k1=1.2, b=0.75), without query expansion."""
    if not eligible:
        return {}
    lengths = [sum(lexical[i].values()) for i in eligible]
    average = max(float(np.mean(lengths)), 1)
    scores = dict.fromkeys(eligible, 0.0)
    for term in set(tokenize(query)):
        frequency = sum(term in lexical[i] for i in eligible)
        idf = float(np.log(1 + (len(eligible) - frequency + 0.5) / (frequency + 0.5)))
        for i, length in zip(eligible, lengths):
            count = lexical[i][term]
            scores[i] += (
                idf * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * length / average))
            )
    return scores


def apply_ranking_policy(
    query: str, passages: list[Passage], scores: ScoreArray
) -> list[Passage]:
    """Order reranked candidates so requested APIs and guides lead.

    Inferred API definitions receive INFERRED_SYMBOL_BONUS. Explicit definitions,
    named guides, and the context that explains them then move ahead of other
    results, which keep their score order.
    """
    plan = plan_query(query)
    eligible = list(range(len(passages)))
    symbols = build_symbol_index(passages)
    definitions = symbols.definitions(plan.explicit, eligible)
    inferred = set(symbols.definitions(plan.hints, eligible)) - set(definitions)
    # A finite recall prior replaces compulsory inferred-API ordering. Strong
    # contrary reranker evidence can still reject a mistaken inference.
    adjusted_scores = np.asarray(scores, dtype=np.float64).copy()
    adjusted_scores[list(inferred)] += INFERRED_SYMBOL_BONUS
    guides = guide_title_matches(plan.expanded, passages, eligible)
    guides = [i for i in guides if adjusted_scores[i] >= 0]
    ranked = [
        p
        for _, p in sorted(
            zip(adjusted_scores, passages), key=lambda item: item[0], reverse=True
        )
    ]
    if definitions and len({passages[i].url for i in definitions}) == 1:
        definition_url = passages[definitions[0]].url
        ranked = [p for p in ranked if p.url == definition_url] + [
            p for p in ranked if p.url != definition_url
        ]
    if 0 in guides and not definitions:
        ranked = [passages[0]] + [p for p in ranked if p != passages[0]]
    # Comparisons need each requested API before incidental mentions or examples.
    if len({passages[i].title for i in definitions}) > 1:
        urls = {passages[i].url for i in definitions}
        ranked = [p for p in ranked if p.url in urls] + [
            p for p in ranked if p.url not in urls
        ]
    elif definitions or (
        inferred
        and adjusted_scores[passages.index(ranked[0])] > 0
        and passages.index(ranked[0]) in inferred
    ):
        # Source guides that explain the requested API are better context than
        # competing API definitions that merely share operation vocabulary.
        # CLI help titles also contain display labels beyond the command name.
        identifiers = {
            name.lower()
            for i in definitions or [passages.index(ranked[0])]
            for name in definition_identifiers(passages[i])
        }
        context = {
            passages[i].url
            for name in identifiers
            for i, weight in symbols.aliases.get(name, ())
            if weight < 100
            and adjusted_scores[i] > 0
            and not passages[i].title.startswith(("mlx.", "mlx::"))
            and explanatory_passage(passages[i])
        }
        # Positive inferred operations can replace competing definitions, while
        # explanatory guides retain their score order and negative hints stay down.
        context.update(passages[i].url for i in inferred if adjusted_scores[i] > 0)
        ranked = (
            ranked[:1]
            + [p for p in ranked[1:] if p.url in context]
            + [p for p in ranked[1:] if p.url not in context]
        )
    if guides and (
        definitions or inferred or len({passages[i].url for i in guides}) > 1
    ):
        requested = {passages[i].url for i in definitions + guides}
        if not definitions:
            # Compound requests need every positively scored inferred operation,
            # not just the leading one, before incidental competing APIs.
            requested.update(
                passages[i].url for i in inferred if adjusted_scores[i] > 0
            )
        ranked = [p for p in ranked if p.url in requested] + [
            p for p in ranked if p.url not in requested
        ]
        guide_urls = {passages[i].url for i in guides}
        if len(guide_urls) == 1 and not definitions:
            # A question mapped to one guide still needs its explanation beside
            # the first inferred API, before additional inferred definitions.
            ranked = (
                ranked[:1]
                + [p for p in ranked[1:] if p.url in guide_urls]
                + [p for p in ranked[1:] if p.url not in guide_urls]
            )
    return ranked


def matches_language(passage: Passage, language: Language) -> bool:
    """Return whether a passage belongs to language.

    API pages are classified by their /python/ or /cpp/ directory; other pages by
    the code they contain. Pages with no language evidence match every filter.
    """
    path = urlsplit(passage.url).path
    for api_language in ("python", "cpp"):
        if f"/{api_language}/" in path:
            return language in ("all", api_language)
    text = passage.segment_text or passage.text
    code = "\n".join(re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL))
    python = bool(
        re.search(r"```(?:python3?|py)\b|\b(?:import mlx|from mlx|mx\.)", text)
        or re.search(
            r"^\s*(?:(?:async\s+)?def \w+\s*\(|class \w+[^\n]*:\s*$|"
            r"(?:from [\w.]+ import|import [\w.]+)\b|>>>)",
            code,
            re.MULTILINE,
        )
    )
    cpp = bool(
        re.search(r"```(?:cpp\b|c\+\+)|#include\s*[<\"]|\b(?:mlx|mx)::", text)
        or re.search(r"\bstd::|\bnamespace \w+\s*[={]", code)
    )
    return (
        language == "all"
        or not (python or cpp)
        or (python if language == "python" else cpp)
    )


def guide_title_matches(
    query: str, passages: list[Passage], eligible: list[int]
) -> list[int]:
    """Return eligible passages whose multiword guide heading the query names.

    API headings, one-word headings, headings shared by several sections, and
    headings contained in a longer matching heading are excluded.
    """
    words = " " + " ".join(re.findall(r"\w+", query.lower())) + " "
    matches: dict[str, list[int]] = {}
    for i in eligible:
        title = passages[i].title.rsplit(" / ", 1)[-1]
        if title.startswith(("mlx.", "mlx::")):
            continue
        title_words = re.findall(r"\w+", title.lower())
        if len(title_words) >= 2 and " " + " ".join(title_words) + " " in words:
            matches.setdefault(title, []).append(i)
    # Multiple named guides can be independent parts of one question. Discard
    # ambiguous headings and shorter phrases contained in a more specific title.
    return [
        i
        for title, rows in matches.items()
        if len({passages[j].url for j in rows}) == 1
        and not any(title != other and title in other for other in matches)
        for i in rows
    ]


def candidate_indices(
    lexical: list[int],
    fused: list[int],
    symbol: list[int] | None = None,
) -> list[int]:
    """Select at most RERANK_CANDIDATE_LIMIT candidates, returned in fused order.

    Places go first to the leading RESERVED_INPUT_LIMIT symbol matches, then to
    fused results among the top RERANK_CANDIDATE_LIMIT lexical results, then to
    the rest of fused. Keyword recall is the only text signal, so it may fill
    every place the reranker has.
    """
    preferred = set(lexical[:RERANK_CANDIDATE_LIMIT])
    # Reserve API matches before filling the budget with lexical matches.
    priority = (
        list(symbol or [])[:RESERVED_INPUT_LIMIT]
        + [i for i in fused if i in preferred]
        + fused
    )
    reserved = set(list(dict.fromkeys(priority))[:RERANK_CANDIDATE_LIMIT])
    return [i for i in fused if i in reserved][:RERANK_CANDIDATE_LIMIT]


class RetrievalIndex:
    """Row-aligned lexical terms and API aliases for a loaded index."""

    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages
        self.lexical = [
            Counter(tokenize(f"{p.title} {retrieval_input(p.title, p.text)}"))
            for p in passages
        ]
        self.symbol_index = build_symbol_index(passages)

    @cached_property
    def vocabulary(self) -> frozenset[str]:
        """Every keyword-index token, including qualified names."""
        return frozenset(token for terms in self.lexical for token in terms)

    def spelling_matches(self, query: str) -> list[str]:
        """Return the closest indexed word for each query word the index lacks.

        Keyword recall cannot match a misspelling such as ``quantizied``, so
        search also scores the nearest indexed word (``quantized``). A match
        that contains the word, or is contained by it (``restore`` and
        ``restored``, ``reproduce`` and ``produce``), is another word form
        rather than a typo and is skipped. Words shorter than five letters, and
        names with digits, dots or underscores, are left alone.
        """
        return [
            match
            for token in dict.fromkeys(tokenize(query))
            if len(token) >= 5 and token.isalpha() and token not in self.vocabulary
            for match in get_close_matches(token, self.vocabulary, n=1, cutoff=0.85)
            if token not in match and match not in token
        ]

    def eligible(
        self, query: str, language: Language, package: Package | None
    ) -> list[int]:
        """Validate query filters and select rows without loading models."""
        if re.search(r"(?<!\S)site:", query, re.IGNORECASE):
            raise ValueError(
                "site: is not supported: this tool searches the local MLX documentation "
                "snapshot. Remove the operator and use a question or API name."
            )
        if language not in {"all", "python", "cpp"}:
            raise ValueError("language must be all, python, or cpp")
        eligible = [
            i
            for i, p in enumerate(self.passages)
            if matches_language(p, language)
            and (package is None or p.package is package)
        ]
        return eligible

    def candidates(self, plan: QueryPlan, eligible: list[int]) -> list[Passage]:
        """Fuse supported lexical and alias recall with reserved API/guide rows."""
        original_query = plan.original
        query = plan.expanded
        symbol = self.symbol_index.rank(query, eligible)
        definitions = self.symbol_index.definitions(query, eligible)
        qualified_query = bool(definitions) or any(
            token.startswith(("mlx.", "mx.", "mlx::", "mx::"))
            for token in tokenize(query)
        )
        scores = lexical_scores(
            " ".join([query, *self.spelling_matches(query)]), self.lexical, eligible
        )
        lexical = sorted(
            (i for i in eligible if scores[i] > 0),
            key=lambda i: scores[i],
            reverse=True,
        )
        # Reciprocal rank fusion combines keyword and API-name ranks without
        # tuning their score scales against each other. Unsupported rows must
        # not fill unused reranker slots when recall is sparse or empty.
        recalled = set(lexical) | set(symbol)
        fused = dict.fromkeys((i for i in eligible if i in recalled), 0.0)
        for rank, i in enumerate(lexical[:48]):
            fused[i] += 1 / (60 + rank + 1)
        anchor = max(fused, key=lambda i: (fused[i], -i), default=None)
        for rank, i in enumerate(symbol[:48]):
            fused[i] += 1 / (60 + rank + 1)
        ranking = sorted(fused, key=lambda i: fused[i], reverse=True)
        candidate_priority = symbol
        guides = guide_title_matches(query, self.passages, eligible)
        original_guides = set(
            guide_title_matches(original_query, self.passages, eligible)
        )
        guides = [
            i
            for i in guides
            if i in lexical[:RESERVED_INPUT_LIMIT] or i not in original_guides
        ]
        # A unique qualified API match leads; other queries keep the fused leader.
        precise = definitions or symbol
        unique_api = len({self.passages[i].url for i in precise}) == 1
        if unique_api and qualified_query:
            anchor = max(precise, key=lambda i: (fused[i], -i))
            candidate_priority = [anchor] + [i for i in precise if i != anchor] + symbol
        if not qualified_query and guides:
            anchor = min(guides)
            candidate_priority = [anchor] + [i for i in guides if i != anchor] + symbol
        if guides and qualified_query:
            # Reserve one passage per requested section before continuation chunks.
            representatives = dict.fromkeys(
                self.passages[i].url for i in definitions + guides
            )
            candidate_priority = [
                next(i for i in definitions + guides if self.passages[i].url == url)
                for url in representatives
            ] + candidate_priority
            if len(
                {self.passages[i].url for i in guides}
            ) == 1 and not self.symbol_index.definitions(original_query, eligible):
                anchor = min(guides)
        if anchor is not None:
            ranking = [anchor] + [i for i in ranking if i != anchor]
        candidates = candidate_indices(lexical, ranking, symbol=candidate_priority)
        return [self.passages[i] for i in candidates]

    def unindexed_names(self, query: str) -> list[str]:
        """Return camelCase or dotted names in the query that no passage mentions.

        These are usually other frameworks' APIs, such as cudaMemcpyAsync or
        torch.compile. Fully qualified MLX names are checked against definitions
        instead, and short MLX spellings such as nn.Conv2d count as mentioned
        when the symbol index knows them.
        """
        names = set(_CAMEL_NAME.findall(query)) | set(_DOTTED_NAME.findall(query))
        return sorted(
            name
            for name in names
            if not name.lower().startswith(("mlx", "mx."))
            and name.rsplit(".", 1)[-1].lower() not in FILE_SUFFIXES
            and name.lower() not in self.vocabulary
            and not self.symbol_index.knows(name)
        )
