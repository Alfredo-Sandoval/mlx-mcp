"""Split canonical passages into windows that fit a model's token budget."""

import re
from dataclasses import replace

from tokenizers import Tokenizer

from mlx_mcp.passages import Passage, chunk_spans, group_passages_by_segment
from mlx_mcp.search_text import retrieval_input

# Maximum tokens in one indexed window's ranking input, counted by the
# reranker's tokenizer. Of 256, 320, 384, 448 and 512, 448 found the most
# facts in the frozen search questions (58 of 62).
DOCUMENT_TOKENS = 448
# Adjacent windows overlap by this many characters or a quarter window, if less.
WINDOW_OVERLAP_CHARS = 160


def _next_window_start(boundaries: list[int], start: int, end: int) -> int:
    """Start the next window with a small overlap, on a line boundary if possible."""
    overlap = max(start + 1, end - min(WINDOW_OVERLAP_CHARS, (end - start) // 4))
    if end in boundaries:
        return next((point for point in boundaries if overlap <= point <= end), end)
    return overlap


class TokenWindows:
    """Canonical source windows measured by the selected model's tokenizer.

    The copied tokenizer measures complete inputs without truncation or padding.
    Long passages are split into overlapping windows. The windows method raises
    ValueError when the query/title or a source character cannot fit its budget.
    """

    def __init__(self, tokenizer: Tokenizer, max_tokens: int) -> None:
        self.tokenizer = Tokenizer.from_str(tokenizer.to_str())
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.max_tokens = max_tokens

    def token_count(self, text: str, pair: str | None = None) -> int:
        """Count model tokens for text, or for the (text, pair) input."""
        return len(self.tokenizer.encode(text, pair).ids)

    def check_query(self, query: str) -> None:
        """Raise ValueError unless the expanded query fits in half the token budget.

        Each (query, window) pair must fit max_tokens, so a longer query would
        leave too little room for passage text. The count includes the terms query
        expansion adds. DocumentationService runs this before recall, and the
        reranker's score repeats it for callers that score without searching.
        """
        limit = self.max_tokens // 2
        count = self.token_count(query)
        if count > limit:
            raise ValueError(
                f"Query too long: {count} tokens after search adds related terms, "
                f"and the limit is {limit}. Shorten the query."
            )

    def windows(self, passage: Passage, query: str | None = None) -> list[Passage]:
        """Split passage into windows that fit the model's token budget.

        Without query, each window's model input fits DOCUMENT_TOKENS for
        the index; with query, each (query, window) pair fits max_tokens.
        Windows end at line breaks where possible, split oversized lines by
        character, and overlap slightly. Structured windows record their range
        within segment_text.
        """

        def fits(text: str) -> bool:
            document = retrieval_input(passage.title, text)
            return (
                self.token_count(document) <= DOCUMENT_TOKENS
                if query is None
                else self.token_count(query, document) <= self.max_tokens
            )

        if fits(passage.text):
            return [passage]
        if not fits(""):
            raise ValueError("The query is too long for the search model. Shorten it.")
        windows: list[Passage] = []
        start = 0
        boundaries = [match.end() for match in re.finditer("\n", passage.text)] + [
            len(passage.text)
        ]
        while start < len(passage.text):
            ends = [point for point in boundaries if point > start]
            low, high = 0, len(ends) - 1
            end = start
            while low <= high:
                middle = (low + high) // 2
                if fits(passage.text[start : ends[middle]]):
                    end, low = ends[middle], middle + 1
                else:
                    high = middle - 1
            # An individual oversized code/data line still needs bounded spans.
            if end == start:
                low, high = start + 1, ends[0]
                while low <= high:
                    middle = (low + high) // 2
                    if fits(passage.text[start:middle]):
                        end, low = middle, middle + 1
                    else:
                        high = middle - 1
            if end == start:
                raise ValueError("Source cannot fit a model ranking window")
            if not passage.text[start:end].strip():
                start = end
                continue
            windows.append(
                replace(
                    passage,
                    text=passage.text[start:end],
                    chunk_start=passage.chunk_start + start
                    if passage.section_path
                    else passage.chunk_start,
                    chunk_end=passage.chunk_start + end
                    if passage.section_path
                    else passage.chunk_end,
                )
            )
            if end == len(passage.text):
                break
            start = _next_window_start(boundaries, start, end)
        return windows

    def prepare_index_windows(self, passages: list[Passage]) -> list[Passage]:
        """Rechunk each canonical segment and split it into index windows."""
        # Re-window canonical segments, not yesterday's windows. This preserves
        # source metadata and makes rebuilds independent of prior boundaries.
        pages: dict[str, list[Passage]] = {}
        for passage in passages:
            pages.setdefault(
                passage.section_urls[0] if passage.section_path else passage.url, []
            ).append(passage)
        prepared: list[Passage] = []
        for page in pages.values():
            for group in group_passages_by_segment(page):
                original = group[0]
                if not original.section_path:
                    prepared.extend(window for p in group for window in self.windows(p))
                    continue
                anchors = tuple(
                    dict.fromkeys(anchor for p in group for anchor in p.anchors)
                )
                spans = chunk_spans(original.segment_text)
                for position, (start, end, chunk) in enumerate(spans):
                    passage = replace(
                        original,
                        text=chunk,
                        chunk_start=start,
                        chunk_end=end,
                        anchors=anchors if position == 0 else (),
                    )
                    prepared.extend(self.windows(passage))
        return prepared
