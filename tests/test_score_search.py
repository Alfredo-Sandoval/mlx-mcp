"""scripts/score_search.py: evidence matching, fetch reading, and case scoring."""

import asyncio

import pytest
from helpers import OPS_PAGE, fixed_order_service
from hypothesis import given
from hypothesis import strategies as st
from pydantic import TypeAdapter

from mlx_mcp.passages import Passage
from mlx_mcp.responses import (
    SourceBlock,
    render_fetch,
    render_sections,
    source_blocks,
)
from mlx_mcp.service import DocumentationService
from scripts.score_search import (
    QUESTIONS,
    UNSEEN_QUESTIONS,
    Case,
    CaseGroup,
    QuestionFile,
    Support,
    contains_support,
    fetch_record,
    measurement_record,
    missing_from_blocks,
    normalized,
    score_cases,
    select_cases,
    source_matches,
    summary,
)


def grouped_case(identity: str, group: CaseGroup | None = None) -> Case:
    """Build a valid case with an optional evaluation group."""
    case: Case = {
        "id": identity,
        "query": "Evidence",
        "limit": 1,
        "support": [("https://docs/text", "Evidence text.")],
    }
    return case if group is None else {**case, "group": group}


def test_canonical_fixture_keeps_regression_and_review_groups_separate() -> None:
    cases = [
        case
        for path in (QUESTIONS, UNSEEN_QUESTIONS)
        for case in TypeAdapter(QuestionFile).validate_json(
            path.read_text(), strict=True
        )["cases"]
    ]
    assert len(select_cases(cases, "regression")) == 57
    assert len(select_cases(cases, "review")) == 30
    assert len(select_cases(cases, "validation")) == 30
    assert len({case["id"] for case in cases}) == len(cases)


def test_summary_separates_group_counts_and_keeps_raw_results_once() -> None:
    service = fixed_order_service(
        [Passage("Evidence", "https://docs/text", "Evidence text.")]
    )
    cases: list[Case] = [
        grouped_case("regression"),
        {
            **grouped_case("review", "review"),
            "support": [("https://docs/missing", "Missing")],
        },
        {**grouped_case("control", "review"), "support": [], "warning": "Evidence"},
        grouped_case("validation", "validation"),
    ]
    try:
        rows = asyncio.run(score_cases(service, cases, 0))
        result = summary(rows, cases)
        counts = {
            "cases": 1,
            "candidate_sources_complete": 1,
            "search_complete": 1,
            "search_fetch_complete": 1,
            "warning_controls": 0,
            "warning_controls_passed": 0,
            "fetch_calls": 0,
        }
        assert result["source_text_retrieval_by_group"] == {
            "regression": counts,
            "review": {
                **counts,
                "candidate_sources_complete": 0,
                "search_complete": 0,
                "search_fetch_complete": 0,
                "warning_controls": 1,
                "warning_controls_passed": 1,
            },
            "validation": counts,
        }
        assert result["cases"] == 3
        assert result["search_complete"] == 2
        assert result["warning_controls"] == 1
        assert result["case_results"] == [measurement_record(row) for row in rows]
    finally:
        service.close()


def test_support_matches_only_its_own_source() -> None:
    url = OPS_PAGE + "#metal-kernel"
    support = [Support(url, "defaults to ensure_row_contiguous=True")]
    text = "**Mechanism:** defaults to\n  ensure_row_contiguous=True"
    assert not missing_from_blocks([SourceBlock(url, text)], support)
    assert missing_from_blocks([SourceBlock(OPS_PAGE + "#other", text)], support)


@pytest.mark.parametrize(
    ("quote", "markdown"),
    [
        (
            "update ( model : Module , gradients : dict )",
            "`update(model: Module, gradients: dict)`",
        ),
        ("The group_size is 64.", "The **`group_size`** is `64`."),
        ("Use the guide.", "Use the [guide](https://docs/guide)."),
        ("Call conv1d(x, groups=1).", "Call `conv1d(x, groups=1)`."),
    ],
)
def test_normalization_matches_plaintext_and_markdown_presentation(
    quote: str, markdown: str
) -> None:
    assert normalized(quote) == normalized(markdown)


@given(
    argument=st.sampled_from(("model", "group_size", "conv1d", "weights2")),
    default=st.integers(min_value=0, max_value=10000),
    gap=st.text(alphabet=" \t", min_size=1, max_size=4),
)
def test_signature_support_ignores_only_spacing_around_punctuation(
    argument: str, default: int, gap: str
) -> None:
    source = f"`update({argument}: Module, gradients: dict = {default})`"
    quote = gap.join(
        f"update ( {argument} : Module , gradients : dict = {default} )".split()
    )
    assert contains_support(source, quote)


def test_rendered_sphinx_definition_prose_preserves_nested_code() -> None:
    text = (
        "`get_peak_memory() -> int`\n:   Get the peak in bytes.\n\n"
        "    Recorded since the last call to [`reset_peak_memory()`](https://docs/reset).\n\n"
        "    ```python\n    :   literal_code_marker\n    ```\n"
    )
    assert contains_support(text, "since the last call to reset_peak_memory()")
    assert contains_support(text, "`:   literal_code_marker`")
    assert normalized("```\n:   literal_code_marker\n```") != normalized(
        "```\n-   literal_code_marker\n```"
    )


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("- `temperature`: (Optional) A float.", "temperature`: (Optional) A float."),
        ("```\ncommand \\\n    --option\n```", "command \\\n    --option"),
    ],
)
def test_literal_fragments_do_not_need_complete_markdown(
    text: str, fragment: str
) -> None:
    assert contains_support(text, fragment)


@pytest.mark.parametrize("span", ("---", "<!-- expected code -->", "", "   "))
def test_empty_rendered_support_cannot_match_an_absent_source(span: str) -> None:
    assert not contains_support("", span)
    assert not contains_support("Unrelated documentation.", span)
    if span.strip():
        assert contains_support(f"```text\n{span}\n```", span)


@given(
    st.text(alphabet="abcXYZ012_()[]{}:,.=+-*/!?<> ", min_size=1, max_size=100).filter(
        str.strip
    )
)
def test_evidence_preserves_code_across_inline_and_fenced_markdown(code: str) -> None:
    support = [Support("https://docs/code", f"` {code} `")]
    assert not missing_from_blocks(
        [SourceBlock("https://docs/code", f"```python\n{code}\n```")], support
    )


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        ("conv1", "conv1d"),
        ("group_size", "my_group_size"),
        ("conv1d(x)", "conv2d(x)"),
        ("group_size=1", "groups=1"),
        ("groups=1", "num_groups=1"),
        ("groups=1", "groups=10"),
        ("x == 1", "x = 1"),
        ("a, b", "a b"),
        ("a / b", "a - b"),
        ("a: int", "a, int"),
    ],
)
def test_support_keeps_identifiers_numbers_and_punctuation_distinct(
    expected: str, actual: str
) -> None:
    support = [Support("https://docs/code", expected)]
    assert (
        missing_from_blocks([SourceBlock("https://docs/code", actual)], support)
        == support
    )


@given(
    owner=st.from_regex(r"[a-z][a-z0-9-]{0,12}", fullmatch=True),
    repository=st.from_regex(r"[a-z][a-z0-9-]{0,12}", fullmatch=True),
    revision=st.text(alphabet="0123456789abcdef", min_size=40, max_size=40),
    path=st.sampled_from(("README.md", "mlx_lm/LORA.md", "src/module.py")),
)
def test_raw_and_blob_sources_match_only_the_same_pinned_file(
    owner: str, repository: str, revision: str, path: str
) -> None:
    raw = f"https://raw.githubusercontent.com/{owner}/{repository}/{revision}/{path}"
    blob = f"https://github.com/{owner}/{repository}/blob/{revision}/{path}"
    changed_revision = ("1" if revision[0] == "0" else "0") + revision[1:]
    assert source_matches(raw + "#section", blob + "#section")
    assert source_matches(blob + "#section", raw + "#section")
    assert source_matches(raw + "#section", blob)
    assert not source_matches(raw + "#other", blob + "#section")
    assert not source_matches(raw, blob.replace(f"/{owner}/", f"/{owner}-other/"))
    assert not source_matches(
        raw, blob.replace(f"/{repository}/", f"/{repository}-other/")
    )
    assert not source_matches(raw, blob.replace(revision, changed_revision))
    assert not source_matches(raw, blob + ".other")


@pytest.mark.parametrize(
    "actual",
    [
        "http://raw.githubusercontent.com/owner/repository/revision/README.md",
        "https://raw.githubusercontent.com.evil/owner/repository/revision/README.md",
        "https://example.com/owner/repository/blob/revision/README.md",
        "https://raw.githubusercontent.com/owner/repository/revision",
    ],
)
def test_source_alias_never_accepts_a_different_host_or_incomplete_path(
    actual: str,
) -> None:
    expected = "https://github.com/owner/repository/blob/revision/README.md"
    assert not source_matches(actual, expected)


@pytest.mark.parametrize(
    "passages",
    [
        [Passage("Other", OPS_PAGE + "#conv1d", "int groups = 1")],
        [Passage("Right", OPS_PAGE + "#conv2d", "int groups = 2")],
        [
            Passage(
                "Spoofed host",
                "https://example.com/mlx/build/html/cpp/ops.html#conv2d",
                "int groups = 1",
            )
        ],
    ],
)
def test_support_rejects_unattributed_or_wrong_section_evidence(
    passages: list[Passage],
) -> None:
    support = [Support("cpp/ops.html#conv2d", "int groups = 1")]
    assert (
        missing_from_blocks(source_blocks(render_sections(passages, limit=2)), support)
        == support
    )


def test_unscoped_arguments_and_warning_controls() -> None:
    passage = Passage("Evaluation", "https://docs/eval", "Evaluate arrays.")
    service = fixed_order_service([passage])
    cases: list[Case] = [
        {
            "id": "first",
            "query": "Evaluate arrays",
            "limit": 1,
            "support": [(passage.url, "Evaluate arrays")],
        },
        {
            "id": "control",
            "query": "Evaluate arrays",
            "limit": 1,
            "support": [],
            "warning": "required warning",
        },
    ]
    try:
        rows = asyncio.run(score_cases(service, cases, 0, "unscoped"))
        assert [r.id for r in rows] == ["first", "control"]
        assert all(r.package is None and r.language == "python" for r in rows)
        assert not rows[1].workflow.complete
    finally:
        service.close()


class TextSource(DocumentationService):
    def __init__(self, text: str) -> None:
        self.text = text

    def fetch(self, url: str, offset: int = 0) -> str:
        return render_fetch(self.text, url, url, offset)


@given(st.text(alphabet="abc XYZ012 λ🌱\n\t`~", min_size=1, max_size=22000))
def test_collector_round_trips_unicode_and_records_exact_offsets(text: str) -> None:
    record = fetch_record(TextSource(text), "https://docs/text", 4)
    assert record.source.text == text
    assert record.complete_source
    assert record.pages[0].offset == 0
    assert [p.offset for p in record.pages[1:]] == [
        p.next_offset for p in record.pages[:-1]
    ]
    assert all(len(p.response) <= 8000 for p in record.pages)


def test_fetch_cap_and_raw_response_cost_are_preserved() -> None:
    source = TextSource("λ value\n" * 3000)
    record = fetch_record(source, "https://docs/text", 1)
    assert not record.complete_source
    assert len(record.pages) == 1
    assert source.text.startswith(record.source.text)
    assert len(record.pages[0].response) > len(record.source.text)
