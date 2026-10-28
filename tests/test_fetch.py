"""DocumentationService.fetch: URL resolution, anchors, and paging."""

import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import pytest
from helpers import (
    REPEATED_PAGE,
    fixed_order_service,
    service_in_memory,
    source_page,
)

from mlx_mcp import (
    retrieval,
)
from mlx_mcp.index_file import write_index
from mlx_mcp.packages import Package
from mlx_mcp.passages import Passage
from mlx_mcp.responses import (
    RESPONSE_CHAR_BUDGET,
    render_sections,
)
from mlx_mcp.service import DocumentationService
from mlx_mcp.sphinx_pages import parse_page


@pytest.mark.parametrize(
    "source",
    [
        "```python\n" + "mx.eval(x)\n" * 3000 + "```\n\nFinal restriction.\n",
        "λ🌱\t" * 15000 + "\n",
    ],
)
def test_fetch_pages_reconstruct_exact_source_with_code_unicode_and_long_lines(
    source: str,
) -> None:
    url = "https://docs/page#section"
    index = fixed_order_service([Passage("Section", url, "Excerpt", source)])
    offset, recovered, pages = 0, "", 0
    while offset < len(source):
        result = index.fetch(url, offset)
        assert len(result) <= RESPONSE_CHAR_BUDGET
        start, end, total, body = source_page(result)
        assert start == offset and end > start and total == len(source)
        assert body == source[start:end]
        recovered += body
        pages += 1
        if end < total:
            assert f'Next: fetch_mlx_docs(url="{url}", offset={end})' in result
        else:
            assert "End of source." in result and "Next:" not in result
        offset = end
    assert pages > 1
    assert recovered == source


@pytest.mark.parametrize("offset", [-1, True, 10**20])
def test_fetch_rejects_invalid_offsets(offset: int) -> None:
    index = fixed_order_service([Passage("Title", "https://docs/page", "Text", "Text")])
    with pytest.raises(ValueError, match="offset"):
        index.fetch("https://docs/page", offset)


def test_sections_and_complete_page_preserve_content() -> None:
    html = """<article class="bd-article"><section id="guide"><h1>Guide</h1>
    <p>Introduction</p><section id="eval"><h2>Evaluation</h2>
    <p>Call eval.</p><pre>mx.eval(x)</pre></section>
    <section id="save"><h2>Saving</h2><p>Save arrays.</p></section>
    </section></article>"""
    passages = parse_page(html, "https://docs/guide.html")
    assert [p.url for p in passages] == [
        "https://docs/guide.html#guide",
        "https://docs/guide.html#eval",
        "https://docs/guide.html#save",
    ]
    assert passages[1].title == "Guide / Evaluation"
    index = service_in_memory(passages)
    section = index.fetch(passages[1].url)
    assert "mx.eval(x)" in section and "Save arrays." not in section
    assert "Save arrays." in index.fetch("https://docs/guide.html#guide")
    page = index.fetch("https://docs/guide.html")
    assert all(text in page for text in ["Introduction", "mx.eval(x)", "Save arrays."])
    with pytest.raises(ValueError, match="isn't in"):
        index.fetch("https://unknown/")


def test_search_parent_link_reaches_intro_and_child(tmp_path: Path) -> None:

    html = '<article class="bd-article"><section id="parent"><h1>Parent</h1><p>Required introduction.</p><section id="tips"><h2>Tips</h2><p>Required qualifier.</p></section></section></article>'
    passages = parse_page(html, "https://docs/guide")
    child = next(p for p in passages if p.url.endswith("#tips"))
    output = render_sections([child], 1)
    assert "Context source: https://docs/guide#parent" in output
    write_index(tmp_path, passages)
    fetched = DocumentationService(tmp_path).fetch("https://docs/guide#parent")
    assert "Required introduction." in fetched and "Required qualifier." in fetched


def test_html_anchors_resolve_only_to_their_owning_section() -> None:

    page = Package.MLX.base_url + "usage/example.html"
    html = """<article class="bd-article"><section id="guide">
      <span id="guide-label"></span><h1>Guide</h1><p>Introduction.</p>
      <section id="class"><h2>Class</h2><dl><dt id="mlx.Example">class Example</dt>
      <dd><span id="mlx.Example.__init__"></span>Constructor details.</dd></dl></section>
      <section id="other"><h2>Other</h2><p id="_CPPv4encoded">C++ details.</p></section>
    </section></article>"""
    passages = parse_page(html, page)
    index = service_in_memory(passages)
    for anchor, section, fact in [
        ("guide-label", "guide", "Introduction."),
        ("mlx.Example.__init__", "class", "Constructor details."),
        ("_CPPv4encoded", "other", "C++ details."),
    ]:
        result = index.fetch(page + "#" + anchor)
        assert f"Source: {page}#{section}" in result
        assert fact in result
    assert "C++ details." not in index.fetch(page + "#mlx.Example")
    for target in [page + "#Guide-label", page + "#unknown"]:
        with pytest.raises(ValueError, match="isn't in the indexed docs"):
            index.fetch(target)


@pytest.mark.parametrize("anchor", ["café", "mlx::core::conv2d"])
def test_fragment_is_decoded_once_and_case_and_page_stay_exact(anchor: str) -> None:
    index = service_in_memory(
        parse_page(
            f'<article class="bd-article"><section id="{anchor}">'
            "<h1>Title</h1><p>Evidence.</p></section></article>",
            REPEATED_PAGE,
        )
    )
    assert "Evidence." in index.fetch(REPEATED_PAGE + "#" + quote(anchor, safe=""))
    for url in [REPEATED_PAGE + "#UNKNOWN", REPEATED_PAGE + "#bad%ff"]:
        with pytest.raises(ValueError):
            index.fetch(url)


def test_model_free_import_and_fetch_in_fresh_process(root: Path) -> None:
    code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from mlx_mcp.service import DocumentationService
index = DocumentationService(Path(sys.argv[1]))
assert 'Section 0' in index.fetch('https://docs/python/api#0')
assert not any(name in sys.modules for name in ('transformers', 'mlx.core', 'mlx_lm'))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(root),
            str(Path(retrieval.__file__).resolve().parents[1]),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_bad_fetch_suggests_indexed_api_and_still_fails() -> None:
    valid = "https://docs/python/_autosummary/mlx.core.array.html#mlx-core-array"
    index = service_in_memory(
        [Passage("mlx.core.array", valid, "Source text", "Source text")]
    )
    with pytest.raises(ValueError) as caught:
        index.fetch("https://docs/usage/array.html#array")
    assert valid in str(caught.value)
    assert "Source text" not in str(caught.value)
    assert "Source text" in index.fetch(valid)


def test_fetch_resolves_exact_api_anchor_on_its_indexed_page(tmp_path: Path) -> None:
    name, section_anchor = "mlx.core.random.bernoulli", "mlx-core-random-bernoulli"

    page = f"https://docs/python/_autosummary/{name}.html"
    html = f'''<article class="bd-article">
    <section id="{section_anchor}"><h1>{name}</h1>
    <dl><dt id="{name}">{name}()</dt><dd>API details.</dd></dl>
    <section id="examples"><h2>Examples</h2><p>Example usage.</p></section>
    </section>
    <section id="related"><h2>Related APIs</h2><p>Other APIs.</p></section>
    </article>'''
    passages = parse_page(html, page)
    write_index(tmp_path, passages)
    index = DocumentationService(tmp_path)

    result = index.fetch(f"{page}#{name}")
    assert result.startswith(
        f"Source: {page}#{section_anchor}\nRequested API anchor: {page}#{name}\n"
        "Package: MLX\nDocumentation version: unknown\n\n"
    )
    assert "API details." in result and "Example usage." in result
    assert "Other APIs." not in result
    assert (
        result.split("\n\n", 1)[1]
        == index.fetch(f"{page}#{section_anchor}").split("\n\n", 1)[1]
    )
    assert "Other APIs." in index.fetch(page)

    for invalid in (
        f"{page}#{name}.missing",
        f"{page}#Related APIs",
        f"{page}#mx.{name.removeprefix('mlx.')}",
        f"https://docs/other.html#{name}",
    ):
        with pytest.raises(ValueError, match="This URL isn't in the indexed docs"):
            index.fetch(invalid)


@pytest.mark.parametrize(
    ("versions", "expected"),
    [
        (("0.32.4",), "0.32.4"),
        (("",), "unknown"),
        (("0.32.4", ""), "0.32.4, unknown"),
        (("0.32.4", "0.31.2"), "0.31.2, 0.32.4"),
    ],
)
def test_fetch_reports_every_returned_documentation_version(
    versions: tuple[str, ...], expected: str
) -> None:
    url = "https://docs/versions"
    index = service_in_memory(
        [
            Passage("Guide", url, f"Source fact {i}.", documentation_version=version)
            for i, version in enumerate(versions)
        ]
    )
    result = index.fetch(url)
    assert f"\nDocumentation version: {expected}\n" in result
    assert all(f"Source fact {i}." in result for i in range(len(versions)))


def test_fetch_continuations_retain_the_documentation_version() -> None:
    url = "https://docs/long-versioned-page"
    source = "A source fact with unicode λ.\n" * 2000
    index = service_in_memory(
        [Passage("Guide", url, source, documentation_version="0.32.4")]
    )
    first = index.fetch(url)
    _, end, _, _ = source_page(first)
    second = index.fetch(url, end)
    for response in (first, second):
        assert "\nDocumentation version: 0.32.4\n" in response
        assert len(response) <= RESPONSE_CHAR_BUDGET
