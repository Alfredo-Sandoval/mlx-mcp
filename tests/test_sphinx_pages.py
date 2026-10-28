"""Sphinx pages parsed into passages."""

from helpers import (
    REPEATED_PAGE,
    service_in_memory,
)

from mlx_mcp.packages import Package
from mlx_mcp.passages import validate_passages
from mlx_mcp.responses import render_sections
from mlx_mcp.sphinx_pages import parse_page


def test_extract_article_preserves_code_and_resolves_links() -> None:
    page = """
    <nav>Unrelated navigation</nav>
    <article class="bd-article">
      <h1>Lazy evaluation<a class="headerlink" href="#lazy">#</a></h1>
      <p>Evaluate arrays with <a href="../python/_autosummary/mlx.core.eval.html">eval</a>.</p>
      <pre>import mlx.core as mx\nmx.eval(x)</pre>
    </article>
    <footer>Unrelated footer</footer>
    """
    passages = parse_page(page, Package.MLX.base_url + "usage/lazy_evaluation.html")
    assert len(passages) == 1
    assert passages[0].title == "Lazy evaluation"
    assert "mx.eval(x)" in passages[0].text
    assert (
        Package.MLX.base_url + "python/_autosummary/mlx.core.eval.html"
        in passages[0].text
    )
    assert "Unrelated" not in passages[0].text


def test_repeated_segments_headingless_containers_and_empty_parents() -> None:
    html = """<article class="bd-article"><section id="outer"><h1>Repeated</h1>
    <p>Identical paragraph.</p><section><p>Headingless child.</p></section>
    <p>Identical paragraph.</p><section id="empty"><section id="deep">
    <h2>Repeated</h2><p>Deep child.</p></section></section><p>Outer tail.</p>
    </section><section id="sibling"><h1>Repeated</h1><p>Other subtree.</p></section></article>"""
    passages = parse_page(html, REPEATED_PAGE)
    validate_passages(passages)
    index = service_in_memory(passages)
    whole = index.fetch(REPEATED_PAGE)
    assert whole.count("Identical paragraph.") == 2
    assert whole.index("Headingless child.") < whole.rindex("Identical paragraph.")
    assert (
        whole.index("Deep child.")
        < whole.index("Outer tail.")
        < whole.index("Other subtree.")
    )
    assert "Other subtree." not in index.fetch(REPEATED_PAGE + "#outer")
    assert "Deep child." in index.fetch(REPEATED_PAGE + "#empty")
    assert "Outer tail." not in index.fetch(REPEATED_PAGE + "#empty")
    parent = [p for p in passages if p.url == REPEATED_PAGE + "#outer"]
    assert render_sections(parent, 1).count("Identical paragraph.") == 2
