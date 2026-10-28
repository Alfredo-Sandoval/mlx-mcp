"""Parse downloaded Sphinx HTML pages into passages."""

import re
from typing import TypedDict
from urllib.parse import quote, urljoin

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify
from pydantic import TypeAdapter

from mlx_mcp.packages import Package
from mlx_mcp.passages import Passage, chunk_spans


class _SphinxIndex(TypedDict):
    docnames: list[str]


def parse_page(
    html: str,
    url: str,
    *,
    retrieved_at: str = "",
    package: Package = Package.MLX,
) -> list[Passage]:
    """Split one Sphinx page into passages that record their section ancestry.

    Content a section owns directly, between its nested sections, forms one
    canonical segment, which split_text cuts into passages. Links become absolute,
    signatures become inline code, and the documentation version comes from the
    page title when the package's titles name its release.
    """
    soup = BeautifulSoup(html, "html.parser")
    title_text = soup.title.get_text(" ", strip=True) if soup.title else ""
    version = package.titled_version and re.search(
        rf"\b{re.escape(package.label)} ([\w.+-]+) documentation$", title_text
    )
    article = soup.select_one("article.bd-article")
    if article is None:
        raise ValueError(f"No documentation article found at {url}")
    for element in article.select(".headerlink, .viewcode-link, script, style"):
        element.decompose()
    # A signature becomes inline code of its visible text. As Markdown, its
    # italic parameters and type links read like `f(*a: [array](...)*)`.
    for signature in article.select("dt.sig"):
        for line_break in signature.select("br"):
            line_break.replace_with(" ")
        code = soup.new_tag("code")
        code.string = " ".join(signature.get_text().split()).replace("→", "->")
        signature.clear()
        signature.append(code)
    # Parameter types are italic pieces, so List[str] becomes *List**[*...*]*.
    # Links to Python's own docs add nothing about MLX.
    for element in article.select(
        'dl.field-list em, a[href^="https://docs.python.org/"]'
    ):
        element.unwrap()
    heading = article.find("h1")
    title = heading.get_text(" ", strip=True) if heading else url.rsplit("/", 1)[-1]
    for link in article.select("a[href]"):
        href = link["href"]
        # BeautifulSoup's HTML parser treats href as a scalar attribute.
        assert isinstance(href, str)
        link["href"] = urljoin(url, href)
    passages: list[Passage] = []

    segment_index = 0

    def collect(
        container: Tag,
        parents: list[str],
        ancestry: tuple[str, ...],
        urls: tuple[str, ...],
        node_id: str,
    ) -> None:
        heading = container.find(re.compile(r"^h[1-6]$"), recursive=False)
        label = heading.get_text(" ", strip=True) if heading else ""
        path = parents + ([label] if label else [])
        section_title = " / ".join(path) or title
        anchor = container.get("id") or (heading.get("id") if heading else None)
        section_url = f"{url}#{quote(str(anchor), safe='.:_-')}" if anchor else url
        ancestry += (node_id,)
        urls += (section_url,)
        pending: list[str] = []
        anchors: list[str] = []

        def flush() -> None:
            nonlocal segment_index
            # No escapes, so names like group_size stay copyable.
            text = markdownify(
                "".join(pending),
                heading_style="ATX",
                bullets="-",
                escape_asterisks=False,
                escape_underscores=False,
            ).strip()
            if text:
                for position, (start, end, chunk) in enumerate(chunk_spans(text)):
                    passages.append(
                        Passage(
                            section_title,
                            section_url,
                            chunk,
                            text,
                            tuple(dict.fromkeys(anchors)) if position == 0 else (),
                            section_path=ancestry,
                            section_urls=urls,
                            segment_index=segment_index,
                            chunk_start=start,
                            chunk_end=end,
                            documentation_version=version.group(1) if version else "",
                            retrieved_at=retrieved_at,
                            package=package,
                        )
                    )
                segment_index += 1
            pending.clear()
            anchors.clear()

        for child_index, child in enumerate(container.children):
            if isinstance(child, Tag) and (
                child.name == "section" or child.find("section") is not None
            ):
                flush()
                collect(child, path, ancestry, urls, f"{node_id}.{child_index}")
            else:
                pending.append(str(child))
                if isinstance(child, Tag):
                    anchors.extend(
                        str(element["id"])
                        for element in [child, *child.select("[id]")]
                        if element.get("id")
                    )
        flush()

    collect(article, [], (), (), "0")
    return passages


def sphinx_docnames(search_index: str, package: Package) -> list[str]:
    """Read safe document names from a Sphinx search index."""
    data = TypeAdapter(_SphinxIndex).validate_json(
        search_index[search_index.index("(") + 1 : search_index.rindex(")")],
        strict=True,
    )
    # Restrict requests to documentation paths on the fixed official origin.
    names = sorted(
        name
        for name in data["docnames"]
        if re.fullmatch(r"[\w-]+(?:[./][\w-]+)*", name)
    )
    if not names:
        raise ValueError(f"No documentation pages listed for {package.label}")
    return names
