"""API definitions, aliases, namespaces, and missing-symbol checks over passages."""

import re
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import urldefrag, urlsplit

from mlx_mcp.packages import EvidenceKind, Package
from mlx_mcp.passages import (
    Passage,
    ranking_text,
)

_IDENTIFIER = r"[A-Za-z_]\w*"
_MLX_PREFIXES = ("mlx.", "mx.", "mlx::", "mx::", "mlx_")
_QUALIFIED = re.compile(rf"(?<![\w.]){_IDENTIFIER}(?:(?:\.|::){_IDENTIFIER})+")
_CALL = re.compile(rf"(?<![\w.])({_IDENTIFIER}(?:(?:\.|::){_IDENTIFIER})?)\s*\(")
_CODE = re.compile(r"```[^\n]*\n(.*?)```|`([^`]+)`", re.DOTALL)
_CPP_SIGNATURE = re.compile(
    rf"^\s*(?:inline\s+)?(?:array|bool|double|float|int|std::vector<[^>]+>)\s+({_IDENTIFIER})\s*\(",
)
_HEADING = re.compile(r"^#{1,6}\s+")
# The Data Types page lists every dtype in one table headed Type | Bytes | Description.
_DTYPE_TABLE_HEADER = re.compile(r"^\|\s*Type\s*\|\s*Bytes\s*\|", re.IGNORECASE)
_DTYPE_TABLE_ROW = re.compile(rf"^\|\s*`({_IDENTIFIER})`\s*\|")
# mlx-lm source files, as they appear in a pinned GitHub blob URL path.
_LM_SOURCE_PATH = re.compile(r"/blob/[^/]+/mlx_lm/(.+)\.py$")
# Kinds of evidence that can define an API name for exact lookup.
_DEFINITION_KINDS = frozenset(
    {
        EvidenceKind.DOCUMENTATION,
        EvidenceKind.API,
    }
)
# Dotted names ending in these are file names, not APIs.
FILE_SUFFIXES = frozenset(
    {
        "cpp", "csv", "gguf", "gputrace", "h", "json", "jsonl", "md", "metal",
        "metallib", "npy", "npz", "onnx", "pt", "pth", "py", "pyi", "safetensors",
        "toml", "txt", "yaml", "yml",
    }
)  # fmt: skip
# Distribution names of the documented packages in module spelling: mlx, mlx_data
# and mlx_lm. Names written with an underscore that are not on this list are
# usually variables or folders (mlx_model), so they need more evidence.
_PACKAGE_NAMES = frozenset(package.value.replace("-", "_") for package in Package)
# An mlx_* name that the query imports, installs or runs is a package, even
# when the index has never heard of it:
#   import mlx_vlm | from mlx_vlm.utils import load | pip install -U mlx_whisper
#   uv add mlx_vlm | python -m mlx_lm.generate
_PACKAGE_USE = re.compile(
    r"""
    (?:\bimport|\binstall|\buv\s+add|\bpoetry\s+add|(?<![\w-])-m
       |\bfrom(?=\s+mlx_\w+(?:\.\w+)*\s+import\b))
    \s+(?:-{1,2}[\w-]+\s+)*      # options such as -U or --upgrade
    ["']?(mlx_[A-Za-z]\w*)
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Query text that shows an mlx_* name is a variable: an assignment or keyword
# argument (mlx_weights = ..., mlx_path=...), a loop variable (for mlx_layer in)
# or an alias (import numpy as mlx_np).
_VARIABLE_USE = re.compile(
    r"(?<![\w.])(mlx_[A-Za-z]\w*)\s*[-+*/]?=(?!=)"
    r"|\bfor\s+(?:\w+\s*,\s*)*(mlx_[A-Za-z]\w*)\b"
    r"|\bas\s+(mlx_[A-Za-z]\w*)\b",
    re.IGNORECASE,
)
_OPTION = re.compile(r"(?<![\w-])--([a-z][a-z0-9-]+)\b")
_OPTION_DEFINITION = re.compile(r"^[ \t]{2,8}--([a-z][a-z0-9-]+)\b", re.MULTILINE)
_GENERIC_ALIASES = frozenset(
    {
        "array",
        "arrays",
        "axis",
        "axes",
        "data",
        "dtype",
        "input",
        "inputs",
        "output",
        "outputs",
        "mlx",
        "mx",
        "shape",
        "shapes",
        "stream",
        "streams",
        "value",
        "values",
    }
)


def _normalise(value: str) -> str:
    """Clean a spelling without folding case-sensitive API identifiers."""
    value = value.strip(" `.,:;()[]{}<>")
    value = re.sub(r"\s+", " ", value).strip()
    # MLX Data's documented import alias is `import mlx.data as dx`.
    return re.sub(r"^dx\.", "mlx.data.", value)


def _spaced_alias_in(alias: str, text: str) -> bool:
    """Match a spaced alias such as "rms norm" or "top k" at word starts.

    The alias must start a word, so "desktop kernel" does not contain "top k".
    A final word of three or more letters may be a stem ("rms norm" matches
    "rms normalization"); a shorter one must end its word.
    """
    end = "" if len(alias.rsplit(" ", 1)[-1]) >= 3 else r"(?!\w)"
    return re.search(rf"(?<!\w){re.escape(alias)}{end}", text) is not None


def _identifier_spellings(identifier: str) -> set[str]:
    """Return case-sensitive symbol spellings and documented import aliases."""
    identifier = _normalise(identifier)
    if not identifier:
        return set()
    aliases = {identifier}
    separator = "::" if "::" in identifier else "."
    parts = re.split(r"\.|::", identifier)
    leaf = parts[-1]
    if len(parts) > 1:
        aliases.update({leaf, separator.join(parts[1:])})
        if identifier.startswith("mlx.data."):
            aliases.add("dx." + ".".join(parts[2:]))
        elif parts[0] in {"mlx", "mx"}:
            aliases.add(separator.join(parts[1:]))
            aliases.add(separator.join(("mx", *parts[1:])))
            if len(parts) > 2 and parts[1] == "core":
                aliases.add(separator.join(parts[2:]))
                aliases.add(separator.join(("mx", *parts[2:])))
    return aliases


def _identifier_aliases(identifier: str) -> set[str]:
    """Fold case and add readable words and acronyms for ranked search only."""
    raw_identifier = identifier.strip(" `.,:;()[]{}<>")
    aliases = {spelling.lower() for spelling in _identifier_spellings(identifier)}
    raw_leaf = re.split(r"\.|::", raw_identifier)[-1]
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", raw_leaf)
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", spaced).replace("_", " ")
    spaced = _normalise(spaced).lower()
    if spaced:
        aliases.add(spaced)
        aliases.add(spaced.replace(" ", ""))
    acronym = re.match(r"([A-Z]{2,})(?=[A-Z][a-z])", raw_leaf)
    if acronym:
        aliases.add(acronym.group(1).lower())
    return {alias for alias in aliases if len(alias) >= 3}


def _looks_like_symbol(token: str) -> bool:
    return (
        "_" in token
        or any(character.isdigit() for character in token)
        or any(character.isupper() for character in token[1:])
        or (token.isupper() and len(token) >= 3)
    )


def _api_identifiers(text: str) -> set[str]:
    """Collect MLX-qualified names, called names in code, and signature names."""
    text = re.sub(r"\bdx\.", "mlx.data.", text)
    identifiers = {
        match.group(0).removesuffix(".html")
        for match in _QUALIFIED.finditer(text)
        if match.group(0).lower().startswith(_MLX_PREFIXES)
    }
    for code in _CODE.finditer(text):
        identifiers.update(
            match.group(1)
            for match in _CALL.finditer(code.group(1) or code.group(2) or "")
        )
    for line in ranking_text(text).splitlines():
        signature = _CPP_SIGNATURE.match(line)
        if signature:
            identifiers.add(signature.group(1))
        python_signature = re.match(
            rf"^\s*(?:class\s+)?({_IDENTIFIER}(?:\.|::){_IDENTIFIER}|{_IDENTIFIER})\s*\(",
            line,
        )
        if python_signature:
            identifiers.add(python_signature.group(1))
    return identifiers


def _lm_module_spelling(passage: Passage) -> str | None:
    """Return the module-qualified name of an MLX LM API passage, if its title lacks it.

    mlx_lm re-exports some functions at package level, so their titles read
    ``mlx_lm.load``. The source URL names the file that defines them
    (``.../mlx_lm/utils.py``), and users write that path: ``mlx_lm.utils.load``.
    Titles that already include the module, such as
    ``mlx_lm.generate.generate_step``, need no second spelling.
    """
    if (
        passage.package is not Package.LM
        or passage.evidence_kind is not EvidenceKind.API
    ):
        return None
    title = passage.title.strip()
    source = _LM_SOURCE_PATH.search(urlsplit(passage.url).path)
    if source is None or not title.startswith("mlx_lm."):
        return None
    module = "mlx_lm." + source.group(1).replace("/", ".")
    if module.endswith(".__init__") or title.startswith(module + "."):
        return None
    return f"{module}.{title.removeprefix('mlx_lm.')}"


def _dtype_identifiers(passage: Passage) -> set[str]:
    """Return ``mlx.core.<name>`` for each dtype in the Data Types table.

    MLX documents ``mx.float32``, ``mx.bfloat16`` and the other dtypes only as
    rows of one table (``| `float32` | 4 | 32-bit float |``), with no section
    of their own, so ``definition_identifiers`` cannot find them. Reading the
    names from the indexed table keeps this list in step with the docs. The
    table's header must be in the same chunk as its rows. Devices (``mx.cpu``,
    ``mx.gpu``) and constants (``mx.pi``, ``mx.inf``) have no table or section
    in the indexed docs, so they stay undefined here.
    """
    if (
        passage.package is not Package.MLX
        or passage.evidence_kind is not EvidenceKind.DOCUMENTATION
    ):
        return set()
    names: set[str] = set()
    in_table = False
    for line in passage.text.splitlines():
        if _DTYPE_TABLE_HEADER.match(line):
            in_table = True
        elif in_table and not line.startswith("|"):
            break
        elif in_table and (row := _DTYPE_TABLE_ROW.match(line)):
            names.add(f"mlx.core.{row.group(1)}")
    return names


def definition_identifiers(passage: Passage) -> set[str]:
    """Return the API names a passage defines, as opposed to merely mentions."""
    title = passage.title.strip()
    identifiers = _api_identifiers(title)
    if module_spelling := _lm_module_spelling(passage):
        identifiers.add(module_spelling)
    # Sphinx environment-variable definitions have no callable signature.
    identifiers.update(
        re.findall(r"^`(MLX_[A-Z0-9_]+)`\n: +", passage.text, re.MULTILINE)
    )
    # C++ operations share one section heading; only the chunk containing the
    # documented signature defines that function, not its entire section.
    if "/cpp/" in urlsplit(passage.url).path:
        # Signatures are inline code; read them without the backticks.
        for line in ranking_text(passage.text).splitlines():
            if signature := _CPP_SIGNATURE.match(line):
                identifiers.add(f"mlx::core::{signature.group(1)}")
    if (
        not identifiers
        and re.fullmatch(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*", title)
        and ("_" in title or re.search(r"[a-z][A-Z]|[A-Z]{2,}[a-z]", title))
    ):
        identifiers.add(title)
    # An unqualified class heading can link to its own fully qualified definition.
    source = passage.segment_text or passage.text
    if re.fullmatch(_IDENTIFIER, title) and re.search(
        rf"^`class {re.escape(title)}[(`]", source, re.MULTILINE
    ):
        page = urldefrag(passage.url)[0]
        for link in re.finditer(r"\]\(([^\s)]+)", source):
            linked_page, fragment = urldefrag(link.group(1))
            if linked_page == page and re.fullmatch(
                rf"(?:mlx|mx)(?:\.{_IDENTIFIER})*\.{re.escape(title)}",
                fragment,
            ):
                identifiers.add(fragment)
    return identifiers


@dataclass(frozen=True)
class SymbolIndex:
    """Alias-to-passage lookup generated from the loaded passages.

    Each alias maps to (passage index, weight) pairs. Weight 100 marks a
    definition; URL (70), heading (45), and text (25) mentions weigh less.
    Search aliases fold case; api_definitions retains native API spellings so
    classes and functions such as ReLU and relu remain distinct.

    A command-line option has three spellings: ``--max-tokens`` (what the help
    text defines), ``max_tokens`` and ``max tokens``. All three score as weight
    100 in rank(), so a question that says "max tokens" finds the command help.
    Only the dashed spelling is a definition for definitions(): a snake_case
    word in a Python question usually names a function argument, not the
    option. ``option_words`` lists the (alias, passage index) pairs that are
    the two plain-word spellings, so definitions() can leave them out.
    """

    aliases: dict[str, tuple[tuple[int, int], ...]]
    api_definitions: dict[str, tuple[int, ...]]
    passage_count: int
    packages: frozenset[Package]
    option_words: frozenset[tuple[str, int]] = frozenset()

    def lookup(self, name: str) -> tuple[int, ...]:
        """Prefer an exact API spelling, then suggest case-insensitive matches.

        Only whole API names and import aliases match; mentions and semantic
        aliases never count. Case-insensitive matches apply only when the
        index has no exact spelling, preserving class/function identity.
        """
        if not re.fullmatch(rf"{_IDENTIFIER}(?:(?:\.|::){_IDENTIFIER})*", name):
            raise ValueError("Use one API name, such as mlx.core.eval or mx.eval")
        name = _normalise(name)
        if name in self.api_definitions:
            return self.api_definitions[name]
        folded = name.lower()
        return tuple(
            sorted(
                {
                    index
                    for spelling, indices in self.api_definitions.items()
                    if spelling.lower() == folded
                    for index in indices
                }
            )
        )

    def knows(self, name: str) -> bool:
        """Report whether any passage defines or mentions this API spelling.

        Short spellings such as ``nn.Conv2d`` or ``core.eval`` are aliases of
        their qualified definitions even when no passage's words contain them.
        """
        name = _normalise(name)
        return (
            name in self.api_definitions
            or name.lower() in self.aliases
            or any(
                spelling.lower() == name.lower() for spelling in self.api_definitions
            )
        )

    def missing_definitions(self, query: str) -> list[str]:
        """Return qualified API names in the query that no passage defines.

        A name is checked when it starts with ``mlx``/``mx``, with the module of
        a documented package (``mlx_lm``), or with a namespace some passage
        defines (``nn.Module``, ``array``). Any other ``mlx_something`` start is
        a variable, folder or file (``mlx_model``, ``mlx_array.shape``), not an
        API. Names ending in a file suffix (``mlx_lm.py``) are never APIs,
        except under ``mlx``/``mx`` where ``mx.metal`` is a real module.
        """
        namespaces = {
            alias.rsplit(".", 1)[0]
            for alias, rows in self.aliases.items()
            if "." in alias and any(weight == 100 for _, weight in rows)
        }
        names = {_normalise(match.group()) for match in _QUALIFIED.finditer(query)}
        for namespace, member in re.findall(
            rf"\b({_IDENTIFIER}(?:\.{_IDENTIFIER})*)\s+({_IDENTIFIER})\s+(?:property|attribute)\b",
            query,
            re.IGNORECASE,
        ):
            if namespace.lower() in namespaces:
                names.add(_normalise(f"{namespace}.{member}"))

        def checked(name: str) -> bool:
            name = name.lower()
            first = re.split(r"\.|::", name, maxsplit=1)[0]
            if first in {"mlx", "mx"}:
                return True
            return name.rpartition(".")[2] not in FILE_SUFFIXES and (
                first in _PACKAGE_NAMES or name.rsplit(".", 1)[0] in namespaces
            )

        return sorted(
            name
            for name in names
            if checked(name)
            and not any(
                weight == 100 for _, weight in self.aliases.get(name.lower(), ())
            )
            and not self.lookup(name)
        )

    def missing_packages(self, query: str) -> list[str]:
        """Return mlx-* package names in the query that the index does not cover.

        These are names the query clearly uses as packages. Names come back in
        the module spelling (``mlx_vlm``), whichever separator the query used.
        ``mlx-community`` is the Hugging Face organization that owns most MLX
        model ids, as in ``mlx-community/Llama-3.2-3B-Instruct-4bit``; it is not
        a package. A name counts when:

        - it is hyphenated (``mlx-vlm``);
        - it is imported, installed or run: ``import mlx_vlm``,
          ``pip install mlx_whisper``, ``python -m mlx_lm``;
        - it is ``mlx_lm`` or ``mlx_data``, the packages this server indexes,
          alone or before a dot.

        A dotted name with no such clue, like ``mlx_vlm.generate``, may be a
        package or a variable; possible_packages returns those.
        """
        return sorted(self._package_mentions(query)[0])

    def possible_packages(self, query: str) -> list[str]:
        """Return mlx_* names that may be packages the index does not cover.

        Spelling alone cannot tell the module in ``mlx_vlm.generate`` from the
        variable in ``mlx_weights.items``, so the caller words this notice as a
        possibility rather than dropping it: a missed notice would let answers
        about another package pass for answers about this one. Some names are
        left out because the query or the index shows what they are:

        - ``mlx_model`` alone is an ordinary name (the default output folder of
          ``mlx_lm.convert``); only a dotted name is a possible package;
        - ``mlx_lm.py`` ends in a file suffix, so it is a file;
        - ``mlx_array.shape`` is a variable named after an MLX array: the index
          knows ``array.shape``;
        - ``mlx_weights = ...``, ``for mlx_layer in ...`` and ``as mlx_np``
          assign the name, so it is a variable.
        """
        return sorted(self._package_mentions(query)[1])

    def _package_mentions(self, query: str) -> tuple[set[str], set[str]]:
        """Return (clear, possible) uncovered package names; see the callers."""
        clear = re.findall(r"(?<![\w.])mlx-[A-Za-z]\w*", query, re.IGNORECASE)
        clear += [match.group(1) for match in _PACKAGE_USE.finditer(query)]
        variables = {
            name.lower()
            for match in _VARIABLE_USE.finditer(query)
            for name in match.groups()
            if name
        }
        possible: list[str] = []
        for match in re.finditer(
            r"(?<![\w.])(mlx_[A-Za-z]\w*)((?:\.[A-Za-z_]\w*)*)", query, re.IGNORECASE
        ):
            module, members = match.group(1), match.group(2).split(".")[1:]
            if members and members[-1].lower() in FILE_SUFFIXES:
                continue
            if module.lower() in _PACKAGE_NAMES:
                clear.append(module)
            elif (
                members
                and module.lower() not in variables
                and not self.knows(f"{module[4:]}.{members[0]}")
            ):
                possible.append(module)
        covered = {package.value.replace("-", "_") for package in self.packages}

        def uncovered(names: list[str]) -> set[str]:
            return (
                {
                    name.lower().replace("-", "_")
                    for name in names
                    if not re.fullmatch(r"MLX_[A-Z0-9_]+", name)
                }
                - {"mlx_community"}
                - covered
            )

        found = uncovered(clear)
        return found, uncovered(possible) - found

    def missing_configuration(self, query: str) -> list[str]:
        """Return MLX_* variable names in the query that no passage defines."""
        return sorted(
            name
            for name in set(re.findall(r"\bMLX_[A-Z0-9_]+\b", query))
            if not any(
                weight == 100 for _, weight in self.aliases.get(name.lower(), ())
            )
        )

    def definitions(self, query: str, eligible: list[int]) -> list[int]:
        """Return eligible passages that define an API the query names.

        A command-line option's help counts only when the query spells the
        option with its dashes (``--max-tokens``). A bare ``max_tokens`` is
        usually a function argument, and ``option_words`` keeps it from making
        every command that has the option a definition.
        """
        names = {_normalise(name) for name in _api_identifiers(query)}
        names.update(_normalise(match.group()) for match in _QUALIFIED.finditer(query))
        names.update(_normalise(match.group(1)) for match in _CALL.finditer(query))
        spelled_options = {name.replace("-", "_") for name in _OPTION.findall(query)}
        names.update(
            _normalise(token)
            for token in re.findall(
                rf"(?<![\w.:]){_IDENTIFIER}(?![\w.:])",
                query,
            )
            if _looks_like_symbol(token)
        )
        normalised_query = _normalise(query).lower()
        for alias in self.aliases:
            if "." in alias and not alias.startswith(("mlx.", "mx.")):
                readable = alias.replace(".", " ")
                if readable in normalised_query and re.search(
                    rf"(?<!\w){re.escape(readable)}(?!\w)", normalised_query
                ):
                    names.add(alias)
        allowed = set(eligible)
        return sorted(
            {
                index
                for name in names | spelled_options
                for index, weight in self.aliases.get(name.lower(), ())
                if weight == 100
                and index in allowed
                and (
                    name not in self.api_definitions
                    or index in self.api_definitions[name]
                )
                and (
                    name in spelled_options
                    or (name.lower(), index) not in self.option_words
                )
            }
        )

    def rank(self, query: str, eligible: list[int]) -> list[int]:
        """Rank eligible passages by weighted alias matches, best first.

        Return an empty list when no alias in the query matches.
        """
        if not eligible:
            return []
        query_aliases: set[str] = set()
        query_aliases.update(name.replace("-", "_") for name in _OPTION.findall(query))
        qualified = set(_api_identifiers(query))
        for identifier in qualified:
            query_aliases.update(_identifier_aliases(identifier))
        for match in _CALL.finditer(query):
            query_aliases.update(_identifier_aliases(match.group(1)))
        for code in _CODE.finditer(query):
            for token in re.findall(
                rf"{_IDENTIFIER}", code.group(1) or code.group(2) or ""
            ):
                for candidate in (token, *token.split("_")):
                    if len(candidate) >= 3 and candidate.lower() in self.aliases:
                        query_aliases.add(candidate.lower())
        for token in re.findall(rf"{_IDENTIFIER}", query):
            if not _looks_like_symbol(token):
                continue
            for candidate in (token, *token.split("_")):
                if len(candidate) >= 3 and candidate.lower() in self.aliases:
                    query_aliases.add(candidate.lower())
        normalised_query = _normalise(query.replace("_", " ")).lower()
        query_aliases.update(
            alias
            for alias in self.aliases
            if " " in alias
            and alias in normalised_query
            and _spaced_alias_in(alias, normalised_query)
        )
        if not query_aliases:
            return []

        allowed = set(eligible)
        scores: defaultdict[int, float] = defaultdict(float)
        for alias in query_aliases:
            references = self.aliases.get(alias, ())
            # Very common words are already covered by BM25. A qualified query
            # is precise enough to use them; bare aliases need to discriminate.
            if alias in _GENERIC_ALIASES and "." not in alias and "::" not in alias:
                continue
            if (
                "." not in alias
                and "::" not in alias
                and len({index for index, _ in references})
                > max(24, self.passage_count // 20)
            ):
                continue
            idf = 1.0 + (self.passage_count / max(len(references), 1)) ** 0.5
            for index, weight in references:
                if index in allowed:
                    scores[index] += weight * idf
        return sorted(scores, key=lambda index: (-scores[index], index))


def build_symbol_index(passages: list[Passage]) -> SymbolIndex:
    """Extract API names from titles, URLs, signatures, headings, and code."""
    references: dict[str, dict[int, int]] = defaultdict(dict)
    api_definitions: dict[str, set[int]] = defaultdict(set)
    option_words: set[tuple[str, int]] = set()
    for index, passage in enumerate(passages):
        if passage.evidence_kind is EvidenceKind.CLI:
            for name in _OPTION_DEFINITION.findall(passage.text):
                for alias in (name.replace("-", "_"), name.replace("-", " ")):
                    references[alias][index] = 100
                    option_words.add((alias, index))
        for identifier in definition_identifiers(passage):
            if passage.evidence_kind in _DEFINITION_KINDS:
                for spelling in _identifier_spellings(identifier):
                    api_definitions[spelling].add(index)
            for alias in _identifier_aliases(identifier):
                references[alias][index] = max(references[alias].get(index, 0), 100)
        # Dtype rows are exact-lookup definitions only. Ranking them like other
        # definitions would make the Data Types table lead every query that
        # mentions float32 or int8, so they get no search alias.
        for identifier in _dtype_identifiers(passage):
            for spelling in _identifier_spellings(identifier):
                api_definitions[spelling].add(index)
        for source, weight in ((passage.url, 70), (passage.text, 25)):
            for identifier in _api_identifiers(source):
                for alias in _identifier_aliases(identifier):
                    references[alias][index] = max(
                        references[alias].get(index, 0), weight
                    )
        # A heading or signature may use an unqualified API name in C++ docs.
        for line in passage.text.splitlines():
            if _HEADING.match(line):
                candidate = _HEADING.sub("", line, count=1).strip()
                if re.fullmatch(rf"{_IDENTIFIER}(?:\.{_IDENTIFIER})?", candidate):
                    for alias in _identifier_aliases(candidate):
                        references[alias][index] = max(
                            references[alias].get(index, 0), 45
                        )
    return SymbolIndex(
        aliases={
            alias: tuple(sorted(rows.items())) for alias, rows in references.items()
        },
        api_definitions={
            name: tuple(sorted(rows)) for name, rows in api_definitions.items()
        },
        passage_count=len(passages),
        packages=frozenset(p.package for p in passages),
        option_words=frozenset(option_words),
    )
