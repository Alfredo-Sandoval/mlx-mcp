r"""API reference passages read from the pinned mlx-lm wheel's source.

Source is parsed with ``ast`` without importing the wheel. Passages are built
directly with their source ranges, so docstring underscores and indentation
survive unchanged; the shared ``chunk_spans`` chunker keeps Markdown block
handling consistent with Sphinx pages.
"""

from __future__ import annotations

import ast
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from mlx_mcp.mlx_lm_sources import LM_SOURCE
from mlx_mcp.packages import EvidenceKind, Package
from mlx_mcp.passages import Passage, chunk_spans

# Black-ish wrap width: past this a single-line signature is reformatted with
# one parameter per line, which is what makes long generation/cache signatures
# readable in a reference page.
_MAX_SIGNATURE_LINE = 88


@dataclass(frozen=True)
class _ModuleSpec:
    """One reference page: a module file plus which top-level names are public API."""

    path: str  # relative to mlx_lm/, e.g. "models/cache.py"
    include: frozenset[str] | None  # None = every public (non-"_") top-level def/class


# Include re-exported operations and documented generation, sampling, tokenizer,
# cache, and tuning helpers. CLI plumbing and model architectures are excluded;
# restricted modules expose only the helpers callers use directly.
MODULES: tuple[_ModuleSpec, ...] = (
    _ModuleSpec(
        "generate.py",
        frozenset(
            {
                "generate",
                "stream_generate",
                "batch_generate",
                "GenerationResponse",
                "generate_step",
                "speculative_generate_step",
            }
        ),
    ),
    _ModuleSpec(
        "convert.py",
        frozenset({"convert"}),
    ),
    _ModuleSpec(
        "utils.py",
        frozenset({"load"}),
    ),
    _ModuleSpec(
        "sample_utils.py",
        None,
    ),
    _ModuleSpec(
        "tokenizer_utils.py",
        frozenset({"TokenizerWrapper"}),
    ),
    _ModuleSpec(
        "models/cache.py",
        frozenset(
            {
                "make_prompt_cache",
                "save_prompt_cache",
                "load_prompt_cache",
                "can_trim_prompt_cache",
                "trim_prompt_cache",
            }
        ),
    ),
    _ModuleSpec(
        "tuner/utils.py",
        None,
    ),
)


def api_passages(wheel: Path, *, retrieved_at: str) -> list[Passage]:
    """Build mlx-lm API reference passages straight from the pinned wheel.

    Raises:
        SyntaxError: A selected module or the re-export source has invalid
            Python syntax. The exception retains its wheel-relative filename.
    """
    data = wheel.read_bytes()
    with zipfile.ZipFile(BytesIO(data)) as archive:
        reexports = _reexports(archive.read("mlx_lm/__init__.py").decode())
        passages = [
            passage
            for spec in MODULES
            for passage in _module_passages(
                spec,
                archive.read(f"mlx_lm/{spec.path}").decode(),
                reexports,
                retrieved_at=retrieved_at,
            )
        ]
    return passages


def _dotted_module(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".")


def _reexports(init_source: str) -> dict[str, str]:
    """Map each name in mlx_lm.__all__ to the mlx_lm/-relative file defining it."""
    tree = ast.parse(init_source, filename="mlx_lm/__init__.py")
    all_names: set[str] = set()
    imports: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            all_names = {
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
        elif isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                imports[alias.asname or alias.name] = f"{node.module}.py"
    if not all_names:
        raise ValueError("mlx_lm/__init__.py has no __all__ to resolve re-exports from")
    return {name: imports[name] for name in all_names if name in imports}


def _module_passages(
    spec: _ModuleSpec,
    module_text: str,
    reexports: dict[str, str],
    *,
    retrieved_at: str,
) -> list[Passage]:
    tree = ast.parse(module_text, filename=f"mlx_lm/{spec.path}")
    page_url = f"{LM_SOURCE.blob_base}mlx_lm/{spec.path}"
    dotted_module = _dotted_module(spec.path)
    passages: list[Passage] = []
    segment_index = 0
    top_level = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
        and (spec.include is None or node.name in spec.include)
    ]
    for index, node in enumerate(top_level):
        title = (
            f"mlx_lm.{node.name}"
            if reexports.get(node.name) == spec.path
            else f"mlx_lm.{dotted_module}.{node.name}"
        )
        segment_index = _emit_definition(
            passages,
            node,
            title,
            page_url,
            (),
            (),
            str(index),
            segment_index,
            retrieved_at=retrieved_at,
        )
        if isinstance(node, ast.ClassDef):
            class_path = (str(index),)
            class_urls = (f"{page_url}#L{node.lineno}",)
            methods = [
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not child.name.startswith("_")
            ]
            for method_index, method in enumerate(methods):
                segment_index = _emit_definition(
                    passages,
                    method,
                    f"{title}.{method.name}",
                    page_url,
                    class_path,
                    class_urls,
                    f"{index}.{method_index}",
                    segment_index,
                    retrieved_at=retrieved_at,
                )
    return passages


def _emit_definition(
    passages: list[Passage],
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    title: str,
    page_url: str,
    parent_path: tuple[str, ...],
    parent_urls: tuple[str, ...],
    node_id: str,
    segment_index: int,
    *,
    retrieved_at: str,
) -> int:
    """Append one definition's chunked passages; return the next segment_index."""
    segment_text = _render_definition(node)
    assert segment_text.strip(), (
        "a rendered definition is never empty (code fence at minimum)"
    )
    section_url = f"{page_url}#L{node.lineno}"
    section_path = (*parent_path, node_id)
    section_urls = (*parent_urls, section_url)
    for position, (start, end, chunk) in enumerate(chunk_spans(segment_text)):
        passages.append(
            Passage(
                title,
                section_url,
                chunk,
                segment_text,
                (title,) if position == 0 else (),
                documentation_version=LM_SOURCE.version,
                package=Package.LM,
                evidence_kind=EvidenceKind.API,
                source_revision=LM_SOURCE.commit,
                retrieved_at=retrieved_at,
                section_path=section_path,
                section_urls=section_urls,
                segment_index=segment_index,
                chunk_start=start,
                chunk_end=end,
            )
        )
    return segment_index + 1


def _render_definition(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> str:
    """Keep explicit constructor contracts in their owning class's segment."""
    docstrings = [ast.get_docstring(node, clean=True) or ""]
    if isinstance(node, ast.ClassDef):
        constructors = [
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == "__init__"
        ]
        code = _class_code(node, constructors)
        for constructor in constructors:
            docstring = ast.get_docstring(constructor, clean=True)
            if docstring:
                docstrings.append(f"Constructor:\n\n{docstring}")
    else:
        code = _format_signature(node)
    fenced = f"```python\n{code}\n```"
    return "\n\n".join([fenced] + [docstring for docstring in docstrings if docstring])


def _decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    return "".join(f"@{ast.unparse(dec)}\n" for dec in node.decorator_list)


def _class_code(
    node: ast.ClassDef, constructors: list[ast.FunctionDef | ast.AsyncFunctionDef]
) -> str:
    """Render fields and explicit constructors in the leading class fence."""
    bases = [ast.unparse(base) for base in node.bases]
    bases += [f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords]
    header = (
        f"class {node.name}({', '.join(bases)}):" if bases else f"class {node.name}:"
    )
    header = _decorators(node) + header
    # Dataclass-style annotated fields are the class's real "signature": the
    # nearest thing to constructor parameters for classes with no explicit
    # __init__ (e.g. GenerationResponse), so they ride in the same fence.
    fields = [
        f"{child.target.id}: {ast.unparse(child.annotation)}"
        + (f" = {ast.unparse(child.value)}" if child.value is not None else "")
        for child in node.body
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
    ]
    members = fields + [_format_signature(constructor) for constructor in constructors]
    return "\n".join(
        [header] + [f"    {line}" for member in members for line in member.splitlines()]
    )


def _format_param(arg: ast.arg, default: ast.expr | None) -> str:
    text = arg.arg
    if arg.annotation is not None:
        text += f": {ast.unparse(arg.annotation)}"
    if default is not None:
        text += (
            f" = {ast.unparse(default)}"
            if arg.annotation is not None
            else f"={ast.unparse(default)}"
        )
    return text


def _format_parameters(args: ast.arguments) -> list[str]:
    """Render parameters one per element in call order, with `/` and `*` markers."""
    positional = args.posonlyargs + args.args
    padding = len(positional) - len(args.defaults)
    positional_defaults: list[ast.expr | None] = [None] * padding + list(args.defaults)
    parts = [
        _format_param(arg, positional_defaults[i])
        for i, arg in enumerate(args.posonlyargs)
    ]
    if args.posonlyargs:
        parts.append("/")
    offset = len(args.posonlyargs)
    parts += [
        _format_param(arg, positional_defaults[offset + i])
        for i, arg in enumerate(args.args)
    ]
    if args.vararg is not None:
        parts.append(f"*{_format_param(args.vararg, None)}")
    elif args.kwonlyargs:
        parts.append("*")
    parts += [
        _format_param(arg, default)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults)
    ]
    if args.kwarg is not None:
        parts.append(f"**{_format_param(args.kwarg, None)}")
    return parts


def _format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    params = _format_parameters(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    oneline = f"{prefix} {node.name}({', '.join(params)}){returns}:"
    if not params or len(oneline) <= _MAX_SIGNATURE_LINE:
        return _decorators(node) + oneline
    wrapped_params = ",\n".join(f"    {param}" for param in params)
    return _decorators(node) + f"{prefix} {node.name}(\n{wrapped_params},\n){returns}:"
