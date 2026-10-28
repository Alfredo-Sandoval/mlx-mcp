"""API name extraction and the symbol index used by lookup and search."""

import asyncio

import pytest
from helpers import (
    FIXTURE_PAGE,
    data_passage,
    definition,
    fixed_order_service,
)
from hypothesis import given
from hypothesis import strategies as st

from mlx_mcp.mlx_lm_sources import LM_SOURCE
from mlx_mcp.packages import EvidenceKind, Package
from mlx_mcp.passages import Passage
from mlx_mcp.symbols import build_symbol_index, definition_identifiers

# Matches the MLX Data Types page, where dtypes are table rows, not sections.
DATA_TYPES_PAGE = Package.MLX.base_url + "python/data_types.html#data-types"
DATA_TYPES_TABLE = """# Data Types

The default floating point type is `float32` and the default integer type is
`int32`.

| Type | Bytes | Description |
| --- | --- | --- |
| `bool_` | 1 | Boolean (`True`, `False`) data type |
| `int32` | 4 | 32-bit signed integer |
| `bfloat16` | 2 | 16-bit brain float (e8, m7) |
| `float32` | 4 | 32-bit float |
| `float64` | 8 | 64-bit double |

Note

Arrays with type `float64` only work with CPU operations.
"""


def test_option_declarations_are_search_definitions_without_becoming_apis() -> None:
    passages = [
        Passage(
            "tool command-line options",
            "https://example/usage",
            "usage: tool [--color-space COLOR]",
            evidence_kind=EvidenceKind.CLI,
        ),
        Passage(
            "tool command-line options",
            "https://example/options",
            "  --color-space COLOR\n      Output coordinate color space.",
            evidence_kind=EvidenceKind.CLI,
        ),
        Passage(
            "Guide",
            "https://example/guide",
            "  --color-space COLOR\n      An incidental example.",
        ),
    ]
    symbols = build_symbol_index(passages)
    assert symbols.definitions("--color-space", [0, 1, 2]) == [1]
    assert symbols.rank("--color-space", [0, 1, 2])[0] == 1
    assert symbols.lookup("color_space") == ()
    assert not symbols.definitions("--color-space", [0, 2])


def command_help(command: str, *options: str) -> Passage:
    """Captured ``--help`` text for one command, as the CLI passages carry it."""
    text = "".join(f"  --{name} VALUE\n      Sets {name}.\n" for name in options)
    return Passage(
        f"{command} command-line options",
        f"https://example/cli#{command}",
        text,
        evidence_kind=EvidenceKind.CLI,
    )


def test_option_spelled_with_dashes_defines_its_command_help() -> None:
    passages = [
        command_help("mlx_lm.generate", "max-tokens", "top-p"),
        command_help("mlx_lm.server", "max-tokens"),
        Passage("Guide", "https://example/guide", "Set max_tokens to 100."),
    ]
    index = build_symbol_index(passages)
    eligible = [0, 1, 2]
    for query in ("--max-tokens", "set --max-tokens to 100"):
        assert index.definitions(query, eligible) == [0, 1]
        assert index.rank(query, eligible) == [0, 1]
    # With both spellings in one query, only the dashed one defines.
    assert index.definitions("--top-p with max_tokens", eligible) == [0]


def test_data_package_names_and_dx_alias_are_recognized() -> None:
    symbols = build_symbol_index([data_passage()])
    for query in ("mlx-data prefetch", "mlx_data prefetch"):
        assert symbols.missing_packages(query) == []
    assert symbols.missing_packages("mlx_lm.generate") == ["mlx_lm"]
    assert symbols.definitions("dx.Stream.prefetch", [0]) == [0]
    assert symbols.missing_definitions("dx.Stream.prefetch") == []
    assert symbols.definitions("mx.data.Stream.prefetch", [0]) == []


def test_symbol_index_derives_qualified_and_camel_case_aliases() -> None:
    passages = [
        Passage("mlx.core.linalg.svd", "https://docs/python/svd", "svd(*a)"),
        Passage("mlx.nn.RMSNorm", "https://docs/python/rms", "class RMSNorm(*dims)"),
    ]
    index = build_symbol_index(passages)

    assert index.rank("mx.linalg.svd", [0, 1]) == [0]
    assert index.rank("rms normalization", [0, 1]) == [1]
    assert "linalg.svd" in index.aliases


def test_explicit_unqualified_symbols_keep_each_requested_definition() -> None:
    passages = [
        Passage("mlx.optimizers.AdamW", "https://docs/adamw", "Optimizer"),
        Passage("mlx.optimizers.cosine_decay", "https://docs/cosine", "Scheduler"),
        Passage("mlx.core.conv1d", "https://docs/conv", "Convolution"),
        Passage("mlx.nn.Conv1d", "https://docs/layer", "Convolution layer"),
        Passage("mlx.nn.Linear", "https://docs/linear", "Linear layer"),
    ]
    index = build_symbol_index(passages)
    eligible = list(range(len(passages)))
    assert index.definitions("AdamW with cosine_decay", eligible) == [0, 1]
    assert index.definitions("mlx.nn.Conv1d", eligible) == [3]
    assert index.definitions("Train a linear regression model", eligible) == []


@pytest.mark.parametrize("class_name", ["ReLU", "GELU", "Sigmoid", "Softmax"])
@pytest.mark.parametrize("namespace", ["mlx.nn.", "mx.nn.", "nn.", ""])
def test_exact_api_spelling_distinguishes_classes_from_functions(
    class_name: str, namespace: str
) -> None:
    function_name = class_name.lower()
    index = build_symbol_index(
        [definition(f"mlx.nn.{class_name}"), definition(f"mlx.nn.{function_name}")]
    )
    for name, expected in ((class_name, 0), (function_name, 1)):
        spelling = namespace + name
        assert index.lookup(spelling) == (expected,)
        assert index.definitions(f"How does `{spelling}()` work?", [0, 1]) == [expected]
        assert index.missing_definitions(spelling) == []
    # Exact identity is independent of the current language/package filter.
    assert index.definitions(f"{namespace}{class_name}()", [1]) == []
    assert index.definitions(f"{namespace}{function_name}()", [0]) == []
    # Folded recall can still retrieve both related APIs.
    assert set(index.rank(f"{namespace}{function_name}()", [0, 1])) == {0, 1}


def test_wrong_case_api_spelling_falls_back_only_without_an_exact_match() -> None:
    index = build_symbol_index([definition("mlx.nn.ReLU"), definition("mlx.nn.relu")])
    assert index.lookup("MLX.NN.RELU") == (0, 1)
    assert index.lookup("nn.rElU") == (0, 1)
    assert index.definitions("How does nn.rElU work?", [0, 1]) == [0, 1]
    assert index.missing_definitions("MLX.NN.RELU nn.rElU") == []
    assert index.missing_definitions("mlx.nn.Unavailable nn.Unavailable") == [
        "mlx.nn.Unavailable",
        "nn.Unavailable",
    ]
    class_only = build_symbol_index([definition("mlx.nn.ReLU")])
    assert class_only.lookup("nn.relu") == (0,)
    assert class_only.missing_definitions("nn.relu") == []


def test_limit_one_search_returns_the_exact_function_instead_of_its_class() -> None:
    service = fixed_order_service(
        [definition("mlx.nn.ReLU"), definition("mlx.nn.relu")]
    )
    try:
        result = asyncio.run(service.search("nn.relu", limit=1))
    finally:
        service.models.worker.close()
    assert "mlx.nn.relu" in result
    assert "mlx.nn.ReLU" not in result
    assert "definitions match `" not in result


def test_cpp_signatures_define_only_their_chunk_and_preserve_language_aliases() -> None:
    convolution = "inline array conv2d(const array &input, const array &weight)"
    matrix = r"inline array quantized\_matmul(const array &input)"
    section = "# Operations\n\n" + convolution + "\n\n" + matrix
    passages = [
        Passage(
            "Operations", "https://docs/cpp/ops.html#operations", convolution, section
        ),
        Passage("Operations", "https://docs/cpp/ops.html#operations", matrix, section),
        Passage(
            "mlx.core.conv2d", "https://docs/python/conv2d", "conv2d(input, weight)"
        ),
        Passage("Guide", "https://docs/dev/guide", "Use mx::eval(x)."),
    ]
    index = build_symbol_index(passages)
    eligible = list(range(len(passages)))
    for query in ("mlx::core::conv2d", "mx::conv2d"):
        assert index.definitions(query, eligible) == [0]
        assert index.missing_definitions(query) == []
    assert index.definitions("mlx::core::quantized_matmul", eligible) == [1]
    assert index.definitions("mx.conv2d", eligible) == [2]
    assert index.definitions("conv2d", eligible) == [0, 2]
    assert index.missing_definitions("mx::eval") == ["mx::eval"]


@given(
    name=st.sampled_from(
        ["split", "meshgrid", "divmod", "depends", "broadcast_arrays"]
    ),
    std=st.sampled_from(
        ["std", '[std](https://docs/std "std")', "`std`", "**std**", "*std*"]
    ),
    array=st.sampled_from(
        ["array", '[array](https://docs/array "array")', "`array`", "**array**"]
    ),
    inline=st.sampled_from(["", "inline "]),
    escaped=st.booleans(),
    entities=st.booleans(),
)
def test_cpp_signature_definition_is_invariant_to_supported_markdown(
    name: str, std: str, array: str, inline: str, escaped: bool, entities: bool
) -> None:
    spelling = name.replace("_", r"\_") if escaped else name
    left, right = ("&lt;", "&gt;") if entities else ("<", ">")
    signature = f"{inline}{std}::vector{left}{array}{right} {spelling}(const array &a)"
    passage = Passage("Operations", "https://docs/cpp/ops.html#operations", signature)
    index = build_symbol_index([passage])
    assert definition_identifiers(passage) == {f"mlx::core::{name}"}
    assert index.lookup(f"mx::{name}") == (0,)
    assert index.missing_definitions(f"mlx::core::{name}") == []


def test_limit_one_cpp_search_leads_with_linked_vector_signature() -> None:
    operation = Passage(
        "Operations",
        "https://docs/cpp/ops.html#operations",
        '[std](https://docs/std "std")::vector<array> split(const array &a, int n)',
    )
    environment = Passage(
        "Environment Variables",
        "https://docs/usage/environment",
        "MLX_FAST_SYNCH\n: off",
    )
    service = fixed_order_service([environment, operation])
    try:
        result = asyncio.run(
            service.search("mlx::core::split", limit=1, language="cpp")
        )
    finally:
        service.models.worker.close()
    assert "Operations" in result
    assert "Environment Variables" not in result
    assert "The indexed docs don't define" not in result


def test_package_coverage_uses_ownership_and_definition_detection_uses_api_titles() -> (
    None
):
    passages = [
        Passage("Guide", "https://docs/guide", "Run mlx_lm.generate(model, prompt)."),
        Passage("mlx_audio.generate", "https://docs/audio", "Generate audio."),
    ]
    index = build_symbol_index(passages)
    assert index.missing_packages("MLX-LM prompt caching") == ["mlx_lm"]
    assert index.missing_definitions("mlx_lm.generate") == ["mlx_lm.generate"]
    assert index.missing_packages("mlx_audio.generate") == []
    assert index.possible_packages("mlx_audio.generate") == ["mlx_audio"]
    assert index.definitions("mlx_audio.generate", [0, 1]) == [1]
    assert index.missing_definitions("mlx_audio.generate") == []
    assert index.missing_definitions("mlx_audio.missing") == ["mlx_audio.missing"]
    assert index.missing_packages("mlx.core.eval and prompt caching") == []


def test_symbol_index_does_not_treat_common_words_as_api_queries() -> None:
    passages = [
        Passage("mlx.nn.Module.train", "https://docs/python/train", "train(mode=True)"),
        Passage("mlx.nn.Linear", "https://docs/python/linear", "Linear(*args)"),
        Passage(
            "Linear Regression",
            "https://docs/examples/linear_regression",
            "gradient descent",
        ),
    ]
    index = build_symbol_index(passages)

    assert (
        index.rank(
            "Train a linear regression model with gradient descent in MLX", [0, 1, 2]
        )
        == []
    )
    assert index.rank("What does `Linear` represent?", [0, 1, 2]) == [1]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "How do I load mlx-community/Llama-3.2-3B-Instruct-4bit with mlx_lm.load?",
            ["mlx_lm"],
        ),
        ("https://huggingface.co/mlx-community/Qwen3-4B-4bit", []),
        # Other mlx-* names are still packages, wherever the model id sits.
        ("mlx-vlm with mlx-community/Qwen2-VL-2B-Instruct-4bit", ["mlx_vlm"]),
    ],
)
def test_hugging_face_organization_is_not_an_external_package(
    query: str, expected: list[str]
) -> None:
    index = build_symbol_index([Passage("Guide", FIXTURE_PAGE, "Core guide text.")])
    assert index.missing_packages(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        # The default --mlx-path folder of mlx_lm.convert.
        "my mlx_model crashes",
        # A variable named after an MLX array; array.shape is documented.
        "what is mlx_array.shape after reshape?",
        # Files, including one named like the documented mlx_lm module.
        "see mlx_lm.py",
    ],
)
def test_ordinary_mlx_underscore_names_get_no_package_or_definition_notice(
    query: str,
) -> None:
    index = build_symbol_index([definition("mlx.core.array.shape")])
    assert index.missing_packages(query) == []
    assert index.missing_definitions(query) == []


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("pip install mlx-vlm", ["mlx_vlm"]),
        ("from mlx_vlm.utils import load_image", ["mlx_vlm"]),
        # The documented package, alone or with a dot, even if the index lacks it.
        ("mlx_lm.generate", ["mlx_lm"]),
        # An uppercase name is an environment variable, not a package.
        ("import MLX_ENABLE_TF32", []),
    ],
)
def test_mlx_packages_the_index_does_not_cover_are_still_reported(
    query: str, expected: list[str]
) -> None:
    index = build_symbol_index([definition("mlx.core.array.shape")])
    assert index.missing_packages(query) == expected


def test_missing_definitions_start_from_mlx_mx_or_a_documented_package() -> None:
    index = build_symbol_index([definition("mlx.core.eval")])
    assert index.missing_definitions("mlx.core.not_a_function") == [
        "mlx.core.not_a_function"
    ]
    assert index.missing_definitions("mlx_lm.not_a_function") == [
        "mlx_lm.not_a_function"
    ]
    # Another package, a variable and a file are not checked against MLX docs.
    assert not index.missing_definitions(
        "mlx_vlm.generate mlx_array.shape mlx_model.parameters mlx_lm.py"
    )
    # Under mlx/mx a file-like last word is still an API name: mx.metal is a module.
    assert index.missing_definitions("mx.cpu mx.gpu mx.pi mx.metal") == [
        "mx.cpu",
        "mx.gpu",
        "mx.metal",
        "mx.pi",
    ]


def test_dtypes_in_the_data_types_table_are_defined_but_not_ranked() -> None:
    passages = [
        definition("mlx.core.matmul"),
        Passage("Data Types", DATA_TYPES_PAGE, DATA_TYPES_TABLE),
    ]
    index = build_symbol_index(passages)
    query = (
        "Is mx.matmul faster with mx.bfloat16 or mx.float32 on mx.gpu versus mx.cpu?"
    )

    # Devices have no table or section, so only they are still reported.
    assert index.missing_definitions(query) == ["mx.cpu", "mx.gpu"]
    assert index.missing_definitions("mx.bool_ mlx.core.int32 mx.float64") == []
    assert index.missing_definitions("mx.float33 mx.float32.shape") == [
        "mx.float32.shape",
        "mx.float33",
    ]
    assert index.lookup("mx.bfloat16") == index.lookup("mlx.core.int32") == (1,)
    # Search ranking is unchanged: float32 appears in too many unrelated
    # questions for the table to lead them all.
    assert index.definitions("mx.float32 float32 int32 mx.matmul", [0, 1]) == [0]


def lm_api(title: str, module_path: str, line: int = 10) -> Passage:
    """An MLX LM API passage, titled and linked as mlx_lm_api builds them."""
    return Passage(
        title,
        f"{LM_SOURCE.blob_base}mlx_lm/{module_path}#L{line}",
        f"```python\ndef {title.rsplit('.', 1)[-1]}():\n```",
        package=Package.LM,
        evidence_kind=EvidenceKind.API,
        source_revision=LM_SOURCE.commit,
    )


def test_lm_reexports_resolve_by_the_module_that_defines_them() -> None:
    passages = [
        lm_api("mlx_lm.load", "utils.py", 453),
        lm_api("mlx_lm.stream_generate", "generate.py", 657),
        lm_api("mlx_lm.generate", "generate.py", 756),
        lm_api("mlx_lm.convert", "convert.py", 85),
        lm_api("mlx_lm.generate.generate_step", "generate.py", 307),
        lm_api("mlx_lm.models.cache.make_prompt_cache", "models/cache.py", 15),
    ]
    index = build_symbol_index(passages)

    assert definition_identifiers(passages[0]) == {"mlx_lm.load", "mlx_lm.utils.load"}
    # Titles that already name their module gain nothing.
    assert definition_identifiers(passages[4]) == {"mlx_lm.generate.generate_step"}
    for spellings, position in (
        (("mlx_lm.load", "mlx_lm.utils.load"), 0),
        (("mlx_lm.stream_generate", "mlx_lm.generate.stream_generate"), 1),
        (("mlx_lm.generate", "mlx_lm.generate.generate"), 2),
        (("mlx_lm.convert", "mlx_lm.convert.convert"), 3),
        (("mlx_lm.generate.generate_step",), 4),
        (("mlx_lm.models.cache.make_prompt_cache",), 5),
    ):
        for name in spellings:
            assert index.lookup(name) == (position,)
            assert index.missing_definitions(f"what does {name} return") == []
    assert index.definitions("mlx_lm.utils.load", list(range(6))) == [0]
    # The module comes from the source URL, never from guessing.
    assert index.lookup("mlx_lm.convert.load") == ()
    assert index.missing_definitions("mlx_lm.utils.generate") == [
        "mlx_lm.utils.generate"
    ]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # A dotted name may be a module or a variable; spelling cannot tell.
        ("mlx_vlm.generate", ["mlx_vlm"]),
        # Assigned, looped over or aliased: the query shows it is a variable.
        ("mlx_weights = load(path); mlx_weights.items()", []),
        # A clear package mention is not repeated as a possible one.
        ("import mlx_vlm; mlx_vlm.generate", []),
    ],
)
def test_dotted_mlx_names_are_possible_packages_unless_the_query_assigns_them(
    query: str, expected: list[str]
) -> None:
    index = build_symbol_index([definition("mlx.core.array.shape")])
    assert index.possible_packages(query) == expected
    assert index.missing_packages(query) == (
        ["mlx_vlm"] if query.startswith("import mlx_vlm") else []
    )
