"""Keyword scoring, candidate recall, filters, and the ranking policy."""

import asyncio
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from helpers import (
    FIXTURE_PAGE,
    service_in_memory,
    service_on_disk,
)

from mlx_mcp.packages import (
    EvidenceKind,
    Language,
    Package,
)
from mlx_mcp.passages import Passage
from mlx_mcp.retrieval import (
    RERANK_CANDIDATE_LIMIT,
    RESERVED_INPUT_LIMIT,
    apply_ranking_policy,
    candidate_indices,
    lexical_scores,
    matches_language,
    tokenize,
)
from mlx_mcp.search_text import plan_query


def test_lexical_score_prefers_the_exact_digit_led_word() -> None:
    lexical = [
        Counter(tokenize(text))
        for text in (
            "Quantize to 4bit weights",
            "Quantize to 8bit weights",
            "Bit packing of weights",
        )
    ]
    scores = lexical_scores("4bit", lexical, [0, 1, 2])
    # The 8bit and bit-only passages still match through "bit".
    assert scores[0] > max(scores[1], scores[2])
    assert min(scores[1], scores[2]) > 0


def test_keyword_recall_finds_a_symbol_among_many_unrelated_passages() -> None:
    target = Passage(
        "mlx.core.conv3d", "https://docs/python/conv3d", "groups parameter"
    )
    index = service_in_memory(
        [Passage("Other", f"https://docs/cpp/{i}", "unrelated") for i in range(60)]
        + [target]
    )
    selected = index.retrieve("mlx.core.conv3d groups")
    assert selected == [target]
    assert index.retrieve("conv3d", "python") == [target]
    assert target not in index.retrieve("conv3d", "cpp")


@pytest.mark.parametrize("query", ["watershed", "DICOM", "qzxvplmn", "12345"])
def test_keyword_recall_returns_nothing_without_supported_candidates(
    query: str,
) -> None:
    index = service_in_memory(
        [Passage("Other", f"https://docs/cpp/{i}", "unrelated") for i in range(60)]
    )
    assert index.retrieve(query) == []


def test_sparse_recall_keeps_each_supported_source_without_unrelated_padding() -> None:
    supported = [
        Passage("First", "https://docs/python/first", "frobnicate inputs"),
        Passage("Second", "https://docs/python/second", "frobnicate outputs"),
    ]
    index = service_in_memory(
        [Passage("Other", f"https://docs/python/{i}", "unrelated") for i in range(60)]
        + supported
    )
    assert index.retrieve("frobnicate") == supported


@pytest.mark.parametrize(
    ("query", "target"),
    [
        (
            "quantizied",
            Passage("Quantized", "https://docs/quantized", "Packed weights."),
        ),
        (
            "Select the GPU",
            Passage(
                "mlx.core.set_default_device", "https://docs/device", "Set target."
            ),
        ),
        (
            "mx.linalg.svd",
            Passage(
                "Factorization",
                "https://docs/python/mlx.core.linalg.svd.html",
                "Decompose matrices.",
            ),
        ),
    ],
)
def test_sparse_recall_preserves_spelling_expansion_and_symbol_signals(
    query: str, target: Passage
) -> None:
    index = service_in_memory(
        [Passage("Other", f"https://docs/python/{i}", "unrelated") for i in range(60)]
        + [target]
    )
    assert index.retrieve(query) == [target]


@pytest.mark.parametrize(
    ("language", "package", "expected"),
    [
        ("python", Package.MLX, True),
        ("cpp", Package.MLX, False),
        ("all", Package.LM, False),
    ],
)
def test_query_support_outside_filters_does_not_recall_unrelated_rows(
    language: Language, package: Package, expected: bool
) -> None:
    target = Passage("Target", "https://docs/python/target", "frobnicate inputs")
    index = service_in_memory(
        [
            Passage("Other", "https://docs/cpp/other", "unrelated"),
            Passage("Other", "https://docs/lm/other", "unrelated", package=Package.LM),
            target,
        ]
    )
    assert index.retrieve("frobnicate", language, package) == (
        [target] if expected else []
    )


def test_symbol_matches_survive_a_full_keyword_budget() -> None:
    # Keyword matches alone fill every place, and API matches rank last.
    lexical = list(range(RERANK_CANDIDATE_LIMIT))
    symbol = list(
        range(RERANK_CANDIDATE_LIMIT, RERANK_CANDIDATE_LIMIT + RESERVED_INPUT_LIMIT)
    )
    fused = lexical + symbol

    reserved = candidate_indices(lexical, fused, symbol)

    # Every API match keeps a place; the best keyword matches fill the rest,
    # all in fused order.
    kept_keywords = RERANK_CANDIDATE_LIMIT - RESERVED_INPUT_LIMIT
    assert reserved == [*range(kept_keywords), *symbol]


@pytest.mark.parametrize(
    ("query", "titles"),
    [
        (
            "Select the GPU and detect CUDA availability",
            ["mlx.core.set_default_device", "mlx.core.cuda.is_available"],
        ),
        (
            "Convert to a NumPy view without copying; is it evaluated?",
            [
                "Conversion to NumPy and Other Frameworks",
                "Lazy Evaluation / When to Evaluate",
            ],
        ),
        ("Mean reduction over two axes", ["mlx.core.mean"]),
    ],
)
def test_operation_intent_retains_sources_over_incidental_matches(
    tmp_path: Path, query: str, titles: list[str]
) -> None:
    noise = [
        Passage(f"Unrelated {i}", f"https://docs/python/noise/{i}", query)
        for i in range(20)
    ]
    wanted = [
        Passage(
            title, f"https://docs/python/wanted/{i}", "Canonical operation explanation."
        )
        for i, title in enumerate(titles)
    ]
    index = service_on_disk(tmp_path, noise + wanted)
    index.score_candidates = lambda query, passages: np.array(
        [2.0 if passage in wanted else 1.0 for passage in passages],
    )
    # Recall reserves these sources; final ranking still requires model evidence.
    assert all(p in index.retrieve(query, "python") for p in wanted)
    result = asyncio.run(index.search(query, limit=len(wanted), language="python"))
    assert all(f"Source: {passage.url}" in result for passage in wanted)


def test_strongly_rejected_inferred_operation_does_not_receive_a_required_slot() -> (
    None
):
    passages = [
        Passage("Relevant guide", "https://docs/guide", "Device selection guidance."),
        Passage(
            "mlx.core.set_default_device", "https://docs/device", "Set default device."
        ),
        Passage("Another guide", "https://docs/other", "Other relevant guidance."),
    ]
    query = "Select the GPU device"
    assert "mlx.core.set_default_device" in plan_query(query).hints
    assert apply_ranking_policy(query, passages, np.array([6.0, -20.0, 5.0]))[:2] == [
        passages[0],
        passages[2],
    ]


def test_convolution_layout_intent_beats_incidental_dropout_layout_text() -> None:
    passages = [
        Passage(
            "mlx.nn.Dropout3d",
            "https://docs/python/dropout3d",
            "Zero channels in a 3D input with NDHWC layout.",
        ),
        Passage(
            "mlx.nn.Conv3d",
            "https://docs/python/conv3d",
            "Apply 3D convolution to channels-last NDHWC inputs.",
        ),
    ]
    query = "3D convolution input layout channels last NDHWC"
    # The live reranker favored Dropout3d's layout wording by less than one
    # point. Operation intent should survive that incidental vocabulary match.
    ranked = apply_ranking_policy(query, passages, np.array([6.1, 5.3]))
    assert ranked[0] == passages[1]
    # The existing finite prior still lets strong contrary evidence reject a
    # mistaken inference instead of forcing a convolution definition to lead.
    rejected = apply_ranking_policy(query, passages, np.array([6.1, -20.0]))
    assert rejected[0] == passages[0]


@pytest.mark.parametrize(
    "eval_score,order",
    [(-3.0, [0, 2, 3, 1]), (-20.0, [0, 2, 1, 3])],
)
def test_compound_guide_request_keeps_positive_operations_before_incidental_apis(
    eval_score: float, order: list[int]
) -> None:
    query = "Slice an array, convert dtype, then force computation"
    passages = [
        Passage("Indexing Arrays", FIXTURE_PAGE + "#indexing", "Use arr[2:8:2]."),
        Passage(
            "mlx.core.slice_update", FIXTURE_PAGE + "#slice", "Update a sub-array."
        ),
        Passage("mlx.core.array.astype", FIXTURE_PAGE + "#astype", "Cast the array."),
        Passage("mlx.core.eval", FIXTURE_PAGE + "#eval", "Evaluate lazy arrays."),
    ]
    ranked = apply_ranking_policy(
        query, passages, np.array([3.0, 6.0, 1.0, eval_score])
    )
    assert [p.url for p in ranked] == [passages[i].url for i in order]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Join two (4, 3) arrays into (8, 3)", "mlx.core.concatenate"),
        ("Combine arrays by adding a new batch dimension", "mlx.core.stack"),
        ("Join arrays without adding a new axis", "mlx.core.concatenate"),
    ],
)
def test_array_join_intent_recovers_the_matching_definition(
    tmp_path: Path, query: str, expected: str
) -> None:
    passages = [
        Passage("mlx.core.matmul", "https://docs/python/matmul", "Batched arrays"),
        Passage(
            "mlx.core.concatenate", "https://docs/python/concatenate", "Existing axis"
        ),
        Passage("mlx.core.stack", "https://docs/python/stack", "New axis"),
    ]
    index = service_on_disk(tmp_path, passages)
    assert index.retrieve(query)[0].title == expected


def test_boolean_selection_prefers_restrictions_over_general_indexing(
    tmp_path: Path,
) -> None:
    query = "Filter an array to extract elements with a boolean mask"
    passages = [
        Passage(
            "Indexing Arrays", "https://docs/indexing", "Indexing works like NumPy."
        ),
        Passage(
            "Indexing Arrays / Boolean Mask Assignment",
            "https://docs/assignment",
            "MLX supports boolean indices using NumPy syntax.",
        ),
        Passage(
            "Indexing Arrays / Differences from NumPy",
            "https://docs/differences",
            "Boolean mask based indexing is supported for assignment only.",
        ),
    ]
    index = service_on_disk(tmp_path, passages)
    index.score_candidates = lambda query, passages: np.array(
        [float("supports boolean" in p.text) for p in passages],
    )
    assert index.retrieve(query)[0] == passages[2]
    result = asyncio.run(index.search(query, limit=2))
    assert "supported for assignment only" in result


def test_model_and_array_evaluation_comparison_preserves_both_definitions() -> None:
    passages = [
        Passage("Llama inference", "https://docs/llama", "Evaluate a model"),
        Passage("mlx.core.eval", "https://docs/python/eval", "Evaluate arrays"),
        Passage(
            "mlx.nn.Module.eval",
            "https://docs/python/module-eval",
            "Set evaluation mode",
        ),
        Passage("mlx.nn.Dropout", "https://docs/python/dropout", "Random masking"),
    ]
    query = "Disable dropout for inference: model.eval() versus mx.eval()?"
    ranked = apply_ranking_policy(query, passages, np.array([9.0, 2.0, 1.0, 8.0]))
    assert ranked[:2] == passages[1:3]


def test_unqualified_question_allows_reranker_to_replace_weak_hybrid_leader() -> None:
    passages = [
        Passage("Numerical Precision", "https://docs/precision", "Precision details"),
        Passage(
            "mlx.core.quantized_matmul", "https://docs/qmm", "Quantized multiplication"
        ),
    ]
    ranked = apply_ranking_policy(
        "How do I use quantizied matmul?", passages, np.array([-3.0, 3.0])
    )
    assert ranked[0] == passages[1]


def test_requested_api_prefers_relevant_guide_over_competing_operation() -> None:
    passages = [
        Passage("mlx.core.distributed.all_sum", "https://docs/sum", "All reduce sum"),
        Passage(
            "mlx.core.distributed.sum_scatter", "https://docs/scatter", "Sum and shard"
        ),
        Passage(
            "Distributed Communication",
            "https://docs/distributed",
            "Use mlx.core.distributed.all_sum to sum arrays across processes.",
        ),
    ]
    scores = np.array([6.0, 5.0, 3.0])
    ranked = apply_ranking_policy(
        "How do I perform an all-reduce sum?", passages, scores
    )
    assert ranked[:2] == [passages[0], passages[2]]
    comparison = "mlx.core.distributed.all_sum versus mlx.core.distributed.sum_scatter"
    assert apply_ranking_policy(comparison, passages, scores)[:2] == passages[:2]


def test_api_catalog_does_not_displace_answer_bearing_operation() -> None:
    passages = [
        Passage("mlx.nn.Conv3d", "https://docs/python/layer", "Convolution layer."),
        Passage(
            "mlx.core.conv3d",
            "https://docs/python/op",
            "Input shape (N, D, H, W, C_in); weight shape (C_out, KD, KH, KW, C_in).",
        ),
        Passage(
            "Neural Networks / Layers",
            "https://docs/python/layers",
            "# Layers\n\n- [Conv3d](https://docs/python/layer#mlx.nn.Conv3d)",
        ),
    ]
    ranked = apply_ranking_policy(
        "mlx.nn.Conv3d input shape and weight shape",
        passages,
        np.array([6.0, 5.0, 3.0]),
    )
    assert ranked[:2] == passages[:2]


@pytest.mark.parametrize("guide_score", [3.0, -3.0])
def test_command_help_keeps_relevant_source_explanations(guide_score: float) -> None:
    passages = [
        Passage(
            "mlx_lm.compress command-line options",
            "https://docs/help#compress",
            "usage: mlx_lm.compress [--bits BITS]",
            evidence_kind=EvidenceKind.CLI,
        ),
        Passage("Other", "https://docs/other", "Unrelated model options."),
        Passage(
            "Compression / Usage",
            "https://docs/compression#usage",
            "Use `mlx_lm.compress` to compress a model before deployment.",
        ),
    ]
    ranked = apply_ranking_policy(
        "How do I use mlx_lm.compress?",
        passages,
        np.array([6.0, 5.0, guide_score]),
    )
    assert ranked[:2] == [
        passages[0],
        passages[2] if guide_score > 0 else passages[1],
    ]


def test_snake_case_argument_leaves_command_help_to_its_reranker_score() -> None:
    word, option = "batch_size", "--batch-size"
    load = Passage(
        "mlx_lm.load",
        "https://docs/api#load",
        f"def load(path_or_hf_repo, {word}=None)",
        package=Package.LM,
        evidence_kind=EvidenceKind.API,
    )
    unrelated = Passage("Other", "https://docs/other", "Unrelated model notes.")
    help_text = Passage(
        "mlx_lm.fuse command-line options",
        "https://docs/help#fuse",
        f"  {option} VALUE\n      Sets {word}.",
        evidence_kind=EvidenceKind.CLI,
    )
    passages = [load, unrelated, help_text]
    scores = np.array([6.0, 2.0, -3.0])

    # The word names a Python argument; the reranker's order stands.
    assert apply_ranking_policy(f"{word} with load()", passages, scores) == passages
    # Spelled as the command option, the help is the definition and leads.
    ranked = apply_ranking_policy(f"what does {option} do?", passages, scores)
    assert ranked[0] == help_text


def test_stream_wait_retrieves_synchronization(tmp_path: Path) -> None:
    passages = [
        Passage("Primitive", "https://docs/dev/primitive", "GPU stream computations"),
        Passage(
            "mlx.core.synchronize",
            "https://docs/python/sync",
            "Synchronize with the stream",
        ),
    ]
    index = service_on_disk(tmp_path, passages)
    assert index.retrieve("How do I wait for a GPU stream to finish?")[0] == passages[1]


@pytest.mark.parametrize(
    "query", ["Cast an array to int32", "Change the dtype of an array"]
)
def test_value_conversion_retrieves_array_astype(tmp_path: Path, query: str) -> None:
    passages = [
        Passage("NumPy conversion", "https://docs/numpy", "Convert arrays"),
        Passage(
            "mlx.core.array.astype", "https://docs/python/astype", "Cast the array"
        ),
    ]
    index = service_on_disk(tmp_path, passages)
    assert index.retrieve(query)[0] == passages[1]


def test_dtype_intent_recovers_the_data_types_guide(tmp_path: Path) -> None:
    passages = [
        Passage("GPU implementation", f"https://docs/dev/{i}", "GPU CPU arrays")
        for i in range(20)
    ]
    guide = Passage(
        "Data Types",
        "https://docs/python/data_types#data-types",
        "The default floating point type is float32. float64 only works on CPU.",
    )
    index = service_on_disk(tmp_path, [*passages, guide])
    assert index.retrieve("Does MLX support float64 on GPU?")[0] == guide


def test_python_search_keeps_usage_guides_and_excludes_cpp_api(tmp_path: Path) -> None:
    passages = [
        Passage(
            "Conversion to NumPy",
            "https://docs/usage/numpy.html#conversion",
            "import mlx.core as mx\nnp.array(a) evaluates the MLX array.",
        ),
        Passage(
            "Lazy Evaluation",
            "https://docs/usage/lazy_evaluation.html#when-to-evaluate",
            "Evaluation placement trades graph size against overhead.",
        ),
        Passage("C++ eval", "https://docs/cpp/eval.html", "Evaluate an array."),
    ]
    index = service_on_disk(tmp_path, passages)
    query = "mx.eval lazy graph inference synchronization"
    assert set(index.retrieve(query, "python")) == set(passages[:2])


@pytest.mark.parametrize(
    ("code", "python", "cpp"),
    [
        ("```python\nmx.eval(x)\n```", True, False),
        ("```cpp\nmlx::core::eval(x);\n```", False, True),
        ("```python\nmx.eval(x)\n```\n```cpp\nmlx::core::eval(x);\n```", True, True),
        ("Evaluation computes a graph.", True, True),
        ("```\ndef fun(x):\n    return x + 1\n```", True, False),
        ("```\nstd::vector<array> inputs;\n```", False, True),
    ],
)
def test_guide_language_uses_content(code: str, python: bool, cpp: bool) -> None:
    passage = Passage("Guide", "https://docs/dev/guide.html", code)
    assert matches_language(passage, "python") == python
    assert matches_language(passage, "cpp") == cpp
    assert matches_language(passage, "all")


def test_exact_eval_definition_beats_async_eval_mentions(tmp_path: Path) -> None:
    passages = [
        Passage(
            "mlx.core.async_eval",
            "https://docs/python/async_eval.html#async-eval",
            "Use mx.eval for synchronous evaluation.",
        ),
        Passage(
            "mlx.nn.Module.eval",
            "https://docs/python/module_eval.html#module-eval",
            "Evaluation mode",
        ),
        Passage(
            "mlx.core.eval", "https://docs/python/eval.html#eval", "Evaluate arrays."
        ),
    ]
    index = service_on_disk(tmp_path, passages)
    result = index.retrieve("mx.eval lazy graph inference synchronization", "python")
    assert result[0] == passages[2]


def test_chunked_exact_api_definition_beats_related_layer(tmp_path: Path) -> None:
    query = "Does mx.conv3d support grouped convolution on the GPU?"
    url = "https://docs/python/conv3d.html#mlx-core-conv3d"
    section = "# mlx.core.conv3d\n\nconv3d(input, weight, groups=1)\n\nOnly groups=1 is supported."
    passages = [
        Passage(
            "mlx.nn.Conv3d",
            "https://docs/python/nn/conv3d.html",
            "GPU convolution layer",
        ),
        Passage("mlx.core.conv3d", url, "# mlx.core.conv3d", section),
        Passage("mlx.core.conv3d", url, "conv3d(input, weight, groups=1)", section),
        Passage("mlx.core.conv3d", url, "Only groups=1 is supported.", section),
    ]
    index = service_on_disk(tmp_path, passages)
    index.score_candidates = lambda query, passages: np.array(
        [float("Only groups" in p.text) for p in passages],
    )
    assert index.retrieve(query, "python")[0].url == url
    assert "Only groups=1 is supported." in asyncio.run(
        index.search(query, 2, "python")
    )


def test_web_operator_is_rejected_before_loading_models() -> None:
    index = service_in_memory([])
    with pytest.raises(ValueError, match="Remove the operator"):
        index.retrieve("site:ml-explore.github.io/mlx array.strides Python")


def test_named_guide_is_preserved_when_training_examples_share_query_terms(
    tmp_path: Path,
) -> None:
    passages = [
        Passage(
            "Multilayer Perceptron",
            "https://docs/mlp",
            "Train a model with gradient descent.",
        ),
        Passage(
            "Linear Regression",
            "https://docs/linear",
            "Fit coefficients to the design matrix.",
        ),
        Passage(
            "MLX", "https://docs/index", "MLX provides machine learning primitives."
        ),
        Passage(
            "Linear Regression",
            "https://docs/linear",
            "More linear regression gradient descent examples.",
        ),
    ]
    index = service_on_disk(tmp_path, passages)
    query = "Train a linear regression model with gradient descent in MLX"
    assert index.retrieve(query)[0] == passages[1]


def test_incidental_guide_title_does_not_override_relevant_results(
    tmp_path: Path,
) -> None:
    passages = [
        Passage(
            f"Conversion Example {i}",
            f"https://docs/example/{i}",
            "numpy conversion ownership lazy evaluation " * 50,
        )
        for i in range(RERANK_CANDIDATE_LIMIT + 4)
    ]
    guide = Passage(
        "Lazy Evaluation", "https://docs/lazy", "Arrays represent computation graphs."
    )
    index = service_on_disk(tmp_path, [*passages, guide])
    result = index.retrieve("NumPy conversion ownership lazy evaluation")
    assert result[0] == passages[0]
    assert len(result) == RERANK_CANDIDATE_LIMIT


def test_compound_candidates_reserve_guides_beside_each_requested_api(
    tmp_path: Path,
) -> None:
    passages = [
        Passage(f"Irrelevant {i}", f"https://docs/python/noise/{i}", "Arrays")
        for i in range(20)
    ]
    requested = [
        Passage(
            "Indexing Arrays", "https://docs/usage/indexing", "Slice mx.array examples"
        ),
        Passage(
            "In Place Updates",
            "https://docs/usage/updates",
            "Indexed mx.array assignment",
        ),
        Passage("mlx.core.array.astype", "https://docs/python/cast", "Cast dtype"),
        Passage("mlx.core.eval", "https://docs/python/eval", "Evaluate arrays"),
    ]
    index = service_on_disk(tmp_path, passages + requested)
    candidates = index.retrieve(
        "Slice an array, update indexed entries, convert dtype, force computation"
    )
    assert all(p in candidates for p in requested)
    assert len(candidates) <= RERANK_CANDIDATE_LIMIT
    assert not index.retrieve("Slice an array and convert dtype", language="cpp")


def test_inferred_apis_keep_the_requested_guide_explanation() -> None:
    passages = [
        Passage("mlx.core.synchronize", "sync", "Wait"),
        Passage("mlx.core.eval", "eval", "Evaluate"),
        Passage("Why Lazy Evaluation", "lazy", "Record a graph"),
    ]
    ranked = apply_ranking_policy(
        "Why is timing my GPU operation instant?", passages, np.array([3.0, 2.0, 1.0])
    )
    assert ranked[:2] == [passages[0], passages[2]]
