"""Query expansion, query plans, and requested documentation versions."""

from dataclasses import asdict

import pytest

from mlx_mcp.packages import Package
from mlx_mcp.passages import Passage
from mlx_mcp.responses import render_sections
from mlx_mcp.search_text import expand_query, plan_query, retrieval_input


def test_plain_language_generation_intent_has_a_core_array_counterexample() -> None:
    question = "Can I discourage repeated phrases in generated text?"
    assert "mlx_lm.sample_utils.make_repetition_penalty" in expand_query(question)
    array_question = "Repeat an array along a new axis"
    assert "mlx_lm" not in expand_query(array_question)
    torch_question = "PyTorch text generation repeats words"
    assert "mlx_lm" not in expand_query(torch_question)


@pytest.mark.parametrize("query", ["MLX LM v0.31.3", "mlx_lm version 0.31.3"])
def test_lm_version_request_keeps_its_package_namespace(query: str) -> None:
    (request,) = plan_query(query).requested_versions
    assert request.package is Package.LM and request.version == "0.31.3"


@pytest.mark.parametrize(
    ("query", "unwanted"),
    [
        ("CUDA graph cache tuning", "mlx.core.set_default_device"),
        ("PyTorch to numpy conversion", "Conversion to NumPy and Other Frameworks"),
        ("Distributed all-reduce sum", "mlx.core.sum"),
        ("Broadcast an array between processes", "mlx.core.broadcast_arrays"),
    ],
)
def test_operation_expansion_does_not_replace_other_intent(
    query: str, unwanted: str
) -> None:
    assert unwanted not in expand_query(query)


def test_search_vocabulary_does_not_change_canonical_evidence() -> None:

    passage = Passage(
        "mlx.core.concatenate",
        "https://docs/concatenate",
        "Concatenate the arrays along the given axis.",
    )
    original = asdict(passage)
    assert "existing axis" in retrieval_input(passage.title, passage.text)
    assert "new axis" in retrieval_input(
        "mlx.core.stack", "Stack arrays along a new axis."
    )
    assert "Join or combine" not in render_sections([passage], 1)
    assert asdict(passage) == original


@pytest.mark.parametrize(
    "query",
    ["Do not select the GPU; select rows with mx.take"],
)
def test_negated_and_mixed_operations_do_not_infer_device_changes(query: str) -> None:
    assert "set_default_device" not in plan_query(query).hints


@pytest.mark.parametrize(
    "query",
    [
        "How do I reshape an array?",
        "mlx.core.stack arrays",
        "Combine arrays using matrix multiplication",
    ],
)
def test_array_join_expansion_preserves_other_operations_and_explicit_apis(
    query: str,
) -> None:

    assert expand_query(query) == query


@pytest.mark.parametrize(
    "query",
    [
        "Convert an MLX array to NumPy",
        "Reinterpret float32 bytes as float16 without converting values",
        "Do not cast MLX arrays",
    ],
)
def test_dtype_conversion_does_not_replace_framework_conversion_or_reinterpretation(
    query: str,
) -> None:

    assert "astype" not in expand_query(query)


@pytest.mark.parametrize(
    ("query", "expected", "unwanted"),
    [
        (
            "3D convolution input layout channels last NDHWC",
            ("mlx.core.conv3d", "mlx.nn.Conv3d"),
            ("conv1d", "conv2d", "conv_transpose"),
        ),
        (
            "2-D transposed convolution input layout",
            ("mlx.core.conv_transpose2d", "mlx.nn.ConvTranspose2d"),
            ("mlx.core.conv2d", "conv_transpose3d"),
        ),
        (
            "Convolution performance channels-last transposed arrays contiguous",
            ("mlx.core.conv1d", "mlx.core.conv2d", "mlx.core.conv3d"),
            ("conv_transpose", "ConvTranspose"),
        ),
        (
            "Compare 3D convolution with transposed 3D convolution",
            ("mlx.core.conv3d", "mlx.core.conv_transpose3d"),
            ("conv1d", "conv2d"),
        ),
    ],
)
def test_convolution_intent_keeps_dimension_and_transpose_scope(
    query: str, expected: tuple[str, ...], unwanted: tuple[str, ...]
) -> None:
    hints = plan_query(query).hints
    assert all(name in hints for name in expected)
    assert not any(name in hints for name in unwanted)


@pytest.mark.parametrize(
    "query",
    [
        "PyTorch 3D convolution input layout",
        "Do not use 3D convolution",
        "Never use transposed 3D convolution",
        "Don't apply 3D deconvolution",
        "mx.conv3d",
        "Does nn.ConvTranspose3d expect channels last?",
        "3D dropout input layout NDHWC",
    ],
)
def test_convolution_inference_preserves_other_frameworks_and_explicit_apis(
    query: str,
) -> None:
    assert "conv" not in plan_query(query).hints.lower()


@pytest.mark.parametrize(
    "query",
    [
        "Why does changing an array captured by mx.compile not change the result?",
        "Compilation constant folding closed-over arrays and dynamic inputs",
        "How do I pass mutable state captured by a compiled function?",
        "Can a compiled function have side effects?",
    ],
)
def test_compiled_state_questions_recall_the_pure_function_contract(query: str) -> None:
    assert "Pure Functions" in plan_query(query).hints
    assert "mlx.core.compile" in plan_query(query).hints


@pytest.mark.parametrize(
    "query",
    [
        "PyTorch compiled function captured arrays",
        "How do I capture a Metal frame?",
        "How do I compile a function with variable input shapes?",
    ],
)
def test_compiled_state_inference_preserves_other_capture_and_compilation_intents(
    query: str,
) -> None:
    assert "Pure Functions" not in plan_query(query).hints
