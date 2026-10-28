"""Query expansion and model input text; indexed passages are never modified."""

import re
from dataclasses import dataclass

from mlx_mcp.packages import Package
from mlx_mcp.passages import ranking_text

# Source-verified descriptions for model input; citations retain original text.
_TITLE_SEARCH_TERMS = {
    "mlx.core.concatenate": "Join or combine arrays along an existing axis or dimension.",
    "mlx.core.stack": "Join arrays along a new axis or dimension.",
    "mlx.core.array.astype": "Cast or convert an array to a different element data type (dtype).",
    "mlx.core.array.view": "Reinterpret array bytes as another dtype without converting the values.",
    "mlx.core.checkpoint": "Recompute intermediate activations during the backward pass instead of storing them, trading computation for lower gradient memory use.",
    "mlx.core.grad": "Compute derivatives for selected positional parameter arrays and keyword arguments using argnums and argnames.",
    "mlx.core.set_memory_limit": "Set an upper memory-use guideline during graph evaluation, training or inference; this is not a strict allocation cap.",
    "mlx.core.reset_peak_memory": "Start a new measurement window for the peak MLX allocation counter by resetting it to zero; read the peak in bytes with get_peak_memory.",
    "mlx.nn.ALiBi": "Bias attention scores by the distance between query and key tokens, using static non-learnable linear biases for input length extrapolation to longer sequences.",
    "Indexing Arrays / Differences from NumPy": "Array indexing past the end has undefined behavior rather than checked bounds errors or a propagated GPU exception. Boolean mask reading is unsupported; assignment is supported.",
    "Custom Metal Kernels / Simple Example": "Build a custom Metal kernel once and reuse it across forward calls; repeated creation creates a new Metal library and can trigger JIT compilation overhead.",
    "Custom Extensions in MLX / Operations and Primitives": "Distinguish the user-facing operation that builds a computation graph from the primitive that evaluates CPU or GPU computation and transforms the graph.",
    "Buffers, Streams and Samples / Buffers": "Choose a finite dataset container with a known length, indexing, random access and shuffling.",
    "Buffers, Streams and Samples / Streams": "Choose a potentially infinite iterable of samples consumed sequentially when the length is unknown or random access is unavailable.",
    "mlx_lm.sample_utils.make_sampler": "Control text generation randomness with temperature, nucleus top-p, min-p, top-k and XTC sampling.",
    "mlx_lm.sample_utils.make_repetition_penalty": "Discourage repeated tokens in generated text with a repetition penalty and context size.",
    "mlx_lm.models.cache.trim_prompt_cache": "Roll back part of a prompt cache in place by trimming tokens; return the number trimmed.",
    "mlx_lm.batch_generate": "Generate text for multiple prompts together as a batch instead of looping over prompts sequentially. Supply tokenized prompts as a list of lists of integer token IDs.",
    "MLX LM / Python API": "Load a language model, format chat messages into a prompt with tokenizer.apply_chat_template and add_generation_prompt, generate text, and convert model weights.",
    "MLX LM / Python API / Streaming": 'Print a language model\'s generated output token by token as it arrives instead of waiting for the whole response; iterate over stream_generate and print response.text with end="" and flush=True.',
    "MLX LM / Long Prompts and Generations": "Save and reuse a prompt prefix with mlx_lm.cache_prompt and mlx_lm.generate --prompt-cache-file; rotating key-value cache size.",
    "Fine-Tuning with LoRA or QLoRA / Run / Fine-tune": "Fine-tune quantized model weights with QLoRA or ordinary weights with LoRA; full fine-tuning, training data and adapter paths.",
    "Fine-Tuning with LoRA or QLoRA / Fuse": "Merge trained low-rank adapter weights into the base model to save one standalone fused model with mlx_lm.fuse; optionally upload or export it.",
    "Learned Quantization / DWQ": "Run mlx_lm.dwq to shrink a model to low 2-bit, 3-bit or 4-bit precision while reducing quality loss through distillation; bits, sample count and batch size control preparation work and memory.",
}
# The indexed MLX docs cover both free functions and array methods.
_TITLE_SEARCH_TERMS.update(
    {
        f"mlx.core.{name}": _TITLE_SEARCH_TERMS[f"mlx.core.array.{name}"]
        for name in ("astype", "view")
    }
)


@dataclass(frozen=True)
class VersionRequest:
    """A documentation version that a query asks for, in one package."""

    package: Package
    version: str


@dataclass(frozen=True)
class QueryPlan:
    """A query and the variants retrieval derives from it.

    explicit rewrites calls such as model.eval( to their documented owner.
    expanded appends inferred search terms, such as API names and guide titles;
    hints holds just those terms.
    requested_versions lists the documentation versions the query names.
    """

    original: str
    explicit: str
    expanded: str
    hints: str
    requested_versions: tuple[VersionRequest, ...]


def plan_query(query: str) -> QueryPlan:
    """Derive the explicit, expanded, and version-aware forms of a query."""
    expanded = expand_query(query)
    # These are explicit instance-method calls, with the documented class owner.
    explicit = re.sub(r"\b(?:model|module)\.eval(?=\s*\()", "mlx.nn.Module.eval", query)
    versions = tuple(
        dict.fromkeys(
            VersionRequest(
                Package(
                    "mlx"
                    + suffix.lower()
                    .replace(" ", "-")
                    .replace(".", "-")
                    .replace("_", "-")
                ),
                version,
            )
            for suffix, version in re.findall(
                r"\bMLX([ ._-](?:data|lm))?\s+(?:(?:version|v)\s*)?"
                r"(\d+\.\d+(?:\.\d+)?(?:[a-zA-Z][\w.-]*)?)\b",
                query,
                re.IGNORECASE,
            )
        )
    )
    return QueryPlan(
        query, explicit, expanded, expanded[len(query) :].strip(), versions
    )


def expand_query(query: str) -> str:
    """Append API names and guide titles for unambiguous operation wording."""
    lower = query.lower()
    query += _operation_query_terms(lower)
    query += _language_model_query_terms(lower)
    query += _runtime_query_terms(lower)
    boolean_selection = re.search(r"\bbool(?:ean)?[ _-]+mask", lower) and re.search(
        r"\b(select\w*|extract\w*|filter\w*|gather\w*|read\w*|"
        r"variable number)\b",
        lower,
    )
    indexing = re.search(r"\b(?:indexing|slic(?:e|ing))\b", lower) and not re.search(
        r"\b(?:mx|mlx\.core)\.slice(?:_update)?\b|"
        r"\b(?:dynamic|start_indices|slice_size)\b",
        lower,
    )
    if indexing and not boolean_selection:
        query += " Indexing Arrays"
    if re.search(r"\b(?:indexed|index|indices)\b", lower) and re.search(
        r"\bupdat(?:e|es|ing)\b", lower
    ):
        query += " In Place Updates"
    if re.search(r"\bforce\s+(?:the\s+)?comput(?:ation|ing)\b", lower):
        query += " mlx.core.eval"
    if indexing and "lazy evaluation" in lower:
        query += " Why Lazy Evaluation"
    if (
        "lazy" in lower
        and re.search(r"\barrays?\b", lower)
        and re.search(r"\b(?:computed|compute|evaluated)\b", lower)
    ):
        query += " Why Lazy Evaluation mlx.core.eval"
    if (
        re.search(r"\bvectoriz(?:e|ing|ation)\b", lower)
        and re.search(r"\b(?:functions?|inputs?|batch\w*)\b", lower)
        and not re.search(r"\b(?:numpy|pytorch|torch|tensorflow|jax)\b", lower)
    ):
        query += " mlx.core.vmap"
    if re.search(r"\bno_grad\b", lower):
        query += " Automatic Differentiation"
    if re.search(r"\ball[-_ ]?reduce\b", lower) and re.search(r"\bsum\b", lower):
        query += " mlx.core.distributed.all_sum"
    if "cosine" in lower and re.search(
        r"\b(?:schedul\w*|learning[ -]rate|anneal\w*|decay)\b", lower
    ):
        query += " mlx.optimizers.cosine_decay"
    dtype = re.search(
        r"\b(?:dtype|data type|precision|(?:bfloat|float|u?int|complex)\d+)\b", lower
    )
    if dtype and re.search(r"\b(?:default|support\w*|gpu|cpu)\b", lower):
        query += " Data Types"
        if "default" in lower:
            query += " default floating point integer types"
    casting = re.search(r"\bcast(?:ing)?\b", lower) and (
        dtype or re.search(r"\b(?:mlx|arrays?)\b|\bmx\.", lower)
    )
    if (
        casting
        or (dtype and re.search(r"\b(?:convert(?:ing)?|chang(?:e|ing))\b", lower))
    ) and not re.search(
        r"\b(?:reinterpret\w*|numpy|pytorch|torch|tensorflow)\b|"
        r"\b(?:without|not)\b[^.?!]*\b(?:cast|convert)",
        lower,
    ):
        query += " mlx.core.array.astype"
    timing = re.search(
        r"\b(benchmark\w*|timing|elapsed|latency|instantly)\b|\bhow long\b|"
        r"\b(?:measure|time)\s+(?:(?:my|the|these)\s+)?(?:gpu|cpu|mlx|array|kernel|code|comput|operation)|"
        r"\b(?:take|takes|taking)\b.{0,30}\b(?:time|seconds?|milliseconds?)\b",
        lower,
    ) and re.search(
        r"\b(gpu|cpu|mlx|computations?|operations?|arrays?|kernels?)\b", lower
    )
    waiting = re.search(
        r"\b(wait(?:ing)?|synchroni[sz](?:e|ation))\b|"
        r"\bblock(?:ing)?\b.{0,30}\b(?:until|for completion)\b",
        lower,
    ) and re.search(r"\b(gpu|streams?|device)\b", lower)
    if timing:
        query += " Why Lazy Evaluation mlx.core.synchronize mlx.core.eval"
    elif waiting:
        query += " mlx.core.synchronize"
    if re.search(r"\b(?:model|module)\.eval\s*\(", lower) or (
        "dropout" in lower
        and re.search(r"\b(inference|evaluation|disable|off)\b", lower)
    ):
        query += " mlx.nn.Module.eval"
    if boolean_selection:
        query += " Differences from NumPy boolean mask selection versus assignment data dependent output shapes"
    if re.search(r"\bstack\b", lower) and re.search(r"\bconcatenate\b", lower):
        return query
    if (
        re.search(r"\barrays?\b", lower)
        and re.search(r"\b(join|combine|stack|concatenate)\b", lower)
        and not re.search(r"\b(?:mlx|mx)\.", lower)
    ):
        shapes = list(re.finditer(r"\(\s*\d+\s*,[\d,\s]*\)", lower))
        ranks = [len(re.findall(r"\d+", shape.group())) for shape in shapes]
        new_axis = bool(re.search(r"\bnew (?:batch )?(?:axis|dimension)\b", lower))
        existing_axis = bool(
            re.search(
                r"\bexisting (?:axis|dimension)\b|"
                r"\b(?:without|not)\b[^.?!]*\bnew (?:batch )?(?:axis|dimension)\b",
                lower,
            )
        )
        if existing_axis:
            new_axis = False
        if len(ranks) >= 2 and re.search(
            r"\b(?:into|to|output|result)\b|->",
            lower[shapes[-2].end() : shapes[-1].start()],
        ):
            new_axis = ranks[-1] == ranks[0] + 1
            existing_axis = ranks[-1] == ranks[0]
        if new_axis or (not existing_axis and re.search(r"\bstack\b", lower)):
            return query + " mlx.core.stack"
        if existing_axis or re.search(r"\b(join|concatenate)\b", lower):
            return query + " mlx.core.concatenate"
    return query


def _language_model_query_terms(lower: str) -> str:
    """Resolve generation and training intents to pinned LM APIs and options."""
    if re.search(r"\b(?:torch|pytorch|jax|tensorflow)\b", lower):
        return ""
    terms: list[str] = []
    generation = re.search(
        r"\b(?:generat\w*|llm|language model|tokens?|phrases?)\b", lower
    )
    if generation and re.search(r"\b(?:repeat\w*|repetition)\b", lower):
        terms.append("mlx_lm.sample_utils.make_repetition_penalty")
    if re.search(r"\bxtc\b", lower):
        if re.search(r"\b(?:command|cli|terminal|flags?)\b", lower):
            terms.extend(("--xtc-probability", "--xtc-threshold"))
        else:
            terms.append("mlx_lm.sample_utils.make_sampler")
    if "chat template" in lower and "server" in lower:
        terms.extend(("--chat-template", "--use-default-chat-template"))
    training = re.search(r"\b(?:train\w*|fine[- ]?tun\w*|loss|learn)\b", lower)
    assistant_only = "assistant" in lower and re.search(
        r"\b(?:only|instead|exclude)\b", lower
    )
    if training and (assistant_only or ("prompt" in lower and "mask" in lower)):
        terms.append("--mask-prompt")
    if "model" in lower and re.search(
        r"\b(?:low[- ]?bit|bit[- ]?width|[23] bits?)\b", lower
    ):
        terms.append("mlx_lm.dwq")
    return " " + " ".join(terms) if terms else ""


def _runtime_query_terms(lower: str) -> str:
    """Name contracts for allocator counters, shared streams and compiled state."""
    terms: list[str] = []
    if "memory" in lower and re.search(
        r"\b(?:current|active|peak|cache|cached)\b", lower
    ):
        terms.append("Memory Management")
    if "thread" in lower and re.search(
        r"\b(?:graph|evaluation|evaluate|evaluating)\b", lower
    ):
        terms.append("mlx.core.new_thread_unsafe_stream")
    if (
        re.search(r"\bcompil(?:e|ed|ation|ing)\b", lower)
        and re.search(
            r"\b(?:captur\w*|closures?|closed[ -]over|side[ -]effects?|"
            r"mutable[ -]state|implicit[ -](?:inputs?|outputs?)|globals?|nonlocal)\b",
            lower,
        )
        and not re.search(r"\b(?:torch|pytorch|tensorflow|jax)\b", lower)
    ):
        # The API signature names inputs/outputs, but the guide demonstrates
        # why closed-over arrays stay constant and how state must be passed.
        terms.extend(("Pure Functions", "mlx.core.compile"))
    return " " + " ".join(terms) if terms else ""


def _convolution_query_terms(lower: str) -> list[str]:
    """Recall convolution definitions without inferring a layout or performance claim.

    Layout wording also appears in dropout and overview tables. Preserve the
    requested spatial dimension, or recall all three when none is specified.
    Only a transposed-convolution phrase selects that operator: transposed
    input arrays do not imply a transposed convolution.
    """
    if (
        not re.search(
            r"\b(?:convolutions?|deconvolutions?|conv(?:transpose)?[123]d)\b", lower
        )
        or re.search(r"\b(?:mx|mlx|nn)\.[\w.]+", lower)
        or re.search(r"\b(?:torch|pytorch|tensorflow|jax)\b", lower)
        or re.search(
            r"\b(?:not|never|without|don't|do not)\s+(?:(?:use|apply|run)\s+)?"
            r"(?:[123][ -]?d\s+)?(?:transpos(?:e|ed)\s+)?"
            r"(?:[123][ -]?d\s+)?(?:de)?convolutions?\b",
            lower,
        )
    ):
        return []
    dimensions = list(
        dict.fromkeys(
            re.findall(r"\b([123])[ -]?d\b", lower)
            + re.findall(r"\bconv(?:transpose)?([123])d\b", lower)
        )
    ) or ["1", "2", "3"]
    transposed = list(
        re.finditer(
            r"\btranspos(?:e|ed)\s+(?:[123][ -]?d\s+)?convolutions?\b|"
            r"\bdeconvolutions?\b|\bconvtranspose[123]d\b",
            lower,
        )
    )
    # A comparison can name both families. Keep ordinary mentions whose spans
    # are outside a transposed-convolution phrase rather than replacing them.
    ordinary = any(
        not any(span.start() <= match.start() < span.end() for span in transposed)
        for match in re.finditer(r"\b(?:convolutions?|conv[123]d)\b", lower)
    )
    families = ([("conv", "Conv")] if ordinary else []) + (
        [("conv_transpose", "ConvTranspose")] if transposed else []
    )
    return [
        name
        for core, layer in families
        for dimension in dimensions
        for name in (f"mlx.core.{core}{dimension}d", f"mlx.nn.{layer}{dimension}d")
    ]


def _operation_query_terms(lower: str) -> str:
    """Return API and guide terms for array, device, and convolution intent."""
    terms: list[str] = []
    gpu = re.search(r"\b(?:gpu|metal|cuda)\b", lower)
    selection = re.search(
        r"\b(?:(?:require|force|select|choose|switch)(?:\s+to)?\s+"
        r"(?:(?:the|a|an)\s+)?(?:gpu|metal|cuda|device|backend)\b|"
        r"default (?:device|gpu|backend)\b)",
        lower,
    )
    negated = re.search(
        r"\b(?:not|never|without|don't|do not)\s+"
        r"(?:(?:need|want)\s+to\s+)?(?:require|force|select|choose|switch)\w*\b",
        lower,
    )
    if gpu and selection and not negated:
        terms.append("mlx.core.set_default_device")
    availability = re.search(
        r"\b(?:available|availability)\b|"
        r"\bdetect (?:the |a )?(?:gpu|metal|cuda)\b|"
        r"\bcheck\b[^.?!]*\bbackend\b",
        lower,
    )
    if gpu and availability:
        backends = [
            name for name in ("metal", "cuda") if re.search(rf"\b{name}\b", lower)
        ]
        terms.extend(
            f"mlx.core.{name}.is_available" for name in backends or ["metal", "cuda"]
        )
    numpy = re.search(r"\b(?:numpy|np)\b", lower)
    ownership = re.search(
        r"\b(?:shar\w*|cop(?:y|ies|ying)|view|ownership|asarray|convert\w*|conversion)\b",
        lower,
    )
    if (
        numpy
        and ownership
        and not re.search(r"\b(?:torch|pytorch|tensorflow|jax)\b", lower)
    ):
        terms.append("Conversion to NumPy and Other Frameworks")
        if re.search(r"\bevaluat\w*\b", lower):
            terms.append("When to Evaluate")
    reduction = re.search(
        r"\b(?:reduc\w*|accumulat\w*|dtype|precision|axis|axes)\b", lower
    )
    explicit = re.search(r"\b(?:mx|mlx\.core)\.[\w.]+", lower)
    if (
        reduction
        and not explicit
        and not re.search(r"\b(?:distributed|all[-_ ]?reduce|cumulative)\b", lower)
    ):
        for name in ("sum", "mean", "prod", "min", "max", "var", "std"):
            if re.search(rf"\b{name}\b", lower):
                terms.append(f"mlx.core.{name}")
    if (
        re.search(r"\bbroadcast(?:ing)?\b", lower)
        and re.search(
            r"\b(?:arrays?|shapes?|dimensions?|elementwise|arithmetic)\b", lower
        )
        and not explicit
        and not re.search(r"\b(?:matmul|matrix|distributed|ranks?|processes)\b", lower)
    ):
        terms.append("mlx.core.broadcast_arrays")
    terms.extend(_convolution_query_terms(lower))
    return " " + " ".join(terms) if terms else ""


def retrieval_input(title: str, markdown: str) -> str:
    """Return a passage's model input: title, API search terms, and ranking text."""
    fields = [title]
    if title in _TITLE_SEARCH_TERMS:
        fields.append(_TITLE_SEARCH_TERMS[title])
    fields.append(ranking_text(markdown))
    return "\n".join(fields)
