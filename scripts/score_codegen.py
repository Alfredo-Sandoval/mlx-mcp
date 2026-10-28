"""Paired MLX coding experiment: task alone versus task plus retrieved docs.

Each trial uses a fresh tool-disabled Codex context. Only the task and, for the
treatment, bounded production search/fetch text enter the prompt. Hidden checks
run the returned module once on the MLX GPU in a write/network-restricted macOS
subprocess. No repair turns or selection of best answers are performed.
"""

import argparse
import ast
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from mlx_mcp.indexing import select_index_directory
from mlx_mcp.model_runtime import ModelAccess
from mlx_mcp.packages import Package
from mlx_mcp.responses import source_blocks
from mlx_mcp.service import DocumentationService
from scripts.code_tasks import TASKS, CodingTask

type Condition = Literal["without_docs", "with_docs"]
CONTEXT_CHARACTERS = 16000
CHECKS_COMPLETED = "MLX_MCP_CHECKS_COMPLETED"
DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "apps",
    "plugins",
    "remote_plugin",
    "memories",
    "multi_agent",
    "multi_agent_v2",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "computer_use",
    "in_app_browser",
    "view_image",
    "image_generation",
    "hooks",
    "skill_search",
    "sleep_tool",
    "goals",
)
REQUIRED_APIS = {
    "grad-with-aux": (("grad", "value_and_grad"),),
    "compiled-state": (("compile",),),
    "conv-layout": (("conv2d",),),
    "per-example-grad": (("vmap",), ("grad",)),
    "metal-strides": (("fast.metal_kernel",),),
}


class CodeOutput(BaseModel):
    """The sole model output; explanations cannot masquerade as Python code."""

    model_config = ConfigDict(extra="forbid", strict=True)
    code: str


@dataclass(frozen=True)
class Execution:
    status: str
    stdout: str
    stderr: str


def checker_source(task: CodingTask, candidate: Path) -> str:
    """Build the private execution driver; only its outcome reaches the report."""
    required = REQUIRED_APIS.get(task.id, ())
    return f"""
import functools
import runpy
import mlx.core as mx
import numpy as np
mx.set_default_device(mx.gpu)
calls = {{}}
observing_candidate = True
def observe(path):
    owner = mx
    parts = path.split('.')
    for part in parts[:-1]:
        owner = getattr(owner, part)
    original = getattr(owner, parts[-1])
    @functools.wraps(original)
    def call(*args, **kwargs):
        if observing_candidate:
            calls.setdefault(path, []).append(kwargs)
        return original(*args, **kwargs)
    setattr(owner, parts[-1], call)
for alternatives in {required!r}:
    for path in alternatives:
        observe(path)
def close(actual, expected):
    assert isinstance(actual, mx.array), 'result must be an MLX array'
    assert actual.shape == np.asarray(expected).shape, 'wrong result shape'
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)
candidate_solve = runpy.run_path({str(candidate)!r})['solve']
observing_candidate = False
def solve(*args):
    global observing_candidate
    observing_candidate = True
    try:
        return candidate_solve(*args)
    finally:
        observing_candidate = False
{task.checks}
assert all(any(path in calls for path in alternatives) for alternatives in {required!r}), 'required MLX API was not called'
if 'fast.metal_kernel' in calls:
    assert len(calls['fast.metal_kernel']) == 1, 'construct the kernel once'
    assert calls['fast.metal_kernel'][0].get('ensure_row_contiguous') is False
print({CHECKS_COMPLETED!r})
"""


def execute(
    task: CodingTask, code: str, directory: Path, timeout: float = 30
) -> Execution:
    """Run one candidate with numerical checks, denying writes and network access.

    The process may read installed Python/MLX files and use Metal. This is a
    correctness experiment for generated code, not a hostile-code security test.
    """
    try:
        ast.parse(code)
    except SyntaxError as error:
        return Execution("syntax_error", "", str(error))
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    candidate = directory / "candidate.py"
    candidate.write_text(code)
    checker = directory / "check.py"
    checker.write_text(checker_source(task, candidate))
    profile = "(version 1) (allow default) (deny network*) (deny file-write*)"
    command = [
        "/usr/bin/sandbox-exec",
        "-p",
        profile,
        sys.executable,
        "-I",
        "-B",
        str(checker),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=directory,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return Execution(
            "execution_timeout",
            captured_text(error.stdout),
            captured_text(error.stderr) + f"\nExceeded {timeout} seconds",
        )
    complete = result.stdout.splitlines()[-1:] == [CHECKS_COMPLETED]
    status = (
        "passed"
        if result.returncode == 0 and complete
        else (
            "checks_incomplete"
            if result.returncode == 0
            else "check_failed"
            if "AssertionError" in result.stderr
            else "runtime_error"
        )
    )
    return Execution(status, result.stdout, result.stderr)


def captured_text(value: str | bytes | None) -> str:
    """Keep timeout diagnostics, including an incomplete UTF-8 output fragment."""
    return (
        value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else value or ""
    )


def prompt(task: CodingTask, context: str) -> str:
    """Exclude checker code, expected values, references, and other trial answers."""
    return (
        "Return a JSON object with a code field containing a complete Python module. "
        "Use MLX on Apple silicon. Do not use tools, read files, access the network, "
        "or run code. The module may import mlx and the Python standard library. "
        "Do not include Markdown fences.\n\nTask:\n"
        + task.prompt
        + ("\n\nRetrieved documentation:\n" + context if context else "")
    )


async def documentation(service: DocumentationService, task: CodingTask) -> str:
    """Read three search-selected sections, one fetch page each, then cap text."""
    response = await service.search(task.query, 3, "python", Package.MLX)
    urls = list(dict.fromkeys(block.url for block in source_blocks(response)))
    content = response + "\n\n" + "\n\n".join(service.fetch(url, 0) for url in urls)
    return content[:CONTEXT_CHARACTERS]


def codex_command(directory: Path, model: str, effort: str) -> list[str]:
    """Use existing CLI authentication while excluding user configuration/tools."""
    executable = shutil.which("codex")
    if executable is None:
        raise FileNotFoundError(
            "Install and sign in to the Codex CLI before this experiment"
        )
    command = [
        executable,
        "exec",
        "--ignore-user-config",
        "--strict-config",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--json",
        "--model",
        model,
        "--config",
        f'model_reasoning_effort="{effort}"',
        "--config",
        'web_search="disabled"',
        "--config",
        "project_doc_max_bytes=0",
        "--enable",
        "skip_host_skill_discovery",
        "--output-schema",
        str(directory / "schema.json"),
        "--output-last-message",
        str(directory / "answer.json"),
        "--cd",
        str(directory),
    ]
    for feature in DISABLED_FEATURES:
        command.extend(("--disable", feature))
    return command + ["-"]


def used_tools(events: str) -> bool:
    """Reject trials containing any completed item other than text/reasoning."""
    rows = TypeAdapter(list[dict[str, object]]).validate_json(
        "[" + ",".join(events.splitlines()) + "]", strict=True
    )
    for row in rows:
        if row.get("type") not in ("item.started", "item.completed"):
            continue
        item = TypeAdapter(dict[str, object]).validate_python(
            row.get("item"), strict=True
        )
        # CLI warnings are emitted as error items (for example an experimental
        # feature notice). They do not give the model external information.
        if item.get("type") not in ("agent_message", "reasoning", "error"):
            return True
    return False


def generate(
    directory: Path, text: str, model: str, effort: str
) -> tuple[str | None, Execution]:
    """Retain raw CLI events; failed generation is an outcome, never a retry."""
    directory.mkdir()
    (directory / "prompt.txt").write_text(text)
    (directory / "schema.json").write_text(json.dumps(CodeOutput.model_json_schema()))
    try:
        result = subprocess.run(
            codex_command(directory, model, effort),
            input=text,
            text=True,
            capture_output=True,
            timeout=180,
            cwd=directory,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout, stderr = captured_text(error.stdout), captured_text(error.stderr)
        (directory / "events.jsonl").write_text(stdout)
        (directory / "generation.stderr").write_text(stderr)
        return None, Execution(
            "generation_timeout", stdout, stderr + "\nExceeded 180 seconds"
        )
    (directory / "events.jsonl").write_text(result.stdout)
    (directory / "generation.stderr").write_text(result.stderr)
    if result.returncode:
        return None, Execution("generation_error", result.stdout, result.stderr)
    try:
        contaminated = used_tools(result.stdout)
    except ValidationError as error:
        return None, Execution("invalid_events", result.stdout, str(error))
    if contaminated:
        return None, Execution("contaminated", result.stdout, "Generation used a tool")
    answer = directory / "answer.json"
    if not answer.is_file():
        return None, Execution("generation_error", result.stdout, "No final answer")
    try:
        output = CodeOutput.model_validate_json(answer.read_text())
    except ValidationError as error:
        return None, Execution("invalid_output", result.stdout, str(error))
    return output.code, Execution("generated", "", "")


def trial_result(
    task: CodingTask,
    condition: Condition,
    repetition: int,
    context: str,
    temporary: Path,
    output: Path,
    model: str,
    effort: str,
) -> dict[str, object]:
    """Generate and grade one fresh answer, preserving files even when it fails."""
    identity = f"{task.id}-{repetition}-{condition}"
    trial = temporary / identity
    code, result = generate(trial, prompt(task, context), model, effort)
    if code is not None:
        result = execute(task, code, trial / "execution")
    shutil.copytree(trial, output / identity)
    return {
        "task": task.id,
        "repetition": repetition,
        "condition": condition,
        "context_characters": len(context),
        **asdict(result),
    }


class Options(argparse.Namespace):
    output: Path
    model: str
    effort: str
    repetitions: int
    index: Path | None


def run(args: Options) -> None:
    """Counterbalance condition order; persist every prompt, answer, and outcome."""
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.index.resolve() if args.index else select_index_directory()
    service = DocumentationService(root, access=ModelAccess.CACHED)
    try:
        contexts = {
            task.id: asyncio.run(documentation(service, task)) for task in TASKS
        }
        snapshot = service.documentation_info()
    finally:
        service.close()
    rows: list[dict[str, object]] = []
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="mlx-codegen-") as name:
        temporary = Path(name)
        for repetition in range(args.repetitions):
            for number, task in enumerate(TASKS):
                conditions: list[Condition] = ["without_docs", "with_docs"]
                if (repetition + number) % 2:
                    conditions.reverse()
                for condition in conditions:
                    context = contexts[task.id] if condition == "with_docs" else ""
                    row = trial_result(
                        task,
                        condition,
                        repetition,
                        context,
                        temporary,
                        args.output,
                        args.model,
                        args.effort,
                    )
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in row.items()
                                if k not in ("stdout", "stderr")
                            }
                        ),
                        flush=True,
                    )
    report = {
        "schema_version": 1,
        "model": args.model,
        "reasoning_effort": args.effort,
        "repetitions": args.repetitions,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "device": "mlx.gpu",
        "versions": {name: version(name) for name in ("mlx", "numpy", "mlx-mcp")},
        "documentation": snapshot,
        "context_character_limit": CONTEXT_CHARACTERS,
        "counts": {
            condition: dict(
                Counter(str(r["status"]) for r in rows if r["condition"] == condition)
            )
            for condition in ("without_docs", "with_docs")
        },
        "trials": rows,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--index", type=Path)
    args = parser.parse_args(namespace=Options())
    args.output = args.output.resolve()
    if args.repetitions < 1:
        raise ValueError("At least one repetition is required")
    if not Path("/usr/bin/sandbox-exec").is_file():
        raise FileNotFoundError(
            "This experiment requires macOS sandbox-exec and MLX GPU"
        )
    run(args)


if __name__ == "__main__":
    main()
