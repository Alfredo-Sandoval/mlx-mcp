"""A client-executed workflow for measured MLX performance improvements."""

PERFORMANCE_WORKFLOW = """Improve the user's code while applying this workflow. Keep
the requested implementation moving; do not replace it with a planning report.
The MCP server supplies guidance and documentation. The coding client inspects,
edits and runs the user's project using its existing tools and permissions.

1. Establish a representative baseline.
Use the project's environment and real inputs. Record hardware, installed MLX
version, device, model/checkpoint, dtype, resolution, batch size and code revision.
Record the imported source location when a checkout can shadow an installed
distribution. Keep dependency installs and source changes outside timed runs;
restart in a fresh process after a runtime change.
Separate model loading, first-call compilation, warmup and steady-state work.
Include partial final batches in sustained measurements: a new batch shape can
trigger compilation even after the usual batch has warmed up. Confirm optimized
flags reach the installed runtime through the project's public configuration.
Evaluate returned MLX arrays and synchronize the workload's streams before
stopping the timer. Include decoding, preprocessing, inference, output conversion
and writing when the user cares about pipeline throughput. Report latency,
throughput and peak active memory with their measurement scope.
For cross-backend comparisons, match the checkpoint/pruning, crop geometry,
actual network input size, precision and output policy. Cropping alone does not
reduce encoder work if every crop is resized to the same fixed network size.

2. Find where wall time goes.
Time the meaningful stages with evaluation at their boundaries. These barriers
can perturb scheduling: use stage timing to locate costs, then measure the
uninstrumented pipeline for the final result. Search the hot path for .item(),
NumPy conversions, Python decisions on device values, repeated eval calls,
model creation, redundant preprocessing and repeated encodings. On Apple unified
memory, a host conversion can impose a synchronization or CPU scan without a
physical device-to-host copy. Inspect actual tensor shapes, padding, layout
changes and which outputs are ultimately consumed.
Time charged to a scalar readback can include previously queued GPU compute;
do not count the whole wait as removable synchronization overhead.

3. Remove measured waste first.
Keep reusable models/encodings resident. Select needed queries on-device before
independent mask generation, expensive upsampling or host materialization;
preserve upstream attention between queries. Batch independent work when the
model supports it; avoid scalar readbacks merely to select an index. Check whether
resolution or batch shapes trigger excessive padding. Treat changes to model,
resolution, dtype, frame sampling or output policy as quality tradeoffs, separately
from equivalent-code optimizations. Do not assume compilation, FP16, larger
batches or a mode named 'fast' will improve the measured workload.
Distinguish stored weight dtype from actual activation/operation dtype. Check
implicit promotion and test pure FP16 separately from mixed-precision policies.
Trace input, weight and output dtypes at projections: FP32 positional features,
geometry or pooled text can promote an entire decoder despite FP16 weights.
If reducing these activations, keep intended normalization/accumulation boundaries
and measure output differences explicitly.
Matrix-to-vector shape changes can select different precision paths; verify
numerical differences against a higher-precision reference before blaming the
optimization. MLX's float32 dtype alone does not guarantee full-precision matmul.
Use fused scaled-dot-product attention where the existing attention semantics
match; verify batch isolation and original-image coordinates when batching.

4. Write a custom kernel when the measured bottleneck justifies it.
Use search_mlx_docs and fetch_mlx_docs for the installed-version-compatible
mlx.core.fast.metal_kernel API and custom-kernel guide. Prefer existing optimized
MLX primitives for matrix multiplication and attention. Fuse memory passes,
layout conversion, normalization or small repeated operations where useful.
Build/cache the kernel once, then dispatch it repeatedly. Preserve rounding,
antialiasing, operation order and boundary behavior when equivalence is required.
Compare against the original implementation on representative shapes, edges and
real inputs. Measure complete-pipeline impact, not just isolated kernel speed.

5. Keep only demonstrated improvements.
Change one major variable at a time. Use the same inputs and explicit correctness
criteria; report exact equality where achieved, otherwise name the numerical or
task-level difference. Compare warm and sustained runs without competing GPU jobs.
Coordinate an exclusive GPU window with other sessions before measuring.
When clock or thermal state can drift, interleave A/B trials with equivalent
warmup and fixed cool-downs, balance their order, and compare paired results and
medians. Report cooled short bursts separately from thermally steady throughput;
an earlier baseline and a later fast run do not by themselves establish a gain.
Reject slower experiments. Run the project's required checks once the final
implementation is ready. Report baseline and final throughput, what changed,
remaining bottlenecks and limitations. Save reproducible benchmark evidence in
the project's normal build/benchmark location, without generating task logs or
inventing validation requirements.

Useful documentation queries:
- lazy evaluation and when to evaluate
- mlx.core.argmax and mlx.core.take_along_axis
- mlx.core.async_eval and streams
- mlx.core.fast.metal_kernel and custom Metal kernels
- profiling memory and performance

Search and fetch the relevant documentation as needed; this workflow does not
claim to have inspected, benchmarked or optimized the user's code.
"""
