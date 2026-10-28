"""Frozen coding tasks with checks withheld from the generation prompt.

The reference and deliberately incorrect solutions validate each checker. These
small MLX-core tasks measure executable behavior, not general agent performance.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CodingTask:
    id: str
    prompt: str
    query: str
    checks: str
    reference: str
    incorrect: str


TASKS = (
    CodingTask(
        "scatter-duplicates",
        "Implement solve(a, indices, updates) returning an MLX array with all updates "
        "added to a one-dimensional float32 a. indices can repeat; every update must "
        "contribute. Leave a unchanged.",
        "How do indexed updates accumulate repeated indices in MLX?",
        """
for indices in ([1, 1, 3, 1], [0, 2, 0, 2]):
    original = np.array([2., 4., 6., 8.], dtype=np.float32)
    updates = np.array([1., -3., 5., 2.], dtype=np.float32)
    expected = original.copy()
    np.add.at(expected, indices, updates)
    a = mx.array(original)
    close(solve(a, mx.array(indices), mx.array(updates)), expected)
    close(a, original)
""",
        """
import mlx.core as mx
def solve(a, indices, updates):
    return a.at[indices].add(updates)
""",
        """
def solve(a, indices, updates):
    a[indices] += updates
    return a
""",
    ),
    CodingTask(
        "grad-with-aux",
        "Implement solve(w, x, y) using MLX automatic differentiation. Return "
        "((loss, predictions), gradient_w), where predictions=x@w and loss is mean "
        "squared error against y. Preserve the predictions as auxiliary output; "
        "differentiate only the loss with respect to w.",
        "mlx.core.value_and_grad auxiliary outputs tuple loss predictions",
        """
for scale in (1., -2.):
    x = np.array([[1., 2.], [-3., 4.], [5., -2.]], dtype=np.float32) * scale
    w = np.array([.3, -.7], dtype=np.float32)
    y = np.array([1., -2., .5], dtype=np.float32)
    (loss, pred), grad = solve(mx.array(w), mx.array(x), mx.array(y))
    close(pred, x @ w)
    close(loss, np.mean((x @ w - y) ** 2))
    close(grad, 2 * x.T @ (x @ w - y) / len(y))
""",
        """
import mlx.core as mx
def solve(w, x, y):
    def loss(params):
        pred = x @ params
        return mx.mean((pred-y)**2), pred
    return mx.value_and_grad(loss)(w)
""",
        """
import mlx.core as mx
def solve(w, x, y):
    pred = x @ w
    return (mx.mean((pred-y)**2), pred), mx.zeros_like(w)
""",
    ),
    CodingTask(
        "compiled-state",
        "Implement solve(initial) returning (step, state). state is a dict with an "
        "MLX scalar array under 'total', initialized from initial. step(delta) must "
        "be compiled with mx.compile, add delta to total, update state, and return "
        "the new total. Repeated calls must see previous updates.",
        "mlx.core.compile inputs outputs captured mutable state dictionary",
        """
for initial in (2., -3.):
    step, state = solve(mx.array(initial))
    total = initial
    for delta in (1., 3., -2., .5):
        total += delta
        close(step(mx.array(delta)), total)
        close(state['total'], total)
""",
        """
import mlx.core as mx
def solve(initial):
    state = {'total': initial}
    def step(delta):
        state['total'] = state['total'] + delta
        return state['total']
    return mx.compile(step, inputs=state, outputs=state), state
""",
        """
import mlx.core as mx
def solve(initial):
    state = {'total': initial}
    def step(delta):
        state['total'] = state['total'] + delta
        return state['total']
    return mx.compile(step), state
""",
    ),
    CodingTask(
        "conv-layout",
        "Implement solve(x, weights) with MLX conv2d. x is NCHW and weights are "
        "OIHW float32 arrays. Return NCHW convolution with stride 1, no padding, "
        "no bias, and no grouping. Handle unequal input and output channels.",
        "mlx.core.conv2d input weight shapes NHWC OHWI transpose NCHW",
        """
for channels, outputs in ((2, 3), (3, 1)):
    x = np.arange(2*channels*4*5, dtype=np.float32).reshape(2, channels, 4, 5) / 31
    w = np.arange(outputs*channels*2*3, dtype=np.float32).reshape(outputs, channels, 2, 3) / 17
    expected = np.empty((2, outputs, 3, 3), dtype=np.float32)
    for n, o, h, k in np.ndindex(expected.shape):
        expected[n,o,h,k] = np.sum(x[n,:,h:h+2,k:k+3] * w[o])
    close(solve(mx.array(x), mx.array(w)), expected)
""",
        """
import mlx.core as mx
def solve(x, weights):
    return mx.conv2d(x.transpose(0,2,3,1), weights.transpose(0,2,3,1)).transpose(0,3,1,2)
""",
        """
import mlx.core as mx
def solve(x, weights):
    return mx.conv2d(x, weights)
""",
    ),
    CodingTask(
        "per-example-grad",
        "Implement solve(w, x, y) returning per-example gradients with respect to "
        "the shared vector w of (dot(w,x_i)-y_i)**2. Use MLX vmap and grad, "
        "without a Python loop over examples. x has shape (batch, features).",
        "mlx.core.vmap grad in_axes None shared weights per example gradient",
        """
for batch in (1, 5):
    x = np.arange(batch*3, dtype=np.float32).reshape(batch,3) / 7
    w = np.array([.2, -.5, 1.], dtype=np.float32)
    y = np.arange(batch, dtype=np.float32) / 3
    expected = 2*(x@w-y)[:,None]*x
    close(solve(mx.array(w), mx.array(x), mx.array(y)), expected)
""",
        """
import mlx.core as mx
def solve(w, x, y):
    def loss(w, row, target):
        return (mx.sum(w*row)-target)**2
    return mx.vmap(mx.grad(loss), in_axes=(None,0,0))(w,x,y)
""",
        """
import mlx.core as mx
def solve(w, x, y):
    return mx.grad(lambda p: mx.mean((x@p-y)**2))(w)
""",
    ),
    CodingTask(
        "explicit-random-key",
        "Implement solve(seed, shape) returning float32 standard-normal MLX samples "
        "of shape. Repeating seed must reproduce the samples. Do not read or "
        "change the global MLX random stream; use an explicit PRNG key.",
        "MLX random normal explicit key seed independent global random state",
        """
for seed in (7, 19):
    mx.random.seed(123)
    expected_next = mx.random.normal((6,))
    mx.eval(expected_next)
    mx.random.seed(123)
    a = solve(seed, (2,3))
    b = solve(seed, (2,3))
    close(a, np.asarray(b))
    assert a.shape == (2,3) and a.dtype == mx.float32
    assert not np.array_equal(np.asarray(a), np.asarray(solve(seed+1, (2,3))))
    samples = solve(seed, (4096,))
    assert isinstance(samples, mx.array) and samples.shape == (4096,)
    assert samples.dtype == mx.float32
    values = np.asarray(samples)
    assert abs(values.mean()) < .08 and .92 < values.std() < 1.08
    close(mx.random.normal((6,)), np.asarray(expected_next))
""",
        """
import mlx.core as mx
def solve(seed, shape):
    return mx.random.normal(shape, dtype=mx.float32, key=mx.random.key(seed))
""",
        """
import mlx.core as mx
def solve(seed, shape):
    mx.random.seed(seed)
    return mx.random.normal(shape)
""",
    ),
    CodingTask(
        "metal-strides",
        "Implement solve(x) returning x squared using a custom MLX Metal kernel. "
        "x is a nonempty float32 MLX array and may be transposed or strided. "
        "Do not make a contiguous copy; set ensure_row_contiguous=False and "
        "honor input strides. Construct the kernel once at module scope.",
        "custom Metal kernel ensure_row_contiguous false elem_to_loc strides shape ndim",
        """
x = mx.arange(60, dtype=mx.float32).reshape(3,4,5) / 13
for view in (x, x.transpose(2,0,1), x[:,::2,1::2]):
    close(solve(view), np.asarray(view)**2)
""",
        """
import mlx.core as mx
kernel = mx.fast.metal_kernel(name='square_strided', input_names=['inp'], output_names=['out'],
    ensure_row_contiguous=False, source='''
    uint i = thread_position_in_grid.x;
    uint loc = elem_to_loc(i, inp_shape, inp_strides, inp_ndim);
    out[i] = inp[loc] * inp[loc];
    ''')
def solve(x):
    return kernel(inputs=[x], grid=(x.size,1,1), threadgroup=(64,1,1),
                  output_shapes=[x.shape], output_dtypes=[x.dtype])[0]
""",
        """
import mlx.core as mx
kernel = mx.fast.metal_kernel(name='wrong_strides', input_names=['inp'], output_names=['out'],
    ensure_row_contiguous=False, source='''
    uint i = thread_position_in_grid.x;
    out[i] = inp[i] * inp[i];
    ''')
def solve(x):
    return kernel(inputs=[x], grid=(x.size,1,1), threadgroup=(64,1,1),
                  output_shapes=[x.shape], output_dtypes=[x.dtype])[0]
""",
    ),
    CodingTask(
        "reinterpret-bits",
        "Implement solve(x) returning an MLX uint32 array with the same shape, "
        "containing the exact IEEE-754 bit patterns of float32 x. Reinterpret "
        "the bytes; do not numerically convert values or copy through NumPy.",
        "MLX reinterpret float32 bytes uint32 view versus astype",
        """
x = np.array([[0., -0., 1.], [-2.5, np.inf, -np.inf]], dtype=np.float32)
actual = solve(mx.array(x))
assert isinstance(actual, mx.array) and actual.dtype == mx.uint32
np.testing.assert_array_equal(np.asarray(actual), x.view(np.uint32))
""",
        """
import mlx.core as mx
def solve(x):
    return x.view(mx.uint32)
""",
        """
import mlx.core as mx
def solve(x):
    return x.astype(mx.uint32)
""",
    ),
)
