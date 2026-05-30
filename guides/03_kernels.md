# Kernel Execution

Kernels are GPU programs that operate on buffers.

## Running a Kernel

```elixir
{:ok, cmd_id} = ExCubecl.run_kernel(name, inputs, output, params \\ %{})
```

- `name` — string kernel name (see below)
- `inputs` — list of input buffer references
- `output` — output buffer reference
- `params` — optional map of kernel-specific parameters

## Available Kernels

```elixir
{:ok, kernels} = ExCubecl.kernels()
# ["elementwise_add", "elementwise_mul", "elementwise_sub",
#  "elementwise_div", "relu", "sigmoid", "tanh", "matmul",
#  "reduce_sum", "reduce_max", "reduce_min", "softmax",
#  "layer_norm", "conv2d", "transpose", "reshape",
#  # Phase 2 — video kernels
#  "yuv_to_rgb", "overlay_alpha", "video_mix", "gaussian_blur",
#  "bicubic_scale", "lut_apply", "chroma_key", "sharpen",
#  # Phase 2 — audio kernels
#  "pcm_mix", "pcm_normalize", "biquad_filter", "fft_convolve",
#  "resample_linear", "dynamics_compress"]
```

## Element-wise Operations

```elixir
{:ok, a} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
{:ok, b} = ExCubecl.buffer([4.0, 5.0, 6.0], [3], :f32)
{:ok, output} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)

{:ok, _cmd} = ExCubecl.run_kernel("elementwise_add", [a, b], output)
{:ok, _cmd} = ExCubecl.run_kernel("elementwise_mul", [a, b], output)
```

## Activation Functions

```elixir
{:ok, input} = ExCubecl.buffer([-1.0, 0.0, 1.0], [3], :f32)
{:ok, output} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)

{:ok, _cmd} = ExCubecl.run_kernel("relu", [input], output)
{:ok, _cmd} = ExCubecl.run_kernel("sigmoid", [input], output)
{:ok, _cmd} = ExCubecl.run_kernel("tanh", [input], output)
```

## Reductions

```elixir
{:ok, input} = ExCubecl.buffer([1.0, 5.0, 3.0, 2.0], [4], :f32)
{:ok, output} = ExCubecl.buffer([0.0], [1], :f32)

{:ok, _cmd} = ExCubecl.run_kernel("reduce_sum", [input], output)
{:ok, _cmd} = ExCubecl.run_kernel("reduce_max", [input], output)
{:ok, _cmd} = ExCubecl.run_kernel("reduce_min", [input], output)
```

## Matrix Multiplication

```elixir
# 2x3 matrix
{:ok, a} = ExCubecl.buffer([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], :f32)
# 3x2 matrix
{:ok, b} = ExCubecl.buffer([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2], :f32)
# Output: 2x2 matrix
{:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 4), [2, 2], :f32)

{:ok, _cmd} = ExCubecl.run_kernel("matmul", [a, b], output)
```

## Convolution

```elixir
# 1x3x3 input (batch=1, channels=1, 3x3 spatial)
{:ok, input} = ExCubecl.buffer([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                                [1, 1, 3, 3], :f32)
# 1x1x2x2 kernel
{:ok, kernel} = ExCubecl.buffer([1.0, 0.0, 0.0, -1.0], [1, 1, 2, 2], :f32)
{:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 4), [1, 1, 2, 2], :f32)

{:ok, _cmd} = ExCubecl.run_kernel("conv2d", [input, kernel], output)
```

## Shape Operations

```elixir
{:ok, input} = ExCubecl.buffer([1.0, 2.0, 3.0, 4.0], [4], :f32)

# Reshape to 2x2
{:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 4), [2, 2], :f32)
{:ok, _cmd} = ExCubecl.run_kernel("reshape", [input], output)

# Transpose
{:ok, transposed} = ExCubecl.buffer(List.duplicate(0.0, 4), [2, 2], :f32)
{:ok, _cmd} = ExCubecl.run_kernel("transpose", [output], transposed)
```

## Custom Kernels (Phase 2+)

Custom CubeCL kernels can be registered at runtime. See the [CubeCL documentation](https://github.com/tracel-ai/cubecl) for kernel authoring.
