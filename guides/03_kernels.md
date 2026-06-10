# Kernel Execution

Kernels are exposed through a CubeCL-style API and execute CPU-side in the compiled Rust NIF. They operate on Rust NIF-managed buffer references (`ResourceArc<Buffer>`).

## Running a Kernel

```elixir
{:ok, cmd_id} = ExCubecl.run_kernel(name, inputs, output, params \\ %{})
```

- `name` — string kernel name (see below)
- `inputs` — list of input buffer references
- `output` — output buffer reference
- `params` — optional map of kernel-specific parameters

## Available Kernels

The kernel list reflects the kernels implemented in the Rust NIF. As of v0.5.0 there are 24 kernels:

```elixir
{:ok, kernels} = ExCubecl.kernels()
# ["elementwise_add", "elementwise_mul", "elementwise_sub",
#  "elementwise_div", "relu", "sigmoid", "tanh",
#  # Video kernels
#  "yuv_to_rgb", "overlay_alpha", "video_mix", "gaussian_blur",
#  "bicubic_scale", "lut_apply", "chroma_key", "sharpen",
#  "brightness_contrast", "denoise", "video_crop",
#  # Audio kernels
#  "pcm_mix", "pcm_normalize", "biquad_filter", "fft_convolve",
#  "resample_linear", "dynamics_compress"
# ]
# length(kernels) == 24
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

## Custom Kernels

Custom kernel registration is not implemented. The kernel list reflects what is compiled into the Rust NIF.
