# Examples

## Basic Buffer Operations

Examples in this guide use the ExCubecl API. Buffers are Rust NIF-managed resource references.

```elixir
# Create and inspect
{:ok, buf} = ExCubecl.buffer([1.0, 2.0, 3.0, 4.0], [4], :f32)
{:ok, [4]} = ExCubecl.shape(buf)
{:ok, "f32"} = ExCubecl.dtype(buf)
{:ok, 16} = ExCubecl.size(buf)

# Read back
{:ok, data} = ExCubecl.read(buf)

# No manual cleanup needed — buffer is automatically freed when GC'd
```

## Image Processing Pipeline

The example below expresses an image-processing pipeline through the API.

```elixir
# Load image data (5x5 grayscale)
pixels = Enum.to_list(10..250//10) |> Enum.take(25)
{:ok, input} = ExCubecl.buffer(pixels, [5, 5], :f32)

# Create intermediate buffers
{:ok, padded} = ExCubecl.buffer(List.duplicate(0.0, 49), [7, 7], :f32)
{:ok, blurred} = ExCubecl.buffer(List.duplicate(0.0, 25), [5, 5], :f32)
{:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 25), [5, 5], :f32)

# Run kernels
{:ok, _cmd1} = ExCubecl.run_kernel("gaussian_blur", [padded], blurred)
{:ok, _cmd2} = ExCubecl.run_kernel("elementwise_add", [blurred], output)

# Read result
{:ok, result} = ExCubecl.read(output)

# No manual cleanup needed — buffers are automatically freed when GC'd
```

## AI Inference Pipeline

This is an API sketch for inference-style operations using available kernels:

```elixir
# Elementwise operations for preprocessing
{:ok, input} = ExCubecl.buffer(raw_pixels, [224 * 224 * 3], :f32)
{:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 224 * 224 * 3), [224 * 224 * 3], :f32)
{:ok, _cmd} = ExCubecl.run_kernel("elementwise_add", [input, input], output)
{:ok, result} = ExCubecl.read(output)
```

## Async Batch Processing

The async API uses NIF-managed command state with thread-pool execution:

```elixir
# Process multiple frames through async command-style APIs
cmd_ids =
  for frame <- camera_frames do
    {:ok, input} = ExCubecl.buffer(frame, [1080, 1920], :f32)
    {:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 1080 * 1920), [1080, 1920], :f32)
    {:ok, cmd_id} = ExCubecl.submit("blur")
    {cmd_id, input, output}
  end

# Wait for all to complete
for {cmd_id, _input, _output} <- cmd_ids do
  :ok = ExCubecl.wait(cmd_id)
  # Buffers are automatically freed when GC'd
end
```

## Multi-Stage Pipeline

Pipeline orchestration executes kernels through the NIF:

```elixir
# Build a 3-stage image processing pipeline
{:ok, input} = ExCubecl.buffer(image_data, [1080, 1920], :f32)
{:ok, denoised} = ExCubecl.buffer(List.duplicate(0.0, 1080 * 1920), [1080, 1920], :f32)
{:ok, edges} = ExCubecl.buffer(List.duplicate(0.0, 1080 * 1920), [1080, 1920], :f32)
{:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 1080 * 1920), [1080, 1920], :f32)

{:ok, pipeline} = ExCubecl.pipeline()

:ok = ExCubecl.pipeline_add(pipeline, "elementwise_add", [input, input], denoised)
:ok = ExCubecl.pipeline_add(pipeline, "relu", [denoised], edges)
:ok = ExCubecl.pipeline_add(pipeline, "elementwise_add", [edges], output)

{:ok, _cmd_ids} = ExCubecl.pipeline_run(pipeline)

{:ok, result} = ExCubecl.read(output)

:ok = ExCubecl.pipeline_free(pipeline)
```
