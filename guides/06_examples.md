# Examples

## Basic Buffer Operations

```elixir
# Create and inspect
{:ok, buf} = ExCubecl.buffer([1.0, 2.0, 3.0, 4.0], [4], :f32)
{:ok, [4]} = ExCubecl.shape(buf)
{:ok, "f32"} = ExCubecl.dtype(buf)
{:ok, 16} = ExCubecl.size(buf)

# Read back
{:ok, data} = ExCubecl.read(buf)

# Cleanup
:ok = ExCubecl.free(buf)
```

## Image Processing Pipeline

```elixir
# Load image data (5x5 grayscale)
pixels = Enum.to_list(10..250//10) |> Enum.take(25)
{:ok, input} = ExCubecl.buffer(pixels, [5, 5], :f32)

# Create intermediate buffers
{:ok, padded} = ExCubecl.buffer(List.duplicate(0.0, 49), [7, 7], :f32)
{:ok, blurred} = ExCubecl.buffer(List.duplicate(0.0, 25), [5, 5], :f32)
{:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 25), [5, 5], :f32)

# Run kernels
{:ok, _cmd1} = ExCubecl.run_kernel("blur", [padded], blurred, %{radius: 2})
{:ok, _cmd2} = ExCubecl.run_kernel("edge_detect", [blurred], output, %{})

# Read result
{:ok, result} = ExCubecl.read(output)

# Cleanup
:ok = ExCubecl.free(input)
:ok = ExCubecl.free(padded)
:ok = ExCubecl.free(blurred)
:ok = ExCubecl.free(output)
```

## AI Inference Pipeline

```elixir
# Preprocess: normalize input pixels
{:ok, input} = ExCubecl.buffer(raw_pixels, [224, 224, 3], :f32)
{:ok, preprocessed} = ExCubecl.buffer(List.duplicate(0.0, 224 * 224 * 3), [224, 224, 3], :f32)
{:ok, _cmd1} = ExCubecl.run_kernel("reshape", [input], preprocessed, %{})

# Inference: fully-connected layer via matmul
{:ok, logits} = ExCubecl.buffer(List.duplicate(0.0, 1000), [1000], :f32)
{:ok, _cmd2} = ExCubecl.run_kernel("matmul", [preprocessed], logits, %{})

# Post-process: softmax for probabilities
{:ok, probs} = ExCubecl.buffer(List.duplicate(0.0, 1000), [1000], :f32)
{:ok, _cmd3} = ExCubecl.run_kernel("softmax", [logits], probs, %{})

# Read predictions
{:ok, predictions} = ExCubecl.read(probs)

# Cleanup
:ok = ExCubecl.free(input)
:ok = ExCubecl.free(preprocessed)
:ok = ExCubecl.free(logits)
:ok = ExCubecl.free(probs)
```

## Matrix Operations

```elixir
# Matrix multiply: (2x3) × (3x2) = (2x2)
{:ok, a} = ExCubecl.buffer([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], :f32)
{:ok, b} = ExCubecl.buffer([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2], :f32)
{:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 4), [2, 2], :f32)

{:ok, _cmd} = ExCubecl.run_kernel("matmul", [a, b], output, %{})
{:ok, result} = ExCubecl.read(output)

# Identity matrix
{:ok, eye} = ExCubecl.buffer([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3], :f32)

# Cleanup
:ok = ExCubecl.free(a)
:ok = ExCubecl.free(b)
:ok = ExCubecl.free(output)
:ok = ExCubecl.free(eye)
```

## Async Batch Processing

```elixir
# Process multiple camera frames concurrently
cmd_ids =
  for frame <- camera_frames do
    {:ok, input} = ExCubecl.buffer(frame, [1080, 1920], :f32)
    {:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 1080 * 1920), [1080, 1920], :f32)
    {:ok, cmd_id} = ExCubecl.submit("blur")
    {cmd_id, input, output}
  end

# Wait for all to complete
for {cmd_id, input, output} <- cmd_ids do
  :ok = ExCubecl.wait(cmd_id)
  :ok = ExCubecl.free(input)
  :ok = ExCubecl.free(output)
end
```

## Multi-Stage Pipeline with Cleanup

```elixir
# Build a 3-stage image processing pipeline
{:ok, input} = ExCubecl.buffer(image_data, [1080, 1920], :f32)
{:ok, denoised} = ExCubecl.buffer(List.duplicate(0.0, 1080 * 1920), [1080, 1920], :f32)
{:ok, edges} = ExCubecl.buffer(List.duplicate(0.0, 1080 * 1920), [1080, 1920], :f32)
{:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 1080 * 1920), [1080, 1920], :f32)

{:ok, pipeline} = ExCubecl.pipeline()

:ok =
  ExCubecl.pipeline_add(pipeline,
    "elementwise_add:#{input},#{input}:#{denoised}"
  )

:ok =
  ExCubecl.pipeline_add(pipeline,
    "relu:#{denoised}:#{edges}"
  )

:ok =
  ExCubecl.pipeline_add(pipeline,
    "reduce_sum:#{edges}:#{output}"
  )

{:ok, _cmd_ids} = ExCubecl.pipeline_run(pipeline)

{:ok, result} = ExCubecl.read(output)

# Cleanup everything
:ok = ExCubecl.free(input)
:ok = ExCubecl.free(denoised)
:ok = ExCubecl.free(edges)
:ok = ExCubecl.free(output)
:ok = ExCubecl.pipeline_free(pipeline)
```
