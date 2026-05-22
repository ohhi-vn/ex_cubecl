# GPU ML Pipeline Example
# Demonstrates async execution and pipeline orchestration for inference
# Run with: mix run examples/ml_pipeline.exs

IO.puts("=== GPU ML Pipeline: Neural Network Inference ===")
IO.puts("")

# Check device
{:ok, info} = ExCubecl.device_info()
IO.puts("Device: #{info.device_name}")
IO.puts("")

# --- Async Execution ---
IO.puts("--- Async Execution ---")

# Create input buffer
{:ok, input} = ExCubecl.buffer([0.5, -0.3, 0.8], [3], :f32)
{:ok, shape} = ExCubecl.shape(input)
IO.puts("Input buffer: shape=#{inspect(shape)}")

# Create output buffer
{:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 3), [3], :f32)

# Submit async work
{:ok, cmd_id} = ExCubecl.submit("run_relu")
IO.puts("Submitted command: #{cmd_id}")

# Poll for completion
case ExCubecl.poll(cmd_id) do
  {:ok, :pending} -> IO.puts("Command still running...")
  {:ok, :completed} -> IO.puts("Command completed!")
  {:ok, status} -> IO.puts("Status: #{status}")
  {:error, reason} -> IO.puts("Error: #{reason}")
end

# Wait for completion
result = ExCubecl.wait(cmd_id)
IO.puts("Wait result: #{inspect(result)}")
IO.puts("")

# --- Pipeline Orchestration ---
IO.puts("--- Pipeline Orchestration ---")

# Create a multi-stage pipeline
{:ok, pipeline} = ExCubecl.pipeline()

# Stage 1: Input processing
{:ok, input_buf} = ExCubecl.buffer([1.0, 2.0, 3.0, 4.0, 5.0], [5], :f32)
{:ok, stage1_out} = ExCubecl.buffer(List.duplicate(0.0, 5), [5], :f32)

ExCubecl.pipeline_add(pipeline, "elementwise_add", [input_buf], stage1_out)

# Stage 2: Activation
{:ok, stage2_out} = ExCubecl.buffer(List.duplicate(0.0, 5), [5], :f32)

ExCubecl.pipeline_add(pipeline, "relu", [stage1_out], stage2_out)

IO.puts("Pipeline created with 2 stages")

# Run the pipeline
{:ok, _cmd_ids} = ExCubecl.pipeline_run(pipeline)
IO.puts("Pipeline executed")

# Read final result
{:ok, final} = ExCubecl.read(stage2_out)
IO.puts("Pipeline output: #{byte_size(final)} bytes")

# Buffers are automatically freed when GC'd — no manual free needed
:ok = ExCubecl.pipeline_free(pipeline)

IO.puts("")
IO.puts("=== Done ===")
