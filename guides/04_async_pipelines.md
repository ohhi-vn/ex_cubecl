# Async Execution & Pipelines

## Async Command-Style API

ExCubecl exposes non-blocking-style command APIs. Command state is managed in the Rust NIF with thread-pool-backed async execution.

### Submit → Poll → Wait

```elixir
# Submit work, get a command ID immediately. The NIF spawns a thread for async execution.
{:ok, cmd_id} = ExCubecl.submit("some_command")

# Poll for status (non-blocking)
{:ok, status} = ExCubecl.poll(cmd_id)
# status is one of: :pending, :running, :completed, :failed

# Or block until completion
:ok = ExCubecl.wait(cmd_id)
```

### Why Async Matters

```
BEAM process → submit command to NIF → return immediately
                                        ↓
                              NIF thread pool executes
                                        ↓
                              command status updates
```

You NEVER want the BEAM process waiting for long-running native work. This would block schedulers, cause latency spikes, and freeze the UI.

### Polling Pattern

```elixir
{:ok, cmd_id} = ExCubecl.submit("work")

result =
  Stream.iterate(0, &(&1 + 1))
  |> Enum.reduce_while(nil, fn _, _ ->
    case ExCubecl.poll(cmd_id) do
      {:ok, :pending} -> {:cont, nil}
      {:ok, :completed} -> {:halt, :ok}
      {:ok, :failed} -> {:halt, :error}
      {:error, r} -> {:halt, {:error, r}}
    end
  end)
```

### Parallel Async Work

```elixir
# Submit multiple command-style work items concurrently.
{:ok, cmd_ids} =
  Enum.map(inputs, fn input ->
    {:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 100), [100], :f32)
    {:ok, cmd_id} = ExCubecl.submit("process")
    {cmd_id, output}
  end)
  |> Enum.map(fn {cmd_id, _} -> cmd_id end)

# Wait for all
for cmd_id <- cmd_ids do
  :ok = ExCubecl.wait(cmd_id)
end
```

## Pipeline Orchestration

Pipelines compose multiple kernel operations into a single executable graph.

### Basic Pipeline

```elixir
{:ok, pipeline} = ExCubecl.pipeline()

:ok = ExCubecl.pipeline_add(pipeline, "elementwise_add", [buf_a, buf_b], buf_out)
:ok = ExCubecl.pipeline_add(pipeline, "relu", [buf_out], buf_result)

{:ok, _cmd_ids} = ExCubecl.pipeline_run(pipeline)
:ok = ExCubecl.pipeline_free(pipeline)
```

### Command Format

Pipeline commands are passed as structured arguments: kernel name,
input buffer references, and output buffer reference.

### Multi-Stage Pipeline

```elixir
{:ok, input} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
{:ok, stage1} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)
{:ok, stage2} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)

{:ok, pipeline} = ExCubecl.pipeline()
:ok = ExCubecl.pipeline_add(pipeline, "elementwise_add", [input, input], stage1)
:ok = ExCubecl.pipeline_add(pipeline, "relu", [stage1], stage2)

{:ok, _cmd_ids} = ExCubecl.pipeline_run(pipeline)

{:ok, result} = ExCubecl.read(stage2)

:ok = ExCubecl.pipeline_free(pipeline)
```

### Combining Async and Pipelines

Run a pipeline:

```elixir
{:ok, pipeline} = ExCubecl.pipeline()
:ok = ExCubecl.pipeline_add(pipeline, "gaussian_blur", [buf_in], buf_blur)
:ok = ExCubecl.pipeline_add(pipeline, "relu", [buf_blur], buf_out)

# Run pipeline through the NIF
{:ok, _cmd_ids} = ExCubecl.pipeline_run(pipeline)
:ok = ExCubecl.pipeline_free(pipeline)
```
