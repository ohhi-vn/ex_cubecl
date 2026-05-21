# Getting Started

ExCubecl is a GPU compute runtime for Elixir. It provides GPU buffer management, kernel execution, async command submission, and pipeline orchestration via CubeCL (Rust NIFs).

## Installation

Add to your `mix.exs`:

```elixir
def deps do
  [{:ex_cubecl, "~> 0.2.0"}]
end
```

Then run `mix deps.get`.

## Architecture

```
Elixir → ExCubecl.NIF → Rust NIF → CubeCL Runtime
```

All GPU state lives in Rust — not in BEAM memory. The Elixir side manages handles, orchestrates pipelines, and schedules work.

## Core Concepts

### Buffers

Buffers are the primary data structure, representing GPU memory holding typed, shaped data:

```elixir
# Create a buffer from a list
{:ok, buf} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)

# Inspect
{:ok, [3]} = ExCubecl.shape(buf)
{:ok, "f32"} = ExCubecl.dtype(buf)
{:ok, 12} = ExCubecl.size(buf)   # bytes

# Read data back
{:ok, data} = ExCubecl.read(buf)

# Free when done
:ok = ExCubecl.free(buf)
```

### Kernels

Kernels are GPU programs that operate on buffers:

```elixir
{:ok, input} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
{:ok, output} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)

{:ok, _cmd} = ExCubecl.run_kernel("elementwise_add", [input], output, %{})
```

### Async Execution

Submit work without blocking the BEAM:

```elixir
{:ok, cmd_id} = ExCubecl.submit("some_command")

# Poll for status
{:ok, :completed} = ExCubecl.poll(cmd_id)

# Or block until done
:ok = ExCubecl.wait(cmd_id)
```

### Pipelines

Compose multiple GPU operations into a single executable graph:

```elixir
{:ok, pipeline} = ExCubecl.pipeline()

:ok = ExCubecl.pipeline_add(pipeline, "elementwise_add:1,2:3")
:ok = ExCubecl.pipeline_add(pipeline, "relu:3,4")

{:ok, _cmd_ids} = ExCubecl.pipeline_run(pipeline)
:ok = ExCubecl.pipeline_free(pipeline)
```

## Supported Types

| Type  | Description             |
|-------|-------------------------|
| `:f32`| 32-bit float            |
| `:f64`| 64-bit float            |
| `:s32`| 32-bit signed integer   |
| `:s64`| 64-bit signed integer   |
| `:u32`| 32-bit unsigned integer |
| `:u8` | 8-bit unsigned integer  |

## Checking Availability

```elixir
ExCubecl.available?()    # true if NIF is loaded
ExCubecl.version()       # "0.2.0"
ExCubecl.device_info()   # %{device_name: "...", device_type: "gpu", ...}
ExCubecl.device_count()  # 1
```

## Next Steps

- [Buffer Management](02_buffers.md) — creating, reading, inspecting, freeing buffers
- [Kernel Execution](03_kernels.md) — running GPU kernels
- [Async & Pipelines](04_async_pipelines.md) — non-blocking execution and pipeline orchestration
- [Mobile Integration](05_mobile.md) — iOS/Android C FFI
- [Examples](06_examples.md) — complete end-to-end examples
