# Getting Started

ExCubecl is a GPU compute runtime for Elixir, powered by CubeCL via Rust NIFs. The current release (v0.5.0) includes a compiled Rust NIF with CPU-side kernel implementations that simulate the CubeCL API surface.

## Installation

Add to your `mix.exs`:

```elixir
def deps do
  [{:ex_cubecl, "~> 0.5"}]
end
```

Then run `mix deps.get`.

## Architecture

```
Elixir → ExCubecl.NIF → Rust NIF → CPU-side kernel execution
```

All resource state lives in Rust-managed memory via `ResourceArc`. The Elixir side manages handles, orchestrates pipelines, and exposes command-style APIs. Kernels execute CPU-side implementations of the CubeCL API surface. Media I/O, transcoding, and video/audio operations use a mix of NIF-backed implementations and Elixir-level validation.

## Core Concepts

### Buffers

Buffers are the primary data structure, representing typed, shaped data in CPU-side resource memory:

```elixir
# Create a buffer from a list
{:ok, buf} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)

# Inspect
{:ok, [3]} = ExCubecl.shape(buf)
{:ok, "f32"} = ExCubecl.dtype(buf)
{:ok, 12} = ExCubecl.size(buf)   # bytes

# Read data back
{:ok, data} = ExCubecl.read(buf)

# Buffers are automatically freed when the Elixir term is garbage collected.
```

### Kernels

Kernels are exposed through the CubeCL-style API and execute CPU-side in the Rust NIF:

```elixir
{:ok, input} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
{:ok, output} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)

{:ok, _cmd} = ExCubecl.run_kernel("elementwise_add", [input], output)
```

### Async Execution

Submit command-style work without blocking the BEAM:

```elixir
{:ok, cmd_id} = ExCubecl.submit("some_command")

# Poll for status
{:ok, :completed} = ExCubecl.poll(cmd_id)

# Or block until done
:ok = ExCubecl.wait(cmd_id)
```

### Pipelines

Compose multiple kernel operations into a single executable graph:

```elixir
{:ok, pipeline} = ExCubecl.pipeline()

:ok = ExCubecl.pipeline_add(pipeline, "elementwise_add", [buf_a, buf_b], buf_out)
:ok = ExCubecl.pipeline_add(pipeline, "relu", [buf_out], buf_result)

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
ExCubecl.version()       # returns the ExCubecl module version (e.g. "0.5.0")
ExCubecl.device_info()   # NIF device metadata (device_type: "gpu", total_memory, compute_units)
ExCubecl.device_count()  # number of available devices
```

## Next Steps

- [Buffer Management](02_buffers.md) — creating, reading, and inspecting buffers
- [Kernel Execution](03_kernels.md) — running kernels via the NIF
- [Async & Pipelines](04_async_pipelines.md) — command-style APIs and pipeline orchestration
- [Mobile Integration](05_mobile.md) — iOS/Android C header prototype
- [Examples](06_examples.md) — API examples
