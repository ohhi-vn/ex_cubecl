# Buffer Management

Buffers are GPU memory regions managed via Rustler `ResourceArc` references. When the Elixir term holding a buffer is garbage collected, the underlying GPU memory is automatically freed.

## Creating Buffers

All buffers are created from a flat list of values with a shape and type:

```elixir
# 1D float vector
{:ok, buf} = ExCubecl.buffer([1.0, 2.0, 3.0, 4.0], [4], :f32)

# 2D matrix
{:ok, matrix} = ExCubecl.buffer([1.0, 2.0, 3.0, 4.0], [2, 2], :f32)

# 3D tensor (e.g. image: height x width x channels)
{:ok, image} = ExCubecl.buffer(List.duplicate(0.0, 1080 * 1920 * 3), [1080, 1920, 3], :f32)

# Integer buffer
{:ok, int_buf} = ExCubecl.buffer([10, 20, 30], [3], :s32)

# Byte buffer (e.g. raw pixel data)
{:ok, bytes} = ExCubecl.buffer([0, 128, 255], [3], :u8)
```

Supported types: `:f32`, `:f64`, `:s32`, `:s64`, `:u32`, `:u8`.

## Reading Data

```elixir
# Returns {:ok, binary}
{:ok, data} = ExCubecl.read(buf)

# Raises on error
data = ExCubecl.read!(buf)
```

## Inspection

```elixir
{:ok, [2, 2]} = ExCubecl.shape(buf)    # dimensions
{:ok, "f32"} = ExCubecl.dtype(buf)     # element type string
{:ok, 16} = ExCubecl.size(buf)         # total bytes
```

## Automatic Memory Management

Buffers are managed via Rustler's `ResourceArc` mechanism. When the Elixir
term holding a buffer reference is garbage collected, Rust's `Drop` is
automatically called to free the underlying GPU memory. **No manual `ExCubecl.free/1`
call is needed or available** — the old `free/1` function has been removed in
favor of automatic cleanup.

```elixir
buf = ExCubecl.buffer!([1.0, 2.0], [2], :f32)
data = ExCubecl.read!(buf)
# Buffer is automatically freed when `buf` is garbage collected
```

## Internal Representation

Internally, buffers are Rustler resource references (`ResourceArc<Buffer>`)
managed by the Rust NIF layer. The Elixir side holds an opaque reference term.
All data lives in Rust memory, not in the BEAM heap. Memory is automatically
released when the reference is garbage collected by the BEAM.
