# Getting Started with ExCubecl

## What is ExCubecl?

ExCubecl is an [Nx](https://github.com/elixir-nx/nx) backend that powers tensor operations through Rust NIFs (Native Implemented Functions). It provides a high-performance compute layer for Elixir's numerical computing ecosystem, with support for CPU computation today and GPU acceleration (via CubeCL) planned for the future.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Elixir / Nx                         │
│   Nx.add(a, b)  →  ExCubecl.Backend.add/3           │
├─────────────────────────────────────────────────────┤
│              ExCubecl.Backend                        │
│   - Type conversion, broadcasting, fallback          │
├─────────────────────────────────────────────────────┤
│              ExCubecl.NIF (Elixir)                   │
│   - NIF function stubs                               │
├─────────────────────────────────────────────────────┤
│              Rust NIF (lib.rs)                       │
│   - Tensor operations on CPU                         │
│   - Integer-aware paths (no f64 roundtrip)           │
├─────────────────────────────────────────────────────┤
│              C FFI (ffi.rs + ex_cubecl.h)            │
│   - Mobile platform interface (iOS/Android)          │
└─────────────────────────────────────────────────────┘
```

## Installation

Add `ex_cubecl` to your `mix.exs`:

```elixir
def deps do
  [
    {:ex_cubecl, "~> 0.1.0"}
  ]
end
```

Then run:

```bash
mix deps.get
mix compile
```

## Quick Start

```elixir
# Create a tensor
t = Nx.tensor([1.0, 2.0, 3.0], backend: ExCubecl.Backend)

# Basic arithmetic
Nx.add(t, t)        # [2.0, 4.0, 6.0]
Nx.multiply(t, t)   # [1.0, 4.0, 9.0]

# Shape operations
Nx.reshape(t, {3, 1})
Nx.transpose(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))

# Reductions
Nx.sum(t)           # 6.0
Nx.argmax(t)        # 2

# Type conversion
Nx.as_type(t, {:s, 32})

# Transfer between backends
binary = Nx.to_binary(t)
Nx.from_binary(binary, {:f, 32}, backend: ExCubecl.Backend)
```

## Supported Data Types

| Type    | Description                | Size    |
|---------|----------------------------|---------|
| `{:f, 32}` | 32-bit float             | 4 bytes |
| `{:f, 64}` | 64-bit float             | 8 bytes |
| `{:s, 32}` | 32-bit signed integer    | 4 bytes |
| `{:s, 64}` | 64-bit signed integer    | 8 bytes |
| `{:u, 32}` | 32-bit unsigned integer  | 4 bytes |
| `{:u, 8}`  | 8-bit unsigned integer   | 1 byte  |

Note: `{:f, 16}` and `{:bf, 16}` are automatically converted to `{:f, 32}`.

## Hardware Support

### Currently Active
- **CPU**: All platforms (macOS x86_64/ARM64, Linux x86_64, Windows x86_64)
- **iOS**: C FFI via static library (Objective-C / Swift bridging)
- **Android**: C FFI via static library (JNI)

### Planned
- **GPU**: Via CubeCL/WGPU (macOS Metal, Linux Vulkan, Windows DirectX/Vulkan, iOS Metal, Android Vulkan)

## Backend Fallback

Operations not yet implemented in the NIF layer automatically fall back to `Nx.BinaryBackend`. This means you can use the full Nx API even if some operations aren't accelerated yet.

```elixir
# This will use BinaryBackend fallback if FFT is not in Nx
Nx.fft(tensor)
```

## Checking Backend Availability

```elixir
# Check if NIF is loaded
ExCubecl.available?()

# Get device info
ExCubecl.device_info()
# %{backend: :cpu, name: "ExCubecl CPU (Rust NIF)", version: "0.1.0", gpu: false, mobile_ffi: true}

# List supported types
ExCubecl.supported_types()
# [{:f, 32}, {:f, 64}, {:s, 32}, {:s, 64}, {:u, 32}, {:u, 8}]
```

## Next Guides

- [Binary Operations](02_binary_ops.md)
- [Unary Operations](03_unary_ops.md)
- [Shape Operations](04_shape_ops.md)
- [Reductions](05_reductions.md)
- [Type Conversion](06_type_conversion.md)
- [Creation Helpers](07_creation.md)
- [Sorting](08_sorting.md)
- [Linear Algebra](09_linalg.md)
- [Window Operations](10_window_ops.md)
- [Indexed Operations](11_indexed_ops.md)
- [Mobile Integration](12_mobile.md)
- [Examples](13_examples.md)
