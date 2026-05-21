[![Hex.pm](https://img.shields.io/hexpm/v/ex_cubecl.svg)](https://hex.pm/packages/ex_cubecl)
[![Docs](https://img.shields.io/badge/docs-hexdocs-blue.svg)](https://hexdocs.pm/ex_cubecl)

> **Status:** Early development. Not yet ready for production use.

# ExCubecl

**ExCubecl** is an [Nx](https://github.com/elixir-nx/nx) backend powered by [CubeCL](https://github.com/tracel-ai/cubecl) via Rust NIFs. It provides efficient tensor operations with support for CPU computation today and GPU acceleration (via CubeCL) coming soon.

## Features

- **Nx Backend**: Full integration with the Nx tensor library
- **Rust NIFs**: High-performance tensor operations via Rust
- **Mobile Support**: C FFI layer for iOS (Objective-C/Swift) and Android (JNI)
- **Graceful Fallback**: Operations not yet implemented in NIF fall back to `Nx.BinaryBackend`
- **Type Support**: `f32`, `f64`, `s32`, `s64`, `u32`, `u8`

## Installation

Add `ex_cubecl` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:ex_cubecl, "~> 0.1.0"}
  ]
end
```

## Quick Start

```elixir
# Create tensors
a = Nx.tensor([1.0, 2.0, 3.0], backend: ExCubecl.Backend)
b = Nx.tensor([4.0, 5.0, 6.0], backend: ExCubecl.Backend)

# Basic operations
Nx.add(a, b)        # [5.0, 7.0, 9.0]
Nx.multiply(a, b)   # [4.0, 10.0, 18.0]
Nx.sum(a)           # 6.0

# Shape operations
Nx.reshape(a, {3, 1})
Nx.transpose(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))

# Reductions
Nx.sum(a, axes: [0])
Nx.argmax(a)

# Type conversion
Nx.as_type(a, {:s, 32})

# Transfer to/from other backends
binary = Nx.to_binary(a)
Nx.from_binary(binary, {:f, 32}, backend: ExCubecl.Backend)
```

## Supported Operations

| Category | Operations |
|----------|-----------|
| **Binary** | `add`, `subtract`, `multiply`, `divide`, `pow`, `remainder`, `atan2`, `min`, `max`, `quotient`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `left_shift`, `right_shift` |
| **Comparison** | `equal`, `not_equal`, `greater`, `less`, `greater_equal`, `less_equal`, `logical_and`, `logical_or`, `logical_xor` |
| **Unary** | `negate`, `abs`, `exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, `sigmoid`, `relu`, `expm1`, `log1p`, `cosh`, `sinh`, `tanh`, `acos`, `asin`, `atan`, `acosh`, `asinh`, `atanh`, `rsqrt`, `cbrt`, `erf`, `erfc`, `erf_inv`, `bitwise_not`, `ceil`, `floor`, `round`, `sign`, `conjugate`, `count_leading_zeros`, `population_count`, `real`, `imag`, `is_nan`, `is_infinity` |
| **Shape** | `reshape`, `squeeze`, `broadcast`, `transpose`, `pad`, `reverse`, `slice`, `concatenate`, `stack`, `select` |
| **Reductions** | `sum`, `product`, `reduce_max`, `reduce_min`, `all`, `any`, `argmax`, `argmin` |
| **Window** | `window_sum`, `window_max`, `window_min` |
| **LinAlg** | `dot`, `conv` |
| **Sorting** | `sort`, `argsort` |
| **Type** | `as_type`, `bitcast`, `constant`, `eye`, `iota` |
| **Indexed** | `indexed_add`, `indexed_put`, `gather`, `put_slice` |

Operations not yet implemented in the NIF layer (e.g., `fft`, `ifft`, `triangular_solve`) automatically fall back to `Nx.BinaryBackend`.

## Mobile Integration (iOS / Android)

ExCubecl includes a C FFI layer for mobile platform integration.

### iOS (Objective-C / Swift)

```objc
#include "ex_cubecl.h"

// Create tensors
float data[] = {1.0f, 2.0f, 3.0f};
size_t shape[] = {3};
ex_cubecl_tensor_handle_t a = ex_cubecl_new_tensor((const uint8_t*)data, shape, 1, EX_CUBECL_DTYPE_F32);
ex_cubecl_tensor_handle_t b = ex_cubecl_new_tensor((const uint8_t*)data, shape, 1, EX_CUBECL_DTYPE_F32);

// Add
ex_cubecl_tensor_handle_t result = ex_cubecl_add(a, b);

// Read result
float out[3];
ex_cubecl_read_tensor(result, (uint8_t*)out, sizeof(out));

// Cleanup
ex_cubecl_deallocate_tensor(a);
ex_cubecl_deallocate_tensor(b);
ex_cubecl_deallocate_tensor(result);
```

### Android (JNI)

```c
#include "ex_cubecl.h"
#include <jni.h>

JNIEXPORT jlong JNICALL
Java_com_example_excubecl_ExCubeclTensor_add(
    JNIEnv *env, jobject thiz, jlong a_handle, jlong b_handle) {
    return (jlong)ex_cubecl_add((ex_cubecl_tensor_handle_t)a_handle,
                                 (ex_cubecl_tensor_handle_t)b_handle);
}
```

See `native/ex_cubecl_nif/include/ex_cubecl.h` for the full API reference.

## Architecture

```
┌─────────────────────────────────────────────┐
│              Elixir / Nx                     │
│  Nx.add(a, b)  →  ExCubecl.Backend.add/3   │
├─────────────────────────────────────────────┤
│           ExCubecl.Backend                   │
│  - Type conversion, broadcasting, fallback   │
├─────────────────────────────────────────────┤
│           ExCubecl.NIF (Elixir)              │
│  - NIF function stubs                        │
├─────────────────────────────────────────────┤
│           Rust NIF (lib.rs)                  │
│  - Tensor operations on CPU                  │
│  - Integer-aware paths (no f64 roundtrip)    │
├─────────────────────────────────────────────┤
│           C FFI (ffi.rs + ex_cubecl.h)       │
│  - Mobile platform interface                 │
│  - Handle-based tensor management            │
└─────────────────────────────────────────────┘
```

## GPU Support (Coming Soon)

GPU acceleration via CubeCL is prepared but requires the CubeCL crate to be published with the needed features. When available, uncomment the `cubecl` dependency in `native/ex_cubecl_nif/Cargo.toml` and enable the `gpu` feature:

```bash
mix compile --features gpu
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
