# Type Conversion

ExCubecl supports converting between different tensor data types and bit-level reinterpretation.

## as_type

Convert a tensor to a different data type.

```elixir
a = Nx.tensor([1.5, 2.7, 3.0], backend: ExCubecl.Backend)

# Float to integer (truncates)
Nx.as_type(a, {:s, 32})
# #Nx.Tensor<s32[3] [1, 2, 3]>

# Integer to float
b = Nx.from_binary(<<10::32-native, 20::32-native>>, {:s, 32}, backend: ExCubecl.Backend)
Nx.as_type(b, {:f, 32})
# #Nx.Tensor<f32[2] [10.0, 20.0]>

# Float32 to Float64
Nx.as_type(Nx.tensor([1.5, 2.5]), {:f, 64})
# #Nx.Tensor<f64[2] [1.5, 2.5]>

# Float64 to Float32
Nx.as_type(Nx.tensor([1.5, 2.5], type: {:f, 64}), {:f, 32})
# #Nx.Tensor<f32[2] [1.5, 2.5]>
```

## Bitcast

Reinterpret the bits of a tensor as a different type without changing the data.

```elixir
# Reinterpret f32 bits as u32
a = Nx.tensor([1.0, 0.0, -0.0], backend: ExCubecl.Backend)
Nx.bitcast(a, {:u, 32})
# #Nx.Tensor<u32[3] [1065353216, 0, 2147483648]>

# Reinterpret u8 as s8
b = Nx.from_binary(<<255::8, 128::8, 0::8>>, {:u, 8}, backend: ExCubecl.Backend)
Nx.bitcast(b, {:s, 8})
# #Nx.Tensor<s8[3] [-1, -128, 0]>
```

## Type Promotion

When operating on tensors of different types, Nx automatically promotes to the wider type.

```elixir
a = Nx.tensor([1.0, 2.0, 3.0], backend: ExCubecl.Backend)  # f32
b = Nx.from_binary(<<10::32-native, 20::32-native, 30::32-native>>, {:s, 32}, backend: ExCubecl.Backend)  # s32

# Result is f32 (wider type)
Nx.add(a, b)
# #Nx.Tensor<f32[3] [11.0, 22.0, 33.0]>
```

## Supported Type Conversions

| From → To | Behavior |
|-----------|----------|
| `f32 → f64` | Widens precision |
| `f64 → f32` | Narrows precision (may lose precision) |
| `f32 → s32` | Truncates to integer |
| `s32 → f32` | Converts to float |
| `u8 → s32` | Zero-extends |
| `s32 → u8` | Truncates to 8 bits |
| `f32 → u32` | Bitcast (reinterprets bits) |
| `u32 → f32` | Bitcast (reinterprets bits) |

## Clipping

Clip values to a range.

```elixir
a = Nx.tensor([1.0, 5.0, 10.0, 15.0, 20.0], backend: ExCubecl.Backend)

Nx.clip(a, Nx.tensor(5.0), Nx.tensor(15.0))
# #Nx.Tensor<f32[5] [5.0, 5.0, 10.0, 15.0, 15.0]>
```
