# Binary Operations

Binary operations perform element-wise operations between two tensors. ExCubecl supports automatic broadcasting when tensor shapes differ.

## Arithmetic Operations

### Add
```elixir
a = Nx.tensor([1.0, 2.0, 3.0], backend: ExCubecl.Backend)
b = Nx.tensor([4.0, 5.0, 6.0], backend: ExCubecl.Backend)
Nx.add(a, b)
# #Nx.Tensor<f32[3] [5.0, 7.0, 9.0]>
```

### Subtract
```elixir
Nx.subtract(a, b)
# #Nx.Tensor<f32[3] [-3.0, -3.0, -3.0]>
```

### Multiply
```elixir
Nx.multiply(a, b)
# #Nx.Tensor<f32[3] [4.0, 10.0, 18.0]>
```

### Divide
```elixir
Nx.divide(a, b)
# #Nx.Tensor<f32[3] [0.25, 0.4, 0.5]>
```

### Power
```elixir
Nx.pow(a, b)
# #Nx.Tensor<f32[3] [1.0, 32.0, 729.0]>
```

### Remainder
```elixir
Nx.remainder(Nx.tensor([10.0, 11.0, 12.0]), Nx.tensor([3.0, 4.0, 5.0]))
# #Nx.Tensor<f32[3] [1.0, 3.0, 2.0]>
```

### Quotient (truncated division)
```elixir
Nx.quotient(Nx.tensor([10.0, 11.0, 12.0]), Nx.tensor([3.0, 4.0, 5.0]))
# #Nx.Tensor<f32[3] [3.0, 2.0, 2.0]>
```

### Atan2
```elixir
Nx.atan2(Nx.tensor([1.0, -1.0]), Nx.tensor([1.0, -1.0]))
# #Nx.Tensor<f32[3] [0.7853981852531433, -2.356194496154785]>
```

## Min / Max

```elixir
a = Nx.tensor([1.0, 5.0, 3.0], backend: ExCubecl.Backend)
b = Nx.tensor([4.0, 2.0, 6.0], backend: ExCubecl.Backend)

Nx.min(a, b)
# #Nx.Tensor<f32[3] [1.0, 2.0, 3.0]>

Nx.max(a, b)
# #Nx.Tensor<f32[3] [4.0, 5.0, 6.0]>
```

## Comparison Operations

All comparison operations return a `u8` tensor with `1` for true and `0` for false.

### Equal
```elixir
Nx.equal(Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([1.0, 0.0, 3.0]))
# #Nx.Tensor<u8[3] [1, 0, 1]>
```

### Not Equal
```elixir
Nx.not_equal(Nx.tensor([1.0, 2.0]), Nx.tensor([1.0, 3.0]))
# #Nx.Tensor<u8[2] [0, 1]>
```

### Greater / Less
```elixir
Nx.greater(Nx.tensor([5.0, 2.0]), Nx.tensor([3.0, 4.0]))
# #Nx.Tensor<u8[2] [1, 0]>

Nx.less(Nx.tensor([5.0, 2.0]), Nx.tensor([3.0, 4.0]))
# #Nx.Tensor<u8[2] [0, 1]>
```

### Greater Equal / Less Equal
```elixir
Nx.greater_equal(Nx.tensor([3.0, 2.0, 5.0]), Nx.tensor([3.0, 4.0, 1.0]))
# #Nx.Tensor<u8[3] [1, 0, 1]>

Nx.less_equal(Nx.tensor([3.0, 2.0, 5.0]), Nx.tensor([3.0, 4.0, 1.0]))
# #Nx.Tensor<u8[3] [1, 1, 0]>
```

## Logical Operations

Work on `u8` tensors. Non-zero values are treated as `true`.

```elixir
a = Nx.tensor([1, 0, 1], backend: ExCubecl.Backend)
b = Nx.tensor([1, 1, 0], backend: ExCubecl.Backend)

Nx.logical_and(a, b)  # [1, 0, 0]
Nx.logical_or(a, b)   # [1, 1, 1]
Nx.logical_xor(a, b)  # [0, 1, 1]
```

## Bitwise Operations

Work on integer tensors (`s32`, `s64`, `u32`, `u8`).

```elixir
a = Nx.from_binary(<<0xFF::32-native, 0x0F::32-native>>, {:u, 32}, backend: ExCubecl.Backend)
b = Nx.from_binary(<<0x0F::32-native, 0xF0::32-native>>, {:u, 32}, backend: ExCubecl.Backend)

Nx.bitwise_and(a, b)  # [0x0F, 0x00]
Nx.bitwise_or(a, b)   # [0xFF, 0xFF]
Nx.bitwise_xor(a, b)  # [0xF0, 0xFF]
```

### Shift Operations
```elixir
Nx.left_shift(Nx.tensor([1::32, 2::32, 4::32]), Nx.tensor([2::32, 3::32, 1::32]))
# [4, 16, 8]

Nx.right_shift(Nx.tensor([16::32, 32::32, 8::32]), Nx.tensor([2::32, 3::32, 1::32]))
# [4, 4, 4]
```

## Broadcasting

When two tensors have different shapes, ExCubecl automatically broadcasts the smaller tensor to match the larger one.

```elixir
# Scalar broadcast
a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: ExCubecl.Backend)
b = Nx.tensor([10.0, 20.0, 30.0], backend: ExCubecl.Backend)
Nx.add(a, b)
# #Nx.Tensor<f32[2][3] [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]

# Broadcasting with different dimensions
a = Nx.tensor([[1.0], [2.0], [3.0]], backend: ExCubecl.Backend)  # {3, 1}
b = Nx.tensor([10.0, 20.0, 30.0], backend: ExCubecl.Backend)      # {3}
Nx.add(a, b)
# #Nx.Tensor<f32[3][3] [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0], [13.0, 23.0, 33.0]]
```

## Integer-Aware Paths

ExCubecl uses optimized integer paths for integer types (`s32`, `s64`, `u32`, `u8`), avoiding the expensive f64 roundtrip that pure-Rust implementations typically use.

```elixir
# Integer addition stays in integer space
a = Nx.from_binary(<<1::32-native, 2::32-native, 3::32-native>>, {:s, 32}, backend: ExCubecl.Backend)
b = Nx.from_binary(<<10::32-native, 20::32-native, 30::32-native>>, {:s, 32}, backend: ExCubecl.Backend)
Nx.add(a, b)
# #Nx.Tensor<s32[3] [11, 22, 33]>
```
