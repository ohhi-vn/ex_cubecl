# Tensor Creation

ExCubecl provides several ways to create tensors.

## From Binary

Create a tensor directly from raw binary data.

```elixir
# Float32 tensor
binary = <<1.0::float-32, 2.0::float-32, 3.0::float-32, 4.0::float-32>>
t = Nx.from_binary(binary, {:f, 32}, backend: ExCubecl.Backend)
Nx.reshape(t, {2, 2})
# #Nx.Tensor<f32[2][2] [[1.0, 2.0], [3.0, 4.0]]

# Integer tensor
int_binary = <<1::32-native, 2::32-native, 3::32-native>>
Nx.from_binary(int_binary, {:s, 32}, backend: ExCubecl.Backend)
# #Nx.Tensor<s32[3] [1, 2, 3]>

# Unsigned 8-bit
bytes = <<255::8, 128::8, 0::8, 64::8>>
Nx.from_binary(bytes, {:u, 8}, backend: ExCubecl.Backend)
# #Nx.Tensor<u8[4] [255, 128, 0, 64]>
```

## Constant

Create a scalar tensor filled with a constant value.

```elixir
Nx.tensor(42.0, backend: ExCubecl.Backend)
# #Nx.Tensor<f32[1] 42.0>

Nx.tensor(0, backend: ExCubecl.Backend)
# #Nx.Tensor<s64[1] 0>
```

## Zeros and Ones

```elixir
Nx.broadcast(Nx.tensor(0.0, backend: ExCubecl.Backend), {3, 3})
# #Nx.Tensor<f32[3][3] [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

Nx.broadcast(Nx.tensor(1.0, backend: ExCubecl.Backend), {2, 4})
# #Nx.Tensor<f32[2][4] [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
```

## Eye (Identity Matrix)

Create an identity-like tensor.

```elixir
Nx.eye({3, 3}, backend: ExCubecl.Backend)
# #Nx.Tensor<f32[3][3] [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

# Batch of identity matrices
Nx.eye({2, 3, 3}, backend: ExCubecl.Backend)
# Shape: {2, 3, 3} - two 3x3 identity matrices
```

## Iota

Create a tensor with values equal to the index along an axis.

```elixir
# Linear indices
Nx.iota({5}, backend: ExCubecl.Backend)
# #Nx.Tensor<f32[5] [0.0, 1.0, 2.0, 3.0, 4.0]>

# 2D indices along axis 0
Nx.iota({3, 4}, axis: 0, backend: ExCubecl.Backend)
# #Nx.Tensor<f32[3][4] [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]>

# 2D indices along axis 1
Nx.iota({3, 4}, axis: 1, backend: ExCubecl.Backend)
# #Nx.Tensor<f32[3][4] [[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]]>
```

## Random Tensors (via Nx)

```elixir
# Uniform random
Nx.random_uniform({3, 3}, backend: ExCubecl.Backend)

# Normal random
Nx.random_normal({2, 4}, backend: ExCubecl.Backend)
```

## From List

```elixir
Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend: ExCubecl.Backend)
|> Nx.reshape({2, 3})
# #Nx.Tensor<f32[2][3] [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
```
