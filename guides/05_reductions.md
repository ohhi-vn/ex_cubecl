# Reductions

Reduction operations collapse one or more dimensions of a tensor, producing a scalar or a smaller tensor.

## Sum

Sum all elements or along specific axes.

```elixir
a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: ExCubecl.Backend)

# Sum all elements
Nx.sum(a)
# #Nx.Tensor<f32[1] 21.0>

# Sum along axis 0 (collapse rows)
Nx.sum(a, axes: [0])
# #Nx.Tensor<f32[3] [5.0, 7.0, 9.0]>

# Sum along axis 1 (collapse columns)
Nx.sum(a, axes: [1])
# #Nx.Tensor<f32[2] [6.0, 15.0]>

# Keep dimensions
Nx.sum(a, axes: [1], keep_axes: true)
# #Nx.Tensor<f32[2][1] [[6.0], [15.0]]
```

## Product

Multiply all elements or along specific axes.

```elixir
a = Nx.tensor([2.0, 3.0, 4.0], backend: ExCubecl.Backend)

Nx.product(a)
# #Nx.Tensor<f32[1] 24.0>

Nx.product(a, axes: [0])
# #Nx.Tensor<f32[1] 24.0>
```

## Max / Min

Find the maximum or minimum value.

```elixir
a = Nx.tensor([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]], backend: ExCubecl.Backend)

Nx.reduce_max(a)
# #Nx.Tensor<f32[1] 9.0>

Nx.reduce_min(a)
# #Nx.Tensor<f32[1] 1.0>

Nx.reduce_max(a, axes: [0])
# #Nx.Tensor<f32[3] [3.0, 5.0, 9.0]>

Nx.reduce_min(a, axes: [1])
# #Nx.Tensor<f32[2] [1.0, 1.0]>
```

## Argmax / Argmin

Find the index of the maximum or minimum value.

```elixir
a = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0], backend: ExCubecl.Backend)

Nx.argmax(a)
# #Nx.Tensor<s64[1] 5>

Nx.argmin(a)
# #Nx.Tensor<s64[1] 1>

# Along a specific axis
b = Nx.tensor([[3.0, 1.0, 4.0], [1.0, 5.0, 2.0]], backend: ExCubecl.Backend)
Nx.argmax(b, axis: 1)
# #Nx.Tensor<s64[2] [2, 1]>
```

## All / Any

Test if all or any elements are non-zero.

```elixir
a = Nx.tensor([1, 2, 3, 4], backend: ExCubecl.Backend)
b = Nx.tensor([1, 0, 3, 0], backend: ExCubecl.Backend)

Nx.all(a)   # 1 (all non-zero)
Nx.all(b)   # 0 (has zeros)
Nx.any(a)   # 1
Nx.any(Nx.tensor([0, 0, 0]))  # 0
```

## Multi-axis Reductions

```elixir
a = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], backend: ExCubecl.Backend)
# Shape: {2, 2, 2}

# Reduce along multiple axes
Nx.sum(a, axes: [0, 1])
# #Nx.Tensor<f32[2] [16.0, 20.0]>

Nx.reduce_max(a, axes: [1, 2])
# #Nx.Tensor<f32[2] [4.0, 8.0]>
```

## Keep Dimensions

Use `keep_axes: true` to maintain the reduced dimensions as size 1.

```elixir
a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: ExCubecl.Backend)

Nx.sum(a, axes: [0])
# #Nx.Tensor<f32[2] [4.0, 6.0]>

Nx.sum(a, axes: [0], keep_axes: true)
# #Nx.Tensor<f32[1][2] [[4.0, 6.0]]>
```
