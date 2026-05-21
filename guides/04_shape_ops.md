# Shape Operations

Shape operations modify the dimensions and layout of tensors without changing their data.

## Reshape

Change the shape of a tensor while keeping the same data and number of elements.

```elixir
a = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend: ExCubecl.Backend)

Nx.reshape(a, {2, 3})
# #Nx.Tensor<f32[2][3] [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

Nx.reshape(a, {3, 2})
# #Nx.Tensor<f32[3][2] [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

# Use -1 to infer a dimension
Nx.reshape(a, {-1, 2})
# #Nx.Tensor<f32[3][2] [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
```

## Transpose

Reverse or permute the axes of a tensor.

```elixir
a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: ExCubecl.Backend)

# Default: reverse all axes
Nx.transpose(a)
# #Nx.Tensor<f32[3][2] [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

# Specify axis permutation
Nx.transpose(a, axes: [1, 0])
# #Nx.Tensor<f32[3][2] [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
```

## Broadcast

Explicitly broadcast a tensor to a target shape.

```elixir
a = Nx.tensor([1.0, 2.0, 3.0], backend: ExCubecl.Backend)

Nx.broadcast(a, {3, 3})
# #Nx.Tensor<f32[3][3] [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
```

## Squeeze

Remove dimensions of size 1.

```elixir
a = Nx.tensor([[[1.0, 2.0, 3.0]]], backend: ExCubecl.Backend)  # {1, 1, 3}

Nx.squeeze(a)
# #Nx.Tensor<f32[3] [1.0, 2.0, 3.0]

# Squeeze specific axes
Nx.squeeze(a, axes: [0])
# #Nx.Tensor<f32[1][3] [[1.0, 2.0, 3.0]]
```

## Pad

Pad a tensor with a constant value.

```elixir
a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: ExCubecl.Backend)

# Pad with 0s: 1 row top, 1 row bottom, 1 col left, 1 col right
Nx.pad(a, 0.0, [{1, 1, 0}, {1, 1, 0}])
# #Nx.Tensor<f32[4][4] [
#   [0.0, 0.0, 0.0, 0.0],
#   [0.0, 1.0, 2.0, 0.0],
#   [0.0, 3.0, 4.0, 0.0],
#   [0.0, 0.0, 0.0, 0.0]
# ]

# Padding config: {before, after, interior}
# interior padding inserts between elements
Nx.pad(a, 0.0, [{0, 0, 1}, {0, 0, 0}])
# #Nx.Tensor<f32[3][2] [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]]
```

## Reverse

Reverse elements along specified axes.

```elixir
a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: ExCubecl.Backend)

# Reverse all elements
Nx.reverse(a)
# #Nx.Tensor<f32[2][3] [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]

# Reverse along axis 0 (rows)
Nx.reverse(a, axes: [0])
# #Nx.Tensor<f32[2][3] [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]

# Reverse along axis 1 (columns)
Nx.reverse(a, axes: [1])
# #Nx.Tensor<f32[2][3] [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]
```

## Slice

Extract a sub-tensor.

```elixir
a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], backend: ExCubecl.Backend)

# slice(starts, lengths, strides)
Nx.slice(a, [0, 0], [2, 2], [1, 1])
# #Nx.Tensor<f32[2][2] [[1.0, 2.0], [4.0, 5.0]]

Nx.slice(a, [1, 1], [2, 2], [1, 1])
# #Nx.Tensor<f32[2][2] [[5.0, 6.0], [8.0, 9.0]]

# With strides (every other element)
Nx.slice(a, [0, 0], [2, 2], [2, 2])
# #Nx.Tensor<f32[2][2] [[1.0, 3.0], [7.0, 9.0]]
```

## Concatenate

Join tensors along an axis.

```elixir
a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: ExCubecl.Backend)
b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]], backend: ExCubecl.Backend)

# Concatenate along axis 0 (rows)
Nx.concatenate([a, b], axis: 0)
# #Nx.Tensor<f32[4][2] [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]

# Concatenate along axis 1 (columns)
Nx.concatenate([a, b], axis: 1)
# #Nx.Tensor<f32[2][4] [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]
```

## Stack

Join tensors along a new axis.

```elixir
a = Nx.tensor([1.0, 2.0, 3.0], backend: ExCubecl.Backend)
b = Nx.tensor([4.0, 5.0, 6.0], backend: ExCubecl.Backend)

Nx.stack([a, b], axis: 0)
# #Nx.Tensor<f2[3] [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

Nx.stack([a, b], axis: 1)
# #Nx.Tensor<f32[3][2] [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
```

## Select

Choose elements from two tensors based on a predicate.

```elixir
pred = Nx.tensor([1, 0, 1, 0], backend: ExCubecl.Backend)
on_true = Nx.tensor([10.0, 20.0, 30.0, 40.0], backend: ExCubecl.Backend)
on_false = Nx.tensor([100.0, 200.0, 300.0, 400.0], backend: ExCubecl.Backend)

Nx.select(pred, on_true, on_false)
# #Nx.Tensor<f32[4] [10.0, 200.0, 30.0, 400.0]
```

## Put Slice

Place a smaller tensor into a larger one at specified indices.

```elixir
t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: ExCubecl.Backend)
slice = Nx.tensor([[10.0, 20.0]], backend: ExCubecl.Backend)

Nx.put_slice(t, [0, 0], slice)
# #Nx.Tensor<f32[2][3] [[10.0, 20.0, 3.0], [4.0, 5.0, 6.0]]
```
