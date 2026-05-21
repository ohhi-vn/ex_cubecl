# Sorting

ExCubecl supports sorting tensors along specified axes.

## Sort

Sort elements along an axis in ascending order.

```elixir
a = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0], backend: ExCubecl.Backend)

# Sort entire tensor
Nx.sort(a)
# #Nx.Tensor<f32[7] [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 9.0]>

# Sort along axis 0 (rows)
b = Nx.tensor([[3.0, 1.0], [6.0, 4.0], [2.0, 8.0]], backend: ExCubecl.Backend)
Nx.sort(b, axis: 0)
# #Nx.Tensor<f32[3][2] [[2.0, 1.0], [3.0, 4.0], [6.0, 8.0]]

# Sort along axis 1 (columns)
Nx.sort(b, axis: 1)
# #Nx.Tensor<f32[3][2] [[1.0, 3.0], [4.0, 6.0], [2.0, 8.0]>
```

## Argsort

Return the indices that would sort the tensor.

```elixir
a = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0], backend: ExCubecl.Backend)

Nx.argsort(a)
# #Nx.Tensor<s64[5] [1, 3, 0, 2, 4]>

# Use argsort to reorder another tensor
indices = Nx.argsort(a)
# indices tell us: element at position 1 is smallest, then position 3, etc.
```

## Top-K Pattern

Combine argsort with slice to get top-k elements.

```elixir
a = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], backend: ExCubecl.Backend)

# Get indices of top 3 largest values
indices = Nx.argsort(a)
# [1, 3, 6, 0, 2, 4, 7, 5]

# Top 3 values (last 3 in sorted order)
top3_indices = Nx.slice(indices, [5], [3], [1])
# [7, 5] - indices of 6.0 and 9.0
```
