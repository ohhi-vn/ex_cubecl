# Window Operations

Window operations apply a sliding window over the tensor and compute a reduction within each window.

## Window Sum

Compute the sum within a sliding window.

```elixir
a = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], backend: ExCubecl.Backend)

# 2x2 window sum
Nx.window_sum(a, [2, 2], [])
# #Nx.Tensor<f32[1][3] [[14.0, 18.0, 22.0]]

# 1D window along axis 1
Nx.window_sum(a, [1, 2], [])
# #Nx.Tensor<f32[2][3] [[3.0, 5.0, 7.0], [11.0, 13.0, 15.0]]
```

## Window Max / Min

Compute the max or min within a sliding window.

```elixir
a = Nx.tensor([[1.0, 3.0, 2.0, 4.0], [5.0, 2.0, 6.0, 1.0]], backend: ExCubecl.Backend)

Nx.window_max(a, [1, 2], [])
# #Nx.Tensor<f32[2][3] [[3.0, 3.0, 4.0], [5.0, 6.0, 6.0]]

Nx.window_min(a, [1, 2], [])
# #Nx.Tensor<f32[2][3] [[1.0, 2.0, 2.0], [2.0, 2.0, 1.0]]
```

## Window Product (Fallback)

Window product falls back to BinaryBackend.

```elixir
Nx.window_product(a, [2, 2], [])
# Uses BinaryBackend fallback
```

## Specifying Axes

```elixir
a = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], backend: ExCubecl.Backend)

# Window along specific axes
Nx.window_sum(a, [2, 2], axes: [1, 2])
# Applies 2x2 window along axes 1 and 2
```

## Keep Dimensions

```elixir
a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: ExCubecl.Backend)

Nx.window_sum(a, [2, 2], keep_axes: true)
# Keeps reduced dimensions as size 1
```
