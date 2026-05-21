# Indexed Operations

Indexed operations allow you to selectively read or write elements at specific positions.

## Gather

Select elements from a tensor using indices.

```elixir
input = Nx.tensor([10.0, 20.0, 30.0, 40.0, 50.0], backend: ExCubecl.Backend)
indices = Nx.tensor([0, 2, 4, 1], backend: ExCubecl.Backend)

Nx.gather(input, indices, axis: 0)
# #Nx.Tensor<f32[4] [10.0, 30.0, 50.0, 20.0]>

# 2D gather
input_2d = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], backend: ExCubecl.Backend)
indices = Nx.tensor([0, 2], backend: ExCubecl.Backend)
Nx.gather(input_2d, indices, axis: 0)
# #Nx.Tensor<f32[2][2] [[1.0, 2.0], [5.0, 6.0]]
```

## Indexed Add

Add updates to a tensor at specified indices.

```elixir
t = Nx.tensor([0.0, 0.0, 0.0, 0.0, 0.0], backend: ExCubecl.Backend)
indices = Nx.tensor([0, 2, 4], backend: ExCubecl.Backend)
updates = Nx.tensor([10.0, 20.0, 30.0], backend: ExCubecl.Backend)

Nx.indexed_add(t, indices, updates, [])
# #Nx.Tensor<f32[5] [10.0, 0.0, 20.0, 0.0, 30.0]>
```

## Indexed Put

Replace values at specified indices.

```elixir
t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], backend: ExCubecl.Backend)
indices = Nx.tensor([1, 3], backend: ExCubecl.Backend)
updates = Nx.tensor([20.0, 40.0], backend: ExCubecl.Backend)

Nx.indexed_put(t, indices, updates, [])
# #Nx.Tensor<f32[5] [1.0, 20.0, 3.0, 40.0, 5.0]>
```

## Put Slice

Place a smaller tensor into a larger one at a specific position.

```elixir
t = Nx.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], backend: ExCubecl.Backend)
slice = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: ExCubecl.Backend)

Nx.put_slice(t, [1, 1], slice)
# #Nx.Tensor<f32[3][3] [
#   [0.0, 0.0, 0.0],
#   [0.0, 1.0, 2.0],
#   [0.0, 3.0, 4.0]
# ]
```

## Select (Conditional)

Choose elements from two tensors based on a condition.

```elixir
condition = Nx.tensor([1, 0, 1, 0, 1], backend: ExCubecl.Backend)
on_true = Nx.tensor([10.0, 20.0, 30.0, 40.0, 50.0], backend: ExCubecl.Backend)
on_false = Nx.tensor([100.0, 200.0, 300.0, 400.0, 500.0], backend: ExCubecl.Backend)

Nx.select(condition, on_true, on_false)
# #Nx.Tensor<f32[5] [10.0, 200.0, 30.0, 400.0, 50.0]
```

## Scatter Operations (Fallback)

Scatter operations fall back to BinaryBackend.

```elixir
Nx.window_scatter_max(tensor, source, init, shape, opts)
Nx.window_scatter_min(tensor, source, init, shape, opts)
```
