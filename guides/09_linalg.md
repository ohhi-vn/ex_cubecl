# Linear Algebra

ExCubecl provides basic linear algebra operations.

## Dot Product / Matrix Multiply

The `dot` operation performs matrix multiplication for 2D tensors.

```elixir
a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: ExCubecl.Backend)
b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]], backend: ExCubecl.Backend)

Nx.dot(a, b)
# #Nx.Tensor<f32[2][2] [[19.0, 22.0], [43.0, 50.0]]

# Non-square matrices
a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: ExCubecl.Backend)  # {2, 3}
b = Nx.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], backend: ExCubecl.Backend)  # {3, 2}
Nx.dot(a, b)
# #Nx.Tensor<f32[2][2] [[58.0, 64.0], [139.0, 154.0]]
```

## Convolution

2D convolution for image-like data.

```elixir
# Input: {batch, height, width}
input = Nx.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], backend: ExCubecl.Backend)

# Kernel: {height, width}
kernel = Nx.tensor([[1.0, 0.0], [0.0, -1.0]], backend: ExCubecl.Backend)

Nx.conv(input, kernel)
# Shape: {1, 2, 2} - output of valid 2D convolution
```

## Triangular Solve (Fallback)

Triangular solve falls back to BinaryBackend.

```elixir
a = Nx.tensor([[1.0, 0.0], [2.0, 3.0]], backend: ExCubecl.Backend)
b = Nx.tensor([1.0, 2.0], backend: ExCubecl.Backend)

Nx.triangular_solve(a, b)
# Uses BinaryBackend fallback
```
