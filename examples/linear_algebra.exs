# Linear Algebra Example
# Run with: mix run examples/linear_algebra.exs

alias ExCubecl.Backend, as: B

IO.puts("=== ExCubecl Linear Algebra ===")
IO.puts("")

# Matrix multiplication
IO.puts("--- Matrix Multiplication ---")
a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: B)
b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]], backend: B)

IO.puts("Matrix A:")
IO.puts("  #{inspect(Nx.to_flat_list(a) |> Enum.chunk_every(2) |> Enum.map(&inspect/1))}")

IO.puts("Matrix B:")
IO.puts("  #{inspect(Nx.to_flat_list(b) |> Enum.chunk_every(2) |> Enum.map(&inspect/1))}")

result = Nx.dot(a, b)
IO.puts("A × B:")
IO.puts("  #{inspect(Nx.to_flat_list(result) |> Enum.chunk_every(2) |> Enum.map(&inspect/1))}")
IO.puts("")

# Non-square matrix multiplication
IO.puts("--- Non-square Matrix Multiply ---")
a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: B)  # 2x3
b = Nx.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], backend: B)  # 3x2

IO.puts("A (2×3) × B (3×2):")
result = Nx.dot(a, b)
IO.puts("  Result shape: #{inspect(Nx.shape(result))}")
IO.puts("  Values: #{inspect(Nx.to_flat_list(result) |> Enum.chunk_every(2) |> Enum.map(&inspect/1))}")
IO.puts("")

# Identity matrix
IO.puts("--- Identity Matrix ---")
eye = Nx.eye({3, 3}, backend: B)
IO.puts("3×3 Identity:")
IO.puts("  #{inspect(Nx.to_flat_list(eye) |> Enum.chunk_every(3) |> Enum.map(&inspect/1))}")

# A × I = A
result = Nx.dot(a |> Nx.reshape({2, 3}), Nx.eye({3, 3}))
IO.puts("A × I = A: #{inspect(Nx.shape(result))}")
IO.puts("")

# Convolution
IO.puts("--- 2D Convolution ---")
# 3x3 input
input = Nx.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], backend: B)
# 2x2 kernel (edge detection-like)
kernel = Nx.tensor([[1.0, -1.0], [0.0, 0.0]], backend: B)

IO.puts("Input shape: #{inspect(Nx.shape(input))}")
IO.puts("Kernel shape: #{inspect(Nx.shape(kernel))}")

result = Nx.conv(input, kernel)
IO.puts("Output shape: #{inspect(Nx.shape(result))}")
IO.puts("Output: #{inspect(Nx.to_flat_list(result))}")
IO.puts("")

# Batch matrix multiply
IO.puts("--- Batch Matrix Multiply ---")
batch_a = Nx.tensor([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]], backend: B)
batch_b = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]], backend: B)

IO.puts("Batch A shape: #{inspect(Nx.shape(batch_a))}")
IO.puts("Batch B shape: #{inspect(Nx.shape(batch_b))}")

result = Nx.dot(batch_a, batch_b)
IO.puts("Result shape: #{inspect(Nx.shape(result))}")
IO.puts("")

IO.puts("=== Done ===")
