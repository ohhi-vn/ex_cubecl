# Basic Operations Example
# Run with: mix run examples/basic_operations.exs

alias ExCubecl.Backend, as: B

IO.puts("=== ExCubecl Basic Operations ===")
IO.puts("")

# Create tensors
a = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], backend: B)
b = Nx.tensor([10.0, 20.0, 30.0, 40.0, 50.0], backend: B)

IO.puts("Tensor a: #{inspect(Nx.to_flat_list(a))}")
IO.puts("Tensor b: #{inspect(Nx.to_flat_list(b))}")
IO.puts("")

# Arithmetic
IO.puts("--- Arithmetic ---")
IO.puts("a + b = #{inspect(Nx.to_flat_list(Nx.add(a, b)))}")
IO.puts("a - b = #{inspect(Nx.to_flat_list(Nx.subtract(a, b)))}")
IO.puts("a * b = #{inspect(Nx.to_flat_list(Nx.multiply(a, b)))}")
IO.puts("a / b = #{inspect(Nx.to_flat_list(Nx.divide(a, b)))}")
IO.puts("")

# Reductions
IO.puts("--- Reductions ---")
IO.puts("sum(a) = #{Nx.sum(a) |> Nx.to_flat_list() |> hd()}")
IO.puts("max(a) = #{Nx.reduce_max(a) |> Nx.to_flat_list() |> hd()}")
IO.puts("min(a) = #{Nx.reduce_min(a) |> Nx.to_flat_list() |> hd()}")
IO.puts("argmax(a) = #{Nx.argmax(a) |> Nx.to_flat_list() |> hd()}")
IO.puts("")

# Unary operations
IO.puts("--- Unary Operations ---")
IO.puts("abs(a - b) = #{inspect(Nx.to_flat_list(Nx.abs(Nx.subtract(a, b))))}")
IO.puts("sqrt(b) = #{inspect(Nx.to_flat_list(Nx.sqrt(b)))}")
IO.puts("sigmoid(a) = #{inspect(Nx.to_flat_list(Nx.sigmoid(a)))}")
IO.puts("relu(a - 3) = #{inspect(Nx.to_flat_list(Nx.relu(Nx.subtract(a, Nx.tensor(3.0)))))}")
IO.puts("")

# Shape operations
IO.puts("--- Shape Operations ---")
m = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: B)
IO.puts("Matrix shape: #{inspect(Nx.shape(m))}")
IO.puts("Transposed shape: #{inspect(Nx.shape(Nx.transpose(m)))}")
IO.puts("Reshaped to {6}: #{inspect(Nx.shape(Nx.reshape(m, {6})))}")
IO.puts("")

# Broadcasting
IO.puts("--- Broadcasting ---")
row = Nx.tensor([10.0, 20.0, 30.0], backend: B)
IO.puts("Matrix + row vector:")
result = Nx.add(m, row)
IO.puts("  Shape: #{inspect(Nx.shape(result))}")
IO.puts("  Values: #{inspect(Nx.to_flat_list(result))}")
IO.puts("")

# Type conversion
IO.puts("--- Type Conversion ---")
int_t = Nx.as_type(a, {:s, 32})
IO.puts("a as s32: #{inspect(Nx.type(int_t))} -> #{inspect(Nx.to_flat_list(int_t))}")
back_to_float = Nx.as_type(int_t, {:f, 32})
IO.puts("Back to f32: #{inspect(Nx.type(back_to_float))} -> #{inspect(Nx.to_flat_list(back_to_float))}")
IO.puts("")

# Backend info
IO.puts("--- Backend Info ---")
IO.puts("Version: #{ExCubecl.version()}")
IO.puts("Device: #{inspect(ExCubecl.device_info())}")
IO.puts("Supported types: #{inspect(ExCubecl.supported_types())}")
IO.puts("")

IO.puts("=== Done ===")
