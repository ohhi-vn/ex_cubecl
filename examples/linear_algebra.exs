# GPU Linear Algebra Example
# Demonstrates matrix operations via GPU buffers
# Run with: mix run examples/linear_algebra.exs

IO.puts("=== GPU Linear Algebra ===")
IO.puts("")

# Create 2x3 matrix
{:ok, a} = ExCubecl.buffer([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], :f32)
{:ok, shape_a} = ExCubecl.shape(a)
IO.puts("Matrix A: shape=#{inspect(shape_a)}")

# Create 3x2 matrix
{:ok, b} = ExCubecl.buffer([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2], :f32)
{:ok, shape_b} = ExCubecl.shape(b)
IO.puts("Matrix B: shape=#{inspect(shape_b)}")

# Output buffer for matmul result (2x2)
{:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 4), [2, 2], :f32)
{:ok, shape_o} = ExCubecl.shape(output)
IO.puts("Output buffer: shape=#{inspect(shape_o)}")

# Run matmul kernel
{:ok, _cmd} = ExCubecl.run_kernel("matmul", [a, b], output, %{})
IO.puts("Matmul kernel executed")

# Read result
{:ok, result} = ExCubecl.read(output)
IO.puts("Result data: #{byte_size(result)} bytes")

# Identity matrix
{:ok, eye} = ExCubecl.buffer([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3], :f32)
{:ok, shape_e} = ExCubecl.shape(eye)
IO.puts("Identity matrix: shape=#{inspect(shape_e)}")

# Cleanup
ExCubecl.free(a)
ExCubecl.free(b)
ExCubecl.free(output)
ExCubecl.free(eye)

IO.puts("")
IO.puts("=== Done ===")
