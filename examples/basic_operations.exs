# Basic GPU Buffer Operations Example
# Run with: mix run examples/basic_operations.exs

IO.puts("=== ExCubecl Basic Operations ===")
IO.puts("")

# Check device
{:ok, info} = ExCubecl.device_info()
IO.puts("Device: #{info.device_name}")
IO.puts("")

# Create buffers from lists
{:ok, a} = ExCubecl.buffer([1.0, 2.0, 3.0, 4.0, 5.0], [5], :f32)
{:ok, b} = ExCubecl.buffer([10.0, 20.0, 30.0, 40.0, 50.0], [5], :f32)

{:ok, shape_a} = ExCubecl.shape(a)
{:ok, dtype_a} = ExCubecl.dtype(a)
{:ok, size_a} = ExCubecl.size(a)
IO.puts("Buffer a: shape=#{inspect(shape_a)}, dtype=#{dtype_a}")
IO.puts("Buffer a size: #{size_a} bytes")
IO.puts("")

# Read back data
{:ok, data} = ExCubecl.read(a)
IO.puts("Data from buffer a: #{byte_size(data)} bytes")
IO.puts("")

# 2D buffer (matrix)
{:ok, matrix} = ExCubecl.buffer([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], :f32)
{:ok, shape_m} = ExCubecl.shape(matrix)
{:ok, dtype_m} = ExCubecl.dtype(matrix)
IO.puts("Matrix shape: #{inspect(shape_m)}")
IO.puts("Matrix dtype: #{dtype_m}")
IO.puts("")

# Integer buffer
{:ok, int_buf} = ExCubecl.buffer([1, 2, 3, 4], [4], :s32)
{:ok, dtype_i} = ExCubecl.dtype(int_buf)
{:ok, size_i} = ExCubecl.size(int_buf)
IO.puts("Integer buffer dtype: #{dtype_i}")
IO.puts("Integer buffer size: #{size_i} bytes")
IO.puts("")

# List available kernels
{:ok, kernels} = ExCubecl.kernels()
IO.puts("Available kernels: #{inspect(kernels)}")
IO.puts("")

# Cleanup
ExCubecl.free(a)
ExCubecl.free(b)
ExCubecl.free(matrix)
ExCubecl.free(int_buf)

IO.puts("=== Done ===")
