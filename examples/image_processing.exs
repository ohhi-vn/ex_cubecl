# GPU Image Processing Example
# Demonstrates buffer management for image processing pipelines
# Run with: mix run examples/image_processing.exs

IO.puts("=== GPU Image Processing Pipeline ===")
IO.puts("")

# Simulate a 5x5 grayscale image as a buffer
image_data = Enum.to_list(10..250//10) |> Enum.take(25)
{:ok, image} = ExCubecl.buffer(image_data, [5, 5], :f32)

{:ok, shape} = ExCubecl.shape(image)
{:ok, size} = ExCubecl.size(image)
IO.puts("Image buffer: shape=#{inspect(shape)}, size=#{size} bytes")

# Create a padded version (7x7 with 1-pixel border)
{:ok, padded} = ExCubecl.buffer(List.duplicate(0.0, 49), [7, 7], :f32)
{:ok, shape_p} = ExCubecl.shape(padded)
IO.puts("Padded buffer: shape=#{inspect(shape_p)}")

# Create output buffer for convolution result
{:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 25), [5, 5], :f32)
{:ok, shape_o} = ExCubecl.shape(output)
IO.puts("Output buffer: shape=#{inspect(shape_o)}")

# Run a blur kernel
{:ok, _cmd} = ExCubecl.run_kernel("blur", [padded], output)
IO.puts("Blur kernel executed")

# Run edge detection
{:ok, edges} = ExCubecl.buffer(List.duplicate(0.0, 25), [5, 5], :f32)
{:ok, _cmd2} = ExCubecl.run_kernel("edge_detect", [padded], edges)
IO.puts("Edge detection kernel executed")

# Read back results
{:ok, result_data} = ExCubecl.read(output)
IO.puts("Output data length: #{byte_size(result_data)} bytes")

# Buffers are automatically freed when GC'd — no manual free needed

IO.puts("")
IO.puts("=== Done ===")
