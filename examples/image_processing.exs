# Image Processing Example
# Demonstrates convolution, padding, and shape operations on image-like data
# Run with: mix run examples/image_processing.exs

alias ExCubecl.Backend, as: B

IO.puts("=== Image Processing Operations ===")
IO.puts("")

# Create a small 5x5 "image" (single channel)
image = Nx.tensor([
  [10, 20, 30, 40, 50],
  [60, 70, 80, 90, 100],
  [110, 120, 130, 140, 150],
  [160, 170, 180, 190, 200],
  [210, 220, 230, 240, 250]
], backend: B)

IO.puts("Original image (5×5):")
IO.puts("  Shape: #{inspect(Nx.shape(image))}")
IO.puts("  Values: #{inspect(Nx.to_flat_list(image))}")
IO.puts("")

# Add batch dimension: {1, 5, 5}
batched = Nx.reshape(image, {1, 5, 5})
IO.puts("Batched shape: #{inspect(Nx.shape(batched))}")
IO.puts("")

# --- Padding ---
IO.puts("--- Padding ---")
# Pad with 1 pixel on each side
padded = Nx.pad(image, 0.0, [{1, 1, 0}, {1, 1, 0}])
IO.puts("Padded shape: #{inspect(Nx.shape(padded))}")
IO.puts("Padded values:")
Nx.to_flat_list(padded)
|> Enum.chunk_every(7)
|> Enum.each(fn row ->
  IO.puts("  #{inspect(row)}")
end)
IO.puts("")

# --- Edge Detection Kernel ---
IO.puts("--- Edge Detection (Sobel-like) ---")
# 3x3 kernel for horizontal edge detection
kernel = Nx.tensor([
  [-1.0, 0.0, 1.0],
  [-2.0, 0.0, 2.0],
  [-1.0, 0.0, 1.0]
], backend: B)

IO.puts("Kernel shape: #{inspect(Nx.shape(kernel))}")

# Apply convolution
batched_padded = Nx.reshape(padded, {1, 7, 7})
result = Nx.conv(batched_padded, kernel)
IO.puts("Conv result shape: #{inspect(Nx.shape(result))}")
IO.puts("")

# --- Blur Kernel ---
IO.puts("--- Box Blur ---")
blur_kernel = Nx.tensor([
  [1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0]
], backend: B)

blur_result = Nx.conv(batched_padded, blur_kernel)
IO.puts("Blur result shape: #{inspect(Nx.shape(blur_result))}")
IO.puts("")

# --- Window Operations (Pooling) ---
IO.puts("--- Max Pooling (2×2) ---")
small = Nx.tensor([
  [1.0, 3.0, 2.0, 4.0],
  [5.0, 6.0, 1.0, 2.0],
  [3.0, 2.0, 7.0, 8.0],
  [1.0, 0.0, 4.0, 3.0]
], backend: B)

IO.puts("Input (4×4):")
Nx.to_flat_list(small)
|> Enum.chunk_every(4)
|> Enum.each(fn row -> IO.puts("  #{inspect(row)}") end)

pooled = Nx.window_max(small, [2, 2], [])
IO.puts("Max pooled (2×2):")
Nx.to_flat_list(pooled)
|> Enum.chunk_every(2)
|> Enum.each(fn row -> IO.puts("  #{inspect(row)}") end)
IO.puts("")

# --- Average via Window Sum ---
IO.puts("--- Average Pooling (2×2) ---")
sum_pooled = Nx.window_sum(small, [2, 2], [])
# Divide by 4 to get average
avg_pooled = Nx.divide(sum_pooled, Nx.tensor(4.0))
IO.puts("Average pooled (2×2):")
Nx.to_flat_list(avg_pooled)
|> Enum.chunk_every(2)
|> Enum.each(fn row -> IO.puts("  #{inspect(row)}") end)
IO.puts("")

# --- Slice (Crop) ---
IO.puts("--- Image Cropping ---")
cropped = Nx.slice(image, [1, 1], [3, 3], [1, 1])
IO.puts("Cropped (3×3 from center):")
Nx.to_flat_list(cropped)
|> Enum.chunk_every(3)
|> Enum.each(fn row -> IO.puts("  #{inspect(row)}") end)
IO.puts("")

# --- Strided Slice (Downsample) ---
IO.puts("--- Strided Slice (Downsample 2x) ---")
downsampled = Nx.slice(image, [0, 0], [3, 3], [2, 2])
IO.puts("Downsampled (every other pixel):")
Nx.to_flat_list(downsampled)
|> Enum.chunk_every(3)
|> Enum.each(fn row -> IO.puts("  #{inspect(row)}") end)
IO.puts("")

IO.puts("=== Done ===")
