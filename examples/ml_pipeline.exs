# Machine Learning Pipeline Example
# Demonstrates a simple forward pass through a neural network layer
# Run with: mix run examples/ml_pipeline.exs

alias ExCubecl.Backend, as: B

IO.puts("=== ML Pipeline: Neural Network Forward Pass ===")
IO.puts("")

# Simulate a simple neural network layer: y = sigmoid(Wx + b)
# Input: 3 features
# Output: 2 neurons

# Input vector
x = Nx.tensor([0.5, -0.3, 0.8], backend: B)
IO.puts("Input: #{inspect(Nx.to_flat_list(x))}")

# Weight matrix (2 neurons × 3 inputs)
w =
  Nx.tensor(
    [
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6]
    ],
    backend: B
  )

IO.puts("Weights shape: #{inspect(Nx.shape(w))}")

# Bias (2 neurons)
b = Nx.tensor([0.1, -0.1], backend: B)
IO.puts("Bias: #{inspect(Nx.to_flat_list(b))}")

# Forward pass: z = Wx + b
z = Nx.dot(w, x)
IO.puts("Wx: #{inspect(Nx.to_flat_list(z))}")

z = Nx.add(z, b)
IO.puts("Wx + b (pre-activation): #{inspect(Nx.to_flat_list(z))}")

# Activation: a = sigmoid(z)
a = Nx.sigmoid(z)
IO.puts("sigmoid(Wx + b) (output): #{inspect(Nx.to_flat_list(a))}")
IO.puts("")

# --- Batch Processing ---
IO.puts("--- Batch Processing ---")

# Batch of 4 samples, each with 3 features
batch_x =
  Nx.tensor(
    [
      [0.5, -0.3, 0.8],
      [1.0, 0.0, -0.5],
      [-0.2, 0.7, 0.1],
      [0.3, 0.3, 0.3]
    ],
    backend: B
  )

IO.puts("Batch input shape: #{inspect(Nx.shape(batch_x))}")

# For batch: z = x @ W^T + b
# Transpose weights for batch multiply
w_t = Nx.transpose(w)
IO.puts("Transposed weights shape: #{inspect(Nx.shape(w_t))}")

batch_z = Nx.dot(batch_x, w_t)
IO.puts("Batch Wx shape: #{inspect(Nx.shape(batch_z))}")

# Broadcast bias across batch
batch_z = Nx.add(batch_z, b)
IO.puts("Batch Wx + b shape: #{inspect(Nx.shape(batch_z))}")

batch_a = Nx.sigmoid(batch_z)
IO.puts("Batch output shape: #{inspect(Nx.shape(batch_a))}")
IO.puts("Batch output:")

Nx.to_flat_list(batch_a)
|> Enum.chunk_every(2)
|> Enum.with_index(1)
|> Enum.each(fn {row, i} ->
  IO.puts("  Sample #{i}: #{inspect(row)}")
end)

IO.puts("")

# --- ReLU Activation ---
IO.puts("--- ReLU Activation ---")
pre_relu = Nx.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], backend: B)
activated = Nx.relu(pre_relu)
IO.puts("Pre-ReLU:  #{inspect(Nx.to_flat_list(pre_relu))}")
IO.puts("Post-ReLU: #{inspect(Nx.to_flat_list(activated))}")
IO.puts("")

# --- Loss Computation (MSE) ---
IO.puts("--- MSE Loss ---")
predicted = Nx.tensor([0.8, 0.2, 0.9], backend: B)
target = Nx.tensor([1.0, 0.0, 1.0], backend: B)

diff = Nx.subtract(predicted, target)
squared = Nx.multiply(diff, diff)
mse = Nx.sum(squared)

IO.puts("Predicted: #{inspect(Nx.to_flat_list(predicted))}")
IO.puts("Target:    #{inspect(Nx.to_flat_list(target))}")
IO.puts("MSE Loss:  #{Nx.to_flat_list(mse) |> hd()}")
IO.puts("")

# --- Top-K Selection ---
IO.puts("--- Top-K Selection ---")
logits = Nx.tensor([0.1, 2.5, 1.0, 3.5, 0.5, 1.5], backend: B)
IO.puts("Logits: #{inspect(Nx.to_flat_list(logits))}")

sorted_indices = Nx.argsort(logits)
IO.puts("Sorted indices: #{inspect(Nx.to_flat_list(sorted_indices))}")

# Get top 3
top3 = Nx.slice(sorted_indices, [3], [3], [1])
IO.puts("Top-3 indices: #{inspect(Nx.to_flat_list(top3))}")

top3_values = Nx.gather(logits, top3, axis: 0)
IO.puts("Top-3 values: #{inspect(Nx.to_flat_list(top3_values))}")
IO.puts("")

IO.puts("=== Done ===")
