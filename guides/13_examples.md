# Examples Overview

This guide provides practical examples of using ExCubecl for common tasks.

## Running Examples

All examples are in the `examples/` directory. Run them with:

```bash
mix run examples/basic_operations.exs
mix run examples/linear_algebra.exs
mix run examples/ml_pipeline.exs
mix run examples/image_processing.exs
```

## Example 1: Basic Operations

Demonstrates arithmetic, reductions, unary ops, shape ops, broadcasting, and type conversion.

```elixir
# Create tensors
a = Nx.tensor([1.0, 2.0, 3.0], backend: ExCubecl.Backend)
b = Nx.tensor([4.0, 5.0, 6.0], backend: ExCubecl.Backend)

# Arithmetic
Nx.add(a, b)        # [5.0, 7.0, 9.0]
Nx.multiply(a, b)   # [4.0, 10.0, 18.0]

# Reductions
Nx.sum(a)           # 6.0
Nx.argmax(a)        # 2

# Broadcasting
m = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: ExCubecl.Backend)
Nx.add(m, Nx.tensor([10.0, 20.0, 30.0]))
# [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]
```

See: [examples/basic_operations.exs](../examples/basic_operations.exs)

## Example 2: Linear Algebra

Matrix operations, convolution, and batch processing.

```elixir
# Matrix multiply
a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: ExCubecl.Backend)
b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]], backend: ExCubecl.Backend)
Nx.dot(a, b)
# [[19.0, 22.0], [43.0, 50.0]]

# 2D Convolution
input = Nx.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], backend: ExCubecl.Backend)
kernel = Nx.tensor([[1.0, 0.0], [0.0, -1.0]], backend: ExCubecl.Backend)
Nx.conv(input, kernel)
```

See: [examples/linear_algebra.exs](../examples/linear_algebra.exs)

## Example 3: ML Pipeline

Neural network forward pass, batch processing, activation functions, and loss computation.

```elixir
# Forward pass: y = sigmoid(Wx + b)
x = Nx.tensor([0.5, -0.3, 0.8], backend: ExCubecl.Backend)
w = Nx.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], backend: ExCubecl.Backend)
b = Nx.tensor([0.1, -0.1], backend: ExCubecl.Backend)

z = Nx.add(Nx.dot(w, x), b)
output = Nx.sigmoid(z)

# MSE Loss
predicted = Nx.tensor([0.8, 0.2, 0.9], backend: ExCubecl.Backend)
target = Nx.tensor([1.0, 0.0, 1.0], backend: ExCubecl.Backend)
mse = Nx.sum(Nx.multiply(Nx.subtract(predicted, target), Nx.subtract(predicted, target)))
```

See: [examples/ml_pipeline.exs](../examples/ml_pipeline.exs)

## Example 4: Image Processing

Convolution, padding, pooling, cropping, and strided operations on image-like data.

```elixir
# Padding
image = Nx.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]], backend: ExCubecl.Backend)
padded = Nx.pad(image, 0.0, [{1, 1, 0}, {1, 1, 0}])

# Max pooling via window_max
Nx.window_max(image, [2, 2], [])

# Cropping via slice
Nx.slice(image, [0, 0], [2, 2], [1, 1])

# Downsampling via strided slice
Nx.slice(image, [0, 0], [2, 2], [2, 2])
```

See: [examples/image_processing.exs](../examples/image_processing.exs)

## Common Patterns

### Chaining Operations

```elixir
# Normalize: (x - mean) / std
x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], backend: ExCubecl.Backend)
mean = Nx.sum(x) |> Nx.divide(Nx.tensor(5.0))
diff = Nx.subtract(x, mean)
variance = Nx.sum(Nx.multiply(diff, diff)) |> Nx.divide(Nx.tensor(5.0))
std = Nx.sqrt(variance)
normalized = Nx.divide(diff, std)
```

### Conditional Selection

```elixir
# ReLU via select
x = Nx.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], backend: ExCubecl.Backend)
mask = Nx.greater(x, Nx.tensor(0.0))
Nx.select(mask, x, Nx.tensor(0.0))
# [0.0, 0.0, 0.0, 1.0, 2.0]
```

### Top-K with Argsort

```elixir
logits = Nx.tensor([0.1, 2.5, 1.0, 3.5, 0.5], backend: ExCubecl.Backend)
indices = Nx.argsort(logits)
top3 = Nx.slice(indices, [2], [3], [1])  # Last 3 = top 3
Nx.gather(logits, top3, axis: 0)
```

### Transfer Between Backends

```elixir
# Create on ExCubecl, compute on BinaryBackend, transfer back
t = Nx.tensor([1.0, 2.0, 3.0], backend: ExCubecl.Backend)
binary = Nx.backend_transfer(t, Nx.BinaryBackend)
result = Nx.some_operation(binary)
back = Nx.backend_transfer(result, ExCubecl.Backend)
```
