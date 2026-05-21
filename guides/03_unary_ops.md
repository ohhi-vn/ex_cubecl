# Unary Operations

Unary operations perform element-wise transformations on a single tensor.

## Basic Math

### Negate
```elixir
a = Nx.tensor([1.0, -2.0, 3.0], backend: ExCubecl.Backend)
Nx.negate(a)
# #Nx.Tensor<f32[3] [-1.0, 2.0, -3.0]>
```

### Absolute Value
```elixir
Nx.abs(a)
# #Nx.Tensor<f32[3] [1.0, 2.0, 3.0]>
```

### Sign
```elixir
Nx.sign(Nx.tensor([-5.0, 0.0, 3.0]))
# #Nx.Tensor<f32[3] [-1.0, 0.0, 1.0]>
```

## Exponential & Logarithmic

### Exp
```elixir
Nx.exp(Nx.tensor([0.0, 1.0, 2.0]))
# #Nx.Tensor<f32[3] [1.0, 2.7182817459106445, 7.389056205749512]>
```

### Log (natural logarithm)
```elixir
Nx.log(Nx.tensor([1.0, 2.7182818, 7.389056]))
# #Nx.Tensor<f32[3] [0.0, 1.0, 2.0]>
```

### Expm1 (e^x - 1, more accurate for small x)
```elixir
Nx.expm1(Nx.tensor([0.0, 0.001, 1.0]))
# #Nx.Tensor<f32[3] [0.0, 0.0010005003213882446, 1.718281865119934]>
```

### Log1p (ln(1 + x), more accurate for small x)
```elixir
Nx.log1p(Nx.tensor([0.0, 0.001, 1.0]))
# #Nx.Tensor<f32[3] [0.0, 0.0009995003213882446, 0.6931471824645996]>
```

## Roots

### Square Root
```elixir
Nx.sqrt(Nx.tensor([4.0, 9.0, 16.0]))
# #Nx.Tensor<f32[3] [2.0, 3.0, 4.0]>
```

### Reciprocal Square Root (1/√x)
```elixir
Nx.rsqrt(Nx.tensor([4.0, 9.0, 16.0]))
# #Nx.Tensor<f32[3] [0.5, 0.3333333432674408, 0.25]>
```

### Cube Root
```elixir
Nx.cbrt(Nx.tensor([8.0, 27.0, 64.0]))
# #Nx.Tensor<f32[3] [2.0, 3.0, 4.0]>
```

## Trigonometric Functions

All trigonometric functions work in radians.

### Sin / Cos / Tan
```elixir
x = Nx.tensor([0.0, :math.pi() / 4, :math.pi() / 2], backend: ExCubecl.Backend)

Nx.sin(x)  # [0.0, 0.7071067690849304, 1.0]
Nx.cos(x)  # [1.0, 0.7071067690849304, 0.0]
Nx.tan(x)  # [0.0, 1.0, 1.633123935319537e16]
```

### Inverse Trigonometric
```elixir
Nx.asin(Nx.tensor([0.0, 0.5, 1.0]))
# #Nx.Tensor<f32[3] [0.0, 0.5235987901687622, 1.5707963705062866]>

Nx.acos(Nx.tensor([1.0, 0.5, 0.0]))
# #Nx.Tensor<f32[3] [0.0, 1.0471975803375244, 1.5707963705062866]>

Nx.atan(Nx.tensor([0.0, 1.0, 1.0e10]))
# #Nx.Tensor<f32[3] [0.0, 0.7853981852531433, 1.5707963705062866]>
```

### Hyperbolic
```elixir
x = Nx.tensor([0.0, 1.0, 2.0], backend: ExCubecl.Backend)

Nx.sinh(x)  # [0.0, 1.175201177597046, 3.6268603801727295]
Nx.cosh(x)  # [1.0, 1.5430806875228882, 3.762195587158203]
Nx.tanh(x)  # [0.0, 0.7615941762924194, 0.9640275835990906]
```

### Inverse Hyperbolic
```elixir
Nx.asinh(Nx.tensor([0.0, 1.0, 2.0]))
Nx.acosh(Nx.tensor([1.0, 2.0, 3.0]))
Nx.atanh(Nx.tensor([0.0, 0.5, 0.9]))
```

## Rounding Operations

```elixir
x = Nx.tensor([1.5, -1.5, 2.3, -2.7], backend: ExCubecl.Backend)

Nx.ceil(x)    # [2.0, -1.0, 3.0, -2.0]
Nx.floor(x)   # [1.0, -2.0, 2.0, -3.0]
Nx.round(x)   # [2.0, -2.0, 2.0, -3.0]
```

## Activation Functions

### Sigmoid
```elixir
Nx.sigmoid(Nx.tensor([0.0, 2.0, -2.0]))
# #Nx.Tensor<f32[3] [0.5, 0.8807970285415649, 0.11920291930437088]>
```

### ReLU
```elixir
Nx.relu(Nx.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
# #Nx.Tensor<f32[5] [0.0, 0.0, 0.0, 1.0, 2.0]>
```

## Error Functions

```elixir
x = Nx.tensor([0.0, 0.5, 1.0, 2.0], backend: ExCubecl.Backend)

Nx.erf(x)     # [0.0, 0.5204998850822449, 0.8427007794380188, 0.9953222870826721]
Nx.erfc(x)    # [1.0, 0.4795001149177551, 0.1572992205619812, 0.004677712917327881]
Nx.erf_inv(Nx.tensor([0.0, 0.5, 0.9]))
# #Nx.Tensor<f32[3] [0.0, 0.4769362807273865, 1.1630871295928955]>
```

## Special Operations

### Conjugate (identity for real numbers)
```elixir
Nx.conjugate(Nx.tensor([1.0, 2.0, 3.0]))
# #Nx.Tensor<f32[3] [1.0, 2.0, 3.0]>
```

### Real / Imaginary parts
```elixir
Nx.real(Nx.tensor([1.0, 2.0, 3.0]))  # [1.0, 2.0, 3.0]
Nx.imag(Nx.tensor([1.0, 2.0, 3.0]))  # [0.0, 0.0, 0.0]
```

### Is NaN / Is Infinity
```elixir
x = Nx.tensor[:nan, :infinity, :neg_infinity, 1.0]

Nx.is_nan(x)        # [1, 0, 0, 0]
Nx.is_infinity(x)   # [0, 1, 1, 0]
```

## Integer-Specific Operations

### Count Leading Zeros
```elixir
Nx.count_leading_zeros(Nx.tensor([1::32, 0::32, 255::32]))
# [31, 32, 24]
```

### Population Count (number of 1-bits)
```elixir
Nx.population_count(Nx.tensor([0::32, 1::32, 255::32, 7::32]))
# [0, 1, 8, 3]
```

### Bitwise Not
```elixir
Nx.bitwise_not(Nx.tensor([0::32, 1::32, 255::32]))
# [-1, -2, -256]
```
