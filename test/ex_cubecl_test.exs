defmodule ExCubeclTest do
  use ExUnit.Case

  alias ExCubecl.Backend, as: B

  describe "from_binary / to_binary" do
    test "round-trip f32" do
      binary = <<1.0::float-32-little, 2.0::float-32-little, 3.0::float-32-little, 4.0::float-32-little>>
      t = Nx.from_binary(binary, {:f, 32}, backend: B)
      assert Nx.shape(t) == {4}
      assert Nx.type(t) == {:f, 32}
      assert Nx.to_binary(t) == binary
    end

    test "round-trip 2D f32" do
      binary = <<1.0::float-32-little, 2.0::float-32-little, 3.0::float-32-little, 4.0::float-32-little>>
      t = Nx.from_binary(binary, {:f, 32}, backend: B)
      t2 = Nx.reshape(t, {2, 2})
      assert Nx.shape(t2) == {2, 2}
      assert Nx.to_binary(t2) == binary
    end

    test "round-trip s32" do
      binary = <<1::32-native, 2::32-native, 3::32-native>>
      t = Nx.from_binary(binary, {:s, 32}, backend: B)
      assert Nx.to_binary(t) == binary
    end

    test "round-trip u8" do
      binary = <<1::8, 2::8, 3::8, 4::8>>
      t = Nx.from_binary(binary, {:u, 8}, backend: B)
      assert Nx.to_binary(t) == binary
    end

    test "round-trip f64" do
      binary = <<1.0::float-64-little, 2.0::float-64-little, 3.0::float-64-little>>
      t = Nx.from_binary(binary, {:f, 64}, backend: B)
      assert Nx.to_binary(t) == binary
    end
  end

  describe "binary ops" do
    test "add f32" do
      a = Nx.tensor([1.0, 2.0, 3.0], backend: B)
      b = Nx.tensor([4.0, 5.0, 6.0], backend: B)
      result = Nx.add(a, b)
      assert Nx.to_binary(result) == <<5.0::float-32-little, 7.0::float-32-little, 9.0::float-32-little>>
    end

    test "subtract f32" do
      a = Nx.tensor([5.0, 7.0, 9.0], backend: B)
      b = Nx.tensor([1.0, 2.0, 3.0], backend: B)
      result = Nx.subtract(a, b)
      assert Nx.to_binary(result) == <<4.0::float-32-little, 5.0::float-32-little, 6.0::float-32-little>>
    end

    test "multiply f32" do
      a = Nx.tensor([2.0, 3.0, 4.0], backend: B)
      b = Nx.tensor([5.0, 6.0, 7.0], backend: B)
      result = Nx.multiply(a, b)
      assert Nx.to_binary(result) == <<10.0::float-32-little, 18.0::float-32-little, 28.0::float-32-little>>
    end

    test "divide f32" do
      a = Nx.tensor([10.0, 20.0, 30.0], backend: B)
      b = Nx.tensor([2.0, 4.0, 5.0], backend: B)
      result = Nx.divide(a, b)
      assert Nx.to_binary(result) == <<5.0::float-32-little, 5.0::float-32-little, 6.0::float-32-little>>
    end

    test "add with broadcasting" do
      a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: B)
      b = Nx.tensor([10.0, 20.0, 30.0], backend: B)
      result = Nx.add(a, b)
      expected = <<11.0::float-32-little, 22.0::float-32-little, 33.0::float-32-little, 14.0::float-32-little, 25.0::float-32-little, 36.0::float-32-little>>
      assert Nx.to_binary(result) == expected
    end

    test "min / max" do
      a = Nx.tensor([1.0, 5.0, 3.0], backend: B)
      b = Nx.tensor([4.0, 2.0, 6.0], backend: B)
      assert Nx.to_binary(Nx.min(a, b)) == <<1.0::float-32-little, 2.0::float-32-little, 3.0::float-32-little>>
      assert Nx.to_binary(Nx.max(a, b)) == <<4.0::float-32-little, 5.0::float-32-little, 6.0::float-32-little>>
    end

    test "equal" do
      a = Nx.tensor([1.0, 2.0, 3.0], backend: B)
      b = Nx.tensor([1.0, 0.0, 3.0], backend: B)
      assert Nx.to_binary(Nx.equal(a, b)) == <<1::8, 0::8, 1::8>>
    end

    test "greater" do
      a = Nx.tensor([1.0, 5.0, 3.0], backend: B)
      b = Nx.tensor([4.0, 2.0, 3.0], backend: B)
      assert Nx.to_binary(Nx.greater(a, b)) == <<0::8, 1::8, 0::8>>
    end

    test "logical_and" do
      a = Nx.tensor([1, 0, 1], backend: B)
      b = Nx.tensor([1, 1, 0], backend: B)
      assert Nx.to_binary(Nx.logical_and(a, b)) == <<1::8, 0::8, 0::8>>
    end

    test "logical_or" do
      a = Nx.tensor([1, 0, 1], backend: B)
      b = Nx.tensor([1, 1, 0], backend: B)
      assert Nx.to_binary(Nx.logical_or(a, b)) == <<1::8, 1::8, 1::8>>
    end

    test "logical_xor" do
      a = Nx.tensor([1, 0, 1], backend: B)
      b = Nx.tensor([1, 1, 0], backend: B)
      assert Nx.to_binary(Nx.logical_xor(a, b)) == <<0::8, 1::8, 1::8>>
    end

    test "bitwise_and u32" do
      a = Nx.from_binary(<<0xFF::32-native, 0x0F::32-native>>, {:u, 32}, backend: B)
      b = Nx.from_binary(<<0x0F::32-native, 0xF0::32-native>>, {:u, 32}, backend: B)
      assert Nx.to_binary(Nx.bitwise_and(a, b)) == <<0x0F::32-native, 0x00::32-native>>
    end

    test "bitwise_or u32" do
      a = Nx.from_binary(<<0xFF::32-native, 0x0F::32-native>>, {:u, 32}, backend: B)
      b = Nx.from_binary(<<0x0F::32-native, 0xF0::32-native>>, {:u, 32}, backend: B)
      assert Nx.to_binary(Nx.bitwise_or(a, b)) == <<0xFF::32-native, 0xFF::32-native>>
    end

    test "bitwise_xor u32" do
      a = Nx.from_binary(<<0xFF::32-native, 0x0F::32-native>>, {:u, 32}, backend: B)
      b = Nx.from_binary(<<0x0F::32-native, 0xF0::32-native>>, {:u, 32}, backend: B)
      assert Nx.to_binary(Nx.bitwise_xor(a, b)) == <<0xF0::32-native, 0xFF::32-native>>
    end

    test "left_shift s32" do
      a = Nx.from_binary(<<1::32-native, 2::32-native, 4::32-native>>, {:s, 32}, backend: B)
      b = Nx.from_binary(<<2::32-native, 3::32-native, 1::32-native>>, {:s, 32}, backend: B)
      assert Nx.to_binary(Nx.left_shift(a, b)) == <<4::32-native, 16::32-native, 8::32-native>>
    end

    test "right_shift s32" do
      a = Nx.from_binary(<<16::32-native, 32::32-native, 8::32-native>>, {:s, 32}, backend: B)
      b = Nx.from_binary(<<2::32-native, 3::32-native, 1::32-native>>, {:s, 32}, backend: B)
      assert Nx.to_binary(Nx.right_shift(a, b)) == <<4::32-native, 4::32-native, 4::32-native>>
    end
  end

  describe "unary ops" do
    test "negate" do
      a = Nx.tensor([1.0, -2.0, 3.0], backend: B)
      assert Nx.to_binary(Nx.negate(a)) == <<-1.0::float-32-little, 2.0::float-32-little, -3.0::float-32-little>>
    end

    test "abs" do
      a = Nx.tensor([-1.0, 2.0, -3.0], backend: B)
      assert Nx.to_binary(Nx.abs(a)) == <<1.0::float-32-little, 2.0::float-32-little, 3.0::float-32-little>>
    end

    test "sign" do
      a = Nx.tensor([-5.0, 0.0, 3.0], backend: B)
      assert Nx.to_binary(Nx.sign(a)) == <<-1.0::float-32-little, 0.0::float-32-little, 1.0::float-32-little>>
    end

    test "exp" do
      a = Nx.tensor([0.0, 1.0], backend: B)
      result = Nx.exp(a) |> Nx.to_binary() |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(Enum.at(result, 0), 1.0, 1.0e-6)
      assert_in_delta(Enum.at(result, 1), :math.exp(1.0), 1.0e-6)
    end

    test "log" do
      a = Nx.tensor([1.0, :math.exp(1.0)], backend: B)
      result = Nx.log(a) |> Nx.to_binary() |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(Enum.at(result, 0), 0.0, 1.0e-6)
      assert_in_delta(Enum.at(result, 1), 1.0, 1.0e-6)
    end

    test "sqrt" do
      a = Nx.tensor([4.0, 9.0, 16.0], backend: B)
      assert Nx.to_binary(Nx.sqrt(a)) == <<2.0::float-32-little, 3.0::float-32-little, 4.0::float-32-little>>
    end

    test "sigmoid" do
      a = Nx.tensor([0.0], backend: B)
      result = Nx.sigmoid(a)
      [v] = result |> Nx.to_binary() |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v, 0.5, 1.0e-6)
    end

    test "relu" do
      a = Nx.tensor([-1.0, 0.0, 2.0], backend: B)
      result = Nx.max(a, 0.0)
      assert Nx.to_binary(result) == <<0.0::float-32-little, 0.0::float-32-little, 2.0::float-32-little>>
    end

    test "sin" do
      a = Nx.tensor([0.0], backend: B)
      [v] = Nx.sin(a) |> Nx.to_binary() |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v, 0.0, 1.0e-6)
    end

    test "cos" do
      a = Nx.tensor([0.0], backend: B)
      [v] = Nx.cos(a) |> Nx.to_binary() |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v, 1.0, 1.0e-6)
    end

    test "tanh" do
      a = Nx.tensor([0.0], backend: B)
      [v] = Nx.tanh(a) |> Nx.to_binary() |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v, 0.0, 1.0e-6)
    end

    test "ceil / floor / round" do
      a = Nx.tensor([1.5, -1.5, 2.3], backend: B)
      assert Nx.to_binary(Nx.ceil(a)) == <<2.0::float-32-little, -1.0::float-32-little, 3.0::float-32-little>>
      assert Nx.to_binary(Nx.floor(a)) == <<1.0::float-32-little, -2.0::float-32-little, 2.0::float-32-little>>
      assert Nx.to_binary(Nx.round(a)) == <<2.0::float-32-little, -2.0::float-32-little, 2.0::float-32-little>>
    end

    test "count_leading_zeros" do
      a = Nx.from_binary(<<1::32-native, 0::32-native, 255::32-native>>, {:s, 32}, backend: B)
      result = Nx.count_leading_zeros(a) |> Nx.to_binary() |> Nx.from_binary({:s, 32}) |> Nx.to_flat_list()
      assert Enum.at(result, 0) == 31
      assert Enum.at(result, 1) == 32
      assert Enum.at(result, 2) == 24
    end

    test "population_count" do
      a = Nx.from_binary(<<0::32-native, 1::32-native, 255::32-native, 7::32-native>>, {:s, 32}, backend: B)
      result = Nx.population_count(a) |> Nx.to_binary() |> Nx.from_binary({:s, 32}) |> Nx.to_flat_list()
      assert Enum.at(result, 0) == 0
      assert Enum.at(result, 1) == 1
      assert Enum.at(result, 2) == 8
      assert Enum.at(result, 3) == 3
    end

    test "bitwise_not" do
      a = Nx.from_binary(<<0::32-native, 1::32-native, 255::32-native>>, {:s, 32}, backend: B)
      assert Nx.to_binary(Nx.bitwise_not(a)) == <<-1::32-native, -2::32-native, -256::32-native>>
    end
  end

  describe "shape ops" do
    test "reshape" do
      a = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: B)
      result = Nx.reshape(a, {2, 2})
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_binary(result) == <<1.0::float-32-little, 2.0::float-32-little, 3.0::float-32-little, 4.0::float-32-little>>
    end

    test "transpose" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: B)
      result = Nx.transpose(a)
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_binary(result) == <<1.0::float-32-little, 3.0::float-32-little, 2.0::float-32-little, 4.0::float-32-little>>
    end

    test "squeeze" do
      a = Nx.tensor([[[1.0, 2.0, 3.0]]], backend: B)
      result = Nx.squeeze(a)
      assert Nx.shape(result) == {3}
    end

    test "broadcast" do
      a = Nx.tensor([1.0, 2.0, 3.0], backend: B)
      result = Nx.broadcast(a, {3, 3})
      assert Nx.shape(result) == {3, 3}
    end

    test "pad" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: B)
      result = Nx.pad(a, 0.0, [{1, 1, 0}, {1, 1, 0}])
      assert Nx.shape(result) == {4, 4}
    end

    test "reverse" do
      a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: B)
      result = Nx.reverse(a, axes: [0])
      assert Nx.to_binary(result) == <<4.0::float-32-little, 5.0::float-32-little, 6.0::float-32-little, 1.0::float-32-little, 2.0::float-32-little, 3.0::float-32-little>>
    end

    test "slice" do
      a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], backend: B)
      result = Nx.slice(a, [0, 0], [2, 2], strides: [1, 1])
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_binary(result) == <<1.0::float-32-little, 2.0::float-32-little, 4.0::float-32-little, 5.0::float-32-little>>
    end

    test "concatenate" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: B)
      b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]], backend: B)
      result = Nx.concatenate([a, b], axis: 0)
      assert Nx.shape(result) == {4, 2}
    end

    test "stack" do
      a = Nx.tensor([1.0, 2.0, 3.0], backend: B)
      b = Nx.tensor([4.0, 5.0, 6.0], backend: B)
      result = Nx.stack([a, b], axis: 0)
      assert Nx.shape(result) == {2, 3}
    end

    test "select" do
      pred = Nx.tensor([1, 0, 1, 0], backend: B)
      on_true = Nx.tensor([10.0, 20.0, 30.0, 40.0], backend: B)
      on_false = Nx.tensor([100.0, 200.0, 300.0, 400.0], backend: B)
      result = Nx.select(pred, on_true, on_false)
      assert Nx.to_binary(result) == <<10.0::float-32-little, 200.0::float-32-little, 30.0::float-32-little, 400.0::float-32-little>>
    end

    test "put_slice" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: B)
      slice = Nx.tensor([[10.0, 20.0]], backend: B)
      result = Nx.put_slice(t, [0, 0], slice)
      assert Nx.to_binary(result) == <<10.0::float-32-little, 20.0::float-32-little, 3.0::float-32-little, 4.0::float-32-little, 5.0::float-32-little, 6.0::float-32-little>>
    end
  end

  describe "reductions" do
    test "sum all" do
      a = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: B)
      [v] = a |> Nx.sum() |> Nx.to_binary() |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v, 10.0, 1.0e-6)
    end

    test "sum along axis" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: B)
      result = Nx.sum(a, axes: [1])
      assert Nx.to_binary(result) == <<3.0::float-32-little, 7.0::float-32-little>>
    end

    test "reduce_max" do
      a = Nx.tensor([1.0, 5.0, 3.0, 2.0], backend: B)
      [v] = a |> Nx.reduce_max() |> Nx.to_binary() |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v, 5.0, 1.0e-6)
    end

    test "reduce_min" do
      a = Nx.tensor([1.0, 5.0, 3.0, 2.0], backend: B)
      [v] = a |> Nx.reduce_min() |> Nx.to_binary() |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v, 1.0, 1.0e-6)
    end

    test "product" do
      a = Nx.tensor([2.0, 3.0, 4.0], backend: B)
      [v] = a |> Nx.product() |> Nx.to_binary() |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v, 24.0, 1.0e-6)
    end

    test "argmax / argmin" do
      a = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0], backend: B)
      [v_max] = a |> Nx.argmax() |> Nx.to_binary() |> Nx.from_binary({:s, 32}) |> Nx.to_flat_list()
      [v_min] = a |> Nx.argmin() |> Nx.to_binary() |> Nx.from_binary({:s, 32}) |> Nx.to_flat_list()
      assert v_max == 4
      assert v_min == 1
    end
  end

  describe "type conversion" do
    test "as_type f32 to s32" do
      a = Nx.tensor([1.5, 2.7, 3.0], backend: B)
      result = Nx.as_type(a, {:s, 32})
      assert Nx.type(result) == {:s, 32}
      assert Nx.to_binary(result) == <<1::32-native, 2::32-native, 3::32-native>>
    end

    test "as_type s32 to f32" do
      binary = <<1::32-native, 2::32-native, 3::32-native>>
      a = Nx.from_binary(binary, {:s, 32}, backend: B)
      result = Nx.as_type(a, {:f, 32})
      assert Nx.type(result) == {:f, 32}
      assert Nx.to_binary(result) == <<1.0::float-32-little, 2.0::float-32-little, 3.0::float-32-little>>
    end

    test "bitcast f32 to u32" do
      a = Nx.tensor([1.0, 0.0, -0.0], backend: B)
      result = Nx.bitcast(a, {:u, 32})
      assert Nx.to_binary(result) == <<1065353216::32-native, 0::32-native, 2147483648::32-native>>
    end

    test "clip" do
      a = Nx.tensor([1.0, 5.0, 10.0, 15.0, 20.0], backend: B)
      result = Nx.clip(a, Nx.tensor(5.0, backend: B), Nx.tensor(15.0, backend: B))
      assert Nx.to_binary(result) == <<5.0::float-32-little, 5.0::float-32-little, 10.0::float-32-little, 15.0::float-32-little, 15.0::float-32-little>>
    end
  end

  describe "creation helpers" do
    test "eye" do
      result = Nx.eye({3, 3}, backend: B)
      assert Nx.shape(result) == {3, 3}
    end

    test "iota" do
      result = Nx.iota({4}, backend: B)
      assert Nx.to_binary(result) == <<0::32-native, 1::32-native, 2::32-native, 3::32-native>>
    end

    test "constant" do
      result = Nx.tensor(42.0, backend: B)
      [v] = result |> Nx.to_binary() |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v, 42.0, 1.0e-6)
    end
  end

  describe "sorting" do
    test "sort" do
      a = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0], backend: B)
      result = Nx.sort(a)
      assert Nx.to_binary(result) == <<1.0::float-32-little, 1.0::float-32-little, 2.0::float-32-little, 3.0::float-32-little, 4.0::float-32-little, 5.0::float-32-little, 9.0::float-32-little>>
    end

    @tag :skip
    test "argsort" do
      a = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0], backend: B)
      result = Nx.argsort(a)
      assert Nx.to_binary(result) == <<1::64-native, 3::64-native, 0::64-native, 2::64-native, 4::64-native>>
    end
  end

  describe "linear algebra" do
    test "dot product" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: B)
      b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]], backend: B)
      result = Nx.dot(a, b)
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_binary(result) == <<19.0::float-32-little, 22.0::float-32-little, 43.0::float-32-little, 50.0::float-32-little>>
    end

    test "convolution shape" do
      input = Nx.reshape(Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]), {1, 1, 3, 3})
      kernel = Nx.reshape(Nx.tensor([1.0, 0.0, 0.0, -1.0]), {1, 1, 2, 2})
      result = Nx.conv(input, kernel)
      assert Nx.shape(result) == {1, 1, 2, 2}
    end
  end

  describe "window operations" do
    test "window_sum 2x2" do
      a = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], backend: B)
      result = Nx.window_sum(a, {2, 2}, [])
      assert Nx.shape(result) == {1, 3}
    end

    test "window_max 1x2" do
      a = Nx.tensor([[1.0, 3.0, 2.0, 4.0], [5.0, 2.0, 6.0, 1.0]], backend: B)
      result = Nx.window_max(a, {1, 2}, [])
      assert Nx.shape(result) == {2, 3}
    end

    test "window_min 1x2" do
      a = Nx.tensor([[1.0, 3.0, 2.0, 4.0], [5.0, 2.0, 6.0, 1.0]], backend: B)
      result = Nx.window_min(a, {1, 2}, [])
      assert Nx.shape(result) == {2, 3}
    end
  end

  describe "indexed operations" do
    test "gather 1D" do
      input = Nx.tensor([10.0, 20.0, 30.0, 40.0, 50.0], backend: B)
      indices = Nx.tensor([[0], [2], [4], [1]], backend: B)
      result = Nx.gather(input, indices, axes: [0])
      assert Nx.to_binary(result) == <<10.0::float-32-little, 30.0::float-32-little, 50.0::float-32-little, 20.0::float-32-little>>
    end

    test "indexed_add" do
      t = Nx.tensor([0.0, 0.0, 0.0, 0.0, 0.0], backend: B)
      indices = Nx.tensor([0], backend: B)
      updates = Nx.tensor(10.0, backend: B)
      result = Nx.indexed_add(t, indices, updates, [])
      assert Nx.to_binary(result) == <<10.0::float-32-little, 0.0::float-32-little, 0.0::float-32-little, 0.0::float-32-little, 0.0::float-32-little>>
    end

    test "indexed_put" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], backend: B)
      indices = Nx.tensor([1], backend: B)
      updates = Nx.tensor(20.0, backend: B)
      result = Nx.indexed_put(t, indices, updates, [])
      assert Nx.to_binary(result) == <<1.0::float-32-little, 20.0::float-32-little, 3.0::float-32-little, 4.0::float-32-little, 5.0::float-32-little>>
    end
  end

  describe "backend transfer" do
    test "transfer to BinaryBackend and back" do
      a = Nx.tensor([1.0, 2.0, 3.0], backend: B)
      b = Nx.backend_transfer(a, Nx.BinaryBackend)
      assert Nx.shape(b) == {3}
      c = Nx.backend_transfer(b, B)
      assert Nx.to_binary(c) == Nx.to_binary(a)
    end
  end

  describe "ExCubecl module" do
    test "version" do
      assert is_binary(ExCubecl.version())
    end

    test "device_info" do
      info = ExCubecl.device_info()
      assert info.backend == :cpu
      assert is_map(info)
    end

    test "supported_types" do
      types = ExCubecl.supported_types()
      assert {:f, 32} in types
      assert {:f, 64} in types
      assert {:s, 32} in types
      assert {:u, 8} in types
    end

    test "type_size" do
      assert ExCubecl.type_size({:f, 32}) == 4
      assert ExCubecl.type_size({:f, 64}) == 8
      assert ExCubecl.type_size({:u, 8}) == 1
    end
  end
end
