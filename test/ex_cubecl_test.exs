defmodule ExCubeclTest do
  use ExUnit.Case
  alias ExCubecl.Backend, as: B

  describe "from_binary / to_binary" do
    test "round-trip f32" do
      binary = <<1.0::float-32, 2.0::float-32, 3.0::float-32, 4.0::float-32>>
      t = Nx.from_binary(binary, {:f, 32}, backend: B)
      assert Nx.shape(t) == {4}
      assert Nx.type(t) == {:f, 32}
      result = Nx.to_binary(t)
      assert result == binary
    end

    test "round-trip 2D f32" do
      binary = <<1.0::float-32, 2.0::float-32, 3.0::float-32, 4.0::float-32>>
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
  end

  describe "binary ops" do
    test "add f32" do
      a = Nx.tensor([1.0, 2.0, 3.0], backend: B)
      b = Nx.tensor([4.0, 5.0, 6.0], backend: B)
      result = Nx.add(a, b)
      assert Nx.to_binary(result) == <<5.0::float-32, 7.0::float-32, 9.0::float-32>>
    end

    test "subtract f32" do
      a = Nx.tensor([5.0, 7.0, 9.0], backend: B)
      b = Nx.tensor([1.0, 2.0, 3.0], backend: B)
      result = Nx.subtract(a, b)
      assert Nx.to_binary(result) == <<4.0::float-32, 5.0::float-32, 6.0::float-32>>
    end

    test "multiply f32" do
      a = Nx.tensor([2.0, 3.0, 4.0], backend: B)
      b = Nx.tensor([5.0, 6.0, 7.0], backend: B)
      result = Nx.multiply(a, b)
      assert Nx.to_binary(result) == <<10.0::float-32, 18.0::float-32, 28.0::float-32>>
    end

    test "divide f32" do
      a = Nx.tensor([10.0, 20.0, 30.0], backend: B)
      b = Nx.tensor([2.0, 4.0, 5.0], backend: B)
      result = Nx.divide(a, b)
      assert Nx.to_binary(result) == <<5.0::float-32, 5.0::float-32, 6.0::float-32>>
    end

    test "add with broadcasting" do
      a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: B)
      b = Nx.tensor([10.0, 20.0, 30.0], backend: B)
      result = Nx.add(a, b)
      expected = <<11.0::float-32, 22.0::float-32, 33.0::float-32, 14.0::float-32, 25.0::float-32, 36.0::float-32>>
      assert Nx.to_binary(result) == expected
    end
  end

  describe "unary ops" do
    test "negate" do
      a = Nx.tensor([1.0, -2.0, 3.0], backend: B)
      result = Nx.negate(a)
      assert Nx.to_binary(result) == <<-1.0::float-32, 2.0::float-32, -3.0::float-32>>
    end

    test "abs" do
      a = Nx.tensor([-1.0, 2.0, -3.0], backend: B)
      result = Nx.abs(a)
      assert Nx.to_binary(result) == <<1.0::float-32, 2.0::float-32, 3.0::float-32>>
    end

    test "exp" do
      a = Nx.tensor([0.0, 1.0], backend: B)
      result = Nx.exp(a)
      [v0, v1] = Nx.to_binary(result) |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v0, 1.0, 1.0e-6)
      assert_in_delta(v1, :math.exp(1.0), 1.0e-6)
    end

    test "log" do
      a = Nx.tensor([1.0, :math.exp(1.0)], backend: B)
      result = Nx.log(a)
      [v0, v1] = Nx.to_binary(result) |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v0, 0.0, 1.0e-6)
      assert_in_delta(v1, 1.0, 1.0e-6)
    end

    test "sqrt" do
      a = Nx.tensor([4.0, 9.0, 16.0], backend: B)
      result = Nx.sqrt(a)
      assert Nx.to_binary(result) == <<2.0::float-32, 3.0::float-32, 4.0::float-32>>
    end

    test "sigmoid" do
      a = Nx.tensor([0.0], backend: B)
      result = Nx.sigmoid(a)
      [v] = Nx.to_binary(result) |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v, 0.5, 1.0e-6)
    end

    test "relu" do
      a = Nx.tensor([-1.0, 0.0, 2.0], backend: B)
      result = Nx.relu(a)
      assert Nx.to_binary(result) == <<0.0::float-32, 0.0::float-32, 2.0::float-32>>
    end

    test "sin" do
      a = Nx.tensor([0.0], backend: B)
      result = Nx.sin(a)
      [v] = Nx.to_binary(result) |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v, 0.0, 1.0e-6)
    end

    test "cos" do
      a = Nx.tensor([0.0], backend: B)
      result = Nx.cos(a)
      [v] = Nx.to_binary(result) |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v, 1.0, 1.0e-6)
    end
  end

  describe "shape ops" do
    test "reshape" do
      a = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: B)
      result = Nx.reshape(a, {2, 2})
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_binary(result) == <<1.0::float-32, 2.0::float-32, 3.0::float-32, 4.0::float-32>>
    end

    test "transpose" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: B)
      result = Nx.transpose(a)
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_binary(result) == <<1.0::float-32, 3.0::float-32, 2.0::float-32, 4.0::float-32>>
    end
  end

  describe "reductions" do
    test "sum all" do
      a = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: B)
      result = Nx.sum(a)
      [v] = Nx.to_binary(result) |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v, 10.0, 1.0e-6)
    end

    test "sum along axis" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: B)
      result = Nx.sum(a, axes: [1])
      assert Nx.to_binary(result) == <<3.0::float-32, 7.0::float-32>>
    end

    test "reduce_max" do
      a = Nx.tensor([1.0, 5.0, 3.0, 2.0], backend: B)
      result = Nx.reduce_max(a)
      [v] = Nx.to_binary(result) |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v, 5.0, 1.0e-6)
    end

    test "reduce_min" do
      a = Nx.tensor([1.0, 5.0, 3.0, 2.0], backend: B)
      result = Nx.reduce_min(a)
      [v] = Nx.to_binary(result) |> Nx.from_binary({:f, 32}) |> Nx.to_flat_list()
      assert_in_delta(v, 1.0, 1.0e-6)
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
end
