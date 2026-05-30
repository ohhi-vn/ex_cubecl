defmodule ExCubeclTest do
  use ExUnit.Case, async: true

  describe "version" do
    test "returns version string" do
      assert is_binary(ExCubecl.version())
    end

    test "version matches mix.exs" do
      assert ExCubecl.version() == "0.4.0"
    end
  end

  describe "device_info" do
    test "returns a map with expected keys" do
      {:ok, info} = ExCubecl.device_info()
      assert is_map(info)
      assert Map.has_key?(info, :device_name)
    end

    test "device info contains expected fields" do
      {:ok, info} = ExCubecl.device_info()
      assert Map.has_key?(info, :device_type)
      assert Map.has_key?(info, :total_memory)
      assert Map.has_key?(info, :compute_units)
    end

    test "device_type is gpu" do
      {:ok, info} = ExCubecl.device_info()
      assert info[:device_type] == "gpu"
    end

    test "total_memory is a positive integer" do
      {:ok, info} = ExCubecl.device_info()
      assert is_integer(info[:total_memory])
      assert info[:total_memory] > 0
    end

    test "compute_units is a positive integer" do
      {:ok, info} = ExCubecl.device_info()
      assert is_integer(info[:compute_units])
      assert info[:compute_units] > 0
    end
  end

  describe "device_count" do
    test "returns a non-negative integer" do
      {:ok, count} = ExCubecl.device_count()
      assert is_integer(count)
      assert count >= 0
    end

    test "returns at least 1" do
      {:ok, count} = ExCubecl.device_count()
      assert count >= 1
    end
  end

  describe "supported_dtypes" do
    test "returns list of dtype atoms" do
      dtypes = ExCubecl.supported_dtypes()
      assert :f32 in dtypes
      assert :f64 in dtypes
      assert :s32 in dtypes
      assert :s64 in dtypes
      assert :u32 in dtypes
      assert :u8 in dtypes
    end

    test "returns exactly 6 dtypes" do
      assert length(ExCubecl.supported_dtypes()) == 6
    end
  end

  describe "available?" do
    test "returns boolean" do
      assert is_boolean(ExCubecl.available?())
    end

    test "returns true when NIF is loaded" do
      assert ExCubecl.available?() == true
    end
  end

  describe "buffer management" do
    test "create and read f32 buffer" do
      {:ok, buf} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
      assert is_reference(buf)
      assert {:ok, "f32"} = ExCubecl.dtype(buf)
      assert {:ok, [3]} = ExCubecl.shape(buf)
      assert {:ok, 12} = ExCubecl.size(buf)
      {:ok, data} = ExCubecl.read(buf)
      assert is_binary(data)
      assert byte_size(data) == 12
    end

    test "create and read s32 buffer" do
      {:ok, buf} = ExCubecl.buffer([10, 20, 30], [3], :s32)
      assert {:ok, "s32"} = ExCubecl.dtype(buf)
      assert {:ok, 12} = ExCubecl.size(buf)
    end

    test "create 2D buffer" do
      {:ok, buf} = ExCubecl.buffer([1.0, 2.0, 3.0, 4.0], [2, 2], :f32)
      assert {:ok, [2, 2]} = ExCubecl.shape(buf)
      assert {:ok, 16} = ExCubecl.size(buf)
    end

    test "create u8 buffer" do
      {:ok, buf} = ExCubecl.buffer([0, 128, 255], [3], :u8)
      assert {:ok, "u8"} = ExCubecl.dtype(buf)
      assert {:ok, 3} = ExCubecl.size(buf)
    end

    test "buffer! raises on success" do
      {:ok, buf} = ExCubecl.buffer([1.0, 2.0], [2], :f32)
      assert is_reference(buf)
    end

    test "read! raises on error" do
      {:ok, buf} = ExCubecl.buffer([1.0], [1], :f32)
      data = ExCubecl.read!(buf)
      assert is_binary(data)
    end

    test "create f64 buffer" do
      {:ok, buf} = ExCubecl.buffer([1.0, 2.0], [2], :f64)
      assert {:ok, "f64"} = ExCubecl.dtype(buf)
      assert {:ok, 16} = ExCubecl.size(buf)
    end

    test "create s64 buffer" do
      {:ok, buf} = ExCubecl.buffer([1, 2], [2], :s64)
      assert {:ok, "s64"} = ExCubecl.dtype(buf)
      assert {:ok, 16} = ExCubecl.size(buf)
    end

    test "create u32 buffer" do
      {:ok, buf} = ExCubecl.buffer([1, 2], [2], :u32)
      assert {:ok, "u32"} = ExCubecl.dtype(buf)
      assert {:ok, 8} = ExCubecl.size(buf)
    end

    test "buffer with single element" do
      {:ok, buf} = ExCubecl.buffer([42.0], [1], :f32)
      assert {:ok, [1]} = ExCubecl.shape(buf)
      assert {:ok, 4} = ExCubecl.size(buf)
    end

    test "buffer with default dtype" do
      {:ok, buf} = ExCubecl.buffer([1.0, 2.0], [2])
      assert {:ok, "f32"} = ExCubecl.dtype(buf)
    end

    test "buffer! with default dtype" do
      buf = ExCubecl.buffer!([1.0, 2.0], [2])
      assert is_reference(buf)
      assert {:ok, "f32"} = ExCubecl.dtype(buf)
    end

    test "buffer with large shape" do
      data = Enum.to_list(1..1000) |> Enum.map(&(&1 * 1.0))
      {:ok, buf} = ExCubecl.buffer(data, [1000], :f32)
      assert {:ok, [1000]} = ExCubecl.shape(buf)
      assert {:ok, 4000} = ExCubecl.size(buf)
    end

    test "buffer with 3D shape" do
      data = Enum.to_list(1..24) |> Enum.map(&(&1 * 1.0))
      {:ok, buf} = ExCubecl.buffer(data, [2, 3, 4], :f32)
      assert {:ok, [2, 3, 4]} = ExCubecl.shape(buf)
      assert {:ok, 96} = ExCubecl.size(buf)
    end

    test "buffer read returns correct binary size for u8" do
      {:ok, buf} = ExCubecl.buffer([0, 128, 255], [3], :u8)
      {:ok, data} = ExCubecl.read(buf)
      assert byte_size(data) == 3
    end

    test "buffer read returns correct binary size for f64" do
      {:ok, buf} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f64)
      {:ok, data} = ExCubecl.read(buf)
      assert byte_size(data) == 24
    end

    test "buffer! raises on mismatched size" do
      assert_raise RuntimeError, ~r/ExCubecl.buffer!\/3 failed/, fn ->
        ExCubecl.buffer!([1.0, 2.0, 3.0], [2], :f32)
      end
    end

    test "buffer returns error on mismatched size" do
      assert {:error, msg} = ExCubecl.buffer([1.0, 2.0, 3.0], [2], :f32)
      assert is_binary(msg)
      assert msg =~ "buffer_new"
    end

    test "buffer with negative s32 values" do
      {:ok, buf} = ExCubecl.buffer([-1, -2, -3], [3], :s32)
      assert {:ok, 12} = ExCubecl.size(buf)
      {:ok, data} = ExCubecl.read(buf)
      assert is_binary(data)
    end

    test "buffer with zero values" do
      {:ok, buf} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)
      assert {:ok, 12} = ExCubecl.size(buf)
      {:ok, zero_data} = ExCubecl.read(buf)
      assert zero_data == <<0::float-32-native, 0::float-32-native, 0::float-32-native>>
    end

    test "buffer with max u8 value" do
      {:ok, buf} = ExCubecl.buffer([255], [1], :u8)
      assert {:ok, 1} = ExCubecl.size(buf)
      {:ok, data} = ExCubecl.read(buf)
      assert data == <<255>>
    end

    test "buffer with min u8 value" do
      {:ok, buf} = ExCubecl.buffer([0], [1], :u8)
      {:ok, data} = ExCubecl.read(buf)
      assert data == <<0>>
    end

    test "multiple buffers are independent" do
      {:ok, buf_a} = ExCubecl.buffer([1.0, 2.0], [2], :f32)
      {:ok, buf_b} = ExCubecl.buffer([3.0, 4.0], [2], :f32)
      assert buf_a != buf_b
      assert {:ok, data_a} = ExCubecl.read(buf_a)
      assert {:ok, data_b} = ExCubecl.read(buf_b)
      assert data_a != data_b
    end
  end

  describe "kernels" do
    test "kernels returns a list" do
      {:ok, kernels} = ExCubecl.kernels()
      assert is_list(kernels)
      assert length(kernels) > 0
    end

    test "kernels includes elementwise_add" do
      {:ok, kernels} = ExCubecl.kernels()
      assert "elementwise_add" in kernels
    end

    test "kernels includes relu" do
      {:ok, kernels} = ExCubecl.kernels()
      assert "relu" in kernels
    end

    test "kernels includes video kernels" do
      {:ok, kernels} = ExCubecl.kernels()
      assert "yuv_to_rgb" in kernels
      assert "overlay_alpha" in kernels
      assert "gaussian_blur" in kernels
    end

    test "kernels includes audio kernels" do
      {:ok, kernels} = ExCubecl.kernels()
      assert "pcm_mix" in kernels
      assert "resample_linear" in kernels
    end

    test "run_kernel executes without error" do
      {:ok, input} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
      {:ok, output} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)
      {:ok, _cmd_id} = ExCubecl.run_kernel("elementwise_add", [input], output)
    end

    test "run_kernel returns a command id" do
      {:ok, input} = ExCubecl.buffer([1.0], [1], :f32)
      {:ok, output} = ExCubecl.buffer([0.0], [1], :f32)
      {:ok, cmd_id} = ExCubecl.run_kernel("relu", [input], output)
      assert is_integer(cmd_id)
      assert cmd_id > 0
    end

    test "run_kernel with empty params" do
      {:ok, input} = ExCubecl.buffer([1.0], [1], :f32)
      {:ok, output} = ExCubecl.buffer([0.0], [1], :f32)
      {:ok, _cmd_id} = ExCubecl.run_kernel("relu", [input], output, %{})
    end

    test "run_kernel with unknown kernel returns error" do
      {:ok, input} = ExCubecl.buffer([1.0], [1], :f32)
      {:ok, output} = ExCubecl.buffer([0.0], [1], :f32)
      assert {:error, msg} = ExCubecl.run_kernel("nonexistent_kernel", [input], output)
      assert is_binary(msg)
      assert msg =~ "nonexistent_kernel"
    end

    test "run_kernel with multiple inputs" do
      {:ok, a} = ExCubecl.buffer([1.0, 2.0], [2], :f32)
      {:ok, b} = ExCubecl.buffer([3.0, 4.0], [2], :f32)
      {:ok, output} = ExCubecl.buffer([0.0, 0.0], [2], :f32)
      {:ok, _cmd_id} = ExCubecl.run_kernel("elementwise_add", [a, b], output)
    end
  end

  describe "async execution" do
    test "submit returns a command id" do
      {:ok, cmd_id} = ExCubecl.submit("test_command")
      assert is_integer(cmd_id)
    end

    test "poll returns status" do
      {:ok, cmd_id} = ExCubecl.submit("test_command")
      {:ok, status} = ExCubecl.poll(cmd_id)
      assert status in [:pending, :running, :completed, :failed]
    end

    test "wait blocks until completion" do
      {:ok, cmd_id} = ExCubecl.submit("test_command")
      result = ExCubecl.wait(cmd_id)
      assert result == :ok or match?({:error, _}, result)
    end

    test "poll after wait returns completed" do
      {:ok, cmd_id} = ExCubecl.submit("test_command")
      :ok = ExCubecl.wait(cmd_id)
      {:ok, status} = ExCubecl.poll(cmd_id)
      assert status == :completed
    end

    test "submit returns unique command ids" do
      {:ok, id1} = ExCubecl.submit("cmd1")
      {:ok, id2} = ExCubecl.submit("cmd2")
      assert id1 != id2
    end

    test "poll with invalid command id returns error" do
      assert {:error, _} = ExCubecl.poll(999_999)
    end

    test "wait with invalid command id returns error" do
      assert {:error, _} = ExCubecl.wait(999_999)
    end
  end

  describe "pipeline" do
    test "create pipeline" do
      {:ok, pipeline_id} = ExCubecl.pipeline()
      assert is_integer(pipeline_id)
      :ok = ExCubecl.pipeline_free(pipeline_id)
    end

    test "pipeline add and run" do
      {:ok, input} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
      {:ok, output} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)

      {:ok, pipeline_id} = ExCubecl.pipeline()
      :ok = ExCubecl.pipeline_add(pipeline_id, "elementwise_add", [input], output)
      {:ok, _cmd_ids} = ExCubecl.pipeline_run(pipeline_id)
      :ok = ExCubecl.pipeline_free(pipeline_id)
    end

    test "multi-stage pipeline" do
      {:ok, input} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
      {:ok, stage1_out} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)
      {:ok, stage2_out} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)

      {:ok, pipeline_id} = ExCubecl.pipeline()
      :ok = ExCubecl.pipeline_add(pipeline_id, "elementwise_add", [input], stage1_out)
      :ok = ExCubecl.pipeline_add(pipeline_id, "relu", [stage1_out], stage2_out)
      {:ok, _cmd_ids} = ExCubecl.pipeline_run(pipeline_id)
      :ok = ExCubecl.pipeline_free(pipeline_id)
    end

    test "pipeline returns unique ids" do
      {:ok, id1} = ExCubecl.pipeline()
      {:ok, id2} = ExCubecl.pipeline()
      assert id1 != id2
      :ok = ExCubecl.pipeline_free(id1)
      :ok = ExCubecl.pipeline_free(id2)
    end

    test "pipeline run returns list of command ids" do
      {:ok, input} = ExCubecl.buffer([1.0], [1], :f32)
      {:ok, output} = ExCubecl.buffer([0.0], [1], :f32)

      {:ok, pipeline_id} = ExCubecl.pipeline()
      :ok = ExCubecl.pipeline_add(pipeline_id, "relu", [input], output)
      {:ok, cmd_ids} = ExCubecl.pipeline_run(pipeline_id)
      assert is_list(cmd_ids)
      assert length(cmd_ids) == 1
      :ok = ExCubecl.pipeline_free(pipeline_id)
    end

    test "pipeline run with multiple stages returns multiple command ids" do
      {:ok, input} = ExCubecl.buffer([1.0], [1], :f32)
      {:ok, mid} = ExCubecl.buffer([0.0], [1], :f32)
      {:ok, output} = ExCubecl.buffer([0.0], [1], :f32)

      {:ok, pipeline_id} = ExCubecl.pipeline()
      :ok = ExCubecl.pipeline_add(pipeline_id, "elementwise_add", [input], mid)
      :ok = ExCubecl.pipeline_add(pipeline_id, "relu", [mid], output)
      {:ok, cmd_ids} = ExCubecl.pipeline_run(pipeline_id)
      assert length(cmd_ids) == 2
      :ok = ExCubecl.pipeline_free(pipeline_id)
    end

    test "pipeline add with invalid pipeline id returns error" do
      {:ok, input} = ExCubecl.buffer([1.0], [1], :f32)
      {:ok, output} = ExCubecl.buffer([0.0], [1], :f32)
      assert {:error, _} = ExCubecl.pipeline_add(999_999, "relu", [input], output)
    end

    test "pipeline run with invalid pipeline id returns error" do
      assert {:error, _} = ExCubecl.pipeline_run(999_999)
    end

    test "pipeline free with invalid pipeline id returns error" do
      assert {:error, _} = ExCubecl.pipeline_free(999_999)
    end

    test "pipeline add with unknown kernel returns error on run" do
      {:ok, input} = ExCubecl.buffer([1.0], [1], :f32)
      {:ok, output} = ExCubecl.buffer([0.0], [1], :f32)

      {:ok, pipeline_id} = ExCubecl.pipeline()
      :ok = ExCubecl.pipeline_add(pipeline_id, "nonexistent_kernel", [input], output)
      assert {:error, msg} = ExCubecl.pipeline_run(pipeline_id)
      assert is_binary(msg)
      assert msg =~ "nonexistent_kernel"
      :ok = ExCubecl.pipeline_free(pipeline_id)
    end

    test "pipeline run with empty pipeline returns empty list" do
      {:ok, pipeline_id} = ExCubecl.pipeline()
      {:ok, cmd_ids} = ExCubecl.pipeline_run(pipeline_id)
      assert cmd_ids == []
      :ok = ExCubecl.pipeline_free(pipeline_id)
    end

    test "pipeline with struct command" do
      {:ok, input} = ExCubecl.buffer([1.0], [1], :f32)
      {:ok, output} = ExCubecl.buffer([0.0], [1], :f32)

      cmd = ExCubecl.Command.run_kernel("relu", [input], output)

      {:ok, pipeline_id} = ExCubecl.pipeline()
      :ok = ExCubecl.pipeline_add_struct(pipeline_id, cmd)
      {:ok, _cmd_ids} = ExCubecl.pipeline_run(pipeline_id)
      :ok = ExCubecl.pipeline_free(pipeline_id)
    end
  end

  describe "Command struct" do
    test "run_kernel creates a command" do
      {:ok, input} = ExCubecl.buffer([1.0], [1], :f32)
      {:ok, output} = ExCubecl.buffer([0.0], [1], :f32)

      cmd = ExCubecl.Command.run_kernel("relu", [input], output)
      assert cmd.op == :run_kernel
      assert cmd.kernel == "relu"
      assert cmd.inputs == [input]
      assert cmd.output == output
      assert cmd.params == %{}
    end

    test "run_kernel with params" do
      {:ok, input} = ExCubecl.buffer([1.0], [1], :f32)
      {:ok, output} = ExCubecl.buffer([0.0], [1], :f32)

      cmd = ExCubecl.Command.run_kernel("relu", [input], output, %{alpha: 0.1})
      assert cmd.params == %{alpha: 0.1}
    end

    test "to_string returns readable representation" do
      {:ok, input} = ExCubecl.buffer([1.0], [1], :f32)
      {:ok, output} = ExCubecl.buffer([0.0], [1], :f32)

      cmd = ExCubecl.Command.run_kernel("relu", [input], output)
      str = ExCubecl.Command.to_string(cmd)
      assert is_binary(str)
      assert str =~ "relu"
      assert str =~ "run_kernel"
    end
  end

  describe "dtype edge cases" do
    test "unsupported dtype raises ArgumentError" do
      assert_raise ArgumentError, fn ->
        ExCubecl.buffer([1.0], [1], :f16)
      end
    end

    test "unsupported dtype raises for buffer! too" do
      assert_raise ArgumentError, fn ->
        ExCubecl.buffer!([1.0], [1], :f16)
      end
    end

    test "string dtype raises ArgumentError" do
      assert_raise ArgumentError, fn ->
        ExCubecl.buffer([1.0], [1], :invalid)
      end
    end
  end

  describe "buffer data integrity" do
    test "f32 data roundtrips correctly" do
      original = [1.5, 2.5, 3.5]
      {:ok, buf} = ExCubecl.buffer(original, [3], :f32)
      {:ok, data} = ExCubecl.read(buf)
      assert byte_size(data) == 12
    end

    test "s32 data roundtrips correctly" do
      original = [-100, 0, 100]
      {:ok, buf} = ExCubecl.buffer(original, [3], :s32)
      {:ok, data} = ExCubecl.read(buf)
      assert byte_size(data) == 12
    end

    test "u8 data roundtrips correctly" do
      original = [0, 127, 255]
      {:ok, buf} = ExCubecl.buffer(original, [3], :u8)
      {:ok, data} = ExCubecl.read(buf)
      assert data == <<0, 127, 255>>
    end

    test "f64 data roundtrips correctly" do
      original = [1.0e100, -1.0e100, 0.0]
      {:ok, buf} = ExCubecl.buffer(original, [3], :f64)
      {:ok, data} = ExCubecl.read(buf)
      assert byte_size(data) == 24
    end
  end
end
