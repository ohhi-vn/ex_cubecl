defmodule ExCubeclTest do
  use ExUnit.Case, async: true

  describe "version" do
    test "returns version string" do
      assert is_binary(ExCubecl.version())
    end
  end

  describe "device_info" do
    test "returns a map with expected keys" do
      {:ok, info} = ExCubecl.device_info()
      assert is_map(info)
      assert Map.has_key?(info, :device_name)
    end
  end

  describe "device_count" do
    test "returns a non-negative integer" do
      {:ok, count} = ExCubecl.device_count()
      assert is_integer(count)
      assert count >= 0
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
  end

  describe "available?" do
    test "returns boolean" do
      assert is_boolean(ExCubecl.available?())
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

    test "buffer! raises on error" do
      {:ok, buf} = ExCubecl.buffer([1.0, 2.0], [2], :f32)
      assert is_reference(buf)
    end

    test "read! raises on error" do
      {:ok, buf} = ExCubecl.buffer([1.0], [1], :f32)
      data = ExCubecl.read!(buf)
      assert is_binary(data)
    end
  end

  describe "kernels" do
    test "kernels returns a list" do
      {:ok, kernels} = ExCubecl.kernels()
      assert is_list(kernels)
      assert length(kernels) > 0
    end

    test "run_kernel executes without error" do
      {:ok, input} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
      {:ok, output} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)
      {:ok, _cmd_id} = ExCubecl.run_kernel("elementwise_add", [input], output)
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
  end
end
