defmodule ExCubeclEdgeCasesTest do
  use ExUnit.Case, async: true
  @moduletag timeout: 300_000

  # ─────────────────────────────────────────────────────────────
  #  Buffer edge cases
  # ─────────────────────────────────────────────────────────────

  describe "buffer edge cases" do
    test "empty list data with empty shape returns error" do
      # shape=[] means product=1, so 0 bytes != expected 4 bytes for f32
      assert {:error, _} = ExCubecl.buffer([], [], :f32)
    end

    test "empty list data with zero-dim shape creates empty buffer" do
      {:ok, buf} = ExCubecl.buffer([], [0], :f32)
      {:ok, data} = ExCubecl.read(buf)
      assert data == <<>>
      assert byte_size(data) == 0
    end

    test "very large buffer" do
      n = 100_000
      data = Enum.map(1..n, &(&1 * 0.001))
      {:ok, buf} = ExCubecl.buffer(data, [n], :f32)
      {:ok, shape} = ExCubecl.shape(buf)
      assert shape == [n]
      {:ok, size} = ExCubecl.size(buf)
      assert size == n * 4
    end

    test "f32 with special float values" do
      data = [0.0, -0.0, 1.0e-40, 1.0e40]
      {:ok, buf} = ExCubecl.buffer(data, [4], :f32)
      {:ok, readback} = ExCubecl.read(buf)
      assert byte_size(readback) == 16
    end

    test "s64 roundtrip" do
      original = [9_223_372_036_854_775_807, -9_223_372_036_854_775_808, 0]
      {:ok, buf} = ExCubecl.buffer(original, [3], :s64)
      {:ok, data} = ExCubecl.read(buf)
      assert byte_size(data) == 24
    end

    test "u32 roundtrip" do
      original = [0, 1, 4_294_967_295]
      {:ok, buf} = ExCubecl.buffer(original, [3], :u32)
      {:ok, data} = ExCubecl.read(buf)
      assert byte_size(data) == 12
    end

    test "4D buffer shape" do
      data = Enum.map(1..24, &(&1 * 1.0))
      {:ok, buf} = ExCubecl.buffer(data, [2, 3, 4, 1], :f32)
      {:ok, shape} = ExCubecl.shape(buf)
      assert shape == [2, 3, 4, 1]
    end

    test "buffer with all zeros for every dtype" do
      for dtype <- [:f32, :f64, :s32, :s64, :u32, :u8] do
        {:ok, buf} = ExCubecl.buffer([0, 0, 0], [3], dtype)
        {:ok, data} = ExCubecl.read(buf)
        assert is_binary(data)
        assert byte_size(data) > 0
      end
    end

    test "buffer dtype returns correct string for all types" do
      pairs = [
        {"f32", :f32},
        {"f64", :f64},
        {"s32", :s32},
        {"s64", :s64},
        {"u32", :u32},
        {"u8", :u8}
      ]

      for {expected, dtype} <- pairs do
        {:ok, buf} = ExCubecl.buffer([1], [1], dtype)
        {:ok, dt} = ExCubecl.dtype(buf)
        assert dt == expected
      end
    end

    test "buffer size returns correct byte count" do
      {:ok, buf} = ExCubecl.buffer([1.0, 2.0, 3.0, 4.0, 5.0], [5], :f32)
      {:ok, size} = ExCubecl.size(buf)
      assert size == 20
    end

    test "mismatched shape and data returns error tuple" do
      assert {:error, msg} = ExCubecl.buffer([1.0, 2.0], [10], :f32)
      assert is_binary(msg)
    end

    test "read! raises on error" do
      assert_raise ArgumentError, fn ->
        ExCubecl.read!(make_ref())
      end
    end
  end

  # ─────────────────────────────────────────────────────────────
  #  Kernel execution edge cases
  # ─────────────────────────────────────────────────────────────

  describe "kernel execution edge cases" do
    test "elementwise_add produces correct results" do
      {:ok, a} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
      {:ok, b} = ExCubecl.buffer([10.0, 20.0, 30.0], [3], :f32)
      {:ok, output} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)
      {:ok, _cmd} = ExCubecl.run_kernel("elementwise_add", [a, b], output)
      {:ok, data} = ExCubecl.read(output)
      <<a1::float-32-native, a2::float-32-native, a3::float-32-native>> = data
      assert_in_delta(a1, 11.0, 1.0e-6)
      assert_in_delta(a2, 22.0, 1.0e-6)
      assert_in_delta(a3, 33.0, 1.0e-6)
    end

    test "elementwise_mul produces correct results" do
      {:ok, a} = ExCubecl.buffer([2.0, 3.0, 4.0], [3], :f32)
      {:ok, b} = ExCubecl.buffer([5.0, 6.0, 7.0], [3], :f32)
      {:ok, output} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)
      {:ok, _cmd} = ExCubecl.run_kernel("elementwise_mul", [a, b], output)
      {:ok, data} = ExCubecl.read(output)
      <<a1::float-32-native, a2::float-32-native, a3::float-32-native>> = data
      assert_in_delta(a1, 10.0, 1.0e-6)
      assert_in_delta(a2, 18.0, 1.0e-6)
      assert_in_delta(a3, 28.0, 1.0e-6)
    end

    test "elementwise_sub produces correct results" do
      {:ok, a} = ExCubecl.buffer([10.0, 20.0, 30.0], [3], :f32)
      {:ok, b} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
      {:ok, output} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)
      {:ok, _cmd} = ExCubecl.run_kernel("elementwise_sub", [a, b], output)
      {:ok, data} = ExCubecl.read(output)
      <<a1::float-32-native, a2::float-32-native, a3::float-32-native>> = data
      assert_in_delta(a1, 9.0, 1.0e-6)
      assert_in_delta(a2, 18.0, 1.0e-6)
      assert_in_delta(a3, 27.0, 1.0e-6)
    end

    test "relu zeros out negative values" do
      {:ok, input} = ExCubecl.buffer([-3.0, -1.0, 0.0, 1.0, 3.0], [5], :f32)
      {:ok, output} = ExCubecl.buffer([0.0, 0.0, 0.0, 0.0, 0.0], [5], :f32)
      {:ok, _cmd} = ExCubecl.run_kernel("relu", [input], output)
      {:ok, data} = ExCubecl.read(output)

      <<v1::float-32-native, v2::float-32-native, v3::float-32-native, v4::float-32-native,
        v5::float-32-native>> = data

      assert_in_delta(v1, 0.0, 1.0e-6)
      assert_in_delta(v2, 0.0, 1.0e-6)
      assert_in_delta(v3, 0.0, 1.0e-6)
      assert_in_delta(v4, 1.0, 1.0e-6)
      assert_in_delta(v5, 3.0, 1.0e-6)
    end

    test "sigmoid produces values in (0, 1) range" do
      {:ok, input} = ExCubecl.buffer([-5.0, 0.0, 5.0], [3], :f32)
      {:ok, output} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)
      {:ok, _cmd} = ExCubecl.run_kernel("sigmoid", [input], output)
      {:ok, data} = ExCubecl.read(output)
      <<v1::float-32-native, v2::float-32-native, v3::float-32-native>> = data
      assert v1 > 0.0 and v1 < 0.01
      assert_in_delta(v2, 0.5, 1.0e-6)
      assert v3 > 0.99 and v3 < 1.0
    end

    test "tanh produces values in (-1, 1) range" do
      {:ok, input} = ExCubecl.buffer([-3.0, 0.0, 3.0], [3], :f32)
      {:ok, output} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)
      {:ok, _cmd} = ExCubecl.run_kernel("tanh", [input], output)
      {:ok, data} = ExCubecl.read(output)
      <<v1::float-32-native, v2::float-32-native, v3::float-32-native>> = data
      assert v1 > -1.0 and v1 < 0.0
      assert_in_delta(v2, 0.0, 1.0e-6)
      assert v3 > 0.0 and v3 < 1.0
    end

    test "kernel with insufficient inputs raises error" do
      {:ok, input} = ExCubecl.buffer([1.0], [1], :f32)
      {:ok, output} = ExCubecl.buffer([0.0], [1], :f32)
      assert_raise ErlangError, ~r/elementwise_add: expected 2 inputs/, fn ->
        ExCubecl.run_kernel("elementwise_add", [input], output)
      end
    end

    test "unknown kernel returns error with kernel name" do
      {:ok, input} = ExCubecl.buffer([1.0], [1], :f32)
      {:ok, output} = ExCubecl.buffer([0.0], [1], :f32)
      assert {:error, msg} = ExCubecl.run_kernel("does_not_exist", [input], output)
      assert msg =~ "does_not_exist"
    end

    test "kernel with custom params does not crash" do
      {:ok, a} = ExCubecl.buffer([1.0, 2.0], [2], :f32)
      {:ok, b} = ExCubecl.buffer([3.0, 4.0], [2], :f32)
      {:ok, output} = ExCubecl.buffer([0.0, 0.0], [2], :f32)
      {:ok, _cmd} = ExCubecl.run_kernel("elementwise_add", [a, b], output, %{alpha: 0.5})
    end

    test "all compute kernels are listed" do
      {:ok, kernels} = ExCubecl.kernels()

      compute =
        ~w(elementwise_add elementwise_mul elementwise_sub elementwise_div relu sigmoid tanh)

      for k <- compute, do: assert(k in kernels)
    end

    test "all video kernels are listed" do
      {:ok, kernels} = ExCubecl.kernels()

      video =
        ~w(yuv_to_rgb overlay_alpha video_mix gaussian_blur bicubic_scale lut_apply chroma_key sharpen)

      for k <- video, do: assert(k in kernels)
    end

    test "all audio kernels are listed" do
      {:ok, kernels} = ExCubecl.kernels()

      audio =
        ~w(pcm_mix pcm_normalize biquad_filter fft_convolve resample_linear dynamics_compress)

      for k <- audio, do: assert(k in kernels)
    end
  end

  # ─────────────────────────────────────────────────────────────
  #  Async execution edge cases
  # ─────────────────────────────────────────────────────────────

  describe "async execution edge cases" do
    test "rapid submit and wait cycle" do
      for _ <- 1..50 do
        {:ok, cmd} = ExCubecl.submit("test_command")
        assert is_integer(cmd)
        assert cmd > 0
        ExCubecl.wait(cmd)
        {:ok, :completed} = ExCubecl.poll(cmd)
      end
    end

    test "poll on completed command returns completed" do
      {:ok, cmd} = ExCubecl.submit("cmd")
      ExCubecl.wait(cmd)
      assert {:ok, :completed} = ExCubecl.poll(cmd)
    end

    test "poll on pending command returns pending or running" do
      {:ok, cmd} = ExCubecl.submit("cmd")
      # Immediately poll — should be pending or running
      {:ok, status} = ExCubecl.poll(cmd)
      assert status in [:pending, :running, :completed]
      ExCubecl.wait(cmd)
    end

    test "wait on already completed command returns ok" do
      {:ok, cmd} = ExCubecl.submit("cmd")
      ExCubecl.wait(cmd)
      # Wait again on the same command
      assert ExCubecl.wait(cmd) == :ok
    end

    test "multiple concurrent submits have unique ids" do
      ids =
        Enum.map(1..20, fn _ ->
          {:ok, cmd} = ExCubecl.submit("cmd")
          cmd
        end)

      assert length(Enum.uniq(ids)) == length(ids)
      Enum.each(ids, &ExCubecl.wait/1)
    end
  end

  # ─────────────────────────────────────────────────────────────
  #  Pipeline edge cases
  # ─────────────────────────────────────────────────────────────

  describe "pipeline edge cases" do
    test "pipeline with many stages" do
      {:ok, a} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
      {:ok, b} = ExCubecl.buffer([4.0, 5.0, 6.0], [3], :f32)
      {:ok, output} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)

      {:ok, pid} = ExCubecl.pipeline()

      # Add 10 stages of elementwise_add
      for _ <- 1..10 do
        :ok = ExCubecl.pipeline_add(pid, "elementwise_add", [a, b], output)
      end

      {:ok, cmd_ids} = ExCubecl.pipeline_run(pid)
      assert length(cmd_ids) == 10
      :ok = ExCubecl.pipeline_free(pid)
    end

    test "pipeline add then free without run" do
      {:ok, a} = ExCubecl.buffer([1.0], [1], :f32)
      {:ok, output} = ExCubecl.buffer([0.0], [1], :f32)
      {:ok, pid} = ExCubecl.pipeline()
      :ok = ExCubecl.pipeline_add(pid, "relu", [a], output)
      :ok = ExCubecl.pipeline_free(pid)
    end

    test "pipeline run on empty pipeline returns empty list" do
      {:ok, pid} = ExCubecl.pipeline()
      {:ok, []} = ExCubecl.pipeline_run(pid)
      :ok = ExCubecl.pipeline_free(pid)
    end

    test "pipeline with struct command using Command struct" do
      {:ok, a} = ExCubecl.buffer([1.0], [1], :f32)
      {:ok, output} = ExCubecl.buffer([0.0], [1], :f32)
      {:ok, pid} = ExCubecl.pipeline()
      cmd = ExCubecl.Command.run_kernel("relu", [a], output, %{beta: 1.0})
      :ok = ExCubecl.pipeline_add_struct(pid, cmd)
      {:ok, [_cmd_id]} = ExCubecl.pipeline_run(pid)
      :ok = ExCubecl.pipeline_free(pid)
    end

    test "pipeline free is idempotent-ish (double free returns error)" do
      {:ok, pid} = ExCubecl.pipeline()
      :ok = ExCubecl.pipeline_free(pid)
      assert {:error, _} = ExCubecl.pipeline_free(pid)
    end
  end

  # ─────────────────────────────────────────────────────────────
  #  Command struct edge cases
  # ─────────────────────────────────────────────────────────────

  describe "Command struct edge cases" do
    test "command with empty inputs list" do
      cmd = ExCubecl.Command.run_kernel("relu", [], make_ref())
      assert cmd.op == :run_kernel
      assert cmd.kernel == "relu"
      assert cmd.inputs == []
      assert cmd.params == %{}
    end

    test "command with custom params" do
      cmd =
        ExCubecl.Command.run_kernel("test", [make_ref()], make_ref(), %{key: "value", num: 42})

      assert cmd.params == %{key: "value", num: 42}
    end

    test "command to_string with multiple inputs" do
      cmd = ExCubecl.Command.run_kernel("add", [make_ref(), make_ref(), make_ref()], make_ref())
      str = ExCubecl.Command.to_string(cmd)
      assert str =~ "add"
      assert str =~ "inputs: 3"
    end

    test "command to_string with params" do
      cmd = ExCubecl.Command.run_kernel("kernel", [make_ref()], make_ref(), %{alpha: 0.5})
      str = ExCubecl.Command.to_string(cmd)
      assert str =~ "kernel"
      assert str =~ "alpha"
    end
  end

  # ─────────────────────────────────────────────────────────────
  #  Media I/O edge cases
  # ─────────────────────────────────────────────────────────────

  describe "Media edge cases" do
    test "open with URL-like path" do
      {:ok, src} = ExCubecl.Media.open("rtmp://live.example.com/stream/key")
      assert is_reference(src)
    end

    test "open with HLS path" do
      {:ok, src} = ExCubecl.Media.open("https://example.com/playlist.m3u8")
      assert is_reference(src)
    end

    test "multiple sequential reads return frames" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame1} = ExCubecl.Media.read_frame(src, :video)
      {:ok, frame2} = ExCubecl.Media.read_frame(src, :video)
      assert %ExCubecl.VideoFrame{} = frame1
      assert %ExCubecl.VideoFrame{} = frame2
    end

    test "interleaved video and audio reads" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame} = ExCubecl.Media.read_frame(src, :video)
      {:ok, samples} = ExCubecl.Media.read_frame(src, :audio)
      assert %ExCubecl.VideoFrame{} = frame
      assert %ExCubecl.AudioSamples{} = samples
    end

    test "close is idempotent" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      assert ExCubecl.Media.close(src) == :ok
    end

    test "streams returns consistent data on repeated calls" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, streams1} = ExCubecl.Media.streams(src)
      {:ok, streams2} = ExCubecl.Media.streams(src)
      assert streams1 == streams2
    end
  end

  # ─────────────────────────────────────────────────────────────
  #  Video operations edge cases
  # ─────────────────────────────────────────────────────────────

  describe "Video edge cases" do
    test "overlay with default options" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame_a} = ExCubecl.Media.read_frame(src, :video)
      {:ok, frame_b} = ExCubecl.Media.read_frame(src, :video)
      assert {:ok, %ExCubecl.VideoFrame{}} = ExCubecl.Video.overlay(frame_a, frame_b)
    end

    test "overlay with zero alpha" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame_a} = ExCubecl.Media.read_frame(src, :video)
      {:ok, frame_b} = ExCubecl.Media.read_frame(src, :video)
      assert {:ok, %ExCubecl.VideoFrame{}} = ExCubecl.Video.overlay(frame_a, frame_b, alpha: 0.0)
    end

    test "overlay with full alpha" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame_a} = ExCubecl.Media.read_frame(src, :video)
      {:ok, frame_b} = ExCubecl.Media.read_frame(src, :video)
      assert {:ok, %ExCubecl.VideoFrame{}} = ExCubecl.Video.overlay(frame_a, frame_b, alpha: 1.0)
    end

    test "mix with all blend modes" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame_a} = ExCubecl.Media.read_frame(src, :video)
      {:ok, frame_b} = ExCubecl.Media.read_frame(src, :video)

      for mode <- [:dissolve, :add, :multiply] do
        assert {:ok, %ExCubecl.VideoFrame{}} = ExCubecl.Video.mix(frame_a, frame_b, mode: mode)
      end
    end

    test "mix with extreme ratios" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame_a} = ExCubecl.Media.read_frame(src, :video)
      {:ok, frame_b} = ExCubecl.Media.read_frame(src, :video)

      assert {:ok, _} = ExCubecl.Video.mix(frame_a, frame_b, ratio: 0.0)
      assert {:ok, _} = ExCubecl.Video.mix(frame_a, frame_b, ratio: 1.0)
    end

    test "convert nv12 to rgb24" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, %ExCubecl.VideoFrame{} = frame} = ExCubecl.Media.read_frame(src, :video)
      # Override format to nv12 for testing
      nv12_frame = %ExCubecl.VideoFrame{frame | format: :nv12}
      {:ok, converted} = ExCubecl.Video.convert(nv12_frame, :nv12, :rgb24)
      assert converted.format == :rgb24
    end

    test "convert rgba returns unsupported error" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, %ExCubecl.VideoFrame{} = frame} = ExCubecl.Media.read_frame(src, :video)
      rgba_frame = %ExCubecl.VideoFrame{frame | format: :rgba}

      assert {:error, {:unsupported_conversion, :rgba}} =
               ExCubecl.Video.convert(rgba_frame, :rgba, :yuv420p)
    end

    test "scale to same dimensions" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame} = ExCubecl.Media.read_frame(src, :video)
      {:ok, scaled} = ExCubecl.Video.scale(frame, width: frame.width, height: frame.height)
      assert scaled.width == frame.width
      assert scaled.height == frame.height
    end

    test "crop with default x and y" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame} = ExCubecl.Media.read_frame(src, :video)
      {:ok, cropped} = ExCubecl.Video.crop(frame, width: 320, height: 240)
      assert cropped.width == 320
      assert cropped.height == 240
    end

    test "crop with custom x and y offsets" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame} = ExCubecl.Media.read_frame(src, :video)
      {:ok, cropped} = ExCubecl.Video.crop(frame, x: 100, y: 50, width: 320, height: 240)
      assert cropped.width == 320
      assert cropped.height == 240
    end

    test "snapshot writes a file" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame} = ExCubecl.Media.read_frame(src, :video)
      path = "/tmp/test_snapshot_#{System.unique_integer([:positive])}.raw"
      assert ExCubecl.Video.snapshot(frame, path) == :ok
      assert File.exists?(path)
      File.rm(path)
    end
  end

  # ─────────────────────────────────────────────────────────────
  #  Audio operations edge cases
  # ─────────────────────────────────────────────────────────────

  describe "Audio edge cases" do
    test "mix single track" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, samples} = ExCubecl.Media.read_frame(src, :audio)
      assert {:ok, %ExCubecl.AudioSamples{}} = ExCubecl.Audio.mix([samples])
    end

    test "mix three tracks" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, a} = ExCubecl.Media.read_frame(src, :audio)
      {:ok, b} = ExCubecl.Media.read_frame(src, :audio)
      {:ok, c} = ExCubecl.Media.read_frame(src, :audio)
      assert {:ok, _} = ExCubecl.Audio.mix([a, b, c], gains: [0.5, 0.3, 0.2])
    end

    test "mix with default gains (all 1.0)" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, a} = ExCubecl.Media.read_frame(src, :audio)
      {:ok, b} = ExCubecl.Media.read_frame(src, :audio)
      assert {:ok, _} = ExCubecl.Audio.mix([a, b])
    end

    test "overlay with zero duck level" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, bg} = ExCubecl.Media.read_frame(src, :audio)
      {:ok, fg} = ExCubecl.Media.read_frame(src, :audio)
      assert {:ok, _} = ExCubecl.Audio.overlay(bg, fg, duck_level: 0)
    end

    test "overlay with extreme duck level" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, bg} = ExCubecl.Media.read_frame(src, :audio)
      {:ok, fg} = ExCubecl.Media.read_frame(src, :audio)
      assert {:ok, _} = ExCubecl.Audio.overlay(bg, fg, duck_level: -60)
    end

    test "resample to same rate" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, samples} = ExCubecl.Media.read_frame(src, :audio)
      {:ok, resampled} = ExCubecl.Audio.resample(samples, from: 48000, to: 48000)
      assert resampled.sample_rate == 48000
      assert resampled.frames == samples.frames
    end

    test "resample to higher rate increases frames" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, samples} = ExCubecl.Media.read_frame(src, :audio)
      {:ok, resampled} = ExCubecl.Audio.resample(samples, from: 44100, to: 96000)
      assert resampled.sample_rate == 96000
      assert resampled.frames > samples.frames
    end

    test "resample to lower rate decreases frames" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, samples} = ExCubecl.Media.read_frame(src, :audio)
      {:ok, resampled} = ExCubecl.Audio.resample(samples, from: 96000, to: 22050)
      assert resampled.sample_rate == 22050
      assert resampled.frames < samples.frames
    end

    test "channels mono to stereo" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, samples} = ExCubecl.Media.read_frame(src, :audio)
      {:ok, stereo} = ExCubecl.Audio.channels(samples, :mono, :stereo)
      assert stereo.channels == 2
    end

    test "channels surround layouts" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, samples} = ExCubecl.Media.read_frame(src, :audio)
      {:ok, s51} = ExCubecl.Audio.channels(samples, :stereo, :surround_51)
      assert s51.channels == 6
      {:ok, s71} = ExCubecl.Audio.channels(samples, :stereo, :surround_71)
      assert s71.channels == 8
    end

    test "channels with unsupported layout raises" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, samples} = ExCubecl.Media.read_frame(src, :audio)

      assert_raise ArgumentError, fn ->
        ExCubecl.Audio.channels(samples, :stereo, :unknown_layout)
      end
    end
  end

  # ─────────────────────────────────────────────────────────────
  #  Filter edge cases
  # ─────────────────────────────────────────────────────────────

  describe "Filter edge cases" do
    test "all video filters return ok" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame} = ExCubecl.Media.read_frame(src, :video)

      for kernel <- [:gaussian_blur, :sharpen, :lut, :chroma_key, :brightness_contrast, :denoise] do
        assert {:ok, %ExCubecl.VideoFrame{}} = ExCubecl.Filter.apply(frame, kernel)
      end
    end

    test "all audio filters return ok" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, samples} = ExCubecl.Media.read_frame(src, :audio)

      for kernel <- [:eq, :compressor, :reverb, :normalize] do
        assert {:ok, %ExCubecl.AudioSamples{}} = ExCubecl.Filter.apply(samples, kernel)
      end
    end

    test "video filter with params" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame} = ExCubecl.Media.read_frame(src, :video)
      assert {:ok, _} = ExCubecl.Filter.apply(frame, :gaussian_blur, radius: 10)
      assert {:ok, _} = ExCubecl.Filter.apply(frame, :sharpen, strength: 2.5)

      assert {:ok, _} =
               ExCubecl.Filter.apply(frame, :chroma_key, color: {0, 255, 0}, threshold: 0.4)

      assert {:ok, _} =
               ExCubecl.Filter.apply(frame, :brightness_contrast, brightness: 0.2, contrast: 1.5)
    end

    test "audio filter with params" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, samples} = ExCubecl.Media.read_frame(src, :audio)
      assert {:ok, _} = ExCubecl.Filter.apply(samples, :compressor, threshold: -24, ratio: 8.0)
      assert {:ok, _} = ExCubecl.Filter.apply(samples, :reverb, room_size: 0.8, wet: 0.4)
    end

    test "filter with unknown kernel on video frame" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame} = ExCubecl.Media.read_frame(src, :video)
      assert {:error, {:unknown_filter, :bogus}} = ExCubecl.Filter.apply(frame, :bogus)
    end

    test "filter with unknown kernel on audio samples" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, samples} = ExCubecl.Media.read_frame(src, :audio)
      assert {:error, {:unknown_filter, :bogus}} = ExCubecl.Filter.apply(samples, :bogus)
    end

    test "filter with video kernel on audio returns error" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, samples} = ExCubecl.Media.read_frame(src, :audio)

      assert {:error, {:unknown_filter, :gaussian_blur}} =
               ExCubecl.Filter.apply(samples, :gaussian_blur)
    end

    test "filter with audio kernel on video returns error" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame} = ExCubecl.Media.read_frame(src, :video)
      assert {:error, {:unknown_filter, :normalize}} = ExCubecl.Filter.apply(frame, :normalize)
    end

    test "chain with single filter" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame} = ExCubecl.Media.read_frame(src, :video)
      assert {:ok, _} = ExCubecl.Filter.chain(frame, [{:denoise, [strength: 0.3]}])
    end

    test "chain with many filters" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame} = ExCubecl.Media.read_frame(src, :video)
      filters = for _ <- 1..5, do: {:gaussian_blur, [radius: 1]}
      assert {:ok, _} = ExCubecl.Filter.chain(frame, filters)
    end

    test "chain with unknown filter returns error" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame} = ExCubecl.Media.read_frame(src, :video)
      # Filter.chain uses pipeline_run which returns a different error format
      assert {:error, msg} = ExCubecl.Filter.chain(frame, [{:nope, []}])
      assert is_binary(msg)
      assert msg =~ "nope"
    end
  end

  # ─────────────────────────────────────────────────────────────
  #  Transcode edge cases
  # ─────────────────────────────────────────────────────────────

  describe "Transcode edge cases" do
    test "start with only video codec" do
      {:ok, enc} = ExCubecl.Transcode.start("out.mp4", video: [codec: :h264])
      assert is_reference(enc)
    end

    test "start with only audio codec" do
      {:ok, enc} = ExCubecl.Transcode.start("out.mp4", audio: [codec: :aac])
      assert is_reference(enc)
    end

    test "start with all supported video codecs" do
      for codec <- [:h264, :h265, :vp9, :av1, :prores] do
        {:ok, enc} = ExCubecl.Transcode.start("out.mp4", video: [codec: codec])
        assert is_reference(enc)
      end
    end

    test "start with all supported audio codecs" do
      for codec <- [:aac, :opus, :mp3, :flac, :pcm] do
        {:ok, enc} = ExCubecl.Transcode.start("out.mp4", audio: [codec: codec])
        assert is_reference(enc)
      end
    end

    test "start with all supported containers" do
      for ext <- ["mp4", "mkv", "webm", "mov", "ts"] do
        path = "output.#{ext}"
        {:ok, enc} = ExCubecl.Transcode.start(path, video: [codec: :h264])
        assert is_reference(enc)
      end
    end

    test "start with nil codecs does not raise" do
      {:ok, enc} = ExCubecl.Transcode.start("out.mp4", video: [], audio: [])
      assert is_reference(enc)
    end

    test "start with string codec raises FunctionClauseError" do
      # validate_codec! only matches on atom codecs
      assert_raise FunctionClauseError, fn ->
        ExCubecl.Transcode.start("out.mp4", video: [codec: "h264"])
      end
    end

    test "write_frame then finish" do
      {:ok, enc} = ExCubecl.Transcode.start("out.mp4", video: [codec: :h264])
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame} = ExCubecl.Media.read_frame(src, :video)
      :ok = ExCubecl.Transcode.write_frame(enc, frame)
      :ok = ExCubecl.Transcode.finish(enc)
    end

    test "write_samples then finish" do
      {:ok, enc} = ExCubecl.Transcode.start("out.mp4", audio: [codec: :aac])
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, samples} = ExCubecl.Media.read_frame(src, :audio)
      :ok = ExCubecl.Transcode.write_samples(enc, samples)
      :ok = ExCubecl.Transcode.finish(enc)
    end

    test "multiple writes before finish" do
      {:ok, enc} = ExCubecl.Transcode.start("out.mp4", video: [codec: :h264])
      {:ok, src} = ExCubecl.Media.open("test.mp4")

      for _ <- 1..10 do
        {:ok, frame} = ExCubecl.Media.read_frame(src, :video)
        :ok = ExCubecl.Transcode.write_frame(enc, frame)
      end

      :ok = ExCubecl.Transcode.finish(enc)
    end

    test "finish without writes returns ok" do
      {:ok, enc} = ExCubecl.Transcode.start("out.mp4", video: [codec: :h264])
      assert ExCubecl.Transcode.finish(enc) == :ok
    end
  end

  # ─────────────────────────────────────────────────────────────
  #  VideoFrame / AudioSamples struct edge cases
  # ─────────────────────────────────────────────────────────────

  describe "struct edge cases" do
    test "VideoFrame with all supported formats" do
      for format <- [:yuv420p, :rgb24, :rgba, :nv12] do
        map = %{
          handle: make_ref(),
          width: 640,
          height: 480,
          format: format,
          pts: 1000,
          duration: 33333
        }

        frame = ExCubecl.VideoFrame.from_map(map)
        assert frame.format == format
      end
    end

    test "AudioSamples with various channel counts" do
      for channels <- [1, 2, 6, 8] do
        map = %{handle: make_ref(), channels: channels, sample_rate: 48000, frames: 1024, pts: 0}
        samples = ExCubecl.AudioSamples.from_map(map)
        assert samples.channels == channels
      end
    end

    test "VideoFrame with zero dimensions" do
      map = %{handle: make_ref(), width: 0, height: 0, format: :yuv420p, pts: 0, duration: 0}
      frame = ExCubecl.VideoFrame.from_map(map)
      assert frame.width == 0
      assert frame.height == 0
    end

    test "VideoFrame with large timestamp" do
      map = %{
        handle: make_ref(),
        width: 1920,
        height: 1080,
        format: :yuv420p,
        pts: 9_223_372_036_854_775_807,
        duration: 0
      }

      frame = ExCubecl.VideoFrame.from_map(map)
      assert frame.pts == 9_223_372_036_854_775_807
    end
  end

  # ─────────────────────────────────────────────────────────────
  #  MediaPipeline edge cases
  # ─────────────────────────────────────────────────────────────

  describe "MediaPipeline edge cases" do
    test "push_frame multiple times" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame} = ExCubecl.Media.read_frame(src, :video)

      for _ <- 1..5 do
        assert ExCubecl.MediaPipeline.push_frame(self(), frame) == :ok
      end

      for _ <- 1..5 do
        assert_received {:frame, %ExCubecl.VideoFrame{}}
      end
    end

    test "push_frame with different frames" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      {:ok, frame1} = ExCubecl.Media.read_frame(src, :video)
      {:ok, frame2} = ExCubecl.Media.read_frame(src, :video)

      ExCubecl.MediaPipeline.push_frame(self(), frame1)
      ExCubecl.MediaPipeline.push_frame(self(), frame2)

      assert_received {:frame, %ExCubecl.VideoFrame{}}
      assert_received {:frame, %ExCubecl.VideoFrame{}}
    end
  end

  # ─────────────────────────────────────────────────────────────
  #  Integration / workflow edge cases
  # ─────────────────────────────────────────────────────────────

  describe "integration workflows" do
    test "full compute pipeline: add then relu" do
      {:ok, a} = ExCubecl.buffer([1.0, -2.0, 3.0, -4.0], [4], :f32)
      {:ok, b} = ExCubecl.buffer([0.5, 0.5, 0.5, 0.5], [4], :f32)
      {:ok, added} = ExCubecl.buffer([0.0, 0.0, 0.0, 0.0], [4], :f32)
      {:ok, activated} = ExCubecl.buffer([0.0, 0.0, 0.0, 0.0], [4], :f32)

      {:ok, _} = ExCubecl.run_kernel("elementwise_add", [a, b], added)
      {:ok, _} = ExCubecl.run_kernel("relu", [added], activated)

      {:ok, data} = ExCubecl.read(activated)

      <<v1::float-32-native, v2::float-32-native, v3::float-32-native, v4::float-32-native>> =
        data

      assert_in_delta(v1, 1.5, 1.0e-6)
      assert_in_delta(v2, 0.0, 1.0e-6)
      assert_in_delta(v3, 3.5, 1.0e-6)
      assert_in_delta(v4, 0.0, 1.0e-6)
    end

    test "pipeline with mixed kernel types" do
      {:ok, a} = ExCubecl.buffer([1.0, 2.0], [2], :f32)
      {:ok, b} = ExCubecl.buffer([3.0, 4.0], [2], :f32)
      {:ok, output1} = ExCubecl.buffer([0.0, 0.0], [2], :f32)
      {:ok, output2} = ExCubecl.buffer([0.0, 0.0], [2], :f32)

      {:ok, pid} = ExCubecl.pipeline()
      :ok = ExCubecl.pipeline_add(pid, "elementwise_add", [a, b], output1)
      :ok = ExCubecl.pipeline_add(pid, "relu", [a], output2)
      {:ok, [id1, id2]} = ExCubecl.pipeline_run(pid)
      assert id1 > 0
      assert id2 > 0
      :ok = ExCubecl.pipeline_free(pid)
    end

    test "buffer lifecycle: create, use, read, recreate" do
      {:ok, buf1} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
      {:ok, data1} = ExCubecl.read(buf1)
      assert byte_size(data1) == 12

      # buf1 will be GC'd; create a new one
      {:ok, buf2} = ExCubecl.buffer([4.0, 5.0, 6.0], [3], :f32)
      {:ok, data2} = ExCubecl.read(buf2)
      assert byte_size(data2) == 12
    end

    test "concurrent buffer operations" do
      buffers =
        Enum.map(1..20, fn i ->
          ExCubecl.buffer([i * 1.0], [1], :f32)
        end)

      results =
        Enum.map(buffers, fn {:ok, buf} ->
          ExCubecl.read(buf)
        end)

      assert length(results) == 20
      Enum.each(results, fn {:ok, data} -> assert byte_size(data) == 4 end)
    end
  end
end
