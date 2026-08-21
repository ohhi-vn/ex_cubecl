defmodule ExCubeclCoverageTest do
  use ExUnit.Case, async: true
  @moduletag timeout: 300_000

  # ── Helpers ─────────────────────────────────────────────────

  defp make_frame(opts \\ []) do
    width = Keyword.get(opts, :width, 64)
    height = Keyword.get(opts, :height, 64)
    format = Keyword.get(opts, :format, :yuv420p)

    bytes =
      case format do
        f when f in [:yuv420p, :nv12] -> div(width * height * 3, 2)
        :rgb24 -> width * height * 3
        :rgba -> width * height * 4
      end

    data = :binary.copy(<<128>>, bytes)
    {:ok, handle} = ExCubecl.buffer(data, [bytes], :u8)

    %ExCubecl.VideoFrame{
      handle: handle,
      width: width,
      height: height,
      format: format,
      pts: 0,
      duration: 33_333
    }
  end

  defp make_samples(opts \\ []) do
    channels = Keyword.get(opts, :channels, 2)
    frames = Keyword.get(opts, :frames, 256)
    sample_rate = Keyword.get(opts, :sample_rate, 48_000)

    data =
      for i <- 1..(frames * channels),
          into: <<>>,
          do: <<:math.sin(i * 0.01) * 0.5::float-32-native>>

    {:ok, handle} = ExCubecl.buffer(data, [frames * channels], :f32)

    %ExCubecl.AudioSamples{
      handle: handle,
      channels: channels,
      sample_rate: sample_rate,
      frames: frames,
      pts: 0
    }
  end

  # ── ExCubecl.Video ──────────────────────────────────────────

  describe "Video.scale validation" do
    test "missing width option" do
      frame = make_frame()
      assert {:error, {:missing_option, :width}} = ExCubecl.Video.scale(frame, height: 32)
    end

    test "missing height option" do
      frame = make_frame()
      assert {:error, {:missing_option, :height}} = ExCubecl.Video.scale(frame, width: 32)
    end

    test "zero width" do
      frame = make_frame()

      assert {:error, {:invalid_dimension, :width, 0}} =
               ExCubecl.Video.scale(frame, width: 0, height: 32)
    end

    test "negative height" do
      frame = make_frame()

      assert {:error, {:invalid_dimension, :height, -5}} =
               ExCubecl.Video.scale(frame, width: 32, height: -5)
    end

    test "non-integer width" do
      frame = make_frame()

      assert {:error, {:invalid_dimension, :width, "big"}} =
               ExCubecl.Video.scale(frame, width: "big", height: 32)
    end

    test "scale up rgb24 frame" do
      frame = make_frame(format: :rgb24)
      assert {:ok, scaled} = ExCubecl.Video.scale(frame, width: 128, height: 128)
      assert scaled.width == 128
      assert scaled.height == 128
      assert scaled.format == :rgb24
    end

    test "scale rgba frame" do
      frame = make_frame(format: :rgba)
      assert {:ok, scaled} = ExCubecl.Video.scale(frame, width: 32, height: 32)
      assert scaled.width == 32
    end

    test "scale nv12 frame" do
      frame = make_frame(format: :nv12)
      assert {:ok, scaled} = ExCubecl.Video.scale(frame, width: 32, height: 32)
      assert scaled.width == 32
    end
  end

  describe "Video.crop validation" do
    test "missing width" do
      frame = make_frame()
      assert {:error, {:missing_option, :width}} = ExCubecl.Video.crop(frame, height: 32)
    end

    test "missing height" do
      frame = make_frame()
      assert {:error, {:missing_option, :height}} = ExCubecl.Video.crop(frame, width: 32)
    end

    test "negative x offset" do
      frame = make_frame()

      assert {:error, {:invalid_offset, :x, -1}} =
               ExCubecl.Video.crop(frame, x: -1, y: 0, width: 32, height: 32)
    end

    test "negative y offset" do
      frame = make_frame()

      assert {:error, {:invalid_offset, :y, -1}} =
               ExCubecl.Video.crop(frame, x: 0, y: -1, width: 32, height: 32)
    end

    test "zero width" do
      frame = make_frame()

      assert {:error, {:invalid_dimension, :width, 0}} =
               ExCubecl.Video.crop(frame, x: 0, y: 0, width: 0, height: 32)
    end

    test "crop out of bounds" do
      frame = make_frame(width: 64, height: 64)

      assert {:error, {:crop_out_of_bounds, 0, 0, 128, 32, 64, 64}} =
               ExCubecl.Video.crop(frame, x: 0, y: 0, width: 128, height: 32)
    end

    test "crop out of bounds vertically" do
      frame = make_frame(width: 64, height: 64)

      assert {:error, {:crop_out_of_bounds, 0, 32, 64, 64, 64, 64}} =
               ExCubecl.Video.crop(frame, x: 0, y: 32, width: 64, height: 64)
    end

    test "crop with offsets within bounds" do
      frame = make_frame(width: 64, height: 64)
      assert {:ok, cropped} = ExCubecl.Video.crop(frame, x: 16, y: 16, width: 32, height: 32)
      assert cropped.width == 32
      assert cropped.height == 32
    end

    test "crop defaults x and y to zero" do
      frame = make_frame(width: 64, height: 64)
      assert {:ok, cropped} = ExCubecl.Video.crop(frame, width: 16, height: 16)
      assert cropped.width == 16
    end
  end

  describe "Video.convert" do
    test "nv12 to rgb24" do
      frame = make_frame(format: :nv12)
      assert {:ok, converted} = ExCubecl.Video.convert(frame, :nv12, :rgb24)
      assert converted.format == :rgb24
    end

    test "rgb24 source returns unsupported conversion" do
      frame = make_frame(format: :rgb24)

      assert {:error, {:unsupported_conversion, :rgb24}} =
               ExCubecl.Video.convert(frame, :rgb24, :yuv420p)
    end

    test "rgba source returns unsupported conversion" do
      frame = make_frame(format: :rgba)

      assert {:error, {:unsupported_conversion, :rgba}} =
               ExCubecl.Video.convert(frame, :rgba, :rgb24)
    end
  end

  describe "Video.snapshot" do
    test "writes raw buffer data to file", %{test: test} do
      frame = make_frame(width: 8, height: 8)
      path = Path.join(System.tmp_dir!(), "ex_cubecl_snapshot_#{test}.bin")
      on_exit(fn -> File.rm(path) end)

      assert :ok = ExCubecl.Video.snapshot(frame, path)
      assert {:ok, data} = File.read(path)
      expected_bytes = div(8 * 8 * 3, 2)
      assert byte_size(data) == expected_bytes
    end

    test "returns error for invalid path" do
      frame = make_frame(width: 8, height: 8)
      assert {:error, _} = ExCubecl.Video.snapshot(frame, "/nonexistent_dir_xyz/snap.bin")
    end
  end

  describe "Video.overlay and mix params" do
    test "overlay with default options" do
      frame_a = make_frame()
      frame_b = make_frame()
      assert {:ok, %ExCubecl.VideoFrame{}} = ExCubecl.Video.overlay(frame_a, frame_b)
    end

    test "mix with add mode" do
      frame_a = make_frame()
      frame_b = make_frame()

      assert {:ok, %ExCubecl.VideoFrame{}} =
               ExCubecl.Video.mix(frame_a, frame_b, mode: :add, ratio: 0.3)
    end

    test "mix with multiply mode" do
      frame_a = make_frame()
      frame_b = make_frame()

      assert {:ok, %ExCubecl.VideoFrame{}} =
               ExCubecl.Video.mix(frame_a, frame_b, mode: :multiply, ratio: 0.8)
    end

    test "mix with default options" do
      frame_a = make_frame()
      frame_b = make_frame()
      assert {:ok, %ExCubecl.VideoFrame{}} = ExCubecl.Video.mix(frame_a, frame_b)
    end
  end

  # ── ExCubecl.Audio ──────────────────────────────────────────

  describe "Audio.mix" do
    test "mix with default gains" do
      a = make_samples()
      b = make_samples()
      assert {:ok, %ExCubecl.AudioSamples{}} = ExCubecl.Audio.mix([a, b])
    end

    test "mix with custom gains" do
      a = make_samples()
      b = make_samples()
      assert {:ok, %ExCubecl.AudioSamples{}} = ExCubecl.Audio.mix([a, b], gains: [0.7, 0.3])
    end

    test "mix with empty track list" do
      assert {:error, :no_tracks} = ExCubecl.Audio.mix([])
    end
  end

  describe "Audio.overlay" do
    test "overlay with default duck level" do
      bg = make_samples()
      fg = make_samples()
      assert {:ok, %ExCubecl.AudioSamples{}} = ExCubecl.Audio.overlay(bg, fg)
    end

    test "overlay with custom duck level" do
      bg = make_samples()
      fg = make_samples()
      assert {:ok, %ExCubecl.AudioSamples{}} = ExCubecl.Audio.overlay(bg, fg, duck_level: -6)
    end
  end

  describe "Audio.resample validation" do
    test "missing from rate" do
      samples = make_samples()
      assert {:error, {:missing_option, :from}} = ExCubecl.Audio.resample(samples, to: 48_000)
    end

    test "missing to rate" do
      samples = make_samples()
      assert {:error, {:missing_option, :to}} = ExCubecl.Audio.resample(samples, from: 44_100)
    end

    test "zero from rate" do
      samples = make_samples()

      assert {:error, {:invalid_rate, :from, 0}} =
               ExCubecl.Audio.resample(samples, from: 0, to: 48_000)
    end

    test "negative to rate" do
      samples = make_samples()

      assert {:error, {:invalid_rate, :to, -1}} =
               ExCubecl.Audio.resample(samples, from: 44_100, to: -1)
    end

    test "float rates are accepted" do
      samples = make_samples()
      assert {:ok, resampled} = ExCubecl.Audio.resample(samples, from: 44_100.0, to: 48_000.0)
      assert resampled.sample_rate == 48_000
    end

    test "resample down" do
      samples = make_samples(frames: 1000, sample_rate: 48_000)
      assert {:ok, resampled} = ExCubecl.Audio.resample(samples, from: 48_000, to: 24_000)
      assert resampled.sample_rate == 24_000
      assert resampled.frames == 500
    end

    test "resample up" do
      samples = make_samples(frames: 1000, sample_rate: 44_100)
      assert {:ok, resampled} = ExCubecl.Audio.resample(samples, from: 44_100, to: 48_000)
      assert resampled.frames == 1088
    end
  end

  describe "Audio.channels" do
    test "mono to stereo" do
      samples = make_samples(channels: 1, frames: 100)
      assert {:ok, converted} = ExCubecl.Audio.channels(samples, :mono, :stereo)
      assert converted.channels == 2
    end

    test "stereo to mono" do
      samples = make_samples(channels: 2, frames: 100)
      assert {:ok, converted} = ExCubecl.Audio.channels(samples, :stereo, :mono)
      assert converted.channels == 1
    end

    test "mono to surround 5.1" do
      samples = make_samples(channels: 1, frames: 100)
      assert {:ok, converted} = ExCubecl.Audio.channels(samples, :mono, :surround_51)
      assert converted.channels == 6
    end

    test "mono to surround 7.1" do
      samples = make_samples(channels: 1, frames: 100)
      assert {:ok, converted} = ExCubecl.Audio.channels(samples, :mono, :surround_71)
      assert converted.channels == 8
    end

    test "stereo to surround 5.1" do
      samples = make_samples(channels: 2, frames: 100)
      assert {:ok, converted} = ExCubecl.Audio.channels(samples, :stereo, :surround_51)
      assert converted.channels == 6
    end

    test "stereo to surround 7.1" do
      samples = make_samples(channels: 2, frames: 100)
      assert {:ok, converted} = ExCubecl.Audio.channels(samples, :stereo, :surround_71)
      assert converted.channels == 8
    end

    test "surround 5.1 to stereo" do
      samples = make_samples(channels: 6, frames: 100)
      assert {:ok, converted} = ExCubecl.Audio.channels(samples, :surround_51, :stereo)
      assert converted.channels == 2
    end

    test "surround 7.1 to stereo" do
      samples = make_samples(channels: 8, frames: 100)
      assert {:ok, converted} = ExCubecl.Audio.channels(samples, :surround_71, :stereo)
      assert converted.channels == 2
    end

    test "surround 5.1 to surround 7.1" do
      samples = make_samples(channels: 6, frames: 100)
      assert {:ok, converted} = ExCubecl.Audio.channels(samples, :surround_51, :surround_71)
      assert converted.channels == 8
    end

    test "surround 7.1 to surround 5.1" do
      samples = make_samples(channels: 8, frames: 100)
      assert {:ok, converted} = ExCubecl.Audio.channels(samples, :surround_71, :surround_51)
      assert converted.channels == 6
    end

    test "same layout is a no-op" do
      samples = make_samples(channels: 2, frames: 100)
      assert {:ok, converted} = ExCubecl.Audio.channels(samples, :stereo, :stereo)
      assert converted.channels == 2
    end

    test "layout mismatch returns error" do
      samples = make_samples(channels: 2, frames: 100)

      assert {:error, {:channel_layout_mismatch, 2, 1}} =
               ExCubecl.Audio.channels(samples, :mono, :stereo)
    end

    test "unsupported target layout" do
      samples = make_samples(channels: 2, frames: 100)

      assert {:error, {:unsupported_channel_layout, :quad}} =
               ExCubecl.Audio.channels(samples, :stereo, :quad)
    end

    test "unsupported source layout" do
      samples = make_samples(channels: 2, frames: 100)

      assert {:error, {:unsupported_channel_layout, :quad}} =
               ExCubecl.Audio.channels(samples, :quad, :stereo)
    end

    test "channel_count for each layout" do
      assert {:ok, 1} = ExCubecl.Audio.channel_count(:mono)
      assert {:ok, 2} = ExCubecl.Audio.channel_count(:stereo)
      assert {:ok, 6} = ExCubecl.Audio.channel_count(:surround_51)
      assert {:ok, 8} = ExCubecl.Audio.channel_count(:surround_71)
      assert {:error, {:unsupported_channel_layout, :quad}} = ExCubecl.Audio.channel_count(:quad)
    end
  end

  # ── ExCubecl.Filter ─────────────────────────────────────────

  describe "Filter.apply video filters" do
    test "gaussian_blur" do
      frame = make_frame()

      assert {:ok, %ExCubecl.VideoFrame{}} =
               ExCubecl.Filter.apply(frame, :gaussian_blur, radius: 2)
    end

    test "sharpen" do
      frame = make_frame()
      assert {:ok, %ExCubecl.VideoFrame{}} = ExCubecl.Filter.apply(frame, :sharpen, strength: 1.5)
    end

    test "lut" do
      frame = make_frame()
      assert {:ok, %ExCubecl.VideoFrame{}} = ExCubecl.Filter.apply(frame, :lut, file: "warm.cube")
    end

    test "chroma_key" do
      frame = make_frame()

      assert {:ok, %ExCubecl.VideoFrame{}} =
               ExCubecl.Filter.apply(frame, :chroma_key, threshold: 0.3)
    end

    test "brightness_contrast" do
      frame = make_frame()

      assert {:ok, %ExCubecl.VideoFrame{}} =
               ExCubecl.Filter.apply(frame, :brightness_contrast, brightness: 0.1, contrast: 1.2)
    end

    test "denoise" do
      frame = make_frame()
      assert {:ok, %ExCubecl.VideoFrame{}} = ExCubecl.Filter.apply(frame, :denoise, strength: 0.5)
    end

    test "video filter on audio samples returns unknown filter" do
      samples = make_samples()

      assert {:error, {:unknown_filter, :gaussian_blur}} =
               ExCubecl.Filter.apply(samples, :gaussian_blur)
    end
  end

  describe "Filter.apply audio filters" do
    test "normalize" do
      samples = make_samples()
      assert {:ok, %ExCubecl.AudioSamples{}} = ExCubecl.Filter.apply(samples, :normalize)
    end

    test "eq" do
      samples = make_samples()

      assert {:ok, %ExCubecl.AudioSamples{}} =
               ExCubecl.Filter.apply(samples, :eq, bands: [{:high_pass, 80}])
    end

    test "compressor" do
      samples = make_samples()

      assert {:ok, %ExCubecl.AudioSamples{}} =
               ExCubecl.Filter.apply(samples, :compressor, threshold: -18, ratio: 4.0)
    end

    test "reverb" do
      samples = make_samples()

      assert {:ok, %ExCubecl.AudioSamples{}} =
               ExCubecl.Filter.apply(samples, :reverb, room_size: 0.5, wet: 0.2)
    end

    test "audio filter on video frame returns unknown filter" do
      frame = make_frame()
      assert {:error, {:unknown_filter, :normalize}} = ExCubecl.Filter.apply(frame, :normalize)
    end

    test "unknown kernel on audio samples" do
      samples = make_samples()

      assert {:error, {:unknown_filter, :nonexistent}} =
               ExCubecl.Filter.apply(samples, :nonexistent)
    end

    test "invalid input type" do
      assert {:error, {:unknown_filter, :blur}} = ExCubecl.Filter.apply("not a buffer", :blur)
    end
  end

  describe "Filter.chain" do
    test "empty chain returns input" do
      frame = make_frame()
      assert {:ok, %ExCubecl.VideoFrame{} = result} = ExCubecl.Filter.chain(frame, [])
      assert result.handle == frame.handle
    end

    test "audio chain" do
      samples = make_samples()

      assert {:ok, %ExCubecl.AudioSamples{}} =
               ExCubecl.Filter.chain(samples, [
                 {:normalize, []},
                 {:compressor, [threshold: -20, ratio: 3.0]}
               ])
    end

    test "chain with invalid filter returns error" do
      frame = make_frame()

      assert {:error, _} =
               ExCubecl.Filter.chain(frame, [
                 {:nonexistent_kernel_xyz, []}
               ])
    end
  end

  # ── ExCubecl.MediaPipeline ──────────────────────────────────

  defmodule TestPipeline do
    use ExCubecl.MediaPipeline

    def handle_frame(frame, state) do
      {:ok, %{state | frames: state.frames + 1, last: frame}}
    end
  end

  defmodule FailingPipeline do
    use ExCubecl.MediaPipeline

    def handle_frame(_frame, _state) do
      {:error, :boom}
    end
  end

  defmodule CustomInitPipeline do
    use ExCubecl.MediaPipeline

    def init(state) do
      {:ok, Map.put(state, :custom, true)}
    end

    def handle_frame(frame, state) do
      {:ok, %{state | frames: state.frames + 1, last: frame}}
    end
  end

  describe "MediaPipeline" do
    test "start_link and push_frame deliver frames to handle_frame" do
      {:ok, pid} = ExCubecl.MediaPipeline.start_link(TestPipeline, %{frames: 0, last: nil})

      frame = make_frame()
      assert :ok = ExCubecl.MediaPipeline.push_frame(pid, frame)

      assert_receive {:frame, ^frame}, 1_000
      # give the GenServer time to process
      wait_until(fn ->
        :sys.get_state(pid).frames == 1
      end)

      state = :sys.get_state(pid)
      assert state.frames == 1
      assert state.last == frame

      GenServer.stop(pid)
    end

    test "multiple frames increment counter" do
      {:ok, pid} = ExCubecl.MediaPipeline.start_link(TestPipeline, %{frames: 0, last: nil})

      for _ <- 1..3, do: ExCubecl.MediaPipeline.push_frame(pid, make_frame())

      wait_until(fn ->
        state = :sys.get_state(pid)
        state.frames == 3
      end)

      assert :sys.get_state(pid).frames == 3
      GenServer.stop(pid)
    end

    test "handle_frame error stops the server" do
      {:ok, pid} = ExCubecl.MediaPipeline.start_link(FailingPipeline, %{frames: 0, last: nil})
      ref = Process.monitor(pid)

      ExCubecl.MediaPipeline.push_frame(pid, make_frame())

      assert_receive {:DOWN, ^ref, :process, ^pid, :boom}, 1_000
    end

    test "custom init/1 is used" do
      {:ok, pid} = ExCubecl.MediaPipeline.start_link(CustomInitPipeline, %{frames: 0, last: nil})

      assert :sys.get_state(pid).custom == true
      GenServer.stop(pid)
    end

    test "start_link with name option" do
      {:ok, pid} =
        ExCubecl.MediaPipeline.start_link(TestPipeline, %{frames: 0, last: nil},
          name: :test_media_pipeline
        )

      assert Process.whereis(:test_media_pipeline) == pid
      GenServer.stop(pid)
    end

    test "push_frame returns :ok" do
      {:ok, pid} = ExCubecl.MediaPipeline.start_link(TestPipeline, %{frames: 0, last: nil})
      assert :ok = ExCubecl.MediaPipeline.push_frame(pid, make_frame())
      GenServer.stop(pid)
    end
  end

  defp wait_until(fun, tries \\ 50)

  defp wait_until(_fun, 0), do: flunk("condition not met")

  defp wait_until(fun, tries) do
    if fun.() do
      :ok
    else
      Process.sleep(10)
      wait_until(fun, tries - 1)
    end
  end

  # ── ExCubecl.Media / ExCubecl.Transcode (NIF-backed) ───────

  describe "Media" do
    test "open returns a source reference" do
      assert {:ok, src} = ExCubecl.Media.open("test.mp4")
      assert is_reference(src)
    end

    test "open rejects non-binary path" do
      assert_raise FunctionClauseError, fn -> ExCubecl.Media.open(123) end
    end

    test "streams returns video and audio stream info" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      assert {:ok, streams} = ExCubecl.Media.streams(src)
      assert is_list(streams)
      assert length(streams) > 0
    end

    test "read_frame :video returns VideoFrame" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      assert {:ok, %ExCubecl.VideoFrame{} = frame} = ExCubecl.Media.read_frame(src, :video)
      assert is_reference(frame.handle)
      assert frame.width > 0
    end

    test "read_frame :audio returns AudioSamples" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      assert {:ok, %ExCubecl.AudioSamples{} = samples} = ExCubecl.Media.read_frame(src, :audio)
      assert samples.sample_rate > 0
    end

    test "read_frame rejects invalid stream type" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      assert_raise FunctionClauseError, fn -> ExCubecl.Media.read_frame(src, :subtitle) end
    end

    test "streams rejects non-reference" do
      assert_raise FunctionClauseError, fn -> ExCubecl.Media.streams("not a ref") end
    end

    test "close returns ok" do
      {:ok, src} = ExCubecl.Media.open("test.mp4")
      assert :ok = ExCubecl.Media.close(src)
    end

    test "supported? returns boolean" do
      assert is_boolean(ExCubecl.Media.supported?())
    end
  end

  describe "Transcode" do
    test "start validates video codec" do
      assert {:error, {:unsupported_codec, :video, :mpeg2, nil}} =
               ExCubecl.Transcode.start("out.mp4", video: [codec: :mpeg2])
    end

    test "start validates audio codec" do
      assert {:error, {:unsupported_codec, :audio, :wma, nil}} =
               ExCubecl.Transcode.start("out.mp4", audio: [codec: :wma])
    end

    test "start validates container" do
      assert {:error, {:unsupported_container, "avi", nil}} =
               ExCubecl.Transcode.start("out.avi", video: [codec: :h264])
    end

    test "start rejects non-atom codec" do
      assert {:error, {:invalid_codec, :video, "h264"}} =
               ExCubecl.Transcode.start("out.mp4", video: [codec: "h264"])
    end

    test "start with no codecs validates container only" do
      result = ExCubecl.Transcode.start("out.mp4")
      assert match?({:ok, _}, result) or match?({:error, :media_unsupported}, result)
    end

    test "start returns encoder reference" do
      assert {:ok, enc} = ExCubecl.Transcode.start("out.mp4", video: [codec: :h264])
      assert is_reference(enc)
    end

    test "write_frame accepts encoder and frame" do
      {:ok, enc} = ExCubecl.Transcode.start("out.mp4", video: [codec: :h264])
      frame = make_frame()
      assert :ok = ExCubecl.Transcode.write_frame(enc, frame)
    end

    test "write_samples accepts encoder and samples" do
      {:ok, enc} = ExCubecl.Transcode.start("out.mp4", audio: [codec: :aac])
      samples = make_samples()
      assert :ok = ExCubecl.Transcode.write_samples(enc, samples)
    end

    test "finish returns ok" do
      {:ok, enc} = ExCubecl.Transcode.start("out.mp4", video: [codec: :h264])
      assert :ok = ExCubecl.Transcode.finish(enc)
    end

    test "run with unsupported input returns error" do
      assert {:error, _} = ExCubecl.Transcode.run("/nonexistent_input_xyz.mp4", "out.mp4")
    end

    test "supported? returns boolean" do
      assert is_boolean(ExCubecl.Transcode.supported?())
    end
  end

  # ── ExCubecl.NIF (direct stub coverage) ────────────────────

  describe "NIF" do
    test "device_info returns ok map" do
      assert {:ok, info} = ExCubecl.NIF.device_info()
      assert is_map(info)
    end

    test "device_count returns ok integer" do
      assert {:ok, count} = ExCubecl.NIF.device_count()
      assert is_integer(count) and count >= 0
    end

    test "buffer_new with mismatched size returns error tuple" do
      assert {:error, _} = ExCubecl.NIF.buffer_new(<<1, 2, 3>>, [4], "u8")
    end

    test "buffer_new with unknown dtype raises" do
      assert_raise ErlangError, fn -> ExCubecl.NIF.buffer_new(<<1, 2, 3, 4>>, [4], "f16") end
    end

    test "buffer_size/shape/dtype roundtrip" do
      data = <<1.0::float-32-native, 2.0::float-32-native>>
      {:ok, buf} = ExCubecl.NIF.buffer_new(data, [2], "f32")

      assert {:ok, 8} = ExCubecl.NIF.buffer_size(buf)
      assert {:ok, [2]} = ExCubecl.NIF.buffer_shape(buf)
      assert {:ok, "f32"} = ExCubecl.NIF.buffer_dtype(buf)
      assert {:ok, ^data} = ExCubecl.NIF.buffer_read(buf)
    end

    test "kernel_list returns all kernels" do
      assert {:ok, kernels} = ExCubecl.NIF.kernel_list()
      assert "elementwise_add" in kernels
      assert "yuv_to_rgb" in kernels
      assert "pcm_mix" in kernels
      assert length(kernels) == 24
    end

    test "kernel_run with unknown kernel returns error" do
      {:ok, buf} = ExCubecl.buffer([1.0], [1], :f32)
      assert {:error, _} = ExCubecl.NIF.kernel_run("no_such_kernel", [buf], buf, %{})
    end

    test "poll with invalid command id returns error" do
      assert {:error, _} = ExCubecl.NIF.poll(999_999_999)
    end

    test "wait with invalid command id returns error" do
      assert {:error, _} = ExCubecl.NIF.wait(999_999_999)
    end

    test "pipeline_add with invalid pipeline id returns error" do
      {:ok, buf} = ExCubecl.buffer([1.0], [1], :f32)
      assert {:error, _} = ExCubecl.NIF.pipeline_add(999_999_999, "relu", [buf], buf, %{})
    end

    test "pipeline_run with invalid pipeline id returns error" do
      assert {:error, _} = ExCubecl.NIF.pipeline_run(999_999_999)
    end

    test "pipeline_free with invalid pipeline id returns error" do
      assert {:error, _} = ExCubecl.NIF.pipeline_free(999_999_999)
    end

    test "submit returns command id" do
      assert {:ok, id} = ExCubecl.NIF.submit("anything")
      assert is_integer(id) and id > 0
    end

    test "media_streams rejects non-reference" do
      assert_raise MatchError, fn -> ExCubecl.NIF.media_streams("bad") end
    end
  end
end
